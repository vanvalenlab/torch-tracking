"""Data loading pipeline for .trk format cell tracking data in PyTorch"""

import numpy as np
import torch
import zarr

import tqdm
import warnings

from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import torchvision.transforms.v2.functional as F
from scipy.spatial.distance import cdist

from torch_tracking.utils import require_mpp, require_lineages, count_nonempty_frames, spatial_adjacency, DISTANCE_THRESHOLD
warnings.filterwarnings("ignore") 

#: The processed-zarr arrays the dataloader reads, by name. Anything else a zarr holds is
#: never loaded, so an array no stage uses costs no memory.
REQUIRED_ARRAYS = ('appearances', 'morphologies', 'centroids', 'mpp')


def get_temporal_adjacency(lineage, start_frame, end_frame, max_cells):
    """Training targets for the frames ``[start_frame, end_frame)`` of one movie.

    Returns an int8 array ``(end_frame - start_frame - 1, max_cells, max_cells)``: entry
    ``[t]`` compares the cells of frame ``start_frame + t`` (rows, the tracks) with the
    cells of the next frame (columns, the candidates). A cell's slot is its label minus
    one.

    An entry is graded only when both of its slots carry a cell: the row's cell in the
    first frame and the column's cell in the next. Otherwise it is ``-1`` (padding) and
    the loss ignores it. That covers unused capacity, and equally a cell not yet born,
    already ended, or missing from one of the two frames. Such a slot holds all-zero
    features, identical to unused capacity, so the targets do not tell them apart.

    * ``1`` on the diagonal: the cell is in both frames.
    * ``2`` from a parent to each daughter, at the parent's final frame.
    * ``0`` for every other graded pair: two cells that are not the same.

    A newborn cell therefore has its column graded in the step into its first frame (it
    is not any existing track), and a cell that ends has its row graded in its last frame
    (it matches no candidate). The division entries are written after padding so that
    they can never be erased.

    Built from the lineage alone, for just the window requested, so there is no frame
    beyond the movie's extent to pad.
    """
    n_frames = end_frame - start_frame
    adjacency = np.zeros((n_frames - 1, max_cells, max_cells), dtype=np.int8)
    present = np.zeros((n_frames, max_cells), dtype=bool)
    divisions = []

    for attr in lineage.values():
        slot = int(attr['label']) - 1
        frames = np.asarray(attr['frames'], dtype=int)
        if frames.size == 0:
            continue
        in_window = frames[(frames >= start_frame) & (frames < end_frame)] - start_frame
        present[in_window, slot] = True

        final = int(frames[-1])
        if attr['daughters'] and start_frame <= final < end_frame - 1:
            divisions.append((final - start_frame, slot, [int(d) - 1 for d in attr['daughters']]))

    tracks, candidates = present[:-1], present[1:]
    np.putmask(adjacency, ~tracks[:, :, None] | ~candidates[:, None, :], -1)

    steps, slots = np.nonzero(tracks & candidates)
    adjacency[steps, slots, slots] = 1

    for step, slot, daughters in divisions:
        adjacency[step, slot, daughters] = 2

    return adjacency


class TrkDataset(Dataset):
    """PyTorch Dataset for .trk format cell tracking data.
    
    Loads data from .trk files containing:
    - X: raw fluorescent nuclear data (B, T, Y, X, C)
    - y: nuclear segmentation masks (B, T, Y, X, C)
    - lineages: lineage records with cell id, frames, divisions
    
    Args:
        trk_path (str or Path): Path to .trk file (train.trk, val.trk, etc.)
        track_length (int): Number of consecutive frames per sample
        max_cells (int): Maximum number of cells per frame
        crop_size (int): Size of appearance crops (will be crop_size x crop_size)
        stride (int): Stride for temporal sampling (1 = every frame)
        appearance_shape (tuple): Output shape for crops (H, W, C)
        data_format (str): 'channels_first' or 'channels_last'
        mode (str): 'training' or 'inference'
        normalize_images (bool): Whether to normalize raw images
        distance_threshold (float): Distance for adjacency matrix generation
    """
    def __init__(
        self,
        trk_path: Union[str, Path],
        track_length: int = 8,
        stride: int = 1,
        data_format: str = 'channels_first',
        distance_threshold: float = DISTANCE_THRESHOLD,
        augment: bool = True,
        rotation_range: int = 180,
        translation_range: float = 512,
        truncate_dataset=None,
    ):
        
        super().__init__()
        self.trk_path = Path(trk_path)
        self.track_length = track_length
        self.stride = stride
        self.data_format = data_format
        self.distance_threshold = distance_threshold
        self.augment = augment
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.truncate_dataset = truncate_dataset

        features = zarr.open(self.trk_path)
        require_mpp(features, self.trk_path)
        lineages = require_lineages(features, self.trk_path)

        for k in REQUIRED_ARRAYS:
            if k not in features:
                raise ValueError(f"{self.trk_path} holds no '{k}' array.")

        # Only the arrays named above are read; a retired array in an older zarr is not.
        self.features = {}
        for k in tqdm.tqdm(REQUIRED_ARRAYS):
            
            if self.truncate_dataset is not None:
                self.features[k] = features[k][:self.truncate_dataset]

            else:
                self.features[k] = features[k][:]

        self.lineages = list(lineages[:self.truncate_dataset])

        # Build sample indices
        self.samples = self._build_sample_indices()
        print(f"  Created {len(self.samples)} samples")

    
    def _build_sample_indices(self) -> List[Dict]:
        """Build list of valid sample indices.
        
        Creates samples by sliding a window of track_length frames
        across each batch with the specified stride.
        """

        samples = []
        B, T, N, Y, X, C = self.features['appearances'].shape
        
        for batch_idx in range(B):
            # Slide window across the movie's occupied frames. The extent is the shared
            # rule evaluation uses, and a window is kept while it ends within it, so
            # every non-empty frame is reachable and none reaches into padding.
            extent = min(T, count_nonempty_frames(self.features['appearances'][batch_idx]))

            for start_frame in range(0, extent - self.track_length + 1, self.stride):
                end_frame = start_frame + self.track_length

                samples.append({
                    'batch_idx': batch_idx,
                    'start_frame': start_frame,
                    'end_frame': end_frame
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)      

    def _apply_augmentation(self, appearances, centroids):
        
        self.rotation_angle = (torch.rand(1) - 0.5) * 2 * self.rotation_range
        angle_rad = torch.deg2rad(torch.tensor(self.rotation_angle))

        rotation_mat =  torch.tensor([
            [torch.cos(angle_rad), -torch.sin(angle_rad)],
            [torch.sin(angle_rad), torch.cos(angle_rad)],
        ])

        # Apply rotation
        appearances = appearances.permute(0, 1, 4, 2, 3)
        appearances = F.rotate(appearances, self.rotation_angle,
                            interpolation=F.InterpolationMode.BILINEAR)
        appearances = appearances.permute(0, 1, 3, 4, 2)

        centroids = torch.matmul(centroids, rotation_mat)
            
        # Apply translation
        random_translate = (torch.rand(2) - 0.5) * 2 * self.translation_range
        self.random_translate = random_translate.unsqueeze(0).unsqueeze(0)
        centroids = centroids + self.random_translate

        return appearances, centroids
    
    def _get_adj_matrix(self, centroids, morphologies):
        """Spatial graph for each frame of a window, by the rule inference also uses."""

        T, N, _ = centroids.shape
        adj_matrix = np.zeros((T, N, N), dtype=bool)

        for time in range(T):
            adj_matrix[time] = spatial_adjacency(
                centroids[time], morphologies[time], self.distance_threshold
            )

        return adj_matrix
    
    def _to_tensors(self, data: Dict) -> Dict[str, torch.Tensor]:
        """Convert numpy arrays to PyTorch tensors."""
        tensors = {}
        
        # Appearances: (T, N, H, W, C)
        
        if self.data_format == 'channels_first':
            # Need to handle the N dimension appropriately
            # For the model, we expect (C, T, H, W) or (T, C, H, W)
            # But we have (T, N, H, W, C) where each N is a separate cell
            
            # We'll process each cell separately in the model
            # Keep as (T, N, H, W, C) for now, model will handle it
            pass

        tensors['morphologies'] = torch.from_numpy(data['morphologies']).float()
        tensors['adj_matrices'] = torch.from_numpy(data['adj_matrices']).float()
        tensors['labels'] = torch.from_numpy(data['labels']).float()


        if self.augment:

            centroids = torch.from_numpy(data['centroids']).float()
            appearances = torch.from_numpy(data['appearances']).float()

            appearances, centroids = self._apply_augmentation(appearances, centroids)

            tensors['appearances'] = appearances
            tensors['centroids'] = centroids

        else:

            tensors['appearances'] = torch.from_numpy(data['appearances']).float()
            tensors['centroids'] = torch.from_numpy(data['centroids']).float()
        
        return tensors

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single sample.
        
        Returns:
            Dictionary containing:
            - appearances: (track_length, max_cells, H, W, C) or channels_first
            - morphologies: (track_length, max_cells, 3)
            - centroids: (track_length, max_cells, 2)
            - adj_matrices: (track_length, max_cells, max_cells)
            - labels: (track_length-1, max_cells, max_cells), built from the lineage
        """

        sample_info = self.samples[idx]
        
        batch_idx = sample_info['batch_idx']
        start_frame = sample_info['start_frame']
        end_frame = sample_info['end_frame']

        time_slice = slice(start_frame, end_frame)

        items = {
            'appearances': self.features['appearances'][batch_idx, time_slice],
            'centroids': self.features['centroids'][batch_idx, time_slice],
            'morphologies': self.features['morphologies'][batch_idx, time_slice]
        }

        items['adj_matrices'] = self._get_adj_matrix(items['centroids'], items['morphologies'])
        
        # Generate labels if in training mode

        items['labels'] = get_temporal_adjacency(
            self.lineages[batch_idx], start_frame, end_frame,
            self.features['appearances'].shape[2],
        )

        # Convert to tensors
        tensor_data = self._to_tensors(items)
        
        return tensor_data
    

    
def create_trk_dataloaders(
    train_path: Optional[Union[str, Path]] = None,
    val_path: Optional[Union[str, Path]] = None,
    test_path: Optional[Union[str, Path]] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    track_length: int = 8,
    stride: int = 1,
    **dataset_kwargs
) -> Tuple[DataLoader, ...]:
    
    """Create dataloaders for .trk format data.
    
    Args:
        train_path: Path to train.trk
        val_path: Path to val.trk (optional)
        test_path: Path to test.trk (optional)
        batch_size: Batch size
        num_workers: Number of data loading workers
        track_length: Number of frames per sequence
        max_cells: Maximum cells per frame
        crop_size: Size of appearance crops
        stride: Temporal stride for sampling
        **dataset_kwargs: Additional arguments for TrkDataset
    
    Returns:
        Tuple of DataLoaders (train_loader, val_loader, test_loader)
        None for loaders where path not provided
    """

    loaders = []

    if train_path is not None:
        # Training loader
        train_dataset = TrkDataset(
            trk_path=train_path,
            track_length=track_length,
            stride=stride,
            **dataset_kwargs
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            )
        loaders.append(train_loader)
    else:
        loaders.append(None)
    
    # Validation loader
    if val_path is not None:
        val_dataset = TrkDataset(
            trk_path=val_path,
            track_length=track_length,
            stride=stride,
            augment=False,
            **dataset_kwargs
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        loaders.append(val_loader)
    else:
        loaders.append(None)
    
    # Test loader
    if test_path is not None:
        test_dataset = TrkDataset(
            trk_path=test_path,
            track_length=track_length,
            stride=stride,
            augment=False,
            **dataset_kwargs
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        loaders.append(test_loader)
    else:
        loaders.append(None)
    
    return tuple(loaders)


if __name__ == '__main__':

    config = {
        "optimizer": "radam",
        "learning_rate": 1e-3,
        "weight_decay": 0,
        "decay": 0.99,
        "scheduler": "reduce_on_plateau",
        "max_epochs": 50,
        "batch_size": 6,
        "n_layers": 1,
        "num_workers": 8,
        "clipnorm": 0.001,
        "step_size": 5,
        "patience": 5,
        "log_and_save": True,
        "enable_early_stopping": False,
        "crop_size": 32,
        "attention": False,
        "truncate_dataset": 4,
        "loss": "wcce",
        "processed": True,
        "dropout": 0,
        "device": "cuda:0",
        "label_smoothing": False,
        "stopping_metric": 'loss'
    }

    train_loader, val_loader, _ = create_trk_dataloaders(
        train_path='data/DynamicNuclearNet-tracking-v1_0/train_proc.zarr',
        val_path='data/DynamicNuclearNet-tracking-v1_0/val_proc.zarr',
        batch_size=config['batch_size'],
        distance_threshold=DISTANCE_THRESHOLD,
        num_workers=config['num_workers'],
        truncate_dataset = config['truncate_dataset'],
    )

    for sample in train_loader:
        print(sample)

    print()

