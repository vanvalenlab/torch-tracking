"""Data loading pipeline for .trk format cell tracking data in PyTorch"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import zarr
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as TF
from scipy.spatial.distance import cdist
import tqdm
import warnings
warnings.filterwarnings("ignore") 

from utils import relabel_sequential_lineage, get_image_features, normalize_adj_matrix, resize

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
        crop_size: int = 32,
        stride: int = 1,
        data_format: str = 'channels_first',
        mode: str = 'training',
        normalize_images: bool = True,
        distance_threshold: float = 100.0,
        augment: bool = True,
        rotation_range: int = 180,
        translation_range: float = 0.1,  # As fraction of image size
        crop_mode: str = 'resize',
        truncate_dataset=None
    ):
        super().__init__()
        self.trk_path = Path(trk_path)
        self.track_length = track_length
        self.crop_size = crop_size
        self.stride = stride
        self.appearance_shape = (crop_size, crop_size, 1)
        self.data_format = data_format
        self.mode = mode
        self.normalize_images = normalize_images
        self.distance_threshold = distance_threshold
        self.augment = augment
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.crop_mode = crop_mode
        self.truncate_dataset = truncate_dataset
        
        # Load .trk file
        print(f"Loading {self.trk_path}...")
        self.trk_data = zarr.open(self.trk_path, mode='r')
        
        if self.truncate_dataset is not None:
            self.X = self.trk_data['X'][:self.truncate_dataset] # Raw images (B, T, Y, X, C)
            self.y = self.trk_data['y'][:self.truncate_dataset] # Segmentation masks (B, T, Y, X, C)
            self.lineages = self.trk_data['lineages'][0][:self.truncate_dataset] # Lineage information

        else:
            self.X = self.trk_data['X'] # Raw images (B, T, Y, X, C)
            self.y = self.trk_data['y'] # Segmentation masks (B, T, Y, X, C)
            self.lineages = self.trk_data['lineages'][0] # Lineage information

        if not len(self.X) == len(self.y) == len(self.lineages):
            raise ValueError(
                'The data do not share the same batch size. '
                'Please make sure you are using a valid .trks file')

        self._correct_lineages()
    
        m_cells = 0
        for i in self.lineages:
            curr_m = len(i.keys())
            if curr_m > m_cells:
                m_cells = curr_m
        self.max_cells = m_cells

        self.features = self._get_features()
        
        print(f"  X shape: {self.X.shape}")
        print(f"  y shape: {self.y.shape}")
        print(f"  Lineages: {len(self.lineages)}")
        
        # Build sample indices
        self.samples = self._build_sample_indices()
        print(f"  Created {len(self.samples)} samples")

        if self.augment:
            self._build_augmentation_pipeline()
    
    def _correct_lineages(self):
        """Ensure valid lineages and sequential labels for all batches"""
        new_X = []
        new_y = []
        new_lineages = []
        for batch in tqdm.tqdm(range(self.y.shape[0])):
            # if is_valid_lineage(self.y[batch], self.lineages[batch]):

            y_relabel, new_lineage = relabel_sequential_lineage(
                self.y[batch], self.lineages[batch])

            new_X.append(self.X[batch])
            new_y.append(y_relabel)
            new_lineages.append(new_lineage)
            # else:
            #     print('Invalid lineage detected.')

        self.X = np.stack(new_X, axis=0)
        self.y = np.stack(new_y, axis=0)
        self.lineages = new_lineages
    
    def _get_max_frames(self, curr_lineage):
        """Helper function for finding the maximum number of cells in a frame of a movie, across
        all frames of the movie. Can be used for batches/tracks interchangeably with frames/cells.

        Args:
            y (np.array): Annotated image data

        Returns:
            int: The maximum number of cells in any frame
        """
        max_frames = 0

        for key, track in curr_lineage.items():

            if track['frames']:
                max_frame = track['frames'][-1]
            else:
                max_frame=0

            if max_frame > max_frames:
                max_frames = max_frame

        return max_frames
    
    def _build_sample_indices(self) -> List[Dict]:
        """Build list of valid sample indices.
        
        Creates samples by sliding a window of track_length frames
        across each batch with the specified stride.
        """

        samples = []
        B, T, Y, X, C = self.X.shape
        
        for batch_idx in range(B):
            # Slide window across time dimension
            for start_frame in range(0, T - self.track_length + 1, self.stride):
                end_frame = start_frame + self.track_length

                if self.X[batch_idx, end_frame-1].sum() == 0:
                    continue


                samples.append({
                    'batch_idx': batch_idx,
                    'start_frame': start_frame,
                    'end_frame': end_frame
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single sample.
        
        Returns:
            Dictionary containing:
            - appearances: (track_length, max_cells, H, W, C) or channels_first
            - morphologies: (track_length, max_cells, 3)
            - centroids: (track_length, max_cells, 2)
            - adj_matrices: (track_length, max_cells, max_cells)
            - labels (optional): (track_length-1, max_cells, max_cells)
        """

        sample_info = self.samples[idx]
        
        batch_idx = sample_info['batch_idx']
        start_frame = sample_info['start_frame']
        end_frame = sample_info['end_frame']
        
        # Extract features from masks

        items = {
            'appearances': self.features['appearances'][batch_idx, start_frame:end_frame],
            'centroids': self.features['centroids'][batch_idx, start_frame:end_frame],
            'morphologies': self.features['morphologies'][batch_idx, start_frame:end_frame],
            'adj_matrices': self.features['adj_matrix'][batch_idx, start_frame:end_frame]
        }
        
        # Generate labels if in training mode
        if self.mode == 'training':
            items['mask'] = self.features['mask'][batch_idx, start_frame:end_frame-1]
            items['labels'] = self.features['labels'][batch_idx, start_frame:end_frame-1]

        
        # Convert to tensors
        tensor_data = self._to_tensors(items)
        
        return tensor_data
    
    def _build_augmentation_pipeline(self):
        """Build torchvision v2 augmentation pipeline."""
        self.transform_both = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=self.rotation_range),
        ])

        self.transform_centroid = transforms.Compose([
            transforms.RandomAffine(degrees = 0, translate=(self.translation_range, self.translation_range))
        ])
    
    
    def _get_features(self):
        """
        Extract the relevant features from the label movie
        Appearance, morphologies, centroids, and adjacency matrices
        """
        max_tracks = self.max_cells
        n_batches = self.X.shape[0]
        n_frames = self.X.shape[1]
        n_channels = self.X.shape[-1]

        batch_shape = (n_batches, n_frames, max_tracks)

        appearance_shape = self.appearance_shape

        appearances = np.zeros(batch_shape + appearance_shape, dtype='float32')

        morphologies = np.zeros(batch_shape + (3,), dtype='float32')

        centroids = np.zeros(batch_shape + (2,), dtype='float32')

        adj_matrix = np.zeros(batch_shape + (max_tracks,), dtype='float32')

        temporal_adj_matrix = np.zeros((n_batches,
                                        n_frames - 1,
                                        max_tracks,
                                        max_tracks,
                                        3), dtype='float32')

        mask = np.zeros(batch_shape, dtype='float32')

        track_length = np.zeros((n_batches, max_tracks, 2), dtype='int32')

        for batch in tqdm.tqdm(range(n_batches)):
            for frame in range(n_frames):

                frame_features = get_image_features(
                    self.X[batch, frame], self.y[batch, frame],
                    appearance_dim=self.appearance_shape[1],
                    crop_mode=self.crop_mode)

                track_ids = frame_features['labels'] - 1
                centroids[batch, frame, track_ids] = frame_features['centroids']
                morphologies[batch, frame, track_ids] = frame_features['morphologies']
                appearances[batch, frame, track_ids] = frame_features['appearances']
                mask[batch, frame, track_ids] = 1

                # Get adjacency matrix, cannot filter on track ids.
                cent = centroids[batch, frame]
                distance = cdist(cent, cent, metric='euclidean')
                distance = distance < self.distance_threshold

                # Disconnect the padded nodes
                morphs = morphologies[batch, frame]
                is_pad = np.matmul(morphs, morphs.T) == 0

                adj = distance * (1 - is_pad)
                adj_matrix[batch, frame] = adj.astype(np.float32)

            # Get track length and temporal adjacency matrix
            for label in self.lineages[batch]:

                # Get track length
                start_frame = self.lineages[batch][label]['frames'][0]
                end_frame = self.lineages[batch][label]['frames'][-1]

                track_id = int(label) - 1
                track_length[batch, track_id, 0] = start_frame
                track_length[batch, track_id, 1] = end_frame

                # Get temporal adjacency matrix
                frames = self.lineages[batch][label]['frames']

                # Assign same
                for f0, f1 in zip(frames[0:-1], frames[1:]):
                    if f1 - f0 == 1:
                        temporal_adj_matrix[batch, f0, track_id, track_id, 0] = 1

                # Assign daughter
                # WARNING: This wont work if there's a time gap between mother
                # cell disappearing and daughter cells appearing
                last_frame = frames[-1]
                daughters = self.lineages[batch][label]['daughters']
                for daughter in daughters:
                    daughter_id = daughter - 1
                    temporal_adj_matrix[batch, last_frame, track_id, daughter_id, 2] = 1

            # Assign different
            same_prob = temporal_adj_matrix[batch, ..., 0]
            daughter_prob = temporal_adj_matrix[batch, ..., 2]
            temporal_adj_matrix[batch, ..., 1] = 1 - same_prob - daughter_prob

            # Identify cell padding
            for i in range(temporal_adj_matrix.shape[2]):
                # index + 1 is the cell label
                if i + 1 not in self.lineages[batch]:
                    temporal_adj_matrix[batch, :, i] = -1
                    temporal_adj_matrix[batch, :, :, i] = -1

            # Identify temporal padding
            for b in range(temporal_adj_matrix.shape[0]):
                sames = temporal_adj_matrix[b, ..., 0]
                sames = np.sum(sames, axis=(1, 2))
                temporal_adj_matrix[b, sames == 0] = -1

        features = {
            'adj_matrix': adj_matrix,
            'appearances': appearances,
            'morphologies': morphologies,
            'centroids': centroids,
            'labels': temporal_adj_matrix,
            'mask': mask,
            'track_length': track_length
            }

        return features
    
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
        
        if self.augment:

            centroids = torch.from_numpy(data['centroids']).float()
            appearances = torch.from_numpy(data['appearances']).float().permute(0, 4, 1, 2, 3)

            appearances, centroids = self.transform_both(appearances, centroids)
            centroids = self.transform_centroid(centroids)

            tensors['appearances'] = appearances.permute(0, 2, 3, 4, 1)
            tensors['centroids'] = centroids

        else:

            tensors['appearances'] = torch.from_numpy(data['appearances']).float()
            tensors['centroids'] = torch.from_numpy(data['centroids']).float()

        if 'labels' in data:
            tensors['labels'] = data['labels']
        
        return tensors

def create_trk_dataloaders(
    train_path: Union[str, Path],
    val_path: Optional[Union[str, Path]] = None,
    test_path: Optional[Union[str, Path]] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    track_length: int = 8,
    crop_size: int = 32,
    stride: int = 1,
    collate_fn = None,
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

    
    # Training loader
    train_dataset = TrkDataset(
        trk_path=train_path,
        track_length=track_length,
        crop_size=crop_size,
        stride=stride,
        mode='training',
        **dataset_kwargs
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
        )
    loaders.append(train_loader)
    
    # Validation loader
    if val_path is not None:
        val_dataset = TrkDataset(
            trk_path=val_path,
            track_length=track_length,
            crop_size=crop_size,
            stride=stride,
            mode='training',
            **dataset_kwargs
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        loaders.append(val_loader)
    else:
        loaders.append(None)
    
    # Test loader
    if test_path is not None:
        test_dataset = TrkDataset(
            trk_path=test_path,
            track_length=track_length,
            crop_size=crop_size,
            stride=stride,
            mode='inference',
            **dataset_kwargs
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,            
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

        loaders.append(test_loader)
    else:
        loaders.append(None)
    
    return tuple(loaders)



# Example usage
if __name__ == "__main__":
    print("Testing .trk Data Pipeline")
    print("=" * 70)
    print()
    
    # Example paths (update with your actual paths)
    train_path = "data/DynamicNuclearNet-tracking-v1_0/train.zarr"
    val_path = "data/DynamicNuclearNet-tracking-v1_0/val.zarr"
    
    if Path(train_path).exists():
        print("1. Creating dataloaders...")
        train_loader, val_loader, test_loader = create_trk_dataloaders(
            train_path=train_path,
            val_path=val_path if Path(val_path).exists() else None,
            test_path=None,
            batch_size=2,
            num_workers=4,  # Use 0 for testing
            track_length=8,
            crop_size=32,
            stride=1  # Sample every 4 frames for faster testing
        )
        
        print(f"   Train batches: {len(train_loader)}")
        if val_loader:
            print(f"   Val batches: {len(val_loader)}")
        print()
        
        print("2. Testing batch loading...")
        for i, batch in enumerate(train_loader):
            print(f"   Batch {i}:")
            for key, value in batch.items():
                print(f"     {key}: {value.shape}, dtype: {value.dtype}")
            
        print()
        
        print("✅ .trk data pipeline working!")
    else:
        print("⚠️  train.trk not found. Please update paths in the example.")
        print()
        print("To use with your data:")
        print("  train_loader, val_loader, _ = create_trk_dataloaders(")
        print("      train_path='path/to/train.trk',")
        print("      val_path='path/to/val.trk',")
        print("      batch_size=4,")
        print("      track_length=8,")
        print("      max_cells=39")
        print("  )")