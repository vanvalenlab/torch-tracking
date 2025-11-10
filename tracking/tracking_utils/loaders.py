"""Data loading pipeline for .trk format cell tracking data in PyTorch"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from skimage.measure import regionprops
import zarr

import numpy as np

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
        appearance_shape: Tuple[int, int, int] = (32, 32, 1),
        data_format: str = 'channels_first',
        mode: str = 'training',
        normalize_images: bool = True,
        distance_threshold: float = 100.0
    ):
        super().__init__()
        self.trk_path = Path(trk_path)
        self.track_length = track_length
        self.crop_size = crop_size
        self.stride = stride
        self.appearance_shape = appearance_shape
        self.data_format = data_format
        self.mode = mode
        self.normalize_images = normalize_images
        self.distance_threshold = distance_threshold
        
        # Load .trk file
        print(f"Loading {self.trk_path}...")
        self.trk_data = zarr.open(self.trk_path, mode='r')
        
        self.X = self.trk_data['X']  # Raw images (B, T, Y, X, C)
        self.y = self.trk_data['y']  # Segmentation masks (B, T, Y, X, C)
        self.lineages = self.trk_data['lineages'][0]  # Lineage information

        m_cells = 0
        for i in self.lineages:
            curr_m = len(i.keys())
            if curr_m>m_cells:
                m_cells=curr_m

        self.max_cells = m_cells
        (self.labels, self.max_cells_list) = self._generate_labels(self.lineages, max_cells = self.max_cells, max_frames=self.X.shape[1])
        
        print(f"  X shape: {self.X.shape}")
        print(f"  y shape: {self.y.shape}")
        print(f"  Lineages: {len(self.lineages)}")
        
        # Build sample indices
        self.samples = self._build_sample_indices()
        print(f"  Created {len(self.samples)} samples")

    def _get_max_cells(self):
        """Helper function for finding the maximum number of cells in a frame of a movie, across
        all frames of the movie. Can be used for batches/tracks interchangeably with frames/cells.

        Args:
            y (np.array): Annotated image data

        Returns:
            int: The maximum number of cells in any frame
        """
        max_cells = 0
        for frame in range(self.y.shape[0]):
            cells = np.unique(self.y[frame])
            n_cells = cells[cells != 0].shape[0]
            if n_cells > max_cells:
                max_cells = n_cells
        return max_cells
    
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
        
        # Extract raw images and masks for this sample
        raw_images = self.X[batch_idx, start_frame:end_frame]  # (T, Y, X, C)
        masks = self.y[batch_idx, start_frame:end_frame]  # (T, Y, X, C)
        
        # Extract features from masks
        features = self._extract_features_from_masks(raw_images, masks)
        
        # Generate labels if in training mode
        if self.mode == 'training':
            
            unpadded_labels = self.labels[batch_idx, start_frame:end_frame-1]
            features['max_cells'] = self.max_cells_list[batch_idx]
            features['labels'] = unpadded_labels
        
        # Convert to tensors
        tensor_data = self._to_tensors(features)
        
        return tensor_data
    
    def _extract_features_from_masks(
        self, 
        raw_images: np.ndarray, 
        masks: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Extract cell features from segmentation masks.
        
        Args:
            raw_images: (T, Y, X, C) raw fluorescent images
            masks: (T, Y, X, C) segmentation masks
        
        Returns:
            Dictionary with:
            - appearances: (T, N, H, W, C) cropped cell images
            - morphologies: (T, N, 3) [area, perimeter, eccentricity]
            - centroids: (T, N, 2) [y, x] positions
            - cell_ids: (T, N) cell IDs from masks
        """
        T = len(masks)
        
        # Lists to store features for each frame
        all_appearances = []
        all_morphologies = []
        all_centroids = []
        all_cell_ids = []
        
        for t in range(T):
            mask_t = masks[t, ..., 0]  # Remove channel dimension
            raw_t = raw_images[t, ..., 0]
            
            # Get region properties
            props = regionprops(mask_t.astype(int))
            
            # Extract features for each cell
            appearances = []
            morphologies = []
            centroids = []
            cell_ids = []
            
            for prop in props:
                cell_id = prop.label
                
                # Centroid (y, x)
                centroid = np.array(prop.centroid)
                
                # Morphology features
                area = prop.area
                perimeter = prop.perimeter
                eccentricity = prop.eccentricity
                morphology = np.array([area, perimeter, eccentricity])
                
                # Extract appearance crop
                y_center, x_center = int(centroid[0]), int(centroid[1])
                half_crop = self.crop_size // 2
                
                y_min = max(0, y_center - half_crop)
                y_max = min(raw_t.shape[0], y_center + half_crop)
                x_min = max(0, x_center - half_crop)
                x_max = min(raw_t.shape[1], x_center + half_crop)
                
                crop = raw_t[y_min:y_max, x_min:x_max]
                
                # Pad if necessary
                if crop.shape[0] < self.crop_size or crop.shape[1] < self.crop_size:
                    padded_crop = np.zeros((self.crop_size, self.crop_size))
                    padded_crop[:crop.shape[0], :crop.shape[1]] = crop
                    crop = padded_crop
                
                # Add channel dimension
                crop = crop[..., np.newaxis]
                
                # Normalize if requested
                if self.normalize_images:
                    crop = (crop - crop.mean()) / (crop.std() + 1e-7)
                
                appearances.append(crop)
                morphologies.append(morphology)
                centroids.append(centroid)
                cell_ids.append(cell_id)
            
            # Convert to arrays and pad/crop to max_cells
            if len(appearances) > 0:
                appearances = np.stack(appearances)  # (N, H, W, C)
                morphologies = np.stack(morphologies)  # (N, 3)
                centroids = np.stack(centroids)  # (N, 2)
                cell_ids = np.array(cell_ids)  # (N,)
            else:
                # No cells in this frame
                appearances = np.zeros((0, self.crop_size, self.crop_size, 1))
                morphologies = np.zeros((0, 3))
                centroids = np.zeros((0, 2))
                cell_ids = np.array([])
            
            # Pad or crop to max_cells
            n_cells = len(appearances)
            if n_cells < self.max_cells:
                # Pad
                pad_n = self.max_cells - n_cells
                appearances = np.pad(
                    appearances, 
                    ((0, pad_n), (0, 0), (0, 0), (0, 0))
                )
                morphologies = np.pad(morphologies, ((0, pad_n), (0, 0)))
                centroids = np.pad(centroids, ((0, pad_n), (0, 0)))
                cell_ids = np.pad(cell_ids, (0, pad_n), constant_values=-1)
            else:
                # Crop
                appearances = appearances[:self.max_cells]
                morphologies = morphologies[:self.max_cells]
                centroids = centroids[:self.max_cells]
                cell_ids = cell_ids[:self.max_cells]
            
            all_appearances.append(appearances)
            all_morphologies.append(morphologies)
            all_centroids.append(centroids)
            all_cell_ids.append(cell_ids)
        
        # Stack across time
        features = {
            'appearances': np.stack(all_appearances),  # (T, N, H, W, C)
            'morphologies': np.stack(all_morphologies),  # (T, N, 3)
            'centroids': np.stack(all_centroids),  # (T, N, 2)
            'cell_ids': np.stack(all_cell_ids)  # (T, N)
        }
        
        # Generate adjacency matrices
        features['adj_matrices'] = self._generate_adjacency_matrices(
            features['centroids']
        )
        
        return features
    
    def _generate_adjacency_matrices(
        self, 
        centroids: np.ndarray
    ) -> np.ndarray:
        """Generate adjacency matrices based on centroid distances.
        
        Args:
            centroids: (T, N, 2) array of centroid positions
        
        Returns:
            adj_matrices: (T, N, N) adjacency matrices
        """
        T, N, _ = centroids.shape
        adj_matrices = np.zeros((T, N, N), dtype=np.float32)
        
        for t in range(T):
            cents = centroids[t]  # (N, 2)
            
            # Compute pairwise distances
            dists = np.sqrt(
                ((cents[:, None, :] - cents[None, :, :]) ** 2).sum(axis=-1)
            )
            
            # Create adjacency based on threshold
            adj = (dists < self.distance_threshold).astype(np.float32)
            
            # Remove self-loops
            np.fill_diagonal(adj, 0)
            
            adj_matrices[t] = adj
        
        return adj_matrices
    
    def _generate_labels(
        self,
        lineage: dict,
        max_cells=None,
        max_frames=None
    ) -> np.ndarray:
        """Generate ground truth tracking labels from lineages.
        
        Args:
            batch_idx: Batch index
            start_frame: Start frame of sequence
            end_frame: End frame of sequence
            cell_ids: (T, N) array of cell IDs in each frame
        
        Returns:
            labels: (T-1, N, N) label matrix where labels[t, i, j] indicates:
                0: no link
                1: different cell
                2: same cell (i in frame t links to j in frame t+1)
        """
        
        curr_max_cells = []
        labels = np.zeros((len(lineage), max_frames, max_cells, max_cells))

        for i, curr_lineage in enumerate(lineage):

            curr_max_cells.append(len(curr_lineage))

            for label, track in curr_lineage.items():
                # The frames where each track is linked to itself (encoded as 1)
                frames_linked = track['frames'][:-1]

                #Identity label
                label = int(label) - 1

                # Place all identities
                labels[i, frames_linked, label, label] = 1

                # Place all mitotic frames as linkages
                if track['frame_div']:
                    frame_divided = track['frame_div']

                    daughter_label = track['daughters']
                    daughter_label = [daughter-1 for daughter in daughter_label]

                    labels[i, frame_divided-1, label, daughter_label] = 2

        return torch.as_tensor(labels, dtype=torch.long), curr_max_cells
    
    def _to_tensors(self, data: Dict) -> Dict[str, torch.Tensor]:
        """Convert numpy arrays to PyTorch tensors."""
        tensors = {}
        
        # Appearances: (T, N, H, W, C)
        appearances = torch.from_numpy(data['appearances']).float()
        
        if self.data_format == 'channels_first':
            # Need to handle the N dimension appropriately
            # For the model, we expect (C, T, H, W) or (T, C, H, W)
            # But we have (T, N, H, W, C) where each N is a separate cell
            
            # We'll process each cell separately in the model
            # Keep as (T, N, H, W, C) for now, model will handle it
            pass
        
        tensors['appearances'] = appearances
        tensors['morphologies'] = torch.from_numpy(data['morphologies']).float()
        tensors['centroids'] = torch.from_numpy(data['centroids']).float()
        tensors['adj_matrices'] = torch.from_numpy(data['adj_matrices']).float()
        
        if 'labels' in data:
            tensors['max_cells'] = data['max_cells']
            tensors['labels'] = data['labels']
        
        return tensors
    
def unpad_collate_fn(batch):

    actual_max_cells = batch[0]['max_cells']

    assert all(sample['max_cells'] == actual_max_cells for sample in batch), \
        "Batch contains samples from different movies with different cell counts!"
    
    collated = {}

    # Stack appearances: (B, T, N, H, W, C) but slice N to actual_max_cells
    appearances = torch.stack([
        sample['appearances'][:, :actual_max_cells]  # (T, N_actual, H, W, C)
        for sample in batch
    ])
    collated['appearances'] = appearances  # (B, T, N_actual, H, W, C)
    
    # Stack morphologies
    morphologies = torch.stack([
        sample['morphologies'][:, :actual_max_cells]  # (T, N_actual, 3)
        for sample in batch
    ])
    collated['morphologies'] = morphologies
    
    # Stack centroids
    centroids = torch.stack([
        sample['centroids'][:, :actual_max_cells]  # (T, N_actual, 2)
        for sample in batch
    ])
    collated['centroids'] = centroids
    
    # Stack adjacency matrices
    adj_matrices = torch.stack([
        sample['adj_matrices'][:, :actual_max_cells, :actual_max_cells]  # (T, N_actual, N_actual)
        for sample in batch
    ])
    collated['adj_matrices'] = adj_matrices
    
    # Stack labels (also unpad!)
    if 'labels' in batch[0]:
        labels = torch.stack([
            sample['labels'][:, :actual_max_cells, :actual_max_cells]  # (T-1, N_actual, N_actual)
            for sample in batch
        ])
        collated['labels'] = labels
    
    # Store actual_max_cells for reference
    collated['actual_max_cells'] = actual_max_cells
    
    return collated

class BatchSampler(Sampler):

    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.movie_groups = {}
        for idx, sample in enumerate(dataset.samples):
            movie_id = sample['batch_idx']
            if movie_id not in self.movie_groups:
                self.movie_groups[movie_id] = []
            self.movie_groups[movie_id].append(idx)

        self.batches = []
        for movie_id, indices in self.movie_groups.items():
            for i in range(0, len(indices), self.batch_size):
                # Split samples from each movie into batches of length batch size

                batch = indices[i:i+self.batch_size]
                if len(batch) == self.batch_size:

                    # Keep full batches only
                    self.batches.append(batch)

    def __iter__(self):

        if self.shuffle:
            indices = torch.randperm(len(self.batches)).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches

        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.batches)


def create_trk_dataloaders(
    train_path: Union[str, Path],
    val_path: Optional[Union[str, Path]] = None,
    test_path: Optional[Union[str, Path]] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    track_length: int = 8,
    crop_size: int = 32,
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
        batch_sampler = BatchSampler(train_dataset, batch_size=batch_size, shuffle=True),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=unpad_collate_fn
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
            batch_sampler = BatchSampler(val_dataset, batch_size=batch_size, shuffle=False),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=unpad_collate_fn
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
            batch_sampler = BatchSampler(test_dataset, batch_size=batch_size, shuffle=False),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=unpad_collate_fn
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
            num_workers=0,  # Use 0 for testing
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
            
            if i == 0:
                break
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