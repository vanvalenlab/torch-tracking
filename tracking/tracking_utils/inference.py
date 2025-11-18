import numpy as np
from typing import Dict
from skimage.measure import regionprops
import torch


def get_max_cells(labels):
        """Helper function for finding the maximum number of cells in a frame of a movie, across
        all frames of the movie. Can be used for batches/tracks interchangeably with frames/cells.

        Args:
            y (np.array): Annotated image data

        Returns:
            int: The maximum number of cells in any frame
        """
        max_cells = 0
        for frame in range(labels.shape[0]):
            cells = np.unique(labels[frame])
            n_cells = cells[cells != 0].shape[0]
            if n_cells > max_cells:
                max_cells = n_cells
        return max_cells

def generate_adjacency_matrices(
    centroids: np.ndarray,
    distance_threshold: int = 64
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
        adj = (dists < distance_threshold).astype(np.float32)
        
        # Remove self-loops
        np.fill_diagonal(adj, 0)
        
        adj_matrices[t] = adj
    
    return adj_matrices

def extract_features(
    X: np.ndarray, 
    y: np.ndarray,
    crop_size=32,
    normalize_images = True
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

    max_cells = np.unique(y)[-1]
    T = y.shape[0]
    
    # Lists to store features for each frame
    all_appearances = []
    all_morphologies = []
    all_centroids = []
    all_cell_ids = []
    
    for t in range(T):
        mask_t = y[t, ..., 0]  # Remove channel dimension
        raw_t = X[t, ..., 0]
        
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
            half_crop = crop_size // 2
            
            y_min = max(0, y_center - half_crop)
            y_max = min(raw_t.shape[0], y_center + half_crop)
            x_min = max(0, x_center - half_crop)
            x_max = min(raw_t.shape[1], x_center + half_crop)
            
            crop = raw_t[y_min:y_max, x_min:x_max]
            
            # Pad if necessary
            if crop.shape[0] < crop_size or crop.shape[1] < crop_size:
                padded_crop = np.zeros((crop_size, crop_size))
                padded_crop[:crop.shape[0], :crop.shape[1]] = crop
                crop = padded_crop
            
            # Add channel dimension
            crop = crop[..., np.newaxis]
            
            # Normalize if requested
            if normalize_images:
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
            appearances = np.zeros((0, crop_size, crop_size, 1))
            morphologies = np.zeros((0, 3))
            centroids = np.zeros((0, 2))
            cell_ids = np.array([])
        
        # Pad or crop to max_cells
        n_cells = len(appearances)
        if n_cells < max_cells:
            # Pad
            pad_n = max_cells - n_cells
            appearances = np.pad(
                appearances, 
                ((0, pad_n), (0, 0), (0, 0), (0, 0))
            )
            morphologies = np.pad(morphologies, ((0, pad_n), (0, 0)))
            centroids = np.pad(centroids, ((0, pad_n), (0, 0)))
            cell_ids = np.pad(cell_ids, (0, pad_n), constant_values=-1)
        else:
            # Crop
            appearances = appearances[:max_cells]
            morphologies = morphologies[:max_cells]
            centroids = centroids[:max_cells]
            cell_ids = cell_ids[:max_cells]
        
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
    features['adj_matrices'] = generate_adjacency_matrices(
        features['centroids']
    )

    features = to_tensors(features)
    
    return features

def to_tensors(data: Dict, 
               unbatched: bool = True
               ) -> Dict[str, torch.Tensor]:
    """Convert numpy arrays to PyTorch tensors."""
    tensors = {}
    
    # Appearances: (T, N, H, W, C)
    appearances = torch.from_numpy(data['appearances']).float()
    
    tensors['appearances'] = appearances
    tensors['morphologies'] = torch.from_numpy(data['morphologies']).float()
    tensors['centroids'] = torch.from_numpy(data['centroids']).float()
    tensors['adj_matrices'] = torch.from_numpy(data['adj_matrices']).float()
    
    if 'labels' in data:
        tensors['max_cells'] = data['max_cells']
        tensors['labels'] = data['labels']
    
    if unbatched:
        tensors['appearances'] = tensors['appearances'].unsqueeze(0)
        tensors['morphologies'] = tensors['morphologies'].unsqueeze(0)
        tensors['centroids'] = tensors['centroids'].unsqueeze(0)
        tensors['adj_matrices'] = tensors['adj_matrices'].unsqueeze(0)
        
    return tensors

def generate_labels(
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

    return labels, curr_max_cells