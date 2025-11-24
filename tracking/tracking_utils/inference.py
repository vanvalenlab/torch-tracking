import numpy as np
from typing import Dict
from skimage.measure import regionprops
import torch
from utils import relabel_sequential_lineage, get_image_features
import tqdm

from scipy.spatial.distance import cdist


def correct_lineages(X, y, lineages):

    new_X = []
    new_y = []
    new_lineages = []
    for batch in tqdm.tqdm(range(y.shape[0])):

        y_relabel, new_lineage = relabel_sequential_lineage(
            y[batch], lineages[batch])

        new_X.append(X[batch])
        new_y.append(y_relabel)
        new_lineages.append(new_lineage)

    new_X = np.stack(new_X, axis=0)
    new_y = np.stack(new_y, axis=0)

    return new_X, new_y, new_lineages

def get_features(X, y, lineages, appearance_shape=(16, 16, 1), crop_mode='fixed', distance_threshold=64):
    """
    Extract the relevant features from the label movie
    Appearance, morphologies, centroids, and adjacency matrices
    """

    max_tracks = get_max_cells(y)
    n_batches = X.shape[0]
    n_frames = X.shape[1]
    n_channels = X.shape[-1]

    batch_shape = (n_batches, n_frames, max_tracks)

    appearance_shape = appearance_shape

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
                X[batch, frame], y[batch, frame],
                appearance_dim=appearance_shape[1],
                crop_mode=crop_mode)

            track_ids = frame_features['labels'] - 1
            centroids[batch, frame, track_ids] = frame_features['centroids']
            morphologies[batch, frame, track_ids] = frame_features['morphologies']
            appearances[batch, frame, track_ids] = frame_features['appearances']
            mask[batch, frame, track_ids] = 1

            # Get adjacency matrix, cannot filter on track ids.
            cent = centroids[batch, frame]
            distance = cdist(cent, cent, metric='euclidean')
            distance = distance < distance_threshold

            # Disconnect the padded nodes
            morphs = morphologies[batch, frame]
            is_pad = np.matmul(morphs, morphs.T) == 0

            adj = distance * (1 - is_pad)
            adj_matrix[batch, frame] = adj.astype(np.float32)

        # Get track length and temporal adjacency matrix
        for label in lineages[batch]:

            # Get track length
            start_frame = lineages[batch][label]['frames'][0]
            end_frame = lineages[batch][label]['frames'][-1]

            track_id = int(label) - 1
            track_length[batch, track_id, 0] = start_frame
            track_length[batch, track_id, 1] = end_frame

            # Get temporal adjacency matrix
            frames = lineages[batch][label]['frames']

            # Assign same
            for f0, f1 in zip(frames[0:-1], frames[1:]):
                if f1 - f0 == 1:
                    temporal_adj_matrix[batch, f0, track_id, track_id, 0] = 1

            # Assign daughter
            # WARNING: This wont work if there's a time gap between mother
            # cell disappearing and daughter cells appearing
            last_frame = frames[-1]
            daughters = lineages[batch][label]['daughters']
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
            if i + 1 not in lineages[batch]:
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

def to_tensors(data: Dict, 
               unbatched: bool = True,
               device=None
               ) -> Dict[str, torch.Tensor]:
    """Convert numpy arrays to PyTorch tensors."""
    tensors = {}
    
    # Appearances: (T, N, H, W, C)
    appearances = torch.from_numpy(data['appearances']).float().to(device)
    
    tensors['appearances'] = appearances
    tensors['morphologies'] = torch.from_numpy(data['morphologies']).float().to(device)
    tensors['centroids'] = torch.from_numpy(data['centroids']).float().to(device)
    tensors['adj_matrix'] = torch.from_numpy(data['adj_matrix']).float().to(device)
    
    if 'labels' in data:
        tensors['labels'] = torch.from_numpy(data['labels']).float().to(device)
    
    if unbatched:
        tensors['appearances'] = tensors['appearances'].unsqueeze(0).to(device)
        tensors['morphologies'] = tensors['morphologies'].unsqueeze(0).to(device)
        tensors['centroids'] = tensors['centroids'].unsqueeze(0).to(device)
        tensors['adj_matrix'] = tensors['adj_matrix'].unsqueeze(0).to(device)
        
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