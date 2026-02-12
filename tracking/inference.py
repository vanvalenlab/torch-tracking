import numpy as np
from typing import Dict
from skimage.measure import regionprops
import torch
from tracking.utils import resize
import tqdm
from skimage.segmentation import relabel_sequential
import warnings


def get_image_features(X, y, appearance_dim=32, crop_mode='fixed', norm=True):
    """Return features for every object in the array.

    Args:
        X (np.array): a 3D numpy array of raw data of shape (x, y, c).
        y (np.array): a 3D numpy array of integer labels of shape (x, y, 1).
        appearance_dim (int): The resized shape of the appearance feature.
        crop_mode (str): Whether to do a fixed crop or to crop and resize
            to create the appearance features
        norm (bool): Whether to remove non cell features and normalize the
            foreground pixels by zero-meaning and dividing by the standard
            deviation. Applies to fixed crop mode only.

    Returns:
        dict: A dictionary of feature names to np.arrays of shape
            (n, c) or (n, x, y, c) where n is the number of objects.
    """
    # X must be float32 for the resize norm option to work correctly
    X = X.astype('float32')
    y = y.astype('int32')

    if crop_mode not in ['resize', 'fixed']:
        raise ValueError('crop_mode must be either resize or fixed')

    appearance_dim = int(appearance_dim)

    # each feature will be ordered based on the label.
    # labels are also stored and can be fetched by index.
    num_labels = len(np.unique(y)) - 1
    labels = np.zeros((num_labels,), dtype='int32')
    centroids = np.zeros((num_labels, 2), dtype='float32')
    morphologies = np.zeros((num_labels, 3), dtype='float32')
    appearances = np.zeros((num_labels, appearance_dim,
                            appearance_dim, X.shape[-1]), dtype='float32')

    if crop_mode == 'fixed':
        # Zero-pad the X array for fixed crop mode
        pad_width = ((appearance_dim, appearance_dim),
                     (appearance_dim, appearance_dim),
                     (0, 0))
        X_padded = np.pad(X, pad_width=pad_width)
        y_padded = np.pad(y, pad_width=pad_width)

        props = regionprops(y_padded[..., 0], cache=False)

    # iterate over all objects in y
    if crop_mode == 'resize':
        props = regionprops(y[..., 0], cache=False)

    for i, prop in enumerate(props):

        # Get label
        labels[i] = prop.label

        # Get centroid
        centroid = np.array(prop.centroid)
        centroids[i] = centroid

        # Get morphology
        morphology = np.array([
            prop.area,
            prop.perimeter,
            prop.eccentricity
        ])
        morphologies[i] = morphology

        if crop_mode == 'resize':
            # Get appearance
            minr, minc, maxr, maxc = prop.bbox
            appearance = np.copy(X[minr:maxr, minc:maxc, :])
            resize_shape = (appearance_dim, appearance_dim)
            appearance = resize(appearance, resize_shape)
            appearances[i] = appearance

        if crop_mode == 'fixed':
            cent = np.array(prop.centroid)
            delta = appearance_dim // 2
            minr = int(cent[0]) - delta
            maxr = int(cent[0]) + delta
            minc = int(cent[1]) - delta
            maxc = int(cent[1]) + delta

            app = np.copy(X_padded[minr:maxr, minc:maxc, :])
            label = np.copy(y_padded[minr:maxr, minc:maxc])

            if norm:
                # Use label as a mask to zero out non-label information
                app = app * (label == prop.label)
                idx = np.nonzero(app)

                # Check data and normalize
                if len(idx) > 0:
                    masked_app = app[idx]
                    mean = np.mean(masked_app)
                    std = np.std(masked_app)
                    app[idx] = (masked_app - mean) / (std + 1e-4)

            appearances[i] = app

    return {
        'appearances': appearances,
        'centroids': centroids,
        'labels': labels,
        'morphologies': morphologies,
    }

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
        cells = np.max(labels[frame])
        if cells > max_cells:
            max_cells = cells
    return max_cells+1

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
               mode: str = 'test',
               device=None
               ) -> Dict[str, torch.Tensor]:
    """Convert numpy arrays to PyTorch tensors."""
    tensors = {}
    
    # Appearances: (T, N, H, W, C)

    if mode=='test':    
        tensors['appearances'] = torch.from_numpy(data['appearances']).float().to(device)
        tensors['morphologies'] = torch.from_numpy(data['morphologies']).float().to(device)
        tensors['centroids'] = torch.from_numpy(data['centroids']).float().to(device)
        tensors['adj_matrix'] = torch.from_numpy(data['adj_matrix']).float().to(device)
        tensors['labels'] = torch.from_numpy(data['labels']).float().to(device)
    
    else:
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

def relabel_sequential_lineage(y, lineage):
    """Ensure the lineage information is sequentially labeled.

    Args:
        y (np.array): Annotated z-stack of image labels.
        lineage (dict): Lineage data for y.

    Returns:
        tuple(np.array, dict): The relabeled array and corrected lineage.
    """


    y_relabel, fw, _ = relabel_sequential(y)

    new_lineage = {}

    cell_ids = np.unique(y)
    cell_ids = cell_ids[cell_ids != 0]
    cell_ids = cell_ids.tolist()
    
    for cell_id in cell_ids:
        
        new_cell_id = int(fw[cell_id])

        new_lineage[new_cell_id] = {}

        # Fix label
        # TODO: label == track ID?
        new_lineage[new_cell_id]['label'] = new_cell_id
        cell_id = str(cell_id)
        # Fix parent
        parent = lineage[cell_id]['parent']
        new_parent = int(fw[parent]) if parent is not None else parent
        new_lineage[new_cell_id]['parent'] = new_parent

        # Fix daughters
        daughters = lineage[cell_id]['daughters']
        new_lineage[new_cell_id]['daughters'] = []
        for d in daughters:
            new_daughter = int(fw[d])
            if not new_daughter:  # missing labels get mapped to 0
                warnings.warn('Cell {} has daughter {} which is not found '
                              'in the label image `y`.'.format(cell_id, d))
            else:
                new_lineage[new_cell_id]['daughters'].append(new_daughter)
                
        # Fix frames
        y_true = np.any(y == np.int64(cell_id), axis=(1, 2))
        y_index = np.nonzero(y_true)[0]

        new_lineage[new_cell_id]['frames'] = y_index.tolist()

    return y_relabel, new_lineage