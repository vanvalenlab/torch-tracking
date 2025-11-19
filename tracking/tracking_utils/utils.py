import numpy as np
import warnings
from skimage.segmentation import relabel_sequential
from skimage.measure import regionprops
import torch
import cv2
from skimage import transform

def normalize_adjacency_symmetric(
    adj: torch.Tensor,
    add_self_loops: bool = True,
    eps: float = 1e-12
) -> torch.Tensor:
    """Symmetric normalization: D^(-1/2) @ A @ D^(-1/2)
    
    This is the most common normalization for GCNs (Kipf & Welling, 2017).
    Results in normalized adjacency where each edge is weighted by the inverse
    square root of the product of node degrees.
    
    Formula: Ã = D^(-1/2) @ A @ D^(-1/2)
    where D is the degree matrix and A is the adjacency matrix.
    
    Args:
        adj: Adjacency matrices of shape (B, T, N, N)
        add_self_loops: If True, adds self-connections before normalization
        eps: Small constant to avoid division by zero
    
    Returns:
        Normalized adjacency matrices of shape (B, T, N, N)
    
    Example:
        >>> adj = torch.rand(2, 8, 39, 39) > 0.5
        >>> adj = adj.float()
        >>> adj_norm = normalize_adjacency_symmetric(adj)
    """
    with torch.no_grad():
        B, T, N, _ = adj.shape
        
        # Add self-loops: A + I
        if add_self_loops:
            identity = torch.eye(N, device=adj.device, dtype=adj.dtype)
            identity = identity.view(1, 1, N, N).expand(B, T, N, N)
            adj = adj + identity
        
        # Compute degree matrix: D[i,i] = sum_j A[i,j]
        degree = adj.sum(dim=-1)  # (B, T, N)
        
        # D^(-1/2)
        degree_inv_sqrt = degree.pow(-0.5)
        degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0.0
        degree_inv_sqrt = torch.clamp(degree_inv_sqrt, min=0.0, max=1e10)
        
        # Create diagonal matrix from degree^(-1/2)
        # D^(-1/2) @ A @ D^(-1/2)
        adj_norm = degree_inv_sqrt.unsqueeze(-1) * adj * degree_inv_sqrt.unsqueeze(-2)
        
        return adj_norm

def normalize_adj_matrix(adj, epsilon=1e-5):
    """Normalize the adjacency matrix

    Args:
        adj (np.array): Adjacency matrix
        epsilon (float): Used to create the degree matrix

    Returns:
        np.array: Normalized adjacency matrix

    Raises:
        ValueError: If ``adj`` has a rank that is not 3 or 4.
    """
    input_rank = len(adj.shape)
    if input_rank not in {3, 4}:
        raise ValueError('Only 3 & 4 dim adjacency matrices are supported')

    if input_rank == 3:
        # temporarily include a batch dimension for consistent processing
        adj = np.expand_dims(adj, axis=0)

    normalized_adj = np.zeros(adj.shape, dtype='float32')

    for t in range(adj.shape[1]):
        adj_frame = adj[:, t]
        # create degree matrix
        degrees = np.sum(adj_frame, axis=1)
        for batch, degree in enumerate(degrees):
            degree = (degree + epsilon) ** -0.5
            degree_matrix = np.diagflat(degree)

            normalized = np.matmul(degree_matrix, adj_frame[batch])
            normalized = np.matmul(normalized, degree_matrix)
            normalized_adj[batch, t] = normalized

    if input_rank == 3:
        # remove batch axis
        normalized_adj = normalized_adj[0]

    return normalized_adj

def is_valid_lineage(y, lineage):
    """Check if a cell lineage of a single movie is valid.

    Daughter cells must exist in the frame after the parent's final frame.

    Args:
        y (numpy.array): The 3D label mask.
        lineage (dict): The cell lineages for a single movie.

    Returns:
        bool: Whether or not the lineage is valid.
    """
    all_cells = np.unique(y)
    all_cells = set([c for c in all_cells if c])

    is_valid = True

    # every lineage should have valid fields
    for cell_label, cell_lineage in lineage.items():
        cell_label = int(cell_label)
        # Get last frame of parent
        if cell_label not in all_cells:
            warnings.warn('Cell {} not found in the label image.'.format(
                cell_label))
            is_valid = False
            continue

        # any cells leftover are missing lineage
        all_cells.remove(cell_label)

        # validate `frames`
        y_true = np.any(y == cell_label, axis=(1, 2))
        y_index = y_true.nonzero()[0]
        frames = list(y_index)
        if frames != cell_lineage['frames']:
            warnings.warn('Cell {} has invalid frames'.format(cell_label))
            is_valid = False
            continue  # no need to test further

        last_parent_frame = cell_lineage['frames'][-1]

        for daughter in cell_lineage['daughters']:
            daughter=str(daughter)
            if daughter not in lineage:
                warnings.warn('Lineage {} has daughter {} not in lineage'.format(
                    cell_label, daughter))
                is_valid = False
                continue  # no need to test further

            # get first frame of daughter
            try:
                first_daughter_frame = lineage[daughter]['frames'][0]
            except IndexError:  # frames is empty?
                warnings.warn('Daughter {} has no frames'.format(daughter))
                is_valid = False
                continue  # no need to test further

            # Check that daughter's start frame is one larger than parent end frame
            if first_daughter_frame - last_parent_frame != 1:
                warnings.warn('Lineage {} has daughter {} in a '
                              'non-subsequent frame.'.format(
                                  cell_label, daughter))
                is_valid = False
                continue  # no need to test further

        # TODO: test parent in lineage
        parent = cell_lineage.get('parent')
        if parent:
            parent = str(parent)
            try:
                parent_lineage = lineage[parent]
            except KeyError:
                warnings.warn('Parent {} is not present in the lineage'.format(
                    cell_lineage['parent']))
                is_valid = False
                continue  # no need to test further
            try:
                last_parent_frame = parent_lineage['frames'][-1]
                first_daughter_frame = cell_lineage['frames'][0]
            except IndexError:  # frames is empty?
                warnings.warn('Cell {} has no frames'.format(parent))
                is_valid = False
                continue  # no need to test further
            # Check that daughter's start frame is one larger than parent end frame
            if first_daughter_frame - last_parent_frame != 1:
                warnings.warn(
                    'Cell {} ends in frame {} but daughter {} first '
                    'appears in frame {}.'.format(
                        parent, last_parent_frame, cell_label,
                        first_daughter_frame))
                is_valid = False
                continue  # no need to test further

    if all_cells:  # all cells with lineages should be removed
        warnings.warn('Cells missing their lineage: {}'.format(
            list(all_cells)))
        is_valid = False

    return is_valid  # if unchanged, all cell lineages are valid!



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


def get_max_cells(y):
    """Helper function for finding the maximum number of cells in a frame of a movie, across
    all frames of the movie. Can be used for batches/tracks interchangeably with frames/cells.

    Args:
        y (np.array): Annotated image data

    Returns:
        int: The maximum number of cells in any frame
    """
    max_cells = 0
    for frame in range(y.shape[0]):
        cells = np.unique(y[frame])
        n_cells = cells[cells != 0].shape[0]
        if n_cells > max_cells:
            max_cells = n_cells
    return max_cells

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


def resize(data, shape, data_format='channels_last', labeled_image=False):
    """Resize the data to the given shape.
    Uses openCV to resize the data if the data is a single channel, as it
    is very fast. However, openCV does not support multi-channel resizing,
    so if the data has multiple channels, use skimage.

    Args:
        data (np.array): data to be reshaped. Must have a channel dimension
        shape (tuple): shape of the output data in the form (x,y).
            Batch and channel dimensions are handled automatically and preserved.
        data_format (str): determines the order of the channel axis,
            one of 'channels_first' and 'channels_last'.
        labeled_image (bool): flag to determine how interpolation and floats are handled based
         on whether the data represents raw images or annotations

    Raises:
        ValueError: ndim of data not 3 or 4
        ValueError: Shape for resize can only have length of 2, e.g. (x,y)

    Returns:
        numpy.array: data reshaped to new shape.
    """
    if len(data.shape) not in {3, 4}:
        raise ValueError('Data must have 3 or 4 dimensions, e.g. '
                         '[batch, x, y], [x, y, channel] or '
                         '[batch, x, y, channel]. Input data only has {} '
                         'dimensions.'.format(len(data.shape)))

    if len(shape) != 2:
        raise ValueError('Shape for resize can only have length of 2, e.g. (x,y).'
                         'Input shape has {} dimensions.'.format(len(shape)))

    original_dtype = data.dtype

    # cv2 resize is faster but does not support multi-channel data
    # If the data is multi-channel, use skimage.transform.resize
    channel_axis = 0 if data_format == 'channels_first' else -1
    batch_axis = -1 if data_format == 'channels_first' else 0

    # Use skimage for multichannel data
    if data.shape[channel_axis] > 1:
        # Adjust output shape to account for channel axis
        if data_format == 'channels_first':
            shape = tuple([data.shape[channel_axis]] + list(shape))
        else:
            shape = tuple(list(shape) + [data.shape[channel_axis]])

        # linear interpolation (order 1) for image data, nearest neighbor (order 0) for labels
        # anti_aliasing introduces spurious labels, include only for image data
        order = 0 if labeled_image else 1
        anti_aliasing = not labeled_image

        _resize = lambda d: transform.resize(d, shape, mode='constant', preserve_range=True,
                                             order=order, anti_aliasing=anti_aliasing)
    # single channel image, resize with cv2
    else:
        shape = tuple(shape)[::-1]  # cv2 expects swapped axes.

        # linear interpolation for image data, nearest neighbor for labels
        # CV2 doesn't support ints for linear interpolation, set to float for image data
        if labeled_image:
            interpolation = cv2.INTER_NEAREST
        else:
            interpolation = cv2.INTER_LINEAR
            data = data.astype('float32')

        _resize = lambda d: np.expand_dims(cv2.resize(np.squeeze(d), shape,
                                                      interpolation=interpolation),
                                           axis=channel_axis)

    # Check for batch dimension to loop over
    if len(data.shape) == 4:
        batch = []
        for i in range(data.shape[batch_axis]):
            d = data[i] if batch_axis == 0 else data[..., i]
            batch.append(_resize(d))
        resized = np.stack(batch, axis=batch_axis)
    else:
        resized = _resize(data)

    return resized.astype(original_dtype)
