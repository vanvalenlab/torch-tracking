import numpy as np
import warnings
import skimage
from skimage.segmentation import relabel_sequential
from skimage.measure import regionprops
import torch
import cv2
from skimage import transform
import pandas as pd
import networkx as nx
from tqdm import tqdm
import torch.optim as optim
from skimage.measure import regionprops

def histogram_normalization(image: np.typing.ArrayLike, kernel_size=None, data_format = 'channels_last'):
    """Pre-process images using Contrast Limited Adaptive Histogram Equalization (CLAHE).

    If one of the inputs is a constant-value array, it will be normalized
    as an array of all zeros of the same shape.

    Parameters
    ----------
    image : numpy.typing.ArrayLike
        Array of phase image data with shape ``(batch, x, y, channel)``
        (or ``(batch, channel, x, y)`` if ``data_format`` is
        ``channels_first``).
    kernel_size : int, optional
        Size of kernel for CLAHE. Defaults to 1/8 of image size.
    data_format : str, optional
        Order of the channel axis, one of ``channels_first`` or
        ``channels_last``. Default is ``channels_last``.

    Returns
    -------
    numpy.ndarray
        Pre-processed image data with dtype ``float32``.
    """

    image = image.astype('float32')

    if data_format == 'channels_first':
        image = np.moveaxis(image, 1, -1)
    
    pbar = tqdm(range(image.shape[0]), leave=False)

    for batch in pbar:
        for channel in range(image.shape[-1]):
            X = image[batch, ..., channel]
            sample_value = X[(0,) * X.ndim]
            if (X == sample_value).all():
                # TODO: Deal with constant value arrays
                # https://github.com/scikit-image/scikit-image/issues/4596
                image[batch, ..., channel] = np.zeros_like(X)
                continue

            # X = rescale_intensity(X, out_range='float')
            X = skimage.exposure.rescale_intensity(X, out_range=(0.0, 1.0))
            X = skimage.exposure.equalize_adapthist(X, kernel_size=kernel_size)
            image[batch, ..., channel] = X
            
    if data_format == 'channels_first':
        image = np.moveaxis(image, -1, 1)

    return image

def weighted_categorical_crossentropy(y_true, y_pred,
                                      n_classes=3, axis=None,
                                      from_logits=False):
    """Compute weighted categorical crossentropy between two tensors.

    Automatically computes the class weights from the target tensor and
    uses them to weight the cross entropy.

    Parameters
    ----------
    y_true : torch.Tensor
        A tensor of the same shape as ``y_pred``.
    y_pred : torch.Tensor
        A tensor resulting from a softmax (unless ``from_logits`` is
        ``True``, in which case ``y_pred`` is expected to be the logits).
    n_classes : int, optional
        Number of classes used to scale the computed class weights.
        Default is 3.
    axis : int or None, optional
        Axis along which the class probabilities sum to 1. Default is
        ``None``, which resolves to the last axis.
    from_logits : bool, optional
        Whether ``y_pred`` is the result of a softmax, or is a tensor of
        logits. Default is False.

    Returns
    -------
    torch.Tensor
        Elementwise weighted crossentropy, with the same shape as
        ``y_true`` and ``y_pred``.

    Raises
    ------
    Exception
        If ``from_logits`` is True, since logits are not supported.
    """
    
    if from_logits:
        raise Exception('weighted_categorical_crossentropy cannot take logits')
    
    if axis is None:
        axis = -1 # if K.image_data_format() == 'channels_first' else K.ndim(y_pred) - 1

    reduce_axis = [x for x in list(range(torch.Tensor.dim(y_pred))) if x != axis]

    # scale preds so that the class probas of each sample sum to 1
    y_pred = y_pred / torch.sum(y_pred, dim=axis, keepdims=True)

    # manual computation of crossentropy
    eps=1e-10
    _epsilon = torch.tensor(eps).type(y_pred.dtype).to(y_pred.device)

    y_pred = torch.clamp(y_pred, min=_epsilon, max=(1. - _epsilon))

    total_sum = torch.sum(y_true)
    class_sum = torch.sum(y_true, dim=reduce_axis, keepdims=True)
    class_weights = 1.0 / n_classes * torch.divide(total_sum, class_sum + 1.)

    return - (y_true * torch.log(y_pred) * class_weights)


def weighted_categorical_crossentropy_v2(
        y_true, 
        y_pred, 
        n_classes=3, 
    ):

    """Compute weighted categorical crossentropy, weighting by inverse class frequency.

    Parameters
    ----------
    y_true : torch.Tensor
        One-hot encoded targets of shape ``(N, C)``.
    y_pred : torch.Tensor
        Predicted class probabilities of shape ``(N, C)``.
    n_classes : int, optional
        Number of classes used to scale the computed class weights.
        Default is 3.

    Returns
    -------
    torch.Tensor
        Elementwise weighted crossentropy, with the same shape as
        ``y_true`` and ``y_pred``.
    """
    eps = 1e-10
    _epsilon = torch.tensor(eps).type(y_pred.dtype).to(y_pred.device)
    
    y_pred = y_pred / torch.sum(y_pred, dim=-1, keepdims=True)
    
    # Clamp predictions to avoid log(0)
    y_pred = torch.clamp(y_pred, min=_epsilon, max=(1. - _epsilon))
        
    # Total number of valid (non-masked) samples across all classes
    total_sum = torch.sum(y_true)
    
    # Sum per class across all samples (N dimension)
    class_sum = torch.sum(y_true, dim=0, keepdims=True)
    
    # Compute weights: inverse frequency normalized by n_classes
    class_weights = 1.0 / n_classes * torch.divide(total_sum, class_sum + 1.).to(y_pred.device)
        
    return - (y_true * torch.log(y_pred) * class_weights)

def normalize_adjacency_symmetric(
    adj: torch.Tensor,
    add_self_loops: bool = True,
    eps: float = 1e-12
) -> torch.Tensor:
    """Apply symmetric normalization to an adjacency matrix: D^(-1/2) @ A @ D^(-1/2).

    This is the most common normalization for GCNs (Kipf & Welling, 2017).
    Results in a normalized adjacency where each edge is weighted by the
    inverse square root of the product of node degrees, where ``D`` is
    the degree matrix and ``A`` is the adjacency matrix.

    Parameters
    ----------
    adj : torch.Tensor
        Adjacency matrices of shape ``(T, N, N)`` or, if batched,
        ``(B, T, N, N)``.
    add_self_loops : bool, optional
        If True, adds self-connections before normalization. Default is
        True.
    eps : float, optional
        Small constant to avoid division by zero. Default is 1e-12.

    Returns
    -------
    torch.Tensor
        Normalized adjacency matrices with the same shape as ``adj``.

    Examples
    --------
    >>> adj = torch.rand(2, 8, 39, 39) > 0.5
    >>> adj = adj.float()
    >>> adj_norm = normalize_adjacency_symmetric(adj)
    """
    with torch.no_grad():

        if adj.ndim == 3:
            T, N, _ = adj.shape
        if adj.ndim == 4:
            # batched
            B, T, N, _ = adj.shape
        
        # Add self-loops: A + I
        if add_self_loops:
            identity = torch.eye(N, device=adj.device, dtype=adj.dtype)
            identity = identity.view(1, N, N).expand(T, N, N)
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

def is_valid_lineage(y, lineage):
    """Check if a cell lineage of a single movie is valid.

    Daughter cells must exist in the frame after the parent's final frame.

    Parameters
    ----------
    y : numpy.ndarray
        The 3D label mask.
    lineage : dict
        The cell lineages for a single movie.

    Returns
    -------
    bool
        Whether or not the lineage is valid.
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

    Parameters
    ----------
    y : numpy.ndarray
        Annotated z-stack of image labels.
    lineage : dict
        Lineage data for ``y``.

    Returns
    -------
    tuple of (numpy.ndarray, dict)
        The relabeled array and the corrected lineage.
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
    """Find the maximum number of cells in any frame of a movie.

    Can be used for batches/tracks interchangeably with frames/cells.

    Parameters
    ----------
    y : numpy.ndarray
        Annotated image data of shape ``(frames, x, y)``.

    Returns
    -------
    int
        The maximum number of cells in any frame.
    """
    max_cells = 0
    for frame in range(y.shape[0]):
        cells = np.unique(y[frame])
        n_cells = cells[cells != 0].shape[0]
        if n_cells > max_cells:
            max_cells = n_cells
    return max_cells

def get_image_features(X, y, appearance_dim=32, crop_mode='fixed', norm=True):
    """Return appearance, centroid, morphology, and label features for every object.

    Parameters
    ----------
    X : numpy.ndarray
        A 3D array of raw image data of shape ``(x, y, c)``.
    y : numpy.ndarray
        A 3D array of integer labels of shape ``(x, y, 1)``.
    appearance_dim : int, optional
        The resized shape of the appearance feature. Default is 32.
    crop_mode : {'fixed', 'resize'}, optional
        Whether to do a fixed crop or to crop and resize to create the
        appearance features. Default is ``'fixed'``.
    norm : bool, optional
        Whether to remove non-cell pixels and normalize the foreground
        pixels by zero-meaning and dividing by the standard deviation.
        Applies to fixed crop mode only. Default is True.

    Returns
    -------
    dict
        A dictionary with the following keys, each mapping to a
        ``numpy.ndarray`` with one entry per object (``n`` objects total):

        - ``appearances`` : shape ``(n, appearance_dim, appearance_dim, c)``.
        - ``centroids`` : shape ``(n, 2)``.
        - ``labels`` : shape ``(n,)``.
        - ``morphologies`` : shape ``(n, 3)``, containing area, perimeter,
          and eccentricity.

    Raises
    ------
    ValueError
        If ``crop_mode`` is not one of ``'resize'`` or ``'fixed'``.
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

            label = np.copy(y[minr:maxr, minc:maxc])

            appearance = appearance * (label == prop.label)
            appearances[i] = resize(appearance, resize_shape)

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

    Parameters
    ----------
    data : numpy.ndarray
        Data to be reshaped. Must have 3 or 4 dimensions and a channel
        dimension.
    shape : tuple of int
        Shape of the output data in the form ``(x, y)``. Batch and
        channel dimensions are handled automatically and preserved.
    data_format : str, optional
        Order of the channel axis, one of ``'channels_first'`` or
        ``'channels_last'``. Default is ``'channels_last'``.
    labeled_image : bool, optional
        Flag to determine how interpolation and floats are handled, based
        on whether the data represents raw images or annotations. Default
        is False.

    Returns
    -------
    numpy.ndarray
        Data reshaped to the new shape.

    Raises
    ------
    ValueError
        If ``data`` does not have 3 or 4 dimensions, or if ``shape`` does
        not have a length of 2.
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

def clean_up_annotations(y, uid=None, data_format='channels_last'):
    """Relabel every frame in the label matrix with globally unique cell IDs.

    Parameters
    ----------
    y : numpy.ndarray
        Annotations to relabel sequentially.
    uid : int or None, optional
        Starting ID to begin labeling cells. If ``None``, starts after the
        total number of unique cell labels across all frames.
    data_format : str, optional
        Order of the channel axis, one of ``'channels_first'`` or
        ``'channels_last'``. Default is ``'channels_last'``.

    Returns
    -------
    numpy.ndarray
        Cleaned up annotations, with dtype ``int32``.
    """
    y = y.astype('int32')
    time_axis = 1 if data_format == 'channels_first' else 0
    num_frames = y.shape[time_axis]

    all_uniques = []
    for f in range(num_frames):
        cells = np.unique(y[:, f] if data_format == 'channels_first' else y[f])
        cells = np.delete(cells, np.where(cells == 0))
        all_uniques.append(cells)

    # The annotations need to be unique across all frames
    uid = sum(len(x) for x in all_uniques) + 1 if uid is None else uid
    for frame, unique_cells in zip(range(num_frames), all_uniques):
        y_frame = y[:, frame] if data_format == 'channels_first' else y[frame]
        y_frame_new = np.zeros(y_frame.shape)
        for cell_label in unique_cells:
            y_frame_new[y_frame == cell_label] = uid
            uid += 1
        if data_format == 'channels_first':
            y[:, frame] = y_frame_new
        else:
            y[frame] = y_frame_new
    return y

def trk_to_graph(lineage, node_key=None):
    """Convert a lineage dictionary into a graph representation of the lineages.

    Parameters
    ----------
    lineage : dict
        Dictionary of lineage data.
    node_key : dict or None, optional
        Map between ground-truth node IDs and result node IDs. Default is
        ``None``, which uses the lineage's own IDs unchanged.

    Returns
    -------
    networkx.DiGraph
        Directed graph representation of the lineage data, with one node
        per ``(cell_id, frame)`` pair and division nodes flagged via the
        ``division`` node attribute.
    """
    edges = []

    all_ids = set()
    single_nodes = set()
    attributes = {}

    for i, lin in lineage.items():
        # Update cell id if node_key is available
        if node_key and (i in node_key):
            idx = node_key[i]
        else:
            idx = i

        cellids = ['{}_{}'.format(idx, t) for t in lin['frames']]

        if len(cellids) == 1:
            single_nodes.add(cellids[0])

        all_ids.update(cellids)
        edges.append(pd.DataFrame({
            'source': cellids[0:-1],
            'target': cellids[1:]
        }))

        # Add connections to any daughters
        source = '{}_{}'.format(idx, max(lin['frames']))
        for d in lin['daughters']:
            # Update cell id if node_key is available
            if node_key and (i in node_key):
                d_idx = node_key[d]
            else:
                d_idx = d

            # Assume daughter appears in next frame
            target = '{}_{}'.format(d_idx, max(lin['frames']) + 1)
            edges.append(pd.DataFrame({
                'source': [source],
                'target': [target]
            }))

            attributes[source] = {'division': True}

    # Create graph
    edges = pd.concat(edges)
    G = nx.from_pandas_edgelist(edges, source='source', target='target', create_using=nx.DiGraph)
    nx.set_node_attributes(G, attributes)

    # Add all isolates to graph
    for cell_id in single_nodes:
        G.add_node(cell_id)

    return G

class MetricsTracker:
    """Accumulate loss and classification metrics across batches.

    Tracks overall accuracy plus per-class recall, precision, and F1 for
    the 3 linkage classes (no-link, link, division).

    Returns
    -------
    MetricsTracker
        An initialized ``MetricsTracker`` object.
    """

    def __init__(self):
        self.reset()
        self.pad_value = -1

    def reset(self):
        """Reset all accumulated loss and classification counts.

        Returns
        -------
        None
        """
        self.total_loss = 0.0
        self.total_samples = 0
        self.correct = 0
        self.total_predictions = 0
        
        # Per-class metrics
        self.class_correct = {0: 0, 1: 0, 2: 0}
        self.class_total = {0: 0, 1: 0, 2: 0}
        self.class_predicted = {0: 0, 1: 0, 2: 0}

        self.cm = np.zeros((3,3))

    
    def update(self, loss, predictions, targets):
        """Accumulate loss and classification counts for a batch.

        Parameters
        ----------
        loss : torch.Tensor
            Scalar loss value for the batch.
        predictions : torch.Tensor
            Predicted class logits of shape ``(B, T, N, M, 3)``.
        targets : torch.Tensor
            Target class indices of shape ``(B, T, N, M)``. Entries less
            than 0 are treated as padding and excluded from the metrics.

        Returns
        -------
        None
        """

        batch_size = predictions.shape[0]
        self.total_loss += loss.item() * batch_size
        self.total_samples += batch_size

        # Get predicted classes
        pred_classes = predictions.argmax(dim=-1).view(-1)  # (B*T*N*M)
        target_classes = targets.view(-1)
        valid_mask = target_classes.view(-1) >= 0

        # Overall accuracy
        correct = (pred_classes == target_classes) & valid_mask
        self.correct += correct.sum().item()
        self.total_predictions += valid_mask.sum().item()
        
        # Per-class metrics
        for class_idx in range(3):
            class_mask = (target_classes == class_idx) & valid_mask
            class_predicted_mask = (pred_classes == class_idx) & valid_mask
            class_correct = (pred_classes == class_idx) & class_mask
            
            self.class_correct[class_idx] += class_correct.sum().item()
            self.class_total[class_idx] += class_mask.sum().item()
            self.class_predicted[class_idx] += class_predicted_mask.sum().item()
    
    def get_metrics(self):
        """Compute the current aggregate and per-class metrics.

        Returns
        -------
        dict
            Dictionary with keys ``loss``, ``accuracy``, and per-class
            ``recall_class_{i}``, ``precision_class_{i}``, and
            ``f1_class_{i}`` for each of the 3 classes.
        """
        avg_loss = self.total_loss / max(self.total_samples, 1)
        accuracy = self.correct / max(self.total_predictions, 1)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy
        }

        running_f1 = 0
        
        # Add per-class metrics
        for class_idx in range(3):
            # Recall: of all true class_idx, how many did we predict correctly?
            recall = (self.class_correct[class_idx] / 
                     max(self.class_total[class_idx], 1))
            
            # Precision: of all predicted class_idx, how many were correct?
            precision = (self.class_correct[class_idx] / 
                        max(self.class_predicted[class_idx], 1))
            
            # F1 score
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            
            metrics[f'recall_class_{class_idx}'] = recall
            metrics[f'precision_class_{class_idx}'] = precision
            metrics[f'f1_class_{class_idx}'] = f1
            running_f1 *= f1
                
        return metrics
    
class EarlyStopping:
    """Track a metric across epochs and signal when training should stop.

    Parameters
    ----------
    patience : int, optional
        Number of epochs to wait for an improvement before stopping.
        Default is 10.
    min_delta : float, optional
        Minimum change in the tracked metric to qualify as an
        improvement. Default is 0.0.
    mode : {'min', 'max'}, optional
        Whether lower values are better (``'min'``, e.g. for loss) or
        higher values are better (``'max'``, e.g. for accuracy). Default
        is ``'min'``.

    Returns
    -------
    EarlyStopping
        An initialized ``EarlyStopping`` object.
    """

    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, metric_value):
        """Update the tracked best value and check the stopping condition.

        Parameters
        ----------
        metric_value : float
            The metric value observed for the current epoch.

        Returns
        -------
        bool
            ``True`` if training should stop (no improvement for
            ``patience`` consecutive calls), ``False`` otherwise.
        """
        if self.best_value is None:
            self.best_value = metric_value
            return False
        
        if self.mode == 'min':
            improved = metric_value < (self.best_value - self.min_delta)
        else:
            improved = metric_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = metric_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    

def create_optimizer(model, config):
    """Create an optimizer based on a training config.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose parameters will be optimized.
    config : object
        Config object with an ``optimizer`` attribute (one of
        ``'radam'`` or ``'adamw'``) and the ``learning_rate`` and
        ``weight_decay`` attributes required by the chosen optimizer.

    Returns
    -------
    torch.optim.Optimizer
        The constructed optimizer.

    Raises
    ------
    ValueError
        If ``config.optimizer`` is not a recognized optimizer name.
    """

    optimizer_name = config.optimizer
    
    if optimizer_name == 'radam':
        optimizer = optim.RAdam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            decoupled_weight_decay=False
        )

    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer, config):
    """Create a learning rate scheduler based on a training config.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer to schedule.
    config : object
        Config object with a ``scheduler`` attribute, one of
        ``'reduce_on_plateau'``, ``'cosine'``, ``'step'``, ``'exp'``, or
        ``'none'``, plus whichever of ``patience``, ``max_epochs``,
        ``step_size``, or ``decay`` are required by the chosen scheduler.

    Returns
    -------
    torch.optim.lr_scheduler.LRScheduler or None
        The constructed scheduler, or ``None`` if ``config.scheduler`` is
        ``'none'``.
    """

    scheduler_name = config.scheduler

    
    if scheduler_name == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            patience=config.patience
            )
        
    elif scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.max_epochs,
            eta_min=1e-6
        )

    elif scheduler_name == 'step':
        step_size = config.get('step_size', 30)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=config.gamma
        )

    elif scheduler_name == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.decay
        )

    elif scheduler_name == 'none':
        scheduler = None

        
    return scheduler
