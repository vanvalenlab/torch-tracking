import numpy as np
import warnings
import skimage
from skimage.segmentation import relabel_sequential
from skimage.measure import regionprops
import torch
import cv2
from skimage import transform
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm
import torch.optim as optim
from skimage.measure import regionprops

# The resolution, in microns per pixel, that the model is trained at. Every movie is
# rescaled to this resolution, in conversion and at inference alike.
MODEL_MPP = 0.55


# Pixel radius within which two cells are linked in the spatial graph, on the grid at
# MODEL_MPP. 89 px keeps the physical reach the model was originally tuned at: 64 px at the
# old effective 0.768 um/px (0.65**2 / 0.55) is 49.2 um, which is 89.4 px at 0.55 um/px.
# Derived rather than swept, because a retune needs a trained model (see
# correct-scale-convention design.md, Decision 4). Used by training, inference and
# evaluation alike so they cannot disagree.
DISTANCE_THRESHOLD = 89


def scale_factor(mpp, model_mpp=MODEL_MPP):
    """Factor by which a movie declared at ``mpp`` is rescaled to the model resolution.

    One output pixel must span ``model_mpp`` microns. A movie coarser than the model
    (larger ``mpp``) has fewer pixels per micron, so it is enlarged: the factor is
    ``mpp / model_mpp``. This is the single definition of the rule; conversion and
    inference both call it so the two cannot disagree.
    """
    # float64 whatever the caller's type: a float32 scalar would round differently from
    # the declared value and could size the same movie one pixel apart on the two sides.
    return float(mpp) / float(model_mpp)


def real_cell_mask(morphologies):
    """Boolean mask, ``True`` where a cell slot holds a real cell.

    ``morphologies`` is ``(..., 3)``, as an ``ndarray`` or a ``Tensor``; the result has
    the shape of ``morphologies`` without its last axis. A padding slot carries an all-zero
    morphology and a real cell has positive area, so a slot is real exactly when any of its
    morphology values is non-zero.

    This is the one definition of "padding" in the repository: the graph rule in
    ``spatial_adjacency`` and every normalization layer in the model use it, so they cannot
    drift apart. It is deliberately *not* read from the centroid. Padding sits at the
    origin in stored data, but training augmentation translates every centroid - padding
    rows included - by a random per-sample offset, so a padding slot's centroid can be
    anywhere.
    """
    return (morphologies != 0).any(-1)


def pair_masks(is_real):
    """Real-pair masks for the two kinds of delta the training branch builds.

    ``is_real`` is ``(B, T, N)``, from ``real_cell_mask``. Returns ``(within, across)``:

    * ``across``, ``(B, T-1, N, N)``: entry ``[b, t, i, j]`` is real only when slot ``i`` of
      frame ``t`` *and* slot ``j`` of frame ``t+1`` are both real. That is the pairing
      ``compute_deltas_across_frames`` and ``Comparison`` use (tracks of the current frame
      against candidates of the next), and the one the training targets use to decide
      which entries are graded, so the three agree by construction.
    * ``within``, ``(B, T, N)``: the delta of slot ``i`` from the previous frame to this
      one. It is real when the slot is real in both frames; the first frame has no
      predecessor, so there it is real when the slot itself is.
    """
    across = is_real[:, :-1, :, None] & is_real[:, 1:, None, :]
    within = is_real.clone()
    within[:, 1:] = is_real[:, 1:] & is_real[:, :-1]
    return within, across


def masked_apply(module, x, is_real, *, fallback_below=2):
    """Apply a batch-statistic module to the real rows of ``x`` only.

    ``x`` is ``(..., C)`` and ``is_real`` is the boolean mask of the leading axes. The
    real rows are gathered into ``(R, C)``, passed through ``module`` - normally an
    ``nn.BatchNorm1d`` - and scattered back into a zero tensor of ``x``'s shape, so padding
    rows come out exactly zero. Because the stock module sees only real rows, its mean,
    variance and running buffers are accumulated over real cells alone, with its momentum,
    ``num_batches_tracked`` and train/eval switch behaving exactly as they always did.

    Two degenerate cases, because ``nn.BatchNorm1d`` in train mode raises at exactly one
    row and a fixture can easily hit it:

    * **No real row.** The module is not called, so no running statistic moves, and the
      result is all zeros.
    * **Fewer than ``fallback_below`` real rows in train mode.** The module is applied to
      every row unmasked instead. Real training carries tens of real cells per window and
      never reaches this; it exists so a tiny batch gives a number rather than an error
      whose cause is three layers from its symptom.

    In eval mode the module uses its running buffers, which depend on no batch, so a single
    real row is simply normalized.
    """
    flat = x.reshape(-1, x.shape[-1])
    mask = is_real.reshape(-1)
    n_real = int(mask.sum())

    if n_real == 0:
        return torch.zeros_like(x)
    if module.training and n_real < fallback_below:
        return module(flat).reshape(x.shape)

    normalized = module(flat[mask])
    out = normalized.new_zeros(flat.shape[0], normalized.shape[-1])
    out[mask] = normalized
    return out.reshape(*x.shape[:-1], normalized.shape[-1])


def spatial_adjacency(centroids, morphologies, distance_threshold):
    """Boolean ``(N, N)`` spatial graph over one frame's cell slots.

    Two slots are linked when both are real cells (see ``real_cell_mask``) and their
    centroid distance is strictly greater than zero and strictly less than
    ``distance_threshold``.

    The single definition of the graph rule: the tracker and the training dataloader
    both call it, so training and inference build the same graph. Padding slots also sit
    at centroid (0, 0), so without the exclusion every real cell near the origin would be
    linked to every padding slot.
    """
    distance = cdist(centroids, centroids, metric='euclidean')
    adjacency = (distance > 0) & (distance < distance_threshold)
    is_real = real_cell_mask(np.asarray(morphologies))
    return adjacency & is_real[:, None] & is_real[None, :]


def count_nonempty_frames(movie):
    """Number of frames of ``movie`` holding any non-zero value.

    The single definition of a movie's extent, shared by evaluation and the training
    sampler so the two cannot disagree about which frames exist. The leading axis is
    time; every other axis, however many there are, is reduced - so the count does not
    depend on the channel axis being singleton. Trailing frames are assumed empty
    padding, so the count is also the exclusive end of the occupied frames.
    """
    movie = np.asarray(movie)
    return int(movie.reshape(movie.shape[0], -1).any(axis=1).sum())


def require_mpp(store, path):
    """Return a converted split's per-movie resolutions, failing if it holds none.

    A zarr written before per-movie resolution was persisted has no ``mpp`` array. It
    was converted under the superseded scale convention, so it must be regenerated;
    silently assuming ``MODEL_MPP`` would hide that and reproduce the mismatch.
    """
    if 'mpp' not in store:
        raise ValueError(
            f"{path} holds no 'mpp' array. It was converted before per-movie resolution "
            "was persisted, under the superseded scale convention, and must be "
            "regenerated with convert_trk_to_zarr. No default resolution is assumed."
        )
    return np.asarray(store['mpp'][:], dtype='float64')


#: Where a processed zarr keeps each movie's corrected lineage (a list, one per movie).
LINEAGES_KEY = 'lineages'


def require_lineages(store, path):
    """Return a processed zarr's per-movie lineages, failing if it carries none.

    The dataloader builds each sample's training targets from the lineage, so a zarr
    converted before the lineage travelled with it cannot drive training. It is rejected
    rather than served from a sibling ``<split>.json``, which nothing guarantees agrees
    with it.
    """
    if LINEAGES_KEY not in store.attrs:
        raise ValueError(
            f"{path} carries no '{LINEAGES_KEY}'. It was converted before the corrected "
            "lineage was stored in the processed zarr, and must be regenerated with "
            "convert_trk_to_zarr. No sibling .json is read in its place."
        )
    return store.attrs[LINEAGES_KEY]


def histogram_normalization(image: np.typing.ArrayLike, kernel_size=None, data_format = 'channels_last'):
    """Pre-process images using Contrast Limited Adaptive
    Histogram Equalization (CLAHE).

    If one of the inputs is a constant-value array, it will
    be normalized as an array of all zeros of the same shape.

    Args:
        image (numpy.array): numpy array of phase image data.
        kernel_size (integer): Size of kernel for CLAHE,
            defaults to 1/8 of image size.

    Returns:
        numpy.array: Pre-processed image data with dtype float32.
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
    """Categorical crossentropy between an output tensor and a target tensor.
    Automatically computes the class weights from the target image and uses
    them to weight the cross entropy

    Args:
        y_true: A tensor of the same shape as ``y_pred``.
        y_pred: A tensor resulting from a softmax
            (unless ``from_logits`` is ``True``, in which
            case ``y_pred`` is expected to be the logits).
        from_logits: Boolean, whether ``y_pred`` is the
            result of a softmax, or is a tensor of logits.

    Returns:
        tensor: Output tensor.
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

    """
    Weighted categorical cross-entropy with masking for padded values.
    
    Args:
        y_pred: Tensor of shape (N, C) - predictions for each sample and class
        y_true: Tensor of shape (N, C) - one-hot encoded targets
        mask: Tensor of shape (N, C) - 1 where included, 0 where excluded (padded)
        n_classes: Number of classes
        from_logits: Whether y_pred contains logits (True) or probabilities (False)
    
    Returns:
        Scalar loss value
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

def add_self_loops_dense(adj: torch.Tensor, is_real: torch.Tensor) -> torch.Tensor:
    """Give every *real* cell of a dense adjacency matrix a self-loop.

    Args:
        adj: Adjacency matrices of shape (B*T, N, N).
        is_real: Boolean mask of shape (B*T, N), ``True`` for a real cell and ``False``
            for a padding slot (see ``real_cell_mask``).

    Returns:
        The same matrices with a non-zero diagonal entry at every real cell and none at a
        padding slot, shape (B*T, N, N).

    This replaced `normalize_adjacency_symmetric`, which computed a full
    `D^-1/2 A D^-1/2` that the only caller then discarded: `NeighborhoodEncoder`
    converts the result to an edge list with `.nonzero()`, which keeps the sparsity
    pattern and throws away every weight. The sole surviving effect of that function was
    its `add_self_loops=True`, which is what this does directly. Divergence 60.

    The graph layers are configured with `add_self_loops=False`, so these are the only
    self-loops in the model, and dropping them would isolate every node. They are added
    to real cells only. A padding slot is given no edge at all, so it cannot exchange
    anything with a real cell and does not contribute an embedding the normalization
    layers downstream would otherwise pool. A node with no edges is safe: `GATConv`
    returns exactly zero for it.
    """
    with torch.no_grad():
        loops = torch.diag_embed(is_real.to(adj.dtype))
        return adj + loops

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

    # Integer keys are the in-memory convention; a lineage decoded from JSON arrives with
    # string keys. Normalize once here so every lookup below, including parent and
    # daughter references, resolves against the same key type.
    lineage = {int(k): v for k, v in lineage.items()}

    # every lineage should have valid fields
    for cell_label, cell_lineage in lineage.items():
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
            daughter = int(daughter)
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
            parent = int(parent)
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

    # Accept either key spelling: a lineage decoded from JSON is keyed by strings, one
    # produced by this package by integers. The output is always integer-keyed, so the
    # function can be applied to its own result.
    lineage = {int(k): v for k, v in lineage.items()}

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
        y_true = np.any(y == cell_id, axis=(1, 2))
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

def get_image_features(X, y, appearance_dim=32):
    """Return features for every object in the array.

    Appearance is what a cell looks like - its texture, shape and contrast - and
    deliberately not how big it is. Size is carried by the morphology vector, which is
    measured after the frame has been rescaled to the model resolution. Each cell's
    bounding box is therefore masked to that cell's own label (so neighbours do not leak
    in), normalized by the mean and standard deviation of its own non-zero pixels,
    padded to a square about the cell (so its proportions are kept), and resized to
    ``appearance_dim`` x ``appearance_dim``. Nothing is clipped and nothing is stretched.

    Normalizing before the resize means the statistics come from the cell's real pixels
    rather than from interpolated ones. A cell with no non-zero pixels is left as zeros.
    Centroids are in image coordinates, with no offset.

    Args:
        X (np.array): a 3D numpy array of raw data of shape (x, y, c).
        y (np.array): a 3D numpy array of integer labels of shape (x, y, 1).
        appearance_dim (int): The side length of the square appearance feature.

    Returns:
        dict: A dictionary of feature names to np.arrays of shape
            (n, c) or (n, x, y, c) where n is the number of objects.
    """
    X = X.astype('float32')
    y = y.astype('int32')

    appearance_dim = int(appearance_dim)

    # each feature will be ordered based on the label.
    # labels are also stored and can be fetched by index.
    #
    # Count distinct non-zero labels directly. Deriving this as
    # len(np.unique(y)) - 1 assumed every frame contains background, so a fully
    # tiled mask - a confluent monolayer filling the field of view - sized these
    # arrays one short and raised IndexError. Divergence 13.
    num_labels = len(np.unique(y[y != 0]))
    labels = np.zeros((num_labels,), dtype='int32')
    centroids = np.zeros((num_labels, 2), dtype='float32')
    morphologies = np.zeros((num_labels, 3), dtype='float32')
    appearances = np.zeros((num_labels, appearance_dim,
                            appearance_dim, X.shape[-1]), dtype='float32')

    for i, prop in enumerate(regionprops(y[..., 0], cache=False)):

        labels[i] = prop.label
        centroids[i] = np.array(prop.centroid)
        morphologies[i] = np.array([
            prop.area,
            prop.perimeter,
            prop.eccentricity
        ])

        # Crop to the bounding box and keep only this cell's own pixels.
        minr, minc, maxr, maxc = prop.bbox
        app = X[minr:maxr, minc:maxc, :] * (y[minr:maxr, minc:maxc, :] == prop.label)

        # Zero-mean, unit-spread over the cell's own non-zero pixels. The guard tests
        # the size of the index, not the arity of the index tuple (divergence 14): an
        # all-zero cell has nothing to normalize and stays zero, without a warning.
        idx = np.nonzero(app)
        if idx[0].size > 0:
            masked_app = app[idx]
            app[idx] = (masked_app - np.mean(masked_app)) / (np.std(masked_app) + 1e-4)

        # Pad the shorter side with background so the box is square about the cell.
        # Resizing a non-square box to a square would stretch the cell.
        height, width = app.shape[:2]
        side = max(height, width)
        pad_r, pad_c = side - height, side - width
        app = np.pad(app, ((pad_r // 2, pad_r - pad_r // 2),
                           (pad_c // 2, pad_c - pad_c // 2),
                           (0, 0)))

        if side == 1:
            # A single-pixel cell: the resize helper squeezes it to a scalar.
            appearances[i] = np.broadcast_to(app, appearances[i].shape)
        else:
            appearances[i] = resize(app, (appearance_dim, appearance_dim))

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

def clean_up_annotations(y, uid=None, data_format='channels_last'):
    """Relabels every frame in the label matrix.

    Args:
        y (np.array): annotations to relabel sequentially.
        uid (int, optional): starting ID to begin labeling cells.
        data_format (str): determines the order of the channel axis,
            one of 'channels_first' and 'channels_last'.

    Returns:
        np.array: Cleaned up annotations.
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

#: Which way is "better" for every metric `MetricsTracker.get_metrics` reports. This is the
#: single place the direction of improvement is decided: best-checkpoint selection, early
#: stopping and the metric-driven learning-rate schedule all read it through
#: `metric_direction`, so none of them can disagree about it.
METRIC_DIRECTIONS = {
    'loss': 'min',
    'accuracy': 'max',
    **{f'{kind}_class_{c}': 'max' for kind in ('precision', 'recall', 'f1') for c in range(3)},
}


def check_choice(setting, value, accepted):
    """Return `value` if it is one of `accepted`, else raise naming both.

    Used wherever a configuration setting selects among named options, so that a bad value
    fails where it is given instead of surfacing later as a missing attribute.
    """
    if value not in accepted:
        raise ValueError(
            f"Unknown {setting} {value!r}; accepted values: {', '.join(map(str, accepted))}"
        )
    return value


def metric_direction(metric_name):
    """'min' if a lower value of `metric_name` is better, 'max' if a higher one is."""
    check_choice('stopping_metric', metric_name, list(METRIC_DIRECTIONS))
    return METRIC_DIRECTIONS[metric_name]


class MetricsTracker:
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_loss = 0.0
        self.total_graded = 0
        self.correct = 0
        self.total_predictions = 0
        
        # Per-class metrics
        self.class_correct = {0: 0, 1: 0, 2: 0}
        self.class_total = {0: 0, 1: 0, 2: 0}
        self.class_predicted = {0: 0, 1: 0, 2: 0}

    def update(self, loss, predictions, targets):
        """
        Args:
            loss: scalar loss for the batch, averaged over its graded entries
            predictions: (B, T, N, M, 3) class probabilities
            targets: (B, T, N, M) integer class labels, -1 marking an ungraded entry
        """

        # Get predicted classes
        pred_classes = predictions.argmax(dim=-1).view(-1)  # (B*T*N*M)
        target_classes = targets.view(-1)
        valid_mask = target_classes.view(-1) >= 0

        # The batch loss is already a mean over graded entries, so weight it by how many
        # there were. Weighting by batch size would over-count a batch that graded little.
        n_graded = valid_mask.sum().item()
        self.total_loss += loss.item() * n_graded
        self.total_graded += n_graded

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
        """Compute and return current metrics."""
        avg_loss = self.total_loss / max(self.total_graded, 1)
        accuracy = self.correct / max(self.total_predictions, 1)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy
        }

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

        return metrics
    
class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        check_choice('early-stopping mode', mode, ('min', 'max'))
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, metric_value):
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
    

OPTIMIZERS = ('radam', 'adamw')
SCHEDULERS = ('reduce_on_plateau', 'cosine', 'step', 'exp', 'none')


def create_optimizer(model, config):
    """Create optimizer based on config."""

    optimizer_name = check_choice('optimizer', config.optimizer, OPTIMIZERS)
    
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
            weight_decay=config.weight_decay,
        )

    return optimizer


def create_scheduler(optimizer, config):
    """Create learning rate scheduler."""

    scheduler_name = check_choice('scheduler', config.scheduler, SCHEDULERS)

    if scheduler_name == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=metric_direction(config.stopping_metric),
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
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.step_gamma
        )

    elif scheduler_name == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.decay
        )

    else:  # 'none'
        scheduler = None

    return scheduler
