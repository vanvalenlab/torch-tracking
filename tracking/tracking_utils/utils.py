import numpy as np
import warnings
from skimage.segmentation import relabel_sequential
from skimage.measure import regionprops


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