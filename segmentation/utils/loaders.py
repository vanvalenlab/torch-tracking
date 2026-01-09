from torch.utils.data import Dataset
from utils.masks import _transform_masks
from utils.toolbox import histogram_normalization
import zarr

from torch.utils.data import DataLoader

from torchvision.transforms import v2 as transforms

import numpy as np

from utils.transforms import inner_distance_transform_2d, outer_distance_transform_2d

# Copied from keras
def to_categorical(x, num_classes=None, dtype="int64"):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with `categorical_crossentropy`.

    Args:
        x: Array-like with class values to be converted into a matrix
            (integers from 0 to `num_classes - 1`).
        num_classes: Total number of classes. If `None`, this would be inferred
            as `max(x) + 1`. Defaults to `None`.

    Returns:
        A binary matrix representation of the input as a NumPy array. The class
        axis is placed last.

    Example:

    >>> a = to_categorical([0, 1, 2, 3], num_classes=4)
    >>> print(a)
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]

    >>> b = np.array([.9, .04, .03, .03,
    ...               .3, .45, .15, .13,
    ...               .04, .01, .94, .05,
    ...               .12, .21, .5, .17],
    ...               shape=[4, 4])
    >>> loss = keras.ops.categorical_crossentropy(a, b)
    >>> print(np.around(loss, 5))
    [0.10536 0.82807 0.1011  1.77196]

    >>> loss = keras.ops.categorical_crossentropy(a, a)
    >>> print(np.around(loss, 5))
    [0. 0. 0. 0.]
    """
    x = np.array(x, dtype="int64")
    input_shape = x.shape

    # Shrink the last dimension if the shape is (..., 1).
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    x = x.reshape(-1)
    if not num_classes:
        num_classes = np.max(x) + 1
    batch_size = x.shape[0]
    categorical = np.zeros((batch_size, num_classes))
    categorical[np.arange(batch_size), x] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical.astype(dtype)


def _transform_masks(y, transform, data_format=None, mask_dtype=np.float32, **kwargs):
    """Based on the transform key, apply a transform function to the masks.

    Refer to :mod:`torch_mesmer.transform_utils` for more information about
    available transforms. Caution for unknown transform keys.

    Args:
        y (numpy.array): Labels of ``ndim`` 4 or 5
        transform (str): Name of the transform, one of
            ``{"deepcell", "disc", "watershed", None}``.
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
        kwargs (dict): Optional transform keyword arguments.

    Returns:
        numpy.array: the output of the given transform function on ``y``.

    Raises:
        ValueError: Rank of ``y`` is not 4 or 5.
        ValueError: Channel dimension of ``y`` is not 1.
        ValueError: ``transform`` is invalid value.
    """
    valid_transforms = {
        'deepcell',  # deprecated for "pixelwise"
        'pixelwise',
        'disc',
        'watershed',  # deprecated for "outer-distance"
        'watershed-cont',  # deprecated for "outer-distance"
        'inner-distance', 'inner_distance',
        'outer-distance', 'outer_distance',
        'centroid',  # deprecated for "inner-distance"
        'fgbg'
    }
    if data_format is None:
        data_format = "channels_last"

    if y.ndim not in {4, 5}:
        raise ValueError('`labels` data must be of ndim 4 or 5.  Got', y.ndim)

    channel_axis = 1 if data_format == 'channels_first' else -1

    if y.shape[channel_axis] != 1:
        raise ValueError('Expected channel axis to be 1 dimension. Got',
                         y.shape[1 if data_format == 'channels_first' else -1])

    if isinstance(transform, str):
        transform = transform.lower()

    if transform not in valid_transforms and transform is not None:
        raise ValueError(f'`{transform}` is not a valid transform')

    elif transform in {'outer-distance', 'outer_distance'}:

        distance_kwargs = {
            'erosion_width': kwargs.pop('erosion_width', 0),
        }

        if data_format == 'channels_first':
            shape = tuple([y.shape[0]] + list(y.shape[2:]))
        else:
            shape = y.shape[0:-1]

        y_transform = np.zeros(shape, dtype=mask_dtype)

        _distance_transform = outer_distance_transform_2d

        for batch in range(y_transform.shape[0]):
            if data_format == 'channels_first':
                mask = y[batch, 0, ...]
            else:
                mask = y[batch, ..., 0]
            y_transform[batch] = _distance_transform(mask, **distance_kwargs)

        y_transform = np.expand_dims(y_transform, axis=-1)

        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, y.ndim - 1, 1)

    elif transform in {'inner-distance', 'inner_distance'}:

        distance_kwargs = {
            'erosion_width': kwargs.pop('erosion_width', 0),
            'alpha': kwargs.pop('alpha', 0.1),
            'beta': kwargs.pop('beta', 1)
        }

        if data_format == 'channels_first':
            shape = tuple([y.shape[0]] + list(y.shape[2:]))
        else:
            shape = y.shape[0:-1]

        y_transform = np.zeros(shape, dtype=mask_dtype)

        _distance_transform = inner_distance_transform_2d

        for batch in range(y_transform.shape[0]):
            if data_format == 'channels_first':
                mask = y[batch, 0, ...]
            else:
                mask = y[batch, ..., 0]
            y_transform[batch] = _distance_transform(mask, **distance_kwargs)

        y_transform = np.expand_dims(y_transform, axis=-1)

        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, y.ndim - 1, 1)

    elif transform == 'fgbg':

        y_transform = np.where(y > 1, 1, y)

        # convert to one hot notation
        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, 1, y.ndim)
        
        # using uint8 since should only be 2 unique values.
        y_transform = to_categorical(y_transform, dtype=np.uint8)

        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, y.ndim - 1, 1)

    return y_transform


def create_data_loaders(
    train,
    val,
    crop_size=256,
    zoom_min=0.75,
    batch_size=16,
    outer_erosion_width=1,
    inner_distance_alpha="auto",
    inner_distance_beta=1,
    inner_erosion_width=0,
    num_workers=4
):
    
    rotation_range = 180
    zoom_range = (zoom_min, 1/zoom_min)

    in_transforms = ["inner-distance", "outer-distance", "fgbg"]

    transforms_kwargs = {

        "outer-distance": {
            "erosion_width": outer_erosion_width
            },

        "inner-distance": {
            "alpha": inner_distance_alpha,
            "beta": inner_distance_beta,
            "erosion_width": inner_erosion_width,
            },

    }

    train_transforms = transforms.Compose([
        transforms.ToImage(),
        transforms.RandomCrop(crop_size),
        transforms.RandomRotation(degrees=rotation_range),
        transforms.RandomResizedCrop(size=256, scale=zoom_range),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5)
        ])
    
    val_transforms = transforms.Compose([
            transforms.ToImage(),
        ])

    train_dataset = SegmentationDataset(
        train['X'], 
        train['y'],
        in_transforms=in_transforms, 
        out_transforms=train_transforms,
        transforms_kwargs=transforms_kwargs)
    
    val_dataset = SegmentationDataset(
        val['X'], 
        val['y'], 
        in_transforms=in_transforms, 
        out_transforms=val_transforms,
        transforms_kwargs=transforms_kwargs)  
      
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return dataloader, valloader

class SegmentationDataset(Dataset):
    def __init__(self, X, y, in_transforms=['outer-distance'], transforms_kwargs={}, out_transforms=None):
        self.X = X
        self.y = y
        self.in_transforms = in_transforms
        self.out_transforms = out_transforms
        self.transforms_kwargs = transforms_kwargs
        self.channel_axis=-1

    def _transform_labels(self, y):
        y_semantic_list = []
        # loop over channels axis of labels in case there are multiple label types
        for label_num in range(y.shape[self.channel_axis]):
    
            if self.channel_axis == 1:
                y_current = y[:, label_num:label_num + 1, ...]
            else:
                y_current = y[..., label_num:label_num + 1]

            data_format='channels_last'
            for transform in self.in_transforms:
                transform_kwargs = self.transforms_kwargs.get(transform, dict())

                y_transform = _transform_masks(y_current, transform,
                                               data_format=data_format,
                                               **transform_kwargs)
                y_semantic_list.append(y_transform)

        y_semantic_list = [ys[0] for ys in y_semantic_list]
        return y_semantic_list

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):

        x = self.X[idx]
        x = histogram_normalization(x)
        y_semantic_list = self._transform_labels(self.y[idx:idx+1]) 

        if self.out_transforms:
            x, y_semantic_list = self.out_transforms(x, y_semantic_list)


        return (x, y_semantic_list)