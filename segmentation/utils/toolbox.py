"""Utility functions that may be used in other transforms."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import cv2
from scipy.signal import windows

from skimage import transform
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_holes
from skimage.segmentation import find_boundaries

import numpy as np
from scipy import ndimage
from skimage import morphology
from skimage import segmentation
from skimage.feature import peak_local_max
from skimage.exposure import equalize_adapthist, rescale_intensity

import torch
import torch.nn as nn

import warnings

import scipy.ndimage as nd

from skimage.morphology import remove_small_objects, h_maxima, disk, ball, square, cube, dilation
from skimage.segmentation import relabel_sequential


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


def tile_image(image, model_input_shape=(512, 512),
               stride_ratio=0.75, pad_mode='constant'):
    """
    Tile large image into many overlapping tiles of size "model_input_shape".

    Args:
        image (numpy.array): The image to tile, must be rank 4.
        model_input_shape (tuple): The input size of the model.
        stride_ratio (float): The stride expressed as a fraction of the tile size.
        pad_mode (str): Padding mode passed to ``np.pad``.

    Returns:
        tuple: (numpy.array, dict): A tuple consisting of an array of tiled
            images and a dictionary of tiling details (for use in un-tiling).

    Raises:
        ValueError: image is not rank 4.
    """
    if image.ndim != 4:
        raise ValueError('Expected image of rank 4, got {}'.format(image.ndim))

    image_size_x, image_size_y = image.shape[1:3]
    tile_size_x = model_input_shape[0]
    tile_size_y = model_input_shape[1]

    ceil = lambda x: int(np.ceil(x))
    round_to_even = lambda x: int(np.ceil(x / 2.0) * 2)

    stride_x = min(round_to_even(stride_ratio * tile_size_x), tile_size_x)
    stride_y = min(round_to_even(stride_ratio * tile_size_y), tile_size_y)

    rep_number_x = max(ceil((image_size_x - tile_size_x) / stride_x + 1), 1)
    rep_number_y = max(ceil((image_size_y - tile_size_y) / stride_y + 1), 1)
    new_batch_size = image.shape[0] * rep_number_x * rep_number_y

    tiles_shape = (new_batch_size, tile_size_x, tile_size_y, image.shape[3])
    tiles = np.zeros(tiles_shape, dtype=image.dtype)

    # Calculate overlap of last tile
    overlap_x = (tile_size_x + stride_x * (rep_number_x - 1)) - image_size_x
    overlap_y = (tile_size_y + stride_y * (rep_number_y - 1)) - image_size_y

    # Calculate padding needed to account for overlap and pad image accordingly
    pad_x = (int(np.ceil(overlap_x / 2)), int(np.floor(overlap_x / 2)))
    pad_y = (int(np.ceil(overlap_y / 2)), int(np.floor(overlap_y / 2)))
    pad_null = (0, 0)
    padding = (pad_null, pad_x, pad_y, pad_null)
    image = np.pad(image, padding, pad_mode)

    counter = 0
    batches = []
    x_starts = []
    x_ends = []
    y_starts = []
    y_ends = []
    overlaps_x = []
    overlaps_y = []

    for b in range(image.shape[0]):
        for i in range(rep_number_x):
            for j in range(rep_number_y):
                x_axis = 1
                y_axis = 2

                # Compute the start and end for each tile
                if i != rep_number_x - 1:  # not the last one
                    x_start, x_end = i * stride_x, i * stride_x + tile_size_x
                else:
                    x_start, x_end = image.shape[x_axis] - tile_size_x, image.shape[x_axis]

                if j != rep_number_y - 1:  # not the last one
                    y_start, y_end = j * stride_y, j * stride_y + tile_size_y
                else:
                    y_start, y_end = image.shape[y_axis] - tile_size_y, image.shape[y_axis]

                # Compute the overlaps for each tile
                if i == 0:
                    overlap_x = (0, tile_size_x - stride_x)
                elif i == rep_number_x - 2:
                    overlap_x = (tile_size_x - stride_x, tile_size_x - image.shape[x_axis] + x_end)
                elif i == rep_number_x - 1:
                    overlap_x = ((i - 1) * stride_x + tile_size_x - x_start, 0)
                else:
                    overlap_x = (tile_size_x - stride_x, tile_size_x - stride_x)

                if j == 0:
                    overlap_y = (0, tile_size_y - stride_y)
                elif j == rep_number_y - 2:
                    overlap_y = (tile_size_y - stride_y, tile_size_y - image.shape[y_axis] + y_end)
                elif j == rep_number_y - 1:
                    overlap_y = ((j - 1) * stride_y + tile_size_y - y_start, 0)
                else:
                    overlap_y = (tile_size_y - stride_y, tile_size_y - stride_y)

                tiles[counter] = image[b, x_start:x_end, y_start:y_end, :]
                batches.append(b)
                x_starts.append(x_start)
                x_ends.append(x_end)
                y_starts.append(y_start)
                y_ends.append(y_end)
                overlaps_x.append(overlap_x)
                overlaps_y.append(overlap_y)
                counter += 1

    tiles_info = {}
    tiles_info['batches'] = batches
    tiles_info['x_starts'] = x_starts
    tiles_info['x_ends'] = x_ends
    tiles_info['y_starts'] = y_starts
    tiles_info['y_ends'] = y_ends
    tiles_info['overlaps_x'] = overlaps_x
    tiles_info['overlaps_y'] = overlaps_y
    tiles_info['stride_x'] = stride_x
    tiles_info['stride_y'] = stride_y
    tiles_info['tile_size_x'] = tile_size_x
    tiles_info['tile_size_y'] = tile_size_y
    tiles_info['stride_ratio'] = stride_ratio
    tiles_info['image_shape'] = image.shape
    tiles_info['dtype'] = image.dtype
    tiles_info['pad_x'] = pad_x
    tiles_info['pad_y'] = pad_y

    return tiles, tiles_info


def spline_window(window_size, overlap_left, overlap_right, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """

    def _spline_window(w_size):
        intersection = int(w_size / 4)
        wind_outer = (abs(2 * (windows.triang(w_size))) ** power) / 2
        wind_outer[intersection:-intersection] = 0

        wind_inner = 1 - (abs(2 * (windows.triang(w_size) - 1)) ** power) / 2
        wind_inner[:intersection] = 0
        wind_inner[-intersection:] = 0

        wind = wind_inner + wind_outer
        wind = wind / np.amax(wind)
        return wind

    # Create the window for the left overlap
    if overlap_left > 0:
        window_size_l = 2 * overlap_left
        l_spline = _spline_window(window_size_l)[0:overlap_left]

    # Create the window for the right overlap
    if overlap_right > 0:
        window_size_r = 2 * overlap_right
        r_spline = _spline_window(window_size_r)[overlap_right:]

    # Put the two together
    window = np.ones((window_size,))
    if overlap_left > 0:
        window[0:overlap_left] = l_spline
    if overlap_right > 0:
        window[-overlap_right:] = r_spline

    return window


def window_2D(window_size, overlap_x=(32, 32), overlap_y=(32, 32), power=2):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    window_x = spline_window(window_size[0], overlap_x[0], overlap_x[1], power=power)
    window_y = spline_window(window_size[1], overlap_y[0], overlap_y[1], power=power)

    window_x = np.expand_dims(np.expand_dims(window_x, -1), -1)
    window_y = np.expand_dims(np.expand_dims(window_y, -1), -1)

    window = window_x * window_y.transpose(1, 0, 2)
    return window


def untile_image(tiles, tiles_info, power=2, **kwargs):
    """Untile a set of tiled images back to the original model shape.

     Args:
         tiles (numpy.array): The tiled images image to untile.
         tiles_info (dict): Details of how the image was tiled (from tile_image).
         power (int): The power of the window function

     Returns:
         numpy.array: The untiled image.
     """
    # Define mininally acceptable tile_size and stride_ratio for spline interpolation
    min_tile_size = 32
    min_stride_ratio = 0.5

    stride_ratio = tiles_info['stride_ratio']
    image_shape = tiles_info['image_shape']
    batches = tiles_info['batches']
    x_starts = tiles_info['x_starts']
    x_ends = tiles_info['x_ends']
    y_starts = tiles_info['y_starts']
    y_ends = tiles_info['y_ends']
    overlaps_x = tiles_info['overlaps_x']
    overlaps_y = tiles_info['overlaps_y']
    tile_size_x = tiles_info['tile_size_x']
    tile_size_y = tiles_info['tile_size_y']
    stride_ratio = tiles_info['stride_ratio']
    x_pad = tiles_info['pad_x']
    y_pad = tiles_info['pad_y']

    image_shape = [image_shape[0], image_shape[1], image_shape[2], tiles.shape[-1]]
    window_size = (tile_size_x, tile_size_y)
    image = np.zeros(image_shape, dtype=float)

    window_cache = {}
    for x, y in zip(overlaps_x, overlaps_y):
        if (x, y) not in window_cache:
            w = window_2D(window_size, overlap_x=x, overlap_y=y, power=power)
            window_cache[(x, y)] = w

    for tile, batch, x_start, x_end, y_start, y_end, overlap_x, overlap_y in zip(
            tiles, batches, x_starts, x_ends, y_starts, y_ends, overlaps_x, overlaps_y):

        # Conditions under which to use spline interpolation
        # A tile size or stride ratio that is too small gives inconsistent results,
        # so in these cases we skip interpolation and just return the raw tiles
        if (min_tile_size <= tile_size_x < image_shape[1] and
                min_tile_size <= tile_size_y < image_shape[2] and
                stride_ratio >= min_stride_ratio):
            window = window_cache[(overlap_x, overlap_y)]
            image[batch, x_start:x_end, y_start:y_end, :] += tile * window
        else:
            image[batch, x_start:x_end, y_start:y_end, :] = tile

    image = image.astype(tiles.dtype)

    x_start = x_pad[0]
    y_start = y_pad[0]
    x_end = image_shape[1] - x_pad[1]
    y_end = image_shape[2] - y_pad[1]

    image = image[:, x_start:x_end, y_start:y_end, :]

    return image


def tile_image_3D(image, model_input_shape=(10, 256, 256), stride_ratio=0.5):
    """
    Tile large image into many overlapping tiles of size "model_input_shape".

    Args:
        image (numpy.array): The 3D image to tile, must be rank 5.
        model_input_shape (tuple): The input size of the model.
        stride_ratio (float): The stride expressed as a fraction of the tile sizet

    Returns:
        tuple(numpy.array, dict): An tuple consisting of an array of tiled
            images and a dictionary of tiling details (for use in un-tiling).

    Raises:
        ValueError: image is not rank 5.
    """
    if image.ndim != 5:
        raise ValueError('Expected image of 5, got {}'.format(
            image.ndim))

    image_size_z, image_size_x, image_size_y = image.shape[1:4]
    tile_size_z = model_input_shape[0]
    tile_size_x = model_input_shape[1]
    tile_size_y = model_input_shape[2]

    ceil = lambda x: int(np.ceil(x))
    round_to_even = lambda x: int(np.ceil(x / 2.0) * 2)

    stride_z = min(round_to_even(stride_ratio * tile_size_z), tile_size_z)
    stride_x = min(round_to_even(stride_ratio * tile_size_x), tile_size_x)
    stride_y = min(round_to_even(stride_ratio * tile_size_y), tile_size_y)

    rep_number_z = max(ceil((image_size_z - tile_size_z) / stride_z + 1), 1)
    rep_number_x = max(ceil((image_size_x - tile_size_x) / stride_x + 1), 1)
    rep_number_y = max(ceil((image_size_y - tile_size_y) / stride_y + 1), 1)
    new_batch_size = image.shape[0] * rep_number_z * rep_number_x * rep_number_y

    # catches error caused by interpolation along z axis with rep number = 1
    # TODO - create a better solution or figure out why it doesn't occur in x and y planes
    if rep_number_z == 1:
        stride_z = tile_size_z

    tiles_shape = (new_batch_size, tile_size_z, tile_size_x, tile_size_y, image.shape[4])
    tiles = np.zeros(tiles_shape, dtype=image.dtype)

    # Calculate overlap of last tile along each axis
    overlap_z = (tile_size_z + stride_z * (rep_number_z - 1)) - image_size_z
    overlap_x = (tile_size_x + stride_x * (rep_number_x - 1)) - image_size_x
    overlap_y = (tile_size_y + stride_y * (rep_number_y - 1)) - image_size_y

    # Calculate padding needed to account for overlap and pad image accordingly
    pad_z = (int(np.ceil(overlap_z / 2)), int(np.floor(overlap_z / 2)))
    pad_x = (int(np.ceil(overlap_x / 2)), int(np.floor(overlap_x / 2)))
    pad_y = (int(np.ceil(overlap_y / 2)), int(np.floor(overlap_y / 2)))
    pad_null = (0, 0)
    padding = (pad_null, pad_z, pad_x, pad_y, pad_null)
    image = np.pad(image, padding, 'constant', constant_values=0)

    counter = 0
    batches = []
    z_starts = []
    z_ends = []
    x_starts = []
    x_ends = []
    y_starts = []
    y_ends = []
    overlaps_z = []
    overlaps_x = []
    overlaps_y = []
    z_axis = 1
    x_axis = 2
    y_axis = 3

    for b in range(image.shape[0]):
        for i in range(rep_number_x):
            for j in range(rep_number_y):
                for k in range(rep_number_z):
                    # Compute the start and end for each tile
                    if i != rep_number_x - 1:  # not the last one
                        x_start, x_end = i * stride_x, i * stride_x + tile_size_x
                    else:
                        x_start, x_end = image.shape[x_axis] - tile_size_x, image.shape[x_axis]

                    if j != rep_number_y - 1:  # not the last one
                        y_start, y_end = j * stride_y, j * stride_y + tile_size_y
                    else:
                        y_start, y_end = image.shape[y_axis] - tile_size_y, image.shape[y_axis]

                    if k != rep_number_z - 1:  # not the last one
                        z_start, z_end = k * stride_z, k * stride_z + tile_size_z
                    else:
                        z_start, z_end = image.shape[z_axis] - tile_size_z, image.shape[z_axis]

                    # Compute the overlaps for each tile
                    if i == 0:
                        overlap_x = (0, tile_size_x - stride_x)
                    elif i == rep_number_x - 2:
                        overlap_x = (tile_size_x - stride_x,
                                     tile_size_x - image.shape[x_axis] + x_end)
                    elif i == rep_number_x - 1:
                        overlap_x = ((i - 1) * stride_x + tile_size_x - x_start, 0)
                    else:
                        overlap_x = (tile_size_x - stride_x, tile_size_x - stride_x)

                    if j == 0:
                        overlap_y = (0, tile_size_y - stride_y)
                    elif j == rep_number_y - 2:
                        overlap_y = (tile_size_y - stride_y,
                                     tile_size_y - image.shape[y_axis] + y_end)
                    elif j == rep_number_y - 1:
                        overlap_y = ((j - 1) * stride_y + tile_size_y - y_start, 0)
                    else:
                        overlap_y = (tile_size_y - stride_y, tile_size_y - stride_y)

                    if k == 0:
                        overlap_z = (0, tile_size_z - stride_z)
                    elif k == rep_number_z - 2:
                        overlap_z = (tile_size_z - stride_z,
                                     tile_size_z - image.shape[z_axis] + z_end)
                    elif k == rep_number_z - 1:
                        overlap_z = ((k - 1) * stride_z + tile_size_z - z_start, 0)
                    else:
                        overlap_z = (tile_size_z - stride_z, tile_size_z - stride_z)

                    tiles[counter] = image[b, z_start:z_end, x_start:x_end, y_start:y_end, :]
                    batches.append(b)
                    x_starts.append(x_start)
                    x_ends.append(x_end)
                    y_starts.append(y_start)
                    y_ends.append(y_end)
                    z_starts.append(z_start)
                    z_ends.append(z_end)
                    overlaps_x.append(overlap_x)
                    overlaps_y.append(overlap_y)
                    overlaps_z.append(overlap_z)
                    counter += 1

    tiles_info = {}
    tiles_info['batches'] = batches
    tiles_info['x_starts'] = x_starts
    tiles_info['x_ends'] = x_ends
    tiles_info['y_starts'] = y_starts
    tiles_info['y_ends'] = y_ends
    tiles_info['z_starts'] = z_starts
    tiles_info['z_ends'] = z_ends
    tiles_info['overlaps_x'] = overlaps_x
    tiles_info['overlaps_y'] = overlaps_y
    tiles_info['overlaps_z'] = overlaps_z
    tiles_info['stride_x'] = stride_x
    tiles_info['stride_y'] = stride_y
    tiles_info['stride_z'] = stride_z
    tiles_info['tile_size_x'] = tile_size_x
    tiles_info['tile_size_y'] = tile_size_y
    tiles_info['tile_size_z'] = tile_size_z
    tiles_info['stride_ratio'] = stride_ratio
    tiles_info['image_shape'] = image.shape
    tiles_info['dtype'] = image.dtype
    tiles_info['pad_x'] = pad_x
    tiles_info['pad_y'] = pad_y
    tiles_info['pad_z'] = pad_z

    return tiles, tiles_info


def window_3D(window_size, overlap_z=(5, 5), overlap_x=(32, 32), overlap_y=(32, 32), power=3):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    window_z = spline_window(window_size[0], overlap_z[0], overlap_z[1], power=power)
    window_x = spline_window(window_size[1], overlap_x[0], overlap_x[1], power=power)
    window_y = spline_window(window_size[2], overlap_y[0], overlap_y[1], power=power)

    window_z = np.expand_dims(np.expand_dims(np.expand_dims(window_z, -1), -1), -1)
    window_x = np.expand_dims(np.expand_dims(np.expand_dims(window_x, -1), -1), -1)
    window_y = np.expand_dims(np.expand_dims(np.expand_dims(window_y, -1), -1), -1)

    window = window_z * window_x.transpose(1, 0, 2, 3) * window_y.transpose(1, 2, 0, 3)

    return window


def untile_image_3D(tiles, tiles_info, power=3, force=False, **kwargs):
    """Untile a set of tiled images back to the original model shape.

     Args:
         tiles (numpy.array): The tiled images image to untile.
         tiles_info (dict): Details of how the image was tiled (from tile_image).
         power (int): The power of the window function
         force (bool): If set to True, forces use spline interpolation regardless of
                       tile size or stride_ratio.

     Returns:
         numpy.array: The untiled image.
     """
    # Define mininally acceptable tile_size and stride_ratios for spline interpolation
    min_tile_size = 32
    min_stride_ratio = 0.5

    if force:
        min_tile_size = 0
        min_stride_ratio = 0

    stride_ratio = tiles_info['stride_ratio']
    image_shape = tiles_info['image_shape']
    batches = tiles_info['batches']

    x_starts = tiles_info['x_starts']
    x_ends = tiles_info['x_ends']
    y_starts = tiles_info['y_starts']
    y_ends = tiles_info['y_ends']
    z_starts = tiles_info['z_starts']
    z_ends = tiles_info['z_ends']

    overlaps_x = tiles_info['overlaps_x']
    overlaps_y = tiles_info['overlaps_y']
    overlaps_z = tiles_info['overlaps_z']

    tile_size_x = tiles_info['tile_size_x']
    tile_size_y = tiles_info['tile_size_y']
    tile_size_z = tiles_info['tile_size_z']
    pad_x = tiles_info['pad_x']
    pad_y = tiles_info['pad_y']
    pad_z = tiles_info['pad_z']

    image_shape = tuple(list(image_shape[:4]) + [tiles.shape[-1]])
    window_size = (tile_size_z, tile_size_x, tile_size_y)
    image = np.zeros(image_shape, dtype=float)

    tile_data_zip = zip(tiles, batches, x_starts, x_ends, y_starts,
                        y_ends, z_starts, z_ends, overlaps_x, overlaps_y, overlaps_z)

    for (tile, batch, x_start, x_end, y_start, y_end, z_start,
         z_end, overlap_x, overlap_y, overlap_z) in tile_data_zip:

        # Conditions under which to use spline interpolation
        # A tile size or stride ratio that is too small gives inconsistent results,
        # so in these cases we skip interpolation and just return the raw tiles
        if (min_tile_size <= tile_size_x < image_shape[2] and
                min_tile_size <= tile_size_y < image_shape[3] and
                min_stride_ratio <= stride_ratio):

            window = window_3D(window_size, overlap_z=overlap_z, overlap_x=overlap_x,
                               overlap_y=overlap_y, power=power)
            image[batch, z_start:z_end, x_start:x_end, y_start:y_end, :] += tile * window
        else:
            image[batch, z_start:z_end, x_start:x_end, y_start:y_end, :] = tile

    image = image.astype(tiles.dtype)

    x_start = pad_x[0]
    y_start = pad_y[0]
    z_start = pad_z[0]
    x_end = image_shape[2] - pad_x[1]
    y_end = image_shape[3] - pad_y[1]
    z_end = image_shape[1] - pad_z[1]

    image = image[:, z_start:z_end, x_start:x_end, y_start:y_end, :]

    return image


def fill_holes(label_img, size=10, connectivity=1):
    """Fills holes located completely within a given label with pixels of the same value

    Args:
        label_img (numpy.array): a 2D labeled image
        size (int): maximum size for a hole to be filled in
        connectivity (int): the connectivity used to define the hole

    Returns:
        numpy.array: a labeled image with no holes smaller than ``size``
            contained within any label.
    """
    output_image = np.copy(label_img)

    props = regionprops(np.squeeze(label_img.astype('int')), cache=False)
    for prop in props:
        if prop.euler_number < 1:

            patch = output_image[prop.slice]

            filled = remove_small_holes(
                ar=(patch == prop.label),
                area_threshold=size,
                connectivity=connectivity)

            output_image[prop.slice] = np.where(filled, prop.label, patch)

    return output_image

def erode_edges(mask, erosion_width):
    """Erode edge of objects to prevent them from touching

    Args:
        mask (numpy.array): uniquely labeled instance mask
        erosion_width (int): integer value for pixel width to erode edges

    Returns:
        numpy.array: mask where each instance has had the edges eroded

    Raises:
        ValueError: mask.ndim is not 2 or 3
    """

    if mask.ndim not in {2, 3}:
        raise ValueError('erode_edges expects arrays of ndim 2 or 3.'
                         'Got ndim: {}'.format(mask.ndim))
    if erosion_width:
        new_mask = np.copy(mask)
        for _ in range(erosion_width):
            boundaries = find_boundaries(new_mask, mode='inner')
            new_mask[boundaries > 0] = 0
        return new_mask

    return mask

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
        axis = 1 # if K.image_data_format() == 'channels_first' else K.ndim(y_pred) - 1
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
    return - torch.sum((y_true * torch.log(y_pred) * class_weights), dim=axis)

def semantic_loss(n_classes):
    def _semantic_loss(y_pred, y_true):
        if n_classes > 1:
            tmp_loss = 0.01 * weighted_categorical_crossentropy(y_true, y_pred, n_classes=n_classes)
            return torch.mean(tmp_loss)
        else:
            loss = nn.MSELoss()
            tmp_loss = loss(y_pred, y_true)
            return tmp_loss

    return _semantic_loss


def normalize(image, epsilon=1e-07):
    """Normalize image data by dividing by the maximum pixel value

    Args:
        image (numpy.array): numpy array of image data
        epsilon (float): fuzz factor used in numeric expressions.

    Returns:
        numpy.array: normalized image data
    """
    if not np.issubdtype(image.dtype, np.floating):
        logging.info('Converting image dtype to float')
    image = image.astype('float32')

    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            img = image[batch, ..., channel]
            normal_image = (img - img.mean()) / (img.std() + epsilon)
            image[batch, ..., channel] = normal_image
    return image


def histogram_normalization(image, kernel_size=None):
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
    if not np.issubdtype(image.dtype, np.floating):
        logging.info('Converting image dtype to float')
    image = image.astype('float32')

    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            X = image[batch, ..., channel]
            sample_value = X[(0,) * X.ndim]
            if (X == sample_value).all():
                # TODO: Deal with constant value arrays
                # https://github.com/scikit-image/scikit-image/issues/4596
                logging.warning('Found constant value array in batch %s and '
                                'channel %s. Normalizing as zeros.',
                                batch, channel)
                image[batch, ..., channel] = np.zeros_like(X)
                continue

            # X = rescale_intensity(X, out_range='float')
            X = rescale_intensity(X, out_range=(0.0, 1.0))
            X = equalize_adapthist(X, kernel_size=kernel_size)
            image[batch, ..., channel] = X
    return image


def percentile_threshold(image, percentile=99.9):
    """Threshold an image to reduce bright spots

    Args:
        image: numpy array of image data
        percentile: cutoff used to threshold image

    Returns:
        np.array: thresholded version of input image
    """

    processed_image = np.zeros_like(image)
    for img in range(image.shape[0]):
        for chan in range(image.shape[-1]):
            current_img = np.copy(image[img, ..., chan])
            non_zero_vals = current_img[np.nonzero(current_img)]

            # only threshold if channel isn't blank
            if len(non_zero_vals) > 0:
                img_max = np.percentile(non_zero_vals, percentile)

                # threshold values down to max
                threshold_mask = current_img > img_max
                current_img[threshold_mask] = img_max

                # update image
                processed_image[img, ..., chan] = current_img

    return processed_image

# def watershed(image, min_distance=10, min_size=50, threshold_abs=0.05):
#     """Use the watershed method to identify unique cells based
#     on their distance transform.

#     # TODO: labels should be the fgbg output, NOT the union of distances

#     Args:
#         image (numpy.array): distance transform of image (model output)
#         min_distance (int): minimum number of pixels separating peaks
#         min_size (int): removes small objects if smaller than min_size.
#         threshold_abs (float): minimum intensity of peaks

#     Returns:
#         numpy.array: image mask where each cell is annotated uniquely
#     """
#     distance = np.argmax(image, axis=-1)
#     labels = (distance > 0).astype('int')

#     local_maxi_coords = peak_local_max(
#         image[..., -1],
#         min_distance=min_distance,
#         threshold_abs=threshold_abs,
#         labels=labels,
#         exclude_border=False)

#     # Construct mask from local_maxi coords
#     local_maxi = np.zeros_like(image[..., -1], dtype=bool)
#     local_maxi[tuple(local_maxi_coords.T)] = True

#     # markers = label(local_maxi)
#     markers = ndimage.label(local_maxi)[0]
#     segments = segmentation.watershed(-distance, markers, mask=labels)
#     results = np.expand_dims(segments, axis=-1)
#     results = morphology.remove_small_objects(
#         results, min_size=min_size, connectivity=1)
#     return results

def pixelwise(prediction, threshold=.8, min_size=50, interior_axis=-2):
    """Post-processing for pixelwise transform predictions.
    Uses the interior predictions to uniquely label every instance.

    Args:
        prediction (numpy.array): pixelwise transform prediction
        threshold (float): confidence threshold for interior predictions
        min_size (int): removes small objects if smaller than min_size.

    Returns:
        numpy.array: post-processed data with each cell uniquely annotated
    """
    # instantiate array to be returned
    labeled_prediction = np.zeros(prediction.shape[:-1] + (1,))

    for batch in range(prediction.shape[0]):
        interior = prediction[[batch], ..., interior_axis] > threshold
        labeled = ndimage.label(interior)[0]
        labeled = morphology.remove_small_objects(
            labeled, min_size=min_size, connectivity=1)

        labeled_prediction[batch, ..., 0] = labeled

    return labeled_prediction

def update_data_split(source, data_dir):
    """Match the nuclear segmentation training data split to the tracking split"""
    # Load all data splits
    X, y, meta, original_split = [], [], [], []
    for split in {"train", "test", "val"}:
        with np.load(os.path.join(data_dir, f"{split}.npz"), allow_pickle=True) as data:
            X.append(data["X"])
            y.append(data["y"])
            meta.append(data["meta"][1:])
            original_split.append([split] * data["X"].shape[0])

    X = np.concatenate(X)
    y = np.concatenate(y)
    meta = np.concatenate(meta)
    original_split = np.concatenate(original_split)

    # Check for missed data
    all_source = np.concatenate(list(source.values()))

    missing = []
    for f in np.unique(meta[:, 0]):
        if f not in all_source[:, 0]:
            missing.append(f)

    data = {}
    for split in {"train", "test", "val"}:
        data[split] = {"X": [], "y": []}
        for src in source[split][:, 0]:
            data[split]["X"].append(X[meta[:, 0] == src])
            data[split]["y"].append(y[meta[:, 0] == src])

    for f in missing:
        # Look up original split
        split = original_split[meta[:, 0] == f][0]
        data[split]["X"].append(X[meta[:, 0] == f])
        data[split]["y"].append(y[meta[:, 0] == f])

    for d in data.values():
        d["X"] = np.concatenate(d["X"])
        d["y"] = np.concatenate(d["y"])

    return data

def peak_concomp(image, maxima_threshold=0.7):

    '''
    Uses connected components via  ``regionprops`` to find centroids 
    weighted by intensity image.

    Args:
        inputs: (list): list of [maximas, interiors] model outputs.
        maxima_threshold (float): the rough thresholding used to find 
            region masks. Should be /very/ strict.

    Returns:
        markers (numpy.array): Mask of points corresponding to predicted
            centroid 'seeds' that is then passed into the watershedding algorithm.

    '''
    markers = np.zeros_like(image.squeeze())
    maxima_thresh = image > maxima_threshold
    maxima_thresh = label(maxima_thresh.astype('uint8').squeeze())
    maxima_props = regionprops(maxima_thresh, intensity_image=image.squeeze())


    for prop in maxima_props:
        x, y = (np.around(prop['centroid_weighted'][0]).astype(int), np.around(prop['centroid_weighted'][1]).astype(int))
        markers[x,y] = prop['label']

    return markers

def deep_watershed(outputs,
                   radius=10,
                   maxima_threshold=0.1,
                   interior_threshold=0.01,
                   maxima_smooth=0,
                   interior_smooth=1,
                   maxima_index=0,
                   interior_index=-1,
                   label_erosion=0,
                   small_objects_threshold=0,
                   fill_holes_threshold=0,
                   pixel_expansion=None,
                   maxima_algorithm='h_maxima',
                   **kwargs):
    """Uses ``maximas`` and ``interiors`` to perform watershed segmentation.
    ``maximas`` are used as the watershed seeds for each object and
    ``interiors`` are used as the watershed mask.

    Args:
        outputs (list): List of [maximas, interiors] model outputs.
            Use `maxima_index` and `interior_index` if list is longer than 2,
            or if the outputs are in a different order.
        radius (int): Radius of disk used to search for maxima
        maxima_threshold (float): Threshold for the maxima prediction.
        interior_threshold (float): Threshold for the interior prediction.
        maxima_smooth (int): smoothing factor to apply to ``maximas``.
            Use ``0`` for no smoothing.
        interior_smooth (int): smoothing factor to apply to ``interiors``.
            Use ``0`` for no smoothing.
        maxima_index (int): The index of the maxima prediction in ``outputs``.
        interior_index (int): The index of the interior prediction in
            ``outputs``.
        label_erosion (int): Number of pixels to erode segmentation labels.
        small_objects_threshold (int): Removes objects smaller than this size.
        fill_holes_threshold (int): Maximum size for holes within segmented
            objects to be filled.
        pixel_expansion (int): Number of pixels to expand ``interiors``.
        maxima_algorithm (str): Algorithm used to locate peaks in ``maximas``.
            One of ``h_maxima`` (default) or ``peak_local_max``.
            ``peak_local_max`` is much faster but seems to underperform when
            given regious of ambiguous maxima.

    Returns:
        numpy.array: Integer label mask for instance segmentation.

    Raises:
        ValueError: ``outputs`` is not properly formatted.
    """
    try:
        maximas = outputs[maxima_index]
        interiors = outputs[interior_index]
    except (TypeError, KeyError, IndexError):
        raise ValueError('`outputs` should be a list of at least two '
                         'NumPy arryas of equal shape.')

    valid_algos = {'h_maxima', 'peak_local_max', 'concomp'}
    if maxima_algorithm not in valid_algos:
        raise ValueError('Invalid value for maxima_algorithm: {}. '
                         'Must be one of {}'.format(
                             maxima_algorithm, valid_algos))

    total_pixels = maximas.shape[1] * maximas.shape[2]
    if maxima_algorithm == 'h_maxima' and total_pixels > 5000**2:
        warnings.warn('h_maxima peak finding algorithm was selected, '
                      'but the provided image is larger than 5k x 5k pixels.'
                      'This will lead to slow prediction performance.')
    # Handle deprecated arguments
    min_distance = kwargs.pop('min_distance', None)
    if min_distance is not None:
        radius = min_distance
        warnings.warn('`min_distance` is now deprecated in favor of `radius`. '
                      'The value passed for `radius` will be used.',
                      DeprecationWarning)

    # distance_threshold vs interior_threshold
    distance_threshold = kwargs.pop('distance_threshold', None)
    if distance_threshold is not None:
        interior_threshold = distance_threshold
        warnings.warn('`distance_threshold` is now deprecated in favor of '
                      '`interior_threshold`. The value passed for '
                      '`distance_threshold` will be used.',
                      DeprecationWarning)

    # detection_threshold vs maxima_threshold
    detection_threshold = kwargs.pop('detection_threshold', None)
    if detection_threshold is not None:
        maxima_threshold = detection_threshold
        warnings.warn('`detection_threshold` is now deprecated in favor of '
                      '`maxima_threshold`. The value passed for '
                      '`detection_threshold` will be used.',
                      DeprecationWarning)

    if maximas.shape[:-1] != interiors.shape[:-1]:
        raise ValueError('All input arrays must have the same shape. '
                         'Got {} and {}'.format(
                             maximas.shape, interiors.shape))

    if maximas.ndim not in {4, 5}:
        raise ValueError('maxima and interior tensors must be rank 4 or 5. '
                         'Rank 4 is 2D data of shape (batch, x, y, c). '
                         'Rank 5 is 3D data of shape (batch, frames, x, y, c).')

    input_is_3d = maximas.ndim > 4

    # fill_holes is not supported in 3D
    if fill_holes_threshold and input_is_3d:
        warnings.warn('`fill_holes` is not supported for 3D data.')
        fill_holes_threshold = 0

    label_images = []
    for maxima, interior in zip(maximas, interiors):
        # squeeze out the channel dimension if passed
        maxima = nd.gaussian_filter(maxima[..., 0], maxima_smooth)
        interior = nd.gaussian_filter(interior[..., 0], interior_smooth)

        if pixel_expansion:
            fn = cube if input_is_3d else square
            interior = dilation(interior, footprint=fn(pixel_expansion * 2 + 1))

        # peak_local_max is much faster but has poorer performance
        # when dealing with more ambiguous local maxima
        if maxima_algorithm == 'peak_local_max':
            coords = peak_local_max(
                maxima,
                min_distance=radius,
                threshold_abs=maxima_threshold,
                exclude_border=kwargs.get('exclude_border', False))

            markers = np.zeros_like(maxima)
            slc = tuple(coords[:, i] for i in range(coords.shape[1]))
            markers[slc] = 1

        elif maxima_algorithm == 'h_maxima':
            # Find peaks and merge equal regions
            fn = ball if input_is_3d else disk
            markers = h_maxima(maxima, h=maxima_threshold, footprint=disk(radius))

        else:           
            # Find peaks and merge equal regions
            markers = peak_concomp(maxima, maxima_threshold=maxima_threshold)

        markers = label(markers)
        label_image = segmentation.watershed(-1 * interior, markers,
                                mask=interior > interior_threshold,
                                watershed_line=0)

        if label_erosion:
            label_image = erode_edges(label_image, label_erosion)

        # Remove small objects
        if small_objects_threshold:
            label_image = remove_small_objects(label_image,
                                               min_size=small_objects_threshold)

        # fill in holes that lie completely within a segmentation label
        if fill_holes_threshold > 0:
            label_image = fill_holes(label_image, size=fill_holes_threshold)

        # Relabel the label image
        label_image, _, _ = relabel_sequential(label_image)

        label_images.append(label_image)

    label_images = np.stack(label_images, axis=0)
    label_images = np.expand_dims(label_images, axis=-1)

    return label_images
