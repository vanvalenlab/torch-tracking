# weighted categorical cross entropy
# PanopticNet from mesmer torch
# count gpus and rate scheduler
# histogram normalization

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
from scipy import ndimage
from skimage import morphology
from skimage import segmentation
from skimage.feature import peak_local_max
from skimage.exposure import equalize_adapthist
from skimage.exposure import rescale_intensity
from skimage.measure import label

import torch
import torch.nn as nn

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

def watershed(image, min_distance=10, min_size=50, threshold_abs=0.05):
    """Use the watershed method to identify unique cells based
    on their distance transform.

    # TODO: labels should be the fgbg output, NOT the union of distances

    Args:
        image (numpy.array): distance transform of image (model output)
        min_distance (int): minimum number of pixels separating peaks
        min_size (int): removes small objects if smaller than min_size.
        threshold_abs (float): minimum intensity of peaks

    Returns:
        numpy.array: image mask where each cell is annotated uniquely
    """
    distance = np.argmax(image, axis=-1)
    labels = (distance > 0).astype('int')

    local_maxi_coords = peak_local_max(
        image[..., -1],
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        labels=labels,
        exclude_border=False)

    # Construct mask from local_maxi coords
    local_maxi = np.zeros_like(image[..., -1], dtype=bool)
    local_maxi[tuple(local_maxi_coords.T)] = True

    # markers = label(local_maxi)
    markers = ndimage.label(local_maxi)[0]
    segments = segmentation.watershed(-distance, markers, mask=labels)
    results = np.expand_dims(segments, axis=-1)
    results = morphology.remove_small_objects(
        results, min_size=min_size, connectivity=1)
    return results

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