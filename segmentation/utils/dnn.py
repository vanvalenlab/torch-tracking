"""Nuclear segmentation application"""

import numpy as np

from utils.toolbox import histogram_normalization
from utils.toolbox import deep_watershed
from utils.toolbox import resize, tile_image, untile_image

import logging

import numpy as np
import time
import torch

torch.set_num_threads(4)

from tqdm import tqdm

import logging

# pre- and post-processing functions
def preprocess(image, **kwargs):
    """Preprocess input data for Mesmer model.

    Args:
        image: array to be processed

    Returns:
        np.array: processed image array
    """

    if len(image.shape) != 4:
        raise ValueError(f"Image data must be 4D, got image of shape {image.shape}")

    output = np.copy(image)

    normalize = kwargs.get('normalize', True)
    if normalize:
        output = histogram_normalization(image=output)

    return output


def format_output_mesmer(output_list):
    """Takes list of model outputs and formats into a dictionary for better readability

    Args:
        output_list (list): predictions from semantic heads

    Returns:
        dict: Dict of predictions for whole cell and nuclear.

    Raises:
        ValueError: if model output list is not len(4)
    """
    expected_length = 3
    if len(output_list) != expected_length:
        raise ValueError('output_list was length {}, expecting length {}'.format(
            len(output_list), expected_length))

    formatted_dict = {
        'whole-cell': [output_list[0], output_list[1][..., 1:2]],
        'nuclear': [output_list[2], output_list[3][..., 1:2]],
    }

    return formatted_dict


def postprocess(model_output, **postprocess_kwargs):

    label_images = deep_watershed(model_output, **postprocess_kwargs)

    return label_images


def resize_input(image, image_mpp, model_mpp):
    """Checks if there is a difference between image and model resolution
    and resizes if they are different. Otherwise returns the unmodified
    image.

    Args:
        image (numpy.array): Input image to resize.
        image_mpp (float): Microns per pixel for the ``image``.

    Returns:
        numpy.array: Input image resized if necessary to match ``model_mpp``
    """
    # Don't scale the image if mpp is the same or not defined
    if image_mpp not in {None, model_mpp}:
        shape = image.shape
        scale_factor = image_mpp / model_mpp
        new_shape = (int(shape[1] * scale_factor),
                        int(shape[2] * scale_factor))
        image = resize(image, new_shape, data_format='channels_last')
    return image

def resize_output(image, original_shape):
        """Rescales input if the shape does not match the original shape
        excluding the batch and channel dimensions.

        Args:
            image (numpy.array): Image to be rescaled to original shape
            original_shape (tuple): Shape of the original input image

        Returns:
            numpy.array: Rescaled image
        """
        if not isinstance(image, list):
            image = [image]

        for i in range(len(image)):
            img = image[i]
            # Compare x,y based on rank of image
            # Check if unnecessary
            if len(img.shape) == 4:
                same = img.shape[1:-1] == original_shape[1:-1]
            elif len(img.shape) == 3:
                same = img.shape[1:] == original_shape[1:-1]
            else:
                same = img.shape == original_shape[1:-1]

            # Resize if same is false
            if not same:
                # Resize function only takes the x,y dimensions for shape
                new_shape = original_shape[1:-1]
                img = resize(img, new_shape,
                             data_format='channels_last',
                             labeled_image=True)
            image[i] = img

        if len(image) == 1:
            image = image[0]

        return image

def tile_input(image, model_image_shape, pad_mode='constant'):
    """Tile the input image to match shape expected by model
    using the ``deepcell_toolbox`` or ``toolbox_utils`` function.

    Only supports 4D images.

    Args:
        image (numpy.array): Input image to tile
        pad_mode (str): The padding mode, one of "constant" or "reflect".

    Raises:
        ValueError: Input images must have only 4 dimensions

    Returns:
        (numpy.array, dict): Tuple of tiled image and dict of tiling
        information.
    """
    if len(image.shape) != 4:
        raise ValueError('toolbox_utils.tile_image only supports 4d images.'
                            f'Image submitted for predict has {len(image.shape)} dimensions')

    # Check difference between input and model image size
    x_diff = image.shape[1] - model_image_shape[0]
    y_diff = image.shape[2] - model_image_shape[1]

    # Check if the input is smaller than model image size
    if x_diff < 0 or y_diff < 0:
        # Calculate padding
        x_diff, y_diff = abs(x_diff), abs(y_diff)
        x_pad = (x_diff // 2, x_diff // 2 + 1) if x_diff % 2 else (x_diff // 2, x_diff // 2)
        y_pad = (y_diff // 2, y_diff // 2 + 1) if y_diff % 2 else (y_diff // 2, y_diff // 2)

        tiles = np.pad(image, [(0, 0), x_pad, y_pad, (0, 0)], 'reflect')
        tiles_info = {'padding': True,
                        'x_pad': x_pad,
                        'y_pad': y_pad}
    # Otherwise tile images larger than model size
    else:
        # Tile images, needs 4d
        tiles, tiles_info = tile_image(image, model_input_shape=model_image_shape,
                                        stride_ratio=0.75, pad_mode=pad_mode)

    return tiles, tiles_info

def batch_predict(tiles, batch_size, model, device):
    """Batch process tiles to generate model predictions.

    Batch processing occurs without loading entire image stack onto
    GPU memory, a problem that exists in other solutions such as
    keras.predict.

    Args:
        tiles (numpy.array): Tiled data which will be fed to model
        batch_size (int): Number of images to predict on per batch

    Returns:
        list: Model outputs
    """

    # list to hold final output
    output_tiles = []

    model.eval()
    batch_outputs_list = []

    for idx, i in enumerate(tqdm(range(0, tiles.shape[0], batch_size))):
        
        batch_inputs = tiles[i:i + batch_size, ...]
        temp_input = torch.tensor(batch_inputs).to(device)
        temp_input = torch.permute(temp_input, (0, 3, 1, 2))    

        with torch.inference_mode():
            outs = model(temp_input)
        
        del temp_input

        batch_outputs = [torch.permute(k, (0, 2, 3, 1)) for k in outs]

        del outs

        # model with only a single output gets temporarily converted to a list
        if not isinstance(batch_outputs, list):
            batch_outputs = [batch_outputs.cpu().detach()]

        else:
            batch_outputs = [b_out.cpu().detach() for b_out in batch_outputs]

        # initialize output list with empty arrays to hold all batches
        if not output_tiles:
            for batch_out in batch_outputs:
                shape = (tiles.shape[0],) + batch_out.shape[1:]
                output_tiles.append(np.zeros(shape, dtype=tiles.dtype))

        # save each batch to corresponding index in output list
        for j, batch_out in enumerate(batch_outputs):
            output_tiles[j][idx*batch_size:(idx+1) * batch_size, ...] = batch_out

    return output_tiles

def untile_output(output_tiles, tiles_info, model_image_shape):
    """Untiles either a single array or a list of arrays
    according to a dictionary of tiling specs

    Args:
        output_tiles (numpy.array or list): Array or list of arrays.
        tiles_info (dict): Tiling specs output by the tiling function.

    Returns:
        numpy.array or list: Array or list according to input with untiled images
    """
    # If padding was used, remove padding
    if tiles_info.get('padding', False):
        def _process(im, tiles_info):
            ((xl, xh), (yl, yh)) = tiles_info['x_pad'], tiles_info['y_pad']
            # Edge-case: upper-bound == 0 - this can occur when only one of
            # either X or Y is smaller than model_img_shape while the other
            # is equal to model_image_shape.
            xh = -xh if xh != 0 else None
            yh = -yh if yh != 0 else None
            return im[:, xl:xh, yl:yh, :]
    # Otherwise untile
    else:
        def _process(im, tiles_info):
            out = untile_image(im, tiles_info, model_input_shape=model_image_shape)
            return out

    if isinstance(output_tiles, list):
        output_images = [_process(o, tiles_info) for o in output_tiles]
    else:
        output_images = _process(output_tiles, tiles_info)

    return output_images

class DNN():
    """Loads a :mod:`deepcell.model_zoo.panopticnet.PanopticNet` model
    for nuclear segmentation with pretrained weights.

    The ``predict`` method handles prep and post processing steps
    to return a labeled image.

    Example:

    .. code-block:: python

        from skimage.io import imread
        from deepcell.applications import NuclearSegmentation

        # Load the image
        im = imread('HeLa_nuclear.png')

        # Expand image dimensions to rank 4
        im = np.expand_dims(im, axis=-1)
        im = np.expand_dims(im, axis=0)

        # Create the application
        app = NuclearSegmentation()

        # create the lab
        labeled_image = app.predict(image)

    Args:
        model (tf.keras.Model): The model to load. If ``None``,
            a pre-trained model will be downloaded.
    """

    def __init__(self, model=None, device=None, postprocess_kwargs=None):

        print(f"using device: {device}")
        if model is None:
            raise Exception("Need to provide a model")

        self.device = device
        self.model = model.to(self.device)
        self.postprocess_kwargs=postprocess_kwargs


        self.model_image_shape = model.input_shape[1:]
        # Require dimension 1 larger than model_input_shape due to addition of batch dimension
        self.required_rank = len(self.model_image_shape) + 1
        self.required_channels = self.model_image_shape[-1]

        self.model_mpp = 0.65
        
        # Not used properly right now
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def predict(self,
                image,
                batch_size=16,
                return_transforms=False,
                image_mpp=None,
                preprocess_kwargs={},
                pad_mode='constant'):
        """Generates a labeled image of the input running prediction with
        appropriate pre and post processing functions.

        Input images are required to have 4 dimensions
        ``[batch, x, y, channel]``.
        Additional empty dimensions can be added using ``np.expand_dims``.

        Args:
            image (numpy.array): Input image with shape
                ``[batch, x, y, channel]``.
            batch_size (int): Number of images to predict on per batch.
            image_mpp (float): Microns per pixel for ``image``.
            compartment (str): Specify type of segmentation to predict.
                Must be one of ``"whole-cell"``, ``"nuclear"``, ``"both"``.
            preprocess_kwargs (dict): Keyword arguments to pass to the
                pre-processing function.
            postprocess_kwargs (dict): Keyword arguments to pass to the
                post-processing function.

        Raises:
            ValueError: Input data must match required rank of the application,
                calculated as one dimension more (batch dimension) than expected
                by the model.

            ValueError: Input data must match required number of channels.

        Returns:
            numpy.array: Instance segmentation mask.
        """

        if self.postprocess_kwargs is None:
            self.postprocess_kwargs = {
                'radius': 10,
                'interior_index': 1,
                'maxima_threshold': 0.1,
                'exclude_border': False,
                'small_objects_threshold': 0,
                'min_distance': 10,
                'maxima_algorithm': 'concomp'
            }


        self.preprocess_kwargs = {
            'normalize': True
        }
        
        # Keep track of original shape for rescaling after processing
        orig_img_shape = image.shape
        resized_image = resize_input(image, image_mpp, self.model_mpp)
        image = preprocess(resized_image, **self.preprocess_kwargs)
        
        # Tile images, raises error if the image is not 4d
        tiles, tiles_info = tile_input(image, pad_mode=pad_mode, model_image_shape=self.model_image_shape)

        output_tiles = batch_predict(tiles=tiles, batch_size=batch_size, model=self.model, device=self.device)

        output_images = untile_output(output_tiles, tiles_info, self.model_image_shape)

        label_image = postprocess(output_images, **self.postprocess_kwargs)

        # Restore channel dimension if not already there
        # TODO: check if unnecessary
        if len(image.shape) == self.required_rank - 1:
            image = np.expand_dims(image, axis=-1)

        label_image = resize_output(label_image, orig_img_shape)
        
        if not return_transforms:
            return label_image
        else:
            return label_image, output_images