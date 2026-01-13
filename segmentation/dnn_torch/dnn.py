"""Nuclear segmentation application"""
import torch

import numpy as np

from utils import histogram_normalization, resize

from model import PanopticNet
from tqdm import tqdm

# pre- and post-processing functions


class DNN():

    def __init__(
            self, 
            model_path=None, 
            device=None, 
            postprocess_kwargs=None,
            batch_size = 16
    ):
        
        print("Initializing model...")
        model = PanopticNet()

        if model_path is None:
            raise Exception("Please provide a path to the model checkpoint file.")

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)

        print(f"Model initialized. \n Using device: {device}")


        self.device = device
        self.model = model.to(self.device)
        self.postprocess_kwargs=postprocess_kwargs
        self.batch_size = batch_size


        self.image_shape = model.crop_size
        self.in_channels = 1
        self.out_channels = 4
        # Require dimension 1 larger than model_input_shape due to addition of batch dimension
        self.model_mpp = 0.65

    def _preprocess(self, image):

        output = np.copy(image)
        output = histogram_normalization(output)

        return output

    def _unfold(self, x):
        _, _, H, W = x.shape
        assert self.in_channels == 1, "Input must have 1 channel"
        assert H % self.image_shape == 0 and W % self.image_shape == 0, "H and W must be multiples of tile_size"
        
        # Calculate number of tiles
        self.n_tiles_h = H // self.image_shape
        self.n_tiles_w = W // self.image_shape
        P = self.n_tiles_h * self.n_tiles_w
        
        # Unfold into tiles
        # Shape: (B, 1, n_tiles_h, n_tiles_w, tile_size, tile_size)
        x_unfold = x.unfold(2, self.image_shape, self.image_shape).unfold(3, self.image_shape, self.image_shape)
        
        # Reshape to (B, P, 1, tile_size, tile_size)
        x_unfold = x_unfold.permute(0, 2, 3, 1, 4, 5).contiguous()
        x_unfold = x_unfold.reshape(self.n_frames, P, 1, self.image_shape, self.image_shape)
        
        # Flatten batch and tile dimensions for processing
        # Shape: (B*P, 1, tile_size, tile_size)
        x_tiled = x_unfold.reshape(self.n_frames * P, 1, self.image_shape, self.image_shape)

        return x_tiled
    
    def _refold(self, x):

        H = self.n_tiles_h * self.image_shape
        W = self.n_tiles_w * self.image_shape
        P = self.n_tiles_h * self.n_tiles_w
        
        # Reshape back to (B, P, C_out, tile_size, tile_size)
        x = x.reshape(self.n_frames, P, self.out_channels, self.image_shape, self.image_shape)
        
        # Reconstruct the full image
        # First reshape to separate tile indices: (B, n_tiles_h, n_tiles_w, C_out, tile_size, tile_size)
        x_tiled = x.reshape(self.n_frames, self.n_tiles_h, self.n_tiles_w, self.out_channels, self.image_shape, self.image_shape)
        
        # Permute to interleave spatial dimensions: (B, C_out, n_tiles_h, tile_size, n_tiles_w, tile_size)
        x_tiled = x_tiled.permute(0, 3, 1, 4, 2, 5).contiguous()
        
        # Final reshape to (B, C_out, H, W)
        x_reconstructed = x_tiled.reshape(self.n_frames, self.out_channels, H, W)
        return x_reconstructed
    
    def _resize_input(self, x):

        T, C, H, W = x.shape
        H_o = H // self.image_shape
        W_o = W // self.image_shape

        # Case where image is smaller than the model
        if H < 256:
            H_o = 1
        if W < 256:
            W_o = 1
        
        # Save number of tiles for inference

        square_size = (H_o * self.image_shape, W_o * self.image_shape)

        reshaped = np.zeros((self.n_frames, self.in_channels) + square_size)

        for t in range(T):

            reshaped[t] = resize(x[t], square_size, data_format='channels_first')

        return reshaped
    
    def _resize_output(self, x):
        
        _, _, H, W = self.input_shape
        output_shape = (self.n_frames, self.out_channels, H, W)
        reshaped = np.zeros(output_shape)    

        for t in range(self.n_frames):

            reshaped[t] = resize(x[t], (H, W), data_format='channels_first')

        return reshaped
    
    def _predict(self, x):

        x_predicted = []

        n_batch = x.shape[0]

        for _, i in enumerate(tqdm(range(0, n_batch, self.batch_size))):

            batch = x[i:i+self.batch_size]

            with torch.inference_mode():
                pred = self.model(batch)
            
            x_predicted.append(pred)

        x_predicted = torch.cat(x_predicted, dim=0)

        return x_predicted
        
    
    def segment(self,
                x):
        
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
                        'interior_threshold': 0.05,
                        'exclude_border': False,
                        'small_objects_threshold': 0,
                        'min_distance': 10,
                        'maxima_algorithm': 'concomp'
                    }


        self.preprocess_kwargs = {
            'normalize': True
        }
        
        # Keep track of original shape for rescaling after processing
        self.H = x.shape[-2]
        self.W = x.shape[-1]
        self.n_frames = x.shape[0]
        self.input_shape = (self.n_frames, 1, self.H, self.W)

        # Preprocess the images and resize to square if necessary
        x = self._resize_input(x)
        x = self._preprocess(x)

        # Move to tensor for unfolding
        x = torch.tensor(x).to(self.device)

        # Unfold images for tiling
        tiles = self._unfold(x)

        output_tiles = self._predict(tiles)

        output_images = self._refold(output_tiles)
        output_images = output_images.cpu().numpy()

        # label_image = postprocess(output_images, **self.postprocess_kwargs)

        label_image = self._resize_output(output_images)
        
        return label_image

if __name__ == '__main__':
    model = DNN(
        model_path='data/segmentation/model/20260109120728/saved_model_best_dict.pth',
        device='cuda:0')
    
    x_test = np.random.rand(35, 1, 512, 512)

    pred = model.segment(x_test)

    print(pred.shape)
    

