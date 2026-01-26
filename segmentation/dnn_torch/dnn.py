import torch
from torch.nn import functional as F
from torchvision.transforms import functional as fvision

import numpy as np
from tqdm import tqdm

from utils import resize, histogram_normalization
from model import PanopticNet
import math
import skimage
from sklearn.cluster import DBSCAN
from postprocess_utils import merge_nearby_points
from skimage.measure import regionprops


class DNN():

    def __init__(
            self, 
            model_path=None, 
            device=None, 
            postprocess_kwargs=None,
            batch_size = 16,
            data_format = 'channels_first'
    ):

        if device is None:
            self.device = 'cpu'
        else:
            self.device=device

        if model_path is None:
            raise Exception("Please provide a path to the model checkpoint file.")
        
        print("Initializing model...")
        model = PanopticNet().to(self.device)

        # Dummy data to make semantic heads
        dummy = torch.rand(1, 1, model.crop_size, model.crop_size).to(self.device)
        _ = model(dummy)
        del dummy

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)

        print(f"Model initialized. \n   Using device: {device}")
        print()

        self.device = device
        self.model = model.eval()
        self.postprocess_kwargs=postprocess_kwargs
        self.batch_size = batch_size
        self.data_format = data_format

        self.image_shape = model.crop_size
        self.in_channels = 1
        self.out_channels = 4
        # Require dimension 1 larger than model_input_shape due to addition of batch dimension
        self.model_mpp = 0.65
            
        self.n_iter = self.postprocess_kwargs.get('n_iter', 200)
        self.step_size = self.postprocess_kwargs.get('step_size', 0.1)
        self.postprocess_method = self.postprocess_kwargs.get('postprocess_method','classical')

        if self.postprocess_kwargs['small_objects_threshold'] == 'auto':
            self.small_objects_threshold = np.pi * self.postprocess_kwargs['radius'] ** 2
        else:
            self.small_objects_threshold = self.postprocess_kwargs['small_objects_threshold']

        self.cluster = DBSCAN(eps=postprocess_kwargs['radius'], min_samples = 5)

    def _preprocess(self, x):

        assert len(x.shape) == 4, 'add batch dimension'

        x = histogram_normalization(x, data_format=self.data_format)

        return x

    def _resize_input(self, x):

        # Handle case where image is smaller than the model
        if self.H < 256:
            self.n_tiles_h = 1
        else:
            self.n_tiles_h = math.ceil(self.H / self.image_shape)

        if self.W < 256:
            self.n_tiles_w = 1
        else:
            self.n_tiles_w = math.ceil(self.W / self.image_shape)
        
        square_size = (self.n_tiles_h * self.image_shape, self.n_tiles_w * self.image_shape)

        x = fvision.resize(x, square_size, interpolation = fvision.InterpolationMode.BILINEAR)

        return x
    
    def _unfold_with_overlap(self, x, overlap=32):
        """
        Extract overlapping tiles from image sequence
        image: (T, C, H, W) tensor
        Returns tiles of shape (T*n_tiles_h*n_tiles_w, C, tile_size, tile_size)
        """
        self.curr_batch_size, C, H, W = x.shape

        stride = self.image_shape - overlap
        
        # Calculate number of tiles needed - ensure we cover the entire image
        nh = int(np.ceil((H - self.image_shape) / stride)) + 1
        nw = int(np.ceil((W - self.image_shape) / stride)) + 1
        
        # Process each frame independently, then stack
        all_tiles = []
        for t in range(self.curr_batch_size):
            frame = x[t]  # (C, H, W)
            frame_tiles = []
            
            for i in range(nh):
                for j in range(nw):
                    # For interior tiles, use regular stride
                    # For the last tile, align to the right/bottom edge
                    if i == nh - 1:
                        h_start = H - self.image_shape
                    else:
                        h_start = i * stride
                        
                    if j == nw - 1:
                        w_start = W - self.image_shape
                    else:
                        w_start = j * stride
                    
                    tile = frame[:, h_start:h_start + self.image_shape, w_start:w_start + self.image_shape]
                    frame_tiles.append(tile)
            
            frame_tiles = torch.stack(frame_tiles, dim=0)  # (nh*nw, C, tile_size, tile_size)
            all_tiles.append(frame_tiles)
        
        # Stack all frames
        all_tiles = torch.cat(all_tiles, dim=0)  # (T*nh*nw, C, tile_size, tile_size)
        
        return all_tiles
    
    def _refold_with_blend(self, tiles, overlap=32):
        """
        Reconstruct image sequence from overlapping tiles using weighted blending
        tiles: (T*n_tiles_h*n_tiles_w, C, tile_size, tile_size)
        original_shape: (T, C, H, W)
        """
        T = self.curr_batch_size
        C = self.out_channels

        stride = self.image_shape - overlap
        
        # Calculate number of tiles per dimension (must match tile_with_overlap)
        nh = int(np.ceil((self.H - self.image_shape) / stride)) + 1
        nw = int(np.ceil((self.W - self.image_shape) / stride)) + 1
        tiles_per_frame = nh * nw
        
        # Create blending weight matrix
        weight_tile = self._create_blend_mask(self.image_shape, overlap, tiles.device)
        
        # Create output tensor for all frames
        output = torch.zeros(T, C, self.H, self.W, device=tiles.device)
        weights = torch.zeros(T, C, self.H, self.W, device=tiles.device)
        
        # Process each frame
        for t in range(T):
            # Get tiles for this frame
            frame_tiles = tiles[t * tiles_per_frame:(t + 1) * tiles_per_frame]
            
            # Reconstruct this frame
            idx = 0
            for i in range(nh):
                for j in range(nw):
                    # Match the tiling logic exactly
                    if i == nh - 1:
                        h_start = self.H - self.image_shape
                    else:
                        h_start = i * stride
                        
                    if j == nw - 1:
                        w_start = self.W - self.image_shape
                    else:
                        w_start = j * stride
                    
                    h_end = h_start + self.image_shape
                    w_end = w_start + self.image_shape
                    
                    # Apply the weight mask to this tile
                    output[t, :, h_start:h_end, w_start:w_end] += \
                        frame_tiles[idx] * weight_tile
                    weights[t, :, h_start:h_end, w_start:w_end] += weight_tile
                    idx += 1
        
        # Avoid division by zero
        weights = torch.clamp(weights, min=1e-8)
        
        return output / weights

    def _create_blend_mask(self, tile_size, overlap, device):
        """Create separable blending mask - computed once, reused for all tiles"""
        mask_1d = torch.ones(tile_size, device=device)
        fade = torch.linspace(0, 1, overlap, device=device)
        mask_1d[:overlap] = fade
        mask_1d[-overlap:] = fade.flip(0)
        
        # Create 2D mask via outer product
        mask = mask_1d.unsqueeze(1) * mask_1d.unsqueeze(0)
        return mask
    
    def _predict(self, x):

        x_predicted = []

        n_batch = x.shape[0]

        pbar = enumerate(tqdm(range(0, n_batch, self.batch_size), desc="Inference on batch", leave=False, colour='#FFDB58'))

        for _, i in pbar:

            batch = x[i:i+self.batch_size]

            with torch.inference_mode():
                pred = self.model(batch)
            
            x_predicted.append(pred)

        x_predicted = torch.cat(x_predicted, dim=0)

        return x_predicted
    
    def _resize_output(self, x):
        
        x = fvision.resize(x, (self.H, self.W), interpolation=fvision.InterpolationMode.BILINEAR)

        return x
    
    def _minmax(self, x):

        arr_flat = x.reshape(x.shape[0], x.shape[1], -1)

        min_vals = arr_flat.min(axis=2, keepdims=True).reshape(x.shape[0], x.shape[1], 1, 1)
        max_vals = arr_flat.max(axis=2, keepdims=True).reshape(x.shape[0], x.shape[1], 1, 1)

        x_normalized = (x - min_vals) / (max_vals - min_vals + 1e-8)

        return x_normalized

    def _get_gradients(self, transform, foreground_tensor):
        # Move to device and ensure correct dtypes

        transform = torch.where(foreground_tensor > self.postprocess_kwargs['transform_thresh'], transform, -1).float()
        
        # Compute gradients of INNER distance (points toward peaks)
        transform_4d = transform.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                            dtype=torch.float32, device=transform.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                            dtype=torch.float32, device=transform.device).view(1, 1, 3, 3)
        
        
        gx = F.conv2d(transform_4d, sobel_x, padding=1)  # (1, 1, H, W)
        gy = F.conv2d(transform_4d, sobel_y, padding=1)  # (1, 1, H, W)

        # Normalize gradients
        grad_mag = torch.sqrt(gx**2 + gy**2) + 1e-8
        gx = gx / grad_mag
        gy = gy / grad_mag

        return gx, gy
    
    def _get_positions(self, transform):

        # Initialize positions for all pixels
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.H, device=transform.device, dtype=torch.float32),
            torch.arange(self.W, device=transform.device, dtype=torch.float32),
            indexing='ij'
        )

        positions = torch.stack([y_coords, x_coords], dim=-1)  # (H, W, 2) -> [y, x]

        return positions
    
    def _follow_flows(self, positions, gx, gy, niter=20, step_size=0.5):

        for step in range(niter):
            # Normalize positions to [-1, 1] for grid_sample
            norm_x = 2 * positions[..., 1] / (self.W - 1) - 1
            norm_y = 2 * positions[..., 0] / (self.H - 1) - 1
            grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)  # (1, H, W, 2) in (x, y) order

            # Sample gradients at current positions
            gx_sample = F.grid_sample(
                gx, 
                grid, 
                align_corners=False, 
            ).squeeze()  # (H, W)

            gy_sample = F.grid_sample(
                gy, 
                grid, 
                align_corners=False, 
            ).squeeze()   # (H, W)

            # Update positions
            positions[..., 0] = positions[..., 0] + step_size * gy_sample
            positions[..., 1] = positions[..., 1] + step_size * gx_sample


            # Clamp to image bounds
            positions[..., 0].clamp_(0, self.H - 1)
            positions[..., 1].clamp_(0, self.W - 1)    

        return positions
    
    def _postprocess(self, x):

        label_image = np.zeros((self.curr_batch_size, 1, self.H, self.W), dtype=int)

        if self.postprocess_method == 'classical':

            x = x.cpu().numpy()

            pbar = tqdm(range(self.curr_batch_size), desc="Postprocessing on batch", leave=False, colour='#CE2029')

            for t in pbar:

                x_temp = x[t]
                
                inner_transform = np.where(x_temp[3] > self.postprocess_kwargs['transform_thresh'], x_temp[0], 0)
                outer_transform = np.where(x_temp[3] > self.postprocess_kwargs['transform_thresh'], x_temp[1], 0)

                inner_transform = skimage.filters.gaussian(inner_transform, sigma=1, channel_axis=0)
                outer_transform = skimage.filters.gaussian(outer_transform, sigma=1, channel_axis=0)

                markers = skimage.morphology.h_maxima(
                    inner_transform, 
                    h=self.postprocess_kwargs['maxima_threshold'], 
                    footprint=skimage.morphology.disk(self.postprocess_kwargs['radius'])
                )
                
                markers = skimage.measure.label(markers)

                label_temp = skimage.segmentation.watershed(
                    -1 * (inner_transform + outer_transform), 
                    markers, 
                    mask= (inner_transform + outer_transform) > self.postprocess_kwargs['reduced_threshold'], 
                    watershed_line=True
                )

                label_image[t], _, _ = skimage.segmentation.relabel_sequential(label_temp)
                label_image[t] = skimage.morphology.area_closing(label_image[t].squeeze())

                if self.small_objects_threshold > 0.:
                    large_objects = skimage.morphology.remove_small_objects(label_image[t] > 0, min_size=self.small_objects_threshold)
                    label_image[t] = label_image[t] * large_objects
                
                if self.postprocess_kwargs['eccentricity'] < 1.:
                    for prop in regionprops(label_image[t].squeeze()):
                        label_ = prop.label
                        if prop.eccentricity > self.postprocess_kwargs['eccentricity']:
                            label_image = np.where(label_image[t] == label_, 0, label_image)


            label_image = label_image.astype(int)

        if self.postprocess_method == 'hybrid':

            pbar = tqdm(range(self.curr_batch_size), desc="Postprocessing", leave=False, colour='#CE2029')

            for t in pbar:

                x_temp = x[t].cpu().numpy()
                
                inds = torch.argwhere(x[t,3] > self.postprocess_kwargs['transform_thresh']).t().cpu().numpy()
                positions = self._get_positions(x[t,0])
                gx, gy = self._get_gradients(x[t,0], x[t,3])

                positions = self._follow_flows(positions, gx, gy, niter=self.n_iter, step_size=self.step_size)

                relevant = positions[inds[0], inds[1]].cpu().numpy().astype(int)
                relevant, relevant_counts = np.unique(relevant, axis=0, return_counts=True)

                x_temp = x[t].cpu().numpy()

                inner_transform = np.where(x_temp[3] > self.postprocess_kwargs['transform_thresh'], x_temp[0], 0)
                outer_transform = np.where(x_temp[3] > self.postprocess_kwargs['transform_thresh'], x_temp[1], 0)

                inner_transform = skimage.filters.gaussian(inner_transform, sigma=1, channel_axis=0)
                outer_transform = skimage.filters.gaussian(outer_transform, sigma=1, channel_axis=0)

                markers = np.zeros((self.H, self.W))

                #TODO: Make keyword argument
                relevant = relevant[relevant_counts > 50]

                for i in range(relevant.shape[0]):
                    curr_point = relevant[i]
                    markers[curr_point[0], curr_point[1]] = 1

                markers = skimage.measure.label(markers)

                label_temp = skimage.segmentation.watershed(
                    -1 * (inner_transform + outer_transform), 
                    markers, 
                    mask= (inner_transform + outer_transform) > self.postprocess_kwargs['reduced_threshold'], 
                    watershed_line=True
                )

                label_image[t], _, _ = skimage.segmentation.relabel_sequential(label_temp)
                label_image[t] = skimage.morphology.area_closing(label_image[t].squeeze())

                if self.small_objects_threshold > 0.:
                    large_objects = skimage.morphology.remove_small_objects(label_image[t] > 0, min_size=self.small_objects_threshold)
                    label_image[t] = label_image[t] * large_objects
                
                if self.postprocess_kwargs['eccentricity'] < 1.:
                    for prop in regionprops(label_image[t].squeeze()):
                        label_ = prop.label
                        if prop.eccentricity > self.postprocess_kwargs['eccentricity']:
                            label_image = np.where(label_image[t] == label_, 0, label_image)

                
            label_image = label_image.astype(int)

        return label_image
        
    def segment(self,
                x,
                data_format='channels_first',
                return_transforms = False):
        
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


        if data_format == 'channels_last':
            x = np.moveaxis(x, -1, 1)

        label_image = np.zeros_like(x)

        # Keep track of original shape for rescaling after processing
        self.H = x.shape[-2]
        self.W = x.shape[-1]
        self.n_frames = x.shape[0]

        pbar = tqdm(range(0, self.n_frames, self.batch_size), leave=False, colour='#008080')

        transforms = torch.zeros((self.n_frames, self.out_channels, self.H, self.W))

        # Preprocess the images and resize to square if necessary
        for i in pbar:
            
            pbar.set_description(f"Log normalizing image")
            x_batch = x[i:i+self.batch_size]
            x_batch = self._preprocess(x_batch)

            x_batch = torch.from_numpy(x_batch).to(self.device)

            pbar.set_description("Inference on tiles")

            # Unfold images for tiling
            tiles = self._unfold_with_overlap(x_batch, overlap=32)
            # Batchwise predictions
            
            tiles = tiles.float()
            output_tiles = self._predict(tiles)

            # Refold batch * tiles into shape (T, C, H_sq, W_sq)
            output_images = self._refold_with_blend(output_tiles, overlap=32)

            # Reshape image back to original size
            pbar.set_description(f"Postprocessing using {self.postprocess_method}")

            if return_transforms:
                transforms[i:i+self.batch_size] = output_images
            
            label_image[i:i+self.batch_size] = self._postprocess(output_images)
                    
        pbar.set_description("Done segmenting images.")

        if return_transforms:
            return label_image, transforms
        else:
            return label_image

    

