"""Data loading pipeline for .trk format cell tracking data in PyTorch"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from skimage.measure import regionprops
import zarr
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as TF
from scipy.spatial.distance import cdist
import tqdm
from skimage.segmentation import relabel_sequential
import warnings
warnings.filterwarnings("ignore") 


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


class TrkDataset(Dataset):
    """PyTorch Dataset for .trk format cell tracking data.
    
    Loads data from .trk files containing:
    - X: raw fluorescent nuclear data (B, T, Y, X, C)
    - y: nuclear segmentation masks (B, T, Y, X, C)
    - lineages: lineage records with cell id, frames, divisions
    
    Args:
        trk_path (str or Path): Path to .trk file (train.trk, val.trk, etc.)
        track_length (int): Number of consecutive frames per sample
        max_cells (int): Maximum number of cells per frame
        crop_size (int): Size of appearance crops (will be crop_size x crop_size)
        stride (int): Stride for temporal sampling (1 = every frame)
        appearance_shape (tuple): Output shape for crops (H, W, C)
        data_format (str): 'channels_first' or 'channels_last'
        mode (str): 'training' or 'inference'
        normalize_images (bool): Whether to normalize raw images
        distance_threshold (float): Distance for adjacency matrix generation
    """
    def __init__(
        self,
        trk_path: Union[str, Path],
        track_length: int = 8,
        crop_size: int = 32,
        stride: int = 1,
        appearance_shape: Tuple[int, int, int] = (32, 32, 1),
        data_format: str = 'channels_first',
        mode: str = 'training',
        normalize_images: bool = True,
        distance_threshold: float = 100.0,
        augment: bool = True,
        rotation_range: int = 180,
        translation_range: float = 0.1,  # As fraction of image size
    ):
        super().__init__()
        self.trk_path = Path(trk_path)
        self.track_length = track_length
        self.crop_size = crop_size
        self.stride = stride
        self.appearance_shape = appearance_shape
        self.data_format = data_format
        self.mode = mode
        self.normalize_images = normalize_images
        self.distance_threshold = distance_threshold
        self.augment = augment
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.crop_mode = 'fixed'
        
        # Load .trk file
        print(f"Loading {self.trk_path}...")
        self.trk_data = zarr.open(self.trk_path, mode='r')
        
        self.X = self.trk_data['X'][0:8]  # Raw images (B, T, Y, X, C)
        self.y = self.trk_data['y'][0:8]  # Segmentation masks (B, T, Y, X, C)
        self.lineages = self.trk_data['lineages'][0][:8] # Lineage information

        if not len(self.X) == len(self.y) == len(self.lineages):
            raise ValueError(
                'The data do not share the same batch size. '
                'Please make sure you are using a valid .trks file')



        self._correct_lineages()
    
        m_cells = 0
        for i in self.lineages:
            curr_m = len(i.keys())
            if curr_m > m_cells:
                m_cells = curr_m
        self.max_cells = m_cells

        self.features = self._get_features()
        
        print(f"  X shape: {self.X.shape}")
        print(f"  y shape: {self.y.shape}")
        print(f"  Lineages: {len(self.lineages)}")
        
        # Build sample indices
        self.samples = self._build_sample_indices()
        print(f"  Created {len(self.samples)} samples")

        if self.augment:
            self._build_augmentation_pipeline()
    
    def _correct_lineages(self):
        """Ensure valid lineages and sequential labels for all batches"""
        new_X = []
        new_y = []
        new_lineages = []
        for batch in tqdm.tqdm(range(self.y.shape[0])):
            if is_valid_lineage(self.y[batch], self.lineages[batch]):

                y_relabel, new_lineage = relabel_sequential_lineage(
                    self.y[batch], self.lineages[batch])

                new_X.append(self.X[batch])
                new_y.append(y_relabel)
                new_lineages.append(new_lineage)
            else:
                print('Invalid lineage detected.')

        self.X = np.stack(new_X, axis=0)
        self.y = np.stack(new_y, axis=0)
        self.lineages = new_lineages
    
    def _get_max_frames(self, curr_lineage):
        """Helper function for finding the maximum number of cells in a frame of a movie, across
        all frames of the movie. Can be used for batches/tracks interchangeably with frames/cells.

        Args:
            y (np.array): Annotated image data

        Returns:
            int: The maximum number of cells in any frame
        """
        max_frames = 0

        for key, track in curr_lineage.items():

            if track['frames']:
                max_frame = track['frames'][-1]
            else:
                max_frame=0

            if max_frame > max_frames:
                max_frames = max_frame

        return max_frames
    
    def _build_sample_indices(self) -> List[Dict]:
        """Build list of valid sample indices.
        
        Creates samples by sliding a window of track_length frames
        across each batch with the specified stride.
        """
        samples = []
        B, T, Y, X, C = self.X.shape
        
        for batch_idx in range(B):
            # Slide window across time dimension
            for start_frame in range(0, T - self.track_length + 1, self.stride):
                end_frame = start_frame + self.track_length

                samples.append({
                    'batch_idx': batch_idx,
                    'start_frame': start_frame,
                    'end_frame': end_frame
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single sample.
        
        Returns:
            Dictionary containing:
            - appearances: (track_length, max_cells, H, W, C) or channels_first
            - morphologies: (track_length, max_cells, 3)
            - centroids: (track_length, max_cells, 2)
            - adj_matrices: (track_length, max_cells, max_cells)
            - labels (optional): (track_length-1, max_cells, max_cells)
        """

        sample_info = self.samples[idx]
        
        batch_idx = sample_info['batch_idx']
        start_frame = sample_info['start_frame']
        end_frame = sample_info['end_frame']

        # Extract raw images and masks for this sample
        raw_images = self.X[batch_idx, start_frame:end_frame]  # (T, Y, X, C)
        masks = self.y[batch_idx, start_frame:end_frame]  # (T, Y, X, C)

        if self.augment:
            raw_images, masks = self._augment_sequence(raw_images, masks)
        
        # Extract features from masks

        items = {
            'appearances': self.features['appearances'][batch_idx, start_frame:end_frame].copy(),
            'centroids': self.features['centroids'][batch_idx, start_frame:end_frame].copy(),
            'morphologies': self.features['morphologies'][batch_idx, start_frame:end_frame].copy(),
            'adj_matrices': self.features['adj_matrix'][batch_idx, start_frame:end_frame].copy(),
        }
        
        # Generate labels if in training mode
        if self.mode == 'training':
            items['labels'] = self.features['labels'][batch_idx, start_frame:end_frame-1].copy()

        
        # Convert to tensors
        tensor_data = self._to_tensors(items)
        
        return tensor_data
    
    def _build_augmentation_pipeline(self):
        """Build torchvision v2 augmentation pipeline."""
        # Note: We manually apply these to maintain temporal consistency
        self.rotation_range_rad = self.rotation_range
        self.translate_range = self.translation_range
    
    def _augment_sequence(
        self,
        raw_images: torch.Tensor,
        masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentations using torchvision.transforms.v2.
        
        Args:
            raw_images: (T, H, W, C) tensor
            masks: (T, H, W, C) tensor (will be treated as masks)
        
        Returns:
            Augmented tensors as numpy arrays
        """
        T, H, W, C = raw_images.shape
        
        # Sample random parameters ONCE for the entire sequence
        angle = torch.FloatTensor(1).uniform_(-self.rotation_range, self.rotation_range).item()
        
        # Translation in pixels (convert from fraction)
        translate_x = int(torch.FloatTensor(1).uniform_(-self.translate_range, self.translate_range).item() * W)
        translate_y = int(torch.FloatTensor(1).uniform_(-self.translate_range, self.translate_range).item() * H)
        
        # Random flips
        do_hflip = torch.rand(1).item() > 0.5
        do_vflip = torch.rand(1).item() > 0.5
        
        augmented_raw = []
        augmented_masks = []
        
        for t in range(T):
            # Get frame: (H, W, C) -> (C, H, W)
            raw_frame = torch.as_tensor(raw_images[t], dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)
            mask_frame = torch.as_tensor(masks[t], dtype=torch.int).permute(2, 0, 1)  # (C, H, W)
            
            # Apply rotation
            raw_frame = TF.rotate(
                raw_frame,
                angle=angle,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0
            )
            mask_frame = TF.rotate(
                mask_frame,
                angle=angle,
                interpolation=transforms.InterpolationMode.NEAREST,  # For masks!
                fill=0
            )
            
            # Apply translation
            raw_frame = TF.affine(
                raw_frame,
                angle=0,
                translate=[translate_x, translate_y],
                scale=1.0,
                shear=0,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0
            )
            mask_frame = TF.affine(
                mask_frame,
                angle=0,
                translate=[translate_x, translate_y],
                scale=1.0,
                shear=0,
                interpolation=transforms.InterpolationMode.NEAREST,
                fill=0
            )
            
            # Apply flips
            if do_hflip:
                raw_frame = TF.hflip(raw_frame)
                mask_frame = TF.hflip(mask_frame)
            
            if do_vflip:
                raw_frame = TF.vflip(raw_frame)
                mask_frame = TF.vflip(mask_frame)
            
            augmented_raw.append(raw_frame.permute(1, 2, 0).numpy())
            augmented_masks.append(mask_frame.permute(1, 2, 0).numpy())
        
        return np.stack(augmented_raw), np.stack(augmented_masks)
    
    def _get_features(self):
        """
        Extract the relevant features from the label movie
        Appearance, morphologies, centroids, and adjacency matrices
        """
        max_tracks = self.max_cells
        n_batches = self.X.shape[0]
        n_frames = self.X.shape[1]
        n_channels = self.X.shape[-1]

        batch_shape = (n_batches, n_frames, max_tracks)

        appearance_shape = self.appearance_shape

        appearances = np.zeros(batch_shape + appearance_shape, dtype='float32')

        morphologies = np.zeros(batch_shape + (3,), dtype='float32')

        centroids = np.zeros(batch_shape + (2,), dtype='float32')

        adj_matrix = np.zeros(batch_shape + (max_tracks,), dtype='float32')

        temporal_adj_matrix = np.zeros((n_batches,
                                        n_frames - 1,
                                        max_tracks,
                                        max_tracks,
                                        3), dtype='float32')

        mask = np.zeros(batch_shape, dtype='float32')

        track_length = np.zeros((n_batches, max_tracks, 2), dtype='int32')

        for batch in tqdm.tqdm(range(n_batches)):
            for frame in range(n_frames):

                frame_features = get_image_features(
                    self.X[batch, frame], self.y[batch, frame],
                    appearance_dim=self.appearance_shape[1],
                    crop_mode=self.crop_mode)

                track_ids = frame_features['labels'] - 1
                centroids[batch, frame, track_ids] = frame_features['centroids']
                morphologies[batch, frame, track_ids] = frame_features['morphologies']
                appearances[batch, frame, track_ids] = frame_features['appearances']
                mask[batch, frame, track_ids] = 1

                # Get adjacency matrix, cannot filter on track ids.
                cent = centroids[batch, frame]
                distance = cdist(cent, cent, metric='euclidean')
                distance = distance < self.distance_threshold

                # Disconnect the padded nodes
                morphs = morphologies[batch, frame]
                is_pad = np.matmul(morphs, morphs.T) == 0

                adj = distance * (1 - is_pad)
                adj_matrix[batch, frame] = adj.astype(np.float32)

            # Get track length and temporal adjacency matrix
            for label in self.lineages[batch]:

                # Get track length
                start_frame = self.lineages[batch][label]['frames'][0]
                end_frame = self.lineages[batch][label]['frames'][-1]

                track_id = int(label) - 1
                track_length[batch, track_id, 0] = start_frame
                track_length[batch, track_id, 1] = end_frame

                # Get temporal adjacency matrix
                frames = self.lineages[batch][label]['frames']

                # Assign same
                for f0, f1 in zip(frames[0:-1], frames[1:]):
                    if f1 - f0 == 1:
                        temporal_adj_matrix[batch, f0, track_id, track_id, 0] = 1

                # Assign daughter
                # WARNING: This wont work if there's a time gap between mother
                # cell disappearing and daughter cells appearing
                last_frame = frames[-1]
                daughters = self.lineages[batch][label]['daughters']
                for daughter in daughters:
                    daughter_id = daughter - 1
                    temporal_adj_matrix[batch, last_frame, track_id, daughter_id, 2] = 1

            # Assign different
            same_prob = temporal_adj_matrix[batch, ..., 0]
            daughter_prob = temporal_adj_matrix[batch, ..., 2]
            temporal_adj_matrix[batch, ..., 1] = 1 - same_prob - daughter_prob

            # Identify cell padding
            for i in range(temporal_adj_matrix.shape[2]):
                # index + 1 is the cell label
                if i + 1 not in self.lineages[batch]:
                    temporal_adj_matrix[batch, :, i] = 0
                    temporal_adj_matrix[batch, :, :, i] = 0

            # Identify temporal padding
            for b in range(temporal_adj_matrix.shape[0]):
                sames = temporal_adj_matrix[b, ..., 0]
                sames = np.sum(sames, axis=(1, 2))
                temporal_adj_matrix[b, sames == 0] = 0

        features = {
            'adj_matrix': adj_matrix,
            'appearances': appearances,
            'morphologies': morphologies,
            'centroids': centroids,
            'labels': temporal_adj_matrix,
            'mask': mask,
            'track_length': track_length
            }

        return features

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

    def _generate_adjacency_matrices(
        self, 
        centroids: np.ndarray
    ) -> np.ndarray:
        """Generate adjacency matrices based on centroid distances.
        
        Args:
            centroids: (T, N, 2) array of centroid positions
        
        Returns:
            adj_matrices: (T, N, N) adjacency matrices
        """
        T, N, _ = centroids.shape
        adj_matrices = np.zeros((T, N, N), dtype=np.float32)
        
        for t in range(T):
            cents = centroids[t]  # (N, 2)
            
            # Compute pairwise distances
            dists = np.sqrt(
                ((cents[:, None, :] - cents[None, :, :]) ** 2).sum(axis=-1)
            )
            
            # Create adjacency based on threshold
            adj = (dists < self.distance_threshold).astype(np.float32)
            
            # Remove self-loops
            np.fill_diagonal(adj, 0)
            
            adj_matrices[t] = adj
        
        return adj_matrices
    
    def _generate_labels(
        self,
        lineage: dict,
        max_cells=None,
        max_frames=None
    ) -> np.ndarray:
        """Generate ground truth tracking labels from lineages.
        
        Args:
            batch_idx: Batch index
            start_frame: Start frame of sequence
            end_frame: End frame of sequence
            cell_ids: (T, N) array of cell IDs in each frame
        
        Returns:
            labels: (T-1, N, N) label matrix where labels[t, i, j] indicates:
                0: no link
                1: different cell
                2: same cell (i in frame t links to j in frame t+1)
        """
        
        curr_max_cells = []
        curr_max_frames = []
        labels = np.full((len(lineage), max_frames, max_cells, max_cells, 3), -1)

        for i, curr_lineage in enumerate(lineage):

            curr_max_cells.append(len(curr_lineage))
            curr_max_frames.append(self._get_max_frames(curr_lineage))
            labels[i,:curr_max_frames[-1], :len(curr_lineage), :len(curr_lineage), :] = 0

            for label, track in curr_lineage.items():
                # The frames where each track is linked to itself (encoded as 1)
                frames_linked = track['frames'][:-1]

                #Identity label
                label = int(label) - 1

                # Place all identities
                labels[i, frames_linked, label, label, 1] = 1

                # Place all mitotic frames as linkages
                if track['frame_div']:
                    frame_divided = track['frame_div']

                    daughter_label = track['daughters']
                    daughter_label = [daughter-1 for daughter in daughter_label]

                    labels[i, frame_divided-1, label, daughter_label, 2] = 1

            # Set "no link" (channel 0) where there's neither identity (channel 1) nor mitotic link (channel 2)
            labels[i, :, :len(curr_lineage), :len(curr_lineage), 0] = np.logical_not(
                np.logical_or(
                    labels[i, :, :len(curr_lineage), :len(curr_lineage), 1],
                    labels[i, :, :len(curr_lineage), :len(curr_lineage), 2]
                ).astype(bool)
            )

        return torch.as_tensor(labels, dtype=torch.float), curr_max_cells, curr_max_frames

    
    def _to_tensors(self, data: Dict) -> Dict[str, torch.Tensor]:
        """Convert numpy arrays to PyTorch tensors."""
        tensors = {}
        
        # Appearances: (T, N, H, W, C)
        appearances = torch.from_numpy(data['appearances']).float()
        
        if self.data_format == 'channels_first':
            # Need to handle the N dimension appropriately
            # For the model, we expect (C, T, H, W) or (T, C, H, W)
            # But we have (T, N, H, W, C) where each N is a separate cell
            
            # We'll process each cell separately in the model
            # Keep as (T, N, H, W, C) for now, model will handle it
            pass
        
        tensors['appearances'] = appearances
        tensors['morphologies'] = torch.from_numpy(data['morphologies']).float()
        tensors['centroids'] = torch.from_numpy(data['centroids']).float()
        tensors['adj_matrices'] = torch.from_numpy(data['adj_matrices']).float()
        
        if 'labels' in data:
            tensors['labels'] = data['labels']
        
        return tensors
    
def unpad_collate_fn(batch):

    actual_max_cells = batch[0]['max_cells']

    assert all(sample['max_cells'] == actual_max_cells for sample in batch), \
        "Batch contains samples from different movies with different cell counts!"
    
    collated = {}

    # Stack appearances: (B, T, N, H, W, C) but slice N to actual_max_cells
    appearances = torch.stack([
        sample['appearances'][:, :actual_max_cells].clone()  # (T, N_actual, H, W, C)
        for sample in batch
    ])
    collated['appearances'] = appearances  # (B, T, N_actual, H, W, C)
    
    # Stack morphologies
    morphologies = torch.stack([
        sample['morphologies'][:, :actual_max_cells].clone()  # (T, N_actual, 3)
        for sample in batch
    ])
    collated['morphologies'] = morphologies
    
    # Stack centroids
    centroids = torch.stack([
        sample['centroids'][:, :actual_max_cells].clone()  # (T, N_actual, 2)
        for sample in batch
    ])
    collated['centroids'] = centroids
    
    # Stack adjacency matrices
    adj_matrices = torch.stack([
        sample['adj_matrices'][:, :actual_max_cells, :actual_max_cells].clone()  # (T, N_actual, N_actual)
        for sample in batch
    ])
    collated['adj_matrices'] = adj_matrices
    
    # Stack labels (also unpad!)
    if 'labels' in batch[0]:
        labels = torch.stack([
            sample['labels'][:, :actual_max_cells, :actual_max_cells].clone()  # (T-1, N_actual, N_actual)
            for sample in batch
        ])
        collated['labels'] = labels
    
    # Store actual_max_cells for reference
    collated['max_cells'] = actual_max_cells
    
    return collated

class BatchSampler(Sampler):

    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.movie_groups = {}
        for idx, sample in enumerate(dataset.samples):
            movie_id = sample['batch_idx']
            if movie_id not in self.movie_groups:
                self.movie_groups[movie_id] = []
            self.movie_groups[movie_id].append(idx)

        self.batches = []
        for movie_id, indices in self.movie_groups.items():
            for i in range(0, len(indices), self.batch_size):
                # Split samples from each movie into batches of length batch size
                

                batch = indices[i:i+self.batch_size]
                if len(batch) == self.batch_size:

                    # Keep full batches only
                    self.batches.append(batch)

    def __iter__(self):

        if self.shuffle:
            indices = torch.randperm(len(self.batches)).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches

        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.batches)


def create_trk_dataloaders(
    train_path: Union[str, Path],
    val_path: Optional[Union[str, Path]] = None,
    test_path: Optional[Union[str, Path]] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    track_length: int = 8,
    crop_size: int = 32,
    stride: int = 1,
    collate_fn = None,
    **dataset_kwargs
) -> Tuple[DataLoader, ...]:
    """Create dataloaders for .trk format data.
    
    Args:
        train_path: Path to train.trk
        val_path: Path to val.trk (optional)
        test_path: Path to test.trk (optional)
        batch_size: Batch size
        num_workers: Number of data loading workers
        track_length: Number of frames per sequence
        max_cells: Maximum cells per frame
        crop_size: Size of appearance crops
        stride: Temporal stride for sampling
        **dataset_kwargs: Additional arguments for TrkDataset
    
    Returns:
        Tuple of DataLoaders (train_loader, val_loader, test_loader)
        None for loaders where path not provided
    """
    loaders = []

    
    # Training loader
    train_dataset = TrkDataset(
        trk_path=train_path,
        track_length=track_length,
        crop_size=crop_size,
        stride=stride,
        mode='training',
        **dataset_kwargs
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler = BatchSampler(train_dataset, batch_size=batch_size, shuffle=True),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
        )
    loaders.append(train_loader)
    
    # Validation loader
    if val_path is not None:
        val_dataset = TrkDataset(
            trk_path=val_path,
            track_length=track_length,
            crop_size=crop_size,
            stride=stride,
            mode='training',
            **dataset_kwargs
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_sampler = BatchSampler(val_dataset, batch_size=batch_size, shuffle=False),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        loaders.append(val_loader)
    else:
        loaders.append(None)
    
    # Test loader
    if test_path is not None:
        test_dataset = TrkDataset(
            trk_path=test_path,
            track_length=track_length,
            crop_size=crop_size,
            stride=stride,
            mode='inference',
            **dataset_kwargs
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_sampler = BatchSampler(test_dataset, batch_size=batch_size, shuffle=False),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

        loaders.append(test_loader)
    else:
        loaders.append(None)
    
    return tuple(loaders)



# Example usage
if __name__ == "__main__":
    print("Testing .trk Data Pipeline")
    print("=" * 70)
    print()
    
    # Example paths (update with your actual paths)
    train_path = "data/DynamicNuclearNet-tracking-v1_0/train.zarr"
    val_path = "data/DynamicNuclearNet-tracking-v1_0/val.zarr"
    
    if Path(train_path).exists():
        print("1. Creating dataloaders...")
        train_loader, val_loader, test_loader = create_trk_dataloaders(
            train_path=train_path,
            val_path=val_path if Path(val_path).exists() else None,
            test_path=None,
            batch_size=2,
            num_workers=4,  # Use 0 for testing
            track_length=8,
            crop_size=32,
            stride=1  # Sample every 4 frames for faster testing
        )
        
        print(f"   Train batches: {len(train_loader)}")
        if val_loader:
            print(f"   Val batches: {len(val_loader)}")
        print()
        
        print("2. Testing batch loading...")
        for i, batch in enumerate(train_loader):
            print(f"   Batch {i}:")
            for key, value in batch.items():
                print(f"     {key}: {value.shape}, dtype: {value.dtype}")
            
        print()
        
        print("✅ .trk data pipeline working!")
    else:
        print("⚠️  train.trk not found. Please update paths in the example.")
        print()
        print("To use with your data:")
        print("  train_loader, val_loader, _ = create_trk_dataloaders(")
        print("      train_path='path/to/train.trk',")
        print("      val_path='path/to/val.trk',")
        print("      batch_size=4,")
        print("      track_length=8,")
        print("      max_cells=39")
        print("  )")