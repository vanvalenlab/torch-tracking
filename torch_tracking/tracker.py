from typing import Dict, Optional, Tuple
from pathlib import Path

import copy
import timeit
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import glob

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from torch_tracking.utils import get_max_cells, get_image_features, resize, clean_up_annotations

from torch_tracking.model import GNNTrackingModel

class CellTracker:
    """
    Args:
        checkpoint_dir (str or None): The directory of the model weight file

        device (str, optional): The device that you want to use for inference.
            Default is `cuda`, can also be `cpu` or `mps`
            If you have a multi-GPU device, specify using `cuda:<N>` where N is the GPU ID


        distance_threshold (int, optional): The distance threshold by which cells are 
            considered "connected" in the GNN in pixels
            Default is 64 pixels

            
        death (float, optional): The threshold probability for a 
            linkage to be considered a "death"
            Default is 0.999


        birth (float, optional): The threshold probability for a 
            linkage to be considered a new cell, or "birth"
            Default is 0.99

        division (float, optional): The threshold probability for a 
            linkage to be considered a mitosis
            Default is 0.05

        track_length (int, optional): The frames of the movie used to 
            make an educated guess on the next frame
            Default is 8.

        mpp (float, optional): The microns per pixel of the movie, 
            used to resize the input image to
            match the input resolution of the model.
            Default is 0.55 microns per pixel

    Returns: Initialized `CellTracker` object
    """
    def __init__(
        self,
        checkpoint_dir = None,
        device = None,
        distance_threshold: int = 64,
        death: float = 0.999,
        birth: float = 0.999,
        division: float = 0.5,
        track_length: int = 8,
        mpp = 0.55,
        verbose=True,
    ):
        
        # Store model and config
        if device is None:
            self.device = 'cpu'
        else:
            self.device = device
        
        self.distance_threshold = distance_threshold
        self.appearance_dim = 32
        self.death = death
        self.birth = birth
        self.division = division
        self.dtype = 'float32'
        self.track_length = track_length
        self.tracks = {}
        self.model_mpp = 0.55
        self.scale_factor = mpp/self.model_mpp
        self.verbose = verbose

        # Tracking state
        self.a_matrix = []
        self.c_matrix = []
        self.assignments = []
        
        # Format config
        self.channel_axis = -1
        self.time_axis = 0
        self.n_batch = 1
        if self.verbose:
            print("Initializing model...")

        self.tracking_model = GNNTrackingModel(
                        graph_layer='gat', 
                        data_format='channels_last',
                        encoder_dim=64,
                        n_layers=2,
                        crop_size=self.appearance_dim,
                    )
        
        if checkpoint_dir is None:
            from deepcell_auth import download_torch_tracking_model

            download_torch_tracking_model()

            canonical_path = Path.home() / ".deepcell/models"
            # Use latest version
            checkpoint_dir = sorted(
                glob.glob(str(canonical_path / "torch-tracking*.pt"))
            )[-1]

        checkpoint = torch.load(checkpoint_dir, map_location=self.device)['model_state_dict']

        self.tracking_model.load_state_dict(checkpoint)  
        self.tracking_model.eval().to(self.device)

        ## Setup placeholder values
        self.X = None
        self.y = None
        self.y_tracked = None
        self.features = None
        self.tensors = None

        # ID mappings (accounting for 0-indexing vs 1-based labels)
        self.id_to_idx = {}  # cell_id -> index in feature arrays
        self.idx_to_id = {}  # (frame, idx) -> cell_id

        if self.verbose:
            print()
            print("Model initialized.")

    def preprocess_movie(self,
                         movie,
                         annotation):
        """Ingest the movie, resize, extract features, and embeddings.

        Args:
            movie (ndarray[np.float32]): Time lapse image with shape (T, H, W, C) 
                where T is the frames dimension
                H and W are the spatial dimension
                C is the number of channels (should always be 1)

            annotation (ndarray[np.int]): Time lapse image of the 
                corresponding nuclear masks to the input movie
                Must be the same shape as the time lapse image 
                except for the final dimension.

        Returns:
            None
        """
        # Validate inputs
        if len(movie.shape) != 4 or len(annotation.shape) != 4:
            raise ValueError(
                f'Input data and labels must be rank 4 (frames, x, y, channels). '
                f'Got rank {len(movie.shape)} (X) and rank {len(annotation.shape)} (y).'
            )
        
        if movie.shape[:-1] != annotation.shape[:-1]:
            raise ValueError(
                f'Input data and labels should have same shape except channels. '
                f'Got {movie.shape} and {annotation.shape}'
            )
        
        if self.verbose:
            print("Processing data and generating embeddings...")

        # Store data
        self.X = copy.copy(movie)
        self._clean_labels(annotation)

        # Rescale if needed
        new_shape = (int(self.X.shape[1]*self.scale_factor), int(self.X.shape[2] * self.scale_factor))
        self.X = resize(self.X, new_shape)
        self.y = resize(self.y, new_shape, labeled_image=True)

        # Extract features and compute embeddings
        adj_matrices, appearances, morphologies, centroids = self._extract_features()
        
        embeddings = self._compute_embeddings(
            appearances, morphologies, centroids, adj_matrices
        )
        
        # Store features (immutable after initialization)
        self.features = {
            'embedding': embeddings,
            'centroid': centroids.squeeze(0),
        }

        if self.verbose:
            print('Embeddings generated. Ready to track.')
        
    def _clean_labels(self, annotation):
        self.y = copy.copy(annotation)
        self.y = clean_up_annotations(self.y)
    
    def _get_frame(self, tensor: np.ndarray, frame: int) -> np.ndarray:
        """Helper to fetch a frame from tensor based on data_format."""
        return tensor[frame]
    
    def _get_cells_in_frame(self, frame: int) -> list:
        """Get all cell labels in the given frame."""
        cells = np.unique(self._get_frame(self.y, frame))
        cells = np.delete(cells, np.where(cells == 0))  # remove background
        return cells.tolist()
    
    def _extract_features(self) -> Tuple[np.ndarray, ...]:
        """Extract appearance, morphology, centroid, and adjacency features.
        
        Returns:
            adj_matrices: (T, max_cells, max_cells) normalized adjacency matrices
            appearances: (T, max_cells, H, W, C) appearance crops
            morphologies: (T, max_cells, 3) morphological features
            centroids: (T, max_cells, 2) centroid positions
        """
        max_cells = get_max_cells(self.y)
        n_frames = self.X.shape[self.time_axis]
        n_channels = self.X.shape[self.channel_axis]

        # if self.norm:
        #     self.X = histogram_normalization(self.X, data_format='channels_last')

        # Initialize feature arrays
        appearances = np.zeros(
            (self.n_batch, n_frames, max_cells, self.appearance_dim, self.appearance_dim, n_channels),
            dtype=np.float32
        )
        morphologies = np.zeros((self.n_batch, n_frames, max_cells, 3), dtype=np.float32)
        centroids = np.zeros((self.n_batch, n_frames, max_cells, 2), dtype=np.float32)
        adj_matrices = np.zeros((self.n_batch, n_frames, max_cells, max_cells), dtype=np.float32)

        # Extract features for each frame
        for batch in range(self.n_batch):
            
            for frame in range(n_frames):
                frame_features = get_image_features(
                    self.X[frame],
                    self.y[frame]
                )
                
                # Build ID mappings
                for cell_idx, cell_id in enumerate(frame_features['labels']):
                    self.id_to_idx[(frame, int(cell_id))] = cell_idx 
                    self.idx_to_id[(frame, cell_idx)] = int(cell_id)

                # Store features
                num_tracks = len(frame_features['labels'])
                centroids[batch, frame, :num_tracks] = frame_features['centroids']
                morphologies[batch, frame, :num_tracks] = frame_features['morphologies']
                appearances[batch, frame, :num_tracks] = frame_features['appearances']
                
                # Compute adjacency based on distance threshold
                distance = cdist(centroids[batch, frame], centroids[batch, frame], metric='euclidean')
                adj = ((distance > 0) & (distance < self.distance_threshold)).astype(np.float32)
                
                # Disconnect padded nodes
                morph = morphologies[batch, frame]
                is_pad = np.matmul(morph, morph.T) == 0
                adj = adj * (1 - is_pad)
                
                adj_matrices[batch, frame] = adj
        return adj_matrices, appearances, morphologies, centroids

    def _to_tensors(
            self,
            appearances,
            morphologies,
            centroids,
            adj_matrices
    ) -> Dict[str, torch.Tensor]:
        """Convert numpy arrays to PyTorch tensors."""
        tensors = {}
        
        # Appearances: (T, N, H, W, C)
        tensors['appearances'] = torch.from_numpy(appearances).float().to(self.device)
        tensors['morphologies'] = torch.from_numpy(morphologies).float().to(self.device)
        tensors['centroids'] = torch.from_numpy(centroids).float().to(self.device)
        tensors['adj_matrix'] = torch.from_numpy(adj_matrices).float().to(self.device)

        self.tensors = tensors

    @torch.no_grad()
    def _compute_embeddings(
        self,
        appearances: np.ndarray,
        morphologies: np.ndarray,
        centroids: np.ndarray,
        adj_matrices: np.ndarray
    ) -> np.ndarray:
        """Compute embeddings using the GNN neighborhood encoder.
        
        Args:
            appearances: (T, N, H, W, C) appearance features
            morphologies: (T, N, 3) morphology features
            centroids: (T, N, 2) centroid positions
            adj_matrices: (T, N, N) adjacency matrices
        
        Returns:
            embeddings: (T, N, embedding_dim) cell embeddings
        """
        # Convert to tensors and add batch dimension
        self._to_tensors(appearances, morphologies, centroids, adj_matrices)
    
        # Get embeddings from model
        embeddings_t, _ = self.tracking_model.get_embeddings(
            appearances=self.tensors['appearances'],
            morphologies=self.tensors['morphologies'],
            centroids=self.tensors['centroids'],
            adj_matrices=self.tensors['adj_matrix']
        )
        
        # Remove batch dimension and convert back to numpy
        embeddings = embeddings_t.squeeze(0).cpu().numpy()
        return embeddings
    
    def _validate_feature_name(self, feature_name: str):
        """Validate that feature name exists."""
        if feature_name not in self.features:
            raise ValueError(
                f'{feature_name} is invalid. Use one of: {list(self.features.keys())}'
            )
    
    def _get_feature(self, frame: int, cell_id: int, feature_name: str = 'embedding') -> np.ndarray:
        """Get feature for a specific cell in a frame."""
        self._validate_feature_name(feature_name)
        cell_idx = self.id_to_idx[(frame, cell_id)]
        return self.features[feature_name][frame, cell_idx, :]
    
    def _get_frame_features(self, frame: int, feature_name: str = 'embedding') -> Dict[int, np.ndarray]:
        """Get features for all cells in a frame."""
        self._validate_feature_name(feature_name)
        
        cells_in_frame = self._get_cells_in_frame(frame)
        frame_features = {}
        
        for cell_id in cells_in_frame:
            frame_features[cell_id] = self._get_feature(frame, cell_id, feature_name)
        return frame_features
    
    def _create_new_track(self, frame: int, old_label: int):
        """Create a new track for a cell."""
        track_id = len(self.tracks)
        new_label = track_id + 1
        
        # Get features
        embedding = self._get_feature(frame, old_label, feature_name='embedding')
        centroid = self._get_feature(frame, old_label, feature_name='centroid')
        
        # Add dimension for temporal axis
        embedding = np.expand_dims(embedding, axis=0)
        centroid = np.expand_dims(centroid, axis=0)
        
        # Initialize track
        self.tracks[track_id] = {
            'label': new_label,
            'frames': [frame],
            'frame_labels': [old_label],
            'daughters': [],
            'capped': False,
            'frame_div': None,
            'parent': None,
            'embedding': embedding,
            'centroid': centroid
        }
        
        # Sanity check
        if frame > 0 and np.any(self._get_frame(self.y, frame) == new_label):
            raise ValueError(
                f'new_label {new_label} already in annotated frame {frame} (frame > 0)'
            )
        
        self.y[frame][self.y[frame] == old_label] = new_label
    
    def _initialize_tracks(self):
        """Initialize tracks from first frame."""
        frame = 0
        cell_ids = self._get_cells_in_frame(frame)
        
        for cell_id in cell_ids:
            self._create_new_track(frame, cell_id)
        
        # Start tracked label array
        self.y_tracked = self.y[[frame]].astype('int32')
    
    def _fetch_tracked_features(
        self,
        before_frame: Optional[int] = None,
        feature_name: str = 'embedding'
    ) -> Dict[int, np.ndarray]:
        """Get feature history for all active tracks.
        
        Args:
            before_frame: Only include frames before this
            feature_name: Which feature to fetch
        
        Returns:
            Dictionary mapping track_idx -> feature array of shape (track_length, feature_dim)
        """
        self._validate_feature_name(feature_name)
        
        if before_frame is None:
            before_frame = self.X.shape[self.time_axis] + 1
        
        # Get valid frames for each track
        track_valid_frames = (
            (n, [f for f in d['frames'] if f < before_frame])
            for n, d in self.tracks.items()
        )
        tracks_with_frames = [(n, f) for n, f in track_valid_frames if len(f) > 0]
        
        # Fetch features with padding if needed
        tracked_features = {}
        for i, (n, valid_frames) in enumerate(tracks_with_frames):
            frame_dict = {frame: j for j, frame in enumerate(valid_frames)}
            frames = valid_frames[-self.track_length:]
            
            # Pad if not enough history
            if len(frames) != self.track_length:
                num_missing = self.track_length - len(frames)
                frames = frames + [frames[-1]] * num_missing
            
            # Get feature data
            fetched = self.tracks[n][feature_name][[frame_dict[f] for f in frames]]
            tracked_features[i] = fetched
        return tracked_features
    
    def _build_cost_matrix(self, assignment_matrix: np.ndarray) -> np.ndarray:
        """Build full cost matrix from assignment matrix.
        
        Cost matrix structure:
        ┌──────────────────┬─────────────┐
        │  Assignment      │   Death     │
        │  (tracks x cells)│ (tracks x   │
        │                  │  tracks)    │
        ├──────────────────┼─────────────┤
        │  Birth           │   Mordor    │
        │  (cells x cells) │ (cells x    │
        │                  │  tracks)    │
        └──────────────────┴─────────────┘
        
        Args:
            assignment_matrix: (num_tracks, num_cells) linking costs
        
        Returns:
            cost_matrix: (num_tracks+num_cells, num_tracks+num_cells)
        """
        num_tracks, num_cells = assignment_matrix.shape
        size = num_tracks + num_cells
        cost_matrix = np.zeros((size, size), dtype=self.dtype)
        
        # Top-left: Assignment costs
        cost_matrix[:num_tracks, :num_cells] = assignment_matrix
        
        # Bottom-left: Birth costs (diagonal = self.birth, rest = 1)
        birth_diagonal = np.array([self.birth] * num_cells)
        birth_matrix = np.zeros((num_cells, num_cells), dtype=self.dtype)
        birth_matrix = np.diag(birth_diagonal) + np.ones(birth_matrix.shape)
        birth_matrix = birth_matrix - np.eye(num_cells)
        cost_matrix[num_tracks:, 0:num_cells] = birth_matrix
        
        # Top-right: Death costs (diagonal = self.death, rest = 1)
        death_matrix = np.ones((num_tracks, num_tracks), dtype=self.dtype)
        death_matrix = self.death * np.eye(num_tracks) + death_matrix
        death_matrix = death_matrix - np.eye(num_tracks)
        cost_matrix[0:num_tracks, num_cells:] = death_matrix
        
        # Bottom-right: Mordor (transpose of assignment)
        cost_matrix[num_tracks:, num_cells:] = assignment_matrix.T
        return cost_matrix
    
    @torch.no_grad()
    def _get_cost_matrix(self, frame: int) -> Tuple[np.ndarray, Dict]:
        """Build cost matrix for assigning cells in current frame.
        
        Args:
            frame: Current frame index
        
        Returns:
            cost_matrix: Full cost matrix for LAP
            predictions_dict: Dict with predictions and track IDs
        """
        # Prepare inputs for inference branch
        relevant_tracks = []
        
        # Get embeddings for tracked cells (history)
        current_embeddings = self._fetch_tracked_features(
            before_frame=frame, feature_name='embedding'
        )
        current_centroids = self._fetch_tracked_features(
            before_frame=frame, feature_name='centroid'
        )
        
        # Get embeddings for current frame cells
        future_embeddings = self._get_frame_features(frame, 'embedding')
        future_centroids = self._get_frame_features(frame, 'centroid')
        
        # Track which tracks we're considering
        for track_id in current_embeddings:
            relevant_tracks.append(track_id)
        
        # Convert to arrays
        current_emb_arr = np.stack([v for k,v in current_embeddings.items()], axis=1)
        current_cent_arr = np.stack([v for k,v in current_centroids.items()], axis=1)
        future_emb_arr = np.stack([v for k,v in future_embeddings.items()], axis=0)
        future_cent_arr = np.stack([v for k,v in future_centroids.items()], axis=0)
        
        # Add time and batch dimensions
        current_emb_arr = np.expand_dims(current_emb_arr, axis=0)  # (1, T, N, D)
        current_cent_arr = np.expand_dims(current_cent_arr, axis=0)  # (1, T, N, 2)
        future_emb_arr = np.expand_dims(future_emb_arr, axis=0)  # (1, M, D)
        future_emb_arr = np.expand_dims(future_emb_arr, axis=0)  # (1, 1, M, D)
        future_cent_arr = np.expand_dims(future_cent_arr, axis=0)  # (1, M, 2)
        future_cent_arr = np.expand_dims(future_cent_arr, axis=0)  # (1, 1, M, 2)
        
        # Convert to tensors
        current_emb_t = torch.from_numpy(current_emb_arr).float().to(self.device)
        current_cent_t = torch.from_numpy(current_cent_arr).float().to(self.device)
        future_emb_t = torch.from_numpy(future_emb_arr).float().to(self.device)
        future_cent_t = torch.from_numpy(future_cent_arr).float().to(self.device)
        
        # Run inference
        t = timeit.default_timer()
        predictions = self.tracking_model.inference_forward(
            current_emb_t, current_cent_t,
            future_emb_t, future_cent_t,
            return_logits=False
        )
        
        # Extract probabilities: (1, 1, N, M, 3) -> (N, M, 3)
        predictions = predictions[0, 0].cpu().numpy()
        
        # Build assignment matrix from "different cell" probabilities (class 1)
        # Cost = 1 - P(same cell)
        assignment_matrix = 1 - predictions[..., 1]
        
        # Set high cost for capped tracks (already divided)
        for i, track_id in enumerate(relevant_tracks):
            if self.tracks[track_id]['capped']:
                assignment_matrix[i, :] = 1.0
        
        # Store predictions for division detection
        self.a_matrix.append(predictions)
        
        # Build full cost matrix
        cost_matrix = self._build_cost_matrix(assignment_matrix)
        self.c_matrix.append(cost_matrix)
        
        
        predictions_dict = {
            'predictions': predictions,
            'track_ids': relevant_tracks
        }
        return cost_matrix, predictions_dict
    
    def _update_tracks(self, assignments: np.ndarray, frame: int, predictions: Dict):
        """Update tracks based on LAP solution.
        
        Args:
            assignments: (K, 2) array of (track_idx, cell_idx) assignments
            frame: Current frame
            predictions: Dict with prediction probabilities and track IDs
        """
        cells_in_frame = self._get_cells_in_frame(frame)
        
        # Initialize tracked labels for this frame
        y_tracked_update = np.zeros(
            (1, self.y.shape[1], self.y.shape[2], 1), dtype='int32'
        )
        
        self.assignments.append(assignments)
        
        # Process each assignment
        for track_idx, cell_idx in assignments:
            # Map cell index to cell ID
            try:
                cell_id = cells_in_frame[cell_idx]
            except IndexError:
                # Cell died or shadow assignment
                continue
            
            # Get features for this cell
            cell_embedding = self._get_feature(frame, cell_id, 'embedding')
            cell_centroid = self._get_feature(frame, cell_id, 'centroid')
            cell_embedding = np.expand_dims(cell_embedding, axis=0)
            cell_centroid = np.expand_dims(cell_centroid, axis=0)
            
            if track_idx in self.tracks:
                # Add to existing track
                self.tracks[track_idx]['frames'].append(frame)
                self.tracks[track_idx]['frame_labels'].append(cell_id)
                self.tracks[track_idx]['embedding'] = np.concatenate([
                    self.tracks[track_idx]['embedding'], cell_embedding
                ], axis=0)
                self.tracks[track_idx]['centroid'] = np.concatenate([
                    self.tracks[track_idx]['centroid'], cell_centroid
                ], axis=0)
                
                # Update labels
                track_label = track_idx + 1
                y_tracked_update[self.y[[frame]] == cell_id] = track_label
                self.y[frame][self.y[frame] == cell_id] = track_label
            
            else:
                # Create new track (birth)
                self._create_new_track(frame, cell_id)
                new_track_id = max(self.tracks)
                new_label = new_track_id + 1
                
                
                # Check for parent (division detection)
                parent = self._get_parent(frame, cell_id, predictions)
                if parent is not None:

                    self.tracks[new_track_id]['parent'] = parent
                    self.tracks[parent]['daughters'].append(new_track_id)
                else:
                    self.tracks[new_track_id]['parent'] = None
                
                # Update labels
                y_tracked_update[self.y[[frame]] == new_label] = new_track_id + 1
                self.y[frame][self.y[frame] == new_label] = new_track_id + 1
        
        # Handle divided cells that were incorrectly assigned
        for track_id in list(self.tracks):
            if not self.tracks[track_id]['daughters']:
                continue
            
            # Cap tracks for divided cells
            if not self.tracks[track_id]['capped']:
                self.tracks[track_id]['frame_div'] = int(frame)
                self.tracks[track_id]['capped'] = True
            
            # Check if this track was assigned in current frame
            try:
                frame_idx = self.tracks[track_id]['frames'].index(frame)
            except ValueError:
                continue
            
            # Create new track for this cell
            new_track_id = len(self.tracks)
            new_label = new_track_id + 1
            old_label = self.tracks[track_id]['frame_labels'][-1]
            
            self._create_new_track(frame, old_label)
            self.tracks[new_track_id]['parent'] = track_id
            
            # Remove from old track
            del self.tracks[track_id]['frames'][frame_idx]
            del self.tracks[track_id]['frame_labels'][frame_idx]
            self.tracks[track_id]['embedding'] = np.delete(
                self.tracks[track_id]['embedding'], frame_idx, axis=0
            )
            self.tracks[track_id]['centroid'] = np.delete(
                self.tracks[track_id]['centroid'], frame_idx, axis=0
            )
            self.tracks[track_id]['daughters'].append(new_track_id)
            
            # Update labels
            old_track_label = self.tracks[track_id]['label']
            y_tracked_update[self.y[[frame]] == old_track_label] = new_label
            self.y[frame][self.y[frame] == old_track_label] = new_label
        
        # Append to tracked labels
        self.y_tracked = np.concatenate([self.y_tracked, y_tracked_update], axis=0)

    def _get_parent(self, frame: int, cell_id: int, predictions: Dict) -> Optional[int]:
        """Find parent track for a cell (division detection).
        
        Args:
            frame: Frame where cell appears
            cell_id: Cell label in frame
            predictions: Dict with prediction probabilities
        
        Returns:
            parent_id: Track ID of parent, or None if no parent
        """
        parent_id = None
        max_prob = self.division
        
        track_ids = predictions['track_ids']
        preds = predictions['predictions']
        
        for track_idx, track_id in enumerate(track_ids):
            # Skip capped tracks
            if self.tracks[track_id]['capped']:
                continue
            
            for cell_idx in range(preds.shape[1]):
                curr_cell_id = self.idx_to_id[(frame, cell_idx)]
                
                if curr_cell_id == cell_id:
                    # Don't call newly-appeared sibling a parent
                    if self.tracks[track_id]['frames'] == [frame]:
                        continue
                    
                    # Check daughter probability (class 2)
                    prob = preds[track_idx, cell_idx, 2]
                    
                    if prob > max_prob:
                        parent_id = track_id
                        max_prob = prob
        return parent_id
    
    def _track_frame(self, frame: int):
        """Track cells in a single frame."""        
        # Get cost matrix and predictions
        cost_matrix, predictions = self._get_cost_matrix(frame)
        
        # Solve LAP with Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignments = np.stack([row_ind, col_ind], axis=1)
        
        # Update tracks based on solution
        self._update_tracks(assignments, frame, predictions)
    
    def track_cells(self):
        """Track all cells across all frames."""        
        # Initialize from first frame
        self._initialize_tracks()
        
        # Track remaining frames
        num_frames = self.X.shape[self.time_axis]

        for frame in tqdm(range(1, num_frames), leave=False):
            self._track_frame(frame)
        
    def _track_review_dict(self) -> Dict:
        """Create dictionary for review/export."""
        def process(key, track_item):
            if track_item is None:
                return track_item
            if key == 'daughters':
                return [x + 1 for x in track_item]
            if key == 'parent':
                return track_item + 1 if track_item is not None else None
            return track_item
        
        track_keys = ['label', 'frames', 'daughters', 'capped', 'frame_div', 'parent']
        return {
            'tracks': {
                track['label']: {key: process(key, track[key]) for key in track_keys}
                for _, track in self.tracks.items()
            },
            'X': self.X,
            'y': self.y,
            'y_tracked': self.y_tracked
        }
    
    def dataframe(self, **kwargs) -> pd.DataFrame:
        """Export tracks to pandas DataFrame.
        
        Args:
            **kwargs: Optional columns (cell_type, set, part, montage)
        
        Returns:
            DataFrame with track information
        """
        extra_columns = ['cell_type', 'set', 'part', 'montage']
        track_columns = ['label', 'daughters', 'frame_div']
        
        # Validate kwargs
        incorrect_args = set(kwargs) - set(extra_columns)
        if incorrect_args:
            raise ValueError(f'Invalid argument: {incorrect_args.pop()}')
        
        # Filter extra columns
        extra_columns = [c for c in extra_columns if c in kwargs]
        extra_column_vals = [kwargs[c] for c in extra_columns]
        
        # Build dataframe
        data = []
        for _, track in self.tracks.items():
            row = extra_column_vals + [track[c] for c in track_columns]
            data.append(row)
        
        df = pd.DataFrame(data, columns=extra_columns + track_columns)
        
        # Convert daughter track IDs to labels
        df['daughters'] = df['daughters'].apply(
            lambda d: [self.tracks[x]['label'] for x in d]
        )
        return df
    
    def _get_assignment_matrix(self):

        n_frames = len(self.a_matrix)
        max_cells = max(arr.shape[0] for arr in self.a_matrix)
        max_tracks = max(arr.shape[1] for arr in self.a_matrix)

        max_dim = max(max_cells, max_tracks)

        assignment_matrix = np.zeros(((n_frames,) + (max_dim, max_dim, 3)))

        for idx, frame in enumerate(self.a_matrix):
            assignment_matrix[(idx,), :frame.shape[0], :frame.shape[1], :] = frame
        return assignment_matrix

    def _get_assignments(self):

        n_frames = len(self.assignments)
        max_cells = max(arr.shape[0] for arr in self.assignments)
        max_tracks = max(arr.shape[1] for arr in self.assignments)

        max_dim = max(max_cells, max_tracks)

        assignment_matrix = np.zeros(((n_frames,) + (max_dim, 2)))

        for idx, frame in enumerate(self.assignments):
            assignment_matrix[(idx,), :frame.shape[0], :frame.shape[1]] = frame
        return assignment_matrix
    
    def get_lineage_dict(self) -> Dict:
        """Export lineage in standard format for .trk files."""
        lineage = {}
        
        for _, track in self.tracks.items():
            label = track['label']
            lineage[label] = {
                'label': label,
                'frames': track['frames'],
                'parent': track['parent'] + 1 if track['parent'] is not None else None,
                'daughters': [self.tracks[d]['label'] for d in track['daughters']],
                'frame_div': track['frame_div'],
                'capped': track['capped']
            }

        return lineage
    