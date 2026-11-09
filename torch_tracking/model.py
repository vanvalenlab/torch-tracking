"""Complete GNN-based cell tracking model in PyTorch"""

import math
import torch
import torch.nn as nn

# Import all components (assumes they're in separate modules)
from torch_tracking.layers import TemporalMerge
from torch_tracking.encoders import (
    AppearanceEncoder, MorphologyEncoder, CentroidEncoder,
    DeltaEncoder, NeighborhoodEncoder
)
from torch_tracking.decoder import TrackingDecoder
from torch_tracking.branches import TrainingBranch, InferenceBranch

class GNNTrackingModel(nn.Module):
    """Complete GNN-based tracking model for single cell tracking.
    
    This model uses Graph Neural Networks to track cells across video frames
    by learning appearance, morphology, and spatial relationships.
    
    The number of cells per frame is not a constructor argument: it is read from the input
    at runtime, so it must agree within a batch but not across batches.

    Args:
        track_length (int): Number of frames in each track sequence
        n_filters (int): Number of convolutional/GNN filters
        encoder_dim (int): Dimension of the feature encoders, and of the embeddings they
            produce. The decoder and the graph layers size themselves from it
        n_layers (int): Number of GNN layers
        graph_layer (str): Type of graph layer ('gcn', 'gat', 'gcs')
        norm_layer (str): Normalization type. Only 'batch' is supported
        appearance_norm (bool): Whether to normalize input images
        n_classes (int): Number of tracking classes (default: 3) - different cell,
            same cell, daughter cell, in that order
        crop_size (int): Side length of the square appearance crops. Must be a power of
            two; `appearance_shape` is derived from it as `(crop_size, crop_size, 1)`
        data_format (str): Must be 'channels_last'. The training branch unpacks a
            channels-last appearance tensor, so channels-first cannot reach a forward pass
        dropout (float): Dropout probability used by the encoders and the decoder

    Example:
        >>> import torch
        >>> model = GNNTrackingModel(track_length=3, n_layers=2, crop_size=32).eval()
        >>> appearances = torch.zeros(1, 3, 4, 32, 32, 1)
        >>> morphologies = torch.zeros(1, 3, 4, 3)
        >>> centroids = torch.zeros(1, 3, 4, 2)
        >>> adj_matrices = torch.zeros(1, 3, 4, 4)
        >>> # Training (returns probabilities, which TrackingLoss requires)
        >>> with torch.no_grad():
        ...     probs = model.training_forward(
        ...         appearances, morphologies, centroids, adj_matrices
        ...     )
        >>> tuple(probs.shape)
        (1, 2, 4, 4, 3)
        >>> # Inference: three tracks with four frames of history, two new detections
        >>> with torch.no_grad():
        ...     probs = model.inference_forward(
        ...         torch.zeros(1, 4, 3, 64), torch.zeros(1, 4, 3, 2),
        ...         torch.zeros(1, 1, 2, 64), torch.zeros(1, 1, 2, 2),
        ...     )
        >>> tuple(probs.shape)
        (1, 1, 3, 2, 3)
    """
    def __init__(
        self,
        track_length=8,
        n_filters=64,
        encoder_dim=64,
        n_layers=3,
        graph_layer='gcn',
        norm_layer='batch',
        appearance_norm=True,
        n_classes=3,
        crop_size=16,
        data_format='channels_last',
        dropout = 0.1
    ):
        super().__init__()
        
        # Store config
        self.track_length = track_length
        self.n_filters = n_filters
        self.encoder_dim = encoder_dim
        self.n_layers = n_layers
        self.graph_layer = graph_layer
        self.appearance_shape = (crop_size, crop_size, 1)
        self.norm_layer = norm_layer
        self.appearance_norm = appearance_norm
        self.n_classes = n_classes
        self.data_format = data_format
        self.dropout = dropout
        
        # Validate inputs
        self._validate_config()
        
        # Build all components
        self._build_encoders()
        self._build_temporal_modules()
        self._build_branches()
        self._build_decoder()
    
    def _validate_config(self):
        """Validate configuration parameters.

        Checks run in dependency order: every value is validated before anything indexes
        or branches on it. The previous ordering read `appearance_shape[1]` and
        `appearance_shape[2]` before checking the shape's length, and left the squareness
        comparison unbound for any `data_format` outside the two it tested.
        """
        if self.data_format != 'channels_last':
            raise ValueError(
                f"data_format must be 'channels_last', got {self.data_format!r}. "
                'The training branch unpacks a channels-last six-dimensional appearance '
                'tensor, so channels-first cannot reach a forward pass.'
            )

        if len(self.appearance_shape) != 3:
            raise ValueError(
                'appearance_shape should be length 3, got '
                f'{len(self.appearance_shape)}'
            )

        height, width, _channels = self.appearance_shape
        if height != width:
            raise ValueError(
                'Appearance shape should have square spatial dimensions, got '
                f'{height}x{width}'
            )

        log2 = math.log2(height)
        if int(log2) != log2:
            raise ValueError(
                f'Spatial dimensions should be power of 2, got {height}'
            )

        graph_layer_name = self.graph_layer.split('-')[0].lower()
        if graph_layer_name not in {'gcn', 'gat', 'gcs'}:
            raise ValueError(f'Invalid graph_layer: {graph_layer_name}')

        if self.norm_layer != 'batch':
            raise ValueError(
                f"norm_layer must be 'batch', got {self.norm_layer!r}. The appearance, "
                'morphology, centroid and delta encoders have no LayerNorm form, so '
                "'layer' built a mostly-BatchNorm model that raised on the first delta "
                'forward. See divergence 59.'
            )
    
    def _build_encoders(self):
        """Build all feature encoders."""
        # Appearance encoder
        self.appearance_encoder = AppearanceEncoder(
            appearance_shape=self.appearance_shape,
            n_filters=self.n_filters,
            encoder_dim=self.encoder_dim,
            appearance_norm=self.appearance_norm,
            data_format=self.data_format,
        )
        
        # Morphology encoder
        self.morphology_encoder = MorphologyEncoder(
            input_dim=3,
            encoder_dim=self.encoder_dim,
        )
        
        # Centroid encoder
        self.centroid_encoder = CentroidEncoder(
            input_dim=2,
            encoder_dim=self.encoder_dim,
        )
        
        # Delta encoders (shared weights for both types)
        self.delta_encoder = DeltaEncoder(
            input_dim=2,
            encoder_dim=self.encoder_dim,
        )
        
        self.delta_across_frames_encoder = DeltaEncoder(
            input_dim=2,
            encoder_dim=self.encoder_dim,
        )
        
        # Neighborhood encoder (combines everything with GNN)
        self.neighborhood_encoder = NeighborhoodEncoder(
            appearance_encoder=self.appearance_encoder,
            morphology_encoder=self.morphology_encoder,
            centroid_encoder=self.centroid_encoder,
            n_filters=self.n_filters,
            embedding_dim=self.encoder_dim,
            n_layers=self.n_layers,
            graph_layer=self.graph_layer,
            norm_layer=self.norm_layer,

        )
    
    def _build_temporal_modules(self):
        """Build temporal processing modules."""
        self.embedding_temporal_merge = TemporalMerge(
            encoder_dim=self.encoder_dim
        )
        
        self.delta_temporal_merge = TemporalMerge(
            encoder_dim=self.encoder_dim
        )
        # No Unmerge layers here: `TrainingBranch` builds and uses its own. The two the
        # model used to build were never read. Divergence 66.
    
    def _build_branches(self):
        """Build training and inference branches."""
        self.training_branch = TrainingBranch(
            neighborhood_encoder=self.neighborhood_encoder,
            embedding_temporal_merge=self.embedding_temporal_merge,
            delta_temporal_merge=self.delta_temporal_merge,
            delta_encoder=self.delta_encoder,
            delta_across_frames_encoder=self.delta_across_frames_encoder,
            track_length=self.track_length
        )
        
        self.inference_branch = InferenceBranch(
            embedding_temporal_merge=self.embedding_temporal_merge,
            delta_temporal_merge=self.delta_temporal_merge,
            delta_encoder=self.delta_encoder,
            delta_across_frames_encoder=self.delta_across_frames_encoder
        )
    
    def _build_decoder(self):

        self.tracking_decoder = TrackingDecoder(
            embedding_dim=self.encoder_dim,
            encoder_dim=self.encoder_dim,
            n_filters=self.n_filters,
            n_classes=self.n_classes,
            norm_layer=self.norm_layer,
            dropout=self.dropout
        )
        
    def training_forward(self, appearances, morphologies, centroids, adj_matrices,
                        return_logits=False):
        """Forward pass for training.
        
        Args:
            appearances: (batch, track_length, max_cells, H, W, C) or channels_first
            morphologies: (batch, track_length, max_cells, 3)
            centroids: (batch, track_length, max_cells, 2)
            adj_matrices: (batch, track_length, max_cells, max_cells)
            return_logits: If True, return logits; if False (the default), return
                probabilities. The default is probabilities because `TrackingLoss` takes
                the logarithm of its input and so requires them; passing logits to it
                yields NaN.
        
        Returns:
            Tensor of shape (batch, track_length-1, max_cells, max_cells, n_classes)
        """

        # Get features from training branch
        embedding_comparisons, deltas, pair_mask = self.training_branch(
            appearances, morphologies, centroids, adj_matrices
        )
        
        # Decode to predictions
        output = self.tracking_decoder(
            embedding_comparisons, deltas,
            apply_softmax=not return_logits,
            is_real=pair_mask
        )
        
        return output
    
    def inference_forward(self, current_embeddings, current_centroids,
                         future_embeddings, future_centroids,
                         return_logits=False):
        """Forward pass for inference/tracking.
        
        Args:
            current_embeddings: (batch, time_history, num_tracks, embedding_dim)
            current_centroids: (batch, time_history, num_tracks, 2)
            future_embeddings: (batch, 1, num_detections, embedding_dim)
            future_centroids: (batch, 1, num_detections, 2)
            return_logits: If True, return logits; if False, return probabilities
        
        Returns:
            Tensor of shape (batch, 1, num_tracks, num_detections, n_classes)
        """
        # Get features from inference branch
        with torch.no_grad():
            embedding_comparisons, deltas = self.inference_branch(
                current_embeddings, current_centroids,
                future_embeddings, future_centroids
            )
            
            # Decode to predictions
            output = self.tracking_decoder(
                embedding_comparisons, deltas,
                apply_softmax=not return_logits
            )
        
        return output
    
    def forward(self, *args, mode='training', **kwargs):
        """Unified forward pass.
        
        Args:
            mode: 'training' or 'inference'
            *args, **kwargs: Arguments for respective forward method
        
        Returns:
            Model outputs based on mode
        """
        if mode == 'training':
            return self.training_forward(*args, **kwargs)
        elif mode == 'inference':
            return self.inference_forward(*args, **kwargs)
        else:
            raise ValueError(f"mode must be 'training' or 'inference', got {mode}")
    
    def get_embeddings(self, appearances, morphologies, centroids, adj_matrices):
        """Extract embeddings for cells (useful for inference setup).
        
        Args:
            appearances: (batch, time, max_cells, H, W, C) or channels_first
            morphologies: (batch, time, max_cells, 3)
            centroids: (batch, time, max_cells, 2)
            adj_matrices: (batch, time, max_cells, max_cells)
        
        Returns:
            embeddings: (batch, time, max_cells, embedding_dim)
            centroids: (batch, time, max_cells, 2)
        """
        batch_size = appearances.shape[0]
        time_steps = appearances.shape[1]
        n_cells = appearances.shape[2]
        
        # CRITICAL FIX: Reshape to merge only batch and time
        # (B, T, N, ...) -> (B*T, N, ...)
        # This matches the corrected encoder expectations
        app_reshaped = appearances.view(batch_size * time_steps, n_cells, *appearances.shape[3:])
        morph_reshaped = morphologies.view(batch_size * time_steps, n_cells, *morphologies.shape[3:])
        cent_reshaped = centroids.view(batch_size * time_steps, n_cells, *centroids.shape[3:])
        adj_reshaped = adj_matrices.view(batch_size * time_steps, *adj_matrices.shape[2:])
        
        # Get embeddings
        # Encoders now return (B*T, N, encoder_dim)
        with torch.no_grad():
            embeddings, centroids_out = self.neighborhood_encoder(
                app_reshaped, morph_reshaped, cent_reshaped, adj_reshaped
            )
        
        # Reshape back: (B*T, N, encoder_dim) -> (B, T, N, encoder_dim)
        embeddings = embeddings.view(batch_size, time_steps, n_cells, -1)
        centroids_out = centroids_out.view(batch_size, time_steps, n_cells, -1)
        
        return embeddings, centroids_out
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Example usage and testing


# Shape checks for these modules live in tests/test_l1_model.py. They used to sit in
# a `__main__` block here, which nothing ran and which had gone stale: it constructed
# modules with arguments they no longer take. Divergence 64.
