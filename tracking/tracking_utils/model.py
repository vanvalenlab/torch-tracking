"""Complete GNN-based cell tracking model in PyTorch"""

import math
import torch
import torch.nn as nn

# Import all components (assumes they're in separate modules)
from layers import Unmerge, TemporalMerge
from encoders import (
    AppearanceEncoder, MorphologyEncoder, CentroidEncoder,
    DeltaEncoder, NeighborhoodEncoder
)
from decoder import TrackingDecoder, TrackingDecoderWithAttention
from branches import TrainingBranch, InferenceBranch

class GNNTrackingModel(nn.Module):
    """Complete GNN-based tracking model for single cell tracking.
    
    This model uses Graph Neural Networks to track cells across video frames
    by learning appearance, morphology, and spatial relationships.
    
    Args:
        max_cells (int): Maximum number of tracks per frame
        track_length (int): Length of track sequences
        n_filters (int): Number of convolutional/GNN filters
        encoder_dim (int): Dimension of feature encoders
        embedding_dim (int): Dimension of final embeddings
        n_layers (int): Number of GNN layers
        graph_layer (str): Type of graph layer ('gcn', 'gat', 'gcs')
        appearance_shape (tuple): Shape of appearance crops (time, H, W, C)
        norm_layer (str): Normalization type ('batch' or 'layer')
        appearance_norm (bool): Whether to normalize input images
        n_classes (int): Number of tracking classes (default: 3)
        data_format (str): 'channels_first' or 'channels_last'
    
    Example:
        >>> model = GNNTrackingModel(
        ...     max_cells=39,
        ...     track_length=8,
        ...     appearance_shape=(1, 32, 32, 1)
        ... )
        >>> 
        >>> # Training
        >>> logits = model.training_forward(appearances, morphologies, 
        ...                                  centroids, adj_matrices)
        >>> 
        >>> # Inference
        >>> probs = model.inference_forward(current_emb, current_cent,
        ...                                  future_emb, future_cent)
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
        crop_size=32,
        data_format='channels_first',
        attention=False,
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
        self.attention = attention
        self.dropout = dropout
        
        # Validate inputs
        self._validate_config()
        
        # Build all components
        self._build_encoders()
        self._build_temporal_modules()
        self._build_branches()
        self._build_decoder()
    
    def _validate_config(self):
        """Validate configuration parameters."""

        spatial_dim = self.appearance_shape[1]

        if self.data_format == 'channels_last':
            spatial_compare = self.appearance_shape[1] == self.appearance_shape[0]
        if self.data_format == 'channels_first':
            spatial_compare = self.appearance_shape[1] == self.appearance_shape[2]

        if len(self.appearance_shape) != 3:
            raise ValueError(f'appearance_shape should be length 4, got {len(self.appearance_shape)}')
        
        # Check if spatial dims are square and power of 2
        if not spatial_compare:
            raise ValueError('Appearance shape should have square spatial dimensions')
        
        log2 = math.log2(spatial_dim)
        if int(log2) != log2:
            raise ValueError('Spatial dimensions should be power of 2')
        
        graph_layer_name = self.graph_layer.split('-')[0].lower()
        if graph_layer_name not in {'gcn', 'gat', 'gcs'}:
            raise ValueError(f'Invalid graph_layer: {graph_layer_name}')
        
        if self.norm_layer not in {'batch', 'layer'}:
            raise ValueError(f'norm_layer must be "batch" or "layer", got {self.norm_layer}')
    
    def _build_encoders(self):
        """Build all feature encoders."""
        # Appearance encoder
        self.appearance_encoder = AppearanceEncoder(
            appearance_shape=self.appearance_shape,
            n_filters=self.n_filters,
            encoder_dim=self.encoder_dim,
            norm_layer=self.norm_layer,
            appearance_norm=self.appearance_norm,
            data_format=self.data_format,
        )
        
        # Morphology encoder
        self.morphology_encoder = MorphologyEncoder(
            input_dim=3,
            encoder_dim=self.encoder_dim,
            norm_layer=self.norm_layer,

        )
        
        # Centroid encoder
        self.centroid_encoder = CentroidEncoder(
            input_dim=2,
            encoder_dim=self.encoder_dim,
            norm_layer=self.norm_layer,

        )
        
        # Delta encoders (shared weights for both types)
        self.delta_encoder = DeltaEncoder(
            input_dim=2,
            encoder_dim=self.encoder_dim,
            norm_layer=self.norm_layer,

        )
        
        self.delta_across_frames_encoder = DeltaEncoder(
            input_dim=2,
            encoder_dim=self.encoder_dim,
            norm_layer=self.norm_layer
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
        
        # Unmerge layers
        self.unmerge_embeddings = Unmerge()
        
        self.unmerge_centroids = Unmerge()
    
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
        """Build tracking decoder."""
        if self.attention:
            self.tracking_decoder = TrackingDecoderWithAttention(
                embedding_dim=self.encoder_dim,
                n_filters=self.n_filters,
                n_classes=self.n_classes,
                norm_layer=self.norm_layer,
                attention_heads = 2
            )

        else:
            self.tracking_decoder = TrackingDecoder(
                embedding_dim=self.encoder_dim,
                n_filters=self.n_filters,
                n_classes=self.n_classes,
                norm_layer=self.norm_layer,
                dropout=self.dropout
            )
        
    def training_forward(self, appearances, morphologies, centroids, adj_matrices,
                        return_logits=True):
        """Forward pass for training.
        
        Args:
            appearances: (batch, track_length, max_cells, H, W, C) or channels_first
            morphologies: (batch, track_length, max_cells, 3)
            centroids: (batch, track_length, max_cells, 2)
            adj_matrices: (batch, track_length, max_cells, max_cells)
            return_logits: If True, return logits; if False, return probabilities
        
        Returns:
            Tensor of shape (batch, track_length-1, max_cells, max_cells, n_classes)
        """

        # Get features from training branch
        embedding_comparisons, deltas = self.training_branch(
            appearances, morphologies, centroids, adj_matrices
        )
        
        # Decode to predictions
        output = self.tracking_decoder(
            embedding_comparisons, deltas,
            apply_softmax=not return_logits
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
if __name__ == "__main__":
    print("="*70)
    print("Complete GNN Tracking Model - PyTorch Implementation")
    print("="*70)
    print()
    
    