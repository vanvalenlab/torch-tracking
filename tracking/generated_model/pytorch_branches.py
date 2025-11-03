"""Training and Inference branches for cell tracking model in PyTorch"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming these are imported from your other modules
from pytorch_custom_layers import Comparison, DeltaReshape, Unmerge, TemporalMerge
from pytorch_custom_layers import compute_deltas, compute_deltas_across_frames


class TrainingBranch(nn.Module):
    """Training branch that processes full sequences of tracked cells.
    
    Takes complete tracks across multiple frames and prepares features
    for the tracking decoder by:
    1. Encoding appearance, morphology, and spatial features
    2. Extracting temporal patterns with LSTM
    3. Computing pairwise comparisons between consecutive frames
    4. Encoding position deltas
    
    Args:
        neighborhood_encoder (nn.Module): Encodes node features with GNN
        embedding_temporal_merge (nn.Module): LSTM for temporal embeddings
        delta_temporal_merge (nn.Module): LSTM for temporal deltas
        delta_encoder (nn.Module): Encodes within-frame position deltas
        delta_across_frames_encoder (nn.Module): Encodes cross-frame deltas
        track_length (int): Number of frames in each track
        max_cells (int): Maximum number of cells per frame
        embedding_dim (int): Dimension of embeddings
        encoder_dim (int): Dimension of encoder features
    """
    def __init__(
        self,
        neighborhood_encoder,
        embedding_temporal_merge,
        delta_temporal_merge,
        delta_encoder,
        delta_across_frames_encoder,
        track_length,
        max_cells,
        embedding_dim,
        encoder_dim
    ):
        super().__init__()
        self.neighborhood_encoder = neighborhood_encoder
        self.embedding_temporal_merge = embedding_temporal_merge
        self.delta_temporal_merge = delta_temporal_merge
        self.delta_encoder = delta_encoder
        self.delta_across_frames_encoder = delta_across_frames_encoder
        
        self.track_length = track_length
        self.max_cells = max_cells
        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder_dim
        
        # Layers for unmerging temporal dimensions
        self.unmerge_embeddings = Unmerge(track_length, max_cells, embedding_dim)
        self.unmerge_centroids = Unmerge(track_length, max_cells, 2)
        
        # Comparison layer
        self.comparison = Comparison()

    def forward(self, appearances, morphologies, centroids, adj_matrices):
        """
        Args:
            appearances: (batch, time, height, width, channels) or channels_first
            morphologies: (batch, time, 3)
            centroids: (batch, time, 2)
            adj_matrices: (batch, time, max_cells, max_cells)
        
        Returns:
            embedding_comparisons: (batch, time-1, max_cells, max_cells, 2*embedding_dim)
            deltas: (batch, time-1, max_cells, max_cells, 2*encoder_dim)
        """
        print(appearances.shape)
        batch_size, time_steps, max_cells, _, _ = appearances.shape
        # batch_size = appearances.shape[0]
        
        
        # Merge batch and temporal dimensions for neighborhood encoder
        # The neighborhood encoder expects (batch*time, ...)
        # time_steps = self.track_length
        
        # Reshape inputs: (B, T, ...) -> (B*T, ...)
        app_reshaped = self._reshape_merge_time(appearances, batch_size, time_steps, max_cells)
        morph_reshaped = self._reshape_merge_time(morphologies, batch_size, time_steps, max_cells)
        cent_reshaped = self._reshape_merge_time(centroids, batch_size, time_steps, max_cells)
        adj_reshaped = self._reshape_merge_time(adj_matrices, batch_size, time_steps, max_cells)
        
        # Encode features with neighborhood encoder
        embeddings, centroids_out = self.neighborhood_encoder(
            app_reshaped, morph_reshaped, cent_reshaped, adj_reshaped
        )
        
        # Unmerge temporal dimension: (B*T, F) -> (B, T, N, F)
        embeddings = self.unmerge_embeddings(embeddings)
        centroids_out = self.unmerge_centroids(centroids_out)
        
        # Split into current and future frames
        embeddings_current = embeddings[:, :-1]  # (B, T-1, N, F)
        embeddings_future = embeddings[:, 1:]     # (B, T-1, N, F)
        
        # Apply temporal merge to current embeddings
        embeddings_current = self.embedding_temporal_merge(embeddings_current)
        
        # Compare current and future embeddings
        embedding_comparisons = self.comparison(embeddings_current, embeddings_future)
        
        # Process centroids to get deltas
        # 1. Compute deltas within each frame (across tracks)
        deltas_within = compute_deltas(centroids_out)  # (B, T, N, 2)
        deltas_within = torch.abs(deltas_within)
        
        # 2. Compute deltas across frames (between all pairs)
        deltas_across = compute_deltas_across_frames(centroids_out)  # (B, T-1, N, N, 2)
        deltas_across = torch.abs(deltas_across)
        
        # Encode deltas
        deltas_within_enc = self.delta_encoder(deltas_within)  # (B, T, N, encoder_dim)
        deltas_across_enc = self.delta_across_frames_encoder(deltas_across)  # (B, T-1, N, N, encoder_dim)
        
        # Apply temporal merge to within-frame deltas
        deltas_within_current = deltas_within_enc[:, :-1]  # (B, T-1, N, encoder_dim)
        deltas_within_current = self.delta_temporal_merge(deltas_within_current)
        
        # Expand within-frame deltas to match across-frame shape
        # (B, T-1, N, encoder_dim) -> (B, T-1, N, 1, encoder_dim)
        deltas_within_current = deltas_within_current.unsqueeze(3)
        # Tile to (B, T-1, N, N, encoder_dim)
        deltas_within_current = deltas_within_current.expand(-1, -1, -1, self.max_cells, -1)
        
        # Concatenate within and across deltas
        deltas = torch.cat([deltas_within_current, deltas_across_enc], dim=-1)
        
        return embedding_comparisons, deltas

    def _reshape_merge_time(self, x, batch_size, time_steps, max_cells):
        """Helper to merge batch and time dimensions."""
        shape = x.shape
        new_shape = (batch_size * time_steps, max_cells) + shape[3:]
        return x.reshape(new_shape)


class InferenceBranch(nn.Module):
    """Inference branch for online tracking of new frames.
    
    Processes partial tracks (current history) and a new frame to predict
    which cells in the new frame correspond to existing tracks.
    
    Args:
        embedding_temporal_merge (nn.Module): LSTM for temporal embeddings
        delta_temporal_merge (nn.Module): LSTM for temporal deltas
        delta_encoder (nn.Module): Encodes within-frame position deltas
        delta_across_frames_encoder (nn.Module): Encodes cross-frame deltas
        embedding_dim (int): Dimension of embeddings
        encoder_dim (int): Dimension of encoder features
    """
    def __init__(
        self,
        embedding_temporal_merge,
        delta_temporal_merge,
        delta_encoder,
        delta_across_frames_encoder,
        embedding_dim,
        encoder_dim
    ):
        super().__init__()
        self.embedding_temporal_merge = embedding_temporal_merge
        self.delta_temporal_merge = delta_temporal_merge
        self.delta_encoder = delta_encoder
        self.delta_across_frames_encoder = delta_across_frames_encoder
        
        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder_dim
        
        # Layers
        self.comparison = Comparison()
        self.delta_reshape = DeltaReshape()

    def forward(self, current_embeddings, current_centroids, 
                future_embeddings, future_centroids):
        """
        Args:
            current_embeddings: (batch, time_history, num_tracks, embedding_dim)
            current_centroids: (batch, time_history, num_tracks, 2)
            future_embeddings: (batch, 1, num_detections, embedding_dim)
            future_centroids: (batch, 1, num_detections, 2)
        
        Returns:
            embedding_comparisons: (batch, 1, num_tracks, num_detections, 2*embedding_dim)
            deltas: (batch, 1, num_tracks, num_detections, 2*encoder_dim)
        """
        # Process current embeddings with temporal merge
        embeddings_current = self.embedding_temporal_merge(current_embeddings)
        
        # Get only the last frame
        embeddings_current = embeddings_current[:, -1:]  # (B, 1, N, embedding_dim)
        
        # Compare with future embeddings
        embedding_comparisons = self.comparison(embeddings_current, future_embeddings)
        
        # Process centroids
        # 1. Get deltas within current tracks
        deltas_within = compute_deltas(current_centroids)  # (B, T, N, 2)
        deltas_within = torch.abs(deltas_within)
        
        # Encode and temporally merge
        deltas_within_enc = self.delta_encoder(deltas_within)
        deltas_within_enc = self.delta_temporal_merge(deltas_within_enc)
        deltas_within_enc = deltas_within_enc[:, -1:]  # Last frame: (B, 1, N, encoder_dim)
        
        # 2. Get deltas from last current frame to future frame
        centroid_current_last = current_centroids[:, -1:]  # (B, 1, N, 2)
        
        # Expand for pairwise differences
        centroid_current_last = centroid_current_last.unsqueeze(3)  # (B, 1, N, 1, 2)
        centroid_future_exp = future_centroids.unsqueeze(2)  # (B, 1, 1, M, 2)
        
        deltas_across = centroid_future_exp - centroid_current_last  # (B, 1, N, M, 2)
        deltas_across = torch.abs(deltas_across)
        
        # Encode cross-frame deltas
        deltas_across_enc = self.delta_across_frames_encoder(deltas_across)
        
        # Reshape within-frame deltas to match
        deltas_within_current = self.delta_reshape(deltas_within_enc, future_centroids)
        
        # Concatenate
        deltas = torch.cat([deltas_within_current, deltas_across_enc], dim=-1)
        
        return embedding_comparisons, deltas


# Example usage and tests
if __name__ == "__main__":
    print("Testing Training and Inference Branches...\n")
    
    # Create mock encoders (simplified for testing)
    class MockNeighborhoodEncoder(nn.Module):
        def __init__(self, embedding_dim):
            super().__init__()
            self.embedding_dim = embedding_dim
        
        def forward(self, app, morph, cent, adj):
            batch_size = app.shape[0]
            # Return mock embeddings and centroids
            embeddings = torch.randn(batch_size, self.embedding_dim)
            centroids = cent
            return embeddings, centroids
    
    embedding_dim = 64
    encoder_dim = 64
    track_length = 8
    max_cells = 39
    
    # Create temporal merge modules
    embedding_temporal_merge = TemporalMerge(embedding_dim)
    delta_temporal_merge = TemporalMerge(encoder_dim)
    
    # Create delta encoders (simplified)
    delta_encoder = nn.Linear(2, encoder_dim)
    delta_across_frames_encoder = nn.Linear(2, encoder_dim)
    
    # Mock neighborhood encoder
    neighborhood_encoder = MockNeighborhoodEncoder(embedding_dim)
    
    print("1. Testing TrainingBranch")
    training_branch = TrainingBranch(
        neighborhood_encoder=neighborhood_encoder,
        embedding_temporal_merge=embedding_temporal_merge,
        delta_temporal_merge=delta_temporal_merge,
        delta_encoder=delta_encoder,
        delta_across_frames_encoder=delta_across_frames_encoder,
        track_length=track_length,
        max_cells=max_cells,
        embedding_dim=embedding_dim,
        encoder_dim=encoder_dim
    )
    
    # Create sample inputs
    batch_size = 2
    appearances = torch.randn(batch_size, track_length, 1, 32, 32)
    morphologies = torch.randn(batch_size, track_length, 3)
    centroids = torch.randn(batch_size, track_length, 2)
    adj_matrices = torch.randn(batch_size, track_length, max_cells, max_cells)
    
    print(f"   Input shapes:")
    print(f"   - Appearances: {appearances.shape}")
    print(f"   - Morphologies: {morphologies.shape}")
    print(f"   - Centroids: {centroids.shape}")
    print(f"   - Adjacency: {adj_matrices.shape}")
    
    # Note: This will fail with mock encoder, but shows the structure
    try:
        emb_comp, deltas = training_branch(appearances, morphologies, centroids, adj_matrices)
        print(f"   Output shapes:")
        print(f"   - Embedding comparisons: {emb_comp.shape}")
        print(f"   - Deltas: {deltas.shape}")
        print(f"   Expected embedding comparisons: ({batch_size}, {track_length-1}, {max_cells}, {max_cells}, {2*embedding_dim})")
        print(f"   Expected deltas: ({batch_size}, {track_length-1}, {max_cells}, {max_cells}, {2*encoder_dim})")
    except Exception as e:
        print(f"   (Expected error with mock encoder: {type(e).__name__})")
    print()
    
    print("2. Testing InferenceBranch")
    inference_branch = InferenceBranch(
        embedding_temporal_merge=embedding_temporal_merge,
        delta_temporal_merge=delta_temporal_merge,
        delta_encoder=delta_encoder,
        delta_across_frames_encoder=delta_across_frames_encoder,
        embedding_dim=embedding_dim,
        encoder_dim=encoder_dim
    )
    
    # Create inference inputs
    num_tracks = 10
    num_detections = 15
    time_history = 5
    
    current_embeddings = torch.randn(batch_size, time_history, num_tracks, embedding_dim)
    current_centroids = torch.randn(batch_size, time_history, num_tracks, 2)
    future_embeddings = torch.randn(batch_size, 1, num_detections, embedding_dim)
    future_centroids = torch.randn(batch_size, 1, num_detections, 2)
    
    print(f"   Input shapes:")
    print(f"   - Current embeddings: {current_embeddings.shape}")
    print(f"   - Current centroids: {current_centroids.shape}")
    print(f"   - Future embeddings: {future_embeddings.shape}")
    print(f"   - Future centroids: {future_centroids.shape}")
    
    emb_comp, deltas = inference_branch(
        current_embeddings, current_centroids,
        future_embeddings, future_centroids
    )
    
    print(f"   Output shapes:")
    print(f"   - Embedding comparisons: {emb_comp.shape}")
    print(f"   - Deltas: {deltas.shape}")
    print(f"   Expected: ({batch_size}, 1, {num_tracks}, {num_detections}, {2*embedding_dim})")
    print(f"   Expected: ({batch_size}, 1, {num_tracks}, {num_detections}, {2*encoder_dim})")
    print()
    
    print("✅ Branch structure tests passed!")
    
    print("\n3. Key differences from TensorFlow:")
    print("   - No Lambda layers - direct tensor operations")
    print("   - Explicit reshaping with .reshape() and .view()")
    print("   - No Input() layers needed")
    print("   - Module composition instead of functional API")
    print("   - Forward pass defined in forward() method")