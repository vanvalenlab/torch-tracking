"""Tracking decoder module for cell tracking model in PyTorch"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrackingDecoder(nn.Module):
    """Decoder that predicts tracking associations between frames.

    Takes embedding comparisons and delta features to predict whether cells
    across frames should be linked (same cell, different cell, or a
    parent-daughter division).

    The decoder processes pairwise comparisons between tracks in consecutive
    frames and outputs a probability distribution over tracking classes.

    Parameters
    ----------
    embedding_dim : int, optional
        Dimension of the embedding and delta/position features; used to
        compute the input dimension of the first dense layer. Default is
        64.
    n_filters : int, optional
        Number of hidden units in the intermediate dense layer. Default is
        64.
    n_classes : int, optional
        Number of output classes. Default is 3:

        - 0 : no link, different cell
        - 1 : same cell
        - 2 : daughter cell (division)
    norm_layer : str, optional
        Normalization applied after the first dense layer, either
        ``batch`` for ``nn.BatchNorm1d`` or ``layer`` for ``nn.LayerNorm``.
        Default is ``batch``.
    dropout : float, optional
        Dropout probability applied after normalization and activation.
        Default is 0.1.

    Returns
    -------
    TrackingDecoder
        An initialized ``TrackingDecoder`` object.
    """
    def __init__(
        self,
        embedding_dim=64,
        n_filters=64,
        n_classes=3,
        norm_layer='batch',
        dropout=0.1
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_filters = n_filters
        self.n_classes = n_classes
        
        # Input dimension: concatenated embeddings (2x) + deltas (2x)
        input_dim = 4 * self.embedding_dim
        
        # First dense layer
        self.dense1 = nn.Linear(input_dim, n_filters)
        self.norm1 = (nn.BatchNorm1d(n_filters) if norm_layer == 'batch' 
                     else nn.LayerNorm(n_filters))
        self.activation1 = nn.ReLU()
        
        # Output layer
        self.dense_out = nn.Linear(n_filters, n_classes)
        
        # Softmax is typically applied in loss function for numerical stability
        # but we include it here for inference
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedding_comparison, deltas, apply_softmax=True):
        """Predict tracking-association scores for pairs of cells.

        Parameters
        ----------
        embedding_comparison : torch.Tensor
            Pairwise comparisons of embeddings, of shape
            ``(batch, time, tracks_current, tracks_future, 2 * embedding_dim)``.
        deltas : torch.Tensor
            Position delta features, of shape
            ``(batch, time, tracks_current, tracks_future, 2 * embedding_dim)``.
        apply_softmax : bool, optional
            Whether to apply softmax to the output. Use ``True`` for
            inference and ``False`` when training with a loss function
            (e.g. ``nn.CrossEntropyLoss``) that expects raw logits. Default
            is True.

        Returns
        -------
        torch.Tensor
            Tensor of shape
            ``(batch, time, tracks_current, tracks_future, n_classes)``. If
            ``apply_softmax`` is True, this is a probability distribution
            over tracking classes for each pair; otherwise it contains raw
            logits.
        """
        # Concatenate embedding comparisons and deltas
        x = torch.cat([embedding_comparison, deltas], dim=-1)
        
        # Get shape info
        batch_size, time_steps, tracks_current, tracks_future, features = x.shape
        
        # Flatten for dense layer: (B, T, X, Y, F) -> (B*T*X*Y, F)
        x = x.view(-1, features)
        
        # First dense layer with normalization
        x = self.dense1(x)
        
        # Apply normalization
        if isinstance(self.norm1, nn.BatchNorm1d):
            x = self.norm1(x)
        else:
            # LayerNorm works directly on flattened features
            x = self.norm1(x)
        
        x = self.activation1(x)

        x = self.dropout(x)
        
        # Output layer
        x = self.dense_out(x)
        
        # Reshape back: (B*T*X*Y, C) -> (B, T, X, Y, C)
        x = x.view(batch_size, time_steps, tracks_current, tracks_future, self.n_classes)
        
        # Apply softmax if requested (for inference)
        if apply_softmax:
            x = self.softmax(x)
        
        return x


# Example usage and tests
if __name__ == "__main__":
    print("Testing tracking decoder modules...\n")
    
    # Test basic TrackingDecoder
    print("1. TrackingDecoder (basic)")
    decoder = TrackingDecoder(
        embedding_dim=64,
        encoder_dim=64,
        n_filters=64,
        n_classes=3
    )
    
    # Create sample inputs
    # Shape: (batch, time-1, tracks_current, tracks_future, features)
    embedding_comp = torch.randn(2, 7, 39, 39, 128)  # 2*64
    deltas = torch.randn(2, 7, 39, 39, 128)  # 2*64
    
    # Forward pass with softmax (inference mode)
    output = decoder(embedding_comp, deltas, apply_softmax=True)
    print(f"   Embedding comparison: {embedding_comp.shape}")
    print(f"   Deltas: {deltas.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Expected: (2, 7, 39, 39, 3)")
    print(f"   Output sum along class dim: {output[0, 0, 0, 0].sum():.4f} (should be ~1.0)")
    print()
    
    # Test without softmax (training mode)
    logits = decoder(embedding_comp, deltas, apply_softmax=False)
    print(f"   Logits (no softmax): {logits.shape}")
    print(f"   Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
    print()
    
    # Test with different batch sizes
    print("3. Testing different input sizes")
    embedding_comp_small = torch.randn(1, 3, 10, 15, 128)
    deltas_small = torch.randn(1, 3, 10, 15, 128)
    output_small = decoder(embedding_comp_small, deltas_small)
    print(f"   Small input: {embedding_comp_small.shape} -> {output_small.shape}")
    print()
    
    # Demonstrate usage with loss function
    print("4. Example training setup")
    print("   For training, use apply_softmax=False and CrossEntropyLoss:")
    print("   ")
    print("   # Get logits")
    print("   logits = decoder(embeddings, deltas, apply_softmax=False)")
    print("   ")
    print("   # Reshape for loss: (B, T, X, Y, C) -> (B*T*X*Y, C)")
    print("   logits_flat = logits.reshape(-1, n_classes)")
    print("   targets_flat = targets.reshape(-1)  # (B*T*X*Y)")
    print("   ")
    print("   # Compute loss")
    print("   criterion = nn.CrossEntropyLoss()")
    print("   loss = criterion(logits_flat, targets_flat)")
    print()
    
    print("✅ All decoder tests passed!")
    
    # Show parameter count
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nTotal parameters in basic decoder: {total_params:,}")
    