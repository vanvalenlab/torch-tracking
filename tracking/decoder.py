"""Tracking decoder module for cell tracking model in PyTorch"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrackingDecoder(nn.Module):
    """Decoder that predicts tracking associations between frames.
    
    Takes embedding comparisons and delta features to predict whether cells
    across frames should be linked (same cell, different cell, or no link).
    
    The decoder processes pairwise comparisons between tracks in consecutive
    frames and outputs a probability distribution over tracking classes.
    
    Args:
        embedding_dim (int): Dimension of embedding features
        encoder_dim (int): Dimension of delta/position features
        n_filters (int): Number of hidden units in intermediate layers
        n_classes (int): Number of output classes (default: 3)
            - Class 0: No link, different cell
            - Class 1: Same cell
            - Class 2: Daughter cell
        norm_layer (str): 'batch' or 'layer' normalization
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
        """
        Args:
            embedding_comparison: Tensor of shape 
                (batch, time, tracks_current, tracks_future, 2*embedding_dim)
                Pairwise comparisons of embeddings
            deltas: Tensor of shape 
                (batch, time, tracks_current, tracks_future, 2*encoder_dim)
                Position delta features
            apply_softmax: Whether to apply softmax (True for inference, 
                False for training with CrossEntropyLoss)
        
        Returns:
            Tensor of shape (batch, time, tracks_current, tracks_future, n_classes)
            Probability distribution over tracking classes for each pair
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


class TrackingDecoderWithAttention(nn.Module):
    """Enhanced tracking decoder with attention mechanism.
    
    Similar to TrackingDecoder but adds an attention layer to better
    capture relationships between embedding and spatial features.
    
    Args:
        embedding_dim (int): Dimension of embedding features
        encoder_dim (int): Dimension of delta/position features
        n_filters (int): Number of hidden units in intermediate layers
        n_classes (int): Number of output classes (default: 3)
        norm_layer (str): 'batch' or 'layer' normalization
        attention_heads (int): Number of attention heads (default: 4)
    """
    def __init__(
        self,
        embedding_dim=64,
        encoder_dim=64,
        n_filters=64,
        n_classes=3,
        norm_layer='batch',
        attention_heads=4
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder_dim
        self.n_filters = n_filters
        self.n_classes = n_classes
        
        # Input dimension
        input_dim = 2 * embedding_dim + 2 * encoder_dim
        
        # Multi-head attention for feature interaction
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=attention_heads,
            batch_first=True
        )
        
        # Dense layers
        self.dense1 = nn.Linear(input_dim, n_filters)
        self.norm1 = (nn.BatchNorm1d(n_filters) if norm_layer == 'batch' 
                     else nn.LayerNorm(n_filters))
        self.activation1 = nn.ReLU()
        
        self.dense2 = nn.Linear(n_filters, n_filters)
        self.norm2 = (nn.BatchNorm1d(n_filters) if norm_layer == 'batch' 
                     else nn.LayerNorm(n_filters))
        self.activation2 = nn.ReLU()
        
        # Output layer
        self.dense_out = nn.Linear(n_filters, n_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, embedding_comparison, deltas, apply_softmax=True):
        """
        Args:
            embedding_comparison: (batch, time, tracks_current, tracks_future, 2*embedding_dim)
            deltas: (batch, time, tracks_current, tracks_future, 2*encoder_dim)
            apply_softmax: Whether to apply softmax
        
        Returns:
            Tensor of shape (batch, time, tracks_current, tracks_future, n_classes)
        """
        # Concatenate features
        x = torch.cat([embedding_comparison, deltas], dim=-1)
        
        batch_size, time_steps, tracks_current, tracks_future, features = x.shape
        
        # Reshape for attention: treat each (current, future) pair as sequence
        x = x.view(batch_size * time_steps, tracks_current * tracks_future, features)
        
        # Self-attention
        x_attn, _ = self.attention(x, x, x)
        
        # Residual connection
        x = x + x_attn
        
        # Flatten for dense layers
        x = x.view(-1, features)
        
        # First dense block
        x = self.dense1(x)
        if isinstance(self.norm1, nn.BatchNorm1d):
            x = self.norm1(x)
        else:
            x = self.norm1(x)
        x = self.activation1(x)
        
        # Second dense block
        x = self.dense2(x)
        if isinstance(self.norm2, nn.BatchNorm1d):
            x = self.norm2(x)
        else:
            x = self.norm2(x)
        x = self.activation2(x)
        
        # Output layer
        x = self.dense_out(x)
        
        # Reshape back
        x = x.view(batch_size, time_steps, tracks_current, tracks_future, self.n_classes)
        
        # Apply softmax if requested
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
    
    # Test TrackingDecoderWithAttention
    print("2. TrackingDecoderWithAttention")
    decoder_attn = TrackingDecoderWithAttention(
        embedding_dim=64,
        encoder_dim=64,
        n_filters=64,
        n_classes=3,
        attention_heads=4
    )
    
    output_attn = decoder_attn(embedding_comp, deltas, apply_softmax=True)
    print(f"   Output with attention: {output_attn.shape}")
    print(f"   Output sum along class dim: {output_attn[0, 0, 0, 0].sum():.4f}")
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
    
    total_params_attn = sum(p.numel() for p in decoder_attn.parameters())
    print(f"Total parameters in attention decoder: {total_params_attn:,}")