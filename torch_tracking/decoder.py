"""Tracking decoder module for cell tracking model in PyTorch"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_tracking.utils import masked_apply


class TrackingDecoder(nn.Module):
    """Decoder that predicts tracking associations between frames.
    
    Takes embedding comparisons and delta features to predict whether cells
    across frames should be linked (same cell, different cell, or no link).
    
    The decoder processes pairwise comparisons between tracks in consecutive
    frames and outputs a probability distribution over tracking classes.
    
    Args:
        embedding_dim (int): Dimension of embedding features
        n_filters (int): Number of hidden units in intermediate layers
        n_classes (int): Number of output classes (default: 3)
            - Class 0: No link, different cell
            - Class 1: Same cell
            - Class 2: Daughter cell
        norm_layer (str): 'batch' or 'layer' normalization
        dropout (float): Dropout probability applied before the output layer
        encoder_dim (int): Dimension of the delta/position features. Defaults to
            `embedding_dim`, which is what the model passes
    """
    def __init__(
        self,
        embedding_dim=64,
        n_filters=64,
        n_classes=3,
        norm_layer='batch',
        dropout=0.1,
        encoder_dim=None
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder_dim = embedding_dim if encoder_dim is None else encoder_dim
        self.n_filters = n_filters
        self.n_classes = n_classes

        # The decoder is handed a pairwise embedding comparison (two embeddings wide) and
        # a pairwise delta (two delta encodings wide), concatenated. Sizing from both
        # widths, rather than `4 * embedding_dim`, stays correct when the delta encoders
        # are not as wide as the embeddings.
        input_dim = 2 * self.embedding_dim + 2 * self.encoder_dim
        
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

    def forward(self, embedding_comparison, deltas, apply_softmax=True, is_real=None):
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
            is_real: Optional boolean mask ``(batch, time, tracks_current,
                tracks_future)``, ``True`` where both ends of the pair are real cells. The
                normalization takes its statistics over those pairs only; a pair with a
                padding end is never graded, so letting it set the statistics would let
                padding decide how real pairs are scored. ``None`` means every pair is
                real, which is the inference branch's case.
        
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
            if is_real is None:
                x = self.norm1(x)
            else:
                x = masked_apply(self.norm1, x, is_real.reshape(-1))
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
