"""Custom PyTorch layers for cell tracking model"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageNormalization2D(nn.Module):
    """Image Normalization layer for 2D data.
    
    Args:
        norm_method (str): Normalization method to use, one of:
            "std", "max", "whole_image", None.
        filter_size (int): The length of the convolution window (not used in current impl).
        data_format (str): The layout of the input, 'channels_first' (B, C, H, W) or
            'channels_last' (B, H, W, C). Normalization is taken over the two spatial
            axes of whichever layout is declared, so the declaration has to be true.
            Anything else raises.
    """
    def __init__(
        self,
        norm_method='std',
        filter_size=61,
        data_format='channels_first'
    ):
        super().__init__()
        valid_modes = {'std', 'max', None, 'whole_image'}
        if norm_method not in valid_modes:
            raise ValueError(f'Invalid `norm_method`: "{norm_method}". '
                           f'Use one of {valid_modes}.')
        
        self.norm_method = norm_method.lower() if isinstance(norm_method, str) else norm_method
        self.filter_size = filter_size

        # This used to store the literal 'channels_first' whatever the caller passed, so
        # a channels_last crop was normalized over (width, channels) - one image row at
        # a time - instead of over its spatial extent. Divergence 80.
        supported = ('channels_first', 'channels_last')
        if data_format not in supported:
            raise ValueError(
                f'Invalid `data_format`: {data_format!r}. Use one of {supported}.'
            )
        self.data_format = data_format

    def forward(self, x):
        """Apply normalization to input tensor."""
        if self.norm_method is None:
            return x
        
        # Ensure channels_first for computation
        if self.data_format == 'channels_last':
            # Convert to channels_first: (B, H, W, C) -> (B, C, H, W)
            x = x.permute(0, 3, 1, 2)
        
        if self.norm_method == 'whole_image':
            # Normalize each image independently
            # Keep dims for channel and batch

            mean = x.mean(dim=(2, 3), keepdim=True)
            std = x.std(dim=(2, 3), keepdim=True)
            x = (x - mean) / (std + 1e-7)
        
        elif self.norm_method == 'std':
            # Standardize across spatial dimensions
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            std = x.std(dim=(1, 2, 3), keepdim=True)
            x = (x - mean) / (std + 1e-7)
        
        elif self.norm_method == 'max':
            # Scale to [0, 1] based on max
            min_val = x.amin(dim=(2, 3), keepdim=True)
            max_val = x.amax(dim=(2, 3), keepdim=True)
            x = (x - min_val) / (max_val - min_val + 1e-7)
        
        # Convert back if needed
        if self.data_format == 'channels_last':
            x = x.permute(0, 2, 3, 1)
        
        return x


class Comparison(nn.Module):
    """Layer for comparing two sequences of inputs.
    
    Expands and tiles x and y to create pairwise comparisons,
    then concatenates them along the last dimension.
    
    Input shapes:
        x: (batch, time, tracks_x, features)
        y: (batch, time, tracks_y, features)
    
    Output shape:
        (batch, time, tracks_x, tracks_y, features * 2)
    """
    def forward(self, x, y):

        # (B, T, X, F) -> (B, T, X, 1, F)
        x = x.unsqueeze(3)

        # (B, T, X, 1, F) -> (B, T, X, Y, F)
        x = x.expand(-1, -1, -1, y.shape[2], -1)
        
        # (B, T, Y, F) -> (B, T, 1, Y, F)
        y = y.unsqueeze(2)

        # (B, T, 1, Y, F) -> (B, T, X, Y, F)
        y = y.expand(-1, -1, x.shape[2], -1, -1)
        
        # Concatenate along feature dimension
        return torch.cat([x, y], dim=-1)


class DeltaReshape(nn.Module):
    """Reshape changes between current and future frames.
    
    Takes current embeddings and tiles them to match the shape of future frame.
    
    Input shapes:
        current: (batch, time, tracks, features)
        future: (batch, time, tracks_future, features)
    
    Output shape:
        (batch, time, tracks, tracks_future, features)
    """
    def forward(self, current, future):
        # Add dimension: (B, T, X, F) -> (B, T, X, 1, F)
        current = current.unsqueeze(3)
        # Tile to match future tracks: (B, T, X, 1, F) -> (B, T, X, Y, F)
        output = current.expand(-1, -1, -1, future.shape[2], -1)
        return output


class Unmerge(nn.Module):
    """Unmerge temporal inputs by reshaping.
    
    Reshapes from (batch, merged_temporal_tracks, features)
    to (batch, track_length, max_cells, features).
    
    Args:
        track_length (int): Length of each track sequence.
        max_cells (int): Maximum number of cells/tracks per frame.
        embedding_dim (int): Dimension of embeddings.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, batch_size, track_length, max_cells):
        return x.view(batch_size, track_length, max_cells, -1)


class TemporalMerge(nn.Module):
    """Layer for merging the time dimension of a Tensor using LSTM.
    
    Runs one LSTM sequence per (batch entry, track), over that track's frames in time
    order. A track's output depends only on its own frames - never on another slot.
    Initialized to match Keras LSTM defaults:
        - Glorot Uniform for input weights
        - Orthogonal for recurrent weights
        - Zero biases with forget gate bias set to 1.0
    
    Args:
        encoder_dim (int): Desired encoder dimension and LSTM hidden size.
    
    Input shape:
        (batch, time, tracks, encoder_dim)
    
    Output shape:
        (batch, time, tracks, encoder_dim)
    """
    def __init__(self, encoder_dim=64):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.lstm = nn.LSTM(
            input_size=encoder_dim,
            hidden_size=encoder_dim,
            batch_first=True
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                # Glorot Uniform — matches Keras `kernel_initializer='glorot_uniform'`
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                # Orthogonal — matches Keras `recurrent_initializer='orthogonal'`
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                # Zero init — matches Keras `bias_initializer='zeros'`
                nn.init.zeros_(param)
                # Forget gate bias set to 1.0 — matches Keras `unit_forget_bias=True`
                # PyTorch concatenates biases as [input, forget, cell, output]
                # There are two bias vectors (bias_ih, bias_hh); the convention is to
                # set the forget gate on bias_ih and leave bias_hh at zero
                if 'bias_ih' in name:
                    hidden_size = self.encoder_dim
                    param.data[hidden_size:hidden_size * 2].fill_(1.0)

    def forward(self, x):
        # x shape: (batch, time, tracks, encoder_dim)
        batch_size, time_steps, num_tracks, features = x.shape
        
        # One LSTM sequence per (batch entry, track): the track axis must come in front
        # of the time axis *before* the reshape. Reshaping (B, T, N, F) directly reads T
        # consecutive memory rows, which are T different tracks of one frame, so the LSTM
        # carried state from one slot to the next and a track's output depended on how
        # many other slots the batch held. Divergence 82.
        # (B, T, N, F) -> (B, N, T, F) -> (B*N, T, F)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * num_tracks, time_steps, features)
        
        # Apply LSTM
        x, _ = self.lstm(x)
        
        # Restore the layout
        # (B*N, T, F) -> (B, N, T, F) -> (B, T, N, F)
        x = x.reshape(batch_size, num_tracks, time_steps, self.encoder_dim).permute(0, 2, 1, 3)
        
        return x


# Utility functions to replace Lambda layers
def compute_deltas(x):
    """Compute deltas between consecutive time steps.
    
    Args:
        x: Tensor of shape (batch, time, ..., features)
    
    Returns:
        Tensor of same shape with deltas, padded at start.
    """
    # Compute differences: x[t] - x[t-1]
    deltas = x[:, 1:] - x[:, :-1]
    
    # Pad at the beginning with zeros
    pad_shape = [0, 0, 0, 0, 1, 0, 0, 0]
    deltas = F.pad(deltas, pad_shape)
    
    return deltas


def compute_deltas_across_frames(centroids):
    """Find deltas across frames between all pairs of tracks.
    
    Args:
        centroids: Tensor of shape (batch, time, tracks, 2)
    
    Returns:
        Tensor of shape (batch, time-1, tracks_current, tracks_future, 2)
        representing differences between all pairs across consecutive frames.
    """
    # Split into current and future frames
    centroid_current = centroids[:, :-1]  # (B, T-1, N, 2)
    centroid_future = centroids[:, 1:]    # (B, T-1, N, 2)
    
    # Add dimensions for broadcasting
    centroid_current = centroid_current.unsqueeze(3)  # (B, T-1, N, 1, 2)
    centroid_future = centroid_future.unsqueeze(2)    # (B, T-1, 1, N, 2)
    
    # Compute pairwise differences
    deltas = centroid_future - centroid_current  # (B, T-1, N, N, 2)
    
    return deltas


# Example usage and tests


# Shape checks for these modules live in tests/test_l1_model.py. They used to sit in
# a `__main__` block here, which nothing ran and which had gone stale: it constructed
# modules with arguments they no longer take. Divergence 64.
