"""Custom PyTorch layers for cell tracking model"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageNormalization2D(nn.Module):
    """Normalize 2D image tensors using a per-image statistic.

    Parameters
    ----------
    norm_method : str or None, optional
        Normalization method to use, one of ``"std"``, ``"max"``,
        ``"whole_image"``, or ``None``. Default is ``"std"``.
    filter_size : int, optional
        The length of the convolution window (not used in the current
        implementation). Default is 61.
    data_format : str, optional
        ``"channels_first"`` or ``"channels_last"``. Default is
        ``"channels_first"``.

    Returns
    -------
    ImageNormalization2D
        An initialized ``ImageNormalization2D`` layer.
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
        self.data_format = 'channels_first'
        
        # Channel axis: 1 for channels_first, -1 for channels_last
        self.channel_axis = 1 if data_format == 'channels_first' else -1

    def forward(self, x):
        """Apply normalization to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Image tensor with shape ``(B, C, H, W)`` or ``(B, H, W, C)``
            depending on ``self.data_format``.

        Returns
        -------
        torch.Tensor
            The normalized tensor, with the same shape and layout as ``x``.
        """
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

    Expands and tiles ``x`` and ``y`` to create pairwise comparisons,
    then concatenates them along the last dimension.

    Returns
    -------
    Comparison
        An initialized ``Comparison`` layer.
    """
    def forward(self, x, y):
        """Compute pairwise concatenated features between ``x`` and ``y``.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(batch, time, tracks_x, features)``.
        y : torch.Tensor
            Tensor of shape ``(batch, time, tracks_y, features)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, time, tracks_x, tracks_y, features * 2)``
            containing the pairwise concatenation of ``x`` and ``y`` along
            the feature dimension.
        """
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

    Takes current embeddings and tiles them to match the shape of the
    future frame.

    Returns
    -------
    DeltaReshape
        An initialized ``DeltaReshape`` layer.
    """
    def forward(self, current, future):
        """Tile ``current`` along a new axis to match ``future``'s track count.

        Parameters
        ----------
        current : torch.Tensor
            Tensor of shape ``(batch, time, tracks, features)``.
        future : torch.Tensor
            Tensor of shape ``(batch, time, tracks_future, features)``. Only
            its size along the tracks dimension is used.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, time, tracks, tracks_future, features)``,
            with ``current`` broadcast along the new ``tracks_future`` axis.
        """
        # Add dimension: (B, T, X, F) -> (B, T, X, 1, F)
        current = current.unsqueeze(3)
        # Tile to match future tracks: (B, T, X, 1, F) -> (B, T, X, Y, F)
        output = current.expand(-1, -1, -1, future.shape[2], -1)
        return output


class Unmerge(nn.Module):
    """Unmerge temporal inputs by reshaping.

    Reshapes from ``(batch, merged_temporal_tracks, features)`` to
    ``(batch, track_length, max_cells, features)``.

    Returns
    -------
    Unmerge
        An initialized ``Unmerge`` layer.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, batch_size, track_length, max_cells):
        """Reshape a merged temporal tensor back into separate time and track axes.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(batch, track_length * max_cells, features)``.
        batch_size : int
            The batch size.
        track_length : int
            Length of each track sequence (the desired time dimension).
        max_cells : int
            Maximum number of cells/tracks per frame.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch_size, track_length, max_cells, features)``.
        """
        return x.view(batch_size, track_length, max_cells, -1)


class TemporalMerge(nn.Module):
    """Merge the time dimension of a tensor using an LSTM.

    Processes temporal sequences at each spatial location (track)
    independently. Weights are initialized to match Keras LSTM defaults:

    - Glorot Uniform for input weights
    - Orthogonal for recurrent weights
    - Zero biases, with the forget gate bias set to 1.0

    Parameters
    ----------
    encoder_dim : int, optional
        Desired encoder dimension and LSTM hidden size. Default is 64.

    Returns
    -------
    TemporalMerge
        An initialized ``TemporalMerge`` layer.
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
        """Run the LSTM over the time axis independently for each track.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(batch, time, tracks, encoder_dim)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, time, tracks, encoder_dim)`` containing
            the LSTM hidden states at each time step.
        """
        # x shape: (batch, time, tracks, encoder_dim)
        batch_size, time_steps, num_tracks, features = x.shape
        
        # Reshape to process all tracks independently
        # (B, T, N, F) -> (B*N, T, F)
        x = x.reshape(batch_size * num_tracks, time_steps, features)
        
        # Apply LSTM
        x, _ = self.lstm(x)
        
        # Reshape back
        # (B*N, T, F) -> (B, T, N, F)
        x = x.reshape(batch_size, time_steps, num_tracks, self.encoder_dim)
        
        return x


# Utility functions to replace Lambda layers
def compute_deltas(x):
    """Compute deltas between consecutive time steps.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape ``(batch, time, ..., features)``.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape as ``x`` containing ``x[:, t] - x[:, t-1]``
        for each time step, zero-padded at the first time step.
    """
    # Compute differences: x[t] - x[t-1]
    deltas = x[:, 1:] - x[:, :-1]
    
    # Pad at the beginning with zeros
    pad_shape = [0, 0, 0, 0, 1, 0, 0, 0]
    deltas = F.pad(deltas, pad_shape)
    
    return deltas


def compute_deltas_across_frames(centroids):
    """Find deltas across frames between all pairs of tracks.

    Parameters
    ----------
    centroids : torch.Tensor
        Tensor of shape ``(batch, time, tracks, 2)``.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(batch, time - 1, tracks, tracks, 2)`` giving the
        pairwise centroid differences (``future - current``) between every
        pair of tracks across each pair of consecutive frames.
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
if __name__ == "__main__":
    print("Testing custom PyTorch layers...\n")
    
    # Test ImageNormalization2D
    print("1. ImageNormalization2D")
    img_norm = ImageNormalization2D(norm_method='whole_image')
    x = torch.randn(2, 3, 32, 32)
    out = img_norm(x)
    print(f"   Input: {x.shape}, Output: {out.shape}")
    print(f"   Output mean: {out.mean():.4f}, std: {out.std():.4f}\n")
    
    # Test Comparison
    print("2. Comparison")
    comp = Comparison()
    x = torch.randn(2, 4, 5, 64)  # batch, time, tracks_x, features
    y = torch.randn(2, 4, 3, 64)  # batch, time, tracks_y, features
    out = comp(x, y)
    print(f"   x: {x.shape}, y: {y.shape}")
    print(f"   Output: {out.shape} (expected: [2, 4, 5, 3, 128])\n")
    
    # Test DeltaReshape
    print("3. DeltaReshape")
    delta_reshape = DeltaReshape()
    current = torch.randn(2, 4, 5, 64)
    future = torch.randn(2, 4, 3, 64)
    out = delta_reshape(current, future)
    print(f"   Current: {current.shape}, Future: {future.shape}")
    print(f"   Output: {out.shape} (expected: [2, 4, 5, 3, 64])\n")
    
    # Test Unmerge
    print("4. Unmerge")
    unmerge = Unmerge(track_length=8, max_cells=39, embedding_dim=64)
    x = torch.randn(2, 312, 64)  # batch, (8*39), features
    out = unmerge(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {out.shape} (expected: [2, 8, 39, 64])\n")
    
    # Test TemporalMerge
    print("5. TemporalMerge")
    temp_merge = TemporalMerge(encoder_dim=64)
    x = torch.randn(2, 8, 39, 64)  # batch, time, tracks, features
    out = temp_merge(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {out.shape} (expected: [2, 8, 39, 64])\n")
    
    # Test compute_deltas
    print("6. compute_deltas")
    x = torch.randn(2, 8, 39, 2)
    deltas = compute_deltas(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {deltas.shape} (expected: [2, 8, 39, 2])\n")
    
    # Test compute_deltas_across_frames
    print("7. compute_deltas_across_frames")
    centroids = torch.randn(2, 8, 39, 2)
    deltas = compute_deltas_across_frames(centroids)
    print(f"   Input: {centroids.shape}")
    print(f"   Output: {deltas.shape} (expected: [2, 7, 39, 39, 2])\n")
    
    print("✅ All tests passed!")