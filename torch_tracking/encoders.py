"""Encoder modules for cell tracking model in PyTorch"""

import math
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, Batch
from torch_tracking.utils import normalize_adjacency_symmetric

from torch_tracking.layers import ImageNormalization2D


class AppearanceEncoder(nn.Module):
    """Encode cell appearance image crops using 3D convolutions.

    Uses ``Conv3d`` with kernel ``(1, 3, 3)`` so the cell dimension ``N`` is
    treated as a pseudo-temporal axis with no mixing between cells, while
    the spatial dimensions are convolved and pooled normally.

    Parameters
    ----------
    appearance_shape : tuple of int, optional
        Shape of a single appearance crop, ``(H, W, C)``. Default is
        ``(32, 32, 1)``.
    n_filters : int, optional
        Number of convolutional filters in each block. Default is 64.
    encoder_dim : int, optional
        Dimension of the output embedding. Default is 64.
    norm_layer : str, optional
        Normalization type. Currently unused; the module always applies
        ``BatchNorm3d``/``BatchNorm1d``. Default is ``'batch'``.
    appearance_norm : bool, optional
        Whether to apply per-cell whole-image normalization to the input
        before convolving. Default is True.
    data_format : str, optional
        Layout of the input tensor passed to ``forward``, either
        ``'channels_first'`` or ``'channels_last'``. Default is
        ``'channels_first'``.
    dropout : float, optional
        Dropout probability. Currently stored but not applied within this
        module. Default is 0.1.

    Returns
    -------
    AppearanceEncoder
        An initialized ``AppearanceEncoder`` module.
    """
    def __init__(
        self,
        appearance_shape=(32, 32, 1),
        n_filters=64,
        encoder_dim=64,
        norm_layer='batch',
        appearance_norm=True,
        data_format='channels_first',
        dropout = 0.1
    ):
        super().__init__()
        self.appearance_shape = appearance_shape
        self.n_filters = n_filters
        self.encoder_dim = encoder_dim
        self.appearance_norm = appearance_norm
        self.data_format = data_format
        self.dropout = dropout
        
        # Calculate number of pooling layers based on spatial dimensions
        spatial_dim = appearance_shape[0]  # Assuming square images
        self.n_layers = int(math.log2(spatial_dim))
        
        # Input normalization (applied per-cell if enabled)
        if self.appearance_norm:
            self.img_norm = ImageNormalization2D(
                norm_method='whole_image',
                data_format=data_format
            )
        
        # Determine input channels
        in_channels = appearance_shape[-1] if data_format == 'channels_last' else appearance_shape[0]
        
        # Build 3D convolutional layers
        # Conv3D kernel: (depth/time, height, width)
        # We use (1, 3, 3) - no conv across cell dimension, only spatial
        self.conv_blocks = nn.ModuleList()
        for i in range(self.n_layers):
            block = nn.Sequential(
                nn.Conv3d(
                    in_channels if i == 0 else n_filters,
                    n_filters,
                    kernel_size=(1, 3, 3),  # CRITICAL: (cells=1, height=3, width=3)
                    stride=1,
                    padding='same',  # No padding on cell dimension
                    bias=False
                ),
                nn.BatchNorm3d(n_filters),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(1, 2, 2))  # Pool only spatial dims
            )
            self.conv_blocks.append(block)
        
        # Final dense layer
        self.dense = nn.Linear(n_filters, encoder_dim)
        self.final_norm = nn.BatchNorm1d(encoder_dim)
        self.final_activation = nn.ReLU()

    def forward(self, x):
        """Encode a batch of cell appearance crops into embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Appearance crops with shape ``(B*T, N, H, W, C)`` if
            ``data_format`` is ``'channels_last'``, or ``(B*T, N, C, H, W)``
            if ``'channels_first'``, where ``B*T`` is the flattened
            batch/time dimension and ``N`` is the number of cells.

        Returns
        -------
        torch.Tensor
            Appearance embeddings with shape ``(B*T, N, encoder_dim)``.
        """
        BT, N = x.shape[:2]
        
        # Apply normalization if enabled
        # Normalization operates on 2D images, so we need to flatten cells
        if self.appearance_norm:
            if self.data_format == 'channels_last':
                # (B*T, N, H, W, C) -> (B*T*N, H, W, C)
                orig_shape = x.shape
                x = x.reshape(-1, *x.shape[2:])
                x = self.img_norm(x)
                # Reshape back: (B*T*N, H, W, C) -> (B*T, N, H, W, C)
                x = x.reshape(orig_shape)
                # Convert to channels_first for Conv3d
                x = x.permute(0, 1, 4, 2, 3)  # (B*T, N, C, H, W)
            else:
                # (B*T, N, C, H, W) -> (B*T*N, C, H, W)
                orig_shape = x.shape
                x = x.reshape(-1, *x.shape[2:])
                x = self.img_norm(x)
                # Reshape back: (B*T*N, C, H, W) -> (B*T, N, C, H, W)
                x = x.reshape(orig_shape)
        else:
            if self.data_format == 'channels_last':
                # Convert to channels_first: (B*T, N, H, W, C) -> (B*T, N, C, H, W)
                x = x.permute(0, 1, 4, 2, 3)

        # We need (B*T, C, N, H, W) - swap channels and cells
        x = x.permute(0, 2, 1, 3, 4)  # (B*T, C, N, H, W)

        # Apply conv blocks
        for block in self.conv_blocks:
            x = block(x)

        # Global spatial pooling: (B*T, filters, N, 1, 1) -> (B*T, filters, N)
        x = x.squeeze(-1).squeeze(-1)
        
        # Permute back: (B*T, filters, N) -> (B*T, N, filters)
        x = x.permute(0, 2, 1)

        # Reshape for dense layer: (B*T, N, filters) -> (B*T*N, filters)
        x = x.reshape(-1, x.shape[-1])
        x = self.dense(x)
        x = self.final_norm(x)
        x = self.final_activation(x)
        
        # Reshape back: (B*T*N, encoder_dim) -> (B*T, N, encoder_dim)
        x = x.view(BT, N, self.encoder_dim)
        
        return x


class MorphologyEncoder(nn.Module):
    """Encode per-cell morphology features with a shared 1x1 convolution.

    Applies a ``Conv1d`` with ``kernel_size=1`` across the cell dimension,
    which is equivalent to a dense layer applied independently to each
    cell, preserving the cell dimension throughout processing.

    Parameters
    ----------
    input_dim : int, optional
        Dimension of the input morphology features. Default is 3.
    encoder_dim : int, optional
        Dimension of the output embedding. Default is 64.
    norm_layer : str, optional
        Normalization type. Currently unused; the module always applies
        ``BatchNorm1d``. Default is ``'batch'``.

    Returns
    -------
    MorphologyEncoder
        An initialized ``MorphologyEncoder`` module.
    """
    def __init__(self, input_dim=3, encoder_dim=64, norm_layer='batch'):
        super().__init__()
        self.encoder_dim = encoder_dim
        
        # Conv1d with kernel_size=1 is equivalent to Dense applied per-cell
        # Input: (batch, channels, length) = (B*T, input_dim, N)
        # Output: (batch, out_channels, length) = (B*T, encoder_dim, N)
        self.conv = nn.Conv1d(input_dim, encoder_dim, kernel_size=1)
        self.norm = nn.BatchNorm1d(encoder_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        """Encode per-cell morphology features.

        Parameters
        ----------
        x : torch.Tensor
            Morphology features with shape ``(B*T, N, input_dim)``.

        Returns
        -------
        torch.Tensor
            Morphology embeddings with shape ``(B*T, N, encoder_dim)``.
        """
        # Conv1d expects (batch, channels, sequence)
        # Permute: (B*T, N, 3) -> (B*T, 3, N)
        x = x.permute(0, 2, 1)
        
        # Apply conv: (B*T, 3, N) -> (B*T, encoder_dim, N)
        x = self.conv(x)
        
        # BatchNorm1d normalizes across (B*T, N) for each channel
        x = self.norm(x)
        x = self.activation(x)
        
        # Permute back: (B*T, encoder_dim, N) -> (B*T, N, encoder_dim)
        x = x.permute(0, 2, 1)
        
        return x


class CentroidEncoder(nn.Module):
    """Encode per-cell centroid positions with a shared 1x1 convolution.

    Applies a ``Conv1d`` with ``kernel_size=1`` across the cell dimension,
    which is equivalent to a dense layer applied independently to each
    cell, preserving the cell dimension throughout processing.

    Parameters
    ----------
    input_dim : int, optional
        Dimension of the input centroid features. Default is 2 (x, y).
    encoder_dim : int, optional
        Dimension of the output embedding. Default is 64.
    norm_layer : str, optional
        Normalization type. Currently unused; the module always applies
        ``BatchNorm1d``. Default is ``'batch'``.

    Returns
    -------
    CentroidEncoder
        An initialized ``CentroidEncoder`` module.
    """
    def __init__(self, input_dim=2, encoder_dim=64, norm_layer='batch'):
        super().__init__()
        self.encoder_dim = encoder_dim
        
        # Conv1d with kernel_size=1 is equivalent to Dense applied per-cell
        # Input: (batch, channels, length) = (B*T, input_dim, N)
        # Output: (batch, out_channels, length) = (B*T, encoder_dim, N)
        self.conv = nn.Conv1d(input_dim, encoder_dim, kernel_size=1)
        self.norm = nn.BatchNorm1d(encoder_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        """Encode per-cell centroid positions.

        Parameters
        ----------
        x : torch.Tensor
            Centroid positions with shape ``(B*T, N, input_dim)``.

        Returns
        -------
        torch.Tensor
            Centroid embeddings with shape ``(B*T, N, encoder_dim)``.
        """
        # Conv1d expects (batch, channels, sequence)
        # Permute: (B*T, N, 2) -> (B*T, 2, N)
        x = x.permute(0, 2, 1)
        
        # Apply conv: (B*T, 2, N) -> (B*T, encoder_dim, N)
        x = self.conv(x)
        
        # BatchNorm1d normalizes across (B*T, N) for each channel
        x = self.norm(x)
        x = self.activation(x)
        
        # Permute back: (B*T, encoder_dim, N) -> (B*T, N, encoder_dim)
        x = x.permute(0, 2, 1)
        
        return x


class DeltaEncoder(nn.Module):
    """Encode position deltas shared across time-steps and track pairs.

    Encodes changes in position between time steps or between tracks using
    a single dense layer with shared weights for both delta types.

    Parameters
    ----------
    input_dim : int, optional
        Dimension of the input delta features. Default is 2.
    encoder_dim : int, optional
        Dimension of the output embedding. Default is 64.
    norm_layer : str, optional
        Normalization applied after the dense layer: ``'batch'`` for
        ``BatchNorm1d``, or any other value for ``LayerNorm``. Default is
        ``'batch'``.
    dropout : float, optional
        Dropout probability applied after normalization. Default is 0.1.

    Returns
    -------
    DeltaEncoder
        An initialized ``DeltaEncoder`` module.
    """
    def __init__(self, input_dim=2, encoder_dim=64, norm_layer='batch', dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(input_dim, encoder_dim)
        self.activation = nn.ReLU()
        self.norm_layer_type = norm_layer
        self.norm = nn.BatchNorm1d(encoder_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        """Encode delta features with a shared dense layer.

        Parameters
        ----------
        x : torch.Tensor
            Delta features with shape ``(batch, time, tracks, input_dim)``
            or ``(batch, time, tracks_1, tracks_2, input_dim)``.

        Returns
        -------
        torch.Tensor
            Encoded deltas with the same leading dimensions as ``x`` but
            with ``encoder_dim`` in the final dimension.
        """
        original_shape = x.shape
        
        # Flatten all but last dimension
        x = x.view(-1, original_shape[-1])
        x = self.dense(x)
        
        # Reshape back
        new_shape = list(original_shape[:-1]) + [x.shape[-1]]
        x = x.view(new_shape)
        
        # Apply normalization - LayerNorm is easier for variable shapes

        if self.norm_layer_type == 'batch':
            # Flatten for BatchNorm1d: needs (N, C) or (N, C, L)
            x_flat = x.reshape(-1, x.shape[-1])
            x_flat = self.norm(x_flat)
            x = x_flat.view(new_shape)
        else:
            x = self.norm(x)
                
        x = self.dropout(x)
        x = self.activation(x)

        return x


class NeighborhoodEncoder(nn.Module):
    """Combine appearance, morphology, and centroid embeddings with a GNN.

    Concatenates the outputs of the appearance, morphology, and centroid
    encoders, projects them to a shared dimension, and applies a stack of
    graph layers over the cell adjacency graph to produce embeddings that
    incorporate neighborhood context.

    Parameters
    ----------
    appearance_encoder : nn.Module
        Encoder module used to embed appearance crops (e.g.
        ``AppearanceEncoder``).
    morphology_encoder : nn.Module
        Encoder module used to embed morphology features (e.g.
        ``MorphologyEncoder``).
    centroid_encoder : nn.Module
        Encoder module used to embed centroid positions (e.g.
        ``CentroidEncoder``).
    n_filters : int, optional
        Number of channels used in the initial projection and graph
        layers. Default is 64.
    embedding_dim : int, optional
        Dimension of the final output embedding. Default is 64.
    n_layers : int, optional
        Number of stacked graph layers. Default is 3.
    graph_layer : str, optional
        Type of graph layer to use, ``'gcn'`` or ``'gat'`` (optionally
        suffixed, e.g. ``'gat-2'``; only the prefix before ``'-'`` is
        used). Default is ``'gcn'``.
    norm_layer : str, optional
        Normalization type: ``'batch'`` for ``BatchNorm1d``, or any other
        value for ``LayerNorm``. Default is ``'batch'``.
    dropout : float, optional
        Dropout probability applied after each graph layer. Default is
        0.1.

    Returns
    -------
    NeighborhoodEncoder
        An initialized ``NeighborhoodEncoder`` module.

    Raises
    ------
    ValueError
        If ``graph_layer`` is not ``'gcn'`` or ``'gat'``.
    """
    def __init__(
        self,
        appearance_encoder,
        morphology_encoder,
        centroid_encoder,
        n_filters=64,
        embedding_dim=64,
        n_layers=3,
        graph_layer='gcn',
        norm_layer='batch',
        dropout=0.1
    ):
        super().__init__()
        self.appearance_encoder = appearance_encoder
        self.morphology_encoder = morphology_encoder
        self.centroid_encoder = centroid_encoder
        self.n_filters = n_filters
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # Initial feature combination
        # Concatenate all encoders' outputs
        combined_dim = (3 * self.embedding_dim)
                
        self.initial_dense = nn.Linear(combined_dim, n_filters)
        self.initial_norm = nn.BatchNorm1d(n_filters) if norm_layer == 'batch' else nn.LayerNorm(n_filters)
        self.initial_activation = nn.ReLU()
        
        # Parse graph layer type
        graph_layer_name = graph_layer.split('-')[0].lower()
        
        # Build GNN layers
        self.graph_layers = nn.ModuleList()
        self.graph_norms = nn.ModuleList()
        self.graph_activations = nn.ModuleList()
        
        for i in range(n_layers):
            if graph_layer_name == 'gcn':
                layer = GCNConv(n_filters, n_filters)
            elif graph_layer_name == 'gat':
                layer = GATConv(n_filters, n_filters, dropout=0.5, heads=1, add_self_loops=False)
            else:
                raise ValueError(f'Unsupported graph layer: {graph_layer_name}')
            
            self.graph_layers.append(layer)
            norm = nn.BatchNorm1d(n_filters) if norm_layer == 'batch' else nn.LayerNorm(n_filters)
            self.graph_norms.append(norm)
            self.graph_activations.append(nn.ReLU())
        
        # Final embedding layer
        final_input_dim = appearance_encoder.encoder_dim + morphology_encoder.encoder_dim + n_filters
        self.final_dense = nn.Linear(final_input_dim, embedding_dim)
        self.final_norm = nn.BatchNorm1d(embedding_dim) if norm_layer == 'batch' else nn.LayerNorm(embedding_dim)
        self.final_activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def _apply_batched_gnn(self, gnn_layer, node_features, adj_matrices):
            
            """Apply GNN using PyG batching (efficient)."""
            batch_size = node_features.shape[0]
        
            # Create list of PyG Data objects
            graph_list = []
        
            for i in range(batch_size):
                nodes = node_features[i]  # (max_cells, features)
                adj = adj_matrices[i]     # (max_cells, max_cells)
                
                # Convert to edge_index
                edge_index = adj.nonzero().t()
                
                graph_list.append(Data(x=nodes, edge_index=edge_index))
            
            # Batch graphs
            batched_graph = Batch.from_data_list(graph_list)
            
            # Apply GNN to batched graph
            out = gnn_layer(batched_graph.x, batched_graph.edge_index)
            
            # Unbatch
            # out is (B*T*max_cells, features)
            # Reshape to (B*T, max_cells, features)
            out = out.view(batch_size, -1, out.shape[-1])
        
            return out

    def forward(self, appearance, morphology, centroids, adj_matrix):
        """Compute neighborhood-aware embeddings for a batch of cell graphs.

        Parameters
        ----------
        appearance : torch.Tensor
            Appearance crops with shape
            ``(batch * time * max_cells, height, width, channels)`` (or the
            ``channels_first`` equivalent), passed to
            ``appearance_encoder``.
        morphology : torch.Tensor
            Morphology features with shape
            ``(batch * time * max_cells, 3)``, passed to
            ``morphology_encoder``.
        centroids : torch.Tensor
            Centroid positions with shape
            ``(batch * time * max_cells, 2)``, passed to
            ``centroid_encoder``.
        adj_matrix : torch.Tensor
            Adjacency matrices with shape
            ``(batch * time, max_cells, max_cells)``.

        Returns
        -------
        node_features : torch.Tensor
            Neighborhood-aware node embeddings with shape
            ``(batch * time, max_cells, embedding_dim)``.
        centroids : torch.Tensor
            The input centroids, reshaped only, with shape
            ``(batch * time, max_cells, 2)``.
        """

        BT, N, _ = adj_matrix.shape  # (B*T, max_cells, max_cells)
        adj_matrix = normalize_adjacency_symmetric(adj_matrix)

        # Encode each feature type
        app_features = self.appearance_encoder(appearance)  # (B * T *  N, encoder_dim)
        morph_features = self.morphology_encoder(morphology)  # (B * T * N, encoder_dim)
        centroid_features = self.centroid_encoder(centroids)  # (B * T * N, encoder_dim)

        app_features = app_features.view(BT, N, -1)
        morph_features = morph_features.view(BT, N, -1)
        centroid_features = centroid_features.view(BT, N, -1)

        # Concatenate features
        node_features = torch.cat([app_features, morph_features, centroid_features], dim=-1)

        # Initial dense layer
        node_features = self.initial_dense(node_features)


        if isinstance(self.initial_norm, nn.BatchNorm1d):
            node_features = node_features.permute(0, 2, 1)
            node_features = self.initial_norm(node_features)
            node_features = node_features.permute(0, 2, 1)
        else:
            node_features = self.initial_norm(node_features)
        
        node_features = self.initial_activation(node_features)

        # For each graph in the batch, apply GNN

        for gnn_layer, norm, activation in zip(
            self.graph_layers, self.graph_norms, self.graph_activations
        ):
            # GNN expects batched graphs
            node_features = self._apply_batched_gnn(
                gnn_layer, node_features, adj_matrix
            )

            node_features = node_features.permute(0,2,1)
            node_features = norm(node_features)
            node_features = node_features.permute(0,2,1)
            node_features = activation(node_features)
            node_features = self.dropout(node_features)

        
        # Final concatenation and dense layer
        node_features = torch.cat([app_features, morph_features, node_features], dim=-1)
        
        node_features = self.final_dense(node_features)
        
        if isinstance(self.final_norm, nn.BatchNorm1d):
            node_features = node_features.permute(0, 2, 1)
            node_features = self.final_norm(node_features)
            node_features = node_features.permute(0, 2, 1)
        else:
            node_features = self.final_norm(node_features)
        
        node_features = self.final_activation(node_features)
        centroids = centroids.view(BT, N, -1)
        
        return node_features, centroids


# Example usage and tests
if __name__ == "__main__":
    print("Testing encoder modules...\n")

    max_cells = 39
    batch_size = 2
    n_frames = 8

    app_encoder = AppearanceEncoder(appearance_shape=(1, 32, 32))
    app_in = torch.randn(batch_size * n_frames * max_cells, 1, 32, 32)  # channels_last: (B * T * N, C, H, W)

    morph_encoder = MorphologyEncoder(input_dim=3, encoder_dim=64)
    morph_in = torch.randn(batch_size * n_frames * max_cells, 3)


    cent_encoder = CentroidEncoder(input_dim=2, encoder_dim=64)
    cent_in = torch.randn(batch_size * n_frames * max_cells, 2)


    # Test NeighborhoodEncoder
    adj_matrix = torch.randn(batch_size * n_frames, max_cells, max_cells)
    n_encoder = NeighborhoodEncoder(appearance_encoder=app_encoder,
                                    morphology_encoder=morph_encoder,
                                    centroid_encoder=cent_encoder, graph_layer='gat'
                                    )
    
    neighborhood = n_encoder(app_in, morph_in, cent_in, adj_matrix)

    print(f"   Output: {neighborhood[0].shape, neighborhood[1].shape}\n")
    
    # Test DeltaEncoder
    print("4. DeltaEncoder")
    delta_encoder = DeltaEncoder(input_dim=2, encoder_dim=64)
    x = torch.randn(16, max_cells, 2)
    out = delta_encoder(x)
    print(f"   Input: {x.shape}, Output: {out.shape}\n")
    
    print("✅ Basic encoder tests passed!")