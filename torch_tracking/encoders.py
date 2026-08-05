"""Encoder modules for cell tracking model in PyTorch"""

import math
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, Batch
from torch_tracking.utils import normalize_adjacency_symmetric

# Import custom layers (assumes they're in the same directory)
from torch_tracking.layers import ImageNormalization2D


class AppearanceEncoder(nn.Module):
    """Encoder for cell appearance images using 3D convolutions.
    
    CRITICAL: Uses Conv3D with kernel (1, 3, 3) to match TensorFlow architecture.
    - Input shape is (B*T, N, H, W, C)
    - Conv3D treats N as "time" dimension but with kernel_size=1 (no mixing between cells)
    - This is semantically correct: cells are a separate dimension, not part of batch
    
    Args:
        appearance_shape (tuple): Shape of appearance input (H, W, C)
        n_filters (int): Number of convolutional filters
        encoder_dim (int): Output feature dimension
        norm_layer (str): 'batch' or 'layer' normalization
        appearance_norm (bool): Whether to apply input normalization
        data_format (str): 'channels_first' or 'channels_last'
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
        """
        Args:
            x: Tensor of shape (B*T, N, H, W, C) for channels_last
               or (B*T, N, C, H, W) for channels_first
        
        Returns:
            Tensor of shape (B*T, N, encoder_dim)
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
    """Encoder for cell morphology features.
    
    Uses Conv1D with kernel_size=1 to match the architectural pattern:
    - Preserves cell dimension throughout processing
    - Consistent with Conv3D in AppearanceEncoder
    - Semantically correct: applies same transformation to each cell
    
    Input shape: (B*T, N, 3) where N is the cell dimension
    Output shape: (B*T, N, encoder_dim)
    
    Args:
        input_dim (int): Dimension of morphology features (default: 3)
        encoder_dim (int): Output feature dimension
        norm_layer (str): 'batch' or 'layer' normalization
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
        """
        Args:
            x: Tensor of shape (B*T, N, input_dim)
        
        Returns:
            Tensor of shape (B*T, N, encoder_dim)
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
    """Encoder for cell centroid positions.
    
    Uses Conv1D with kernel_size=1 to match the architectural pattern:
    - Preserves cell dimension throughout processing
    - Consistent with Conv3D in AppearanceEncoder
    - Semantically correct: applies same transformation to each cell
    
    Input shape: (B*T, N, 2) where N is the cell dimension
    Output shape: (B*T, N, encoder_dim)
    
    Args:
        input_dim (int): Dimension of centroid features (default: 2 for x,y)
        encoder_dim (int): Output feature dimension
        norm_layer (str): 'batch' or 'layer' normalization
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
        """
        Args:
            x: Tensor of shape (B*T, N, input_dim)
        
        Returns:
            Tensor of shape (B*T, N, encoder_dim)
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
    """Encoder for position deltas.
    
    Encodes changes in position between time steps or between tracks.
    Shared weights for both delta types.
    
    Args:
        input_dim (int): Dimension of delta features (default: 2)
        encoder_dim (int): Output feature dimension
        norm_layer (str): 'batch' or 'layer' normalization
    """
    def __init__(self, input_dim=2, encoder_dim=64, norm_layer='batch', dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(input_dim, encoder_dim)
        self.activation = nn.ReLU()
        self.norm_layer_type = norm_layer
        self.norm = nn.BatchNorm1d(encoder_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, time, tracks, input_dim) 
               or (batch, time, tracks_1, tracks_2, input_dim)
        
        Returns:
            Tensor of same shape but with encoder_dim in last dimension
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
    """Encoder that integrates appearance, morphology, and spatial context using GNNs.
    
    Combines multiple feature types and applies graph convolutions to capture
    neighborhood relationships between cells.
    
    Args:
        appearance_encoder (nn.Module): Appearance encoder module
        morphology_encoder (nn.Module): Morphology encoder module
        centroid_encoder (nn.Module): Centroid encoder module
        n_filters (int): Number of filters for GNN layers
        embedding_dim (int): Final embedding dimension
        n_layers (int): Number of GNN layers
        graph_layer (str): Type of graph layer ('gcn', 'gat', etc.)
        norm_layer (str): 'batch' or 'layer' normalization
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
        """

        Args:
            appearance: (batch * time * max_cells, height, width, channels)
            morphology: (batch * time * max_cells, 3)
            centroids: (batch * time * max_cells, 2)
            adj_matrix: (batch * time, max_cells, max_cells)
            batch_size: (int) size of batch
            n_frames: (int) number of frames in batch
            max_cells: (int) max cells in training batch
        
        Returns:
            node_features: (batch * time, max_cells, embedding_dim)
            centroids: (batch * time, max_cells, 2) - passed through only reshaped
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