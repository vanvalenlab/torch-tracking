"""Encoder modules for cell tracking model in PyTorch"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv

# Import custom layers (assumes they're in the same directory)
from layers import ImageNormalization2D


class AppearanceEncoder(nn.Module):
    """Encoder for cell appearance images using 3D convolutions.
    
    Processes image crops through a series of 3D convolutions with pooling
    to extract appearance features.
    
    Args:
        appearance_shape (tuple): Shape of appearance input (time, height, width, channels)
        n_filters (int): Number of convolutional filters
        encoder_dim (int): Output feature dimension
        norm_layer (str): 'batch' or 'layer' normalization
        appearance_norm (bool): Whether to apply input normalization
        data_format (str): 'channels_first' or 'channels_last'
    """
    def __init__(
        self,
        appearance_shape=(1, 32, 32, 1),
        n_filters=64,
        encoder_dim=64,
        norm_layer='batch',
        appearance_norm=True,
        data_format='channels_first'
    ):
        super().__init__()
        self.appearance_shape = appearance_shape
        self.n_filters = n_filters
        self.encoder_dim = encoder_dim
        self.appearance_norm = appearance_norm
        self.data_format = data_format
        
        # Calculate number of pooling layers based on spatial dimensions
        spatial_dim = appearance_shape[-1]  # Assuming square images
        self.n_layers = int(math.log2(spatial_dim))
        
        # Input normalization
        if self.appearance_norm:
            self.img_norm = ImageNormalization2D(
                norm_method='whole_image',
                data_format=data_format
            )
        
        # Build convolutional layers
        # PyTorch Conv3d expects (batch, channels, depth, height, width)
        in_channels = appearance_shape[-1] if data_format == 'channels_last' else appearance_shape[0]
        
        self.conv_blocks = nn.ModuleList()
        for i in range(self.n_layers):
            block = nn.Sequential(
                nn.Conv3d(
                    in_channels if i == 0 else n_filters,
                    n_filters,
                    kernel_size=(1, 3, 3),
                    stride=1,
                    padding=(0, 1, 1),
                    bias=False
                ),
                nn.BatchNorm3d(n_filters) if norm_layer == 'batch' else nn.LayerNorm([n_filters]),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(1, 2, 2))
            )
            self.conv_blocks.append(block)
        
        # Final dense layer
        self.dense = nn.Linear(n_filters, encoder_dim)
        self.final_norm = nn.BatchNorm1d(encoder_dim) if norm_layer == 'batch' else nn.LayerNorm(encoder_dim)
        self.final_activation = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, cells, time, height, width, channels) if channels_last
               or (batch, channels, cells, time, height, width) if channels_first
        
        Returns:
            Tensor of shape (batch, cells, encoder_dim)
        """

        # Merge tensors to be of shape (B*T, C, N, H, W) (rank 4)
        if x.ndim > 5:
            x = x.permute(0, 2, 1, 3, 4, 5)
            x = x.view(x.shape[0]* x.shape[1], *x.shape[2:])
        elif x.ndim < 5:
            raise ValueError(f'''Input tensor to appearance encoder must be of rank 5, tensor x is of rank {x.ndim}.
                             Check shape of tensor, which must be either of shape (B, C, N, T, H, W) or (B*T, C, N, H, W)''')
        
        # Apply input normalization if needed
        if self.appearance_norm:
            # Need to handle temporal dimension
            batch_size = x.shape[0]
            if self.data_format == 'channels_last':
                # (B, T, H, W, C) -> (B*T, H, W, C)
                time_steps = x.shape[1]
                x = x.view(batch_size * time_steps, *x.shape[2:])
                x = self.img_norm(x)
                # Back to (B, T, H, W, C)
                x = x.view(batch_size, time_steps, *x.shape[1:])
                # Convert to channels_first: (B, T, H, W, C) -> (B, C, T, H, W)
                x = x.permute(0, 4, 1, 2, 3)
            else:
                # Need to move channel dimension to after big B (b*t)
                x = self.img_norm(x)
        else:
            if self.data_format == 'channels_last':
                # Convert to channels_first
                x = x.permute(0, 4, 1, 2, 3)
        
        # Before it goes into the conv blocks, it has to have the shape of (B*T, C, N, H, W)

        # Apply conv blocks
        for block in self.conv_blocks:
            x = block(x)
        
        # After pooling, spatial dimensions should be 1x1
        # Squeeze them: (B, C, T, 1, 1) -> (B, C, T)
        x = x.squeeze(-1).squeeze(-1)
        
        # Permute to (B, T, C) for dense layer
        x = x.permute(0, 2, 1)
        
        # Apply final dense layer
        batch_size, time_steps, _ = x.shape
        x = x.reshape(batch_size * time_steps, -1)
        x = self.dense(x)
        x = x.reshape(batch_size, time_steps, -1)
        
        # Apply normalization (handling batch norm carefully)
        if isinstance(self.final_norm, nn.BatchNorm1d):
            x = x.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)
            x = self.final_norm(x)
            x = x.permute(0, 2, 1)  # Back to (B, T, C)
        else:
            x = self.final_norm(x)
        
        x = self.final_activation(x)
        
        return x


class MorphologyEncoder(nn.Module):
    """Encoder for cell morphology features.
    
    Simple MLP to encode morphological measurements.
    
    Args:
        input_dim (int): Dimension of morphology features (default: 3)
        encoder_dim (int): Output feature dimension
        norm_layer (str): 'batch' or 'layer' normalization
    """
    def __init__(self, input_dim=3, encoder_dim=64, norm_layer='batch'):
        super().__init__()
        self.dense = nn.Linear(input_dim, encoder_dim)
        self.encoder_dim = encoder_dim
        self.norm = nn.BatchNorm1d(encoder_dim) if norm_layer == 'batch' else nn.LayerNorm(encoder_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, time, n_cells, input_dim)
        
        Returns:
            Tensor of shape (batch, time, n_cells, encoder_dim)
        """
        
        # Reshape for dense layer
        x = self.dense(x)
        
        # Apply normalization
        if isinstance(self.norm, nn.BatchNorm1d):
            x = x.permute(0, 2, 1)
            x = self.norm(x)
            x = x.permute(0, 2, 1)
        else:
            x = self.norm(x)
        
        x = self.activation(x)
        return x


class CentroidEncoder(nn.Module):
    """Encoder for cell centroid positions.
    
    Simple MLP to encode centroid coordinates.
    
    Args:
        input_dim (int): Dimension of centroid features (default: 2 for x,y)
        encoder_dim (int): Output feature dimension
        norm_layer (str): 'batch' or 'layer' normalization
    """
    def __init__(self, input_dim=2, encoder_dim=64, norm_layer='batch'):
        super().__init__()
        self.dense = nn.Linear(input_dim, encoder_dim)
        self.norm = nn.BatchNorm1d(encoder_dim) if norm_layer == 'batch' else nn.LayerNorm(encoder_dim)
        self.encoder_dim = encoder_dim
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, time, input_dim)
        
        Returns:
            Tensor of shape (batch, time, encoder_dim)
        """
        
        # Reshape for dense layer
        x = self.dense(x)
        
        # Apply normalization
        if isinstance(self.norm, nn.BatchNorm1d):
            x = x.permute(0, 2, 1)
            x = self.norm(x)
            x = x.permute(0, 2, 1)
        else:
            x = self.norm(x)
        
        x = self.activation(x)
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
    def __init__(self, input_dim=2, encoder_dim=64, norm_layer='batch'):
        super().__init__()
        self.dense = nn.Linear(input_dim, encoder_dim)
        self.activation = nn.ReLU()
        self.norm_layer_type = norm_layer
        
        # We'll create norms dynamically based on input shape
        # since deltas can have different dimensions

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
        x = x.reshape(-1, original_shape[-1])
        x = self.dense(x)
        
        # Reshape back
        new_shape = list(original_shape[:-1]) + [x.shape[-1]]
        x = x.reshape(new_shape)
        
        # Apply normalization - LayerNorm is easier for variable shapes
        if self.norm_layer_type == 'batch':
            # For simplicity with variable shapes, use LayerNorm
            x = F.layer_norm(x, [x.shape[-1]])
        else:
            x = F.layer_norm(x, [x.shape[-1]])
        
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
        norm_layer='batch'
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
        combined_dim = (appearance_encoder.encoder_dim + 
                       morphology_encoder.encoder_dim + 
                       centroid_encoder.encoder_dim)
        
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
                layer = GATv2Conv(n_filters, n_filters, heads=1)
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

    def forward(self, appearance, morphology, centroids, adj_matrix):
        """
        Args:
            appearance: (batch, time, max_cells, height, width, channels)
            morphology: (batch, time, max_cells, 3)
            centroids: (batch, time, max_cells, 2)
            adj_matrix: (batch, time, max_cells, max_cells)
        
        Returns:
            node_features: (batch, time, embedding_dim)
            centroids: (batch, time, 2) - passed through unchanged
        """
        # Encode each feature type
        app_features = self.appearance_encoder(appearance)  # (B,  N, encoder_dim)
        morph_features = self.morphology_encoder(morphology)  # (B, N, encoder_dim)
        centroid_features = self.centroid_encoder(centroids)  # (B, N, encoder_dim)
        
        batch_size, time_steps = app_features.shape[0], app_features.shape[1]
        
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
        
        # Apply graph convolutions
        # Need to flatten batch and time for PyG
        node_features_flat = node_features
        
        # Convert adj_matrix to edge_index format for PyG
        # adj_matrix shape: (B, T, N, N)
        adj_flat = adj_matrix
        
        # For each graph in the batch, apply GNN
        # This is simplified - in practice you'd want to create proper batched graphs
        # Here we assume fully connected within each time step
        for i, (layer, norm, activation) in enumerate(zip(self.graph_layers, self.graph_norms, self.graph_activations)):
            # Simplified: treat each (B*T) as separate graph
            # In production, you'd create proper PyG Data objects
            
            # For GCN, we need edge_index and edge_weight
            # This is a simplified version - proper implementation would batch graphs correctly
            node_features_list = []
            for b in range(app_features.shape[0]):
                adj_b = adj_flat[b]
                edge_index = adj_b.nonzero().t()

                nodes_b = node_features_flat[b:b+1]

                out_b = layer(nodes_b.squeeze(0), edge_index)
                node_features_list.append(out_b.unsqueeze(0))
            
            node_features_flat = torch.cat(node_features_list, dim=0)
            
            # Apply normalization
            if isinstance(norm, nn.BatchNorm1d):
                node_features_flat = norm(node_features_flat.permute(0, 2, 1))
            else:
                node_features_flat = norm(node_features_flat)
            
            node_features_flat = activation(node_features_flat.permute(0, 2, 1))
        
        # Reshape back
        node_features = node_features_flat.reshape(batch_size, time_steps, -1)
        
        # Final concatenation and dense layer
        concat = torch.cat([app_features, morph_features, node_features], dim=-1)
        
        concat = concat.reshape(batch_size * time_steps, -1)
        node_features = self.final_dense(concat)
        node_features = node_features.reshape(batch_size, time_steps, -1)
        
        if isinstance(self.final_norm, nn.BatchNorm1d):
            node_features = node_features.permute(0, 2, 1)
            node_features = self.final_norm(node_features)
            node_features = node_features.permute(0, 2, 1)
        else:
            node_features = self.final_norm(node_features)
        
        node_features = self.final_activation(node_features)
        
        return node_features, centroids


# Example usage and tests
if __name__ == "__main__":
    print("Testing encoder modules...\n")

    max_cells = 39
    
    # Test AppearanceEncoder
    print("1. AppearanceEncoder")
    app_encoder = AppearanceEncoder(
        appearance_shape=(1, max_cells, 32, 32),
        n_filters=64,
        encoder_dim=64
    )

    app_in = torch.randn(16, 1, max_cells, 32, 32)  # channels_first: (B, C, N, H, W)
    app_out = app_encoder(app_in)
    print(f"   Input: {app_in.shape}, Output: {app_out.shape}")

    print()
    print("1b. Appearance encoder with both batch and time dimensions")
    app_in = torch.randn(2, 1, 8, max_cells, 32, 32)  # channels_first: (B, C, T, N, H, W)
    app_out = app_encoder(app_in)
    print(f"   Input: {app_in.shape}, Output: {app_out.shape}")
    print()

    # Test MorphologyEncoder
    print("2. MorphologyEncoder")
    morph_encoder = MorphologyEncoder(input_dim=3, encoder_dim=64)
    morph_in = torch.randn(16, max_cells, 3)
    morph_out = morph_encoder(morph_in)
    print(f"   Input: {morph_in.shape}, Output: {morph_out.shape}\n")
    print()

    # Test CentroidEncoder
    print("3. CentroidEncoder")
    cent_encoder = CentroidEncoder(input_dim=2, encoder_dim=64)
    cent_in = torch.randn(16, max_cells, 2)
    cent_out = cent_encoder(cent_in)
    print(f"   Input: {cent_in.shape}, Output: {cent_out.shape}\n")

    # Test NeighborhoodEncoder
    adj_matrix = torch.randn(16, max_cells, max_cells)
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