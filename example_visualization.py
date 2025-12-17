"""
Example: Visualize cell tracking with network overlay using your existing data.

This script shows how to load your .zarr data and create network overlay movies.
"""

import numpy as np
import zarr
from visualize_network import (
    create_network_movie, 
    save_frames_as_images,
    create_network_movie_no_save
)


def load_tracking_data(zarr_path: str, batch_idx: int = 0):
    """Load tracking data from your .zarr file.
    
    Args:
        zarr_path: Path to .zarr file
        batch_idx: Which batch to load
    
    Returns:
        X: Raw images (T, H, W, C)
        y: Label movie (T, H, W, C)
        adj_matrix: Adjacency matrices if available
    """
    z = zarr.open(zarr_path, mode='r')
    
    X = z['X'][batch_idx]  # (T, H, W, C)
    y = z['y'][batch_idx]  # (T, H, W, C)
    
    print(f"Loaded data shapes:")
    print(f"  Raw images (X): {X.shape}")
    print(f"  Labels (y): {y.shape}")
    
    # Squeeze channel dimension if it's 1
    if y.shape[-1] == 1:
        y = y.squeeze(-1)
    
    return X, y


def create_adjacency_from_labels(
    y: np.ndarray,
    distance_threshold: float = 64.0,
    max_cells: int = None
) -> np.ndarray:
    """Create adjacency matrices from labeled images.
    
    Args:
        y: (T, H, W) labeled movie
        distance_threshold: Maximum distance for connections
        max_cells: Maximum number of cells (auto-detect if None)
    
    Returns:
        adj_matrices: (T, max_cells, max_cells) adjacency matrices
    """
    from scipy.spatial.distance import cdist
    from skimage.measure import regionprops
    
    T = y.shape[0]
    
    # Find max cells if not provided
    if max_cells is None:
        max_cells = 0
        for t in range(T):
            n = len(np.unique(y[t])) - 1  # Subtract background
            if n > max_cells:
                max_cells = n
        max_cells = max_cells + 5  # Add buffer
    
    print(f"Creating adjacency matrices with max_cells={max_cells}")
    
    adj_matrices = np.zeros((T, max_cells, max_cells), dtype=np.float32)
    
    for t in range(T):
        # Get cell properties
        props = regionprops(y[t].astype(int))
        
        if len(props) == 0:
            continue
        
        # Get centroids for all cells
        centroids = np.array([prop.centroid for prop in props])
        labels = [prop.label for prop in props]
        
        # Compute pairwise distances
        distances = cdist(centroids, centroids, metric='euclidean')
        
        # Create adjacency based on distance threshold
        adj = (distances < distance_threshold).astype(np.float32)
        
        # Remove self-loops
        np.fill_diagonal(adj, 0)
        
        # Place in adjacency matrix (using 0-indexed positions)
        for i, label_i in enumerate(labels):
            for j, label_j in enumerate(labels):
                if label_i <= max_cells and label_j <= max_cells:
                    adj_matrices[t, label_i - 1, label_j - 1] = adj[i, j]
    
    print(f"Created adjacency matrices: {adj_matrices.shape}")
    
    return adj_matrices


def visualize_tracking_with_network(
    zarr_path: str,
    batch_idx: int = 0,
    output_path: str = 'tracking_network.mp4',
    distance_threshold: float = 64.0,
    fps: int = 5,
    save_frames: bool = True,
    frames_dir: str = './tracking_frames',
    **viz_kwargs
):
    """Complete pipeline to visualize tracking with network overlay.
    
    Args:
        zarr_path: Path to .zarr data file
        batch_idx: Which batch to visualize
        output_path: Output video path
        distance_threshold: Maximum distance for network connections
        fps: Frames per second for video
        save_frames: Whether to also save individual frames
        frames_dir: Directory for saving frames
        **viz_kwargs: Additional visualization arguments
    """
    print("=" * 70)
    print("Cell Tracking Network Visualization")
    print("=" * 70)
    print()
    
    # Load data
    print("Step 1: Loading data...")
    X, y = load_tracking_data(zarr_path, batch_idx)
    print()
    
    # Create adjacency matrices
    print("Step 2: Creating adjacency matrices...")
    adj_matrices = create_adjacency_from_labels(
        y, 
        distance_threshold=distance_threshold
    )
    print()
    
    # Save individual frames if requested
    if save_frames:
        print("Step 3: Saving individual frames...")
        save_frames_as_images(
            y, adj_matrices,
            output_dir=frames_dir,
            **viz_kwargs
        )
        print()
    
    # Create movie
    print("Step 4: Creating animated movie...")
    try:
        anim = create_network_movie(
            y, adj_matrices,
            output_path=output_path,
            fps=fps,
            **viz_kwargs
        )
        print(f"âœ… Movie saved to: {output_path}")
    except Exception as e:
        print(f"âš ï¸  Movie creation failed: {e}")
        print(f"Individual frames are available in: {frames_dir}")
    
    print()
    print("=" * 70)
    print("Visualization complete!")
    print("=" * 70)


# Example usage
if __name__ == "__main__":
    
    # Example 1: Basic usage with your data
    print("\nExample 1: Basic visualization")
    print("-" * 70)
    
    # Adjust this path to your data
    zarr_path = 'data/DynamicNuclearNet-tracking-v1_0/test.zarr'
    
    try:
        visualize_tracking_with_network(
            zarr_path=zarr_path,
            batch_idx=0,
            output_path='cell_tracking_network.mp4',
            distance_threshold=64.0,
            fps=3,
            save_frames=True,
            frames_dir='./cell_tracking_frames',
            # Visualization options
            show_labels=True,
            edge_color='cyan',
            edge_width=1.5,
            node_color='yellow',
            node_size=80,
            edge_alpha=0.7,
            node_alpha=0.9,
            threshold=0.0  # Show all edges
        )
    except FileNotFoundError:
        print(f"Data file not found: {zarr_path}")
        print("Please update the zarr_path to point to your data.")
    
    print("\n")
    
    # Example 2: Advanced usage with custom styling
    print("\nExample 2: Custom styling")
    print("-" * 70)
    print("Customize the visualization with these parameters:")
    print()
    print("# Color schemes:")
    print("  edge_color='red', node_color='blue'  # Blue nodes with red edges")
    print("  edge_color='lime', node_color='magenta'  # Neon style")
    print()
    print("# Size and visibility:")
    print("  edge_width=3.0, node_size=150  # Larger, more visible")
    print("  edge_alpha=0.5, node_alpha=0.5  # More transparent")
    print()
    print("# Network filtering:")
    print("  threshold=0.5  # Only show strong connections")
    print("  show_labels=False  # Hide cell ID numbers")
    print()
    
    # Example 3: Using with features from your model
    print("\nExample 3: Using pre-computed adjacency matrices")
    print("-" * 70)
    print("""
# If you already have adjacency matrices from your model:
from visualize_network_movie import create_network_movie

# Assuming you have:
# - features: dict with 'adj_matrix' key (from your loader)
# - y: labeled movie

adj_matrices = features['adj_matrix'][batch_idx]  # (T, max_cells, max_cells)
y_movie = features['labels'][batch_idx]  # (T, H, W)

create_network_movie(
    y_movie, adj_matrices,
    output_path='my_tracking.mp4',
    fps=5,
    show_labels=True
)
    """)
    
    print("\n")
    print("=" * 70)
    print("Tips:")
    print("  - Use lower fps (2-3) for slow-motion viewing")
    print("  - Adjust distance_threshold to control network density")
    print("  - Set threshold>0 to filter weak connections")
    print("  - Save frames if video encoding fails")
    print("=" * 70)