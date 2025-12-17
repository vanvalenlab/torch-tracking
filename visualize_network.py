"""
Visualize labeled nuclei movie with network overlay from adjacency matrix.

This script creates an animation showing nuclei labels with their spatial
connections as a network graph overlaid on each frame.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import networkx as nx
from skimage.measure import regionprops
import warnings
from scipy.spatial.distance import cdist
warnings.filterwarnings('ignore')


def get_centroids_from_labels(label_frame: np.ndarray) -> dict:
    """Extract centroids from a labeled image frame.
    
    Args:
        label_frame: 2D array of cell labels (H, W)
    
    Returns:
        Dictionary mapping cell_id -> (y, x) centroid position
    """

    centroids = np.zeros((np.max(label_frame) + 1, 2))
    props = regionprops(label_frame)
    
    for i in range(len(props)):
        prop = props[i]
        centroids[prop.label] = prop.centroid  # (y, x) format

    return centroids


def create_graph_from_adjacency(
    adj_matrix: np.ndarray,
    centroids: np.ndarray,
    threshold: float = 0.0
) -> nx.Graph:
    """Create NetworkX graph from adjacency matrix and centroids.
    
    Args:
        adj_matrix: (N, N) adjacency matrix
        centroids: Dictionary mapping node_id -> (y, x) position
        threshold: Minimum edge weight to include (default 0.0)
    
    Returns:
        NetworkX Graph with positions
    """

    G = nx.Graph()
    
    # Add nodes with positions
    for node_id in range(centroids.shape[0]):
        if centroids[node_id].sum() > 0:
            G.add_node(node_id, pos=(centroids[node_id, 1], centroids[node_id,0]))  # NetworkX uses (x, y) format
    
    # Add edges from adjacency matrix
    for node_i in G.nodes():
        for node_j in G.nodes():
            # Check if there's an edge in the adjacency matrix
            weight = adj_matrix[node_i, node_j]  # Assuming 1-indexed labels
            
            if (weight < threshold) and weight > 0:
                G.add_edge(node_i, node_j, weight=weight)
    
    return G


def visualize_frame_with_network(
    label_frame: np.ndarray,
    ax: plt.Axes,
    frame,
    centroids: np.ndarray = None,
    show_labels: bool = True,
    edge_color: str = 'cyan',
    edge_width: float = 1.5,
    node_color: str = 'yellow',
    node_size: int = 50,
    edge_alpha: float = 0.7,
    node_alpha: float = 0.8,
    threshold: float = 0.0,
    cmap: str = 'tab20'
) -> None:
    """Visualize a single frame with network overlay.
    
    Args:
        label_frame: (H, W) labeled image
        adj_matrix: (max_cells, max_cells) adjacency matrix for this frame
        ax: Matplotlib axes to draw on
        show_labels: Whether to show cell ID labels
        edge_color: Color for network edges
        edge_width: Width of network edges
        node_color: Color for network nodes
        node_size: Size of network nodes
        edge_alpha: Transparency of edges
        node_alpha: Transparency of nodes
        threshold: Minimum adjacency value to draw edge
        cmap: Colormap for cell labels
    """
    ax.clear()
    ax.set_xlim(0, 700)
    ax.set_ylim(0, 700)

    # Display the labeled image
    ax.imshow(label_frame, cmap=cmap, interpolation='nearest', alpha=1)

    
    # Get centroids
    if centroids is None:
        centroids = get_centroids_from_labels(label_frame)
    else:
        centroids = centroids[frame]

    adj_matrix = cdist(centroids, centroids, metric='euclidean')
    
    # Create graph
    G = create_graph_from_adjacency(adj_matrix, centroids, threshold)
    
    # Get positions for NetworkX
    pos = nx.get_node_attributes(G, 'pos')
    
    
    # Draw edges
    if G.number_of_edges() > 0:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color=edge_color,
            width=edge_width,
            alpha=1
        )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_color,
        node_size=node_size,
        alpha=1
    )
    
    # Draw labels if requested
    if show_labels:
        labels = {node: str(node) for node in G.nodes()}
        nx.draw_networkx_labels(
            G, pos, labels, ax=ax,
            font_size=8,
            font_color='white',
            font_weight='bold'
        )
    
    ax.axis('off')


def create_network_movie(
    label_movie: np.ndarray,
    centroids = None,
    output_path: str = 'network_movie.mp4',
    fps: int = 5,
    dpi: int = 100,
    figsize = (10, 10),
    show_labels: bool = True,
    edge_color: str = 'cyan',
    edge_width: float = 1.5,
    node_color: str = 'yellow',
    node_size: int = 50,
    threshold: float = 0.0,
    **kwargs
) -> animation.FuncAnimation:
    """Create an animated movie of labeled nuclei with network overlay.
    
    Args:
        label_movie: (T, H, W) array of labeled images over time
        adj_matrices: (T, max_cells, max_cells) adjacency matrices
        output_path: Path to save the movie
        fps: Frames per second for the movie
        dpi: Resolution of the saved movie
        figsize: Figure size (width, height) in inches
        show_labels: Whether to show cell ID labels
        edge_color: Color for network edges
        edge_width: Width of network edges
        node_color: Color for network nodes
        node_size: Size of network nodes
        threshold: Minimum adjacency value to draw edge
        **kwargs: Additional arguments for visualize_frame_with_network
    
    Returns:
        matplotlib.animation.FuncAnimation object
    """
    n_frames = label_movie.shape[0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    def update_frame(frame_idx):
        """Update function for animation."""
        label_frame = label_movie[frame_idx]

        if centroids is not None:
            centroids = centroids[frame_idx]
        
        visualize_frame_with_network(
            label_frame, ax, centroids=centroids,
            show_labels=show_labels,
            edge_color=edge_color,
            edge_width=edge_width,
            node_color=node_color,
            node_size=node_size,
            threshold=threshold,
            **kwargs
        )
        
        ax.set_title(f"Frame {frame_idx + 1}/{n_frames}", fontsize=14)
        return ax,
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update_frame,
        frames=n_frames,
        interval=1000 / fps,
        blit=False,
        repeat=True
    )
    
    # Save movie
    print(f"Saving movie to {output_path}...")
    writer = animation.FFMpegWriter(
        fps=fps,
        bitrate=1800,
        metadata={'artist': 'Cell Tracking Visualization'}
    )
    anim.save(output_path, writer=writer, dpi=dpi)
    print(f"Movie saved successfully!")
    
    return anim


def create_network_movie_no_save(
    label_movie: np.ndarray,
    centroids = None,
    fps: int = 5,
    figsize=(10, 10),
    **kwargs
) -> animation.FuncAnimation:
    """Create animation without saving (for Jupyter notebooks).
    
    Same as create_network_movie but doesn't save to file.
    Use with %matplotlib notebook or display with HTML(anim.to_jshtml())
    """
    n_frames = label_movie.sum(axis=(1,2)) > 0
    n_frames = np.sum(n_frames)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    def update_frame(frame_idx):
        label_frame = label_movie[frame_idx]

        visualize_frame_with_network(
            label_frame, ax, frame_idx, centroids=centroids, **kwargs
        )
        
        return ax,
    
    anim = animation.FuncAnimation(
        fig, update_frame,
        frames=n_frames,
        interval=1000 / fps,
        blit=False,
        repeat=True
    )
    
    return anim


def save_frames_as_images(
    label_movie: np.ndarray,
    adj_matrices: np.ndarray,
    output_dir: str = './frames',
    prefix: str = 'frame',
    **kwargs
) -> None:
    """Save each frame as a separate image file.
    
    Useful if video encoding fails or you want individual frames.
    
    Args:
        label_movie: (T, H, W) array of labeled images
        adj_matrices: (T, max_cells, max_cells) adjacency matrices
        output_dir: Directory to save frames
        prefix: Prefix for frame filenames
        **kwargs: Additional arguments for visualize_frame_with_network
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    n_frames = label_movie.shape[0]
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for frame_idx in range(n_frames):
        label_frame = label_movie[frame_idx]
        adj_matrix = adj_matrices[frame_idx]
        
        visualize_frame_with_network(
            label_frame, adj_matrix, ax, **kwargs
        )
        
        ax.set_title(f"Frame {frame_idx + 1}/{n_frames}")
        
        output_path = os.path.join(output_dir, f"{prefix}_{frame_idx:04d}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        if (frame_idx + 1) % 10 == 0:
            print(f"Saved {frame_idx + 1}/{n_frames} frames")
    
    plt.close(fig)
    print(f"All frames saved to {output_dir}")


# Example usage
if __name__ == "__main__":
    print("Network Movie Visualization Tool")
    print("=" * 70)
    print()
    
    # Example 1: Create synthetic data for demonstration
    print("Example 1: Creating synthetic data for demonstration")
    
    # Create synthetic label movie (5 frames, 100x100 pixels)
    np.random.seed(42)
    n_frames = 5
    height, width = 100, 100
    max_cells = 10
    
    # Generate random cell positions
    label_movie = np.zeros((n_frames, height, width), dtype=np.int32)
    adj_matrices = np.zeros((n_frames, max_cells, max_cells), dtype=np.float32)
    
    for t in range(n_frames):
        # Create some circular cells
        n_cells = np.random.randint(5, 8)
        for i in range(n_cells):
            cy = np.random.randint(20, height - 20)
            cx = np.random.randint(20, width - 20)
            radius = np.random.randint(5, 10)
            
            y, x = np.ogrid[:height, :width]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            label_movie[t][mask] = i + 1
        
        # Create adjacency matrix (connect nearby cells)
        props = regionprops(label_movie[t])
        for i, prop_i in enumerate(props):
            for j, prop_j in enumerate(props):
                if i < j:
                    dist = np.linalg.norm(
                        np.array(prop_i.centroid) - np.array(prop_j.centroid)
                    )
                    if dist < 40:  # Connect if within 40 pixels
                        adj_matrices[t, i, j] = 1.0
                        adj_matrices[t, j, i] = 1.0
    
    print(f"Created synthetic movie: {label_movie.shape}")
    print(f"Adjacency matrices: {adj_matrices.shape}")
    print()
    
    # Example 2: Save frames as images
    print("Example 2: Saving individual frames")
    save_frames_as_images(
        label_movie, adj_matrices,
        output_dir='./network_frames',
        show_labels=True,
        edge_color='cyan',
        node_color='yellow',
        edge_width=2.0,
        node_size=100
    )
    print()
    
    # Example 3: Create movie
    print("Example 3: Creating animated movie")
    try:
        anim = create_network_movie(
            label_movie, adj_matrices,
            output_path='network_overlay_movie.mp4',
            fps=2,
            dpi=100,
            show_labels=True,
            edge_color='cyan',
            edge_width=2.0,
            node_color='yellow',
            node_size=100
        )
        print("Movie created successfully!")
    except Exception as e:
        print(f"Movie creation failed: {e}")
        print("Try saving individual frames instead with save_frames_as_images()")
    
    print()
    print("=" * 70)
    print("Usage with your data:")
    print()
    print("from visualize_network_movie import create_network_movie")
    print()
    print("# Load your data")
    print("# y: (T, H, W) labeled nuclei movie")
    print("# adj_matrices: (T, max_cells, max_cells) adjacency matrices")
    print()
    print("anim = create_network_movie(")
    print("    y, adj_matrices,")
    print("    output_path='my_tracking_movie.mp4',")
    print("    fps=5,")
    print("    show_labels=True,")
    print("    edge_color='cyan',")
    print("    node_color='yellow',")
    print("    threshold=0.5  # Only show edges with weight > 0.5")
    print(")")