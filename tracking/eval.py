"""Inference and evaluation scripts for GNN cell tracking model"""

import torch
import numpy as np
from typing import Dict
from tracking.model import GNNTrackingModel
from tracker import CellTracker
import zarr

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def create_timelapse_gif(im1, im2, output_path='timelapse.gif', fps=10, 
                         titles=('Predicted', 'True'), 
                         cmap='gray'):
    """
    Create a GIF from a time lapse image with two channels.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Time lapse image of shape (T, H, W, 2)
    output_path : str
        Path to save the output GIF
    fps : int
        Frames per second for the GIF
    titles : tuple
        Titles for the two subplots
    cmap : str
        Colormap to use for display
    vmin, vmax : float, optional
        Min/max values for intensity scaling. If None, uses data min/max
    """

    T, H, W, C = im1.shape
    
    # Set up the figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    
    im1_plot = axes[0].imshow(im1[0], cmap=cmap)
    im2_plot = axes[1].imshow(im2[0], cmap=cmap)
    
    axes[0].set_title(titles[0])
    axes[1].set_title(titles[1])
    axes[0].axis('off')
    axes[1].axis('off')
    
    # Add colorbars
    plt.colorbar(im1_plot, ax=axes[0], fraction=0.046, pad=0.04)
    plt.colorbar(im2_plot, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Add frame counter
    frame_text = fig.text(0.5, 0.02, f'Frame: 0/{T-1}', 
                          ha='center', fontsize=12)
    
    plt.tight_layout()
    
    def update(frame):
        """Update function for animation"""
        im1_plot.set_data(im1[frame])
        im2_plot.set_data(im2[frame])
        frame_text.set_text(f'Frame: {frame}/{T-1}')
        return im1_plot, im2_plot, frame_text
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=T, 
                                   interval=1000/fps, blit=True)
    
    # Save as GIF
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close()
    
    print(f"GIF saved to {output_path}")


def build_indices(X):

    samples = []

    for batch in range(X.shape[0]):

        end_frame = np.sum(np.sum(X[batch], axis=(1, 2, 3)) != 0) - 1
        samples.append(end_frame.item())
    
    return samples


if __name__ == "__main__":

    config = {
        'batch_size': 16,
        'n_layers': 1,
        'crop_size': 32
    }

    # Initialize model

    model = GNNTrackingModel(
                             graph_layer='gat', 
                             data_format='channels_last',
                             encoder_dim=64,
                             n_layers=config['n_layers'],
                             crop_size=config['crop_size'],
                             )

    checkpoint_dir = 'checkpoints/20260131-201751/checkpoint_epoch_49.pt'
    checkpoint = torch.load(checkpoint_dir) 
    model.load_state_dict(checkpoint['model_state_dict'])   
    
    z = zarr.open('data/DynamicNuclearNet-tracking-v1_0/test.zarr')
    z2 = zarr.open('data/DynamicNuclearNet-tracking-v1_0/test_proc.zarr')
    batch = 6

    X = z['X'][:]
    y = z['y'][:]

    samples = build_indices(X)

    for batch in range(8,9):
        end_frame = samples[batch]

        tracker = CellTracker(
            movie=X[batch, :end_frame],  # (T, Y, X, C)
            annotation=y[batch, :end_frame],  # (T, Y, X, C)
            tracking_model=model,
            device='cuda:0',
            appearance_dim=32,
            division=0.99,  # Threshold for detecting mitosis,
            track_length=8,
            distance_threshold=72,
            crop_mode='fixed'
        )

        tracker.track_cells()

        track_review = tracker._track_review_dict()
        y_tracked = track_review['y_tracked']
        gt_movie = y[batch, :end_frame]
        outname = f"timelapse_batch_{batch}.gif"
        create_timelapse_gif(y_tracked, gt_movie, output_path=outname, cmap='viridis')
    




    
