"""Inference and evaluation scripts for GNN cell tracking model"""

import torch
import numpy as np
from typing import Dict
from tracking.model import GNNTrackingModel
from tracking.tracker import CellTracker
import zarr
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import json
from tracking.metrics import TrackingMetrics

from tracking.visualization import create_timelapse_gif_with_lineage


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
        'batch_size': 6,
        'n_layers': 1,
        'crop_size': 16,
        'crop_mode': 'fixed'
    }

    # Initialize model

    model = GNNTrackingModel(
                            graph_layer='gat', 
                            data_format='channels_last',
                            encoder_dim=64,
                            n_layers=config['n_layers'],
                            crop_size=config['crop_size'],
                            )

    checkpoint_dir = 'checkpoints/20260227-112127/best_model.pt'
    checkpoint = torch.load(checkpoint_dir) 
    model.load_state_dict(checkpoint['model_state_dict'])     
    
    z = zarr.open('data/DynamicNuclearNet-tracking-v1_0/test.zarr')
    z2 = zarr.open('data/DynamicNuclearNet-tracking-v1_0/test_proc.zarr')

    with open('data/DynamicNuclearNet-tracking-v1_0/test.json') as file:
        gt_lineage = json.load(file)

    X = z['X'][:]
    y = z['y'][:]

    samples = build_indices(X)

    metrics_out = []

    compiled_metrics = {
        'correct_division': 0,
        'mismatch_division': 0,
        'false_positive_division': 0,
        'false_negative_division': 0,
        'total_divisions': 0,
        'aa_tp': 0,
        'aa_total': 0,
        'te_tp': 0,
        'te_total': 0
    }

    for batch in range(X.shape[0]):

        curr_gt_lineage= gt_lineage[batch]
        X = z['X'][batch]
        y = z['y'][batch]
        gt = z2['labels'][batch]
        end_frame = samples[batch]

        tracker = CellTracker(
            movie=X[:end_frame],  # (T, Y, X, C)
            annotation=y[:end_frame],  # (T, Y, X, C)
            tracking_model=model,
            device='cuda:0',
            appearance_dim=32,
            track_length=8,
            division=0.3,
            crop_mode=config['crop_mode'],
            data_format = 'channels_last',
        )

        tracker.track_cells()

        y_tracked = tracker.y_tracked
        lineage = tracker.get_lineage_dict()

        metrics = TrackingMetrics(curr_gt_lineage, y[:end_frame], lineage, y_tracked, threshold=0.8)

        for k, v in metrics.stats.items():
            compiled_metrics[k] += v
        metrics_out.append(metrics.stats)

        track_review = tracker._track_review_dict()
        y_tracked = track_review['y_tracked']
        gt_movie = y[:end_frame]
        outname = f"movies/timelapse_batch_{batch}.gif"

        create_timelapse_gif_with_lineage(
            y_tracked, 
            gt_movie, 
            lineage1=lineage, 
            lineage2=curr_gt_lineage, 
            output_path=outname, 
            cmap='viridis'
        )

    df = pd.DataFrame(metrics_out)

    df['division_precision'] = (df['correct_division'])/(df['correct_division'] + df['false_positive_division'])
    df['division_recall'] = df['correct_division']/(df['correct_division'] + df['false_negative_division'])
    df['division_f1'] = (2 * df['division_recall'] * df['division_precision'])/(df['division_precision'] + df['division_recall'])


    df['aa_accuracy'] = df['aa_tp']/df['aa_total']
    df['te_accuracy'] = df['te_tp']/df['te_total']
    

    print(df['division_f1'].mean())
    print(df['division_recall'].mean())
    print(df['division_precision'].mean())


    
