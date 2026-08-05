"""Inference and evaluation scripts for GNN cell tracking model"""

import pandas as pd

import numpy as np
from torch_tracking.tracker import CellTracker
import zarr
import json
import tqdm

import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.animation as animation

from torch_tracking.metrics import TrackingMetrics

def build_indices(X):

    samples = []

    for batch in range(X.shape[0]):

        end_frame = np.sum(np.sum(X[batch], axis=(1, 2, 3)) != 0) - 1
        samples.append(end_frame.item())
    
    return samples

def pretty_print(df):
    
    print()
    print('='*25)
    print('Results')
    print('='*25)

    full_precision = df['correct_division'].sum() / (df['correct_division'].sum() + df['false_positive_division'].sum())
    full_recall = df['correct_division'].sum() / (df['correct_division'].sum() + df['false_negative_division'].sum())
    full_f1 = 2*full_precision*full_recall / (full_recall+full_precision)

    metrics = {
        'Precision': full_precision,
        'Recall': full_recall,
        "F1": full_f1
    }

    for k,v in metrics.items():
        print(f"{k}: {v:.4}")
        print()


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

if __name__ == "__main__":

    write_movies = False



    # Make output directories

    movies_out = Path('movies')
    metrics_out = Path('metrics')

    if not movies_out.exists():
        movies_out.mkdir()
    if not metrics_out.exists():
        metrics_out.mkdir()

    # Initialize model


    z = zarr.open(Path.home() / '.deepcell/tracking/test.zarr')

    with open(Path.home() / '.deepcell/tracking/test.json') as file:
        gt_lineage = json.load(file)

    metrics_out = []

    X = z['X'][:]
    y = z['y'][:]

    for batch in tqdm.tqdm(range(X.shape[0]), leave=False):

        curr_gt_lineage= gt_lineage[batch]

        samples = build_indices(y)
        end_frame = samples[batch]

        tracker = CellTracker(
            checkpoint_dir=Path.home() / 'torch-tracking/checkpoints/20260607-083350/best_model.pt',
            device='cuda:1',
            division=0.5,
            birth=0.999,
            death=0.999,
            track_length=8,
            verbose=False
        )

        tracker.preprocess_movie(movie=X[batch, :end_frame],
                                 annotation=y[batch, :end_frame])
        
        tracker.track_cells()

        y_tracked = tracker.y_tracked
        lineage = tracker.get_lineage_dict()

        curr_gt_lineage  = {int(k): v for k, v in curr_gt_lineage.items()}
        lineage = {int(k): v for k, v in lineage.items()}

        metrics = TrackingMetrics(curr_gt_lineage, y[batch, :end_frame].squeeze(), lineage, y_tracked.squeeze(), threshold=0.8, verbose=False).stats

        metrics_out.append(metrics)

        if write_movies:
            track_review = tracker._track_review_dict()
            y_tracked = track_review['y_tracked']
            gt_movie = y[batch, :end_frame]
            outname = f"movies/timelapse_batch_{batch}.gif"

            create_timelapse_gif(
                y_tracked, 
                gt_movie,  
                output_path=outname, 
                cmap='viridis'
            )

    df = pd.DataFrame(metrics_out)

    df['division_precision'] = (df['correct_division'])/(df['correct_division'] + df['false_positive_division'])
    df['division_recall'] = df['correct_division']/(df['correct_division'] + df['false_negative_division'])
    df['division_f1'] = (2 * df['division_recall'] * df['division_precision'])/(df['division_precision'] + df['division_recall'])

    df['aa_accuracy'] = df['aa_tp']/df['aa_total']
    df['te_accuracy'] = df['te_tp']/df['te_total']

    df.to_csv('eval_results.csv')

    pretty_print(df)



