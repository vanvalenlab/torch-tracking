"""Inference and evaluation scripts for GNN cell tracking model"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import tqdm
from collections import defaultdict
import json
from model import GNNTrackingModel
from tracker import CellTracker
import zarr
import pprint

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.colors import ListedColormap

from metrics import TrackingEvaluator

def create_timelapse_gif(im1, im2, output_path='timelapse.gif', fps=10, 
                         titles=('Channel 1', 'Channel 2'), 
                         cmap='gray', vmin=None, vmax=None):
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


class ModelEvaluator:
    """Evaluate tracking model on test data.
    
    Computes various tracking metrics including:
    - Accuracy, precision, recall
    - Track-based metrics (track purity, completeness)
    - CLEAR MOT metrics (MOTA, MOTP)
    
    Args:
        model: Trained model
        device: Device for inference
    """
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    @torch.no_grad()
    def evaluate_batch(
        self,
        appearances: torch.Tensor,
        morphologies: torch.Tensor,
        centroids: torch.Tensor,
        adj_matrices: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate model on a single batch.
        
        Returns metrics for this batch.
        """
        # Move to device
        appearances = appearances.to(self.device)
        morphologies = morphologies.to(self.device)
        centroids = centroids.to(self.device)
        adj_matrices = adj_matrices.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass
        predictions = self.model.training_forward(
            appearances, morphologies, centroids, adj_matrices,
            return_logits=False
        )
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, labels)
        
        return metrics
    
    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute various metrics.
        
        Args:
            predictions: (B, T-1, N, M, 3) probabilities
            labels: (B, T-1, N, M) ground truth
        """
        pred_classes = predictions.argmax(dim=-1)
        
        # Valid mask (ignore padding)
        
        # Overall accuracy
        correct = (pred_classes == labels) & valid_mask
        accuracy = correct.sum().float() / valid_mask.sum().float()
        
        # Per-class metrics
        metrics = {'accuracy': accuracy.item()}
        
        for class_idx in range(3):
            # Precision and recall for this class
            pred_positive = (pred_classes == class_idx) & valid_mask
            true_positive = (labels == class_idx) & valid_mask
            
            tp = (pred_positive & true_positive).sum().float()
            fp = (pred_positive & ~true_positive).sum().float()
            fn = (~pred_positive & true_positive).sum().float()
            
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            
            metrics[f'precision_class_{class_idx}'] = precision.item()
            metrics[f'recall_class_{class_idx}'] = recall.item()
            metrics[f'f1_class_{class_idx}'] = f1.item()
        
        # Special focus on "same cell" (class 2) - most important for tracking
        metrics['link_precision'] = metrics['precision_class_2']
        metrics['link_recall'] = metrics['recall_class_2']
        metrics['link_f1'] = metrics['f1_class_2']
        
        return metrics
    
    def print_evaluation(self, metrics: Dict[str, float]):
        """Print evaluation results in a nice format."""
        print("\n" + "="*70)
        print("Evaluation Results")
        print("="*70)
        print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
        print(f"\nLink Detection (Class 2 - Most Important):")
        print(f"  Precision: {metrics['link_precision']:.4f}")
        print(f"  Recall:    {metrics['link_recall']:.4f}")
        print(f"  F1 Score:  {metrics['link_f1']:.4f}")
        
        print(f"\nPer-Class Breakdown:")
        for class_idx in range(3):
            class_name = ['No Link', 'Different Cell', 'Same Cell'][class_idx]
            print(f"\n  {class_name} (Class {class_idx}):")
            print(f"    Precision: {metrics[f'precision_class_{class_idx}']:.4f}")
            print(f"    Recall:    {metrics[f'recall_class_{class_idx}']:.4f}")
            print(f"    F1 Score:  {metrics[f'f1_class_{class_idx}']:.4f}")
        
        print("\n" + "="*70 + "\n")

def build_indices(X):

    samples = []

    for batch in range(X.shape[0]):

        end_frame = np.sum(np.sum(X[batch], axis=(1, 2, 3)) != 0) - 1
        samples.append(end_frame.item())
    
    return samples

# Example usage
if __name__ == "__main__":

    config = {
        'batch_size': 6,
        'n_layers': 1,
        'crop_size': 16
    }

    # Initialize model

    model = GNNTrackingModel(
                             graph_layer='gat', 
                             data_format='channels_last',
                             encoder_dim=64,
                             n_layers=config['n_layers'],
                             crop_size=config['crop_size'],
                             )

    checkpoint_dir = 'checkpoints/20251215-164022/best_model.pt'
    checkpoint = torch.load(checkpoint_dir) 
    model.load_state_dict(checkpoint['model_state_dict'])   
    
    z = zarr.open('data/DynamicNuclearNet-tracking-v1_0/test.zarr')
    z2 = zarr.open('data/DynamicNuclearNet-tracking-v1_0/test_proc.zarr')
    batch = 9

    X = z['X'][:]
    y = z['y'][:]
    gt = z2['labels'][:][batch]
    gt_mask = z2['mask'][:][batch]

    samples = build_indices(X)
    end_frame = samples[batch]

    tracker = CellTracker(
        movie=X[batch, :end_frame],  # (T, Y, X, C)
        annotation=y[batch, :end_frame],  # (T, Y, X, C)
        tracking_model=model,
        device='cuda:0',
        appearance_dim=16,
        division=0.99,  # Threshold for detecting mitosis,
        track_length=8
    )



    tracker.track_cells()

    track_review = tracker._track_review_dict()
    y_tracked = track_review['y_tracked']
    gt_movie = y[batch, :end_frame]

    frame_max = np.max(y_tracked, axis=(1,2,3))
    print(frame_max[-1] + 1)

    the_rest = [np.random.rand((3)) for _ in range(frame_max[-1] + 1)]
    the_rest[0] = np.array([0.,0.,0.])

    rand_cmap = ListedColormap(the_rest, N=frame_max[-1]+1)

    create_timelapse_gif(y_tracked, gt_movie, cmap='viridis')

    # predictions = tracker._get_assignment_matrix()

    # pt, ph, pw, pc = predictions.shape

    # gt_cropped = gt[:pt, :ph, :pw, :pc]

    # metrics = TrackingEvaluator()
    # all_metrics = metrics.evaluate_all(predictions, gt_cropped)

    # pprint.pprint(all_metrics)

    




    
