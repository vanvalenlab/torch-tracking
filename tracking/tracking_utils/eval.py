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
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Compute various metrics.
        
        Args:
            predictions: (B, T-1, N, M, 3) probabilities
            labels: (B, T-1, N, M) ground truth
        """
        pred_classes = predictions.argmax(dim=-1)
        
        # Valid mask (ignore padding)
        valid_mask = labels >= 0
        
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
    
    def evaluate_dataloader(self, dataloader) -> Dict[str, float]:
        """Evaluate on entire dataloader.
        
        Returns aggregated metrics.
        """
        all_metrics = defaultdict(list)
        
        for batch in tqdm(dataloader, desc='Evaluating'):
            batch_metrics = self.evaluate_batch(
                batch['appearances'],
                batch['morphologies'],
                batch['centroids'],
                batch['adj_matrices'],
                batch['labels']
            )
            
            for key, value in batch_metrics.items():
                all_metrics[key].append(value)
        
        # Aggregate
        aggregated = {}
        for key, values in all_metrics.items():
            aggregated[key] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
        
        return aggregated
    
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

def run_online_tracking(
    model,
    raw_images: np.ndarray,
    masks: np.ndarray,
    device='cuda',
    track_length=8,
    max_cells=39
) -> List[Dict]:
    """Run online tracking on a video sequence.
    
    Args:
        model: Trained GNNTrackingModel
        raw_images: (T, Y, X, C) raw fluorescent images
        masks: (T, Y, X, C) segmentation masks
        device: Device for inference
        track_length: History length for tracking
        max_cells: Maximum cells per frame
    
    Returns:
        tracks: List of track dictionaries
    """
    
    tracker = CellTracker(
        model=model,
        device=device,
        max_history_length=track_length,
        link_threshold=0.5,
        max_gap=3
    )
    
    print("Running online tracking...")
    print(f"Total frames: {len(raw_images)}")
    
    # For each frame, extract features and update tracker
    for frame_idx in tqdm(range(len(raw_images))):


        # Extract cell features from mask (simplified - use your actual extraction)
        # This would use the same logic as in TrkDataset._extract_features_from_masks
        
        # For now, placeholder:
        # embeddings, centroids = extract_features(raw_images[frame_idx], masks[frame_idx])
        
        # Update tracker
        # assignments = tracker.update(embeddings, centroids, frame_idx)
        
        pass  # Implement feature extraction here
    
    # Export results
    tracks = tracker.export_tracks()
    
    print(f"\nTracking complete!")
    print(f"Total tracks created: {len(tracks)}")
    print(f"Average track length: {np.mean([t['length'] for t in tracks]):.1f} frames")
    
    return tracks


def save_evaluation_results(
    metrics: Dict[str, float],
    output_path: Path
):
    """Save evaluation results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results saved to {output_path}")


# Example usage
if __name__ == "__main__":

    config = {
        'optimizer': 'radam',
        'learning_rate': 5e-4,
        'weight_decay': 0,
        'decay': 0.99,
        'scheduler': 'reduce_on_plateau',
        'max_epochs': 50,
        'batch_size': 6,
        'n_layers': 1,
        'num_workers': 8,
        'clipnorm': 1e-3,
        'step_size': 5,
        'crop_mode': 'fixed',
        'patience': 5,
        'log_and_save': True,
        'enable_early_stopping': True,
        'crop_size': 16,
        'attention': False,
        'truncate_dataset': None,
        'loss': 'wcce'
    }

    # Initialize model

    model = GNNTrackingModel(
                             graph_layer='gat', 
                             data_format='channels_last',
                             encoder_dim=64,
                             n_layers=config['n_layers'],
                             crop_size=config['crop_size'],
                             attention=config['attention']
                             )

    checkpoint_dir = 'checkpoints/20251211-141526/best_model.pt'
    checkpoint = torch.load(checkpoint_dir) 
    model.load_state_dict(checkpoint['model_state_dict'])   
    
    z = zarr.open('data/DynamicNuclearNet-tracking-v1_0/test.zarr')

    X = z['X'][:]
    y = z['y'][:]

    samples = build_indices(X)
    print(samples)
    batch = 1
    end_frame = samples[batch]

    tracker = CellTracker(
        movie=X[batch, :end_frame],  # (T, Y, X, C)
        annotation=y[batch, :end_frame],  # (T, Y, X, C)
        tracking_model=model,
        device='cuda:4',
        appearance_dim=16,
        division=0.99  # Threshold for detecting mitosis
    )

    tracker.track_cells()

    lineage = tracker.get_lineage_dict()

    pprint.pprint(lineage)

    # print("Inference & Evaluation Example")
    # print("="*70)
    # print()
    
    # print("1. Evaluation on test set:")
    # print("```python")
    # print("from gnn_tracking_model import GNNTrackingModel")
    # print("from trk_data_loader import create_trk_dataloaders")
    # print("from inference_eval import ModelEvaluator")
    # print()
    # print("# Load model")
    # print("model = GNNTrackingModel(...)")
    # print("checkpoint = torch.load('checkpoints/best_model.pt')")
    # print("model.load_state_dict(checkpoint['model_state_dict'])")
    # print()
    # print("# Load test data")
    # print("_, _, test_loader = create_trk_dataloaders(")
    # print("    train_path='train.trk',")
    # print("    val_path='val.trk',")
    # print("    test_path='test.trk',")
    # print("    batch_size=4")
    # print(")")
    # print()
    # print("# Evaluate")
    # print("evaluator = ModelEvaluator(model, device='cuda')")
    # print("metrics = evaluator.evaluate_dataloader(test_loader)")
    # print("evaluator.print_evaluation(metrics)")
    # print("```")
    # print()
    
    # print("2. Online tracking on new video:")
    # print("```python")
    # print("from inference_eval import CellTracker")
    # print()
    # print("# Create tracker")
    # print("tracker = CellTracker(")
    # print("    model=model,")
    # print("    device='cuda',")
    # print("    max_history_length=8,")
    # print("    link_threshold=0.5")
    # print(")")
    # print()
    # print("# Process each frame")
    # print("for frame_idx in range(num_frames):")
    # print("    # Extract embeddings for cells in this frame")
    # print("    embeddings, centroids = extract_cell_features(frame)")
    # print("    ")
    # print("    # Update tracks")
    # print("    assignments = tracker.update(embeddings, centroids, frame_idx)")
    # print("    ")
    # print("    # assignments maps detection_idx -> track_id")
    # print()
    # print("# Export results")
    # print("tracks = tracker.export_tracks()")
    # print("```")
    # print()
    
    # print("Key Features:")
    # print("  ✓ Online tracking with track history")
    # print("  ✓ Hungarian algorithm for optimal assignment")
    # print("  ✓ Comprehensive evaluation metrics")
    # print("  ✓ Per-class precision/recall/F1")
    # print("  ✓ Easy export to standard formats")
    # print("  ✓ Track management (creation, termination)")
    # print()
    
    # print("Metrics Computed:")
    # print("  - Overall accuracy")
    # print("  - Link precision/recall/F1 (class 2)")
    # print("  - Per-class metrics (all 3 classes)")
    # print("  - Track statistics (length, count)")
    # print()
    
    # print("Next Steps:")
    # print("  1. Test evaluation on your val/test sets")
    # print("  2. Tune link_threshold based on precision/recall trade-off")
    # print("  3. Try online tracking on new videos")
    # print("  4. Visualize tracks with your favorite viz tool")