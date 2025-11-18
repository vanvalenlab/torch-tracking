"""
Diagnostic script to analyze why the model fails to learn certain classes
"""

import torch
import numpy as np
from collections import defaultdict

def diagnose_label_distribution(dataloader, num_batches=10):
    """Analyze the class distribution in the dataset.
    
    This will reveal the severity of class imbalance.
    """
    print("="*70)
    print("DIAGNOSING LABEL DISTRIBUTION")
    print("="*70)
    
    class_counts = defaultdict(int)
    total_pairs = 0
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        labels = batch['labels']
        
        # If one-hot, convert to indices
        if labels.dim() == 5:  # (B, T, N, M, 3)
            labels = labels.argmax(dim=-1)
        
        # Count each class
        for cls in range(3):
            count = (labels == cls).sum().item()
            class_counts[cls] += count
        
        total_pairs += labels.numel()
    
    print(f"\nAnalyzed {i+1} batches with {total_pairs:,} total cell pairs\n")
    print("Class Distribution:")
    for cls in range(3):
        count = class_counts[cls]
        pct = 100 * count / total_pairs
        class_name = ['No Link', 'Same Cell', 'Mitosis'][cls]
        print(f"  Class {cls} ({class_name:12s}): {count:8,} ({pct:5.2f}%)")
    
    # Compute imbalance ratios
    print("\nImbalance Ratios:")
    print(f"  No Link : Same Cell = {class_counts[0]/max(class_counts[1],1):.1f} : 1")
    print(f"  No Link : Mitosis   = {class_counts[0]/max(class_counts[2],1):.1f} : 1")
    print(f"  Same Cell : Mitosis = {class_counts[1]/max(class_counts[2],1):.1f} : 1")
    
    # Warning
    if class_counts[0] / max(class_counts[1], 1) > 100:
        print("\n⚠️  CRITICAL: Severe class imbalance detected!")
        print("   Recommendation: Use focal loss + heavy class weights")
    
    return class_counts


def diagnose_model_predictions(model, dataloader, device='cuda', num_batches=5):
    """Analyze what the model is actually predicting.
    
    This reveals if the model is collapsing to always predict one class.
    """
    print("\n" + "="*70)
    print("DIAGNOSING MODEL PREDICTIONS")
    print("="*70)
    
    model.eval()
    
    pred_counts = defaultdict(int)
    true_counts = defaultdict(int)
    confusion_matrix = np.zeros((3, 3), dtype=np.int64)
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            # Move to device
            appearances = batch['appearances'].to(device)
            morphologies = batch['morphologies'].to(device)
            centroids = batch['centroids'].to(device)
            adj_matrices = batch['adj_matrices'].to(device)
            labels = batch['labels']
            
            # If one-hot, convert to indices
            if labels.dim() == 5:
                labels = labels.argmax(dim=-1)
            
            labels = labels.to(device)
            
            # Get predictions
            predictions = model.training_forward(
                appearances, morphologies, centroids, adj_matrices,
                return_logits=False
            )
            
            pred_classes = predictions.argmax(dim=-1)
            
            # Count predictions
            for cls in range(3):
                pred_counts[cls] += (pred_classes == cls).sum().item()
                true_counts[cls] += (labels == cls).sum().item()
            
            # Update confusion matrix
            for true_cls in range(3):
                for pred_cls in range(3):
                    mask = labels == true_cls
                    confusion_matrix[true_cls, pred_cls] += (pred_classes[mask] == pred_cls).sum().item()
    
    # Print results
    total_preds = sum(pred_counts.values())
    
    print("\nPrediction Distribution:")
    for cls in range(3):
        count = pred_counts[cls]
        pct = 100 * count / total_preds
        class_name = ['No Link', 'Same Cell', 'Mitosis'][cls]
        print(f"  Class {cls} ({class_name:12s}): {count:8,} ({pct:5.2f}%)")
    
    print("\nTrue Label Distribution (in these batches):")
    for cls in range(3):
        count = true_counts[cls]
        pct = 100 * count / total_preds
        class_name = ['No Link', 'Same Cell', 'Mitosis'][cls]
        print(f"  Class {cls} ({class_name:12s}): {count:8,} ({pct:5.2f}%)")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print("               Predicted")
    print("           ", "  ".join([f"C{i}" for i in range(3)]))
    for i in range(3):
        row_sum = confusion_matrix[i].sum()
        row_pcts = confusion_matrix[i] / max(row_sum, 1) * 100
        print(f"  True C{i}: " + " ".join([f"{int(val):3d}" for val in confusion_matrix[i]]) + 
              f"  ({row_pcts[0]:.0f}%, {row_pcts[1]:.0f}%, {row_pcts[2]:.0f}%)")
    
    # Check for collapse
    max_pred_pct = max(pred_counts.values()) / total_preds * 100
    if max_pred_pct > 95:
        print("\n⚠️  CRITICAL: Model has collapsed!")
        print(f"   Predicting class {max(pred_counts, key=pred_counts.get)} {max_pred_pct:.1f}% of the time")
        print("   Recommendation: Adjust loss weights, check label generation")
    
    # Check per-class recall
    print("\nPer-Class Recall:")
    for cls in range(3):
        if confusion_matrix[cls].sum() > 0:
            recall = confusion_matrix[cls, cls] / confusion_matrix[cls].sum() * 100
            class_name = ['No Link', 'Same Cell', 'Mitosis'][cls]
            print(f"  {class_name:12s}: {recall:5.1f}%")
            
            if recall < 10 and cls > 0:
                print(f"    ⚠️  WARNING: Model is NOT learning class {cls}!")


def diagnose_gradient_flow(model, dataloader, device='cuda'):
    """Check if gradients are flowing to all parts of the model.
    
    This can reveal if some components are not being updated.
    """
    print("\n" + "="*70)
    print("DIAGNOSING GRADIENT FLOW")
    print("="*70)
    
    model.train()
    
    # Get one batch
    batch = next(iter(dataloader))
    
    appearances = batch['appearances'].to(device)
    morphologies = batch['morphologies'].to(device)
    centroids = batch['centroids'].to(device)
    adj_matrices = batch['adj_matrices'].to(device)
    labels = batch['labels']
    
    # If one-hot, convert to indices
    if labels.dim() == 5:
        labels = labels.argmax(dim=-1)
    
    labels = labels.to(device)
    
    # Forward pass
    predictions = model.training_forward(
        appearances, morphologies, centroids, adj_matrices,
        return_logits=True
    )
    
    # Compute loss
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(predictions.view(-1, 3), labels.view(-1).long())
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    print("\nGradient Statistics by Component:")
    
    components = {
        'Appearance Encoder': model.appearance_encoder,
        'Morphology Encoder': model.morphology_encoder,
        'Centroid Encoder': model.centroid_encoder,
        'Neighborhood Encoder': model.neighborhood_encoder,
        'Tracking Decoder': model.tracking_decoder
    }
    
    for name, component in components.items():
        grad_norms = []
        for param in component.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        if grad_norms:
            mean_grad = np.mean(grad_norms)
            max_grad = np.max(grad_norms)
            print(f"  {name:20s}: mean={mean_grad:.6f}, max={max_grad:.6f}")
            
            if mean_grad < 1e-7:
                print(f"    ⚠️  WARNING: Very small gradients - may not be learning!")
        else:
            print(f"  {name:20s}: NO GRADIENTS")
            print(f"    ⚠️  CRITICAL: Component not being updated!")


def diagnose_data_format(dataloader):
    """Check if data format and shapes are correct.
    
    This catches shape mismatches early.
    """
    print("\n" + "="*70)
    print("DIAGNOSING DATA FORMAT")
    print("="*70)
    
    batch = next(iter(dataloader))
    
    print("\nBatch Contents:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:20s}: {str(value.shape):30s} dtype={value.dtype}")
        else:
            print(f"  {key:20s}: {value}")
    
    # Check labels specifically
    labels = batch['labels']
    print("\nLabel Analysis:")
    print(f"  Shape: {labels.shape}")
    print(f"  Dtype: {labels.dtype}")
    
    if labels.dim() == 5:
        print("  Format: ONE-HOT ENCODED")
        print("  ⚠️  WARNING: Your loss function may expect CLASS INDICES, not one-hot!")
        print("  Recommendation: Change label generation to output integer class indices")
    elif labels.dim() == 4:
        print("  Format: CLASS INDICES")
        unique_values = torch.unique(labels)
        print(f"  Unique values: {unique_values.tolist()}")
        
        if unique_values.max() > 2:
            print("  ⚠️  WARNING: Labels contain values > 2 (expected 0, 1, 2)")
    
    # Check for NaN or inf
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if torch.isnan(value).any():
                print(f"  ⚠️  WARNING: {key} contains NaN values!")
            if torch.isinf(value).any():
                print(f"  ⚠️  WARNING: {key} contains Inf values!")


def run_full_diagnostics(model, dataloader, device='cuda'):
    """Run all diagnostic checks."""
    print("\n" + "="*70)
    print("RUNNING FULL DIAGNOSTIC SUITE")
    print("="*70 + "\n")

    model.to(device)
    
    # 1. Data format
    diagnose_data_format(dataloader)
    
    # 2. Label distribution
    diagnose_label_distribution(dataloader, num_batches=20)
    
    # 3. Model predictions
    diagnose_model_predictions(model, dataloader, device, num_batches=10)
    
    # 4. Gradient flow
    diagnose_gradient_flow(model, dataloader, device)
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70 + "\n")


# Example usage
if __name__ == "__main__":
    print("Cell Tracking Model Diagnostic Tool")
    print("="*70)
    print()
    print("This script will help identify why your model fails to learn certain classes.")
    print()
    print("Usage:")
    print("  from diagnostics import run_full_diagnostics")
    print("  from model import GNNTrackingModel")
    print("  from loaders import create_trk_dataloaders")
    print()
    print("  # Load your model and data")
    print("  model = GNNTrackingModel(...)")
    print("  train_loader, _, _ = create_trk_dataloaders(...)")
    print()
    print("  # Run diagnostics")
    print("  run_full_diagnostics(model, train_loader, device='cuda')")

    from model import GNNTrackingModel
    from loaders import create_trk_dataloaders

    # Load your model and data
    model = GNNTrackingModel(
                             graph_layer='gat', 
                             data_format='channels_last',
                             encoder_dim=64
                             )
    
    
    train_loader, val_loader, _ = create_trk_dataloaders(train_path='/home/sholtzen/torch-tracking/data/DynamicNuclearNet-tracking-v1_0/train.zarr',
                                                         test_path='/home/sholtzen/torch-tracking/data/DynamicNuclearNet-tracking-v1_0/test.zarr',
                                                         val_path='/home/sholtzen/torch-tracking/data/DynamicNuclearNet-tracking-v1_0/val.zarr',
                                                         batch_size=4,
                                                         distance_threshold=64,
                                                         augment=True)
    # Run diagnostics
    run_full_diagnostics(model, train_loader, device='cuda:6')