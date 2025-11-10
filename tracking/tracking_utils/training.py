"""Complete training loop for GNN cell tracking model"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import json
from typing import Dict
import time

from model import GNNTrackingModel
from loaders import create_trk_dataloaders


class TrackingLoss(nn.Module):
    """Custom loss function for cell tracking.
    
    Combines cross-entropy loss with optional class weights to handle
    class imbalance (many more "no link" pairs than "same cell" pairs).
    
    Args:
        class_weights (list or tensor): Weights for each class [no_link, different, same]
        ignore_index (int): Index to ignore in loss computation (for padding)
    """
    def __init__(self, class_weights=None, ignore_index=-100):
        super().__init__()
        
        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
        
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            reduction='mean'
        )
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, T-1, N, M, 3) logits
            targets: (B, T-1, N, M) class labels
        
        Returns:
            loss: scalar tensor
        """
        # Flatten predictions and targets
        B, T, N, M, C = predictions.shape
        predictions_flat = predictions.reshape(-1, C)  # (B*T*N*M, 3)
        targets_flat = targets.reshape(-1)  # (B*T*N*M,)
        
        loss = self.criterion(predictions_flat, targets_flat)
        return loss


class MetricsTracker:
    """Track training and validation metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_loss = 0.0
        self.total_samples = 0
        self.correct = 0
        self.total_predictions = 0
        
        # Per-class metrics
        self.class_correct = {0: 0, 1: 0, 2: 0}
        self.class_total = {0: 0, 1: 0, 2: 0}
    
    def update(self, loss, predictions, targets):
        """
        Args:
            loss: scalar loss value
            predictions: (B, T-1, N, M, 3) logits
            targets: (B, T-1, N, M) labels
        """
        batch_size = predictions.shape[0]
        self.total_loss += loss.item() * batch_size
        self.total_samples += batch_size
        
        # Compute accuracy
        pred_classes = predictions.argmax(dim=-1)  # (B, T-1, N, M)
        
        # Mask out padding (assuming -100 or negative values are padding)
        valid_mask = targets >= 0
        
        correct = (pred_classes == targets) & valid_mask
        self.correct += correct.sum().item()
        self.total_predictions += valid_mask.sum().item()
        
        # Per-class accuracy
        for class_idx in range(3):
            class_mask = (targets == class_idx) & valid_mask
            class_correct = (pred_classes == class_idx) & class_mask
            
            self.class_correct[class_idx] += class_correct.sum().item()
            self.class_total[class_idx] += class_mask.sum().item()
    
    def get_metrics(self) -> Dict[str, float]:
        """Compute and return current metrics."""
        avg_loss = self.total_loss / max(self.total_samples, 1)
        accuracy = self.correct / max(self.total_predictions, 1)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy
        }
        
        # Add per-class accuracies
        for class_idx in range(3):
            class_acc = (self.class_correct[class_idx] / 
                        max(self.class_total[class_idx], 1))
            metrics[f'accuracy_class_{class_idx}'] = class_acc
        
        return metrics


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, metric_value):
        if self.best_value is None:
            self.best_value = metric_value
            return False
        
        if self.mode == 'min':
            improved = metric_value < (self.best_value - self.min_delta)
        else:
            improved = metric_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = metric_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class Trainer:
    """Main trainer class for GNN tracking model.
    
    Args:
        model: GNNTrackingModel instance
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler (optional)
        loss_fn: Loss function
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for tensorboard logs
        max_epochs: Maximum number of training epochs
        gradient_clip: Max gradient norm for clipping
        early_stopping_patience: Patience for early stopping
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        scheduler=None,
        loss_fn=None,
        device='cuda',
        checkpoint_dir='./checkpoints',
        log_dir='./logs',
        max_epochs=100,
        gradient_clip=1.0,
        early_stopping_patience=10
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_epochs = max_epochs
        self.gradient_clip = gradient_clip
        
        # Loss function
        if loss_fn is None:
            # Default: weighted cross-entropy to handle class imbalance
            # Typically: no_link >> different > same_cell in frequency
            class_weights = [0.1, 1.0, 2.0]  # Adjust based on your data
            self.loss_fn = TrackingLoss(class_weights=class_weights)
        else:
            self.loss_fn = loss_fn
        
        # Move to device
        self.model = self.model.to(device)
        self.loss_fn = self.loss_fn.to(device)
        
        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode='min'
        )
        
        # Tracking
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = []
        self.val_history = []
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metrics_tracker = MetricsTracker()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            appearances = batch['appearances'].to(self.device)
            morphologies = batch['morphologies'].to(self.device)
            centroids = batch['centroids'].to(self.device)
            adj_matrices = batch['adj_matrices'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            predictions = self.model.training_forward(
                appearances, morphologies, centroids, adj_matrices,
                return_logits=True
            )
            
            # Compute loss
            loss = self.loss_fn(predictions, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.gradient_clip
                )
            
            self.optimizer.step()
            
            # Update metrics
            with torch.no_grad():
                predictions_probs = torch.softmax(predictions, dim=-1)
                metrics_tracker.update(loss, predictions_probs, labels)
            
            # Update progress bar
            current_metrics = metrics_tracker.get_metrics()
            pbar.set_postfix({
                'loss': f"{current_metrics['loss']:.4f}",
                'acc': f"{current_metrics['accuracy']:.4f}"
            })
        
        return metrics_tracker.get_metrics()
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        metrics_tracker = MetricsTracker()
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch} [Val]')
        
        for batch in pbar:
            # Move batch to device
            appearances = batch['appearances'].to(self.device)
            morphologies = batch['morphologies'].to(self.device)
            centroids = batch['centroids'].to(self.device)
            adj_matrices = batch['adj_matrices'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            predictions = self.model.training_forward(
                appearances, morphologies, centroids, adj_matrices,
                return_logits=True
            )
            
            # Compute loss
            loss = self.loss_fn(predictions, labels)
            
            # Update metrics
            predictions_probs = torch.softmax(predictions, dim=-1)
            metrics_tracker.update(loss, predictions_probs, labels)
            
            # Update progress bar
            current_metrics = metrics_tracker.get_metrics()
            pbar.set_postfix({
                'loss': f"{current_metrics['loss']:.4f}",
                'acc': f"{current_metrics['accuracy']:.4f}"
            })
        
        return metrics_tracker.get_metrics()
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"  💾 Saved best model (val_loss: {self.best_val_loss:.4f})")
        
        # Keep only last 3 checkpoints to save space
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if len(checkpoints) > 3:
            for old_checkpoint in checkpoints[:-3]:
                old_checkpoint.unlink()
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop."""
        print("="*70)
        print("Starting Training")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Max epochs: {self.max_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("="*70)
        print()
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            self.val_history.append(val_metrics)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('Learning_rate', current_lr, epoch)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch} Summary ({epoch_time:.1f}s):")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
                break
            
            print()
        
        total_time = time.time() - start_time
        print("="*70)
        print(f"Training completed in {total_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*70)
        
        self.writer.close()
        
        # Save final training history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump({
                'train': self.train_history,
                'val': self.val_history
            }, f, indent=2)


def create_optimizer(model, config):
    """Create optimizer based on config."""
    optimizer_name = config.get('optimizer', 'adam').lower()
    lr = config.get('learning_rate', 1e-3)
    weight_decay = config.get('weight_decay', 1e-5)
    
    if optimizer_name == 'adam':
        optimizer = optim.RAdam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'sgd':
        momentum = config.get('momentum', 0.9)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer, config):
    """Create learning rate scheduler."""
    scheduler_name = config.get('scheduler', 'reduce_on_plateau').lower()
    
    if scheduler_name == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
            )
    elif scheduler_name == 'cosine':
        T_max = config.get('max_epochs', 100)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=1e-6
        )
    elif scheduler_name == 'step':
        step_size = config.get('step_size', 30)
        gamma = config.get('gamma', 0.1)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif scheduler_name == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    return scheduler


# Example usage
if __name__ == "__main__":

    train_loader, val_loader, _ = create_trk_dataloaders(train_path='/home/sholtzen/torch-tracking/data/DynamicNuclearNet-tracking-v1_0/train.zarr',
                                                         test_path='/home/sholtzen/torch-tracking/data/DynamicNuclearNet-tracking-v1_0/test.zarr',
                                                         val_path='/home/sholtzen/torch-tracking/data/DynamicNuclearNet-tracking-v1_0/val.zarr',
                                                         batch_size=2)
    
    model = GNNTrackingModel(max_cells = train_loader.dataset.max_cells, graph_layer='gat', data_format='channels_last')

    config = {
        'optimizer': 'adam',
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'scheduler': 'reduce_on_plateau',
        'max_epochs': 100
    }

    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device='cuda:6',
        checkpoint_dir='./checkpoints',
        max_epochs=100,
        gradient_clip=1.0
    )   

    trainer.train()
    
    print("To use this training loop:")
    print()
    print("```python")
    print("from gnn_tracking_model import GNNTrackingModel")
    print("from trk_data_loader import create_trk_dataloaders")
    print("from trainer import Trainer, create_optimizer, create_scheduler")
    print()
    print("# Create model")
    print("model = GNNTrackingModel(")
    print("    max_cells=39,")
    print("    track_length=8,")
    print("    n_filters=64,")
    print("    encoder_dim=64,")
    print("    embedding_dim=64")
    print(")")
    print()
    print("# Create dataloaders")
    print("train_loader, val_loader, _ = create_trk_dataloaders(")
    print("    train_path='train.trk',")
    print("    val_path='val.trk',")
    print("    batch_size=4,")
    print("    track_length=8")
    print(")")
    print()
    print("# Create optimizer and scheduler")
    print("config = {")
    print("    'optimizer': 'adam',")
    print("    'learning_rate': 1e-3,")
    print("    'weight_decay': 1e-5,")
    print("    'scheduler': 'reduce_on_plateau',")
    print("    'max_epochs': 100")
    print("}")
    print()
    print("optimizer = create_optimizer(model, config)")
    print("scheduler = create_scheduler(optimizer, config)")
    print()
    print("# Create trainer")
    print("trainer = Trainer(")
    print("    model=model,")
    print("    train_loader=train_loader,")
    print("    val_loader=val_loader,")
    print("    optimizer=optimizer,")
    print("    scheduler=scheduler,")
    print("    device='cuda',")
    print("    checkpoint_dir='./checkpoints',")
    print("    max_epochs=100,")
    print("    gradient_clip=1.0")
    print(")")
    print()
    print("# Train!")
    print("trainer.train()")
    print("```")
    print()
    print("Features:")
    print("  ✓ Automatic checkpointing")
    print("  ✓ TensorBoard logging")
    print("  ✓ Early stopping")
    print("  ✓ Learning rate scheduling")
    print("  ✓ Gradient clipping")
    print("  ✓ Per-class accuracy tracking")
    print("  ✓ Progress bars with tqdm")
    print("  ✓ Weighted loss for class imbalance")