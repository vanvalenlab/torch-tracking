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
from loader import create_trk_dataloaders

class TrackingLoss(nn.Module):
    
    def __init__(self, alpha=None, gamma=2.0, use_focal=True):
        super().__init__()
        
        if alpha is None:
            # Default: aggressive weighting for minority classes
            # Adjust these based on your actual class distribution
            alpha = torch.tensor([1.0, 20.0, 30.0])  # [no_link, same_cell, mitosis]
        
        self.alpha = alpha
        self.gamma = gamma
        self.use_focal = use_focal
        self.pad_value = -1
        
        # Standard CrossEntropyLoss with class weights
        self.criterion = nn.CrossEntropyLoss(weight=alpha, reduction='none', ignore_index=self.pad_value)

    def _weighted_categorical_crossentropy(self, y_true, y_pred,
                                        n_classes=3, axis=None,
                                        from_logits=False):
        
        """Categorical crossentropy between an output tensor and a target tensor.
        Automatically computes the class weights from the target image and uses
        them to weight the cross entropy

        Args:
            y_true: A tensor of the same shape as ``y_pred``.
            y_pred: A tensor resulting from a softmax
                (unless ``from_logits`` is ``True``, in which
                case ``y_pred`` is expected to be the logits).
            from_logits: Boolean, whether ``y_pred`` is the
                result of a softmax, or is a tensor of logits.

        Returns:
            tensor: Output tensor.
        """

        # scale preds so that the class probas of each sample sum to 1
        y_pred = y_pred / torch.sum(y_pred, dim=axis, keepdims=True)
        # manual computation of crossentropy
        eps=1e-10
        _epsilon = torch.tensor(eps).type(y_pred.dtype).to(y_pred.device)
        y_pred = torch.clamp(y_pred, min=_epsilon, max=(1. - _epsilon))
        total_sum = torch.sum(y_true)
        class_sum = torch.sum(y_true, dim=0, keepdims=True)
        class_weights = 1.0 / n_classes * torch.divide(total_sum, class_sum + 1.)
        return - torch.mean((y_true * torch.log(y_pred) * class_weights), dim=axis)
    
    def forward(self, predictions, targets):
        
        """
        Args:
            predictions: (B, T, N, M, 3) logits from model
            targets: (B, T, N, M) integer class labels [0, 1, 2]
        
        Returns:
            loss: scalar loss value
        """

        # # Reshape for CrossEntropyLoss
        predictions_flat = predictions.view(-1, predictions.shape[-1])  # (B*T*N*M, 3)
        targets_flat = targets.view(-1, targets.shape[-1]).long()  # (B*T*N*M)
        
        # Compute weighted cross-entropy
        ce_loss = self._weighted_categorical_crossentropy(targets_flat, predictions_flat, targets_flat.shape[-1])
        
        # Apply focal loss modulation
        if self.use_focal:
            with torch.no_grad():
                # Get probability of true class
                pt = torch.exp(-ce_loss)
            
            # Apply focal weight: focus on hard examples
            focal_weight = (1 - pt) ** self.gamma
            loss = focal_weight * ce_loss
        else:
            loss = ce_loss
        
        return loss.mean()


class MetricsTracker:
    """FIXED: Proper masking and per-class metrics.
    
    Replace the MetricsTracker class in training.py with this version.
    """
    
    def __init__(self):
        self.reset()
        self.pad_value = -1
    
    def reset(self):
        self.total_loss = 0.0
        self.total_samples = 0
        self.correct = 0
        self.total_predictions = 0
        
        # Per-class metrics
        self.class_correct = {0: 0, 1: 0, 2: 0}
        self.class_total = {0: 0, 1: 0, 2: 0}
        self.class_predicted = {0: 0, 1: 0, 2: 0}
    
    def update(self, loss, predictions, targets):
        """
        Args:
            loss: scalar loss value
            predictions: (B, T, N, M, 3) logits
            targets: (B, T, N, M) integer labels
            actual_max_cells: int, actual number of cells in this batch
        """

        batch_size = predictions.shape[0]
        self.total_loss += loss.item() * batch_size
        self.total_samples += batch_size
        
        # Get predicted classes
        pred_classes = predictions.argmax(dim=-1)  # (B*T*N*M)
        target_classes = targets.argmax(dim=-1)
                
        # Only consider pairs within actual_max_cells range
        valid_mask = target_classes != self.pad_value

        # Overall accuracy
        correct = (pred_classes == target_classes)
        self.correct += correct.sum().item()
        self.total_predictions += target_classes.sum().item()
        
        # Per-class metrics
        for class_idx in range(3):
            class_mask = (target_classes == class_idx)
            class_predicted_mask = (pred_classes == class_idx)
            class_correct = (pred_classes == class_idx) & class_mask 
            
            self.class_correct[class_idx] += class_correct.sum().item()
            self.class_total[class_idx] += class_mask.sum().item()
            self.class_predicted[class_idx] += class_predicted_mask.sum().item()
    
    def get_metrics(self):
        """Compute and return current metrics."""
        avg_loss = self.total_loss / max(self.total_samples, 1)
        accuracy = self.correct / max(self.total_predictions, 1)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy
        }
        
        # Add per-class metrics
        for class_idx in range(3):
            # Recall: of all true class_idx, how many did we predict correctly?
            recall = (self.class_correct[class_idx] / 
                     max(self.class_total[class_idx], 1))
            
            # Precision: of all predicted class_idx, how many were correct?
            precision = (self.class_correct[class_idx] / 
                        max(self.class_predicted[class_idx], 1))
            
            # F1 score
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            
            metrics[f'recall_class_{class_idx}'] = recall
            metrics[f'precision_class_{class_idx}'] = precision
            metrics[f'f1_class_{class_idx}'] = f1
        
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
            self.loss_fn = TrackingLoss(use_focal=False, gamma=3.0)
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
            
            if batch_idx >= 512:
                break

            # Unpad and move batch to device
            appearances = batch['appearances'].to(self.device)
            morphologies = batch['morphologies'].to(self.device)
            centroids = batch['centroids'].to(self.device)
            adj_matrices = batch['adj_matrices'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            predictions = self.model.training_forward(
                appearances, morphologies, centroids, adj_matrices,
                return_logits=False
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
                metrics_tracker.update(loss, predictions, labels)
            
            # Update progress bar
            current_metrics = metrics_tracker.get_metrics()
            pbar.set_postfix({
                'mit_acc': f"{current_metrics['precision_class_2']:.4f}",
                'same_acc': f"{current_metrics['precision_class_1']:.4f}",
                'no_acc': f"{current_metrics['precision_class_0']:.4f}",
                'loss': f"{current_metrics['loss']:.4f}",

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
            # mask = batch['mask'].to(self.device)

            # Forward pass
            predictions = self.model.training_forward(
                appearances, morphologies, centroids, adj_matrices,
                return_logits=False
            )
            
            # Compute loss
            loss = self.loss_fn(predictions, labels)
            
            # Update metrics
            metrics_tracker.update(loss, predictions, labels)
            
            # Update progress bar
            current_metrics = metrics_tracker.get_metrics()
            pbar.set_postfix({
                'mit_acc': f"{current_metrics['precision_class_2']:.4f}",
                'same_acc': f"{current_metrics['precision_class_1']:.4f}",
                'no_acc': f"{current_metrics['precision_class_0']:.4f}",
                'loss': f"{current_metrics['loss']:.4f}"
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

            for i in range(3):
                self.writer.add_scalar(f'Precision/train/class_{i}', train_metrics[f'precision_class_{i}'], epoch)
                self.writer.add_scalar(f'Precision/val/class_{i}', val_metrics[f'precision_class_{i}'], epoch)
                self.writer.add_scalar(f'Recall/train/class_{i}', train_metrics[f'recall_class_{i}'], epoch)
                self.writer.add_scalar(f'Recall/val/class_{i}', val_metrics[f'recall_class_{i}'], epoch)

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
            factor=0.1,
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

    model = GNNTrackingModel(
                             graph_layer='gat', 
                             data_format='channels_last',
                             encoder_dim=64
                             )
    
    train_loader, val_loader, _ = create_trk_dataloaders(train_path='/home/sholtzen/torch-tracking/data/DynamicNuclearNet-tracking-v1_0/train.zarr',
                                                         val_path='/home/sholtzen/torch-tracking/data/DynamicNuclearNet-tracking-v1_0/val.zarr',
                                                         batch_size=6,
                                                         distance_threshold=64,
                                                         augment=True,
                                                         num_workers=4)
    
    config = {
        'optimizer': 'adam',
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'scheduler': 'reduce_on_plateau',
        'max_epochs': 50
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
        max_epochs=config['max_epochs'],
        gradient_clip=0.001
    )   

    trainer.train()
