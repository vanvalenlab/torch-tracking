import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import json
from typing import Dict
import time
import datetime
from model import GNNTrackingModel
from loader import create_trk_dataloaders
from utils import weighted_categorical_crossentropy_v2
import numpy as np

import torch.nn.functional as F

def focal_loss(logits, labels, alpha=None, gamma=2):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      logits: A float tensor of size [batch, num_classes].
      labels: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    bc_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * bc_loss

    if alpha is not None:
        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)
    else:
        focal_loss = torch.sum(loss)

    focal_loss /= torch.sum(labels)
    return focal_loss

class Loss(torch.nn.Module):
    def __init__(
        self,
        loss_type: str = "cross_entropy",
        beta: float = 0.999,
        fl_gamma=2,
        samples_per_class=None,
        class_balanced=False,
        safe: bool = False,
    ):
        """
        Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.

        reference: https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf

        Args:
            loss_type: string. One of "focal_loss", "cross_entropy",
                "binary_cross_entropy", "softmax_binary_cross_entropy".
            beta: float. Hyperparameter for Class balanced loss.
            fl_gamma: float. Hyperparameter for Focal loss.
            samples_per_class: A python list of size [num_classes].
                Required if class_balance is True.
            class_balanced: bool. Whether to use class balanced loss.
            safe: bool. Whether to allow labels with no samples.
        Returns:
            Loss instance
        """
        super(Loss, self).__init__()

        if class_balanced is True and samples_per_class is None:
            raise ValueError("samples_per_class cannot be None when class_balanced is True")

        self.loss_type = loss_type
        self.beta = beta
        self.fl_gamma = fl_gamma
        self.samples_per_class = samples_per_class
        self.class_balanced = class_balanced
        self.safe = safe

    def forward(self, logits: torch.tensor, labels: torch.tensor):
        """
        Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.

        Args:
            logits: A float tensor of size [batch, num_classes].
            labels: An int tensor of size [batch].
        Returns:
            cb_loss: A float tensor representing class balanced loss
        """

        batch_size = logits.size(0)
        num_classes = logits.size(1)
        labels_one_hot = F.one_hot(labels, num_classes).float()

        if self.class_balanced:
            effective_num = 1.0 - np.power(self.beta, self.samples_per_class)
            # Avoid division by 0 error for test cases without all labels present.
            if self.safe:
                effective_num_classes = np.sum(effective_num != 0)
                effective_num[effective_num == 0] = np.inf

            else:
                effective_num_classes = num_classes

            weights = (1.0 - self.beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * effective_num_classes
            weights = torch.tensor(weights, device=logits.device).float()

            if self.loss_type != "cross_entropy":
                weights = weights.unsqueeze(0)
                weights = weights.repeat(batch_size, 1) * labels_one_hot
                weights = weights.sum(1)
                weights = weights.unsqueeze(1)
                weights = weights.repeat(1, num_classes)
        else:
            weights = None

        if self.loss_type == "focal_loss":
            cb_loss = focal_loss(logits, labels_one_hot, alpha=weights, gamma=self.fl_gamma)
        elif self.loss_type == "cross_entropy":
            cb_loss = F.cross_entropy(input=logits, target=labels_one_hot, weight=weights, reduction='none')
        elif self.loss_type == "binary_cross_entropy":
            cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
        elif self.loss_type == "softmax_binary_cross_entropy":
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
        return cb_loss

class TrackingLoss(nn.Module):
    
    def __init__(self, gamma=2.0, loss='wcce', label_smoothing=False, class_weights='batch', class_samples=None):
        super().__init__()
        
        self.weights = class_weights
        self.gamma = gamma
        self.loss = loss
        self.pad_value = -1
        self.label_smoothing = label_smoothing
        self.class_samples = class_samples

        if self.loss == 'balanced':
            self.criterion = Loss(loss_type='cross_entropy', 
                                  class_balanced=True, 
                                  samples_per_class=self.class_samples,
                                  safe=True)
            
        # Standard CrossEntropyLoss with class weights
        elif self.loss == 'cce':
            self.criterion = nn.CrossEntropyLoss(reduction='none', 
                                                 ignore_index=self.pad_value)

    
    def forward(self, predictions, targets, mask):
        
        """
        Args:
            predictions: (B, T, N, M, 3) logits from model
            targets: (B, T, N, M) integer class labels [0, 1, 2]
        
        Returns:
            loss: scalar loss value
        """

        # # Reshape for CrossEntropyLoss
        predictions_flat = predictions.view(-1, predictions.shape[-1])  # (B*T*N*M, 3)
        targets_flat = targets.view(-1, targets.shape[-1])
        valid_mask = mask.view(-1)

        # Compute weighted cross-entropy
        if self.loss == 'wcce':

            loss = weighted_categorical_crossentropy_v2(
                targets_flat, 
                predictions_flat, 
                n_classes=3, 
                label_smoothing=self.label_smoothing, 
                class_weights=self.weights
            )

            return loss[valid_mask,:].mean()

        else:
            loss = self.criterion(predictions_flat, targets_flat.argmax(dim=-1))
            return loss[valid_mask].mean()
            
class MetricsTracker:
    
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
    
    def update(self, loss, predictions, targets, valid_mask):
        """
        Args:
            loss: scalar loss value
            predictions: (B, T, N, M, 3) logits
            targets: (B, T, N, M, 3) one hot encoding of labels
        """

        batch_size = predictions.shape[0]
        self.total_loss += loss.item() * batch_size
        self.total_samples += batch_size


        # Get predicted classes
        pred_classes = predictions.argmax(dim=-1)  # (B*T*N*M)
        target_classes = targets.argmax(dim=-1)
    

        # Overall accuracy
        correct = (pred_classes == target_classes) & valid_mask
        self.correct += correct.sum().item()
        self.total_predictions += valid_mask.sum().item()
        
        # Per-class metrics
        for class_idx in range(3):
            class_mask = (target_classes == class_idx) & valid_mask
            class_predicted_mask = (pred_classes == class_idx) & valid_mask
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

        running_f1 = 0
        
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
            running_f1 *= f1
        
        metrics['geom_f1'] = f1 ** (1/3)
        
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
        checkpoint_dir='./checkpoints/',
        log_dir='./logs/',
        max_epochs=100,
        gradient_clip=1.0,
        early_stopping_patience=10,
        enable_early_stopping=True,
        log_and_save=True,
        config=None,
        loss='wcce',
        label_smoothing=False,
        class_weights='batch',
        stopping_metric = 'loss'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_epochs = max_epochs
        self.gradient_clip = gradient_clip
        self.enable_early_stopping = enable_early_stopping
        self.log_and_save = log_and_save
        self.config=config
        self.loss=loss
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
        self.stopping_metric = stopping_metric

        if loss=='wcce':
            self.return_logits = False
        else:
            self.return_logits = True

        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.log_suffix = curr_time
        self.log_dir = Path(log_dir +  self.log_suffix)
        self.checkpoint_dir = Path(checkpoint_dir + self.log_suffix)

        self.loss_fn = TrackingLoss(loss=loss, 
                                    label_smoothing=self.label_smoothing, 
                                    class_weights=self.class_weights,
                                    class_samples = train_loader.dataset.class_samples)
        
        # Move to device
        self.model = self.model.to(device)
        self.loss_fn = self.loss_fn.to(device)
        
        # Setup directories and Tensorboard writer
        if self.log_and_save:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)

            with open(f"{self.checkpoint_dir}/config.json", 'w') as f:
                json.dump(self.config, f, indent=4)
        
        else:
            self.writer = None
        
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

        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch} [Train]', dynamic_ncols=True)
        
        for batch_idx, batch in enumerate(pbar):

            # Unpad and move batch to device
            appearances = batch['appearances'].to(self.device)
            morphologies = batch['morphologies'].to(self.device)
            centroids = batch['centroids'].to(self.device)
            adj_matrices = batch['adj_matrices'].to(self.device)
            labels = batch['labels'].to(self.device)
            mask = batch['mask'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            
            predictions = self.model.training_forward(
                appearances, morphologies, centroids, adj_matrices,
                return_logits=self.return_logits
            )
            
            # Compute loss
            loss = self.loss_fn(predictions, labels, mask)
            
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
                metrics_tracker.update(loss, predictions, labels, mask)
            
            # Update progress bar
            current_metrics = metrics_tracker.get_metrics()
            pbar.set_postfix({
                'mit_f1': f"{current_metrics['f1_class_2']:.4f}",
                'same_f1': f"{current_metrics['f1_class_0']:.4f}",
                'no_f1': f"{current_metrics['f1_class_1']:.4f}",
                'loss': f"{current_metrics['loss']:.4f}",
            })

        metrics = metrics_tracker.get_metrics()

        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        metrics_tracker = MetricsTracker()
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch} [Val]', dynamic_ncols=True)
        
        for batch_idx, batch in enumerate(pbar):

            # Move batch to device
            appearances = batch['appearances'].to(self.device)
            morphologies = batch['morphologies'].to(self.device)
            centroids = batch['centroids'].to(self.device)
            adj_matrices = batch['adj_matrices'].to(self.device)
            labels = batch['labels'].to(self.device)
            mask = batch['mask'].to(self.device)

            # Forward pass
            predictions = self.model.training_forward(
                appearances, morphologies, centroids, adj_matrices,
                return_logits=self.return_logits
            )
            
            # Compute loss
            loss = self.loss_fn(predictions, labels, mask)

            # Update metrics
            metrics_tracker.update(loss, predictions, labels, mask)
            
            # Update progress bar
            current_metrics = metrics_tracker.get_metrics()
            pbar.set_postfix({
                'mit_f1': f"{current_metrics['f1_class_2']:.4f}",
                'same_f1': f"{current_metrics['f1_class_0']:.4f}",
                'no_f1': f"{current_metrics['f1_class_1']:.4f}",
                'loss': f"{current_metrics['loss']:.4f}"
            })

        metrics = metrics_tracker.get_metrics()

        return metrics
    
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
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch:01d}.pt'
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print()
            print(f"    Saved best model (val_loss: {self.best_val_loss:.4f})")
        else:
            torch.save(checkpoint, checkpoint_path)

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
                    self.scheduler.step(val_metrics[self.stopping_metric])
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']

            # Save checkpoint
            is_best = val_metrics[self.stopping_metric] < self.best_val_loss
            
            if is_best:
                self.best_val_loss = val_metrics[self.stopping_metric]

            if self.writer is not None:
            # Log to tensorboard
                self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
                self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
                self.writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
                self.writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
                self.writer.add_scalar('Learning_rate', current_lr, epoch)
                self.writer.add_scalar('Accuracy/train_geom', train_metrics['geom_f1'], epoch)
                self.writer.add_scalar('Accuracy/val_geom', val_metrics['geom_f1'], epoch)

                for i in range(3):
                    self.writer.add_scalar(f'Precision/train/class_{i}', train_metrics[f'precision_class_{i}'], epoch)
                    self.writer.add_scalar(f'Precision/val/class_{i}', val_metrics[f'precision_class_{i}'], epoch)
                    self.writer.add_scalar(f'Recall/train/class_{i}', train_metrics[f'recall_class_{i}'], epoch)
                    self.writer.add_scalar(f'Recall/val/class_{i}', val_metrics[f'recall_class_{i}'], epoch)
                    self.writer.add_scalar(f'F1/train/class_{i}', train_metrics[f'f1_class_{i}'], epoch)
                    self.writer.add_scalar(f'F1/val/class_{i}', val_metrics[f'f1_class_{i}'], epoch)

                self.save_checkpoint(is_best=is_best)

            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch} Summary ({epoch_time:.1f}s):")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            if self.enable_early_stopping:
                # Early stopping
                if self.early_stopping(val_metrics[self.stopping_metric]):
                    print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
                    break
            
            print()

        self.save_checkpoint(is_best=is_best)

        total_time = time.time() - start_time
        print("="*70)
        print(f"Training completed in {total_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*70)

        if self.writer is not None:
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
    
    if optimizer_name == 'radam':
        optimizer = optim.RAdam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
    elif optimizer_name == 'muon':
        optimizer = optim.Muon(
            model.parameters(),
            lr=lr
            )

    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
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
        patience = config.get('patience', 5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=patience
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

    elif scheduler_name == 'exp':
        decay = config.get('decay', 0.99)
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=decay
        )

    elif scheduler_name == 'none':
        scheduler = None

    elif scheduler_name == 'caliban': 
        step_size = config.get('step_size', 30)
        decay = config.get('decay', 0.99)
        scheduler=optim.lr_scheduler.ChainedScheduler([        
            optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=decay
            ),
            optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size
            )
        ])
        
    return scheduler


# Example usage
if __name__ == "__main__":

    # Make config dictionary

    config = {
        "optimizer": "radam",
        "learning_rate": 0.00001,
        "weight_decay": 0,
        "decay": 0.99,
        "scheduler": "cosine",
        "max_epochs": 30,
        "batch_size": 4,
        "n_layers": 1,
        "num_workers": 8,
        "clipnorm": 0.001,
        "step_size": 5,
        "crop_mode": "fixed",
        "patience": 5,
        "log_and_save": True,
        "enable_early_stopping": False,
        "crop_size": 32,
        "attention": False,
        "truncate_dataset": None,
        "loss": "balanced",
        "t_direction": "forward",
        "processed": True,
        "dropout": 0,
        "device": "cuda:0",
        "label_smoothing": False,
        "stopping_metric": 'loss'
    }

    # Initialize model

    model = GNNTrackingModel(
                             graph_layer='gat', 
                             data_format='channels_last',
                             encoder_dim=64,
                             n_layers=config['n_layers'],
                             crop_size=config['crop_size'],
                             attention=config['attention'],
                             dropout=config['dropout']
                             )
    

    # Create optimizer and rate scheduler
    
    train_loader, val_loader, _ = create_trk_dataloaders(
        train_path='data/DynamicNuclearNet-tracking-v1_0/train_proc.zarr',
        val_path='data/DynamicNuclearNet-tracking-v1_0/val_proc.zarr',
        batch_size=config['batch_size'],
        distance_threshold=64,
        augment=True,
        crop_mode=config['crop_mode'],
        num_workers=config['num_workers'],
        crop_size=config['crop_size'],
        truncate_dataset = config['truncate_dataset'],
        t_direction=config['t_direction'],
        processed=config['processed']
        )

    val_loader.dataset.augment = False

    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config['device'],
        checkpoint_dir='./checkpoints/',
        max_epochs=config['max_epochs'],
        gradient_clip=config['clipnorm'],
        enable_early_stopping=config['enable_early_stopping'],
        log_and_save = True if config['truncate_dataset'] is None else True,
        config=config,
        loss=config['loss'],
        label_smoothing = config['label_smoothing'],
        class_weights='batch',
        stopping_metric = config['stopping_metric'],
    )   

    # torch.tensor([2.177e-2, 6.713e-5, 2.97815741]).float()

    trainer.train()
