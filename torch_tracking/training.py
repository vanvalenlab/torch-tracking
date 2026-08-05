import torch
import time
import datetime
import json

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from typing import Dict

from torch_tracking.model import GNNTrackingModel
from torch_tracking.loader import create_trk_dataloaders
from torch_tracking.loss import TrackingLoss
from torch_tracking.utils import MetricsTracker, EarlyStopping, create_optimizer, create_scheduler


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
        device='cuda',
        checkpoint_dir='./checkpoints/',
        log_dir='./logs/',
        max_epochs=50,
        gradient_clip=1.0,
        early_stopping_patience=10,
        enable_early_stopping=False,
        log_and_save=True,
        config=None,
        class_weights=None,
        loss='wcce',
        stopping_metric = 'loss',
        gamma=0.1,
        data_precision = 'bfloat16'
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
        self.stopping_metric = stopping_metric
        self.return_logits = False
        self.gamma = gamma
        self.metrics_tracker = MetricsTracker()
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.class_weights = class_weights

        self.log_suffix = curr_time
        self.log_dir = Path(log_dir +  self.log_suffix)
        self.checkpoint_dir = Path(checkpoint_dir + self.log_suffix)

        self.loss_fn = TrackingLoss(loss=self.loss, gamma=self.gamma, class_weights=self.class_weights)
        
        # Move to device
        self.model = self.model.to(device)
        self.loss_fn = self.loss_fn.to(device)

        # Altering data precision values
        if data_precision == 'bfloat16':
            self.autocast = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
            self.scaler = torch.amp.GradScaler()

        elif data_precision == 'float32':
            self.autocast = torch.amp.autocast(device_type='cuda', dtype=torch.float32)
            self.scaler = torch.amp.GradScaler(enabled=False)

        elif data_precision == 'float16':
            self.autocast = torch.amp.autocast(device_type='cuda', dtype=torch.float16)
            self.scaler = torch.amp.GradScaler()

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

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch} [Train]', dynamic_ncols=True)
        
        for _, batch in enumerate(pbar):

            # Unpad and move batch to device
            appearances = batch['appearances'].to(self.device)
            morphologies = batch['morphologies'].to(self.device)
            centroids = batch['centroids'].to(self.device)
            adj_matrices = batch['adj_matrices'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            with self.autocast:

                predictions = self.model.training_forward(
                    appearances, morphologies, centroids, adj_matrices,
                    return_logits=self.return_logits
                )
            
                # Compute loss
                loss = self.loss_fn(predictions, labels)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer) # Unscales the gradients in-place

            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {self.current_epoch}, skipping batch")
                optimizer.zero_grad()
                continue

            if torch.isnan(loss) or torch.isinf(loss):
                # Log which batch and what the probs looked like
                print(f"  min prob: {predictions.min().item():.2e}")
                print(f"  max prob: {predictions.max().item():.2e}")
                optimizer.zero_grad()
                continue

            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.gradient_clip
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update metrics
            with torch.no_grad():
                self.metrics_tracker.update(loss, predictions, labels)


            # Update progress bar
            current_metrics = self.metrics_tracker.get_metrics()
            pbar.set_postfix({
                'no_f1': f"{current_metrics['f1_class_0']:.4f}",
                'same_f1': f"{current_metrics['f1_class_1']:.4f}",
                'mit_f1': f"{current_metrics['f1_class_2']:.4f}",
                'loss': f"{current_metrics['loss']:.4f}"
            })

        metrics = self.metrics_tracker.get_metrics()

        self.metrics_tracker.reset()

        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch} [Val]', dynamic_ncols=True)
        
        for batch_idx, batch in enumerate(pbar):

            # Move batch to device
            appearances = batch['appearances'].to(self.device)
            morphologies = batch['morphologies'].to(self.device)
            centroids = batch['centroids'].to(self.device)
            adj_matrices = batch['adj_matrices'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            predictions = self.model.training_forward(
                appearances, morphologies, centroids, adj_matrices,
                return_logits=self.return_logits
            )
            
            # Compute loss
            loss = self.loss_fn(predictions, labels)

            if torch.isnan(loss):
                continue


            # Update metrics
            self.metrics_tracker.update(loss, predictions, labels)

            # Update progress bar
            current_metrics = self.metrics_tracker.get_metrics()
            pbar.set_postfix({
                'no_f1': f"{current_metrics['f1_class_0']:.4f}",
                'same_f1': f"{current_metrics['f1_class_1']:.4f}",
                'mit_f1': f"{current_metrics['f1_class_2']:.4f}",
                'loss': f"{current_metrics['loss']:.4f}"
            })

        metrics = self.metrics_tracker.get_metrics()

        self.metrics_tracker.reset()

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
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics[self.stopping_metric])
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']

            # Save checkpoint
            is_best = val_metrics[self.stopping_metric] < self.best_val_loss
            
            if is_best:
                self.best_val_loss = val_metrics[self.stopping_metric]
                self.save_checkpoint(is_best=True)


            if self.writer is not None:
            # Log to tensorboard
                for k, v in train_metrics.items():
                    self.writer.add_scalar(f'train/{k}', v, epoch)
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f'val/{k}', v, epoch)

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


# Example usage
if __name__ == "__main__":

    # Make config dictionary

    config = {
        "optimizer": "radam",
        "learning_rate": 0.001,
        "weight_decay": 0,
        "decay": 0.99,
        "scheduler": "reduce_on_plateau",
        "max_epochs": 50,
        "batch_size": 8,
        "n_layers": 2,
        "num_workers": 4,
        "clipnorm": 1.0,
        "step_size": 5,
        "crop_mode": "fixed",
        "patience": 5,
        "log_and_save": True,
        "enable_early_stopping": False,
        "crop_size": 32,
        "truncate_dataset": None,
        "loss": "wcce",
        "dropout": 0,
        "device": "cuda:0",
        "label_smoothing": False,
        "stopping_metric": "loss",
        'data_precision': 'bfloat16',
        'gamma': 1.0
    }

    # Initialize model

    model = GNNTrackingModel(
        graph_layer='gat', 
        data_format='channels_last',
        encoder_dim=64,
        n_layers=config['n_layers'],
        crop_size=config['crop_size'],
        dropout=config['dropout']
    )
    

    # Create optimizer and rate scheduler
    
    train_loader, val_loader, _ = create_trk_dataloaders(
        train_path=Path.home() / '.deepcell/tracking/train_proc.zarr',
        val_path=Path.home() / '.deepcell/tracking/val_proc.zarr',
        batch_size=config['batch_size'],
        distance_threshold=64,
        num_workers=config['num_workers'],
        truncate_dataset = config['truncate_dataset'],
    )

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
        log_and_save = True,
        config=config,
        loss=config['loss'],
        stopping_metric = config['stopping_metric'],
        data_precision=config['data_precision'],
        gamma=config['gamma'],
        class_weights=[1,10,100]
    )   

    trainer.train()
