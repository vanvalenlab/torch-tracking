import torch
import time
import datetime
import json
from dataclasses import dataclass, field
from typing import Any

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from typing import Dict

from torch_tracking.model import GNNTrackingModel
from torch_tracking.loader import create_trk_dataloaders
from torch_tracking.loss import TrackingLoss
from torch_tracking.utils import MetricsTracker, EarlyStopping, create_optimizer, create_scheduler

@dataclass
class TrainingConfig:
    """Configuration for a :class:`Trainer` run.

    Parameters
    ----------
    model : torch.nn.Module or None, optional
        The model to train. Default is None.
    train_loader : torch.utils.data.DataLoader or None, optional
        DataLoader providing training batches. Default is None.
    val_loader : torch.utils.data.DataLoader or None, optional
        DataLoader providing validation batches. Default is None.
    optimizer : str, optional
        Name of the optimizer to build via ``create_optimizer``. Default is
        ``'radam'``.
    scheduler : str, optional
        Name of the learning rate scheduler to build via
        ``create_scheduler``. Default is ``'reduce_on_plateau'``.
    crop_mode : str, optional
        Cropping mode used when extracting appearance features. Default is
        ``'fixed'``.
    device : str, optional
        Device to train on. Default is ``'cuda'``.
    checkpoint_dir : str, optional
        Base directory in which checkpoints are saved. A timestamped
        subdirectory is created under this path for each run. Default is
        ``'./checkpoints/'``.
    log_dir : str, optional
        Base directory for TensorBoard logs. A timestamped subdirectory is
        created under this path for each run. Default is ``'./logs/'``.
    step_size : int, optional
        Step size (in epochs) for step-based learning rate schedulers.
        Default is 5.
    max_epochs : int, optional
        Maximum number of training epochs. Default is 50.
    num_workers : int, optional
        Number of worker processes for data loading. Default is 4.
    n_layers : int, optional
        Number of GNN layers in the model. Default is 2.
    clipnorm : float, optional
        Maximum gradient norm used for gradient clipping. Default is 1.0.
    batch_size : int, optional
        Number of samples per training batch. Default is 8.
    learning_rate : float, optional
        Initial learning rate. Default is 0.001.
    patience : int, optional
        Number of epochs with no improvement before early stopping
        triggers. Default is 5.
    enable_early_stopping : bool, optional
        Whether to stop training early when the stopping metric stops
        improving. Default is False.
    log_and_save : bool, optional
        Whether to write TensorBoard logs and save checkpoints/config to
        disk. Default is True.
    class_weights : list of float, optional
        Per-class weights used by the loss function. Default is
        ``[1, 10, 100]``.
    loss : str, optional
        Name of the loss function used by :class:`~torch_tracking.loss.TrackingLoss`.
        Default is ``'wcce'``.
    stopping_metric : str, optional
        Key into the validation metrics dict used to select the best
        checkpoint and drive early stopping/scheduling. Default is
        ``'loss'``.
    gamma : float, optional
        Focal loss focusing parameter, passed to
        :class:`~torch_tracking.loss.TrackingLoss`. Default is 1.0.
    truncate_dataset : int or None, optional
        If set, limits the dataset to this many samples. Default is None.
    data_precision : str, optional
        Floating point precision used for autocast/gradient scaling.
        One of ``'bfloat16'``, ``'float32'``, or ``'float16'``. Default is
        ``'bfloat16'``.
    label_smoothing : bool, optional
        Whether to apply label smoothing in the loss function. Default is
        False.
    dropout : float, optional
        Dropout probability used in the model. Default is 0.
    crop_size : int, optional
        Size of the appearance crop fed to the model. Default is 32.
    decay : float, optional
        Decay factor used by the optimizer/scheduler. Default is 0.99.
    weight_decay : float, optional
        Weight decay (L2 penalty) used by the optimizer. Default is 0.0.

    Returns
    -------
    TrainingConfig
        An initialized ``TrainingConfig`` object.
    """
    model: nn.Module = None
    train_loader: torch.utils.data.DataLoader = None
    val_loader: torch.utils.data.DataLoader = None
    optimizer: str = 'radam'
    scheduler: str = 'reduce_on_plateau'
    crop_mode: str = 'fixed'
    device: str = 'cuda'
    checkpoint_dir: str = './checkpoints/'
    log_dir: str = './logs/'
    step_size: int = 5
    max_epochs: int = 50
    num_workers: int = 4
    n_layers: int = 2
    clipnorm: float = 1.0
    batch_size: int = 8
    learning_rate: float = 0.001
    patience: int = 5
    enable_early_stopping: bool = False
    log_and_save: bool = True
    class_weights: list[float] = field(default_factory=lambda: [1, 10, 100])
    loss: str = 'wcce'
    stopping_metric: str = 'loss'
    gamma: float = 1.0
    truncate_dataset: int = None
    data_precision: str = 'bfloat16'
    label_smoothing: bool = False
    dropout: float = 0
    crop_size: int = 32
    decay: float = 0.99
    weight_decay: float = 0.0

    def to_dict(self):
        """Serialize the config's scalar training settings to a dictionary.

        Excludes non-serializable fields (``model``, ``train_loader``,
        ``val_loader``) and ``class_weights``. Used to persist the run
        configuration alongside checkpoints.

        Returns
        -------
        dict
            A JSON-serializable dictionary of training hyperparameters.
        """
        return {
        "log_dir": self.log_dir,
        "optimizer": self.optimizer,
        "learning_rate": self.learning_rate,
        "weight_decay": self.weight_decay,
        "decay": self.decay,
        "checkpoint_dir": self.checkpoint_dir,
        "scheduler": self.scheduler,
        "max_epochs": self.max_epochs,
        "batch_size": self.batch_size,
        "n_layers": self.n_layers,
        "num_workers": self.num_workers,
        "clipnorm": self.clipnorm,
        "step_size": self.step_size,
        "crop_mode": self.crop_mode,
        "patience": self.patience,
        "log_and_save": self.log_and_save,
        "enable_early_stopping": self.enable_early_stopping,
        "crop_size": self.crop_size,
        "truncate_dataset": self.truncate_dataset,
        "loss": self.loss,
        "dropout": self.dropout,
        "device": self.device,
        "label_smoothing": self.label_smoothing,
        "stopping_metric": self.stopping_metric,
        'data_precision': self.data_precision,
        'gamma': self.gamma
    }


class Trainer:
    """Train and validate a :class:`~torch_tracking.model.GNNTrackingModel`.

    Handles the training/validation loop, mixed-precision autocast,
    gradient clipping, learning rate scheduling, early stopping, and
    checkpoint/TensorBoard logging.

    Parameters
    ----------
    model : torch.nn.Module
        The GNN tracking model to train.
    train_loader : torch.utils.data.DataLoader
        DataLoader providing training batches.
    val_loader : torch.utils.data.DataLoader
        DataLoader providing validation batches.
    config : TrainingConfig
        Training configuration specifying the optimizer, scheduler, device,
        checkpoint/log directories, and other hyperparameters.

    Returns
    -------
    Trainer
        An initialized ``Trainer`` object.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: TrainingConfig
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = create_optimizer(self.model, config)
        self.scheduler = create_scheduler(self.optimizer, config)
        self.device = self.config.device
        self.max_epochs = self.config.max_epochs
        self.gradient_clip = self.config.clipnorm
        self.enable_early_stopping = self.config.enable_early_stopping
        self.patience = self.config.patience
        self.log_and_save = self.config.log_and_save
        self.loss=self.config.loss
        self.stopping_metric = self.config.stopping_metric
        self.return_logits = False
        self.gamma = self.config.gamma
        self.metrics_tracker = MetricsTracker()
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.class_weights = self.config.class_weights

        self.log_suffix = curr_time
        self.log_dir = Path(self.config.log_dir +  self.log_suffix)
        self.checkpoint_dir = Path(self.config.checkpoint_dir + self.log_suffix)

        self.loss_fn = TrackingLoss(loss=self.loss, gamma=self.gamma, class_weights=self.class_weights)
        
        # Move to device
        self.model = self.model.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)

        # Altering data precision values
        if self.config.data_precision == 'bfloat16':
            self.autocast = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
            self.scaler = torch.amp.GradScaler()

        elif self.config.data_precision == 'float32':
            self.autocast = torch.amp.autocast(device_type='cuda', dtype=torch.float32)
            self.scaler = torch.amp.GradScaler(enabled=False)

        elif self.config.data_precision == 'float16':
            self.autocast = torch.amp.autocast(device_type='cuda', dtype=torch.float16)
            self.scaler = torch.amp.GradScaler()

        # Setup directories and Tensorboard writer
        if self.log_and_save:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)

            with open(f"{self.checkpoint_dir}/config.json", 'w') as f:
                json.dump(self.config.to_dict(), f, indent=4)
        
        else:
            self.writer = None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.patience,
            mode='min'
        )
        
        # Tracking
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = []
        self.val_history = []

    
    def train_epoch(self) -> Dict[str, float]:
        """Run one epoch of training over ``self.train_loader``.

        For each batch, runs the model's training forward pass, computes
        the loss, backpropagates under mixed precision, clips gradients,
        steps the optimizer, and updates the running metrics. Batches that
        produce a NaN or infinite loss are skipped.

        Returns
        -------
        dict of str to float
            The accumulated training metrics for the epoch (as returned by
            :meth:`~torch_tracking.utils.MetricsTracker.get_metrics`),
            including ``loss``, ``accuracy``, and per-class F1 scores.
        """
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
                self.optimizer.zero_grad()
                continue

            if torch.isnan(loss) or torch.isinf(loss):
                # Log which batch and what the probs looked like
                print(f"  min prob: {predictions.min().item():.2e}")
                print(f"  max prob: {predictions.max().item():.2e}")
                self.optimizer.zero_grad()
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
        """Evaluate the model on ``self.val_loader`` without updating weights.

        For each batch, runs the model's training forward pass in
        evaluation mode, computes the loss, and updates the running
        metrics. Batches that produce a NaN loss are skipped.

        Returns
        -------
        dict of str to float
            The accumulated validation metrics (as returned by
            :meth:`~torch_tracking.utils.MetricsTracker.get_metrics`),
            including ``loss``, ``accuracy``, and per-class F1 scores.
        """
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
        """Save the current model/optimizer state to ``self.checkpoint_dir``.

        If ``is_best`` is True, the checkpoint is written to
        ``best_model.pt``; otherwise it is written to
        ``checkpoint_epoch_<N>.pt`` and only the 3 most recent such
        per-epoch checkpoints are retained on disk.

        Parameters
        ----------
        is_best : bool, optional
            Whether this checkpoint corresponds to the best validation
            score seen so far. Default is False.

        Returns
        -------
        None
        """
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
        """Restore model, optimizer, and training history from a checkpoint.

        Parameters
        ----------
        checkpoint_path : str or pathlib.Path
            Path to a checkpoint file previously written by
            :meth:`save_checkpoint`.

        Returns
        -------
        None
            Updates ``self.model``, ``self.optimizer``, ``self.current_epoch``,
            ``self.best_val_loss``, ``self.train_history``, and
            ``self.val_history`` in place.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """Run the full training loop from ``self.current_epoch`` to ``max_epochs``.

        For each epoch, trains and validates the model, steps the
        scheduler, saves a checkpoint whenever the stopping metric
        improves, logs metrics to TensorBoard (if enabled), and applies
        early stopping (if enabled). Writes the combined train/validation
        history to ``training_history.json`` in ``self.checkpoint_dir``
        when training completes.

        Returns
        -------
        None
        """
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

    # Make config object with default params
    config = TrainingConfig()
    config.device='cuda:3'

    # Initialize model

    model = GNNTrackingModel(
        graph_layer='gat', 
        data_format='channels_last',
        encoder_dim=64,
        n_layers=config.n_layers,
        crop_size=config.crop_size,
        dropout=config.dropout
    )

    config.model = model

    # Create optimizer and rate scheduler
    
    train_loader, val_loader, _ = create_trk_dataloaders(
        train_path=Path.home() / '.deepcell/tracking/train_proc.zarr',
        val_path=Path.home() / '.deepcell/tracking/val_proc.zarr',
        batch_size=config.batch_size,
        distance_threshold=64,
        num_workers=config.num_workers,
        truncate_dataset = config.truncate_dataset,
    )

    config.train_loader = train_loader
    config.val_loader = val_loader
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )   

    trainer.train()
