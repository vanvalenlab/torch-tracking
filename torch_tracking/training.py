"""Training loop, training configuration and the command-line entry point for it.

Launch a run with `python -m torch_tracking.training --data-dir <dir>` (or the installed
`torch-tracking-train` command). Every setting in `TrainingConfig` is also a flag; run with
`--help` for the list. See the README's Training section for a worked recipe.
"""

import argparse
import dataclasses
import datetime
import json
import re
import shutil
import time
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torch_tracking.loader import create_trk_dataloaders
from torch_tracking.loss import LOSSES, WEIGHTED_LOSSES, TrackingLoss
from torch_tracking.model import GNNTrackingModel
from torch_tracking.utils import (
    DISTANCE_THRESHOLD,
    OPTIMIZERS,
    SCHEDULERS,
    EarlyStopping,
    MetricsTracker,
    check_choice,
    create_optimizer,
    create_scheduler,
    metric_direction,
)

#: Accepted values of `TrainingConfig.data_precision`.
PRECISIONS = ('bfloat16', 'float16', 'float32')

_CHECKPOINT_PATTERN = re.compile(r'checkpoint_epoch_(\d+)\.pt$')


def _setting(default, help, **extra):
    """A configuration field whose `help` text also documents its command-line flag."""
    metadata = {'help': help, **extra}
    if isinstance(default, list):
        return field(default_factory=lambda: list(default), metadata=metadata)
    return field(default=default, metadata=metadata)


@dataclass
class TrainingConfig:
    """Everything that distinguishes one training run from another.

    Every field is recorded in the run's `config.json`, is read by some code path, and is
    settable from the command line as `--<field-name>` with underscores as hyphens.
    """
    data_dir: Optional[str] = _setting(
        None, 'directory holding train_proc.zarr and val_proc.zarr (required)', required=True)
    optimizer: str = _setting('radam', f'one of {", ".join(OPTIMIZERS)}')
    scheduler: str = _setting('reduce_on_plateau', f'one of {", ".join(SCHEDULERS)}')
    device: str = _setting(
        'cuda', "torch device; on a shared machine pick the card with CUDA_VISIBLE_DEVICES")
    checkpoint_dir: str = _setting('./checkpoints/', 'parent directory for per-run checkpoint folders')
    log_dir: str = _setting('./logs/', 'parent directory for per-run TensorBoard logs')
    step_size: int = _setting(5, "epochs between decays under the 'step' scheduler")
    step_gamma: float = _setting(
        0.1, "learning-rate multiplier at each decay under the 'step' scheduler")
    max_epochs: int = _setting(50, 'number of epochs to train')
    num_workers: int = _setting(
        4, 'dataloader worker processes; each holds its own copy of the training set in RAM')
    n_layers: int = _setting(2, 'graph layers in the model')
    clipnorm: float = _setting(1.0, 'gradient-norm clip; 0 disables clipping')
    batch_size: int = _setting(8, 'windows per batch')
    learning_rate: float = _setting(0.001, 'initial learning rate')
    patience: int = _setting(5, "epochs without improvement before 'reduce_on_plateau' decays")
    early_stopping_patience: int = _setting(
        10, 'epochs without improvement before stopping (with --enable-early-stopping)')
    enable_early_stopping: bool = _setting(False, 'stop when the stopping metric stops improving')
    log_and_save: bool = _setting(
        True, 'write checkpoints, history and TensorBoard logs; off writes nothing')
    class_weights: list[float] = _setting(
        [1, 10, 100], 'loss weight per class, in [different, same, mitosis] order')
    loss: str = _setting('wcce', f'one of {", ".join(LOSSES)}')
    stopping_metric: str = _setting(
        'loss', 'validation metric that selects the best epoch and drives early stopping; '
                'loss is minimized, every other metric maximized')
    gamma: float = _setting(1.0, 'focusing parameter of the focal loss modes')
    truncate_dataset: Optional[int] = _setting(None, 'use only the first N movies (for smoke tests)')
    data_precision: str = _setting('bfloat16', f'one of {", ".join(PRECISIONS)}')
    dropout: float = _setting(0, 'dropout rate in the model')
    crop_size: int = _setting(32, 'appearance crop size the model expects')
    decay: float = _setting(0.99, "learning-rate multiplier per epoch under the 'exp' scheduler")
    weight_decay: float = _setting(0.0, 'weight decay passed to the optimizer')
    max_checkpoints: Optional[int] = _setting(
        None, 'keep only the most recent N per-epoch checkpoints; default keeps every epoch')

    def to_dict(self):
        """Every field, so the written record cannot omit a setting that acts on the run."""
        return dataclasses.asdict(self)


def validate_config(config):
    """Reject a bad configuration where it is given, naming the value and what is accepted."""
    check_choice('optimizer', config.optimizer, OPTIMIZERS)
    check_choice('scheduler', config.scheduler, SCHEDULERS)
    check_choice('loss', config.loss, LOSSES)
    check_choice('data_precision', config.data_precision, PRECISIONS)
    metric_direction(config.stopping_metric)
    if config.max_checkpoints is not None and config.max_checkpoints < 1:
        raise ValueError(f"max_checkpoints must be at least 1 or None, got {config.max_checkpoints}")


class Trainer:
    """Trains a tracking model and writes its checkpoints.

    Everything that varies between runs lives in `config`; the constructor takes the model,
    the two dataloaders and that one configuration object.

    Args:
        model: the model to train (a `GNNTrackingModel`)
        train_loader: training DataLoader
        val_loader: validation DataLoader
        config: a `TrainingConfig`. Its choices (optimizer, scheduler, loss, precision,
            stopping metric) are checked here, so a bad value fails at construction.

    A run writes, beneath `<config.checkpoint_dir>/<timestamp>/` (only when
    `config.log_and_save` is on): `config.json`, one `checkpoint_epoch_NNNN.pt` per completed
    epoch, `best_model.pt` (a copy of the best epoch), and `training_history.json`.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: TrainingConfig
    ):
        validate_config(config)

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
        self.early_stopping_patience = self.config.early_stopping_patience
        self.log_and_save = self.config.log_and_save
        self.loss = self.config.loss
        self.stopping_metric = self.config.stopping_metric
        self.max_checkpoints = self.config.max_checkpoints
        self.gamma = self.config.gamma
        self.metrics_tracker = MetricsTracker()
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.class_weights = self.config.class_weights

        self.log_suffix = curr_time
        self.log_dir = Path(self.config.log_dir) / self.log_suffix
        self.checkpoint_dir = Path(self.config.checkpoint_dir) / self.log_suffix

        # The loss applies class weights in 'wcce' and 'focal_wcce' only.
        if self.loss in WEIGHTED_LOSSES:
            loss_weights = self.class_weights
        else:
            loss_weights = None
            if self.class_weights:
                print(f"Note: loss {self.loss!r} does not use class_weights; "
                      f"{self.class_weights} is recorded in the config but has no effect")
        self.loss_fn = TrackingLoss(loss=self.loss, gamma=self.gamma, class_weights=loss_weights)

        # Move to device
        self.model = self.model.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)

        # Mixed precision follows the configured device. Full precision turns autocasting
        # off rather than asking for an autocast dtype the accelerator runtime rejects.
        device_type = torch.device(self.device).type
        if self.config.data_precision == 'float32':
            self.autocast = torch.amp.autocast(device_type=device_type, enabled=False)
            self.scaler = torch.amp.GradScaler(device_type, enabled=False)
        else:
            dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.config.data_precision]
            self.autocast = torch.amp.autocast(device_type=device_type, dtype=dtype)
            self.scaler = torch.amp.GradScaler(device_type)

        # Setup directories and Tensorboard writer
        if self.log_and_save:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)

            with open(self.checkpoint_dir / "config.json", 'w') as f:
                json.dump(self.config.to_dict(), f, indent=4)

        else:
            self.writer = None

        # Which way is better is a property of the metric, not of each call site.
        self.metric_mode = metric_direction(self.stopping_metric)
        self.early_stopping = EarlyStopping(
            patience=self.early_stopping_patience,
            mode=self.metric_mode
        )

        # Tracking
        self.current_epoch = 0
        self.best_value = float('inf') if self.metric_mode == 'min' else float('-inf')
        self.best_epoch = None
        self.train_history = []
        self.val_history = []

    def _is_improvement(self, value):
        """Whether `value` beats the best stopping-metric value seen so far."""
        if self.metric_mode == 'min':
            return value < self.best_value
        return value > self.best_value

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch} [Train]', dynamic_ncols=True)

        for batch_idx, batch in enumerate(pbar):

            # Unpad and move batch to device
            appearances = batch['appearances'].to(self.device)
            morphologies = batch['morphologies'].to(self.device)
            centroids = batch['centroids'].to(self.device)
            adj_matrices = batch['adj_matrices'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            with self.autocast:

                # TrackingLoss takes the log of its input, so it is given probabilities.
                predictions = self.model.training_forward(
                    appearances, morphologies, centroids, adj_matrices,
                    return_logits=False
                )

                # Compute loss
                loss = self.loss_fn(predictions, labels)

            # Skip a batch whose loss is not finite *before* touching the gradient scaler.
            # Calling `unscale_` and then abandoning the batch would leave the scaler
            # mid-update, and the next batch would raise.
            if not torch.isfinite(loss):
                print(f"Non-finite loss ({loss.item()}) at epoch {self.current_epoch}, "
                      f"batch {batch_idx}; skipping batch "
                      f"(prediction range {predictions.min().item():.2e} to "
                      f"{predictions.max().item():.2e})")
                self.optimizer.zero_grad()
                continue

            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer) # Unscales the gradients in-place

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
        """Validate on validation set.

        Validation does not run under autocast, so its forward pass is full precision while
        training's is not. That is divergence 71 in `openspec/DIVERGENCES.md`; it is left
        as it is here because aligning it changes recorded validation losses.
        """
        self.model.eval()

        pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch} [Val]', dynamic_ncols=True)

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
                return_logits=False
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
        """Write the checkpoint for the epoch just completed; return its path.

        Every call writes `checkpoint_epoch_NNNN.pt`. When `is_best`, the same file is also
        copied to `best_model.pt`. Writes nothing when the run has `log_and_save` off.
        """
        if not self.log_and_save:
            return None

        checkpoint = {
            # Index of the last epoch that finished; a resumed run continues after it.
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'early_stopping_state': {
                'best_value': self.early_stopping.best_value,
                'counter': self.early_stopping.counter,
                'should_stop': self.early_stopping.should_stop,
            },
            'stopping_metric': self.stopping_metric,
            'best_metric': self.best_value,
            'best_epoch': self.best_epoch,
            'train_history': self.train_history,
            'val_history': self.val_history
        }

        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch:04d}.pt'
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            shutil.copyfile(checkpoint_path, self.checkpoint_dir / 'best_model.pt')
            print()
            print(f"    Saved best model ({self.stopping_metric}: {self.best_value:.4f})")

        self._prune_checkpoints()
        return checkpoint_path

    def _prune_checkpoints(self):
        """Apply the retention cap to per-epoch checkpoints; `best_model.pt` is never pruned."""
        if self.max_checkpoints is None:
            return
        numbered = []
        for path in self.checkpoint_dir.glob('checkpoint_epoch_*.pt'):
            match = _CHECKPOINT_PATTERN.search(path.name)
            if match:
                numbered.append((int(match.group(1)), path))
        numbered.sort()
        for _, old_checkpoint in numbered[:-self.max_checkpoints]:
            old_checkpoint.unlink()

    def load_checkpoint(self, checkpoint_path):
        """Restore a run from a checkpoint; `train()` then continues after its epoch."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        scheduler_state = checkpoint.get('scheduler_state_dict')
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)
        elif self.scheduler is not None:
            print("Note: this checkpoint holds no scheduler state; the learning-rate "
                  "schedule restarts from the optimizer's current rate")
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        stopping = checkpoint.get('early_stopping_state')
        if stopping is not None:
            self.early_stopping.best_value = stopping['best_value']
            self.early_stopping.counter = stopping['counter']
            self.early_stopping.should_stop = stopping['should_stop']

        # Older checkpoints name the best value `best_val_loss`.
        self.best_value = checkpoint.get('best_metric', checkpoint.get('best_val_loss'))
        self.best_epoch = checkpoint.get('best_epoch')
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']

        # The saved epoch finished; continue with the one after it.
        self.current_epoch = checkpoint['epoch'] + 1

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}; resuming at epoch {self.current_epoch}")

    def train(self):
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

            metric_value = val_metrics[self.stopping_metric]
            is_best = self._is_improvement(metric_value)
            if is_best:
                self.best_value = metric_value
                self.best_epoch = epoch

            # Decided before saving, so the checkpoint holds the state after this epoch.
            should_stop = self.enable_early_stopping and self.early_stopping(metric_value)

            # Every epoch leaves a checkpoint; the best one is also copied to best_model.pt.
            self.save_checkpoint(is_best=is_best)

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

            if should_stop:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
                break

            print()

        total_time = time.time() - start_time
        print("="*70)
        print(f"Training completed in {total_time/3600:.2f} hours")
        print(f"Best validation {self.stopping_metric}: {self.best_value:.4f}")
        print("="*70)

        if self.writer is not None:
            self.writer.close()

        # Save final training history
        if self.log_and_save:
            history_path = self.checkpoint_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump({
                    'train': self.train_history,
                    'val': self.val_history,
                    'stopping_metric': self.stopping_metric,
                    'best_epoch': self.best_epoch,
                    'best_value': self.best_value,
                }, f, indent=2)


def build_model(config):
    """The model a run trains. The architecture is fixed to what `CellTracker` loads."""
    return GNNTrackingModel(
        graph_layer='gat',
        data_format='channels_last',
        encoder_dim=64,
        n_layers=config.n_layers,
        crop_size=config.crop_size,
        dropout=config.dropout
    )


def run_training(config):
    """Build the model and dataloaders a configuration describes, train, and return the Trainer."""
    if config.data_dir is None:
        raise ValueError("config.data_dir is required: the directory holding "
                         "train_proc.zarr and val_proc.zarr")
    data_dir = Path(config.data_dir).expanduser()

    model = build_model(config)

    train_loader, val_loader, _ = create_trk_dataloaders(
        train_path=data_dir / 'train_proc.zarr',
        val_path=data_dir / 'val_proc.zarr',
        batch_size=config.batch_size,
        distance_threshold=DISTANCE_THRESHOLD,
        num_workers=config.num_workers,
        truncate_dataset=config.truncate_dataset,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    trainer.train()
    return trainer


def _argument_options(annotation):
    """argparse keyword arguments for a configuration field of the given type."""
    origin = typing.get_origin(annotation)
    if annotation is bool:
        return {'action': argparse.BooleanOptionalAction}
    if origin is list:
        return {'type': typing.get_args(annotation)[0], 'nargs': '+'}
    if origin is typing.Union:
        inner = [a for a in typing.get_args(annotation) if a is not type(None)]
        return {'type': inner[0]}
    return {'type': annotation}


def build_parser():
    """A parser with one flag per `TrainingConfig` field, generated from the fields.

    Generating the flags from the dataclass is what keeps the two in step: a new field gets
    a flag with no further edit. Unset flags are left out of the namespace so the dataclass
    stays the only place defaults are written.
    """
    parser = argparse.ArgumentParser(
        prog='torch-tracking-train',
        description=__doc__.strip().split('\n')[0],
    )
    hints = typing.get_type_hints(TrainingConfig)
    for f in dataclasses.fields(TrainingConfig):
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
        parser.add_argument(
            '--' + f.name.replace('_', '-'),
            dest=f.name,
            default=argparse.SUPPRESS,
            required=f.metadata.get('required', False),
            help=f"{f.metadata.get('help', '')} (default: {default})",
            **_argument_options(hints[f.name]),
        )
    return parser


def config_from_args(argv=None):
    """Parse command-line arguments into a `TrainingConfig`."""
    return TrainingConfig(**vars(build_parser().parse_args(argv)))


def main(argv=None):
    """Command-line entry point; returns the process exit status."""
    run_training(config_from_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
