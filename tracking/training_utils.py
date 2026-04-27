import torch
import numpy as np
import torch.optim as optim

class CalibrationMetrics:
    """
    Tracks per-class confidence calibration.
    
    Expected Calibration Error (ECE): weighted average gap between
    model confidence and actual accuracy across confidence bins.
    
    A perfectly calibrated model has ECE = 0.
    Overconfident models have high confidence but low accuracy -> ECE > 0.
    Underconfident models have low confidence but high accuracy -> ECE > 0.
    """

    def __init__(self, n_classes=3, n_bins=10):
        self.n_classes = n_classes
        self.n_bins = n_bins
        self.bin_edges = torch.linspace(0, 1, n_bins + 1)
        self.reset()

    def reset(self):
        # Per class: accumulate confidence and correctness per bin
        # Shape: (n_classes, n_bins)
        self.bin_confidence_sum = torch.zeros(self.n_classes, self.n_bins)
        self.bin_accuracy_sum   = torch.zeros(self.n_classes, self.n_bins)
        self.bin_counts         = torch.zeros(self.n_classes, self.n_bins)

    def update(self, probs, targets):
        """
        Args:
            probs:   (B, T, N, M, C) softmax probabilities
            targets: (B, T, N, M)    integer class labels
        """

        valid_mask = targets.view(-1) >= 0

        probs_flat   = probs.view(-1, self.n_classes)[valid_mask]  # (K, C)
        targets_flat = targets.view(-1)[valid_mask]                # (K,)

        for c in range(self.n_classes):
            # Confidence = the probability the model assigned to class c
            confidence = probs_flat[:, c]              # (K,)
            # Correct = whether the true label is class c
            correct = (targets_flat == c).float()      # (K,)

            for b in range(self.n_bins):
                lo = self.bin_edges[b]
                hi = self.bin_edges[b + 1]
                in_bin = (confidence >= lo) & (confidence < hi)

                count = in_bin.sum().item()
                if count == 0:
                    continue

                self.bin_counts[c, b]         += count
                self.bin_confidence_sum[c, b] += confidence[in_bin].sum().item()
                self.bin_accuracy_sum[c, b]   += correct[in_bin].sum().item()

    def compute(self):
        metrics = {}
        class_names = ['no_match', 'match', 'division']

        for c in range(self.n_classes):
            counts = self.bin_counts[c]
            total  = counts.sum().item()

            if total == 0:
                continue

            avg_confidence = self.bin_confidence_sum[c] / counts.clamp(min=1)
            avg_accuracy   = self.bin_accuracy_sum[c]   / counts.clamp(min=1)

            ece      = ((counts / total) * (avg_confidence - avg_accuracy).abs()).sum().item()
            mean_gap = ((counts / total) * (avg_confidence - avg_accuracy)).sum().item()

            # Mean confidence: where is the model actually operating on the 0-1 scale?
            mean_confidence = (
                (counts / total) * avg_confidence
            ).sum().item()

            # Mean accuracy: what is the model's actual correctness rate?
            mean_accuracy = (
                (counts / total) * avg_accuracy
            ).sum().item()

            # Relative gap: gap as a fraction of mean accuracy
            # This normalises for the fact that rare classes have low base rates
            relative_gap = mean_gap / (mean_accuracy + 1e-10)

            metrics[f'ece_class_{c}_{class_names[c]}']              = ece
            metrics[f'calibration_gap_class_{c}_{class_names[c]}']  = mean_gap
            metrics[f'mean_confidence_class_{c}_{class_names[c]}']  = mean_confidence
            metrics[f'mean_accuracy_class_{c}_{class_names[c]}']    = mean_accuracy
            metrics[f'relative_gap_class_{c}_{class_names[c]}']     = relative_gap

        return metrics

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

        self.cm = np.zeros((3,3))

    
    def update(self, loss, predictions, targets):
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
        pred_classes = predictions.argmax(dim=-1).view(-1)  # (B*T*N*M)
        target_classes = targets.view(-1)
        valid_mask = target_classes.view(-1) >= 0

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
    

def create_optimizer(model, config):
    """Create optimizer based on config."""

    optimizer_name = config.get('optimizer', 'radam').lower()
    lr = config.get('learning_rate', 1e-3)
    weight_decay = config.get('weight_decay', 1e-5)
    
    if optimizer_name == 'radam':
        optimizer = optim.RAdam(
            model.parameters(),
            lr=lr,
            weight_decay=0,
            decoupled_weight_decay=True
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