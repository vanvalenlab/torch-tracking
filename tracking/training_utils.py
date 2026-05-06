import numpy as np
import torch.optim as optim

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
    weight_decay = config.get('weight_decay', 0)
    
    if optimizer_name == 'radam':
        optimizer = optim.RAdam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            decoupled_weight_decay=False
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
            factor=0.2,
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
        scheduler=[        
            optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=decay
            ),
            optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=decay
            )
        ]
        
    return scheduler