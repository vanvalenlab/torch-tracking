import torch
from torch_tracking.utils import weighted_categorical_crossentropy_v2
from torch import nn
from torch.nn import functional as F


def multiclass_focal_loss(probs, targets_one_hot, gamma=2.0):
    """
    Args:
        probs:          (N, C) softmax probabilities from model
        targets_one_hot:(N, C) one-hot encoded targets (float)
        gamma:          focusing parameter. 0 = standard CE, 2 is typical default.
    """
    eps = 1e-7
    probs = probs.clamp(eps, 1 - eps)

    p_t = (probs * targets_one_hot).sum(dim=-1, keepdim=True)  # (N, 1)
    modulator = (1.0 - p_t) ** gamma                           # (N, 1)

    ce = -(targets_one_hot * torch.log(probs))                 # (N, C)

    return modulator * ce


class TrackingLoss(nn.Module):

    def __init__(self, loss='wcce', gamma=2.0, class_weights = None):
        super().__init__()

        assert loss in ('wcce', 'focal', 'focal_wcce'), \
            f"loss must be 'wcce', 'focal', or 'focal_wcce', got '{loss}'"

        self.loss = loss
        self.gamma = gamma
        self.pad_value = -1
        self.class_weights = class_weights

    def forward(self, predictions, targets, label_smoothing=0.0):
        """
        Args:
            predictions:     (B, T, N, M, C) softmax probabilities from model
            targets:         (B, T, N, M) integer class labels [0, 1, 2], -1 = padding
            label_smoothing: float in [0, 1). 0 = disabled.
        """
        B, T, N, _, C = predictions.shape

        valid_mask = (targets >= 0).view(-1)

        probs_flat = predictions.view(-1, C)[valid_mask]                          # (K, C)

        targets_flat = targets.view(-1).long()
        targets_one_hot = F.one_hot(targets_flat[valid_mask], num_classes=C).float()  # (K, C)

        if label_smoothing > 0.0:
            targets_one_hot = (1 - label_smoothing) * targets_one_hot + label_smoothing / C

        if self.loss == 'wcce':
            
            # weighted categorical cross entropy
            eps = 1e-7
            probs = probs_flat.clamp(eps, 1 - eps)

            total_sum = targets_one_hot.sum()
            class_sum = targets_one_hot.sum(dim=0, keepdim=True)  # (1, C)

            if self.class_weights is None:
                # Inverse frequency: rare classes get higher weight
                class_weights = total_sum / (class_sum.clamp(min=1.0) * C)  # (1, C)

                # Normalize so the most-frequent class has weight=1
                # class_weights = inv_freq / inv_freq.min()
            else:
                class_weights = torch.tensor(self.class_weights).to(total_sum.device)

            loss = -(targets_one_hot * torch.log(probs) * class_weights)    # (K, C)

        elif self.loss == 'focal':
            loss = multiclass_focal_loss(probs_flat, targets_one_hot, gamma=self.gamma)

        elif self.loss == 'focal_wcce':
            eps = 1e-7
            probs = probs_flat.clamp(eps, 1 - eps)

            total_sum = targets_one_hot.sum()
            class_sum = targets_one_hot.sum(dim=0, keepdim=True)
            if self.class_weights is None:
                class_weights = (1.0 / C) * (total_sum / (class_sum + 1.0))  # (1, C)
            else:
                class_weights = torch.tensor(self.class_weights).to(total_sum.device)

            ce = -(targets_one_hot * torch.log(probs) * class_weights)    # (K, C)

            p_t = (probs * targets_one_hot).sum(dim=-1, keepdim=True)     # (K, 1)
            modulator = (1.0 - p_t) ** self.gamma                         # (K, 1)
            loss = modulator * ce                                           # (K, C)

        return loss.mean()