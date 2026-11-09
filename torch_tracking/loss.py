import torch
from torch_tracking.utils import check_choice
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


LOSSES = ('wcce', 'focal', 'focal_wcce')

#: Loss modes that apply `class_weights`. `focal` does not, and refuses them.
WEIGHTED_LOSSES = ('wcce', 'focal_wcce')


class TrackingLoss(nn.Module):
    """Classification loss over the track-by-cell score grid.

    The loss takes the logarithm of its input, so it requires **probabilities** (a softmax
    over the three classes), never logits. `GNNTrackingModel.training_forward` returns
    probabilities by default for exactly this reason.

    Args:
        loss: one of `LOSSES`.
        gamma: focusing parameter for the focal modes. Unrelated to any learning-rate
            schedule; the step schedule has its own `step_gamma`.
        class_weights: per-class weights, in the model's `[different, same, mitosis]`
            class order. Applied by `wcce` and `focal_wcce`; `focal` has no use for them
            and raises rather than accepting and discarding them.

    The returned loss is on the per-entry scale: the weighted cross-entropy summed over the
    classes and averaged over the graded entries. It is not also averaged over the class
    axis. Losses recorded before this was corrected (divergence 53) are one third of this
    figure.
    """

    def __init__(self, loss='wcce', gamma=2.0, class_weights = None):
        super().__init__()

        check_choice('loss', loss, LOSSES)
        if loss not in WEIGHTED_LOSSES and class_weights is not None:
            raise ValueError(
                f"loss {loss!r} does not apply class_weights; pass class_weights=None, "
                f"or use one of {', '.join(WEIGHTED_LOSSES)}"
            )

        self.loss = loss
        self.gamma = gamma
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

        # Per-entry scale: sum over the class axis, mean over graded entries.
        return loss.sum(dim=-1).mean()