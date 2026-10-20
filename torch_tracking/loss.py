import torch
from torch_tracking.utils import weighted_categorical_crossentropy_v2
from torch import nn
from torch.nn import functional as F


def multiclass_focal_loss(probs, targets_one_hot, gamma=2.0):
    """Compute the per-sample, per-class multiclass focal loss.

    Parameters
    ----------
    probs : torch.Tensor
        Tensor of shape ``(N, C)`` containing softmax probabilities from
        the model.
    targets_one_hot : torch.Tensor
        Tensor of shape ``(N, C)`` containing one-hot encoded targets
        (float).
    gamma : float, optional
        Focusing parameter. 0 gives standard cross entropy; 2 is a typical
        default. Default is 2.0.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(N, C)`` containing the focal loss for each
        sample and class.
    """
    eps = 1e-7
    probs = probs.clamp(eps, 1 - eps)

    p_t = (probs * targets_one_hot).sum(dim=-1, keepdim=True)  # (N, 1)
    modulator = (1.0 - p_t) ** gamma                           # (N, 1)

    ce = -(targets_one_hot * torch.log(probs))                 # (N, C)

    return modulator * ce


class TrackingLoss(nn.Module):
    """Loss module for the cell-tracking model's linkage predictions.

    Supports weighted categorical cross entropy, multiclass focal loss, or
    a combination of the two.

    Parameters
    ----------
    loss : str, optional
        Which loss variant to use, one of ``"wcce"`` (weighted categorical
        cross entropy), ``"focal"`` (multiclass focal loss), or
        ``"focal_wcce"`` (focal loss with class weighting). Default is
        ``"wcce"``.
    gamma : float, optional
        Focusing parameter used by the ``"focal"`` and ``"focal_wcce"``
        variants. Default is 2.0.
    class_weights : sequence of float or None, optional
        Fixed per-class weights to use instead of weights computed from
        class frequency in each batch. Default is None.

    Returns
    -------
    TrackingLoss
        An initialized ``TrackingLoss`` module.
    """

    def __init__(self, loss='wcce', gamma=2.0, class_weights = None):
        super().__init__()

        assert loss in ('wcce', 'focal', 'focal_wcce'), \
            f"loss must be 'wcce', 'focal', or 'focal_wcce', got '{loss}'"

        self.loss = loss
        self.gamma = gamma
        self.pad_value = -1
        self.class_weights = class_weights

    def forward(self, predictions, targets, label_smoothing=0.0):
        """Compute the tracking loss between predicted and target linkages.

        Parameters
        ----------
        predictions : torch.Tensor
            Tensor of shape ``(B, T, N, M, C)`` containing softmax
            probabilities from the model.
        targets : torch.Tensor
            Tensor of shape ``(B, T, N, M)`` containing integer class
            labels (``0``, ``1``, ``2``); a value of ``-1`` marks padding
            and is excluded from the loss.
        label_smoothing : float, optional
            Amount of label smoothing to apply, in ``[0, 1)``. ``0``
            disables smoothing. Default is 0.0.

        Returns
        -------
        torch.Tensor
            Scalar tensor containing the mean loss over all valid
            (non-padding) entries.
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