
import torch
import torch.nn as nn

class TrackingLoss(nn.Module):
    
    def __init__(self, alpha=None, gamma=2.0, use_focal=True):
        super().__init__()
        
        if alpha is None:
            # Default: aggressive weighting for minority classes
            # Adjust these based on your actual class distribution
            alpha = torch.tensor([1.0, 5.0, 10.0])  # [no_link, same_cell, mitosis]
        
        self.alpha = alpha
        self.gamma = gamma
        self.use_focal = use_focal
        
        # Standard CrossEntropyLoss with class weights
        self.criterion = nn.CrossEntropyLoss(weight=alpha, reduction='none')

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
        ce_loss = self._weighted_categorical_crossentropy(predictions_flat, targets_flat, targets_flat.shape[-1])
        
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
        
        return loss.sum()
    
if __name__ == "__main__":

    y_true = torch.rand((4,8, 405, 405, 3))
    y_pred = torch.randint(low=-5, high=5, size=(4,8,405,405,3))

    loss = TrackingLoss(use_focal=False)

    loss_out = loss(y_true, y_pred)
    print(loss_out)