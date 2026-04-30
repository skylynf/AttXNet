"""
Loss functions: Focal Loss and standard Cross-Entropy wrapper.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., ICCV 2017).
    Addresses class imbalance and hard-sample mining by down-weighting
    well-classified examples.

    Args:
        alpha: balancing factor per class, float or list.
        gamma: focusing parameter (default 2.0).
        num_classes: number of classes.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, num_classes: int = 2):
        super().__init__()
        self.gamma = gamma
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([1 - alpha, alpha])
        else:
            self.alpha = torch.tensor(alpha)
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, self.num_classes).float()

        alpha = self.alpha.to(logits.device)
        alpha_t = (targets_onehot * alpha.unsqueeze(0)).sum(dim=1)

        pt = (probs * targets_onehot).sum(dim=1)
        focal_weight = alpha_t * (1 - pt) ** self.gamma

        ce = F.cross_entropy(logits, targets, reduction='none')
        loss = focal_weight * ce
        return loss.mean()


def build_loss(
    loss_type: str = "focal",
    alpha: float = 0.75,
    gamma: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
):
    """Factory function for loss.

    ``wce`` = weighted Cross-Entropy with per-class weights (inverse frequency on train set).
    Pass ``class_weights`` on the correct device when constructing the module if needed;
    train.py moves weights to device before training.
    """
    if loss_type == "focal":
        return FocalLoss(alpha=alpha, gamma=gamma)
    elif loss_type == "ce":
        return nn.CrossEntropyLoss()
    elif loss_type == "wce":
        if class_weights is None:
            raise ValueError("loss=wce requires class_weights from dataset meta")
        return nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
