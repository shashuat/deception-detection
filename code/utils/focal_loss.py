# utils/focal_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for dealing with class imbalance.
    Focal Loss: https://arxiv.org/abs/1708.02002
    
    Args:
        alpha (float, optional): Weighing factor for the rare class. Default: 0.25
        gamma (float, optional): Focusing parameter. Default: 2.0
        reduction (str, optional): Specifies the reduction to apply to the output.
            'none' | 'mean' | 'sum'. Default: 'mean'
        device (str, optional): Device to use. Default: 'cuda:0' if available else 'cpu'
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', device=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Tensor of shape [B, C] where C is the number of classes
            targets: Tensor of shape [B] with class indices
        """
        # Convert targets to one-hot encoding
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        
        # Compute focal loss
        F_loss = self.alpha * (1-pt)**self.gamma * CE_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for improved generalization.
    
    Args:
        smoothing (float): Label smoothing factor between 0 and 1.
        dim (int): Dimension over which to apply smoothing (class dimension).
    """
    def __init__(self, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # Create a one-hot encoding of the target
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(self.dim) - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))