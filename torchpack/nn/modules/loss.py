import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SmoothCrossEntropyLoss']


class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, gamma=0.1):
        super(SmoothCrossEntropyLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        targets = torch.zeros_like(inputs).scatter(1, targets.view(-1, 1), 1)
        targets = targets * (1 - self.gamma) + (1 - targets) * self.gamma / (num_classes - 1)
        return torch.mean(torch.sum(-targets * F.log_softmax(inputs), 1))
