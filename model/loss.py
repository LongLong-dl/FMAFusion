import torch
import torch.nn as nn


class NormalizedMAELoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        abs_error = (pred - target).abs().mean(dim=0)
        norm_factor = target.abs().mean(dim=0) + self.eps
        return (abs_error / norm_factor).sum()
