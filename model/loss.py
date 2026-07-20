import torch
import torch.nn as nn


class NormalizedMAELoss(nn.Module):
    """Multi-task normalized MAE loss (Loss Function, Eq.8).
    Per-target MAE normalized by target mean, unweighted sum across 5 targets.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # Eq.8: l_k = Σ|ŷ - y| / Σy  for k ∈ {cal, mass, fat, carb, protein}
        abs_error = (pred - target).abs().mean(dim=0)
        norm_factor = target.abs().mean(dim=0) + self.eps
        # Eq.9: L = l_cal + l_mass + l_fat + l_carb + l_protein
        return (abs_error / norm_factor).sum()
