"""Masked losses — only frequency bins a device actually covers contribute."""
from __future__ import annotations

import torch


def masked_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over valid bins only. Shapes: (B, G)."""
    se = (pred - y) ** 2 * mask
    return se.sum() / mask.sum().clamp_min(1.0)
