"""Model interface: a graph batch -> absorption spectra of shape (batch, G).

``G`` (spectrum length) is supplied at construction from the auto-detected target grid,
never hardcoded. Keeping this thin lets idea #2 (another GNN) and idea #3 (a CNN) drop in
as new modules behind the same ``build(config, in_dim, G)`` factory and training loop.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SpectrumModel(nn.Module):
    """Base class; subclasses implement ``forward(data) -> (batch, G)`` in (0, 1)."""

    def __init__(self, G: int):
        super().__init__()
        self.G = G


def build_model(name: str, *, G: int, in_dim: int | None = None,
                dims: dict | None = None, **kw) -> SpectrumModel:
    if name == "dualgnn":
        from .dualgnn import DualGNN
        return DualGNN(G=G, dims=dims, **kw)
    if name == "segegnn":
        from .seg_egnn import SegEGNN
        return SegEGNN(G=G, dims=dims, **kw)
    if name == "layercnn":
        from .cnn import LayerCNN
        return LayerCNN(G=G, dims=dims, **kw)
    raise ValueError(f"Unknown model {name!r}")
