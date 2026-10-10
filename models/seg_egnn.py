"""Rep-1 v2 model (``seg_egnn``): edge-feature-aware GNN over the single-node-type
``HeteroData`` with three relations (path / open / spatial) -> absorption spectrum.

Unlike ``rgnn`` (RGCNConv, which *ignores* edge features), every relation here is an
edge-conditioned ``GINEConv`` that reads its own edge-feature vector. Each message-passing
layer runs one GINEConv per relation over the SHARED ``seg`` node embedding and sums the
three results (a relation that has no edges for a given graph contributes only its self term).
K such layers -> global mean+max pool over ``seg`` nodes -> optional resonance-comb encoder
-> MLP head -> G outputs -> sigmoid. Heavily regularized for the tiny dataset.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool

from .base import SpectrumModel

SEG = "seg"
PATH = (SEG, "path", SEG)
OPEN = (SEG, "open", SEG)
SPATIAL = (SEG, "spatial", SEG)


def _mlp(din: int, dout: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(din, dout), nn.ReLU(), nn.Linear(dout, dout))


class _RelLayer(nn.Module):
    """One message-passing layer: a GINEConv per relation, summed over relations."""

    def __init__(self, hidden: int, edge_dims: dict[str, int]):
        super().__init__()
        self.convs = nn.ModuleDict({
            "path": GINEConv(_mlp(hidden, hidden), edge_dim=edge_dims["path_edge"]),
            "open": GINEConv(_mlp(hidden, hidden), edge_dim=edge_dims["open_edge"]),
            "spatial": GINEConv(_mlp(hidden, hidden), edge_dim=edge_dims["spatial_edge"]),
        })

    def forward(self, h, edges):
        out = self.convs["path"](h, edges[PATH][0], edges[PATH][1])
        out = out + self.convs["open"](h, edges[OPEN][0], edges[OPEN][1])
        out = out + self.convs["spatial"](h, edges[SPATIAL][0], edges[SPATIAL][1])
        return out


class SegEGNN(SpectrumModel):
    def __init__(
        self,
        G: int,
        dims: dict,
        hidden: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__(G)
        self.dropout = dropout
        self.encoder = nn.Sequential(
            nn.Linear(dims["node_in"], hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.layers = nn.ModuleList(_RelLayer(hidden, dims) for _ in range(num_layers))

        self.comb_dim = dims.get("comb_dim", 0)
        fuse_dim = 2 * hidden
        if self.comb_dim > 0:
            self.comb_enc = nn.Sequential(nn.Linear(self.comb_dim, hidden), nn.ReLU(),
                                          nn.Linear(hidden, hidden))
            fuse_dim += hidden
        self.head = nn.Sequential(
            nn.Linear(fuse_dim, 2 * hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2 * hidden, G),
        )

    def forward(self, data) -> torch.Tensor:
        store = data[SEG]
        batch = getattr(store, "batch", None)
        if batch is None:
            batch = store.x.new_zeros(store.x.size(0), dtype=torch.long)
        edges = {rel: (data[rel].edge_index, data[rel].edge_attr)
                 for rel in (PATH, OPEN, SPATIAL)}

        h = self.encoder(store.x)
        for layer in self.layers:
            h = layer(h, edges)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        pooled = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)
        if self.comb_dim > 0:
            pooled = torch.cat([pooled, self.comb_enc(data.comb)], dim=1)
        return torch.sigmoid(self.head(pooled))     # (batch, G)
