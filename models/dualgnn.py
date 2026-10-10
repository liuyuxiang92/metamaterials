"""Method #2 model: two-stream GNN over the {vertex, surface} HeteroData -> spectrum.

Stream 1 (vertex / tube shape): encode positions -> K edge-aware GINEConv layers on the 9-d
    segment edges (intra-tube only) -> global mean+max pool -> z1.
Stream 2 (surface / coupling): a DeepSets sub-encoder turns each surface's variable-count,
    masked boundary vertices into a fixed embedding, concatenated with the 12-d surface
    features -> encode -> K GINEConv layers on the 6-d coupling edges (intra+inter) ->
    global mean+max pool -> z2.
Fusion: concat [z1, z2] -> MLP head -> G outputs -> sigmoid. One model, trained jointly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool

from .base import SpectrumModel


def _mlp(din: int, dout: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(din, dout), nn.ReLU(), nn.Linear(dout, dout))


class _GineStack(nn.Module):
    """K GINEConv layers at a fixed width, each reading `edge_dim`-wide edge features."""

    def __init__(self, hidden: int, edge_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList(
            GINEConv(_mlp(hidden, hidden), edge_dim=edge_dim) for _ in range(num_layers)
        )

    def forward(self, h, edge_index, edge_attr):
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


def _pool(h, batch):
    return torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)


class DualGNN(SpectrumModel):
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

        # ---- stream 1: vertex / tube shape ----
        self.vert_enc = nn.Sequential(nn.Linear(dims["vertex_in"], hidden), nn.ReLU(),
                                      nn.Linear(hidden, hidden))
        self.vert_gnn = _GineStack(hidden, dims["seg_edge"], num_layers, dropout)

        # ---- stream 2: surface / coupling ----
        # DeepSets over each surface's masked boundary vertices -> 2*hidden shape embedding
        self.vpoint = _mlp(dims["surf_vert"], hidden)
        self.surf_enc = nn.Sequential(
            nn.Linear(2 * hidden + dims["surf_feat"], hidden), nn.ReLU(),
            nn.Linear(hidden, hidden))
        self.surf_gnn = _GineStack(hidden, dims["couple_edge"], num_layers, dropout)

        # ---- optional physics prior: resonance comb -> hidden ----
        self.comb_dim = dims.get("comb_dim", 0)
        fuse_dim = 4 * hidden
        if self.comb_dim > 0:
            self.comb_enc = nn.Sequential(nn.Linear(self.comb_dim, hidden), nn.ReLU(),
                                          nn.Linear(hidden, hidden))
            fuse_dim += hidden

        # ---- fusion + head ----
        self.head = nn.Sequential(
            nn.Linear(fuse_dim, 2 * hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2 * hidden, G),
        )

    def _deepsets(self, vpad, vmask):
        """(N, Vmax, 3), (N, Vmax) -> (N, 2*hidden) masked mean+max over real vertices."""
        h = self.vpoint(vpad)                       # (N, Vmax, hidden)
        m = vmask.unsqueeze(-1)                      # (N, Vmax, 1)
        summ = (h * m).sum(dim=1)
        cnt = m.sum(dim=1).clamp_min(1.0)
        mean = summ / cnt
        maxp = h.masked_fill(m == 0, float("-inf")).max(dim=1).values
        maxp = torch.nan_to_num(maxp, neginf=0.0)   # surfaces always have >=3 verts, but be safe
        return torch.cat([mean, maxp], dim=1)

    def forward(self, data) -> torch.Tensor:
        vstore = data["vertex"]
        sstore = data["surface"]
        vedge = data["vertex", "seg", "vertex"]
        sedge = data["surface", "couple", "surface"]

        bv = getattr(vstore, "batch", None)
        if bv is None:
            bv = vstore.x.new_zeros(vstore.x.size(0), dtype=torch.long)
        bs = getattr(sstore, "batch", None)
        if bs is None:
            bs = sstore.feat.new_zeros(sstore.feat.size(0), dtype=torch.long)

        # stream 1
        hv = self.vert_enc(vstore.x)
        hv = self.vert_gnn(hv, vedge.edge_index, vedge.edge_attr)
        z1 = _pool(hv, bv)

        # stream 2
        shape_emb = self._deepsets(sstore.vpad, sstore.vmask)
        hs = self.surf_enc(torch.cat([shape_emb, sstore.feat], dim=1))
        hs = self.surf_gnn(hs, sedge.edge_index, sedge.edge_attr)
        z2 = _pool(hs, bs)

        parts = [z1, z2]
        if self.comb_dim > 0:
            comb = getattr(data, "comb")            # (B, G)
            parts.append(self.comb_enc(comb))
        z = torch.cat(parts, dim=1)
        return torch.sigmoid(self.head(z))          # (B, G)
