"""Method #3 model: an ordered stack of per-layer CNNs -> absorption spectrum.

Pipeline (see representations/voxel.py for the input):
  per-layer : each layer image (1,H,W) --[ONE shared 2D CNN]--> embedding (E)
              weights shared across layers (14 separate CNNs would just multiply params at n=6)
              concat per-layer scalars [thickness, z_center]        -> (B, D, E+scalar)
  over z    : a LOCAL 1D conv stack ALONG the layer axis                -> (B, D, F)
              kernel `z_kernel`, `z_depth` blocks; padded layers masked out at every step
  pool      : masked mean + max over the REAL layers                     -> (B, 2F)
              with rich_pool (§3.3.2): + h_front + h_back + attention     -> (B, 5F)
  fuse+head : concat optional resonance-comb embedding -> MLP -> G -> sigmoid

Two properties the z-stage MUST have, and why (both established for the Ver2-1a/1b twins,
which are the SAME set of layer images in a DIFFERENT order — see ver2-twin-leakage):
  * ORDER-AWARE. A permutation-invariant pool over layers would map the two twins to identical
    outputs, reproducing the dual_gnn copying failure by construction. A 1D conv along z is
    order-sensitive.
  * LOCALITY-BIASED. Adjacent layers share one 1 mm partition and couple through it; distant
    layers have solid between them. Stacked local kernels make influence decay with z-distance
    (layers k apart mix only after k blocks) — the right physical prior, and far safer than
    global attention over layers at n=6. The interaction RANGE is a knob (`z_kernel`,`z_depth`):
    receptive field = 1 + z_depth*(z_kernel-1); `z_depth=1` is strictly nearest-neighbour.

No positional encoding by default: the twin signal is purely RELATIVE adjacency, and a conv
along z is already translation-equivariant, the correct symmetry. `z_posenc=True` adds a
learned per-index bias for a possible top/bottom boundary effect.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import SpectrumModel


def _conv2d_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
        nn.MaxPool2d(2),
    )


class _AttentionPool(nn.Module):
    """Weighted layer pool: score each layer, softmax over the REAL layers, weighted sum.

    A learned complement to mean/max — it can up-weight the acoustically important layers
    instead of averaging them away. Padded layers get -inf scores so they never contribute.
    """

    def __init__(self, feat: int):
        super().__init__()
        self.score = nn.Linear(feat, 1)

    def forward(self, z, lmask):                              # z:(B,D,F)  lmask:(B,D)
        s = self.score(z).squeeze(-1)                        # (B, D)
        s = s.masked_fill(lmask == 0, float("-inf"))
        w = torch.softmax(s, dim=1).unsqueeze(-1)            # (B, D, 1)
        return (w * z).sum(dim=1)                             # (B, F)


class _LayerEncoder(nn.Module):
    """Shared 2D CNN: (N, 1, H, W) -> (N, E). Adaptive pool makes it H/W-agnostic.

    A stride-2 stem (cheap: 1 input channel) quarters the feature map before the wide conv
    blocks, so the dominant activation stays well within a 16 GB shared-memory budget when all
    B*D layers of a full batch pass through at once. The finest wall is 4 px at 0.25 mm, so a
    single stride-2 downsample keeps it resolvable.
    """

    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        c = max(8, hidden // 4)
        self.stem = nn.Sequential(nn.Conv2d(1, c, 3, stride=2, padding=1),
                                  nn.BatchNorm2d(c), nn.ReLU())
        self.net = nn.Sequential(
            _conv2d_block(c, c),
            _conv2d_block(c, 2 * c),
            _conv2d_block(2 * c, hidden),
            nn.AdaptiveAvgPool2d(1),                      # -> (N, hidden, 1, 1)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                 # x: (N, 1, H, W)
        h = self.net(self.stem(x)).flatten(1)             # (N, hidden)
        return self.drop(h)


class LayerCNN(SpectrumModel):
    def __init__(
        self,
        G: int,
        dims: dict,
        hidden: int = 64,
        z_kernel: int = 3,
        z_depth: int = 3,
        z_posenc: bool = False,
        dmax: int = 14,
        dropout: float = 0.3,
        rich_pool: bool = False,
    ):
        super().__init__(G)
        self.dropout = dropout
        self.z_posenc = z_posenc
        self.rich_pool = rich_pool

        self.layer_enc = _LayerEncoder(hidden, dropout)
        feat = hidden + dims["scalar_dim"]                # per-layer vector before the z-stage
        if z_posenc:
            self.pos = nn.Parameter(torch.zeros(dmax, feat))

        pad = z_kernel // 2
        self.zconvs = nn.ModuleList()
        cin = feat
        for _ in range(z_depth):
            self.zconvs.append(nn.Conv1d(cin, hidden, z_kernel, padding=pad))
            cin = hidden
        self.F = hidden

        # pooling over the layer axis. Default: masked mean + max (2F). Rich (§3.3.2):
        # h_device = [h_front, h_back, mean, max, attention] (5F).
        if rich_pool:
            self.attn_pool = _AttentionPool(hidden)
            fuse = 5 * hidden
        else:
            fuse = 2 * hidden

        self.comb_dim = dims.get("comb_dim", 0)
        if self.comb_dim > 0:
            self.comb_enc = nn.Sequential(nn.Linear(self.comb_dim, hidden), nn.ReLU(),
                                          nn.Linear(hidden, hidden))
            fuse += hidden
        self.head = nn.Sequential(
            nn.Linear(fuse, 2 * hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2 * hidden, G),
        )

    def forward(self, data) -> torch.Tensor:
        vox, lay, lmask = data.vox, data.lay, data.lmask  # (B,D,H,W) (B,D,S) (B,D)
        B, D, H, W = vox.shape

        # ---- per-layer 2D encoding (all B*D layers through the shared CNN at once) ----
        h = self.layer_enc(vox.reshape(B * D, 1, H, W)).reshape(B, D, -1)   # (B, D, hidden)
        h = torch.cat([h, lay], dim=-1)                                     # (B, D, hidden+S)
        if self.z_posenc:
            h = h + self.pos[:D].unsqueeze(0)
        h = h * lmask.unsqueeze(-1)                                         # zero padded layers

        # ---- local conv along the layer axis (order-aware, locality-biased) ----
        z = h.transpose(1, 2)                                              # (B, C, D)
        for conv in self.zconvs:
            z = conv(z)
            z = F.relu(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
            z = z * lmask.unsqueeze(1)                                     # re-mask padded cols
        z = z.transpose(1, 2)                                             # (B, D, F)

        # ---- masked mean + max pool over real layers ----
        m = lmask.unsqueeze(-1)                                            # (B, D, 1)
        cnt = m.sum(dim=1).clamp_min(1.0)
        mean = (z * m).sum(dim=1) / cnt
        mx = z.masked_fill(m == 0, float("-inf")).max(dim=1).values
        pooled = torch.cat([mean, mx], dim=1)                              # (B, 2F)

        if self.rich_pool:
            # append encoded first/last physical layer + attention pool -> (B, 5F)
            h_front = z[:, 0]                                              # index 0 = bottom
            last = (lmask.sum(dim=1) - 1).clamp_min(0).long()             # last real layer idx
            h_back = z[torch.arange(B, device=z.device), last]
            h_attn = self.attn_pool(z, lmask)
            pooled = torch.cat([h_front, h_back, pooled, h_attn], dim=1)

        if self.comb_dim > 0:
            pooled = torch.cat([pooled, self.comb_enc(data.comb)], dim=1)
        return torch.sigmoid(self.head(pooled))                            # (B, G)
