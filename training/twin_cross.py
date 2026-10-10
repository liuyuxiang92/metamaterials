"""Minimal cross-plot: each Ver2 device's prediction against its TWIN's COMSOL.

Two panels, two curves each — nothing else:

    panel 1   COMSOL 1a  vs  prediction 1b
    panel 2   COMSOL 1b  vs  prediction 1a

If the model copies its training twin, each prediction lies on top of the other device's
COMSOL curve. For the fuller diagnostic (own-truth overlays, pair-difference correlation)
use ``gnn.training.twin_check``.

    python -m gnn.training.twin_cross
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from ..data.devices import GNN_DIR

C_A = "#2a78d6"   # Ver2-1a identity
C_B = "#eb6834"   # Ver2-1b identity
INK, INK2 = "#0b0b0b", "#52514e"

DEV_A = "AI2026_Trial-Ver2-1a"
DEV_B = "A12026_Trial_Ver2-1b"


def twin_cross(run: str = "dual_gnn_res", dev_a: str = DEV_A, dev_b: str = DEV_B,
               out: str | None = None):
    import matplotlib
    if out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = GNN_DIR / "results" / run
    d = torch.load(res / "train_output.pt", map_location="cpu", weights_only=False)
    g, F = np.asarray(d["grid"]), d["folds"]
    m = F[dev_a]["mask"].astype(bool) & F[dev_b]["mask"].astype(bool)
    f = g[m]
    ya, pa = F[dev_a]["y"][m], F[dev_a]["pred"][m]
    yb, pb = F[dev_b]["y"][m], F[dev_b]["pred"][m]
    mae = lambda x, y: float(np.abs(x - y).mean())

    sa, sb = "Ver2-1a", "Ver2-1b"
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, sharey=True)

    panels = [
        (axes[0], ya, C_A, f"COMSOL {sa}", pb, C_B, f"prediction {sb}"),
        (axes[1], yb, C_B, f"COMSOL {sb}", pa, C_A, f"prediction {sa}"),
    ]
    for ax, y, cy, ly, p, cp, lp in panels:
        ax.plot(f, y, color=cy, lw=2.0, label=ly)
        ax.plot(f, p, color=cp, lw=2.0, ls="--", label=lp)
        ax.set_title(f"{ly}   vs   {lp}      MAE = {mae(y, p):.4f}",
                     fontsize=11, color=INK, loc="left")
        ax.set_ylabel("absorption", color=INK2)
        ax.grid(alpha=0.25, lw=0.6, color="0.75")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=9.5, loc="lower right", framealpha=0.95)
    axes[1].set_xlabel("frequency (Hz)", color=INK2)

    print(f"== {run} cross-check, {f.min():.0f}-{f.max():.0f} Hz ==")
    print(f"  COMSOL {sa} vs prediction {sb} : MAE {mae(ya, pb):.4f}")
    print(f"  COMSOL {sb} vs prediction {sa} : MAE {mae(yb, pa):.4f}")
    print(f"  (own-truth MAE for reference   : {sa} {mae(ya, pa):.4f}, {sb} {mae(yb, pb):.4f})")

    fig.tight_layout()
    out = out or str(res / "twin_cross.png")
    fig.savefig(out, dpi=160)
    print(f"  wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="dual_gnn_res")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    twin_cross(a.run, out=a.out)
