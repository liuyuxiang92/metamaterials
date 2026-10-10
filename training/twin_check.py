"""Did the model learn the geometry, or just copy its near-duplicate training twin?

Ver2-1a and Ver2-1b differ only in layer ORDERING, so their COMSOL spectra are nearly
equal. Under leave-one-out, whichever one is held out has its twin sitting in the training
set — so a model that ignores geometry and regurgitates the twin's spectrum scores a
deceptively good MAE. This script separates the two explanations.

The decisive quantity is not either device's MAE but the DIFFERENCE between the pair:

    delta_true = comsol_B - comsol_A        the geometric signal the model must resolve
    delta_pred = pred_B   - pred_A          what the model actually says the difference is

If the model learned the geometry, ``corr(delta_true, delta_pred)`` is strongly positive.
Near zero means the prediction difference is noise; negative means it resolves the pair
backwards. Compare each MAE against the ``copy the twin`` baseline (predict A's COMSOL for
B) — a model that cannot beat it has learned nothing about the pair.

    python -m gnn.training.twin_check                       # dual_gnn_res, the Ver2 pair
    python -m gnn.training.twin_check --run dual_gnn -o /tmp/twin.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..data.devices import GNN_DIR

# validated categorical slots (dataviz reference palette, light surface)
C_A = "#2a78d6"     # device A identity
C_B = "#eb6834"     # device B identity
C_TRUE = "#1baf7a"  # delta_true
C_PRED = "#4a3aa7"  # delta_pred
INK = "#0b0b0b"
INK2 = "#52514e"

DEV_A = "AI2026_Trial-Ver2-1a"
DEV_B = "A12026_Trial_Ver2-1b"


def _label(ax, x, y, text, color, va="bottom", ha="left"):
    """Direct label: text in ink, hue carried by the box edge (never colored text)."""
    ax.annotate(text, (x, y), color=INK, fontsize=8.5, va=va, ha=ha, zorder=8,
                bbox=dict(fc="w", ec=color, lw=1.6, alpha=0.92, boxstyle="round,pad=0.22"))


def twin_check(run: str = "dual_gnn_res", dev_a: str = DEV_A, dev_b: str = DEV_B,
               out: str | None = None):
    import matplotlib
    if out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = GNN_DIR / "results" / run
    d = torch.load(res / "train_output.pt", map_location="cpu", weights_only=False)
    g, F = np.asarray(d["grid"]), d["folds"]
    for dev in (dev_a, dev_b):
        if dev not in F:
            raise SystemExit(f"{run}: no fold for {dev}. have: {list(F)}")

    m = F[dev_a]["mask"].astype(bool) & F[dev_b]["mask"].astype(bool)
    f = g[m]
    ya, pa = F[dev_a]["y"][m], F[dev_a]["pred"][m]
    yb, pb = F[dev_b]["y"][m], F[dev_b]["pred"][m]
    mae = lambda x, y: float(np.abs(x - y).mean())

    dt, dp = yb - ya, pb - pa
    corr = float(np.corrcoef(dt, dp)[0, 1])
    sign = float((np.sign(dt) == np.sign(dp)).mean())
    twin_base = mae(ya, yb)

    print(f"== {run}: twin check on {dev_a} / {dev_b} ==")
    print(f"  band {f.min():.0f}-{f.max():.0f} Hz, {m.sum()} bins")
    print(f"  MAE(comsol_A, comsol_B)      = {twin_base:.4f}  <- signal to resolve")
    print(f"  pred_A vs own comsol_A       = {mae(pa, ya):.4f}")
    print(f"  pred_A vs TWIN comsol_B      = {mae(pa, yb):.4f}")
    print(f"  pred_B vs own comsol_B       = {mae(pb, yb):.4f}")
    print(f"  pred_B vs TWIN comsol_A      = {mae(pb, ya):.4f}")
    print(f"  corr(delta_true, delta_pred) = {corr:+.4f}")
    print(f"  sign agreement               = {sign:.1%}")

    short_a, short_b = dev_a.split("_")[-1], dev_b.split("_")[-1]
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 1, 0.85], hspace=0.16))
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.6, color="0.75")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # --- panels 1-2: each held-out device vs its own truth AND its twin's truth ----
    for ax, (held, c_h, p_h, y_h, y_tw, c_tw, s_h, s_tw) in zip(
            axes[:2],
            [(dev_b, C_B, pb, yb, ya, C_A, short_b, short_a),
             (dev_a, C_A, pa, ya, yb, C_B, short_a, short_b)]):
        ax.plot(f, y_tw, color=c_tw, lw=2.0, alpha=0.85,
                label=f"COMSOL {s_tw} (the TWIN, in training set)")
        ax.plot(f, y_h, color=c_h, lw=2.0, label=f"COMSOL {s_h} (own ground truth)")
        ax.plot(f, p_h, color=c_h, lw=2.0, ls="--", label=f"prediction {s_h} (held out)")
        e_own, e_twin = mae(p_h, y_h), mae(p_h, y_tw)
        verdict = "CLOSER TO TWIN" if e_twin < e_own else "closer to own truth"
        ax.set_title(f"Held out: {held}   —   pred vs own COMSOL {e_own:.4f}  ·  "
                     f"pred vs twin COMSOL {e_twin:.4f}   →  {verdict}",
                     fontsize=10, color=INK, loc="left")
        ax.set_ylabel("absorption", color=INK2)
        ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95)

    # --- panel 3: the difference — the quantity that actually decides it ----------
    ax = axes[2]
    ax.axhline(0, color="0.6", lw=1.0)
    ax.plot(f, dt, color=C_TRUE, lw=2.0, label=f"delta_true = COMSOL {short_b} - COMSOL {short_a}")
    ax.plot(f, dp, color=C_PRED, lw=2.0, ls="--", label=f"delta_pred = pred {short_b} - pred {short_a}")
    k = int(np.argmax(np.abs(dt)))
    _label(ax, f[k], dt[k], "delta_true", C_TRUE, va="bottom")
    k = int(np.argmax(np.abs(dp)))
    _label(ax, f[k], dp[k], "delta_pred", C_PRED, va="top")
    ax.set_title(f"The pair difference — corr = {corr:+.3f}, sign agreement = {sign:.0%} "
                 f"(chance 50%)   →  {'resolves the pair' if corr > 0.3 else 'does NOT resolve the pair'}",
                 fontsize=10, color=INK, loc="left")
    ax.set_ylabel("absorption difference", color=INK2)
    ax.set_xlabel("frequency (Hz)", color=INK2)
    lim = 1.05 * max(np.abs(dt).max(), np.abs(dp).max())
    ax.set_ylim(-lim, lim * 1.7)          # headroom so the legend clears the traces
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95)

    fig.suptitle(f"Is `{run}` learning geometry, or copying its near-duplicate twin?\n"
                 f"dual GNN MAE {mae(pa, ya):.4f} / {mae(pb, yb):.4f}   vs   "
                 f"'copy the twin' baseline {twin_base:.4f}",
                 fontsize=12, color=INK)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.925, bottom=0.06, hspace=0.22)

    out = out or str(res / "twin_check.png")
    fig.savefig(out, dpi=160)
    print(f"  wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="dual_gnn_res")
    ap.add_argument("--dev-a", default=DEV_A)
    ap.add_argument("--dev-b", default=DEV_B)
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    twin_check(a.run, a.dev_a, a.dev_b, a.out)
