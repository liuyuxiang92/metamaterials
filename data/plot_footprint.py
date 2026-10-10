"""Visual check of the footprint wall-split used by ``brep_extract``.

Plots one tube's "curly" footprint face (the extruded cross-section of the folded duct) and
labels the two parallel curly walls that ``brep_extract.split_walls`` pairs up to build the
centerline. Use it to confirm by eye that:

  * the open-cap edge was found at the right end of the maze,
  * wall A and wall B run alongside each other (not one wrapping the whole loop),
  * their vertex-wise average lands on the duct centerline.

    python -m gnn.data.plot_footprint AI2026_Trial_Ver1R-4 58
    python -m gnn.data.plot_footprint AI2026_Trial_Ver1R-4 58 -o /tmp/air058.png

The tube number is 1-based and matches ``tube_index`` in ``gnn/tube_geometry.csv``.
"""
from __future__ import annotations

import argparse

import numpy as np

from .brep_extract import split_walls
from .devices import discover_devices
from .face_extract import TopAbs_SOLID, _as_solid, _iter, _load_shape

WALL_A_COLOR = "#1f77b4"
WALL_B_COLOR = "#d62728"


def plot_tube(device: str, tube_index: int, out_path: str | None = None,
              show_pairing: bool = True):
    import matplotlib
    if out_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dev = discover_devices()[device]
    solids = _iter(_load_shape(dev.step), TopAbs_SOLID)
    if not 1 <= tube_index <= len(solids):
        raise SystemExit(f"{device}: tube_index must be 1..{len(solids)}, got {tube_index}")

    res = split_walls(_as_solid(solids[tube_index - 1]))
    if res is None:
        raise SystemExit(f"{device} tube {tube_index}: no usable footprint face")
    V, wall1, wall2, i_open, W = res
    n = len(V)
    cl = (wall1 + wall2) / 2.0

    # --- consistency numbers printed alongside the picture -------------------
    d = np.linalg.norm(wall1 - wall2, axis=1)
    s1 = np.diff(wall1[:, :2], axis=0)
    s2 = np.diff(wall2[:, :2], axis=0)
    cos = np.sum(s1 * s2, axis=1) / (np.linalg.norm(s1, axis=1) * np.linalg.norm(s2, axis=1)
                                     + 1e-12)
    L = float(np.linalg.norm(np.diff(cl, axis=0), axis=1).sum())
    print(f"== {device} tube {tube_index} (air{tube_index:03d}) ==")
    print(f"  footprint loop      : {n} vertices -> {len(wall1)} per wall")
    print(f"  channel width W     : {W:.3f} mm")
    print(f"  wall separation     : min {d.min():.3f}  max {d.max():.3f} mm  (should be ~W)")
    print(f"  paired-segment cos  : min {cos.min():.6f}  (1.0 = perfectly parallel)")
    print(f"  centerline length   : {L:.2f} mm")

    # --- plot ---------------------------------------------------------------
    span = V[:, :2].max(axis=0) - V[:, :2].min(axis=0)
    h = 9.0
    fig, ax = plt.subplots(figsize=(max(5.5, min(12.0, h * span[0] / max(span[1], 1e-9) + 4.2)), h))
    loop = np.vstack([V, V[:1]])
    ax.fill(loop[:, 0], loop[:, 1], color="0.90", zorder=0, label="air domain (footprint)")

    if show_pairing:
        for a, b in zip(wall1, wall2):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="0.65", lw=0.6, zorder=1)

    ax.plot(wall1[:, 0], wall1[:, 1], color=WALL_A_COLOR, lw=2.2, zorder=3,
            label=f"curly wall A ({len(wall1)} verts)")
    ax.plot(wall2[:, 0], wall2[:, 1], color=WALL_B_COLOR, lw=2.2, zorder=3,
            label=f"curly wall B ({len(wall2)} verts)")
    ax.plot(cl[:, 0], cl[:, 1], color="k", lw=1.0, ls="--", zorder=4,
            label=f"centerline (A+B)/2, {L:.1f} mm")

    cap = np.array([V[i_open], V[(i_open + 1) % n]])
    ax.plot(cap[:, 0], cap[:, 1], color="#2ca02c", lw=3.5, zorder=5, label="open cap")
    end = np.array([wall1[-1], wall2[-1]])
    ax.plot(end[:, 0], end[:, 1], color="#9467bd", lw=3.5, zorder=5, label="closed end")

    # start markers so the open->end direction is unambiguous
    for p, name, color, dx in ((wall1[0], "A", WALL_A_COLOR, -6),
                               (wall2[0], "B", WALL_B_COLOR, 6)):
        ax.scatter(*p[:2], s=55, color=color, zorder=6, edgecolor="w")
        ax.annotate(f"{name} start", p[:2], textcoords="offset points", xytext=(dx, -16),
                    ha="right" if dx < 0 else "left", color=color, fontsize=9,
                    fontweight="bold", zorder=7)

    # label each wall on its longest segment, offset outward away from its partner
    label_pos = []
    for wall, other, name, color in ((wall1, wall2, "A", WALL_A_COLOR),
                                     (wall2, wall1, "B", WALL_B_COLOR)):
        k = int(np.argmax(np.linalg.norm(np.diff(wall[:, :2], axis=0), axis=1)))
        anchor = (wall[k, :2] + wall[k + 1, :2]) / 2.0
        off = anchor - (other[k, :2] + other[k + 1, :2]) / 2.0
        off = off / (np.linalg.norm(off) + 1e-12) * (0.9 * W)
        pos = anchor + off
        label_pos.append(pos)
        ax.annotate(f"curly wall {name}", xy=anchor, xytext=pos, color=color, fontsize=11,
                    fontweight="bold", ha="center", va="center", zorder=7,
                    rotation=90 if abs(off[0]) > abs(off[1]) else 0,
                    bbox=dict(fc="w", ec=color, alpha=0.9, boxstyle="round,pad=0.25"),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2,
                                    shrinkA=2, shrinkB=0))

    # annotations don't participate in autoscaling -> fold them into the limits by hand
    pts = np.vstack([V[:, :2]] + [np.asarray(label_pos)])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.6 * W
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.suptitle(f"{device} — air{tube_index:03d} footprint\n"
                 f"W={W:.2f} mm · {n} loop vertices · wall gap "
                 f"{d.min():.2f}–{d.max():.2f} mm · cos={cos.min():.4f}", fontsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, framealpha=0.9,
              borderaxespad=0.0)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=160)
        print(f"  wrote {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("device", nargs="?", default="AI2026_Trial_Ver1R-4")
    ap.add_argument("tube", nargs="?", type=int, default=58, help="1-based tube index")
    ap.add_argument("-o", "--out", default=None, help="save to PNG instead of showing")
    ap.add_argument("--no-pairing", action="store_true", help="hide wall-pairing rungs")
    a = ap.parse_args()
    plot_tube(a.device, a.tube, a.out, show_pairing=not a.no_pairing)
