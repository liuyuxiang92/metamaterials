"""Visual check of the voxel representation: rasterized layer images + footprint outlines.

Renders every layer of a device as its binary air-occupancy image (method #3 input) with the
exact footprint polygons drawn on top, so you can confirm by eye that the rasterization fills
the right cells and that the layers are stacked in the right order.

    python -m gnn.representations.plot_slices AI2026_Trial_Ver1R-4
    python -m gnn.representations.plot_slices AI2026_Trial-Ver2-1a -o /tmp/slices.png
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from ..data.brep_extract import _footprint_face
from ..data.devices import discover_devices
from ..data.face_extract import (TopAbs_SOLID, _as_solid, _iter, _load_shape,
                                 _ordered_vertices)
from ..data.step_extract import _bbox
from .voxel import PX_MM, _canvas, _layer_images


def plot_slices(device: str, px_mm: float = PX_MM, out: str | None = None):
    import matplotlib
    if out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    origin, H, W = _canvas(px_mm)
    vox, lay, areas = _layer_images(device, px_mm, origin, H, W)
    L = len(vox)

    # collect footprint outlines per layer for overlay
    shape = _load_shape(discover_devices()[device].step)
    by_layer = defaultdict(list)
    for shp in _iter(shape, TopAbs_SOLID):
        sol = _as_solid(shp)
        (_, _, z0), (_, _, z1) = _bbox(sol)
        f = _footprint_face(sol)
        if f is not None:
            by_layer[round(0.5 * (z0 + z1), 1)].append(_ordered_vertices(f)[:, :2])
    zs = sorted(by_layer)

    ncols = min(5, L)
    nrows = int(np.ceil(L / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    extent = [origin[0], origin[0] + W * px_mm, origin[1], origin[1] + H * px_mm]
    for i in range(nrows * ncols):
        ax = axes.ravel()[i]
        if i >= L:
            ax.axis("off"); continue
        ax.imshow(vox[i], origin="lower", extent=extent, cmap="Greys", vmin=0, vmax=1)
        for V in by_layer[zs[i]]:
            loop = np.vstack([V, V[:1]])
            ax.plot(loop[:, 0], loop[:, 1], color="#d62728", lw=0.7)
        ax.set_title(f"layer {i}  z={lay[i,1]:.1f}  t={lay[i,0]:.1f}mm", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    ras = vox.sum(axis=(1, 2)) * px_mm * px_mm
    rel = np.abs(ras - areas) / np.clip(areas, 1e-9, None)
    fig.suptitle(f"{device} — {L} layers @ {px_mm} mm  (max area err {rel.max()*100:.2f}% vs exact)")
    fig.tight_layout()
    out = out or f"/tmp/{device}_slices.png"
    fig.savefig(out, dpi=110)
    print(f"wrote {out}  (max rel area err {rel.max()*100:.2f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("device", nargs="?", default="AI2026_Trial_Ver1R-4")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--px", type=float, default=PX_MM)
    a = ap.parse_args()
    plot_slices(a.device, a.px, a.out)
