"""Representation #3: a device as an ordered stack of rasterized layer images.

Every solid in these STEPs is an exact Z-extruded prism (``volume == footprint_area *
thickness`` to 0.0000%), and the tubes are organized into ``L`` uniform-pitch layers along
Z. So a device is losslessly described by, per layer, ONE binary occupancy image of the air
footprint plus that layer's thickness and z-position — no need to sample redundant slices
through the extrusion. This is the "several CNN along z" idea (method #3) with per-layer
sampling instead of a uniform 1 mm grid (which would duplicate 4-10 identical slices per
layer and, being incommensurate with the 5.5/7.7/11 mm layer pitches, would encode thickness
as an unstable run length).

Per device we build a ``torch_geometric.data.Data`` (used only as a typed attribute bag, NOT
as a graph) with:
  * ``vox``   (Dmax, H, W)  binary air-occupancy per layer, zero-padded above the real layers
  * ``lay``   (Dmax, 2)     per-layer ``[thickness_mm, z_center_mm]``
  * ``lmask`` (Dmax,)       1 for real layers, 0 for padding
  * ``y, mask, comb``       as in the other representations (targets / optional resonance comb)

Why these scalars: thickness is passed EXACTLY here rather than as a slice count. The 1.0 mm
partition GAP is deliberately omitted — it is 1.0 mm in every current device (zero variance,
would divide-by-zero in the scaler); re-add it as a third column when a device with a
different partition appears.

The rasterization canvas is SHARED across all devices (common origin = global bbox-min, common
extent = max over devices) so every image lands in the same physical frame and is directly
comparable layer-to-layer. Cached to ``results/cache/<device>_vox_<px_mm>.npz`` — the cache key
includes ``px_mm`` so changing resolution never silently reuses stale voxels.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from matplotlib.path import Path as MplPath
from torch_geometric.data import Data

from ..data.brep_extract import _footprint_face
from ..data.devices import GNN_DIR, discover_devices
from ..data.face_extract import (TopAbs_SOLID, _as_solid, _iter, _load_shape,
                                 _ordered_vertices, _surface_props)
from ..data.step_extract import _bbox

PX_MM = 0.25          # pixel size (mm); 0.25 resolves the 3 tube widths (18/20/32 px) exactly
DMAX = 14             # max layer count across devices (Ver2); shorter stacks are padded
LAYER_Z_TOL = 1.0     # solids whose bbox z-centre is within this are the same layer (mm)


# --------------------------------------------------------------------------- #
# Shared rasterization frame
# --------------------------------------------------------------------------- #
def _canvas(px_mm: float) -> tuple[np.ndarray, int, int]:
    """(origin_xy, H, W) for a frame that contains every device's footprint at ``px_mm``."""
    origin = None
    hi = None
    for dev in discover_devices().values():
        (xmn, ymn, _), (xmx, ymx, _) = _bbox(_load_shape(dev.step))
        lo = np.array([xmn, ymn]); h = np.array([xmx, ymx])
        origin = lo if origin is None else np.minimum(origin, lo)
        hi = h if hi is None else np.maximum(hi, h)
    ext = hi - origin
    W = int(np.ceil(ext[0] / px_mm))
    H = int(np.ceil(ext[1] / px_mm))
    return origin, H, W


def _rasterize(verts_xy: np.ndarray, origin: np.ndarray, H: int, W: int,
               px_mm: float, grid: np.ndarray) -> None:
    """OR one footprint polygon (absolute XY) into the boolean canvas ``grid`` (H, W).

    Only the pixels inside the polygon's own bbox are tested — testing the whole canvas is
    ~H*W points per polygon (millions) for no reason.
    """
    p = (verts_xy - origin) / px_mm                       # polygon in pixel coords
    x0 = max(int(np.floor(p[:, 0].min())), 0)
    x1 = min(int(np.ceil(p[:, 0].max())), W)
    y0 = max(int(np.floor(p[:, 1].min())), 0)
    y1 = min(int(np.ceil(p[:, 1].max())), H)
    if x1 <= x0 or y1 <= y0:
        return
    xs = np.arange(x0, x1) + 0.5                           # pixel centres
    ys = np.arange(y0, y1) + 0.5
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    inside = MplPath(p).contains_points(pts).reshape(y1 - y0, x1 - x0)
    grid[y0:y1, x0:x1] |= inside


# --------------------------------------------------------------------------- #
# Per-device build
# --------------------------------------------------------------------------- #
def _layer_images(device: str, px_mm: float, origin: np.ndarray, H: int, W: int):
    """Return (vox[L,H,W] bool, lay[L,2], areas[L]) for the real layers, bottom->top."""
    shape = _load_shape(discover_devices()[device].step)
    solids = _iter(shape, TopAbs_SOLID)

    # group solids into layers by bbox z-centre
    by_layer: dict[float, list] = defaultdict(list)
    for shp in solids:
        sol = _as_solid(shp)
        (_, _, z0), (_, _, z1) = _bbox(sol)
        by_layer[round(0.5 * (z0 + z1), 1)].append((sol, z0, z1))

    zs = sorted(by_layer)
    vox = np.zeros((len(zs), H, W), dtype=bool)
    lay = np.zeros((len(zs), 2), dtype=np.float32)
    areas = np.zeros(len(zs), dtype=np.float64)
    for i, zc in enumerate(zs):
        thick = np.mean([z1 - z0 for _, z0, z1 in by_layer[zc]])
        lay[i] = [thick, zc]
        for sol, _, _ in by_layer[zc]:
            face = _footprint_face(sol)
            if face is None:
                continue
            V = _ordered_vertices(face)[:, :2]
            _rasterize(V, origin, H, W, px_mm, vox[i])
            areas[i] += _surface_props(face)[0]
    return vox, lay, areas


def _cache_path(device: str, px_mm: float):
    d = GNN_DIR / "results" / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{device}_vox_{px_mm:g}.npz"


def build(
    device: str,
    *,
    px_mm: float = PX_MM,
    dmax: int = DMAX,
    canvas: tuple[np.ndarray, int, int] | None = None,
    use_cache: bool = True,
    y: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    add_resonance: bool = False,
    resonance_k: int = 3,
    resonance_mode: str = "quarter",
    resonance_sigma: float = 40.0,
    grid: np.ndarray | None = None,
    add_gaps: bool = False,
    # accepted-and-ignored so the shared training loop can pass one kwargs set:
    **_ignored,
) -> Data:
    cache = _cache_path(device, px_mm)
    if use_cache and cache.exists():
        z = np.load(cache)
        shp = tuple(int(x) for x in z["vox_shape"])
        vox = np.unpackbits(z["vox"], count=int(np.prod(shp))).astype(bool).reshape(shp)
        lay = z["lay"]
    else:
        if canvas is None:
            canvas = _canvas(px_mm)
        origin, H, W = canvas
        vox, lay, _ = _layer_images(device, px_mm, origin, H, W)
        # store bit-packed to keep the cache small (a full canvas is mostly empty)
        np.savez_compressed(cache, vox=np.packbits(vox), vox_shape=np.asarray(vox.shape),
                            lay=lay)

    L, H, W = vox.shape
    if L > dmax:
        raise ValueError(f"{device}: {L} layers exceeds dmax={dmax}")

    if add_gaps:
        # Inter-layer gaps from the cached [thickness, z_center] (layers are sorted bottom->top):
        #   gap[i] = (zc[i+1]-t[i+1]/2) - (zc[i]+t[i]/2)   (partition between layer i and i+1)
        # g_minus = gap below a layer, g_plus = gap above it; boundary layers have no neighbour
        # so their outward gap is 0 (a real, distinguishing "front/back" marker). See models/cnn.py.
        t, zc = lay[:, 0], lay[:, 1]
        gap = (zc[1:] - t[1:] / 2.0) - (zc[:-1] + t[:-1] / 2.0)   # (L-1,)
        g_minus = np.concatenate([[0.0], gap]).astype(np.float32)  # gap below layer i
        g_plus = np.concatenate([gap, [0.0]]).astype(np.float32)   # gap above layer i
        lay = np.column_stack([lay, g_minus, g_plus]).astype(np.float32)  # (L, 4)

    vpad = np.zeros((dmax, H, W), dtype=np.float32)
    vpad[:L] = vox.astype(np.float32)
    lpad = np.zeros((dmax, lay.shape[1]), dtype=np.float32)
    lpad[:L] = lay
    lmask = np.zeros(dmax, dtype=np.float32)
    lmask[:L] = 1.0

    data = Data()
    data.vox = torch.from_numpy(vpad)                     # (Dmax, H, W)
    data.lay = torch.from_numpy(lpad)                     # (Dmax, 2)
    data.lmask = torch.from_numpy(lmask)                  # (Dmax,)
    data.num_layers = L
    data.device_name = device
    data.num_nodes = dmax                                 # silences PyG's num_nodes inference

    if add_resonance and grid is not None:
        from ..data.resonance import resonance_comb
        comb = resonance_comb(device, grid, k=resonance_k, mode=resonance_mode,
                              sigma=resonance_sigma)
        data.comb = torch.tensor(comb, dtype=torch.float32).unsqueeze(0)  # (1, G)
    if y is not None:
        data.y = torch.tensor(np.asarray(y), dtype=torch.float32).unsqueeze(0)
    if mask is not None:
        data.mask = torch.tensor(np.asarray(mask), dtype=torch.float32).unsqueeze(0)
    return data


def collate_voxel(samples: list[Data]) -> Data:
    """Stack per-device voxel ``Data`` into one batched ``Data`` with a leading batch dim.

    vox -> (B, Dmax, H, W), lay -> (B, Dmax, 2), lmask -> (B, Dmax),
    y/mask/comb -> (B, G). Kept separate from PyG's node-concat collation on purpose: these
    are dense per-device tensors, not variable-size node sets.
    """
    b = Data()
    b.vox = torch.stack([s.vox for s in samples], dim=0)
    b.lay = torch.stack([s.lay for s in samples], dim=0)
    b.lmask = torch.stack([s.lmask for s in samples], dim=0)
    b.num_graphs = len(samples)
    b.device_names = [s.device_name for s in samples]
    for attr in ("y", "mask", "comb"):
        if hasattr(samples[0], attr):
            setattr(b, attr, torch.cat([getattr(s, attr) for s in samples], dim=0))
    return b


def model_dims(sample: Data) -> dict:
    return dict(
        scalar_dim=sample.lay.size(-1),
        comb_dim=int(sample.comb.size(1)) if hasattr(sample, "comb") else 0,
    )


if __name__ == "__main__":
    from ..data.targets import load_targets
    ts = load_targets()
    cvs = _canvas(PX_MM)
    print(f"shared canvas: H={cvs[1]} W={cvs[2]} @ {PX_MM} mm  (origin {np.round(cvs[0],1)})")
    for dev in ts.y:
        g = build(dev, canvas=cvs, y=ts.y[dev], mask=ts.mask[dev])
        occ = g.vox[: g.num_layers].sum().item()
        print(f"  {dev:26s} layers={g.num_layers:2d}  vox{tuple(g.vox.shape)}  occ_px={occ:.0f}")
