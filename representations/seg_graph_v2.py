"""Representation #1 v2 (``seg_egnn``): one node per straight tube segment, THREE typed
edges each carrying its own edge-feature vector, packed as a ``HeteroData`` with a single
node type ``"seg"``.

Node  : a straight segment. 12-d feature = ``[c(3), d_hat(3), L, w, h, D_h, s, L_tube]``
        (see ``representations/base.py``; the config picks these by name).

Edges (three relations, each with its own ``edge_attr`` width):
  * ``("seg","path","seg")``    — directly-adjacent segments of the SAME tube (a fold).
        ``[Δc(3), d, cosθ, τ]``  (6)  — τ = z-component of d_hat_i × d_hat_j (turn sign).
  * ``("seg","open","seg")``    — segment-0 "opening" nodes within a cutoff on the inlet plane.
        ``[Δo(3), d_open, Δz, A_a, A_b]``  (7)  — o = opening center (p0 of segment 0).
  * ``("seg","spatial","seg")`` — non-adjacent intra-tube + inter-tube segments whose true
        segment-segment min distance ``d_min`` ≤ cutoff.
        ``[Δc(3), d_min, cosθ, r_overlap, Δz]``  (7).

``r_overlap`` (projection-only): project segment j onto segment i's own axis d_hat_i through
c_i; overlap of the two projected intervals / min(L_i, L_j). Parallel-alongside → 1;
perpendicular → j projects to ~its own width → ~0, with no angle threshold. Edges are stored
BOTH directions, so the reverse edge projects onto d_hat_j (asymmetric, as intended).

Normalization (in ``training/dataset.py``): only the *physical* (mm / mm²) columns are
standardized per fold; the unit/bounded columns listed in the ``*_RAW_COLS`` constants below
(direction, cosθ, τ, r_overlap) are left raw. Those constants live here so builder and scaler
cannot drift.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import HeteroData

from ..data.fold_coords import get_segments
from ..data.schema import SegmentRecord
from .base import compute_features, node_feature_dim

SEG = "seg"
PATH = (SEG, "path", SEG)
OPEN = (SEG, "open", SEG)
SPATIAL = (SEG, "spatial", SEG)

_PARTITION_MM = 2.0
_Z = np.array([0.0, 0.0, 1.0])

# Canonical v2 node layout (configs/seg_egnn.yaml feature_names). The builder asserts the
# seg_direction span sits at [3:6] so these raw-column constants stay valid.
NODE_RAW_COLS = (3, 4, 5)          # seg_direction (unit vector) — leave raw
PATH_RAW_COLS = (4, 5)             # cosθ, τ
OPEN_RAW_COLS: tuple[int, ...] = ()  # Δo, d_open, Δz, A_a, A_b — all physical
SPATIAL_RAW_COLS = (4, 5)          # cosθ, r_overlap

PATH_EDGE_DIM = 6
OPEN_EDGE_DIM = 7
SPATIAL_EDGE_DIM = 7


# --------------------------------------------------------------------------- #
# geometry helpers (pure numpy)
# --------------------------------------------------------------------------- #
def _segment_segment_dist(p1: np.ndarray, q1: np.ndarray,
                          p2: np.ndarray, q2: np.ndarray) -> float:
    """Minimum distance between two 3-D line segments [p1,q1] and [p2,q2].

    Clamped closest-point algorithm (Ericson, Real-Time Collision Detection).
    """
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(d1 @ d1)
    e = float(d2 @ d2)
    f = float(d2 @ r)
    eps = 1e-12
    if a <= eps and e <= eps:                       # both degenerate to points
        return float(np.linalg.norm(p1 - p2))
    if a <= eps:                                    # segment 1 is a point
        s = 0.0
        t = min(max(f / e, 0.0), 1.0)
    else:
        c = float(d1 @ r)
        if e <= eps:                                # segment 2 is a point
            t = 0.0
            s = min(max(-c / a, 0.0), 1.0)
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            s = min(max((b * f - c * e) / denom, 0.0), 1.0) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = min(max(-c / a, 0.0), 1.0)
            elif t > 1.0:
                t = 1.0
                s = min(max((b - c) / a, 0.0), 1.0)
    c1 = p1 + d1 * s
    c2 = p2 + d2 * t
    return float(np.linalg.norm(c1 - c2))


def _overlap_ratio(ci: np.ndarray, di: np.ndarray, Li: float,
                   cj: np.ndarray, dj: np.ndarray, Lj: float) -> float:
    """Project segment j onto segment i's axis d_hat_i through c_i; return the interval
    overlap / min(L_i, L_j). Parallel-alongside → ~1; perpendicular → ~0 (no threshold)."""
    if Li < 1e-9 or Lj < 1e-9:
        return 0.0
    hi = 0.5 * Li                                   # i spans [-hi, +hi] about its center
    cos = float(np.clip(di @ dj, -1.0, 1.0))
    tj = float((cj - ci) @ di)                      # j's center projected onto d_hat_i
    hj = 0.5 * Lj * abs(cos)                        # j's projected half-extent
    lo = max(-hi, tj - hj)
    hi_ = min(hi, tj + hj)
    overlap = max(0.0, hi_ - lo)
    return overlap / min(Li, Lj)


def _cutoff(areas: np.ndarray) -> float:
    """Wall-adjacent centerline separation ≈ one duct side + partition."""
    side = math.sqrt(float(np.median(areas)))
    return side + _PARTITION_MM


def _sample_points(P0: np.ndarray, P1: np.ndarray, spacing: float):
    """Sample each segment at ≤``spacing`` intervals (≥ endpoints); return (points, owner)."""
    pts: list[np.ndarray] = []
    owner: list[int] = []
    for i in range(len(P0)):
        a, b = P0[i], P1[i]
        L = float(np.linalg.norm(b - a))
        k = max(2, int(math.ceil(L / spacing)) + 1)
        for t in np.linspace(0.0, 1.0, k):
            pts.append(a + (b - a) * t)
            owner.append(i)
    return np.asarray(pts), np.asarray(owner)


def _spatial_candidates(P0: np.ndarray, P1: np.ndarray, cutoff: float) -> set[tuple[int, int]]:
    """Segment pairs whose sampled points come within ~cutoff — a tight superset of the
    pairs with d_min ≤ cutoff (exact d_min is checked afterwards)."""
    if len(P0) < 2:
        return set()
    pts, owner = _sample_points(P0, P1, cutoff)
    tree = cKDTree(pts)
    cand: set[tuple[int, int]] = set()
    for a, b in tree.query_pairs(r=1.5 * cutoff):
        oa, ob = int(owner[a]), int(owner[b])
        if oa != ob:
            cand.add((oa, ob) if oa < ob else (ob, oa))
    return cand


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build(
    device: str,
    *,
    source: str = "brep",
    feature_names: tuple[str, ...] = ("seg_center", "seg_direction", "seg_length",
                                      "width_height", "hydraulic_diameter", "dist_to_open",
                                      "tube_length"),
    open_cutoff_scale: float = 1.5,
    spatial_cutoff_scale: float = 1.5,
    y: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    add_resonance: bool = False,
    resonance_k: int = 3,
    resonance_mode: str = "quarter",
    resonance_sigma: float = 40.0,
    grid: np.ndarray | None = None,
) -> HeteroData:
    feature_names = list(feature_names)
    _assert_layout(feature_names)
    segs = get_segments(device, source=source)

    # order nodes tube-by-tube, open->end
    by_tube: dict[int, list[SegmentRecord]] = {}
    for s in segs:
        by_tube.setdefault(s.tube_index, []).append(s)
    for lst in by_tube.values():
        lst.sort(key=lambda s: s.segment_index)

    node_segs: list[SegmentRecord] = []
    node_tube: list[int] = []
    node_sidx: list[int] = []
    feats: list[list[float]] = []
    path_src: list[int] = []
    path_dst: list[int] = []
    for tube, lst in sorted(by_tube.items()):
        tube_length = float(sum(s.length for s in lst))
        cum_len = 0.0
        prev: SegmentRecord | None = None
        start = len(node_segs)
        for i, s in enumerate(lst):
            ctx = {"prev": prev, "cum_len": cum_len, "tube_length": tube_length}
            feats.append(compute_features(s, ctx, feature_names))
            node_segs.append(s)
            node_tube.append(tube)
            node_sidx.append(s.segment_index)
            if i > 0:
                a, b = start + i - 1, start + i
                path_src += [a, b]                  # both directions
                path_dst += [b, a]
            cum_len += s.length
            prev = s

    n = len(node_segs)
    C = np.array([s.midpoint for s in node_segs])                       # (n,3) centers
    P0 = np.array([s.p0_arr for s in node_segs])
    P1 = np.array([s.p1_arr for s in node_segs])
    L = np.array([s.length for s in node_segs])
    D = np.zeros((n, 3))
    nz = L > 1e-9
    D[nz] = (P1[nz] - P0[nz]) / L[nz, None]                             # (n,3) unit dirs
    areas = np.array([s.cross_section_mm2 for s in node_segs])
    tube_arr = np.array(node_tube)
    sidx_arr = np.array(node_sidx)

    data = HeteroData()

    # ---- optional resonance appended to node features (per-tube broadcast) ----
    if add_resonance:
        from ..data.resonance import tube_resonances
        res = tube_resonances(device, k=resonance_k, mode=resonance_mode)
        zero = [0.0] * resonance_k
        for i, s in enumerate(node_segs):
            fn = res[s.tube_index].tolist()[:resonance_k] if s.tube_index in res else zero
            feats[i] = feats[i] + list(fn)

    data[SEG].x = torch.tensor(feats, dtype=torch.float32)
    data[SEG].num_nodes = n

    # ---- path edges ----
    data[PATH].edge_index = torch.tensor([path_src, path_dst], dtype=torch.long)
    data[PATH].edge_attr = torch.tensor(
        [_path_feat(C, D, a, b) for a, b in zip(path_src, path_dst)],
        dtype=torch.float32).reshape(-1, PATH_EDGE_DIM)

    # ---- opening edges (segment-0 nodes) ----
    open_nodes = np.where(sidx_arr == 0)[0]
    o_pts = P0[open_nodes]                                              # opening centers
    osrc: list[int] = []
    odst: list[int] = []
    oattr: list[list[float]] = []
    if len(open_nodes) > 1:
        ocut = _cutoff(areas) * open_cutoff_scale
        otree = cKDTree(o_pts)
        for ia, ib in otree.query_pairs(r=ocut):
            a, b = int(open_nodes[ia]), int(open_nodes[ib])
            for u, v in ((a, b), (b, a)):
                osrc.append(u); odst.append(v)
                oattr.append(_open_feat(P0, areas, u, v))
    data[OPEN].edge_index = torch.tensor([osrc, odst], dtype=torch.long).reshape(2, -1)
    data[OPEN].edge_attr = torch.tensor(oattr, dtype=torch.float32).reshape(-1, OPEN_EDGE_DIM)

    # ---- spatial edges (non-adjacent intra + inter, filtered by exact d_min) ----
    ssrc: list[int] = []
    sdst: list[int] = []
    sattr: list[list[float]] = []
    if n > 1:
        scut = _cutoff(areas) * spatial_cutoff_scale
        for a, b in _spatial_candidates(P0, P1, scut):
            if tube_arr[a] == tube_arr[b] and abs(sidx_arr[a] - sidx_arr[b]) == 1:
                continue                                               # that's a path edge
            dmin = _segment_segment_dist(P0[a], P1[a], P0[b], P1[b])
            if dmin > scut:
                continue
            for u, v in ((a, b), (b, a)):
                ssrc.append(u); sdst.append(v)
                sattr.append(_spatial_feat(C, D, L, areas, u, v, dmin))
    data[SPATIAL].edge_index = torch.tensor([ssrc, sdst], dtype=torch.long).reshape(2, -1)
    data[SPATIAL].edge_attr = torch.tensor(sattr, dtype=torch.float32).reshape(-1, SPATIAL_EDGE_DIM)

    data.device_name = device
    if add_resonance and grid is not None:
        from ..data.resonance import resonance_comb
        comb = resonance_comb(device, grid, k=resonance_k, mode=resonance_mode,
                              sigma=resonance_sigma)
        data.comb = torch.tensor(comb, dtype=torch.float32).unsqueeze(0)
    if y is not None:
        data.y = torch.tensor(np.asarray(y), dtype=torch.float32).unsqueeze(0)
    if mask is not None:
        data.mask = torch.tensor(np.asarray(mask), dtype=torch.float32).unsqueeze(0)
    return data


def _path_feat(C, D, a, b) -> list[float]:
    dc = C[b] - C[a]
    cos = float(np.clip(D[a] @ D[b], -1.0, 1.0))
    tau = float(np.cross(D[a], D[b]) @ _Z)          # turn sign in the fold plane
    return [dc[0], dc[1], dc[2], float(np.linalg.norm(dc)), cos, tau]


def _open_feat(P0, areas, a, b) -> list[float]:
    do = P0[b] - P0[a]                              # opening-center displacement
    return [do[0], do[1], do[2], float(np.linalg.norm(do)),
            float(P0[b][2] - P0[a][2]), float(areas[a]), float(areas[b])]


def _spatial_feat(C, D, L, areas, a, b, dmin) -> list[float]:
    dc = C[b] - C[a]
    cos = float(np.clip(D[a] @ D[b], -1.0, 1.0))
    r = _overlap_ratio(C[a], D[a], float(L[a]), C[b], D[b], float(L[b]))
    return [dc[0], dc[1], dc[2], float(dmin), cos, r, float(C[a][2] - C[b][2])]


def _assert_layout(feature_names: list[str]) -> None:
    """The raw-column constants assume seg_direction occupies node columns [3:6]."""
    off = 0
    for name in feature_names:
        from .base import FEATURE_DIMS
        if name == "seg_direction":
            if off != 3:
                raise ValueError(
                    f"seg_graph_v2 expects seg_direction at node cols [3:6], got offset {off}; "
                    f"keep the canonical feature order in the config.")
            return
        off += FEATURE_DIMS[name]
    raise ValueError("seg_graph_v2 requires 'seg_direction' in feature_names")


def model_dims(sample: HeteroData) -> dict:
    """Input dims for the model factory, read off a built graph."""
    d = {
        "node_in": int(sample[SEG].x.size(1)),
        "path_edge": PATH_EDGE_DIM,
        "open_edge": OPEN_EDGE_DIM,
        "spatial_edge": SPATIAL_EDGE_DIM,
    }
    if hasattr(sample, "comb"):
        d["comb_dim"] = int(sample.comb.size(1))
    return d


if __name__ == "__main__":
    from ..data.targets import load_targets
    ts = load_targets()
    names = ("seg_center", "seg_direction", "seg_length", "width_height",
             "hydraulic_diameter", "dist_to_open", "tube_length")
    print(f"node feature dim = {node_feature_dim(list(names))}")
    for dev in ts.y:
        g = build(dev, feature_names=names, y=ts.y[dev], mask=ts.mask[dev])
        n = g[SEG].num_nodes
        npath = g[PATH].edge_index.size(1)
        nopen = g[OPEN].edge_index.size(1)
        nspat = g[SPATIAL].edge_index.size(1)
        # sanity: every non-degenerate segment's direction row is unit-norm
        dnorm = g[SEG].x[:, 3:6].norm(dim=1)
        ok = bool((dnorm[g[SEG].x[:, 6] > 1e-6] - 1.0).abs().max() < 1e-4)
        print(f"{dev:28s} nodes={n:4d} x={tuple(g[SEG].x.shape)} "
              f"path={npath:4d} open={nopen:4d} spatial={nspat:5d} dhat_unit={ok}")
