"""Extract per-tube folded centerlines from a STEP air-domain B-rep.

Strategy (robust for constant-cross-section, orthogonally-folded ducts):
  1. Load STEP; iterate the individual tube solids (one solid per resonator tube).
  2. Voxelize each solid via point-in-solid tests on a regular grid.
  3. BFS from one geometric end over the occupied-voxel graph; the centroid of each BFS
     "layer" (≈ a cross-sectional slice) traces the centerline, ordered end-to-end.
  4. Simplify the centerline polyline (Ramer–Douglas–Peucker); each retained vertex is a
     fold, so consecutive vertices become straight-segment nodes (k folds -> k+1 segments).
  5. Orient open->end and emit absolute coords in a canonical global frame.

Validated against ``tube_geometry.csv``: Σ‖segment‖ per tube must match the known length.

Requires OpenCASCADE bindings: ``cadquery-ocp`` (import ``OCP.*``) or ``pythonocc-core``
(import ``OCC.Core.*``) — both are tried.
"""
from __future__ import annotations

import heapq
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from .devices import TUBE_GEOMETRY_CSV, discover_devices
from .schema import SegmentRecord

# ---- OCCT bindings: try cadquery-ocp (OCP) then pythonocc-core (OCC.Core) ----
try:
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_IN
    from OCP.TopoDS import TopoDS
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.gp import gp_Pnt
    _OCP = True
except ImportError:  # pragma: no cover
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_IN
    from OCC.Core.TopoDS import topods as TopoDS
    from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib as BRepBndLib
    from OCC.Core.gp import gp_Pnt
    _OCP = False

VOXEL_MM = 1.0      # grid spacing; < wall (2mm) so parallel channels stay separated
RDP_TOL_MM = 2.0    # centerline simplification tolerance (legacy path)
MIN_RUN_MM = 3.0    # min straight-run extent to count as a real fold (hysteresis)


def _load_shape(path: Path):
    reader = STEPControl_Reader()
    reader.ReadFile(str(path))
    reader.TransferRoots()
    return reader.OneShape()


def _iter(shape, kind):
    exp = TopExp_Explorer(shape, kind)
    out = []
    while exp.More():
        out.append(exp.Current())
        exp.Next()
    return out


def _bbox(shape):
    box = Bnd_Box()
    (BRepBndLib.Add_s if _OCP else BRepBndLib.Add)(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return (xmin, ymin, zmin), (xmax, ymax, zmax)


def _as_solid(shp):
    return (TopoDS.Solid_s if _OCP else TopoDS.Solid)(shp)


def _occupancy(solid, h: float):
    """Return (coords dict idx->center, set of occupied idx) via point-in-solid tests."""
    (xmin, ymin, zmin), (xmax, ymax, zmax) = _bbox(solid)
    clf = BRepClass3d_SolidClassifier(solid)
    nx = max(1, int((xmax - xmin) / h) + 1)
    ny = max(1, int((ymax - ymin) / h) + 1)
    nz = max(1, int((zmax - zmin) / h) + 1)
    occ: set[tuple[int, int, int]] = set()
    centers: dict[tuple[int, int, int], np.ndarray] = {}
    for i in range(nx):
        x = xmin + (i + 0.5) * h
        for j in range(ny):
            y = ymin + (j + 0.5) * h
            for k in range(nz):
                z = zmin + (k + 0.5) * h
                clf.Perform(gp_Pnt(x, y, z), 1e-6)
                if clf.State() == TopAbs_IN:
                    occ.add((i, j, k))
                    centers[(i, j, k)] = np.array([x, y, z])
    return centers, occ


_NEIGH = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def _bfs_layers(occ: set, start: tuple) -> list[list[tuple]]:
    """BFS over 6-connected occupied voxels; return voxels grouped by distance from start."""
    dist = {start: 0}
    q = deque([start])
    while q:
        c = q.popleft()
        for d in _NEIGH:
            nb = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
            if nb in occ and nb not in dist:
                dist[nb] = dist[c] + 1
                q.append(nb)
    maxd = max(dist.values())
    layers: list[list[tuple]] = [[] for _ in range(maxd + 1)]
    for v, d in dist.items():
        layers[d].append(v)
    return layers


def _farthest(occ: set, start: tuple) -> tuple:
    layers = _bfs_layers(occ, start)
    return layers[-1][0]


def _smooth(P: np.ndarray, w: int = 3) -> np.ndarray:
    """Moving-average smoothing of a polyline (suppresses voxel jitter).

    Edge-padded so endpoints are preserved (a plain ``mode='same'`` convolution would pull
    the first/last points toward the origin).
    """
    if len(P) <= w:
        return P
    pad = w // 2
    Pp = np.pad(P, ((pad, pad), (0, 0)), mode="edge")
    k = np.ones(w) / w
    return np.stack([np.convolve(Pp[:, d], k, mode="valid") for d in range(3)], axis=1)


def _centerline(centers: dict, occ: set) -> np.ndarray:
    """Ordered polyline of layer centroids from one end to the other."""
    seed = next(iter(occ))
    endA = _farthest(occ, seed)          # double-sweep for a true graph-diameter endpoint
    _farthest(occ, endA)
    layers = _bfs_layers(occ, endA)
    pts = [np.mean([centers[v] for v in layer], axis=0) for layer in layers if layer]
    return _smooth(np.asarray(pts))


def _squared_polyline(P: np.ndarray) -> np.ndarray:
    """Reconstruct an axis-aligned (Manhattan) centerline from dense centroid points.

    The BFS-centroid path cuts corners at folds (underestimating length). These ducts are
    axis-aligned, so we (1) split the path into runs of constant signed direction, (2) fix
    each run's two transverse coords to the run mean (the true duct-center offsets), and
    (3) place each fold vertex at the intersection of consecutive runs' axes — recovering
    the true corner and hence the true length.
    """
    if len(P) < 3:
        return P
    sdir = []  # signed axis per step: (axis, sign)
    for j in range(len(P) - 1):
        d = P[j + 1] - P[j]
        a = int(np.argmax(np.abs(d)))
        sdir.append((a, 1.0 if d[a] >= 0 else -1.0))

    # Hysteresis run segmentation: only commit a direction change once the CURRENT run has
    # travelled >= MIN_RUN_MM along its axis. This keeps real transverse shifts between
    # serpentine arms (~ side+wall) while ignoring sub-wall voxel jitter.
    runs = []  # (axis, start_idx, end_idx) inclusive
    s = 0
    cur = sdir[0]
    for j in range(1, len(sdir)):
        if sdir[j] != cur:
            extent = abs(P[j][cur[0]] - P[s][cur[0]])
            if extent >= MIN_RUN_MM:
                runs.append((cur[0], s, j)); s = j; cur = sdir[j]
            # else: treat as jitter, stay on current run
    runs.append((cur[0], s, len(P) - 1))

    info = [{"axis": ax, "si": si, "ei": ei, "c": P[si:ei + 1].mean(axis=0)}
            for ax, si, ei in runs]

    verts = []
    r0 = info[0]
    start = r0["c"].copy(); start[r0["axis"]] = P[r0["si"]][r0["axis"]]
    verts.append(start)
    for a, b in zip(info, info[1:]):
        aa, ba = a["axis"], b["axis"]
        v = np.zeros(3)
        if aa != ba:
            third = ({0, 1, 2} - {aa, ba}).pop()
            v[aa] = b["c"][aa]           # along a's axis: where run b sits
            v[ba] = a["c"][ba]           # along b's axis: where run a sits
            v[third] = 0.5 * (a["c"][third] + b["c"][third])
        else:  # rare same-axis reversal (U-turn): fall back to midpoint
            v = 0.5 * (P[a["ei"]] + P[b["si"]])
        verts.append(v)
    rl = info[-1]
    end = rl["c"].copy(); end[rl["axis"]] = P[rl["ei"]][rl["axis"]]
    verts.append(end)
    return np.asarray(verts)


def _rdp(points: np.ndarray, eps: float) -> np.ndarray:
    """Ramer–Douglas–Peucker simplification of a 3-D polyline."""
    if len(points) < 3:
        return points
    a, b = points[0], points[-1]
    ab = b - a
    nab = np.linalg.norm(ab)
    if nab < 1e-9:
        d = np.linalg.norm(points - a, axis=1)
    else:
        d = np.linalg.norm(np.cross(points - a, ab), axis=1) / nab
    i = int(np.argmax(d))
    if d[i] > eps:
        left = _rdp(points[: i + 1], eps)
        right = _rdp(points[i:], eps)
        return np.vstack([left[:-1], right])
    return np.vstack([a, b])


def _orient_open_to_end(poly: np.ndarray) -> np.ndarray:
    """Heuristic: open/inlet end is the lower-z endpoint. Flip so p0=open."""
    return poly if poly[0][2] <= poly[-1][2] else poly[::-1]


def _tube_records(device: str, tube_index: int, poly: np.ndarray, cs: float,
                  origin: np.ndarray) -> list[SegmentRecord]:
    poly = _orient_open_to_end(poly) - origin  # canonical frame
    recs = []
    for s in range(len(poly) - 1):
        recs.append(SegmentRecord(
            device=device, tube_index=tube_index, segment_index=s,
            p0=tuple(map(float, poly[s])), p1=tuple(map(float, poly[s + 1])),
            cross_section_mm2=cs,
        ))
    return recs


# --------------------------------------------------------------------------- #
# Skeleton centerline (robust for folded-back serpentines where shell-centroid
# collapses). Thin the occupancy to a 1-voxel curve, trace it end to end, and
# extend the terminals to the physical caps.
# --------------------------------------------------------------------------- #
_NEIGH26 = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
            if (i, j, k) != (0, 0, 0)]


def _skeletonize(arr: np.ndarray) -> np.ndarray:
    try:
        from skimage.morphology import skeletonize_3d as _skel
    except ImportError:  # newer skimage folds 3-D into skeletonize
        from skimage.morphology import skeletonize as _skel
    return _skel(arr) > 0


def _dijkstra(svset: set, src: tuple):
    """Weighted shortest paths over the 26-connected skeleton (Euclidean step costs)."""
    dist = {src: 0.0}
    parent = {src: None}
    pq = [(0.0, src)]
    while pq:
        dcur, c = heapq.heappop(pq)
        if dcur > dist.get(c, 1e18):
            continue
        for d in _NEIGH26:
            nb = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
            if nb in svset:
                nd = dcur + float(np.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]))
                if nd < dist.get(nb, 1e18):
                    dist[nb] = nd
                    parent[nb] = c
                    heapq.heappush(pq, (nd, nb))
    return dist, parent


def _skeleton_polyline(centers: dict, occ: set) -> np.ndarray:
    """Ordered centerline polyline (mm) from the 3-D skeleton of the occupancy."""
    keys = np.array(list(occ))
    mn = keys.min(axis=0)
    arr = np.zeros(tuple(keys.max(axis=0) - mn + 1), dtype=bool)
    shifted = keys - mn
    arr[shifted[:, 0], shifted[:, 1], shifted[:, 2]] = True

    sv = [tuple(p) for p in np.argwhere(_skeletonize(arr))]
    if len(sv) < 2:  # degenerate: fall back to occupancy centroid
        return np.mean([centers[tuple(k)] for k in occ], axis=0)[None, :]
    svset = set(sv)

    # two graph-diameter endpoints (double sweep), then trace a -> b via parents
    d0, _ = _dijkstra(svset, sv[0])
    a = max(d0, key=d0.get)
    d1, parent = _dijkstra(svset, a)
    b = max(d1, key=d1.get)
    path = []
    cur = b
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()

    mn = tuple(int(x) for x in mn)
    return np.array([centers[(v[0] + mn[0], v[1] + mn[1], v[2] + mn[2])] for v in path])


def _extend_ends(poly: np.ndarray, solid, h: float, max_ext: float = 4.0) -> np.ndarray:
    """March each terminal outward (along its local direction) to the solid boundary."""
    if len(poly) < 2:
        return poly
    clf = BRepClass3d_SolidClassifier(solid)

    def inside(p) -> bool:
        clf.Perform(gp_Pnt(float(p[0]), float(p[1]), float(p[2])), 1e-6)
        return clf.State() == TopAbs_IN

    def march(p0, direction):
        step = 0.25 * h
        last = p0.copy()
        p = p0.copy()
        travelled = 0.0
        while travelled < max_ext:
            p = p + direction * step
            if not inside(p):
                break
            last = p.copy()
            travelled += step
        return last

    out = poly.copy()
    d_start = poly[0] - poly[1]
    d_end = poly[-1] - poly[-2]
    if np.linalg.norm(d_start) > 1e-9:
        out[0] = march(poly[0], d_start / np.linalg.norm(d_start))
    if np.linalg.norm(d_end) > 1e-9:
        out[-1] = march(poly[-1], d_end / np.linalg.norm(d_end))
    return out


def _polyline_len(p: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def _candidates(centers, occ, solid, voxel_mm) -> list[np.ndarray]:
    """All plausible centerline reconstructions from the occupancy.

    Two centerline families cover the two tube geometries; each is offered both as a mildly
    simplified raw path and as a sharp-corner (``_squared_polyline``) reconstruction:
      * **skeleton** (3-D medial axis) — robust for tight folded serpentines (Ver2), but
        ``skeletonize_3d`` collapses short/thick ducts to a point;
      * **BFS shell-centroid** — robust for simple/straight ducts (Ver1R), collapses on
        serpentines.
    """
    out: list[np.ndarray] = []
    sk = _extend_ends(_skeleton_polyline(centers, occ), solid, voxel_mm)
    if len(sk) >= 2:
        out.append(_rdp(sk, 1.0))
        out.append(_squared_polyline(sk))
    cl = _centerline(centers, occ)
    if len(cl) >= 2:
        out.append(_rdp(cl, 1.0))
        out.append(_squared_polyline(cl))
    return [c for c in out if len(c) >= 2]


def _solid_polyline(solid, voxel_mm: float, known: float | None = None) -> np.ndarray | None:
    """Trace one tube's folded centerline (absolute mm).

    Build every candidate reconstruction (see ``_candidates``) and keep the one whose length
    best matches the known design length (``tube_geometry.csv``); with no known length, keep
    the longest (all candidates under-estimate, never over-, so longest is closest to truth).
    """
    centers, occ = _occupancy(solid, voxel_mm)
    if len(occ) < 2:
        return None
    cands = _candidates(centers, occ, solid, voxel_mm)
    if not cands:
        return None
    if known is not None and known > 0:
        return min(cands, key=lambda p: abs(_polyline_len(p) - known))
    return max(cands, key=_polyline_len)


def _worker(args):
    """Subprocess: re-read the STEP (OCC objects can't be pickled) and trace a slice of solids."""
    step_path, indices, voxel_mm, known = args
    shape = _load_shape(Path(step_path))
    solids = _iter(shape, TopAbs_SOLID)
    return [(i, _solid_polyline(_as_solid(solids[i]), voxel_mm, known[i])) for i in indices]


def extract_segments(device: str, voxel_mm: float = VOXEL_MM,
                     validate: bool = True, parallel: bool = True) -> list[SegmentRecord]:
    dev = discover_devices()[device]
    shape = _load_shape(dev.step)
    solids = _iter(shape, TopAbs_SOLID)
    if not solids:
        raise RuntimeError(f"{device}: no solids in STEP (inspect with step_inspect)")
    n = len(solids)

    origin = np.asarray(_bbox(shape)[0])  # canonical origin = global bbox min corner

    # known per-tube cross-section (for the feature) + length (for validation)
    geo = pd.read_csv(TUBE_GEOMETRY_CSV)
    geo = geo[geo["device"] == device].sort_values("tube_index").reset_index(drop=True)

    # known design length per solid index (for per-tube method selection), aligned to ti+1
    known = [float(geo.loc[i, "length_mm"]) if i < len(geo) else None for i in range(n)]

    # trace every solid's centerline (parallel across CPU cores; solids are independent)
    polys: dict[int, np.ndarray | None] = {}
    if parallel and n > 1:
        import multiprocessing as mp
        nproc = min(mp.cpu_count(), n)
        chunks = [list(range(k, n, nproc)) for k in range(nproc)]  # strided load balance
        args = [(str(dev.step), ch, voxel_mm, known) for ch in chunks if ch]
        with mp.Pool(nproc) as pool:
            for res in pool.map(_worker, args):
                polys.update(dict(res))
    else:
        for i in range(n):
            polys[i] = _solid_polyline(_as_solid(solids[i]), voxel_mm, known[i])

    records: list[SegmentRecord] = []
    for ti in range(n):
        poly = polys.get(ti)
        if poly is None or len(poly) < 2:
            continue
        cs = float(geo.loc[ti, "cross_section_mm2"]) if ti < len(geo) else 20.25
        records.extend(_tube_records(device, ti + 1, poly, cs, origin))

    if validate and len(geo):
        _validate_lengths(device, records, geo)
    return records


def _validate_lengths(device, records, geo, rtol: float = 0.05):
    by_tube: dict[int, float] = {}
    for r in records:
        by_tube[r.tube_index] = by_tube.get(r.tube_index, 0.0) + r.length
    known = dict(zip(geo["tube_index"], geo["length_mm"]))
    bad = []
    for t, L in by_tube.items():
        k = known.get(t)
        if k and abs(L - k) > rtol * k:
            bad.append((t, L, k))
    if bad:
        worst = max(bad, key=lambda x: abs(x[1] - x[2]) / x[2])
        print(f"[warn] {device}: {len(bad)}/{len(by_tube)} tubes exceed {rtol:.0%} length "
              f"error (worst tube {worst[0]}: extracted {worst[1]:.1f} vs known {worst[2]:.1f} mm)")


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else next(iter(discover_devices()))
    recs = extract_segments(name)
    ntubes = len({r.tube_index for r in recs})
    print(f"{name}: {ntubes} tubes, {len(recs)} segments, "
          f"total_len={sum(r.length for r in recs):.1f} mm")
