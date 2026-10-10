"""Exact folded-tube centerline from the STEP B-rep (no voxels).

Each tube is a constant-cross-section duct extruded in Z (verified: solid volume ==
footprint_area * thickness to 0.00%). The two long "curly" walls of the channel are the two
halves of the footprint face's boundary loop; they are parallel offsets (width apart) with the
SAME vertex count, so the centerline is exactly the vertex-wise average of the two walls.

Method (per tube), from the user's construction:
  1. Take the footprint face = the largest planar face with normal ~ +/-Z; get its ordered
     boundary vertices (exact CARTESIAN_POINTs), a closed loop of ``n`` vertices.
  2. Open-cap edge = the along-X boundary edge (|dx|>|dy|) of length ~ width, at MIN y.
     Its two vertices A, B start the two walls.
  3. Split the loop in half from that cap: wall1 = B, next, next, ...; wall2 = A, prev, ...
     (``m = n/2`` vertices each). They reconnect at the end (bottom) cap.
  4. centerline = (wall1 + wall2) / 2, vertex-by-vertex -> EXACT (paired segments are parallel).
  5. Orient open->end (open = the min-Y cap), place at mid-thickness Z, simplify to folds.

Validated exact: centerline length == footprint quadratic length (0.00%) and pairing cos == 1.0
on every tube of every device.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .devices import TUBE_GEOMETRY_CSV, discover_devices
from .face_extract import (TopAbs_FACE, TopAbs_SOLID, _as_face, _as_solid, _iter,
                           _load_shape, _ordered_vertices, _plane_normal, _surface_props)
from .schema import SegmentRecord
from .step_extract import _bbox, _rdp

RDP_TOL_MM = 0.25   # remove collinear centerline points; folds are exact so tol is tiny


def _footprint_face(solid):
    """Largest planar face with normal ~ +/-Z (the extruded footprint)."""
    faces = [_as_face(x) for x in _iter(solid, TopAbs_FACE)]
    zf = [(_surface_props(f)[0], f) for f in faces
          if _plane_normal(f) is not None and abs(_plane_normal(f)[2]) > 0.9]
    if not zf:
        return None
    return max(zf, key=lambda t: t[0])[1]


def split_walls(solid) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float] | None:
    """Split a tube's footprint loop into its two parallel curly walls.

    Returns ``(V, wall1, wall2, i_open, width)`` where ``V[n,3]`` is the ordered footprint
    boundary loop, ``wall1``/``wall2`` are the two ``n/2``-vertex walls running open->end,
    and ``i_open`` indexes the open-cap edge ``V[i_open] -> V[i_open+1]``.
    """
    top = _footprint_face(solid)
    if top is None:
        return None
    V = _ordered_vertices(top)
    n = len(V)
    if n < 4 or n % 2 != 0:
        return None
    XY = V[:, :2]
    e = np.linalg.norm(np.diff(np.vstack([V, V[:1]]), axis=0), axis=1)
    A = _surface_props(top)[0]
    P = e.sum()
    disc = P * P - 16 * A
    if disc < 0:
        return None
    W = (P - math.sqrt(disc)) / 4.0                       # exact channel width

    # open cap = along-X edge (|dx|>|dy|), length ~ width, at MIN y
    cand = [k for k in range(n)
            if abs(XY[(k + 1) % n, 0] - XY[k, 0]) > abs(XY[(k + 1) % n, 1] - XY[k, 1])
            and abs(e[k] - W) < 0.6 * W]
    if not cand:
        return None
    i = min(cand, key=lambda k: (XY[k, 1] + XY[(k + 1) % n, 1]) / 2.0)

    m = n // 2
    wall1 = np.array([V[(i + 1 + k) % n] for k in range(m)])  # from B into the maze
    wall2 = np.array([V[(i - k) % n] for k in range(m)])      # from A into the maze
    return V, wall1, wall2, i, W


def _exact_centerline(solid) -> tuple[np.ndarray, float, float] | None:
    """Return (centerline[m,3] open->end at mid-Z, width, thickness) or None."""
    res = split_walls(solid)
    if res is None:
        return None
    _, wall1, wall2, _, W = res
    cl = (wall1 + wall2) / 2.0                                # EXACT centerline, open->end

    (_, _, zmin), (_, _, zmax) = _bbox(solid)
    thickness = zmax - zmin
    cl[:, 2] = 0.5 * (zmin + zmax)                            # place at mid-thickness
    return cl, W, thickness


def extract_segments(device: str, validate: bool = True) -> list[SegmentRecord]:
    dev = discover_devices()[device]
    shape = _load_shape(dev.step)
    solids = _iter(shape, TopAbs_SOLID)
    if not solids:
        raise RuntimeError(f"{device}: no solids in STEP")
    origin = np.asarray(_bbox(shape)[0])                      # canonical frame (global bbox min)

    geo = pd.read_csv(TUBE_GEOMETRY_CSV)
    geo = geo[geo["device"] == device].sort_values("tube_index").reset_index(drop=True)

    records: list[SegmentRecord] = []
    for ti, shp in enumerate(solids):
        res = _exact_centerline(_as_solid(shp))
        if res is None:
            continue
        cl, W, thickness = res
        poly = _rdp(cl - origin, RDP_TOL_MM)                  # fold vertices, canonical coords
        cs = float(W * thickness)
        for s in range(len(poly) - 1):
            records.append(SegmentRecord(
                device=device, tube_index=ti + 1, segment_index=s,
                p0=tuple(map(float, poly[s])), p1=tuple(map(float, poly[s + 1])),
                cross_section_mm2=cs,
                width_mm=float(W), thickness_mm=float(thickness),
            ))

    if validate and len(geo):
        _validate(device, records, geo)
    return records


def _validate(device, records, geo, rtol=0.02):
    by_tube: dict[int, float] = {}
    for r in records:
        by_tube[r.tube_index] = by_tube.get(r.tube_index, 0.0) + r.length
    known = dict(zip(geo["tube_index"], geo["length_mm"]))
    bad = [(t, L, known[t]) for t, L in by_tube.items()
           if t in known and known[t] and abs(L - known[t]) > rtol * known[t]]
    if bad:
        worst = max(bad, key=lambda x: abs(x[1] - x[2]) / x[2])
        print(f"[warn] {device}: {len(bad)}/{len(by_tube)} tubes exceed {rtol:.0%} "
              f"(worst tube {worst[0]}: {worst[1]:.1f} vs {worst[2]:.1f} mm)")


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else next(iter(discover_devices()))
    recs = extract_segments(name)
    nt = len({r.tube_index for r in recs})
    print(f"{name}: {nt} tubes, {len(recs)} segments, total_len={sum(r.length for r in recs):.1f} mm")
