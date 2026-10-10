"""Extract planar surfaces + facing-surface couplings from a STEP air-domain B-rep.

Method #2 (two-graph GNN) needs, per device:
  * ``SurfaceRecord`` — one per planar B-rep face of a tube's air domain (Graph-2 node),
    carrying ordered boundary vertices, exact area/centroid (mass properties), unit outward
    normal, perimeter and bbox extents.
  * ``CouplingRecord`` — an edge between two surfaces that are parallel, anti-parallel-facing,
    separated only by a thin partition wall ("no air between them"), and actually overlapping.
    Couplings are INTRA-tube (same folded tube's adjacent arms) or INTER-tube (different tubes).

Coupling test (per candidate pair):
  1. anti-parallel normals            n_a · n_b <= NORMAL_DOT_TOL
  2. thin gap along the normal        |(c_b - c_a) · n_a| <= MAX_GAP_MM   (~partition thickness)
  3. no air between them              midpoint of the two centroids is OUTSIDE the air solid(s)
                                      -> in partition material, not the duct lumen
  4. real facing overlap             projected-polygon overlap area > MIN_OVERLAP_MM2 (shapely)

Coordinates are ABSOLUTE, in the same canonical global frame as ``step_extract`` (origin =
global bbox min corner), so surfaces and segments share one frame.

Requires OpenCASCADE bindings: ``cadquery-ocp`` (``OCP.*``) or ``pythonocc-core`` (``OCC.Core.*``).
Face extraction is heavy (loads the STEP, iterates every face); results are cached like
``fold_coords``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

from .devices import GNN_DIR, discover_devices
from .schema import CouplingRecord, SurfaceRecord

# ---- OCCT bindings: try cadquery-ocp (OCP) then pythonocc-core (OCC.Core) ----
try:
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_REVERSED, TopAbs_OUT
    from OCP.TopoDS import TopoDS
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Plane
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
    from OCP.BRep import BRep_Tool
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.gp import gp_Pnt
    _OCP = True
except ImportError:  # pragma: no cover
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_REVERSED, TopAbs_OUT
    from OCC.Core.TopoDS import topods as TopoDS
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_Plane
    from OCC.Core.BRepGProp import brepgprop as BRepGProp
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepTools import BRepTools_WireExplorer, breptools as BRepTools
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib as BRepBndLib
    from OCC.Core.gp import gp_Pnt
    _OCP = False

MIN_FACE_AREA_MM2 = 2.0    # drop tiny fillet/connector faces
NORMAL_DOT_TOL = -0.98     # n_a . n_b <= this  => anti-parallel (facing)
MAX_GAP_MM = 4.0           # gap along normal must be ~partition thickness (< duct width)
MIN_OVERLAP_MM2 = 1.0      # min projected overlap area to count as a real coupling


# --------------------------------------------------------------------------- #
# OCCT helpers (mirror step_extract's shim so the two stay import-compatible)
# --------------------------------------------------------------------------- #
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


def _bbox_min(shape) -> np.ndarray:
    box = Bnd_Box()
    (BRepBndLib.Add_s if _OCP else BRepBndLib.Add)(shape, box)
    xmin, ymin, zmin, _, _, _ = box.Get()
    return np.array([xmin, ymin, zmin])


def _face_bbox_extents(face) -> np.ndarray:
    box = Bnd_Box()
    (BRepBndLib.Add_s if _OCP else BRepBndLib.Add)(face, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return np.array([xmax - xmin, ymax - ymin, zmax - zmin])


def _as_solid(shp):
    return (TopoDS.Solid_s if _OCP else TopoDS.Solid)(shp)


def _as_face(shp):
    return (TopoDS.Face_s if _OCP else TopoDS.Face)(shp)


def _surface_props(face) -> tuple[float, np.ndarray]:
    props = GProp_GProps()
    (BRepGProp.SurfaceProperties_s if _OCP else BRepGProp.SurfaceProperties)(face, props)
    c = props.CentreOfMass()
    return float(props.Mass()), np.array([c.X(), c.Y(), c.Z()])


def _plane_normal(face) -> np.ndarray | None:
    surf = BRepAdaptor_Surface(face)
    if surf.GetType() != GeomAbs_Plane:
        return None
    d = surf.Plane().Axis().Direction()
    n = np.array([d.X(), d.Y(), d.Z()], dtype=np.float64)
    if face.Orientation() == TopAbs_REVERSED:   # orient outward w.r.t. the solid
        n = -n
    nn = np.linalg.norm(n)
    return n / nn if nn > 1e-12 else None


def _ordered_vertices(face) -> np.ndarray:
    """Ordered boundary-wire vertices of the outer wire (V, 3)."""
    wire = (BRepTools.OuterWire_s if _OCP else BRepTools.OuterWire)(face)
    we = BRepTools_WireExplorer(wire)
    pts = []
    while we.More():
        p = (BRep_Tool.Pnt_s if _OCP else BRep_Tool.Pnt)(we.CurrentVertex())
        pts.append([p.X(), p.Y(), p.Z()])
        we.Next()
    return np.asarray(pts, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Geometry: 2-D projection + overlap area
# --------------------------------------------------------------------------- #
def _plane_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal in-plane axes (u, v) spanning the plane with normal n."""
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def _overlap_area(va: np.ndarray, vb: np.ndarray, n: np.ndarray) -> float:
    """Exact area of the two faces' overlap, projected onto their shared plane."""
    u, w = _plane_basis(n)
    pa = np.column_stack([va @ u, va @ w])
    pb = np.column_stack([vb @ u, vb @ w])
    try:
        A = Polygon(pa); B = Polygon(pb)
        if not A.is_valid:
            A = A.buffer(0)
        if not B.is_valid:
            B = B.buffer(0)
        return float(A.intersection(B).area)
    except Exception:
        return 0.0


def _perimeter(verts: np.ndarray) -> float:
    if len(verts) < 2:
        return 0.0
    d = verts - np.roll(verts, -1, axis=0)          # closed loop
    return float(np.linalg.norm(d, axis=1).sum())


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _extract_surfaces(shape, origin: np.ndarray, device: str):
    """Return (surfaces, tube_of, solids) — one SurfaceRecord per planar face."""
    solids = [_as_solid(s) for s in _iter(shape, TopAbs_SOLID)]
    surfaces: list[SurfaceRecord] = []
    tube_of: list[int] = []
    for ti, solid in enumerate(solids, start=1):
        for f in _iter(solid, TopAbs_FACE):
            face = _as_face(f)
            n = _plane_normal(face)
            if n is None:
                continue
            area, centroid = _surface_props(face)
            if area < MIN_FACE_AREA_MM2:
                continue
            verts = _ordered_vertices(face)
            if len(verts) < 3:
                continue
            surfaces.append(SurfaceRecord(
                device=device, tube_index=ti, surface_index=len(surfaces),
                vertices=tuple(map(tuple, verts - origin)),
                area_mm2=area, normal=tuple(n),
                centroid=tuple(centroid - origin),
                perimeter_mm=_perimeter(verts),
                bbox_extents=tuple(_face_bbox_extents(face)),
            ))
            tube_of.append(ti)
    return surfaces, tube_of, solids


def _detect_couplings(device, surfaces, tube_of, solids, origin) -> list[CouplingRecord]:
    F = len(surfaces)
    if F < 2:
        return []
    C = np.array([s.centroid for s in surfaces])       # canonical-frame centroids
    N = np.array([s.normal for s in surfaces])

    # solid classifiers keyed by tube index (lazily reused)
    clf = {ti: BRepClass3d_SolidClassifier(sol) for ti, sol in enumerate(solids, start=1)}

    def outside(ti: int, p_abs: np.ndarray) -> bool:
        c = clf[ti]
        c.Perform(gp_Pnt(float(p_abs[0]), float(p_abs[1]), float(p_abs[2])), 1e-6)
        return c.State() == TopAbs_OUT

    # KD-tree prune: only pairs whose centroids are within a generous radius are considered.
    diag = np.array([np.linalg.norm(s.bbox_extents) for s in surfaces])
    radius = float(diag.max() + MAX_GAP_MM)
    tree = cKDTree(C)

    couplings: list[CouplingRecord] = []
    seen: set[tuple[int, int]] = set()
    for i in range(F):
        for j in tree.query_ball_point(C[i], radius):
            if j <= i:
                continue
            key = (i, j)
            if key in seen:
                continue
            if float(N[i] @ N[j]) > NORMAL_DOT_TOL:       # (1) not anti-parallel
                continue
            off = C[j] - C[i]
            gap = abs(float(off @ N[i]))
            if gap > MAX_GAP_MM:                          # (2) too thick / lumen walls
                continue
            # surfaces store canonical (origin-subtracted) coords; the classifier lives on the
            # original solids, so shift the midpoint back into the absolute STEP frame.
            m_abs = 0.5 * (C[i] + C[j]) + origin
            ti, tj = tube_of[i], tube_of[j]
            if not outside(ti, m_abs):                    # (3) no air between them
                continue
            if tj != ti and not outside(tj, m_abs):
                continue
            ov = _overlap_area(surfaces[i].vertices_arr, surfaces[j].vertices_arr, N[i])
            if ov < MIN_OVERLAP_MM2:                      # (4) real facing overlap
                continue
            seen.add(key)
            couplings.append(CouplingRecord(
                device=device, surf_a=i, surf_b=j,
                gap_vec=tuple(off), gap_dist=gap, overlap_area_mm2=ov,
                coupling_type=0 if ti == tj else 1,
            ))
    return couplings


def extract_faces(device: str) -> tuple[list[SurfaceRecord], list[CouplingRecord]]:
    dev = discover_devices()[device]
    shape = _load_shape(dev.step)
    origin = _bbox_min(shape)
    surfaces, tube_of, solids = _extract_surfaces(shape, origin, device)
    couplings = _detect_couplings(device, surfaces, tube_of, solids, origin)
    return surfaces, couplings


# --------------------------------------------------------------------------- #
# Caching (mirrors fold_coords)
# --------------------------------------------------------------------------- #
def _cache_dir() -> Path:
    d = GNN_DIR / "results" / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _surf_path(device: str) -> Path:
    return _cache_dir() / f"{device}_surfaces.csv"


def _coup_path(device: str) -> Path:
    return _cache_dir() / f"{device}_couplings.csv"


def _save_cache(device, surfaces, couplings) -> None:
    srows = [{
        "surface_index": s.surface_index, "tube_index": s.tube_index,
        "vertices_json": json.dumps([list(v) for v in s.vertices]),
        "area_mm2": s.area_mm2,
        "nx": s.normal[0], "ny": s.normal[1], "nz": s.normal[2],
        "cx": s.centroid[0], "cy": s.centroid[1], "cz": s.centroid[2],
        "perimeter_mm": s.perimeter_mm,
        "ex": s.bbox_extents[0], "ey": s.bbox_extents[1], "ez": s.bbox_extents[2],
    } for s in surfaces]
    pd.DataFrame(srows).to_csv(_surf_path(device), index=False)

    crows = [{
        "surf_a": c.surf_a, "surf_b": c.surf_b,
        "gx": c.gap_vec[0], "gy": c.gap_vec[1], "gz": c.gap_vec[2],
        "gap_dist": c.gap_dist, "overlap_area_mm2": c.overlap_area_mm2,
        "coupling_type": c.coupling_type,
    } for c in couplings]
    pd.DataFrame(crows, columns=["surf_a", "surf_b", "gx", "gy", "gz",
                                 "gap_dist", "overlap_area_mm2", "coupling_type"]
                 ).to_csv(_coup_path(device), index=False)


def _load_cache(device) -> tuple[list[SurfaceRecord], list[CouplingRecord]]:
    sdf = pd.read_csv(_surf_path(device))
    surfaces = [SurfaceRecord(
        device=device, tube_index=int(r.tube_index), surface_index=int(r.surface_index),
        vertices=tuple(tuple(v) for v in json.loads(r.vertices_json)),
        area_mm2=float(r.area_mm2), normal=(r.nx, r.ny, r.nz),
        centroid=(r.cx, r.cy, r.cz), perimeter_mm=float(r.perimeter_mm),
        bbox_extents=(r.ex, r.ey, r.ez),
    ) for r in sdf.itertuples()]
    cdf = pd.read_csv(_coup_path(device))
    couplings = [CouplingRecord(
        device=device, surf_a=int(r.surf_a), surf_b=int(r.surf_b),
        gap_vec=(r.gx, r.gy, r.gz), gap_dist=float(r.gap_dist),
        overlap_area_mm2=float(r.overlap_area_mm2), coupling_type=int(r.coupling_type),
    ) for r in cdf.itertuples()]
    return surfaces, couplings


def get_faces(device: str, use_cache: bool = True
              ) -> tuple[list[SurfaceRecord], list[CouplingRecord]]:
    if use_cache and _surf_path(device).exists() and _coup_path(device).exists():
        return _load_cache(device)
    surfaces, couplings = extract_faces(device)
    _save_cache(device, surfaces, couplings)
    return surfaces, couplings


def _report(device, surfaces, couplings) -> None:
    intra = sum(c.coupling_type == 0 for c in couplings)
    inter = sum(c.coupling_type == 1 for c in couplings)
    gaps = np.array([c.gap_dist for c in couplings]) if couplings else np.zeros(0)
    ov = np.array([c.overlap_area_mm2 for c in couplings]) if couplings else np.zeros(0)
    ntubes = len({s.tube_index for s in surfaces})
    print(f"{device:28s} surfaces={len(surfaces):4d} tubes={ntubes:3d} "
          f"couplings={len(couplings):4d} (intra={intra} inter={inter})")
    if len(gaps):
        print(f"    gap_dist[min,med,max]=[{gaps.min():.2f},{np.median(gaps):.2f},{gaps.max():.2f}] mm"
              f"   overlap[min,med,max]=[{ov.min():.1f},{np.median(ov):.1f},{ov.max():.1f}] mm^2")


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else next(iter(discover_devices()))
    surfaces, couplings = get_faces(name, use_cache=False)
    _report(name, surfaces, couplings)
