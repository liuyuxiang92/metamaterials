"""Get a device's folded geometry as ``list[SegmentRecord]`` from a chosen source.

Sources
-------
``step``   : real extractor (``step_extract``) — parse the STEP B-rep into per-tube,
             per-segment absolute endpoints. The only physically-correct source.
``scalar`` : smoke-test fallback built from ``gnn/tube_geometry.csv`` — one segment per
             tube, tubes laid on a synthetic grid with a vertical extent equal to the known
             length. Not physical; exists only so the graph/model/training loop is runnable
             before STEP extraction lands. Its per-tube length equals the ground-truth length,
             so it also exercises the length-validation path.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .devices import TUBE_GEOMETRY_CSV
from .schema import SegmentRecord

_PARTITION_MM = 2.0  # inner partition thickness from the design sheet


def _scalar_segments(device: str) -> list[SegmentRecord]:
    df = pd.read_csv(TUBE_GEOMETRY_CSV)
    df = df[df["device"] == device].sort_values("tube_index")
    if df.empty:
        raise ValueError(f"No rows for device {device!r} in {TUBE_GEOMETRY_CSV}")

    n = len(df)
    ncols = max(1, math.ceil(math.sqrt(n)))
    recs: list[SegmentRecord] = []
    for k, (_, row) in enumerate(df.iterrows()):
        cs = float(row["cross_section_mm2"])
        length = float(row["length_mm"])
        side = math.sqrt(cs)
        spacing = side + _PARTITION_MM
        col, r = k % ncols, k // ncols
        base = np.array([col * spacing, r * spacing, 0.0])
        p0 = base
        p1 = base + np.array([0.0, 0.0, length])  # vertical segment of the right length
        recs.append(
            SegmentRecord(
                device=device,
                tube_index=int(row["tube_index"]),
                segment_index=0,
                p0=tuple(p0), p1=tuple(p1),
                cross_section_mm2=cs,
                width_mm=side, thickness_mm=side,   # square-duct stand-in for the smoke path
            )
        )
    return recs


def _cache_path(device: str, source: str = "brep") -> "Path":
    from pathlib import Path
    from .devices import GNN_DIR
    d = GNN_DIR / "results" / "cache"
    d.mkdir(parents=True, exist_ok=True)
    suffix = "" if source == "brep" else f"_{source}"   # brep is the primary cache
    return d / f"{device}_segments{suffix}.csv"


def _save_cache(device: str, segs: list[SegmentRecord], source: str) -> None:
    rows = [
        {"tube_index": s.tube_index, "segment_index": s.segment_index,
         "x0": s.p0[0], "y0": s.p0[1], "z0": s.p0[2],
         "x1": s.p1[0], "y1": s.p1[1], "z1": s.p1[2],
         "cross_section_mm2": s.cross_section_mm2,
         "width_mm": s.width_mm, "thickness_mm": s.thickness_mm}
        for s in segs
    ]
    pd.DataFrame(rows).to_csv(_cache_path(device, source), index=False)


def _load_cache(device: str, source: str) -> list[SegmentRecord]:
    df = pd.read_csv(_cache_path(device, source))
    has_wh = "width_mm" in df.columns and "thickness_mm" in df.columns   # tolerate old caches
    out: list[SegmentRecord] = []
    for r in df.itertuples():
        w = getattr(r, "width_mm", None) if has_wh else None
        h = getattr(r, "thickness_mm", None) if has_wh else None
        out.append(SegmentRecord(
            device=device, tube_index=int(r.tube_index),
            segment_index=int(r.segment_index),
            p0=(r.x0, r.y0, r.z0), p1=(r.x1, r.y1, r.z1),
            cross_section_mm2=float(r.cross_section_mm2),
            width_mm=None if w is None or pd.isna(w) else float(w),
            thickness_mm=None if h is None or pd.isna(h) else float(h),
        ))
    return out


def get_segments(device: str, source: str = "brep", use_cache: bool = True) -> list[SegmentRecord]:
    """Folded-tube segments. Sources:
      ``brep``   : exact B-rep centerline (wall-averaging; default, fast, no voxels).
      ``step``   : legacy voxel/skeleton centerline (approximate; kept for comparison).
      ``scalar`` : synthetic smoke-test stand-in.
    """
    if source == "scalar":
        return _scalar_segments(device)
    if source in ("brep", "step"):
        if use_cache and _cache_path(device, source).exists():
            return _load_cache(device, source)
        if source == "brep":
            from .brep_extract import extract_segments
        else:
            from .step_extract import extract_segments  # legacy voxel path
        segs = extract_segments(device)
        _save_cache(device, segs, source)
        return segs
    raise ValueError(f"Unknown fold-coord source {source!r} (use 'brep', 'step', or 'scalar')")


if __name__ == "__main__":
    from .devices import discover_devices
    for name in discover_devices():
        segs = get_segments(name, source="scalar")
        tot = sum(s.length for s in segs)
        print(f"{name:28s} tubes={len(segs):3d}  total_len={tot:8.1f} mm")
