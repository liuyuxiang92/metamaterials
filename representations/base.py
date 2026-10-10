"""Node-feature registry shared by graph representations.

Each feature is a callable ``fn(seg, ctx) -> list[float]``. ``ctx`` carries per-node
context needed by some features:
  * ``ctx["prev"]``     : previous ``SegmentRecord`` in the same tube (or ``None`` at the open end)
  * ``ctx["cum_len"]``  : centerline distance from the open end to this segment's start (mm)
  * ``ctx["tube_length"]`` : total centerline length of this segment's tube (mm)

Selecting features in a config (e.g. ``["abs_endpoints", "cross_section"]``) is all it takes
to change the node vector — adding ``position_along_tube`` / ``turn_angle`` later is one edit.
"""
from __future__ import annotations

import numpy as np

from ..data.schema import SegmentRecord


def _abs_endpoints(seg: SegmentRecord, ctx: dict) -> list[float]:
    return [*seg.p0, *seg.p1]                      # 6: absolute open- & end-side points


def _cross_section(seg: SegmentRecord, ctx: dict) -> list[float]:
    return [seg.cross_section_mm2]                 # 1


def _displacement_vector(seg: SegmentRecord, ctx: dict) -> list[float]:
    return list(seg.vector)                        # 3: relative open->end displacement


def _position_along_tube(seg: SegmentRecord, ctx: dict) -> list[float]:
    return [float(ctx["cum_len"])]                 # 1: depth from the open end


def _turn_angle(seg: SegmentRecord, ctx: dict) -> list[float]:
    prev: SegmentRecord | None = ctx["prev"]
    if prev is None:
        return [0.0]
    a, b = prev.vector, seg.vector
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return [0.0]
    cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return [float(np.arccos(cos))]                 # 1: bend angle vs previous segment


# --------------------------------------------------------------------------- #
# Rep-1 v2 (seg_egnn) node features: physically-meaningful per-segment descriptors.
# --------------------------------------------------------------------------- #
def _seg_center(seg: SegmentRecord, ctx: dict) -> list[float]:
    return list(seg.midpoint)                      # 3: c_i = (p0+p1)/2


def _seg_direction(seg: SegmentRecord, ctx: dict) -> list[float]:
    v = seg.vector
    n = np.linalg.norm(v)
    return list(v / n) if n > 1e-9 else [0.0, 0.0, 0.0]   # 3: d_hat_i, unit open->end


def _seg_length(seg: SegmentRecord, ctx: dict) -> list[float]:
    return [seg.length]                            # 1: L_i


def _width_height(seg: SegmentRecord, ctx: dict) -> list[float]:
    return [seg.width, seg.thickness]              # 2: w_i (in-plane), h_i (z-thickness)


def _hydraulic_diameter(seg: SegmentRecord, ctx: dict) -> list[float]:
    return [seg.hydraulic_diameter]                # 1: D_h,i = 2wh/(w+h)


def _dist_to_open(seg: SegmentRecord, ctx: dict) -> list[float]:
    return [float(ctx["cum_len"]) + 0.5 * seg.length]   # 1: s_i, open-cap -> segment center


def _tube_length(seg: SegmentRecord, ctx: dict) -> list[float]:
    return [float(ctx["tube_length"])]             # 1: L_tube (whole tube)


FEATURES = {
    "abs_endpoints": _abs_endpoints,
    "cross_section": _cross_section,
    "displacement_vector": _displacement_vector,
    "position_along_tube": _position_along_tube,
    "turn_angle": _turn_angle,
    # Rep-1 v2:
    "seg_center": _seg_center,
    "seg_direction": _seg_direction,
    "seg_length": _seg_length,
    "width_height": _width_height,
    "hydraulic_diameter": _hydraulic_diameter,
    "dist_to_open": _dist_to_open,
    "tube_length": _tube_length,
}
FEATURE_DIMS = {"abs_endpoints": 6, "cross_section": 1, "displacement_vector": 3,
                "position_along_tube": 1, "turn_angle": 1,
                "seg_center": 3, "seg_direction": 3, "seg_length": 1, "width_height": 2,
                "hydraulic_diameter": 1, "dist_to_open": 1, "tube_length": 1}


def node_feature_dim(names: list[str]) -> int:
    return sum(FEATURE_DIMS[n] for n in names)


def compute_features(seg: SegmentRecord, ctx: dict, names: list[str]) -> list[float]:
    out: list[float] = []
    for n in names:
        out.extend(FEATURES[n](seg, ctx))
    return out
