"""Core data contract shared by every geometry source and every representation.

A device's folded-tube geometry is expressed as a list of ``SegmentRecord`` — one
per straight segment of a tube (a tube folded ``k`` times has ``k + 1`` segments,
hence ``k + 1`` records / graph nodes). Coordinates are ABSOLUTE, in a canonical
global frame shared across devices (see ``step_extract.canonicalize``), ordered so
that ``p0`` is the open/inlet-side endpoint and ``p1`` is the end/terminal-side
endpoint of the segment. This ordering encodes the user's open->end direction:
``vector = p1 - p0``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class SegmentRecord:
    device: str
    tube_index: int          # which tube (resonator) this segment belongs to
    segment_index: int       # position along the tube, 0 = at the open end
    p0: Vec3                  # open-side endpoint, absolute global coords (mm)
    p1: Vec3                  # end-side endpoint, absolute global coords (mm)
    cross_section_mm2: float  # duct cross-sectional area (constant along a tube)
    # in-plane channel width and out-of-plane (z) thickness of the duct cross-section.
    # Optional/backward-compatible: when absent, both fall back to sqrt(cross_section)
    # (a square-duct approximation) via the ``width``/``thickness`` properties below.
    width_mm: float | None = None
    thickness_mm: float | None = None

    @property
    def width(self) -> float:
        """Channel width (mm); sqrt(area) fallback when not extracted."""
        return float(self.width_mm) if self.width_mm is not None else math.sqrt(self.cross_section_mm2)

    @property
    def thickness(self) -> float:
        """Cross-section z-thickness (mm); sqrt(area) fallback when not extracted."""
        return float(self.thickness_mm) if self.thickness_mm is not None else math.sqrt(self.cross_section_mm2)

    @property
    def hydraulic_diameter(self) -> float:
        """D_h = 2*w*h/(w+h)."""
        w, h = self.width, self.thickness
        return 2.0 * w * h / (w + h) if (w + h) > 1e-9 else 0.0

    @property
    def p0_arr(self) -> np.ndarray:
        return np.asarray(self.p0, dtype=np.float64)

    @property
    def p1_arr(self) -> np.ndarray:
        return np.asarray(self.p1, dtype=np.float64)

    @property
    def vector(self) -> np.ndarray:
        """open->end displacement (direction * length)."""
        return self.p1_arr - self.p0_arr

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.vector))

    @property
    def midpoint(self) -> np.ndarray:
        return 0.5 * (self.p0_arr + self.p1_arr)


# --------------------------------------------------------------------------- #
# Method #2 (two-graph GNN) geometry contract: surfaces + their couplings.
# A ``SurfaceRecord`` is one planar B-rep face of a tube's air domain; a
# ``CouplingRecord`` is an edge between two facing surfaces (see face_extract).
# Coordinates are ABSOLUTE, in the same canonical global frame as SegmentRecord.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SurfaceRecord:
    device: str
    tube_index: int                 # which tube (solid) this face belongs to
    surface_index: int              # unique per device (0..N-1), the Graph-2 node id
    vertices: tuple[Vec3, ...]      # ordered boundary-wire vertices, absolute coords (mm)
    area_mm2: float                 # exact face area (GProp)
    normal: Vec3                    # unit outward normal (points out of the solid)
    centroid: Vec3                  # exact area-weighted centroid (GProp)
    perimeter_mm: float             # sum of boundary edge lengths
    bbox_extents: Vec3              # (dx, dy, dz) of the face's axis-aligned bbox

    @property
    def vertices_arr(self) -> np.ndarray:
        return np.asarray(self.vertices, dtype=np.float64)     # (V, 3)

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def centroid_arr(self) -> np.ndarray:
        return np.asarray(self.centroid, dtype=np.float64)

    @property
    def normal_arr(self) -> np.ndarray:
        return np.asarray(self.normal, dtype=np.float64)


@dataclass(frozen=True)
class CouplingRecord:
    device: str
    surf_a: int                     # surface_index of the first face
    surf_b: int                     # surface_index of the second face
    gap_vec: Vec3                   # centroid_b - centroid_a (mm)
    gap_dist: float                 # partition thickness, |offset| along the shared normal (mm)
    overlap_area_mm2: float         # exact projected-polygon overlap area
    coupling_type: int              # 0 = intra-tube (same tube), 1 = inter-tube
