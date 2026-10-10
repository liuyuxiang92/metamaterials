"""Representation #2: two graphs per device, packed into one ``HeteroData``.

Graph 1 — tube shape (node type ``vertex``, edge type ``(vertex, seg, vertex)``):
    node = a fold vertex of a tube; edge = the straight segment between consecutive vertices
    OF THE SAME TUBE (in tube order). Nodes + edges reconstruct every tube's 3-D path.

Graph 2 — surface coupling (node type ``surface``, edge type ``(surface, couple, surface)``):
    node = a planar B-rep face; edge = a facing-surface coupling (intra- or inter-tube).

Two feature modes (``invariant`` flag):
  * absolute (default, method-#2 v1): node/edge features use ABSOLUTE coordinates/normals
    -> reconstructable but pose-dependent.
  * invariant (Ver1R upgrade): every feature is translation- AND rotation-invariant.
      Graph 1 node = [cross_section, cum_len, bend_angle, dihedral];  edge = [length].
      Graph 2 node = [area, perimeter, vcount, e1, e2] + DeepSets over per-boundary-vertex
        invariants [edge_len_to_next, interior_angle, dist_to_centroid];
        edge = [gap_dist, lateral_offset, overlap_area, coupling_type].
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import HeteroData

from ..data.face_extract import get_faces
from ..data.fold_coords import get_segments
from ..data.schema import CouplingRecord, SurfaceRecord

SURF_FEAT_DIM = 12  # absolute mode: area(1)+normal(3)+centroid(3)+perimeter(1)+vcount(1)+extents(3)


# --------------------------------------------------------------------------- #
# Small invariant-geometry helpers
# --------------------------------------------------------------------------- #
def _angle(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))


def _torsion(p0, p1, p2, p3) -> float:
    """Signed dihedral angle about the p1-p2 bond (radians)."""
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    if np.linalg.norm(n1) < 1e-9 or np.linalg.norm(n2) < 1e-9:
        return 0.0
    b2n = b2 / np.linalg.norm(b2)
    m1 = np.cross(n1, b2n)
    return float(np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))


def _bend_dihedral(verts: np.ndarray):
    """Per-vertex bend + dihedral (0 where undefined) and per-segment lengths."""
    M = len(verts)
    segs = verts[1:] - verts[:-1]
    lengths = np.linalg.norm(segs, axis=1)
    bend = np.zeros(M)
    dih = np.zeros(M)
    for k in range(1, M - 1):                       # interior vertices have a bend
        bend[k] = _angle(segs[k - 1], segs[k])
    for k in range(2, M - 1):                        # dihedral needs 3 consecutive segments
        dih[k] = _torsion(verts[k - 2], verts[k - 1], verts[k], verts[k + 1])
    return bend, dih, lengths


def _plane_basis(n: np.ndarray):
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def _principal_extents(verts: np.ndarray, normal: np.ndarray) -> tuple[float, float]:
    """Face length/width along its OWN principal axes (rotation-invariant)."""
    n = normal / (np.linalg.norm(normal) + 1e-12)
    u, w = _plane_basis(n)
    c = verts.mean(axis=0)
    d = verts - c
    P = np.column_stack([d @ u, d @ w])             # (V, 2) in-plane coords
    if len(P) < 2:
        return 0.0, 0.0
    cov = np.cov(P.T)
    _, evecs = np.linalg.eigh(cov)
    proj = P @ evecs
    ext = np.sort(proj.max(axis=0) - proj.min(axis=0))[::-1]
    return float(ext[0]), float(ext[1])


def _polygon_vertex_invariants(verts: np.ndarray) -> np.ndarray:
    """Per-boundary-vertex [edge_len_to_next, interior_angle, dist_to_centroid] (V, 3)."""
    V = len(verts)
    c = verts.mean(axis=0)
    out = np.zeros((V, 3))
    for i in range(V):
        cur, nxt, prev = verts[i], verts[(i + 1) % V], verts[(i - 1) % V]
        out[i, 0] = np.linalg.norm(nxt - cur)
        out[i, 1] = _angle(prev - cur, nxt - cur)
        out[i, 2] = np.linalg.norm(cur - c)
    return out


def _lateral_offset(gap_vec, gap_dist: float) -> float:
    g2 = float(np.dot(gap_vec, gap_vec))
    return float(np.sqrt(max(0.0, g2 - gap_dist * gap_dist)))


# --------------------------------------------------------------------------- #
# Graph 1 (tube shape)
# --------------------------------------------------------------------------- #
def _tube_vertices(device: str, source: str):
    """Yield (tube_index, cross_section, verts[M,3]) per tube, vertices open->end order."""
    segs = get_segments(device, source=source)
    by_tube: dict[int, list] = {}
    for s in segs:
        by_tube.setdefault(s.tube_index, []).append(s)
    for tube, lst in sorted(by_tube.items()):
        lst.sort(key=lambda s: s.segment_index)
        verts = np.array([list(lst[0].p0)] + [list(s.p1) for s in lst])
        yield tube, lst[0].cross_section_mm2, verts


def _vertex_graph_absolute(device: str, source: str):
    pos, node_tube, e_src, e_dst, e_attr = [], [], [], [], []
    for tube, _, verts in _tube_vertices(device, source):
        start = len(pos)
        pos.extend(verts.tolist())
        node_tube += [tube] * len(verts)
        for k in range(len(verts) - 1):
            a, b = start + k, start + k + 1
            p0, p1 = verts[k], verts[k + 1]
            e_src += [a, b]; e_dst += [b, a]
            e_attr += [[*(p1 - p0), *p0, *p1], [*(p0 - p1), *p1, *p0]]
    x = torch.tensor(pos, dtype=torch.float32)
    return (x, torch.tensor([e_src, e_dst], dtype=torch.long),
            torch.tensor(e_attr, dtype=torch.float32).reshape(-1, 9), node_tube)


def _vertex_graph_invariant(device: str, source: str):
    feat, node_tube, e_src, e_dst, e_attr = [], [], [], [], []
    for tube, cs, verts in _tube_vertices(device, source):
        bend, dih, lengths = _bend_dihedral(verts)
        cum = np.concatenate([[0.0], np.cumsum(lengths)])
        start = len(feat)
        for k in range(len(verts)):
            feat.append([cs, cum[k], bend[k], dih[k]])
        node_tube += [tube] * len(verts)
        for k in range(len(verts) - 1):
            a, b = start + k, start + k + 1
            L = float(lengths[k])
            e_src += [a, b]; e_dst += [b, a]; e_attr += [[L], [L]]
    x = torch.tensor(feat, dtype=torch.float32)
    return (x, torch.tensor([e_src, e_dst], dtype=torch.long),
            torch.tensor(e_attr, dtype=torch.float32).reshape(-1, 1), node_tube)


# --------------------------------------------------------------------------- #
# Graph 2 (surface coupling)
# --------------------------------------------------------------------------- #
def _surface_feat_absolute(s: SurfaceRecord) -> list[float]:
    return [s.area_mm2, *s.normal, *s.centroid, s.perimeter_mm,
            float(s.n_vertices), *s.bbox_extents]


def _surface_graph(surfaces, couplings, vmax: int, *, invariant: bool):
    n = len(surfaces)
    vdim = 3
    vpad = np.zeros((n, vmax, vdim), dtype=np.float32)
    vmask = np.zeros((n, vmax), dtype=np.float32)
    feat_dim = 5 if invariant else SURF_FEAT_DIM
    feat = np.zeros((n, feat_dim), dtype=np.float32)
    for i, s in enumerate(surfaces):
        V = s.vertices_arr
        vfeat = _polygon_vertex_invariants(V) if invariant else V
        v = min(len(V), vmax)
        vpad[i, :v] = vfeat[:v]
        vmask[i, :v] = 1.0
        if invariant:
            e1, e2 = _principal_extents(V, s.normal_arr)
            feat[i] = [s.area_mm2, s.perimeter_mm, float(s.n_vertices), e1, e2]
        else:
            feat[i] = _surface_feat_absolute(s)

    e_src, e_dst, e_attr = [], [], []
    for c in couplings:
        a, b = c.surf_a, c.surf_b
        if invariant:
            lat = _lateral_offset(np.asarray(c.gap_vec), c.gap_dist)
            fwd = [c.gap_dist, lat, c.overlap_area_mm2, float(c.coupling_type)]
            rev = fwd
        else:
            gv = np.asarray(c.gap_vec)
            fwd = [*gv, c.gap_dist, c.overlap_area_mm2, float(c.coupling_type)]
            rev = [*(-gv), c.gap_dist, c.overlap_area_mm2, float(c.coupling_type)]
        e_src += [a, b]; e_dst += [b, a]; e_attr += [fwd, rev]

    ew = 4 if invariant else 6
    edge_index = torch.tensor([e_src, e_dst], dtype=torch.long) if e_src \
        else torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(e_attr, dtype=torch.float32).reshape(-1, ew) if e_attr \
        else torch.zeros((0, ew), dtype=torch.float32)
    return (torch.from_numpy(vpad), torch.from_numpy(vmask), torch.from_numpy(feat),
            edge_index, edge_attr)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build(
    device: str,
    *,
    source: str = "step",
    vmax: int | None = None,
    y: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    use_cache: bool = True,
    invariant: bool = False,
    add_resonance: bool = False,
    resonance_k: int = 3,
    resonance_mode: str = "quarter",
    resonance_sigma: float = 40.0,
    graph1_resonance: bool = False,
    grid: np.ndarray | None = None,
) -> HeteroData:
    surfaces, couplings = get_faces(device, use_cache=use_cache)
    if vmax is None:
        vmax = max((s.n_vertices for s in surfaces), default=4)

    if invariant:
        xv, ev, ea_v, node_tube = _vertex_graph_invariant(device, source)
    else:
        xv, ev, ea_v, node_tube = _vertex_graph_absolute(device, source)
    vpad, vmask, feat, es, ea_s = _surface_graph(surfaces, couplings, vmax, invariant=invariant)

    if add_resonance:
        from ..data.resonance import tube_resonances
        res = tube_resonances(device, k=resonance_k, mode=resonance_mode)
        # tube_f1 on each surface node (for coupling to reason about resonance splitting)
        f1col = torch.tensor([[float(res[s.tube_index][0]) if s.tube_index in res else 0.0]
                              for s in surfaces], dtype=torch.float32)
        feat = torch.cat([feat, f1col], dim=1)
        # optional: per-tube f_n on Graph-1 vertices
        if graph1_resonance:
            zero = np.zeros(resonance_k)
            fcols = torch.tensor([res.get(t, zero)[:resonance_k].tolist() if t in res
                                  else zero.tolist() for t in node_tube], dtype=torch.float32)
            xv = torch.cat([xv, fcols], dim=1)

    data = HeteroData()
    data["vertex"].x = xv
    data["vertex"].num_nodes = xv.size(0)
    data["vertex", "seg", "vertex"].edge_index = ev
    data["vertex", "seg", "vertex"].edge_attr = ea_v

    data["surface"].vpad = vpad
    data["surface"].vmask = vmask
    data["surface"].feat = feat
    data["surface"].num_nodes = feat.size(0)
    data["surface", "couple", "surface"].edge_index = es
    data["surface", "couple", "surface"].edge_attr = ea_s

    data.device_name = device
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


def model_dims(g: HeteroData) -> dict:
    return dict(
        vertex_in=g["vertex"].x.size(1),
        seg_edge=g["vertex", "seg", "vertex"].edge_attr.size(1),
        surf_vert=g["surface"].vpad.size(2),
        surf_feat=g["surface"].feat.size(1),
        couple_edge=g["surface", "couple", "surface"].edge_attr.size(1),
        vmax=g["surface"].vpad.size(1),
        comb_dim=int(g.comb.size(1)) if hasattr(g, "comb") else 0,
    )


if __name__ == "__main__":
    from ..data.targets import load_targets
    ts = load_targets()
    vmax = max(max((r.n_vertices for r in get_faces(dev)[0]), default=4) for dev in ts.y)
    print(f"global Vmax = {vmax}")
    for inv in (False, True):
        g = build(next(iter(ts.y)), source="step", vmax=vmax, invariant=inv,
                  y=ts.y[next(iter(ts.y))], mask=ts.mask[next(iter(ts.y))])
        print(f"invariant={inv}: dims={model_dims(g)}")
