"""Build per-device graphs once, plus per-fold feature standardization.

Supports both representations behind one interface:
  * ``seg_graph`` (method #1) -> ``Data`` with a single node-feature matrix ``x``.
  * ``dual_graph`` (method #2) -> ``HeteroData`` with {vertex, surface} nodes and typed edges.

Standardization stats are computed on the TRAINING devices of a fold only and applied to
every graph, so no information leaks from the held-out device into training. ``fit_scaler``
returns a ``Scaler`` that knows how to standardize whichever graph type it was fit on.
"""
from __future__ import annotations

import torch
from torch_geometric.data import Data, HeteroData

from ..data.targets import TargetSet, load_targets

VERT = "vertex"
SURF = "surface"
SEG = ("vertex", "seg", "vertex")
COUPLE = ("surface", "couple", "surface")


def build_graphs(
    targets: TargetSet | None = None,
    *,
    representation: str = "seg_graph_v2",
    source: str = "step",
    feature_names: tuple[str, ...] = ("abs_endpoints", "cross_section"),
    coupling_scale: float = 1.5,
    invariant: bool = False,
    add_resonance: bool = False,
    resonance_k: int = 3,
    resonance_mode: str = "quarter",
    resonance_sigma: float = 40.0,
    graph1_resonance: bool = False,
    px_mm: float = 0.25,
    open_cutoff_scale: float = 1.5,
    spatial_cutoff_scale: float = 1.5,
    add_gaps: bool = False,
) -> tuple[dict[str, object], TargetSet]:
    if targets is None:
        targets = load_targets()

    if representation == "voxel":
        from ..representations import voxel
        canvas = voxel._canvas(px_mm)                    # shared frame, built once
        graphs = {
            dev: voxel.build(
                dev, px_mm=px_mm, canvas=canvas, y=targets.y[dev], mask=targets.mask[dev],
                add_resonance=add_resonance, resonance_k=resonance_k,
                resonance_mode=resonance_mode, resonance_sigma=resonance_sigma,
                grid=targets.grid, add_gaps=add_gaps,
            )
            for dev in targets.y
        }
        return graphs, targets

    if representation == "seg_graph_v2":
        from ..representations import seg_graph_v2
        graphs = {
            dev: seg_graph_v2.build(
                dev, source=source, feature_names=feature_names,
                open_cutoff_scale=open_cutoff_scale, spatial_cutoff_scale=spatial_cutoff_scale,
                y=targets.y[dev], mask=targets.mask[dev],
                add_resonance=add_resonance, resonance_k=resonance_k,
                resonance_mode=resonance_mode, resonance_sigma=resonance_sigma,
                grid=targets.grid,
            )
            for dev in targets.y
        }
        return graphs, targets

    if representation == "dual_graph":
        from ..data.face_extract import get_faces
        from ..representations import dual_graph
        # pass 1: global Vmax across all devices so every graph pads to one width (batchable)
        vmax = 4
        for dev in targets.y:
            surfaces, _ = get_faces(dev)
            vmax = max(vmax, max((s.n_vertices for s in surfaces), default=4))
        graphs = {
            dev: dual_graph.build(
                dev, source=source, vmax=vmax, invariant=invariant,
                add_resonance=add_resonance, resonance_k=resonance_k,
                resonance_mode=resonance_mode, resonance_sigma=resonance_sigma,
                graph1_resonance=graph1_resonance, grid=targets.grid,
                y=targets.y[dev], mask=targets.mask[dev],
            )
            for dev in targets.y
        }
        return graphs, targets

    raise ValueError(f"Unknown representation {representation!r}")


# --------------------------------------------------------------------------- #
# Standardization
# --------------------------------------------------------------------------- #
def _stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x.mean(dim=0), x.std(dim=0).clamp_min(1e-6)


def _is_voxel(g) -> bool:
    return hasattr(g, "vox")


class Scaler:
    """Feature standardizer for a fold; fit on training graphs, applied to any graph."""

    def __init__(self, graphs: list):
        self.seg = False
        self.voxel = _is_voxel(graphs[0])
        if self.voxel:
            # standardize the per-layer scalars only, over REAL layers of the training set;
            # leave vox as raw 0/1 occupancy (already normalized; standardizing kills sparsity)
            real = torch.cat([g.lay[g.lmask > 0] for g in graphs], dim=0)   # (sum L, S)
            self.lay_mean, self.lay_std = _stats(real)
            return

        self.hetero = isinstance(graphs[0], HeteroData)
        if not self.hetero:
            self.mean, self.std = _stats(torch.cat([g.x for g in graphs], dim=0))
            return

        self.seg = "seg" in graphs[0].node_types      # Rep-1 v2 (seg_graph_v2)
        if self.seg:
            self._fit_seg(graphs)
            return
        # HeteroData: per node-/edge-tensor stats, coords for vpad from real vertices only
        self.s = {}
        self.s["vertex_x"] = _stats(torch.cat([g[VERT].x for g in graphs], dim=0))
        self.s["seg_edge"] = _stats(torch.cat([g[SEG].edge_attr for g in graphs], dim=0))
        self.s["surf_feat"] = _stats(torch.cat([g[SURF].feat for g in graphs], dim=0))
        real = torch.cat([g[SURF].vpad[g[SURF].vmask > 0] for g in graphs], dim=0)  # (M,3)
        self.s["surf_vert"] = _stats(real)
        ce = torch.cat([g[COUPLE].edge_attr for g in graphs], dim=0)
        cont = graphs[0][COUPLE].edge_attr.size(1) - 1  # all but the last col (coupling_type)
        # standardize the continuous coupling features; leave coupling_type (last col) raw
        self.s["couple_edge"] = _stats(ce[:, :cont]) if ce.numel() else (
            torch.zeros(cont), torch.ones(cont))

    def _fit_seg(self, graphs: list):
        """Rep-1 v2: standardize PHYSICAL columns only (leave unit/bounded features raw).
        Which columns are physical lives in ``seg_graph_v2`` so builder & scaler can't drift."""
        from ..representations import seg_graph_v2 as V2
        self._v2 = V2
        n0 = V2.NODE_RAW_COLS
        W = graphs[0][V2.SEG].x.size(1)
        self._seg_node_phys = [i for i in range(W) if i not in n0]
        x = torch.cat([g[V2.SEG].x for g in graphs], dim=0)[:, self._seg_node_phys]
        self._seg_node_stats = _stats(x)

        self._seg_edge = {}   # rel -> (phys_idx, mean, std)
        for rel, dim, raw in ((V2.PATH, V2.PATH_EDGE_DIM, V2.PATH_RAW_COLS),
                              (V2.OPEN, V2.OPEN_EDGE_DIM, V2.OPEN_RAW_COLS),
                              (V2.SPATIAL, V2.SPATIAL_EDGE_DIM, V2.SPATIAL_RAW_COLS)):
            phys = [i for i in range(dim) if i not in raw]
            ea = torch.cat([g[rel].edge_attr for g in graphs], dim=0)
            stats = _stats(ea[:, phys]) if ea.numel() and phys else (
                torch.zeros(len(phys)), torch.ones(len(phys)))
            self._seg_edge[rel] = (phys, *stats)

    def _apply_seg(self, graph):
        V2 = self._v2
        g = graph.clone()
        phys = self._seg_node_phys
        m, s = self._seg_node_stats
        x = g[V2.SEG].x.clone()
        x[:, phys] = (x[:, phys] - m) / s
        g[V2.SEG].x = x
        for rel, (pidx, m, s) in self._seg_edge.items():
            ea = g[rel].edge_attr
            if ea.numel() and pidx:
                ea = ea.clone()
                ea[:, pidx] = (ea[:, pidx] - m) / s
                g[rel].edge_attr = ea
        return g

    def apply(self, graph):
        if self.voxel:
            g = graph.clone()
            lay = (g.lay - self.lay_mean) / self.lay_std
            g.lay = lay * g.lmask.unsqueeze(-1)          # keep padded rows at zero
            return g

        if not self.hetero:
            g = graph.clone()
            g.x = (g.x - self.mean) / self.std
            return g

        if self.seg:
            return self._apply_seg(graph)

        g = graph.clone()
        m, s = self.s["vertex_x"]; g[VERT].x = (g[VERT].x - m) / s
        m, s = self.s["seg_edge"]; g[SEG].edge_attr = (g[SEG].edge_attr - m) / s
        m, s = self.s["surf_feat"]; g[SURF].feat = (g[SURF].feat - m) / s

        m, s = self.s["surf_vert"]                      # standardize coords, then re-zero pads
        vpad = (g[SURF].vpad - m) / s
        g[SURF].vpad = vpad * g[SURF].vmask.unsqueeze(-1)

        ea = g[COUPLE].edge_attr
        if ea.numel():
            m, s = self.s["couple_edge"]
            cont = m.numel()
            ea = ea.clone()
            ea[:, :cont] = (ea[:, :cont] - m) / s
            g[COUPLE].edge_attr = ea
        return g


def fit_scaler(graphs: list) -> Scaler:
    return Scaler(graphs)


def collate(graphs: list):
    """Batch a list of built graphs: voxel -> stacked Data, otherwise a PyG Batch."""
    from torch_geometric.data import Batch
    if _is_voxel(graphs[0]):
        from ..representations.voxel import collate_voxel
        return collate_voxel(graphs)
    return Batch.from_data_list(graphs)
