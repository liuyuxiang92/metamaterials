# Folded-Tube Metamaterial Absorption Surrogates

Deep-learning surrogates that predict the **sound-absorption spectrum** of folded-tube acoustic metamaterials directly from their CAD geometry (STEP), trained against COMSOL simulations.

The repository provides three model families that share one data pipeline, training loop and evaluation protocol:

| Method | Representation | Model | Config |
|---|---|---|---|
| **Segment GNN** | One node per straight tube segment, with three typed, feature-carrying edge relations (path, opening, spatial) | Edge-conditioned `GINEConv` per relation | `configs/seg_egnn.yaml` |
| **Dual GNN** | Two graphs per device: a tube-shape graph (fold vertices and segments) and a surface-coupling graph (B-rep faces and facing-surface couplings) | Two `GINEConv` streams + DeepSets face encoder, with resonance features | `configs/dual_gnn_res.yaml` |
| **Layer CNN** | A z-ordered stack of per-layer air-occupancy images (0.25 mm/px) with layer thickness, position and inter-layer gaps | Shared 2D CNN per layer + local 1D convolution along z, rich pooling | `configs/layer_cnn_v2.yaml` |

---

## Contents

- [Installation](#installation)
- [Data layout](#data-layout)
- [Quick start](#quick-start)
- [Methods](#methods)
- [Evaluation protocol](#evaluation-protocol)
- [Outputs](#outputs)
- [Repository structure](#repository-structure)
- [Configuration reference](#configuration-reference)
- [Utilities](#utilities)

---

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -r requirements.txt
```

Main dependencies are PyTorch, PyTorch Geometric, `cadquery-ocp` (OpenCASCADE bindings for STEP/B-rep extraction), `shapely`, `scipy`, `scikit-image`, `pandas` and `matplotlib`.

> **Note.** Use `cadquery-ocp < 8`. Version 8.x changes the `Bnd_Box` API and breaks the geometry extraction. As an alternative, `pythonocc-core` from conda-forge is also supported.

---

## Data layout

This repository is the `gnn` Python package. It contains **code only**: no geometry, spectra or trained models. Clone it into a folder named `gnn`, next to the device folders:

```
code/
├── gnn/                         <- this repository
├── AI2026_Trial_Ver1R-1/
│   ├── AI2026_Ver1R-1.step      CAD geometry
│   └── AI2026_Ver1R-1.txt       COMSOL absorption spectrum
├── AI2026_Trial-Ver2-1a/
│   ├── AI2026_Ver2-1a.step
│   └── AI2026_Ver2-1a_Report.txt
└── ...
```

Devices are discovered automatically from every folder that contains a `.step` file and a spectrum report. Adding a new device only requires adding its folder. The frequency grid is built as the union of all devices' grids, with per-device masks for bins a device does not cover.

The derived files `tube_geometry.csv`, `csv/` and the `results/` cache are generated locally and are not tracked.

---

## Quick start

Run from the parent `code/` directory:

```bash
cd code
git clone https://github.com/liuyuxiang92/metamaterials.git gnn

PYTHONPATH=. python -m gnn.run --config gnn/configs/seg_egnn.yaml       # Segment GNN
PYTHONPATH=. python -m gnn.run --config gnn/configs/dual_gnn_res.yaml   # Dual GNN
PYTHONPATH=. python -m gnn.run --config gnn/configs/layer_cnn_v2.yaml   # Layer CNN
```

Any config key can be overridden from the command line:

```bash
PYTHONPATH=. python -m gnn.run --config gnn/configs/seg_egnn.yaml --set epochs=250 device=cpu run_name=seg_250
```

Each run trains leave-one-device-out and then writes plots and metrics automatically (see [Outputs](#outputs)).

---

## Methods

### Segment GNN (`seg_graph_v2` + `segegnn`)

- **Nodes** are the straight segments of each folded tube. Each has 12 features: center, unit direction (open to closed end), segment length, width, thickness, hydraulic diameter, distance from the opening, and whole-tube length.
- **Path edges** connect consecutive segments of the same tube. Their features are the center offset, distance, `cos θ` and turn sign.
- **Opening edges** couple tube inlets on the shared inlet plane. Their features are the opening offset, distance, Δz and both opening areas.
- **Spatial edges** connect nearby non-adjacent segments within and across tubes. Their features are the exact segment-to-segment minimum distance, alignment, overlap ratio and Δz.
- Every relation has its own edge-conditioned `GINEConv`. Physical length and area features are standardized per fold, using the training devices only.

### Dual GNN (`dual_graph` + `dualgnn`)

- **Graph 1, tube shape:** nodes are fold vertices and edges are the straight segments between consecutive vertices of the same tube. Together they rebuild each tube's 3D path.
- **Graph 2, surface coupling:** nodes are planar B-rep faces. Each face is encoded with DeepSets over its boundary vertices, plus area, normal, centroid, perimeter and bounding box. Edges are facing-surface couplings: thin-gap, anti-parallel and overlapping faces, within a tube or between tubes.
- **Resonance features:** per-tube quarter-wave frequencies are added on the faces, and a device-level resonance comb is added at the head.
- Each graph has its own `GINEConv` stream with mean and max pooling. The two are concatenated and passed to an MLP head.

### Layer CNN (`voxel` + `layercnn`)

- Every solid is a Z-extruded prism, so a device is represented exactly as a z-ordered stack of layer images, plus `[thickness, z_center, gap_below, gap_above]` per layer.
- A shared 2D CNN encodes each layer image.
- A stack of local 1D convolutions along z then fuses neighbouring layers. This stage is **order-aware**: devices made of the same layers in a different order, such as Ver2-1a and Ver2-1b, get different representations.
- The pooling is `[front layer, back layer, mean, max, attention]`, then an MLP head.

All models end in a sigmoid, so predicted absorption lies in [0, 1]. Training minimizes the masked MSE.

---

## Evaluation protocol

Training is **leave-one-device-out**. With *N* devices, *N* models are trained, and each holds one device out.

- **Nested LOO (default, `inner_folds: 5`).** An inner leave-one-out over the training devices selects the epoch budget. The model is then retrained on all training devices for that budget and evaluated once on the untouched held-out device. This gives an unbiased test error.
- **Held-out-as-validation (`inner_folds: 1`).** The held-out device drives early stopping (patience `patience`), and the best checkpoint is kept. This is faster, but it is optimistic, because the evaluated device also selects the epoch.
- `training/plain_loo.py` compares both stopping rules on the same folds.

---

## Outputs

Each run writes the following to `results/<run_name>/`:

| File | Content |
|---|---|
| `metrics.csv` | Per-device MAE / RMSE on the held-out device |
| `summary.md` | Run configuration and aggregate metrics |
| `learning_curves.png` | Training and validation loss per fold |
| `predictions.png` | Predicted vs. COMSOL spectra for every held-out device |
| `per_frequency_error.png` | Error as a function of frequency |
| `train_output.pt` | Predictions, histories and configuration, for re-analysis |

---

## Repository structure

```
.
├── run.py                   Entry point: train + analyze for one config
├── configs/                 seg_egnn.yaml, dual_gnn_res.yaml, layer_cnn_v2.yaml
├── data/
│   ├── devices.py           Device discovery
│   ├── targets.py           Union frequency grid and masks
│   ├── brep_extract.py      Exact tube centerlines from the B-rep
│   ├── face_extract.py      Planar faces and facing-surface couplings
│   ├── step_extract.py      Legacy voxel/skeleton centerline extractor
│   ├── fold_coords.py       Segment coordinates (B-rep, STEP or scalar)
│   ├── resonance.py         Quarter/half-wave resonance features
│   └── schema.py            Data records
├── representations/
│   ├── seg_graph_v2.py      Segment graph
│   ├── dual_graph.py        Two-graph HeteroData
│   └── voxel.py             Layer-stack images
├── models/
│   ├── seg_egnn.py          Segment GNN
│   ├── dualgnn.py           Dual GNN
│   ├── cnn.py               Layer CNN
│   └── base.py              Model factory
└── training/
    ├── train.py             Leave-one-out training loop (nested / plain)
    ├── dataset.py           Graph building, per-fold scaling, collation
    ├── losses.py            Masked losses
    ├── metrics.py           MAE / RMSE
    ├── analyze.py           Plots and summary
    ├── plain_loo.py         Stopping-rule diagnostic
    ├── twin_check.py        Order-sensitivity check on the Ver2-1a/1b pair
    └── twin_cross.py        Cross-twin prediction check
```

---

## Configuration reference

| Key | Meaning |
|---|---|
| `representation`, `model` | Method selection (`seg_graph_v2`/`segegnn`, `dual_graph`/`dualgnn`, `voxel`/`layercnn`) |
| `source` | Geometry source: `brep` (exact, default), `step` (voxel skeleton) or `scalar` (smoke test) |
| `hidden`, `num_layers`, `dropout` | Model size and regularization |
| `lr`, `weight_decay`, `epochs`, `patience`, `seed` | Optimization |
| `inner_folds` | `5` = nested LOO; `1` = held-out device used for early stopping |
| `add_resonance`, `resonance_k`, `resonance_mode`, `resonance_sigma` | Physics resonance features |
| `open_cutoff_scale`, `spatial_cutoff_scale` | Edge proximity cutoffs (Segment GNN) |
| `coupling_scale`, `invariant` | Coupling cutoff and pose invariance (Dual GNN) |
| `px_mm`, `add_gaps`, `rich_pool`, `z_kernel`, `z_depth` | Layer CNN resolution and architecture |
| `device` | `cpu`, `cuda`, `mps` or `auto` |

---

## Utilities

```bash
PYTHONPATH=. python -m gnn.representations.seg_graph_v2              # node/edge counts per device
PYTHONPATH=. python -m gnn.representations.voxel                     # layer-stack shapes per device
PYTHONPATH=. python -m gnn.representations.plot_slices <device>      # layer images
PYTHONPATH=. python -m gnn.data.face_extract <device>                # surfaces and couplings report
PYTHONPATH=. python -m gnn.data.step_inspect <device>                # STEP topology
PYTHONPATH=. python -m gnn.training.twin_check --run <run_name>      # order-sensitivity check
PYTHONPATH=. python -m gnn.training.plain_loo --config gnn/configs/seg_egnn.yaml
```
