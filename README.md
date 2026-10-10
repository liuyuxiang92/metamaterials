# Folded-tube GNN surrogate

Predicts a metamaterial's absorption spectrum from its folded-tube geometry with an
edge-type-aware graph neural network. See `../.claude/plans/zesty-questing-reef.md` for the
full design rationale.

## Idea (Representation #1)

- **Node** = one straight segment of a tube. A tube folded *k* times → *k+1* nodes.
- **Node features** = `[x0,y0,z0, x1,y1,z1, cross_section]` — absolute open- and end-side
  endpoint coordinates (open→end ordered) in a canonical global frame, plus duct area.
- **Edges** (two kinds, separate GNN weights):
  - type 0 **intra-tube** — consecutive segments of the same tube (a fold);
  - type 1 **inter-tube** — segments of different tubes that are physically close (coupling).
- **Target** = COMSOL absorption spectrum on an auto-detected union frequency grid (length `G`),
  with per-device masks for bins a device doesn't cover.

## Idea (Representation #2) — two-graph GNN

A second, richer formulation (same targets/training/analysis). Each device becomes **two
graphs** packed in one `HeteroData`, fused by concatenation:

- **Graph 1 — tube shape** (`vertex` nodes, `(vertex,seg,vertex)` edges, intra-tube only):
  node = a fold **vertex** (feature = absolute position); edge = the straight **segment**
  between consecutive vertices of the *same* tube (feature `[p1−p0, p0, p1]`). Rebuilds each
  tube's 3-D path.
- **Graph 2 — surface coupling** (`surface` nodes, `(surface,couple,surface)` edges): node =
  a planar B-rep **face**, encoded by **DeepSets** over its (variable-count, masked) boundary
  vertices + exact `[area, normal, centroid, perimeter, vcount, bbox_extents]`; edge = a
  facing-surface **coupling** (`[gap_vec, gap_dist, overlap_area, coupling_type]`), found where
  two faces are anti-parallel, thin-gap, non-lumen ("no air between"), and overlapping.
  Couplings are **intra-tube** (same folded tube's adjacent arms) or **inter-tube**.

Model: two GINEConv streams (edge-feature-aware) → global mean+max pool each → concat → head.

## Layout

```
data/    schema, device discovery, targets (auto grid+mask), fold coords (step | scalar),
         STEP centerline extractor (step_extract) + STEP face/coupling extractor (face_extract)
representations/  feature registry (base) + seg_graph (Rep-1) + dual_graph (Rep-2, HeteroData)
models/  base factory + rgnn (RGCNConv) + dualgnn (two-stream GINEConv + DeepSets)
training/ dataset (per-fold standardization, both graph types), losses, metrics, train (LOO), analyze
configs/ seg_rgnn.yaml  dual_gnn.yaml     run.py   requirements.txt   results/<run>/
```

## Run

```bash
pip install -r gnn/requirements.txt

# method #1 (Rep-1) — real geometry (needs cadquery-ocp); writes gnn/results/seg_rgnn/
python -m gnn.run --config gnn/configs/seg_rgnn.yaml

# method #2 (two-graph GNN) — writes gnn/results/dual_gnn/ (needs cadquery-ocp + shapely)
python -m gnn.run --config gnn/configs/dual_gnn.yaml

# pipeline smoke test with the scalar geometry stand-in (no CAD kernel needed)
python -m gnn.run --config gnn/configs/seg_rgnn.yaml --set source=scalar run_name=smoke epochs=60

# extract/inspect surfaces + couplings for one device (sanity report; caches to results/cache/)
python -m gnn.data.face_extract AI2026_Trial_Ver1R-2

# inspect a STEP's topology (helps tune extraction)
python -m gnn.data.step_inspect AI2026_Trial_Ver1R-1

# visually verify the footprint wall-split for one tube (labels the two curly walls)
python -m gnn.data.plot_footprint AI2026_Trial_Ver1R-4 58 -o /tmp/air058.png
```

Outputs per run in `gnn/results/<run>/`: `learning_curves.png`, `predictions.png`,
`per_frequency_error.png`, `metrics.csv`, `summary.md`, `train_output.pt`.

## Status / caveats

- Trains leave-one-out over the available devices (currently 6). With so few samples this is a
  **pipeline/overfit test**, not an accurate surrogate — success = it runs end-to-end and produces
  plausible curves. Scales automatically as `(step, txt)` device folders are added.
- **`source=brep` (default) extracts the centerline EXACTLY from the B-rep** (`data/brep_extract.py`),
  no voxels. Each tube is a Z-extruded constant-section duct; its footprint face's boundary loop is
  the two parallel channel walls (n/2 vertices each). Split the loop in half at the open cap (the
  min-Y along-X edge) and average the two walls vertex-by-vertex → the exact centerline (length
  matches the footprint quadratic to **0.00%**, paired walls perfectly parallel), in ~0.5–2 s/device.
  The legacy voxel/skeleton method is still available as `source=step` (approximate).
- `tube_geometry.csv` was regenerated from the B-rep (length = footprint quadratic,
  cross-section = width×thickness); every row satisfies `cross_section×length == volume`. It had
  13 wrong Ver1R-4 tubes (a 5×10 rectangle mislabeled as a 10×10 square); the bad version is kept as
  `tube_geometry.csv.bak_wrong`.

## Results so far (leave-one-out, n=6)

| run | config | mean MAE | Ver2 MAE | Ver1R MAE |
|---|---|---|---|---|
| `seg_rgnn` | method #1 (segment nodes) | 0.149 | 0.058 / 0.060 | 0.17–0.23 |
| `seg_rgnn_res` | method #1 + resonance | 0.142 | 0.055 / 0.039 | 0.16–0.23 |
| `dual_gnn` | method #2 (two graphs, absolute) | 0.138 | 0.039 / 0.039 | 0.16–0.22 |
| `dual_gnn_inv` | method #2, translation+rotation invariant | 0.144 | ~0.05 | ~0.17–0.24 |
| **`dual_gnn_res`** | **method #2 + resonance (best)** | **0.128** | **0.016 / 0.017** | 0.16–0.22 |

**Physics resonance features** (per-tube quarter-wave `f_n` on nodes + a device "resonance comb"
at the head; `gnn/data/resonance.py`) help **both** architectures on **Ver2** (where isolated-tube
resonances match the peaks) and are neutral on **Ver1R**. Ver1R are *broadband* absorbers with
dense ripple — no sparse peaks to place — so the comb can't help; the residual Ver1R error is
device-specific ripple that is **not learnable from n=6** (a data ceiling, not a model gap).
**Invariance** (`invariant: true`) is proven exact but was neutral-to-slightly-negative here, since
the devices share a frame and pose was never the bottleneck. Enable resonance via
`configs/*_res.yaml`; toggle `add_resonance` / `invariant` / `graph1_resonance` in any config.

## Method #3 — layer-stack CNN (`layer_cnn`)

`representations/voxel.py` + `models/cnn.py`. Every solid is an exact Z-extruded prism and the
tubes form uniform-pitch **layers**, so a device is losslessly a **z-ordered stack of layer
images**: per layer, one binary air-occupancy image (0.25 mm px — resolves the 3 tube widths at
18/20/32 px) plus per-layer scalars `[thickness, z_center]`. A shared 2D CNN encodes each layer;
a **local 1D conv along z** then fuses neighbouring layers.

The z-stage is deliberately **order-aware and locality-biased**: adjacent layers share one 1 mm
partition and couple through it, distant layers do not, so a stacked local kernel makes influence
decay with z-distance. This is the point of the method — Ver2-1a and Ver2-1b are the *same* set of
layer images in a *different* order (1b = 1a's layers unweaved), so a permutation-invariant model
maps them to identical outputs. `dual_gnn_res` does exactly that and **copies the twin**
(`corr(Δtrue,Δpred) = −0.64`); the CNN's order-sensitivity is checked with
`python -m gnn.training.twin_check --run layer_cnn`. Interaction range is tunable
(`z_kernel`,`z_depth`; `z_depth: 1` = strictly nearest-neighbour, a clean ablation).

**v2 upgrade (§3.3.2, `layer_cnn_v2`, `add_gaps`+`rich_pool`).** Two additions, gated behind
config flags so the baseline stays reproducible:
- **Inter-layer gap features** — the per-layer scalars gain `[g⁻, g⁺]`, the partition gaps to the
  layers below/above (`g_{i,i+1} = (z_{i+1}−t_{i+1}/2) − (z_i+t_i/2)`), so a layer's vector is
  `[e_i, z_i, t_i, g⁻, g⁺]`. Boundary layers use 0 (no neighbour). Gaps are derived at build time
  from the cached `[thickness, z_center]` — no cache rebuild.
- **Richer pooling** — device vector `[h_front, h_back, MeanPool, MaxPool, AttentionPool]` (5·hidden)
  instead of mean+max (2·hidden): the first/last physical layer explicitly, plus a learned
  attention over layers. The 3× 1D conv along z (kernel 3 → i−3…i+3) already matched the spec, so
  it is unchanged.

Inspect / verify:
```bash
python -m gnn.representations.voxel                      # build + shapes for all devices
python -m gnn.representations.plot_slices AI2026_Trial_Ver1R-4   # layer images + outlines
python -m gnn.run --config gnn/configs/layer_cnn.yaml    # (and layer_cnn_v2.yaml)
```

## Method #1 v2 — edge-conditioned segment GNN (`seg_egnn`)

`representations/seg_graph_v2.py` + `models/seg_egnn.py`. Same segment=node idea as Rep-1, but
with **physical node features** and **three typed edges that each carry their own edge-feature
vector, read by an edge-conditioned `GINEConv` per relation** (Rep-1's `RGCNConv` *ignored* edge
features entirely). Packed as a `HeteroData` with a single `seg` node type.

- **Node** (12-d): `[c(3), d̂(3), L, w, h, D_h, s, L_tube]` — center, unit open→end direction,
  segment length, in-plane width, z-thickness, hydraulic diameter, open-cap→center distance,
  whole-tube length. (Width/thickness are now carried on `SegmentRecord`; `brep_extract` already
  computed them.)
- **Path edge** (6-d, adjacent same-tube): `[Δc, d, cosθ, τ]` — τ = z-component of d̂_i×d̂_j
  (turn sign; folds here are 90° so cosθ≈0 and τ=±1 carries the bend).
- **Opening edge** (7-d, segment-0 nodes within cutoff): `[Δo, d_open, Δz, A_a, A_b]` — coupling
  of tube inlets on the shared y=0 plane.
- **Spatial edge** (7-d, non-adjacent intra + inter, within cutoff): `[Δc, d_min, cosθ, r_overlap,
  Δz]` — `d_min` = exact segment-segment min distance; `r_overlap` = project j onto d̂_i, interval
  overlap / min(L_i,L_j) (parallel-alongside→1, perpendicular→~0, no threshold).

Cutoffs are a physical proximity gate (`open_cutoff_scale`, `spatial_cutoff_scale`): base =
`√(median area) + partition` (wall-adjacent duct separation), scaled to catch diagonal neighbours.
The spatial gate uses `d_min` (via a sampled-point KD-tree superset then exact filter), so long
offset-parallel tubes aren't missed. **Normalization** (per fold, training devices only) standardizes
only the physical mm/mm² columns; unit/bounded features (`d̂, cosθ, τ, r_overlap`) are left raw —
the raw-column constants live in `seg_graph_v2.py` so builder and scaler can't drift.

Inspect / verify:
```bash
python -m gnn.representations.seg_graph_v2               # per-device node/edge counts + sanity
python -m gnn.run --config gnn/configs/seg_egnn.yaml     # (and seg_egnn_res.yaml)
```

## Evaluation protocol — nested LOO (leak fixed)

Earlier runs used the held-out **test** device as the early-stopping validation set, so every
reported MAE was the best of ~300 epochs *chosen by peeking at the answer*. The loop
(`training/train.py`) now does **nested LOO**: an inner LOO over the 5 training devices picks an
epoch budget, then the model retrains on all 5 for that budget with **no** early stopping and
predicts the untouched test device. Controlled by `inner_folds` (default 5). The MAEs in the
first table predate this fix (leaky) and are **not comparable** to the honest numbers below.

### Honest nested-LOO results (leak-free; `results/method_comparison.csv`)

| run (honest protocol) | mean MAE | mean RMSE | Ver2 MAE | Ver1R MAE |
|---|---|---|---|---|
| **`seg_egnn` (Method #1 v2, best)** | **0.155** | **0.186** | 0.088 | **0.188** |
| `dual_gnn_res_nested` | 0.164 | 0.199 | 0.092 | 0.200 |
| `seg_rgnn_nested` | 0.168 | 0.204 | 0.116 | 0.195 |
| `seg_egnn_res` (v2 + resonance) | 0.175 | 0.212 | 0.131 | 0.197 |
| `layer_cnn_v2` (Method #3 v2, §3.3.2) | 0.178 | 0.214 | 0.146 | 0.194 |
| `layer_cnn` (Method #3) | 0.180 | 0.217 | 0.149 | 0.195 |
| `seg_rgnn_res_nested` | 0.186 | 0.223 | 0.160 | 0.199 |

**`layer_cnn_v2`** (gap features + `[front, back, mean, max, attention]` pooling) edges past the
baseline `layer_cnn` on every metric (0.178 vs 0.180 MAE, 0.214 vs 0.217 RMSE; Ver2 0.146 vs
0.149), but the gain is marginal and it stays the weakest of the four method families — the z-order
signal it targets is still below the n=6 noise floor. Its twin `corr(Δtrue,Δpred) = −0.19`
(unchanged from baseline): the richer pooling does not resolve the 0.010 pair either.

**`seg_egnn` is the best honest model on every metric**, including **Ver1R** (the hard broadband
devices where the Ver2 twin-leakage can't help) — so the richer physical node features + the three
edge-conditioned relations are a genuine gain, not a twin artifact. Unlike the older architectures,
**resonance features HURT here** (`seg_egnn_res` 0.175 > `seg_egnn` 0.155): the 12-d physical node
vector already encodes per-tube length (`L_tube`, `s`), so the quarter-wave comb is redundant and
just adds noise at n=6. On the twin test `seg_egnn`'s
`corr(Δtrue,Δpred) = −0.14` (vs `dual_gnn_res` −0.64, `layer_cnn` −0.19): it largely **stops
copying the twin** but still does not positively resolve the 0.010 pair difference — model error
(~0.09) is ~9× that signal, the same n=6 data ceiling seen elsewhere. Check with
`python -m gnn.training.twin_check --run seg_egnn`.

## Extending (planned)

- **Idea #3** — ✅ built (`voxel` + `layercnn`, above).
- **Hierarchical fusion** — let Graph-2 coupling edges modulate Graph-1 tube embeddings before
  pooling (drop-in alternative to the current concat fusion in `models/dualgnn.py`).

## Using this repository

This repository is the `gnn` package itself. Clone it as `gnn/` next to the device folders:

```bash
cd code
git clone https://github.com/liuyuxiang92/metamaterials.git gnn
python -m pip install -r gnn/requirements.txt
PYTHONPATH=. python -m gnn.run --config gnn/configs/seg_egnn.yaml       # segment GNN v2
PYTHONPATH=. python -m gnn.run --config gnn/configs/dual_gnn_res.yaml   # dual GNN
PYTHONPATH=. python -m gnn.run --config gnn/configs/layer_cnn_v2.yaml   # layer CNN v2
```
