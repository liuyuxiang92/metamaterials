"""Nested leave-one-out training loop for the folded-tube spectrum surrogate.

For each held-out TEST device: pick an epoch budget by inner LOO over the 5 training devices
(early-stopping on an inner validation device the test device never touches), then retrain on
all 5 for that many epochs with NO early stopping and predict the test device. This removes the
earlier leak where the test device doubled as the early-stopping validation set — every
reported number was previously the best of ~300 epochs chosen by peeking at the answer.

Saves per-fold loss histories and the held-out prediction to
``gnn/results/<run>/train_output.pt`` for the analysis stage.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from ..data.devices import GNN_DIR
from ..models.base import build_model
from .dataset import build_graphs, collate, fit_scaler
from .losses import masked_mse

DEFAULTS = dict(
    run_name="seg_egnn", representation="seg_graph_v2", model="segegnn", source="brep",
    feature_names=("abs_endpoints", "cross_section"), coupling_scale=1.5,
    open_cutoff_scale=1.5, spatial_cutoff_scale=1.5,
    invariant=False, add_resonance=False, resonance_k=3, resonance_mode="quarter",
    resonance_sigma=40.0, graph1_resonance=False,
    px_mm=0.25, z_kernel=3, z_depth=3, z_posenc=False, add_gaps=False, rich_pool=False,
    hidden=64, num_layers=3, dropout=0.3,
    lr=1e-3, weight_decay=1e-4, epochs=400, patience=80, inner_folds=5,
    seed=0, device="auto",
)


def pick_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build(cfg, sample, G):
    """Construct the configured model, deriving input dims from a built graph."""
    name = cfg["model"]
    if name == "dualgnn":
        from ..representations.dual_graph import model_dims
        return build_model("dualgnn", G=G, dims=model_dims(sample), hidden=cfg["hidden"],
                           num_layers=cfg["num_layers"], dropout=cfg["dropout"])
    if name == "segegnn":
        from ..representations.seg_graph_v2 import model_dims
        return build_model("segegnn", G=G, dims=model_dims(sample), hidden=cfg["hidden"],
                           num_layers=cfg["num_layers"], dropout=cfg["dropout"])
    if name == "layercnn":
        from ..representations.voxel import model_dims
        return build_model("layercnn", G=G, dims=model_dims(sample), hidden=cfg["hidden"],
                           z_kernel=cfg["z_kernel"], z_depth=cfg["z_depth"],
                           z_posenc=cfg["z_posenc"], dmax=int(sample.vox.size(0)),
                           dropout=cfg["dropout"], rich_pool=cfg["rich_pool"])
    raise ValueError(f"Unknown model {name!r}")


def _train(model, train_batch, cfg, max_epochs, val_batch=None):
    """One training run. With ``val_batch`` -> early-stop, return the best epoch and its state.
    Without -> train exactly ``max_epochs`` and return the final state. Never selects on a
    batch that isn't passed in here, so the caller controls what the model is allowed to see."""
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    tr_hist, va_hist = [], []
    best_val, best_state, best_epoch, wait = float("inf"), None, max_epochs, 0

    for ep in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss = masked_mse(model(train_batch), train_batch.y, train_batch.mask)
        loss.backward()
        opt.step()
        tr_hist.append(float(loss.detach()))

        if val_batch is None:
            continue
        model.eval()
        with torch.no_grad():
            vloss = masked_mse(model(val_batch), val_batch.y, val_batch.mask)
        va_hist.append(float(vloss.detach()))
        if vloss < best_val - 1e-6:
            best_val, best_state, best_epoch, wait = float(vloss), \
                copy.deepcopy(model.state_dict()), ep + 1, 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break

    if val_batch is not None and best_state is not None:
        model.load_state_dict(best_state)
    return dict(best_val=best_val, best_epoch=best_epoch, tr_hist=tr_hist, va_hist=va_hist)


def _fit_batch(devs, graphs, cfg, device):
    """Scaler fit on ``devs`` + their collated, standardized training batch on ``device``."""
    scaler = fit_scaler([graphs[d] for d in devs])
    batch = collate([scaler.apply(graphs[d]) for d in devs]).to(device)
    return scaler, batch


def _train_one_fold(held: str, graphs, cfg, device) -> dict:
    """Nested LOO for one outer (test) fold — the test device is never used for selection."""
    G = graphs[held].y.shape[1]
    pool = [d for d in graphs if d != held]
    torch.manual_seed(cfg["seed"])

    # ---- inner LOO over the training pool: choose an epoch budget without the test device ----
    n_inner = max(1, min(int(cfg["inner_folds"]), len(pool)))
    inner_vals = pool[:n_inner]
    best_epochs, last_inner = [], None
    for v in inner_vals:
        tr = [d for d in pool if d != v]
        scaler, tb = _fit_batch(tr, graphs, cfg, device)
        vb = collate([scaler.apply(graphs[v])]).to(device)
        model = _build(cfg, tb, G).to(device)
        r = _train(model, tb, cfg, cfg["epochs"], val_batch=vb)
        best_epochs.append(r["best_epoch"])
        last_inner = r
    budget = int(np.median(best_epochs))

    # ---- final fit on ALL 5 training devices for that budget, no early stopping ----
    scaler, tb = _fit_batch(pool, graphs, cfg, device)
    model = _build(cfg, tb, G).to(device)
    fin = _train(model, tb, cfg, max(budget, 1))

    model.eval()
    hb = collate([scaler.apply(graphs[held])]).to(device)
    with torch.no_grad():
        pred = model(hb).detach().cpu().numpy().ravel()

    return {
        "pred": pred,
        "y": graphs[held].y.numpy().ravel(),
        "mask": graphs[held].mask.numpy().ravel(),
        "train_hist": fin["tr_hist"],            # final-fit training curve
        "val_hist": last_inner["va_hist"],       # INNER validation curve (not the test device)
        "best_val": last_inner["best_val"],
        "epoch_budget": budget,
    }


def run(cfg: dict | None = None) -> Path:
    cfg = {**DEFAULTS, **(cfg or {})}
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = pick_device(cfg["device"])

    graphs, targets = build_graphs(
        representation=cfg["representation"], source=cfg["source"],
        feature_names=tuple(cfg["feature_names"]), coupling_scale=cfg["coupling_scale"],
        invariant=cfg["invariant"], add_resonance=cfg["add_resonance"],
        resonance_k=cfg["resonance_k"], resonance_mode=cfg["resonance_mode"],
        resonance_sigma=cfg["resonance_sigma"], graph1_resonance=cfg["graph1_resonance"],
        px_mm=cfg["px_mm"], add_gaps=cfg["add_gaps"],
        open_cutoff_scale=cfg["open_cutoff_scale"], spatial_cutoff_scale=cfg["spatial_cutoff_scale"],
    )
    print(f"Device: {device} | {len(graphs)} graphs | G={targets.G} | "
          f"repr={cfg['representation']} | model={cfg['model']} | source={cfg['source']}")

    try:
        folds = {h: _train_one_fold(h, graphs, cfg, device) for h in graphs}
    except (RuntimeError, NotImplementedError) as e:
        if device.type != "cpu":
            print(f"[warn] {device} backend failed ({e}); falling back to CPU")
            device = torch.device("cpu")
            folds = {h: _train_one_fold(h, graphs, cfg, device) for h in graphs}
        else:
            raise

    for h, r in folds.items():
        print(f"  held-out {h:28s} inner-val MSE={r['best_val']:.5f} "
              f"(epoch budget={r['epoch_budget']})")

    out_dir = GNN_DIR / "results" / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"grid": targets.grid, "folds": folds, "config": cfg}, out_dir / "train_output.pt")
    print(f"Saved -> {out_dir/'train_output.pt'}")
    return out_dir


if __name__ == "__main__":
    run({"source": "scalar", "run_name": "smoke", "epochs": 60, "patience": 30})
