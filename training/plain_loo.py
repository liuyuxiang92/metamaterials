"""Plain 6-fold LOO (NOT nested) — a diagnostic to compare stopping rules.

Standard leave-one-out: 5 devices train, the 6th is held out. This module deliberately does the
thing nested LOO avoids — it lets you use the held-out device as a validation set — so you can
SEE the difference between:

  Case A  early-stop on the held-out device   (peek at the answer -> optimistic / LEAKY)
          train on the 5, and per epoch measure the held-out device's MAE; keep the epoch with
          the lowest held-out MAE. This is the old leaky protocol, shown on purpose.

  Case B  fixed epoch budgets (honest)         (no peeking)
          train on the 5 for a FIXED number of epochs (no early stopping), evaluate the held-out
          device once. Sweep several budgets (e.g. 20/40/60/80/100) to watch MAE/RMSE evolve and
          where overfitting sets in.

Case A's number is what you could NOT know without the test labels; Case B is what an honest run
delivers at a pre-chosen budget. The gap between them is the size of the early-stopping leak.

    python -m gnn.training.plain_loo --config gnn/configs/seg_egnn.yaml
    python -m gnn.training.plain_loo --config gnn/configs/seg_egnn.yaml --epochs 20,40,60,80,100
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from ..data.devices import GNN_DIR
from .dataset import build_graphs, collate, fit_scaler
from .losses import masked_mse
from .metrics import spectrum_metrics
from .train import DEFAULTS, _build, pick_device


def _eval(model, batch, y, mask, grid) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        pred = model(batch).detach().cpu().numpy().ravel()
    m = spectrum_metrics(pred, y, mask, grid)
    return m["mae"], m["rmse"]


def _fit_batch(devs, graphs, device):
    sc = fit_scaler([graphs[d] for d in devs])
    return sc, collate([sc.apply(graphs[d]) for d in devs]).to(device)


def _fold_early(cfg, graphs, pool, held, G, grid, device):
    """Train on the 5 pool devices; early-stop on the HELD-OUT device's MAE (leaky). Returns the
    (mae, rmse, epoch) of the lowest-held-out-MAE checkpoint."""
    torch.manual_seed(cfg["seed"])
    sc, tb = _fit_batch(pool, graphs, device)
    hb = collate([sc.apply(graphs[held])]).to(device)
    y, mask = graphs[held].y.numpy().ravel(), graphs[held].mask.numpy().ravel()
    model = _build(cfg, tb, G).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best = (float("inf"), float("inf"), 0)   # (mae, rmse, epoch)
    wait = 0
    for ep in range(cfg["epochs"]):
        model.train(); opt.zero_grad()
        masked_mse(model(tb), tb.y, tb.mask).backward(); opt.step()
        mae, rmse = _eval(model, hb, y, mask, grid)       # peek at the held-out device
        if mae < best[0] - 1e-6:
            best, wait = (mae, rmse, ep + 1), 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    return best


def _fold_fixed(cfg, graphs, pool, held, G, grid, device, n_epochs):
    """Train on the 5 pool devices for exactly ``n_epochs`` (no early stopping); eval held once."""
    torch.manual_seed(cfg["seed"])
    sc, tb = _fit_batch(pool, graphs, device)
    hb = collate([sc.apply(graphs[held])]).to(device)
    y, mask = graphs[held].y.numpy().ravel(), graphs[held].mask.numpy().ravel()
    model = _build(cfg, tb, G).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    for _ in range(max(int(n_epochs), 1)):
        model.train(); opt.zero_grad()
        masked_mse(model(tb), tb.y, tb.mask).backward(); opt.step()
    return _eval(model, hb, y, mask, grid)


def run(cfg: dict, epochs_grid: list[int]) -> pd.DataFrame:
    cfg = {**DEFAULTS, **cfg}
    device = pick_device(cfg["device"])
    graphs, targets = build_graphs(
        representation=cfg["representation"], source=cfg["source"],
        feature_names=tuple(cfg["feature_names"]), coupling_scale=cfg["coupling_scale"],
        invariant=cfg["invariant"], add_resonance=cfg["add_resonance"],
        resonance_k=cfg["resonance_k"], resonance_mode=cfg["resonance_mode"],
        resonance_sigma=cfg["resonance_sigma"], graph1_resonance=cfg["graph1_resonance"],
        px_mm=cfg["px_mm"], add_gaps=cfg.get("add_gaps", False),
        open_cutoff_scale=cfg["open_cutoff_scale"], spatial_cutoff_scale=cfg["spatial_cutoff_scale"])
    G = targets.G
    print(f"plain 6-fold LOO | {cfg['run_name']} | repr={cfg['representation']} | device={device}")

    # ---- Case A: early-stop on the held-out device (leaky) ----
    a_mae, a_rmse, a_ep = [], [], []
    for held in graphs:
        pool = [d for d in graphs if d != held]
        mae, rmse, ep = _fold_early(cfg, graphs, pool, held, G, targets.grid, device)
        a_mae.append(mae); a_rmse.append(rmse); a_ep.append(ep)
    print("\n[Case A] early-stop on HELD-OUT device (peeks at test -> optimistic/LEAKY)")
    print(f"   mean MAE={np.mean(a_mae):.4f}  mean RMSE={np.mean(a_rmse):.4f}  "
          f"(best epochs: {a_ep})")

    # ---- Case B: fixed epoch budgets (honest) ----
    rows = []
    for n in epochs_grid:
        maes, rmses = [], []
        for held in graphs:
            pool = [d for d in graphs if d != held]
            mae, rmse = _fold_fixed(cfg, graphs, pool, held, G, targets.grid, device, n)
            maes.append(mae); rmses.append(rmse)
        rows.append(dict(epochs=n, mean_mae=np.mean(maes), mean_rmse=np.mean(rmses)))
    dfB = pd.DataFrame(rows)
    print("\n[Case B] FIXED epochs, no peeking (honest)")
    print(dfB.round(4).to_string(index=False))

    out = GNN_DIR / "results" / f"plain_loo_{cfg['run_name']}.csv"
    dfB_out = dfB.copy()
    dfB_out.attrs = {}
    dfB.round(4).to_csv(out, index=False)
    # append Case A as a labeled row for the record
    with open(out, "a") as f:
        f.write(f"early_stop_on_holdout,{np.mean(a_mae):.4f},{np.mean(a_rmse):.4f}\n")
    print(f"\nsaved -> {out}")
    return dfB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--epochs", default="20,40,60,80,100",
                    help="comma-separated fixed epoch budgets for Case B")
    ap.add_argument("--set", nargs="*", default=[], metavar="k=v")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text()) or {}
    for item in args.set:
        k, _, v = item.partition("=")
        cfg[k] = yaml.safe_load(v)
    grid = [int(x) for x in str(args.epochs).split(",")]
    run(cfg, grid)


if __name__ == "__main__":
    main()
