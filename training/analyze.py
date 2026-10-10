"""Post-training analysis: turn saved LOO outputs into figures + a metrics report.

Reads ``gnn/results/<run>/train_output.pt`` and writes, in the same folder:
  learning_curves.png, predictions.png, per_frequency_error.png,
  metrics.csv, summary.md
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .metrics import spectrum_metrics


def _grid_shape(n: int) -> tuple[int, int]:
    ncols = math.ceil(math.sqrt(n))
    return math.ceil(n / ncols), ncols


def _learning_curves(folds: dict, path: Path):
    nrows, ncols = _grid_shape(len(folds))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for ax, (dev, r) in zip(axes.ravel(), folds.items()):
        ax.plot(r["train_hist"], label="train (final fit)")
        if r.get("val_hist"):
            ax.plot(r["val_hist"], label="inner val")
        ax.set_title(dev, fontsize=8)
        ax.set_xlabel("epoch"); ax.set_ylabel("masked MSE"); ax.legend(fontsize=7)
    for ax in axes.ravel()[len(folds):]:
        ax.axis("off")
    fig.suptitle("Learning curves — final-fit train vs inner-validation (nested LOO)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _predictions(grid: np.ndarray, folds: dict, path: Path):
    nrows, ncols = _grid_shape(len(folds))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for ax, (dev, r) in zip(axes.ravel(), folds.items()):
        v = r["mask"] > 0
        ax.plot(grid[v], r["y"][v], label="COMSOL", lw=1.5)
        ax.plot(grid[v], r["pred"][v], label="GNN", lw=1.2, ls="--")
        ax.set_title(dev, fontsize=8); ax.set_ylim(0, 1)
        ax.set_xlabel("freq (Hz)"); ax.set_ylabel("absorption"); ax.legend(fontsize=7)
    for ax in axes.ravel()[len(folds):]:
        ax.axis("off")
    fig.suptitle("Held-out prediction vs COMSOL")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _per_frequency_error(grid: np.ndarray, folds: dict, path: Path):
    G = grid.shape[0]
    acc = np.zeros(G); cnt = np.zeros(G)
    for r in folds.values():
        m = r["mask"] > 0
        acc[m] += np.abs(r["pred"][m] - r["y"][m]); cnt[m] += 1
    prof = np.divide(acc, cnt, out=np.full(G, np.nan), where=cnt > 0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(grid, prof)
    ax.set_title("Per-frequency mean |error| across held-out devices")
    ax.set_xlabel("freq (Hz)"); ax.set_ylabel("mean |absorption error|")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def analyze(out_dir: Path | str) -> Path:
    out_dir = Path(out_dir)
    blob = torch.load(out_dir / "train_output.pt", weights_only=False)
    grid, folds = np.asarray(blob["grid"]), blob["folds"]

    _learning_curves(folds, out_dir / "learning_curves.png")
    _predictions(grid, folds, out_dir / "predictions.png")
    _per_frequency_error(grid, folds, out_dir / "per_frequency_error.png")

    rows = []
    for dev, r in folds.items():
        m = spectrum_metrics(r["pred"], r["y"], r["mask"], grid)
        rows.append({"device": dev, **m})
    df = pd.DataFrame(rows).set_index("device")
    df.loc["mean"] = df.mean(numeric_only=True)
    df.loc["std"] = df.iloc[:-1].std(numeric_only=True)
    df.to_csv(out_dir / "metrics.csv")

    with open(out_dir / "summary.md", "w") as f:
        f.write(f"# Run `{out_dir.name}` — LOO summary\n\n")
        cfg = blob["config"]
        f.write(f"- devices: {len(folds)} | grid: {grid[0]:.0f}-{grid[-1]:.0f} Hz, G={grid.shape[0]}\n")
        f.write(f"- representation: `{cfg.get('representation', 'seg_graph')}` | "
                f"model: `{cfg.get('model', 'rgnn')}` | source: `{cfg.get('source')}`\n")
        if cfg.get("model", "rgnn") == "rgnn":
            f.write(f"- features: `{list(cfg.get('feature_names'))}`\n")
        f.write("\n")
        f.write("## Metrics (masked, held-out)\n\n")
        f.write(df.round(4).to_markdown() + "\n\n")
        f.write("## Figures\n\n")
        for p in ("learning_curves.png", "predictions.png", "per_frequency_error.png"):
            f.write(f"- ![{p}]({p})\n")
    print(f"Analysis written -> {out_dir}")
    print(df.round(4).to_string())
    return out_dir


if __name__ == "__main__":
    import sys
    from ..data.devices import GNN_DIR
    d = sys.argv[1] if len(sys.argv) > 1 else GNN_DIR / "results" / "smoke"
    analyze(d)
