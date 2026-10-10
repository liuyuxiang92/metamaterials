"""Per-device spectrum metrics, evaluated over valid (masked) bins only.

All inputs are 1-D numpy arrays of length G plus the frequency grid; ``mask`` is 1 where
the device has ground truth. Returns a plain dict of scalars (Hz for frequency quantities).
"""
from __future__ import annotations

import numpy as np

_ABSORB_THRESH = 0.5  # "absorbing" if absorption exceeds this


def _bandwidth_hz(sig: np.ndarray, grid: np.ndarray) -> float:
    step = float(grid[1] - grid[0]) if grid.size > 1 else 1.0
    return float(np.count_nonzero(sig > _ABSORB_THRESH) * step)


def spectrum_metrics(pred: np.ndarray, y: np.ndarray, mask: np.ndarray,
                     grid: np.ndarray) -> dict[str, float]:
    v = mask > 0
    g, p, t = grid[v], pred[v], y[v]
    if g.size == 0:
        return {k: float("nan") for k in
                ("mae", "rmse", "in_band_mae", "peak_freq_err", "bandwidth_err")}

    err = p - t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    in_band = t >= _ABSORB_THRESH
    in_band_mae = float(np.mean(np.abs(err[in_band]))) if in_band.any() else float("nan")

    peak_freq_err = float(abs(g[int(np.argmax(p))] - g[int(np.argmax(t))]))
    bandwidth_err = float(abs(_bandwidth_hz(p, g) - _bandwidth_hz(t, g)))
    return {"mae": mae, "rmse": rmse, "in_band_mae": in_band_mae,
            "peak_freq_err": peak_freq_err, "bandwidth_err": bandwidth_err}
