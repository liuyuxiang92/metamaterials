"""Physics-informed resonance features from tube lengths.

A folded tube is a quarter-wave resonator (open inlet, closed end): it absorbs near
``f_n = (2n-1)*c/(4L)`` (odd harmonics). We use these analytic frequencies two ways:
  * per-tube ``f_n`` scalars (fed to the graphs), and
  * a device ``resonance_comb`` — a soft length-G spectrum with a bump at every tube's f_n,
    a physics prior on where absorption peaks belong (fed to the model head).

Lengths come from ``tube_geometry.csv`` (already loaded elsewhere). ``mode="half"`` switches to
an open-open half-wave model ``f_n = n*c/(2L)`` for the quarter-vs-half validation check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .devices import TUBE_GEOMETRY_CSV

C_MM_S = 343_000.0  # speed of sound, mm/s (343 m/s), so f is in Hz when L is in mm


def tube_resonances(device: str, k: int = 3, mode: str = "quarter") -> dict[int, np.ndarray]:
    """{tube_index: [f_1..f_k] Hz} from each tube's length."""
    df = pd.read_csv(TUBE_GEOMETRY_CSV)
    df = df[df["device"] == device]
    out: dict[int, np.ndarray] = {}
    for r in df.itertuples():
        L = float(r.length_mm)
        if L <= 0:
            continue
        if mode == "quarter":
            f = [(2 * n - 1) * C_MM_S / (4.0 * L) for n in range(1, k + 1)]
        else:  # half-wave
            f = [n * C_MM_S / (2.0 * L) for n in range(1, k + 1)]
        out[int(r.tube_index)] = np.asarray(f, dtype=np.float64)
    return out


def resonance_comb(device: str, grid: np.ndarray, k: int = 3, mode: str = "quarter",
                   sigma: float = 40.0) -> np.ndarray:
    """Length-G soft spectrum: a Gaussian bump at every tube's in-band f_n, summed."""
    grid = np.asarray(grid, dtype=np.float64)
    comb = np.zeros(grid.shape[0], dtype=np.float64)
    fmin, fmax = float(grid.min()), float(grid.max())
    for fs in tube_resonances(device, k, mode).values():
        for f in fs:
            if fmin <= f <= fmax:
                comb += np.exp(-0.5 * ((grid - f) / sigma) ** 2)
    m = comb.max()
    if m > 0:
        comb /= m
    return comb.astype(np.float32)
