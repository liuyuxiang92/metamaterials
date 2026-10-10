"""Load absorption-spectrum targets and build an AUTO-SIZED common frequency grid.

Two on-disk formats are auto-detected per device:
  * COMSOL export (Ver1R ``*.txt``): ``%``-comment header, data cols ``freq, Reflc, Absorp``;
    target = the ``Absorp`` column (located by parsing the header, not a fixed index).
  * Report (Ver2 ``*_Report.txt``): 4 whitespace cols, no header; target = col 2 (COMSOL).

The union grid length ``G`` is whatever the supplied devices imply — never hardcoded. Each
device is placed on the grid and masked to the bins it actually covers, so shorter-range
devices (Ver2 today) and any future longer/finer device are handled by the same rule.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .devices import Device, discover_devices

_MULTISPACE = re.compile(r"\s{2,}")


def _read_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (freq, absorption) for one device, auto-detecting the file format."""
    comment_hdr: str | None = None
    rows: list[list[float]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("%"):
            if "Absorp" in line:  # COMSOL column header
                comment_hdr = line.lstrip("%").strip()
            continue
        parts = line.split()
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            continue  # skip any stray non-numeric line
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"No numeric data parsed from {path}")

    if comment_hdr is not None:  # COMSOL export -> find the Absorp column
        tokens = _MULTISPACE.split(comment_hdr)
        absorp_col = next(i for i, t in enumerate(tokens) if "Absorp" in t)
    else:  # Ver2 report: col1=freq, col2=COMSOL absorption
        absorp_col = 1
    return arr[:, 0], arr[:, absorp_col]


@dataclass
class TargetSet:
    grid: np.ndarray                 # (G,) shared frequency axis, Hz
    y: dict[str, np.ndarray]         # device -> (G,) absorption, 0 where missing
    mask: dict[str, np.ndarray]      # device -> (G,) 1.0 where device has ground truth

    @property
    def G(self) -> int:
        return int(self.grid.shape[0])


def _infer_step(freq: np.ndarray) -> float:
    d = np.diff(np.unique(freq))
    return float(np.min(d)) if d.size else 1.0


def load_targets(devices: dict[str, Device] | None = None) -> TargetSet:
    if devices is None:
        devices = discover_devices()
    spectra = {name: _read_spectrum(dev.target_txt) for name, dev in devices.items()}

    # --- pass 1: auto-detect the union grid ---
    step = min(_infer_step(f) for f, _ in spectra.values())
    fmin = min(float(f.min()) for f, _ in spectra.values())
    fmax = max(float(f.max()) for f, _ in spectra.values())
    grid = np.arange(fmin, fmax + step / 2.0, step)
    G = grid.shape[0]
    tol = step / 2.0

    # --- pass 2: place each device on the grid + build its mask ---
    y: dict[str, np.ndarray] = {}
    mask: dict[str, np.ndarray] = {}
    for name, (freq, absorp) in spectra.items():
        yy = np.zeros(G, dtype=np.float32)
        mm = np.zeros(G, dtype=np.float32)
        idx = np.round((freq - fmin) / step).astype(int)
        ok = (idx >= 0) & (idx < G) & (np.abs(grid[np.clip(idx, 0, G - 1)] - freq) < tol)
        yy[idx[ok]] = absorp[ok].astype(np.float32)
        mm[idx[ok]] = 1.0
        y[name], mask[name] = yy, mm
    return TargetSet(grid=grid, y=y, mask=mask)


if __name__ == "__main__":
    ts = load_targets()
    print(f"Auto grid: G={ts.G}  {ts.grid[0]:.0f}-{ts.grid[-1]:.0f} Hz  step={ts.grid[1]-ts.grid[0]:.0f}")
    for name in ts.y:
        cov = int(ts.mask[name].sum())
        print(f"  {name:28s} covered bins={cov:4d}/{ts.G}  "
              f"absorp[min,max]=[{ts.y[name][ts.mask[name]>0].min():.3f},"
              f"{ts.y[name][ts.mask[name]>0].max():.3f}]")
