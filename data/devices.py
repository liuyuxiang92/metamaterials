"""Discover device ``(STEP, target-txt)`` pairs under the repository root.

A "device" is a trial folder such as ``AI2026_Trial_Ver1R-1`` containing a ``.step``
file and a results ``.txt``. The folder name is the canonical device id, matching the
``device`` column in ``gnn/tube_geometry.csv`` and the per-device files in ``gnn/csv/``.
The pipeline is source-agnostic: point it at any folder of such trials and it scales.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# gnn/data/devices.py -> parents[2] == repo root (…/code)
REPO_ROOT = Path(__file__).resolve().parents[2]
GNN_DIR = REPO_ROOT / "gnn"
CSV_DIR = GNN_DIR / "csv"
TUBE_GEOMETRY_CSV = GNN_DIR / "tube_geometry.csv"


@dataclass(frozen=True)
class Device:
    device: str          # canonical id == trial folder name
    folder: Path
    step: Path
    target_txt: Path

    @property
    def csv(self) -> Path:
        """Per-device tube geometry CSV in gnn/csv/ (may not exist)."""
        return CSV_DIR / f"{self.device}.csv"


def _pick_target_txt(folder: Path) -> Path | None:
    txts = [p for p in folder.glob("*.txt") if p.name.lower() != "readme.txt"]
    if not txts:
        return None
    # Prefer an explicit *_Report.txt (Ver2), else the sole remaining txt (Ver1R export).
    reports = [p for p in txts if p.name.endswith("_Report.txt")]
    return reports[0] if reports else txts[0]


def discover_devices(root: Path | str = REPO_ROOT) -> dict[str, Device]:
    """Return {device_id: Device} for every trial folder with a STEP + target txt."""
    root = Path(root)
    devices: dict[str, Device] = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        steps = list(folder.glob("*.step"))
        target = _pick_target_txt(folder)
        if not steps or target is None:
            continue
        devices[folder.name] = Device(
            device=folder.name, folder=folder, step=steps[0], target_txt=target
        )
    return devices


if __name__ == "__main__":
    for name, dev in discover_devices().items():
        print(f"{name:28s}  step={dev.step.name:28s}  target={dev.target_txt.name}")
