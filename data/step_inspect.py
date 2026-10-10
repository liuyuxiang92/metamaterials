"""Diagnostic: report the topological structure of a device's STEP air-domain.

Used to design/verify ``step_extract`` — tells us how many solids/shells/faces the air
domain decomposes into (are tubes separate solids? one connected manifold?) and its bbox.

    python -m gnn.data.step_inspect [device_id]
"""
from __future__ import annotations

import sys

from .devices import discover_devices
from .step_extract import _load_shape, _iter, _bbox  # reuse loaders

try:
    from OCP.TopAbs import (TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE,
                            TopAbs_EDGE, TopAbs_VERTEX)
except ImportError:  # pragma: no cover
    from OCC.Core.TopAbs import (TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE,
                                 TopAbs_EDGE, TopAbs_VERTEX)


def inspect(device: str):
    devs = discover_devices()
    shape = _load_shape(devs[device].step)
    counts = {
        "solids": len(_iter(shape, TopAbs_SOLID)),
        "shells": len(_iter(shape, TopAbs_SHELL)),
        "faces": len(_iter(shape, TopAbs_FACE)),
        "edges": len(_iter(shape, TopAbs_EDGE)),
        "vertices": len(_iter(shape, TopAbs_VERTEX)),
    }
    (xmin, ymin, zmin), (xmax, ymax, zmax) = _bbox(shape)
    print(f"== {device} ==")
    for k, v in counts.items():
        print(f"  {k:9s}= {v}")
    print(f"  bbox: x[{xmin:.1f},{xmax:.1f}] y[{ymin:.1f},{ymax:.1f}] z[{zmin:.1f},{zmax:.1f}] mm")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else next(iter(discover_devices()))
    inspect(name)
