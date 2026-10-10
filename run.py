"""CLI: train the LOO surrogate then run post-training analysis.

    python -m gnn.run --config gnn/configs/seg_egnn.yaml
    python -m gnn.run --config gnn/configs/seg_egnn.yaml --set source=scalar run_name=smoke

``--set k=v`` overrides any config key (values are parsed as YAML scalars/lists).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .training import train
from .training.analyze import analyze


def _coerce(v: str):
    return yaml.safe_load(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="k=v", help="override config keys")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text()) or {}
    for item in args.set:
        k, _, v = item.partition("=")
        cfg[k] = _coerce(v)

    out_dir = train.run(cfg)
    analyze(out_dir)


if __name__ == "__main__":
    main()
