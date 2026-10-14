"""Entry point: train FOUND FLOQ (ur5e_benchmark.floq.train.train_floq,
imported unmodified) on our bimanual arm-arm avoidance episodes.

Usage:
    python train_bimanual_floq.py --config configs/bimanual_floq_smoke.yaml \
        --output-dir outputs/runs/bimanual_floq_smoke

Config is a plain YAML dict consumed directly by
``ur5e_benchmark.floq.train.train_floq`` (same schema as
``configs/floq/floq_v1_*.yaml`` in the VLAism repo) -- see
hj_reachability/FLOQ_SETUP.md for field-by-field notes on our adaptation
(state/dynamics/boundary/discount).
"""
from __future__ import annotations

import argparse
import json
import os

import yaml

import paths  # noqa: F401  (sys.path setup for ur5e_benchmark.floq)
from ur5e_benchmark.floq import train_floq


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dry-run", action="store_true", help="single forward+backward+step, no epoch loop")
    p.add_argument("--device", default=None, help="override config['device'], e.g. cuda:0 or cpu")
    p.add_argument("--epochs", type=int, default=None, help="override config['epochs']")
    p.add_argument("--max-train-batches", type=int, default=None, help="override config['max_train_batches']")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.device is not None:
        config["device"] = args.device
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.max_train_batches is not None:
        config["max_train_batches"] = args.max_train_batches

    os.makedirs(args.output_dir, exist_ok=True)
    result = train_floq(config=config, output_dir=args.output_dir, dry_run=bool(args.dry_run))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
