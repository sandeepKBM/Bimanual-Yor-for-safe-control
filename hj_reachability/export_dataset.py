#!/usr/bin/env python3
"""Emit a small bimanual arm-arm-avoidance DeepReach bootstrap dataset.

Adapted from the format written by
``ur5e_benchmark.grasp_boundary.deepreach_export.DeepReachBootstrapDatasetExporter``
(train/val/test CSVs + feature_schema.json + label_policy.json +
split_manifest.json + dataset_description.json + dataset_manifest.json +
README.md), but the *source* of rows here is uniform random sampling over
``dynamics.BimanualPairAvoidance.state_test_range()`` (there are no recorded
episodes yet -- see hj_reachability/README.md), not recorded rollouts.

Split assignment reuses the ur5e exporter's deterministic sha1-hash-of-key
scheme (``_sha_split`` there) so re-running with the same --seed reproduces
the same train/val/test membership.

Usage (smoke, seconds):
    python export_dataset.py --num-samples 200 --output-root /tmp/bimanual_hj_smoke

Usage (later, real dataset -- still CPU/seconds-scale, this is not training):
    python export_dataset.py --num-samples 200000 \\
        --output-root hj_reachability/datasets/bimanual_yor_v1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dynamics import BimanualPairAvoidance, BimanualPairAvoidanceConfig


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha_split(split_seed: int, key: str) -> float:
    """Verbatim port of DeepReachBootstrapDatasetExporter._sha_split."""
    digest = hashlib.sha1(f"{split_seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def assign_split(split_seed: int, sample_id: str, train_fraction: float, val_fraction: float) -> str:
    value = _sha_split(split_seed, sample_id)
    if value < train_fraction:
        return "train"
    if value < train_fraction + val_fraction:
        return "val"
    return "test"


def build_rows(
    dynamics: BimanualPairAvoidance,
    *,
    num_samples: int,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    task_region_frac: float = 0.0,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    if task_region_frac > 0.0:
        states = dynamics.sample_states_mixed(num_samples, rng, task_region_frac=task_region_frac)
    else:
        states = dynamics.sample_states(num_samples, rng)
    rows = []
    for idx in range(num_samples):
        state = states[idx]
        min_dist = dynamics.min_pair_distance(state)
        # l_value = dynamics.boundary_fn(state) == the FAITHFUL proxy's h at
        # the worst keypoint pair (per-pair radius baked in) -- NOT
        # `min_dist - combined_radius` any more (that assumed one fixed
        # scalar combined radius, which the ~25-keypoint-per-arm proxy no
        # longer has; see dynamics.py/mujoco_fk.py 2026-09-06 upgrade notes).
        l_value = dynamics.boundary_fn(state)
        target_is_safe = 1 if l_value > 0.0 else 0
        # DeepReach bootstrap sign convention (safe_when_leq_zero): mirrors
        # ur5e_benchmark's label_policy.json deepreach_sign_convention block,
        # which is the OPPOSITE sign of dynamics.boundary_fn's l(x)>0=safe.
        target_value = -1.0 if target_is_safe else 1.0
        sample_id = f"sample_{idx:08d}"
        split = assign_split(seed, sample_id, train_fraction, val_fraction)
        row = {
            "sample_id": sample_id,
            "split": split,
            "min_pair_distance": round(min_dist, 6),
            "l_value": round(l_value, 6),
            "target_value": target_value,
            "target_is_safe": target_is_safe,
        }
        for name, value in zip(dynamics.state_feature_order, state.tolist()):
            row[name] = round(float(value), 6)
        rows.append(row)
    return rows


def write_dataset(rows: list[dict], dynamics: BimanualPairAvoidance, *, output_root: Path, args: argparse.Namespace) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)

    csv_fieldnames = [
        "sample_id",
        "split",
        *dynamics.state_feature_order,
        "min_pair_distance",
        "l_value",
        "target_value",
        "target_is_safe",
    ]
    split_rows = {"train": [], "val": [], "test": []}
    for row in rows:
        split_rows[row["split"]].append(row)
    for split_name, split_row_list in split_rows.items():
        with (output_root / f"{split_name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_fieldnames)
            writer.writeheader()
            for row in split_row_list:
                writer.writerow(row)

    achieved_bounds = {}
    for j, name in enumerate(dynamics.state_feature_order):
        values = [row[name] for row in rows]
        achieved_bounds[name] = {"min": min(values), "max": max(values)}

    feature_schema = {
        "created_at": _utc_now_iso(),
        "feature_set": "bimanual_pair_avoidance_joint_state_v0",
        "state_feature_order": dynamics.state_feature_order,
        "state_dim": dynamics.state_dim,
        "metadata_columns": ["sample_id", "split", "min_pair_distance", "l_value"],
    }
    (output_root / "feature_schema.json").write_text(json.dumps(feature_schema, indent=2, sort_keys=True))

    label_policy = {
        "version": "bimanual_hj_bootstrap_v1_faithful_proxy",
        "source": f"mixed sampling: {getattr(args, 'task_region_frac', 0.0):.2f} task-region "
        "(both EEs near the shared workspace point) + rest uniform joint-state (no recorded "
        "rollouts yet -- see README.md, dynamics.sample_states_mixed)",
        "boundary_fn": "l(x) = h at the worst (min-h) FAITHFUL ~25-keypoint-per-arm proxy pair "
        "(per-pair radius baked in: dist - (r_ego+r_int+d_min)), IDENTICAL to "
        "experiments/ab_smooth_vs_bangbang/sim_common.py's h (dynamics.boundary_fn -> "
        "mujoco_fk.MuJoCoFKOracle.boundary_fn). Superseded the old 3-point-proxy "
        "`min_pair_distance - combined_radius` form (single fixed scalar radius).",
        "deepreach_sign_convention": {
            "safe_leq_zero": True,
            "unsafe_gt_zero": True,
            "safe_target": -1.0,
            "unsafe_target": 1.0,
            "note": "target_value/target_is_safe are the negation of dynamics.boundary_fn's l(x) sign "
            "(l(x)>0 is safe in dynamics.py; target_value<=0 is safe in this CSV, matching "
            "ur5e_benchmark's DeepReach bootstrap convention).",
        },
        "dynamics_config": asdict(dynamics.cfg),
    }
    (output_root / "label_policy.json").write_text(json.dumps(label_policy, indent=2, sort_keys=True, default=list))

    split_manifest = {
        "split_config": {
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "split_seed": args.seed,
        },
        "counts": {split: len(v) for split, v in split_rows.items()},
    }
    (output_root / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, sort_keys=True))

    dataset_description = {
        "data_format": {
            "supervised_files": {
                "train": "train.csv",
                "val": "val.csv",
                "test": "test.csv",
                "target_column": "target_value",
                "safe_indicator_column": "target_is_safe",
            },
            "trajectory_files": {
                "root": None,
                "state_order": dynamics.state_feature_order,
                "note": "No trajectories yet -- rows are i.i.d. sampled states, not rollouts. "
                "temporal_consistency training is not available until real MuJoCo "
                "rollouts are collected (see sim_common.py-style episode capture).",
            },
        },
        "achieved_bounds": achieved_bounds,
    }
    (output_root / "dataset_description.json").write_text(json.dumps(dataset_description, indent=2, sort_keys=True))

    manifest = {
        "created_at": _utc_now_iso(),
        "output_root": str(output_root.resolve()),
        "num_samples": len(rows),
        "split_row_counts": {split: len(v) for split, v in split_rows.items()},
        "class_counts": {
            "safe": sum(1 for r in rows if r["target_is_safe"] == 1),
            "unsafe": sum(1 for r in rows if r["target_is_safe"] == 0),
        },
        "state_dim": dynamics.state_dim,
        "args": vars(args),
    }
    (output_root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))

    readme = "\n".join(
        [
            "# Bimanual Arm-Arm Avoidance HJ Bootstrap Dataset (scaffold, v0)",
            "",
            f"Created at: `{_utc_now_iso()}`",
            "",
            "Generated by `hj_reachability/export_dataset.py` from uniform random",
            "joint-state samples scored by `hj_reachability/dynamics.py`'s",
            "`BimanualPairAvoidance.boundary_fn` (analytic sphere/capsule proxy, NOT",
            "recorded rollouts). See `hj_reachability/README.md` for status/next steps.",
            "",
            "Contents:",
            "- `train.csv`, `val.csv`, `test.csv`: state -> target_value/target_is_safe rows",
            "- `feature_schema.json`: state feature order",
            "- `label_policy.json`: sign convention and dynamics config used to generate labels",
            "- `dataset_description.json`, `dataset_manifest.json`, `split_manifest.json`",
        ]
    )
    (output_root / "README.md").write_text(readme + "\n", encoding="utf-8")

    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-samples", type=int, default=200, help="total sampled states to generate")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--u-bar-ego", type=float, default=BimanualPairAvoidanceConfig().u_bar_ego)
    p.add_argument("--u-bar-int", type=float, default=BimanualPairAvoidanceConfig().u_bar_int)
    p.add_argument("--d-min", type=float, default=BimanualPairAvoidanceConfig().d_min)
    p.add_argument(
        "--task-region-frac", type=float, default=0.5,
        help="fraction of samples drawn from the task-region sampler (both EEs near the "
        "shared workspace point, mirroring COLLISION_FIDELITY.md) instead of uniform "
        "joint-space sampling; 0.0 reproduces the old pure-uniform behavior.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BimanualPairAvoidanceConfig(u_bar_ego=args.u_bar_ego, u_bar_int=args.u_bar_int, d_min=args.d_min)
    dynamics = BimanualPairAvoidance(cfg)
    rows = build_rows(
        dynamics,
        num_samples=args.num_samples,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        task_region_frac=args.task_region_frac,
    )
    manifest = write_dataset(rows, dynamics, output_root=args.output_root, args=args)
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
