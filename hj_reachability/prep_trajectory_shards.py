#!/usr/bin/env python3
"""CPU data-prep: emit trajectory-shard files the DeepReach CartPoleDataset
loader requires, from the i.i.d. bimanual_yor_v2 CSV rows.

Why this exists
---------------
The DeepReach trajectory-backed loader (utils/dataio.py CartPoleDataset) is
used for BimanualArmArmAvoidance. Its constructor UNCONDITIONALLY requires
`<data_root>/trajectories/sequence_*.txt` files (each row = one comma-separated
`state_dim`-vector), even for training_objective=supervised_bootstrap. In
bootstrap mode the sampled HJ-collocation batch drawn from these files is
DISCARDED (only the supervised CSV MSE drives the loss), so the states here
serve only as a legal, in-distribution collocation pool. The real supervised
labels are still read directly from train.csv via supervised_labels_file.

This is a one-time CPU prep (infra rule: prep shards on CPU, ship to GPU). It
does NOT run the MuJoCo oracle -- it just reshapes existing CSV states into the
sequence-file layout the loader parses with `[float(x) for x in line.split(",")]`.

Usage:
  python3 prep_trajectory_shards.py \
      --data-root datasets/bimanual_yor_v2 [--split train] [--rows-per-shard 512]
"""
import argparse
import csv
import json
import os


def load_state_order(data_root):
    schema_path = os.path.join(data_root, "feature_schema.json")
    if os.path.exists(schema_path):
        with open(schema_path) as f:
            order = list(json.load(f).get("state_feature_order") or [])
        if order:
            return order
    desc_path = os.path.join(data_root, "dataset_description.json")
    with open(desc_path) as f:
        desc = json.load(f)
    return list((((desc.get("data_format") or {}).get("trajectory_files") or {}).get("state_order")) or [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="train", help="CSV split to shard (train/val/test)")
    ap.add_argument("--rows-per-shard", type=int, default=512)
    args = ap.parse_args()

    data_root = os.path.abspath(args.data_root)
    order = load_state_order(data_root)
    if not order:
        raise SystemExit(f"Could not determine state_feature_order under {data_root}")

    csv_path = os.path.join(data_root, f"{args.split}.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"Missing {csv_path}")

    traj_dir = os.path.join(data_root, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    n_shards = (n + args.rows_per_shard - 1) // args.rows_per_shard
    for s in range(n_shards):
        chunk = rows[s * args.rows_per_shard : (s + 1) * args.rows_per_shard]
        out = os.path.join(traj_dir, f"sequence_{s:05d}.txt")
        with open(out, "w") as g:
            for row in chunk:
                g.write(",".join(f"{float(row[name]):.6f}" for name in order) + "\n")
    print(f"[prep] wrote {n_shards} shard files ({n} states, {args.rows_per_shard}/shard) -> {traj_dir}")


if __name__ == "__main__":
    main()
