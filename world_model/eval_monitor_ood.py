"""world_model/eval_monitor_ood.py -- OOD eval of an EXISTING trained
monitor head + WM checkpoint (no retraining, no fine-tuning).

Loads a frozen WorldModel checkpoint (`--wm-checkpoint`) and a frozen
MonitorHead checkpoint (`--head-checkpoint`, as saved by
`monitor_head.py --head-ckpt-out`) and evaluates BOTH, as-is, on every
episode in `--dreamer-fmt-dir` (no train/eval split -- the whole DR set is
held out relative to training). Reuses `monitor_head.py`'s own
`compute_latents_and_labels` / `precision_recall` / `roc_curve` /
`calibration_curve` / `save_metrics_plot` so the metric definitions are
IDENTICAL to the in-distribution 0.999-AUC run this is being compared
against -- no reimplementation drift.

Usage:
    CUDA_VISIBLE_DEVICES=<n> /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python \\
        -m world_model.eval_monitor_ood \\
        --dreamer-fmt-dir world_model/data/dreamer_fmt/cup_dr_test \\
        --collection-dir collection/data/cup_dr_test \\
        --wm-checkpoint world_model/logdir/wm_cup_full/best.pt \\
        --head-checkpoint world_model/logdir/wm_cup_full/monitor_head_cup_full.pt \\
        --spill-signal per_step --tag cup_dr_ood
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
import yaml

WORLD_MODEL_ROOT = Path(__file__).resolve().parent
if str(WORLD_MODEL_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(WORLD_MODEL_ROOT.parent))

from world_model.train_wm import split_episodes  # noqa: E402
from world_model.eval_wm import build_wm_from_checkpoint  # noqa: E402
from world_model.dataset_adapter import convert_dir  # noqa: E402
from world_model.monitor_head import (  # noqa: E402
    LABEL_NAMES,
    MonitorHead,
    calibration_curve,
    compute_latents_and_labels,
    precision_recall,
    roc_curve,
    save_metrics_plot,
    stack_split,
)

DREAMERV3_ROOT = "/common/users/ss5772/dreamerv3"
if DREAMERV3_ROOT not in sys.path:
    sys.path.insert(0, DREAMERV3_ROOT)
import tools as dreamer_tools  # noqa: E402

ARTIFACTS_DIR = Path("/common/users/ss5772/bimanual/artifacts")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(WORLD_MODEL_ROOT / "train_wm_config.yaml"))
    p.add_argument("--collection-dir", required=True)
    p.add_argument("--dreamer-fmt-dir", required=True)
    p.add_argument("--wm-checkpoint", required=True)
    p.add_argument("--head-checkpoint", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--spill-signal", choices=["broadcast", "per_step"], default="per_step")
    p.add_argument("--tag", default="ood")
    p.add_argument("--skip-convert", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    with open(args.config) as f:
        cfg = dict(yaml.safe_load(f))
    if args.device:
        cfg["device"] = args.device
    if not torch.cuda.is_available():
        cfg["device"] = "cpu"
    config = types.SimpleNamespace(**cfg)
    device = config.device

    collection_dir = Path(args.collection_dir)
    dreamer_dir = Path(args.dreamer_fmt_dir)
    if not args.skip_convert:
        print(f"[eval_monitor_ood] converting {collection_dir} -> {dreamer_dir}")
        convert_dir(collection_dir, dreamer_dir)

    episodes = dreamer_tools.load_episodes(dreamer_dir)
    print(f"[eval_monitor_ood] loaded {len(episodes)} OOD episodes: {sorted(episodes.keys())}")

    wm, image_shape, proprio_dim, num_actions = build_wm_from_checkpoint(config, episodes, args.wm_checkpoint)
    print(f"[eval_monitor_ood] frozen WorldModel loaded from {args.wm_checkpoint} "
          f"(image={image_shape} proprio_dim={proprio_dim}) -- NOT retrained.")

    head_ckpt = torch.load(args.head_checkpoint, map_location=device)
    head = MonitorHead(head_ckpt["latent_dim"], hidden=head_ckpt["hidden"]).to(device)
    head.load_state_dict(head_ckpt["head_state_dict"])
    head.eval()
    print(f"[eval_monitor_ood] frozen MonitorHead loaded from {args.head_checkpoint} "
          f"(trained on {head_ckpt.get('wm_checkpoint')}, spill_signal={head_ckpt.get('spill_signal')}) -- NOT retrained.")

    latents = compute_latents_and_labels(wm, episodes, collection_dir, device, spill_signal=args.spill_signal)
    latent_dim = next(iter(latents.values()))["z"].shape[-1]
    assert latent_dim == head_ckpt["latent_dim"], (
        f"latent_dim mismatch: OOD data gives {latent_dim}, head was trained on {head_ckpt['latent_dim']}"
    )

    # No split: evaluate on ALL OOD episodes (every one of them is unseen --
    # neither the WM nor the head ever trained on collection/data/cup_dr_test).
    all_keys = list(episodes.keys())
    z_all, y_all = stack_split(latents, all_keys)
    print(f"[eval_monitor_ood] OOD timesteps={z_all.shape[0]} "
          f"(spill_pos={int(y_all[:,0].sum())}, collision_pos={int(y_all[:,1].sum())})")

    with torch.no_grad():
        logits = head(z_all)
        probs = torch.sigmoid(logits).cpu().numpy()
    labels = y_all.cpu().numpy()

    results = {}
    for i, name in enumerate(LABEL_NAMES):
        yt, yp = labels[:, i], probs[:, i]
        precision, recall = precision_recall(yt, yp)
        roc = roc_curve(yt, yp)
        cal = calibration_curve(yt, yp)
        results[name] = {"precision": precision, "recall": recall, "roc": roc, "calibration": cal}
        auc_str = f"{roc[2]:.3f}" if roc is not None else "undefined (single class)"
        print(f"[eval_monitor_ood] OOD/{name}: precision={precision:.3f} recall={recall:.3f} "
              f"roc_auc={auc_str} n_pos={int(yt.sum())}/{len(yt)}")

    out_path = ARTIFACTS_DIR / f"monitor_{args.tag}.png"
    save_metrics_plot(results, out_path)
    print(f"[eval_monitor_ood] metrics plot saved to {out_path}")

    metrics_path = ARTIFACTS_DIR / f"monitor_{args.tag}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(
            {
                name: {
                    "precision": results[name]["precision"],
                    "recall": results[name]["recall"],
                    "roc_auc": results[name]["roc"][2] if results[name]["roc"] is not None else None,
                    "n_pos": int(labels[:, i].sum()),
                    "n_total": int(len(labels[:, i])),
                }
                for i, name in enumerate(LABEL_NAMES)
            },
            f,
            indent=2,
        )
    print(f"[eval_monitor_ood] metrics json saved to {metrics_path}")


if __name__ == "__main__":
    main()
