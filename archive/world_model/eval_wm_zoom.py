"""world_model/eval_wm_zoom.py

Diagnostic companion to eval_wm.py, written for THIS task (honest visual
check of whether the trained WM models the cup/ball dynamics, not just the
static background). eval_wm.py's own filmstrip (artifacts/wm_eval_cup_full.png)
looks visually static at 162px/frame because the ZED view is dominated by
static background (sky, table, arms) -- the cup+8 balls occupy a handful of
pixels. This script:

  1. Picks a SPECIFIC eval episode by id (not eval_wm.py's dreamer_tools
     .sample_episodes, which samples a short/possibly-early random window)
     so we can force a genuine spill episode (spill_shake/spill_tilt) and a
     long enough imagined rollout to cover the actual scatter event.
  2. Renders BOTH the full frame and a zoomed crop (table region only, where
     the cup + scattering balls live) at higher DPI / fewer, more widely
     spaced timesteps so the ball dots are actually visible to the eye.

Reuses eval_wm.py's build_wm_from_checkpoint / open_loop_rollout /
_channels_to_panels unchanged -- no eval_wm.py edits.

Usage:
    python -m world_model.eval_wm_zoom --episode-id episode_000181 \\
        --checkpoint world_model/logdir/wm_cup_full/best.pt \\
        --dreamer-fmt-dir world_model/data/dreamer_fmt/cup_yor_full \\
        --prefix-len 8 --rollout-len 150 --tag spillshake_zoom
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch
import yaml

DREAMERV3_ROOT = "/common/users/ss5772/dreamerv3"
if DREAMERV3_ROOT not in sys.path:
    sys.path.insert(0, DREAMERV3_ROOT)
import tools as dreamer_tools  # noqa: E402

WORLD_MODEL_ROOT = Path(__file__).resolve().parent
if str(WORLD_MODEL_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(WORLD_MODEL_ROOT.parent))

from world_model.eval_wm import (  # noqa: E402
    build_wm_from_checkpoint,
    open_loop_rollout,
    _channels_to_panels,
)

ARTIFACTS_DIR = Path("/common/users/ss5772/bimanual/artifacts")
# Table region within the 64x64 ZED frame (rows, cols) -- where the cup +
# scattering balls actually live (see /tmp/spill_zed_frames.png inspection:
# sky fills rows 0-16, table body rows ~16-55, cols ~4-60).
CROP = (14, 58, 2, 62)  # (r0, r1, c0, c1)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dreamer-fmt-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=str(WORLD_MODEL_ROOT / "train_wm_config.yaml"))
    p.add_argument("--episode-id", required=True, help="e.g. episode_000181 (id prefix, matched against loaded keys)")
    p.add_argument("--prefix-len", type=int, default=8)
    p.add_argument("--rollout-len", type=int, default=150)
    p.add_argument("--stride", type=int, default=10, help="show every Nth frame in the strip")
    p.add_argument("--tag", default="zoom")
    p.add_argument("--device", default=None)
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

    episodes = dreamer_tools.load_episodes(Path(args.dreamer_fmt_dir))
    match = [k for k in episodes if k.startswith(args.episode_id)]
    assert match, f"no episode key starts with {args.episode_id!r}; have {sorted(episodes.keys())[:5]}..."
    key = match[0]
    ep = episodes[key]
    print(f"[eval_wm_zoom] using episode {key}, length={ep['action'].shape[0]}")

    wm, image_shape, proprio_dim, num_actions = build_wm_from_checkpoint(config, episodes, args.checkpoint)

    t_len = min(ep["action"].shape[0], args.prefix_len + args.rollout_len)
    batch = {}
    for k, v in ep.items():
        arr = np.asarray(v[:t_len])
        batch[k] = torch.as_tensor(arr, device=config.device)[None]
    data = wm.preprocess(batch)

    rollout = open_loop_rollout(wm, data, args.prefix_len, args.rollout_len)
    print(f"[eval_wm_zoom] prefix={rollout['prefix_len']} rollout={rollout['rollout_len']} "
          f"image_mse_prefix={rollout['image_mse_prefix']:.6f} "
          f"image_mse_imagined={rollout['image_mse_imagined']:.6f}")

    truth = rollout["truth_image"][0].detach().cpu().numpy().clip(0, 1)  # (T,H,W,C)
    pred = rollout["pred_image"][0].detach().cpu().numpy().clip(0, 1)
    prefix_len = rollout["prefix_len"]
    t = truth.shape[0]
    idxs = list(range(0, t, args.stride))
    if idxs[-1] != t - 1:
        idxs.append(t - 1)

    r0, r1, c0, c1 = CROP

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(idxs)
    fig, axes = plt.subplots(4, n, figsize=(max(6, n * 1.6), 6.4))
    for j, i in enumerate(idxs):
        true_panel = _channels_to_panels(truth[i])
        pred_panel = _channels_to_panels(pred[i])
        # true_panel may be (H, W*ncam, 3) if multi-cam tiled; crop each cam tile.
        w_each = true_panel.shape[1] // (true_panel.shape[1] // true_panel.shape[0] if true_panel.shape[0] else 1)
        # Simpler: just crop first W columns (single ZED tile repeated -> same content).
        H = truth.shape[1]
        true_full = true_panel[:, :H]
        pred_full = pred_panel[:, :H]
        true_crop = true_full[r0:r1, c0:c1]
        pred_crop = pred_full[r0:r1, c0:c1]
        err_crop = np.abs(pred_crop - true_crop)

        axes[0, j].imshow(true_full)
        axes[1, j].imshow(true_crop, interpolation="nearest")
        axes[2, j].imshow(pred_crop, interpolation="nearest")
        axes[3, j].imshow(err_crop, interpolation="nearest")
        for r in range(4):
            axes[r, j].axis("off")
        tag = "obs" if i < prefix_len else "imag"
        axes[0, j].set_title(f"t={i} ({tag})", fontsize=7)
    axes[0, 0].set_ylabel("full", fontsize=8)
    axes[1, 0].set_ylabel("true (zoom)", fontsize=8)
    axes[2, 0].set_ylabel("pred (zoom)", fontsize=8)
    axes[3, 0].set_ylabel("|err| (zoom)", fontsize=8)
    fig.suptitle(
        f"{key} -- table/cup/ball crop, prefix={prefix_len} obs + {rollout['rollout_len']} imagined "
        f"(stride={args.stride}); imagined image MSE={rollout['image_mse_imagined']:.5f}"
    )
    fig.tight_layout()
    out_path = ARTIFACTS_DIR / f"wm_eval_{args.tag}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[eval_wm_zoom] saved {out_path}")


if __name__ == "__main__":
    main()
