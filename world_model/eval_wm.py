"""world_model/eval_wm.py

WM eval on a held-out split (SCAFFOLD + SMOKE ONLY -- see world_model/README.md).

(a) reconstruction error -- teacher-forced encode/decode over the eval split,
    image + proprio MSE (in the decoder's own -log_prob loss, plus a plain
    pixel-space MSE for readability).
(b) open-loop imagined rollout -- encode a short prefix (posterior), then
    `dynamics.imagine_with_action` for the remaining steps using only the
    known action sequence (no further observations), decode, and compare
    predicted vs. true image + proprio.
(c) a filmstrip PNG: true / reconstructed(-or-imagined) / abs-error rows over
    time, saved to artifacts/wm_eval_<tag>.png.

Shapes (image channels, proprio dim) are read off the checkpoint's own
`obs_space` (saved by train_wm.py's convert step / re-inferred from the same
converted data) -- never hardcoded; the filmstrip renderer also does not
assume any fixed channel count (see `_channels_to_panels`).

Usage (real run):
    python -m world_model.eval_wm --dreamer-fmt-dir world_model/data/dreamer_fmt/train_001 \\
        --checkpoint world_model/logdir/wm_run_001/best.pt --prefix-len 5 --rollout-len 15

Usage (smoke, what this task ran):
    CUDA_VISIBLE_DEVICES=3 python -m world_model.eval_wm \\
        --dreamer-fmt-dir world_model/data/dreamer_fmt/smoke \\
        --checkpoint world_model/logdir/smoke_run/latest.pt \\
        --prefix-len 3 --rollout-len 5 --eval-holdout 1 --tag smoke
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch
import yaml

DREAMERV3_ROOT = "../dreamerv3"
if DREAMERV3_ROOT not in sys.path:
    sys.path.insert(0, DREAMERV3_ROOT)

import models  # noqa: E402
import tools as dreamer_tools  # noqa: E402

WORLD_MODEL_ROOT = Path(__file__).resolve().parent
if str(WORLD_MODEL_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(WORLD_MODEL_ROOT.parent))
from world_model.train_wm import (  # noqa: E402
    build_spaces,
    infer_obs_shapes,
    load_config,
    split_episodes,
)
from world_model.dataset_adapter import convert_dir  # noqa: E402

import gymnasium as gym  # noqa: E402

ARTIFACTS_DIR = Path("artifacts")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(WORLD_MODEL_ROOT / "train_wm_config.yaml"))
    p.add_argument("--collection-dir", default=None)
    p.add_argument("--dreamer-fmt-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--eval-holdout", type=int, default=1)
    p.add_argument("--prefix-len", type=int, default=5, help="#observed steps before imagining")
    p.add_argument("--rollout-len", type=int, default=15, help="#imagined steps after the prefix")
    p.add_argument("--batch-size", type=int, default=None, help="#episodes shown in the filmstrip (<= eval episodes)")
    p.add_argument("--tag", default="run")
    p.add_argument("--skip-convert", action="store_true")
    return p.parse_args(argv)


@torch.no_grad()
def reconstruction_error(wm, batch) -> dict:
    """(a) teacher-forced reconstruction error over the whole eval batch."""
    data = wm.preprocess(batch)
    embed = wm.encoder(data)
    post, prior = wm.dynamics.observe(embed, data["action"], data["is_first"])
    feat = wm.dynamics.get_feat(post)
    metrics = {}
    image_pred = wm.heads["decoder"](feat)["image"].mode()
    metrics["image_mse"] = float(torch.mean((image_pred - data["image"]) ** 2))
    if "proprio" in data:
        proprio_pred = wm.heads["decoder"](feat)["proprio"].mode()
        metrics["proprio_mse"] = float(torch.mean((proprio_pred - data["proprio"]) ** 2))
    kl_loss, kl_value, _, _ = wm.dynamics.kl_loss(
        post, prior, wm._config.kl_free, wm._config.dyn_scale, wm._config.rep_scale
    )
    metrics["kl"] = float(kl_value.mean())
    return metrics, data, feat


@torch.no_grad()
def open_loop_rollout(wm, data, prefix_len: int, rollout_len: int):
    """(b) encode a prefix (posterior states), then imagine forward using
    only the true action sequence -- no further observations. Mirrors
    models.WorldModel.video_pred's logic but generalized to arbitrary
    prefix/rollout lengths and also decoding `proprio`, not just `image`."""
    embed = wm.encoder(data)
    b, t = data["action"].shape[:2]
    prefix_len = min(prefix_len, t - 1)
    rollout_len = min(rollout_len, t - prefix_len)

    post, _ = wm.dynamics.observe(
        embed[:, :prefix_len], data["action"][:, :prefix_len], data["is_first"][:, :prefix_len]
    )
    recon_feat = wm.dynamics.get_feat(post)
    recon_image = wm.heads["decoder"](recon_feat)["image"].mode()

    init_state = {k: v[:, -1] for k, v in post.items()}
    imagine_actions = data["action"][:, prefix_len : prefix_len + rollout_len]
    prior = wm.dynamics.imagine_with_action(imagine_actions, init_state)
    imag_feat = wm.dynamics.get_feat(prior)
    imag_image = wm.heads["decoder"](imag_feat)["image"].mode()
    imag_proprio = None
    if "proprio" in data:
        imag_proprio = wm.heads["decoder"](imag_feat)["proprio"].mode()

    truth_image = data["image"][:, : prefix_len + rollout_len]
    pred_image = torch.cat([recon_image, imag_image], 1)

    out = {
        "truth_image": truth_image,
        "pred_image": pred_image,
        "prefix_len": prefix_len,
        "rollout_len": rollout_len,
        "image_mse_prefix": float(torch.mean((recon_image - truth_image[:, :prefix_len]) ** 2)),
        "image_mse_imagined": float(torch.mean((imag_image - truth_image[:, prefix_len:]) ** 2)),
    }
    if "proprio" in data and imag_proprio is not None:
        truth_proprio = data["proprio"][:, prefix_len : prefix_len + rollout_len]
        out["proprio_mse_imagined"] = float(torch.mean((imag_proprio - truth_proprio) ** 2))
    return out


def _channels_to_panels(img: np.ndarray) -> np.ndarray:
    """img: (H, W, C) float in [0,1]. Returns an (H, W', 3) RGB panel with NO
    assumption about what C is (1, 3, 6, or anything else) -- multiples of 3
    are split into side-by-side 3-channel views (e.g. 6ch = 2 concatenated
    cams -> 2 RGB tiles); anything else falls back to the first channel shown
    as grayscale, so this keeps working whether the WM image is wrist-cam
    (6ch, 2 cams) or ZED (3ch, or ZED-RGB+depth as 4ch, etc.)."""
    h, w, c = img.shape
    if c == 3:
        return np.clip(img, 0, 1)
    if c == 1:
        return np.repeat(img, 3, axis=-1)
    if c % 3 == 0:
        tiles = [img[..., i : i + 3] for i in range(0, c, 3)]
        return np.clip(np.concatenate(tiles, axis=1), 0, 1)
    gray = np.repeat(img[..., :1], 3, axis=-1)
    return np.clip(gray, 0, 1)


def save_filmstrip(rollout: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # wm.preprocess() puts "image" in [0,1] (obs["image"] / 255.0); the mse
    # image_dist decoder's .mode() targets that same range (cnn_sigmoid=False
    # in train_wm_config.yaml means no output squashing, so it can slightly
    # over/undershoot [0,1] -- clip for display only).
    truth = rollout["truth_image"][0].detach().cpu().numpy()  # (T,H,W,C) in [0,1]
    pred = rollout["pred_image"][0].detach().cpu().numpy()
    truth = truth.clip(0, 1)
    pred = pred.clip(0, 1)
    err = np.abs(pred - truth)

    t = truth.shape[0]
    prefix_len = rollout["prefix_len"]
    fig, axes = plt.subplots(3, t, figsize=(max(6, t * 1.5), 4.5))
    if t == 1:
        axes = axes[:, None]
    for i in range(t):
        axes[0, i].imshow(_channels_to_panels(truth[i]))
        axes[1, i].imshow(_channels_to_panels(pred[i]))
        axes[2, i].imshow(_channels_to_panels(err[i]))
        for r in range(3):
            axes[r, i].axis("off")
        tag = "obs" if i < prefix_len else "imagined"
        axes[0, i].set_title(f"t={i}\n({tag})", fontsize=8)
    axes[0, 0].set_ylabel("true", fontsize=9)
    axes[1, 0].set_ylabel("recon/imag", fontsize=9)
    axes[2, 0].set_ylabel("|error|", fontsize=9)
    fig.suptitle(
        f"WM eval filmstrip -- prefix={prefix_len} observed, "
        f"{rollout['rollout_len']} imagined; "
        f"imagined image MSE={rollout['image_mse_imagined']:.4f}"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def build_wm_from_checkpoint(config, episodes, checkpoint_path):
    image_shape, proprio_dim, num_actions = infer_obs_shapes(episodes)
    obs_space, act_space = build_spaces(image_shape, proprio_dim, num_actions)
    wm = models.WorldModel(obs_space, act_space, step=0, config=config).to(config.device)
    ckpt = torch.load(checkpoint_path, map_location=config.device)
    wm.load_state_dict(ckpt["wm_state_dict"])
    wm.eval()
    return wm, image_shape, proprio_dim, num_actions


def main(argv=None) -> None:
    args = parse_args(argv)
    with open(args.config) as f:
        cfg = dict(yaml.safe_load(f))
    if args.device:
        cfg["device"] = args.device
    if not torch.cuda.is_available():
        cfg["device"] = "cpu"
    config = types.SimpleNamespace(**cfg)

    dreamer_dir = Path(args.dreamer_fmt_dir)
    if not args.skip_convert:
        collection_dir = args.collection_dir or "collection/data/smoke"
        print(f"[eval_wm] converting {collection_dir} -> {dreamer_dir}")
        convert_dir(collection_dir, dreamer_dir)

    episodes = dreamer_tools.load_episodes(dreamer_dir)
    print(f"[eval_wm] loaded {len(episodes)} episodes: {sorted(episodes.keys())}")
    _, eval_episodes = split_episodes(episodes, args.eval_holdout)
    assert eval_episodes, "no held-out episodes -- increase --eval-holdout or the dataset size"
    print(f"[eval_wm] eval split: {sorted(eval_episodes.keys())}")

    wm, image_shape, proprio_dim, num_actions = build_wm_from_checkpoint(config, episodes, args.checkpoint)
    print(f"[eval_wm] built WorldModel from checkpoint={args.checkpoint} "
          f"(inferred image={image_shape} proprio_dim={proprio_dim} num_actions={num_actions})")

    max_len = min(ep["action"].shape[0] for ep in eval_episodes.values())
    batch_length = min(max_len, args.prefix_len + args.rollout_len)
    batch_size = args.batch_size or len(eval_episodes)
    gen = dreamer_tools.sample_episodes(eval_episodes, batch_length)
    dataset = dreamer_tools.from_generator(gen, min(batch_size, len(eval_episodes)))
    batch = next(dataset)

    # (a) reconstruction error
    recon_metrics, data, _ = reconstruction_error(wm, batch)
    print(f"[eval_wm] (a) reconstruction: {recon_metrics}")

    # (b) open-loop imagined rollout
    rollout = open_loop_rollout(wm, data, args.prefix_len, args.rollout_len)
    print(f"[eval_wm] (b) open-loop rollout: prefix={rollout['prefix_len']} "
          f"rollout={rollout['rollout_len']} image_mse_prefix={rollout['image_mse_prefix']:.4f} "
          f"image_mse_imagined={rollout['image_mse_imagined']:.4f} "
          f"proprio_mse_imagined={rollout.get('proprio_mse_imagined', float('nan')):.4f}")

    # (c) filmstrip
    out_path = ARTIFACTS_DIR / f"wm_eval_{args.tag}.png"
    save_filmstrip(rollout, out_path)
    print(f"[eval_wm] (c) filmstrip saved to {out_path}")


if __name__ == "__main__":
    main()
