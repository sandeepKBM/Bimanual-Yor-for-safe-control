"""world_model/train_wm.py

REAL offline DreamerV3 RSSM world-model training driver (scaffold task --
this file is meant to run a full training loop; only smoke-tested here, per
task instructions: no full training, no sbatch was executed from this repo).

Same code path as smoke_train_wm.py (build `models.WorldModel` directly, no
`dreamer.py` main()/make_env()/Dreamer agent, no online env -- see
world_model/README.md section 4 "Offline-training support verdict"), but:
  - obs/action SHAPES ARE INFERRED FROM THE LOADED DATA, never hardcoded.
    `image` channel count and `proprio` dim come from the converted npz's
    actual arrays (`episodes[k]["image"].shape[-1]` etc.), not from
    train_wm_config.yaml's `image_channels`/`proprio_dim` fields (those are
    documentation/defaults only, checked-not-trusted -- see
    `infer_obs_shapes()`). This matters because the ZED-view /
    26-D-proprio switch is still being finalized elsewhere (dataset_adapter.py
    may change channel count without this file needing an edit).
  - real training loop over `config.train_steps` (not 3 smoke steps),
  - checkpoint save/resume: `latest.pt` every `--ckpt-every` steps, `best.pt`
    on eval-loss improvement (mirrors dreamer.py's own
    agent_state_dict/optims_state_dict pattern -- see README.md),
  - tensorboard + jsonl logging via the vendored `tools.Logger` (same class
    dreamer.py itself uses) for recon/kl/reward/cont loss components,
  - a periodic eval hook (held-out split forward pass, no grad, no backward).

Usage (real run, NOT executed by this task):
    python -m world_model.train_wm --config world_model/train_wm_config.yaml \\
        --collection-dir collection/data/run_001 \\
        --dreamer-fmt-dir world_model/data/dreamer_fmt/train_001

Usage (smoke, seconds, what this task DID run):
    CUDA_VISIBLE_DEVICES=3 python -m world_model.train_wm \\
        --collection-dir collection/data/smoke \\
        --dreamer-fmt-dir world_model/data/dreamer_fmt/smoke \\
        --logdir world_model/logdir/smoke_run \\
        --train-steps 3 --batch-size 2 --batch-length 16 \\
        --eval-every 2 --ckpt-every 2 --eval-holdout 1
"""
from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import yaml

DREAMERV3_ROOT = "../dreamerv3"
if DREAMERV3_ROOT not in sys.path:
    sys.path.insert(0, DREAMERV3_ROOT)

import models  # noqa: E402  (vendored dreamerv3-torch)
import tools as dreamer_tools  # noqa: E402

WORLD_MODEL_ROOT = Path(__file__).resolve().parent
if str(WORLD_MODEL_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(WORLD_MODEL_ROOT.parent))  # so `world_model.*` imports work
from world_model.dataset_adapter import convert_dir  # noqa: E402

import gymnasium as gym  # noqa: E402


# --------------------------------------------------------------------------
# Config / CLI
# --------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(WORLD_MODEL_ROOT / "train_wm_config.yaml"))
    p.add_argument("--collection-dir", default=None, help="overrides config's implicit source dir")
    p.add_argument("--dreamer-fmt-dir", default=None, help="converted-npz output dir (also read dir)")
    p.add_argument("--logdir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--train-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--batch-length", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=100, help="steps between eval-hook forward passes")
    p.add_argument("--ckpt-every", type=int, default=100, help="steps between latest.pt saves")
    p.add_argument("--eval-holdout", type=int, default=1, help="#episodes (by sorted id, last N) held out for eval")
    p.add_argument("--no-resume", action="store_true", help="ignore an existing latest.pt")
    p.add_argument("--skip-convert", action="store_true", help="dreamer-fmt-dir already populated")
    return p.parse_args(argv)


def load_config(args: argparse.Namespace) -> types.SimpleNamespace:
    with open(args.config) as f:
        cfg = dict(yaml.safe_load(f))
    if args.dreamer_fmt_dir:
        cfg["offline_traindir"] = args.dreamer_fmt_dir
    if args.logdir:
        cfg["logdir"] = args.logdir
    if args.device:
        cfg["device"] = args.device
    if not torch.cuda.is_available():
        cfg["device"] = "cpu"
    if args.train_steps is not None:
        cfg["train_steps"] = args.train_steps
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.batch_length is not None:
        cfg["batch_length"] = args.batch_length
    return types.SimpleNamespace(**cfg)


# --------------------------------------------------------------------------
# Shape inference -- NEVER hardcode image channels / proprio dim. The image
# view (ZED vs. wrist) and proprio width (26-D vs. a future revision) are
# still being finalized in dataset_adapter.py by another task; this file
# must not assume either.
# --------------------------------------------------------------------------
def infer_obs_shapes(episodes: dict) -> tuple[tuple[int, int, int], int, int]:
    """Returns ((H, W, C_image), proprio_dim, num_actions) read directly off
    one loaded episode's actual arrays -- not off the yaml config."""
    any_ep = next(iter(episodes.values()))
    image_shape = tuple(any_ep["image"].shape[1:])  # (H, W, C)
    proprio_dim = int(any_ep["proprio"].shape[-1])
    num_actions = int(any_ep["action"].shape[-1])
    assert len(image_shape) == 3, f"expected (T,H,W,C) image, got trailing shape {image_shape}"
    return image_shape, proprio_dim, num_actions


def build_spaces(image_shape, proprio_dim, num_actions):
    obs_space = gym.spaces.Dict(
        {
            "image": gym.spaces.Box(0, 255, image_shape, np.uint8),
            "proprio": gym.spaces.Box(-np.inf, np.inf, (proprio_dim,), np.float32),
        }
    )
    act_space = gym.spaces.Box(-1.0, 1.0, (num_actions,), np.float32)
    return obs_space, act_space


def split_episodes(episodes: dict, n_holdout: int) -> tuple[dict, dict]:
    """Holds out the last `n_holdout` episodes (by sorted key) for eval.
    With very few episodes (e.g. the 3-episode smoke set) this can leave the
    train split with only 1-2 episodes -- fine for a scaffold/smoke, NOT a
    real held-out split; a real run should point --dreamer-fmt-dir at a
    dataset large enough that eval episodes are truly disjoint in content."""
    keys = sorted(episodes.keys())
    n_holdout = max(0, min(n_holdout, len(keys) - 1))  # always keep >=1 train ep
    eval_keys = keys[-n_holdout:] if n_holdout > 0 else []
    train_keys = keys[: len(keys) - n_holdout] if n_holdout > 0 else keys
    return {k: episodes[k] for k in train_keys}, {k: episodes[k] for k in eval_keys}


# --------------------------------------------------------------------------
# Checkpointing (mirrors dreamerv3-torch's own dreamer.py pattern, but for a
# bare WorldModel rather than the full Dreamer agent -- see README.md).
# --------------------------------------------------------------------------
def save_checkpoint(path: Path, wm, step: int, best_eval_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "wm_state_dict": wm.state_dict(),
        "optims_state_dict": dreamer_tools.recursively_collect_optim_state_dict(wm),
        "step": step,
        "best_eval_loss": best_eval_loss,
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, wm) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=next(wm.parameters()).device)
    wm.load_state_dict(ckpt["wm_state_dict"])
    dreamer_tools.recursively_load_optim_state_dict(wm, ckpt["optims_state_dict"])
    return int(ckpt.get("step", 0)), float(ckpt.get("best_eval_loss", float("inf")))


@torch.no_grad()
def eval_forward(wm, batch) -> dict:
    """One held-out forward pass, no grad/backward -- the periodic eval hook.
    Mirrors WorldModel._train's math (encode -> observe -> heads -> losses)
    without the optimizer step, so it's cheap and side-effect-free."""
    data = wm.preprocess(batch)
    embed = wm.encoder(data)
    post, prior = wm.dynamics.observe(embed, data["action"], data["is_first"])
    kl_loss, kl_value, _, _ = wm.dynamics.kl_loss(
        post, prior, wm._config.kl_free, wm._config.dyn_scale, wm._config.rep_scale
    )
    feat = wm.dynamics.get_feat(post)
    losses = {}
    for name, head in wm.heads.items():
        pred = head(feat)
        preds = pred if isinstance(pred, dict) else {name: pred}
        for k, dist in preds.items():
            losses[k] = -dist.log_prob(data[k]).mean()
    model_loss = sum(losses.values()) + kl_loss.mean()
    metrics = {f"eval_{k}_loss": float(v) for k, v in losses.items()}
    metrics["eval_kl"] = float(kl_value.mean())
    metrics["eval_model_loss"] = float(model_loss)
    return metrics


# --------------------------------------------------------------------------
# Main training loop
# --------------------------------------------------------------------------
def main(argv=None) -> None:
    args = parse_args(argv)
    config = load_config(args)

    collection_dir = args.collection_dir or "collection/data/smoke"
    dreamer_dir = Path(config.offline_traindir)

    if not args.skip_convert:
        print(f"[train_wm] converting {collection_dir} -> {dreamer_dir}")
        convert_dir(collection_dir, dreamer_dir)

    episodes = dreamer_tools.load_episodes(dreamer_dir)
    print(f"[train_wm] loaded {len(episodes)} episodes: {sorted(episodes.keys())}")
    train_episodes, eval_episodes = split_episodes(episodes, args.eval_holdout)
    print(f"[train_wm] split: {len(train_episodes)} train / {len(eval_episodes)} eval episodes")

    image_shape, proprio_dim, num_actions = infer_obs_shapes(episodes)
    print(f"[train_wm] inferred obs shapes from data: image={image_shape} "
          f"proprio_dim={proprio_dim} num_actions={num_actions} "
          f"(config.yaml says image_channels={getattr(config, 'image_channels', '?')} "
          f"proprio_dim={getattr(config, 'proprio_dim', '?')} -- data wins if they differ)")
    obs_space, act_space = build_spaces(image_shape, proprio_dim, num_actions)

    train_gen = dreamer_tools.sample_episodes(train_episodes, config.batch_length)
    train_dataset = dreamer_tools.from_generator(train_gen, config.batch_size)
    eval_gen = (
        dreamer_tools.sample_episodes(eval_episodes, config.batch_length)
        if eval_episodes
        else None
    )
    eval_dataset = (
        dreamer_tools.from_generator(eval_gen, min(config.batch_size, len(eval_episodes)))
        if eval_gen
        else None
    )

    print(f"[train_wm] device={config.device} batch_size={config.batch_size} "
          f"batch_length={config.batch_length} train_steps={config.train_steps}")
    wm = models.WorldModel(obs_space, act_space, step=0, config=config).to(config.device)

    logdir = Path(config.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    logger = dreamer_tools.Logger(logdir, 0)

    latest_path = logdir / "latest.pt"
    best_path = logdir / "best.pt"
    start_step = 0
    best_eval_loss = float("inf")
    if latest_path.exists() and not args.no_resume:
        start_step, best_eval_loss = load_checkpoint(latest_path, wm)
        print(f"[train_wm] resumed from {latest_path} at step={start_step} "
              f"best_eval_loss={best_eval_loss:.4f}")
    else:
        print(f"[train_wm] no resume ({'--no-resume passed' if args.no_resume else 'no checkpoint found'}); starting from step 0")

    t0 = time.time()
    step = start_step
    while step < config.train_steps:
        batch = next(train_dataset)
        post, context, metrics = wm._train(batch)
        loss_keys = [k for k in metrics if k.endswith("_loss") or k == "kl"]
        for k in loss_keys:
            logger.scalar(f"train/{k}", metrics[k])
        step += 1
        logger.step = step

        if step % max(1, args.eval_every) == 0 and eval_dataset is not None:
            eval_batch = next(eval_dataset)
            eval_metrics = eval_forward(wm, eval_batch)
            for k, v in eval_metrics.items():
                logger.scalar(f"eval/{k}", v)
            eval_loss = eval_metrics["eval_model_loss"]
            print(f"[train_wm] step {step}: train_model_loss={float(metrics['model_loss']):.4f} "
                  f"eval_model_loss={eval_loss:.4f}")
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save_checkpoint(best_path, wm, step, best_eval_loss)
                print(f"[train_wm] step {step}: new best eval_model_loss={best_eval_loss:.4f} -> {best_path}")
        else:
            print(f"[train_wm] step {step}: train_model_loss={float(metrics['model_loss']):.4f}")

        logger.write(fps=True)

        if step % max(1, args.ckpt_every) == 0 or step == config.train_steps:
            save_checkpoint(latest_path, wm, step, best_eval_loss)

    dt = time.time() - t0
    print(f"[train_wm] done: {step - start_step} steps in {dt:.1f}s "
          f"({dt / max(1, step - start_step):.3f}s/step). logdir={logdir}")


if __name__ == "__main__":
    main()
