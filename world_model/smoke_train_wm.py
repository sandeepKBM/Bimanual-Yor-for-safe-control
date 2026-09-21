"""world_model/smoke_train_wm.py

SETUP-ONLY SMOKE TEST -- a handful of DreamerV3 RSSM world-model train steps
on our real `collection/data/smoke/` episodes (converted via
dataset_adapter.py). No online RL, no env interaction, no sbatch: this runs
directly on whatever GPU is free, for seconds.

What this proves: `models.WorldModel._train(batch)` from the vendored
dreamerv3-torch clone (/common/users/ss5772/dreamerv3) executes end-to-end
(encoder -> RSSM dynamics.observe -> decoder/reward/cont heads -> loss ->
backward -> optimizer step) on OUR RGB+action data, on this machine's GPU.

Does NOT use dreamer.py's main()/make_env/Dreamer agent -- see
world_model/README.md ("Offline support verdict") for why: that path always
builds a gym env purely for observation_space/action_space discovery, which
we don't have (and don't need) for a pure offline world-model run. Instead
we hand-build the obs/action spaces from dataset_adapter.py's known output
shapes and construct `models.WorldModel` directly.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch
import yaml

DREAMERV3_ROOT = "/common/users/ss5772/dreamerv3"
if DREAMERV3_ROOT not in sys.path:
    sys.path.insert(0, DREAMERV3_ROOT)

import models  # noqa: E402  (vendored dreamerv3-torch)
import tools as dreamer_tools  # noqa: E402

WORLD_MODEL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORLD_MODEL_ROOT.parent))  # so `world_model.*` imports work
from world_model.dataset_adapter import convert_dir  # noqa: E402

import gymnasium as gym  # noqa: E402


def load_config(smoke_batch_size: int, smoke_batch_length: int) -> types.SimpleNamespace:
    with open(WORLD_MODEL_ROOT / "train_wm_config.yaml") as f:
        cfg = yaml.safe_load(f)
    # Smoke overrides: tiny batch/length so this is seconds, not minutes.
    cfg["batch_size"] = smoke_batch_size
    cfg["batch_length"] = smoke_batch_length
    if not torch.cuda.is_available():
        cfg["device"] = "cpu"
    return types.SimpleNamespace(**cfg)


def build_spaces(config: types.SimpleNamespace):
    obs_space = gym.spaces.Dict(
        {
            "image": gym.spaces.Box(
                0, 255, (config.size[0], config.size[1], config.image_channels), np.uint8
            ),
            "proprio": gym.spaces.Box(
                -np.inf, np.inf, (config.proprio_dim,), np.float32
            ),
        }
    )
    act_space = gym.spaces.Box(-1.0, 1.0, (config.num_actions,), np.float32)
    return obs_space, act_space


def main() -> None:
    smoke_collection_dir = "/common/users/ss5772/bimanual/collection/data/smoke"
    smoke_dreamer_dir = WORLD_MODEL_ROOT / "data" / "dreamer_fmt" / "smoke"

    print(f"[smoke] converting {smoke_collection_dir} -> {smoke_dreamer_dir}")
    convert_dir(smoke_collection_dir, smoke_dreamer_dir)

    config = load_config(smoke_batch_size=2, smoke_batch_length=16)
    print(f"[smoke] device={config.device} batch_size={config.batch_size} "
          f"batch_length={config.batch_length}")

    episodes = dreamer_tools.load_episodes(smoke_dreamer_dir)
    print(f"[smoke] loaded {len(episodes)} episodes: {list(episodes.keys())}")

    generator = dreamer_tools.sample_episodes(episodes, config.batch_length)
    dataset = dreamer_tools.from_generator(generator, config.batch_size)

    obs_space, act_space = build_spaces(config)
    wm = models.WorldModel(obs_space, act_space, step=0, config=config).to(config.device)
    print(f"[smoke] WorldModel constructed on {config.device}")

    n_steps = 3
    for i in range(n_steps):
        batch = next(dataset)
        post, context, metrics = wm._train(batch)
        loss_keys = [k for k in metrics if k.endswith("_loss") or k == "kl"]
        loss_report = {k: float(metrics[k]) for k in loss_keys}
        print(f"[smoke] step {i}: {loss_report}")

    print("[smoke] RSSM world-model train step ran successfully on our bimanual data.")


if __name__ == "__main__":
    main()
