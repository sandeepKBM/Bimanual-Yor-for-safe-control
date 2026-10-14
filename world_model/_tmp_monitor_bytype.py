"""Diagnostic: break down the trained monitor head's spill precision/recall
by perturbation_level (spill_shake vs spill_tilt) on the held-out split, to
check whether the near-perfect AUC=1.000 is driven mostly by the "easy"
spill_shake sub-case (per-step spill flag fires at t=2 and stays on ~99% of
the episode -- effectively a constant per-episode label) vs the harder
spill_tilt case (flag onsets mid-episode, ~60% through, ~40% episode
coverage -- a genuine temporal-onset prediction problem)."""
import json
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
from world_model.train_wm import split_episodes  # noqa: E402
from world_model.eval_wm import build_wm_from_checkpoint  # noqa: E402
from world_model.monitor_head import (  # noqa: E402
    MonitorHead, compute_latents_and_labels, precision_recall, roc_curve,
)

COLLECTION_DIR = Path("collection/data/cup_yor_full")

with open("world_model/train_wm_config.yaml") as f:
    cfg = dict(yaml.safe_load(f))
config = types.SimpleNamespace(**cfg)
device = config.device

episodes = dreamer_tools.load_episodes(Path("world_model/data/dreamer_fmt/cup_yor_full"))
train_eps, eval_eps = split_episodes(episodes, 20)
wm, image_shape, proprio_dim, num_actions = build_wm_from_checkpoint(
    config, episodes, "world_model/logdir/wm_cup_full/best.pt"
)
latents = compute_latents_and_labels(wm, episodes, COLLECTION_DIR, device, spill_signal="per_step")

head_ckpt = torch.load("world_model/logdir/wm_cup_full/monitor_head_cup_full.pt", map_location=device)
head = MonitorHead(head_ckpt["latent_dim"], hidden=head_ckpt["hidden"]).to(device)
head.load_state_dict(head_ckpt["head_state_dict"])
head.eval()

def _ep_id(key):
    return key.rsplit("-", 1)[0]

level_by_ep = {}
for key in eval_eps:
    meta = json.loads((COLLECTION_DIR / _ep_id(key) / "metadata.json").read_text())
    level_by_ep[key] = meta["perturbation_level"]

by_level = {}
for key, lvl in level_by_ep.items():
    by_level.setdefault(lvl, []).append(key)

print("[bytype] eval episodes per perturbation_level:", {k: len(v) for k, v in by_level.items()})

with torch.no_grad():
    for lvl, keys in sorted(by_level.items()):
        z = torch.cat([latents[k]["z"] for k in keys], dim=0)
        y_spill = torch.cat([latents[k]["spill"] for k in keys], dim=0).cpu().numpy()
        y_coll = torch.cat([latents[k]["collision"] for k in keys], dim=0).cpu().numpy()
        probs = torch.sigmoid(head(z)).cpu().numpy()
        p_spill, r_spill = precision_recall(y_spill, probs[:, 0])
        roc_spill = roc_curve(y_spill, probs[:, 0])
        p_coll, r_coll = precision_recall(y_coll, probs[:, 1])
        roc_coll = roc_curve(y_coll, probs[:, 1])
        auc_spill = f"{roc_spill[2]:.3f}" if roc_spill else "undefined"
        auc_coll = f"{roc_coll[2]:.3f}" if roc_coll else "undefined"
        print(f"[bytype] {lvl}: n_t={len(y_spill)} spill_pos={int(y_spill.sum())} "
              f"spill(P={p_spill:.3f} R={r_spill:.3f} AUC={auc_spill}) | "
              f"coll_pos={int(y_coll.sum())} coll(P={p_coll:.3f} R={r_coll:.3f} AUC={auc_coll})")
