"""world_model/train_latent_safety_value.py

Latent HJ-style safety value `V_theta(z)` trained via the Fisac et al.
discounted safety-Bellman recursion ("Bridging Hamilton-Jacobi Safety
Analysis and Reinforcement Learning", Fisac, Lugovoy, Rubies-Royo, Ghosh,
Tomlin, 2019) directly on the FROZEN WorldModel's latent
`z = wm.dynamics.get_feat(post)` (deter+stoch), over REAL recorded
`(z_t, l_t, z_{t+1})` transitions read out of `collection/` episodes -- NO
imagined / model-rollout transitions in this pass (see "NOT done here"
below). This is the scaffold for the thing that will eventually replace the
analytic barrier `h(x)` in the QP -- that wiring is explicitly NOT done
here.

SIGN CONVENTION -- read this before touching any sign in this file
--------------------------------------------------------------------------
  - `l_t in {0,1}`: l_t=1 iff a failure occurred at step t.
    `l_t = collision[t] | dropped[t] | cube_dropped[t]`, i.e.
      * collision[t]    -- real per-step arm-arm contact bool.
      * dropped[t]      -- repurposed per-step SPILL flag
                            (`count_spilled(model,data) > 0`), the
                            "collect_cup*.py" convention (see
                            collection/collect_cup_collect.py's docstring
                            and monitor_head.py's `--spill-signal
                            per_step`).
      * cube_dropped[t] -- the target cube fumbled OUTSIDE the cup (a
                            third, distinct failure mode present on
                            cup_collect_v1-style episodes only; degrades to
                            all-zeros where the field is absent, same
                            graceful-degradation pattern as monitor_head's
                            `topple`).
    This is the task's "spill OR collision OR drop" -- see
    `_load_failure_signal` below.
  - Margin `g(l) = 1 - 2*l`  ->  g=+1 on safe steps, g=-1 on failure steps.
    STANDARD HJ SAFETY-VALUE CONVENTION USED THROUGHOUT THIS FILE:
        V(z) < 0  <=>  z is judged UNSAFE (inside the discounted
                       reachable-failure set)
        V(z) > 0  <=>  z is judged safe
        V(z) = 0  is the level set that would stand in for the analytic
                   barrier h(x)=0 in a future QP (NOT wired up here).
  - Discounted safety Bellman target (Fisac et al.'s discounted variant of
    the safety value): for every REAL transition `(z_t, l_t, z_{t+1})`,
        y_t = min( g(l_t), (1-gamma)*g(l_t) + gamma*V_target(z_{t+1}) )
    i.e. "the value of a state is the worse of its own margin right now,
    and a discounted blend of that margin with the successor's value" --
    this is what makes V the (discounted) WORST margin seen along the
    trajectory, not a sum of costs. `V_target` is a periodically-hard-
    copied target network (Polyak/EMA would also work; this file picks the
    periodic-copy convention -- see `--target-update-interval`), exactly
    the DQN-style target-network trick Fisac et al. also use.
  - At a FAILURE step (l_t=1) the target is additionally hard-clamped to
    `g(l_t) = -1` directly via `torch.where`, bypassing the bootstrap term.
    This is mathematically already implied by the `min()` above whenever
    `V_target` stays inside `[-1, 1]` (g=-1 is the global minimum of that
    range, so `(1-gamma)*(-1) + gamma*V_target >= (1-gamma)*(-1) +
    gamma*(-1) = -1 = g(l_t)`, i.e. the bootstrap term can never be
    STRICTLY smaller than g(l_t) at a failure step, so `min` always picks
    g(l_t) there on its own) -- the explicit `torch.where` is added anyway
    per this task's "be explicit and consistent" ask, and as a hedge in
    case numerical drift ever pushes `V_target` outside `(-1, 1)`.
  - `V_theta(z) = tanh(mlp(z)) in (-1, 1)` -- bounded to the same open
    range as `g(l)`'s extremes by construction, which is exactly what
    makes the failure-step clamp above provably redundant (not merely
    hoped to be).
  - `gamma=0.99` (a Fisac-style discounted safety value; gamma->1 recovers
    the undiscounted worst-case HJ value but loses the contraction
    property that makes a target-network TD scheme well-behaved).

Training loop skeleton (frozen WM, Adam, held-out episode split, epochs)
deliberately mirrors monitor_head.py's; `compute_latents_and_failure` below
mirrors monitor_head.compute_latents_and_labels's frozen-WM forward pass
byte-for-byte (same wm.preprocess -> wm.encoder -> wm.dynamics.observe ->
get_feat pipeline) and differs only in which raw per-step arrays it reads
off the episode (it needs `cube_dropped` too, which monitor_head.ALL_LABELS
does not expose, so it is not reused as an import -- see module docstring
there for why "or its pattern" was explicitly allowed for this file).

Eval (go/no-go): on held-out episodes, label every REAL state `z_t` as
"imminent-unsafe" (1) if any failure occurs within the next
`--imminent-horizon` steps, else "safe" (0), then report the ROC-AUC of
`-V_theta(z_t)` (min-max-normalized into [0,1] so monitor_head.roc_curve's
[0,1]-threshold sweep applies unchanged -- AUC is invariant to monotonic
rescaling) against that binarized label. This checks whether the learned
Bellman value actually RANKS imminent-failure states below safe ones --
the thing an analytic-barrier replacement would need to get right.

NOT done here (explicitly out of scope for this pass)
--------------------------------------------------------------------------
  - No imagined / model-rollout transitions (the WM's own imagination
    rollout) -- only REAL recorded `(z_t, z_{t+1})` pairs from
    `collection/` episodes are used as Bellman-backup transitions.
  - No QP/controller/demo wiring -- this file only trains and evaluates
    `V_theta` in isolation. A later pass swaps it in for the analytic
    `h(x)` barrier in experiments/ab_smooth_vs_bangbang's HOCBF-QP.
  - Honest risk: `V_theta` inherits every OOD brittleness of the frozen WM
    (garbage-in-garbage-out on any latent far from the WM's training
    distribution -- nothing here makes the WM itself conservative there),
    and is trained on real transitions from one scripted-policy dataset
    (`collection/data/cup_collect_v1`) only -- its learned "reachable
    failure set" is only as good as what that policy actually visited.

Usage (smoke, what this task ran -- see the task's returned exact command
for the intended FULL run, which this file does NOT execute):
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=5 \\
        /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python \\
        -m world_model.train_latent_safety_value \\
        --dreamer-fmt-dir world_model/data/dreamer_fmt/cup_collect_v1 \\
        --collection-dir collection/data/cup_collect_v1 \\
        --checkpoint world_model/logdir/wm_cup_collect/best.pt \\
        --skip-convert --max-episodes 8 --eval-holdout 2 \\
        --epochs 3 --batch-size 512 --target-update-interval 20 \\
        --tag smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from world_model.dataset_adapter import convert_dir  # noqa: E402
from world_model.monitor_head import _episode_id_from_key, roc_curve  # noqa: E402

ARTIFACTS_DIR = Path("/common/users/ss5772/bimanual/artifacts")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(WORLD_MODEL_ROOT / "train_wm_config.yaml"))
    p.add_argument("--collection-dir", default="/common/users/ss5772/bimanual/collection/data/cup_collect_v1")
    p.add_argument("--dreamer-fmt-dir", default="/common/users/ss5772/bimanual/world_model/data/dreamer_fmt/cup_collect_v1")
    p.add_argument("--checkpoint", default=str(WORLD_MODEL_ROOT / "logdir" / "wm_cup_collect" / "best.pt"),
                    help="a trained WorldModel checkpoint (frozen -- V_theta never backprops into it)")
    p.add_argument("--device", default=None)
    p.add_argument("--skip-convert", action="store_true", help="pass this if --dreamer-fmt-dir is already populated (e.g. cup_collect_v1 already converted).")
    p.add_argument("--max-episodes", type=int, default=None, help="cap the number of loaded episodes (sorted by key, first N kept) -- for a SHORT smoke only; omit for a full run.")
    p.add_argument("--eval-holdout", type=int, default=20, help="number of episodes (by sorted key, last N) held out for the separation-AUC eval; NOT used as training transitions.")
    p.add_argument("--seed", type=int, default=0)

    # Value-net / Bellman-target hyperparameters.
    p.add_argument("--hidden", type=int, default=64, help="mirrors monitor_head.MonitorHead's hidden width.")
    p.add_argument("--gamma", type=float, default=0.99, help="Fisac discounted-safety-Bellman discount factor.")
    p.add_argument("--target-update-interval", type=int, default=200, help="hard-copy V_theta -> V_target every this many GRADIENT STEPS (periodic target network, DQN/Fisac-style).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=100, help="passes over the shuffled train-transition set.")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--imminent-horizon", type=int, default=20, help="a held-out real state within this many steps of a recorded failure is labeled 'imminent-unsafe' for the go/no-go separation-AUC eval.")

    p.add_argument("--tag", default="run")
    p.add_argument("--ckpt-out", default=None, help="if set, saves V_theta's state_dict + metadata to this path. NOT saved by default (mirrors monitor_head's --head-ckpt-out opt-in pattern, so a smoke run can't accidentally clobber a real checkpoint).")
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# Value net: tiny MLP, same shape family as monitor_head.MonitorHead, single
# scalar tanh-bounded output -- see module docstring's sign-convention
# section for why tanh (bounds V to the same (-1,1) range as g(l)).
# --------------------------------------------------------------------------
class LatentSafetyValue(nn.Module):
    def __init__(self, latent_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, z):
        return torch.tanh(self.net(z))  # (..., 1) in (-1, 1); V<0 <=> unsafe (see module docstring).


def margin(l: torch.Tensor) -> torch.Tensor:
    """g(l) = +1 on safe steps (l=0), -1 on failure steps (l=1)."""
    return 1.0 - 2.0 * l.float()


def _load_failure_signal(collection_dir: Path, dreamer_key: str) -> np.ndarray:
    """l_t = collision[t] | dropped[t] | cube_dropped[t], read straight off
    the RAW collection/ episode.npz (never from anything the WM encoder
    saw) -- see module docstring's sign-convention section for what each
    array means. `cube_dropped` is absent on non-cup_collect_v1-style
    episodes (e.g. cup_yor_full, which uses collect_cup.py without that
    field) -- degrades to an all-zeros contribution there, same
    graceful-degradation pattern as monitor_head.py's `topple`. Returns
    shape (T,) float32 in {0., 1.}."""
    ep_dir = collection_dir / _episode_id_from_key(dreamer_key)
    with np.load(ep_dir / "episode.npz") as f:
        collision = np.asarray(f["collision"], dtype=np.float32)
        dropped = np.asarray(f["dropped"], dtype=np.float32)
        if "cube_dropped" in f:
            cube_dropped = np.asarray(f["cube_dropped"], dtype=np.float32)
        else:
            cube_dropped = np.zeros_like(collision)
    l = np.clip(collision + dropped + cube_dropped, 0.0, 1.0).astype(np.float32)
    return l


@torch.no_grad()
def compute_latents_and_failure(wm, episodes: dict, collection_dir: Path, device):
    """Mirrors monitor_head.compute_latents_and_labels's frozen-WM
    forward pass (full-length teacher-forced posterior, `wm` stays in
    eval()/no_grad throughout this file) but returns the combined
    3-way-OR failure signal instead of per-label spill/collision/topple.
    Returns {dreamer_key -> {'z': (T,D) tensor, 'l': (T,) tensor}}."""
    out = {}
    for key, ep in sorted(episodes.items()):
        batch = {
            "image": ep["image"][None],
            "proprio": ep["proprio"][None],
            "action": ep["action"][None],
            "is_first": ep["is_first"][None],
            "is_terminal": ep["is_terminal"][None],  # required by wm.preprocess(); unused past that
        }
        data = wm.preprocess(batch)
        embed = wm.encoder(data)
        post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
        feat = wm.dynamics.get_feat(post)[0]  # (T, latent_dim), detached (no_grad context)
        l = _load_failure_signal(collection_dir, key)
        t = min(feat.shape[0], l.shape[0])
        out[key] = {
            "z": feat[:t].detach().to(device),
            "l": torch.as_tensor(l[:t], device=device),
        }
    return out


def build_transitions(latents: dict, keys: list[str]):
    """Real recorded (z_t, l_t, z_{t+1}) transitions, t=0..T-2 per episode
    (the final frame of each episode has no z_{t+1} and is simply excluded
    from the Bellman-backup transition set -- see module docstring's "NOT
    done here": no imagined transitions fill that gap)."""
    zs, ls, zns = [], [], []
    for k in keys:
        z, l = latents[k]["z"], latents[k]["l"]
        if z.shape[0] < 2:
            continue
        zs.append(z[:-1])
        ls.append(l[:-1])
        zns.append(z[1:])
    return torch.cat(zs, dim=0), torch.cat(ls, dim=0), torch.cat(zns, dim=0)


def imminent_unsafe_labels(l: torch.Tensor, horizon: int) -> np.ndarray:
    """Binarized eval label per real state z_t: 1 if a failure (l=1) occurs
    anywhere in [t, t+horizon] (inclusive), else 0. O(T) backward scan
    (distance-to-next-failure), not O(T*horizon)."""
    l_np = l.detach().cpu().numpy()
    T = len(l_np)
    dist = np.full(T, np.inf, dtype=np.float64)
    running = np.inf
    for t in range(T - 1, -1, -1):
        running = 0.0 if l_np[t] > 0.5 else (running + 1.0 if np.isfinite(running) else np.inf)
        dist[t] = running
    return (dist <= horizon).astype(np.float32)


def save_metrics_plot(v_eval: np.ndarray, y_imminent: np.ndarray, y_now: np.ndarray, roc, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax_roc, ax_hist = axes
    if roc is not None:
        fpr, tpr, auc = roc
        ax_roc.plot(fpr, tpr, marker=".", label=f"AUC={auc:.3f}")
        ax_roc.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
        ax_roc.legend(fontsize=8)
    else:
        ax_roc.text(0.5, 0.5, "AUC undefined\n(single class in eval split)", ha="center", va="center", fontsize=9)
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    ax_roc.set_title("imminent-unsafe separation (score=-V_theta)", fontsize=9)

    ax_hist.hist(v_eval[y_now < 0.5], bins=30, alpha=0.6, label="safe now (l=0)", density=True)
    ax_hist.hist(v_eval[y_now >= 0.5], bins=30, alpha=0.6, label="failure now (l=1)", density=True)
    ax_hist.axvline(0.0, color="k", linestyle="--", linewidth=0.8)
    ax_hist.set_xlabel("V_theta(z)")
    ax_hist.set_title("V_theta histogram: safe-now vs failure-now", fontsize=9)
    ax_hist.legend(fontsize=8)

    fig.suptitle("Latent safety value V_theta(z) -- Fisac discounted-safety-Bellman TD")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
        print(f"[latent_safety_value] converting {collection_dir} -> {dreamer_dir}")
        convert_dir(collection_dir, dreamer_dir)

    episodes = dreamer_tools.load_episodes(dreamer_dir)
    print(f"[latent_safety_value] loaded {len(episodes)} episodes from {dreamer_dir}")
    if args.max_episodes is not None:
        keep_keys = sorted(episodes.keys())[: args.max_episodes]
        episodes = {k: episodes[k] for k in keep_keys}
        print(f"[latent_safety_value] --max-episodes={args.max_episodes} -> using {len(episodes)} episodes (SMOKE-style cap; omit for a full run)")

    train_eps, eval_eps = split_episodes(episodes, args.eval_holdout)
    print(f"[latent_safety_value] split: {len(train_eps)} train / {len(eval_eps)} eval episodes")

    wm, image_shape, proprio_dim, num_actions = build_wm_from_checkpoint(config, episodes, args.checkpoint)
    print(f"[latent_safety_value] frozen WorldModel loaded from {args.checkpoint} "
          f"(image={image_shape} proprio_dim={proprio_dim}); V_theta trains ONLY on its latent output.")

    latents = compute_latents_and_failure(wm, episodes, collection_dir, device)
    latent_dim = next(iter(latents.values()))["z"].shape[-1]
    print(f"[latent_safety_value] latent_dim={latent_dim}")

    z_train, l_train, znext_train = build_transitions(latents, list(train_eps.keys()))
    n_train = z_train.shape[0]
    n_fail_train = int(l_train.sum().item())
    print(f"[latent_safety_value] train transitions={n_train} (failure l=1 count={n_fail_train}, "
          f"{100.0 * n_fail_train / max(1, n_train):.1f}%)")

    value_net = LatentSafetyValue(latent_dim, hidden=args.hidden).to(device)
    target_net = LatentSafetyValue(latent_dim, hidden=args.hidden).to(device)
    target_net.load_state_dict(value_net.state_dict())
    target_net.eval()
    for p_ in target_net.parameters():
        p_.requires_grad_(False)

    opt = torch.optim.Adam(value_net.parameters(), lr=args.lr)

    global_step = 0
    for epoch in range(args.epochs):
        perm = torch.randperm(n_train, device=device)
        losses = []
        for start in range(0, n_train, args.batch_size):
            idx = perm[start:start + args.batch_size]
            zb, lb, znb = z_train[idx], l_train[idx], znext_train[idx]

            with torch.no_grad():
                gb = margin(lb)
                v_next = target_net(znb).squeeze(-1)
                bootstrap = (1.0 - args.gamma) * gb + args.gamma * v_next
                soft_target = torch.minimum(gb, bootstrap)
                y = torch.where(lb > 0.5, gb, soft_target)  # explicit failure/terminal clamp -- see module docstring

            pred = value_net(zb).squeeze(-1)
            loss = F.mse_loss(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            global_step += 1
            if global_step % args.target_update_interval == 0:
                target_net.load_state_dict(value_net.state_dict())
            losses.append(loss.item())

        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
            print(f"[latent_safety_value] epoch {epoch}: train_mse_loss={np.mean(losses):.5f} "
                  f"(global_step={global_step})")

    # Final target sync (cosmetic -- eval below uses value_net directly, never target_net).
    target_net.load_state_dict(value_net.state_dict())

    # --------------------------------------------------------------------
    # Eval: imminent-unsafe separation AUC on held-out episodes.
    # --------------------------------------------------------------------
    eval_keys = list(eval_eps.keys())
    if not eval_keys:
        print("[latent_safety_value] WARNING: no held-out episodes -- evaluating on the train split (smoke-only fallback).")
        eval_keys = list(train_eps.keys())

    z_list, l_list, imm_list = [], [], []
    for k in eval_keys:
        z, l = latents[k]["z"], latents[k]["l"]
        z_list.append(z)
        l_list.append(l)
        imm_list.append(torch.as_tensor(imminent_unsafe_labels(l, args.imminent_horizon), device=device))
    z_eval = torch.cat(z_list, dim=0)
    y_now = torch.cat(l_list, dim=0).cpu().numpy()
    y_imminent = torch.cat(imm_list, dim=0).cpu().numpy()

    with torch.no_grad():
        v_eval = value_net(z_eval).squeeze(-1).cpu().numpy()

    score = -v_eval  # higher score = judged MORE unsafe (V<0 <=> unsafe, see sign convention)
    lo, hi = float(score.min()), float(score.max())
    prob = (score - lo) / (hi - lo + 1e-8)  # min-max into [0,1] for monitor_head.roc_curve's threshold sweep; AUC is scale-invariant
    roc = roc_curve(y_imminent, prob)
    auc_str = f"{roc[2]:.4f}" if roc is not None else "undefined (single class in eval split)"
    n_imminent = int(y_imminent.sum())
    print(f"[latent_safety_value] eval states={len(y_imminent)} imminent-unsafe(horizon={args.imminent_horizon})="
          f"{n_imminent} ({100.0 * n_imminent / max(1, len(y_imminent)):.1f}%) | separation_roc_auc={auc_str}")

    mean_v_safe_now = float(v_eval[y_now < 0.5].mean()) if np.any(y_now < 0.5) else float("nan")
    mean_v_fail_now = float(v_eval[y_now >= 0.5].mean()) if np.any(y_now >= 0.5) else float("nan")
    print(f"[latent_safety_value] sanity: mean V(safe-now)={mean_v_safe_now:.4f} "
          f"mean V(failure-now)={mean_v_fail_now:.4f} (expect safe > 0 > failure, and safe > failure)")

    out_path = ARTIFACTS_DIR / f"latent_safety_value_{args.tag}.png"
    save_metrics_plot(v_eval, y_imminent, y_now, roc, out_path)
    print(f"[latent_safety_value] plot saved to {out_path}")

    metrics_path = ARTIFACTS_DIR / f"latent_safety_value_{args.tag}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "n_train_transitions": n_train,
                "n_train_failure": n_fail_train,
                "n_eval_states": int(len(y_imminent)),
                "n_eval_imminent_unsafe": n_imminent,
                "imminent_horizon": args.imminent_horizon,
                "separation_roc_auc": roc[2] if roc is not None else None,
                "mean_v_safe_now": mean_v_safe_now,
                "mean_v_failure_now": mean_v_fail_now,
                "gamma": args.gamma,
                "epochs": args.epochs,
                "final_train_mse_loss": float(np.mean(losses)),
            },
            f,
            indent=2,
        )
    print(f"[latent_safety_value] metrics json saved to {metrics_path}")

    if args.ckpt_out:
        ckpt_path = Path(args.ckpt_out)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "value_state_dict": value_net.state_dict(),
                "latent_dim": latent_dim,
                "hidden": args.hidden,
                "gamma": args.gamma,
                "sign_convention": "V<0 <=> unsafe; g(l)=1-2*l; margin range (-1,1) via tanh output",
                "image_shape": tuple(image_shape),
                "proprio_dim": int(proprio_dim),
                "num_actions": int(num_actions),
                "wm_checkpoint": str(args.checkpoint),
                "failure_signal": "collision | dropped(spill) | cube_dropped",
            },
            ckpt_path,
        )
        print(f"[latent_safety_value] V_theta checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
