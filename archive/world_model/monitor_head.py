"""world_model/monitor_head.py

The FAILURE-MONITOR head (SCAFFOLD + SMOKE ONLY -- see world_model/README.md).

A small MLP classifier on the WM latent z_t (= `dynamics.get_feat(post)`,
i.e. `[stoch, deter]` from a *frozen, already-trained* WorldModel's
posterior) -> P(spill), P(collision). Input is ONLY the latent -- per
docs/design/state_spaces.md's sim2real line, object pose / spill / collision
are LABEL fields, never WM inputs, and this file never feeds them to the
classifier's input (they are read purely as targets, from the RAW
`collection/` episode -- not from anything the WM encoder ever saw).

Label sourcing (current schema, see collection/episode.py):
  - `collision[t]`  (T,) bool  -- real PER-STEP signal, read straight from
    the episode's raw `episode.npz` (dataset_adapter.py's converted
    "is_terminal" combines collision|dropped, which is NOT what we want
    here, so this file re-reads the raw per-step `collision` array itself).
  - `spilled_count`  (episode-level int, from `metadata.json`) -- there is
    NO per-step spill signal in the schema yet (no cup/balls scene wired up
    -- see episode.py's build_episode_metadata docstring: spilled_count is
    always 0 today). We broadcast `spilled_count > 0` as a constant
    per-episode proxy label across every timestep of that episode. This is
    a documented placeholder, not a real per-step spill signal; once a cup
    scene lands with a true per-step spill flag, replace `_load_raw_labels`
    accordingly (one line change).
  - On `collection/data/smoke` specifically, BOTH labels are degenerate (all
    3 smoke episodes have collision=False, spilled_count=0) -- see this
    file's own smoke run output. `collection/data/cup_verify` (already
    collected elsewhere) has real positive spill/collision episodes and is
    the right dataset for a *meaningful* monitor-head run once the WM is
    trained on comparable data; this task's smoke intentionally stays on
    collection/data/smoke per instructions.

Metrics (precision/recall, ROC-AUC, calibration curve) are computed with
plain numpy -- `sklearn` is not installed in the `mujoco_ur5e` env used here
and this task must not add new dependencies -- see the manual
`precision_recall`/`roc_curve`/`calibration_curve` helpers below.

Usage (smoke, what this task ran):
    CUDA_VISIBLE_DEVICES=3 python -m world_model.monitor_head \\
        --dreamer-fmt-dir world_model/data/dreamer_fmt/smoke \\
        --collection-dir collection/data/smoke \\
        --checkpoint world_model/logdir/smoke_run/latest.pt \\
        --epochs 20 --eval-holdout 1 --tag smoke
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
import yaml

DREAMERV3_ROOT = "/common/users/ss5772/dreamerv3"
if DREAMERV3_ROOT not in sys.path:
    sys.path.insert(0, DREAMERV3_ROOT)

import tools as dreamer_tools  # noqa: E402

WORLD_MODEL_ROOT = Path(__file__).resolve().parent
if str(WORLD_MODEL_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(WORLD_MODEL_ROOT.parent))
from world_model.train_wm import build_spaces, infer_obs_shapes, split_episodes  # noqa: E402
from world_model.eval_wm import build_wm_from_checkpoint  # noqa: E402
from world_model.dataset_adapter import convert_dir  # noqa: E402

ARTIFACTS_DIR = Path("/common/users/ss5772/bimanual/artifacts")

# All labels this file knows how to read straight off a raw collection/
# episode.npz (see _load_raw_labels). LABEL_NAMES is the ACTIVE subset for a
# given run -- set from --labels in main() (default stays ("spill",
# "collision") for back-compat with existing cup/-smoke invocations).
#   - "spill"/"collision": unchanged, see _load_raw_labels docstring.
#   - "topple": per-step `count_toppled` array (float, #blocks near the
#     tower zone that are tilted-over/knocked-off -- see
#     scripts/stack_task_utils.py::count_toppled), written by
#     collection/collect_stack.py's StackEpisodeRecorder._on_step as an
#     "extra: dense failure label" on every stacking episode. This is the
#     stacking task's WM-visible failure mode (cup's spill has no analogue
#     here -- stacking episodes have no `dropped`-as-spill repurposing, see
#     collect_stack.py's own docstring: dropped is block_dropped(), and
#     spilled_count in metadata.json is repurposed as the EPISODE-FINAL
#     count_toppled()). Only present in stacking-collected episodes; absent
#     (e.g. on cup/smoke data) it degrades gracefully to all-zeros.
ALL_LABELS = ("spill", "collision", "topple")
LABEL_NAMES = ("spill", "collision")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(WORLD_MODEL_ROOT / "train_wm_config.yaml"))
    p.add_argument("--collection-dir", default="/common/users/ss5772/bimanual/collection/data/smoke")
    p.add_argument("--dreamer-fmt-dir", required=True)
    p.add_argument("--checkpoint", required=True, help="a trained WorldModel checkpoint (frozen -- monitor head never backprops into it)")
    p.add_argument("--device", default=None)
    p.add_argument("--eval-holdout", type=int, default=1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--tag", default="run")
    p.add_argument("--skip-convert", action="store_true")
    p.add_argument(
        "--spill-signal", choices=["broadcast", "per_step"], default="broadcast",
        help="'broadcast' (default, back-compat): spilled_count>0 from metadata.json, "
             "constant across every timestep of the episode (the only signal available "
             "on collection/data/smoke, which has no cup scene). 'per_step': the raw "
             "per-step `dropped` array, which collection/collect_cup.py's cup-task "
             "collector REPURPOSES as a true per-step spill flag "
             "(`dropped=bool(count_spilled(model,data)>0)` at every step -- see that "
             "file's own comment) -- use this on cup datasets (e.g. cup_yor_full) for a "
             "non-degenerate, temporally-real spill-onset label instead of the constant "
             "per-episode proxy.")
    p.add_argument(
        "--labels", default="spill,collision",
        help="comma-separated subset of ALL_LABELS=('spill','collision','topple') to train/eval "
             "the head on (default 'spill,collision', back-compat with cup runs). Use "
             "'topple,collision' on stacking datasets (collect_stack.py episodes carry a real "
             "per-step `count_toppled` array; cup datasets have no such field and spill has no "
             "stacking analogue -- see _load_raw_labels).")
    p.add_argument(
        "--head-ckpt-out", default=None,
        help="if set, saves the trained MonitorHead's state_dict + latent_dim + obs "
             "shapes to this path (for reuse outside this script, e.g. eval/ harness "
             "integration) -- NOT saved by default.")
    return p.parse_args(argv)


class MonitorHead(nn.Module):
    """Tiny MLP: latent z_t -> logits for [spill, collision]. Deliberately
    separate from dreamerv3-torch's own networks.MLP -- this head is trained
    independently of the WM (frozen encoder/dynamics; only this head's
    params get gradients), so it has no dependency on WorldModel internals
    beyond consuming its `get_feat(post)` output tensor."""

    def __init__(self, latent_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, len(LABEL_NAMES)),
        )

    def forward(self, z):
        return self.net(z)  # raw logits, shape (..., 2)


def _episode_id_from_key(dreamer_key: str) -> str:
    """'episode_000000-60' -> 'episode_000000' (dreamerv3-torch's
    save_episodes filename convention is '<id>-<length>.npz')."""
    return dreamer_key.rsplit("-", 1)[0]


def _load_raw_labels(
    collection_dir: Path, dreamer_key: str, spill_signal: str = "broadcast"
) -> dict[str, np.ndarray]:
    """Reads collision[t]/spill[t]/topple[t] straight from the RAW
    collection/ episode -- never from anything routed through the WM encoder.
    spill_signal='broadcast': per-episode spilled_count>0 constant (old
    placeholder, back-compat). spill_signal='per_step': raw dropped[t] array,
    which collect_cup.py's cup-task collector repurposes as a real per-step
    spill flag -- see this file's module docstring / --spill-signal help.
    topple[t]: raw per-step `count_toppled` array (>0 -> 1.0), written by
    collect_stack.py on stacking episodes only (see ALL_LABELS comment
    above); absent on other datasets (e.g. cup/smoke), where it degrades to
    an all-zeros array of the same length as collision -- so requesting
    "topple" on non-stacking data is well-defined (constant-negative) rather
    than an error. Returns {"spill","collision","topple"}, each shape (T,)
    float32 in {0,1}."""
    ep_dir = collection_dir / _episode_id_from_key(dreamer_key)
    with np.load(ep_dir / "episode.npz") as f:
        collision = np.asarray(f["collision"], dtype=np.float32)
        if spill_signal == "per_step":
            spill = np.asarray(f["dropped"], dtype=np.float32)
        else:
            meta = json.loads((ep_dir / "metadata.json").read_text())
            spilled = float(meta.get("spilled_count", 0) > 0)
            spill = np.full_like(collision, spilled, dtype=np.float32)
        if "count_toppled" in f:
            topple = (np.asarray(f["count_toppled"], dtype=np.float32) > 0).astype(np.float32)
        else:
            topple = np.zeros_like(collision, dtype=np.float32)
    return {"spill": spill, "collision": collision, "topple": topple}


@torch.no_grad()
def compute_latents_and_labels(wm, episodes: dict, collection_dir: Path, device, spill_signal: str = "broadcast"):
    """For every episode: full-length teacher-forced posterior through the
    FROZEN world model (`wm` stays in eval() / no_grad throughout this
    file -- the monitor head trains on top of, never through, the WM), then
    read the labels straight off the raw collection/ episode. Returns a dict
    keyed by dreamer episode key -> {'z': (T,D), 'spill': (T,), 'collision': (T,)}."""
    out = {}
    for key, ep in sorted(episodes.items()):
        batch = {
            "image": ep["image"][None],
            "proprio": ep["proprio"][None],
            "action": ep["action"][None],
            "is_first": ep["is_first"][None],
            "is_terminal": ep["is_terminal"][None],  # required by wm.preprocess(); unused past that (cont_head is not called here)
        }
        data = wm.preprocess(batch)
        embed = wm.encoder(data)
        post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
        feat = wm.dynamics.get_feat(post)[0]  # (T, latent_dim), detached (no_grad context)
        raw = _load_raw_labels(collection_dir, key, spill_signal=spill_signal)
        t = min([feat.shape[0]] + [v.shape[0] for v in raw.values()])
        out[key] = {"z": feat[:t].detach().to(device)}
        for name, arr in raw.items():
            out[key][name] = torch.as_tensor(arr[:t], device=device)
    return out


def stack_split(latents: dict, keys: list[str]):
    z = torch.cat([latents[k]["z"] for k in keys], dim=0)
    y = torch.stack(
        [torch.cat([latents[k][name] for k in keys], dim=0) for name in LABEL_NAMES],
        dim=-1,
    )
    return z, y


# --------------------------------------------------------------------------
# Manual metrics (no sklearn in the mujoco_ur5e env -- see module docstring).
# --------------------------------------------------------------------------
def precision_recall(y_true: np.ndarray, y_prob: np.ndarray, thresh: float = 0.5) -> tuple[float, float]:
    y_pred = (y_prob >= thresh).astype(int)
    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return precision, recall


def roc_curve(y_true: np.ndarray, y_prob: np.ndarray, n_thresh: int = 50):
    """Returns (fpr, tpr, auc), or None if y_true has a single class (AUC is
    undefined -- reported as such rather than silently returning 0.5/nan)."""
    if len(np.unique(y_true)) < 2:
        return None
    thresholds = np.linspace(1.0, 0.0, n_thresh)
    P = float(np.sum(y_true == 1))
    N = float(np.sum(y_true == 0))
    tprs, fprs = [], []
    for th in thresholds:
        pred = (y_prob >= th).astype(int)
        tp = float(np.sum((pred == 1) & (y_true == 1)))
        fp = float(np.sum((pred == 1) & (y_true == 0)))
        tprs.append(tp / P if P > 0 else 0.0)
        fprs.append(fp / N if N > 0 else 0.0)
    fprs, tprs = np.array(fprs), np.array(tprs)
    order = np.argsort(fprs)
    # np.trapz was removed in numpy>=2.0 (renamed np.trapezoid); fall back to
    # a manual trapezoidal sum so this runs on either numpy version.
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    if _trapz is not None:
        auc = float(_trapz(tprs[order], fprs[order]))
    else:
        x, y = fprs[order], tprs[order]
        auc = float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))
    return fprs[order], tprs[order], auc


def calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bins[1:-1]), 0, n_bins - 1)
    mean_pred, frac_pos, counts = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        mean_pred.append(float(y_prob[mask].mean()))
        frac_pos.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))
    return np.array(mean_pred), np.array(frac_pos), np.array(counts)


def save_metrics_plot(results: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(LABEL_NAMES), figsize=(5 * len(LABEL_NAMES), 8))
    for i, name in enumerate(LABEL_NAMES):
        r = results[name]
        ax_roc, ax_cal = axes[0, i], axes[1, i]
        if r["roc"] is not None:
            fpr, tpr, auc = r["roc"]
            ax_roc.plot(fpr, tpr, marker=".", label=f"AUC={auc:.3f}")
            ax_roc.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
            ax_roc.legend(fontsize=8)
        else:
            ax_roc.text(0.5, 0.5, "AUC undefined\n(single class in eval split)",
                        ha="center", va="center", fontsize=9)
        ax_roc.set_xlim(0, 1)
        ax_roc.set_ylim(0, 1)
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.set_title(f"{name}: ROC (precision={r['precision']:.2f}, recall={r['recall']:.2f})", fontsize=9)

        mean_pred, frac_pos, counts = r["calibration"]
        if len(mean_pred) > 0:
            ax_cal.plot(mean_pred, frac_pos, marker="o", label="model")
        ax_cal.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8, label="perfect")
        ax_cal.set_xlim(0, 1)
        ax_cal.set_ylim(0, 1)
        ax_cal.set_xlabel("mean predicted P")
        ax_cal.set_ylabel("empirical frequency")
        ax_cal.set_title(f"{name}: calibration", fontsize=9)
        ax_cal.legend(fontsize=8)

    fig.suptitle("Failure-monitor head: latent-only P(spill), P(collision)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main(argv=None) -> None:
    args = parse_args(argv)
    global LABEL_NAMES
    LABEL_NAMES = tuple(s.strip() for s in args.labels.split(","))
    assert all(n in ALL_LABELS for n in LABEL_NAMES), f"--labels must be a subset of {ALL_LABELS}, got {LABEL_NAMES}"
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
        print(f"[monitor_head] converting {collection_dir} -> {dreamer_dir}")
        convert_dir(collection_dir, dreamer_dir)

    episodes = dreamer_tools.load_episodes(dreamer_dir)
    print(f"[monitor_head] loaded {len(episodes)} episodes: {sorted(episodes.keys())}")
    train_eps, eval_eps = split_episodes(episodes, args.eval_holdout)
    print(f"[monitor_head] split: {len(train_eps)} train / {len(eval_eps)} eval episodes")

    wm, image_shape, proprio_dim, num_actions = build_wm_from_checkpoint(config, episodes, args.checkpoint)
    print(f"[monitor_head] frozen WorldModel loaded from {args.checkpoint} "
          f"(image={image_shape} proprio_dim={proprio_dim}); monitor head trains ONLY on its latent output.")

    latents = compute_latents_and_labels(wm, episodes, collection_dir, device, spill_signal=args.spill_signal)
    print(f"[monitor_head] spill_signal={args.spill_signal}")
    latent_dim = next(iter(latents.values()))["z"].shape[-1]
    print(f"[monitor_head] latent_dim={latent_dim}")

    z_train, y_train = stack_split(latents, list(train_eps.keys()))
    z_eval, y_eval = stack_split(latents, list(eval_eps.keys())) if eval_eps else (None, None)
    print(f"[monitor_head] train timesteps={z_train.shape[0]} "
          f"(spill_pos={int(y_train[:,0].sum())}, collision_pos={int(y_train[:,1].sum())}) | "
          f"eval timesteps={0 if z_eval is None else z_eval.shape[0]}")

    head = MonitorHead(latent_dim, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        opt.zero_grad()
        logits = head(z_train)
        loss = bce(logits, y_train)
        loss.backward()
        opt.step()
        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs - 1:
            print(f"[monitor_head] epoch {epoch}: train_bce_loss={loss.item():.4f}")

    results = {}
    with torch.no_grad():
        if z_eval is not None and z_eval.shape[0] > 0:
            eval_logits = head(z_eval)
            eval_probs = torch.sigmoid(eval_logits).cpu().numpy()
            eval_labels = y_eval.cpu().numpy()
        else:
            print("[monitor_head] WARNING: no held-out episodes -- evaluating on the train split (smoke-only fallback).")
            eval_logits = head(z_train)
            eval_probs = torch.sigmoid(eval_logits).cpu().numpy()
            eval_labels = y_train.cpu().numpy()

    for i, name in enumerate(LABEL_NAMES):
        yt, yp = eval_labels[:, i], eval_probs[:, i]
        precision, recall = precision_recall(yt, yp)
        roc = roc_curve(yt, yp)
        cal = calibration_curve(yt, yp)
        results[name] = {"precision": precision, "recall": recall, "roc": roc, "calibration": cal}
        auc_str = f"{roc[2]:.3f}" if roc is not None else "undefined (single class)"
        print(f"[monitor_head] eval/{name}: precision={precision:.3f} recall={recall:.3f} "
              f"roc_auc={auc_str} n_pos={int(yt.sum())}/{len(yt)}")

    out_path = ARTIFACTS_DIR / f"monitor_{args.tag}.png"
    save_metrics_plot(results, out_path)
    print(f"[monitor_head] metrics plot saved to {out_path}")

    metrics_path = ARTIFACTS_DIR / f"monitor_{args.tag}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(
            {
                name: {
                    "precision": results[name]["precision"],
                    "recall": results[name]["recall"],
                    "roc_auc": results[name]["roc"][2] if results[name]["roc"] is not None else None,
                }
                for name in LABEL_NAMES
            },
            f,
            indent=2,
        )
    print(f"[monitor_head] metrics json saved to {metrics_path}")

    if args.head_ckpt_out:
        head_ckpt_path = Path(args.head_ckpt_out)
        head_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "head_state_dict": head.state_dict(),
                "latent_dim": latent_dim,
                "hidden": args.hidden,
                "image_shape": tuple(image_shape),
                "proprio_dim": int(proprio_dim),
                "num_actions": int(num_actions),
                "label_names": LABEL_NAMES,
                "spill_signal": args.spill_signal,
                "wm_checkpoint": str(args.checkpoint),
            },
            head_ckpt_path,
        )
        print(f"[monitor_head] head checkpoint saved to {head_ckpt_path}")


if __name__ == "__main__":
    main()
