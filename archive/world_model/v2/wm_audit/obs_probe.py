"""wm_audit/obs_probe.py -- is the cup CONTENTS / FILL observable from the current sensors?
(Kim, Nakamura, Bajcsy 2025 'estimation gap' test, cheap version.) SAVED DATA ONLY.

Probes, trained on TRAIN shards and scored on TEST shards (same %5 split as latent_hj):
  * ZED DINOv2 patch tokens (cache_zed_px112, 64 x 384) of the first 5 frames (cup upright, idle,
    contents settled) -> fill height pend_h, n_balls (beads), kind (beads vs particles)
  * the same at a mid-motion frame before any spill (0.3-0.6 s before onset for spill eps)
  * proprio + cup kinematics are INDEPENDENT of contents by construction (rig arms are kinematic
    playback; the plan is sampled without reference to the contents) -- checked empirically too:
    the plan statistics (peak cup accel / tilt) vs fill.
Kernel ridge (linear kernel, dual form) with lambda picked on a VAL fold; R^2 and AUROC reported
with a learning curve (25/50/100% of train episodes) so 'more data' can be read off directly.

  python -m world_model.v2.wm_audit.obs_probe --out $AUD/results/obs_probe.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from world_model.v2.wm_audit import common as C

DATA = Path("/common/users/ss5772/datasets/bimanual_wm_v2/spill_v2")
CACHE = Path("/common/users/ss5772/bimanual_data/dinowm/cache_zed_px112")


def load(max_shards=1000):
    man = json.load(open(DATA / "manifest_merged.json"))
    meta = {int(e["seed"]): e for e in man["episodes"]}
    shards = sorted(glob.glob(str(DATA / "shard_*.npz")))
    split_of = {Path(p).name: ("test" if i % 5 == 4 else ("val" if i % 5 == 3 else "train")) for i, p in enumerate(shards)}
    rows = []
    for p in shards[:max_shards]:
        name = Path(p).name
        cf = CACHE / name.replace(".npz", ".zed_px112.npy")
        if not cf.exists():
            continue
        tok = np.load(cf, mmap_mode="r")
        with np.load(p) as f:
            st, ln, seeds, onset = f["ep_start"], f["ep_len"], f["ep_seed"], f["ep_onset_frame"]
            cup_acc = f["cup_acc"]; cup_pose = f["cup_pose"]
        for s, L, sd, on in zip(st, ln, seeds, onset):
            e = meta.get(int(sd))
            if e is None:
                continue
            f0 = tok[s:s + 5].astype(np.float32).mean(0)                        # (64, 384) idle, upright
            # mid-motion pre-spill frame: 0.45 s before onset for spill eps, else 60% of the episode
            fm = int(on - 7) if on >= 0 else int(0.6 * L)
            fm = int(np.clip(fm, 5, L - 1))
            f1 = tok[s + fm].astype(np.float32)
            acc = np.linalg.norm(cup_acc[s:s + L, :3], axis=1)
            rows.append(dict(split=split_of[name], seed=int(sd), kind=e["contents"]["kind"],
                             n_balls=int(e["contents"]["n_balls"]), h=float(e.get("pend_h", np.nan)),
                             family=e["family"], tok0=f0.reshape(-1), tok1=f1.reshape(-1),
                             tok0_pool=f0.mean(0), plan_acc_max=float(acc.max()),
                             plan_acc_mean=float(acc.mean())))
    return rows


def kridge(Xtr, ytr, Xte, lam):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-4
    A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    K = A @ A.T
    ym = ytr.mean()
    alpha = np.linalg.solve(K + lam * np.trace(K) / len(K) * np.eye(len(K)), ytr - ym)
    return B @ (A.T @ alpha) + ym


def r2(p, y):
    return float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def probe(rows, feat, target, mask_fn, fracs=(0.25, 0.5, 1.0), lams=(1e-3, 1e-2, 1e-1, 1.0, 10.0), cls=False, seed=0):
    rng = np.random.default_rng(seed)
    R = [r for r in rows if mask_fn(r)]
    tr = [r for r in R if r["split"] == "train"]; va = [r for r in R if r["split"] == "val"]
    te = [r for r in R if r["split"] == "test"]
    y = lambda S: np.array([float(r[target]) if not cls else float(r[target] == cls) for r in S])
    X = lambda S: np.stack([r[feat] for r in S])
    out = {}
    for fr in fracs:
        idx = rng.choice(len(tr), max(20, int(fr * len(tr))), replace=False)
        trs = [tr[i] for i in idx]
        best = None
        for lam in lams:
            pv = kridge(X(trs), y(trs), X(va), lam)
            sc = C.auroc(pv, y(va) > 0.5) if cls else r2(pv, y(va))
            if best is None or sc > best[0]:
                best = (sc, lam)
        pt = kridge(X(trs), y(trs), X(te), best[1])
        sc = C.auroc(pt, y(te) > 0.5) if cls else r2(pt, y(te))
        out[f"frac{fr}"] = dict(n_train=len(trs), n_test=len(te), lam=best[1], **({"auroc": sc} if cls else {"r2": sc}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-shards", type=int, default=1000)
    a = ap.parse_args()
    rows = load(a.max_shards)
    beads = lambda r: r["kind"] == "beads"
    allk = lambda r: True
    res = dict(n_eps=len(rows), n_by_split={s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")})
    res["fill_h_beads_idle_tokens"] = probe(rows, "tok0", "h", beads)
    res["n_balls_beads_idle_tokens"] = probe(rows, "tok0", "n_balls", beads)
    res["fill_h_beads_idle_pooled"] = probe(rows, "tok0_pool", "h", beads)
    res["kind_particles_idle_tokens"] = probe(rows, "tok0", "kind", allk, cls="particles")
    res["fill_h_beads_prespill_tokens"] = probe(rows, "tok1", "h", beads)
    # contents vs plan independence (kinematic playback): correlation of fill with plan statistics
    Rb = [r for r in rows if beads(r)]
    h = np.array([r["h"] for r in Rb])
    res["corr_fill_vs_plan_acc_max"] = float(np.corrcoef(h, [r["plan_acc_max"] for r in Rb])[0, 1])
    res["corr_fill_vs_plan_acc_mean"] = float(np.corrcoef(h, [r["plan_acc_mean"] for r in Rb])[0, 1])
    res["fill_h_beads_stats"] = dict(mean=float(h.mean()), std=float(h.std()), min=float(h.min()), max=float(h.max()))
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
