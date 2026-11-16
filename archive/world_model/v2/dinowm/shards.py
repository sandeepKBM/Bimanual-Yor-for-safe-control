"""world_model/v2/dinowm/shards.py -- light, direct reader of gen_spill_data.py shards.

SIMULATION DATA ONLY (reads .npz files; no robot, no network).

Why not model/data.py prep: prep writes an uncompressed canonical copy of every
episode (3 cams x 112 px -> ~15 MB/episode, ~65 GB for 4.4k eps). Here episodes are
read straight from the compressed shards; images are optional (per-key lazy load).

Returned episodes follow the CANONICAL contract of model/data.py (so eval/harness,
model/common.step_features, SloshPhysicsPredictor and the GRU all consume them), plus
  meta.contents      TRUE contents (replayed from ep_seed with the generator's rng
                     draw order -- checked against the shard's ep_n_balls)
  meta.kind          'beads' | 'particles'
  meta.fill_h        contents fill height [m] (== generator's pend_h before clip)
  meta.family        rig motion family
  dense              200 Hz cup kinematics {acc (T200,3), axis (T200,3), idx (T,)}
  label              'spill' (beads/particles rim-exit count) or 'pend' (liquid
                     pendulum-proxy: spill_count := pend_spill, onset := first 1)

Split: deterministic per ep_seed (70/15/15), independent of which shards exist yet,
so a model trained on today's shards is never tested on its own train episodes later.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DATA = Path("/common/users/ss5772/datasets/bimanual_wm_v2/spill_v2")
STRIDE = 13
DT200 = 0.005
HZ = 1.0 / (STRIDE * DT200)          # 15.38 Hz
CAMS = ("zed", "lwrist", "rwrist")
LOWKEYS = ("proprio", "action", "action_cup", "cup_pose", "cup_vel", "cup_acc", "spilled", "pend_spill",
           "pend_phi_deg", "collision")


def split_of(seed: int) -> str:
    u = np.random.default_rng(int(seed) * 7919 + 17).random()
    return "test" if u < 0.15 else ("val" if u < 0.30 else "train")


# --------------------------------------------------------------------------- contents replay
_GEN = {}


def _gen():
    if not _GEN:
        sys.path.insert(0, str(REPO / "world_model" / "v2" / "data"))
        import contents as Cn  # noqa: E402  (read-only import; builds nothing at import)
        from gen_spill_data import RIG_FAMILY_W  # noqa: E402
        _GEN.update(C=Cn, fw=dict(RIG_FAMILY_W))
    return _GEN


def replay_contents(seed: int, particle_frac=0.15):
    """Re-draw the generator's rng sequence of episode_rig() up to the contents
    (family choice -> particle coin -> contents). Pure numpy; no MuJoCo run."""
    G = _gen(); Cn = G["C"]; fw = G["fw"]
    rng = np.random.default_rng(int(seed))
    fams = list(fw); w = np.array([fw[f] for f in fams], float)
    family = str(rng.choice(fams, p=w / w.sum()))
    if particle_frac > 0 and rng.random() < particle_frac:
        cont = Cn.Contents(n_balls=int(rng.integers(40, 81)), radius=float(rng.uniform(0.0034, 0.0040)),
                           density=1000.0, friction=0.0, dampratio=1.0, condim=1, solref_tc=0.01,
                           solimp=(0.8, 0.95, 0.002), torsional=0.0, rolling=0.0, kind="particles", poly=0.3)
        if Cn.layout(cont) is None:
            cont = Cn.Contents(**Cn.NOMINAL)
    else:
        cont = Cn.sample_contents(rng)
    d = cont.to_dict(); d["fill_h"] = float(Cn.fill_height(cont))
    return family, d


# --------------------------------------------------------------------------- loading
def shard_paths(data=DATA, min_age_s=60.0):
    """Complete shards only: skip files still being written (young mtime / bad zip)."""
    import os
    import time
    import zipfile
    out = []
    for p in sorted(glob.glob(str(Path(data) / "shard_*.npz"))):
        if time.time() - os.path.getmtime(p) < min_age_s:
            continue
        try:
            with zipfile.ZipFile(p) as z:
                z.getinfo("ep_len.npy")
        except Exception:
            continue
        out.append(p)
    return out


def load_shard(path, images=False, cams=CAMS, label="spill", particle_frac=0.15):
    """-> list of canonical episodes from one shard."""
    with np.load(path, allow_pickle=False) as f:
        keys = set(f.files)
        st, ln = f["ep_start"], f["ep_len"]
        seeds, onf, nb = f["ep_seed"], f["ep_onset_frame"], f["ep_n_balls"]
        low = {k: f[k] for k in LOWKEYS if k in keys}
        dst, dln = f["ep_dense_start"], f["ep_dense_len"]
        dacc, dax = f["cup_acc_200hz"][:, :3], f["cup_axis_200hz"]
        dpos, dquat = f["cup_pos_200hz"], f["cup_quat_200hz"]
        imgs = {c: f[f"img_{c}"] for c in cams} if images else {}
    eps = []
    for i in range(len(st)):
        s, l = int(st[i]), int(ln[i]); sl = slice(s, s + l)
        fam, cont = replay_contents(int(seeds[i]), particle_frac)
        assert cont["n_balls"] == int(nb[i]), f"contents replay mismatch {path}#{i}: {cont['n_balls']} vs {nb[i]}"
        sc = low["spilled"][sl].astype(np.int32)
        ep = dict(id=f"{Path(path).stem}#{i}", seed=int(seeds[i]), hz=HZ, T=l, t_start=0,
                  proprio=low["proprio"][sl].astype(np.float32), action=low["action"][sl].astype(np.float32),
                  action_cup=low["action_cup"][sl].astype(np.float32),
                  cup_pos=low["cup_pose"][sl, :3].astype(np.float32), cup_quat=low["cup_pose"][sl, 3:7].astype(np.float32),
                  cup_vel=low["cup_vel"][sl].astype(np.float32), cup_acc=low["cup_acc"][sl].astype(np.float32),
                  collision=low["collision"][sl].astype(bool),
                  images={c: v[sl] for c, v in imgs.items()},
                  meta=dict(contents=cont, kind=cont["kind"], fill_h=cont["fill_h"], family=fam,
                            split=split_of(int(seeds[i]))))
        ds, dl = int(dst[i]), int(dln[i])
        idx = np.minimum(np.arange(l) * STRIDE, dl - 1)
        ep["dense"] = dict(acc=dacc[ds:ds + dl].astype(np.float64), axis=dax[ds:ds + dl].astype(np.float64), idx=idx,
                           pos=dpos[ds:ds + dl], quat=dquat[ds:ds + dl])
        ep["spill_raw"] = sc; ep["onset_raw"] = int(onf[i])
        ps = low.get("pend_spill")
        ep["pend_spill"] = ps[sl].astype(np.int32) if ps is not None else None
        pp = low.get("pend_phi_deg")
        ep["pend_phi"] = pp[sl].astype(np.float64) if pp is not None else None
        set_label(ep, label)
        eps.append(ep)
    return eps


def set_label(ep, label):
    """Switch the episode's target between the real rim-exit spill and the liquid
    pendulum proxy (in place)."""
    if label == "spill":
        sc = ep["spill_raw"]
        on = np.flatnonzero(sc > sc[0])
        ep["spill_count"] = sc; ep["onset"] = int(on[0]) if len(on) else -1
    elif label == "pend":
        ps = ep["pend_spill"]
        if ps is None or (ps < 0).all():
            raise KeyError("no pend labels in this episode")
        ps = np.maximum(ps, 0)
        on = np.flatnonzero(ps > 0)
        ep["spill_count"] = ps; ep["onset"] = int(on[0]) if len(on) else -1
    else:
        raise ValueError(label)
    ep["label"] = label
    return ep


def load_all(data=DATA, images=False, label="spill", limit_shards=None, workers=8, cams=CAMS):
    ps = shard_paths(data)[:limit_shards]
    if workers > 1 and not images:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            parts = pool.starmap(load_shard, [(p, False, cams, label) for p in ps])
    else:
        parts = [load_shard(p, images, cams, label) for p in ps]
    return [e for part in parts for e in part]


def by_split(eps):
    out = {"train": [], "val": [], "test": []}
    for e in eps:
        out[e["meta"]["split"]].append(e)
    return out
