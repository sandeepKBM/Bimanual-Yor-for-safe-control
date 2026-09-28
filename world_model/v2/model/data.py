"""world_model/v2/model/data.py -- canonical spill-episode format, loaders, prep.

CANONICAL EPISODE (dict; what the eval harness and every v2 model consume)
  id        str
  hz        float                       step rate (v2 target: 15 Hz)
  T         int
  t_start   int                         first scorable step (motion start / release)
  spill_count (T,) int                  balls out of the cup (oracle), per step
  onset     int                         first rise above spill_count[t_start]; -1 never
  collision (T,) bool                   SEPARATE label, never mixed into spill
  proprio   (T,P) float32               robot-knowable state (q, qd, grippers, ...)
  action    (T,A) float32               action applied AFTER observing step t
  images    {cam: (T,H,W,3) uint8}      e.g. zed_crop, wrist_left, wrist_right
  cup_pos (T,3), cup_quat (T,4 wxyz)    cup kinematics from FK (+ fixed grasp offset)
  kin_hi    optional {hz, pos, quat, idx}  native-rate cup kinematics + index map
  meta      dict                        contents / lighting / behaviour params (OOD splits)

SOURCES
  legacy:<collection dir>   collection/data/<ds>/episode_*/episode.npz (200 Hz,
            64x64 frames: rgb_zed + rgb_wrist_{left,right}_ego). Resampled to
            --hz. Cup kinematics = MuJoCo FK of the holding arm's `*_ee_grasp`
            site composed with a FIXED cup-in-hand offset (estimated once per
            episode from the logged cup pose label -- what a real robot knows
            from its grasp); the GT cup pose itself is never a model input.
  v2:<dir>                  the new 15 Hz npz shards (world_model/v2/data, other
            agent). Key names are matched through ALIASES below so a schema
            tweak there needs a one-line alias here, not a rewrite. One episode
            per file, or multi-episode files with an `ep_len`/`episode_index` array.

PREP (westeros-CPU stage; GPU jobs only read the prepared dir):
  python -m world_model.v2.model.data prep --source legacy:collection/data/cup_collect_v1 \
      --out $PREP/cup_collect_v1 --hz 15 --workers 16
  -> <out>/ep_XXXXX.npz (canonical, uncompressed) + <out>/index.json (splits).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from world_model.v2.eval import metrics as M  # noqa: E402

ALIASES = dict(
    spill_count=("spill_count", "spilled_count", "n_spilled", "spilled"),
    onset=("onset", "spill_onset", "ep_onset_frame", "onset_frame"),   # FRAME index (not 200 Hz step)
    t_start=("t_start", "t_motion0", "release_step"),
    collision=("collision", "arm_collision"),
    proprio=("proprio",),
    action=("action", "actions", "qdd_cmd"),
    cup_pos=("cup_pos", "cup_fk_pos"),
    cup_quat=("cup_quat", "cup_fk_quat"),
    hz=("hz", "rate_hz", "control_hz"),
)
IMAGE_PREFIXES = ("rgb_", "img_", "image_", "crop_", "cam_")


# --------------------------------------------------------------------------
# small utils
# --------------------------------------------------------------------------
def _first(d, names, default=None):
    for n in names:
        if n in d:
            return d[n]
    return default


def resample_index(T_native: int, native_hz: float, hz: float) -> np.ndarray:
    return np.unique(np.clip(np.round(np.arange(0, T_native, native_hz / hz)).astype(int), 0, T_native - 1))


def window_max(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """max of x over (idx[k-1], idx[k]] so short events between samples are kept."""
    out = np.empty(len(idx), x.dtype); prev = -1
    for k, i in enumerate(idx):
        out[k] = x[prev + 1:i + 1].max() if i > prev else x[i]; prev = i
    return out


def quat_mul(a, b):
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0); w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2], -1)


def quat_conj(q):
    return q * np.array([1, -1, -1, -1])


def quat_rot(q, v):
    qv = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], -1)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[..., 1:]


# --------------------------------------------------------------------------
# FK (offline MuJoCo, simulation model only)
# --------------------------------------------------------------------------
_FK = {}


def _fk_model():
    if "m" not in _FK:
        import mujoco
        sys.path.insert(0, str(REPO / "scripts"))
        from scripts.cup_task_utils import load_yor_cup_model
        r = load_yor_cup_model(); m = r[0] if isinstance(r, tuple) else r
        d = mujoco.MjData(m)
        adr = {s: [m.jnt_qposadr[m.joint(f"{s}_arm_joint{i}").id] for i in range(1, 7)] for s in ("left", "right")}
        site = {s: m.site(f"{s}_arm_ee_grasp").id for s in ("left", "right")}
        _FK.update(m=m, d=d, adr=adr, site=site, mujoco=mujoco)
    return _FK


def fk_grasp_sites(left_q, right_q):
    """(T,6)x2 -> {'left': (pos (T,3), quat (T,4))...} of the *_arm_ee_grasp sites."""
    F = _fk_model(); m, d, mj = F["m"], F["d"], F["mujoco"]
    T = len(left_q); out = {s: (np.zeros((T, 3)), np.zeros((T, 4))) for s in ("left", "right")}
    q = np.zeros(4)
    for t in range(T):
        d.qpos[F["adr"]["left"]] = left_q[t]; d.qpos[F["adr"]["right"]] = right_q[t]
        mj.mj_kinematics(m, d)
        for s in ("left", "right"):
            sid = F["site"][s]
            out[s][0][t] = d.site_xpos[sid]
            mj.mju_mat2Quat(q, d.site_xmat[sid]); out[s][1][t] = q
    return out


def cup_kin_from_fk(left_q, right_q, cup_pose_label):
    """FK cup kinematics: holding hand's grasp site (hand closest to the cup, by
    median distance) composed with ONE fixed cup-in-hand offset (median of the
    per-step offsets while the cup is within 12 cm of that hand)."""
    ee = fk_grasp_sites(left_q, right_q)
    cp, cq = cup_pose_label[:, :3].astype(float), cup_pose_label[:, 3:7].astype(float)
    dist = {s: np.linalg.norm(ee[s][0] - cp, axis=1) for s in ee}
    hand = min(dist, key=lambda s: np.median(dist[s]))
    ep, eq = ee[hand]
    rel_p = quat_rot(quat_conj(eq), cp - ep)
    rel_q = quat_mul(quat_conj(eq), cq)
    near = dist[hand] < 0.12
    if not near.any():
        near[:] = True
    k = np.nonzero(near)[0]
    med = np.median(rel_p[k], 0)
    j = k[np.argmin(np.linalg.norm(rel_p[k] - med, axis=1))]
    pos = ep + quat_rot(eq, np.repeat(rel_p[j][None], len(ep), 0))
    quat = quat_mul(eq, np.repeat(rel_q[j][None], len(eq), 0))
    return pos.astype(np.float32), quat.astype(np.float32), hand, float(np.median(dist[hand]))


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------
LEGACY_CAMS = {"zed": "rgb_zed", "wrist_left": "rgb_wrist_left_ego", "wrist_right": "rgb_wrist_right_ego"}


def load_legacy_episode(ep_dir, hz=15.0, native_hz=200.0, cams=LEGACY_CAMS, fk=True) -> dict:
    ep_dir = Path(ep_dir)
    meta = json.loads((ep_dir / "metadata.json").read_text()) if (ep_dir / "metadata.json").exists() else {}
    with np.load(ep_dir / "episode.npz") as f:
        keys = set(f.files)
        # older collectors (collect_cup.py: cup_dr_*) only log the per-step flag
        sc = np.asarray(f["spilled_count"] if "spilled_count" in keys else f["dropped"]).astype(np.int32)
        if "rgb_zed" not in keys and "rgb_head" in keys:
            cams = {**cams, "zed": "rgb_head"}
        Tn = len(sc); idx = resample_index(Tn, native_hz, hz)
        lq, rq = f["left_q"], f["right_q"]
        proprio = np.concatenate([lq, f["left_qd"], rq, f["right_qd"],
                                  np.asarray(f["left_gripper_state"])[:, None],
                                  np.asarray(f["right_gripper_state"])[:, None]], 1).astype(np.float32)
        action = np.concatenate([f["left_action"], f["right_action"]], 1).astype(np.float32)
        coll = np.asarray(f["collision"]).astype(bool)
        images = {c: np.ascontiguousarray(f[k][idx]) for c, k in cams.items() if k in keys}
        cup_label = np.asarray(f["box_pose"]) if "box_pose" in keys else None
        lq, rq = np.asarray(lq), np.asarray(rq)
    ep = dict(id=f"{ep_dir.parent.name}/{ep_dir.name}", hz=float(hz), T=len(idx), t_start=0,
              spill_count=window_max(sc, idx), collision=window_max(coll, idx),
              proprio=proprio[idx], action=action[idx], images=images, meta=meta, native_index=idx)
    ep["onset"] = M.onset_from_counts(ep["spill_count"], 0)
    if fk and cup_label is not None:
        pos, quat, hand, d = cup_kin_from_fk(lq, rq, cup_label)
        ep.update(cup_pos=pos[idx], cup_quat=quat[idx],
                  kin_hi=dict(hz=float(native_hz), pos=pos, quat=quat, idx=idx))
        ep["meta"] = {**meta, "cup_kin_source": f"fk:{hand}_arm_ee_grasp+fixed_offset", "cup_hand_dist_med": d}
    return ep


def _split_multi(d: dict):
    dense = None
    if "ep_dense_start" in d and "spilled_200hz" in d:       # gen_spill_data.py dense curve
        dense = [d["spilled_200hz"][s:s + l] for s, l in zip(d["ep_dense_start"], d["ep_dense_len"])]
        d = {k: v for k, v in d.items() if k != "spilled_200hz"}
    if "ep_len" in d:
        lens = np.asarray(d["ep_len"]); starts = np.r_[0, np.cumsum(lens)[:-1]]
    elif "episode_index" in d:
        ei = np.asarray(d["episode_index"]); u, starts = np.unique(ei, return_index=True)
        o = np.argsort(starts); starts = starts[o]; lens = np.diff(np.r_[starts, len(ei)])
    else:
        return [d]
    Ttot = int(lens.sum()); out = []
    for i, (s, l) in enumerate(zip(starts, lens)):
        e = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == Ttot:
                e[k] = v[s:s + l]
            elif isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(lens):
                e[k] = v[i]
            else:
                e[k] = v
        if dense is not None:
            e["spilled_200hz"] = dense[i]
        out.append(e)
    return out


def _manifest_meta(path: Path):
    """gen_spill_data.py writes <dir>/manifest.json with per-episode dicts carrying
    their shard id; return them in shard order (matched to ep order in the shard)."""
    for mf in (path.parent / "manifest.json", path.parent.parent / "manifest.json"):
        if mf.exists():
            try:
                man = json.loads(mf.read_text())
            except Exception:
                return []
            eps = man.get("episodes", man.get("eps", [])) if isinstance(man, dict) else man
            sid = int(path.stem.split("_")[-1]) if path.stem.split("_")[-1].isdigit() else None
            top = {k: v for k, v in man.items() if not isinstance(v, (list, dict))} if isinstance(man, dict) else {}
            return [e for e in eps if isinstance(e, dict) and e.get("shard") == sid], top
    return [], {}


def load_v2_file(path, default_hz=15.0) -> list[dict]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as f:
        raw = {k: f[k] for k in f.files}
    man, top = _manifest_meta(path)
    if "wm_dt" in top:
        default_hz = 1.0 / float(top["wm_dt"])
    eps = []
    for j, d in enumerate(_split_multi(raw)):
        meta = dict(man[j]) if j < len(man) else {}
        if meta and "contents" not in meta and "n_balls" in meta:
            meta["contents"] = {"n_balls": meta["n_balls"]}
        for mk in ("meta_json", "meta", "metadata_json"):
            if mk in d:
                try:
                    meta = json.loads(str(d[mk]))
                except Exception:
                    pass
        side = path.with_suffix(".json")
        if not meta and side.exists() and side.name != "manifest.json":
            meta = json.loads(side.read_text())
        sc = np.asarray(_first(d, ALIASES["spill_count"])).astype(np.int32).reshape(-1)
        T = len(sc)
        images = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray) and v.dtype == np.uint8 and v.ndim == 4 and v.shape[0] == T and v.shape[-1] == 3:
                name = k
                for p in IMAGE_PREFIXES:
                    if name.startswith(p):
                        name = name[len(p):]
                images[name] = v
        t0 = _first(d, ALIASES["t_start"], 0); t0 = int(np.asarray(t0).reshape(-1)[0])
        onset = _first(d, ALIASES["onset"])
        onset = int(np.asarray(onset).reshape(-1)[0]) if onset is not None else M.onset_from_counts(sc, t0)
        hz = _first(d, ALIASES["hz"], default_hz); hz = float(np.asarray(hz).reshape(-1)[0])
        ep = dict(id=f"{path.stem}#{j}", hz=hz, T=T, t_start=t0, spill_count=sc, onset=onset,
                  collision=np.asarray(_first(d, ALIASES["collision"], np.zeros(T, bool))).astype(bool),
                  proprio=np.asarray(_first(d, ALIASES["proprio"])).astype(np.float32),
                  action=np.asarray(_first(d, ALIASES["action"])).astype(np.float32),
                  images=images, meta=meta)
        cp, cq = _first(d, ALIASES["cup_pos"]), _first(d, ALIASES["cup_quat"])
        if cp is None and "cup_pose" in d:
            cp, cq = d["cup_pose"][:, :3], d["cup_pose"][:, 3:7]
        if cp is not None:
            ep["cup_pos"], ep["cup_quat"] = np.asarray(cp, np.float32), np.asarray(cq, np.float32)
        for extra in ("time_to_spill", "spill_within", "action_cup", "cup_vel", "cup_acc", "spill_frac",
                      "right_cup_contact", "right_ball_contact", "t", "spilled_200hz"):
            if extra in d:
                ep[extra] = d[extra]
        eps.append(ep)
    return eps


# --------------------------------------------------------------------------
# prepared (canonical) dir: write / read
# --------------------------------------------------------------------------
def save_canonical(ep: dict, path) -> None:
    flat = {k: v for k, v in ep.items() if isinstance(v, np.ndarray)}
    for c, im in ep.get("images", {}).items():
        flat[f"img__{c}"] = im
    if "kin_hi" in ep:
        for k, v in ep["kin_hi"].items():
            flat[f"kinhi__{k}"] = np.asarray(v)
    scal = {k: ep[k] for k in ("id", "hz", "T", "t_start", "onset")}
    flat["__scalars__"] = np.array(json.dumps(scal))
    flat["__meta__"] = np.array(json.dumps(ep.get("meta", {}), default=str))
    np.savez(path, **flat)


def load_canonical(path, with_images=True, cams=None) -> dict:
    with np.load(path, allow_pickle=False) as f:
        ep = json.loads(str(f["__scalars__"])); ep["meta"] = json.loads(str(f["__meta__"]))
        ep["images"] = {}; kin = {}
        for k in f.files:
            if k.startswith("__"):
                continue
            if k.startswith("img__"):
                c = k[5:]
                if with_images and (cams is None or c in cams):
                    ep["images"][c] = f[k]
            elif k.startswith("kinhi__"):
                kin[k[7:]] = f[k]
            else:
                ep[k] = f[k]
        if kin:
            kin["hz"] = float(kin["hz"]); ep["kin_hi"] = kin
    return ep


def make_splits(eps_meta: list[dict], seed=0, val_frac=0.15, test_frac=0.15, ood_rule: str | None = None):
    """Trajectory-level split stratified by (family, spilled). Optional OOD rule
    'meta.dotted.key>value' / '<' / '==' carves an OOD set out BEFORE splitting."""
    rng = np.random.default_rng(seed)
    ood = set()
    if ood_rule:
        import operator, re
        mt = re.match(r"(.+?)(>=|<=|==|>|<)(.+)", ood_rule)
        key, op, val = mt.group(1), mt.group(2), mt.group(3)
        opf = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le, "==": operator.eq}[op]
        from world_model.v2.eval.harness import meta_get
        for e in eps_meta:
            v = meta_get(e, key)
            try:
                hit = opf(float(v), float(val))
            except (TypeError, ValueError):
                hit = str(v) == val if op == "==" else False
            if hit:
                ood.add(e["id"])
    strata = {}
    for e in eps_meta:
        if e["id"] in ood:
            continue
        key = (str(e.get("meta", {}).get("perturbation_family", "")) + "/" +
               str(e.get("meta", {}).get("perturbation_level", "")), e["onset"] >= 0)
        strata.setdefault(key, []).append(e["id"])
    sp = {"train": [], "val": [], "test": [], "ood": sorted(ood)}
    for key in sorted(strata):
        g = strata[key]; rng.shuffle(g)
        nv, nt = int(round(val_frac * len(g))), int(round(test_frac * len(g)))
        sp["val"] += g[:nv]; sp["test"] += g[nv:nv + nt]; sp["train"] += g[nv + nt:]
    return sp


def _prep_one(args):
    kind, src, out, i, hz, native_hz = args
    os.environ.setdefault("MUJOCO_GL", "disable")
    if kind == "legacy":
        eps = [load_legacy_episode(src, hz=hz, native_hz=native_hz)]
    else:
        eps = load_v2_file(src, default_hz=hz)
    res = []
    for j, ep in enumerate(eps):
        fn = Path(out) / f"ep_{i:05d}_{j:02d}.npz"
        save_canonical(ep, fn)
        res.append(dict(file=fn.name, id=ep["id"], T=ep["T"], hz=ep["hz"], onset=ep["onset"],
                        t_start=ep["t_start"], meta=ep["meta"], cams=sorted(ep["images"])))
    return res


def prep(source: str, out, hz=15.0, native_hz=200.0, workers=8, limit=None, seed=0, ood_rule=None):
    kind, src = source.split(":", 1)
    src = Path(src) if Path(src).is_absolute() else REPO / src
    if kind == "legacy":
        items = sorted(p for p in src.glob("episode_*") if (p / "episode.npz").exists())
    elif kind == "v2":
        items = sorted(src.rglob("*.npz"))
    else:
        raise ValueError(kind)
    if limit:
        items = items[:limit]
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    jobs = [(kind, str(p), str(out), i, hz, native_hz) for i, p in enumerate(items)]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers, maxtasksperchild=8) as pool:
            recs = [r for rs in pool.imap(_prep_one, jobs, chunksize=1) for r in rs]
    else:
        recs = [r for j in jobs for r in _prep_one(j)]
    splits = make_splits(recs, seed=seed, ood_rule=ood_rule)
    index = dict(source=source, hz=hz, native_hz=native_hz, episodes=recs, splits=splits)
    (out / "index.json").write_text(json.dumps(index, indent=1, default=str))
    return index


def load_prepared(prep_dir, split: str | None = None, with_images=True, cams=None, limit=None) -> list[dict]:
    prep_dir = Path(prep_dir)
    index = json.loads((prep_dir / "index.json").read_text())
    recs = index["episodes"]
    if split:
        keep = set(index["splits"][split]); recs = [r for r in recs if r["id"] in keep]
    if limit:
        recs = recs[:limit]
    return [load_canonical(prep_dir / r["file"], with_images, cams) for r in recs]


def main(argv=None):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--source", required=True, help="legacy:<collection dir> | v2:<shard dir>")
    p.add_argument("--out", required=True)
    p.add_argument("--hz", type=float, default=15.0)
    p.add_argument("--native-hz", type=float, default=200.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ood-rule", default=None, help="e.g. 'contents.n_balls>10' or 'lighting.level==dark'")
    a = ap.parse_args(argv)
    idx = prep(a.source, a.out, a.hz, a.native_hz, a.workers, a.limit, a.seed, a.ood_rule)
    n = len(idx["episodes"]); ns = sum(r["onset"] >= 0 for r in idx["episodes"])
    print(f"[prep] {n} episodes ({ns} spill) -> {a.out}; splits " +
          ", ".join(f"{k}={len(v)}" for k, v in idx["splits"].items()), flush=True)


if __name__ == "__main__":
    main()
