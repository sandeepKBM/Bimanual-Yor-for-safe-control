"""gen_nominal.py -- sharded, resumable generator of NOMINAL tray carries with a free ball (MuJoCo truth).
SIMULATION ONLY (MuJoCo CPU; no GPU, no rendering, no hardware, no network).

Nominal carry law (b_tray_ball/carry.py, imported unchanged; parameters are ASSUMED, a stand-in for an operator)
  3 min-jerk transports of the tray centre: direction ~ U(0, 2pi), distance D ~ U(0.15, 0.45) m * intensity,
  duration T ~ U(1, 2) s / sqrt(intensity), rests 0.1-0.3 s, 0.5 s tail. Episode length ~ 4-5 s (about 200-250 ticks).
  A noisy anti-roll tilt loop aligns the tray normal with the effective-gravity normal
  c* = normalize([kappa a_x, kappa a_y, g]): kappa ~ U(0.6, 1.1) (imperfect compensation), loop lag tau ~ U(0.12, 0.25) s,
  OU tracking noise ~ U(0, 0.25) rad/s. a is clipped to +-5 m/s^2, w_xy to +-2 rad/s. intensity = 1.7 (default).
  Ball starts near the tray centre: tray-frame (x, y) ~ N(0, 5 mm).
  Hidden per episode: mu_roll ~ log-uniform [5e-4, 6e-3] m (recorded as `hid`, not in obs).
  The episode continues after the ball leaves the tray (the ball falls; l stays at its floor of -0.5).

Phase 1b options (all default OFF, defaults reproduce phase-1a data):
  --config piper     a_scale 0.3, w_scale 0.5, slew alpha 10 rad/s^2, carry-mode clip, intensity 3.0 (configs/piper.json)
  --carry-mode clip  same carry paths; accel VECTOR clipped at 5*a_scale, tilt rate at 2*w_scale (authority/scaled_probe)
  --tracker realistic | --obs-noise-m --obs-latency-ticks --obs-rate-hz --obs-dropout-p   ball-tracker model, `full` mode only
  The tray part of obs (first 9) is always exact. Parameters are stored in meta.json.

Unit of work = one SHARD of `--eps-per-shard` episodes (default 25, a few seconds each). Shard g gets the split
shard_split(g) = test if g%5==4, val if g%5==3, else train (so 60/20/20 by episode, like the pendulum wmdata).
Episode seed = seed0 + g * eps_per_shard + j. Each finished shard writes
    <out>/<split>/shard_<g>.npz   obs act l n nd priv (per tick); lens onset seed hid tunnel (per episode)
    <out>/<split>/snap_<g>.npz    MuJoCo snapshots at grid ticks (every snap_stride from snap_min_tick), restartable
    <out>/<split>/shard_<g>._DONE sentinel (a shard with a sentinel is skipped on re-run)
`--merge` builds <out>/<split>.npz, readable by wm_audit.common.load_split(split, data=<out>).

  python -m world_model.v2.tray_rep.gen_nominal --obs-mode full --n-eps 3000 --workers 16
  python -m world_model.v2.tray_rep.gen_nominal --obs-mode full --n-eps 3000 --shard 0/10       (Slurm-array style)
  python -m world_model.v2.tray_rep.gen_nominal --obs-mode full --n-eps 3000 --merge
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep import env as E
from world_model.v2.bead_rep import util as U
from world_model.v2.testbeds.b_tray_ball.carry import make_carry, carry_u, mj_acc, G

DT_WM = E.DT_WM
SEED0 = 100_000
INTENSITY = 1.7
EPS_PER_SHARD = 25
SNAP_STRIDE, SNAP_MIN_TICK = 2, 20
CARRY_MODES = ("orig", "clip")
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
BASE_KEYS = ("obs", "act", "l", "n", "nd", "priv")
EPS_KEYS = ("lens", "onset", "seed", "hid", "tunnel")


def carry_u_clip(cr, t, c, a_scale, w_scale):
    """`clip` carry mode (the authority agent's scaled_probe.carry_u_scaled, same maths, same RNG use).
    Same paths and durations as the original carry. The accel is clipped at 5*a_scale and the tilt rate at 2*w_scale:
    an operator limited by the arm's authority. a_scale == 1: the original per-axis clip (reproduces carry_u).
    a_scale < 1: the accel VECTOR is clipped (the torque limit is per direction). The env then applies the slew limit."""
    a = np.zeros(3)
    for (t0, T, Dv) in cr["segs"]:
        if t0 <= t < t0 + T:
            a = Dv / T ** 2 * mj_acc((t - t0) / T)
    if a_scale == 1.0:
        a = np.clip(a, -5.0, 5.0)
    else:
        a = a / max(1.0, float(np.hypot(a[0], a[1])) / (5.0 * a_scale))
    ct = np.array([cr["kappa"] * a[0], cr["kappa"] * a[1], G]); ct /= np.linalg.norm(ct)
    w = np.cross(c, ct) / cr["tau"]
    cr["ou"] = 0.9 * cr["ou"] + cr["noise"] * cr["rng"].normal(size=2) * np.sqrt(1 - 0.81)
    w[:2] = np.clip(w[:2] + cr["ou"], -2.0 * w_scale, 2.0 * w_scale)
    w[2] = 0.0
    return np.concatenate([a, w])


def run_episode(renv: E.TrayRepEnv, seed, intensity=INTENSITY, snap_stride=SNAP_STRIDE, snap_min_tick=SNAP_MIN_TICK,
                with_snaps=True, carry_mode="orig"):
    """one nominal carry. Row t = state at tick t (before the action), act[t] = action applied t -> t+1.
    carry_mode 'orig': b_tray_ball carry_u (clip 5 m/s^2, 2 rad/s). 'clip': carry_u_clip at the env's a_scale / w_scale."""
    rng = np.random.default_rng(seed)
    mu = E.sample_mu_roll(rng)
    ball_xy = rng.normal(0, 0.005, 2)
    cr = make_carry(rng, intensity)
    T = int(cr["T_end"] / DT_WM)
    rec = renv.reset(ball_xy=ball_xy, mu_roll=mu, track_seed=int(seed) + 777_000_000)   # own noise stream
    obs = np.zeros((T, renv.obs_dim), np.float32); act = np.zeros((T, 6), np.float32)
    l = np.zeros(T, np.float32); priv = np.zeros((T, E.PRIV_DIM), np.float32)
    snaps = []
    tunnel = False
    for t in range(T):
        obs[t], l[t], priv[t] = rec["obs"], rec["l"], rec["priv"]
        if with_snaps and t >= snap_min_tick and (t - snap_min_tick) % snap_stride == 0:
            snaps.append((t, renv.snapshot()))
        if carry_mode == "clip":
            u = carry_u_clip(cr, t * DT_WM, renv.env.axis(), renv.a_scale, renv.w_scale)
        else:
            u = carry_u(cr, t * DT_WM, renv.env.axis())
        rec = renv.tick(u)
        act[t] = rec["act"]
        tunnel |= bool(rec["tunnel"])
    bad = np.flatnonzero(l < 0)
    return dict(obs=obs, act=act, l=l, priv=priv, n=priv[:, 0:3].copy(), nd=priv[:, 3:6].copy(),
                onset=int(bad[0]) if len(bad) else -1, hid=mu, tunnel=int(tunnel), seed=int(seed), snaps=snaps)


def load_config(name_or_path):
    """JSON config by path, or by name from tray_rep/configs/<name>.json."""
    p = Path(name_or_path)
    if not p.exists():
        p = CONFIG_DIR / f"{name_or_path}.json"
    return json.loads(p.read_text())


def process_shard(args):
    (g, out, obs_mode, a_scale, lip, intensity, eps_per_shard, seed0, snap_stride, snap_min_tick,
     w_scale, alpha, carry_mode, trk) = args
    split = K.shard_split(g)
    d = Path(out) / split
    sent = d / f"shard_{g:04d}._DONE"
    if sent.exists():
        return dict(g=g, split=split, skipped=True)
    t0 = time.perf_counter()
    renv = E.TrayRepEnv(obs_mode, a_scale, lip, w_scale=w_scale, alpha=alpha, **trk)
    eps = [run_episode(renv, seed0 + g * eps_per_shard + j, intensity, snap_stride, snap_min_tick, carry_mode=carry_mode)
           for j in range(eps_per_shard)]
    cat = lambda k: np.concatenate([e[k] for e in eps], 0)
    S = dict(state=[], prev=[], has_prev=[], kin=[], status=[], k=[], w_slew=[], trk=[], ep=[], tick=[])
    for j, e in enumerate(eps):
        for t, sn in e["snaps"]:
            a = sn.arrays()
            for key in ("state", "prev", "has_prev", "kin", "status", "k", "w_slew", "trk"):
                S[key].append(a[key])
            S["ep"].append(j); S["tick"].append(t)
    U.safe_mkdir(d)
    U.atomic_savez(d / f"shard_{g:04d}.npz", **{k: cat(k) for k in BASE_KEYS},
                   lens=np.array([len(e["obs"]) for e in eps], np.int64),
                   onset=np.array([e["onset"] for e in eps], np.int64), seed=np.array([e["seed"] for e in eps], np.int64),
                   hid=np.array([e["hid"] for e in eps], np.float64), tunnel=np.array([e["tunnel"] for e in eps], np.int8))
    U.atomic_savez(d / f"snap_{g:04d}.npz", snap_state=np.array(S["state"]), snap_prev=np.array(S["prev"]),
                   snap_has_prev=np.array(S["has_prev"], np.int8), snap_kin=np.array(S["kin"]),
                   snap_status=np.array(S["status"], np.int8), snap_k=np.array(S["k"], np.int32),
                   snap_wslew=np.array(S["w_slew"], np.float64).reshape(-1, 3),
                   snap_trk=np.array(S["trk"], np.float64),
                   snap_ep=np.array(S["ep"], np.int32), snap_tick=np.array(S["tick"], np.int32))
    U.write_sentinel(sent)
    return dict(g=g, split=split, skipped=False, wall_s=time.perf_counter() - t0, n_eps=len(eps),
                ticks=int(sum(len(e["obs"]) for e in eps)), fail_eps=int(sum(e["onset"] >= 0 for e in eps)),
                tunnel_eps=int(sum(e["tunnel"] for e in eps)))


def merge_split(root, split, allow_partial=False, expect=None):
    root = Path(root)
    files = sorted(f for f in (root / split).glob("shard_*.npz") if ".tmp" not in f.name)
    done = [f for f in files if f.with_suffix("._DONE").exists()]
    if expect is not None and len(done) != expect and not allow_partial:
        raise SystemExit(f"[merge] {split}: {len(done)} / {expect} shards done (use --allow-partial)")
    acc = {k: [] for k in BASE_KEYS + EPS_KEYS}
    for f in done:
        with np.load(f, allow_pickle=False) as z:
            for k in acc:
                acc[k].append(z[k])
    out = {k: np.concatenate(v, 0) for k, v in acc.items()}
    out["start"] = np.concatenate([[0], np.cumsum(out["lens"])[:-1]]).astype(np.int64)
    U.atomic_savez(root / f"{split}.npz", **out)
    return dict(split=split, shards=len(done), eps=int(len(out["lens"])), ticks=int(out["lens"].sum()),
                fail_frac=float((out["onset"] >= 0).mean()), tunnel_eps=int(out["tunnel"].sum()),
                mean_len=float(out["lens"].mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs-mode", default="full", choices=list(E.OBS_DIMS))
    ap.add_argument("--out", default="", help="default <root>/data_<obs_mode>, or data_<config>_<obs_mode> with --config")
    ap.add_argument("--config", default="", help="name or path of a JSON in configs/ (e.g. piper). Sets a_scale, w_scale, "
                    "alpha, carry_mode, intensity; explicit flags win")
    ap.add_argument("--n-eps", type=int, default=3000)
    ap.add_argument("--eps-per-shard", type=int, default=EPS_PER_SHARD)
    ap.add_argument("--intensity", type=float, default=None, help=f"carry intensity (default {INTENSITY})")
    ap.add_argument("--a-scale", type=float, default=None, help="stored in meta; consumers (branches/oracle) use it (default 1)")
    ap.add_argument("--w-scale", type=float, default=None, help="tilt-rate scale (default: = a-scale)")
    ap.add_argument("--alpha", type=float, default=None, help="tilt-rate slew limit [rad/s^2] (default: off)")
    ap.add_argument("--carry-mode", default=None, choices=list(CARRY_MODES), help="orig (default) or clip")
    ap.add_argument("--tracker", default=None, help="name of a tracker entry in the config (tracker_<name>, e.g. realistic)")
    ap.add_argument("--obs-noise-m", type=float, default=None, help="ball x,y noise std [m] (full mode; default 0)")
    ap.add_argument("--obs-latency-ticks", type=int, default=None, help="ball obs delay in ticks (default 0)")
    ap.add_argument("--obs-rate-hz", type=float, default=None, help="ball sample-and-hold rate (default 50 = every tick)")
    ap.add_argument("--obs-dropout-p", type=float, default=None, help="per-tick sample loss probability (default 0)")
    ap.add_argument("--lip", action="store_true", help="dustpan (back lip) instead of the flat tray")
    ap.add_argument("--seed0", type=int, default=SEED0)
    ap.add_argument("--snap-stride", type=int, default=SNAP_STRIDE)
    ap.add_argument("--snap-min-tick", type=int, default=SNAP_MIN_TICK)
    ap.add_argument("--shard", default="0/1", help="i/n: this task takes every n-th shard")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else {}
    pick = lambda v, key, dflt: v if v is not None else cfg.get(key, dflt)
    a.intensity, a.a_scale = pick(a.intensity, "intensity", INTENSITY), pick(a.a_scale, "a_scale", 1.0)
    a.w_scale, a.alpha = pick(a.w_scale, "w_scale", None), pick(a.alpha, "alpha", None)
    a.carry_mode = pick(a.carry_mode, "carry_mode", "orig")
    tcfg = cfg.get(f"tracker_{a.tracker}", {}) if a.tracker else {}
    trk = dict(obs_noise_m=a.obs_noise_m if a.obs_noise_m is not None else tcfg.get("obs_noise_m", 0.0),
               obs_latency_ticks=a.obs_latency_ticks if a.obs_latency_ticks is not None else tcfg.get("obs_latency_ticks", 0),
               obs_rate_hz=a.obs_rate_hz if a.obs_rate_hz is not None else tcfg.get("obs_rate_hz", 50.0),
               obs_dropout_p=a.obs_dropout_p if a.obs_dropout_p is not None else tcfg.get("obs_dropout_p", 0.0))
    cname = Path(a.config).stem if a.config else ""
    out = Path(a.out) if a.out else K.ROOT / (f"data_{cname}_{a.obs_mode}" if cname else f"data_{a.obs_mode}")
    n_shards = -(-a.n_eps // a.eps_per_shard)
    if a.merge:
        for sp in K.SPLITS:
            exp = sum(1 for g in range(n_shards) if K.shard_split(g) == sp)
            print("[merge]", merge_split(out, sp, a.allow_partial, exp), flush=True)
        return
    U.safe_mkdir(out)
    meta_path = out / "meta.json"
    if not meta_path.exists():
        U.jdump(dict(kind="tray_rep nominal", obs_mode=a.obs_mode, obs_dim=E.OBS_DIMS[a.obs_mode], priv_dim=E.PRIV_DIM,
                     feat_dim=E.FEAT_DIM, n_eps=a.n_eps, eps_per_shard=a.eps_per_shard, intensity=a.intensity,
                     a_scale=a.a_scale, w_scale=a.w_scale, alpha=a.alpha, carry_mode=a.carry_mode, config=cname,
                     **trk, lip=a.lip, seed0=a.seed0, snap_stride=a.snap_stride,
                     snap_min_tick=a.snap_min_tick, mu_roll_range=[E.MU_LO, E.MU_HI], dt_wm=DT_WM,
                     split_rule="shard g: g%5==4 test, g%5==3 val, else train",
                     priv_cols="loc(3) vrel(3) omega_tray(3) d lift",
                     n_nd="n = ball tray-frame position (3), nd = its tray-frame velocity (3)"), meta_path)
    i, n = U.parse_shard(a.shard)
    mine = [g for g in range(n_shards) if g % n == i]
    print(f"[gen] obs_mode={a.obs_mode} shard {a.shard}: {len(mine)}/{n_shards} shards, out={out}", flush=True)
    tasks = [(g, str(out), a.obs_mode, a.a_scale, a.lip, a.intensity, a.eps_per_shard, a.seed0, a.snap_stride,
              a.snap_min_tick, a.w_scale, a.alpha, a.carry_mode, trk) for g in mine]
    t0 = time.perf_counter()
    res = []
    with Pool(a.workers, maxtasksperchild=8) as P:
        for r in P.imap_unordered(process_shard, tasks):
            res.append(r)
            if not r["skipped"]:
                print(f"[gen] shard {r['g']:04d} {r['split']}: {r['n_eps']} eps {r['ticks']} ticks "
                      f"fail {r['fail_eps']}/{r['n_eps']} tunnel {r['tunnel_eps']} {r['wall_s']:.1f}s", flush=True)
    new = [r for r in res if not r["skipped"]]
    print(f"[gen] done: {len(new)} new, {len(res) - len(new)} skipped, wall {time.perf_counter() - t0:.0f}s, "
          f"cpu-s in shards {sum(r['wall_s'] for r in new):.0f}, "
          f"maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
