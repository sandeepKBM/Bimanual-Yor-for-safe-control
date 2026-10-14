"""scaled_probe.py -- tray + ball savable-set measurement at a REDUCED actuator authority.  SIMULATION ONLY, CPU only.

Reuses b_tray_ball (imported, not edited): TrayEnv, make_carry, the 85-element set (probe.U85), `_brake`, QS/NE analytics.
What is new here:
  * the action set is scaled per axis:  U[:, 0] *= sx, U[:, 1] *= sy, U[:, 3:] *= sw  (brake limits scale the same way);
  * the commanded tilt rate w passes a slew limit |dw| <= ALPHA * dt per tick (the arm cannot step its angular rate);
  * the nominal carry is the SAME synthetic carry with its accel clip at 5*sa and its tilt-rate clip at 2*sw, in one of two modes
      'clip'    : same paths and durations; the accel clip truncates the profile (an authority-limited operator)
      'stretch' : durations x 1/sqrt(sa) so the peak accel scales with sa (a slower operator). Rest gaps unchanged.
  python -m world_model.v2.tray_rep.authority.scaled_probe nominal --sx 0.5 --sy 0.5 --sw 1 --mode stretch
  python -m world_model.v2.tray_rep.authority.scaled_probe savable --sx .. --sy .. --sw .. --mode .. --eps 24
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import numba as nb

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from world_model.v2.testbeds.b_tray_ball import probe as P                      # noqa: E402
from world_model.v2.testbeds.b_tray_ball import analytic as AN                  # noqa: E402
from world_model.v2.testbeds.b_tray_ball.carry import make_carry, G              # noqa: E402
from world_model.v2.testbeds.b_tray_ball.tray_env import TrayEnv, DT_WM, ST_OFF, ST_TUNNEL, STRIDE, _kin_step_nb  # noqa: E402

OUT = Path("/common/users/ss5772/bimanual_data/tray_rep/authority")
ALPHA_DEFAULT = 10.0            # rad/s^2 angular-rate slew (from recommend.py / grid.py; see report)
LEADS = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 45, 60]
D_HOLD, K_H = P.D_HOLD, P.K_H
U85 = P.U85


def scaled_U(sx, sy, sw):
    U = U85.copy()
    U[:, 0] *= sx; U[:, 1] *= sy; U[:, 3:] *= sw
    return U


def slew(w_cmd, w_prev, alpha, dt=DT_WM):
    return w_prev + np.clip(w_cmd - w_prev, -alpha * dt, alpha * dt)


def stretch(cr, f):
    """time-stretch a carry's segments by f (segment durations x f, rest gaps unchanged)."""
    segs, t, prev_end = [], 0.0, 0.0
    old = cr["segs"]
    for k, (t0, T, Dv) in enumerate(old):
        gap = t0 - (old[k - 1][0] + old[k - 1][1]) if k else t0
        t += gap
        segs.append((t, T * f, Dv)); t += T * f
    cr = dict(cr); cr["segs"] = segs; cr["T_end"] = t + 0.5
    return cr


def carry_u_scaled(cr, t, c, a_cap, w_cap):
    """as b_tray_ball.carry.carry_u, but accel clip a_cap (3,) and tilt-rate clip w_cap (scalar)."""
    from world_model.v2.testbeds.b_tray_ball.carry import mj_acc
    a = np.zeros(3)
    for (t0, T, Dv) in cr["segs"]:
        if t0 <= t < t0 + T:
            a = Dv / T ** 2 * mj_acc((t - t0) / T)
    if a_cap[0] == 5.0 and a_cap[1] == 5.0:
        a = np.clip(a, -a_cap, a_cap)                     # full authority: the original per-axis clip (reproduces b_tray_ball)
    else:
        nrm = np.hypot(a[0] / a_cap[0], a[1] / a_cap[1])  # reduced authority: clip the VECTOR (the torque limit is per direction)
        a = a / max(1.0, nrm)
    ct = np.array([cr["kappa"] * a[0], cr["kappa"] * a[1], G]); ct /= np.linalg.norm(ct)
    w = np.cross(c, ct) / cr["tau"]
    cr["ou"] = 0.9 * cr["ou"] + cr["noise"] * cr["rng"].normal(size=2) * np.sqrt(1 - 0.81)
    w[:2] = np.clip(w[:2] + cr["ou"], -w_cap, w_cap)
    w[2] = 0.0
    return np.concatenate([a, w])


def run_carry_s(args):
    """like probe.run_carry with scaled authority. args = (seed, inten, cfg, snap_spec). Also records last realized (a, w)."""
    seed, inten, cfg, snap_leads = args
    sx, sy, sw, mode, alpha = cfg["sx"], cfg["sy"], cfg["sw"], cfg["mode"], cfg["alpha"]
    rng = np.random.default_rng(seed)
    env = TrayEnv(lip=False)
    env.reset(env.home, np.array([1.0, 0, 0, 0]), ball_xy=rng.normal(0, 0.005, 2))
    cr = make_carry(rng, inten)
    if mode == "stretch":
        cr = stretch(cr, 1.0 / np.sqrt(min(sx, sy)))
    a_cap = np.array([5.0 * sx, 5.0 * sy, 0.0]); w_cap = 2.0 * sw
    T = int(cr["T_end"] / DT_WM)
    snaps, info = {}, dict(seed=seed, fail_tick=-1, T=T, tunnel=False, max_tilt=0.0, max_a=0.0, max_v=0.0)
    wl = np.zeros(3); w_real = np.zeros(3); a_real = np.zeros(3)
    for k in range(T):
        if snap_leads is not None and snap_leads.get("onset", -1) >= 0:
            lead = snap_leads["onset"] - k
            if lead in snap_leads["leads"]:
                snaps[lead] = (env.snapshot(), wl.copy(), w_real.copy())
        u = carry_u_scaled(cr, k * DT_WM, env.axis(), a_cap, w_cap)
        w_real = slew(u[3:], w_real, alpha)
        u = np.concatenate([u[:3], w_real])
        info["max_a"] = max(info["max_a"], float(np.hypot(u[0], u[1])))
        r = env.tick(u)
        info["max_v"] = max(info["max_v"], float(np.hypot(env.v[0], env.v[1])))
        wl = r["obs"][6:9]
        info["max_tilt"] = max(info["max_tilt"], float(np.arccos(np.clip(r["obs"][2], -1, 1))))
        if env.status:
            break
    info["tunnel"] = env.status == ST_TUNNEL
    if env.status == ST_OFF:
        info["fail_tick"] = env.fail_tick // STRIDE
    return info, snaps


# ---------------------------------------------------------------------------------------------- sequences with slew
def lib_sequences_slew(pos, v, quat, U, d_hold, K, a_max, w_max, w0, alpha):
    """(n+2, K, 6) REALIZED sequences. rows 0..n-1: hold U[i] for d_hold ticks then brake; n: hold-still; n+1: brake only.
    The tilt rate follows the command through the slew limit, starting from the realized rate w0 at the snapshot."""
    n = U.shape[0]
    out = np.zeros((n + 2, K, 6))
    for i in range(n + 2):
        p, vv, q = pos.copy(), v.copy(), quat.copy()
        wprev = np.asarray(w0, float).copy()
        for t in range(K):
            if i < n and t < d_hold:
                u = U[i].copy()
            elif i == n:
                u = np.zeros(6)
            else:
                u = P._brake(vv, P._axis(q), a_max, w_max, 0.02)
            wprev = slew(u[3:], wprev, alpha)
            u = np.concatenate([u[:3], wprev])
            out[i, t] = u
            for _ in range(4):
                p, vv, q = _kin_step_nb(p, vv, q, u[:3].copy(), u[3:].copy(), 1.5, 0.005)
    return out


def classify(fail, n):
    h, br = fail[n], fail[n + 1]
    lib = (~fail[:n]).any()
    if not h:
        return "hold_ok"
    return "savable" if (lib or not br) else "not_certified"


def oracle_episode_s(args):
    seed, inten, cfg, onset = args
    info, snaps = run_carry_s((seed, inten, cfg, dict(onset=onset, leads=set(LEADS))))
    env = TrayEnv(lip=False)
    AN.set_lip(False)
    U = scaled_U(cfg["sx"], cfg["sy"], cfg["sw"])
    a_max, w_max = 5.0 * min(cfg["sx"], cfg["sy"]), 2.0 * cfg["sw"]
    n = len(U)
    rows = []
    for lead, (sn, wl, wr) in sorted(snaps.items()):
        env.restore(sn)
        seqs = lib_sequences_slew(env.pos.copy(), env.v.copy(), env.quat.copy(), U, D_HOLD, K_H, a_max, w_max, wr, cfg["alpha"])
        fail = np.zeros(n + 2, bool); tun = np.zeros(n + 2, bool)
        for i in range(n + 2):
            fail[i], _, tun[i], _ = P.run_seq(env, sn, seqs[i])
        rows.append(dict(seed=seed, lead=lead, fail=fail, tun=tun, cls=classify(fail, n)))
    return rows


def nominal(args):
    cfg = dict(sx=args.sx, sy=args.sy, sw=args.sw, mode=args.mode, alpha=args.alpha)
    res = {}
    for inten in args.inten:
        tasks = [(10_000 + s, inten, cfg, None) for s in range(args.n)]
        t0 = time.perf_counter()
        with Pool(args.workers) as pool:
            out = pool.map(run_carry_s, tasks, chunksize=4)
        infos = [o[0] for o in out]
        ft = np.array([i["fail_tick"] for i in infos]); fail = ft >= 0
        res[f"int{inten}"] = dict(n=args.n, fail_rate=float(fail.mean()), tunnel=int(sum(i["tunnel"] for i in infos)),
                                  median_fail_tick=float(np.median(ft[fail])) if fail.any() else None,
                                  median_len_ticks=float(np.median([i["T"] for i in infos])),
                                  max_a_p99=float(np.percentile([i["max_a"] for i in infos], 99)),
                                  max_speed_p50=float(np.median([i["max_v"] for i in infos])), max_speed_p99=float(np.percentile([i["max_v"] for i in infos], 99)),
                                  max_tilt_p99=float(np.percentile([i["max_tilt"] for i in infos], 99)),
                                  cpu_s=(time.perf_counter() - t0) * args.workers)
        print(cfg, f"int{inten}", res[f"int{inten}"], flush=True)
    merge(f"nominal_{tag(cfg)}", dict(cfg=cfg, res=res))


def tag(cfg):
    return f"sx{cfg['sx']}_sy{cfg['sy']}_sw{cfg['sw']}_{cfg['mode']}"


def merge(key, val):
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "scaled_probe.json"
    cur = json.loads(f.read_text()) if f.exists() else {}
    cur[key] = val
    f.write_text(json.dumps(cur, indent=1, default=float))


def savable(args):
    cfg = dict(sx=args.sx, sy=args.sy, sw=args.sw, mode=args.mode, alpha=args.alpha)
    inten = args.inten[0]
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        infos = [o[0] for o in pool.map(run_carry_s, [(10_000 + s, inten, cfg, None) for s in range(args.scan)], chunksize=4)]
    fails = [(i["seed"], inten, cfg, i["fail_tick"]) for i in infos if i["fail_tick"] > 30][:args.eps]
    print(cfg, "failing eps used", len(fails), "of", len(infos), f"(fail rate {np.mean([i['fail_tick']>=0 for i in infos]):.3f})", flush=True)
    if not fails:
        merge(f"savable_{tag(cfg)}", dict(cfg=cfg, n_failing=0, scan=len(infos)))
        return
    with Pool(args.workers) as pool:
        res = pool.map(oracle_episode_s, fails, chunksize=1)
    rows = [r for rr in res for r in rr]
    wall = time.perf_counter() - t0
    cl = np.array([r["cls"] for r in rows]); leads = np.array([r["lead"] for r in rows]); seeds = np.array([r["seed"] for r in rows])
    out = dict(cfg=cfg, n_states=len(rows), n_eps=len(fails), scan=len(infos),
               nominal_fail_rate_scan=float(np.mean([i["fail_tick"] >= 0 for i in infos])),
               frac={k: float((cl == k).mean()) for k in ("hold_ok", "savable", "not_certified")},
               n_certified_savable=int((cl == "savable").sum()),
               by_lead={int(L): {k: float(((cl == k) & (leads == L)).sum() / max(1, (leads == L).sum()))
                                 for k in ("hold_ok", "savable", "not_certified")} for L in sorted(set(leads))})
    last_any, last_sav = [], []
    for s in sorted(set(seeds)):
        m = [(l, c) for l, c, sd in zip(leads, cl, seeds) if sd == s]
        g = [l for l, c in m if c in ("savable", "hold_ok")]; gs = [l for l, c in m if c == "savable"]
        last_any.append(min(g) if g else 999); last_sav.append(min(gs) if gs else 999)
    out["last_certified_lead_ticks_median"] = float(np.median(last_any))
    out["last_certified_lead_savable_only_median"] = float(np.median(last_sav))
    out["frac_eps_never_certified"] = float(np.mean(np.array(last_any) == 999))
    out["states_with_tunnel"] = int(sum(r["tun"].any() for r in rows))
    out["wall_s"] = wall; out["cpu_s_est"] = wall * args.workers
    print(json.dumps({k: v for k, v in out.items()}, default=float), flush=True)
    merge(f"savable_{tag(cfg)}", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--sx", type=float, default=1.0)
    ap.add_argument("--sy", type=float, default=1.0)
    ap.add_argument("--sw", type=float, default=1.0)
    ap.add_argument("--mode", default="clip")
    ap.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    ap.add_argument("--inten", type=float, nargs="+", default=[1.7])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--eps", type=int, default=24)
    ap.add_argument("--scan", type=int, default=160)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    {"nominal": nominal, "savable": savable}[a.cmd](a)
