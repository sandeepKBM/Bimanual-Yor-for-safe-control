"""probe.py -- feasibility probe for the tray-ball testbed (SIMULATION, CPU only, no rendering).
  python -m world_model.v2.testbeds.b_tray_ball.probe nominal   [--n 200] [--workers 8]
  python -m world_model.v2.testbeds.b_tray_ball.probe savable   [--eps 48]  [--workers 8]
  python -m world_model.v2.testbeds.b_tray_ball.probe selftest
Writes /common/users/ss5772/bimanual_data/testbeds/b_tray_ball/feasibility.json (merge of stage results).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import numba as nb

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from world_model.v2.testbeds.b_tray_ball.tray_env import (TrayEnv, BallParams, B, DT, DT_WM, ST_OFF, ST_TUNNEL, STRIDE,
                                                          _kin_step_nb, sdf_rect, kin_step_fast)
from world_model.v2.testbeds.b_tray_ball import analytic as AN
from world_model.v2.testbeds.b_tray_ball.carry import make_carry, carry_u

OUT = Path("/common/users/ss5772/bimanual_data/testbeds/b_tray_ball")
JSON = OUT / "feasibility.json"
K_H = 100                    # 2 s horizon in WM ticks
LEADS = [2, 5, 10, 15, 20, 30, 40, 60, 80]
D_HOLD = 10                  # library: hold the fallback action for 10 ticks (0.2 s), then "brake" (level + stop)
U85 = B.fallback_U()


def merge_json(key, val):
    OUT.mkdir(parents=True, exist_ok=True)
    cur = json.loads(JSON.read_text()) if JSON.exists() else {}
    cur[key] = val
    JSON.write_text(json.dumps(cur, indent=1, default=float))


# ---------------------------------------------------------------------------------------------- tray-only sequences
@nb.njit(cache=True)
def _brake(v, c, a_max, w_max, dt_wm):
    sp = np.sqrt(v[0] * v[0] + v[1] * v[1])
    u = np.zeros(6)
    if sp > 1e-9:
        m = min(a_max, sp / dt_wm)
        u[0] = -v[0] / sp * m
        u[1] = -v[1] / sp * m
    wh = np.cross(c, np.array([0.0, 0.0, 1.0]))
    nh = np.sqrt(wh @ wh)
    if nh > 1e-9:
        tilt = np.arcsin(min(1.0, nh))
        m = min(w_max, tilt / dt_wm)
        u[3:] = wh / nh * m
    return u


@nb.njit(cache=True)
def _axis(q):
    return np.array([2.0 * (q[1] * q[3] + q[0] * q[2]), 2.0 * (q[2] * q[3] - q[0] * q[1]), 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])])


@nb.njit(cache=True)
def lib_sequences(pos, v, quat, U, d_hold, K, a_max=5.0, w_max=2.0):
    """(n+2, K, 6): rows 0..n-1 = hold U[i] for d_hold ticks then brake (level + stop); row n = hold-still (u = 0);
    row n+1 = brake only. Ball-independent (the tray is kinematic)."""
    n = U.shape[0]
    out = np.zeros((n + 2, K, 6))
    for i in range(n + 2):
        p, vv, q = pos.copy(), v.copy(), quat.copy()
        for t in range(K):
            if i < n and t < d_hold:
                u = U[i]
            elif i == n:
                u = np.zeros(6)
            else:
                u = _brake(vv, _axis(q), a_max, w_max, 0.02)
            out[i, t] = u
            for _ in range(4):
                p, vv, q = _kin_step_nb(p, vv, q, u[:3].copy(), u[3:].copy(), 1.5, 0.005)
    return out


def run_seq(env, snap, seq):
    """execute seq (K,6) from snapshot; returns (failed, fail_tick, tunnel, min margin d)."""
    env.restore(snap)
    dmin = 1e9
    for t in range(len(seq)):
        r = env.tick(seq[t])
        dmin = min(dmin, r["d"])
        if env.status:
            break
    return env.status == ST_OFF, (t if env.status == ST_OFF else -1), env.status == ST_TUNNEL, dmin


def ball_world(env):
    pb = env.d.qpos[env.bq:env.bq + 3].copy()
    vb = env.d.qvel[env.bv:env.bv + 3].copy()
    Rb = env._R(env.d.qpos[env.bq + 3:env.bq + 7])
    wb = Rb @ env.d.qvel[env.bv + 3:env.bv + 6]
    return pb, vb, wb


# ---------------------------------------------------------------------------------------------- nominal carries
def run_carry(args):
    seed, inten, lip, snap_leads = args
    rng = np.random.default_rng(seed)
    env = TrayEnv(lip=bool(lip))
    env.reset(env.home, np.array([1.0, 0, 0, 0]), ball_xy=rng.normal(0, 0.005, 2))
    cr = make_carry(rng, inten)
    T = int(cr["T_end"] / DT_WM)
    snaps, info = {}, dict(seed=seed, fail_tick=-1, T=T, tunnel=False, max_tilt=0.0)
    wl = np.zeros(3)
    for k in range(T):
        if snap_leads is not None and snap_leads.get("onset", -1) >= 0:
            lead = snap_leads["onset"] - k
            if lead in snap_leads["leads"]:
                snaps[lead] = (env.snapshot(), wl.copy())
        u = carry_u(cr, k * DT_WM, env.axis())
        r = env.tick(u)
        wl = r["obs"][6:9]
        info["max_tilt"] = max(info["max_tilt"], float(np.arccos(np.clip(r["obs"][2], -1, 1))))
        if env.status:
            break
    info["tunnel"] = env.status == ST_TUNNEL
    if env.status == ST_OFF:
        info["fail_tick"] = env.fail_tick // STRIDE
    return info, snaps


def nominal(args):
    res = {}
    for lip in (0, 1):
        for inten in ((1.0, 1.7, 2.0) if lip == 0 else (1.7,)):
            n = args.n if inten == 1.7 else args.n // 2
            tasks = [(10_000 + s, inten, lip, None) for s in range(n)]
            t0 = time.perf_counter()
            with Pool(args.workers) as P:
                out = P.map(run_carry, tasks, chunksize=4)
            wall = time.perf_counter() - t0
            infos = [o[0] for o in out]
            ft = np.array([i["fail_tick"] for i in infos])
            fail = ft >= 0
            res[f"lip{lip}_int{inten}"] = dict(n=n, fail_rate=float(fail.mean()), tunnel=int(sum(i["tunnel"] for i in infos)),
                                               median_fail_tick=float(np.median(ft[fail])) if fail.any() else None,
                                               median_len_ticks=float(np.median([i["T"] for i in infos])),
                                               max_tilt_p99_rad=float(np.percentile([i["max_tilt"] for i in infos], 99)),
                                               wall_s=wall)
            print(lip, inten, res[f"lip{lip}_int{inten}"], flush=True)
    merge_json("nominal_failure", res)


# ---------------------------------------------------------------------------------------------- savable oracle
def oracle_episode(args):
    """re-run a failing carry, snapshot at LEADS ticks before onset, and run the MuJoCo library oracle + NE/QS predictions."""
    seed, inten, lip, onset = args
    info, snaps = run_carry((seed, inten, lip, dict(onset=onset, leads=set(LEADS))))
    env = TrayEnv(lip=bool(lip))
    AN.set_lip(bool(lip))
    p = env.p
    rows = []
    n = len(U85)
    for lead, (sn, wl) in sorted(snaps.items()):
        t0 = time.perf_counter()
        env.restore(sn)
        pb, vb, wb = ball_world(env)
        seqs = lib_sequences(env.pos.copy(), env.v.copy(), env.quat.copy(), U85, D_HOLD, K_H)
        fail = np.zeros(n + 2, bool); ftk = -np.ones(n + 2, int); tun = np.zeros(n + 2, bool); dmin = np.zeros(n + 2)
        for i in range(n + 2):
            fail[i], ftk[i], tun[i], dmin[i] = run_seq(env, sn, seqs[i])
        t_mj = time.perf_counter() - t0
        # NE / QS predictions on the same sequences
        t1 = time.perf_counter()
        env.restore(sn)
        R = env._R(env.quat)
        c = env.axis()
        ne_fail = np.zeros(n + 2, bool); ne_ftk = -np.ones(n + 2, int)
        qs_fail = np.zeros(n + 2, bool)
        vrel = R.T @ (vb - env.v - np.cross(wl, pb - env.pos))
        loc0 = (pb - env.pos) @ R
        for i in range(n + 2):
            locs, ds, _ = AN.ne_rollout(pb, vb, wb, env.pos, env.v, env.quat, seqs[i], p, Om_prev=wl, lag=1)
            bad = np.nonzero(ds < 0)[0]
            ne_fail[i] = len(bad) > 0
            ne_ftk[i] = bad[0] if len(bad) else -1
            xs, ys = AN.qs_rollout(loc0[0], loc0[1], vrel[0], vrel[1], env.pos.copy(), env.v.copy(), env.quat.copy(), seqs[i],
                                   p.radius, p.inertia_k, p.mu_roll, 4, 1.5, DT)
            qd = np.array([sdf_rect(x, y, bool(lip)) for x, y in zip(xs, ys)])
            qs_fail[i] = (qd < 0).any()
        t_an = time.perf_counter() - t1
        rows.append(dict(seed=seed, lead=lead, fail=fail, ftk=ftk, tun=tun, dmin=dmin, ne_fail=ne_fail, ne_ftk=ne_ftk,
                         qs_fail=qs_fail, t_mj=t_mj, t_an=t_an, d0=float(sdf_rect(loc0[0], loc0[1], bool(lip))),
                         tilt=float(np.arccos(np.clip(c[2], -1, 1))), speed=float(np.hypot(*(vrel[:2])))))
    return rows


def summarize_rows(rows, n=len(U85)):
    out = dict(n_states=len(rows))
    def cls(r):
        h, br = r["fail"][n], r["fail"][n + 1]
        lib = (~r["fail"][:n]).any()
        if not h:
            return "hold_ok"
        if lib or not br:
            return "savable"
        return "not_certified"
    cl = np.array([cls(r) for r in rows])
    leads = np.array([r["lead"] for r in rows])
    for name in ("hold_ok", "savable", "not_certified"):
        out[f"frac_{name}"] = float((cl == name).mean())
    out["by_lead"] = {int(L): {nm: float(((cl == nm) & (leads == L)).sum() / max(1, (leads == L).sum()))
                               for nm in ("hold_ok", "savable", "not_certified")} for L in sorted(set(leads))}
    # per-episode: smallest lead at which a state is still certified savable or hold_ok-with-nominal-failure = "last savable lead"
    seeds = sorted(set(r["seed"] for r in rows))
    last_sav, n_eps_any = [], 0
    for s in seeds:
        ls = [r["lead"] for r, c in zip(rows, cl) if r["seed"] == s and c in ("savable", "hold_ok")]
        if ls:
            last_sav.append(min(ls)); n_eps_any += 1
    out["episodes"] = len(seeds)
    out["episodes_with_savable_state"] = n_eps_any
    out["last_savable_lead_ticks_median"] = float(np.median(last_sav)) if last_sav else None
    out["last_savable_lead_ticks_all"] = [int(x) for x in last_sav]
    # tunnels
    out["states_with_tunnel_in_any_branch"] = int(sum(r["tun"].any() for r in rows))
    # analytic vs MuJoCo label agreement over all (state, sequence) pairs
    mj = np.concatenate([r["fail"] for r in rows]); ne = np.concatenate([r["ne_fail"] for r in rows]); qs = np.concatenate([r["qs_fail"] for r in rows])
    tun = np.concatenate([r["tun"] for r in rows])
    keep = ~tun
    for nm, a in (("NE", ne), ("QS", qs)):
        m, x = mj[keep], a[keep]
        out[f"{nm}_label_agreement"] = float((m == x).mean())
        out[f"{nm}_false_safe"] = float(((~x) & m).sum() / max(1, m.sum()))      # MuJoCo fails, model says safe
        out[f"{nm}_false_fail"] = float((x & (~m)).sum() / max(1, (~m).sum()))
    # certified-savable label from the analytic model vs MuJoCo
    def lab(failarr):
        return "hold_ok" if not failarr[n] else ("savable" if ((~failarr[:n]).any() or not failarr[n + 1]) else "not_certified")
    for nm, key in (("NE", "ne_fail"), ("QS", "qs_fail")):
        a = np.array([lab(r[key]) for r in rows])
        out[f"{nm}_state_class_agreement"] = float((a == cl).mean())
    # timing
    out["mj_s_per_state"] = float(np.mean([r["t_mj"] for r in rows])); out["analytic_s_per_state"] = float(np.mean([r["t_an"] for r in rows]))
    # nominal-state geometry
    out["d0_median"] = float(np.median([r["d0"] for r in rows])); out["tilt_median_rad"] = float(np.median([r["tilt"] for r in rows]))
    return out


def savable(args):
    nom = json.loads(JSON.read_text())["nominal_failure"]
    inten, lip = 1.7, 0
    tasks = [(10_000 + s, inten, lip, -1) for s in range(args.scan)]
    with Pool(args.workers) as P:
        infos = [o[0] for o in P.map(run_carry, [(s, i, l, None) for (s, i, l, _) in tasks], chunksize=4)]
    fails = [(i["seed"], inten, lip, i["fail_tick"]) for i in infos if i["fail_tick"] >= 0 and i["fail_tick"] > 30][:args.eps]
    print("failing episodes used:", len(fails), "of", len(infos), flush=True)
    t0 = time.perf_counter()
    with Pool(args.workers) as P:
        res = P.map(oracle_episode, fails, chunksize=1)
    rows = [r for rr in res for r in rr]
    wall = time.perf_counter() - t0
    summ = summarize_rows(rows)
    summ["wall_s"] = wall; summ["cpu_s_est"] = wall * args.workers
    print(json.dumps({k: v for k, v in summ.items() if k != "last_savable_lead_ticks_all"}, indent=1, default=float), flush=True)
    merge_json("savable_flat_int1.7", summ)
    np.savez_compressed(OUT / "savable_rows.npz", lead=[r["lead"] for r in rows], seed=[r["seed"] for r in rows],
                        fail=np.array([r["fail"] for r in rows]), ne_fail=np.array([r["ne_fail"] for r in rows]),
                        qs_fail=np.array([r["qs_fail"] for r in rows]), d0=[r["d0"] for r in rows], tilt=[r["tilt"] for r in rows],
                        speed=[r["speed"] for r in rows])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--eps", type=int, default=48)
    ap.add_argument("--scan", type=int, default=160)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    {"nominal": nominal, "savable": savable}[a.cmd](a)
