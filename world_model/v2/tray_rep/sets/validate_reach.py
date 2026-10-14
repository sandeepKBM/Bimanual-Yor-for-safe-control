"""sets/validate_reach.py -- do MuJoCo ball trajectories stay inside the predicted reachable sets? SIMULATION / SAVED DATA ONLY. CPU.

collect:  run F0 carries of a split (nominal law, no filter) with the ball estimator (estimate.BallKF) running on the obs stream.
          At every `stride`-th tick (from tick `t0`) take a BRANCH: copy the env (snapshot), apply the same plan the predictor assumes,
              tick 0: u0 = u_nom (the command the filter would examine)     ticks 1..H-1: the BACKUP law (reach.backup_cmd)
          in MuJoCo (the hidden mu_roll of the carry is in the shadow env) and record the true tray-frame ball path (H+1 states).
          The backup law reads the TRUE ball state if kd > 0 (ideal sensor; documented). Records are saved to an npz.
evaluate: for a ReachParams, predict the set of every branch from (estimate, P, tray state, u0) and score the true path:
          traj coverage = fraction of branches whose ball stays inside the set at EVERY tick it is still on the tray;
          step coverage = fraction of (branch, tick) pairs inside; mean half-width by tick (tightness).
The model-error bound eps_a is tuned on the VAL split only (see tune_eps).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep.store import Store
from world_model.v2.tray_rep.sets import reach as RC
from world_model.v2.tray_rep.sets.estimate import BallKF
from world_model.v2.tray_rep.bridge import episode as EP
from world_model.v2.testbeds.b_tray_ball.tray_env import HX, HY

OUT = K.ROOT / "sets"


def make_kf(meta, sigma_a=None):
    nf = meta.get("obs_noise_m", 0.0) == 0.0 and meta.get("obs_latency_ticks", 0) == 0 and meta.get("obs_dropout_p", 0.0) == 0.0 \
        and meta.get("obs_rate_hz", 50.0) >= 50.0
    return BallKF(sigma_meas=meta.get("obs_noise_m", 0.003) or 0.003, lat=meta.get("obs_latency_ticks", 3),
                  sigma_a=(0.05 if nf else 0.10) if sigma_a is None else sigma_a, noise_free=nf)


def branch(shadow, snap, hid, u0, par, H):
    """-> true (H+1, 4) [x, y, vx, vy] tray-frame ball path (backward-difference velocity) for plan (u0, backup)."""
    shadow.restore(snap, mu_roll=hid)
    rec = shadow.current()
    p = rec["priv"]
    s = np.array([p[0], p[1], p[3], p[4]], np.float64)
    tr = [s.copy()]
    for k in range(H):
        if k == 0:
            u = np.asarray(u0, np.float64)
        else:
            c = shadow.env.axis()
            a, w = RC.backup_cmd(c, shadow.env.v.copy(), s, par)
            u = np.concatenate([a, w])
        r = shadow.tick(u)
        pp = r["priv"]
        s = np.array([pp[0], pp[1], pp[3], pp[4]], np.float64)
        tr.append(s.copy())
        if r["failed"] or r["tunnel"]:
            tr += [s.copy()] * (H - k - 1)
            break
    return np.array(tr), (r["failed"] or r["tunnel"])


def collect(setup="piper_trk", split="val", n_eps=40, ep_start=0, stride=10, t0=10, H=60, rp=None, out=None, seed=0):
    rp = rp or RC.ReachParams()
    data = K.ROOT / f"data_{setup}_full"
    st = Store(data, split)
    meta = st.meta
    renv = E.env_from_meta(meta)
    shadow = E.TrayRepEnv("full", renv.a_scale, renv.lip, w_scale=renv.w_scale, alpha=renv.alpha)
    par = RC.ReachParams(**{**rp.__dict__, "a_max": renv.a_max, "w_max": renv.w_max, "alpha": renv.alpha or 10.0}).par()
    recs = []
    for ep in range(ep_start, min(st.n_eps, ep_start + n_eps)):
        info = st.ep_info(ep)
        nm = EP.Nominal(meta, renv, info["seed"], "closedloop")
        rec = renv.reset(ball_xy=nm.ball_xy, mu_roll=nm.mu, track_seed=int(info["seed"]) + EP.TRK_SEED_BASE + EP.TRK_SEED_STRIDE * seed)
        obs = np.zeros((nm.T + 1, renv.obs_dim), np.float32)
        obs[0] = rec["obs"]
        kf = make_kf(meta)
        x, P = kf.step(obs[:1])
        for t in range(nm.T):
            u_nom = nm.u(t)
            if t >= t0 and (t - t0) % stride == 0 and renv.status == 0:
                tr, dropped = branch(shadow, renv.snapshot(), nm.mu, u_nom, par, H)
                o = obs[t]
                recs.append(dict(ep=ep, t=t, x=x.copy(), P=P.copy(), c0=o[0:3].astype(np.float64), vt0=o[3:6].astype(np.float64),
                                 w0=o[6:9].astype(np.float64), u0=np.asarray(u_nom, np.float64), true=tr, mu=nm.mu, drop=bool(dropped)))
            r = renv.tick(np.asarray(u_nom, np.float64))
            obs[t + 1] = r["obs"]
            x, P = kf.step(obs[:t + 2])
            if r["failed"] or r["tunnel"]:
                break
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **{k: np.array([r[k] for r in recs]) for k in recs[0]}, meta=json.dumps(dict(setup=setup, split=split, H=H)))
    return recs


def load_records(path):
    z = np.load(path, allow_pickle=False)
    n = len(z["ep"])
    return [{k: z[k][i] for k in z.files if k != "meta"} for i in range(n)]


def evaluate(recs, rp: RC.ReachParams, a_max, w_max, alpha, on_tray_tol=0.0, detail=False):
    p = RC.ReachParams(**{**rp.__dict__, "a_max": a_max, "w_max": w_max, "alpha": alpha})
    model = RC.ReachModel(p)
    H = p.H
    traj_ok, step_in, step_n, hw_sum = [], 0, 0, np.zeros(H + 1)
    first_out = []
    by_k = np.zeros(H + 1)
    n_by_k = np.zeros(H + 1)
    worst = []
    for r in recs:
        pr = model.predict(r["x"], r["P"], r["c0"], r["vt0"], r["w0"], r["u0"], jac=False)
        tr = r["true"][:H + 1, :2]
        # only while the ball is on the tray (after it left, MuJoCo clips; the set is irrelevant)
        on = (np.abs(tr[:, 0]) <= HX + on_tray_tol) & (np.abs(tr[:, 1]) <= HY + on_tray_tol)
        last = len(on) if on.all() else int(np.argmin(on)) + 1
        ins = np.all(np.abs(tr - pr.centre) <= pr.hw + 1e-12, axis=1)[:last]
        traj_ok.append(bool(ins.all()))
        step_in += int(ins.sum())
        step_n += len(ins)
        by_k[:last] += ins
        n_by_k[:last] += 1
        hw_sum += pr.hw.max(1)
        if not ins.all():
            k = int(np.argmin(ins))
            first_out.append(k)
            worst.append(float(np.max(np.abs(tr[:last] - pr.centre[:last]) - pr.hw[:last])))
    out = dict(n=len(recs), traj_cov=float(np.mean(traj_ok)), step_cov=step_in / max(step_n, 1),
               mean_hw_mm_at=[float(1e3 * hw_sum[k] / len(recs)) for k in (0, 10, 20, 30, 40, 50, H)],
               first_out_median=float(np.median(first_out)) if first_out else None,
               worst_excess_mm_max=float(1e3 * max(worst)) if worst else 0.0)
    if detail:
        out["step_cov_by_k"] = (by_k / np.maximum(n_by_k, 1)).tolist()
    return out


def tune_eps(recs, rp, a_max, w_max, alpha, target=0.95, grid=None):
    """smallest eps_a on the grid whose trajectory coverage >= target (VAL records only)."""
    grid = grid if grid is not None else [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5]
    rows = []
    for e in grid:
        res = evaluate(recs, RC.ReachParams(**{**rp.__dict__, "eps_a": e}), a_max, w_max, alpha)
        rows.append((e, res["traj_cov"], res["step_cov"], res["mean_hw_mm_at"]))
        if res["traj_cov"] >= target:
            return e, rows
    return None, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", default="piper_trk", choices=["piper", "piper_trk"])
    ap.add_argument("--split", default="val", choices=["val"])
    ap.add_argument("--n-eps", type=int, default=40)
    ap.add_argument("--ep-start", type=int, default=0)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--H", type=int, default=60)
    ap.add_argument("--kd", type=float, default=3.0)
    ap.add_argument("--kp", type=float, default=2.0)
    ap.add_argument("--out", default=str(OUT / "tmp"))
    ap.add_argument("--tune", action="store_true")
    a = ap.parse_args()
    rp = RC.ReachParams(H=a.H, kd=a.kd, kp=a.kp)
    t0 = time.perf_counter()
    path = Path(a.out) / f"branches_{a.setup}_{a.split}_kd{a.kd:g}_kp{a.kp:g}_H{a.H}_n{a.n_eps}_s{a.ep_start}.npz"
    recs = collect(a.setup, a.split, a.n_eps, a.ep_start, a.stride, 10, a.H, rp, path)
    print(f"collected {len(recs)} branches in {time.perf_counter() - t0:.1f}s -> {path}", flush=True)
    meta = json.loads((K.ROOT / f"data_{a.setup}_full" / "meta.json").read_text())
    a_max, w_max = 5.0 * meta["a_scale"], 2.0 * meta["w_scale"]
    if a.tune:
        e, rows = tune_eps(recs, rp, a_max, w_max, meta["alpha"])
        for r in rows:
            print("eps_a %.2f traj_cov %.3f step_cov %.4f hw_mm %s" % (r[0], r[1], r[2], [round(x, 1) for x in r[3]]), flush=True)
        print("chosen eps_a:", e, flush=True)
    else:
        print(json.dumps(evaluate(recs, rp, a_max, w_max, meta["alpha"], detail=False), indent=1), flush=True)


if __name__ == "__main__":
    main()
