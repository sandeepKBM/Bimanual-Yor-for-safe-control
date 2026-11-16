"""model_uq/collect.py -- residual dataset of the analytic ball model vs MuJoCo, FIXED physics. SIMULATION / SAVED DATA ONLY. CPU.

For VAL episodes of the tray env (initial conditions: seed -> mu, ball start, carry plan, via bridge.episode.Nominal) it
  (1) replays the nominal carry (no filter) and records it as a "program 0" trajectory, and
  (2) at every `stride`-th tick branches the MuJoCo env (snapshot/restore, with a FRESH hidden mu_roll, log-uniform) and runs
      L ticks under a command program, recording the true ball path in the tray frame and the tray axis / command per tick.
Programs (u = [tray accel (3), tray tilt rate (3)], the same 6-vector the env takes):
  0 nominal      the operator carry itself (closed loop, as run_carry with F0)
  1 tilt-hold    P controller to a tilt whose in-plane gravity is rho* x the (calibrated) rolling-resistance threshold, rho* log-U[0.2, 6],
                 re-drawn every 15-30 ticks, plus optional accel segments  -> dense coverage of the stuck / roll boundary (gn/f ~ 1)
  2 random       piecewise-constant random accel (incl. a_z) and tilt rate, segments of 3-10 ticks
  3 backup       a random first command then reach.backup_cmd (random gains)  -> the plans F5 reasons about
Output: npz of padded arrays (n_branch, L+1, ...). Nothing in tray_rep / testbeds is edited.
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep.store import Store
from world_model.v2.tray_rep.sets import reach as RC
from world_model.v2.tray_rep.bridge import episode as EP

DT = RC.DT
L_DEF = 40
G9 = 9.81


def _tilt_cmd(c, cdes_xy, w_max, kappa=6.0):
    w = np.zeros(3)
    w[1] = np.clip(kappa * (cdes_xy[0] - c[0]), -w_max, w_max)
    w[0] = np.clip(-kappa * (cdes_xy[1] - c[1]), -w_max, w_max)
    return w


class Program:
    """a stateful command generator: u = prog(shadow_env, state)"""

    def __init__(self, kind, rng, renv, mu):
        self.kind, self.rng, self.renv = kind, rng, renv
        self.a_max, self.w_max = renv.a_max, renv.w_max
        self.mu = mu
        self.f_level = (RC.MU_EFF(mu) / RC.R_BALL) * G9
        self.k = 0
        self.seg_left = 0
        self.seg_acc = 0
        self.u_hold = np.zeros(6)
        self.cdes = np.zeros(2)
        self.a_hold = np.zeros(3)
        self.acc_on = rng.random() < 0.5
        self.par = None
        if kind == 3:
            self.kd = float(rng.choice([0.0, 1.0, 3.0, 5.0]))
            self.kp = float(rng.choice([0.0, 2.0, 4.0]))
            p = RC.ReachParams(kd=self.kd, kp=self.kp, a_max=self.a_max, w_max=self.w_max, alpha=renv.alpha or 10.0)
            self.par = p.par()
            self.u0 = self._rand_cmd()
            self.nloc = None

    def _rand_cmd(self):
        rng = self.rng
        ang = rng.uniform(0, 2 * np.pi)
        mag = self.a_max * rng.uniform(0, 1) ** 0.7
        a = np.array([mag * np.cos(ang), mag * np.sin(ang), rng.uniform(-0.5, 0.5) * self.a_max * rng.random()])
        w = rng.uniform(-1, 1, 3) * self.w_max * np.array([1, 1, 0])
        return np.concatenate([a, w])

    def __call__(self, c, s_loc):
        rng = self.rng
        if self.kind == 2:
            if self.seg_left <= 0:
                self.u_hold = self._rand_cmd()
                self.seg_left = int(rng.integers(3, 11))
            self.seg_left -= 1
            return self.u_hold.copy()
        if self.kind == 1:
            if self.seg_left <= 0:
                rho = float(np.exp(rng.uniform(np.log(0.2), np.log(6.0))))
                th = float(np.arcsin(min(rho * self.f_level / G9, 0.19)))
                phi = rng.uniform(0, 2 * np.pi)
                self.cdes = np.array([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi)])
                self.seg_left = int(rng.integers(15, 31))
                if self.acc_on:
                    self.a_hold = self._rand_cmd()[:3] * rng.uniform(0, 0.6)
            self.seg_left -= 1
            w = _tilt_cmd(c, self.cdes, self.w_max)
            a = self.a_hold if self.acc_on else np.zeros(3)
            return np.concatenate([a, w])
        if self.kind == 3:
            if self.k == 0:
                self.k = 1
                return self.u0.copy()
            vt = self.renv.env.v.copy()
            a, w = RC.backup_cmd(c, vt, np.array([s_loc[0], s_loc[1], s_loc[2], s_loc[3]], np.float64), self.par)
            return np.concatenate([a, w])
        raise ValueError(self.kind)


def _rec_state(renv):
    e = renv.env
    return e.local().copy(), e.axis().copy()


def run_branch(shadow, snap, mu, kind, L, rng):
    """-> dict(pos (L+1, 3), c (L+1, 3), u (L, 6), vt (L+1, 3), n (valid ticks), dropped)"""
    shadow.restore(snap, mu_roll=mu)
    prog = Program(kind, rng, shadow, mu)
    pos = np.zeros((L + 1, 3)); cc = np.zeros((L + 1, 3)); vt = np.zeros((L + 1, 3)); U = np.zeros((L, 6))
    p0, c0 = _rec_state(shadow)
    pos[0], cc[0], vt[0] = p0, c0, shadow.env.v
    prev = p0
    n, dropped = L, False
    for k in range(L):
        c = cc[k]
        v_est = (pos[k] - pos[k - 1]) / DT if k > 0 else np.zeros(3)
        u = prog(c, np.array([pos[k, 0], pos[k, 1], v_est[0], v_est[1]]))
        r = shadow.tick(u)
        U[k] = u
        pos[k + 1], cc[k + 1] = _rec_state(shadow)
        vt[k + 1] = shadow.env.v
        if r["failed"] or r["tunnel"]:
            n, dropped = k + 1, True
            break
    if n < L:
        pos[n + 1:], cc[n + 1:], vt[n + 1:] = pos[n], cc[n], vt[n]
    return dict(pos=pos, c=cc, u=U, vt=vt, n=n, dropped=dropped)


G = {}


def _init(data, split, L, stride, t0, progs):
    G["st"] = Store(data, split)
    G["meta"] = G["st"].meta
    G["renv"] = E.env_from_meta(G["meta"])
    r = G["renv"]
    G["shadow"] = E.TrayRepEnv("full", r.a_scale, r.lip, w_scale=r.w_scale, alpha=r.alpha)
    G["cfg"] = dict(L=L, stride=stride, t0=t0, progs=progs)


def _work(ep):
    st, meta, renv, shadow, cfg = G["st"], G["meta"], G["renv"], G["shadow"], G["cfg"]
    L = cfg["L"]
    info = st.ep_info(ep)
    seed = int(info["seed"])
    rng = np.random.default_rng(seed * 7 + 13)
    nm = EP.Nominal(meta, renv, seed, "closedloop")
    rec = renv.reset(ball_xy=nm.ball_xy, mu_roll=nm.mu, track_seed=seed + EP.TRK_SEED_BASE)
    out = []
    # program 0: the nominal carry itself, cut into windows of L ticks (non-overlapping) is done in the analysis; store whole carry
    T = nm.T
    pos = np.zeros((T + 1, 3)); cc = np.zeros((T + 1, 3)); U = np.zeros((T, 6))
    pos[0], cc[0] = _rec_state(renv)
    n = T
    for t in range(T):
        # branches first (state at tick t)
        if t >= cfg["t0"] and (t - cfg["t0"]) % cfg["stride"] == 0 and renv.status == 0:
            snap = renv.snapshot()
            for kind in cfg["progs"]:
                if kind == 0:
                    continue
                mu = float(np.exp(rng.uniform(np.log(E.MU_LO), np.log(E.MU_HI))))
                b = run_branch(shadow, snap, mu, kind, L, rng)
                b.update(ep=ep, t=t, kind=kind, mu=mu)
                out.append(b)
        u = nm.u(t)
        r = renv.tick(np.asarray(u, np.float64))
        U[t] = u
        pos[t + 1], cc[t + 1] = _rec_state(renv)
        if r["failed"] or r["tunnel"]:
            n = t + 1
            break
    if 0 in cfg["progs"]:
        # nominal carry as chunks of L ticks (hidden mu = nm.mu)
        for s0 in range(0, max(n - L, 0) + 1, L):
            seg = slice(s0, s0 + L + 1)
            if s0 + L > n:
                break
            out.append(dict(pos=pos[seg].copy(), c=cc[seg].copy(), u=U[s0:s0 + L].copy(), vt=np.zeros((L + 1, 3)), n=L,
                            dropped=False, ep=ep, t=s0, kind=0, mu=nm.mu))
    return out


def collect(setup="piper_trk", data=None, split="val", n_eps=None, ep_start=0, L=L_DEF, stride=12, t0=10, progs=(0, 1, 2, 3),
            workers=8, out=None):
    data = Path(data) if data else K.ROOT / f"data_{setup}_full"
    st = Store(data, split)
    eps = list(range(ep_start, st.n_eps if n_eps is None else min(st.n_eps, ep_start + n_eps)))
    t_start = time.perf_counter()
    res = []
    if workers <= 1:
        _init(data, split, L, stride, t0, progs)
        for i, e in enumerate(eps):
            res += _work(e)
    else:
        with Pool(workers, initializer=_init, initargs=(data, split, L, stride, t0, progs), maxtasksperchild=20) as P:
            for i, r in enumerate(P.imap_unordered(_work, eps, chunksize=2)):
                res += r
                if (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{len(eps)} eps, {len(res)} branches, {time.perf_counter() - t_start:.0f}s", flush=True)
    res.sort(key=lambda b: (b["ep"], b["t"], b["kind"]))
    arr = {k: np.array([b[k] for b in res]) for k in ("pos", "c", "u", "vt", "n", "dropped", "ep", "t", "kind", "mu")}
    meta = dict(setup=setup, data=str(data), split=split, L=L, stride=stride, t0=t0, progs=list(progs), n_eps=len(eps),
                wall_s=time.perf_counter() - t_start, a_max=float(G_meta_amax(data)), note="fixed pin_cup physics")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **arr, meta=json.dumps(meta))
    return arr, meta


def G_meta_amax(data):
    m = json.loads((Path(data) / "meta.json").read_text())
    return 5.0 * m["a_scale"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", default="piper_trk")
    ap.add_argument("--data", default="", help="dataset dir (default data_<setup>_full; use ..._full_v2 if present)")
    ap.add_argument("--n-eps", type=int, default=None)
    ap.add_argument("--ep-start", type=int, default=0)
    ap.add_argument("--L", type=int, default=L_DEF)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arr, meta = collect(a.setup, a.data or None, "val", a.n_eps, a.ep_start, a.L, a.stride, 10, (0, 1, 2, 3), a.workers, a.out)
    kinds = np.bincount(arr["kind"], minlength=4)
    print(f"{len(arr['ep'])} branches by program {kinds.tolist()}, dropped {int(arr['dropped'].sum())}, {meta['wall_s']:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
