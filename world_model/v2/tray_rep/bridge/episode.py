"""bridge/episode.py -- replay one nominal carry in MuJoCo with a filter acting every WM tick (50 Hz).
SIMULATION / SAVED DATA ONLY (MuJoCo CPU; no hardware, no network).

The nominal carry is rebuilt from the stored episode seed with the SAME code and random draws as gen_nominal.run_episode
(mu_roll, ball start, carry plan, tracker noise stream), so with the filter F0 the carry reproduces the stored one bitwise.
(The stored snapshots start at tick 20, so the carry is regenerated from tick 0 instead of restored.)

nominal = "closedloop" (default): u_nom(t) = gen_nominal.carry_u_clip(...) -- the operator law. It reads the TRUE tray axis, so when a
          filter bends the tray the operator reacts, like a person or a policy that sees the tray. Its noise stream advances
          one step per tick whatever the filter does, so the carry stays time-aligned.
nominal = "replay": u_nom(t) = the logged action [a, realised tilt rate] of the stored episode, open loop.
"""
from __future__ import annotations

import time

import numpy as np

from world_model.v2.tray_rep import common as K          # noqa: F401
from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep import gen_nominal as GN
from world_model.v2.testbeds.b_tray_ball.carry import make_carry, carry_u
from world_model.v2.tray_rep.bridge.metrics import PathProgress

DT_WM = E.DT_WM
TRK_SEED_BASE = 777_000_000          # gen_nominal.run_episode: track_seed = seed + 777_000_000
TRK_SEED_STRIDE = 1_000_003          # run seed s > 0 shifts the tracker noise stream (s = 0 reproduces the stored data)


class Nominal:
    def __init__(self, meta, renv, seed, mode="closedloop", logged_act=None):
        rng = np.random.default_rng(int(seed))
        self.mu = E.sample_mu_roll(rng)
        self.ball_xy = rng.normal(0, 0.005, 2)
        self.cr = make_carry(rng, meta["intensity"])
        self.T = int(self.cr["T_end"] / DT_WM)
        self.carry_mode = meta.get("carry_mode", "orig")
        self.mode, self.logged = mode, logged_act
        self.renv = renv

    def u(self, t):
        if self.mode == "replay":
            return np.array(self.logged[min(t, len(self.logged) - 1)], np.float64)
        ax = self.renv.env.axis()
        if self.carry_mode == "clip":
            return GN.carry_u_clip(self.cr, t * DT_WM, ax, self.renv.a_scale, self.renv.w_scale)
        return carry_u(self.cr, t * DT_WM, ax)


def run_carry(renv, meta, seed, filt, run_seed=0, nominal="closedloop", logged_act=None, onset_stored=-1):
    """-> record dict (see metrics.py). The carry stops at the first drop (ball centre off the tray)."""
    nm = Nominal(meta, renv, seed, nominal, logged_act)
    T = nm.T
    rec = renv.reset(ball_xy=nm.ball_xy, mu_roll=nm.mu, track_seed=int(seed) + TRK_SEED_BASE + TRK_SEED_STRIDE * int(run_seed))
    obs = np.zeros((T + 1, renv.obs_dim), np.float32)
    act = np.zeros((T, 6), np.float32)
    obs[0] = rec["obs"]
    filt.begin(dict(renv=renv, hid=nm.mu, seed=seed, T=T))
    a_max, w_max = renv.a_max, renv.w_max
    scale = np.array([a_max] * 3 + [w_max] * 3)
    n_int = 0
    dev_raw = dev_norm = 0.0
    dropped = tunnel = False
    drop_tick = -1
    n_ticks = 0
    t_f = t_s = 0.0
    from world_model.v2.tray_rep.bridge.filters import project_box
    prog = PathProgress(renv.env.pos[:2], [sg[2] for sg in nm.cr["segs"]])          # path-progress metric (added fields)
    for t in range(T):
        t0 = time.perf_counter()
        u_nom = nm.u(t)
        u = project_box(filt(obs[:t + 1], act[:t], u_nom, t), a_max, w_max)
        t1 = time.perf_counter()
        d = u - u_nom
        if np.any(np.abs(d) > 1e-9):
            n_int += 1
        dev_raw += float(np.linalg.norm(d))
        dev_norm += float(np.linalg.norm(d / scale))
        r = renv.tick(u)
        prog.update(renv.env.pos[:2], t + 1)
        t_f += t1 - t0
        t_s += time.perf_counter() - t1
        obs[t + 1] = r["obs"]
        act[t] = r["act"]
        n_ticks = t + 1
        tunnel |= bool(r["tunnel"])
        if r["failed"] or r["tunnel"]:
            dropped, drop_tick = True, t + 1
            break
    out = dict(seed=int(seed), L=int(T), onset_stored=int(onset_stored), dropped=bool(dropped), drop_tick=int(drop_tick),
               tunnel=bool(tunnel), n_ticks=int(n_ticks), n_interv=int(n_int), dev_raw_sum=dev_raw, dev_norm_sum=dev_norm,
               completion=float((drop_tick if dropped else T) / T), filt_s=t_f, sim_s=t_s, mu=float(nm.mu))
    out.update(prog.record())            # path_len_nom, path_len_done, progress, goal_tick, goal_reached (new; old fields unchanged)
    out.update(filt.stats())
    return out
