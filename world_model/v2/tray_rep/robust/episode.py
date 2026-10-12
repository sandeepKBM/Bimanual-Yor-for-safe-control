"""robust/episode.py -- replay one nominal carry with a filter acting every tick on a CORRUPTED ball observation.
SIMULATION / SAVED DATA ONLY (MuJoCo CPU). Copy of bridge/episode.run_carry with three changes:
  * the env has NO tracker (exact ball in its obs); the filter sees obs_c = [exact tray part, corrupted ball part];
  * the corruption (robust.corrupt.Corruptor) is a function of the TRUE ball stream, the condition, and a seed, so every
    filter sees the same corrupted stream on the same carry (as long as the carry itself is the same);
  * per-tick phase bookkeeping: ticks and interventions by phase, the phase at the drop, and the carry's class
    (steady / dropout / reacq = the worst phase inside the critical window [onset-25, onset] of a nominally failing carry;
    the whole carry if the nominal carry does not fail).
The carry is rebuilt from tick 0 from the stored seed (same random draws as gen_nominal.run_episode), like the bridge.
Record fields (on top of the bridge's): cond, ticks_ph[3], interv_ph[3], cls, drop_phase, burst, nom_drop, x_* (filter extras).
"""
from __future__ import annotations

import time

import numpy as np

from world_model.v2.tray_rep import common as K          # noqa: F401
from world_model.v2.tray_rep.bridge.episode import Nominal
from world_model.v2.tray_rep.bridge.filters import project_box
from world_model.v2.tray_rep.robust import corrupt as CR

DT_WM = CR.DT
CRIT = 25                     # ticks before the nominal failure that define the critical window


def carry_key(seed, run_seed):
    return (int(seed) * 1000003 + int(run_seed) * 7919 + 5) & 0x7FFFFFFF


def run_carry_c(renv, meta, seed, filt, cond, run_seed=0, nominal="closedloop", logged_act=None, onset_stored=-1):
    nm = Nominal(meta, renv, seed, nominal, logged_act)
    T = nm.T
    key = carry_key(seed, run_seed)
    bs = CR.burst_start(cond, key, t_fail=onset_stored, T=T)
    burst = (bs, cond.blen) if bs is not None else None
    cr = CR.Corruptor(cond, key, burst)
    rec = renv.reset(ball_xy=nm.ball_xy, mu_roll=nm.mu, track_seed=0)
    info = cr.reset(rec["priv"][0:2], rec["priv"][3:5])
    obs = np.zeros((T + 1, renv.obs_dim), np.float32)
    act = np.zeros((T, 6), np.float32)
    obs[0, :9] = rec["obs"][:9]
    obs[0, 9:] = info["ball"]
    filt.begin(dict(renv=renv, hid=nm.mu, seed=seed, T=T))
    a_max, w_max = renv.a_max, renv.w_max
    scale = np.array([a_max] * 3 + [w_max] * 3)
    ticks_ph = np.zeros(3, int)
    interv_ph = np.zeros(3, int)
    phases = []
    n_int = 0
    dev_raw = dev_norm = 0.0
    dropped = tunnel = False
    drop_tick, drop_phase, n_ticks = -1, -1, 0
    t_f = t_s = 0.0
    for t in range(T):
        t0 = time.perf_counter()
        u_nom = nm.u(t)
        filt.info = info
        u = project_box(filt(obs[:t + 1], act[:t], u_nom, t), a_max, w_max)
        t1 = time.perf_counter()
        d = u - u_nom
        iv = bool(np.any(np.abs(d) > 1e-9))
        ph = info["phase"]
        phases.append(ph)
        ticks_ph[ph] += 1
        interv_ph[ph] += int(iv)
        n_int += int(iv)
        dev_raw += float(np.linalg.norm(d))
        dev_norm += float(np.linalg.norm(d / scale))
        r = renv.tick(u)
        info = cr.update(r["priv"][0:2], r["priv"][3:5])
        t_f += t1 - t0
        t_s += time.perf_counter() - t1
        obs[t + 1, :9] = r["obs"][:9]
        obs[t + 1, 9:] = info["ball"]
        act[t] = r["act"]
        n_ticks = t + 1
        tunnel |= bool(r["tunnel"])
        if r["failed"] or r["tunnel"]:
            dropped, drop_tick, drop_phase = True, t + 1, ph
            break
    ph_arr = np.array(phases)
    win = ph_arr[max(0, onset_stored - CRIT):onset_stored + 1] if onset_stored >= 0 else ph_arr
    cls = CR.PH_DROP if (win == CR.PH_DROP).any() else (CR.PH_REACQ if (win == CR.PH_REACQ).any() else CR.PH_STEADY)
    out = dict(seed=int(seed), L=int(T), onset_stored=int(onset_stored), nom_drop=bool(onset_stored >= 0), dropped=bool(dropped),
               drop_tick=int(drop_tick), tunnel=bool(tunnel), n_ticks=int(n_ticks), n_interv=int(n_int),
               dev_raw_sum=dev_raw, dev_norm_sum=dev_norm, completion=float((drop_tick if dropped else T) / T),
               filt_s=t_f, sim_s=t_s, mu=float(nm.mu), cond=cond.name, ticks_ph=ticks_ph.tolist(), interv_ph=interv_ph.tolist(),
               cls=int(cls), drop_phase=int(drop_phase), burst=(list(burst) if burst else None))
    out.update(filt.stats())
    return out
