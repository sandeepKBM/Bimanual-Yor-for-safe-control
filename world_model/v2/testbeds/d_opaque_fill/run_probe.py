"""run_probe.py -- feasibility probe for the opaque-fill testbed (CPU only, no GPU, no rendering). SIMULATION ONLY.

  python -m world_model.v2.testbeds.d_opaque_fill.run_probe --phase all --procs 8

Phases: p1 (replay spill_v2 carries x fill levels), p2 (savable-set search from sampled pre-failure states, MuJoCo and
analytic pendulum), probe (identifiability: static + dynamic load vs fill/type), torque (joint-torque difference through the
YOR left-arm Jacobian). Writes /common/users/ss5772/bimanual_data/testbeds/d_opaque_fill/feasibility.json (merged by phase).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[k] = "1"

import numpy as np

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from world_model.v2.testbeds.d_opaque_fill import lib as L           # noqa: E402
from world_model.v2.testbeds.d_opaque_fill import opaque_env as O    # noqa: E402
from world_model.v2.latent_hj import slosh_pend as sp                # noqa: E402

OUT = Path("/common/users/ss5772/bimanual_data/testbeds/d_opaque_fill")
FILL_N = (40, 80, 110, 140, 165)           # grains; settled fill height measured per env (env.top)
N_CARRIES = 14
TAUS = (10, 30, 60)                         # target ticks before first spill for the pre-failure state samples
U = L.B.fallback_U()
ORDER = L.gentle_order(U)
EPS = None


def init():
    global EPS
    EPS = L.load_carries(4, N_CARRIES)


def mkenv(n, mu, drag):
    return O.OpaqueEnv(O.Fill(n=n, mu=mu, drag=drag))


# ----------------------------------------------------------------------------------------------- phase 1
def p1_task(task):
    n, mu, drag, ci = task
    t0 = time.process_time()
    env = mkenv(n, mu, drag)
    ep = EPS[ci]
    r = L.replay(env, ep)
    onset = r["onset"]
    snaps = []
    if onset >= 0 and not r["tunnel"]:
        for tau in TAUS:
            j = ((onset - tau) // 5) * 5
            if j >= 20:
                s = [sn for (jj, sn) in r["snaps"] if jj == j]
                if s:
                    snaps.append((tau, int(j), s[0]))
    # tilt at the first spill tick, max tilt overall
    til = r["tilt"]
    # analytic pendulum along the same carry, hidden fill height = the settled height (true parameter)
    P = sp.PendParams(h=env.top)
    pm = O.PendModel(env.top)
    P.w1, P.L, P.phi_star = pm.w1, pm.L, pm.phi_star
    Nn, ND, Cc, PHI = sp.replay_np(ep["pos"] + env.top * ep["axis"], ep["axis"], P)
    pend_spill = bool((PHI >= pm.phi_star).any())
    pend_onset = int(np.argmax(PHI >= pm.phi_star) // 4) if pend_spill else -1
    return dict(n=n, mu=mu, drag=drag, ci=ci, top=env.top, mass=env.fill.mass, onset=onset, spilled=onset >= 0,
                tunnel=bool(r["tunnel"]), n_rim=int(r["n_rim"]), lmin=float(r["l"].min()),
                tilt_onset=float(til[min(onset, len(til) - 1)]) if onset >= 0 else None, tilt_max=float(til.max()),
                T4=int(r["T4"]), pend_spill=pend_spill, pend_onset=pend_onset, phi_star=float(pm.phi_star),
                snaps=snaps, cpu=time.process_time() - t0)


# ----------------------------------------------------------------------------------------------- phase 1b (fine snapshots near onset)
TAUS_B = (2, 4, 6)


def p1b_task(task):
    """replay a known-spilling carry with a snapshot at EVERY tick; keep the snapshots TAUS_B ticks before the first spill."""
    n, mu, drag, ci = task
    t0 = time.process_time()
    env = mkenv(n, mu, drag)
    r = L.replay(env, EPS[ci], snap_stride=1)
    onset = r["onset"]
    snaps = []
    for tau in TAUS_B:
        j = onset - tau
        s = [sn for (jj, sn) in r["snaps"] if jj == j]
        if j >= 20 and s:
            snaps.append((tau, int(j), s[0]))
    return dict(n=n, mu=mu, drag=drag, ci=ci, onset=onset, snaps=snaps, cpu=time.process_time() - t0)


# ----------------------------------------------------------------------------------------------- phase 2
def pend_savability(pm: "O.PendModel", n0, nd0, c0, U, order, T=L.T_BR):
    def run(u):
        n, nd, c = n0.copy(), nd0.copy(), c0.copy()
        for _ in range(T):
            n, nd, c = pm.tick(n, nd, c, u)
            if pm.margin(n, c) < 0:
                return True
        return False
    i0 = int(np.argmin(np.abs(U).sum(1)))
    out = dict(hold_spills=run(U[i0]), savable=False, tried=1)
    if not out["hold_spills"]:
        return out
    for i in order:
        if i == i0:
            continue
        out["tried"] += 1
        if not run(U[i]):
            out["savable"] = True
            return out
    return out


def p2_task(task):
    n, mu, drag, ci, tau, j, snap, ep_i = task
    t0 = time.process_time()
    env = mkenv(n, mu, drag)
    res = L.savability(env, snap, U, ORDER)
    c1 = time.process_time() - t0
    # the analytic pendulum labels the SAME state (true hidden fill height)
    ep = EPS[ci]
    pm = O.PendModel(env.top)
    P = sp.PendParams(h=env.top)
    P.w1, P.L, P.phi_star = pm.w1, pm.L, pm.phi_star
    k = 4 * j
    Nn, ND, Cc, PHI = sp.replay_np(ep["pos"][:k + 1] + env.top * ep["axis"][:k + 1], ep["axis"][:k + 1], P)
    pres = pend_savability(pm, Nn[-1], ND[-1], Cc[-1], U, ORDER)
    # MuJoCo margin at that state
    env.restore(snap)
    lstate = O.margin(env.local())
    return dict(n=n, mu=mu, drag=drag, ci=ci, tau=tau, j=j, top=env.top, mj=res, pend=pres, l_state=lstate,
                cpu_mj=c1, cpu=time.process_time() - t0)


# ----------------------------------------------------------------------------------------------- identification probe
def probe_task(task):
    """gentle probe: hold 0.3 s, lateral accel pulse a = 1.5 m/s^2 for 0.2 s (then -1.5 for 0.2 s), hold 0.6 s.
    returns the wrench (F on container by grains) trace at 50 Hz and summary features."""
    n, mu, drag = task
    t0 = time.process_time()
    env = mkenv(n, mu, drag)
    env.reset_settled()
    seq = [np.zeros(6)] * 15 + [np.array([1.5, 0, 0, 0, 0, 0])] * 10 + [np.array([-1.5, 0, 0, 0, 0, 0])] * 10 + [np.zeros(6)] * 30
    F, M = [], []
    for u in seq:
        r = env.tick(u, log=True)
        F.append(r["F"]); M.append(r["M"])
    F, M = np.array(F), np.array(M)
    m = env.fill.mass
    Fz0 = float(-F[5:15, 2].mean())                    # static load (N) from the vertical force before the pulse
    # force in x during the +a pulse relative to rigid m*a: momentum coupling of the contents
    fx = F[15:25, 0]
    rigid = m * 1.5
    return dict(n=n, mu=mu, drag=drag, mass=m, top=env.top, Fz_static=Fz0, m_est=Fz0 / O.G,
                fx_peak=float(np.abs(fx).max()), fx_mean_pulse=float(fx[2:].mean()), rigid=rigid,
                coupling=float(np.abs(fx[2:].mean()) / rigid), ringdown=float(np.abs(F[40:, 0] - F[40:, 0].mean()).std()),
                Fx=F[:, 0].tolist(), cpu=time.process_time() - t0)


# ----------------------------------------------------------------------------------------------- torque stage
def arm_jacobian():
    import mujoco
    sys.path.insert(0, str(REPO / "experiments" / "ab_smooth_vs_bangbang"))
    from world_model.v2.data import contents as C
    m = mujoco.MjModel.from_xml_path(str(C.ROBOT))
    d = mujoco.MjData(m)
    names = [f"left_arm_joint{i}" for i in range(1, 7)]
    qa = [m.jnt_qposadr[m.joint(nm).id] for nm in names]
    da = [m.jnt_dofadr[m.joint(nm).id] for nm in names]
    q = np.array([-0.00944173, 2.69291536, -1.83582068, -0.47028722, 1.03733985, -1.30204097])   # sim_common.LEFT_GRASP_Q
    d.qpos[qa] = q
    m.body_gravcomp[:] = 0.0
    mujoco.mj_forward(m, d)
    sid = m.site("left_arm_ee_grasp").id
    Jp, Jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, Jp, Jr, sid)
    return Jp[:, da], Jr[:, da], d.qfrc_bias[da].copy(), d.site_xmat[sid].reshape(3, 3).copy()


def torque_task(task):
    n, ci = task
    t0 = time.process_time()
    env = mkenv(n, 0.5, 0.0)
    ep = EPS[ci]
    r = L.replay(env, ep, log_wrench=True)
    W = r["wrench"]                                     # (nt,6): F (world), M about container origin (world)
    ax = ep["axis"][::4][:len(W)]
    Jp, Jr, grav, _ = arm_jacobian()
    Mg = W[:, 3:] + np.cross(-0.06 * ax, W[:, :3])     # moment about a grasp point 6 cm up the container axis
    tau = -(W[:, :3] @ Jp + Mg @ Jr)                    # torque the arm must add to carry the grains (sign: opposes load)
    tau_static = -(np.array([0, 0, -O.G * env.fill.mass]) @ Jp)
    dyn = tau - tau_static
    end = r["onset"] if r["onset"] >= 0 else len(W)          # only count ticks before the first spill (mass is lost afterwards)
    tau, dyn = tau[:end], dyn[:end]
    return dict(n=n, ci=ci, mass=env.fill.mass, tau_static=tau_static.tolist(), tau_peak=np.abs(tau[20:]).max(0).tolist(),
                dyn_peak=np.abs(dyn[20:]).max(0).tolist(), dyn_rms=np.sqrt((dyn[20:] ** 2).mean(0)).tolist(), n_ticks=int(end),
                grav=grav.tolist(), spilled=bool(r["onset"] >= 0), cpu=time.process_time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--states", type=int, default=36)
    ap.add_argument("--cpu-cap", type=float, default=1500.0, help="stop launching phase-2 states beyond this total CPU [s]")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jf = OUT / "feasibility.json"
    J = json.loads(jf.read_text()) if jf.exists() else {}
    ctx = mp.get_context("fork")
    t_wall = time.time()
    if a.phase in ("p1", "all"):
        tasks = [(n, 0.5, 0.0, ci) for n in FILL_N for ci in range(N_CARRIES)] + [(140, 0.03, 3.0, ci) for ci in range(N_CARRIES)]
        with ctx.Pool(a.procs, initializer=init, maxtasksperchild=4) as pool:
            R = pool.map(p1_task, tasks, chunksize=1)
        np.save(OUT / "tmp" / "p1_snaps.npy", np.array([(r["n"], r["mu"], r["ci"], len(r["snaps"])) for r in R], float))
        import pickle
        (OUT / "tmp").mkdir(exist_ok=True)
        pickle.dump(R, open(OUT / "tmp" / "p1.pkl", "wb"))
        J["p1_cpu_s"] = float(sum(r["cpu"] for r in R))
        J["p1_wall_s"] = time.time() - t_wall
        J["p1_rows"] = [{k: v for k, v in r.items() if k != "snaps"} for r in R]
        jf.write_text(json.dumps(J, indent=1))
        print("p1 done cpu", J["p1_cpu_s"], "wall", J["p1_wall_s"], flush=True)
    if a.phase in ("p2", "all"):
        import pickle
        R = pickle.load(open(OUT / "tmp" / "p1.pkl", "rb"))
        pool_states = [(r["n"], r["mu"], r["drag"], r["ci"], tau, j, sn) for r in R if r["mu"] == 0.5 for (tau, j, sn) in r["snaps"]]
        rng = np.random.default_rng(0)
        rng.shuffle(pool_states)
        sel = pool_states[:a.states]
        tasks = [(n, mu, drag, ci, tau, j, sn, i) for i, (n, mu, drag, ci, tau, j, sn) in enumerate(sel)]
        t1 = time.time()
        with ctx.Pool(a.procs, initializer=init, maxtasksperchild=2) as pool:
            R2 = pool.map(p2_task, tasks, chunksize=1)
        J["p2_n_pool"] = len(pool_states)
        J["p2_cpu_s"] = float(sum(r["cpu"] for r in R2))
        J["p2_wall_s"] = time.time() - t1
        J["p2_rows"] = R2
        jf.write_text(json.dumps(J, indent=1))
        print("p2 done cpu", J["p2_cpu_s"], "wall", J["p2_wall_s"], flush=True)
    if a.phase in ("p1b", "all"):
        import pickle
        R = pickle.load(open(OUT / "tmp" / "p1.pkl", "rb"))
        tasks = [(r["n"], r["mu"], r["drag"], r["ci"]) for r in R if r["spilled"] and r["mu"] == 0.5 and not r["tunnel"]]
        t1 = time.time()
        with ctx.Pool(a.procs, initializer=init, maxtasksperchild=2) as pool:
            RB = pool.map(p1b_task, tasks, chunksize=1)
        pickle.dump(RB, open(OUT / "tmp" / "p1b.pkl", "wb"))
        J["p1b_cpu_s"] = float(sum(r["cpu"] for r in RB))
        jf.write_text(json.dumps(J, indent=1))
        print("p1b done cpu", J["p1b_cpu_s"], "wall", time.time() - t1, flush=True)
    if a.phase in ("p2b", "all"):
        import pickle
        RB = pickle.load(open(OUT / "tmp" / "p1b.pkl", "rb"))
        tasks = [(r["n"], r["mu"], r["drag"], r["ci"], tau, j, sn, 0) for r in RB for (tau, j, sn) in r["snaps"]]
        t1 = time.time()
        with ctx.Pool(a.procs, initializer=init, maxtasksperchild=2) as pool:
            R2 = pool.map(p2_task, tasks, chunksize=1)
        J["p2b_cpu_s"] = float(sum(r["cpu"] for r in R2))
        J["p2b_rows"] = R2
        jf.write_text(json.dumps(J, indent=1))
        print("p2b done cpu", J["p2b_cpu_s"], "wall", time.time() - t1, flush=True)
    if a.phase in ("probe", "all"):
        tasks = [(n, 0.5, 0.0) for n in (40, 80, 110, 140, 165)] + [(n, 0.03, 3.0) for n in (80, 140)] + [(110, 0.2, 0.0)]
        with ctx.Pool(min(a.procs, len(tasks)), initializer=init) as pool:
            R3 = pool.map(probe_task, tasks, chunksize=1)
        J["probe_rows"] = R3
        J["probe_cpu_s"] = float(sum(r["cpu"] for r in R3))
        jf.write_text(json.dumps(J, indent=1))
        print("probe done cpu", J["probe_cpu_s"], flush=True)
    if a.phase in ("torque", "all"):
        tasks = [(n, ci) for n in (40, 100, 165) for ci in (2, 4)]
        with ctx.Pool(min(a.procs, len(tasks)), initializer=init) as pool:
            R4 = pool.map(torque_task, tasks, chunksize=1)
        J["torque_rows"] = R4
        J["torque_cpu_s"] = float(sum(r["cpu"] for r in R4))
        jf.write_text(json.dumps(J, indent=1))
        print("torque done cpu", J["torque_cpu_s"], flush=True)


if __name__ == "__main__":
    main()
