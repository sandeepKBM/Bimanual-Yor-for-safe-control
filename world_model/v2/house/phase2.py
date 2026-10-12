"""phase2.py -- Phase 2 part A: route-level tray carry through a doorway with a MOVING (mocap) base.  SIMULATION only.

    python -m world_model.v2.house.phase2 plan   --route h5  --amax 0.25 [--clear-min 0.05]
    python -m world_model.v2.house.phase2 sweep  --route h5  --caps 0.15 0.25 0.4 --seeds 10 --workers 8
    python -m world_model.v2.house.phase2 det    --route h5  --amax 0.25 --seeds 10
    python -m world_model.v2.house.phase2 one    --route h5  --amax 0.25 --seed 0

Per step: set the mocap base from the trajectory -> FL-MPC (unmodified mpc_fl via phase1.Ctl, plus the tray tilt cone
through left_ee_body / tilt_max_rad / cup_cone_facets) -> qfrc_applied -> mj_step.  The base is kinematic: base
acceleration reaches the arms ONLY through the two tray welds (measured here via qfrc_constraint on the arm DoFs).
The controller does not know the base acceleration (the effective-gravity cone sees only the arm-produced tray accel).

Seed = base start jitter (xy 5 mm sigma, yaw 0.5 deg, tapered to zero over the first metre so the door pass is not
shifted) + ball position on the tray (+-2 cm) + joint-encoder noise on what the controller reads (1e-4 rad, 2e-3 rad/s).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

from . import build as B
from . import nav as N
from . import phase1 as P1

HERE = Path(__file__).resolve().parent
RUNS = Path(os.environ.get("P2_RUNS", "/data/local/ss5772/molmospaces/p2"))
DT = 0.005
TILT_MAX = 0.10          # rad, FL-MPC tray tilt bound (scenario.py uses 0.10 box / 0.30 cone)
CONE_FACETS = 8
TAU_REAL = 8.0           # Nm, real Piper feed-forward cap (AGENTS / memory); sim TAU_MAX = 100
SETTLE_S = 0.6           # extra rest time before the trajectory
ENC_Q, ENC_V = 1e-4, 2e-3

# route registry: start/goal xy are free base poses found with nav.best_free_pose (see results/phase2_*.json)
ROUTES = {
    "h17": dict(house=17, door="door|2|7", start=(6.9, 10.6), goal=(4.85, 5.35),
                note="house 17 single door LivingRoom->Bedroom (0.884 m clear)"),
    "h5": dict(house=5, door="door|4|5", start=(3.1, 6.09), goal=(2.135, 2.0),
               note="house 5 double door Bedroom->Bathroom (1.806 m clear)"),
}


def register(name, **kw):
    ROUTES[name] = kw


# ------------------------------------------------------------------------------------------------- planning
def plan(route, amax=0.25, vmax=0.3, jmax=None, clear_min=0.05, verbose=True):
    r = ROUTES[route]
    idx = r["house"]
    m, d = B.load_house(idx, True)
    mujoco.mj_forward(m, d)
    F = N.build_occupancy(idx, m=m, d=d)
    C = N.coarsen(F)
    door = next(x for x in N.door_list(idx, m, d) if x["id"] == r["door"])
    out = N.plan_route(idx, r["start"], r["goal"], door, vmax=vmax, amax=amax, jmax=jmax, dt=DT, clear_min=clear_min,
                       F=F, C=C, verbose=verbose)
    out["door_obj"] = door
    return out, F


def save_plan(out, route, tag):
    RUNS.mkdir(parents=True, exist_ok=True)
    rt = out["route"]
    js = {k: v for k, v in out.items() if k not in ("route", "door_obj", "door_u_curve")}
    js["door_u_curve_nonzero"] = [c for c in out.get("door_u_curve", []) if c[1] > 0]
    if rt:
        tr = rt["traj"]
        js["route"] = {k: v for k, v in rt.items() if k != "traj"}
        js["route"]["traj_meta"] = tr.meta
        np.savez_compressed(RUNS / f"traj_{route}_{tag}.npz", t=tr.t, x=tr.x, y=tr.y, yaw=tr.yaw)
    (RUNS / f"plan_{route}_{tag}.json").write_text(json.dumps(js, indent=1, default=str))
    return js


def load_traj(route, tag):
    z = np.load(RUNS / f"traj_{route}_{tag}.npz")
    return dict(t=z["t"], x=z["x"], y=z["y"], yaw=z["yaw"])


# ------------------------------------------------------------------------------------------------- controller
class Ctl2(P1.Ctl):
    """phase1.Ctl + tray tilt cone (and optional per-joint 8 Nm torque box through FLMPC.tau_max_vec)."""

    def __init__(self, C, tilt=True, tau_cap=None):
        super().__init__(C)
        self.lee = C.m.body("left_arm_gripper_base").id
        self.tilt = tilt
        if tau_cap is not None:
            # real-arm emulation: the sim arm has built-in gravcomp (FL-MPC's tau excludes gravity); the real feed-forward
            # must also carry gravity (hold-pose |qfrc_gravcomp| up to ~5.4 Nm), so the box on the FL part is 8 - |g|.
            mujoco.mj_forward(C.m, C.d)
            g = np.maximum(np.abs(C.d.qfrc_gravcomp[self.ld]), np.abs(C.d.qfrc_gravcomp[self.rd]))
            self.fl.tau_max_vec = np.maximum(float(tau_cap) - g, 0.5)

    def tau(self, d):
        mujoco.mj_forward(self.m, d)
        pairs = self.sc.all_pair_barriers(self.m, d, self.ld, self.rd)
        kw = dict(left_ee_body=self.lee, tilt_max_rad=TILT_MAX, cup_cone_facets=CONE_FACETS) if self.tilt else {}
        te, ti, ok = self.fl.step(self.m, d, self.lq, self.ld, self.rq, self.rd,
                                  self.qe_ref, self.z, self.qi_ref, self.z, pairs, **kw)
        if not ok:
            self.n_infeasible += 1
        self.last_ok = ok
        return te, ti


# ------------------------------------------------------------------------------------------------- distances
class DistMon:
    """Sparse min-distance monitors: arm<->arm, robot<->house (sphere prefilter + mj_geomDistance), and robot<->door
    frame/leaf geoms (exact, few geoms)."""

    def __init__(self, C, door_bodies):
        m = C.m
        self.m = m
        self.cls = P1.geom_class(m)
        col = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
        notplane = m.geom_type != mujoco.mjtGeom.mjGEOM_PLANE
        self.L = [g for g in range(m.ngeom) if self.cls[g] == "L" and col[g]]
        self.R = [g for g in range(m.ngeom) if self.cls[g] == "R" and col[g]]
        self.own = [g for g in range(m.ngeom) if self.cls[g] in ("L", "R", "B", "T") and col[g] and notplane[g]]
        self.house = np.array([g for g in range(m.ngeom) if self.cls[g] == "H" and col[g] and notplane[g]])
        bid = {m.body(b).id for b in door_bodies}
        self.door = [g for g in self.house if int(m.geom_bodyid[g]) in bid]
        self.ft = np.zeros(6)

    def _gd(self, d, a, b, dmax):
        v = mujoco.mj_geomDistance(self.m, d, a, b, dmax, self.ft)
        if v < dmax and abs(np.linalg.norm(self.ft[3:] - self.ft[:3]) - abs(v)) > 1e-4:
            return dmax                       # spurious mesh-mesh zero beyond distmax
        return v

    def arm_arm(self, d, dmax=0.3):
        best = dmax
        for a in self.L:
            for b in self.R:
                best = min(best, self._gd(d, a, b, dmax))
        return best

    def robot_house(self, d, dmax=0.5):
        m = self.m
        hx = d.geom_xpos[self.house]
        hr = np.minimum(m.geom_rbound[self.house], 6.0)
        base = d.mocap_pos[0]
        near = self.house[np.linalg.norm(hx[:, :2] - base[:2], axis=1) - hr < 1.6 + dmax]
        best, bn = dmax, ""
        for a in self.own:
            pa, ra = d.geom_xpos[a], m.geom_rbound[a]
            sel = near[np.linalg.norm(d.geom_xpos[near] - pa, axis=1) - np.minimum(m.geom_rbound[near], 6.0) - ra < dmax]
            for b in sel:
                v = self._gd(d, a, int(b), dmax)
                if v < best:
                    best, bn = v, f"{self.cls[a]}:{m.geom(a).name or a}|{m.geom(int(b)).name or int(b)}"
        return best, bn

    def robot_door(self, d, dmax=0.5):
        best = dmax
        for a in self.own:
            for b in self.door:
                best = min(best, self._gd(d, a, int(b), dmax))
        return best


# ------------------------------------------------------------------------------------------------- episode
def tray_stats(m, d, tray, ball):
    R = d.xmat[tray].reshape(3, 3)
    rel = R.T @ (d.xpos[ball] - d.xpos[tray])
    return rel, float(math.degrees(math.acos(np.clip(R[2, 2], -1, 1))))


def run_episode(house, traj, seed, door_bodies=(), tau_cap=None, tilt=True, dist_every=40, static_s=None,
                hash_state=False, snap_at=None, restore=False, max_steps=None, record=True, frame_cb=None):
    """One carry. traj = dict(t,x,y,yaw) at DT (None + static_s => stationary carry of that many seconds).
    Returns (summary dict, arrays dict).  hash_state: returns sha256 over every step's qpos/qvel/warmstart/mocap/tau."""
    import hashlib as _h
    from world_model.v2.testbeds.b_tray_ball import tray_env as TE
    rng = np.random.default_rng(seed)
    if traj is None:
        n_t = int(round(static_s / DT))
        x0, y0, yaw0 = 0.0, 0.0, 0.0
    else:
        n_t = len(traj["t"])
        x0, y0, yaw0 = traj["x"][0], traj["y"][0], traj["yaw"][0]
    jit_xy = np.clip(rng.normal(0, 0.005, 2), -0.01, 0.01)
    jit_yaw = float(np.clip(rng.normal(0, math.radians(0.5)), -math.radians(1), math.radians(1)))
    ball_off = rng.uniform(-0.02, 0.02, 2)
    spawn = (x0 + jit_xy[0], y0 + jit_xy[1]) if traj is not None else (x0, y0)
    C = P1.new_hold(house, spawn, yaw0 + jit_yaw, DT)
    m, d = C.m, C.d
    ctl = Ctl2(C, tilt=tilt, tau_cap=tau_cap)
    tray, ball = m.body("tray").id, m.body("ball").id
    # ball jitter on the tray (tray frame x,y), at rest
    ba = m.jnt_qposadr[m.joint("ball_free").id]
    Rt = d.xmat[tray].reshape(3, 3)
    d.qpos[ba:ba + 3] += Rt[:, :2] @ ball_off
    mujoco.mj_forward(m, d)
    dm = DistMon(C, door_bodies)
    cls = dm.cls
    dn = mujoco.MjData(m)
    n_set = int(round(SETTLE_S / DT)) if traj is not None else 0
    N_ = n_set + n_t if max_steps is None else min(max_steps, n_set + n_t)
    if traj is not None:
        s_prog = np.concatenate([[0], np.cumsum(np.hypot(np.diff(traj["x"]), np.diff(traj["y"])))])
        taper = np.clip(1.0 - s_prog / 1.0, 0.0, 1.0)
    ld, rd = ctl.ld, ctl.rd
    gc_log = np.zeros((N_, 12)); tau_log = np.zeros((N_, 12)); qfc_log = np.zeros((N_, 12)); ball_log = np.zeros((N_, 3)); tilt_log = np.zeros(N_)
    inf_log = np.zeros(N_, bool); tc = np.zeros(N_); tp = np.zeros(N_)
    pose_log = np.zeros((N_, 3))
    contacts = {}
    dist_t, d_aa, d_rh, d_door, d_nm = [], [], [], [], []
    h = _h.sha256() if hash_state else None
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    snap = None
    lost_at = None
    arm_idx = np.concatenate([ld, rd])
    k0 = 0
    for k in range(k0, N_):
        if snap_at is not None and k == snap_at and not restore:
            snap = (np.zeros(mujoco.mj_stateSize(m, spec)), ctl.state(), rng.bit_generator.state, h.copy() if h else None, k)
            mujoco.mj_getState(m, d, snap[0], spec)
        if traj is not None:
            kk = min(max(k - n_set, 0), n_t - 1)
            tp_ = taper[kk]
            # jitter taper: the spawn jitter decays to zero within 1 m of progress (door pass is not shifted)
            x = traj["x"][kk] + jit_xy[0] * tp_
            y = traj["y"][kk] + jit_xy[1] * tp_
            yaw = traj["yaw"][kk] + jit_yaw * tp_
            d.mocap_pos[0] = (x, y, 0.0)
            d.mocap_quat[0] = B.yaw_quat(yaw)
        # controller reads encoder-noisy joint state (noise drawn here, in fixed order => deterministic)
        mujoco.mj_copyData(dn, m, d)
        dn.qpos[ctl.lq] += rng.normal(0, ENC_Q, 6); dn.qpos[ctl.rq] += rng.normal(0, ENC_Q, 6)
        dn.qvel[ld] += rng.normal(0, ENC_V, 6); dn.qvel[rd] += rng.normal(0, ENC_V, 6)
        t0 = time.perf_counter()
        te, ti = ctl.tau(dn)
        t1 = time.perf_counter()
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[ld] = te
        d.qfrc_applied[rd] = ti
        mujoco.mj_step(m, d)
        t2 = time.perf_counter()
        tc[k], tp[k] = (t1 - t0) * 1e3, (t2 - t1) * 1e3
        tau_log[k, :6], tau_log[k, 6:] = te, ti
        qfc_log[k] = d.qfrc_constraint[arm_idx]
        gc_log[k] = d.qfrc_gravcomp[arm_idx]
        inf_log[k] = not ctl.last_ok
        rel, tilt_deg = tray_stats(m, d, tray, ball)
        ball_log[k] = rel; tilt_log[k] = tilt_deg
        pose_log[k] = d.mocap_pos[0, 0], d.mocap_pos[0, 1], 0.0
        for c in range(d.ncon):
            cc = d.contact[c]
            a, b = sorted((cls[cc.geom1], cls[cc.geom2]))
            contacts[a + b] = contacts.get(a + b, 0) + 1
        if lost_at is None and (abs(rel[0]) > TE.HX or abs(rel[1]) > TE.HY or rel[2] < 0.0):
            lost_at = (k - n_set) * DT
        if dist_every and k % dist_every == 0:
            dist_t.append((k - n_set) * DT)
            d_aa.append(dm.arm_arm(d))
            r, nm = dm.robot_house(d)
            d_rh.append(r); d_nm.append(nm)
            d_door.append(dm.robot_door(d) if dm.door else np.nan)
        if h is not None:
            h.update(d.qpos.tobytes()); h.update(d.qvel.tobytes()); h.update(d.qacc_warmstart.tobytes())
            h.update(d.mocap_pos.tobytes()); h.update(d.mocap_quat.tobytes()); h.update(tau_log[k].tobytes())
        if frame_cb is not None:          # rendering hook (demo_route.py); read-only on the simulation
            frame_cb(k, k - n_set, m, d, dict(rel=rel, tilt=tilt_deg, d_rh=(d_rh[-1] if d_rh else None),
                                              d_aa=(d_aa[-1] if d_aa else None), tau=tau_log[k], ctl_ms=tc[k],
                                              infeasible=bool(inf_log[k])))
    out_hash = h.hexdigest() if h is not None else None
    # ---- restore leg of the determinism test: rerun from the snapshot in the SAME process on fresh data
    restore_hash = None
    if snap is not None and h is not None:
        restore_hash = _rerun_from_snapshot(C, ctl, snap, traj, taper, jit_xy, jit_yaw, n_set, n_t, N_, rng, spec,
                                            ld, rd)
    # ---- summary
    sdf = np.array([TE.sdf_rect(r[0], r[1], False) for r in ball_log])
    sl = slice(int(1.0 / DT), None)
    teq = tau_log + gc_log                     # torque a real arm (no built-in gravcomp) would need as feed-forward
    ov8 = np.abs(teq) > TAU_REAL
    ov8_fl = np.abs(tau_log) > TAU_REAL
    summ = dict(
        house=house, seed=seed, steps=N_, route_s=(N_ - n_set) * DT, tau_cap=tau_cap, tilt=tilt,
        ball_lost=lost_at is not None, ball_lost_at_s=lost_at,
        ball_max_xy_offset_m=float(np.hypot(ball_log[:, 0], ball_log[:, 1]).max()),
        ball_drift_max_m=float(np.hypot(*(ball_log[n_set:, :2] - ball_log[max(n_set - 1, 0), :2]).T).max()),
        ball_final_offset_m=[float(ball_log[-1, 0]), float(ball_log[-1, 1])],
        ball_min_edge_margin_m=float(sdf.min()), ball_min_z=float(ball_log[:, 2].min()),
        tray_tilt_max_deg=float(tilt_log.max()), tray_tilt_p95_deg=float(np.percentile(tilt_log, 95)),
        contacts=contacts, n_infeasible=int(ctl.n_infeasible), n_infeasible_log=int(inf_log.sum()),
        ctl_ms_median=float(np.median(tc[sl])), ctl_ms_p95=float(np.percentile(tc[sl], 95)), ctl_ms_max=float(tc[sl].max()),
        phys_ms_median=float(np.median(tp[sl])), phys_ms_p95=float(np.percentile(tp[sl], 95)),
        step_ms_median=float(np.median((tc + tp)[sl])),
        dist_arm_arm_min_m=float(np.min(d_aa)) if d_aa else None,
        dist_robot_house_min_m=float(np.min(d_rh)) if d_rh else None,
        dist_robot_house_argmin=d_nm[int(np.argmin(d_rh))] if d_rh else None,
        dist_door_geoms_min_m=float(np.nanmin(d_door)) if d_door and not np.all(np.isnan(d_door)) else None,
        tau_peak_per_joint=np.abs(tau_log).max(0).tolist(), tau_peak_max=float(np.abs(tau_log).max()),
        frac_steps_any_gt_8Nm_flpart=float(ov8_fl.any(1).mean()),
        tau_equiv_peak_per_joint=np.abs(teq).max(0).tolist(), tau_equiv_peak_max=float(np.abs(teq).max()),
        gravcomp_absmax=np.abs(gc_log).max(0).tolist(),
        frac_steps_any_gt_8Nm=float(ov8.any(1).mean()), frac_per_joint_gt_8Nm=ov8.mean(0).tolist(),
        tau_mean=tau_log[sl].mean(0).tolist(), tau_std=tau_log[sl].std(0).tolist(),
        qfc_mean=qfc_log[sl].mean(0).tolist(), qfc_std=qfc_log[sl].std(0).tolist(),
        qfc_absmax=np.abs(qfc_log).max(0).tolist(),
        jitter=dict(xy=jit_xy.tolist(), yaw_deg=math.degrees(jit_yaw), ball_off=ball_off.tolist()),
        hash=out_hash, restore_hash=restore_hash,
    )
    if traj is not None:
        summ["end_err_to_goal_m"] = float(math.hypot(pose_log[-1, 0] - traj["x"][-1], pose_log[-1, 1] - traj["y"][-1]))
    arr = dict(gc=gc_log, tau=tau_log, qfc=qfc_log, ball=ball_log, tilt=tilt_log, infeas=inf_log, tc=tc, tp=tp, pose=pose_log,
               dist_t=np.array(dist_t), d_aa=np.array(d_aa), d_rh=np.array(d_rh), d_door=np.array(d_door))
    return summ, arr


def _rerun_from_snapshot(C, ctl, snap, traj, taper, jit_xy, jit_yaw, n_set, n_t, N_, rng, spec, ld, rd):
    """Restore (mj_setState + FL-MPC prev accel + rng state) into FRESH MjData and re-run to the end; return hash of the
    same quantities (continuing the hash object copied at the snapshot)."""
    m = C.m
    d2 = mujoco.MjData(m)
    state, cs, rngstate, hobj, k_snap = snap
    mujoco.mj_setState(m, d2, state, spec)
    ctl.set_state(cs)
    r2 = np.random.default_rng()
    r2.bit_generator.state = rngstate
    h = hobj
    dn = mujoco.MjData(m)
    for k in range(k_snap, N_):
        if traj is not None:
            kk = min(max(k - n_set, 0), n_t - 1)
            tp_ = taper[kk]
            d2.mocap_pos[0] = (traj["x"][kk] + jit_xy[0] * tp_, traj["y"][kk] + jit_xy[1] * tp_, 0.0)
            d2.mocap_quat[0] = B.yaw_quat(traj["yaw"][kk] + jit_yaw * tp_)
        mujoco.mj_copyData(dn, m, d2)
        dn.qpos[ctl.lq] += r2.normal(0, ENC_Q, 6); dn.qpos[ctl.rq] += r2.normal(0, ENC_Q, 6)
        dn.qvel[ld] += r2.normal(0, ENC_V, 6); dn.qvel[rd] += r2.normal(0, ENC_V, 6)
        te, ti = ctl.tau(dn)
        d2.qfrc_applied[:] = 0.0
        d2.qfrc_applied[ld] = te; d2.qfrc_applied[rd] = ti
        mujoco.mj_step(m, d2)
        tau = np.concatenate([te, ti])
        h.update(d2.qpos.tobytes()); h.update(d2.qvel.tobytes()); h.update(d2.qacc_warmstart.tobytes())
        h.update(d2.mocap_pos.tobytes()); h.update(d2.mocap_quat.tobytes()); h.update(tau.tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------- workers
def _worker(a):
    (route, tag, seed, tau_cap, tilt, kw) = a
    r = ROUTES[route]
    ref = kw.get("ref_tag", tag)
    traj = load_traj(route, ref)
    if tag == "static":                      # stationary carry at the route's start pose
        n = int(round(kw["static_s"] / DT))
        traj = dict(t=np.arange(n) * DT, x=np.full(n, traj["x"][0]), y=np.full(n, traj["y"][0]), yaw=np.full(n, traj["yaw"][0]))
    door_bodies = json.loads((RUNS / f"plan_{route}_{ref}.json").read_text()).get("door_bodies", [])
    t = time.time()
    summ, arr = run_episode(r["house"], traj, seed, door_bodies=door_bodies, tau_cap=tau_cap, tilt=tilt,
                            static_s=None, hash_state=kw.get("hash_state", False),
                            snap_at=kw.get("snap_at"), max_steps=kw.get("max_steps"))
    summ["wall_s"] = time.time() - t
    summ["route"], summ["tag"] = route, tag
    if kw.get("save", True):
        np.savez_compressed(RUNS / f"run_{route}_{tag}_s{seed}_cap{tau_cap}_{'tilt' if tilt else 'notilt'}.npz", **arr)
    return summ


def _pool(tasks, workers):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(workers, 8), maxtasksperchild=1) as p:
        return list(p.imap(_worker, tasks))


# ------------------------------------------------------------------------------------------------- analysis
def _lp(x, w):
    k = np.ones(w) / w
    xp = np.concatenate([np.repeat(x[:1], w, 0), x, np.repeat(x[-1:], w, 0)])
    return np.stack([np.convolve(xp[:, j], k, mode="same") for j in range(x.shape[1])], 1)[w:-w]


def disturbance(route, tag, seeds, static_seed=0, win_s=0.5):
    """Base-motion disturbance on arm torque.  Quantities per run (FL-MPC tau and qfrc_constraint on the 12 arm DoFs):
    (a) RMS / peak deviation from the stationary-carry mean, raw and after a 0.5 s moving average (encoder noise
    dominates the raw signal), (b) least-squares gain of the filtered signal on base acceleration (world x, y)."""
    st = np.load(RUNS / f"run_{route}_static_s{static_seed}_capNone_tilt.npz")
    n0 = int(1.0 / DT)
    mu_tau, mu_qfc = st["tau"][n0:].mean(0), st["qfc"][n0:].mean(0)
    w = int(win_s / DT)
    sd_raw_tau = (st["tau"][n0:] - mu_tau).std(0)
    sd_f_tau = (_lp(st["tau"], w)[n0:] - mu_tau).std(0)
    sd_f_qfc = (_lp(st["qfc"], w)[n0:] - mu_qfc).std(0)
    rows = []
    for sd in seeds:
        f = RUNS / f"run_{route}_{tag}_s{sd}_capNone_tilt.npz"
        if not f.exists():
            continue
        z = np.load(f)
        pose = z["pose"]
        v = np.gradient(pose[:, :2], DT, axis=0)
        acc = np.gradient(_lp(v, 40), DT, axis=0)
        moving = np.hypot(v[:, 0], v[:, 1]) > 0.01
        tf, qf = _lp(z["tau"], w), _lp(z["qfc"], w)
        dtau, dqfc = tf - mu_tau, qf - mu_qfc
        A = np.c_[acc, np.ones(len(acc))][moving]
        coef_t = np.linalg.lstsq(A, dtau[moving], rcond=None)[0][:2]       # (2, 12) Nm per m/s^2
        coef_q = np.linalg.lstsq(A, dqfc[moving], rcond=None)[0][:2]
        pred = A @ np.linalg.lstsq(A, dtau[moving], rcond=None)[0]
        r2 = 1 - ((dtau[moving] - pred) ** 2).sum(0) / ((dtau[moving] - dtau[moving].mean(0)) ** 2).sum(0)
        rows.append(dict(seed=sd, peak_acc=float(np.hypot(acc[:, 0], acc[:, 1]).max()),
                         tau_dev_rms_filtered=np.sqrt((dtau[moving] ** 2).mean(0)).tolist(),
                         tau_dev_peak_filtered=np.abs(dtau).max(0).tolist(),
                         qfc_dev_rms_filtered=np.sqrt((dqfc[moving] ** 2).mean(0)).tolist(),
                         qfc_dev_peak_filtered=np.abs(dqfc).max(0).tolist(),
                         gain_tau_per_ms2_norm=float(np.linalg.norm(coef_t, axis=0).max()),
                         gain_qfc_per_ms2_norm=float(np.linalg.norm(coef_q, axis=0).max()),
                         r2_tau_on_acc_max=float(np.nanmax(r2)),
                         tau_raw_rms=float(np.sqrt(((z["tau"] - mu_tau)[moving] ** 2).mean(0)).max())))
    return dict(static=dict(tau_mean=mu_tau.tolist(), raw_std_max=float(sd_raw_tau.max()),
                            filtered_std_max_tau=float(sd_f_tau.max()), filtered_std_max_qfc=float(sd_f_qfc.max())),
                rows=rows)


def fictitious_estimate(house=5, accs=(0.25, 0.65, 1.5)):
    """ANALYTIC estimate (not simulated) of what the mocap base leaves out: for a base accel a_b the arm DoFs would feel
    the generalised force -J_com^T m_arm a_b (arm link inertia) -- the sim's mocap base gives the arms no such force.
    Returns per-arm, per-joint |torque| [Nm] at the hold pose for a_b along robot x (forward) and y (lateral)."""
    C = P1.new_hold(house, (3.1, 6.09), 0.0, DT)
    m, d = C.m, C.d
    mujoco.mj_forward(m, d)
    out = {}
    for side, dofs in (("left", P1.B.LEFT_J), ("right", P1.B.RIGHT_J)):
        ids = [m.joint(j).id for j in dofs]
        da = np.array([m.jnt_dofadr[i] for i in ids])
        root = m.body(f"{side}_arm_link1").id
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        mujoco.mj_jacSubtreeCom(m, d, jp, root)
        mass = float(m.body_subtreemass[root])
        for ax, name in ((0, "x_fwd"), (1, "y_lat")):
            e = np.zeros(3); e[ax] = 1.0
            tau1 = -(jp[:, da].T @ e) * mass          # per 1 m/s^2
            for a in accs:
                out[f"{side}_{name}_a{a}"] = np.abs(tau1 * a).round(3).tolist()
        out[f"{side}_mass_kg"] = mass
    return out


# ------------------------------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "one", "sweep", "det", "static", "dist", "fict"])
    ap.add_argument("--route", default="h17")
    ap.add_argument("--amax", type=float, default=0.25)
    ap.add_argument("--vmax", type=float, default=0.3)
    ap.add_argument("--caps", type=float, nargs="*", default=[0.15, 0.25, 0.4])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--clear-min", type=float, default=0.05)
    ap.add_argument("--tau-cap", type=float, default=None)
    ap.add_argument("--no-tilt", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--static-s", type=float, default=10.0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"a{a.amax}_v{a.vmax}"
    if a.cmd == "plan":
        out, F = plan(a.route, a.amax, a.vmax, clear_min=a.clear_min)
        out["door_bodies"] = out["door_obj"]["bodies"]
        js = save_plan(out, a.route, tag)
        js["door_bodies"] = out["door_bodies"]
        (RUNS / f"plan_{a.route}_{tag}.json").write_text(json.dumps(js, indent=1, default=str))
        print(json.dumps({k: v for k, v in js.items() if k not in ("door_u_curve_nonzero", "polyline", "fp")}, indent=1, default=str))
        return
    if a.cmd == "fict":
        print(json.dumps(fictitious_estimate(ROUTES[a.route]["house"]), indent=1))
        return
    if a.cmd == "dist":
        print(json.dumps(disturbance(a.route, tag, range(a.seeds)), indent=1))
        return
    if a.cmd == "static":
        s = _worker((a.route, "static", a.seed, a.tau_cap, not a.no_tilt, dict(static_s=a.static_s, ref_tag=tag)))
        print(json.dumps(s, indent=1, default=str))
        return
    if a.cmd == "one":
        s = _worker((a.route, tag, a.seed, a.tau_cap, not a.no_tilt, dict(max_steps=a.max_steps)))
        print(json.dumps(s, indent=1, default=str))
        return
    if a.cmd == "sweep":
        res = []
        for cap in a.caps:
            tg = f"a{cap}_v{a.vmax}"
            if not (RUNS / f"traj_{a.route}_{tg}.npz").exists():
                out, F = plan(a.route, cap, a.vmax, clear_min=a.clear_min, verbose=False)
                out["door_bodies"] = out["door_obj"]["bodies"]
                js = save_plan(out, a.route, tg)
                js["door_bodies"] = out["door_bodies"]
                (RUNS / f"plan_{a.route}_{tg}.json").write_text(json.dumps(js, indent=1, default=str))
                if not out["route"]:
                    print("no route at cap", cap, out["reason"]); continue
            tasks = [(a.route, tg, s, a.tau_cap, not a.no_tilt, dict(max_steps=a.max_steps)) for s in range(a.seeds)]
            res += _pool(tasks, a.workers)
            ok = [r for r in res if r["tag"] == tg]
            print(f"cap {cap}: ball kept {sum(not r['ball_lost'] for r in ok)}/{len(ok)}", flush=True)
            Path(a.out or RUNS / f"sweep_{a.route}.json").write_text(json.dumps(res, indent=1, default=str))
        return
    if a.cmd == "det":
        tasks = []
        for s in range(a.seeds):
            for rep in range(2):
                tasks.append((a.route, tag, s, a.tau_cap, not a.no_tilt,
                              dict(hash_state=True, snap_at=1500 if rep == 0 else None, max_steps=a.max_steps, save=False)))
        res = _pool(tasks, a.workers)
        rows = []
        for s in range(a.seeds):
            r0, r1 = res[2 * s], res[2 * s + 1]
            rows.append(dict(seed=s, hash_run0=r0["hash"], hash_run1=r1["hash"], same=r0["hash"] == r1["hash"],
                             restore_same=(r0["restore_hash"] == r0["hash"]) if r0["restore_hash"] else None))
        print(json.dumps(rows, indent=1))
        print("bitwise repeat:", sum(r["same"] for r in rows), "/", len(rows),
              " restore:", sum(bool(r["restore_same"]) for r in rows), "/", len(rows))
        Path(a.out or RUNS / f"det_{a.route}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
