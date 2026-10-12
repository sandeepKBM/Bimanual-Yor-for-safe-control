"""execute.py -- carry the tray to the chosen spot with the unmodified FL-MPC (phase2.Ctl2) while the holonomic mocap base follows the planned path;
kinematic replay of the naive straight-line counterfactual; true (MuJoCo geom) clearance + contact monitors.  SIMULATION ONLY.

The base is kinematic (as in phase2): base acceleration reaches the arms only through the two tray welds; FL-MPC regulates both arms to the hold pose and
keeps the tray tilt inside the cup cone.  The tray path = the base path (the tray is rigid in the hands)."""
from __future__ import annotations

import math
import time

import numpy as np
import mujoco

from world_model.v2.house import build as B
from world_model.v2.house import nav as N
from world_model.v2.house import phase1 as P1
from world_model.v2.house import phase2 as P2
from world_model.v2.house.office import compose as CP
from . import perceive as PC

DT = 0.005
SETTLE_S = 0.6
HOLD_END_S = 1.2
FRAME_EVERY = 8                     # 8 * 5 ms = 40 ms = 25 fps


def make_traj(path_t, base0, yaw0, vmax=0.3, amax=0.25, dt=DT):
    """world path of the base origin from the path offsets t (n,2) + start; time-parameterised with nav.time_parameterize; yaw held (holonomic)."""
    Q = np.asarray(path_t, float) + np.asarray(base0, float)[None]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    keep = np.concatenate([[True], np.diff(s) > 1e-6])
    Q, s = Q[keep], s[keep]
    tr = N.time_parameterize(Q, s, vmax=vmax, amax=amax, dt=dt)
    tr.yaw = np.full_like(tr.x, yaw0)
    return tr


class ClearMon:
    """true clearance: min mj_geomDistance between every robot geom (arms, hands, tray, ball, chassis) and every collidable house geom, with the
    argmin and the per-object (GT key) minimum.  Floor planes are excluded (the wheels stand on it by design)."""

    def __init__(self, C, keys):
        m = C.m
        self.m = m
        self.cls = P1.geom_class(m)
        col = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
        notplane = m.geom_type != mujoco.mjtGeom.mjGEOM_PLANE
        self.own = [g for g in range(m.ngeom) if self.cls[g] in ("L", "R", "B", "T", "P") and col[g] and notplane[g]]
        self.house = np.array([g for g in range(m.ngeom) if self.cls[g] == "H" and col[g] and notplane[g]])
        self.keys = keys
        self.ft = np.zeros(6)
        self.obj_min = {}
        self.floor = {g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE}

    def _gd(self, d, a, b, dmax):
        v = mujoco.mj_geomDistance(self.m, d, a, b, dmax, self.ft)
        if v < dmax and abs(np.linalg.norm(self.ft[3:] - self.ft[:3]) - abs(v)) > 1e-4 and v >= 0:
            return dmax
        return v

    def measure(self, d, dmax=0.4, track=True):
        m = self.m
        base = d.mocap_pos[0]
        hx = d.geom_xpos[self.house]
        hr = np.minimum(m.geom_rbound[self.house], 6.0)
        near = self.house[np.linalg.norm(hx[:, :2] - base[:2], axis=1) - hr < 1.6 + dmax]
        best, bn = dmax, ""
        for a in self.own:
            pa, ra = d.geom_xpos[a], m.geom_rbound[a]
            sel = near[np.linalg.norm(d.geom_xpos[near] - pa, axis=1) - np.minimum(m.geom_rbound[near], 6.0) - ra < dmax]
            for b in sel:
                v = self._gd(d, a, int(b), dmax)
                if track:
                    k = str(self.keys[int(b)])
                    if v < self.obj_min.get(k, 9.0):
                        self.obj_min[k] = float(v)
                if v < best:
                    best, bn = v, f"{self.cls[a]}:{m.geom(a).name or a}|{m.geom(int(b)).name or int(b)}"
        return float(best), bn

    def contacts(self, d):
        """number of active contacts between a robot geom and a house geom (floor excluded) + the pair names."""
        out = []
        for c in range(d.ncon):
            cc = d.contact[c]
            g1, g2 = int(cc.geom1), int(cc.geom2)
            if g1 in self.floor or g2 in self.floor:
                continue
            a, b = self.cls[g1], self.cls[g2]
            if (a == "H") != (b == "H"):
                out.append((self.m.geom(g1).name or str(g1), self.m.geom(g2).name or str(g2)))
        return out


def snapshot(d):
    return dict(qpos=d.qpos.copy(), mocap_pos=d.mocap_pos.copy(), mocap_quat=d.mocap_quat.copy())


def restore(m, d, s):
    d.qpos[:] = s["qpos"]
    d.mocap_pos[:] = s["mocap_pos"]
    d.mocap_quat[:] = s["mocap_quat"]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)


def run_tracking(C, traj, keys, seed=0, dist_every=10, progress=None):
    """FL-MPC carry along traj (mocap base).  Returns (summary, frames) ; frames = list of dict(snapshot, t, info) every FRAME_EVERY steps."""
    from world_model.v2.testbeds.b_tray_ball import tray_env as TE
    m, d = C.m, C.d
    rng = np.random.default_rng(seed)
    ctl = P2.Ctl2(C, tilt=True)
    mon = ClearMon(C, keys)
    tray, ball = m.body("tray").id, m.body("ball").id
    ld, rd = ctl.ld, ctl.rd
    dn = mujoco.MjData(m)
    n_set, n_t, n_end = int(SETTLE_S / DT), len(traj.t), int(HOLD_END_S / DT)
    N_ = n_set + n_t + n_end
    frames = []
    tc = np.zeros(N_)
    tau_peak = np.zeros(12)
    tilt_max, lost_at, n_inf = 0.0, None, 0
    contacts = {}
    dmin, dmin_name, dmin_t = 9.0, "", 0.0
    trace_t, trace_d, trace_tray = [], [], []
    last_d = (9.0, "")
    for k in range(N_):
        kk = min(max(k - n_set, 0), n_t - 1)
        d.mocap_pos[0] = (traj.x[kk], traj.y[kk], 0.0)
        d.mocap_quat[0] = B.yaw_quat(traj.yaw[kk])
        mujoco.mj_copyData(dn, m, d)
        dn.qpos[ctl.lq] += rng.normal(0, P2.ENC_Q, 6)
        dn.qpos[ctl.rq] += rng.normal(0, P2.ENC_Q, 6)
        dn.qvel[ld] += rng.normal(0, P2.ENC_V, 6)
        dn.qvel[rd] += rng.normal(0, P2.ENC_V, 6)
        t0 = time.perf_counter()
        te, ti = ctl.tau(dn)
        tc[k] = (time.perf_counter() - t0) * 1e3
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[ld] = te
        d.qfrc_applied[rd] = ti
        mujoco.mj_step(m, d)
        tau_peak = np.maximum(tau_peak, np.abs(np.concatenate([te, ti])))
        for (a, b) in mon.contacts(d):
            contacts[f"{a}|{b}"] = contacts.get(f"{a}|{b}", 0) + 1
        rel, tilt = P2.tray_stats(m, d, tray, ball)
        tilt_max = max(tilt_max, tilt)
        if lost_at is None and (abs(rel[0]) > TE.HX or abs(rel[1]) > TE.HY or rel[2] < 0.0):
            lost_at = (k - n_set) * DT
        if k % dist_every == 0:
            dd, nm = mon.measure(d)
            last_d = (dd, nm)
            trace_t.append((k - n_set) * DT)
            trace_d.append(dd)
            if dd < dmin:
                dmin, dmin_name, dmin_t = dd, nm, (k - n_set) * DT
        if k % FRAME_EVERY == 0:
            tr = d.xpos[tray].copy()
            frames.append(dict(snap=snapshot(d), t=(k - n_set) * DT, d=last_d[0], d_name=last_d[1], tilt=float(tilt), tray=tr, k=k,
                               ncon=int(sum(contacts.values())), ball_ok=lost_at is None))
        if progress and k % 400 == 0:
            progress(k, N_)
    summ = dict(duration_s=float(N_ * DT), traj_s=float(traj.t[-1]), n_steps=int(N_), min_clearance_m=float(dmin), min_clearance_pair=dmin_name,
                min_clearance_t=float(dmin_t), contacts=contacts, n_contacts_total=int(sum(contacts.values())), ball_lost=lost_at is not None,
                ball_lost_at_s=lost_at, tray_tilt_max_deg=float(tilt_max), n_infeasible=int(ctl.n_infeasible), ctl_ms_median=float(np.median(tc)),
                ctl_ms_p95=float(np.percentile(tc, 95)), tau_peak_max_Nm=float(tau_peak.max()),
                per_object_min_m={k: round(v, 4) for k, v in sorted(mon.obj_min.items()) if v < 0.4},
                final_tray_xy=[float(d.xpos[tray][0]), float(d.xpos[tray][1])], final_tray_z=float(d.xpos[tray][2]),
                trace_t=trace_t, trace_d=[float(x) for x in trace_d])
    return summ, frames


def replay_kinematic(C, traj, keys, dist_every=4):
    """naive-plan counterfactual: drive the base along `traj` with the hold pose frozen (set_base_pose carries the tray rigidly); NO controller -- this is a
    geometric replay of what the placement WOULD touch.  Returns (summary, frames)."""
    m, d = C.m, C.d
    mon = ClearMon(C, keys)
    frames = []
    contacts = {}
    dmin, dmin_name, dmin_t = 9.0, "", 0.0
    pen_pairs = {}
    trace_t, trace_d = [], []
    n = len(traj.t)
    last = (9.0, "")
    for k in range(0, n, dist_every):
        CP.set_base_pose(C, float(traj.x[k]), float(traj.y[k]), float(traj.yaw[k]))
        mujoco.mj_forward(m, d)
        dd, nm = mon.measure(d, dmax=0.4)
        last = (dd, nm)
        trace_t.append(float(traj.t[k]))
        trace_d.append(dd)
        if dd < dmin:
            dmin, dmin_name, dmin_t = dd, nm, float(traj.t[k])
        for (a, b) in mon.contacts(d):
            contacts[f"{a}|{b}"] = contacts.get(f"{a}|{b}", 0) + 1
        if dd < 0:
            pen_pairs[nm] = min(pen_pairs.get(nm, 0.0), dd)
        if k % FRAME_EVERY == 0 or dist_every >= FRAME_EVERY:
            frames.append(dict(snap=snapshot(d), t=float(traj.t[k]), d=dd, d_name=nm, tilt=0.0, tray=d.xpos[m.body("tray").id].copy(), k=k,
                               ncon=int(sum(contacts.values())), ball_ok=True))
    summ = dict(min_clearance_m=float(dmin), min_clearance_pair=dmin_name, min_clearance_t=float(dmin_t), n_contacts_total=int(sum(contacts.values())),
                contacts=contacts, penetrating_pairs={k: round(v, 4) for k, v in pen_pairs.items()}, collides=bool(dmin < 0),
                per_object_min_m={k: round(v, 4) for k, v in sorted(mon.obj_min.items()) if v < 0.4}, trace_t=trace_t, trace_d=trace_d)
    return summ, frames
