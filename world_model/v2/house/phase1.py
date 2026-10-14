"""phase1.py -- Phase 1 acceptance 1-4 (docs/status/molmospaces_integration_plan.md section 4).  SIMULATION only.

    python -m world_model.v2.house.phase1 [--house 2] [--out results/phase1.json]

 1. spawn: zero contacts between YOR (+tray, +ball) and house geoms (+ min distance, since a mocap base is static and
    never produces contacts against static house geoms).
 2. FL-MPC (mpc_fl, UNMODIFIED) torque on the composed model == torque on scene_yor_canonical for the same arm state.
 3. 5 s static carry (FL-MPC holding the hold pose, tray welded between the hands): ball stays, zero arm-arm and
    arm-house contacts, step time vs 1.5 ms.
 4. snapshot/restore bitwise (mj_getState/mj_setState incl. mocap + warmstart), physics-only and with FL-MPC in loop.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

from . import build as B

HERE = Path(__file__).resolve().parent
REPO = B.REPO
if str(REPO / "experiments" / "ab_smooth_vs_bangbang") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments" / "ab_smooth_vs_bangbang"))

FL_KW = dict(NC=4, L=8, W_POS_EGO=3000.0, W_VEL_EGO=15.0)     # same as latent_hj.scenario.FL_KW


# ------------------------------------------------------------------------------------------ contact classification
def geom_class(m):
    """geom -> 'L','R' (arm links+fingers), 'B' (rest of YOR: base/lift/wheels), 'T' tray, 'P' ball, 'H' house."""
    cls = np.empty(m.ngeom, dtype="<U1")
    root = m.body("yor_root").id
    bcls = {}
    for b in range(m.nbody):
        nm = m.body(b).name or ""
        x = b
        under = False
        while x > 0:
            if x == root:
                under = True
                break
            x = int(m.body_parentid[x])
        if nm == "tray":
            bcls[b] = "T"
        elif nm == "ball":
            bcls[b] = "P"
        elif under:
            bcls[b] = "L" if nm.startswith("left_arm_") else ("R" if nm.startswith("right_arm_") else "B")
        else:
            bcls[b] = "H"
    for g in range(m.ngeom):
        cls[g] = bcls[int(m.geom_bodyid[g])]
    return cls


def contact_counts(m, d, cls):
    """dict pair-class -> count, over active contacts (penetrating or within margin)."""
    out = {}
    for k in range(d.ncon):
        c = d.contact[k]
        a, b = sorted((cls[c.geom1], cls[c.geom2]))
        key = a + b
        out[key] = out.get(key, 0) + 1
    return out


def robot_house_min_distance(m, d, cls, radius=3.0, distmax=3.0):
    """Min signed distance between any YOR/tray/ball geom and any house geom (mj_geomDistance), house geoms
    pre-filtered to a sphere of `radius` around the robot root. Returns (min_dist, robot_geom_name, house_geom_name)."""
    root = d.mocap_pos[0].copy()
    own = [g for g in range(m.ngeom) if cls[g] != "H" and (m.geom_contype[g] or m.geom_conaffinity[g])]
    # fromto result buffer
    ft = np.zeros(6)
    hs = [g for g in range(m.ngeom) if cls[g] == "H" and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE   # floor touches the wheels by design
          and np.linalg.norm(d.geom_xpos[g, :2] - root[:2]) < radius + float(min(m.geom_rbound[g], 4.0))
          and (m.geom_contype[g] or m.geom_conaffinity[g])]
    best = (distmax, "", "")
    for gi in own:
        for hj in hs:
            dist = mujoco.mj_geomDistance(m, d, gi, hj, distmax, ft)
            if dist < distmax and abs(np.linalg.norm(ft[3:] - ft[:3]) - abs(dist)) > 1e-4:
                continue                      # mesh-mesh pairs beyond distmax can return a spurious 0
            if dist < best[0]:
                best = (float(dist), m.geom(gi).name or f"g{gi}", m.geom(hj).name or f"g{hj}")
    return best


# ------------------------------------------------------------------------------------------ controller plumbing
class Ctl:
    """Thin driver around the unmodified FLMPC: holds q_hold for both arms, applies qfrc_applied."""

    def __init__(self, C, dt_ctrl=None):
        import mpc_fl
        import sim_common as sc
        self.sc, self.m = sc, C.m
        self.lj, self.lq, self.ld = sc.joint_addrs(C.m, sc.LEFT_ARM_JOINTS)
        self.rj, self.rq, self.rd = sc.joint_addrs(C.m, sc.RIGHT_ARM_JOINTS)
        kw = dict(FL_KW)
        if dt_ctrl is not None:
            kw["DT"] = dt_ctrl
        # FL-MPC implementation: 'py' (mpc_fl.FLMPC, default, unchanged) or 'rt' (RTFLMPC, exact Goldfarb-Idnani QP) via
        # env BIMANUAL_FLMPC_IMPL; the read-only monitor records feasible/fallback/table-row violations (qp_stats()).
        from ..qp_safety import QPRecorder, StepMonitor, make_flmpc
        self.fl = make_flmpc(C.m, **kw)
        self.qpmon = StepMonitor.attach(self.fl)
        if self.fl.impl == "py":
            QPRecorder.install(sc)
        NP = self.fl.NP
        self.qe_ref = np.tile(np.array(C.info["hold_q"]["left"]), (NP + 1, 1))
        self.qi_ref = np.tile(np.array(C.info["hold_q"]["right"]), (NP + 1, 1))
        self.z = np.zeros((NP + 1, 6))
        self.n_infeasible = 0

    def tau(self, d):
        # mj_forward first: after mj_step, d.qM / qfrc_bias / xpos are the PRE-integration values (one step stale) and
        # after mj_setState they are not restored at all; FLMPC reads them directly.  Forward makes the call a pure
        # function of (qpos, qvel, warmstart) -> bitwise restorable.
        mujoco.mj_forward(self.m, d)
        pairs = self.sc.all_pair_barriers(self.m, d, self.ld, self.rd)
        te, ti, ok = self.fl.step(self.m, d, self.lq, self.ld, self.rq, self.rd,
                                  self.qe_ref, self.z, self.qi_ref, self.z, pairs)
        if not ok:
            self.n_infeasible += 1
        return te, ti

    def qp_stats(self):
        """Read-only solver/constraint report: this controller's StepMonitor (+ the process-wide py-path QPRecorder)."""
        from ..qp_safety import QPRecorder
        out = self.qpmon.summary()
        if self.fl.impl == "py":
            out["py_qp"] = QPRecorder.summary()
        return out

    def state(self):
        return (self.fl.a_ego_prev.copy(), self.fl.a_int_prev.copy())

    def set_state(self, s):
        self.fl.a_ego_prev, self.fl.a_int_prev = s[0].copy(), s[1].copy()


def new_hold(house, spawn, yaw, dt, with_tray=True, gravity=B.CANON_GRAVITY):
    C = B.compose(house, spawn, yaw=yaw, with_tray=with_tray, dt=dt, gravity=gravity)
    B.set_hold_state(C)
    B.disable_arm_servos(C.m)
    fin = C.info["finger"]
    for side in ("left", "right"):
        C.d.ctrl[C.m.actuator(f"{side}_arm_joint7_pos").id] = fin
        C.d.ctrl[C.m.actuator(f"{side}_arm_joint8_pos").id] = -fin
    mujoco.mj_forward(C.m, C.d)
    return C


# ------------------------------------------------------------------------------------------ footprint
def footprint(C):
    """Axis-aligned extents (robot frame, m) of all YOR collision geoms (+tray) in the hold pose, from geom AABBs
    (conservative).  y = width across the travel direction (x forward)."""
    m, d = C.m, C.d
    cls = geom_class(m)
    yaw = C.info["yaw"]
    R = B._rotz(yaw)
    base = np.array([*C.info["spawn"], 0.0])
    lo = np.full(3, 1e9); hi = np.full(3, -1e9)
    for g in range(m.ngeom):
        if cls[g] in ("H",) or not (m.geom_contype[g] or m.geom_conaffinity[g]) or m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        c = R.T @ (d.geom_xpos[g] - base)
        Rg = R.T @ d.geom_xmat[g].reshape(3, 3)
        half = np.abs(Rg) @ m.geom_aabb[g, 3:]
        cc = c + Rg @ m.geom_aabb[g, :3]
        lo = np.minimum(lo, cc - half); hi = np.maximum(hi, cc + half)
    return dict(x=[float(lo[0]), float(hi[0])], y=[float(lo[1]), float(hi[1])], z=[float(lo[2]), float(hi[2])],
                width_y_m=float(hi[1] - lo[1]), depth_x_m=float(hi[0] - lo[0]))


# ------------------------------------------------------------------------------------------ acceptance 1
def acc1(C):
    m, d = C.m, C.d
    cls = geom_class(m)
    mujoco.mj_forward(m, d)
    cc = contact_counts(m, d, cls)
    house_con = {k: v for k, v in cc.items() if "H" in k}
    md = robot_house_min_distance(m, d, cls)
    return dict(ncon_total=int(d.ncon), pair_counts=cc, contacts_with_house=int(sum(house_con.values())),
                min_robot_house_dist_m=md[0], nearest=md[1:], passed=bool(sum(house_con.values()) == 0 and md[0] >= 0.0))


def acc1_arm_filter(C):
    """Do arm geoms actually collide with walls and furniture?  (a) bit-logic over ALL house collision geoms;
    (b) physical: shift the mocap base so a finger geom sits inside a chosen house geom, look for the contact."""
    m, d = C.m, C.d
    cls = geom_class(m)
    res = {}
    hg = [g for g in range(m.ngeom) if cls[g] == "H" and (m.geom_contype[g] or m.geom_conaffinity[g])]
    for side, c in (("left", "L"), ("right", "R")):
        ag = [g for g in range(m.ngeom) if cls[g] == c and (m.geom_contype[g] or m.geom_conaffinity[g])]
        act = np.bitwise_or.reduce(m.geom_contype[ag]); aca = np.bitwise_or.reduce(m.geom_conaffinity[ag])
        ok = [bool((m.geom_contype[h] & aca) or (act & m.geom_conaffinity[h])) for h in hg]
        res[f"{side}_bits_pass_frac"] = float(np.mean(ok)); res["n_house_collision_geoms"] = len(hg)
    # physical probes: one house geom per distinct (group, contype, conaffinity, geom type); shift the mocap base so a
    # collidable finger geom of the arm under test sits at the probe geom's centre, then look for ANY contact
    # between that probe and ANY geom of that arm.
    probes, seen = [], {}
    for g in hg:
        key = (int(m.geom_group[g]), int(m.geom_contype[g]), int(m.geom_conaffinity[g]), int(m.geom_type[g]))
        if key not in seen:
            seen[key] = g
            probes.append(g)
    hits = {"left": 0, "right": 0}
    tot = 0
    base0 = d.mocap_pos[0].copy()
    for g in probes:
        for side, c in (("left", "L"), ("right", "R")):
            tip = m.body(f"{side}_arm_link8").id
            gt = next(x for x in range(m.ngeom) if m.geom_bodyid[x] == tip and cls[x] == c and (m.geom_contype[x] or m.geom_conaffinity[x]))
            d.mocap_pos[0] = base0
            mujoco.mj_forward(m, d)
            d.mocap_pos[0] = base0 + (d.geom_xpos[g] - d.geom_xpos[gt])
            mujoco.mj_forward(m, d)
            hit = any((d.contact[k].geom1 == g and cls[d.contact[k].geom2] == c) or
                      (d.contact[k].geom2 == g and cls[d.contact[k].geom1] == c) for k in range(d.ncon))
            hits[side] += int(hit)
            if not hit:
                res.setdefault(f"probe_miss_{side}", []).append([m.geom(g).name[:40], list(k for k, v in seen.items() if v == g)[0]])
        tot += 1
    d.mocap_pos[0] = base0
    mujoco.mj_forward(m, d)
    res["probes"] = tot
    res["probe_hits_left"], res["probe_hits_right"] = hits["left"], hits["right"]
    res["probe_keys(group,contype,conaff,type)"] = sorted(seen)
    return res


# ------------------------------------------------------------------------------------------ acceptance 2
def acc2(C_house_kw, n_states=6, seed=0):
    import sim_common as sc
    out = {}
    canon = sc.load_model()
    dc = mujoco.MjData(canon)
    variants = {
        "composed_notray_g981": dict(with_tray=False, gravity=B.CANON_GRAVITY),
        "composed_tray_weld_g981": dict(with_tray=True, gravity=B.CANON_GRAVITY),
        "composed_notray_house_g98": dict(with_tray=False, gravity=None),
    }
    import mpc_fl
    rng = np.random.default_rng(seed)
    _, clq, cld = sc.joint_addrs(canon, sc.LEFT_ARM_JOINTS)
    _, crq, crd = sc.joint_addrs(canon, sc.RIGHT_ARM_JOINTS)
    flc = mpc_fl.FLMPC(canon, **FL_KW)
    NP = flc.NP
    states = []
    for _ in range(n_states):
        states.append(dict(ql=rng.normal(0, 0.4, 6) + np.array([0, 1.5, -0.6, 0, -0.9, 0.78]),
                           qr=rng.normal(0, 0.4, 6) + np.array([0, 1.5, -0.6, 0, -0.9, -0.78]),
                           vl=rng.normal(0, 0.6, 6), vr=rng.normal(0, 0.6, 6),
                           dql=rng.normal(0, 0.05, 6), dqr=rng.normal(0, 0.1, 6),
                           ap=(rng.normal(0, 1, 6), rng.normal(0, 1, 6))))
    z = np.zeros((NP + 1, 6))

    def run(m, d, fl, lq, ld, rq, rd, s):
        d.qpos[lq] = s["ql"]; d.qpos[rq] = s["qr"]; d.qvel[ld] = s["vl"]; d.qvel[rd] = s["vr"]
        for sd in ("left", "right"):        # finger opening is part of the arm state (changes M slightly): identical in both
            d.qpos[m.jnt_qposadr[m.joint(f"{sd}_arm_joint7").id]] = 0.0175
            d.qpos[m.jnt_qposadr[m.joint(f"{sd}_arm_joint8").id]] = -0.0175
        mujoco.mj_forward(m, d)
        fl.a_ego_prev, fl.a_int_prev = s["ap"][0].copy(), s["ap"][1].copy()
        qer = np.tile(d.qpos[lq] + s["dql"], (NP + 1, 1)); qir = np.tile(d.qpos[rq] + s["dqr"], (NP + 1, 1))
        pairs = sc.all_pair_barriers(m, d, ld, rd)
        te, ti, ok = fl.step(m, d, lq, ld, rq, rd, qer, z, qir, z, pairs)
        return te, ti, fl.last_M_e.copy(), fl.last_c_e.copy(), fl.last_M_i.copy(), fl.last_c_i.copy(), ok, len(pairs)

    ref = [run(canon, dc, flc, clq, cld, crq, crd, s) for s in states]
    for name, kw in variants.items():
        C = B.compose(**C_house_kw, **kw)
        if kw["with_tray"]:
            B.set_hold_state(C)
        m, d = C.m, C.d
        _, lq, ld = sc.joint_addrs(m, sc.LEFT_ARM_JOINTS)
        _, rq, rd = sc.joint_addrs(m, sc.RIGHT_ARM_JOINTS)
        fl = mpc_fl.FLMPC(m, **FL_KW)
        dmax = dict(tau_e=0.0, tau_i=0.0, M=0.0, c=0.0)
        nfe = 0
        for s, r in zip(states, ref):
            # tray variant: arm state is overwritten by the test state; weld off so constraint forces cannot enter
            if kw["with_tray"]:
                d.eq_active[:] = 0
            o = run(m, d, fl, lq, ld, rq, rd, s)
            dmax["tau_e"] = max(dmax["tau_e"], float(np.abs(o[0] - r[0]).max()))
            dmax["tau_i"] = max(dmax["tau_i"], float(np.abs(o[1] - r[1]).max()))
            dmax["M"] = max(dmax["M"], float(np.abs(o[2] - r[2]).max()), float(np.abs(o[4] - r[4]).max()))
            dmax["c"] = max(dmax["c"], float(np.abs(o[3] - r[3]).max()), float(np.abs(o[5] - r[5]).max()))
            nfe += int(o[6]) != int(r[6])
        dmax["feasible_flag_mismatches"] = nfe
        dmax["max_abs_tau_ref"] = float(max(np.abs(r[0]).max() for r in ref))
        dmax["passed_1e-6"] = bool(max(dmax["tau_e"], dmax["tau_i"]) < 1e-6)
        out[name] = dmax
    out["n_states"] = n_states
    return out


# ------------------------------------------------------------------------------------------ acceptance 3
def carry(C, seconds=5.0, ctl_dt=None, record_every=1):
    m, d = C.m, C.d
    cls = geom_class(m)
    ctl = Ctl(C, dt_ctrl=ctl_dt)
    n = int(round(seconds / m.opt.timestep))
    tray = m.body("tray").id
    ball = m.body("ball").id
    ts_phys, ts_ctl = [], []
    pair_tot = {}
    maxoff = 0.0; min_bz = 1e9; max_tilt = 0.0; max_dq = 0.0; max_tray_drift = 0.0
    q_hold = np.concatenate([C.info["hold_q"]["left"], C.info["hold_q"]["right"]])
    tray0 = d.xpos[tray].copy()
    lost_at = None
    for k in range(n):
        t0 = time.perf_counter()
        te, ti = ctl.tau(d)
        t1 = time.perf_counter()
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[ctl.ld] = te
        d.qfrc_applied[ctl.rd] = ti
        mujoco.mj_step(m, d)
        t2 = time.perf_counter()
        ts_ctl.append((t1 - t0) * 1e3); ts_phys.append((t2 - t1) * 1e3)
        cc = contact_counts(m, d, cls)
        for kk, v in cc.items():
            pair_tot[kk] = pair_tot.get(kk, 0) + v
        R = d.xmat[tray].reshape(3, 3)
        rel = R.T @ (d.xpos[ball] - d.xpos[tray])
        maxoff = max(maxoff, float(np.hypot(rel[0], rel[1])))
        min_bz = min(min_bz, float(rel[2]))
        max_tilt = max(max_tilt, float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))))
        q = np.concatenate([d.qpos[ctl.lq], d.qpos[ctl.rq]])
        max_dq = max(max_dq, float(np.abs(q - q_hold).max()))
        max_tray_drift = max(max_tray_drift, float(np.linalg.norm(d.xpos[tray] - tray0)))
        if lost_at is None and (abs(rel[0]) > 0.15 or abs(rel[1]) > 0.10 or rel[2] < 0.0):
            lost_at = k * m.opt.timestep
    p = np.array(ts_phys)
    c = np.array(ts_ctl)
    w = slice(len(p) // 10, None)       # drop warm-up (first 10 %) for timing stats
    return dict(
        dt=float(m.opt.timestep), steps=n, seconds=seconds, ctl_DT=ctl.fl.DT,
        phys_ms_median=float(np.median(p[w])), phys_ms_mean=float(p[w].mean()), phys_ms_p95=float(np.percentile(p[w], 95)),
        phys_ms_max=float(p[w].max()), ctl_ms_median=float(np.median(c[w])), ctl_ms_p95=float(np.percentile(c[w], 95)),
        ball_max_xy_offset_m=maxoff, ball_min_z_in_tray_m=min_bz, tray_max_tilt_deg=max_tilt,
        max_joint_drift_rad=max_dq, tray_max_drift_m=max_tray_drift, ball_lost_at_s=lost_at,
        contact_step_counts=pair_tot, n_infeasible=ctl.n_infeasible,
        passed_ball=bool(lost_at is None),
        passed_no_armarm=bool(pair_tot.get("LR", 0) == 0),
        passed_no_armhouse=bool(pair_tot.get("HL", 0) + pair_tot.get("HR", 0) == 0),
        passed_time_1p5ms=bool(np.median(p[w]) < 1.5),
    )


# ------------------------------------------------------------------------------------------ acceptance 4
def _bits(d, m):
    return (d.qpos.copy(), d.qvel.copy(), d.qacc_warmstart.copy(), d.mocap_pos.copy(), d.mocap_quat.copy(),
            d.qfrc_constraint.copy(), int(d.ncon), np.array([d.contact[k].dist for k in range(d.ncon)]),
            d.eq_active.copy(), float(d.time))


def _same(a, b):
    return all(np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y for x, y in zip(a, b))


def acc4(C, n_pre=100, n_post=200, with_ctl=True):
    m, d = C.m, C.d
    ctl = Ctl(C) if with_ctl else None
    ld = np.array([m.jnt_dofadr[m.joint(f"left_arm_joint{i}").id] for i in range(1, 7)])
    rd = np.array([m.jnt_dofadr[m.joint(f"right_arm_joint{i}").id] for i in range(1, 7)])
    base0 = d.mocap_pos[0].copy()
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    q_hold = np.concatenate([C.info["hold_q"]["left"], C.info["hold_q"]["right"]])
    qa = np.concatenate([[m.jnt_qposadr[m.joint(f"left_arm_joint{i}").id] for i in range(1, 7)],
                         [m.jnt_qposadr[m.joint(f"right_arm_joint{i}").id] for i in range(1, 7)]])

    def drive(dd, k):
        dd.mocap_pos[0] = base0 + np.array([0.05 * np.sin(0.05 * k), 0.03 * np.sin(0.031 * k), 0.0])  # base wiggle
        if ctl is not None:
            te, ti = ctl.tau(dd)
        else:                               # deterministic PD hold + a time-varying kick so dynamics are not trivial
            kick = 1.0 * np.sin(0.07 * k) * np.ones(12)
            tq = 20.0 * (q_hold - dd.qpos[qa]) - 2.0 * dd.qvel[np.concatenate([ld, rd])] + kick
            te, ti = tq[:6], tq[6:]
        dd.qfrc_applied[:] = 0.0
        dd.qfrc_applied[ld] = te
        dd.qfrc_applied[rd] = ti
        mujoco.mj_step(m, dd)

    for k in range(n_pre):
        drive(d, k)
    snap = np.zeros(mujoco.mj_stateSize(m, spec))
    mujoco.mj_getState(m, d, snap, spec)
    ctl_state = ctl.state() if ctl else None
    for k in range(n_pre, n_pre + n_post):
        drive(d, k)
    ref = _bits(d, m)                                      # straight run, step n_pre+n_post

    res = {}
    # (a) restore into the SAME data after the run
    mujoco.mj_setState(m, d, snap, spec)
    if ctl:
        ctl.set_state(ctl_state)
    for k in range(n_pre, n_pre + n_post):
        drive(d, k)
    res["same_data_bitwise"] = bool(_same(ref, _bits(d, m)))
    # (b) restore into a FRESH data
    d2 = mujoco.MjData(m)
    mujoco.mj_setState(m, d2, snap, spec)
    if ctl:
        ctl.set_state(ctl_state)
    for k in range(n_pre, n_pre + n_post):
        drive(d2, k)
    res["fresh_data_bitwise"] = bool(_same(ref, _bits(d2, m)))
    res["max_abs_qpos_diff_fresh"] = float(np.abs(ref[0] - d2.qpos).max())
    res["max_abs_qvel_diff_fresh"] = float(np.abs(ref[1] - d2.qvel).max())
    res["steps"] = n_post
    # guard: MuJoCo silently resets on instability (warning only) -> time would not equal the expected value
    res["time_ok"] = bool(abs(ref[-1] - (n_pre + n_post) * m.opt.timestep) < 1e-9 and abs(d2.time - ref[-1]) < 1e-12)
    res["qpos_moved_from_start_rad"] = float(np.abs(ref[0][qa] - q_hold).max())
    res["with_fl_mpc"] = bool(with_ctl)
    res["state_size"] = int(len(snap))
    return res


def sweep1(out):
    p0 = json.loads((HERE / "results" / "phase0.json").read_text())["rows"]
    rows = []
    for r in p0:
        t = time.time()
        i = r["house"]
        cs = B.find_clear_spots(i, top=1)[0]          # re-picked with the low (3 cm) ray height included
        row = dict(house=i, spot=[cs["x"], cs["y"]], free_r=cs["free_r"])
        try:
            C = new_hold(i, (cs["x"], cs["y"]), 0.3, dt=0.005)
            a1 = acc1(C)
            row.update(ik_pairs=C.info["ik_pairs_tried"], contacts_with_house=a1["contacts_with_house"],
                       ncon_total=a1["ncon_total"], min_dist=a1["min_robot_house_dist_m"], nearest=a1["nearest"],
                       footprint=footprint(C), passed=a1["passed"])
        except Exception as e:
            row["error"] = repr(e)[:200]
        row["s"] = time.time() - t
        rows.append(row)
        print(row, flush=True)
        Path(out).write_text(json.dumps(rows, indent=1, default=str))


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--house", type=int, default=2)
    ap.add_argument("--x", type=float, default=None)
    ap.add_argument("--y", type=float, default=None)
    ap.add_argument("--yaw", type=float, default=0.3)
    ap.add_argument("--out", default=str(HERE / "results" / "phase1.json"))
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--sweep1", action="store_true", help="acceptance 1 (+footprint) in all 20 val houses at their clear spot")
    a = ap.parse_args()
    if a.sweep1:
        return sweep1(a.out)
    if a.x is None:
        spot = B.find_clear_spots(a.house, top=1)[0]
        a.x, a.y = spot["x"], spot["y"]
        print("clear spot", spot, flush=True)
    kw = dict(idx=a.house, spawn_xy=(a.x, a.y), yaw=a.yaw)
    res = dict(house=a.house, spawn=[a.x, a.y], yaw=a.yaw, load_avg_start=os.getloadavg())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    def save():
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))

    want = lambda k: (not a.only) or (k in a.only)
    C = None
    if want("1") or want("3") or want("4"):
        t = time.time()
        C = new_hold(a.house, (a.x, a.y), a.yaw, dt=0.005)
        res["compose"] = dict(C.info, build_s=time.time() - t)
        print("composed", {k: C.info[k] for k in ("nbody", "ngeom", "nv", "nq", "nu", "model_hash")}, flush=True)
    if want("1"):
        res["acc1"] = acc1(C); res["footprint_hold"] = footprint(C); print("acc1", res["acc1"], res["footprint_hold"], flush=True)
        Cn = B.compose(**kw, with_tray=False, dt=0.005)
        B.set_hold_state(Cn)
        res["acc1_no_tray"] = acc1(Cn)
        res["acc1_arm_filter"] = acc1_arm_filter(Cn); print("filter", res["acc1_arm_filter"], flush=True)
        save()
    if want("2"):
        res["acc2"] = acc2(dict(idx=a.house, spawn_xy=(a.x, a.y), yaw=a.yaw, dt=0.005)); print("acc2", res["acc2"], flush=True)
        save()
    if want("3"):
        res["acc3_dt0.005"] = carry(C, 5.0); print("acc3 dt.005", res["acc3_dt0.005"], flush=True)
        save()
        C2 = new_hold(a.house, (a.x, a.y), a.yaw, dt=0.002)
        res["acc3_dt0.002"] = carry(C2, 5.0, ctl_dt=0.002); print("acc3 dt.002", res["acc3_dt0.002"], flush=True)
        save()
    if want("4"):
        C4 = new_hold(a.house, (a.x, a.y), a.yaw, dt=0.005)
        res["acc4_physics"] = acc4(C4, with_ctl=False); print("acc4 phys", res["acc4_physics"], flush=True)
        C5 = new_hold(a.house, (a.x, a.y), a.yaw, dt=0.005)
        res["acc4_flmpc"] = acc4(C5, with_ctl=True); print("acc4 fl", res["acc4_flmpc"], flush=True)
        save()
    res["load_avg_end"] = os.getloadavg()
    save()


if __name__ == "__main__":
    main()
