"""pan/under_rim.py -- two-claw lift via the proposal's fallback, adapted: SLIDE the covered pan so its heavy far side overhangs the
support edge, then the second claw clamps the overhanging rim (one finger under the pan bottom, one on the lid) and both arms lift.
SIMULATION ONLY.  Same physics/driver as pan/handle_grasp.py (FL-MPC NV=7, torque via qfrc_applied, real friction, no weld/teleport).

Why slide instead of "set on the counter": one claw cannot lift the covered pan at all (handle_grasp results 10-28: <= 0.3 Nm
pitch capacity at 10 N vs 0.43-1.95 Nm needed), so it cannot carry it to a counter.  The hidden weights sit on the far side (-u, away
from the handle), so the overhang is made over the stove's own far edge by pushing the pan along the handle axis (the left claw's palm
rests against the handle end, so the push goes through the palm, not friction).  The counter in the covered scene is not used.

Timeline [s]: settle 0.5 | close left 0.7 | settle 0.3 | push PUSH along -u 1.5 | settle 0.3 | right claw moves in APPROACH_IN 1.0 |
close right 0.7 | settle 0.3 | lift both LIFT 1.5 | hold 2.0.  The right claw waits at a stand-off pose from t=0 (initial configuration).

    python -m world_model.v2.nero.pan.under_rim --placement far_centre [--push 0.07] [--out x.json]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time

import numpy as np
import mujoco

from .. import config as C
from .. import adapters as A
from .. import gripper as G
from ..hold_test import Ctl
from ..ik import ArmIK
from ..democ import kin as K
from . import covered as CV
from . import scene as S
from .handle_grasp import (DT, LIFT, FINGER_OPEN, LEFT_APPROACH, LEFT_JAW, TwoCtl, find_pregrasp, _set_fingers, _grip_force_mode)

from .rim_geometry import RIGHT_APPROACH, RIGHT_JAW, APPROACH_IN, EDGE_CLEAR, _perturb_pose   # moved 2026-11-03 (single source)
SEG = [("settle", 0.5), ("close_L", 0.7), ("settle", 0.3), ("push", 1.5), ("settle", 0.3), ("approach_R", 1.0),
       ("close_R", 0.7), ("settle", 0.3), ("lift", 1.5), ("hold", 2.0)]


def _segments():
    t, out = 0.0, []
    for name, dur in SEG:
        out.append((name, t, t + dur))
        t += dur
    return out


def run(placement="far_centre", hidden_mass=None, push=0.07, grasp_frac=0.6, grip_force=10.0, condim=None, snap=None,
        perturb=None, payload_mass=None, observe=None):
    """perturb (robustness sweeps): dict(dx, dy [m], yaw_deg = TRUE pan pose error vs the believed nominal pose used for planning;
    mu_scale = multiply every geom's sliding friction).  Plans (claw targets) come from the NOMINAL settled scene unless
    observe = dict(noise=<observe_pose.NOISE key>, seed=int, retries=3): then observe_pose estimates the pan pose from one depth
    image (a fresh image per retry while fit_ok is False; refuse after `retries`) and the left pinch target + approach, the push
    direction and the right clamp target are re-aimed at the ESTIMATE.  observe=None runs the original code path unchanged."""
    perturb = dict(perturb or {})
    cp = CV.CoveredParams(placement=placement)
    if hidden_mass is not None:
        cp = dataclasses.replace(cp, hidden_mass=hidden_mass, overload_mass=hidden_mass)
    cp = dataclasses.replace(cp, pan=dataclasses.replace(cp.pan, handle_grasp_frac=grasp_frac))
    p = cp.pan
    m = CV.compose(cp)
    m.opt.timestep = DT
    if condim is not None:
        m.geom_condim[m.geom("pan_handle").id] = condim
    if perturb.get("mu_scale"):
        m.geom_friction[:, 0] *= perturb["mu_scale"]
    tau = C.tau_limit()      # F3 11-03: FL-MPC torque limit, one source (config.tau_limit)
    d = mujoco.MjData(m)
    S.settle(m, d, 1.0)
    handle_site_nominal = d.site_xpos[m.site("pan_handle_grasp").id].copy()     # BELIEF: planning uses the nominal pose
    if observe is not None:
        from . import observe_pose as O
        rdr = mujoco.Renderer(m, O.CAM["height"], O.CAM["width"])
        tmpl = O.template(m, d, p, renderer=rdr)
        pos0, R0 = d.xpos[m.body("pan").id].copy(), d.xmat[m.body("pan").id].reshape(3, 3).copy()
    if any(perturb.get(k) for k in ("dx", "dy", "yaw_deg")):
        _perturb_pose(m, d, p.pan_centre_xy, perturb.get("dx", 0.0), perturb.get("dy", 0.0), perturb.get("yaw_deg", 0.0))
        S.settle(m, d, 0.5)
    statics = CV.probe_statics(m, d, p)
    u = p.handle_axis()
    pan = m.body("pan").id
    left_target, left_app, left_jaw = handle_site_nominal, LEFT_APPROACH, LEFT_JAW
    centre_aim = np.array([p.pan_centre_xy[0], p.pan_centre_xy[1], 0.0])
    obs_rec = None
    if observe is not None:
        rng = np.random.default_rng(observe.get("seed", 0))
        tries = []
        for _ in range(observe.get("retries", 3)):
            est = O.estimate(m, d, p, tmpl, observe.get("noise", "clean"), rng, renderer=rdr)
            tries.append(est)
            if est["fit_ok"]:
                break
        rdr.close()
        tru = O.true_pose(m, d, p, pos0, R0)
        obs_rec = dict(est=est, true=tru, n_images=len(tries), noise=observe.get("noise", "clean"),
                       residual_mm=1e3 * float(np.hypot(est["dx"] - tru["dx"], est["dy"] - tru["dy"])),
                       residual_yaw_deg=float(est["yaw_deg"] - tru["yaw_deg"]))
        if not est["fit_ok"]:
            return dict(placement=placement, hidden_mass=None if placement == "none" else cp.hidden_mass, perturb=perturb,
                        push_m=push, grasp_frac=grasp_frac, grip_force_N=grip_force, observe=obs_rec, held=False,
                        error=f"observe_pose: no good fit in {len(tries)} images (refused)")
        aim, Rz = O.aim_transform(est, p.pan_centre_xy)
        left_target, left_app, left_jaw = aim(handle_site_nominal), Rz @ LEFT_APPROACH, Rz @ LEFT_JAW
        u = Rz @ u
        centre_aim = aim(centre_aim)
    # left pre-grasp on the handle, fingers open
    preL = find_pregrasp(m, d, "left", left_target, left_app, left_jaw)
    d.qpos[ArmIK(m, "left").qa] = preL["q"]
    _set_fingers(m, d, "left", FINGER_OPEN)
    mujoco.mj_forward(m, d)
    # right clamp target AFTER the push: far rim outer face, fingertips reach in until EDGE_CLEAR short of the stove edge
    stove_edge = p.pan_centre_xy[1] - S.ASSUMED["stove_half"][1]          # far (-y) edge of the stove top (handle points +y)
    if observe is None:
        rim_after = p.pan_centre_xy[1] - push - (p.pan_radius + p.pan_wall_t)
        x_R = p.pan_centre_xy[0]
    else:                                   # pan centre after pushing `push` along the ESTIMATED handle axis (-u)
        c_after = centre_aim - push * u
        rim_after = c_after[1] - (p.pan_radius + p.pan_wall_t)
        x_R = c_after[0]
    overhang = stove_edge - rim_after
    tip_y = stove_edge - EDGE_CLEAR
    z_mid = p.pan_base_z() + 0.5 * (p.pan_bottom_t + p.pan_wall_h + CV.ASSUMED_COVERED["lid_t"])
    target_R = np.array([x_R, tip_y, z_mid])
    standoff = target_R - APPROACH_IN * RIGHT_APPROACH
    preR = find_pregrasp(m, d, "right", standoff, RIGHT_APPROACH, RIGHT_JAW)
    d.qpos[ArmIK(m, "right").qa] = preR["q"]
    _set_fingers(m, d, "right", FINGER_OPEN)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    A.disable_arm_servos(m)
    addr = {s: A.arm_addrs(m, s)[0] for s in C.SIDES}
    q0 = {s: d.qpos[addr[s]].copy() for s in C.SIDES}
    gb = {s: m.body(f"{s}_arm_gripper_base").id for s in C.SIDES}
    m_nom = float(m.body_mass[pan] + m.body_mass[m.body("lid").id])
    if payload_mass == "true":          # as if the probe measured the total (pan + lid + hidden) mass exactly
        m_nom += float(m.body_mass[m.body("hidden").id]) if placement != "none" else 0.0
    elif payload_mass is not None:
        m_nom = float(payload_mass)
    ctl = TwoCtl(m, dict(left=q0["left"].tolist(), right=q0["right"].tolist()), tau, pan, gb, m_nom, share=0.5)
    NP, FDT = ctl.fl.NP, ctl.fl.DT
    act = {s: m.actuator(f"{s}_arm_gripper_pos").id for s in C.SIDES}
    fing = {m.geom(f"{s}_arm_finger{i}_col").id: s for s in C.SIDES for i in (1, 2)}
    pan_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == pan}
    skip = {m.body("lid").id} | ({m.body("hidden").id} if placement != "none" else set())
    segs = _segments()
    seg_t = {n: (a, b) for n, a, b in segs}
    t_end = segs[-1][2]

    paths = {s: [] for s in C.SIDES}          # list of (t0, t1, waypoints) per arm

    def add_path(side, t0, t1, delta):
        pik = K.PoseIK(m, side)
        q = d.qpos[addr[side]].copy()
        p0, R0 = pik.ik.pose(q)
        wp = [q.copy()]
        for i in range(1, 61):
            a = i / 60
            q, _ = pik.solve(p0 + delta * a * a * (3 - 2 * a), R0, q, q_post=wp[0], k_post=0.1, iters=60, tol=1e-9)
            wp.append(q.copy())
        paths[side].append((t0, t1, wp))

    def cmd(side, t):
        q = q0[side]
        for t0, t1, wp in paths[side]:
            if t < t0:
                break
            a = np.clip((t - t0) / (t1 - t0), 0, 1) * (len(wp) - 1)
            i = min(int(a), len(wp) - 2)
            q = (1 - (a - i)) * wp[i] + (a - i) * wp[i + 1]
        return q

    started = set()
    log = dict(t=[], pan_xyz=[], tilt=[], support=[], fz={s: [] for s in C.SIDES}, squeeze={s: [] for s in C.SIDES})
    fv = np.zeros(6)
    ref = {}
    slip = {s: [] for s in C.SIDES}
    ts_c = []
    y_pan0 = float(d.xpos[pan][1])
    n = int(round(t_end / DT))
    for k in range(n):
        t = k * DT
        for name, delta, side in (("push", -push * u, "left"), ("approach_R", APPROACH_IN * RIGHT_APPROACH, "right"),
                                  ("lift", np.array([0, 0, LIFT]), "both")):
            a, b = seg_t[name]
            if name not in started and t >= a - 1e-9:
                started.add(name)
                for s in (C.SIDES if side == "both" else (side,)):
                    add_path(s, a, b, delta)
        ctl.qe_ref = np.stack([cmd("left", t + i * FDT) for i in range(NP + 1)])
        ctl.qi_ref = np.stack([cmd("right", t + i * FDT) for i in range(NP + 1)])
        for s, close_seg in (("left", "close_L"), ("right", "close_R")):
            t_c = seg_t[close_seg][0]
            if abs(t - t_c) < DT / 2:
                _grip_force_mode(m, act[s], grip_force)
            d.ctrl[act[s]] = G.open_frac_to_ctrl(FINGER_OPEN) if t < t_c else G.force_ctrl(grip_force)
        ctl.on = t >= seg_t["lift"][0]                       # nominal empty-pan payload split 50/50 once both claws hold
        t0 = time.perf_counter()
        te, ti = ctl.tau(d)
        ts_c.append((time.perf_counter() - t0) * 1e3)
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[ctl.ld] = te
        d.qfrc_applied[ctl.rd] = ti
        S.step(m, d)                                     # 11-01: 1 ms physics substeps (config.substeps_default)
        if snap is not None:
            snap(t, m, d)
        fz = {s: 0.0 for s in C.SIDES}
        sq = {s: 0.0 for s in C.SIDES}
        sup = 0.0
        for j in range(d.ncon):
            c = d.contact[j]
            g1, g2 = c.geom1, c.geom2
            if (g1 in pan_geoms) == (g2 in pan_geoms) and not (g1 in fing or g2 in fing):
                continue
            mujoco.mj_contactForce(m, d, j, fv)
            f_w = c.frame.reshape(3, 3).T @ fv[:3]
            if g1 in fing or g2 in fing:                     # finger on pan, lid or anything: squeeze = sum of normals
                sq[fing[g1 if g1 in fing else g2]] += float(fv[0])
            if (g1 in pan_geoms) == (g2 in pan_geoms):
                continue
            if g1 in pan_geoms:
                f_w = -f_w
            other = g2 if g1 in pan_geoms else g1
            if other in fing:
                fz[fing[other]] += f_w[2]
            elif m.geom_bodyid[other] not in skip:
                sup += f_w[2]
        Rp = d.xmat[pan].reshape(3, 3)
        if k % 10 == 0:
            log["t"].append(round(t, 3))
            log["pan_xyz"].append([round(float(x), 4) for x in d.xpos[pan]])
            log["tilt"].append(round(float(np.degrees(np.arccos(np.clip(Rp[2, 2], -1, 1)))), 2))
            log["support"].append(round(sup, 3))
            for s in C.SIDES:
                log["fz"][s].append(round(fz[s], 3))
                log["squeeze"][s].append(round(sq[s], 2))
        if t >= seg_t["lift"][0] - 1e-9:
            for s in C.SIDES:
                Rg = d.xmat[gb[s]].reshape(3, 3)
                rel_p, rel_R = Rg.T @ (d.xpos[pan] - d.xpos[gb[s]]), Rg.T @ Rp
                ref.setdefault(s, (rel_p, rel_R))
                c_ = (np.trace(ref[s][1].T @ rel_R) - 1) / 2
                slip[s].append((float(np.linalg.norm(rel_p - ref[s][0])), float(np.degrees(np.arccos(np.clip(c_, -1, 1))))))
    i_push_end = int(round(seg_t["settle"][1] / DT))
    lift0 = int(round(seg_t["lift"][0] / (10 * DT)))
    z_pre = log["pan_xyz"][lift0][2]
    lift = float(d.xpos[pan][2] - z_pre)
    lid_off = float(np.linalg.norm(d.xpos[m.body("lid").id] - d.xpos[pan] - Rp @ np.array(
        [0, 0, p.pan_bottom_t + p.pan_wall_h + CV.ASSUMED_COVERED["lid_t"] / 2])))
    hidden_in = None
    if placement != "none":
        h = d.xpos[m.body("hidden").id] - d.xpos[pan]
        hidden_in = bool(np.linalg.norm(h[:2]) < p.pan_radius and abs(h[2]) < 0.06)
    last = slice(-20, None)                                   # last 1 s of the log (every 10 steps)
    res = dict(
        placement=placement, hidden_mass=None if placement == "none" else (cp.overload_mass if placement == "overload" else cp.hidden_mass),
        mode="real_friction_contact", push_m=push, perturb=perturb, payload_model_mass_kg=m_nom, grasp_frac=grasp_frac, grip_force_N=grip_force, condim=int(m.geom_condim[m.geom("pan_handle").id]),
        overhang_planned_m=float(overhang), statics=statics,
        pregrasp=dict(left=dict(margin=preL["margin"]), right=dict(margin=preR["margin"])),
        pan_dy_after_push_m=float(log["pan_xyz"][int(round(seg_t["approach_R"][0] / (10 * DT)))][1] - y_pan0),
        lift_achieved_m=lift, final_tilt_deg=log["tilt"][-1], max_tilt_deg_lift_hold=float(max(log["tilt"][lift0:])),
        hold_last1s_support_N=float(np.mean(log["support"][last])),
        hold_last1s_fz_N={s: float(np.mean(log["fz"][s][last])) for s in C.SIDES},
        hold_last1s_squeeze_N={s: float(np.mean(log["squeeze"][s][last])) for s in C.SIDES},
        slip_by_claw={s: dict(mm=1e3 * slip[s][-1][0], deg=slip[s][-1][1]) for s in C.SIDES},
        lid_offset_mm=1e3 * lid_off, hidden_still_in_pan=hidden_in,
        n_infeasible=int(ctl.n_infeasible), n_fallback=int(ctl.n_fallback), ctrl_ms_mean=float(np.mean(ts_c[5:])),
        observe=obs_rec, log=log,
    )
    # ASSUMED success thresholds: lift >= 80 % of LIFT, no support force, tilt < 6 deg, per-claw slip < 15 mm / 6 deg
    res["held"] = bool(lift >= 0.8 * LIFT and res["hold_last1s_support_N"] < 0.1 and res["final_tilt_deg"] < 6.0
                       and all(v["deg"] < 6.0 and v["mm"] < 15.0 for v in res["slip_by_claw"].values()))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--placement", default="far_centre", choices=list(CV.ASSUMED_COVERED["placements"]))
    ap.add_argument("--hidden-mass", type=float, default=None)
    ap.add_argument("--push", type=float, default=0.07)
    ap.add_argument("--grasp-frac", type=float, default=0.6)
    ap.add_argument("--grip-force", type=float, default=10.0)
    ap.add_argument("--condim", type=int, default=None, choices=[3, 4])
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    r = run(a.placement, a.hidden_mass, a.push, a.grasp_frac, a.grip_force, a.condim)
    s = json.dumps(r, indent=1, default=float)
    if a.out:
        open(a.out, "w").write(s)
    print(json.dumps({k: v for k, v in r.items() if k not in ("log", "statics")}, indent=1, default=float))


if __name__ == "__main__":
    main()
