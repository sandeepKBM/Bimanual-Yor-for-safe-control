"""pan/handle_grasp.py -- step 2: REAL-friction one-claw pinch of the covered pan's handle, lift, hold.  SIMULATION ONLY.

Adapted from democ/grasp_smoke.py (same FL-MPC NV=7 driver, same timeline, torque control through qfrc_applied, servos disabled):
settle 0.5 s (claw open around the handle) -> close 0.7 s (--grip servo: stock position target 0; --grip force: constant F N)
-> settle 0.3 s -> lift LIFT over 1.5 s (Cartesian, rotation fixed, IK continuation from the ACTUAL hand pose) -> hold 2 s.
Left claw on the handle (approach -y, jaw z: fingers above/below the handle); the right arm holds its initial pose, or, with
second_frac, ALSO grips the handle (top-down, jaw across the width) and both arms lift together (two-claw lift).
No weld, no teleport: whether the pan stays in the claw is decided by MuJoCo contact + friction only.

Payload model inside FL-MPC ("nominal"): the controller knows the EMPTY covered pan (pan + lid mass at the pan body's COM),
never the hidden weight -- that is the parameter the demo is about.  `--payload off` = unmodified FL-MPC.

    python -m world_model.v2.nero.pan.handle_grasp --placement far_centre [--hidden-mass 0.2] [--out x.json]
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
from ..hold_test import Ctl
from ..ik import ArmIK
from ..democ import kin as K
from ..democ import payload as PL
from . import covered as CV
from . import scene as S

DT = 0.005
T_SETTLE, T_CLOSE, T_SETTLE2, T_LIFT, T_HOLD = 0.5, 0.7, 0.3, 1.5, 2.0
LIFT = 0.05
FINGER_OPEN = 0.03          # per finger: 6 cm gap around the 1.4 cm handle


class PanCtl(Ctl):
    """FL-MPC with the payload hooks on the LEFT arm only: nominal (empty covered pan) mass at the pan COM."""

    def __init__(self, m, hold_q, tau_vec, pan_id, grip_id, mass, **kw):
        super().__init__(m, hold_q, tau_vec, **kw)
        self.pan, self.grip, self.mass, self.on = pan_id, grip_id, mass, False

    def tau(self, d):
        if not self.on:
            return super().tau(d)
        mujoco.mj_forward(self.m, d)
        Me, be = PL.arm_payload_terms(self.m, d, self.pan, self.grip, self.ld, 1.0, self.mass)
        pairs = self.pairs(d)
        te, ti, ok = self.fl.step(self.m, d, self.lq, self.ld, self.rq, self.rd, self.qe_ref, self.z, self.qi_ref, self.z, pairs,
                                  M_ego_extra=Me, bias_ego_extra=be)
        self.n_infeasible += int(not ok)
        self.n_fallback += int(bool(self.fl.last_fallback))
        return te, ti


def _arm_contacts(m, d, side, allow=("pan",)):
    worst = 0.0
    for c in d.contact[:d.ncon]:
        bs = [m.body(m.geom_bodyid[g]).name for g in (c.geom1, c.geom2)]
        if not any(b.startswith(side) for b in bs):
            continue
        other = [b for b in bs if not b.startswith(side)]
        if other and other[0] in allow:
            continue
        worst = min(worst, c.dist)
    return worst


def _set_fingers(m, d, side, v):
    for j in range(m.njnt):
        nm = m.joint(j).name
        if nm.startswith(f"{side}_arm_gripper_joint"):
            d.qpos[m.jnt_qposadr[j]] = v if m.jnt_range[j][1] > 0 else -v     # joint2 range is [-0.05, 0]


RIGHT_GRIPS = {   # (approach, jaw) for the right claw on the handle; straight top-down hits the table with the wrist (checked 10-28)
    "tilt": (np.array([0.5, 0.0, -0.866]), np.array([0.866, 0.0, 0.5])),   # 30 deg off vertical, from the robot side; jaw ~across width
    "side": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),        # horizontal from the robot side, fingers above/below
}
LEFT_APPROACH, LEFT_JAW = np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])     # from the handle end, fingers above/below


def handle_point(m, d, p, frac):
    """World point on the handle centre line at `frac` of its length from the wall (same formula as scene._pan_local)."""
    ax = p.handle_axis()
    loc = ax * (p.pan_radius + p.pan_wall_t + frac * p.handle_len) + np.array([0, 0, p.pan_bottom_t + p.pan_wall_h - p.handle_t / 2])
    pan = m.body("pan").id
    return d.xpos[pan] + d.xmat[pan].reshape(3, 3) @ loc


def find_pregrasp(m, d0, side, target, approach, jaw, n_sol=10, seeds=60):
    """IK at `target`, fingers open; keep solutions where nothing touches the arm (pan, table, the other arm); best joint margin."""
    ik = ArmIK(m, side)
    best = None
    for q, e in ik.collect(target, approach, jaw, n_sol=n_sol, seeds=seeds):
        d = mujoco.MjData(m)
        d.qpos[:] = d0.qpos
        d.qpos[ik.qa] = q
        _set_fingers(m, d, side, FINGER_OPEN)
        mujoco.mj_forward(m, d)
        if _arm_contacts(m, d, side, allow=()) > -1e-3 and (best is None or ik.margin(q) > best[1]):
            best = (q, ik.margin(q), e)
    if best is None:
        raise RuntimeError(f"no contact-free pre-grasp for the {side} claw")
    return dict(q=best[0], margin=best[1], ik_err=[float(x) for x in best[2]], target=target)


def _grip_force_mode(m, act, force):
    """ASSUMPTION (labelled in the output): the claw runs in force/current mode at `force` N closing (stock MJCF = position servo,
    kp 200 N/m, which on a 14 mm handle squeezes only ~kp * 7 mm = 1.4 N).  10 N = the MJCF actuator forcerange limit."""
    m.actuator_gainprm[act, :3] = (1.0, 0.0, 0.0)
    m.actuator_biasprm[act, :3] = 0.0
    m.actuator_ctrlrange[act] = (-force, force)


class TwoCtl(Ctl):
    """FL-MPC with payload hooks on BOTH arms (nominal empty-pan mass split by `share`), like democ.grasp_smoke.PayloadCtl."""

    def __init__(self, m, hold_q, tau_vec, pan_id, grips, mass, share=0.5, **kw):
        super().__init__(m, hold_q, tau_vec, **kw)
        self.pan, self.grips, self.mass, self.share, self.on = pan_id, grips, mass, share, False

    def tau(self, d):
        if not self.on:
            return super().tau(d)
        mujoco.mj_forward(self.m, d)
        Me, be = PL.arm_payload_terms(self.m, d, self.pan, self.grips["left"], self.ld, self.share, self.mass)
        Mi, bi = PL.arm_payload_terms(self.m, d, self.pan, self.grips["right"], self.rd, 1.0 - self.share, self.mass)
        te, ti, ok = self.fl.step(self.m, d, self.lq, self.ld, self.rq, self.rd, self.qe_ref, self.z, self.qi_ref, self.z,
                                  self.pairs(d), M_ego_extra=Me, bias_ego_extra=be, M_int_extra=Mi, bias_int_extra=bi)
        self.n_infeasible += int(not ok)
        self.n_fallback += int(bool(self.fl.last_fallback))
        return te, ti


def run(placement="far_centre", hidden_mass=None, payload="nominal", tau_preset=None, grip="servo", grip_force=10.0,
        grasp_frac=None, second_frac=None, second_grip="tilt", snap=None, second_depth=None, condim=None, left_grasp=None):
    """grasp_frac: left pinch point along the handle (default scene 0.6).  second_frac: if set, the RIGHT claw also grips the
    handle there (top-down, jaw across the handle width) and both arms lift together (two-claw lift)."""
    cp = CV.CoveredParams(placement=placement)
    if hidden_mass is not None:
        cp = dataclasses.replace(cp, hidden_mass=hidden_mass, overload_mass=hidden_mass)
    if grasp_frac is not None:
        cp = dataclasses.replace(cp, pan=dataclasses.replace(cp.pan, handle_grasp_frac=grasp_frac))
    two = second_frac is not None
    sides = ("left", "right") if two else ("left",)
    m = CV.compose(cp)
    m.opt.timestep = DT
    if condim is not None:   # sensitivity: 4 adds torsional (twist) friction, 6 also rolling; coefficients = geom friction[1:], max-combined
        m.geom_condim[m.geom("pan_handle").id] = condim
    tau = C.tau_limits(tau_preset)
    d = mujoco.MjData(m)
    S.settle(m, d, 1.0)                                    # pan, lid and puck at rest on the stove (arms at their initial pose)
    statics = CV.probe_statics(m, d, cp.pan)
    if left_grasp is None:      # default: pinch at the handle site from the handle end
        pre = {"left": find_pregrasp(m, d, "left", d.site_xpos[m.site("pan_handle_grasp").id].copy(), LEFT_APPROACH, LEFT_JAW)}
    else:                       # externally proposed grasp (e.g. graspnet): world fingertip target, approach, jaw (unit vectors)
        pre = {"left": find_pregrasp(m, d, "left", np.asarray(left_grasp["target"], float), np.asarray(left_grasp["approach"], float),
                                     np.asarray(left_grasp["jaw"], float))}
    d.qpos[ArmIK(m, "left").qa] = pre["left"]["q"]
    _set_fingers(m, d, "left", FINGER_OPEN)
    mujoco.mj_forward(m, d)
    if two:                                                # right pre-grasp checked against the left arm already in place
        ap_r, jw_r = RIGHT_GRIPS[second_grip]
        # the IK target is the FINGERTIP point (render check 10-28): push it past the handle along the approach so the fingers wrap
        # the whole cross-section (half-extent of the handle box along the approach + 4 mm), unless second_depth is given
        if second_depth is None:
            second_depth = 0.5 * (abs(ap_r[0]) * cp.pan.handle_w + abs(ap_r[2]) * cp.pan.handle_t) + 0.004
        pre["right"] = find_pregrasp(m, d, "right", handle_point(m, d, cp.pan, second_frac) + second_depth * ap_r, ap_r, jw_r)
        d.qpos[ArmIK(m, "right").qa] = pre["right"]["q"]
        _set_fingers(m, d, "right", FINGER_OPEN)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    A.disable_arm_servos(m)
    addr = {s: A.arm_addrs(m, s)[0] for s in C.SIDES}
    q0 = {s: d.qpos[addr[s]].copy() for s in C.SIDES}
    pan = m.body("pan").id
    gb = {s: m.body(f"{s}_arm_gripper_base").id for s in C.SIDES}
    m_nom = float(m.body_mass[pan] + m.body_mass[m.body("lid").id])
    hold = dict(left=q0["left"].tolist(), right=q0["right"].tolist())
    if payload == "nominal":
        ctl = TwoCtl(m, hold, tau, pan, gb, m_nom, share=0.5) if two else PanCtl(m, hold, tau, pan, gb["left"], m_nom)
    else:
        ctl = Ctl(m, hold, tau)
    NP, FDT = ctl.fl.NP, ctl.fl.DT
    act_g = {s: m.actuator(f"{s}_arm_gripper_pos").id for s in sides}
    fing = {m.geom(f"{s}_arm_finger{i}_col").id: (s, f"f{i}") for s in sides for i in (1, 2)}
    handle = m.geom("pan_handle").id
    t_close0, t_close1 = T_SETTLE, T_SETTLE + T_CLOSE
    t_lift0 = t_close1 + T_SETTLE2
    t_lift1, t_end = t_lift0 + T_LIFT, t_lift0 + T_LIFT + T_HOLD
    n = int(round(t_end / DT))
    wp = {s: [] for s in sides}

    def capture_lift():
        for s in sides:
            pik = K.PoseIK(m, s)
            q = d.qpos[addr[s]].copy()
            p0, R0 = pik.ik.pose(q)
            wp[s].append(q.copy())
            for i in range(1, 61):
                u = i / 60
                q, _ = pik.solve(p0 + np.array([0, 0, LIFT * u * u * (3 - 2 * u)]), R0, q, q_post=wp[s][0], k_post=0.1, iters=60, tol=1e-9)
                wp[s].append(q.copy())

    def cmd(s, t):
        if s not in sides or not wp[s]:
            return q0[s]
        u = np.clip((t - t_lift0) / T_LIFT, 0, 1) * (len(wp[s]) - 1)
        i = min(int(u), len(wp[s]) - 2)
        a = u - i
        return (1 - a) * wp[s][i] + a * wp[s][i + 1]

    z_pan0 = float(d.xpos[pan][2])
    ref = {}
    slip = {s: [] for s in sides}
    fnorm, tilt, ts_c = [], [], []
    tau_sat = 0
    pan_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == pan}
    skip_bodies = {m.body("lid").id} | ({m.body("hidden").id} if placement != "none" else set())
    pitch_ax = np.cross([0.0, 0.0, 1.0], cp.pan.handle_axis())   # horizontal, normal to the handle: the probe's pitch axis
    gsite = m.site("pan_handle_grasp").id
    wrench = []                  # per step: {side: (Fz, pitch moment about the LEFT grasp point)}, support Fz (stove/counter/table)
    fv = np.zeros(6)
    for k in range(n):
        t = k * DT
        if not wp["left"] and t >= t_lift0 - 1e-9:
            capture_lift()
        ctl.qe_ref = np.stack([cmd("left", t + i * FDT) for i in range(NP + 1)])
        ctl.qi_ref = np.stack([cmd("right", t + i * FDT) for i in range(NP + 1)])
        for s in sides:
            if grip == "force" and abs(t - t_close0) < DT / 2:
                _grip_force_mode(m, act_g[s], grip_force)
            d.ctrl[act_g[s]] = FINGER_OPEN if t < t_close0 else (-grip_force if grip == "force" else 0.0)
        if payload == "nominal" and t >= t_close1:
            ctl.on = True
        t0 = time.perf_counter()
        te, ti = ctl.tau(d)
        ts_c.append((time.perf_counter() - t0) * 1e3)
        tau_sat += int(np.any(np.abs(te) >= 0.98 * tau) or np.any(np.abs(ti) >= 0.98 * tau))
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[ctl.ld] = te
        d.qfrc_applied[ctl.rd] = ti
        mujoco.mj_step(m, d)
        if snap is not None:
            snap(t, m, d)
        fn = {k_: 0.0 for k_ in fing.values()}
        cw = {s: [0.0, 0.0] for s in sides}
        oz = 0.0
        gp_now = d.site_xpos[gsite]
        for j in range(d.ncon):
            c = d.contact[j]
            g1, g2 = c.geom1, c.geom2
            if (g1 in pan_geoms) == (g2 in pan_geoms):
                continue
            mujoco.mj_contactForce(m, d, j, fv)
            f_w = c.frame.reshape(3, 3).T @ fv[:3]             # world frame, acting on geom2
            if g1 in pan_geoms:
                f_w = -f_w
            other = g2 if g1 in pan_geoms else g1
            if other in fing:
                s, f = fing[other]
                if handle in (g1, g2):
                    fn[(s, f)] += float(fv[0])
                cw[s][0] += f_w[2]
                cw[s][1] += float(np.dot(np.cross(c.pos - gp_now, f_w), pitch_ax))
            elif m.geom_bodyid[other] not in skip_bodies:
                oz += f_w[2]
        fnorm.append(fn)
        wrench.append((cw, oz))
        Rp = d.xmat[pan].reshape(3, 3)
        tilt.append(float(np.degrees(np.arccos(np.clip(Rp[2, 2], -1, 1)))))
        if t >= t_lift0 - 1e-9:
            for s in sides:
                Rg = d.xmat[gb[s]].reshape(3, 3)
                rel_p, rel_R = Rg.T @ (d.xpos[pan] - d.xpos[gb[s]]), Rg.T @ Rp
                ref.setdefault(s, (rel_p, rel_R))
                c_ = (np.trace(ref[s][1].T @ rel_R) - 1) / 2
                slip[s].append((float(np.linalg.norm(rel_p - ref[s][0])), float(np.degrees(np.arccos(np.clip(c_, -1, 1))))))
    i_hold = int(round(t_lift1 / DT))
    lift = float(d.xpos[pan][2] - z_pan0)
    lid_ok = float(np.linalg.norm(d.xpos[m.body("lid").id] - d.xpos[pan] - (d.xmat[pan].reshape(3, 3) @ np.array(
        [0, 0, cp.pan.pan_bottom_t + cp.pan.pan_wall_h + CV.ASSUMED_COVERED["lid_t"] / 2]))))
    hidden_in = None
    if placement != "none":
        h = d.xpos[m.body("hidden").id] - d.xpos[pan]
        hidden_in = bool(np.linalg.norm(h[:2]) < cp.pan.pan_radius and abs(h[2]) < 0.06)
    last = wrench[-200:]
    sl = slip["left"]
    res = dict(
        placement=placement, hidden_mass=None if placement == "none" else (cp.overload_mass if placement == "overload" else cp.hidden_mass),
        payload_model=payload, nominal_mass_kg=m_nom, mode="real_friction_contact", claws=len(sides),
        grasp_frac=cp.pan.handle_grasp_frac if left_grasp is None else None, left_grasp=left_grasp, second_frac=second_frac, second_grip=second_grip if two else None, second_depth=second_depth,
        grip=grip, grip_force_N=grip_force if grip == "force" else None,
        assumed=dict(finger_mu=float(m.geom_friction[m.geom("left_arm_finger1_col").id][0]), handle_mu=float(m.geom_friction[handle][0]),
                     finger_force_limit_N=float(m.actuator_forcerange[act_g["left"]][1]), condim=int(m.geom_condim[handle]),
                     twist_friction_m=float(max(m.geom_friction[handle][1], m.geom_friction[m.geom("left_arm_finger1_col").id][1])),
                     cone="elliptic" if m.opt.cone == 1 else "pyramidal", finger_open=FINGER_OPEN, dt=DT),
        statics=statics, pregrasp={s: dict(margin=pre[s]["margin"], ik_err=pre[s]["ik_err"]) for s in sides},
        lift_achieved_m=lift, dropped=bool(lift < -0.02), final_tilt_deg=tilt[-1], max_tilt_deg_after_lift=float(max(tilt[int(t_lift0 / DT):])),
        slip_final_mm=1e3 * sl[-1][0], slip_final_deg=sl[-1][1], slip_max_deg=max(x[1] for x in sl),
        creep_last_1s_deg=sl[-1][1] - sl[-200][1],
        slip_by_claw={s: dict(mm=1e3 * slip[s][-1][0], deg=slip[s][-1][1]) for s in sides},
        grip_normal_N_hold={f"{s}_{f}": float(np.mean([x[(s, f)] for x in fnorm[i_hold:]])) for (s, f) in fing.values()},
        lid_offset_mm=1e3 * lid_ok, hidden_still_in_pan=hidden_in,
        hold_last1s_claw_Fz_N=float(np.mean([sum(w[0][s][0] for s in sides) for w in last])),
        hold_last1s_claw_pitch_moment_Nm=float(np.mean([w[0]["left"][1] for w in last])),
        hold_last1s_by_claw={s: dict(Fz_N=float(np.mean([w[0][s][0] for w in last])),
                                     pitch_moment_about_left_grasp_Nm=float(np.mean([w[0][s][1] for w in last]))) for s in sides},
        hold_last1s_support_Fz_N=float(np.mean([w[1] for w in last])),
        far_rim_still_supported=bool(np.mean([w[1] for w in last]) > 0.1),
        n_infeasible=int(ctl.n_infeasible), n_fallback=int(ctl.n_fallback), tau_saturated_steps=tau_sat,
        ctrl_ms_mean=float(np.mean(ts_c[5:])),
    )
    # HELD (smoke criterion, not a safety claim): not dropped, lifted >= 80% of LIFT, pan clear of every support, every claw's droop < 6 deg
    # and slip < 15 mm.
    res["held"] = bool(not res["dropped"] and lift >= 0.8 * LIFT and not res["far_rim_still_supported"]
                       and all(v["deg"] < 6.0 and v["mm"] < 15.0 for v in res["slip_by_claw"].values()))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--placement", default="far_centre", choices=list(CV.ASSUMED_COVERED["placements"]))
    ap.add_argument("--hidden-mass", type=float, default=None)
    ap.add_argument("--payload", default="nominal", choices=["nominal", "off"])
    ap.add_argument("--grip", default="servo", choices=["servo", "force"])
    ap.add_argument("--grip-force", type=float, default=10.0)
    ap.add_argument("--grasp-frac", type=float, default=None)
    ap.add_argument("--second-frac", type=float, default=None, help="two-claw lift: right claw on the handle at this fraction")
    ap.add_argument("--second-grip", default="tilt", choices=["tilt", "side"])
    ap.add_argument("--condim", type=int, default=None, choices=[3, 4, 6])
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    r = run(a.placement, a.hidden_mass, a.payload, grip=a.grip, grip_force=a.grip_force, grasp_frac=a.grasp_frac, second_frac=a.second_frac,
          second_grip=a.second_grip, condim=a.condim)
    s = json.dumps(r, indent=1, default=float)
    if a.out:
        open(a.out, "w").write(s)
    print(s)


if __name__ == "__main__":
    main()
