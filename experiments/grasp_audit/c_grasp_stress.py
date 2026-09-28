"""Part C: REAL friction-grasp stress test (cup: left arm, cube: right arm). SIMULATION ONLY.

After a verified grasp + 6 cm lift, the EE is driven by an accurate task-space inverse-dynamics
rig (common.TaskSpaceID; arm-joint frictionloss zeroed IN THE RIG so the commanded excitation is
realised -- finger joints untouched) through excitations of increasing magnitude, each started from
the SAME post-grasp snapshot:
  lin_{x,y,z}  : Hann-windowed 3-cycle sine of EE position, peak accel ladder
  rot_{pinch,perp,yaw}: Hann-windowed 3-cycle sine rotation about the grasp point (pad midpoint)
  tilt_{pinch,perp}: slow quasi-static tilt to an angle, hold 1 s
Measured per test: peak EE |a| (FD of exact velocity), peak |alpha|, object slip vs gripper
(translation, EE frame) and relative rotation, jaw contact counts, normal force, max friction-cone
utilisation, min contact distance (penetration), contact-count toggles (chatter), drop.

usage: python c_grasp_stress.py OBJ VARIANT OUT.json      OBJ in {cup,cube}
VARIANT keys (comma list k=v): mode=servo|motor, F=8, pad_mu=2.0, noslip=2, solref=0.005,
solimp0=0.95, impratio=10, cone=ell|pyr, cup_mu=0.6, ladder=full|quick
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import mujoco

from common import (REPO, sc, quat_mul, quat_conj, quat_angle, axis_angle_quat, body_vel, rel_pose,
                    TaskSpaceID, jaw_geoms, contact_stats)

sys.path.insert(0, str(REPO))
from scripts.cup_collect_task_utils import load_yor_cup_collect_model  # noqa: E402
from scripts import cup_task_utils as ct  # noqa: E402
from control import grasp_routines as gr  # noqa: E402
from control import cube_grasp as cg  # noqa: E402
from control.arm_controller import ArmController  # noqa: E402

DT = 0.005
TRUE_RANGE: dict = {}


def parse_variant(s):
    v = dict(mode="servo", F=8.0, pad_mu=2.0, noslip=2, solref=0.005, solimp0=0.95, impratio=10.0,
             cone="ell", cup_mu=0.6, ladder="full", lift=0.06, widejl="1")
    if s and s != "nominal":
        for kv in s.split(","):
            k, val = kv.split("=")
            v[k] = type(v[k])(val) if not isinstance(v[k], str) else val
    return v


def apply_model_params(model, side, v, obj_bid):
    for pn in (f"{side}_arm_link7_pad", f"{side}_arm_link8_pad"):
        g = model.geom(pn).id
        model.geom_friction[g, 0] = v["pad_mu"]
        model.geom_solref[g, 0] = v["solref"]
        model.geom_solimp[g, 0] = v["solimp0"]
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == obj_bid:
            model.geom_friction[g, 0] = v["cup_mu"] if v.get("_obj") == "cup" else model.geom_friction[g, 0]
    model.opt.noslip_iterations = int(v["noslip"])
    model.opt.impratio = float(v["impratio"])
    model.opt.cone = 1 if v["cone"] == "ell" else 0


def zero_arm_fricloss(model, side):
    joints = sc.LEFT_ARM_JOINTS if side == "left" else sc.RIGHT_ARM_JOINTS
    _, _, dadr = sc.joint_addrs(model, joints)
    model.dof_frictionloss[dadr] = 0.0


def smooth(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


# ---------------------------------------------------------------------------------------------
# grasp setups -> (model, data, side, obj_bid, finger-hold callable)
# ---------------------------------------------------------------------------------------------
def setup_cup(v):
    model, data = load_yor_cup_collect_model(restore_finger_object_collision=True)
    cup = model.body(ct.CUP_BODY_NAME).id
    v["_obj"] = "cup"
    apply_model_params(model, "left", v, cup)
    gr.place_cup_in_open_gripper(model, data, side="left", z_frac=0.5, carry_balls=True)
    if v["mode"] == "motor":
        cg.set_finger_motor_mode(model, "left", 20.0)
    arm = ArmController(model, data, "left", cfg=gr.transport_cfg(),
                        grip_mode="motor_direct" if v["mode"] == "motor" else "servo_emulated")
    arm.hold_pose()
    for k in range(200):   # grasp_cup_real's close ramp (identical schedule)
        arm.set_grip_force(min(1.0, (k + 1) / 100) * v["F"])
        arm.step(); mujoco.mj_step(model, data)
    arm.set_grip_force(v["F"])
    for _ in range(200):
        arm.step(); mujoco.mj_step(model, data)

    def fingers():
        arm.grip.apply(data)
    return model, data, "left", cup, fingers, arm.grip


def setup_cube(v, cube="cube_3"):
    from scripts.smoke_realgrasp_cube import JointFLTracker
    from teleop import bimanual_teleop as bt
    model, data = load_yor_cup_collect_model(restore_finger_object_collision=True)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    cb = model.body(cube).id
    v["_obj"] = "cube"
    apply_model_params(model, "right", v, cb)
    cg.disable_right_arm_position_servos(model)
    grip = cg.CubeGripper(model, data, "right", mode=v["mode"])
    grip.open_fingers(1.0)
    ik = bt.ArmIK(model, "right")
    pg = cg.plan_top_down_pregrasp(model, data, cube, "right", ik=ik, tip_clearance=0.004)
    jt = JointFLTracker(model, data)

    def run_joint(q_a, q_b, T, settle=0.0):
        for k in range(int((T + settle) / DT)):
            jt.step(q_a + smooth(k * DT / T) * (q_b - q_a)); grip.step(); mujoco.mj_step(model, data)
    run_joint(jt.q(), pg.q_hover, 2.5, settle=0.5)
    site_off = pg.site_pos_at - pg.ee_pos_at
    qs = []
    ik.last_q = jt.q()
    for s in np.linspace(0, 1, 11):
        q, _ = ik.solve(data, pg.ee_pos_hover + s * (pg.ee_pos_at - pg.ee_pos_hover) + site_off, pg.site_quat,
                        ori_weight=1.0, iters=200, allow_coarse_restart=False)
        qs.append(q)
    qs = np.array(qs)
    for k in range(int(1.9 / DT)):
        s = smooth(k * DT / 1.5)
        x = s * (len(qs) - 1); i = min(int(x), len(qs) - 2)
        jt.step(qs[i] + (x - i) * (qs[i + 1] - qs[i])); grip.step(); mujoco.mj_step(model, data)
    grip.close_on_cube(cube, force_n=v["F"], ramp_s=0.3)
    for k in range(int(0.9 / DT)):
        jt.step(qs[-1]); grip.step(); mujoco.mj_step(model, data)

    def fingers():
        grip.step()
    return model, data, "right", cb, fingers, grip


# ---------------------------------------------------------------------------------------------
def run_test(model, data, side, obj, fingers, jaw7, jaw8, snap, profile, T, held_bid):
    mujoco.mj_setState(model, data, snap, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    ts = TaskSpaceID(model, data, side, held_bid=held_bid)
    ee = ts.ee
    p0, q0 = data.xpos[ee].copy(), data.xquat[ee].copy()
    rp0, rq0 = rel_pose(data, ee, obj)
    pad = [model.geom(f"{side}_arm_link7_pad").id, model.geom(f"{side}_arm_link8_pad").id]
    c0 = 0.5 * (data.geom_xpos[pad[0]] + data.geom_xpos[pad[1]])
    pinch = data.geom_xpos[pad[1]] - data.geom_xpos[pad[0]]; pinch /= np.linalg.norm(pinch)
    pm0 = data.xmat[ee].reshape(3, 3).T @ (c0 - data.xpos[ee])
    n = int(T / DT)
    jr = np.array([TRUE_RANGE[int(model.dof_jntid[d_])] for d_ in ts.dadr]); jlo, jhi = jr[:, 0], jr[:, 1]
    out = dict(joint_limit_hit=False, slip_mm=0.0, rot_deg=0.0, util_max=0.0, util95_frac=0.0, pen_mm=0.0, fn_min=1e9, fn_max=0.0,
               min_jaw=99, toggles=0, noc_run=0, a_peak=0.0, al_peak=0.0, ee_err_mm=0.0, dropped=False)
    vprev = wprev = None
    ncprev = None
    run = 0
    util_hi = 0
    for k in range(n):
        t = k * DT
        p, q, v, w, a, al = profile(t, p0, q0, c0, pinch)
        fingers()
        ts.step(p, q, v, w, a, al)
        mujoco.mj_step(model, data)
        lv, lw = body_vel(model, data, ee)
        if vprev is not None:
            out["a_peak"] = max(out["a_peak"], float(np.linalg.norm(lv - vprev) / DT))
            out["al_peak"] = max(out["al_peak"], float(np.linalg.norm(lw - wprev) / DT))
        vprev, wprev = lv, lw
        rp, rq = rel_pose(data, ee, obj)
        out["slip_mm"] = max(out["slip_mm"], float(np.linalg.norm(rp - rp0) * 1e3))
        if out["slip_mm"] > 2.0 and "t_first_slip2" not in out:
            out["t_first_slip2"] = t
        Re = data.xmat[ee].reshape(3, 3)
        pm = Re.T @ (0.5 * (data.geom_xpos[pad[0]] + data.geom_xpos[pad[1]]) - data.xpos[ee])
        out["padmid_drift_mm"] = max(out.get("padmid_drift_mm", 0.0), float(np.linalg.norm(pm - pm0) * 1e3))
        out["obj_vs_padmid_mm"] = max(out.get("obj_vs_padmid_mm", 0.0), float(np.linalg.norm((rp - pm) - (rp0 - pm0)) * 1e3))
        out["rot_deg"] = max(out["rot_deg"], float(np.degrees(quat_angle(quat_mul(quat_conj(rq0), rq)))))
        n7, n8, f7, f8, util, dmin = contact_stats(model, data, obj, jaw7, jaw8)
        out["util_max"] = max(out["util_max"], util)
        util_hi += int(util > 0.95)
        out["pen_mm"] = max(out["pen_mm"], -dmin * 1e3 if dmin < 1 else 0.0)
        out["fn_min"] = min(out["fn_min"], f7 + f8); out["fn_max"] = max(out["fn_max"], f7 + f8)
        out["min_jaw"] = min(out["min_jaw"], min(n7, n8))
        nc = (n7 > 0, n8 > 0)
        if ncprev is not None and nc != ncprev:
            out["toggles"] += 1
        ncprev = nc
        run = run + 1 if (n7 + n8) == 0 else 0
        out["noc_run"] = max(out["noc_run"], run)
        out["ee_err_mm"] = max(out["ee_err_mm"], float(np.linalg.norm(data.xpos[ee] - p) * 1e3))
        qj = data.qpos[ts.qadr]
        if np.any(np.minimum(qj - jlo, jhi - qj) < 0.01):
            out["joint_limit_hit"] = True
        if not np.all(np.isfinite(data.qpos)):
            out["dropped"] = True; break
    out["util95_frac"] = util_hi / n
    out["dropped"] = bool(out["dropped"] or out["noc_run"] >= 20 or out["slip_mm"] > 30.0)
    out["tau_sat_steps"] = ts.sat_steps
    out["rig_ok"] = bool(out["ee_err_mm"] < 10.0 and ts.sat_steps == 0)
    return out


def hann_sine(A, f, cycles=3):
    Tw = cycles / f
    w = 2 * np.pi * f

    def s(t):
        if t > Tw:
            return 0.0, 0.0, 0.0
        h = 0.5 * (1 - np.cos(2 * np.pi * t / Tw)); hd = 0.5 * (2 * np.pi / Tw) * np.sin(2 * np.pi * t / Tw)
        hdd = 0.5 * (2 * np.pi / Tw) ** 2 * np.cos(2 * np.pi * t / Tw)
        x = A * h * np.sin(w * t)
        xd = A * (hd * np.sin(w * t) + h * w * np.cos(w * t))
        xdd = A * (hdd * np.sin(w * t) + 2 * hd * w * np.cos(w * t) - h * w * w * np.sin(w * t))
        return x, xd, xdd
    return s, Tw


def lin_profile(axis, a_peak):
    f = max(3.0, np.sqrt(a_peak / 0.05) / (2 * np.pi))
    A = a_peak / (2 * np.pi * f) ** 2
    s, Tw = hann_sine(A, f)
    e = np.zeros(3); e[axis] = 1.0

    def prof(t, p0, q0, c0, pinch):
        x, xd, xdd = s(t)
        return p0 + x * e, q0, xd * e, np.zeros(3), xdd * e, np.zeros(3)
    return prof, Tw + 0.5


def rot_axis(which, pinch):
    zup = np.array([0, 0, 1.0])
    if which == "pinch":
        return pinch
    if which == "yaw":
        return zup
    ax = np.cross(zup, pinch); return ax / np.linalg.norm(ax)


def rot_profile(which, al_peak):
    f = max(2.0, np.sqrt(al_peak / 0.5) / (2 * np.pi))
    A = al_peak / (2 * np.pi * f) ** 2
    s, Tw = hann_sine(A, f)

    def prof(t, p0, q0, c0, pinch):
        ax = rot_axis(which, pinch)
        th, thd, thdd = s(t)
        R = np.zeros(9); mujoco.mju_quat2Mat(R, axis_angle_quat(ax, th)); R = R.reshape(3, 3)
        r = p0 - c0
        p = c0 + R @ r
        wv = thd * ax; al = thdd * ax
        rr = R @ r
        v = np.cross(wv, rr); a = np.cross(al, rr) + np.cross(wv, np.cross(wv, rr))
        return p, quat_mul(axis_angle_quat(ax, th), q0), v, wv, a, al
    return prof, Tw + 0.5


def tilt_profile(which, deg):
    Tr, Th = 1.0, 1.0
    th1 = np.radians(deg)

    def prof(t, p0, q0, c0, pinch):
        ax = rot_axis(which, pinch)
        s = smooth(t / Tr)
        th = th1 * s
        R = np.zeros(9); mujoco.mju_quat2Mat(R, axis_angle_quat(ax, th)); R = R.reshape(3, 3)
        return c0 + R @ (p0 - c0), quat_mul(axis_angle_quat(ax, th), q0), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    return prof, Tr + Th


def main():
    obj, vs, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    v = parse_variant(vs)
    t0 = time.time()
    model, data, side, ob, fingers, grip = setup_cup(v) if obj == "cup" else setup_cube(v)
    jaw7, jaw8 = jaw_geoms(model, side)
    TRUE_RANGE.clear(); TRUE_RANGE.update({j: model.jnt_range[j].copy() for j in range(model.njnt)})
    n7, n8, f7, f8, util, dmin = contact_stats(model, data, ob, jaw7, jaw8)
    grasp = dict(n7=n7, n8=n8, fn7=f7, fn8=f8, util=util, pen_mm=-dmin * 1e3,
                 finger_q=[float(data.qpos[model.joint(f'{side}_arm_joint7').qposadr[0]]),
                           float(data.qpos[model.joint(f'{side}_arm_joint8').qposadr[0]])],
                 act_force=[float(data.actuator_force[model.actuator(f'{side}_arm_joint7_pos').id]),
                            float(data.actuator_force[model.actuator(f'{side}_arm_joint8_pos').id])])
    zero_arm_fricloss(model, side)
    if v.get("widejl", "1") == "1":
        # RIG-ONLY: widen the excited arm's 6 joint ranges so the excitation device does not slam
        # into a joint limit (LEFT_GRASP_Q has joint5 only 0.11 rad from its limit) -- the grasp
        # physics (fingers, pads, object) is untouched. joint_limit_hit is still reported vs the TRUE range.
        joints = sc.LEFT_ARM_JOINTS if side == "left" else sc.RIGHT_ARM_JOINTS
        for jn in joints:
            model.jnt_range[model.joint(jn).id] = (-6.5, 6.5)
    if side == "left":
        sc.disable_left_arm_position_servos(model)
    # lift clear of the table with the rig
    ts = TaskSpaceID(model, data, side, held_bid=ob)
    ee = ts.ee
    p0, q0 = data.xpos[ee].copy(), data.xquat[ee].copy()
    rp0, rq0 = rel_pose(data, ee, ob)
    for k in range(int(1.5 / DT)):
        s = smooth(k * DT / 1.0)
        fingers(); ts.step(p0 + np.array([0, 0, v["lift"] * s]), q0); mujoco.mj_step(model, data)
    rp1, rq1 = rel_pose(data, ee, ob)
    grasp["lift_slip_mm"] = float(np.linalg.norm(rp1 - rp0) * 1e3)
    grasp["lift_rot_deg"] = float(np.degrees(quat_angle(quat_mul(quat_conj(rq0), rq1))))
    n7, n8, f7, f8, util, dmin = contact_stats(model, data, ob, jaw7, jaw8)
    grasp.update(after_lift_n=[n7, n8], after_lift_fn=[f7, f8], after_lift_util=util)
    obj_mass = float(model.body_subtreemass[ob]) + (sum(model.body_mass[b] for b in ct.ball_body_ids(model))
                                                     if obj == "cup" else 0.0)
    grasp["obj_mass_kg"] = obj_mass
    size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)
    snap = np.zeros(size)
    mujoco.mj_getState(model, data, snap, mujoco.mjtState.mjSTATE_INTEGRATION)

    tests = [("hold", 0, (lambda t, p0, q0, c0, pn: (p0, q0, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))), 2.0)]
    lin_l = [1, 2, 4, 8, 16, 32, 64] if v["ladder"] == "full" else [4, 16, 64]
    rot_l = [5, 10, 20, 40, 80, 160, 320] if v["ladder"] == "full" else [20, 80, 320]
    tilt_l = [15, 30, 45, 60, 90] if v["ladder"] == "full" else [30, 60, 90]
    for ax in range(3):
        for a in lin_l:
            pr, T = lin_profile(ax, a); tests.append((f"lin_{'xyz'[ax]}", a, pr, T))
    for w in ("pinch", "perp", "yaw"):
        for a in rot_l:
            pr, T = rot_profile(w, a); tests.append((f"rot_{w}", a, pr, T))
    for w in ("pinch", "perp"):
        for d in tilt_l:
            pr, T = tilt_profile(w, d); tests.append((f"tilt_{w}", d, pr, T))
    res = []
    for name, lvl, pr, T in tests:
        r = run_test(model, data, side, ob, fingers, jaw7, jaw8, snap, pr, T, ob)
        r.update(test=name, level=lvl)
        res.append(r)
        print(json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items()}), flush=True)
    json.dump(dict(obj=obj, variant=v, grasp=grasp, tests=res, wall_s=time.time() - t0), open(outp, "w"), indent=1,
              default=float)
    print("GRASP", json.dumps(grasp, default=float), "wall", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
