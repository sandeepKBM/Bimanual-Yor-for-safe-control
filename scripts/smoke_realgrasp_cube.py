#!/usr/bin/env python
"""Stage-4 smoke: RIGHT-arm REAL friction pick of a cube (no weld, no teleport).

Headless, simulation only. Sequence (right arm torque-controlled throughout, its 6
position servos disabled in-memory; left arm stays on its stock servos at home):
  1. approach: computed-torque joint PD (tau = M a + bias) along a smoothstep from home
     to the IK pre-grasp HOVER config (pads 10 cm above the cube, top-down); fingers open;
  2. OSC ArmController (Khatib OSC, transport gains) descends to the grasp pose;
  3. control.cube_grasp.CubeGripper closes under force (ramp 0 -> F) until both pads touch;
  4. lift 5 cm, then move 5 cm laterally (+x), hold;
  5. release (fingers open), retreat 6 cm up, let the cube fall/settle.

Assertions (numbers printed): both-pad contact (>=1 each, >=2 total) throughout the
carry; cube-vs-gripper slip < 1.5 cm; cube lifted >= 4 cm; finger actuator force within
+-20 N; after release the cube separates from the pads and comes to rest on the table;
no NaNs.

Run:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python scripts/smoke_realgrasp_cube.py \
      [--cube cube_3] [--force 8] [--grip-mode motor|servo]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.cup_collect_task_utils import load_yor_cup_collect_model  # noqa: E402
from control.arm_controller import ArmController  # noqa: E402
from control.cartesian_impedance import CartesianImpedanceConfig  # noqa: E402
from control import cube_grasp as cg  # noqa: E402
import sim_common as sc  # noqa: E402  (path set up by control.*)

_FAILS: list[str] = []
TABLE_TOP_Z = 0.55


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not cond:
        _FAILS.append(name)


def smoothstep(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


def osc_cfg() -> CartesianImpedanceConfig:
    # grasp_routines.transport_cfg gains (kp_rot 80 / kd_rot 120 = the verified sweet spot)
    return CartesianImpedanceConfig(
        n_joints=6, tau_max_nm=np.full(6, sc.TAU_MAX),
        kp_pos=np.full(3, 225.0), kd_pos=np.full(3, 30.0),
        kp_rot=80.0, kd_rot=120.0,
        kp_posture=1.0, kd_posture=2.0, kd_joint=0.5, use_osc_inertia=True,
    )


class JointFLTracker:
    """Joint-space computed-torque tracker for the right arm (same structure as FL-MPC's
    inner law, tau = M a + c): tau = M (kp e + kd e_dot) + Kq e + Dq e_dot + (bias - passive)
    [+ reflected payload gravity]. The small Nm-level Kq/Dq term overcomes the model's
    0.3 Nm joint frictionloss on the light wrist joints, which a pure M-scaled PD cannot."""

    def __init__(self, model, data):
        self.model, self.data = model, data
        _, self.qadr, self.dadr = sc.joint_addrs(model, sc.RIGHT_ARM_JOINTS)
        self.kp, self.kd = 100.0, 20.0
        self.Kq = np.array([30.0, 30.0, 30.0, 15.0, 15.0, 15.0])
        self.Dq = np.array([3.0, 3.0, 3.0, 1.0, 1.0, 1.0])
        self.ee_bid = model.body("right_arm_gripper_base").id
        self.held_bid = None
        self.max_tau = 0.0

    def q(self):
        return self.data.qpos[self.qadr].copy()

    def step(self, q_ref, qd_ref=None):
        m, d = self.model, self.data
        qd_ref = np.zeros(6) if qd_ref is None else qd_ref
        e = q_ref - d.qpos[self.qadr]
        ed = qd_ref - d.qvel[self.dadr]
        M = sc.full_mass_matrix(m, d)[np.ix_(self.dadr, self.dadr)]
        tau = M @ (self.kp * e + self.kd * ed) + self.Kq * e + self.Dq * ed
        tau += d.qfrc_bias[self.dadr] - d.qfrc_passive[self.dadr]
        if self.held_bid is not None:
            _, bias_box = sc.payload_reflected_terms(m, d, self.ee_bid, self.held_bid, self.dadr)
            tau += bias_box
        tau = np.clip(tau, -sc.TAU_MAX, sc.TAU_MAX)
        self.max_tau = max(self.max_tau, float(np.max(np.abs(tau))))
        d.qfrc_applied[self.dadr] = tau


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", default="cube_3")
    ap.add_argument("--force", type=float, default=8.0)
    ap.add_argument("--grip-mode", default="motor", choices=["motor", "servo"])
    ap.add_argument("--arm", default="joint", choices=["joint", "osc"],
                    help="arm driver after the approach: joint-space FL tracker through IK "
                         "waypoints (default) or the OSC ArmController (Cartesian)")
    ap.add_argument("--tip-clearance", type=float, default=0.004)
    ap.add_argument("--flagship-collision", action="store_true",
                    help="load WITHOUT restore_finger_object_collision (the flagship's default): "
                         "jaw<->cube contact then relies on the cube's CUBE_ARM_BITS")
    args = ap.parse_args()
    t_wall = time.time()

    print(f"=== Stage-4 real-grasp CUBE smoke (right arm, {args.cube}, arm={args.arm}, "
          f"grip={args.grip_mode}, F={args.force} N) ===")
    model, data = load_yor_cup_collect_model(restore_finger_object_collision=not args.flagship_collision)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    dt = model.opt.timestep

    cube_bid = model.body(args.cube).id
    ee_bid = model.body("right_arm_gripper_base").id
    check("no weld equality in model (neq == 0)", model.neq == 0, f"neq={model.neq}")
    check("jaw<->cube contact enabled (bitmasks)", cg.finger_cube_contact_enabled(model, cube_bid, "right"))

    # --- torque control owns the right arm; fingers -> force motors (Option B) ---------
    cg.disable_right_arm_position_servos(model)
    grip = cg.CubeGripper(model, data, "right", mode=args.grip_mode)
    if args.grip_mode == "motor":
        check("finger actuators are force motors, |forcerange|<=20", cg.is_finger_motor_mode(model, "right"))
    grip.open_fingers(1.0)

    from teleop import bimanual_teleop as bt
    ik = bt.ArmIK(model, "right")
    pg = cg.plan_top_down_pregrasp(model, data, args.cube, "right", ik=ik, tip_clearance=args.tip_clearance)
    print(f"  pregrasp: jaw-axis yaw={np.degrees(pg.yaw):.0f} deg  tip_local={np.round(pg.tip_local, 4)}  "
          f"IK err hover={pg.ik_err_hover*1e3:.2f} mm at={pg.ik_err_at*1e3:.2f} mm")
    check("pre-grasp IK reachable (< 5 mm)", max(pg.ik_err_hover, pg.ik_err_at) < 5e-3)
    if _FAILS:
        print(f"RESULT: FAIL (setup) {_FAILS} -- pick a reachable cube (cube_2/cube_3 are)")
        return 1

    jt = JointFLTracker(model, data)
    nan_bad = [False]

    def physics():
        mujoco.mj_step(model, data)
        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
            nan_bad[0] = True

    def run_joint(q_a, q_b, T, settle=0.0):
        for k in range(int((T + settle) / dt)):
            jt.step(q_a + smoothstep(k * dt / T) * (q_b - q_a))
            grip.step()
            physics()

    # site-space Cartesian segment -> dense IK path (for the joint tracker)
    _path_cache: dict = {}

    def ik_path(site_a, site_b, n=11):
        """Dense warm-started IK along a site-space segment (cached; no random restarts,
        so an unreachable target fails fast instead of stalling every tick)."""
        key = (tuple(np.round(site_a, 6)), tuple(np.round(site_b, 6)))
        if key in _path_cache:
            return _path_cache[key]
        qs = []
        for s in np.linspace(0.0, 1.0, n if np.linalg.norm(site_b - site_a) > 1e-9 else 1):
            q, e = ik.solve(data, site_a + s * (site_b - site_a), pg.site_quat, ori_weight=1.0,
                            iters=200, allow_coarse_restart=False)
            qs.append(q)
        if len(qs) == 1:
            qs.append(qs[0])
        _path_cache[key] = np.array(qs)
        return _path_cache[key]

    def q_on_path(path, s):
        x = s * (len(path) - 1)
        i = min(int(x), len(path) - 2)
        return path[i] + (x - i) * (path[i + 1] - path[i])

    # --- 1. approach (both modes): joint FL tracker home -> q_hover ----------------------
    run_joint(jt.q(), pg.q_hover, 2.5, settle=0.5)
    print(f"  approach: EE err to hover = {np.linalg.norm(data.xpos[ee_bid]-pg.ee_pos_hover)*1e3:.1f} mm")

    arm = None
    if args.arm == "osc":
        arm = ArmController(model, data, "right", cfg=osc_cfg(), disable_servos=False)

    site_off = pg.site_pos_at - pg.ee_pos_at        # constant (fixed orientation)

    def carry_segment(ee_a, ee_b, T, per_tick=None):
        """Move the EE body from ee_a to ee_b (fixed top-down orientation) in T s."""
        path = ik_path(ee_a + site_off, ee_b + site_off) if arm is None else None
        for k in range(int(T / dt)):
            s = smoothstep(k * dt / T)
            if arm is None:
                jt.step(q_on_path(path, s))
            else:
                arm.set_ee_target(ee_a + s * (ee_b - ee_a), quat=pg.ee_quat)
                arm.step()
            grip.step()        # AFTER the arm step: the cube gripper owns the finger ctrl
            physics()
            if per_tick is not None:
                per_tick()

    # --- 2. descend to the grasp pose ----------------------------------------------------
    ik.last_q = jt.q()
    carry_segment(pg.ee_pos_hover, pg.ee_pos_at, 1.5)
    carry_segment(pg.ee_pos_at, pg.ee_pos_at, 0.4)
    pre = grip.contacts(cube_bid)
    tip_w = data.xpos[ee_bid] + data.xmat[ee_bid].reshape(3, 3) @ pg.tip_local
    print(f"  at grasp pose: EE err={np.linalg.norm(data.xpos[ee_bid]-pg.ee_pos_at)*1e3:.1f} mm  "
          f"tip-z above table={(tip_w[2]-TABLE_TOP_Z)*1e3:.1f} mm  jaw-centre xy vs cube="
          f"{np.round((tip_w[:2]-data.xpos[cube_bid][:2])*1e3, 1)} mm  pre-close contacts jaw7/jaw8/other="
          f"{pre.n_jaw7}/{pre.n_jaw8}/{pre.n_other_arm}")
    cube_z0 = float(data.xpos[cube_bid][2])
    cube_xy0 = data.xpos[cube_bid][:2].copy()

    # --- 3. close under force ------------------------------------------------------------
    grip.close_on_cube(args.cube, force_n=args.force, ramp_s=0.3)
    t_close = 0.0
    while grip.state != "holding" and t_close < 1.5:
        carry_segment(pg.ee_pos_at, pg.ee_pos_at, dt)
        t_close += dt
    held_state = grip.state
    carry_segment(pg.ee_pos_at, pg.ee_pos_at, 0.3)
    pc = grip.contacts()
    print(f"  close: reached '{held_state}' in {t_close:.2f}s  contacts jaw7={pc.n_jaw7} jaw8={pc.n_jaw8} "
          f"(named-pad {pc.n_named_pad}) other-arm={pc.n_other_arm}  Fn jaw7={pc.fn_jaw7:.2f} N "
          f"jaw8={pc.fn_jaw8:.2f} N  finger act force={np.round(grip.actuator_forces(), 2)} N  "
          f"finger q={np.round(grip.finger_q()*1e3, 1)} mm")
    check("grasp established (is_held: both jaws + Fn floor)", grip.is_held() and held_state == "holding")

    def cube_in_gripper():
        R = data.xmat[ee_bid].reshape(3, 3)
        return R.T @ (data.xpos[cube_bid] - data.xpos[ee_bid])

    rel0 = cube_in_gripper()
    # reflected payload gravity feed-forward (8 g cube) -- the arm's gravcomp does not cover it
    if arm is None:
        jt.held_bid = cube_bid
    else:
        arm.set_held_body(args.cube)

    # --- 4. lift 5 cm, move 5 cm laterally ------------------------------------------------
    p_at = pg.ee_pos_at.copy()
    p_lift = p_at + np.array([0.0, 0.0, 0.05])
    p_move = p_lift + np.array([0.05, 0.0, 0.0])
    st = dict(min7=10 ** 6, min8=10 ** 6, mintot=10 ** 6, slip=0.0, act=0.0, zmax=cube_z0,
              minfn=1e9, maxfn=0.0, other=0, npad=0)

    def monitor():
        c = grip.contacts()
        st["min7"], st["min8"] = min(st["min7"], c.n_jaw7), min(st["min8"], c.n_jaw8)
        st["mintot"] = min(st["mintot"], c.n_total)
        st["minfn"], st["maxfn"] = min(st["minfn"], c.fn_total), max(st["maxfn"], c.fn_total)
        st["other"] = max(st["other"], c.n_other_arm)
        st["npad"] = max(st["npad"], c.n_named_pad)
        st["slip"] = max(st["slip"], float(np.linalg.norm(cube_in_gripper() - rel0)))
        st["act"] = max(st["act"], float(np.max(np.abs(grip.actuator_forces()))))
        st["zmax"] = max(st["zmax"], float(data.xpos[cube_bid][2]))

    for a_, b_, T in [(p_at, p_lift, 1.5), (p_lift, p_lift, 0.3), (p_lift, p_move, 1.5), (p_move, p_move, 0.5)]:
        carry_segment(a_, b_, T, per_tick=monitor)
    lift = st["zmax"] - cube_z0
    carried_xy = float(np.linalg.norm(data.xpos[cube_bid][:2] - cube_xy0))
    ee_track = float(np.linalg.norm(data.xpos[ee_bid] - p_move))
    print(f"  carry: min contacts jaw7={st['min7']} jaw8={st['min8']} total={st['mintot']}  "
          f"Fn(sum) min/max={st['minfn']:.2f}/{st['maxfn']:.2f} N  other-arm<->cube max={st['other']}  "
          f"max slip={st['slip']*100:.2f} cm  cube lift={lift*100:.2f} cm  cube xy moved={carried_xy*100:.2f} cm  "
          f"EE track err={ee_track*1e3:.1f} mm  max |finger act|={st['act']:.2f} N  max |arm tau| (joint tracker phases)={jt.max_tau:.1f} Nm")
    check("both jaws in contact throughout carry (each >=1, total >=2)",
          st["min7"] >= 1 and st["min8"] >= 1 and st["mintot"] >= 2,
          f"jaw7>={st['min7']} jaw8>={st['min8']} total>={st['mintot']}")
    check("cube-vs-gripper slip < 1.5 cm", st["slip"] < 0.015, f"{st['slip']*100:.2f} cm")
    check("cube lifted >= 4 cm", lift >= 0.04, f"{lift*100:.2f} cm")
    check("cube carried laterally ~5 cm (>= 4 cm)", carried_xy >= 0.04, f"{carried_xy*100:.2f} cm")
    check("finger actuator force within +-20 N", st["act"] <= 20.0 + 1e-6, f"max {st['act']:.2f} N")

    # --- 5. release, retreat, settle -------------------------------------------------------
    if arm is None:
        jt.held_bid = None
    else:
        arm.set_held_body(None)
    grip.release()
    carry_segment(p_move, p_move, 0.4)
    p_up = p_move + np.array([0.0, 0.0, 0.06])
    carry_segment(p_move, p_up, 0.8)
    carry_segment(p_up, p_up, 1.2)
    pc = grip.contacts()
    v_cube = float(np.linalg.norm(data.cvel[cube_bid][3:]))
    w_cube = float(np.linalg.norm(data.cvel[cube_bid][:3]))
    cz = float(data.xpos[cube_bid][2])
    gap = float(np.linalg.norm(cg.pad_midpoint(model, data, "right") - data.xpos[cube_bid]))
    moved = float(np.linalg.norm(data.xpos[cube_bid][:2] - cube_xy0))
    rest_z = TABLE_TOP_Z + cg_half()
    print(f"  after release: jaw contacts={pc.n_total} other-arm={pc.n_other_arm}  padmid-cube dist={gap*100:.1f} cm  "
          f"cube z={cz:.4f} (rest z={rest_z:.4f})  |v|={v_cube*1e3:.2f} mm/s  |w|={w_cube:.3f} rad/s  "
          f"cube displaced {moved*100:.1f} cm from pick spot")
    check("cube separated from jaws after release", pc.n_total == 0 and pc.n_other_arm == 0 and gap > 0.03,
          f"contacts={pc.n_total} gap={gap*100:.1f} cm")
    check("cube at rest on the table (z within 3 mm, |v|<5 mm/s)",
          abs(cz - rest_z) < 3e-3 and v_cube < 5e-3, f"z={cz:.4f} v={v_cube*1e3:.2f} mm/s")
    check("no NaNs", not nan_bad[0])

    print(f"  wall time {time.time()-t_wall:.1f}s, sim time {data.time:.2f}s")
    if _FAILS:
        print(f"RESULT: FAIL ({len(_FAILS)}): {_FAILS}")
        return 1
    print("RESULT: PASS -- right arm REALLY picked, carried and released a cube (no weld/teleport)")
    return 0


def cg_half() -> float:
    from scripts.cup_collect_task_utils import CUBE_HALF
    return CUBE_HALF


if __name__ == "__main__":
    raise SystemExit(main())
