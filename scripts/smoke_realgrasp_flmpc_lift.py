#!/usr/bin/env python
"""Step-1 smoke: REAL friction grasp, then an FL-MPC-driven lift (SIM ONLY).

Grasp: same loader as scripts/smoke_realgrasp_lift.py (restore_finger_object_
collision=True), grip force 8 N, and either
  --grasp stage1  : control.grasp_routines.grasp_cup_real EXACTLY as the Stage-1
                    smoke (LEFT_GRASP_Q; its jaw axis is ~6 deg off horizontal,
                    so the pinch leaves the cup ~5.4 deg tilted before any motion);
  --grasp leveled : (default) the same routine with the jaw axis leveled first
                    (control.realgrasp_flmpc.grasp_cup_real_leveled).
Then the LEFT arm is handed to FL-MPC
(experiments/ab_smooth_vs_bangbang/mpc_fl.FLMPC, same construction as the
flagship demo's flmpc branch: NC=4, L=8, W_POS_EGO=800) which tracks a
joint-space lift reference (dense orientation-holding IK, 11 cm straight up,
1.2 s smoothstep, same as the demo's LIFT_HEIGHT_M/T_LIFT_MOVE) followed by a
3 s hold, with ALL of FL-MPC's hard rows active:
  arm-arm HOCBF clearance (incl. cup keypoints re-built each step from the
  ACTUAL cup pose), torque box +-100 Nm, cup-upright tilt rows (anchored on
  the TRUE cup axis each step), cup-arm leash, table floor;
and the real cup's mass/inertia folded in via sim_common.payload_reflected_terms
(M_ego_extra/bias_ego_extra), plus (default --friction-ff on) a joint dry-friction
feedforward: the arm joints' frictionloss=0.3 Nm is a MuJoCo constraint force
that FL's tau = M a + c does not cancel, and without it the light wrist stalls
and the EE rolls ~12 deg during the lift (measured; --friction-ff off). The RIGHT arm is held at its start pose by the
same joint FL-MPC QP (intruder reference = constant). The grip-force
controller keeps squeezing every step.

NO teleport, NO weld, NO velocity zeroing: nothing in this script writes
cup/ball qpos/qvel after grasp_cup_real returns.

Assertions (numbers printed):
  (1) TRUE cup tilt (cup body +z vs world +z) < 5.73 deg throughout lift+hold
  (2) both pads touching the cup and >= 2 pad-cup contacts throughout
  (3) grip slip (cup position in the EE frame vs start) < 2 cm
  (4) cup rises >= 7 cm
  (5) finger actuator force within +-20 N (and pad normal-force sum reported)
  (6) no NaNs
Also reports: EE tilt, max |tau|, QP-infeasible fraction, solve time.

Run (CPU-only, no GL):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python scripts/smoke_realgrasp_flmpc_lift.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts"),
           str(_REPO_ROOT / "experiments" / "ab_smooth_vs_bangbang")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sim_common as sc  # noqa: E402
import mpc_fl  # noqa: E402
from scripts.cup_collect_task_utils import load_yor_cup_collect_model  # noqa: E402
from scripts import cup_task_utils as ct  # noqa: E402
from teleop import bimanual_teleop as bt  # noqa: E402
from control.grasp_routines import grasp_cup_real  # noqa: E402
from control import realgrasp_flmpc as rg  # noqa: E402

# Flagship-demo constants (scripts/make_cup_collect_safety_demo.py), duplicated
# here as plain numbers so this smoke does not import the (heavy, torch-loading)
# demo module. Keep in sync.
LIFT_HEIGHT_M = 0.11
T_LIFT_MOVE = 1.2
CUP_TILT_MAX_RAD = 0.10       # demo.CUP_TILT_MAX_RAD (5.73 deg)
CUP_LEASH_M = 0.08            # demo.CUP_LEASH_M
TABLE_FLOOR_MARGIN_M = 0.006  # demo.TABLE_FLOOR_MARGIN_M
CUP_KP_RADIUS = 0.004
CUP_RIM_N = 6
CUP_RING_Z_FRACS = (0.0125, 0.5, 1.0)

_FAILS: list[str] = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        _FAILS.append(name)


def cup_keypoints_from_actual(data, ee_body, cup_body):
    """Cup barrier keypoints (same 3 rings + base point as the demo's
    build_cup_barrier_keypoints) from the ACTUAL cup pose, expressed as
    EE-local offsets (Jacobian through the arm chain)."""
    rel_pos, rel_quat = rg.cup_rel_pose_in_ee(data, ee_body, cup_body)
    R_rel = np.zeros(9)
    mujoco.mju_quat2Mat(R_rel, rel_quat / np.linalg.norm(rel_quat))
    R_rel = R_rel.reshape(3, 3)
    pts = []
    for zf in CUP_RING_Z_FRACS:
        z = zf * ct.CUP_HEIGHT
        for i in range(CUP_RIM_N):
            a = 2.0 * np.pi * i / CUP_RIM_N
            pts.append(np.array([ct.CUP_RADIUS * np.cos(a), ct.CUP_RADIUS * np.sin(a), z]))
    pts.append(np.zeros(3))
    return [(ee_body, rel_pos + R_rel @ p, CUP_KP_RADIUS) for p in pts]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grip-n", type=float, default=8.0)
    ap.add_argument("--hold-s", type=float, default=3.0)
    ap.add_argument("--grasp", choices=("stage1", "leveled"), default="leveled",
                    help="stage1: control.grasp_routines.grasp_cup_real exactly as smoke_realgrasp_lift.py "
                         "(LEFT_GRASP_Q; jaw axis ~6 deg off horizontal -> cup starts ~5.4 deg tilted). "
                         "leveled (default): same routine with the jaw axis leveled first "
                         "(control.realgrasp_flmpc.grasp_cup_real_leveled).")
    ap.add_argument("--friction-ff", choices=("on", "off"), default="on",
                    help="joint dry-friction feedforward on FL-MPC's commanded motion direction "
                         "(control.realgrasp_flmpc.friction_feedforward). off reproduces the wrist stall.")
    ap.add_argument("--debug", action="store_true", help="print a trace every 40 steps")
    ap.add_argument("--no-filter", action="store_true",
                    help="apply FL-MPC torque raw (default: same TorqueCommandFilter as the demo)")
    args = ap.parse_args()

    print("=== Step-1: real grasp + FL-MPC lift smoke (SIMULATION ONLY) ===")
    model, data = load_yor_cup_collect_model(restore_finger_object_collision=True)
    if args.grasp == "stage1":
        arm, res = grasp_cup_real(model, data, grip_force=args.grip_n, verbose=True)
    else:
        arm, res, _ginfo = rg.grasp_cup_real_leveled(model, data, grip_force=args.grip_n, verbose=True)
    print(f"  config: grasp={args.grasp} friction_ff={args.friction_ff} filter={'off' if args.no_filter else 'on'}")
    check("grasp established (>=2 pad-cup contacts)", res.n_pad_contacts >= 2,
          f"contacts={res.n_pad_contacts}, grip={res.grip_force_measured:.2f} N")
    grip = arm.grip                      # keep ONLY the grip-force controller; OSC is retired
    grip.set_target_force(args.grip_n)

    _, lq, ld = sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)
    _, rq, rd = sc.joint_addrs(model, sc.RIGHT_ARM_JOINTS)
    sc.disable_left_arm_position_servos(model)
    sc.disable_right_arm_position_servos(model)
    ee = model.body("left_arm_gripper_base").id
    cup = model.body(ct.CUP_BODY_NAME).id
    pads = arm.idx.pad_geom_ids
    mujoco.mj_forward(model, data)

    q_grasp = data.qpos[lq].copy()
    q_right = data.qpos[rq].copy()
    ik = bt.ArmIK(model, "left")
    track, q_lift, ik_err, ik_ori = rg.build_dense_lift_track(
        model, data, ik, q_grasp, np.array([0.0, 0.0, LIFT_HEIGHT_M]), T_LIFT_MOVE, args.hold_s)
    print(f"  lift IK: max pos err={ik_err*1000:.2f} mm, max EE-orientation drift along knots="
          f"{np.degrees(ik_ori):.2f} deg")

    fl = mpc_fl.FLMPC(model, NC=4, L=8, W_POS_EGO=800.0)
    tfl = sc.TorqueCommandFilter(6, sc.FILTER_LOWPASS_ALPHA, np.full(6, sc.FILTER_RATE_LIMIT_NM_S))
    tfr = sc.TorqueCommandFilter(6, sc.FILTER_LOWPASS_ALPHA, np.full(6, sc.FILTER_RATE_LIMIT_NM_S))
    left_table_kps = sc.barrier_keypoints(model, sc.EGO_PREFIX)
    right_table_kps = sc.barrier_keypoints(model, sc.INT_PREFIX)
    table_gid = model.geom("table_top").id
    table_top_z = float(data.geom_xpos[table_gid][2] + model.geom_size[table_gid][2])

    cup_z0 = float(data.xpos[cup][2])
    rel0, _ = rg.cup_rel_pose_in_ee(data, ee, cup)
    ee_axis_local = data.xmat[ee].reshape(3, 3).T @ np.array([0.0, 0.0, 1.0])   # EE "up" at grasp
    tilt_cup0 = rg.cup_true_tilt_rad(data, cup)
    print(f"  after grasp: TRUE cup tilt={np.degrees(tilt_cup0):.2f} deg  cup_z={cup_z0:.4f}  "
          f"pads touching={rg.pads_touching_cup(model, data, pads, cup)}")

    n_steps = int(round((T_LIFT_MOVE + args.hold_s) / sc.DT))
    NP = fl.NP
    L = dict(tilt=[], ee_tilt=[], pads=[], ncon=[], slip=[], fing=[], fn=[], tau=[], feas=[], st=[], z=[])
    nan = False
    t_wall = time.perf_counter()
    for k in range(n_steps):
        t = k * sc.DT
        grip.apply(data)
        mujoco.mj_forward(model, data)

        q_ego_ref = np.stack([track.sample(t + j * sc.DT) for j in range(NP + 1)])
        qd_ego_ref = np.stack([track.sample_dot(t + j * sc.DT) for j in range(NP + 1)])
        q_int_ref = np.tile(q_right, (NP + 1, 1))
        qd_int_ref = np.zeros((NP + 1, 6))

        cup_kps = cup_keypoints_from_actual(data, ee, cup)
        pairs = sc.all_pair_barriers(model, data, ld, rd, extra_ego_keypoints=cup_kps)
        M_box, bias_box = sc.payload_reflected_terms(model, data, ee, cup, ld)
        M_full = sc.full_mass_matrix(model, data)
        M_ll = M_full[np.ix_(ld, ld)] + M_box
        bias_l = data.qfrc_bias[ld] + bias_box
        M_rr = M_full[np.ix_(rd, rd)]
        bias_r = data.qfrc_bias[rd]
        a_te, b_te = sc.table_plane_barrier_rows(model, data, left_table_kps, ld, np.linalg.inv(M_ll),
                                                 bias_l, table_top_z, margin=TABLE_FLOOR_MARGIN_M)
        a_ti, b_ti = sc.table_plane_barrier_rows(model, data, right_table_kps, rd, np.linalg.inv(M_rr),
                                                 bias_r, table_top_z, margin=TABLE_FLOOR_MARGIN_M)
        # cup-upright rows anchored on the TRUE cup axis (not a captured EE axis)
        fl._cup_axis_local = rg.cup_axis_in_ee(data, ee, cup)
        t0 = time.perf_counter()
        tau_l, tau_r, feas = fl.step(
            model, data, lq, ld, rq, rd, q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref, pairs,
            M_ego_extra=M_box, bias_ego_extra=bias_box, table_rows=(a_te, b_te, a_ti, b_ti),
            left_ee_body=ee, tilt_max_rad=CUP_TILT_MAX_RAD,
            leash_ee_body=ee, leash_max_m=CUP_LEASH_M, cup_cone_facets=0, ego_hold=True)
        st = time.perf_counter() - t0
        if args.friction_ff == "on":
            tau_l = tau_l + rg.friction_feedforward(model, ld, data.qvel[ld], fl.a_ego_prev, sc.DT)
            tau_r = tau_r + rg.friction_feedforward(model, rd, data.qvel[rd], fl.a_int_prev, sc.DT)
        if not args.no_filter:
            tau_l = tfl.apply(tau_l, sc.DT)
            tau_r = tfr.apply(tau_r, sc.DT)
        tau_l = np.clip(tau_l, -sc.TAU_MAX, sc.TAU_MAX)
        tau_r = np.clip(tau_r, -sc.TAU_MAX, sc.TAU_MAX)
        data.qfrc_applied[ld] = tau_l
        data.qfrc_applied[rd] = tau_r

        # measurements (state at this step, before integration)
        rel, _ = rg.cup_rel_pose_in_ee(data, ee, cup)
        L["tilt"].append(rg.cup_true_tilt_rad(data, cup))
        L["ee_tilt"].append(rg.ee_tilt_rad(data, ee, ee_axis_local))
        L["pads"].append(rg.pads_touching_cup(model, data, pads, cup))
        L["ncon"].append(rg.pad_cup_contacts(model, data, pads, cup))
        L["slip"].append(float(np.linalg.norm(rel - rel0)))
        L["fing"].append(float(np.max(np.abs(rg.finger_actuator_forces(model, data)))))
        L["fn"].append(rg.pad_normal_force_sum(model, data, pads, cup))
        L["tau"].append(float(max(np.max(np.abs(tau_l)), np.max(np.abs(tau_r)))))
        L["feas"].append(bool(feas))
        L["st"].append(st)
        L["z"].append(float(data.xpos[cup][2]))

        if args.debug and k % 40 == 0:
            ax = rg.cup_axis_in_ee(data, ee, cup)
            b = data.xmat[ee].reshape(3, 3) @ ax
            print(f"   k={k:4d} |q-qref|={np.linalg.norm(data.qpos[lq]-q_ego_ref[0]):.4f} "
                  f"qd={np.linalg.norm(data.qvel[ld]):.3f} ee_tilt={np.degrees(L['ee_tilt'][-1]):.2f} "
                  f"cup_tilt={np.degrees(L['tilt'][-1]):.2f} tilt0=({-b[1]:+.3f},{b[0]:+.3f}) feas={feas} "
                  f"tau_l={np.round(tau_l,2)}")
        mujoco.mj_step(model, data)
        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
            nan = True
            print(f"  NaN at step {k}")
            break
    mujoco.mj_forward(model, data)
    wall = time.perf_counter() - t_wall

    A = {k: np.asarray(v) for k, v in L.items()}
    rise = float(data.xpos[cup][2]) - cup_z0
    k_lift_end = int(round(T_LIFT_MOVE / sc.DT))
    tilt_deg = np.degrees(A["tilt"])
    print()
    print(f"  steps={len(A['tilt'])}  wall={wall:.1f}s  ({1000*wall/max(1,len(A['tilt'])):.1f} ms/step)")
    print(f"  TRUE cup tilt [deg]: max={tilt_deg.max():.3f} (lift max={tilt_deg[:k_lift_end].max():.3f}, "
          f"hold max={tilt_deg[k_lift_end:].max():.3f}), final={tilt_deg[-1]:.3f}")
    print(f"  EE tilt from grasp orientation [deg]: max={np.degrees(A['ee_tilt']).max():.3f}")
    print(f"  pads touching cup: min={A['pads'].min()}  pad-cup contacts: min={A['ncon'].min()} "
          f"mean={A['ncon'].mean():.1f}")
    print(f"  grip slip (cup pos in EE frame vs start): max={A['slip'].max()*100:.3f} cm, "
          f"final={A['slip'][-1]*100:.3f} cm")
    print(f"  cup rise: {rise*100:.2f} cm (commanded EE lift {LIFT_HEIGHT_M*100:.0f} cm)")
    print(f"  finger actuator |force| max={A['fing'].max():.2f} N (limit {rg.FINGER_FORCE_LIMIT_N:.0f}); "
          f"pad normal-force sum: min={A['fn'].min():.2f} mean={A['fn'].mean():.2f} max={A['fn'].max():.2f} N")
    print(f"  |tau| max={A['tau'].max():.2f} Nm (TAU_MAX {sc.TAU_MAX:.0f}); QP infeasible frac="
          f"{1.0 - A['feas'].mean():.4f}; solve mean={1000*A['st'].mean():.1f} ms max={1000*A['st'].max():.1f} ms")
    print()
    check("(1) TRUE cup tilt < 5.73 deg throughout lift + hold", tilt_deg.max() < np.degrees(CUP_TILT_MAX_RAD),
          f"max={tilt_deg.max():.3f} deg")
    check("(2) both pads touching + >=2 pad-cup contacts throughout", A["pads"].min() >= 2 and A["ncon"].min() >= 2,
          f"min pads={A['pads'].min()}, min contacts={A['ncon'].min()}")
    check("(3) grip slip < 2 cm", A["slip"].max() < 0.02, f"max={A['slip'].max()*100:.3f} cm")
    check("(4) cup rises >= 7 cm", rise >= 0.07, f"rise={rise*100:.2f} cm")
    check("(5) finger force within +-20 N", A["fing"].max() <= rg.FINGER_FORCE_LIMIT_N + 1e-6,
          f"max={A['fing'].max():.2f} N")
    check("(6) no NaNs", not nan)
    if _FAILS:
        print(f"RESULT: FAIL ({len(_FAILS)}): {_FAILS}")
        return 1
    print("RESULT: PASS -- FL-MPC lifts a REALLY-GRIPPED cup (no teleport/weld/vel-zeroing) within the cup cone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
