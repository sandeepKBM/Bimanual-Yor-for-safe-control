#!/usr/bin/env python
"""collection/collect_cup_collect.py

Data collector for the FLAGSHIP "cube-collection into a held cup" scene
(scenes/scene_yor_cup_collect.xml, scripts/cup_collect_task_utils.py). Adapts
collection/collect_cup.py's collector loop/CLI shape to this scene, reuses
collection/episode.py verbatim (the exact episode/record format
world_model/logdir/wm_cup_full was trained from) and the ZED camera render
convention `collection/cameras.py`'s YorCupCameraRenderer already established
for the cup-only task (image = zed_cam frame duplicated into both
`rgb_head`/`rgb_wrist_right` halves, so world_model/dataset_adapter.py's
hardcoded `image = concat([rgb_head, rgb_wrist_right])` needs zero changes).

WHY THIS COLLECTOR EXISTS: `world_model/logdir/wm_cup_full/` (WM + monitor
head) was trained ONLY on `collection/data/cup_yor_full` -- collect_cup.py's
cup-ONLY scene (no cubes, cup teleported not grasped). On the cup-collect
scene (cubes present, cup physically held aloft by the left arm, right arm
crossing near it) that WM/monitor is OOD and reports spill risk ~1.0
constantly (a false positive, see scripts/make_cup_collect_safety_demo.py's
"geometric_spill" config). This collector produces cup-collect-scene
episodes -- REAL ZED image + REAL proprio + the SAME per-step spill/collision
label convention monitor_head.py already reads (`--spill-signal per_step`) --
so the WM + head can be retrained/calibrated on THIS scene's actual visual
and dynamic distribution.

Episode construction (real torque-controlled dynamics for BOTH arms; only
the joint-space PD REFERENCE trajectories below are scripted, exactly the
repo's standing "IK-solved joint-space reference + real tracking" convention
already used by scripts/make_cup_collect_safety_demo.py / eval/run_eval.py):
  - LEFT arm: `eval/scene_setup.init_cup_grasped_state` grasps the cup (+8
    balls) ONCE at t=0 via the just-fixed loader
    (`scripts.cup_collect_task_utils.load_yor_cup_collect_model`'s
    `restore_finger_object_collision` now defaults OFF, so the closing
    fingers no longer disturb/empty the cup at grasp-init -- see this
    module's `verify_no_poison_at_init` and the module docstring's Critical
    Prerequisite). The left arm then lifts the cup ~11cm (IK-solved, same
    height as the safety demo) and holds; every step the cup body is
    kinematically re-pinned to the left EE (`scene_setup.reattach_cup_rigidly`
    -- required: `init_cup_grasped_state`'s finger-pad grasp never actually
    contacts the cup wall in this scene, see that function's own docstring)
    while the 8 balls stay fully free/dynamic (what can actually spill).
    Some episode LEVELS additionally drive a `scene_setup.transfer_tilt_offset`
    burst on top of the lift hold -- the repo's own validated "physically
    grasped cup, tilt the wrist" spill mechanism (eval/run_eval.py) -- to
    deliberately induce real spills for label balance.
  - RIGHT arm: an IK-solved (`teleop.bimanual_teleop.ArmIK`, the same DLS IK
    already used by the safety demo) pick -> carry -> deposit waypoint chain
    targeting ONE of the scene's 6 cubes (varied per episode -- see
    `_choose_target_cube`), tracked by real computed-torque PD
    (`scene_setup.task_torque`, generalized to the right arm + cube payload).
    The cube is kinematically pinned to the right EE only while "grasped"
    (`reattach_cup_rigidly` again, generalized -- same scripted-attach
    convention `collection/collect_stack.py` already uses for its own
    cube/block payloads), detached before the deposit release so the actual
    drop into the cup is real contact physics.
  - NO HOCBF-QP safety filter is applied (unlike the safety demo/eval
    harness): this is a DATA COLLECTOR, not a safety demo, and we WANT a mix
    of near/actual arm-arm crossings for label balance -- adding the
    cooperative QP filter would suppress exactly the collision episodes this
    dataset needs.

Per-episode BEHAVIOR LEVELS (COLLECT_BEHAVIOR_LEVELS below, cycled by
`seed % len(...)` exactly like collect_cup.py's CUP_BEHAVIOR_LEVELS, so the
default full collection is an even 4-way mix):
  - "clean_deposit":     normal-speed pick/carry/deposit, no left-arm tilt,
                         generous hover/release clearance -> the common
                         "everything goes fine" case (no spill, no collision,
                         usually a successful deposit).
  - "spill_tilt":        clean_deposit's right-arm motion PLUS a left-arm
                         wrist tilt burst (70-100 deg) partway through ->
                         deliberately spills via the grasped-cup tipping
                         mechanism (independent of the right arm).
  - "collision_intrude": right arm targets a cube near the cup's side of the
                         table (cube_4/cube_5, see scene docstring) with
                         reduced hover/release clearance and faster (~2x)
                         phase timing, no tilt -> biases toward real arm-arm
                         contact and/or an incidental bump-induced spill.
  - "fast_aggressive":   collision_intrude's fast/close right-arm motion
                         PLUS a moderate left-arm tilt burst -> the "worst
                         case" mix (collision and/or spill and/or a fumbled
                         cube).
A monitor can only learn spill/collision from ZED+proprio context if the
dataset actually contains both positive and negative examples of each -- this
4-way cycle is what gives the resulting dataset that mix (see
`main()`'s printed per-level/outcome summary).

Per-step record (STEP_ARRAY_KEYS from collection/episode.py, unmodified,
plus a few EXTRA per-step arrays -- see EpisodeWriter.save's docstring for
why extras are safe to add):
  - rgb_head / rgb_wrist_right: the ZED (`zed_cam`) frame, duplicated (see
    module docstring above) -- world_model/dataset_adapter.py's `image`
    (T,64,64,6) source, UNCHANGED contract.
  - rgb_zed / rgb_wrist_left_ego / rgb_wrist_right_ego: EXTRA, auxiliary-only
    (same convention as collect_cup.py).
  - left_q/left_qd/right_q/right_qd/left_gripper_state/right_gripper_state:
    dataset_adapter.py's `proprio` (T,26) source, unchanged layout.
  - left_action/right_action: the joint-space PD REFERENCE this step (6 arm
    "ctrl-equivalent" position targets + 2 finger ctrl values) -- NOT the
    raw torque (position servos are disabled and driven by qfrc_applied
    instead, so `data.ctrl` on the arm actuators is stale/inert; the PD
    reference is the right analogue of "commanded action" and keeps the
    action DISTRIBUTION/SCALE consistent with wm_cup_full's collect_cup.py
    data, which recorded true position-actuator ctrl targets).
  - box_pose: repurposed (per episode.py's established cup-task convention)
    to carry the CUP body's (xyz, wxyz-quat) pose -- LABEL only, routed to
    `object_pose` by dataset_adapter.py, never fed to the WM encoder.
  - collision: real per-step arm-arm (left_arm_* vs right_arm_* body prefix)
    contact bool -- read directly by monitor_head.py's "collision" label.
  - dropped: repurposed (same collect_cup.py convention) to carry the
    per-step SPILL flag `count_spilled(model, data) > 0` -- read directly by
    monitor_head.py's `--spill-signal per_step` "spill" label.
  - EXTRA "spilled_count" (T,) int32: the raw count_spilled(...) value each
    step (magnitude, not just the `dropped` threshold).
  - EXTRA "deposited_count" (T,) int32: raw count_deposited_cubes(...) each
    step -- this task's OWN success signal (no analogue in the cup-only
    task).
  - EXTRA "cube_dropped" (T,) bool: the target cube fumbled outside the cup
    after the scripted release (on the table, not deposited) -- a genuine
    "drop" signal distinct from ball-spill, per the task's LABEL list.

Critical prerequisite check (run this file's `--verify-only` mode, or see
`main()`'s default behavior on first run): every episode's t=0 (right after
grasp+lift settle) must have count_spilled==0 (balls start IN the cup). If
any episode starts already-spilled, the loader fix did not take and the
WHOLE dataset would be poisoned -- this collector raises/aborts rather than
silently writing poisoned episodes (see `collect_episode`'s assertion).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SIM_COMMON_DIR = REPO_ROOT / "experiments" / "ab_smooth_vs_bangbang"
if str(SIM_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_COMMON_DIR))
import sim_common as sc  # noqa: E402  (read-only import; DT/TAU_MAX/joint_addrs/
                          # disable_*_position_servos/full_mass_matrix/
                          # payload_reflected_terms reused verbatim)

EVAL_DIR = REPO_ROOT / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
import scene_setup  # noqa: E402  (read-only import; init_cup_grasped_state/
                     # reattach_cup_rigidly/cup_rel_pose/task_torque/
                     # left_ee_jac_rot_pinv/transfer_tilt_offset/
                     # left_right_contact/EE_BODY_NAME reused verbatim)

from scripts import cup_collect_task_utils as cct  # noqa: E402  (read-only import)
from teleop import bimanual_teleop as bt  # noqa: E402  (read-only import; ArmIK/
                                           # finger_actuators/set_gripper reused)

from collection import cameras as cam_mod  # noqa: E402  (ZED_CAM name only)
from collection.episode import (  # noqa: E402
    EpisodeWriter,
    GRIPPER_JOINT_RANGE_M,
    build_episode_metadata,
)
from collection.perturbations import PerturbationSpec  # noqa: E402

COLLECT_BEHAVIOR_LEVELS = ("clean_deposit", "spill_tilt", "collision_intrude", "fast_aggressive")

# -- Right-arm pick/carry/deposit waypoint timing (base; scaled per level) --
HOVER = 0.14
GRASP_Z_OFFSET = 0.006
LIFT_HEIGHT_M = 0.11          # left-arm cup lift (matches make_cup_collect_safety_demo.py)
T_LIFT_MOVE = 1.2
T_LIFT_HOLD = 0.3
RIGHT_START_T = T_LIFT_MOVE + T_LIFT_HOLD + 0.1   # right arm waits for the cup lift to settle
SETTLE_TAIL_S = 1.0

# Base per-phase (duration_s, hold_s) values, scaled by params["speed_scale"]
# each episode -- see collect_episode's `phase_specs` construction (which
# also optionally inserts a "transit" phase between "lift" and "carry").
T_TILT_START = 1.8   # left-arm tilt burst start (after the lift settles + right arm underway)
TAIL_AFTER_TILT_S = 0.8

# Cubes biased toward the cup's side of the table (see scene docstring:
# "cube_4, cube_5 ... near/across the midline toward the cup") -- used by
# the collision-biased levels to make an arm-arm crossing likely.
NEAR_CUP_CUBE_IDX = (4, 5)


class _WaypointTrack:
    """Smoothstep-eased joint-space waypoint track (move `duration`, then
    hold `hold`), identical construction to
    scripts/make_cup_collect_safety_demo.py's JointWaypointTrack --
    reimplemented here (not imported) to keep this collector independent of
    that demo script's QP/video machinery."""

    def __init__(self, start_q: np.ndarray):
        self.start_q = np.asarray(start_q, dtype=float).copy()
        self.waypoints: list[tuple[np.ndarray, float, float]] = []

    def add(self, q_target, duration, hold):
        self.waypoints.append((np.asarray(q_target, dtype=float).copy(), float(duration), float(hold)))

    def sample(self, t: float) -> np.ndarray:
        q = self.start_q
        t0 = 0.0
        for q_target, duration, hold in self.waypoints:
            t_end_move = t0 + duration
            t_end_hold = t_end_move + hold
            if t <= t_end_move:
                s = 0.0 if duration <= 0 else float(np.clip((t - t0) / duration, 0.0, 1.0))
                s = s * s * (3.0 - 2.0 * s)
                return q + s * (q_target - q)
            elif t <= t_end_hold:
                return q_target
            q = q_target
            t0 = t_end_hold
        return q

    def bounds(self, names: list[str]) -> dict:
        out = {}
        t0 = 0.0
        for (q_target, duration, hold), name in zip(self.waypoints, names):
            move_end = t0 + duration
            hold_end = move_end + hold
            out[name] = (t0, move_end, hold_end)
            t0 = hold_end
        return out


def _smoothstep(t, t0, t1):
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    s = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def _gripper_state(model: mujoco.MjModel, data: mujoco.MjData, prefix: str) -> float:
    jid = model.joint(f"{prefix}_joint7").id
    qadr = model.jnt_qposadr[jid]
    return float(np.clip(data.qpos[qadr] / GRIPPER_JOINT_RANGE_M, 0.0, 1.0))


def _right_gripper_opening(t: float, bounds: dict) -> float:
    at_cube_move_end, at_cube_hold_end = bounds["at_cube"][1], bounds["at_cube"][2]
    release_move_end, release_hold_end = bounds["release"][1], bounds["release"][2]
    if t < at_cube_move_end:
        return 1.0
    if t < at_cube_hold_end:
        s = (t - at_cube_move_end) / max(at_cube_hold_end - at_cube_move_end, 1e-6)
        return float(1.0 - np.clip(s, 0.0, 1.0))
    if t < release_move_end:
        return 0.0
    if t < release_hold_end:
        s = (t - release_move_end) / max(release_hold_end - release_move_end, 1e-6)
        return float(np.clip(s, 0.0, 1.0))
    return 1.0


def _cube_pin_active(t: float, bounds: dict) -> bool:
    return bounds["at_cube"][2] <= t < bounds["release"][1]


def _cube_inside_cup(model, data, cube_bid, cup_bid, cup_radius, cup_height, margin=0.002) -> bool:
    """Same formula as cup_collect_task_utils.count_deposited_cubes, applied
    to a single cube (used for the per-step `cube_dropped` fumble label)."""
    cup_pos = data.xpos[cup_bid]
    cup_mat = data.xmat[cup_bid].reshape(3, 3)
    local = cup_mat.T @ (data.xpos[cube_bid] - cup_pos)
    radial = float(np.hypot(local[0], local[1]))
    height = float(local[2])
    return radial < cup_radius and -margin <= height < cup_height


def _sample_level_params(level: str, seed: int) -> dict:
    """Per-episode randomized parameters for `level` -- mirrors
    collection/perturbations.py's Random(seed)-per-episode convention (kept
    local since this task's behaviors have no home in perturbations.py's
    existing FAMILIES)."""
    rng = np.random.default_rng(seed)
    if level == "clean_deposit":
        p = dict(speed_scale=rng.uniform(0.9, 1.1), hover_clear=0.11, release_clear=0.045,
                  tilt_deg=0.0, tilt_dur_s=0.0, target_pool="any", transit_extra_z=0.30)
    elif level == "spill_tilt":
        p = dict(speed_scale=rng.uniform(0.9, 1.1), hover_clear=0.11, release_clear=0.045,
                  tilt_deg=rng.uniform(75.0, 100.0), tilt_dur_s=rng.uniform(0.7, 1.0), target_pool="any",
                  transit_extra_z=0.30)
    elif level == "collision_intrude":
        p = dict(speed_scale=rng.uniform(0.40, 0.55), hover_clear=0.02, release_clear=0.02,
                  tilt_deg=0.0, tilt_dur_s=0.0, target_pool="near_cup", transit_extra_z=0.0)
    elif level == "fast_aggressive":
        p = dict(speed_scale=rng.uniform(0.40, 0.55), hover_clear=0.025, release_clear=0.025,
                  tilt_deg=rng.uniform(45.0, 70.0), tilt_dur_s=rng.uniform(0.5, 0.8), target_pool="near_cup",
                  transit_extra_z=0.0)
    else:  # pragma: no cover
        raise AssertionError(level)
    return p


def _choose_target_cube(model, seed: int, pool: str) -> tuple[int, str]:
    ids = cct.cube_body_ids(model)
    names = [model.body(bid).name for bid in ids]
    if pool == "near_cup":
        candidates = [i for i, n in enumerate(names) if int(n.split("_")[1]) in NEAR_CUP_CUBE_IDX]
    else:
        candidates = list(range(len(ids)))
    idx = candidates[seed % len(candidates)]
    return ids[idx], names[idx]


def collect_episode(
    episode_index: int,
    seed: int,
    out_dir: Path,
    *,
    render_size: int = 64,
    render_every: int = 1,
    level: str | None = None,
) -> dict:
    if level is None:
        level = COLLECT_BEHAVIOR_LEVELS[seed % len(COLLECT_BEHAVIOR_LEVELS)]
    params = _sample_level_params(level, seed)

    model, _ = cct.load_yor_cup_collect_model()  # restore_finger_object_collision=False (the FIX)
    data = scene_setup.init_cup_grasped_state(model, seed=seed, cup_jitter_m=0.0)

    # -- CRITICAL PREREQUISITE: balls must start IN the cup (fixed loader). --
    spilled_at_grasp = int(cct.count_spilled(model, data))
    if spilled_at_grasp > 0:
        raise RuntimeError(
            f"[collect_cup_collect] POISON DETECTED: episode seed={seed} starts with "
            f"count_spilled={spilled_at_grasp} > 0 right after grasp -- the finger-object "
            f"collision fix did not take (or regressed). Aborting before writing any data."
        )

    dt = model.opt.timestep
    assert abs(dt - sc.DT) < 1e-12

    _, left_qposadr, left_dofadr = sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)
    _, right_qposadr, right_dofadr = sc.joint_addrs(model, sc.RIGHT_ARM_JOINTS)

    left_ee_body = model.body(scene_setup.EE_BODY_NAME).id
    right_ee_body = model.body("right_arm_gripper_base").id
    cup_body = model.body(cct.CUP_BODY_NAME).id

    q_grasp = data.qpos[left_qposadr].copy()

    # Tilt joint-space direction, computed ONCE at the settled grasp pose
    # (before any lift motion) -- same convention as eval/run_eval.py.
    Jr_pinv = scene_setup.left_ee_jac_rot_pinv(model, data, left_ee_body, left_dofadr)
    q_transfer_dir = Jr_pinv @ scene_setup.TRANSFER_ROT_AXIS

    cup_rel_pos0, cup_rel_quat0 = scene_setup.cup_rel_pose(model, data, left_ee_body, cup_body)

    left_ik = bt.ArmIK(model, "left")
    left_ik.last_q = q_grasp.copy()
    site_pos0 = data.site_xpos[left_ik.site_id].copy()
    q_lift, lift_err = left_ik.solve(data, site_pos0 + np.array([0.0, 0.0, LIFT_HEIGHT_M]))

    track_l = _WaypointTrack(q_grasp)
    track_l.add(q_lift, T_LIFT_MOVE, T_LIFT_HOLD)

    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.qpos[left_qposadr] = q_lift
    mujoco.mj_forward(model, scratch)
    ee_lift_pos = scratch.xpos[left_ee_body].copy()
    ee_lift_mat = scratch.xmat[left_ee_body].reshape(3, 3).copy()
    cup_lift_pos = ee_lift_pos + ee_lift_mat @ cup_rel_pos0
    cup_rim_z = cup_lift_pos[2] + cct.CUP_HEIGHT

    cube_body, cube_name = _choose_target_cube(model, seed, params["target_pool"])
    cube_xyz = data.xpos[cube_body].copy()

    right_ik = bt.ArmIK(model, "right")
    q_home = data.qpos[right_qposadr].copy()
    right_ik.last_q = q_home.copy()

    hover_clear, release_clear = params["hover_clear"], params["release_clear"]
    p_hover_cube = cube_xyz + np.array([0.0, 0.0, HOVER])
    p_at_cube = cube_xyz + np.array([0.0, 0.0, GRASP_Z_OFFSET])
    p_hover_cup = np.array([cup_lift_pos[0], cup_lift_pos[1], cup_rim_z + hover_clear])
    p_release = np.array([cup_lift_pos[0], cup_lift_pos[1], cup_rim_z + release_clear])

    q_hover_cube, e1 = right_ik.solve(data, p_hover_cube)
    q_at_cube, e2 = right_ik.solve(data, p_at_cube)
    q_lift_cube, e3 = right_ik.solve(data, p_hover_cube)

    # phase_specs: (name, q_target, base_duration_s, base_hold_s). "transit"
    # is an OPTIONAL extra waypoint (only for non-collision-biased levels,
    # params["transit_extra_z"] > 0): routes the carried cube well above
    # both arms' reach before descending onto the cup, instead of a direct
    # low joint-space sweep from `lift` straight to `hover_cup` -- without
    # it, that direct sweep reliably clips the LEFT arm/cup (this scene's
    # whole point, per scripts/make_cup_collect_safety_demo.py's docstring:
    # "the RIGHT arm sweeps close" -- WITHOUT a QP filter that is a near-
    # guaranteed contact, not just for the deliberately aggressive levels).
    # collision_intrude/fast_aggressive deliberately SKIP this (transit_extra_z=0)
    # to keep their low, fast, close sweep intact.
    phase_specs: list[tuple[str, np.ndarray, float, float]] = [
        ("hover_cube", q_hover_cube, 1.0, 0.10),
        ("at_cube", q_at_cube, 0.7, 0.20),
        ("lift", q_lift_cube, 0.6, 0.10),
    ]
    e_transit = None
    transit_extra_z = params["transit_extra_z"]
    if transit_extra_z > 0.0:
        mid_xy = 0.5 * (p_hover_cube[:2] + p_hover_cup[:2])
        transit_z = max(p_hover_cube[2], p_hover_cup[2]) + transit_extra_z
        p_transit = np.array([mid_xy[0], mid_xy[1], transit_z])
        q_transit, e_transit = right_ik.solve(data, p_transit)
        phase_specs.append(("transit", q_transit, 0.6, 0.05))

    q_hover_cup, e4 = right_ik.solve(data, p_hover_cup)
    q_release, e5 = right_ik.solve(data, p_release)
    phase_specs.append(("carry", q_hover_cup, 1.3, 0.25))
    phase_specs.append(("release", q_release, 0.5, 0.20))
    phase_specs.append(("park", q_home, 0.8, 0.30))

    speed_scale = params["speed_scale"]
    phase_names = [n for n, _, _, _ in phase_specs]
    track_r = _WaypointTrack(q_home)
    for _name, qt, dur, hold in phase_specs:
        track_r.add(qt, max(0.12, dur * speed_scale), max(0.05, hold * speed_scale))
    bounds = track_r.bounds(phase_names)
    right_total_s = bounds["park"][2]

    tilt_deg, tilt_dur_s = params["tilt_deg"], params["tilt_dur_s"]
    total_s = RIGHT_START_T + right_total_s + SETTLE_TAIL_S
    if tilt_deg > 0.0:
        total_s = max(total_s, T_TILT_START + tilt_dur_s + TAIL_AFTER_TILT_S)
    num_steps = int(round(total_s / dt))

    left_j7 = model.actuator("left_arm_joint7_pos").id
    left_j8 = model.actuator("left_arm_joint8_pos").id
    right_finger_act = bt.finger_actuators(model, "right")

    sc.disable_left_arm_position_servos(model)
    sc.disable_right_arm_position_servos(model)

    ego_cams = []
    for key, cname in (("rgb_wrist_left_ego", "left_arm_egocentric"), ("rgb_wrist_right_ego", "right_arm_egocentric")):
        if model.camera(cname).id >= 0:
            ego_cams.append((key, cname))

    writer = EpisodeWriter()
    ever_collision = False
    ever_spilled = False
    cube_pin_rel = None
    cube_home_z = float(cube_xyz[2])

    renderer = mujoco.Renderer(model, height=render_size, width=render_size)
    try:
        for step in range(num_steps):
            t = step * dt

            data.ctrl[left_j7] = 0.0
            data.ctrl[left_j8] = 0.0
            t_r = t - RIGHT_START_T
            grip_r = _right_gripper_opening(t_r, bounds) if t_r >= 0 else 1.0
            bt.set_gripper(data, right_finger_act, grip_r)

            mujoco.mj_forward(model, data)
            scene_setup.reattach_cup_rigidly(model, data, left_ee_body, cup_body, cup_rel_pos0, cup_rel_quat0)

            pin_now = t_r >= 0 and _cube_pin_active(t_r, bounds)
            if pin_now:
                if cube_pin_rel is None:
                    cube_pin_rel = scene_setup.cup_rel_pose(model, data, right_ee_body, cube_body)
                scene_setup.reattach_cup_rigidly(model, data, right_ee_body, cube_body, *cube_pin_rel)
            else:
                if t_r < 0 or t_r >= bounds["release"][1]:
                    cube_pin_rel = None
            mujoco.mj_forward(model, data)

            q_left_des = track_l.sample(t)
            if tilt_deg > 0.0:
                tilt_off = scene_setup.transfer_tilt_offset(
                    t, start_s=T_TILT_START, dur_s=tilt_dur_s, amp_rad=np.deg2rad(tilt_deg))
                q_left_des = q_left_des + q_transfer_dir * tilt_off
            q_right_des = q_home if t_r < 0 else track_r.sample(t_r)

            tau_l, _, _ = scene_setup.task_torque(
                model, data, q_left_des, left_dofadr, left_qposadr, left_ee_body, cup_body,
                couple_payload=True, freeze=False)
            tau_r, _, _ = scene_setup.task_torque(
                model, data, q_right_des, right_dofadr, right_qposadr, right_ee_body, cube_body,
                couple_payload=pin_now, freeze=False)
            data.qfrc_applied[left_dofadr] = np.clip(tau_l, -sc.TAU_MAX, sc.TAU_MAX)
            data.qfrc_applied[right_dofadr] = np.clip(tau_r, -sc.TAU_MAX, sc.TAU_MAX)

            collision = scene_setup.left_right_contact(model, data)
            spilled_count_t = int(cct.count_spilled(model, data))
            spilled_now = spilled_count_t > 0
            deposited_count_t = int(cct.count_deposited_cubes(model, data))
            ever_collision = ever_collision or collision
            ever_spilled = ever_spilled or spilled_now

            cube_z_now = float(data.xpos[cube_body][2])
            cube_dropped_now = bool(
                t_r >= bounds["release"][1]
                and not _cube_inside_cup(model, data, cube_body, cup_body, cct.CUP_RADIUS, cct.CUP_HEIGHT)
                and cube_z_now < cube_home_z + 0.03
            )

            if step % render_every == 0 or step == 0:
                renderer.update_scene(data, camera=cam_mod.ZED_CAM)
                zed_img = renderer.render().copy()
                imgs = {"rgb_head": zed_img, "rgb_wrist_right": zed_img.copy(), "rgb_zed": zed_img.copy()}
                for key, cname in ego_cams:
                    renderer.update_scene(data, camera=cname)
                    imgs[key] = renderer.render().copy()

            cup_quat = data.xquat[cup_body].copy()
            cup_pose = np.concatenate([data.xpos[cup_body], cup_quat]).astype(np.float32)

            step_fields = dict(
                rgb_head=imgs["rgb_head"],
                rgb_wrist_right=imgs["rgb_wrist_right"],
                rgb_zed=imgs["rgb_zed"],
                left_q=data.qpos[left_qposadr].astype(np.float32).copy(),
                left_qd=data.qvel[left_dofadr].astype(np.float32).copy(),
                right_q=data.qpos[right_qposadr].astype(np.float32).copy(),
                right_qd=data.qvel[right_dofadr].astype(np.float32).copy(),
                left_action=np.concatenate([q_left_des, [0.0, 0.0]]).astype(np.float32),
                right_action=np.concatenate([q_right_des, data.ctrl[right_finger_act]]).astype(np.float32),
                left_gripper_state=np.float32(_gripper_state(model, data, "left_arm")),
                right_gripper_state=np.float32(_gripper_state(model, data, "right_arm")),
                box_pose=cup_pose,
                collision=bool(collision),
                dropped=bool(spilled_now),
                spilled_count=np.int32(spilled_count_t),
                deposited_count=np.int32(deposited_count_t),
                cube_dropped=bool(cube_dropped_now),
            )
            for key, _ in ego_cams:
                step_fields[key] = imgs[key]
            writer.add_step(**step_fields)

            mujoco.mj_step(model, data)

        final_spilled_count = int(cct.count_spilled(model, data))
        final_deposited_count = int(cct.count_deposited_cubes(model, data))
        target_deposited = _cube_inside_cup(model, data, cube_body, cup_body, cct.CUP_RADIUS, cct.CUP_HEIGHT)
        grasp_success = not ever_spilled  # repurposed: "cup contents intact"
        success = grasp_success and not ever_collision and target_deposited

        spec = PerturbationSpec(
            family="cup_collect_behavior", level=level, seed=int(seed),
            parameters={**params, "target_cube": cube_name},
        )
        metadata = build_episode_metadata(
            episode_index=episode_index,
            seed=seed,
            perturbation=spec,
            policy_name=f"cup_collect_scripted_{level}_yor_v1",
            num_steps=len(writer),
            success=success,
            grasp_success=grasp_success,
            collision=ever_collision,
            spilled_count=final_spilled_count,
        )
        metadata["spilled_at_grasp"] = spilled_at_grasp
        metadata["final_deposited_count"] = final_deposited_count
        metadata["target_cube_deposited"] = bool(target_deposited)
        metadata["target_cube"] = cube_name
        metadata["left_lift_ik_err_m"] = float(lift_err)
        right_ik_errs = dict(hover_cube=float(e1), at_cube=float(e2),
                              lift=float(e3), hover_cup=float(e4), release=float(e5))
        if e_transit is not None:
            right_ik_errs["transit"] = float(e_transit)
        metadata["right_ik_errs_m"] = right_ik_errs
        episode_dir = out_dir / metadata["episode_id"]
        writer.save(episode_dir, metadata)
        return metadata
    finally:
        renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "collection" / "data" / "cup_collect_v1")
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--render-size", type=int, default=64,
                         help="MUST be 64 to match world_model/dataset_adapter.py's hardcoded (T,64,64,6) assert")
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--level", type=str, default=None, choices=(None, *COLLECT_BEHAVIOR_LEVELS),
                         help="force one behavior for all episodes; default cycles seed %% 4 for a mix")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = []
    for i in range(args.num_episodes):
        seed = args.seed_start + i
        meta = collect_episode(
            i, seed, args.out_dir,
            render_size=args.render_size, render_every=args.render_every, level=args.level,
        )
        results.append(meta)
        print(
            f"[collect_cup_collect] episode={meta['episode_id']} seed={seed} level={meta['perturbation_level']} "
            f"target={meta['target_cube']} spilled_at_grasp={meta['spilled_at_grasp']} "
            f"success={meta['success']} spilled_count={meta['spilled_count']} "
            f"deposited={meta['final_deposited_count']} collision={meta['collision']} steps={meta['num_steps']}",
            flush=True,
        )
    dt = time.time() - t0
    n_spilled = sum(1 for r in results if r["spilled_count"] > 0)
    n_collision = sum(1 for r in results if r["collision"])
    n_deposited = sum(1 for r in results if r["final_deposited_count"] > 0)
    n_clean = sum(1 for r in results if r["spilled_count"] == 0 and not r["collision"])
    print(f"[collect_cup_collect] wrote {len(results)} episodes to {args.out_dir} in {dt:.1f}s "
          f"(spilled={n_spilled} collision={n_collision} deposited={n_deposited} clean={n_clean})", flush=True)


if __name__ == "__main__":
    main()
