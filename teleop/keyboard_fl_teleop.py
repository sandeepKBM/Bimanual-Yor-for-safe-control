#!/usr/bin/env python
"""teleop/keyboard_fl_teleop.py -- browser-keyboard teleop of the RIGHT arm's
end-effector, with the LEFT arm holding a lifted cup, both arms driven by the
feedback-linearized cooperative MPC (experiments/ab_smooth_vs_bangbang/mpc_fl.py
FLMPC) so collision / cup-upright / torque constraints stay HARD -- the user
cannot command a collision or tip the cup.

SIMULATION ONLY. No robot SDK / ROS / serial / CAN is imported or reachable
from this file. The only sockets opened are (1) an aiohttp HTTP+WebSocket
server bound for SSH port-forwarding (keyboard page) and (2) an optional
mjviser/viser HTTP+WebSocket server (3D view), both purely local visualization
services -- nothing here ever connects out to a robot, motor driver, or any
non-loopback remote machine.

START STATE -- how it's built (mirrors scripts/make_cup_collect_safety_demo.py)
--------------------------------------------------------------------------
`TeleopSim.__init__` calls `demo.build_episode(seed=...)` (unmodified) to get
the standard cup-collect scene/model/data + info dict, then calls
`demo.run_episode(info, controller="flmpc", max_steps=lift_steps)` where
`lift_steps = round(demo.RIGHT_START_T / sc.DT)` -- i.e. it runs the demo's
OWN v4c-ramp lift phase (t < RIGHT_START_T: LEFT arm grasps + lifts the cup on
its computed-torque ramp + QP filter; RIGHT arm sits at its `q_home` the whole
time) and stops there. `info["data"]` is mutated in place by run_episode, so
after that call `info["model"]`/`info["data"]` ARE the live sim objects at the
"cup lifted and settled, right arm at home" instant -- exactly the start state
this task asks for. We snapshot qpos/qvel/ctrl/time right after this call for
the `R` reset key.

From that instant on, this module takes over the per-step control loop itself
(it does NOT keep calling demo.run_episode): every tick it mirrors EXACTLY the
demo's `controller == "flmpc"` branch (scripts/make_cup_collect_safety_demo.py
lines ~1667-1720) --
  - M_full = sc.full_mass_matrix(model, data); M_box, bias_box =
    sc.payload_reflected_terms(model, data, left_ee_body, cup_body, left_dofadr)
    (cup-payload reflection onto the left/ego arm's dynamics);
  - pairs = sc.all_pair_barriers(model, data, left_dofadr, right_dofadr,
    extra_ego_keypoints=info["cup_barrier_keypoints"]) (arm-arm + cup HOCBF
    keypoint pairs);
  - optional table-floor HOCBF rows via sc.table_plane_barrier_rows, same as
    the demo's `table_floor=True` path;
  - fl.step(model, data, left_qposadr, left_dofadr, right_qposadr,
    right_dofadr, q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref, pairs,
    M_ego_extra=M_box, bias_ego_extra=bias_box, table_rows=...,
    left_ee_body=left_ee_body, tilt_max_rad=demo.CUP_TILT_MAX_RAD,
    leash_ee_body=left_ee_body, leash_max_m=demo.CUP_LEASH_M,
    cup_cone_facets=demo.CUP_CONE_FACETS, ego_hold=...)
    -- returns (tau_ego=tau_left, tau_int=tau_right, feasible);
  - tau clipped+rate/low-pass filtered via sc.TorqueCommandFilter (same
    FILTER_LOWPASS_ALPHA/FILTER_RATE_LIMIT_NM_S the demo uses) then written to
    data.qfrc_applied, then mujoco.mj_step -- the demo's `flmpc` branch NEVER
    overrides fl.step's returned torque with the crude INFEAS_BRAKE_KD brake
    (see AGENTS.md's "known failure modes" note -- that override is v4c-only),
    so this module doesn't either.

The ONLY thing replaced relative to the demo's flmpc branch is WHICH arm's
reference (q_ego_ref/q_int_ref) is scripted vs. keyboard-driven, and whether
the ego (left) position-tracking cost is released (`ego_hold`):
  - active arm == "right" (default): q_ego_ref = tile(info["q_lift"]) (constant
    hold, byte-identical to the demo), ego_hold=False (byte-identical to the
    demo's rationale: the left/cup arm has no restoring pull, it only yields
    and stays put -- see mpc_fl.FLMPC.step's `ego_hold` docstring); q_int_ref =
    tile(IK solution of the keyboard-driven right-EE target) instead of the
    demo's scripted `track_r` sample.
  - active arm == "left": q_ego_ref = tile(IK solution of the keyboard-driven
    left-EE target), ego_hold=True (so the tracking cost actually pulls the
    arm toward the moving keyboard target -- ego_hold=False would zero that
    cost and the arm would never follow); q_int_ref = tile(the right arm's own
    last IK solution, frozen -- it holds still while not being driven).
  Either way, the cup-upright (tilt) and cup-arm leash CONSTRAINT ROWS are
  passed to fl.step UNCONDITIONALLY on every call (same as the demo) -- they
  are QP rows, not a cost term, so they stay enforced regardless of which arm
  is being teleoperated or of `ego_hold`. This satisfies the task's
  requirement that the left arm's cup-upright rows stay enforced even when
  IT is the one being driven by keyboard.

KNOWN LIMITATION: mpc_fl.FLMPC's cup-arm displacement LEASH is anchored at
`q_ego_ref[0]` internally (see mpc_fl.py step()'s block (4)) -- when the LEFT
arm is the active/teleoperated arm, q_ego_ref[0] IS the moving keyboard target,
so the leash's excursion bound follows the target rather than bounding
excursion from the original lift pose. The cup-UPRIGHT tilt bound (the
primary spill-relevant hard constraint) is anchored to world-vertical, not to
q_ego_ref, so it is unaffected and stays fully enforced in both modes. This is
a property of mpc_fl.FLMPC's existing API (not modified here, per this task's
scope), not a bug introduced by this file.

VIEW: mjviser (see teleop/mjviser_panel.py for the reference pattern). That
module's own `Viewer` class owns its own physics stepping loop (calls
`step_fn(model, data)` on ITS OWN timer), which cannot share ownership of
`mj_step` with this module's 200 Hz FL-MPC control loop. Its docstring says
so explicitly: "For full control over the loop, use `ViserMujocoScene`
directly and call `update_from_mjdata` ... yourself." -- confirmed by
inspecting `mjviser.ViserMujocoScene.__init__`/`update_from_mjdata` signatures.
So this module uses `mjviser.ViserMujocoScene` directly: our own control
thread calls `scene.update_from_mjdata(data)` after every `mj_step` (throttled
to ~20 Hz to bound bandwidth) -- a genuinely externally-stepped model/data,
no EGL/GPU rendering needed (viser sends geometry, not rendered pixels, so
this stays CPU-only and does not need MUJOCO_GL/CUDA_VISIBLE_DEVICES).

REAL-TIME VIA REPLAN DECIMATION (--replan-every N)
--------------------------------------------------
A full FL-MPC re-solve (IK + all_pair_barriers over ~625 keypoint pairs +
table rows + fl.step's QP) costs ~25-30 ms wall, for 5 ms of sim -- so
re-solving every sim step runs at ~1/6 real time. Instead the full solve
runs only every N sim steps; between solves the controller ZERO-ORDER-HOLDS
the planned first-block ACCELERATION (a_ego, a_int = fl.a_ego_prev /
fl.a_int_prev after the solve) and re-realizes the feedback-linearizing
torque tau = M(q) a + c(q, qdot) from the CURRENT state every sim step
(mass matrix + bias only, ~0.05 ms), scaled toward a=0 by FLMPC's exact
closed-form torque-box backstop (_torque_safe_scale) and then filtered +
clipped to +-TAU_MAX exactly like a solve tick. Horizon context: FLMPC here is
NC=4 blocks x L=8 steps = NP=32 steps x 5 ms = 160 ms of look-ahead, and its
decision variable holds the acceleration CONSTANT within each L=8-step
(40 ms) block -- so for N <= L the held acceleration IS the plan's own
committed first block (the QP already verified the clearance / table /
torque-box / cup-tilt rows along the predicted trajectory under exactly
that constant-acceleration first block). N > L is refused. N=1 reproduces
the original every-step solve. On hold ticks the HUD's h_min/feasible are
the values from the last solve (<= N-1 steps old); contact-based collision
detection still runs every step.

GRASP MODES (--grasp-mode {faked,real})
---------------------------------------
The start state + the per-step grasp maintenance come from a pluggable
builder (see GRASP_BUILDERS / build_start_state_faked /
build_start_state_realgrasp). "faked" (default) is the demo's kinematic
rigid re-attach of the cup to the left EE every step -- behavior identical
to before. "real" is a STUB that raises NotImplementedError: another work
item builds the real friction grasp + FL-MPC-driven lift; its builder must
return the StartState dict documented on build_start_state_realgrasp.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import mujoco

# --- path setup: identical pattern to scripts/make_cup_collect_safety_demo.py ---
REPO_ROOT = Path(__file__).resolve().parents[1]
AB4_DIR = REPO_ROOT / "experiments" / "ab_smooth_vs_bangbang"
EVAL_DIR = REPO_ROOT / "eval"
for _p in (str(REPO_ROOT), str(AB4_DIR), str(EVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sim_common as sc  # noqa: E402  (read-only import)
import mpc_fl  # noqa: E402  (read-only import; FLMPC used unmodified)
import run_ab4  # noqa: E402  (read-only import; left_right_geom_sets only)
from teleop import bimanual_teleop as bt  # noqa: E402  (read-only import; BimanualIK/set_gripper/finger_actuators)
from scripts import make_cup_collect_safety_demo as demo  # noqa: E402  (read-only import; build_episode/run_episode/constants)

from aiohttp import web  # noqa: E402

# ---------------------------------------------------------------------------
# Key map (exact, per task spec)
# ---------------------------------------------------------------------------
DEADMAN_KEY = " "        # Space, HOLD
GRIPPER_KEY = "k"
SWITCH_ARM_KEY = "tab"
SPEED_DOWN_KEY = "-"
SPEED_UP_KEY = "="
RESET_KEY = "r"
ESTOP_KEY = "escape"

# world-frame axis + sign per position key (first-listed letter is +).
POS_KEYS = {"w": (0, +1.0), "s": (0, -1.0),
            "a": (1, +1.0), "d": (1, -1.0),
            "q": (2, +1.0), "e": (2, -1.0)}
# world-frame rot axis (roll=x, pitch=y, yaw=z) + sign per key.
ROT_KEYS = {"z": (0, +1.0), "x": (0, -1.0),
            "t": (1, +1.0), "g": (1, -1.0),
            "c": (2, +1.0), "v": (2, -1.0)}
ALL_MOTION_KEYS = set(POS_KEYS) | set(ROT_KEYS)
TOGGLE_KEYS = {GRIPPER_KEY, SWITCH_ARM_KEY, SPEED_DOWN_KEY, SPEED_UP_KEY, RESET_KEY, ESTOP_KEY}
BOUND_KEYS = ALL_MOTION_KEYS | TOGGLE_KEYS | {DEADMAN_KEY}

LIN_SPEED_DEFAULT = 0.05     # [m/s]
LIN_SPEED_MIN = 0.0125
LIN_SPEED_MAX = 0.2
ANG_SPEED_DEFAULT = np.deg2rad(30.0)   # [rad/s]
STALE_KEY_S = 0.25           # server-side auto-release: no refresh within this -> released
KEY_REFRESH_HINT_S = 0.10    # what the JS is told to use (for the HUD/docstring only)
WORKSPACE_HALF_M = 0.35      # +/- box around each side's initial EE target (x,y)
SAFETY_OVERRIDE_ERR_M = 0.03     # realized-vs-target EE deviation that flags "SAFETY OVERRIDING"
SAFETY_OVERRIDE_H_M = 0.04       # worst arm-arm clearance below which we flag "SAFETY OVERRIDING"
REPLAN_EVERY_DEFAULT = 8         # full FL-MPC re-solve every N sim steps (N*5 ms); see module docstring
REPLAN_NEAR_M_DEFAULT = 0.0      # solve EVERY step while last arm-arm h_min < this (or infeasible); 0 = off
K_DOUBLE_TAP_S = 1.0            # cup-holding arm: K must be pressed twice within this to open (drop)
MAX_LAG_S = 0.10                 # sim loop: catch up after bursty solve ticks, resync only past this lag


def _light_forward(model, data) -> None:
    """Subset of mj_forward for hold ticks: kinematics, tendons, qM (mj_makeM:
    CRB + armature), com velocities, passive forces (damping, gravcomp), RNE
    bias. No collision/constraint/acceleration stages (mj_step redoes the full
    forward pass itself)."""
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_tendon(model, data)
    mujoco.mj_makeM(model, data)
    mujoco.mj_comVel(model, data)
    mujoco.mj_passive(model, data)
    mujoco.mj_rne(model, data, 0, data.qfrc_bias)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _axis_angle_quat(axis: int, angle: float) -> np.ndarray:
    ax = np.zeros(3)
    ax[axis] = 1.0
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, ax, angle)
    return q


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def _mat2quat(mat: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.reshape(9))
    return q


# ---------------------------------------------------------------------------
# Thread-safe key state with server-side stuck-key auto-release.
# ---------------------------------------------------------------------------
class KeyState:
    def __init__(self):
        self._lock = threading.Lock()
        self._last_refresh: dict[str, float] = {}

    def refresh(self, key: str) -> bool:
        """Mark `key` as held right now. Returns True iff this is a FRESH
        press (key was not already considered held) -- used for edge-
        triggered toggle actions (TAB/K/-/=/R/Esc)."""
        now = time.monotonic()
        with self._lock:
            was_held = key in self._last_refresh and (now - self._last_refresh[key]) < STALE_KEY_S
            self._last_refresh[key] = now
            return not was_held

    def release(self, key: str) -> None:
        with self._lock:
            self._last_refresh.pop(key, None)

    def release_all(self) -> None:
        with self._lock:
            self._last_refresh.clear()

    def held(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            t = self._last_refresh.get(key)
            if t is None:
                return False
            if (now - t) >= STALE_KEY_S:
                del self._last_refresh[key]
                return False
            return True

    def snapshot_held(self) -> set[str]:
        now = time.monotonic()
        with self._lock:
            stale = [k for k, t in self._last_refresh.items() if (now - t) >= STALE_KEY_S]
            for k in stale:
                del self._last_refresh[k]
            return set(self._last_refresh.keys())


def apply_ws_message(sim: "TeleopSim", msg: dict) -> None:
    """Single entry point for a decoded {"type", "key"} client message --
    used by BOTH the real aiohttp WebSocket handler and the headless smoke
    test, so the smoke test exercises the exact same code path a browser
    would drive."""
    mtype = msg.get("type")
    key = str(msg.get("key", "")).lower()
    if mtype == "keydown":
        fresh = sim.key_state.refresh(key)
        if fresh and key in TOGGLE_KEYS:
            sim.on_toggle_key(key)
    elif mtype == "keyup":
        sim.key_state.release(key)
    elif mtype == "release_all":
        sim.key_state.release_all()


# ---------------------------------------------------------------------------
# Grasp-mode start-state builders (pluggable; the control loop is agnostic).
# ---------------------------------------------------------------------------
def build_start_state_faked(seed: int = 0, ee_speed_limit: bool = True) -> dict:
    """DEFAULT grasp mode: the demo's kinematic rigid re-attach (see module
    docstring "START STATE"). Returns a StartState dict -- the contract is
    documented on build_start_state_realgrasp."""
    info = demo.build_episode(seed=seed, ee_speed_limit=ee_speed_limit)
    lift_steps = int(round(demo.RIGHT_START_T / sc.DT))
    # Run the demo's OWN (unmodified) lift phase up to (not including) the
    # crossing/flmpc phase -- see module docstring "START STATE" above.
    demo.run_episode(info, controller="flmpc", qp_enabled=True, capture=False,
                      max_steps=lift_steps, cup_in_barrier=True, table_floor=True,
                      ee_speed_limit=ee_speed_limit)
    left_ee_body, cup_body = info["left_ee_body"], info["cup_body"]
    rel_pos0, rel_quat0 = info["cup_rel_pos0"], info["cup_rel_quat0"]

    def grasp_hold_fn(model, data, gripper_open):
        # Faked grasp: re-pin the cup to the left EE every sim step,
        # unconditionally (byte-identical to the pre-grasp-mode loop; the
        # finger ctrl has already been written from `gripper_open`).
        demo.scene_setup.reattach_cup_rigidly(model, data, left_ee_body, cup_body, rel_pos0, rel_quat0)

    def payload_fn(model, data):
        return sc.payload_reflected_terms(model, data, left_ee_body, cup_body, info["left_dofadr"])

    return dict(info=info, lift_steps=lift_steps, holder_side="left",
                gripper_open0={"left": 0.0, "right": 1.0},
                grasp_hold_fn=grasp_hold_fn, payload_fn=payload_fn,
                cup_keypoints_fn=None, tilt_body=left_ee_body, fl=None)


def build_start_state_realgrasp(seed: int = 0, ee_speed_limit: bool = True) -> dict:
    """STUB -- real friction grasp + FL-MPC-driven lift (--grasp-mode real).

    TODO(real-grasp integration point): another work item is building the
    real friction grasp (finger pads actually squeezing the cup wall, no
    kinematic re-attach) + an FL-MPC-driven lift. Plug it in HERE: build the
    scene, run the grasp + lift, and return a StartState dict at the
    "cup lifted and settled, right arm at home" instant. TeleopSim consumes
    ONLY these keys (so nothing else in this file needs to change):

      info            dict, same schema as demo.build_episode()'s return:
                      model, data (LIVE, already at the post-lift state),
                      left/right_qposadr, left/right_dofadr, left/right_ee_body,
                      cup_body, q_lift (6,) left hold pose, q_home (6,) right
                      pose, table_top_z, cup_barrier_keypoints (list of
                      (body_id, local_offset(3,), radius) on the LEFT EE body).
      lift_steps      int, sim steps already consumed by the grasp+lift.
      holder_side     "left" -- which arm holds the cup (K double-tap guard).
      gripper_open0   {"left": 0.0, "right": 1.0} initial gripper commands.
      grasp_hold_fn   callable(model, data, gripper_open: dict) -> None, called
                      EVERY sim step after bt.set_gripper wrote the finger ctrl
                      from gripper_open and after mj_forward, before torques
                      are computed. Real mode: maintain the grip force here
                      (e.g. overwrite the holder's finger ctrl with a squeeze
                      command while gripper_open[holder] < 0.5); do NOT
                      re-pin the cup.
      payload_fn      callable(model, data) -> (M_box (6,6), bias_box (6,)) or
                      None: the cup's reflected dynamics on the left arm for
                      FL (faked: sc.payload_reflected_terms rigid-grasp model).
      cup_keypoints_fn  callable(model, data) -> keypoint list (same format as
                      info["cup_barrier_keypoints"], offsets on the LEFT EE
                      body, recomputed from the ACTUAL cup-body pose relative
                      to the EE so a slipped cup is still covered), or None
                      to use the static info["cup_barrier_keypoints"].
      tilt_body       body id whose world-up axis the HUD's cup_tilt reads
                      (real: info["cup_body"], the actual cup).
      fl              an already-built/warmed mpc_fl.FLMPC to reuse, or None.
    """
    raise NotImplementedError(
        "--grasp-mode real is a stub: build_start_state_realgrasp() must be provided by the "
        "real-friction-grasp + FL-MPC-lift work item (see its docstring for the StartState "
        "contract). Use --grasp-mode faked.")


GRASP_BUILDERS = {"faked": build_start_state_faked, "real": build_start_state_realgrasp}


# ---------------------------------------------------------------------------
# The simulation + control loop.
# ---------------------------------------------------------------------------
class TeleopSim:
    def __init__(self, seed: int = 0, ee_speed_limit: bool = True,
                 replan_every: int = REPLAN_EVERY_DEFAULT, grasp_mode: str = "faked",
                 replan_near_m: float = REPLAN_NEAR_M_DEFAULT):
        start = GRASP_BUILDERS[grasp_mode](seed=seed, ee_speed_limit=ee_speed_limit)
        info = start["info"]
        lift_steps = int(start["lift_steps"])
        self.grasp_mode = grasp_mode
        self.holder_side = start["holder_side"]
        self._gripper_open0 = dict(start["gripper_open0"])
        self._grasp_hold_fn = start["grasp_hold_fn"]
        self._payload_fn = start["payload_fn"]
        self._cup_keypoints_fn = start["cup_keypoints_fn"]
        self._tilt_body = start["tilt_body"]

        self.info = info
        self.model = model = info["model"]
        self.data = data = info["data"]
        self.left_qposadr = info["left_qposadr"]
        self.left_dofadr = info["left_dofadr"]
        self.right_qposadr = info["right_qposadr"]
        self.right_dofadr = info["right_dofadr"]
        self.left_ee_body = info["left_ee_body"]
        self.right_ee_body = info["right_ee_body"]
        self.cup_body = info["cup_body"]
        self.q_lift = info["q_lift"].copy()
        self.q_home = info["q_home"].copy()
        self.table_top_z = info["table_top_z"]
        self.cup_kps_for_barrier = info["cup_barrier_keypoints"]

        self.left_table_kps = sc.barrier_keypoints(model, sc.EGO_PREFIX)
        self.right_table_kps = sc.barrier_keypoints(model, sc.INT_PREFIX)
        self.left_geoms, self.right_geoms = run_ab4.left_right_geom_sets(model)

        self.ik = bt.BimanualIK(model)
        self.left_gripper_act = bt.finger_actuators(model, "left")
        self.right_gripper_act = bt.finger_actuators(model, "right")

        self.fl = start["fl"] if start["fl"] is not None else mpc_fl.FLMPC(model, NC=4, L=8, W_POS_EGO=800.0)
        replan_every = int(replan_every)
        if not (1 <= replan_every <= self.fl.L):
            raise ValueError(f"--replan-every must be in [1, L={self.fl.L}] (the FL-MPC's constant-"
                             f"acceleration first block); got {replan_every}")
        self.replan_every = replan_every
        self.replan_near_m = float(replan_near_m)
        self._a_hold = None              # (a_left, a_right) planned first-block accel, ZOH'd between solves
        self._ticks_since_replan = 0
        self.replan_count = 0
        self._last_feasible = True
        self._last_h_min = None
        self._ik_err = 0.0
        # all_pair_barriers orders pairs ego-major with the extra (cup)
        # keypoints appended AFTER the arm keypoints, so the first
        # n_ego_arm*n_int pairs are exactly the arm-arm pairs
        # sc.min_pair_distance scans -- min(h) over them == its h_min, for free.
        self._n_arm_pairs = len(sc.barrier_keypoints(model, sc.EGO_PREFIX)) * \
            len(sc.barrier_keypoints(model, sc.INT_PREFIX))
        self.rtf = None                   # measured sim-s per wall-s (set by the sim thread)
        # FLMPC.__init__ already JIT-warms its own `_assemble_fl` kernel, but
        # `sc.solve_constrained_box_qp` (the separate njit box-QP solver
        # fl.step() calls) is NOT warmed by that -- our lift-phase construction
        # never actually calls fl.step() (t_r never reaches >=0 during the
        # capped lift-only run_episode call above), so without this the FIRST
        # real tick() would pay the JIT compile cost live, stalling for enough
        # wall-clock time to trip the 250ms stuck-key auto-release. Same
        # warmup call scripts/make_cup_collect_safety_demo.py's own --controller
        # v7c setup makes for the same underlying solver.
        from controller_core.constrained_box_qp import numba_warmup as _box_qp_warmup
        _box_qp_warmup()

        self.tau_filter_l = sc.TorqueCommandFilter(6, sc.FILTER_LOWPASS_ALPHA,
                                                     np.full(6, sc.FILTER_RATE_LIMIT_NM_S))
        self.tau_filter_r = sc.TorqueCommandFilter(6, sc.FILTER_LOWPASS_ALPHA,
                                                     np.full(6, sc.FILTER_RATE_LIMIT_NM_S))

        # world-"up" expressed in the left EE's local frame at the settled
        # q_lift hold pose -- SAME "vertical" definition mpc_fl.FLMPC's own
        # cup-upright rows and the demo's own logging use.
        _scratch = mujoco.MjData(model)
        _scratch.qpos[:] = data.qpos
        _scratch.qpos[self.left_qposadr] = self.q_lift
        mujoco.mj_forward(model, _scratch)
        self._cup_axis_local = (_scratch.xmat[self._tilt_body].reshape(3, 3).T
                                 @ np.array([0.0, 0.0, 1.0]))

        self.key_state = KeyState()
        self.lock = threading.RLock()   # protects data/targets/hud vs. the ws thread reading hud

        self.active_side = "right"
        self.speed_scale = 1.0
        self.estop = False
        self._k_armed_t = None           # monotonic time of the first K tap on the cup-holding arm
        self._warning = ""
        self._warning_until = 0.0

        self.gripper_open = dict(self._gripper_open0)  # 0=closed (holding cup), 1=open
        self.q_ref = {"left": self.q_lift.copy(), "right": self.q_home.copy()}
        self.target_pos = {}
        self.target_quat = {}
        self.workspace_center = {}
        for side in ("left", "right"):
            site_id = self.ik.arms[side].site_id
            self.target_pos[side] = data.site_xpos[site_id].copy()
            self.target_quat[side] = _mat2quat(data.site_xmat[site_id])
            self.workspace_center[side] = data.site_xpos[site_id].copy()

        # Snapshot for the R reset key -- the "post-lift start state".
        self._snapshot = dict(qpos=data.qpos.copy(), qvel=data.qvel.copy(),
                               ctrl=data.ctrl.copy(), time=float(data.time))

        self.lift_steps = lift_steps
        self.t_sim = float(data.time)
        self.step_count = lift_steps
        self.hud = {}
        self._update_hud(feasible=True, h_min=None, collision=False, track_err=0.0)

    # -- toggle-key actions (edge-triggered, see apply_ws_message) --
    def on_toggle_key(self, key: str) -> None:
        with self.lock:
            if key == SWITCH_ARM_KEY:
                self.active_side = "left" if self.active_side == "right" else "right"
                site_id = self.ik.arms[self.active_side].site_id
                # re-anchor the newly-active side's target at its CURRENT
                # realized EE pose so switching never snaps the arm.
                self.target_pos[self.active_side] = self.data.site_xpos[site_id].copy()
                self.target_quat[self.active_side] = _mat2quat(self.data.site_xmat[site_id])
                self.workspace_center[self.active_side] = self.target_pos[self.active_side].copy()
                self._k_armed_t = None       # a pending drop confirmation never carries across arms
            elif key == GRIPPER_KEY:
                side = self.active_side
                now = time.monotonic()
                if side == self.holder_side and self.gripper_open[side] < 0.5:
                    # Opening the cup-holding gripper DROPS the cup: require a
                    # second K within K_DOUBLE_TAP_S of the first.
                    if self._k_armed_t is not None and (now - self._k_armed_t) <= K_DOUBLE_TAP_S:
                        self.gripper_open[side] = 1.0
                        self._k_armed_t = None
                        self._set_warning("")
                    else:
                        self._k_armed_t = now
                        self._set_warning("press K again to DROP the cup", K_DOUBLE_TAP_S)
                else:
                    self.gripper_open[side] = 1.0 if self.gripper_open[side] < 0.5 else 0.0
                    self._k_armed_t = None
            elif key == SPEED_DOWN_KEY:
                self.speed_scale = _clamp(self.speed_scale * 0.5, LIN_SPEED_MIN / LIN_SPEED_DEFAULT,
                                           LIN_SPEED_MAX / LIN_SPEED_DEFAULT)
            elif key == SPEED_UP_KEY:
                self.speed_scale = _clamp(self.speed_scale * 2.0, LIN_SPEED_MIN / LIN_SPEED_DEFAULT,
                                           LIN_SPEED_MAX / LIN_SPEED_DEFAULT)
            elif key == RESET_KEY:
                self._reset_locked()
            elif key == ESTOP_KEY:
                self.estop = not self.estop
                if self.estop:
                    for side in ("left", "right"):
                        site_id = self.ik.arms[side].site_id
                        self.target_pos[side] = self.data.site_xpos[site_id].copy()
                        self.target_quat[side] = _mat2quat(self.data.site_xmat[site_id])

    def _set_warning(self, text: str, duration_s: float = 0.0) -> None:
        self._warning = text
        self._warning_until = time.monotonic() + duration_s

    def _reset_locked(self) -> None:
        data = self.data
        data.qpos[:] = self._snapshot["qpos"]
        data.qvel[:] = self._snapshot["qvel"]
        data.ctrl[:] = self._snapshot["ctrl"]
        data.time = self._snapshot["time"]
        mujoco.mj_forward(self.model, data)
        self.fl.a_ego_prev[:] = 0.0
        self.fl.a_int_prev[:] = 0.0
        self.fl.last_feasible = True
        self.active_side = "right"
        self.speed_scale = 1.0
        self.estop = False
        self._k_armed_t = None
        self._set_warning("")
        self._a_hold = None               # force a fresh solve on the first post-reset tick
        self._ticks_since_replan = 0
        self._last_feasible = True
        self._last_h_min = None
        self.gripper_open = dict(self._gripper_open0)
        self.q_ref = {"left": self.q_lift.copy(), "right": self.q_home.copy()}
        for side in ("left", "right"):
            site_id = self.ik.arms[side].site_id
            self.target_pos[side] = data.site_xpos[site_id].copy()
            self.target_quat[side] = _mat2quat(data.site_xmat[site_id])
            self.workspace_center[side] = self.target_pos[side].copy()
            self.ik.arms[side].last_q = self.q_ref[side].copy()
        self.step_count = self.lift_steps
        self.t_sim = float(data.time)

    def _integrate_targets_locked(self, dt: float) -> None:
        if self.estop:
            return
        deadman = self.key_state.held(DEADMAN_KEY)
        if not deadman:
            return
        held = self.key_state.snapshot_held()
        side = self.active_side
        lin_speed = _clamp(LIN_SPEED_DEFAULT * self.speed_scale, LIN_SPEED_MIN, LIN_SPEED_MAX)
        ang_speed = ANG_SPEED_DEFAULT * self.speed_scale
        moved = False
        for key, (axis, sign) in POS_KEYS.items():
            if key in held:
                self.target_pos[side][axis] += sign * lin_speed * dt
                moved = True
        if moved:
            lo = self.workspace_center[side] - WORKSPACE_HALF_M
            hi = self.workspace_center[side] + WORKSPACE_HALF_M
            lo[2] = self.table_top_z + 0.03
            hi[2] = self.table_top_z + 0.5
            self.target_pos[side] = np.clip(self.target_pos[side], lo, hi)
        for key, (axis, sign) in ROT_KEYS.items():
            if key in held:
                dq = _axis_angle_quat(axis, sign * ang_speed * dt)
                q = _quat_mul(dq, self.target_quat[side])
                self.target_quat[side] = q / max(np.linalg.norm(q), 1e-9)

    def tick(self, dt: float | None = None) -> None:
        """One 200 Hz control step: keyboard -> target -> [every N steps: IK ->
        FL-MPC solve] -> torque -> mj_step. A solve tick mirrors
        scripts/make_cup_collect_safety_demo.py's `controller == "flmpc"`
        branch; the other N-1 ticks zero-order-hold the planned acceleration
        and re-realize tau = M(q) a + c(q, qdot) at the current state -- see
        module docstring "REAL-TIME VIA REPLAN DECIMATION"."""
        dt = sc.DT if dt is None else dt
        with self.lock:
            model, data = self.model, self.data
            self._integrate_targets_locked(dt)

            active = self.active_side
            # Near-contact guard: fall back to an every-step solve while the
            # last solve saw arm-arm clearance below replan_near_m or was
            # infeasible (safety over speed; the sim slows down there).
            near = (self.replan_near_m > 0.0 and
                    ((self._last_h_min is not None and self._last_h_min < self.replan_near_m)
                     or not self._last_feasible))
            replan = self._a_hold is None or near or self._ticks_since_replan >= self.replan_every
            if replan:
                # ArmIK.solve warm-starts from its own persisted `last_q`
                # internally. allow_coarse_restart=False: when the target is
                # unreachable (DLS err > 2 cm -- e.g. driven into the cup
                # arm) the default random-restart `coarse_seed` does ~6000
                # full-model mj_forward calls (~10-20 s per tick measured)
                # AND can return a different arm configuration (branch flip),
                # i.e. a discontinuous joint reference. For teleop the
                # reference must stay continuous: pure warm-started DLS only.
                q_sol, self._ik_err = self.ik.solve(data, active, self.target_pos[active],
                                                    self.target_quat[active], allow_coarse_restart=False)
                self.q_ref[active] = q_sol

            bt.set_gripper(data, self.left_gripper_act, self.gripper_open["left"])
            bt.set_gripper(data, self.right_gripper_act, self.gripper_open["right"])
            if replan:
                mujoco.mj_forward(model, data)
            else:
                # Hold tick: only the stages the ZOH torque + cup re-attach
                # need (kinematics, qM incl. armature, velocity terms,
                # passive incl. gravcomp/damping, RNE bias) -- ~0.02 ms vs
                # ~0.9 ms for mj_forward (whose collision/constraint work
                # mj_step redoes anyway). Verified bit-identical M/bias/
                # passive/xpos/xmat vs mj_forward (MuJoCo 3.9).
                _light_forward(model, data)
            self._grasp_hold_fn(model, data, self.gripper_open)   # faked: rigid cup re-attach

            M_full = sc.full_mass_matrix(model, data)
            payload = self._payload_fn(model, data) if self._payload_fn is not None else None
            if payload is None:
                M_box, bias_box = np.zeros((6, 6)), np.zeros(6)
            else:
                M_box, bias_box = payload

            if replan:
                ego_hold = (active == "left")
                q_ego_ref = np.tile(self.q_ref["left"], (self.fl.NP + 1, 1))
                qd_ego_ref = np.zeros((self.fl.NP + 1, 6))
                q_int_ref = np.tile(self.q_ref["right"], (self.fl.NP + 1, 1))
                qd_int_ref = np.zeros((self.fl.NP + 1, 6))
                cup_kps = (self._cup_keypoints_fn(model, data) if self._cup_keypoints_fn is not None
                           else self.cup_kps_for_barrier)
                pairs = sc.all_pair_barriers(model, data, self.left_dofadr, self.right_dofadr,
                                              extra_ego_keypoints=cup_kps)

                M_ll = M_full[np.ix_(self.left_dofadr, self.left_dofadr)] + M_box
                bias_l = data.qfrc_bias[self.left_dofadr] + bias_box
                M_rr = M_full[np.ix_(self.right_dofadr, self.right_dofadr)]
                bias_r = data.qfrc_bias[self.right_dofadr]
                a_te, b_te = sc.table_plane_barrier_rows(model, data, self.left_table_kps, self.left_dofadr,
                                                           np.linalg.inv(M_ll), bias_l, self.table_top_z,
                                                           margin=demo.TABLE_FLOOR_MARGIN_M)
                a_ti, b_ti = sc.table_plane_barrier_rows(model, data, self.right_table_kps, self.right_dofadr,
                                                           np.linalg.inv(M_rr), bias_r, self.table_top_z,
                                                           margin=demo.TABLE_FLOOR_MARGIN_M)

                tau_l, tau_r, feasible = self.fl.step(
                    model, data, self.left_qposadr, self.left_dofadr, self.right_qposadr, self.right_dofadr,
                    q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref, pairs,
                    M_ego_extra=M_box, bias_ego_extra=bias_box,
                    table_rows=(a_te, b_te, a_ti, b_ti),
                    left_ee_body=self.left_ee_body, tilt_max_rad=demo.CUP_TILT_MAX_RAD,
                    leash_ee_body=self.left_ee_body, leash_max_m=demo.CUP_LEASH_M,
                    cup_cone_facets=demo.CUP_CONE_FACETS,
                    ego_hold=ego_hold)
                # fl.step stores the applied first-block accelerations here
                # (after its infeasibility fallback + torque backstop).
                self._a_hold = (self.fl.a_ego_prev.copy(), self.fl.a_int_prev.copy())
                self._ticks_since_replan = 0
                self.replan_count += 1
                self._last_feasible = bool(feasible)
                # arm-arm h_min (== sc.min_pair_distance's) from the pairs we
                # already have -- pre-step, same instant the old per-tick call used.
                self._last_h_min = float(min(p["h"] for p in pairs[:self._n_arm_pairs]))
            else:
                # ZOH of the planned acceleration; exact FL torque at the
                # CURRENT state (same M/c definitions fl.step uses, incl. the
                # cup payload reflection on the left arm).
                ld, rd = self.left_dofadr, self.right_dofadr
                M_e = M_full[np.ix_(ld, ld)] + M_box
                c_e = data.qfrc_bias[ld] - data.qfrc_passive[ld] + bias_box
                M_i = M_full[np.ix_(rd, rd)]
                c_i = data.qfrc_bias[rd] - data.qfrc_passive[rd]
                a_e = self.fl._torque_safe_scale(M_e, c_e, self._a_hold[0], sc.TAU_MAX)
                a_i = self.fl._torque_safe_scale(M_i, c_i, self._a_hold[1], sc.TAU_MAX)
                tau_l = M_e @ a_e + c_e
                tau_r = M_i @ a_i + c_i
                feasible = self._last_feasible
            self._ticks_since_replan += 1
            h_min = self._last_h_min

            tau_l_cmd = np.clip(self.tau_filter_l.apply(tau_l, sc.DT), -sc.TAU_MAX, sc.TAU_MAX)
            tau_r_cmd = np.clip(self.tau_filter_r.apply(tau_r, sc.DT), -sc.TAU_MAX, sc.TAU_MAX)
            data.qfrc_applied[self.left_dofadr] = tau_l_cmd
            data.qfrc_applied[self.right_dofadr] = tau_r_cmd

            mujoco.mj_step(model, data)
            self.t_sim = float(data.time)
            self.step_count += 1

            # Arm-arm contact check on the contacts mj_step's own internal
            # forward pass just computed (= this tick's pre-integration,
            # post-re-attach state) -- fresh on EVERY tick, incl. hold ticks
            # that skip mj_forward.
            collision = False
            for i in range(data.ncon):
                con = data.contact[i]
                if ((con.geom1 in self.left_geoms and con.geom2 in self.right_geoms) or
                        (con.geom2 in self.left_geoms and con.geom1 in self.right_geoms)):
                    collision = True
                    break

            site_id = self.ik.arms[active].site_id
            track_err = float(np.linalg.norm(data.site_xpos[site_id] - self.target_pos[active]))
            self._update_hud(feasible=feasible, h_min=h_min, collision=collision, track_err=track_err)

    def _update_hud(self, *, feasible: bool, h_min, collision: bool, track_err: float) -> None:
        cup_axis_world = self.data.xmat[self._tilt_body].reshape(3, 3) @ self._cup_axis_local
        tilt_rad = float(np.arccos(np.clip(cup_axis_world[2], -1.0, 1.0)))
        safety_overriding = bool((not feasible) or collision or (track_err > SAFETY_OVERRIDE_ERR_M) or
                                  (h_min is not None and h_min < SAFETY_OVERRIDE_H_M))
        if self._warning and time.monotonic() > self._warning_until:
            self._warning = ""
            self._k_armed_t = None
        self.hud = dict(
            warning=self._warning, grasp_mode=self.grasp_mode, holder_arm=self.holder_side,
            replan_every=self.replan_every, replan_near_m=self.replan_near_m, rtf=None if self.rtf is None else round(self.rtf, 2),
            t=self.t_sim, step=self.step_count, active_arm=self.active_side,
            speed_scale=round(self.speed_scale, 3),
            speed_mps=round(_clamp(LIN_SPEED_DEFAULT * self.speed_scale, LIN_SPEED_MIN, LIN_SPEED_MAX), 4),
            deadman=self.key_state.held(DEADMAN_KEY), estop=self.estop,
            safety_overriding=safety_overriding, feasible=bool(feasible), collision=bool(collision),
            h_min_m=None if h_min is None else round(float(h_min), 4),
            cup_tilt_deg=round(np.degrees(tilt_rad), 3),
            cup_tilt_max_deg=round(np.degrees(demo.CUP_TILT_MAX_RAD), 3),
            track_err_m=round(track_err, 4), ik_err_m=round(float(self._ik_err), 4),
            ee_target_xyz={s: [round(float(v), 4) for v in self.target_pos[s]] for s in ("left", "right")},
            left_gripper_open=self.gripper_open["left"], right_gripper_open=self.gripper_open["right"],
        )


# ---------------------------------------------------------------------------
# aiohttp web server: one page + /ws.
# ---------------------------------------------------------------------------
PAGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>FL-MPC keyboard teleop</title>
<style>
*{box-sizing:border-box}
body{background:#111;color:#ddd;font-family:monospace;padding:16px;margin:0}
#hud{white-space:pre-wrap;overflow-wrap:anywhere;background:#000;border:1px solid #444;padding:12px;
     border-radius:6px;max-width:720px;width:100%}
.warn{color:#ff5555;font-weight:bold}
.ok{color:#55ff88}
#keys{margin-top:12px;font-size:12px;color:#999;max-width:720px;overflow-wrap:anywhere}
</style></head>
<body>
<h2>Bimanual FL-MPC keyboard teleop</h2>
<div id="hud">connecting...</div>
<div id="keys">
Space=deadman(hold) W/S=+-x A/D=+-y Q/E=+-z Z/X=roll T/G=pitch C/V=yaw
K=gripper (cup arm: double-tap K within 1 s to drop) TAB=switch arm -/= =speed R=reset Esc=e-stop
</div>
<script>
let ws = new WebSocket("ws://" + location.host + "/ws");
const BOUND = new Set([" ","w","s","a","d","q","e","z","x","t","g","c","v",
                        "k","tab","-","=","r","escape"]);
const held = new Set();
function keyName(e){ return e.key.length === 1 ? e.key.toLowerCase() : e.key.toLowerCase(); }
function send(type, key){ if (ws.readyState === 1) ws.send(JSON.stringify({type:type, key:key})); }
window.addEventListener("keydown", (e) => {
  const k = keyName(e);
  if (!BOUND.has(k)) return;
  e.preventDefault();
  held.add(k);
  send("keydown", k);
});
window.addEventListener("keyup", (e) => {
  const k = keyName(e);
  if (!BOUND.has(k)) return;
  e.preventDefault();
  held.delete(k);
  send("keyup", k);
});
window.addEventListener("blur", () => { held.forEach(k => send("keyup", k)); held.clear(); send("release_all", ""); });
document.addEventListener("visibilitychange", () => {
  if (document.hidden) { held.forEach(k => send("keyup", k)); held.clear(); send("release_all", ""); }
});
// Browser key-repeat is unreliable -- resend keydown for everything currently
// held every 100ms so the server's stuck-key auto-release (>250ms silence)
// never fires for a genuinely-held key.
setInterval(() => { held.forEach(k => send("keydown", k)); }, 100);
ws.onmessage = (ev) => {
  const h = JSON.parse(ev.data);
  const el = document.getElementById("hud");
  const cls = h.safety_overriding ? "warn" : "ok";
  el.innerHTML =
    (h.warning ? "<span class='warn'>!! " + h.warning + " !!</span>\\n" : "") +
    "t=" + h.t.toFixed(2) + "s  step=" + h.step + "  rtf=" + (h.rtf === null ? "-" : h.rtf + "x") + "  replan_every=" + h.replan_every +
    "  grasp=" + h.grasp_mode + "\\n" +
    "active_arm=" + h.active_arm + "  speed=" + h.speed_mps + " m/s (x" + h.speed_scale + ")\\n" +
    "deadman=" + h.deadman + "  estop=" + h.estop + "\\n" +
    "<span class='" + cls + "'>SAFETY_OVERRIDING=" + h.safety_overriding + "</span>" +
    "  feasible=" + h.feasible + "  collision=" + h.collision + "\\n" +
    "h_min=" + h.h_min_m + " m   track_err=" + h.track_err_m + " m   ik_err=" + h.ik_err_m + " m\\n" +
    "cup_tilt=" + h.cup_tilt_deg + " / " + h.cup_tilt_max_deg + " deg\\n" +
    "target_right=" + JSON.stringify(h.ee_target_xyz.right) + "\\n" +
    "target_left=" + JSON.stringify(h.ee_target_xyz.left) + "\\n" +
    "grippers: left=" + h.left_gripper_open + " right=" + h.right_gripper_open;
};
</script>
</body></html>"""


def build_app(sim: TeleopSim) -> web.Application:
    async def index(_request):
        return web.Response(text=PAGE_HTML, content_type="text/html")

    async def ws_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        stop = asyncio.Event()

        async def push_hud():
            try:
                while not stop.is_set():
                    with sim.lock:
                        payload = json.dumps(sim.hud)
                    await ws.send_str(payload)
                    await asyncio.sleep(0.1)  # ~10 Hz
            except ConnectionResetError:
                pass

        pusher = asyncio.ensure_future(push_hud())
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        apply_ws_message(sim, json.loads(msg.data))
                    except (json.JSONDecodeError, TypeError):
                        pass
                elif msg.type == web.WSMsgType.ERROR:
                    break
        finally:
            stop.set()
            pusher.cancel()
        return ws

    app = web.Application()
    app["sim"] = sim
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    return app


def start_sim_thread(sim: TeleopSim, scene=None) -> threading.Thread:
    """Real-time-paced 200 Hz control loop in its own daemon thread. Solve
    ticks are bursty (one ~25 ms solve, then N-1 ~1-2 ms hold ticks), so the
    loop CATCHES UP after a solve (no sleep while behind) and only resyncs
    once it lags by more than MAX_LAG_S (then it is genuinely too slow)."""
    def _loop():
        period = sc.DT
        next_t = time.perf_counter()
        render_count = 0
        rtf_wall0, rtf_sim0 = time.perf_counter(), sim.t_sim
        while True:
            sim.tick(period)
            render_count += 1
            if scene is not None and render_count % 10 == 0:   # ~20 Hz view refresh
                try:
                    with sim.lock:
                        scene.update_from_mjdata(sim.data)
                except Exception:
                    pass
            now = time.perf_counter()
            if now - rtf_wall0 >= 1.0:
                ds = sim.t_sim - rtf_sim0
                sim.rtf = ds / (now - rtf_wall0) if ds >= 0 else None   # None across an R reset
                rtf_wall0, rtf_sim0 = now, sim.t_sim
            next_t += period
            slack = next_t - now
            if slack > 0:
                time.sleep(slack)
            elif slack < -MAX_LAG_S:
                next_t = time.perf_counter()   # too far behind; resync instead of spiraling

    th = threading.Thread(target=_loop, daemon=True, name="fl-teleop-sim")
    th.start()
    return th


def start_view(sim: TeleopSim, view_port: int, host: str = "0.0.0.0"):
    """Best-effort mjviser 3D view via ViserMujocoScene, externally stepped
    (see module docstring "VIEW"). Returns the scene object (for the sim
    thread to push updates into) or None on any failure -- the keyboard
    server still works without it."""
    try:
        import viser
        import mjviser
        server = viser.ViserServer(host=host, port=view_port)
        scene = mjviser.ViserMujocoScene(server, sim.model, num_envs=1)
        scene.update_from_mjdata(sim.data)
        print(f"[keyboard_fl_teleop] mjviser 3D view on http://localhost:{view_port}")
        return scene
    except Exception as exc:  # pragma: no cover -- best-effort only
        print(f"[keyboard_fl_teleop] 3D view unavailable ({exc!r}); continuing without it.")
        return None


# ---------------------------------------------------------------------------
# Headless smoke test.
# ---------------------------------------------------------------------------
def _hold_and_tick(sim: TeleopSim, keys: list[str], duration_s: float, refresh_every_s: float = 0.09):
    """Programmatic key injection (test a/d): resend keydown for `keys` +
    deadman every `refresh_every_s` of WALL-CLOCK time (not tick count --
    the server's stuck-key/deadman staleness check is wall-clock based, and
    this control loop's own per-tick compute (~30ms/step in this sandbox,
    see the module's known-limitations note: it does not sustain a true
    200Hz wall-clock rate) already consumes real time, so refreshing on a
    fixed TICK cadence -- assuming near-200Hz -- would let real wall-clock
    time between refreshes drift past the 250ms stale threshold and
    spuriously auto-release a genuinely-held key. Advances exactly
    `duration_s` of SIMULATED time regardless of how long that takes in
    real wall-clock time.)"""
    n_steps = int(round(duration_s / sc.DT))
    last_refresh = -1.0
    for _ in range(n_steps):
        now = time.perf_counter()
        if now - last_refresh >= refresh_every_s:
            apply_ws_message(sim, {"type": "keydown", "key": DEADMAN_KEY})
            for k in keys:
                apply_ws_message(sim, {"type": "keydown", "key": k})
            last_refresh = now
        sim.tick(sc.DT)


def run_smoke(seed: int = 0, replan_every: int = REPLAN_EVERY_DEFAULT, grasp_mode: str = "faked",
              replan_near_m: float = REPLAN_NEAR_M_DEFAULT, approach_scale: float = 2.0) -> None:
    print(f"=== keyboard_fl_teleop SMOKE START (replan_every={replan_every}, replan_near_m={replan_near_m}, "
          f"approach_scale={approach_scale}, "
          f"grasp_mode={grasp_mode}) ===")
    sim = TeleopSim(seed=seed, replan_every=replan_every, grasp_mode=grasp_mode, replan_near_m=replan_near_m)
    print(f"[smoke] start state reached: t={sim.t_sim:.3f}s step={sim.step_count} "
          f"active_arm={sim.active_side} left_gripper_open={sim.gripper_open['left']} "
          f"right_gripper_open={sim.gripper_open['right']} cup_tilt_deg={sim.hud['cup_tilt_deg']:.3f}")
    assert sim.gripper_open["left"] == 0.0, "left gripper must start closed (holding the cup)"

    left_geoms, right_geoms = sim.left_geoms, sim.right_geoms

    # -- (a) hold Space+W 1.5s then Space+Q 1.5s, release -----------------
    target0 = sim.target_pos["right"].copy()
    t_wall_a, t_sim_a = time.perf_counter(), sim.t_sim
    _hold_and_tick(sim, ["w"], 1.5)
    target_after_w = sim.target_pos["right"].copy()
    dx = target_after_w[0] - target0[0]
    expected_dx = LIN_SPEED_DEFAULT * 1.5
    _hold_and_tick(sim, ["q"], 1.5)
    target_after_q = sim.target_pos["right"].copy()
    dz = target_after_q[2] - target_after_w[2]
    expected_dz = LIN_SPEED_DEFAULT * 1.5
    print(f"[smoke a] compute-bound RTF over (a) = {(sim.t_sim - t_sim_a) / (time.perf_counter() - t_wall_a):.3f}x "
          f"(no sleeping; replan_every={sim.replan_every})")
    site_id = sim.ik.arms["right"].site_id
    realized = sim.data.site_xpos[site_id].copy()
    track_err = float(np.linalg.norm(realized - sim.target_pos["right"]))
    h_min, *_ = sc.min_pair_distance(sim.model, sim.data)
    collision_a = False
    for i in range(sim.data.ncon):
        con = sim.data.contact[i]
        if ((con.geom1 in left_geoms and con.geom2 in right_geoms) or
                (con.geom2 in left_geoms and con.geom1 in right_geoms)):
            collision_a = True
    apply_ws_message(sim, {"type": "keyup", "key": "w"})
    apply_ws_message(sim, {"type": "keyup", "key": "q"})
    apply_ws_message(sim, {"type": "keyup", "key": DEADMAN_KEY})
    print(f"[smoke a] dx={dx:.4f} (expected~{expected_dx:.4f})  dz={dz:.4f} (expected~{expected_dz:.4f})")
    print(f"[smoke a] realized right EE track_err={track_err:.4f} m  h_min={h_min:.4f} m  "
          f"collision={collision_a}  cup_tilt_deg={sim.hud['cup_tilt_deg']:.3f} "
          f"(max {sim.hud['cup_tilt_max_deg']:.3f})  left_gripper_open={sim.gripper_open['left']}")
    assert abs(dx - expected_dx) < 0.02, f"target +x motion off: dx={dx}"
    assert abs(dz - expected_dz) < 0.02, f"target +z motion off: dz={dz}"
    assert track_err < 0.05, f"right EE tracking error too large: {track_err}"
    assert not collision_a, "unexpected arm-arm collision in test (a)"
    assert sim.hud["cup_tilt_deg"] <= np.degrees(demo.CUP_TILT_MAX_RAD) + 1e-6, "cup tilt exceeded its cone"
    assert sim.gripper_open["left"] == 0.0, "left arm dropped the cup (gripper opened) during (a)"

    # -- (b) stuck key: W+Space keydown, NO refresh, no keyup -------------
    # KeyState staleness (STALE_KEY_S=0.25s) is WALL-CLOCK based, so this
    # samples target_pos["right"][0] against actual elapsed wall time (not
    # tick count / assumed Hz) until ~600ms of real time has passed, and
    # checks that motion actually stopped within ~300ms of the single
    # keydown (no further refresh sent).
    apply_ws_message(sim, {"type": "keydown", "key": DEADMAN_KEY})
    apply_ws_message(sim, {"type": "keydown", "key": "w"})
    t_wall0 = time.perf_counter()
    samples = []   # (elapsed_wall_s, x)
    while True:
        sim.tick(sc.DT)
        elapsed = time.perf_counter() - t_wall0
        samples.append((elapsed, float(sim.target_pos["right"][0])))
        if elapsed >= 0.6:
            break
    last_move_t = 0.0
    for i in range(1, len(samples)):
        if abs(samples[i][1] - samples[i - 1][1]) > 1e-9:
            last_move_t = samples[i][0]
    x_end = samples[-1][1]
    still_at_end = all(abs(x - x_end) < 1e-9 for _, x in samples if _ > last_move_t)
    print(f"[smoke b] {len(samples)} ticks over {samples[-1][0]:.3f}s wall; last target motion at "
          f"t={last_move_t*1000:.1f}ms; x frozen thereafter={still_at_end}; x_final={x_end:.4f}")
    assert last_move_t <= 0.30, f"stuck W key kept moving the target past ~300ms: last move at {last_move_t}s"
    assert still_at_end, "target kept moving after the stuck key should have auto-released"
    apply_ws_message(sim, {"type": "release_all", "key": ""})

    # -- (c) deadman: W held, Space NOT held -------------------------------
    target_before = sim.target_pos["right"].copy()
    apply_ws_message(sim, {"type": "keydown", "key": "w"})
    for _ in range(60):
        sim.tick(sc.DT)
    target_after = sim.target_pos["right"].copy()
    dmove = float(np.linalg.norm(target_after - target_before))
    print(f"[smoke c] target motion without deadman over 300ms = {dmove:.6f} m")
    assert dmove < 1e-9, f"target moved without the deadman held: {dmove}"
    apply_ws_message(sim, {"type": "release_all", "key": ""})

    # -- (d) drive the right EE target at the left arm / cup ---------------
    # approach_scale=2 -> 0.1 m/s (what this test has always effectively run
    # at: the old code sent two keydowns WITHOUT a keyup, and only the first
    # is a fresh edge-triggered press). approach_scale=4 -> 0.2 m/s (max) is a
    # harder stress variant (--smoke-approach-scale 4).
    n_press = int(round(np.log2(approach_scale)))
    for _ in range(n_press):   # FRESH presses (keyup between)
        apply_ws_message(sim, {"type": "keydown", "key": SPEED_UP_KEY})
        apply_ws_message(sim, {"type": "keyup", "key": SPEED_UP_KEY})
    assert abs(sim.speed_scale - approach_scale) < 1e-9, sim.speed_scale
    left_pos = sim.data.xpos[sim.left_ee_body].copy()
    right_pos = sim.data.xpos[sim.right_ee_body].copy()
    delta = left_pos - right_pos
    approach_keys = []
    inv_pos = {(axis, sign): key for key, (axis, sign) in POS_KEYS.items()}
    for axis in range(3):
        sign = 1.0 if delta[axis] >= 0 else -1.0
        approach_keys.append(inv_pos[(axis, sign)])
    print(f"[smoke d] approach keys toward left arm/cup: {approach_keys}  speed_scale={sim.speed_scale}")
    worst_h = 1e9
    worst_h_every_step = 1e9   # independent per-step check (sc.min_pair_distance), not the HUD's solve-tick value
    saw_override = False
    saw_collision = False
    n_infeasible_solves = 0
    solves0 = sim.replan_count
    n_steps = int(round(4.0 / sc.DT))
    refresh_every_s = 0.09   # wall-clock cadence, see _hold_and_tick's docstring
    last_refresh = -1.0
    for _ in range(n_steps):
        now = time.perf_counter()
        if now - last_refresh >= refresh_every_s:
            apply_ws_message(sim, {"type": "keydown", "key": DEADMAN_KEY})
            for kk in approach_keys:
                apply_ws_message(sim, {"type": "keydown", "key": kk})
            last_refresh = now
        n_before = sim.replan_count
        sim.tick(sc.DT)
        if sim.replan_count != n_before and not sim.hud["feasible"]:
            n_infeasible_solves += 1
        worst_h = min(worst_h, sim.hud["h_min_m"])
        worst_h_every_step = min(worst_h_every_step, float(sc.min_pair_distance(sim.model, sim.data)[0]))
        saw_override = saw_override or sim.hud["safety_overriding"]
        saw_collision = saw_collision or sim.hud["collision"]
    print(f"[smoke d] worst h_min over approach = {worst_h:.4f} m (solve ticks)  "
          f"{worst_h_every_step:.4f} m (EVERY sim step, post-step)   "
          f"saw SAFETY_OVERRIDING={saw_override}  saw collision={saw_collision}  "
          f"infeasible solves={n_infeasible_solves}/{sim.replan_count - solves0}  "
          f"final target_right={sim.target_pos['right']}")
    worst_h = min(worst_h, worst_h_every_step)
    assert not saw_collision, "FL-MPC failed to prevent an arm-arm collision under a direct approach command"
    assert worst_h >= -1e-3, f"h_min went meaningfully negative (deep penetration): {worst_h}"
    assert saw_override, "SAFETY_OVERRIDING never engaged despite commanding straight at the held cup"
    apply_ws_message(sim, {"type": "release_all", "key": ""})

    # -- (e) loopback WebSocket ---------------------------------------------
    asyncio.run(_smoke_ws(sim))

    # -- (f) K guard: cup-holding arm needs a double-tap to open ------------
    def _tap(key):
        apply_ws_message(sim, {"type": "keydown", "key": key})
        apply_ws_message(sim, {"type": "keyup", "key": key})

    holder = sim.holder_side
    other = "right" if holder == "left" else "left"
    if sim.active_side != holder:
        _tap(SWITCH_ARM_KEY)
    assert sim.active_side == holder and sim.gripper_open[holder] < 0.5
    _tap(GRIPPER_KEY)
    sim.tick(sc.DT)
    warn1 = sim.hud["warning"]
    print(f"[smoke f] single K on cup arm ({holder}): gripper_open={sim.gripper_open[holder]} HUD warning={warn1!r}")
    assert sim.gripper_open[holder] < 0.5, "single K on the cup-holding arm opened the gripper"
    assert "press K again to DROP the cup" in warn1, "no HUD warning after the first K on the cup arm"
    time.sleep(K_DOUBLE_TAP_S + 0.15)             # let the confirmation window expire
    sim.tick(sc.DT)
    assert sim.hud["warning"] == "", "K warning did not expire"
    _tap(GRIPPER_KEY)                              # new first tap (the old one expired)
    assert sim.gripper_open[holder] < 0.5, "a K after the 1 s window expired opened the gripper"
    _tap(GRIPPER_KEY)                              # second tap within the window -> open
    print(f"[smoke f] expired-window K ignored; double-tap -> gripper_open={sim.gripper_open[holder]}")
    assert sim.gripper_open[holder] > 0.5, "double-tap K on the cup arm did not open the gripper"
    _tap(GRIPPER_KEY)                              # closing needs no confirmation
    assert sim.gripper_open[holder] < 0.5, "single K did not re-close the cup arm's gripper"
    _tap(SWITCH_ARM_KEY)
    g0 = sim.gripper_open[other]
    _tap(GRIPPER_KEY)
    print(f"[smoke f] other arm ({other}): single K toggled gripper_open {g0} -> {sim.gripper_open[other]}")
    assert sim.gripper_open[other] != g0, "single K on the non-holding arm did not toggle its gripper"
    _tap(GRIPPER_KEY)
    _tap(RESET_KEY)
    sim.tick(sc.DT)
    assert sim.active_side == "right" and sim.gripper_open == sim._gripper_open0
    print("[smoke f] K guard OK")

    print("=== keyboard_fl_teleop SMOKE PASSED ===")


async def _smoke_ws(sim: TeleopSim) -> None:
    import websockets

    app = build_app(sim)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    print(f"[smoke e] loopback aiohttp server bound on 127.0.0.1:{port}")
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:
            before = sim.key_state.held("e")
            await ws.send(json.dumps({"type": "keydown", "key": "e"}))
            await asyncio.sleep(0.05)
            after = sim.key_state.held("e")
            print(f"[smoke e] server key_state['e'] held before={before} after={after}")
            assert (not before) and after, "server key_state did not react to the loopback keydown message"
            hud_msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
            hud = json.loads(hud_msg)
            print(f"[smoke e] received HUD push over the loopback socket: "
                  f"active_arm={hud.get('active_arm')} t={hud.get('t')}")
            assert "active_arm" in hud and "t" in hud, "HUD payload missing expected fields"
            await ws.send(json.dumps({"type": "keyup", "key": "e"}))
    finally:
        await runner.cleanup()
    print("[smoke e] loopback client + server shut down cleanly")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8081, help="aiohttp keyboard/HUD server port.")
    ap.add_argument("--view-port", type=int, default=8080, help="mjviser 3D-view server port.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="run the headless smoke test and exit (no browser).")
    ap.add_argument("--smoke-approach-scale", type=float, default=2.0, choices=[1.0, 2.0, 4.0],
                    help="smoke test (d) approach speed scale (2 -> 0.1 m/s default; 4 -> 0.2 m/s stress).")
    ap.add_argument("--no-view", action="store_true", help="skip the mjviser 3D view entirely.")
    ap.add_argument("--host", default="0.0.0.0",
                    help="bind address for BOTH servers (keyboard page + mjviser view). Default 0.0.0.0 "
                         "(all interfaces); use 127.0.0.1 for SSH-port-forward-only access on a shared machine.")
    ap.add_argument("--replan-every", type=int, default=REPLAN_EVERY_DEFAULT,
                    help=f"full FL-MPC re-solve every N sim steps (N*5 ms; 1..L=8), zero-order-holding the "
                         f"planned acceleration in between (default {REPLAN_EVERY_DEFAULT}).")
    ap.add_argument("--replan-near-m", type=float, default=REPLAN_NEAR_M_DEFAULT,
                    help="near-contact guard: solve EVERY step while the last arm-arm clearance h_min is below "
                         f"this [m] or the last solve was infeasible; 0 disables (default {REPLAN_NEAR_M_DEFAULT}).")
    ap.add_argument("--grasp-mode", choices=sorted(GRASP_BUILDERS), default="faked",
                    help="faked (default): kinematic rigid cup re-attach; real: STUB, see "
                         "build_start_state_realgrasp.")
    args = ap.parse_args()

    if args.smoke:
        run_smoke(seed=args.seed, replan_every=args.replan_every, grasp_mode=args.grasp_mode,
                  replan_near_m=args.replan_near_m, approach_scale=args.smoke_approach_scale)
        return

    if args.host == "0.0.0.0":
        print("!" * 78)
        print("WARNING: --host 0.0.0.0 -> both servers listen on ALL interfaces. On a shared machine anyone who")
        print("can reach these ports can open the page and DRIVE this sim. For SSH-forward-only use, rerun with")
        print("  --host 127.0.0.1")
        print("!" * 78)

    sim = TeleopSim(seed=args.seed, replan_every=args.replan_every, grasp_mode=args.grasp_mode,
                    replan_near_m=args.replan_near_m)
    scene = None if args.no_view else start_view(sim, args.view_port, host=args.host)
    start_sim_thread(sim, scene=scene)

    print("=" * 78)
    print("Bimanual FL-MPC keyboard teleop -- SIMULATION ONLY, no hardware connection.")
    print(f"bind host={args.host}  replan_every={sim.replan_every} ({sim.replan_every * sc.DT * 1e3:.0f} ms; "
          f"FL-MPC horizon {sim.fl.NP * sc.DT * 1e3:.0f} ms)  grasp_mode={args.grasp_mode}")
    print(f"On your laptop:")
    print(f"  ssh -L {args.port}:localhost:{args.port} -L {args.view_port}:localhost:{args.view_port} <user>@<box>")
    print(f"Then open:")
    print(f"  http://localhost:{args.port}   (keyboard teleop page)")
    if scene is not None:
        print(f"  http://localhost:{args.view_port}   (3D view)")
    print("=" * 78)

    app = build_app(sim)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
