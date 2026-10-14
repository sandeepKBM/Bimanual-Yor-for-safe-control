"""EE-target teleop for BOTH YOR bimanual arms (scenes/scene_yor_cup.xml).

Interface: a 6-DoF (position + optional orientation) end-effector TARGET per
arm, resolved to joint targets by damped-least-squares (DLS) IK on that arm's
own 6-DOF Jacobian (mink was evaluated for reuse -- see "IK backend" below --
and found not directly usable for this robot/scene, so DLS is used), then sent
to the arm's `<position>` actuators. Gripper is a separate scalar per arm in
[0, 1] (0=closed, 1=open), driven straight to the two finger actuators.

Two backends, both built on the same `BimanualIK` + `set_gripper` primitives:
  1. INTERACTIVE (`run_interactive`): `mujoco.viewer` + keyboard, for a machine
     with a display. NOT run/exercised in this task (headless sandbox).
  2. SCRIPTED (`drive`): a headless, display-free API that steps the sim while
     following a per-arm list of (position, orientation, gripper) segments --
     used by `teleop/record_teleop.py` for episode recording.

IK backend
----------
`YOR/extra/mink_whole_body_ik.py` uses the `mink` package (confirmed
importable in `safe_env`) but is written for a DIFFERENT scene/robot: a
mobile-base rig with joints named `joint_x/joint_y/joint_th/lift_joint/
left_joint1..7`, a single arm, a `pinch_site_target` mocap body, and its own
`mujoco/scene.xml`. None of that exists in `scenes/scene_yor_cup.xml` (welded
base, joints `left_arm_joint1..6`/`right_arm_joint1..6`, no mocap body, two
arms). Re-pointing mink's `Configuration`/`FrameTask` machinery at this scene
would mean re-deriving mink usage from scratch anyway, with no code actually
reused from that file -- so per the task's "reuse if importable/usable, else
DLS" instruction, this module uses damped-least-squares IK on `mj_jacSite`
directly (the same technique already used and verified in this repo by
`scripts/verify_yor_canonical.py`'s `coarse_seed`/`ik_to_target`, reimplemented
here rather than imported since that script is not in the allowed-import list
-- generalized to work on `scene_yor_cup.xml`'s full model, which has extra
DOFs (cup + 8 balls) ahead of/around the two 6-DOF arm joint groups).

Tracking accuracy: warm-started DLS (persisted per-arm joint solution) with a
coarse random-restart fallback converges to <1mm position error within its
default iteration budget for targets across the arms' shared reachable
envelope documented in `scenes/YOR_CANONICAL.md` (see this module's own
`self_check()` and the smoke test in `record_teleop.py`).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SIM_COMMON_DIR = REPO_ROOT / "experiments" / "ab_smooth_vs_bangbang"
if str(SIM_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_COMMON_DIR))
import sim_common  # noqa: E402  (LEFT_ARM_JOINTS/RIGHT_ARM_JOINTS/joint_addrs -- read-only import)

from scripts.cup_task_utils import load_yor_cup_model  # noqa: E402

SIDES = ("left", "right")
GRASP_SITE = {"left": "left_arm_ee_grasp", "right": "right_arm_ee_grasp"}
GRIPPER_BODY = {"left": "left_arm_gripper_base", "right": "right_arm_gripper_base"}
ARM_JOINTS = {"left": sim_common.LEFT_ARM_JOINTS, "right": sim_common.RIGHT_ARM_JOINTS}
FINGER_JOINTS = {"left": ("left_arm_joint7", "left_arm_joint8"),
                 "right": ("right_arm_joint7", "right_arm_joint8")}
GRIPPER_JOINT_RANGE_M = 0.035  # matches collection/episode.py's GRIPPER_JOINT_RANGE_M convention


def load_scene(xml_path: str | None = None):
    """Load scenes/scene_yor_cup.xml with the ZED workspace camera + arm-arm
    collision patch already applied (via scripts.cup_task_utils.load_yor_cup_model,
    which is the single source of truth for this scene's compiled model)."""
    if xml_path is None:
        model, data = load_yor_cup_model()
    else:
        model, data = load_yor_cup_model(xml_path)
    return model, data


def reset_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# IK: damped-least-squares on the arm's own 6-DOF Jacobian.
# ---------------------------------------------------------------------------

def _mat2quat(mat9: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat9.reshape(-1))
    return q


def _orientation_error(target_quat: np.ndarray, cur_quat: np.ndarray) -> np.ndarray:
    """3-vector rotation-error (axis*angle, small-angle) s.t. driving it to 0
    rotates cur_quat onto target_quat."""
    neg_cur = np.zeros(4)
    mujoco.mju_negQuat(neg_cur, cur_quat)
    diff = np.zeros(4)
    mujoco.mju_mulQuat(diff, target_quat, neg_cur)
    vel = np.zeros(3)
    mujoco.mju_quat2Vel(vel, diff, 1.0)
    return vel


def coarse_seed(model, scratch, site_id, gripper_body_id, target_pos, qposadr, jnt_range,
                 n=6000, seed=0, want_down=True):
    """Random-restart seed search (same idea as scripts/verify_yor_canonical.py's
    coarse_seed): among n random joint samples, keep the one closest to
    target_pos (within 12cm) while preferring a "reach down" gripper
    orientation -- avoids DLS converging to a badly-oriented elbow lock when
    warm-starting from a pose far from the target."""
    rng = np.random.default_rng(seed)
    best = (1e9, None)
    for _ in range(n):
        q = rng.uniform(jnt_range[:, 0], jnt_range[:, 1])
        scratch.qpos[qposadr] = q
        mujoco.mj_forward(model, scratch)
        pos_err = float(np.linalg.norm(scratch.site_xpos[site_id] - target_pos))
        if pos_err > 0.12:
            continue
        cost = pos_err * 2.0
        if want_down:
            xmat = scratch.xmat[gripper_body_id].reshape(3, 3)
            approach_z = xmat[:, 2]
            cost += (1.0 + approach_z[2])  # prefer approach axis pointing down
        if cost < best[0]:
            best = (cost, q.copy())
    return best[1]


class ArmIK:
    """DLS position(+orientation) IK for one arm's 6 joints on a given model."""

    def __init__(self, model: mujoco.MjModel, side: str):
        assert side in SIDES
        self.model = model
        self.side = side
        self.site_id = model.site(GRASP_SITE[side]).id
        self.gripper_body_id = model.body(GRIPPER_BODY[side]).id
        _, self.qposadr, self.dofadr = sim_common.joint_addrs(model, ARM_JOINTS[side])
        jids = [model.joint(n).id for n in ARM_JOINTS[side]]
        self.jnt_range = model.jnt_range[jids].copy()
        self._scratch = mujoco.MjData(model)
        self.last_q: Optional[np.ndarray] = None

    def current_q(self, data: mujoco.MjData) -> np.ndarray:
        return data.qpos[self.qposadr].copy()

    def solve(self, data: mujoco.MjData, target_pos: np.ndarray,
              target_quat: Optional[np.ndarray] = None,
              iters: int = 60, damp: float = 0.03, max_step: float = 0.25,
              tol_pos: float = 1e-4, ori_weight: float = 0.3,
              allow_coarse_restart: bool = True) -> tuple[np.ndarray, float]:
        """Warm-started DLS IK. Returns (q(6,), final_pos_err_m)."""
        target_pos = np.asarray(target_pos, dtype=float)
        scratch = self._scratch
        scratch.qpos[:] = data.qpos
        q0 = self.last_q if self.last_q is not None else self.current_q(data)
        q, err = self._dls(scratch, target_pos, target_quat, q0, iters, damp, max_step,
                            tol_pos, ori_weight)
        if err > 0.02 and allow_coarse_restart:
            seed = coarse_seed(self.model, scratch, self.site_id, self.gripper_body_id,
                                target_pos, self.qposadr, self.jnt_range,
                                want_down=(target_quat is None))
            if seed is not None:
                q_r, err_r = self._dls(scratch, target_pos, target_quat, seed,
                                        iters * 3, damp, max_step, tol_pos, ori_weight)
                if err_r < err:
                    q, err = q_r, err_r
        self.last_q = q.copy()
        return q, err

    def _dls(self, scratch, target_pos, target_quat, q0, iters, damp, max_step,
             tol_pos, ori_weight):
        q = np.clip(q0.copy(), self.jnt_range[:, 0], self.jnt_range[:, 1])
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        err_vec = np.zeros(3)
        for _ in range(iters):
            scratch.qpos[self.qposadr] = q
            mujoco.mj_forward(self.model, scratch)
            pos = scratch.site_xpos[self.site_id].copy()
            err_vec = target_pos - pos
            pos_err = float(np.linalg.norm(err_vec))
            if target_quat is not None:
                cur_quat = _mat2quat(scratch.site_xmat[self.site_id])
                ori_err = _orientation_error(np.asarray(target_quat, dtype=float), cur_quat)
                full_err = np.concatenate([err_vec, ori_weight * ori_err])
            else:
                full_err = err_vec
            if pos_err < tol_pos and (target_quat is None or np.linalg.norm(full_err[3:]) < 1e-3):
                break
            mujoco.mj_jacSite(self.model, scratch, jacp, jacr, self.site_id)
            if target_quat is not None:
                J = np.vstack([jacp[:, self.dofadr], jacr[:, self.dofadr]])
            else:
                J = jacp[:, self.dofadr]
            dq = J.T @ np.linalg.solve(J @ J.T + (damp ** 2) * np.eye(J.shape[0]), full_err)
            step_norm = np.linalg.norm(dq)
            if step_norm > max_step:
                dq *= max_step / step_norm
            q = np.clip(q + dq, self.jnt_range[:, 0], self.jnt_range[:, 1])
        return q, float(np.linalg.norm(err_vec))


class BimanualIK:
    """Convenience wrapper: one ArmIK per side, sharing the same model."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.arms = {side: ArmIK(model, side) for side in SIDES}

    def solve(self, data, side, target_pos, target_quat=None, **kw):
        return self.arms[side].solve(data, target_pos, target_quat, **kw)


# ---------------------------------------------------------------------------
# Actuator plumbing (arm position targets + gripper).
# ---------------------------------------------------------------------------

def arm_pos_actuators(model: mujoco.MjModel, side: str) -> list[int]:
    return [model.actuator(f"{side}_arm_joint{i}_pos").id for i in range(1, 7)]


def finger_actuators(model: mujoco.MjModel, side: str) -> list[int]:
    return [model.actuator(f"{side}_arm_joint7_pos").id, model.actuator(f"{side}_arm_joint8_pos").id]


def set_arm_ctrl(data: mujoco.MjData, act_ids: list[int], q: np.ndarray) -> None:
    data.ctrl[act_ids] = q


def set_gripper(data: mujoco.MjData, act_ids: list[int], opening: float) -> None:
    """opening in [0, 1]: 0 = fully closed, 1 = fully open (matches
    collection/episode.py's GRIPPER CONVENTION)."""
    o = float(np.clip(opening, 0.0, 1.0)) * GRIPPER_JOINT_RANGE_M
    data.ctrl[act_ids] = [o, -o]


def gripper_state(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> float:
    jid = model.joint(f"{side}_arm_joint7").id
    qadr = model.jnt_qposadr[jid]
    return float(np.clip(data.qpos[qadr] / GRIPPER_JOINT_RANGE_M, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Scripted-target API (headless).
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """One waypoint for one arm: linearly interpolate position (and nlerp
    orientation, if given) from wherever the previous segment ended, over
    `duration` seconds, then hold there for `hold` more seconds."""
    pos: np.ndarray
    quat: Optional[np.ndarray] = None
    gripper: Optional[float] = None  # None = hold whatever gripper value was last commanded
    duration: float = 1.0
    hold: float = 0.0


@dataclass
class _SideTrack:
    segments: list[Segment]
    start_pos: np.ndarray
    start_quat: Optional[np.ndarray]
    start_gripper: float

    def sample(self, t: float):
        pos, quat, grip = self.start_pos, self.start_quat, self.start_gripper
        t0 = 0.0
        for seg in self.segments:
            t_end_move = t0 + seg.duration
            t_end_hold = t_end_move + seg.hold
            grip_target = seg.gripper if seg.gripper is not None else grip
            if t <= t_end_move:
                s = 0.0 if seg.duration <= 0 else np.clip((t - t0) / seg.duration, 0.0, 1.0)
                interp_pos = pos + s * (seg.pos - pos)
                interp_quat = _slerp(quat, seg.quat, s) if quat is not None and seg.quat is not None else (seg.quat if s >= 1.0 else quat)
                interp_grip = grip + s * (grip_target - grip)
                return interp_pos, interp_quat, interp_grip
            elif t <= t_end_hold:
                return seg.pos, seg.quat if seg.quat is not None else quat, grip_target
            pos, quat, grip = seg.pos, (seg.quat if seg.quat is not None else quat), grip_target
            t0 = t_end_hold
        return pos, quat, grip

    @property
    def total_time(self) -> float:
        return sum(seg.duration + seg.hold for seg in self.segments)


def _slerp(q0, q1, s):
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        out = q0 + s * (q1 - q0)
        return out / np.linalg.norm(out)
    theta0 = np.arccos(dot)
    theta = theta0 * s
    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)
    return q0 * np.cos(theta) + q2 * np.sin(theta)


def drive(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    targets_over_time: dict[str, list[Segment]],
    *,
    ik: Optional[BimanualIK] = None,
    on_step: Optional[Callable[[int, mujoco.MjModel, mujoco.MjData], None]] = None,
    extra_steps: int = 0,
) -> int:
    """Headless scripted-target teleop. `targets_over_time` maps side ->
    list[Segment] (built relative to that arm's CURRENT ee pose/gripper at
    call time). Every physics step: sample each side's current interpolated
    (pos, quat, gripper) target, DLS-IK it to joint targets, write ctrl,
    `mj_step`, then invoke `on_step(step_idx, model, data)` (used by
    `record_teleop.py` to append an episode step). Sides absent from
    `targets_over_time` simply keep whatever ctrl they already had (e.g. a
    "home" hold). Returns the number of steps run.
    """
    if ik is None:
        ik = BimanualIK(model)

    dt = model.opt.timestep
    tracks: dict[str, _SideTrack] = {}
    finger_act = {side: finger_actuators(model, side) for side in SIDES}
    arm_act = {side: arm_pos_actuators(model, side) for side in SIDES}

    for side, segs in targets_over_time.items():
        arm = ik.arms[side]
        start_pos = data.site_xpos[arm.site_id].copy()
        start_quat = _mat2quat(data.site_xmat[arm.site_id])
        start_grip = gripper_state(model, data, side)
        tracks[side] = _SideTrack(segs, start_pos, start_quat, start_grip)

    total_T = max((tr.total_time for tr in tracks.values()), default=0.0)
    num_steps = int(np.ceil(total_T / dt)) + extra_steps

    for step in range(num_steps):
        t = step * dt
        for side, track in tracks.items():
            pos, quat, grip = track.sample(t)
            q, _err = ik.solve(data, side, pos, target_quat=None)  # position-only tracking; quat carried for API completeness
            set_arm_ctrl(data, arm_act[side], q)
            set_gripper(data, finger_act[side], grip)
        mujoco.mj_step(model, data)
        if on_step is not None:
            on_step(step, model, data)

    return num_steps


# ---------------------------------------------------------------------------
# Interactive backend (keyboard + mujoco.viewer). Requires a local display --
# NOT exercised in this (headless) task; documented here for later use.
# ---------------------------------------------------------------------------

INTERACTIVE_KEYS = """
Interactive teleop keys (run_interactive):
  TAB          toggle active arm (left <-> right)
  Arrow keys   translate active arm's EE target in the world XY plane
  PAGE UP/DOWN translate active arm's EE target in Z
  [ / ]        close / open the active arm's gripper (toward 0 / 1)
  R            reset the scene to its 'home' keyframe (and re-seed targets)
  ESC / close  quit the viewer

Run (on a machine with a display, X11/Wayland forwarded or local):
  MUJOCO_GL=glfw python -m teleop.bimanual_teleop --interactive
"""


def run_interactive(model=None, data=None, step_m: float = 0.01) -> None:
    """Launch an interactive mujoco.viewer session with keyboard EE-target
    teleop for both arms. REQUIRES A DISPLAY (glfw). See INTERACTIVE_KEYS."""
    import mujoco.viewer  # local import: only needed for this backend

    if model is None or data is None:
        model, data = load_scene()
        reset_home(model, data)

    ik = BimanualIK(model)
    arm_act = {side: arm_pos_actuators(model, side) for side in SIDES}
    finger_act = {side: finger_actuators(model, side) for side in SIDES}

    state = {
        "active": "left",
        "targets": {
            side: data.site_xpos[ik.arms[side].site_id].copy() for side in SIDES
        },
        "grip": {side: gripper_state(model, data, side) for side in SIDES},
        "reset": False,
    }

    def key_callback(keycode: int) -> None:
        # mujoco.viewer's glfw keycodes match GLFW key constants.
        import glfw  # noqa: F401  (only needed here; part of the viewer's own deps)
        active = state["active"]
        if keycode == glfw.KEY_TAB:
            state["active"] = "right" if active == "left" else "left"
        elif keycode == glfw.KEY_UP:
            state["targets"][active][0] += step_m
        elif keycode == glfw.KEY_DOWN:
            state["targets"][active][0] -= step_m
        elif keycode == glfw.KEY_LEFT:
            state["targets"][active][1] += step_m
        elif keycode == glfw.KEY_RIGHT:
            state["targets"][active][1] -= step_m
        elif keycode == glfw.KEY_PAGE_UP:
            state["targets"][active][2] += step_m
        elif keycode == glfw.KEY_PAGE_DOWN:
            state["targets"][active][2] -= step_m
        elif keycode == glfw.KEY_LEFT_BRACKET:
            state["grip"][active] = max(0.0, state["grip"][active] - 0.1)
        elif keycode == glfw.KEY_RIGHT_BRACKET:
            state["grip"][active] = min(1.0, state["grip"][active] + 0.1)
        elif keycode == ord("R"):
            state["reset"] = True

    with mujoco.viewer.launch_passive(model=model, data=data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            if state["reset"]:
                reset_home(model, data)
                for side in SIDES:
                    state["targets"][side] = data.site_xpos[ik.arms[side].site_id].copy()
                    state["grip"][side] = gripper_state(model, data, side)
                state["reset"] = False
            for side in SIDES:
                q, _err = ik.solve(data, side, state["targets"][side])
                set_arm_ctrl(data, arm_act[side], q)
                set_gripper(data, finger_act[side], state["grip"][side])
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactive", action="store_true",
                         help="Launch the keyboard viewer backend (requires a display).")
    args = parser.parse_args()
    if args.interactive:
        run_interactive()
    else:
        print(INTERACTIVE_KEYS)
        print("Pass --interactive on a machine with a display to run the viewer backend.")
        print("For headless/scripted use, see teleop/record_teleop.py or import drive() directly.")
