"""Controller-agnostic REAL friction grasp of a cube with one YOR gripper (default: RIGHT).

Replaces the flagship's magnet weld (``cup_collect_task_utils.activate_weld``) with a
genuine two-jaw friction pinch. This module ONLY commands the two finger actuators and reads
contacts, so it can run alongside ANY arm controller (the OSC ``ArmController`` now,
FL-MPC later). Per physics tick the caller does::

    arm_controller_step()          # writes data.qfrc_applied on the 6 arm DOFs
    gripper.step()                 # writes data.ctrl for the 2 finger actuators (LAST: wins)
    mujoco.mj_step(model, data)

(``ArmController.step`` also writes finger ctrl via its own GripForceController; calling
``CubeGripper.step`` AFTER it overwrites that, so the cube gripper owns the fingers.)

Grip actuation modes (``CubeGripper(mode=...)``):

* ``"motor"`` (default) -- plan Option B, a DOCUMENTED FIDELITY CHANGE: the side's two
  finger position servos are switched IN MEMORY (compiled MjModel only; no MJCF edit) to
  direct-force motors, ``gain=1, bias=0``, with ``ctrlrange = forcerange = +-20 N`` (the
  UNMODIFIED YOR finger limit -- never exceeded). Positioning (open / pre-shape) is a
  software PD with the stock servo gain ``kp=200`` (identical to the stock servo law,
  which has no velocity term); gripping writes the closing force ``F`` directly
  (``ctrl7=-F, ctrl8=+F``) -> true closed-form grip force.
* ``"servo"`` -- stock position servos untouched; the grip is "command fully closed", so
  the contact force is ``kp * q_contact`` = 200 * 0.0105 ~= 2.1 N per pad on the 2.1 cm
  cube (the ctrlrange clip at 0 caps it) regardless of the requested force. Full model
  fidelity, but no force control. Kept for comparison / the stock-model story.

Finger conventions (yor_robot_derived.mjcf): joint7 in [0, 0.035] (axis +z), joint8 in
[-0.035, 0] (axis -z); (0, 0) is closed. Pad face gap = 2*q7 (pad thickness 5 mm), so
an unloaded 2.1 cm cube is touched at q7 ~= 0.0105. Finger slides have damping 100
N s/m, so an 8 N close moves at <= ~8 cm/s -- no slam.

Collision prerequisite: jaw<->cube contact must be ON. Load with
``restore_finger_object_collision=True`` (jaw geoms gain bit0 -> contact any cube
regardless of the cube's CUBE_ARM_BITS), or keep the cube's arm bits set (jaw contype has
the arm bit) -- i.e. do NOT call ``sim_common.set_cube_arm_collision(..., False)`` unless
bit0 was restored. ``finger_cube_contact_enabled`` checks the bitmasks.

Pre-grasp: ``plan_top_down_pregrasp`` returns a top-down (approach -z) pose whose jaws
straddle the cube, aligned with the cube's faces, as BOTH an EE-body pose
(for Cartesian controllers, e.g. OSC on ``*_gripper_base``) and an EE-site pose + IK
joint configs (for joint-space controllers, e.g. FL-MPC waypoints; reuses
``teleop.bimanual_teleop.ArmIK`` read-only, site ``*_arm_ee_grasp``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SIM_COMMON_DIR = _REPO_ROOT / "experiments" / "ab_smooth_vs_bangbang"
for _p in (str(_SIM_COMMON_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GRIPPER_JOINT_RANGE_M = 0.035
FINGER_KP = 200.0               # stock class="finger" position gain
FINGER_FORCE_LIMIT = 20.0       # stock class="finger" forcerange (+-20) -- never exceeded
DEFAULT_GRIP_FORCE_N = 8.0

_ARM_ACTS = {s: [f"{s}_arm_joint{i}_pos" for i in range(1, 7)] for s in ("left", "right")}
_ARM_JOINTS = {s: [f"{s}_arm_joint{i}" for i in range(1, 7)] for s in ("left", "right")}
_FINGER_ACTS = {s: (f"{s}_arm_joint7_pos", f"{s}_arm_joint8_pos") for s in ("left", "right")}
_FINGER_JOINTS = {s: (f"{s}_arm_joint7", f"{s}_arm_joint8") for s in ("left", "right")}
_PADS = {s: (f"{s}_arm_link7_pad", f"{s}_arm_link8_pad") for s in ("left", "right")}
_EE_BODY = {s: f"{s}_arm_gripper_base" for s in ("left", "right")}
_EE_SITE = {s: f"{s}_arm_ee_grasp" for s in ("left", "right")}


# --------------------------------------------------------------------------------------
# In-memory model patches (compiled MjModel only; scene files never touched)
# --------------------------------------------------------------------------------------
def disable_arm_position_servos(model, side: str = "right") -> None:
    """Zero gain/bias of the side's 6 ARM joint position actuators so torque control via
    ``data.qfrc_applied`` owns those joints (mirror of
    sim_common.disable_left_arm_position_servos). Finger actuators untouched."""
    for name in _ARM_ACTS[side]:
        aid = model.actuator(name).id
        model.actuator_gainprm[aid, :] = 0.0
        model.actuator_biasprm[aid, :] = 0.0


def disable_right_arm_position_servos(model) -> None:
    disable_arm_position_servos(model, "right")


def set_finger_motor_mode(model, side: str = "right",
                          force_limit: float = FINGER_FORCE_LIMIT) -> None:
    """FIDELITY CHANGE (plan Option B): switch the side's two finger position servos to
    direct-force motors (ctrl = force in N on the slide), keeping forcerange at the
    stock +-20 N. In-memory only. Idempotent."""
    force_limit = float(min(abs(force_limit), FINGER_FORCE_LIMIT))
    for name in _FINGER_ACTS[side]:
        aid = model.actuator(name).id
        model.actuator_gainprm[aid, :] = 0.0
        model.actuator_gainprm[aid, 0] = 1.0
        model.actuator_biasprm[aid, :] = 0.0
        model.actuator_ctrllimited[aid] = 1
        model.actuator_ctrlrange[aid] = (-force_limit, force_limit)
        model.actuator_forcelimited[aid] = 1
        model.actuator_forcerange[aid] = (-force_limit, force_limit)


def is_finger_motor_mode(model, side: str = "right") -> bool:
    ok = True
    for name in _FINGER_ACTS[side]:
        aid = model.actuator(name).id
        ok &= bool(model.actuator_gainprm[aid, 0] == 1.0 and not np.any(model.actuator_biasprm[aid, :3]))
        ok &= bool(model.actuator_forcerange[aid, 1] <= FINGER_FORCE_LIMIT + 1e-9)
    return ok


def finger_cube_contact_enabled(model, cube_body_id: int, side: str = "right") -> bool:
    """True iff EVERY jaw collision geom (pad + fingertip box, both fingers) passes
    MuJoCo's contype/conaffinity test against every collision geom of the cube body
    (the prerequisite for a real pinch)."""
    j7, j8 = jaw_geom_ids(model, side)
    pads = list(j7) + list(j8)
    cube_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == cube_body_id
                  and (model.geom_contype[g] or model.geom_conaffinity[g])]
    if not cube_geoms:
        return False
    for p in pads:
        for g in cube_geoms:
            if not ((model.geom_contype[p] & model.geom_conaffinity[g])
                    or (model.geom_contype[g] & model.geom_conaffinity[p])):
                return False
    return True


# --------------------------------------------------------------------------------------
# Contact sensing
# --------------------------------------------------------------------------------------
# A YOR jaw (finger body link7 / link8) has TWO coplanar collision boxes on its inner
# face: the named "*_pad" (3x3 cm, s ~ 7.6-10.6 cm from gripper_base along the approach
# axis) and an unnamed fingertip box (s ~ 10.6-13.6 cm). A cube resting on a table can
# only be reached by the fingertip half (the tip would otherwise hit the table), so
# contacts are counted per JAW (any collision geom of the finger body), with the named-pad
# subset reported separately.
def jaw_geom_ids(model, side: str = "right") -> tuple[tuple[int, ...], tuple[int, ...]]:
    out = []
    for fb in (f"{side}_arm_link7", f"{side}_arm_link8"):
        bid = model.body(fb).id
        out.append(tuple(g for g in range(model.ngeom) if model.geom_bodyid[g] == bid
                         and (model.geom_contype[g] or model.geom_conaffinity[g])))
    return out[0], out[1]


@dataclass
class JawContact:
    n_jaw7: int              # contacts jaw7 (finger link7, pad+tip) <-> cube
    n_jaw8: int
    fn_jaw7: float           # summed normal force jaw7 <-> cube [N]
    fn_jaw8: float
    n_named_pad: int         # subset of the above on the named *_pad geoms
    n_other_arm: int         # contacts cube <-> any OTHER geom of this arm (palm capsule...)

    @property
    def n_total(self) -> int:
        return self.n_jaw7 + self.n_jaw8

    @property
    def fn_total(self) -> float:
        return self.fn_jaw7 + self.fn_jaw8

    @property
    def both_jaws(self) -> bool:
        return self.n_jaw7 > 0 and self.n_jaw8 > 0


def jaw_cube_contacts(model, data, cube_body_id: int, side: str = "right") -> JawContact:
    j7, j8 = jaw_geom_ids(model, side)
    pads = {model.geom(n).id for n in _PADS[side]}
    prefix = f"{side}_arm_"
    n7 = n8 = n_pad = n_other = 0
    f7 = f8 = 0.0
    f6 = np.zeros(6)
    for c in range(data.ncon):
        con = data.contact[c]
        g1, g2 = con.geom1, con.geom2
        b1, b2 = model.geom_bodyid[g1], model.geom_bodyid[g2]
        if b1 == cube_body_id:
            other, ob = g2, b2
        elif b2 == cube_body_id:
            other, ob = g1, b1
        else:
            continue
        if other in j7 or other in j8:
            mujoco.mj_contactForce(model, data, c, f6)
            fn = abs(float(f6[0]))
            if other in j7:
                n7 += 1
                f7 += fn
            else:
                n8 += 1
                f8 += fn
            n_pad += int(other in pads)
        elif (model.body(ob).name or "").startswith(prefix):
            n_other += 1
    return JawContact(n7, n8, f7, f8, n_pad, n_other)


def pad_midpoint(model, data, side: str = "right") -> np.ndarray:
    """Midpoint of the two named pad geoms (world)."""
    p7, p8 = model.geom(_PADS[side][0]).id, model.geom(_PADS[side][1]).id
    return 0.5 * (data.geom_xpos[p7] + data.geom_xpos[p8])


# --------------------------------------------------------------------------------------
# The controller-agnostic gripper
# --------------------------------------------------------------------------------------
class CubeGripper:
    """Finger-only cube grip state machine. Call ``step()`` once per physics tick,
    AFTER the arm controller's step. Modes: 'open' -> 'closing' -> 'holding';
    'release' (opening, drops the cube). ``is_held()`` = both jaws touching the cube
    with summed normal force >= ``force_floor_n``."""

    def __init__(self, model, data, side: str = "right", *, mode: str = "motor",
                 kp: float = FINGER_KP, force_limit: float = FINGER_FORCE_LIMIT,
                 force_floor_n: float = 0.5, ramp_s: float = 0.3) -> None:
        if mode not in ("motor", "servo"):
            raise ValueError(f"mode must be motor|servo; got {mode!r}")
        self.model, self.data, self.side, self.mode = model, data, side, mode
        self.kp = float(kp)
        self.force_limit = float(min(force_limit, FINGER_FORCE_LIMIT))
        self.force_floor_n = float(force_floor_n)
        self.ramp_steps = max(1, int(round(ramp_s / model.opt.timestep)))
        self.act7, self.act8 = (model.actuator(n).id for n in _FINGER_ACTS[side])
        self.q7adr, self.q8adr = (int(model.joint(n).qposadr[0]) for n in _FINGER_JOINTS[side])
        self.pad_ids = tuple(model.geom(n).id for n in _PADS[side])
        if mode == "motor":
            set_finger_motor_mode(model, side, self.force_limit)
        self.state = "open"
        self._opening = 1.0
        self._force_target = 0.0
        self._force_cmd = 0.0
        self._k = 0
        self.cube_body_id: int | None = None

    # ---- commands --------------------------------------------------------------------
    def open_fingers(self, opening: float = 1.0) -> None:
        """Position-mode opening in [0,1] (1 = fully open, 0.035 m per finger)."""
        self._opening = float(np.clip(opening, 0.0, 1.0))
        self.state = "open"
        self._force_cmd = 0.0

    def close_on_cube(self, cube, force_n: float = DEFAULT_GRIP_FORCE_N,
                      ramp_s: float | None = None) -> None:
        """Start a force-ramped close on ``cube`` (body name or id). Force ramps 0 ->
        ``force_n`` (clipped to the +-20 N limit) over ``ramp_s``; state becomes
        'holding' once ramp is done and ``is_held()``."""
        self.cube_body_id = self._bid(cube)
        self._force_target = float(np.clip(force_n, 0.0, self.force_limit))
        if ramp_s is not None:
            self.ramp_steps = max(1, int(round(ramp_s / self.model.opt.timestep)))
        self._k = 0
        self.state = "closing"

    def hold(self, force_n: float | None = None) -> None:
        """Maintain (or change) the grip force every step."""
        if force_n is not None:
            self._force_target = float(np.clip(force_n, 0.0, self.force_limit))
        self._k = self.ramp_steps
        self.state = "holding"

    def release(self, opening: float = 1.0) -> None:
        self._opening = float(np.clip(opening, 0.0, 1.0))
        self._force_cmd = 0.0
        self.state = "release"

    # ---- per-tick --------------------------------------------------------------------
    def step(self) -> None:
        d = self.data
        if self.state in ("closing", "holding"):
            self._k = min(self._k + 1, self.ramp_steps)
            self._force_cmd = self._force_target * self._k / self.ramp_steps
            if self.mode == "motor":
                f = self._force_cmd
                d.ctrl[self.act7] = -f          # joint7 -> 0 (close)
                d.ctrl[self.act8] = +f          # joint8 -> 0 (close)
            else:  # stock servo: command closed; realised force = kp*q (capped +-20)
                d.ctrl[self.act7] = 0.0
                d.ctrl[self.act8] = 0.0
            if self.state == "closing" and self._k >= self.ramp_steps and self.is_held():
                self.state = "holding"
        else:  # open / release: position the fingers
            o = self._opening * GRIPPER_JOINT_RANGE_M
            if self.mode == "motor":
                q7 = float(d.qpos[self.q7adr])
                q8 = float(d.qpos[self.q8adr])
                lim = self.force_limit
                d.ctrl[self.act7] = float(np.clip(self.kp * (o - q7), -lim, lim))
                d.ctrl[self.act8] = float(np.clip(self.kp * (-o - q8), -lim, lim))
            else:
                d.ctrl[self.act7] = o
                d.ctrl[self.act8] = -o

    # ---- sensing ---------------------------------------------------------------------
    def contacts(self, cube=None) -> JawContact:
        bid = self._bid(cube) if cube is not None else self.cube_body_id
        if bid is None:
            return JawContact(0, 0, 0.0, 0.0, 0, 0)
        return jaw_cube_contacts(self.model, self.data, bid, self.side)

    def is_held(self, cube=None) -> bool:
        pc = self.contacts(cube)
        return pc.both_jaws and pc.fn_total >= self.force_floor_n

    def closed_empty(self, cube_half: float = 0.0105, tol: float = 0.003) -> bool:
        """True if the jaws have closed past the cube width without holding it (a miss)."""
        return float(self.data.qpos[self.q7adr]) < cube_half - tol and not self.is_held()

    def actuator_forces(self) -> np.ndarray:
        """Realised finger actuator forces [N] (post forcerange clamp)."""
        return np.array([self.data.actuator_force[self.act7], self.data.actuator_force[self.act8]])

    def finger_q(self) -> np.ndarray:
        return np.array([self.data.qpos[self.q7adr], self.data.qpos[self.q8adr]])

    @property
    def force_cmd(self) -> float:
        return self._force_cmd

    def _bid(self, cube) -> int:
        return int(cube) if isinstance(cube, (int, np.integer)) else self.model.body(cube).id


# --------------------------------------------------------------------------------------
# Pre-grasp pose (top-down, jaws straddling the cube)
# --------------------------------------------------------------------------------------
@dataclass
class PreGrasp:
    cube_body_id: int
    cube_pos: np.ndarray
    ee_quat: np.ndarray        # EE BODY (gripper_base) quat wxyz, approach axis = -world z
    ee_pos_hover: np.ndarray   # EE BODY targets (Cartesian controllers)
    ee_pos_at: np.ndarray
    site_quat: np.ndarray      # EE SITE (*_ee_grasp) targets (ArmIK / joint-space controllers)
    site_pos_hover: np.ndarray
    site_pos_at: np.ndarray
    q_hover: np.ndarray        # IK joint configs (6,)
    q_at: np.ndarray
    ik_err_hover: float
    ik_err_at: float
    yaw: float                 # world yaw of the finger-slide axis (EE-body y)
    tip_local: np.ndarray      # jaw-centre fingertip point in the EE-body frame


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=np.float64).reshape(9))
    return q


def jaw_tip_in_ee_frame(model, data, side: str = "right") -> np.ndarray:
    """Jaw-centre fingertip point in the EE-body frame: x,y = pad midpoint (centre
    between the jaws), z = farthest extent of any jaw collision box along the approach
    axis (EE-body +z). Computed on a scratch copy with symmetric open fingers (the value
    is independent of the symmetric opening). YOR: ~(0, 0, 0.136) m."""
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    q7a = int(model.joint(_FINGER_JOINTS[side][0]).qposadr[0])
    q8a = int(model.joint(_FINGER_JOINTS[side][1]).qposadr[0])
    scratch.qpos[q7a], scratch.qpos[q8a] = 0.03, -0.03
    mujoco.mj_kinematics(model, scratch)
    b = model.body(_EE_BODY[side]).id
    R, p = scratch.xmat[b].reshape(3, 3), scratch.xpos[b]
    mid = R.T @ (pad_midpoint(model, scratch, side) - p)
    zmax = -np.inf
    j7, j8 = jaw_geom_ids(model, side)
    for g in j7 + j8:
        Rg = scratch.geom_xmat[g].reshape(3, 3)
        h = model.geom_size[g]
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corner = scratch.geom_xpos[g] + Rg @ (np.array([sx, sy, sz]) * h)
                    zmax = max(zmax, float((R.T @ (corner - p))[2]))
    return np.array([mid[0], mid[1], zmax])


def _cube_half_height(model, data, bid: int) -> float:
    """Half extent of the cube body's (box) geom along world z."""
    best = 0.0
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == bid and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            Rg = data.geom_xmat[g].reshape(3, 3)
            best = max(best, float(np.abs(Rg[2, :]) @ model.geom_size[g]))
    return best


def ee_targets_for_tip(tip_world: np.ndarray, R: np.ndarray, tip_local: np.ndarray,
                       site_local_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(EE-body position, EE-site position) that put the jaw tip point at ``tip_world``
    with EE-body rotation ``R``."""
    ee = np.asarray(tip_world, float) - R @ tip_local
    return ee, ee + R @ site_local_pos


def plan_top_down_pregrasp(model, data, cube, side: str = "right", *,
                           hover: float = 0.10, tip_clearance: float = 0.004,
                           ik=None, prefer_q: np.ndarray | None = None) -> PreGrasp:
    """Top-down grasp of an upright cube: approach axis (EE-body +z) = -world z, finger
    slide axis (EE-body y) aligned with one of the cube's horizontal face normals (4 yaw
    candidates; chosen by IK error, then joint travel from ``prefer_q``/current q).

    Depth: the jaw FINGERTIPS stop ``tip_clearance`` above the cube's bottom face (= the
    table), so the jaws cover the cube's full side (the fingertip box spans 3 cm) and the
    gripper_base palm capsule (lowest point ~2.6 cm above the tips) clears the cube top.
    (The named *_pad geoms sit ABOVE the fingertip boxes; on a 2.1 cm cube they cannot be
    lowered to the cube without driving the tips into the table -- the grip is on the
    fingertip half of each jaw.)

    ``ik`` defaults to ``teleop.bimanual_teleop.ArmIK`` (read-only reuse, pure sim DLS on
    ``*_arm_ee_grasp``)."""
    if ik is None:
        from teleop import bimanual_teleop as bt   # read-only reuse (pure sim IK)
        ik = bt.ArmIK(model, side)
    bid = int(cube) if isinstance(cube, (int, np.integer)) else model.body(cube).id
    cpos = data.xpos[bid].copy()
    Rc = data.xmat[bid].reshape(3, 3)
    ax = Rc[:, 0] if abs(Rc[2, 0]) < 0.7 else Rc[:, 1]    # a horizontal cube axis
    cube_yaw = float(np.arctan2(ax[1], ax[0]))
    half_z = _cube_half_height(model, data, bid)

    tip_local = jaw_tip_in_ee_frame(model, data, side)
    s_local_pos = model.site_pos[ik.site_id].copy()
    s_local_quat = model.site_quat[ik.site_id].copy()
    q_now = data.qpos[ik.qposadr].copy() if prefer_q is None else np.asarray(prefer_q, float)
    tip_at = cpos + np.array([0.0, 0.0, -half_z + tip_clearance])
    tip_hover = tip_at + np.array([0.0, 0.0, hover])
    ee_bid = model.body(_EE_BODY[side]).id
    scratch = mujoco.MjData(model)

    best = None
    for k in range(4):
        psi = cube_yaw + k * 0.5 * np.pi
        z_b = np.array([0.0, 0.0, -1.0])
        y_b = np.array([np.cos(psi), np.sin(psi), 0.0])
        x_b = np.cross(y_b, z_b)
        R = np.column_stack([x_b, y_b, z_b])
        qb = _rotmat_to_quat(R)
        qs = np.zeros(4)
        mujoco.mju_mulQuat(qs, qb, s_local_quat)
        ee_at, s_at = ee_targets_for_tip(tip_at, R, tip_local, s_local_pos)
        ee_hover, s_hover = ee_targets_for_tip(tip_hover, R, tip_local, s_local_pos)
        ik.last_q = q_now.copy()
        q_h, e_h = ik.solve(data, s_hover, qs, ori_weight=1.0, iters=200)
        ik.last_q = q_h.copy()
        q_a, e_a = ik.solve(data, s_at, qs, ori_weight=1.0, iters=200)
        scratch.qpos[:] = data.qpos
        scratch.qpos[ik.qposadr] = q_a
        mujoco.mj_kinematics(model, scratch)
        ori_err = float(np.linalg.norm(scratch.xmat[ee_bid].reshape(3, 3) - R))
        score = (round(max(e_h, e_a) + 0.05 * ori_err, 3), float(np.linalg.norm(q_h - q_now)))
        cand = PreGrasp(bid, cpos, qb, ee_hover, ee_at, qs, s_hover, s_at, q_h, q_a,
                        float(e_h), float(e_a), float(psi), tip_local)
        if best is None or score < best[0]:
            best = (score, cand)
    return best[1]
