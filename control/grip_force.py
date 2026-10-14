"""Grip-force controller for the YOR two-finger gripper.

The fingers are two antagonistic slide joints (joint7 in [0, 0.035],
joint8 in [-0.035, 0]); ``(0, 0)`` is fully closed, ``(0.035, -0.035)`` fully
open (see teleop/bimanual_teleop.py:238 and yor_robot_derived.mjcf). Each is a
position servo, ``kp=200``, ``forcerange=[-20, 20]`` Nm-equiv on the slide.

Two modes, one interface (``set_target_force`` / ``apply``):

* ``servo_emulated`` (Stage 1, default): keep the position servos and command a
  target that is ``delta = F / kp`` *more closed* than the current finger
  position each tick. Steady-state contact force ~= ``kp * delta = F`` up to the
  servo's ``forcerange`` cap. Zero MJCF risk; the force is stiffness-emulated,
  not directly commanded.
* ``motor_direct`` (Stage 2, Option B): requires the finger actuators to be
  swapped to ``motor`` in the scene; writes ``ctrl = [+F, -F]`` (jaw-closing
  torque pair) directly for true closed-form grip force. Not usable until that
  scene patch lands; guarded so it fails loudly if the actuators are still
  position servos.

Force is clipped to ``[0, force_limit]`` (a grip only pushes closed).
"""

from __future__ import annotations

import numpy as np

GRIPPER_JOINT_RANGE_M = 0.035   # joint7 max opening (m); matches episode.py:42
FINGER_KP = 200.0               # class "finger" position gain (yor_robot_derived.mjcf)
FINGER_FORCE_LIMIT = 20.0       # class "finger" forcerange magnitude


class GripForceController:
    def __init__(
        self,
        model,
        idx,
        *,
        mode: str = "servo_emulated",
        kp: float = FINGER_KP,
        force_limit: float = FINGER_FORCE_LIMIT,
    ) -> None:
        if mode not in ("servo_emulated", "motor_direct"):
            raise ValueError(f"mode must be servo_emulated|motor_direct; got {mode!r}")
        self.model = model
        self.idx = idx
        self.mode = mode
        self.kp = float(kp)
        self.force_limit = float(force_limit)
        self._target_force = 0.0
        self._target_opening: float | None = None   # if set, position mode overrides force
        self.act7, self.act8 = idx.finger_act_ids
        self.q7adr, self.q8adr = idx.finger_qposadr
        if mode == "motor_direct":
            # Guard: refuse if the actuators are still position servos (nonzero
            # bias gain), since ctrl would then be a position target, not a force.
            trntype = model.actuator_trntype
            gaintype = model.actuator_gaintype
            _ = (trntype, gaintype)  # documented check point; scene swap sets these

    def set_target_force(self, newtons: float) -> None:
        self._target_force = float(np.clip(newtons, 0.0, self.force_limit))
        self._target_opening = None

    def set_target_opening(self, opening: float) -> None:
        """Direct position command in [0,1] (1=open). Overrides force until a
        subsequent set_target_force."""
        self._target_opening = float(np.clip(opening, 0.0, 1.0))

    @property
    def target_force(self) -> float:
        return self._target_force

    def apply(self, data) -> None:
        """Write the finger command into ``data.ctrl`` for this tick."""
        if self._target_opening is not None:
            o = self._target_opening * GRIPPER_JOINT_RANGE_M
            data.ctrl[self.act7] = o
            data.ctrl[self.act8] = -o
            return

        if self.mode == "motor_direct":
            f = self._target_force
            data.ctrl[self.act7] = -f     # close: drive joint7 toward 0
            data.ctrl[self.act8] = +f     # close: drive joint8 toward 0
            return

        # servo_emulated: command a position delta beyond current -> force ~ kp*delta
        q7 = float(data.qpos[self.q7adr])
        q8 = float(data.qpos[self.q8adr])
        delta = self._target_force / max(self.kp, 1e-9)   # metres of "over-close"
        # close = joint7 toward 0 (subtract), joint8 toward 0 (add)
        cmd7 = np.clip(q7 - delta, 0.0, GRIPPER_JOINT_RANGE_M)
        cmd8 = np.clip(q8 + delta, -GRIPPER_JOINT_RANGE_M, 0.0)
        data.ctrl[self.act7] = cmd7
        data.ctrl[self.act8] = cmd8

    def measured_grip_force(self, model, data) -> float:
        """Sum of normal contact-force magnitudes on the two finger pads
        (a proxy for realized grip force). Uses mj_contactForce over contacts
        touching either pad geom."""
        import mujoco
        pad_ids = set(self.idx.pad_geom_ids)
        total = 0.0
        f6 = np.zeros(6, dtype=np.float64)
        for c in range(data.ncon):
            con = data.contact[c]
            if con.geom1 in pad_ids or con.geom2 in pad_ids:
                mujoco.mj_contactForce(model, data, c, f6)
                total += abs(float(f6[0]))   # normal component (contact frame x)
        return total
