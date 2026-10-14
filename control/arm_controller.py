"""ArmController — programmatic control API for one YOR arm.

One honest interface replacing the scattered teleport/weld hacks:

    arm = ArmController(model, data, "left")
    arm.set_ee_target(pos, quat)      # world-frame Cartesian target
    arm.set_grip_force(10.0)          # Newtons
    for _ in range(N):
        arm.step()                    # ONE control tick (writes qfrc_applied + grip)
        mujoco.mj_step(model, data)   # caller advances physics

Per tick, ``step()``:
  1. reads per-arm state (state_adapter.build_state),
  2. runs the 6-DOF Cartesian-impedance QP (cartesian_impedance),
  3. writes the resulting joint torque to ``data.qfrc_applied`` on the arm DOFs
     (the arm's position servos are disabled in __init__ so torque control is
     exclusive), clipped to TAU_MAX,
  4. applies the grip command (grip_force).

Control rate = physics rate (dt=0.005, 200 Hz); one ``step()`` per ``mj_step``.
``set_ee_target`` is absolute world metres/quaternion (ZOH between updates);
``set_grip_force`` is Newtons.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SIM_COMMON_DIR = _REPO_ROOT / "experiments" / "ab_smooth_vs_bangbang"
if str(_SIM_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_COMMON_DIR))
import sim_common as sc  # noqa: E402

from .cartesian_impedance import CartesianImpedanceConfig, CartesianImpedanceController
from .grip_force import GripForceController
from .state_adapter import ArmIndex, build_state


@dataclass
class GraspResult:
    held: bool
    n_pad_contacts: int
    grip_force_measured: float
    ee_pos: np.ndarray


class ArmController:
    def __init__(
        self,
        model,
        data,
        side: str,
        *,
        cfg: CartesianImpedanceConfig | None = None,
        grip_mode: str = "servo_emulated",
        held_body_name: str | None = None,
        disable_servos: bool = True,
    ) -> None:
        self.model = model
        self.data = data
        self.side = side
        self.idx = ArmIndex(model, side)
        if cfg is None:
            cfg = CartesianImpedanceConfig(n_joints=self.idx.n, tau_max_nm=np.full(self.idx.n, sc.TAU_MAX))
        self.ctl = CartesianImpedanceController(cfg)
        self.grip = GripForceController(model, self.idx, mode=grip_mode)
        self.held_body_id = model.body(held_body_name).id if held_body_name else None

        if disable_servos:
            if side == "left":
                sc.disable_left_arm_position_servos(model)
            else:
                sc.disable_right_arm_position_servos(model)

        mujoco.mj_forward(model, data)
        self.ctl.reset_from_state(build_state(model, data, self.idx))
        self._last_out = None

    # ---- command API -----------------------------------------------------
    def set_ee_target(self, pos, quat=None, vel=None) -> None:
        self._target_pos = np.asarray(pos, dtype=np.float64).reshape(3).copy()
        self._target_quat = None if quat is None else np.asarray(quat, dtype=np.float64).reshape(4).copy()
        self._target_vel = None if vel is None else np.asarray(vel, dtype=np.float64).reshape(3).copy()
        self._hold = False

    def hold_pose(self) -> None:
        """Freeze at the current EE pose using the impedance hold reference."""
        self._hold = True

    def set_grip_force(self, newtons: float) -> None:
        self.grip.set_target_force(newtons)

    def set_grip_opening(self, opening: float) -> None:
        self.grip.set_target_opening(opening)

    def set_held_body(self, held_body_name: str | None) -> None:
        self.held_body_id = self.model.body(held_body_name).id if held_body_name else None

    # ---- state -----------------------------------------------------------
    def get_state(self) -> dict:
        return build_state(self.model, self.data, self.idx)

    def n_pad_contacts(self) -> int:
        pad_ids = set(self.idx.pad_geom_ids)
        return sum(
            1 for c in range(self.data.ncon)
            if self.data.contact[c].geom1 in pad_ids or self.data.contact[c].geom2 in pad_ids
        )

    # ---- control tick ----------------------------------------------------
    def step(self) -> None:
        model, data, idx = self.model, self.data, self.idx
        st = build_state(model, data, idx)
        if getattr(self, "_hold", False):
            st["hold_current_pose"] = True
        else:
            st["target_ee_pos"] = getattr(self, "_target_pos", st["ee_pos"])
            if getattr(self, "_target_quat", None) is not None:
                st["target_ee_quat"] = self._target_quat
            if getattr(self, "_target_vel", None) is not None:
                st["target_ee_vel"] = self._target_vel

        # Optional held-object gravity feedforward (support the payload weight).
        if self.held_body_id is not None:
            _, bias_box = sc.payload_reflected_terms(
                model, data, idx.ee_body_id, self.held_body_id, idx.dofadr
            )
            st["gravity_torque"] = st["gravity_torque"] + np.asarray(bias_box, dtype=np.float64)

        out = self.ctl.compute(st)
        self._last_out = out
        data.qfrc_applied[idx.dofadr] = out.tau
        self.grip.apply(data)

    @property
    def last_output(self):
        return self._last_out
