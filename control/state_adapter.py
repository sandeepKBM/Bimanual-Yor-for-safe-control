"""Adapter: MuJoCo ``MjModel``/``MjData`` -> per-arm controller state dict.

Produces the dict consumed by ``CartesianImpedanceController.compute`` for one
arm (``left`` or ``right``) of the YOR bimanual robot. All quantities are
world-frame; the Jacobian is restricted to the arm's own 6 DOF columns (exact,
since the arm's EE is moved only by that arm's joints).

Feedforward convention: ``gravity_torque = qfrc_bias - qfrc_passive`` restricted
to the arm DOFs. Writing this as ``qfrc_applied`` cancels gravity + Coriolis +
joint damping/springs so the impedance law sees an (approximate) unit-mass
double integrator — matching the FL pattern in mpc_fl.py:525-537.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mujoco

# sim_common lives under experiments/ and inserts real_Cartpole on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SIM_COMMON_DIR = _REPO_ROOT / "experiments" / "ab_smooth_vs_bangbang"
if str(_SIM_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_COMMON_DIR))

import sim_common as sc  # noqa: E402  (read-only import; do not edit)

ARM_JOINTS = {
    "left": [f"left_arm_joint{i}" for i in range(1, 7)],
    "right": [f"right_arm_joint{i}" for i in range(1, 7)],
}
EE_BODY = {"left": "left_arm_gripper_base", "right": "right_arm_gripper_base"}
FINGER_ACTUATORS = {
    "left": ("left_arm_joint7_pos", "left_arm_joint8_pos"),
    "right": ("right_arm_joint7_pos", "right_arm_joint8_pos"),
}
FINGER_JOINTS = {
    "left": ("left_arm_joint7", "left_arm_joint8"),
    "right": ("right_arm_joint7", "right_arm_joint8"),
}
FINGER_PAD_GEOMS = {
    "left": ("left_arm_link7_pad", "left_arm_link8_pad"),
    "right": ("right_arm_link7_pad", "right_arm_link8_pad"),
}


class ArmIndex:
    """Cached MuJoCo index bundle for one arm (joints, dofs, EE body, fingers)."""

    def __init__(self, model, side: str) -> None:
        if side not in ARM_JOINTS:
            raise ValueError(f"side must be 'left' or 'right'; got {side!r}")
        self.side = side
        _, self.qposadr, self.dofadr = sc.joint_addrs(model, ARM_JOINTS[side])
        self.ee_body_id = model.body(EE_BODY[side]).id
        self.finger_act_ids = tuple(model.actuator(a).id for a in FINGER_ACTUATORS[side])
        fj = FINGER_JOINTS[side]
        _, self.finger_qposadr, _ = sc.joint_addrs(model, list(fj))
        self.pad_geom_ids = tuple(model.geom(g).id for g in FINGER_PAD_GEOMS[side])
        self.n = len(self.dofadr)


def build_state(model, data, idx: ArmIndex, *, sim_time: float | None = None) -> dict:
    """Build the per-arm controller state dict from current MjData."""
    dofadr = idx.dofadr
    qposadr = idx.qposadr
    q = np.asarray(data.qpos[qposadr], dtype=np.float64)
    qd = np.asarray(data.qvel[dofadr], dtype=np.float64)

    ee = idx.ee_body_id
    ee_pos = np.asarray(data.xpos[ee], dtype=np.float64).copy()
    ee_quat = np.asarray(data.xquat[ee], dtype=np.float64).copy()  # MuJoCo is [w,x,y,z]

    jacp, jacr = sc.body_jac_full(model, data, ee)   # each 3 x nv (world frame)
    jp = jacp[:, dofadr]
    jr = jacr[:, dofadr]
    jacobian = np.vstack([jp, jr])                   # 6 x n

    qd_full = np.asarray(data.qvel, dtype=np.float64)
    ee_lin_vel = jacp @ qd_full                       # exact world-frame EE velocity
    ee_ang_vel = jacr @ qd_full

    # FL feedforward: cancel gravity + Coriolis + passive (damping/springs).
    # NOTE the YOR model has built-in gravcomp on the arm bodies, so this term
    # nets ~0 for the arm's own weight; it still carries Coriolis and any
    # residual, and payload gravity is added separately by ArmController.
    grav = np.asarray(data.qfrc_bias[dofadr] - data.qfrc_passive[dofadr], dtype=np.float64)

    # Arm-restricted mass matrix (for OSC operational-space inertia shaping).
    M_full = sc.full_mass_matrix(model, data)
    M_arm = M_full[np.ix_(dofadr, dofadr)]

    return {
        "time": float(data.time if sim_time is None else sim_time),
        "q": q,
        "qd": qd,
        "ee_pos": ee_pos,
        "ee_quat": ee_quat,
        "ee_lin_vel": ee_lin_vel,
        "ee_ang_vel": ee_ang_vel,
        "jacobian": jacobian,
        "gravity_torque": grav,
        "mass_matrix": M_arm,
    }
