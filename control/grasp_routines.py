"""Real friction grasp routines — replace the faked cup teleport / cube weld.

``grasp_cup_real`` establishes a genuine two-finger friction pinch on the cup:
the left arm is posed so its open finger pads straddle the cup diameter (the
cup's axis passes through the pad midpoint, pads at ~±0.035 m, cup wall at
r=0.03 m), then the fingers close under grip-FORCE control (gravity on, no
gravity-off cheat) until BOTH pads register contact with the cup. From then on
the cup is held only by friction — no ``reattach_cup_rigidly`` pin.

This is the "core physics win": the cup follows the end-effector through a lift
because the fingers actually grip it, and it can be dropped/slip like a real
grasp. Requires the model loaded with ``restore_finger_object_collision=True``
so the pad geoms can contact the cup.
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
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import sim_common as sc  # noqa: E402
from scripts import cup_task_utils as ct  # noqa: E402

from .arm_controller import ArmController  # noqa: E402
from .cartesian_impedance import CartesianImpedanceConfig  # noqa: E402


@dataclass
class GraspResult:
    held: bool
    n_pad_contacts: int
    grip_force_measured: float
    cup_pos: np.ndarray


def _cup_pad_contacts(model, data, pad_geom_ids, cup_body_id) -> int:
    """Count contacts between either finger pad geom and any geom of the cup body."""
    pad_ids = set(pad_geom_ids)
    n = 0
    for c in range(data.ncon):
        con = data.contact[c]
        b1 = model.geom_bodyid[con.geom1]
        b2 = model.geom_bodyid[con.geom2]
        g1_pad = con.geom1 in pad_ids
        g2_pad = con.geom2 in pad_ids
        if (g1_pad and b2 == cup_body_id) or (g2_pad and b1 == cup_body_id):
            n += 1
    return n


def place_cup_in_open_gripper(model, data, *, side: str = "left",
                              z_frac: float = 0.55, carry_balls: bool = True):
    """Reset to home, pose the arm at the grasp pose, and place the cup so its
    axis passes through the open pad midpoint (a wall on each side of the jaw).
    Leaves fingers OPEN. Does NOT close or pin — that's the caller's grip loop.
    """
    key = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)

    _, left_qposadr, _ = sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)
    data.qpos[left_qposadr] = sc.LEFT_GRASP_Q
    left_act_ids = [model.actuator(f"left_arm_joint{i}_pos").id for i in range(1, 7)]
    data.ctrl[left_act_ids] = sc.LEFT_GRASP_Q
    # open fingers
    data.ctrl[model.actuator("left_arm_joint7_pos").id] = 0.035
    data.ctrl[model.actuator("left_arm_joint8_pos").id] = -0.035
    data.qpos[model.joint("left_arm_joint7").qposadr[0]] = 0.035
    data.qpos[model.joint("left_arm_joint8").qposadr[0]] = -0.035
    mujoco.mj_forward(model, data)

    pad7 = model.geom("left_arm_link7_pad").id
    pad8 = model.geom("left_arm_link8_pad").id
    pad_mid = 0.5 * (data.geom_xpos[pad7] + data.geom_xpos[pad8])

    cup_jid = model.joint("cup_free").id
    cup_qposadr = int(model.jnt_qposadr[cup_jid])
    cup_dofadr = int(model.jnt_dofadr[cup_jid])
    old_cup_pos = data.qpos[cup_qposadr:cup_qposadr + 3].copy()
    new_cup_pos = pad_mid - np.array([0.0, 0.0, z_frac * ct.CUP_HEIGHT])
    delta = new_cup_pos - old_cup_pos
    data.qpos[cup_qposadr:cup_qposadr + 3] = new_cup_pos
    data.qpos[cup_qposadr + 3:cup_qposadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[cup_dofadr:cup_dofadr + 6] = 0.0

    if carry_balls:
        for bid in ct.ball_body_ids(model):
            jadr = model.body_jntadr[bid]
            b_qposadr = int(model.jnt_qposadr[jadr])
            b_dofadr = int(model.jnt_dofadr[jadr])
            data.qpos[b_qposadr:b_qposadr + 3] += delta
            data.qvel[b_dofadr:b_dofadr + 6] = 0.0

    mujoco.mj_forward(model, data)
    return new_cup_pos


def transport_cfg() -> CartesianImpedanceConfig:
    """Cartesian-impedance gains tuned for stable cup TRANSPORT on the YOR arm.

    kp_rot is kept at the stability sweet spot (80 -- higher destabilises the
    orientation loop at 200 Hz), and orientation DAMPING is raised (kd_rot=120)
    to suppress the wrist swing during the lift reconfiguration. Result: ~15 deg
    EE-tilt lift, 1.3 cm grip slip, cup rigid in the grip.
    """
    return CartesianImpedanceConfig(
        n_joints=6, tau_max_nm=np.full(6, sc.TAU_MAX),
        kp_pos=np.full(3, 225.0), kd_pos=np.full(3, 30.0),
        kp_rot=80.0, kd_rot=120.0,
        kp_posture=1.0, kd_posture=2.0, kd_joint=0.5, use_osc_inertia=True,
    )


def grasp_cup_real(model, data, *, grip_force: float = 8.0, z_frac: float = 0.5,
                   close_steps: int = 200, settle_steps: int = 200,
                   carry_balls: bool = True, cfg: CartesianImpedanceConfig | None = None,
                   noslip_iterations: int = 2, verbose: bool = False):
    """Establish a real friction grasp of the cup with the LEFT arm.

    Returns (arm, GraspResult). The arm holds its pose (impedance) while the
    grip force ramps up; success = both pads contacting the cup after settle.
    ``noslip_iterations`` is applied to the model (reduces residual tangential
    creep, MuJoCo #3328); ``cfg`` defaults to the transport-tuned gains.
    """
    if noslip_iterations is not None:
        model.opt.noslip_iterations = int(noslip_iterations)
    if cfg is None:
        cfg = transport_cfg()
    place_cup_in_open_gripper(model, data, side="left", z_frac=z_frac, carry_balls=carry_balls)

    arm = ArmController(model, data, "left", cfg=cfg)
    arm.hold_pose()
    cup_bid = model.body(ct.CUP_BODY_NAME).id

    # Ramp grip force 0 -> grip_force while holding the arm pose (gravity ON).
    for k in range(close_steps):
        frac = min(1.0, (k + 1) / max(1, close_steps // 2))
        arm.set_grip_force(frac * grip_force)
        arm.step()
        mujoco.mj_step(model, data)

    # Settle at full grip.
    arm.set_grip_force(grip_force)
    for _ in range(settle_steps):
        arm.step()
        mujoco.mj_step(model, data)

    n_contacts = _cup_pad_contacts(model, data, arm.idx.pad_geom_ids, cup_bid)
    gforce = arm.grip.measured_grip_force(model, data)
    cup_pos = data.xpos[cup_bid].copy()
    held = n_contacts >= 2 and gforce > 0.5
    if verbose:
        print(f"[grasp_cup_real] pad-cup contacts={n_contacts} grip={gforce:.2f} N held={held}")
    return arm, GraspResult(held=held, n_pad_contacts=n_contacts,
                            grip_force_measured=float(gforce), cup_pos=cup_pos)
