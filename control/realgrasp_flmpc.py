"""Real friction grasp + FL-MPC glue (SIMULATION ONLY).

Helpers shared by ``scripts/smoke_realgrasp_flmpc_lift.py`` (Step-1 smoke) and
``scripts/make_cup_collect_safety_demo.py --grasp-mode real`` (flagship):

* ``restore_left_pad_object_collision`` -- post-compile, PAD-LOCAL version of
  ``cup_collect_task_utils._restore_finger_object_collision`` restricted to the
  LEFT fingers (OR bit0 onto the left link7/link8 group-3 collision geoms), so
  the left pads can physically pinch the cup while the RIGHT fingers keep the
  exact same (faked-mode) collision bits -- the right arm's cube weld path is
  untouched.
* true-cup-state measurements from the CUP BODY pose (not the EE): absolute cup
  tilt from world vertical, cup axis expressed in the EE frame (fed to
  FL-MPC's cup-upright rows so they bound the TRUE cup tilt), cup pose in the
  EE frame (grip slip), pad<->cup contact count, finger actuator force.
* ``DenseLiftTrack`` -- joint-space lift reference from a dense set of
  orientation-holding IK knots (straight vertical EE path, EE orientation held
  at the grasp orientation), smoothstep-timed; same ``sample``/``sample_dot``
  interface as the demo's ``JointWaypointTrack``.

No MuJoCo state is written by the measurement helpers. Nothing here talks to
hardware.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mujoco

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SIM_COMMON_DIR = _REPO_ROOT / "experiments" / "ab_smooth_vs_bangbang"
for _p in (str(_SIM_COMMON_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sim_common as sc  # noqa: E402  (read-only import)

LEFT_FINGER_BODIES = ("left_arm_link7", "left_arm_link8")
LEFT_PAD_GEOMS = ("left_arm_link7_pad", "left_arm_link8_pad")
LEFT_FINGER_ACTUATORS = ("left_arm_joint7_pos", "left_arm_joint8_pos")
FINGER_FORCE_LIMIT_N = 20.0   # class "finger" forcerange (unmodified YOR model)
_EZ = np.array([0.0, 0.0, 1.0])


def restore_left_pad_object_collision(model) -> list[int]:
    """OR contact bit0 onto the LEFT finger group-3 collision geoms only
    (post-compile, in-memory). Mirrors cup_collect_task_utils.
    _restore_finger_object_collision (which does all four fingers at MjSpec
    time) but leaves the right fingers bit-identical to the faked pipeline.
    Returns the patched geom ids."""
    patched = []
    body_ids = {model.body(b).id for b in LEFT_FINGER_BODIES}
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) in body_ids and int(model.geom_group[g]) == 3:
            model.geom_contype[g] |= 1
            model.geom_conaffinity[g] |= 1
            patched.append(g)
    # MuJoCo >= 3 also caches the per-BODY OR of its geoms' bits (body_contype/
    # body_conaffinity) for the broadphase body-pair filter at compile time --
    # without updating these, the geom-level patch above is silently ignored.
    if hasattr(model, "body_contype"):
        for b in body_ids:
            model.body_contype[b] |= 1
            model.body_conaffinity[b] |= 1
    return patched


def cup_true_tilt_rad(data, cup_body: int) -> float:
    """Angle between the CUP BODY's own +z axis and world +z [rad]."""
    z_cup = data.xmat[cup_body].reshape(3, 3)[:, 2]
    return float(np.arccos(np.clip(z_cup[2], -1.0, 1.0)))


def ee_tilt_rad(data, ee_body: int, axis_local: np.ndarray) -> float:
    """Angle of an EE-fixed axis (captured as world-up at some reference
    time) from world +z [rad] -- the EE-based 'cup tilt' proxy the faked
    pipeline logs (left_cup_tilt_rad)."""
    a = data.xmat[ee_body].reshape(3, 3) @ axis_local
    return float(np.arccos(np.clip(a[2], -1.0, 1.0)))


def cup_axis_in_ee(data, ee_body: int, cup_body: int) -> np.ndarray:
    """The cup body's +z axis expressed in the EE's local frame. Feeding this
    to FLMPC as ``_cup_axis_local`` makes its cup-upright rows' tilt anchor
    (tilt0 = R_ee @ axis -> small-angle xy) equal the TRUE cup tilt, with the
    rotation Jacobian taken at the EE (the cup rotates with the EE while it is
    held; it is not on the arm's kinematic chain, so its own Jacobian w.r.t.
    the arm dofs is structurally zero)."""
    R_ee = data.xmat[ee_body].reshape(3, 3)
    R_cup = data.xmat[cup_body].reshape(3, 3)
    return R_ee.T @ R_cup[:, 2]


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def cup_rel_pose_in_ee(data, ee_body: int, cup_body: int):
    """(rel_pos(3,), rel_quat(4,)) of the ACTUAL cup pose in the EE frame
    (same convention as eval/scene_setup.cup_rel_pose)."""
    R_ee = data.xmat[ee_body].reshape(3, 3)
    rel_pos = R_ee.T @ (data.xpos[cup_body] - data.xpos[ee_body])
    rel_quat = _quat_mul(_quat_conj(data.xquat[ee_body]), data.xquat[cup_body])
    return rel_pos, rel_quat


def pad_cup_contacts(model, data, pad_geom_ids, cup_body: int) -> int:
    """Number of active contacts between either LEFT pad geom and any cup geom."""
    pads = set(int(g) for g in pad_geom_ids)
    n = 0
    for c in range(data.ncon):
        con = data.contact[c]
        g1, g2 = int(con.geom1), int(con.geom2)
        if (g1 in pads and model.geom_bodyid[g2] == cup_body) or \
           (g2 in pads and model.geom_bodyid[g1] == cup_body):
            n += 1
    return n


def pads_touching_cup(model, data, pad_geom_ids, cup_body: int) -> int:
    """How many DISTINCT pads (0/1/2) currently contact the cup -- the
    'both-pad contact' signal (2 = genuine two-sided pinch)."""
    touching = set()
    pads = set(int(g) for g in pad_geom_ids)
    for c in range(data.ncon):
        con = data.contact[c]
        g1, g2 = int(con.geom1), int(con.geom2)
        if g1 in pads and model.geom_bodyid[g2] == cup_body:
            touching.add(g1)
        elif g2 in pads and model.geom_bodyid[g1] == cup_body:
            touching.add(g2)
    return len(touching)


def pad_normal_force_sum(model, data, pad_geom_ids, cup_body: int) -> float:
    """Sum of pad<->cup contact NORMAL forces [N] (both pads together)."""
    pads = set(int(g) for g in pad_geom_ids)
    f6 = np.zeros(6)
    tot = 0.0
    for c in range(data.ncon):
        con = data.contact[c]
        g1, g2 = int(con.geom1), int(con.geom2)
        if (g1 in pads and model.geom_bodyid[g2] == cup_body) or \
           (g2 in pads and model.geom_bodyid[g1] == cup_body):
            mujoco.mj_contactForce(model, data, c, f6)
            tot += abs(float(f6[0]))
    return tot


def finger_actuator_forces(model, data) -> np.ndarray:
    """Realized LEFT finger actuator forces [N] (joint7, joint8) -- bounded by
    the MJCF forcerange +-20 N by construction; logged to PROVE it."""
    ids = [model.actuator(a).id for a in LEFT_FINGER_ACTUATORS]
    return np.asarray(data.actuator_force[ids], dtype=float).copy()


class DenseLiftTrack:
    """Joint-space lift reference: smoothstep time-scaling s(t) over
    ``t_move``, then hold; the joint path is a piecewise-linear interpolation
    through ``n_knots`` IK solutions of a STRAIGHT vertical EE path with the
    EE orientation HELD at its start orientation (orientation-weighted DLS
    IK). Same sample()/sample_dot() interface as JointWaypointTrack."""

    def __init__(self, q_knots: np.ndarray, t_move: float, t_hold: float):
        self.q_knots = np.asarray(q_knots, dtype=float)
        self.s_knots = np.linspace(0.0, 1.0, self.q_knots.shape[0])
        self.t_move = float(t_move)
        self.t_hold = float(t_hold)
        self.start_q = self.q_knots[0].copy()
        self.end_q = self.q_knots[-1].copy()

    def _q_of_s(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0.0, 1.0))
        return np.array([np.interp(s, self.s_knots, self.q_knots[:, j])
                         for j in range(self.q_knots.shape[1])])

    def sample(self, t: float) -> np.ndarray:
        if t <= 0.0:
            return self.start_q.copy()
        u = float(np.clip(t / max(self.t_move, 1e-9), 0.0, 1.0))
        s = u * u * (3.0 - 2.0 * u)
        return self._q_of_s(s)

    def sample_dot(self, t: float, h: float = 1e-3) -> np.ndarray:
        return (self.sample(t + h) - self.sample(t - h)) / (2.0 * h)


def build_dense_lift_track(model, data, arm_ik, q_start: np.ndarray, lift_vec: np.ndarray,
                           t_move: float, t_hold: float, n_knots: int = 12):
    """IK the straight lift path (orientation held at the CURRENT grasp-site
    orientation) at ``n_knots`` evenly spaced points. ``arm_ik`` is a
    teleop.bimanual_teleop.ArmIK for the left arm (warm-started knot to knot,
    restarts disabled so the branch never jumps). Returns
    (DenseLiftTrack, q_end, max_pos_err_m, max_ori_err_rad)."""
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.qpos[arm_ik.qposadr] = q_start
    mujoco.mj_forward(model, scratch)
    site_pos0 = scratch.site_xpos[arm_ik.site_id].copy()
    site_quat0 = np.zeros(4)
    mujoco.mju_mat2Quat(site_quat0, scratch.site_xmat[arm_ik.site_id].copy())

    arm_ik.last_q = np.asarray(q_start, dtype=float).copy()
    knots = [np.asarray(q_start, dtype=float).copy()]
    max_err = 0.0
    for s in np.linspace(0.0, 1.0, n_knots)[1:]:
        q, err = arm_ik.solve(scratch, site_pos0 + s * np.asarray(lift_vec, dtype=float),
                              target_quat=site_quat0, iters=200, allow_coarse_restart=False)
        knots.append(q.copy())
        max_err = max(max_err, float(err))
    # orientation drift along the knots (FK check, not trusted from IK)
    max_ori = 0.0
    for q in knots:
        scratch.qpos[arm_ik.qposadr] = q
        mujoco.mj_forward(model, scratch)
        qq = np.zeros(4)
        mujoco.mju_mat2Quat(qq, scratch.site_xmat[arm_ik.site_id].copy())
        d = abs(float(np.dot(qq, site_quat0)))
        max_ori = max(max_ori, 2.0 * float(np.arccos(min(1.0, d))))
    track = DenseLiftTrack(np.stack(knots), t_move, t_hold)
    return track, knots[-1].copy(), max_err, max_ori


# ---------------------------------------------------------------------------
# Joint dry-friction feedforward for the FL-MPC lift.
#
# WHY: the YOR arm joints carry frictionloss=0.3 Nm each (unmodified model).
# MuJoCo resolves frictionloss as a CONSTRAINT force (qfrc_constraint), so it
# is NOT in qfrc_bias/qfrc_passive and FL-MPC's tau = M a + c does not cancel
# it. With the arm's built-in gravcomp the lift needs only ~0.1-0.7 Nm, so on
# the light wrist joints (M_jj ~ 0.008 kg m^2) the planned M a is BELOW the
# 0.3 Nm friction: the wrist stalls while joints 1-3 move, the EE rolls ~12 deg
# and the cup tilts with it (measured, Step-1 smoke without this term).
# Fix (model-based, same spirit as FL's own c term): add the known Coulomb
# friction in the direction FL-MPC is COMMANDING the joint to move,
#     tau_f = f_loss * tanh((qd + a0*DT) / V_EPS),
# where a0 is FL-MPC's applied first-block acceleration. ~0 when the plan is
# to hold still (friction then helps the hold). Added AFTER the QP, so the
# final torque is re-clipped to TAU_MAX by the caller (torque box stays hard).
# ---------------------------------------------------------------------------
FRICTION_FF_V_EPS = 0.02   # [rad/s] tanh smoothing of the sign()


def friction_feedforward(model, dofadr, qd, a0, dt, v_eps: float = FRICTION_FF_V_EPS) -> np.ndarray:
    v_plan = np.asarray(qd, dtype=float) + np.asarray(a0, dtype=float) * float(dt)
    return np.asarray(model.dof_frictionloss[dofadr], dtype=float) * np.tanh(v_plan / v_eps)


# ---------------------------------------------------------------------------
# Leveled real grasp (optional variant of control.grasp_routines.grasp_cup_real)
#
# WHY: sim_common.LEFT_GRASP_Q puts the gripper's jaw (pad-to-pad) axis ~6 deg
# off horizontal (pads at different heights, 8 mm apart in z). Closing two
# flat pads forces the cup axis PERPENDICULAR to the jaw axis, so the Stage-1
# grasp leaves the cup tilted ~5.4 deg before any motion -- 94% of the
# 5.73 deg cup cone. This variant rotates the open gripper (orientation IK,
# grasp-site position held) so the jaw axis is horizontal, re-centres the cup
# (+balls, rigid translation) on the new pad midpoint, then runs the SAME
# close/settle loop as grasp_cup_real (same ArmController/transport_cfg, same
# grip ramp, gravity ON, no pin).
# ---------------------------------------------------------------------------
def _leveled_grasp_q(model, data, arm_ik, pad7: int, pad8: int):
    pad_ax = data.geom_xpos[pad8] - data.geom_xpos[pad7]
    pad_ax = pad_ax / np.linalg.norm(pad_ax)
    pad_h = np.array([pad_ax[0], pad_ax[1], 0.0])
    pad_h = pad_h / np.linalg.norm(pad_h)
    axis = np.cross(pad_ax, pad_h)
    s = np.linalg.norm(axis)
    ang = float(np.arctan2(s, float(np.dot(pad_ax, pad_h))))
    if s < 1e-9:
        return data.qpos[arm_ik.qposadr].copy(), 0.0, 0.0
    axis = axis / s
    q_corr = np.array([np.cos(ang / 2), *(np.sin(ang / 2) * axis)])
    site_q = np.zeros(4)
    mujoco.mju_mat2Quat(site_q, data.site_xmat[arm_ik.site_id].copy())
    tgt_q = np.zeros(4)
    mujoco.mju_mulQuat(tgt_q, q_corr, site_q)
    arm_ik.last_q = data.qpos[arm_ik.qposadr].copy()
    q, err = arm_ik.solve(data, data.site_xpos[arm_ik.site_id].copy(), target_quat=tgt_q,
                          iters=300, allow_coarse_restart=False)
    return q, float(err), ang


def grasp_cup_real_leveled(model, data, *, grip_force: float = 8.0, z_frac: float = 0.5,
                           close_steps: int = 200, settle_steps: int = 200,
                           noslip_iterations: int = 2, level: bool = True,
                           ball_jitter_m: float = 0.0, rng=None, verbose: bool = False):
    """Like control.grasp_routines.grasp_cup_real but with the jaw axis
    leveled first (see block comment above). level=False skips the leveling
    (then the placement + close/settle loop is identical to grasp_cup_real).
    ball_jitter_m>0 (+ rng): per-ball xy offset U(-j, j) added to the canonical
    in-cup ball layout BEFORE the close (episode-to-episode variation; the
    balls then settle under real gravity/contact). Returns (arm, GraspResult,
    info_dict)."""
    from control import grasp_routines as gr
    from control.arm_controller import ArmController
    from scripts import cup_task_utils as ct
    from teleop import bimanual_teleop as bt

    if noslip_iterations is not None:
        model.opt.noslip_iterations = int(noslip_iterations)
    gr.place_cup_in_open_gripper(model, data, side="left", z_frac=z_frac, carry_balls=True)

    pad7 = model.geom(LEFT_PAD_GEOMS[0]).id
    pad8 = model.geom(LEFT_PAD_GEOMS[1]).id
    ik = bt.ArmIK(model, "left")
    if level:
        q_lvl, ik_err, ang = _leveled_grasp_q(model, data, ik, pad7, pad8)
    else:
        q_lvl, ik_err, ang = sc.LEFT_GRASP_Q.copy(), 0.0, 0.0
    _, lq, _ = sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)
    data.qpos[lq] = q_lvl
    left_act_ids = [model.actuator(f"left_arm_joint{i}_pos").id for i in range(1, 7)]
    data.ctrl[left_act_ids] = q_lvl
    mujoco.mj_forward(model, data)

    # re-centre cup (+ balls, same rigid translation) on the new pad midpoint
    pad_mid = 0.5 * (data.geom_xpos[pad7] + data.geom_xpos[pad8])
    cup_jid = model.joint("cup_free").id
    cqa = int(model.jnt_qposadr[cup_jid])
    cda = int(model.jnt_dofadr[cup_jid])
    new_cup = pad_mid - np.array([0.0, 0.0, z_frac * ct.CUP_HEIGHT])
    delta = new_cup - data.qpos[cqa:cqa + 3]
    data.qpos[cqa:cqa + 3] = new_cup
    data.qpos[cqa + 3:cqa + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[cda:cda + 6] = 0.0
    for bid in ct.ball_body_ids(model):
        jadr = model.body_jntadr[bid]
        bqa = int(model.jnt_qposadr[jadr])
        bda = int(model.jnt_dofadr[jadr])
        data.qpos[bqa:bqa + 3] += delta
        if ball_jitter_m > 0.0 and rng is not None:
            data.qpos[bqa:bqa + 2] += rng.uniform(-ball_jitter_m, ball_jitter_m, size=2)
        data.qvel[bda:bda + 6] = 0.0
    mujoco.mj_forward(model, data)

    arm = ArmController(model, data, "left", cfg=gr.transport_cfg())
    arm.hold_pose()
    cup_bid = model.body(ct.CUP_BODY_NAME).id
    for k in range(close_steps):
        frac = min(1.0, (k + 1) / max(1, close_steps // 2))
        arm.set_grip_force(frac * grip_force)
        arm.step()
        mujoco.mj_step(model, data)
    arm.set_grip_force(grip_force)
    for _ in range(settle_steps):
        arm.step()
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    n_contacts = pad_cup_contacts(model, data, arm.idx.pad_geom_ids, cup_bid)
    gforce = arm.grip.measured_grip_force(model, data)
    held = n_contacts >= 2 and gforce > 0.5
    res = gr.GraspResult(held=held, n_pad_contacts=n_contacts,
                         grip_force_measured=float(gforce), cup_pos=data.xpos[cup_bid].copy())
    info = dict(level_rot_deg=float(np.degrees(ang)), level_ik_err_m=ik_err,
                cup_delta_m=delta.copy().tolist(), leveled=bool(level),
                ball_jitter_m=float(ball_jitter_m))
    if verbose:
        print(f"[grasp_cup_real_leveled] jaw leveled by {np.degrees(ang):.2f} deg (IK err "
              f"{ik_err*1000:.2f} mm, cup shift {np.round(delta*1000, 1)} mm); pad-cup contacts="
              f"{n_contacts} grip={gforce:.2f} N held={held}")
    return arm, res, info
