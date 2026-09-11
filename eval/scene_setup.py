"""Scene setup for the cup-task eval harness (scenes/scene_yor_cup.xml).

Physically grasps the cup (+ its 8 balls) with the LEFT arm, using the exact
teleport-to-fingers technique already validated for the box task in
sim_common.init_grasped_state, generalized to a multi-body payload (the cup
body + 8 independent free-body balls -- none of this repo's existing cup-task
code grasps the cup: collection/collect_cup.py deliberately scripts the cup's
pose directly, "the cup is never grasped -- no bimanual grasp/IK for the cup
exists in this repo"). This lets the geometric safety module's evasive ego
motion actually couple into the payload (jerk/drift -> spill), which is the
whole point of this eval (mirrors experiments/ab_smooth_vs_bangbang/
RESULTS_v4.md's collision-vs-grasp coupling, on the cup task).

sim_common.LEFT_GRASP_Q (the box task's grasp pose) is reused as the arm
target: collection/collect_cup.py's own docstring notes this pose "sits
inside/against this scene's cup rim" in scene_yor_cup.xml -- i.e. it is
already a plausible grasp approach for the cup in this scene, and
sim_common.RIGHT_INTRUDE_CTRL was IK-solved specifically to intrude on a left
arm sitting at LEFT_GRASP_Q, so reusing it here keeps the intruder's severity
directly comparable to the (well-tested) box-task numbers.

Only imports (never edits) sim_common.py / scripts/cup_task_utils.py /
scenes/scene_yor_cup.xml.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mujoco

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SIM_COMMON_DIR = _REPO_ROOT / "experiments" / "ab_smooth_vs_bangbang"
for _p in (str(_REPO_ROOT), str(_SIM_COMMON_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sim_common as sc  # noqa: E402  (read-only import; do not edit)
from scripts import cup_task_utils as ct  # noqa: E402  (read-only import; do not edit)

EE_BODY_NAME = "left_arm_gripper_base"
CUP_GRASP_Z_FRAC = 0.55   # fraction of CUP_HEIGHT above the cup's own bottom
                          # origin that is placed at the fingers' pad_mid --
                          # i.e. the fingers pinch the cup wall a bit above
                          # its vertical center (biases the grasp point above
                          # the balls' resting layer, not down where they sit).


def load_model():
    """(model,) for scene_yor_cup.xml with the arm-arm contype/conaffinity
    fix + zed_cam applied, via cup_task_utils.load_yor_cup_model (read-only
    import; discards its throwaway MjData -- grasp init below builds its
    own)."""
    model, _unused_data = ct.load_yor_cup_model()
    return model


def init_cup_grasped_state(model: mujoco.MjModel, seed: int = 0, cup_jitter_m: float = 0.0):
    """Reset to 'home', teleport the LEFT arm to sc.LEFT_GRASP_Q, then
    teleport the cup body (+ all 8 balls, by the SAME rigid translation, so
    they stay in their original cup-relative layout) to the open left
    fingers' pad midpoint, close the fingers with gravity off (transient,
    same technique as sim_common.init_grasped_state), restore gravity, and
    settle. Returns the MjData with the grasp established.

    `cup_jitter_m`: optional small x/y offset applied to the PRE-grasp cup
    position (before computing the translation applied to cup+balls) purely
    to vary episodes across seeds -- mirrors collect_cup.py's own
    `cup_jitter_m` perturbation knob.
    """
    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    key = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)

    _, left_qposadr, _ = sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)
    data.qpos[left_qposadr] = sc.LEFT_GRASP_Q
    left_act_ids = [model.actuator(f"left_arm_joint{i}_pos").id for i in range(1, 7)]
    data.ctrl[left_act_ids] = sc.LEFT_GRASP_Q
    mujoco.mj_forward(model, data)

    pad7 = model.geom("left_arm_link7_pad").id
    pad8 = model.geom("left_arm_link8_pad").id
    pad_mid = 0.5 * (data.geom_xpos[pad7] + data.geom_xpos[pad8])

    cup_bid = model.body(ct.CUP_BODY_NAME).id
    cup_jid = model.joint("cup_free").id
    cup_qposadr = int(model.jnt_qposadr[cup_jid])
    cup_dofadr = int(model.jnt_dofadr[cup_jid])
    old_cup_pos = data.qpos[cup_qposadr:cup_qposadr + 3].copy()
    if cup_jitter_m > 0.0:
        old_cup_pos = old_cup_pos + np.array(
            [rng.uniform(-cup_jitter_m, cup_jitter_m), rng.uniform(-cup_jitter_m, cup_jitter_m), 0.0])

    new_cup_pos = pad_mid - np.array([0.0, 0.0, CUP_GRASP_Z_FRAC * ct.CUP_HEIGHT])
    delta = new_cup_pos - old_cup_pos

    data.qpos[cup_qposadr:cup_qposadr + 3] = new_cup_pos
    data.qpos[cup_qposadr + 3:cup_qposadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[cup_dofadr:cup_dofadr + 6] = 0.0

    for bid in ct.ball_body_ids(model):
        jadr = model.body_jntadr[bid]
        b_qposadr = int(model.jnt_qposadr[jadr])
        b_dofadr = int(model.jnt_dofadr[jadr])
        data.qpos[b_qposadr:b_qposadr + 3] += delta
        data.qvel[b_dofadr:b_dofadr + 6] = 0.0

    mujoco.mj_forward(model, data)

    original_gravity = model.opt.gravity.copy()
    model.opt.gravity[:] = 0.0

    j7_act = model.actuator("left_arm_joint7_pos").id
    j8_act = model.actuator("left_arm_joint8_pos").id
    data.ctrl[j7_act] = 0.0
    data.ctrl[j8_act] = 0.0
    for _ in range(500):
        mujoco.mj_step(model, data)

    model.opt.gravity[:] = original_gravity
    for _ in range(300):
        mujoco.mj_step(model, data)
    return data


def task_torque(model, data, q0_left, left_dofadr, left_qposadr, ee_body_id, cup_body_id,
                 couple_payload: bool = True, freeze: bool = False):
    """Computed-torque joint-space PD toward q0 (hold the grasp pose), with
    the cup's mass/inertia/gravity reflected into M_ll/bias_l via
    sim_common.payload_reflected_terms (reused verbatim -- only the payload
    BODY differs from the box task, the method does not). `freeze=True`
    (used by the semantic-veto intervention) drops the position-attraction
    term and keeps pure velocity damping -- "hold where you are" rather than
    "return to q0"."""
    M = sc.full_mass_matrix(model, data)
    M_ll = M[np.ix_(left_dofadr, left_dofadr)]
    bias_l = data.qfrc_bias[left_dofadr]
    if couple_payload:
        M_box, bias_box = sc.payload_reflected_terms(model, data, ee_body_id, cup_body_id, left_dofadr)
        M_ll = M_ll + M_box
        bias_l = bias_l + bias_box
    q = data.qpos[left_qposadr]
    qd = data.qvel[left_dofadr]
    if freeze:
        qacc_des = -sc.KD_TASK * qd
    else:
        qacc_des = sc.KP_TASK * (q0_left - q) - sc.KD_TASK * qd
    tau = M_ll @ qacc_des + bias_l
    return tau, M_ll, bias_l


TRANSFER_ROT_AXIS = np.array([1.0, 0.0, 0.0])  # world-X tilt axis, same axis
                                                # collect_cup.py's own
                                                # "spill_tilt" behavior level
                                                # tips the cup about.


def left_ee_jac_rot_pinv(model, data, ee_body_id, left_dofadr) -> np.ndarray:
    """Pseudoinverse (6x3) of the left EE's ROTATIONAL Jacobian (3x6,
    restricted to the left arm's 6 dofs), evaluated ONCE at the settled
    grasp pose. Converts a small Cartesian angular PD-target offset into a
    joint-space PD-target offset for `transfer_tilt_offset` below.

    Translational transfer was tried FIRST (a lateral EE Cartesian-offset
    PD target, converted the same way via the translational Jacobian) and
    calibrated empirically (see eval/_calib_probe.py, deleted after use) up
    to a 0.25 m / 0.15 s burst -- it never spilled a single ball. Instrumenting
    world-frame cup position during the burst showed why: `data.xpos[cup_body]`
    stayed bit-for-bit frozen while the EE moved ~15 cm and back -- this
    grasp's finger-pad contact provides enough NORMAL (radially-pinching)
    friction to hold the cup against gravity (validated by the ordinary
    settle in `init_cup_grasped_state`) but not enough TANGENTIAL friction to
    drag it sideways; a lateral PD-target burst just slides the gripper
    across the cup wall instead of carrying it. Tilting the wrist instead
    (this function) loads the grasp through the SAME friction that already
    holds the cup up against gravity, so it transmits -- and once tilted
    past the cup's own rim geometry, gravity (not friction) does the rest,
    same mechanism as the validated spill_tilt behavior in
    collection/perturbations.py / scripts/test_cup_scene.py's run_tilt_test
    (90-115 deg over 0.8-1.2 s -> all 8 balls spill)."""
    _jacp, jacr = sc.body_jac_full(model, data, ee_body_id)
    Jr = jacr[:, left_dofadr]  # 3x6
    return np.linalg.pinv(Jr)  # 6x3


def transfer_tilt_offset(t: float, start_s: float = 0.5, dur_s: float = 1.6,
                          amp_rad: float = np.deg2rad(70.0)) -> float:
    """Scalar PD-target angular offset [rad] (about `TRANSFER_ROT_AXIS`) for
    the LEFT EE's fast transfer/tilt motion -- the SEMANTIC failure of
    interest is the resulting SPILL, which the geometric (arm-arm distance)
    layer is structurally blind to (it never reacts to payload sloshing/
    tipping at all, only to arm-arm proximity).

    Quintic (minimum-jerk) 0->amp_rad ramp over [start_s, start_s+dur_s]
    (zero angular velocity/accel at both endpoints -- no PD-target
    discontinuity), then HOLDS at amp_rad (the "transfer" completes; the cup
    stays tipped for the rest of the episode so the spill finishes/balls
    land before the episode ends, mirroring run_tilt_test's tilt_dur/
    hold_dur). Defaults (70 deg over 1.6 s) are deliberately SLOWER than
    collect_cup.py's own validated spill_tilt range (90-115 deg over
    0.8-1.2s) -- calibrated down from an initial 100 deg/1.0s probe that
    spilled all 8 balls by ~30 deg (~0.15s after the tilt starts), too fast
    for a semantic monitor to see rising risk and veto before the spill is
    already done. The slower ramp still spills under `none`/`geometric_only`
    but gives the semantic role an actual window to intervene first. See
    `left_ee_jac_rot_pinv`'s docstring for why this eval tilts the EE at all
    rather than using a lateral accel-shake (which does not transmit through
    this grasp's contact-free finger placement, unlike the kinematic-
    teleport shake numbers in collection/perturbations.py's "spill_shake"
    level, which bypass arm dynamics entirely)."""
    if t <= start_s:
        return 0.0
    tau = min(t - start_s, dur_s)
    s = tau / dur_s
    return amp_rad * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)


def cup_rel_pose(model, data, ee_body_id, cup_body_id):
    """(rel_pos (3,), rel_quat (4,)) of the cup's CURRENT pose expressed in
    the EE's own frame -- captured once (right after grasp settle) to
    define the rigid-attachment offset used by `reattach_cup_rigidly`."""
    R_gb = data.xmat[ee_body_id].reshape(3, 3)
    rel_pos = R_gb.T @ (data.xpos[cup_body_id] - data.xpos[ee_body_id])
    rel_quat = quat_mul(quat_conj(data.xquat[ee_body_id]), data.xquat[cup_body_id])
    return rel_pos, rel_quat


def reattach_cup_rigidly(model, data, ee_body_id, cup_body_id, rel_pos, rel_quat) -> None:
    """Kinematically pin the cup body to the EE using a FIXED rigid offset
    (from `cup_rel_pose`, captured once at episode start), each step.

    WHY THIS EXISTS (discovered while calibrating this eval's transfer/tilt
    motion, via eval/_calib_probe.py and eval/_calib_probe2.py -- deleted
    after use, not part of the harness): `init_cup_grasped_state`'s
    teleport-to-fingers technique places the cup's central AXIS at the open
    fingers' pad-midpoint, not a cup WALL between the two pads -- when the
    fingers close they end up hovering ~2.8-3.3 cm from every one of the
    cup's 12 wall segments (i.e. near-equidistant, dead center in the empty
    cup), never actually touching the wall. Instrumenting `ncon`/contact
    pairs through the whole grasp-settle sequence confirmed ZERO pad-cup
    contacts at any point, before or after gravity is restored -- the cup
    just rests on the table, geometrically near the open gripper but
    mechanically unconnected to it. Any EE motion (translation OR the
    tilt this eval uses) previously left the cup completely undisturbed
    (`data.xpos[cup_body]` bit-for-bit frozen across an 8 cm/100 deg EE
    excursion) -- i.e. the pre-existing "LEFT arm grasps the cup" premise
    this eval (and the original collision-only smoke) was built on did not
    actually hold; nothing was ever spillable via arm motion.

    This function is the fix: it kinematically re-pins the cup to the EE
    every step -- the SAME technique collection/collect_cup.py's own
    "cup_behavior" motions already use and this project already validates
    for accel/tilt-induced spill (set cup qpos directly, zero its qvel, then
    `mj_step`) -- so the cup is now genuinely carried by (and only by) the
    EE's motion. Only the cup body is overridden; the 8 balls are NEVER
    touched by this function and remain ordinary free bodies simulated by
    MuJoCo's own contact dynamics for the whole episode -- they are what can
    actually lag/slosh/tip out of the (now rigidly carried) cup, which is
    the semantic spill failure this eval exists to exercise."""
    cup_jid = model.body_jntadr[cup_body_id]
    cup_qposadr = int(model.jnt_qposadr[cup_jid])
    cup_dofadr = int(model.jnt_dofadr[cup_jid])
    R_gb = data.xmat[ee_body_id].reshape(3, 3)
    cup_pos = data.xpos[ee_body_id] + R_gb @ rel_pos
    cup_quat = quat_mul(data.xquat[ee_body_id], rel_quat)
    cup_quat = cup_quat / np.linalg.norm(cup_quat)
    data.qpos[cup_qposadr:cup_qposadr + 3] = cup_pos
    data.qpos[cup_qposadr + 3:cup_qposadr + 7] = cup_quat
    data.qvel[cup_dofadr:cup_dofadr + 6] = 0.0


def capture_ball_rel(model, data, cup_body_id) -> dict:
    """Capture each ball's pose RELATIVE TO THE CUP (from free-joint qpos, which
    for a free body IS its world pose), for carry_balls_with_cup below. Returns
    {ball_body_id: (rel_pos(3,), rel_quat(4,))}. Call once when the balls are
    settled inside the cup (episode start), before any lift motion."""
    cup_jid = model.body_jntadr[cup_body_id]
    cq = int(model.jnt_qposadr[cup_jid])
    cup_pos = data.qpos[cq:cq + 3].copy()
    cup_quat = data.qpos[cq + 3:cq + 7].copy()
    conj = np.empty(4); mujoco.mju_negQuat(conj, cup_quat)
    rel = {}
    for bid in ct.ball_body_ids(model):
        jadr = model.body_jntadr[bid]
        bq = int(model.jnt_qposadr[jadr])
        d = data.qpos[bq:bq + 3] - cup_pos
        rp = np.empty(3); mujoco.mju_rotVecQuat(rp, d, conj)          # world->cup frame
        rq = np.empty(4); mujoco.mju_mulQuat(rq, conj, data.qpos[bq + 3:bq + 7])
        rel[int(bid)] = (rp, rq)
    return rel


def cup_velocity_from_motion(model, prev_qpos, cur_qpos, dt):
    """Finite-difference the cup's velocity from its pinned pose motion, in
    MuJoCo's EXACT free-joint qvel convention (via mj_differentiatePos, which
    handles the quaternion tangent correctly for every joint type). Returns the
    FULL (model.nv,) qvel of the prev->cur motion; the caller extracts the cup's
    6 dofs and writes ONLY those into data.qvel (leaving the arms' physics qvel
    untouched). Used for the INERTIAL CUP-CARRY fidelity fix: reattach_cup_rigidly
    pins the cup each step with ZEROED velocity, so the cup walls have no velocity
    in contact resolution and the loose balls feel only geometric wall-sweep, not
    the cup's true inertial forces -- an empirically-confirmed infidelity (the
    effective-gravity cone gave no spill benefit because spill here isn't
    inertial). Setting the cup qvel to this finite-difference BEFORE mj_step makes
    the moving walls impart proper momentum to the balls (real sloshing). The cup
    POSITION is still re-pinned to the EE next step (kinematic carry unchanged);
    only its velocity in this step's dynamics is corrected."""
    qvel = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, qvel, dt, prev_qpos, cur_qpos)
    return qvel


def canonical_ball_rel(model, cup_body_id) -> dict:
    """The CANONICAL ball-in-cup layout, captured from the scene's `home`
    keyframe (where the balls are, by scene design, correctly settled inside the
    cup -- count_spilled==0). Use this (NOT capture_ball_rel on a live episode)
    as the carry layout: init_cup_grasped_state's grasp does not actually hold
    the cup (see reattach_cup_rigidly's docstring -- the fingers never contact
    the walls), so by the time init returns the cup+balls have fallen/scattered
    and ~7/8 balls already read 'spilled'. Carrying with the canonical layout
    FORCES the balls back inside the (kinematically re-pinned) cup from step 0,
    so count_spilled starts at ~0 and the spill metric measures only
    dodge/tilt-induced spill during the episode."""
    scratch = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, scratch, model.key("home").id)
    mujoco.mj_forward(model, scratch)
    return capture_ball_rel(model, scratch, cup_body_id)


def carry_balls_with_cup(model, data, cup_body_id, ball_rel: dict) -> None:
    """Rigidly carry the balls WITH the cup: set each ball's free-joint qpos to
    cup_pose (x) its captured rel pose, and zero its qvel. Used ONLY during the
    initial lift phase (see make_cup_collect_safety_demo run_episode): the
    kinematic cup re-pin (reattach_cup_rigidly) teleports the cup with ZEROED
    velocity, so a moving cup would otherwise sweep its walls THROUGH the loose
    balls and scatter them out ("lift-loss") -- an artifact that corrupts the
    spill metric (count_spilled read 6-8/8 balls out from step 0). Freezing the
    balls in the cup through the lift, then RELEASING them once the cup is at its
    settled hold pose, makes count_spilled ~0 at the start of the measured
    (crossing/dodge) window so the spill signal reflects only DODGE-induced
    spill. Balls are ordinary free bodies again the instant this stops being
    called."""
    cup_jid = model.body_jntadr[cup_body_id]
    cq = int(model.jnt_qposadr[cup_jid])
    cup_pos = data.qpos[cq:cq + 3]
    cup_quat = data.qpos[cq + 3:cq + 7]
    for bid, (rp, rq) in ball_rel.items():
        jadr = model.body_jntadr[bid]
        bq = int(model.jnt_qposadr[jadr])
        bd = int(model.jnt_dofadr[jadr])
        world = np.empty(3); mujoco.mju_rotVecQuat(world, rp, cup_quat)   # cup->world frame
        data.qpos[bq:bq + 3] = cup_pos + world
        wq = np.empty(4); mujoco.mju_mulQuat(wq, cup_quat, rq)
        data.qpos[bq + 3:bq + 7] = wq
        data.qvel[bd:bd + 6] = 0.0


def left_right_contact(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """Body-name-prefix arm-arm contact check (identical logic to
    collection/collect_cup.py's `_left_right_contact` -- reimplemented here,
    not imported, since collect_cup.py is a standalone script, not a
    library)."""
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.body(model.geom_bodyid[c.geom1]).name
        b2 = model.body(model.geom_bodyid[c.geom2]).name
        s1 = "left" if b1.startswith("left_arm_") else ("right" if b1.startswith("right_arm_") else None)
        s2 = "left" if b2.startswith("left_arm_") else ("right" if b2.startswith("right_arm_") else None)
        if {s1, s2} == {"left", "right"}:
            return True
    return False


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_angle(q):
    w = np.clip(abs(q[0]), -1.0, 1.0)
    return 2.0 * np.arccos(w)
