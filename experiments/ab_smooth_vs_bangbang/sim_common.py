"""Shared setup/utilities for the A/B/C smooth-HOCBF-QP vs bang-bang HJ-style
fail-safe experiment (E1: single-arm pick + intrusion), on
scenes/scene_yor_canonical.xml (v3 -- re-pointed from the old simplified rig,
scenes/scene_bimanual_pickup.xml, to the proper YOR robot; see
scenes/YOR_CANONICAL.md, the locked interface, and RESULTS_v3.md for the full
re-point changelog/diff vs v2).

Ego arm = LEFT (grasps test_box, torque-controlled on its 6 arm joints).
Intruder arm = RIGHT (scripted, open-loop position-servo sweep, same for all
conditions).

This module is read-only with respect to the scene files (yor_robot_derived
.mjcf / scene_yor_canonical.xml are the locked canonical scene) and does not
modify them on disk; all per-run setup (grasp init, the cross-arm-collision
contype/conaffinity patch below) happens on the in-memory MjModel/MjData.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import mujoco

SCENE = "/common/users/ss5772/bimanual/scenes/scene_yor_canonical.xml"

REAL_CARTPOLE_ROOT = "/common/users/ss5772/real_Cartpole"
if REAL_CARTPOLE_ROOT not in sys.path:
    sys.path.insert(0, REAL_CARTPOLE_ROOT)

from controller_core.box_qp import build_weighted_least_squares_qp  # noqa: E402
from controller_core.constrained_box_qp import solve_constrained_box_qp  # noqa: E402
from controller_core.filters import TorqueCommandFilter  # noqa: E402

LEFT_ARM_JOINTS = [f"left_arm_joint{i}" for i in range(1, 7)]
RIGHT_ARM_JOINTS = [f"right_arm_joint{i}" for i in range(1, 7)]

# left_arm_joint1..6 ranges (verified via mj_name2id/jnt_range in
# scenes/YOR_CANONICAL.md -- IDENTICAL to the old rig's left/joint1..6 ranges,
# same underlying Piper arm; copied here as plain data for the fabric's
# joint-limit barrier, NOT re-parsed from the XML at runtime, to keep this
# module's only scene dependency the model load itself).
LEFT_JOINT_RANGES = np.array([
    [-2.61, 2.61],
    [0.0, 3.13],
    [-2.965, 0.0],
    [-1.74, 1.74],
    [-1.2, 1.2],
    [-3.0, 3.0],
])

BOX_BODY_NAME = "test_box"
EE_BODY_NAME = "left_arm_gripper_base"

# ---------------------------------------------------------------------------
# COLLISION-FIDELITY FIX (see experiments/ab_smooth_vs_bangbang/
# COLLISION_FIDELITY.md for the full investigation): the original v3 proxy
# was 3 spheres per arm (link3/link5/gripper_base), radius/origin only --
# measured false-safe rate (proxy says h>=0 but the REAL per-link fitted-
# capsule/box geoms already overlap) = 19.8% of true collisions (3.55% of
# all sampled configs, worst case 5.5cm undetected penetration), missing
# link0/1/2/4/6/7/8 entirely and the wrist iphone_body mount (itself a real,
# frequently-colliding protrusion not covered by ANY of the 3 old bodies).
#
# Fix: cover ALL major links per arm, each represented by 2 endpoint spheres
# fit to that link's REAL fitted-capsule geom (+1 extra midpoint sphere for
# long capsules, half-length > 0.05m, to avoid a shaft-center gap) or 1
# bounding sphere per box geom (finger pads). Verified (collision_fidelity_
# assess.py / collision_fidelity_fix_test.py, 2x independent 2000-config
# samples, uniform + task-region near the shared workspace) to bring
# false-safe to 0.00% (0/699 true collisions across both seeds).
# ---------------------------------------------------------------------------
LINK_NAMES = [
    "link0", "link1", "link2", "link3", "link4", "link5", "link6",
    "gripper_base", "link7", "link8", "iphone_body",
]
EGO_PREFIX = "left_arm_"
INT_PREFIX = "right_arm_"
# Kept for any external code that still names the (now-superseded) 3-body
# proxy set -- NOT used by min_pair_distance/all_pair_barriers/barrier below.
EGO_BARRIER_BODIES = ["left_arm_link3", "left_arm_link5", "left_arm_gripper_base"]
INT_BARRIER_BODIES = ["right_arm_link3", "right_arm_link5", "right_arm_gripper_base"]
_LONG_CAPSULE_HALFLEN_THRESH = 0.05  # [m] add a 3rd midpoint sphere above this

# ---------------------------------------------------------------------------
# Tunable experiment parameters (SAME across C0/C1/C2/C3 -- only the ego
# control law changes). RETUNED for v3's much tighter geometry (arms 0.185m
# apart, mounted at z~0.67, vs. the old rig's 0.5m floor-mounted spacing):
# a probe (probe_v3.py) found the ego (left) and intruder (right) grippers'
# collision-proxy bodies are ~0.23m apart at the intruder's home/ready pose
# and ~0.04m apart (gripper_base-to-gripper_base) at full IK intrusion --
# roughly half the old rig's corresponding separations -- so R_EGO/R_INT/
# D_MIN and H_TRIGGER are all scaled down by ~2x from v1/v2 to keep the same
# qualitative margin structure (safe positive h at rest, hard negative h at
# full intrusion) on the new scale.
# ---------------------------------------------------------------------------
DT = 0.005              # = model timestep (scenes/scene_yor_canonical.xml uses 0.005, NOT the old
                        # rig's 0.002 -- verified via model.opt.timestep); control runs at the physics rate
DURATION_S = 6.0
N_STEPS = int(round(DURATION_S / DT))

# R_EGO/R_INT (fixed 3.5cm "stand-in" radii for every sphere) are SUPERSEDED
# by the collision-fidelity fix: each of the ~25 keypoint spheres per arm now
# carries its OWN radius, fit to the real per-link geometry (see LINK_NAMES /
# _body_keypoints below), not a single guessed constant. Kept only as
# documentation of the old scale.
R_EGO = 0.035            # [m] (legacy; NOT used by the corrected proxy below)
R_INT = 0.035            # [m] (legacy; NOT used by the corrected proxy below)
D_MIN = 0.03             # [m] extra required clearance margin, STILL used: h = dist - (r_e + r_i + D_MIN)
                         # per keypoint pair (v2: 0.08)
# Legacy/representative scalar, kept ONLY for make_plots*.py's reference
# horizontal line -- the real per-pair combined radius now varies (r_e+r_i
# ranges ~0.017-0.07m across the 25x25 keypoint pairs); this is just
# R_EGO+R_INT+D_MIN from the old fixed-radius scheme, not used in any h
# computation below.
COMBINED_RADIUS = R_EGO + R_INT + D_MIN   # = 0.10, v2 was 0.18

H_TRIGGER = 0.04          # [m] C1 (bang-bang) switches when h < H_TRIGGER (v2: 0.05)
# HOCBF class-K gains, DOUBLED from v2 (15->30): a smaller COMBINED_RADIUS buffer
# (0.10 vs v2's 0.18) gives the class-K term less distance-budget to act within,
# so it must react proportionally faster to hold the same qualitative margin --
# an alpha sweep {15,25,30,40,50,70,90} at fixed R_EGO/R_INT/D_MIN/H_TRIGGER
# confirmed this is a real, monotonic-ish effect (higher alpha -> smaller min|h|
# violation), not noise; see RESULTS_v3.md "C2 alpha retune" for the full sweep
# table and the honest finding that NO alpha in this sweep achieves both
# collision=False AND box_dropped=False simultaneously on the tighter rig.
ALPHA1 = 30.0             # HOCBF class-K gain (psi1 = hdot + alpha1*h)
ALPHA2 = 30.0             # HOCBF class-K gain (psi2 = psi1dot + alpha2*psi1)

TAU_MAX = 100.0           # [Nm] per-arm-joint torque limit -- from the UNMODIFIED YOR GitHub model
                          # (piper URDF effort=100 == MJCF piper-class forcerange +-100). Was 40 (conservative).
                          # NOTE: raising 40->100 did NOT change the collision/infeasibility cliff (analysis
                          # 2026-09-08); the cliff is speed/horizon-governed, not torque-limited.
                          # own qfrc_applied torque budget, independent of the model's built-in
                          # position-actuator forcerange; unchanged from v1/v2.

# computed-torque task PD gains (joint-space, critically damped: Kd = 2*sqrt(Kp)).
KP_TASK = 600.0
KD_TASK = 2.0 * (KP_TASK ** 0.5)

# TorqueCommandFilter (smoothing primitive from real_Cartpole/controller_core/filters.py)
FILTER_LOWPASS_ALPHA = 0.25          # first-order low-pass coefficient
FILTER_RATE_LIMIT_NM_S = 400.0       # Nm/s slew-rate limit per joint

# ---------------------------------------------------------------------------
# C3 (geometric fabric) tuning -- soft-safety layer, see EgoController.fabric_qacc.
# FABRIC_REPULSION_MARGIN kept at 2x H_TRIGGER (same ratio as v1/v2: 0.10/0.05).
# ---------------------------------------------------------------------------
FABRIC_M_ATTR = 1.0
FABRIC_KP_ATTR = 40.0
FABRIC_KD_ATTR = 2.0 * (FABRIC_KP_ATTR ** 0.5)
FABRIC_JOINT_MARGIN = 0.15           # [rad] joint-limit soft margin
FABRIC_K_JL = 2.0
FABRIC_D_JL = 0.5
FABRIC_W_JL = 0.05
FABRIC_REPULSION_MARGIN = 2.0 * H_TRIGGER   # [m] = 0.08 (v2: 0.10, same 2x-H_TRIGGER ratio)
FABRIC_K_REP = 4.0
FABRIC_D_REP = 18.0
FABRIC_W_REP = 0.35
FABRIC_QDD_CAP = 60.0
FABRIC_M_CAP = 3.0e3
FABRIC_A_CAP = 25.0
FABRIC_X_FLOOR = 0.02

# ---------------------------------------------------------------------------
# Intruder (right arm) scripted sweep -- SAME timeline/oscillation structure
# as v2 (ramp in, oscillate, ramp out; see _smoothstep/intruder_ctrl below),
# but RIGHT_HOME_CTRL / RIGHT_INTRUDE_CTRL are new poses found for the v3
# geometry by probe_v3.py:
#   - RIGHT_HOME_CTRL = the canonical scene's own "home"/ready keyframe pose
#     for the right arm (scenes/scene_yor_canonical.xml <keyframe>).
#   - RIGHT_INTRUDE_CTRL = a damped-least-squares IK solution (seeded by the
#     same orientation-aware coarse random search used for the left-arm grasp
#     IK in scripts/verify_yor_canonical.py) that aims the right arm's grasp
#     site (right_arm_ee_grasp) AT the left arm's grasp site position while
#     the left arm is frozen at LEFT_GRASP_Q -- i.e. the deepest reachable
#     "both grippers try to occupy the same point" intrusion. At this pose,
#     probe_v3.py measured the closest EGO/INT_BARRIER_BODIES pair
#     (gripper_base-to-gripper_base) at 0.040m, vs. 0.229m at RIGHT_HOME_CTRL
#     -- i.e. h goes from +0.129 (safe) to -0.060 (violated) under
#     COMBINED_RADIUS=0.10, a comparable safety-margin swing to v1/v2's
#     0.5m-spacing rig scaled to the new 0.185m mount separation.
LEFT_GRASP_Q = np.array([-0.00944173, 2.69291536, -1.83582068, -0.47028722, 1.03733985, -1.30204097])
RIGHT_HOME_CTRL = np.array([0.0, 1.58065, -0.578175, 0.0, -0.912, -0.78])
RIGHT_INTRUDE_CTRL = np.array([-0.15229176, 2.56294267, -1.48373594, 0.93891875, 1.01939931, 0.80796997])
# timeline (seconds): ramp in [T_IN0, T_IN1], oscillating hold until
# T_HOLD_END, ramp out to T_OUT_END, hold at home thereafter. Unchanged from v2.
T_IN0, T_IN1 = 0.4, 0.8
T_HOLD_END = 5.2
T_OUT_END = 5.8
OSC_AMP = 0.14          # additive penetration-depth oscillation amplitude (of the home->intrude span);
                        # clipped to [0, 1.08] factor range below (checked by hand against all 6
                        # right-arm joint ranges for the v3 RIGHT_HOME_CTRL/RIGHT_INTRUDE_CTRL pair --
                        # tightest margin is joint5 at 1.08x: -0.912+1.08*1.9314=+1.174 rad, inside
                        # [-1.2, 1.2]).
OSC_FREQ_HZ = 5.0       # near/above the ego task-PD's natural frequency (sqrt(Kp_task)/2pi ~= 3.9 Hz)
                        # -- unchanged from v2; still what makes bang-bang (C1) chatter/destabilize.


def _smoothstep(t, t0, t1):
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    s = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def intruder_ctrl(t: float) -> np.ndarray:
    """Deterministic, open-loop right-arm ctrl trajectory (same for all conditions)."""
    if t < T_IN0:
        s = 0.0
    elif t < T_IN1:
        s = _smoothstep(t, T_IN0, T_IN1)
    elif t < T_HOLD_END:
        osc = OSC_AMP * np.sin(2.0 * np.pi * OSC_FREQ_HZ * (t - T_IN1))
        s = np.clip(1.0 + osc, 0.0, 1.08)
    elif t < T_OUT_END:
        s = 1.0 - _smoothstep(t, T_HOLD_END, T_OUT_END)
    else:
        s = 0.0
    return RIGHT_HOME_CTRL + s * (RIGHT_INTRUDE_CTRL - RIGHT_HOME_CTRL)


# ---------------------------------------------------------------------------
# COLLISION-FIDELITY FIX (root cause + fix; full investigation in
# experiments/ab_smooth_vs_bangbang/COLLISION_FIDELITY.md; supersedes the v3
# "add synthetic proxy spheres" workaround described in RESULTS_v3.md).
#
# ROOT CAUSE (single, concrete, confirmed -- no separate broadphase bug):
# yor_robot_derived.mjcf's <default class="collision"> sets contype="0"
# conaffinity="1" for every robot link's REAL collision geom (the
# already-present fitted capsules/boxes per link). Two contype=0 geoms never
# generate a contact against each other regardless of proximity (MuJoCo's
# contact test is (contype1 & conaffinity2) || (contype2 & conaffinity1) !=
# 0, identically 0 when both contypes are 0) -- so AS SHIPPED, the two arms
# are structurally incapable of ever colliding with EACH OTHER (table/floor/
# box collisions are unaffected, since those geoms use the ordinary
# contype=conaffinity=1 default).
#
# The v3 write-up (RESULTS_v3.md) additionally reported a *second*,
# unexplained drop (mj_geomDistance said -3.7cm overlap for the two
# gripper_base capsules, but data.contact stayed empty even after patching
# contype/conaffinity on the 3 EGO/INT_BARRIER_BODIES). Re-investigated here
# (experiments/ab_smooth_vs_bangbang/diag_contact_drop.py): patching
# contype/conaffinity on ALL real per-link "collision"-class geoms (group==3)
# -- not just the 3 barrier bodies -- and re-running mj_forward at the exact
# LEFT_GRASP_Q/RIGHT_INTRUDE_CTRL pose reproduces ZERO missing contacts (10/10
# true-penetrating cross-arm geom pairs, incl. gripper_base-vs-gripper_base,
# all present in data.contact, matching mj_geomDistance exactly). No second
# bug is reproducible once the filter fix is applied to the real geoms; the
# "silently dropped" case in RESULTS_v3.md is best explained by only 3 of the
# ~13 collision geoms per arm having been patched, so most cross-arm pairs
# (including possibly the specific one probed) were still contype=0.
#
# FIX: patch contype/conaffinity directly on the REAL per-link collision
# geoms (fitted capsules from the mesh + finger-pad boxes, group==3) for
# every left_arm_*/right_arm_* body -- no synthetic sphere geoms needed.
# Bit scheme unchanged from v3 (two HIGH, otherwise-unused bits, NOT bit0
# which every other geom in the scene uses for its ordinary
# contype=0/conaffinity=1 or contype=conaffinity=1 defaults):
#   left vs right:                  contype_left(0b0100) & conaffinity_right(0b0100) != 0 -> COLLIDE (intended)
#   left vs left / right vs right:  bit patterns never overlap                             -> no self-collision
#   real geom vs table/box/etc (all bit0-only masks): 0b0100 & 0b0001 == 0 both ways        -> no spurious contact
# The scene files on disk are never touched (in-memory MjSpec patch only).
# ---------------------------------------------------------------------------
_LEFT_BIT, _RIGHT_BIT = 0b0100, 0b1000        # 4, 8  -- arm<->arm (cross-arm) identity
_LEFT_SELF, _RIGHT_SELF = 0b1_0000, 0b10_0000  # 16, 32 -- per-arm SELF-collision group
CUBE_ARM_BITS = _LEFT_BIT | _RIGHT_BIT         # 12 -- put on a cube so BOTH arms avoid it
# runtime helper below (set_cube_arm_collision) toggles CUBE_ARM_BITS on a cube's
# conaffinity: cleared while the cube is welded to the gripper (so the held cube
# does not fight the weld), restored on release.


def _patch_real_collision_geoms(spec):
    """Full-fidelity contype/conaffinity scheme for the bimanual scene. Each real
    per-link "collision" geom ships as (contype=0, conaffinity=1) -> two such
    geoms never collide. We rewrite them so the RIGHT things collide and the
    grasp/weld isolations stay intact. Bit scheme:
      left arm : contype = LEFT_BIT|LEFT_SELF (4|16),  conaffinity = RIGHT_BIT|LEFT_SELF (8|16)
      right arm: contype = RIGHT_BIT|RIGHT_SELF(8|32), conaffinity = LEFT_BIT|RIGHT_SELF (4|32)
    => arm<->arm collide (via 4/8), each arm SELF-collides (via its own 16/32 bit,
       which the OTHER arm lacks, so self bits never cross arms), and neither self
       bit reaches the objects. Then, on the object/world side (conaffinity ORed,
       contype left at 1 so object<->object/world is unchanged):
      table/floor/leg : += (4|8)  -> BOTH arms collide with fixed world structure
      cup             : += 8      -> RIGHT arm collides with the cup, LEFT (holder) does NOT
      cube            : += (4|8)  -> both arms collide with a cube... EXCEPT the welded/held
                                     one, whose bits are cleared at pick time (see
                                     set_cube_arm_collision) so it doesn't fight the weld.
    Adjacent same-arm links overlap at the shared joint, which would spawn spurious
    self-contacts; those pairs are excluded via _exclude_adjacent_links below."""
    L_ct, L_ca = _LEFT_BIT | _LEFT_SELF,  _RIGHT_BIT | _LEFT_SELF
    R_ct, R_ca = _RIGHT_BIT | _RIGHT_SELF, _LEFT_BIT | _RIGHT_SELF
    for b in spec.bodies:
        if b.name.startswith(EGO_PREFIX):
            contype, conaffinity = L_ct, L_ca
        elif b.name.startswith(INT_PREFIX):
            contype, conaffinity = R_ct, R_ca
        else:
            continue
        for g in b.geoms:
            if g.contype == 0 and g.conaffinity == 1:
                g.contype, g.conaffinity = contype, conaffinity

    # --- object / fixed-world side: OR arm bits into CONAFFINITY only (contype
    # stays 1, so object<->object and object<->world collisions are unchanged) ---
    _bodies = list(spec.bodies)
    try:
        _bodies.append(spec.worldbody)
    except Exception:
        pass
    _arm_both = _LEFT_BIT | _RIGHT_BIT   # 12
    for b in _bodies:
        for g in b.geoms:
            gn = (g.name or "").lower()
            if not (g.contype or g.conaffinity):
                continue
            if "cup" in gn:                                   # right arm avoids the cup; left holds it
                g.conaffinity |= _RIGHT_BIT
            elif "cube" in gn:                                # both arms avoid un-held cubes
                g.conaffinity |= _arm_both
            elif "table" in gn or "floor" in gn or "leg" in gn:
                g.conaffinity |= _arm_both
    _exclude_adjacent_links(spec)


def _exclude_adjacent_links(spec):
    """Exclude collisions between ADJACENT (parent<->child) links within each arm,
    whose fitted capsules overlap at the shared joint and would otherwise generate
    a continuous spurious self-contact. Non-adjacent same-arm links are left to
    collide (that IS the self-collision we want). Adds a MjSpec <exclude> for each
    parent/child body pair whose names both start with an arm prefix."""
    id2name = {b.id: b.name for b in spec.bodies}
    added = 0
    for b in spec.bodies:
        nm = b.name or ""
        if not (nm.startswith(EGO_PREFIX) or nm.startswith(INT_PREFIX)):
            continue
        pid = b.parent.id if b.parent is not None else -1
        pnm = id2name.get(pid, "")
        if pnm and (pnm.startswith(EGO_PREFIX) or pnm.startswith(INT_PREFIX)):
            try:
                spec.add_exclude(bodyname1=pnm, bodyname2=nm)
                added += 1
            except Exception:
                pass
    # Also exclude the two gripper FINGERS of each arm (link7<->link8): they are
    # SIBLINGS (both children of gripper_base, so not caught by the parent-child
    # loop above) and sit close together -- their jaw geoms overlap at rest and
    # would spawn ~25 spurious self-contacts. The two jaws must never self-collide
    # (they close on grasped objects, not each other).
    for pre in (EGO_PREFIX, INT_PREFIX):
        try:
            spec.add_exclude(bodyname1=f"{pre}link7", bodyname2=f"{pre}link8")
            added += 1
        except Exception:
            pass
    return added


def set_cube_arm_collision(model, cube_body_id: int, enabled: bool):
    """Runtime toggle for the held cube: clear the arm bits (CUBE_ARM_BITS) from
    every collision geom of the cube BODY while it is welded/held (enabled=False,
    so no arm collides with the moving held cube and fights the mjEQ_WELD), and
    restore them on release (enabled=True). Operates on the compiled MjModel in
    place. Call on the weld rising edge (False) and falling edge (True)."""
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != cube_body_id:
            continue
        if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            continue
        if enabled:
            model.geom_conaffinity[gid] |= CUBE_ARM_BITS
        else:
            model.geom_conaffinity[gid] &= ~CUBE_ARM_BITS


def load_model():
    spec = mujoco.MjSpec.from_file(SCENE)
    _patch_real_collision_geoms(spec)
    model = spec.compile()
    assert abs(model.opt.timestep - DT) < 1e-12, (model.opt.timestep, DT)
    return model


# ---------------------------------------------------------------------------
# Corrected multi-keypoint barrier proxy (replaces the old 3-sphere-per-arm,
# fixed-radius EGO_BARRIER_BODIES/INT_BARRIER_BODIES + R_EGO/R_INT scheme).
# Each LINK_NAMES body contributes 2 endpoint spheres (+1 midpoint sphere if
# its fitted capsule's half-length exceeds _LONG_CAPSULE_HALFLEN_THRESH) at
# the capsule's OWN radius, read directly from the compiled model's real
# per-link collision geometry -- so the proxy always matches whatever
# geometry the scene actually ships, with no hand-guessed radius.
# ---------------------------------------------------------------------------
def _quat_rotate(q, v):
    w, x, y, z = q
    return np.array([
        (1 - 2 * (y * y + z * z)) * v[0] + 2 * (x * y - z * w) * v[1] + 2 * (x * z + y * w) * v[2],
        2 * (x * y + z * w) * v[0] + (1 - 2 * (x * x + z * z)) * v[1] + 2 * (y * z - x * w) * v[2],
        2 * (x * z - y * w) * v[0] + 2 * (y * z + x * w) * v[1] + (1 - 2 * (x * x + y * y)) * v[2],
    ])


def _body_keypoints(model, body_id):
    """(offset_local(3,), radius) sphere list approximating a body's REAL
    "collision"-class (group==3) geom(s): 2 (or 3, for long capsules)
    spheres per fitted capsule at its true radius, 1 bounding sphere per box
    geom (finger pads)."""
    kps = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != body_id or model.geom_group[gid] != 3:
            continue
        gtype = model.geom_type[gid]
        pos = model.geom_pos[gid].copy()
        size = model.geom_size[gid].copy()
        if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            axis = _quat_rotate(model.geom_quat[gid], np.array([0.0, 0.0, 1.0]))
            hl, r = float(size[1]), float(size[0])
            kps.append((pos + hl * axis, r))
            kps.append((pos - hl * axis, r))
            if hl > _LONG_CAPSULE_HALFLEN_THRESH:
                kps.append((pos, r))
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            kps.append((pos, float(np.linalg.norm(size))))
        else:
            kps.append((pos, float(size[0])))
    return kps


_KEYPOINT_CACHE: dict = {}


def barrier_keypoints(model, prefix):
    """List of (body_id, local_offset(3,), radius) for all LINK_NAMES bodies
    under `prefix` ("left_arm_" or "right_arm_"). Cached per model instance
    (geometry is static after compile)."""
    key = (id(model), prefix)
    cached = _KEYPOINT_CACHE.get(key)
    if cached is None:
        out = []
        for ln in LINK_NAMES:
            bid = model.body(prefix + ln).id
            for offset, r in _body_keypoints(model, bid):
                out.append((bid, offset, r))
        cached = _KEYPOINT_CACHE[key] = out
    return cached


def _kp_world(data, body_id, offset):
    return data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ offset


def joint_addrs(model, names):
    jids = [model.joint(n).id for n in names]
    qposadr = np.array([model.jnt_qposadr[j] for j in jids])
    dofadr = np.array([model.jnt_dofadr[j] for j in jids])
    return jids, qposadr, dofadr


def disable_right_arm_position_servos(model):
    """Mirror of disable_left_arm_position_servos, for the RIGHT arm's 6
    ARM joint position actuators only (added for the v4 cooperative-intruder
    experiment, RESULTS_v4.md: the intruder becomes torque-controlled under
    its own HOCBF-QP filter in conditions B/C, exactly as the ego already is
    under C2). Right FINGER actuators (joint7_pos/joint8_pos) are left
    untouched (fingers stay open, position-servo driven -- the intruder
    grasps nothing)."""
    for name in [f"right_arm_joint{i}_pos" for i in range(1, 7)]:
        aid = model.actuator(name).id
        model.actuator_gainprm[aid, :] = 0.0
        model.actuator_biasprm[aid, :] = 0.0


def disable_left_arm_position_servos(model):
    """Zero out gain/bias for the 6 left ARM joint position actuators so that
    driving them via data.ctrl produces zero force -- torque control for these
    6 DOF happens exclusively through data.qfrc_applied. Left FINGER actuators
    (joint7_pos/joint8_pos) and the entire RIGHT arm/finger actuators are left
    untouched (right arm stays position-servo driven; left fingers stay
    position-servo driven closed, holding the grasp)."""
    for name in [f"left_arm_joint{i}_pos" for i in range(1, 7)]:
        aid = model.actuator(name).id
        model.actuator_gainprm[aid, :] = 0.0
        model.actuator_biasprm[aid, :] = 0.0


def init_grasped_state(model):
    """Reset to the 'home' keyframe (right arm ready pose, left arm ready
    pose, fingers open, box resting on the table), then:
      1. Teleport the LEFT ARM (only) to LEFT_GRASP_Q -- the pre-computed IK
         grasp pose (probe_v3.py) that places the open left fingers straddling
         the box top. (v3 NEW STEP: the canonical scene's own "home" keyframe
         is a generic ready pose, not already positioned at the box the way
         the old rig's home keyframe was -- so the arm itself, not just the
         box, must be teleported.)
      2. Reproduce the exact grasp-init technique validated in
         scripts/test_pickup_scene.py / v1/v2 sim_common.py: teleport the box
         to the midpoint of the (now correctly-positioned) OPEN left finger
         pads, close the fingers with gravity off (transient, to avoid an
         artificial free-fall race against the closing fingers), then restore
         gravity. Returns the MjData with the grasp established and the box
         supported purely by finger friction -- the SAME initial condition
         for all four conditions (C0/C1/C2/C3).
    """
    data = mujoco.MjData(model)
    key = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)

    _, left_qposadr, _ = joint_addrs(model, LEFT_ARM_JOINTS)
    data.qpos[left_qposadr] = LEFT_GRASP_Q
    left_act_ids = [model.actuator(f"left_arm_joint{i}_pos").id for i in range(1, 7)]
    data.ctrl[left_act_ids] = LEFT_GRASP_Q
    mujoco.mj_forward(model, data)

    pad7 = model.geom("left_arm_link7_pad").id
    pad8 = model.geom("left_arm_link8_pad").id
    pad_mid = 0.5 * (data.geom_xpos[pad7] + data.geom_xpos[pad8])

    box_jid = model.joint("test_box_free").id
    box_qposadr = model.jnt_qposadr[box_jid]
    box_qveladr = model.jnt_dofadr[box_jid]
    data.qpos[box_qposadr:box_qposadr + 3] = pad_mid
    data.qpos[box_qposadr + 3:box_qposadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[box_qveladr:box_qveladr + 6] = 0.0
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
    # settle briefly with gravity restored (fingers already closed & gripping)
    for _ in range(200):
        mujoco.mj_step(model, data)
    return data


def min_pair_distance(model, data, ego_prefix=EGO_PREFIX, int_prefix=INT_PREFIX):
    """Closest pair over the FULL corrected multi-keypoint proxy (LINK_NAMES x
    2-3 spheres per link, see barrier_keypoints). "Closest" is measured by
    h = dist - (r_ego + r_int + D_MIN), not raw distance, since keypoint radii
    now vary per link (unlike the old fixed-R_EGO/R_INT scheme).

    Returns (h_min, dist, ego_body_id*, int_body_id*, unit_vector n =
    (p_ego-p_int)/||.||, ego_offset*, int_offset*) for the worst (min-h) pair.
    """
    ego_kps = barrier_keypoints(model, ego_prefix)
    int_kps = barrier_keypoints(model, int_prefix)
    best = None
    for eb, eoff, er in ego_kps:
        pe = _kp_world(data, eb, eoff)
        for ib, ioff, ir in int_kps:
            pi = _kp_world(data, ib, ioff)
            d = pe - pi
            dist = float(np.linalg.norm(d))
            h = dist - (er + ir + D_MIN)
            if best is None or h < best[0]:
                n = d / max(dist, 1e-9)
                best = (h, dist, eb, ib, n, eoff, ioff)
    return best


def point_jac(model, data, body_id):
    """Translational Jacobian (3 x nv) of the body-frame origin."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, data.xpos[body_id], body_id)
    return jacp


def point_jac_at(model, data, body_id, offset):
    """Translational Jacobian (3 x nv) of an arbitrary point rigidly attached
    to `body_id`, given as a LOCAL (body-frame) offset -- used for the
    corrected proxy's keypoint spheres (which are generally not at the body
    origin)."""
    p_world = _kp_world(data, body_id, offset)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, p_world, body_id)
    return jacp


def point_jacdot_at(model, data, body_id, offset):
    """Time-derivative (3 x nv) of point_jac_at, at current qvel."""
    p_world = _kp_world(data, body_id, offset)
    jacp_dot = np.zeros((3, model.nv))
    jacr_dot = np.zeros((3, model.nv))
    mujoco.mj_jacDot(model, data, jacp_dot, jacr_dot, p_world, body_id)
    return jacp_dot


def body_jac_full(model, data, body_id):
    """Translational + rotational Jacobian (each 3 x nv) of a body's origin,
    full nv width (caller restricts columns)."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, data.xpos[body_id], body_id)
    return jacp, jacr


def skew(v):
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def point_jacdot(model, data, body_id):
    """Time-derivative (3 x nv) of the translational Jacobian, at current qvel."""
    jacp_dot = np.zeros((3, model.nv))
    jacr_dot = np.zeros((3, model.nv))
    mujoco.mj_jacDot(model, data, jacp_dot, jacr_dot, data.xpos[body_id], body_id)
    return jacp_dot


def full_mass_matrix(model, data):
    m = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, m, data.qM)
    return m


def sparc(speed: np.ndarray, dt: float, padlevel=4, fc=10.0, amp_th=0.05) -> float:
    """Spectral ARC length (Balasubramanian et al. 2015), standard implementation."""
    n = len(speed)
    if n < 4:
        return float("nan")
    nfft = int(2 ** (np.ceil(np.log2(n)) + padlevel))
    f = np.fft.rfftfreq(nfft, d=dt)
    Mf = np.abs(np.fft.rfft(speed, nfft))
    if Mf.max() < 1e-12:
        return float("nan")
    Mf = Mf / Mf.max()
    fc_idx = np.where(f <= fc)[0]
    if len(fc_idx) < 2:
        return float("nan")
    f_sel = f[fc_idx]
    Mf_sel = Mf[fc_idx]
    inx = np.where(Mf_sel >= amp_th)[0]
    if len(inx) < 2:
        return float("nan")
    fc_final = range(inx[0], inx[-1] + 1)
    f_c = f_sel[fc_final]
    Mf_c = Mf_sel[fc_final]
    dArc = np.sqrt((np.diff(f_c) / (f_c[-1] - f_c[0] + 1e-12)) ** 2 + np.diff(Mf_c) ** 2)
    return -float(np.sum(dArc))


def jerk_metrics(pos_series: np.ndarray, dt: float):
    """pos_series: (T, D). Returns (rms_jerk_norm, max_jerk_norm) where jerk is
    the 3rd time-derivative computed by successive np.gradient (central
    differences), and norm is taken over the D axes at each timestep."""
    v = np.gradient(pos_series, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    j = np.gradient(a, dt, axis=0)
    jn = np.linalg.norm(j, axis=1) if pos_series.ndim == 2 else np.abs(j)
    return float(np.sqrt(np.mean(jn ** 2))), float(np.max(jn))


def all_pair_barriers(model, data, ego_dofadr, int_dofadr,
                       ego_prefix=EGO_PREFIX, int_prefix=INT_PREFIX,
                       extra_ego_keypoints=None):
    """Barrier info for EVERY (ego_keypoint, intruder_keypoint) sphere pair
    over the CORRECTED multi-keypoint proxy (~25x25 = 625 pairs; see
    barrier_keypoints/LINK_NAMES), not just the argmin. Used by the C2
    HOCBF-QP so that the safety constraint does not depend on a single,
    discretely-switching closest-pair choice, and by C3's fabric repulsion.

    Per-keypoint Jacobians are computed ONCE per keypoint (not once per pair)
    and reused across all pairs it participates in -- O(n_ego + n_int) mj_jac
    calls, not O(n_ego * n_int).

    extra_ego_keypoints: OPTIONAL list of (body_id, local_offset(3,), radius)
    tuples -- the SAME per-keypoint format barrier_keypoints/_body_keypoints
    already produce -- appended to the ego (left, by default) keypoint set
    BEFORE the pairwise states are built. Intended use: a payload rigidly
    carried by the ego arm (e.g. a held cup), represented as a few extra
    spheres given as a FIXED local offset on one of the ego arm's own bodies
    (typically its end-effector, since mj_jac must be taken on a body that is
    actually on the ego arm's kinematic chain -- a separately-jointed payload
    body that is only kinematically re-pinned each step, not welded, would
    yield a structurally-zero Jacobian w.r.t. ego_dofadr). Once appended, the
    payload spheres flow through the EXACT SAME pairwise distance/Jacobian/
    HOCBF machinery below as any ordinary ego-arm keypoint -- so e.g. the
    intruder's own avoidance QP rows (built from these same `pairs`, see
    callers) automatically gain "avoid the payload too" rows, with no new
    barrier type and no change to the QP math.
    Default None => byte-identical to the original behavior (ego_kps
    unchanged, no extra work) -- existing callers that never pass this
    (run_ab4.py, run_sweep.py, run_ab5/6/7.py, mpc.py, safety_module.py, ...)
    are completely unaffected.
    """
    ego_kps = barrier_keypoints(model, ego_prefix)
    if extra_ego_keypoints:
        ego_kps = list(ego_kps) + list(extra_ego_keypoints)
    int_kps = barrier_keypoints(model, int_prefix)

    def _states(kps, dofadr):
        states = []
        for bid, off, r in kps:
            p = _kp_world(data, bid, off)
            J = point_jac_at(model, data, bid, off)[:, dofadr]
            Jd = point_jacdot_at(model, data, bid, off)[:, dofadr]
            v = J @ data.qvel[dofadr]
            states.append((bid, p, r, J, Jd, v))
        return states

    ego_states = _states(ego_kps, ego_dofadr)
    int_states = _states(int_kps, int_dofadr)

    out = []
    for eid, pe, er, Je, JeDot, v_ego in ego_states:
        for iid, pi, ir, Ji, JiDot, v_int in int_states:
            d = pe - pi
            dist = float(np.linalg.norm(d))
            n = d / max(dist, 1e-9)
            hdot = float(n @ (v_ego - v_int))
            out.append(dict(h=dist - (er + ir + D_MIN), dist=dist, eid=eid, iid=iid,
                             n=n, Je=Je, Ji=Ji, JeDot=JeDot, JiDot=JiDot,
                             v_ego=v_ego, v_int=v_int, hdot=hdot))
    return out


# ---------------------------------------------------------------------------
# TABLE-PLANE (static half-space) HOCBF-QP rows -- additive, OPTIONAL helper
# for callers that want to keep an arm's own collision-proxy keypoints above a
# fixed world-z floor (e.g. scripts/make_cup_collect_safety_demo.py's
# cup-collect scene, whose table_top surface is not in ANY existing barrier:
# arm<->table physics is deliberately masked off -- see
# _patch_real_collision_geoms's bit scheme above, which only ever turns on
# left<->right collision -- and all_pair_barriers is arm-vs-arm/payload only,
# never arm-vs-scene-geometry). Does NOT touch all_pair_barriers or any
# existing caller (run_ab4.py, run_sweep.py, run_ab5/6/7.py, mpc.py,
# safety_module.py, ...) -- purely additive, opt-in, called nowhere by
# default.
# ---------------------------------------------------------------------------
def table_plane_barrier_rows(model, data, kps, dofadr, M_inv, bias, z_floor,
                              margin=0.0, alpha1=ALPHA1, alpha2=ALPHA2, kp_radius=False):
    """HOCBF-QP rows for a STATIC horizontal half-space floor barrier
    (h_plane = z_keypoint - (z_floor - margin) >= 0) over an arbitrary set of
    (body_id, local_offset(3,), radius) keypoints -- the SAME per-keypoint
    format barrier_keypoints/all_pair_barriers already use (e.g.
    barrier_keypoints(model, EGO_PREFIX) for one arm's own real collision-
    proxy geometry).

    A simpler special case of the pairwise relative-degree-2 HOCBF
    construction all_pair_barriers' callers already build by hand (see e.g.
    scripts/make_cup_collect_safety_demo.py's QP construction): the "other
    side" here is a STATIC world plane, not a moving intruder arm, so the
    unit normal n=(0,0,1) is CONSTANT (ndot=0, unlike the cross-arm rows'
    n(t) which tracks the live relative-position unit vector) and there is no
    intruder velocity/acceleration term (v_int=0, a_int=0, so no `ndot @
    (v_ego-v_int)` or `a_int` correction) -- only this arm's own J/Jdot at
    each keypoint enter hddot. SAME cascade otherwise: psi1 = hdot +
    alpha1*h, psi2 = hddot + alpha1*hdot + alpha2*psi1 >= 0, and the SAME
    box-QP row convention (a_row @ tau <= b_row, i.e. `-A_h @ tau <=
    const_term` so psi2>=0 once the QP's solution satisfies the row) the
    arm-arm rows use -- callers just np.vstack/np.concatenate these onto
    their existing a_ineq/b_ineq, no other QP-construction code changes.

    dofadr/M_inv/bias must all be for the SAME arm as `kps` (e.g. left_dofadr
    + M_ll_inv + bias_l for `kps=barrier_keypoints(model, EGO_PREFIX)`).

    margin [m]: SUBTRACTED from z_floor before comparison, i.e. the enforced
    threshold is (z_floor - margin) -- a small POSITIVE margin therefore
    RELAXES the barrier (a keypoint may sit up to `margin` below z_floor
    before h goes negative), giving the gripper/EE room to still reach
    objects resting ON the table surface (e.g. a cube whose center sits a
    few mm above the table). margin=0.0 enforces the table's own surface
    height exactly.

    kp_radius (default False = unchanged): True -> each keypoint's proxy-sphere
    radius is included, h = z_kp - r - (z_floor - margin), i.e. the sphere's
    SURFACE (not its centre) is kept above the threshold (2026-09-25 --ctrlfixes
    F2: without it the r=0.05 palm capsule touches the table while its centre is
    still ~55 mm above the threshold).

    Returns (a_rows, b_rows): (len(kps), len(dofadr)) / (len(kps),) arrays,
    ready to np.vstack/np.concatenate alongside an existing a_ineq/b_ineq.
    """
    n = np.array([0.0, 0.0, 1.0])
    z_thresh = z_floor - margin
    a_rows, b_rows = [], []
    for bid, off, _r in kps:
        p = _kp_world(data, bid, off)
        J = point_jac_at(model, data, bid, off)[:, dofadr]
        Jd = point_jacdot_at(model, data, bid, off)[:, dofadr]
        v = J @ data.qvel[dofadr]
        h = float(p[2] - z_thresh) if not kp_radius else float(p[2] - _r - z_thresh)
        hdot = float(v[2])
        hddot_rest = float(n @ (Jd @ data.qvel[dofadr]))
        psi1 = hdot + alpha1 * h
        A_h = n @ J @ M_inv
        const_term = -float(A_h @ bias) + hddot_rest + alpha1 * hdot + alpha2 * psi1
        a_rows.append(-A_h)
        b_rows.append(const_term)
    return np.stack(a_rows, axis=0), np.array(b_rows)


def payload_reflected_terms(model, data, ee_body_id, box_body_id, left_dofadr):
    """Reflect the grasped box's mass/inertia + gravity into the LEFT arm's
    joint-space dynamics, via the grasp Jacobian, under a rigid-grasp
    assumption. See v1/v2 sim_common.py for the full derivation comment
    (unchanged here -- only body/joint NAMES differ, not the method)."""
    m_box = float(model.body_mass[box_body_id])
    inertia_diag = model.body_inertia[box_body_id]
    R_box = data.xmat[box_body_id].reshape(3, 3)
    I_box_world = R_box @ np.diag(inertia_diag) @ R_box.T

    Jp_full, Jr_full = body_jac_full(model, data, ee_body_id)
    Jp_ee = Jp_full[:, left_dofadr]
    Jr_ee = Jr_full[:, left_dofadr]

    r = data.xpos[box_body_id] - data.xpos[ee_body_id]
    Jp_box = Jp_ee - skew(r) @ Jr_ee
    Jr_box = Jr_ee

    J_box = np.vstack([Jp_box, Jr_box])  # 6x6
    I_spatial = np.zeros((6, 6))
    I_spatial[:3, :3] = m_box * np.eye(3)
    I_spatial[3:, 3:] = I_box_world
    M_box_reflected = J_box.T @ I_spatial @ J_box

    g_vec = model.opt.gravity.copy()
    bias_box = -Jp_box.T @ (m_box * g_vec)
    return M_box_reflected, bias_box


class EgoController:
    """Base class: computes the ego (left arm) task torque each step. Subclasses
    C0/C1/C2/C3 layer the safety behavior on top of `task_torque`."""

    def __init__(self, model, q0_left, couple_payload=True):
        self.model = model
        self.q0_left = q0_left.copy()
        _, self.left_qposadr, self.left_dofadr = joint_addrs(model, LEFT_ARM_JOINTS)
        _, self.right_qposadr, self.right_dofadr = joint_addrs(model, RIGHT_ARM_JOINTS)
        self.couple_payload = couple_payload
        self.ee_body_id = model.body(EE_BODY_NAME).id
        self.box_body_id = model.body(BOX_BODY_NAME).id

    def payload_terms(self, data):
        if not self.couple_payload:
            return np.zeros((6, 6)), np.zeros(6)
        return payload_reflected_terms(self.model, data, self.ee_body_id,
                                        self.box_body_id, self.left_dofadr)

    def task_torque(self, data):
        """Computed-torque joint-space PD toward q0 (hold initial EE pose),
        with the grasped box's mass/inertia/gravity reflected into M_ll and
        bias_l -- used identically by every condition."""
        M = full_mass_matrix(self.model, data)
        M_ll = M[np.ix_(self.left_dofadr, self.left_dofadr)]
        bias_l = data.qfrc_bias[self.left_dofadr]
        M_box, bias_box = self.payload_terms(data)
        M_ll = M_ll + M_box
        bias_l = bias_l + bias_box
        q = data.qpos[self.left_qposadr]
        qd = data.qvel[self.left_dofadr]
        qacc_des = KP_TASK * (self.q0_left - q) - KD_TASK * qd
        tau = M_ll @ qacc_des + bias_l
        return tau, M_ll, bias_l

    def fabric_qacc(self, data, pairs):
        """C3: minimal self-rolled geometric fabric (Ratliff et al.,
        "RMPflow" / "Optimization Fabrics"); see v1/v2 sim_common.py for the
        full derivation comment (unchanged -- only body/joint NAMES differ)."""
        q = data.qpos[self.left_qposadr]
        qd = data.qvel[self.left_dofadr]
        n = 6
        M_total = np.zeros((n, n))
        f_total = np.zeros(n)

        # 1. attractor (always-on)
        J = np.eye(n)
        M_attr = FABRIC_M_ATTR * np.eye(n)
        a_attr = FABRIC_KP_ATTR * (self.q0_left - q) - FABRIC_KD_ATTR * qd
        M_total += J.T @ M_attr @ J
        f_total += J.T @ M_attr @ a_attr

        # 2. joint-limit barrier, both sides, per joint
        margin = FABRIC_JOINT_MARGIN
        for j in range(n):
            lo, hi = LEFT_JOINT_RANGES[j]
            x_lo = q[j] - lo
            if x_lo < margin:
                x_lo = max(x_lo, FABRIC_X_FLOOR)
                Jr = np.zeros((1, n)); Jr[0, j] = 1.0
                xdot = qd[j]
                a_raw = -FABRIC_K_JL * (1.0 / x_lo - 1.0 / margin) - FABRIC_D_JL * xdot
                a = np.array([np.clip(a_raw, -FABRIC_A_CAP, FABRIC_A_CAP)])
                m = min(FABRIC_W_JL / (x_lo ** 2), FABRIC_M_CAP)
                M_i = np.array([[m]])
                M_total += Jr.T @ M_i @ Jr
                f_total += Jr.T @ M_i @ a
            x_hi = hi - q[j]
            if x_hi < margin:
                x_hi = max(x_hi, FABRIC_X_FLOOR)
                Jr = np.zeros((1, n)); Jr[0, j] = -1.0
                xdot = -qd[j]
                a_raw = -FABRIC_K_JL * (1.0 / x_hi - 1.0 / margin) - FABRIC_D_JL * xdot
                a = np.array([np.clip(a_raw, -FABRIC_A_CAP, FABRIC_A_CAP)])
                m = min(FABRIC_W_JL / (x_hi ** 2), FABRIC_M_CAP)
                M_i = np.array([[m]])
                M_total += Jr.T @ M_i @ Jr
                f_total += Jr.T @ M_i @ a

        # 3. arm-arm repulsion, all ego/intruder sphere pairs
        margin_rep = FABRIC_REPULSION_MARGIN
        for p in pairs:
            h_p = p["h"]
            if h_p < margin_rep:
                h_safe = max(h_p, FABRIC_X_FLOOR)
                n_p, Je_p = p["n"], p["Je"]
                Jr = (n_p @ Je_p).reshape(1, n)
                a_raw = (-FABRIC_K_REP * (1.0 / h_safe - 1.0 / margin_rep)
                         - FABRIC_D_REP * p["hdot"])
                a = np.array([np.clip(a_raw, -FABRIC_A_CAP, FABRIC_A_CAP)])
                m = min(FABRIC_W_REP / (h_safe ** 2), FABRIC_M_CAP)
                M_i = np.array([[m]])
                M_total += Jr.T @ M_i @ Jr
                f_total += Jr.T @ M_i @ a

        M_total = M_total + 1e-8 * np.eye(n)  # numerical safety only (attractor already p.d.)
        q_dd_f = np.linalg.solve(M_total, f_total)
        q_dd_f = np.clip(q_dd_f, -FABRIC_QDD_CAP, FABRIC_QDD_CAP)
        return q_dd_f

    def barrier(self, data):
        """Worst (min-h) pair over the corrected multi-keypoint proxy. Only
        the 2 keypoints of that argmin pair need their Jacobians (cheap);
        the search itself (over ~625 position-only pairs) has no mj_jac
        calls."""
        h, dist, eid, iid, n, eoff, ioff = min_pair_distance(self.model, data)
        Je = point_jac_at(self.model, data, eid, eoff)[:, self.left_dofadr]      # 3x6
        Ji = point_jac_at(self.model, data, iid, ioff)[:, self.right_dofadr]     # 3x6
        v_ego = Je @ data.qvel[self.left_dofadr]
        v_int = Ji @ data.qvel[self.right_dofadr]
        hdot = float(n @ (v_ego - v_int))
        return dict(h=h, dist=dist, eid=eid, iid=iid, n=n, Je=Je, Ji=Ji,
                    v_ego=v_ego, v_int=v_int, hdot=hdot)
