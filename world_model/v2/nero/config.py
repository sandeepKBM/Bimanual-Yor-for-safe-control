"""config.py -- NERO constants: provenance, official joint limits, and the (ASSUMED) torque-limit config.

Everything not read from the official URDF/docs is tagged ASSUMED here and nowhere else (one place to change).
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / "assets_nero"
URDF_ARM = ASSETS / "nero" / "urdf" / "nero_description.urdf"               # world->base_link->link1..7 (joint1..7)
URDF_FLANGE = ASSETS / "nero" / "urdf" / "nero_with_flange_description.xacro"   # link7->flange_link (fixed)
URDF_GRIPPER = ASSETS / "nero" / "urdf" / "nero_with_gripper_description.xacro"  # flange->gripper_base + 2 prismatic fingers
MESH_DIR = {"nero": ASSETS / "nero" / "meshes", "agx_gripper": ASSETS / "agx_gripper" / "meshes"}
SCENE_OUT = REPO / "scenes" / "yor_nero_derived.mjcf"   # old (AgileX-gripper) file name, not written any more; use scene_path("yor_nero_derived") (defined below)
ARM_OUT = ASSETS / "nero_arm.xml"

URDF_REPO = "https://github.com/agilexrobotics/agx_arm_urdf"
URDF_COMMIT = "983788b58d7eae76511177f768d85877534917fe"       # master HEAD when downloaded, 2026-10-09

N_ARM = 7
# ---- OFFICIAL joint position limits (rad), from nero_description.urdf <limit lower upper>
URDF_LIMITS = np.array([[-2.70526, 2.70526], [-1.74, 1.74], [-2.75, 2.75], [-1.01, 2.14],
                        [-2.75, 2.75], [-0.73, 0.95], [-1.5707963, 1.5707963]])
# ---- OFFICIAL pyAgxArm nero_api.md create_agx_arm_config() default joint_limits (rad); same joints, ~1e-2 rad wider
SDK_DOC_LIMITS = np.array([[-2.705261, 2.705261], [-1.745330, 1.745330], [-2.757621, 2.757621],
                           [-1.012291, 2.146755], [-2.757621, 2.757621], [-0.733039, 0.959932],
                           [-1.570797, 1.570797]])
URDF_VEL = 5.0          # rad/s in URDF <limit velocity> for all 7 joints (SolidWorks-exporter placeholder)
URDF_EFFORT = 100.0     # Nm  in URDF <limit effort> for all 7 joints (placeholder-looking; identical to the Piper URDF value)
GRIPPER_STROKE = 0.1    # m total opening (URDF 'gripper' joint 0..0.1); each finger 0..0.05
GRIPPER_EFFORT = 10.0   # N (URDF) -- forcerange written into the base arm MJCF (urdf_to_mjcf); the NYU build overrides it with NUT_FORCE_N

# ---- 2026-11-01: NERO datasheet values (reference only; the sim defaults above are NOT changed by these).
# OFFICIAL = NERO_USER_MANUAL.pdf, "Download Full Specs" on global.agilex.ai/products/nero (fetched 2026-11-01), sec. 2.1 table;
# its footnote: "test results ... in a controlled test environment".  No joint torque rating and no gripper force in it.
MANUAL = dict(
    payload_kg=3.0, mass_kg=4.8, reach_m=0.58, repeat_mm=0.1, max_power_W=150.0,
    # max joint speed [deg/s]; URDF_VEL 5 rad/s (286 deg/s) is a placeholder above these
    joint_speed_deg_s=(180.0, 180.0, 180.0, 225.0, 225.0, 225.0, 225.0),
    # joint ranges [deg]; within ~3 deg of URDF_LIMITS except J2, whose manual zero differs (-15..190 vs URDF +-99.7, both ~205 deg span)
    joint_range_deg=((-157, 157), (-15, 190), (-160, 160), (-60, 125), (-160, 160), (-43, 58), (-90, 90)),
)
# ---- 2026-11-03: the pan/pot scenes run the NYU gripper (nut force NUT_FORCE_N below) with 1 ms physics inside each 5 ms control step.
PHYSICS_SUBSTEPS = 5    # MODEL choice: 1 ms physics inside each 5 ms control step, torque held. Needed above 10 N (5 ms made the lid walk,
                        #    results/contact_model_2026-11-01/claw40/RESULT.txt). Control rate stays 200 Hz.
# (companion switch, defined with tau_limit() below: BIMANUAL_LEGACY_TAU=1 = the 100 Nm FL-MPC torque box of every pre-11-03 hold/pan/democ run)


def claw_default():
    """pan-scene gripper actuator force limit [N]: the NUT force, NUT_FORCE_N (231 N MODEL)."""
    return NUT_FORCE_N


def substeps_default():
    """physics substeps per 5 ms control step in the pan scenes: PHYSICS_SUBSTEPS (the nyu jaw/nut servo oscillates at 5 ms, nyu_gripper.py: always 1 ms)."""
    return PHYSICS_SUBSTEPS


# ---- torque limits: NOT published as motor specs anywhere official.  Presets (one config):
TORQUE_PRESETS = {
    # official URDF <limit effort>; looks like an exporter default (same 100 as the Piper URDF) -> ASSUMED upper bound
    "urdf_effort": np.full(7, 100.0),
    # official pyAgxArm docs/nero/firmware_reference.md (re-read 2026-11-01): MIT-mode t_ff COMMAND range, fw <= 1.10 (8-bit).  A command encoding
    # range, not a motor rating -- but it is the only per-joint number AgileX publishes.
    "mit_fw_le_1.10": np.array([24.0, 24.0, 16.0, 16.0, 8.0, 8.0, 8.0]),
    # same doc, fw >= 1.11 (12-bit), all joints
    "mit_fw_ge_1.11": np.full(7, 16.0),
}
TORQUE_PRESET = "urdf_effort"      # ASSUMED default; change here (or pass tau_limits=) -- the one torque config
TORQUE_STATUS = "ASSUMED (no official rated/peak motor torque found; see TORQUE_PRESETS for the official-but-indirect numbers)"


def tau_limits(preset: str | None = None) -> np.ndarray:
    return TORQUE_PRESETS[preset or TORQUE_PRESET].copy()


# ---- 2026-11-03 F3: ONE torque limit for every NERO runner that builds FLMPC (tau_max_vec / torque-box rows / saturation reports).
# tau_limits() above is unchanged and still feeds the MJCF actuator forcerange only (servos are disabled in the FL-MPC runners, so
# forcerange never acts on the applied qfrc).  FL-MPC runners call tau_limit() instead.
TAU_MIT_NM = 16.0       # ASSUMED: = NERO MIT t_ff command range +-16 Nm (fw >= 1.11); AgileX publishes no rated joint torque
LEGACY_TAU = os.environ.get("BIMANUAL_LEGACY_TAU") == "1"   # 1 = every pre-11-03 pan/democ/hold run: the 100 Nm URDF-effort placeholder in FL-MPC


def tau_limit(preset: str | None = None) -> np.ndarray:
    """per-joint torque limit [Nm] (7,) given to FLMPC(tau_max_vec=...): 16 Nm (ASSUMED, MIT command range) by default, the
    100 Nm placeholder under BIMANUAL_LEGACY_TAU=1; an explicit `preset` name still selects that TORQUE_PRESETS entry."""
    if preset is not None:
        return TORQUE_PRESETS[preset].copy()
    return np.full(N_ARM, 100.0 if LEGACY_TAU else TAU_MIT_NM)


# ---- servo/passive params: ALL ASSUMED (the URDF has none).  Piper-class values in yor_robot_derived: frictionloss 0.3,
# armature 0.005, no damping.  NERO inherits those; damping small.
ASSUMED = dict(
    armature=0.005, damping=0.5, frictionloss=0.3,
    kp=(200.0, 200.0, 200.0, 100.0, 100.0, 100.0, 100.0), kv=(20.0, 20.0, 20.0, 10.0, 10.0, 10.0, 10.0),
    finger_kp=200.0, finger_damping=5.0, finger_frictionloss=0.05,
    # mount: NERO base_link frame == Piper link0 frame (same pos/euler on Lift_Top); NERO base flange pattern unverified
    mount_left=dict(pos="-0.005 -0.0925 0.0575", euler="0.7854 0 3.14"),
    mount_right=dict(pos="-0.005 0.0925 0.0575", euler="-0.7854 0 3.14"),
    ee_grasp_z=0.120,    # m along gripper_base z: 18 mm back from the finger tips (z=0.138), inside the pad span (0.062..0.138)
)

# ---- names (variant uses the YOR prefixes so house/phase1-style code that keys on 'left_arm_' keeps working)
SIDES = ("left", "right")


def joint_names(side: str):
    return [f"{side}_arm_joint{i}" for i in range(1, N_ARM + 1)]


LEFT_ARM_JOINTS = joint_names("left")
RIGHT_ARM_JOINTS = joint_names("right")

# ---- 2026-11-03 USER: "remove it entirely, we only use the NYU gripper".  The NYU RUM/Stick-V2-style angular-jaw gripper (the YOR gripper_body mesh +
# D435i-matched wrist cameras; nyu_gripper.py, assets_nero/nyu_gripper/README.md) is the only gripper build; GRIPPER is a provenance label only.
GRIPPER = "nyu"
ARM_NYUGRIP_OUT = ASSETS / "nero_arm_nyugrip.xml"
# nyu nut force limit [N]. MODEL (owner 11-03): XL430 stall 1.4 N m x PWM cap 300/885 (USER 11-03) x lead-screw efficiency 0.48 (friction coeff 0.2 ASSUMED)
# x 2 pi / 6.2 mm lead (MEASURED from the mesh) = 231 N  (~25-44 N per jaw at the tip).
NUT_FORCE_N = 231.0
# vessel handle / tab thickness [m] (11-03 USER-approved): 0.020 ASSUMED: at the TCP (40 % of pivot-to-tip) the
# fully closed NYU jaws leave an inner gap of 22.2 mm (gripper.jaw_gap_m), so a 14 mm bar + 2 x 2 mm clearance (18 mm) cannot be straddled; 20 mm + 4 mm = 24 mm can.
HANDLE_T = 0.020
SCENE_SUFFIX = "_nyu"      # generated scene files carry this suffix (the earlier AgileX-gripper files are never rewritten)


def scene_path(name: str) -> "Path":
    """scenes/<name>_nyu.mjcf."""
    return REPO / "scenes" / f"{name}{SCENE_SUFFIX}.mjcf"


# ---- 2026-11-03 sim-fix list (items 5, 6, 8, 9): each behaviour change has a legacy switch that reproduces the pre-fix setup.
# item 5: arm-arm contacts.  The 'nero_collision' geoms ship contype 0 / conaffinity 1, so in the contents / pan scenes (scenes._base, pan.scene.build_spec;
# NOT adapters.compose_scene / democ.scene, which patch bits themselves) the LEFT and RIGHT arm never touched (0 collidable pairs).
LEGACY_ARMCOL = os.environ.get("BIMANUAL_LEGACY_ARMCOL") == "1"   # 1 = no arm-arm contacts (every result before 11-03)
ARM_ARM_BIT = 64        # MODEL: a bit unused by every other geom (checked with the collision matrix); LEFT arm contype |= it, RIGHT arm conaffinity |= it
                        #   -> exactly left<->right pairs collide; same-arm pairs and arm<->world/object pairs are unchanged.

# item 6: ONE physics default for every carry entry point.
LEGACY_CARRY_PHYS = os.environ.get("BIMANUAL_LEGACY_CARRY_PHYS") == "1"   # 1 = CarryEpisode defaults fric_comp=None, weld_solref=None (MuJoCo default 0.02 s weld)
WELD_STIFF = (0.004, 1.0)   # ASSUMED (11-02 tilt sweep): stiff weld solref so the "rigid" grasp bends < 0.5 deg


def carry_physics() -> dict:
    """the carry-episode physics defaults: fric_comp=True (joint friction compensation, carry.FRIC_COMP) + weld_solref=WELD_STIFF; the legacy pair
    (None, None) under BIMANUAL_LEGACY_CARRY_PHYS=1."""
    if LEGACY_CARRY_PHYS:
        return dict(fric_comp=None, weld_solref=None)
    return dict(fric_comp=True, weld_solref=WELD_STIFF)


# item 8: the coop controller's wrench source (joint-torque estimate vs the sim-only weld constraint wrench).
LEGACY_COOP_SENSOR = os.environ.get("BIMANUAL_LEGACY_COOP_SENSOR") == "1"   # 1 = the sim-only constraint wrench (no real counterpart)
COOP_TAU_NOISE_NM = 0.0         # ASSUMED: joint-torque feedback noise std [Nm] for the F_hat estimate (0 = noise-free; no NERO joint-torque noise spec published)
COOP_TAU_NOISE_NM_LATER = 0.05  # ASSUMED placeholder for later runs (NOT used by default)
# item 9: coop friction-compensation velocity term.
LEGACY_COOP_FRIC = os.environ.get("BIMANUAL_LEGACY_COOP_FRIC") == "1"   # 1 = the pre-fix coop friction term
LEGACY_COOP_FC = os.environ.get("BIMANUAL_LEGACY_COOP_FC") == "1"       # 1 = keep the friction wrapper's joint kp/ki under ctl=coopimp (pre-11-03; USER 11-03 dropped them)
LEGACY_COOP_SQUEEZE = os.environ.get("BIMANUAL_LEGACY_COOP_SQUEEZE") == "1"   # 1 = the old 5 N ASSUMED coopimp squeeze target (USER 11-03: 0 N)


def coop_f_sq():
    """coopimp squeeze target [N]: 0 (USER 11-03) or the old 5 N ASSUMED under BIMANUAL_LEGACY_COOP_SQUEEZE=1."""
    return 5.0 if LEGACY_COOP_SQUEEZE else 0.0
