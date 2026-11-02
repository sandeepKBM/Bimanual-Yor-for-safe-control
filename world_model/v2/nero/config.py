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
SCENE_OUT = REPO / "scenes" / "yor_nero_derived.mjcf"
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
GRIPPER_EFFORT = 10.0   # N (URDF) -- the sim claw limit (MJCF forcerange); see GRIPPER_DATASHEET (4-5x higher)

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
# Gripper: RESELLER pages only (mybotshop.de AgileX-Nero-Gripper, generationrobots.com PiPER/NERO gripper, openelab.io), 2026-11-01:
# rated 40 N, max 50 N clamping, 0-70 mm opening (one page also says 100 mm), 500 g, rubber fingers.  The manual says a PiPER gripper
# works on NERO after an ID change, so the two grippers are separate but compatible products.  NOT confirmed by AgileX.
GRIPPER_DATASHEET = dict(rated_N=40.0, max_N=50.0, stroke_m=0.07,
                         source="reseller pages; 40 N rating confirmed by the user 2026-11-01 (relayed via peer session)")
# Named claw force limits (USER 11-01: 40 N variant approved).  "urdf" = the MJCF forcerange every result before 11-01 used, and stays
# the default; the others raise the gripper actuator forcerange IN MEMORY (pan.handle_grasp.set_claw_limit), the MJCF is not edited.
CLAW_PRESETS = {"urdf": GRIPPER_EFFORT, "nero_rated": 40.0, "nero_max": 50.0}   # N; nero_* = GRIPPER_DATASHEET

# ---- 2026-11-01 USER decision (relayed): 40 N rating confirmed -> PAN/POT scene sim default = 40 N claw + 1 ms physics step.
# Applied by pan.scene.apply_sim_defaults (at compose) and pan.scene.step (physics substeps); other scenes (democ, house, ...) unchanged.
# The MJCF/URDF file still says 10 N (not edited); the limit is raised in memory.
CLAW_N_DEFAULT = 40.0   # N. MEASURED (vendor datasheet rating: GRIPPER_DATASHEET rated_N, user-confirmed 11-01). A rated spec, not
                        #    our own measurement: no n / statistic / noise model; actual force vs current not measured.
PHYSICS_SUBSTEPS = 5    # MODEL choice: 1 ms physics inside each 5 ms control step, torque held. Needed above 10 N (5 ms made the lid walk,
                        #    results/contact_model_2026-11-01/claw40/RESULT.txt). Control rate stays 200 Hz.
LEGACY_CLAW = os.environ.get("BIMANUAL_LEGACY_CLAW") == "1"   # 1 = the setup of every result before 11-01 (10 N, 5 ms physics)


def claw_default():
    """pan-scene claw force limit [N]: 40 (default) or the MJCF 10 N under BIMANUAL_LEGACY_CLAW=1."""
    return GRIPPER_EFFORT if LEGACY_CLAW else CLAW_N_DEFAULT


def substeps_default():
    """physics substeps per 5 ms control step in the pan scenes: 5 (1 ms) or 1 (5 ms) under BIMANUAL_LEGACY_CLAW=1."""
    return 1 if LEGACY_CLAW else PHYSICS_SUBSTEPS

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
