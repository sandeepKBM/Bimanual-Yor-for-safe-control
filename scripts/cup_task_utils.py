"""
Utilities for scenes/scene_bimanual_cup.xml (the "don't-spill-the-cup" task).

Provides:
  - build_cup_spec(xml_path) / load_cup_model(xml_path): load the scene and attach
    the wrist camera ("right/wrist_cam") to body "right/gripper_base" at load time
    via mujoco.MjSpec, WITHOUT editing scenes/bimanual_piper_pickup.xml (that file
    is treated as an existing/derived file we must not touch -- only new filenames
    are written for this task). The world-fixed "head_cam" is declared directly in
    scene_bimanual_cup.xml since it does not need to attach to an existing body.
  - count_spilled(model, data): ground-truth spill detector used to label episodes
    for the world model.

Cup geometry constants below must match scenes/scene_bimanual_cup.xml (see that
file's header comment and scenes/CUP_TASK.md for the derivation).
"""

import sys
from pathlib import Path

import numpy as np
import mujoco

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SIM_COMMON_DIR = _REPO_ROOT / "experiments" / "ab_smooth_vs_bangbang"
if str(_SIM_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_COMMON_DIR))

CUP_BODY_NAME = "cup"
CUP_RADIUS = 0.030   # wall centerline radius (m); interior clear radius is slightly less
CUP_HEIGHT = 0.080   # cup height: bottom (local z=0) to rim (local z=CUP_HEIGHT)
BALL_PREFIX = "ball_"
N_BALLS_DEFAULT = 8

WRIST_CAM_BODY = "right/gripper_base"
WRIST_CAM_NAME = "right/wrist_cam"
# gripper_base's local +Z axis points toward the fingers (link7/link8 attach at
# pos="0 0 0.1358" on this body). Mounting the camera near the fingertip level
# (z=0.12) and rotating it 180 deg about Y (so its view axis -Z_cam aligns with
# +Z of gripper_base, i.e. "forward" past the fingers) plus a ~23 deg downward
# tilt about X gives a clear forward/down view of the workspace without the
# frame being filled by the gripper's own mesh (verified empirically -- see
# scenes/CUP_TASK.md for the candidate sweep that ruled out euler=(0,-pi/2,0),
# which stares back into the gripper body instead).
WRIST_CAM_POS = (0.0, 0.0, 0.12)
WRIST_CAM_EULER = (0.4, 3.14159, 0.0)
WRIST_CAM_FOVY = 70.0
HEAD_CAM_FOVY = 45.0


def build_cup_spec(xml_path):
    """Parse scene_bimanual_cup.xml into an editable mujoco.MjSpec and attach the
    wrist camera to body WRIST_CAM_BODY. Does not touch the XML file on disk."""
    spec = mujoco.MjSpec.from_file(xml_path)
    gripper_body = spec.body(WRIST_CAM_BODY)
    gripper_body.add_camera(
        name=WRIST_CAM_NAME,
        pos=list(WRIST_CAM_POS),
        euler=list(WRIST_CAM_EULER),
        fovy=WRIST_CAM_FOVY,
    )
    return spec


def load_cup_model(xml_path):
    """Return (model, data) for scene_bimanual_cup.xml with the wrist camera attached."""
    spec = build_cup_spec(xml_path)
    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data


def count_spilled(model, data, cup_radius=CUP_RADIUS, cup_height=CUP_HEIGHT,
                   cup_body_name=CUP_BODY_NAME, ball_prefix=BALL_PREFIX,
                   height_margin=0.002):
    """Ground-truth spill detector.

    A ball is counted as "spilled" if its position, expressed in the cup body's
    own bottom-centered local frame (translated by the cup's world position and
    rotated by the cup's current world orientation -- so this stays correct even
    while the cup is tilted), falls outside the upright cylinder approximating
    the cup's interior:
        radial = hypot(local_x, local_y)   (distance from the cup's central axis)
        height = local_z
        spilled if radial > cup_radius OR height < -height_margin OR height > cup_height

    i.e. "outside the cup interior" == beyond the cup's radius from its axis, OR
    below the cup's own bottom, OR past the rim height along the cup's axis.
    Balls resting on the bottom have radial < cup_radius and height roughly in
    [0.005, 0.03] (well under cup_height=0.08), so resting balls are never
    falsely flagged; balls that roll/fall out over the rim end up with
    radial > cup_radius (and/or height < 0 once they land on the table, which is
    below the cup's own bottom plane) and are correctly flagged as spilled.

    Returns: int, number of spilled balls.
    """
    cup_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cup_body_name)
    if cup_bid < 0:
        raise ValueError(f"body '{cup_body_name}' not found in model")
    cup_pos = data.xpos[cup_bid].copy()
    cup_mat = data.xmat[cup_bid].reshape(3, 3).copy()

    spilled = 0
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not name or not name.startswith(ball_prefix):
            continue
        ball_pos = data.xpos[bid]
        local = cup_mat.T @ (ball_pos - cup_pos)
        radial = float(np.hypot(local[0], local[1]))
        height = float(local[2])
        if radial > cup_radius or height < -height_margin or height > cup_height:
            spilled += 1
    return spilled


YOR_CUP_XML = str(_REPO_ROOT / "scenes" / "scene_yor_cup.xml")


def load_yor_cup_model(xml_path=YOR_CUP_XML):
    """Load scenes/scene_yor_cup.xml (cup+balls on the PROPER YOR robot),
    with the arm-arm collision contype/conaffinity fix applied IN-MEMORY, plus
    the real ZED workspace camera (`zed_cam`) attached at load time.

    yor_robot_derived.mjcf ships every robot link's real collision geom with
    contype=0 (see experiments/ab_smooth_vs_bangbang/sim_common.py's
    `_patch_real_collision_geoms` docstring for the full root-cause writeup):
    two contype=0 geoms never contact each other, so as shipped the two arms
    are structurally incapable of colliding with EACH OTHER (they still
    collide fine with the table/cup/balls, which use ordinary contype=1
    defaults -- only arm-vs-arm needs the patch). collect_cup.py's
    "collision_intrusion" behavior needs real left/right arm contacts to
    label `collision`, so this loader imports and reuses sim_common's exact
    patch function (not a re-implementation) rather than editing
    yor_robot_derived.mjcf or scene_yor_cup.xml on disk.

    `zed_cam` (scripts/zed_camera.py's `add_zed_camera`) is the real ZED
    workspace camera -- mounted on `Lift_Top` (torso), framing the tabletop +
    both arms -- and is this pipeline's world-model `image` source (see
    collection/CUP_DATA_V2.md). Added here, in-memory, via the same MjSpec
    the collision patch already uses, so scene_yor_cup.xml is never edited on
    disk for this either."""
    import sim_common  # noqa: E402  (experiments/ab_smooth_vs_bangbang, read-only import)
    from scripts.zed_camera import add_zed_camera  # noqa: E402

    spec = mujoco.MjSpec.from_file(xml_path)
    add_zed_camera(spec)
    sim_common._patch_real_collision_geoms(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data


def ball_body_ids(model, ball_prefix=BALL_PREFIX):
    ids = []
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if name and name.startswith(ball_prefix):
            ids.append(bid)
    return ids
