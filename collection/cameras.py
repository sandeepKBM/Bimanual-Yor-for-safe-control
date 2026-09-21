"""Camera injection/selection for the bimanual scenes (pickup + cup).

== Pickup scene (scene_bimanual_pickup.xml / bimanual_piper_pickup.xml, both
read-only, untouched by this module) ==
Defines no <camera> elements, so MuJoCo 3.x's MjSpec API is used to add two
cameras programmatically at load time, then compile. This keeps the checked-in
scene files pristine.

Two cameras (fixed, "FIXED" cam mode -- i.e. rigidly attached to the parent
frame, not tracking/targeting):
  - "head_cam":        attached to worldbody, static overhead/oblique view of
                        the table + both arms.
  - "right/wrist_cam":  attached to body "right/gripper_base", moves with the
                        right end-effector (eye-in-hand view).

Only right/wrist_cam is wired up per the task spec ("RGB from head_cam &
right/wrist_cam"). A symmetric "left/wrist_cam" would be a one-line addition
(same add_camera call on "left/gripper_base") if bimanual eye-in-hand on both
arms is needed later.

== Cup scene on the PROPER YOR robot (scene_yor_cup.xml) ==
Cup+balls on scenes/scene_yor_cup.xml (the welded-base/lift YOR robot, see
scenes/YOR_CANONICAL.md), superseding the SIMPLIFIED-rig cup scene below.
Per the sim2real decision (collection/CUP_DATA_V2.md): the world model's
`image` observation comes from the real ZED workspace camera
(`zed_cam`, scripts/zed_camera.py::add_zed_camera, mounted on Lift_Top --
frames the tabletop + both arms, sees spills), NOT a synthetic external
"close-up" cam like the old scene's cup_cam_a/cup_cam_b (not physically
realizable on hardware). `world_model/dataset_adapter.py` is owned by a
separate in-flight task and is intentionally NOT edited here -- it
unconditionally builds `image = concat([rgb_head, rgb_wrist_right],
axis=-1)`, so `YorCupCameraRenderer` stores the SAME zed_cam frame into BOTH
of those field names (image ends up as the ZED view duplicated across both
halves, not two different cameras) rather than changing the adapter's
contract. The two REAL egocentric wrist cams
(`left_arm_egocentric`/`right_arm_egocentric`, yor_robot_derived.mjcf,
unmodified) are recorded separately as auxiliary data
(`rgb_wrist_left_ego`/`rgb_wrist_right_ego`), NOT part of `image`.
`build_yor_cup_model()` delegates to `scripts.cup_task_utils.
load_yor_cup_model` (single source of truth for the scene + the in-memory
arm-arm collision contype fix + the zed_cam attachment).

== Cup scene on the SIMPLIFIED rig (scene_bimanual_cup.xml) == (superseded)
UNLIKE the pickup scene, the cup scene's world-model cameras ("cup_cam_a",
"cup_cam_b" -- fixed, close-in, angled so the ball level inside the cup is
actually resolvable, see scenes/scene_bimanual_cup.xml's camera comments and
collection/CUP_DATA.md) are declared DIRECTLY in the XML, not injected here
via MjSpec -- there is no duplicate camera definition to reconcile: this
module's `build_cup_model()` just calls `scripts.cup_task_utils.load_cup_model`
(the single source of truth for what the cup scene's compiled model contains,
including the MjSpec-injected "right/wrist_cam" kept for compatibility/future
eye-in-hand use) and `CupDualCameraRenderer` renders cup_cam_a/cup_cam_b from
whatever that returns. If cup_task_utils.py ever changes which cameras exist,
this module does not need a matching edit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SCENE_PICKUP = "/common/users/ss5772/bimanual/scenes/scene_bimanual_pickup.xml"
SCENE_CUP = "/common/users/ss5772/bimanual/scenes/scene_bimanual_cup.xml"
SCENE_YOR_CUP = "/common/users/ss5772/bimanual/scenes/scene_yor_cup.xml"

CUP_CAM_A = "cup_cam_a"
CUP_CAM_B = "cup_cam_b"

LEFT_EGOCENTRIC = "left_arm_egocentric"
RIGHT_EGOCENTRIC = "right_arm_egocentric"

HEAD_CAM_NAME = "head_cam"
RIGHT_WRIST_CAM_NAME = "right/wrist_cam"

# Overhead-oblique view of the table/box area from world +X, looking back
# toward the origin and down. Hand-picked for the pickup scene's table height
# (~0.19m per bimanual_piper_pickup.xml resting pose) and arm base spacing
# (+/-0.25m in Y) -- good enough for smoke-testing the plumbing; not claimed
# to be a tuned "nice" render.
_HEAD_CAM_POS = (0.9, 0.0, 0.85)
_HEAD_CAM_QUAT = (0.6532815, 0.2705981, 0.2705981, 0.6532815)  # ~-50deg pitch, facing -X

_WRIST_CAM_POS = (0.0, 0.0, 0.06)
_WRIST_CAM_QUAT = (1.0, 0.0, 0.0, 0.0)  # camera -Z along gripper_base -Z (down the approach axis)


def build_model(scene_path: str = SCENE_PICKUP) -> mujoco.MjModel:
    """Load `scene_path` via MjSpec, inject head_cam + right/wrist_cam, compile."""
    spec = mujoco.MjSpec.from_file(scene_path)

    spec.worldbody.add_camera(
        name=HEAD_CAM_NAME,
        pos=list(_HEAD_CAM_POS),
        quat=list(_HEAD_CAM_QUAT),
        mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED,
    )

    right_gripper = spec.body("right/gripper_base")
    right_gripper.add_camera(
        name=RIGHT_WRIST_CAM_NAME,
        pos=list(_WRIST_CAM_POS),
        quat=list(_WRIST_CAM_QUAT),
        mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED,
    )

    model = spec.compile()
    assert model.camera(HEAD_CAM_NAME).id >= 0
    assert model.camera(RIGHT_WRIST_CAM_NAME).id >= 0
    return model


class DualCameraRenderer:
    """Thin wrapper around mujoco.Renderer for the two cameras above."""

    def __init__(self, model: mujoco.MjModel, height: int = 84, width: int = 84):
        self.model = model
        self.height = height
        self.width = width
        self._renderer = mujoco.Renderer(model, height=height, width=width)

    def render(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        out = {}
        for key, cam_name in (("rgb_head", HEAD_CAM_NAME), ("rgb_wrist_right", RIGHT_WRIST_CAM_NAME)):
            self._renderer.update_scene(data, camera=cam_name)
            out[key] = self._renderer.render().copy()  # (H, W, 3) uint8
        return out

    def close(self) -> None:
        self._renderer.close()


def build_cup_model(scene_path: str = SCENE_CUP) -> mujoco.MjModel:
    """Load `scene_path` (the cup scene) via cup_task_utils.load_cup_model --
    the single source of truth for that scene's cameras (cup_cam_a/cup_cam_b
    declared in the XML + the MjSpec-injected right/wrist_cam) -- and assert
    the two world-model-facing cameras are present."""
    from scripts.cup_task_utils import load_cup_model  # noqa: E402 (local import: avoid a hard dep on scripts/ for pickup-only callers)

    model, _ = load_cup_model(scene_path)
    assert model.camera(CUP_CAM_A).id >= 0
    assert model.camera(CUP_CAM_B).id >= 0
    return model


def build_yor_cup_model(scene_path: str = SCENE_YOR_CUP) -> mujoco.MjModel:
    """Load `scene_path` (the cup scene on the PROPER YOR robot) via
    cup_task_utils.load_yor_cup_model -- the single source of truth for that
    scene's compiled model, including the in-memory arm-arm collision
    contype/conaffinity fix (see that function's docstring) -- and assert the
    two real egocentric wrist cams are present."""
    from scripts.cup_task_utils import load_yor_cup_model  # noqa: E402 (local import, mirrors build_cup_model)

    model, _ = load_yor_cup_model(scene_path)
    assert model.camera(LEFT_EGOCENTRIC).id >= 0
    assert model.camera(RIGHT_EGOCENTRIC).id >= 0
    assert model.camera(ZED_CAM).id >= 0
    return model


ZED_CAM = "zed_cam"


class YorCupCameraRenderer:
    """Renders the PROPER-YOR cup scene's cameras. IMPORTANT constraint:
    world_model/dataset_adapter.py is owned by another in-flight task and is
    NOT edited here -- it unconditionally builds `image = concat([rgb_head,
    rgb_wrist_right], axis=-1)`. So to make the real ZED workspace camera
    (`zed_cam`, scripts/zed_camera.py, mounted on Lift_Top -- frames the
    tabletop + both arms, sees spills) the world model's actual image source
    WITHOUT touching the adapter, this renderer stores the SAME zed_cam frame
    into BOTH `rgb_head` and `rgb_wrist_right` (the two field names the
    adapter already reads) -- so `image` ends up as the ZED view duplicated
    across both 3-channel halves, not two different cameras. Also stored:
      - `rgb_zed`: the same ZED frame again, under its own key, for direct
        inspection/debugging (not read by the adapter).
      - `rgb_wrist_left_ego` / `rgb_wrist_right_ego`: the two REAL egocentric
        wrist cams (left_arm_egocentric/right_arm_egocentric) -- auxiliary
        data only, NOT part of `image` and not read by the adapter today.
    All of these are EXTRA keys beyond episode.py's required STEP_ARRAY_KEYS
    except rgb_head/rgb_wrist_right themselves (see EpisodeWriter.save's
    docstring for how extras are persisted) -- the box-pickup task's
    collector (collect.py) is completely unaffected."""

    def __init__(self, model: mujoco.MjModel, height: int = 64, width: int = 64):
        self.model = model
        self.height = height
        self.width = width
        self._renderer = mujoco.Renderer(model, height=height, width=width)

    def render(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        self._renderer.update_scene(data, camera=ZED_CAM)
        zed_img = self._renderer.render().copy()  # (H, W, 3) uint8

        out = {
            "rgb_head": zed_img,             # ZED frame -- adapter's image[..., :3]
            "rgb_wrist_right": zed_img.copy(),  # ZED frame again -- adapter's image[..., 3:6]
            "rgb_zed": zed_img.copy(),          # same frame, dedicated key (debug/inspection)
        }
        for key, cam_name in (
            ("rgb_wrist_left_ego", LEFT_EGOCENTRIC),
            ("rgb_wrist_right_ego", RIGHT_EGOCENTRIC),
        ):
            self._renderer.update_scene(data, camera=cam_name)
            out[key] = self._renderer.render().copy()
        return out

    def close(self) -> None:
        self._renderer.close()


class CupDualCameraRenderer:
    """Renders the cup scene's two fixed world-model cameras (cup_cam_a,
    cup_cam_b). Reuses episode.py's existing `rgb_head`/`rgb_wrist_right`
    array keys (repurposed field names for the cup task -- see
    scenes/scene_bimanual_cup.xml's camera comments and
    collection/CUP_DATA.md) so world_model/dataset_adapter.py's
    2-cam -> 6-channel `image` contract needs zero changes."""

    def __init__(self, model: mujoco.MjModel, height: int = 64, width: int = 64):
        self.model = model
        self.height = height
        self.width = width
        self._renderer = mujoco.Renderer(model, height=height, width=width)

    def render(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        out = {}
        for key, cam_name in (("rgb_head", CUP_CAM_A), ("rgb_wrist_right", CUP_CAM_B)):
            self._renderer.update_scene(data, camera=cam_name)
            out[key] = self._renderer.render().copy()  # (H, W, 3) uint8
        return out

    def close(self) -> None:
        self._renderer.close()
