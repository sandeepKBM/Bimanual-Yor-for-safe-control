"""Utilities for scenes/scene_yor_stack.xml (the bimanual STACKING task).

Provides:
  - load_yor_stack_model(xml_path): load the scene with the real ZED workspace
    camera (scripts.zed_camera.add_zed_camera) attached + the arm-arm
    collision contype/conaffinity fix applied, both IN-MEMORY via
    mujoco.MjSpec (same pattern as scripts/cup_task_utils.py's
    load_yor_cup_model -- scene_yor_stack.xml and yor_robot_derived.mjcf are
    never edited on disk).
  - tower_height / count_toppled / block_dropped / collision: ground-truth
    label functions for the WM-visible failure mode (a topple of the shared
    central tower), plus is_stacked_success for the task-success predicate.

See scenes/STACK_TASK.md for the full scene/task writeup.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import mujoco

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SIM_COMMON_DIR = _REPO_ROOT / "experiments" / "ab_smooth_vs_bangbang"
if str(_SIM_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_COMMON_DIR))

YOR_STACK_XML = str(_REPO_ROOT / "scenes" / "scene_yor_stack.xml")

# --- Scene constants (must match scenes/scene_yor_stack.xml) ---------------
BLOCK_HALF = 0.02
TABLE_TOP_Z = 0.55
TOWER_XY = (0.42, -0.21)          # tower_zone_marker center, table center
TOWER_RADIUS = 0.06               # horizontal radius counted as "in the tower"
TOPPLE_RADIUS = 0.12              # wider radius catching blocks knocked away
TOPPLE_TILT_DEG = 35.0            # local z-axis vs world z beyond this = "toppled"
DROP_Z = 0.30                     # below this z = fell off table/platform onto floor
LEVEL_TOL = 0.018                 # z-tolerance for "resting at stack level i"
UPRIGHT_COS_MIN = 0.85            # cos(~32deg): min uprightness to count as "stacked"

BLOCK_NAMES = (
    "block_l1", "block_l2", "block_l3",   # left-only (right arm cannot reach)
    "block_r1", "block_r2", "block_r3",   # right-only (left arm cannot reach)
    "block_s1", "block_s2",               # shared (either arm can reach)
)
BLOCK_SIDE = {
    "block_l1": "left", "block_l2": "left", "block_l3": "left",
    "block_r1": "right", "block_r2": "right", "block_r3": "right",
    "block_s1": "shared", "block_s2": "shared",
}


def load_yor_stack_model(xml_path: str = YOR_STACK_XML):
    """Return (model, data) for scenes/scene_yor_stack.xml with the arm-arm
    collision fix applied in-memory (yor_robot_derived.mjcf ships every real
    collision geom with contype=0, i.e. the two arms are structurally
    incapable of colliding with EACH OTHER as shipped -- see
    experiments/ab_smooth_vs_bangbang/sim_common.py's
    `_patch_real_collision_geoms` docstring) plus the real ZED workspace
    camera (`zed_cam`, scripts/zed_camera.py's `add_zed_camera`) attached,
    mounted on Lift_Top, framing the tabletop + both platforms + both arms.
    Both are applied via the same MjSpec, reused (not re-implemented) from
    the modules that already own them -- scene_yor_stack.xml is never edited
    on disk."""
    import sim_common  # noqa: E402  (experiments/ab_smooth_vs_bangbang, read-only import)
    from scripts.zed_camera import add_zed_camera  # noqa: E402

    spec = mujoco.MjSpec.from_file(xml_path)
    add_zed_camera(spec)
    sim_common._patch_real_collision_geoms(spec)
    _restore_finger_object_collision(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data


_FINGER_BODIES = (
    "left_arm_link7", "left_arm_link8", "right_arm_link7", "right_arm_link8",
)


def _restore_finger_object_collision(spec) -> None:
    """Compensate a side-effect of `sim_common._patch_real_collision_geoms`
    (reused, unmodified, above -- needed so left/right ARMS can collide with
    each other for the `collision` label): that patch rewrites contype/
    conaffinity for EVERY geom under any left_arm_*/right_arm_* body that
    shipped as (contype=0, conaffinity=1) -- which includes the finger pad
    geoms (left_arm_link7/8, right_arm_link7/8 -- same "group=3 collision
    class" as the arm-arm barrier proxies), not just the arm-link collision
    proxies it's meant for. Post-patch, a finger pad's (contype, conaffinity)
    becomes e.g. (4, 8) (left) -- which no longer intersects a block's
    ordinary (contype=1, conaffinity=1), i.e. the fingers become physically
    incapable of grasping anything (verified empirically: perfect finger-pad/
    block alignment, zero contacts, until this fix). OR-ing bit 0 back into
    both fields on just the finger geoms restores their original
    everything-collides behavior (so blocks are graspable again) while
    leaving the OR'd-in left/right bits intact (so arm-arm contact via the
    fingers, e.g. during collision_bump, still also registers)."""
    for b in spec.bodies:
        if b.name in _FINGER_BODIES:
            for g in b.geoms:
                g.contype |= 1
                g.conaffinity |= 1


def _block_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _upright_cos(xmat_flat: np.ndarray) -> float:
    """cos(angle) between the body's own local +Z axis (xmat's 3rd column)
    and world +Z -- 1.0 = perfectly upright, 0.0 = lying on its side."""
    return float(xmat_flat.reshape(3, 3)[2, 2])


def tower_height(model: mujoco.MjModel, data: mujoco.MjData, xy=TOWER_XY,
                  radius: float = TOWER_RADIUS, table_z: float = TABLE_TOP_Z,
                  block_half: float = BLOCK_HALF, level_tol: float = LEVEL_TOL,
                  upright_cos_min: float = UPRIGHT_COS_MIN) -> int:
    """Number of blocks forming a CONTIGUOUS, upright stack centered at `xy`,
    counted from the table surface up. A block is a stack candidate if its
    horizontal distance from `xy` is < radius and it is upright (local
    z-axis within ~32deg of world z, i.e. not toppled/on its side). Sorted by
    height, a candidate counts toward level i only if its center height
    matches the expected resting height for a clean stack
    (table_z + block_half + i*2*block_half) within `level_tol`; the count
    stops at the first gap/mismatch (so a knocked-loose block above a good
    stack does not inflate the count)."""
    candidates = []
    for name in BLOCK_NAMES:
        bid = _block_id(model, name)
        if bid < 0:
            continue
        pos = data.xpos[bid]
        horiz = float(np.hypot(pos[0] - xy[0], pos[1] - xy[1]))
        if horiz < radius and _upright_cos(data.xmat[bid]) > upright_cos_min:
            candidates.append(float(pos[2]))
    candidates.sort()

    height = 0
    expected = table_z + block_half
    for z in candidates:
        if abs(z - expected) < level_tol:
            height += 1
            expected += 2.0 * block_half
        else:
            break
    return height


def count_toppled(model: mujoco.MjModel, data: mujoco.MjData, xy=TOWER_XY,
                   radius: float = TOPPLE_RADIUS, table_z: float = TABLE_TOP_Z,
                   tilt_deg: float = TOPPLE_TILT_DEG) -> int:
    """Number of blocks NEAR the tower zone (within the wider `radius`, so it
    also catches blocks knocked sideways off the exact center) that are
    either tilted over (local z-axis more than `tilt_deg` from world z -- on
    their side, not standing) or knocked off the table entirely (center below
    the table surface). This is the WM-visible TOWER FAILURE signal: a clean
    stack has count_toppled==0; a bumped/misplaced/collided stack has
    count_toppled>0."""
    cos_thresh = math.cos(math.radians(tilt_deg))
    count = 0
    for name in BLOCK_NAMES:
        bid = _block_id(model, name)
        if bid < 0:
            continue
        pos = data.xpos[bid]
        horiz = float(np.hypot(pos[0] - xy[0], pos[1] - xy[1]))
        if horiz < radius:
            fell_over = _upright_cos(data.xmat[bid]) < cos_thresh
            knocked_off = pos[2] < table_z - 0.03
            if fell_over or knocked_off:
                count += 1
    return count


def block_dropped(model: mujoco.MjModel, data: mujoco.MjData, drop_z: float = DROP_Z) -> bool:
    """True if ANY block has fallen off its table/platform toward the floor
    (center height below `drop_z`, well under the table top at 0.55m) --
    e.g. pushed off the table edge during a bimanual mishap."""
    for name in BLOCK_NAMES:
        bid = _block_id(model, name)
        if bid < 0:
            continue
        if float(data.xpos[bid][2]) < drop_z:
            return True
    return False


def collision(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """Ground-truth left/right ARM-ARM contact this step (faithful proxy,
    same body-name-prefix convention as collection/collect_cup.py's
    `_left_right_contact` / teleop/record_teleop.py's copy -- geom names on
    this robot don't uniformly carry the side prefix, but every BODY does)."""
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.body(model.geom_bodyid[c.geom1]).name
        b2 = model.body(model.geom_bodyid[c.geom2]).name
        s1 = "left" if b1.startswith("left_arm_") else ("right" if b1.startswith("right_arm_") else None)
        s2 = "left" if b2.startswith("left_arm_") else ("right" if b2.startswith("right_arm_") else None)
        if {s1, s2} == {"left", "right"}:
            return True
    return False


def is_stacked_success(model: mujoco.MjModel, data: mujoco.MjData, k: int = 3) -> bool:
    """Task success: a tower of at least `k` blocks standing, none toppled."""
    return tower_height(model, data) >= k and count_toppled(model, data) == 0
