"""Utilities for scenes/scene_yor_cup_collect.xml (the flagship bimanual
"cube-collection into a held cup" scene).

Provides:
  - load_yor_cup_collect_model(xml_path): load the scene with the real ZED
    workspace camera (scripts.zed_camera.add_zed_camera) attached + the
    arm-arm collision contype/conaffinity fix applied, both IN-MEMORY via
    mujoco.MjSpec -- EXACT same load pattern as
    scripts/cup_task_utils.py's load_yor_cup_model (this scene's cup+balls
    are a verbatim copy of that one's, so the same loader logic applies
    unchanged; also restores finger/object collision the way
    scripts/stack_task_utils.py's load_yor_stack_model does, since this
    scene's whole point is a RIGHT-arm gripper picking up the new cubes --
    see `_restore_finger_object_collision`'s docstring below for why that
    extra step is needed on top of the cup loader's patch).
  - cube_body_ids(model): body ids of the 6 scattered cubes (cube_0..cube_5).
  - count_deposited_cubes(model, data): ground-truth "cube is inside the
    held cup" detector, structured exactly like scripts/cup_task_utils.py's
    count_spilled but INVERTED (inside-the-interior counts, not
    outside-the-interior).
  - count_spilled is re-exported (unchanged) from scripts.cup_task_utils for
    convenience -- this module does not redefine ball/spill logic.

This file is UTILITIES ONLY -- no controller/sequencer/task-policy logic
lives here (that is a separate, later build). scenes/scene_yor_cup_collect.xml,
scenes/scene_yor_cup.xml, and scenes/yor_robot_derived.mjcf are never edited
on disk by anything in this module.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mujoco

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SIM_COMMON_DIR = _REPO_ROOT / "experiments" / "ab_smooth_vs_bangbang"
if str(_SIM_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_COMMON_DIR))

from scripts.cup_task_utils import (  # noqa: E402  (re-exported, unchanged)
    CUP_BODY_NAME,
    CUP_RADIUS,
    CUP_HEIGHT,
    BALL_PREFIX,
    N_BALLS_DEFAULT,
    count_spilled,
    ball_body_ids,
)

YOR_CUP_COLLECT_XML = str(_REPO_ROOT / "scenes" / "scene_yor_cup_collect.xml")

CUBE_PREFIX = "cube_"
N_CUBES_DEFAULT = 6
CUBE_HALF = 0.0105  # must match scenes/scene_yor_cup_collect.xml's cube geom size

CUBE_WELD_EQ_NAME = "cube_weld"  # name of the (initially-inactive) mjEQ_WELD
                                  # equality added by add_cube_weld/load_yor_cup_collect_model
                                  # below -- look it up post-compile via
                                  # mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, CUBE_WELD_EQ_NAME).

TABLE_TOP_GEOM_NAME = "table_top"  # scenes/scene_yor_cup_collect.xml's table surface geom (axis-aligned
                                    # box; world-top-z = geom pos.z + geom size.z) -- used at MjSpec (pre-
                                    # compile) time by add_keep_zone_marker/load_yor_cup_collect_model below
                                    # to place the keep-zone marker disk exactly on the table surface.

KEEP_ZONE_MARKER_NAME = "keep_zone_marker"  # name of the visual (non-colliding) reference-circle geom
                                             # added by add_keep_zone_marker below -- look it up post-compile
                                             # via mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, KEEP_ZONE_MARKER_NAME).
KEEP_ZONE_RADIUS_DEFAULT_M = 0.10  # fallback keep-zone radius [m] when a caller of
                                    # load_yor_cup_collect_model doesn't override it. make_cup_collect_
                                    # safety_demo.py defines its OWN KEEP_ZONE_RADIUS_M and passes it in
                                    # explicitly end-to-end (both here and into count_placed_cubes's
                                    # `radius`), so this default only matters for other/ad-hoc callers.


_FINGER_BODIES = (
    "left_arm_link7", "left_arm_link8", "right_arm_link7", "right_arm_link8",
)


def _restore_finger_object_collision(spec) -> None:
    """Compensate a side-effect of sim_common._patch_real_collision_geoms
    (reused, unmodified, in load_yor_cup_collect_model below): that patch
    rewrites contype/conaffinity for EVERY geom under any
    left_arm_*/right_arm_* body that shipped as (contype=0, conaffinity=1)
    -- which includes the finger pad geoms (left_arm_link7/8,
    right_arm_link7/8), not just the arm-arm collision proxies it's meant
    for. Post-patch, a finger pad's (contype, conaffinity) no longer
    intersects a cube's ordinary (contype=1, conaffinity=1), i.e. the
    fingers become physically incapable of grasping anything. OR-ing bit 0
    back into both fields on just the finger geoms restores their original
    everything-collides behavior (cubes stay graspable) while leaving the
    OR'd-in left/right bits intact (arm-arm contact via the fingers still
    also registers). Identical fix to
    scripts/stack_task_utils.py's `_restore_finger_object_collision` --
    reproduced here (not imported) to keep this module self-contained and
    because scripts/cup_task_utils.py's load_yor_cup_model does not need it
    (the cup task's grasp is a kinematic reattach, not a real finger-contact
    grasp -- see eval/scene_setup.py's `reattach_cup_rigidly` docstring) but
    THIS scene's cube-collection task is built around a real gripper pinch,
    so the fix is required here."""
    for b in spec.bodies:
        if b.name in _FINGER_BODIES:
            for g in b.geoms:
                # ONLY the tuned finger-pad collision geoms (group==3), NOT the
                # visual mesh geom (group 2, ships contype=0/conaffinity=0). OR-ing
                # bit 0 onto the visual mesh made it a universal collider whose big
                # convex hull flung the cup's balls out during the grasp-close and
                # produced false arm-arm collisions in the crossing metric.
                if g.group == 3:
                    g.contype |= 1
                    g.conaffinity |= 1


def add_cube_weld(spec, ee_body_name: str, cube_body_name: str,
                   eq_name: str = CUBE_WELD_EQ_NAME) -> None:
    """Add an mjEQ_WELD equality constraint between `ee_body_name` and
    `cube_body_name` to `spec` (an MjSpec, not yet compiled), starting
    INACTIVE (active=0) -- a "magnet" attach: the physical connection is a
    real constraint force once activated (see `activate_weld` below), not a
    kinematic teleport/pin. Must be called BEFORE `spec.compile()`."""
    spec.add_equality(name=eq_name, type=mujoco.mjtEq.mjEQ_WELD,
                       objtype=mujoco.mjtObj.mjOBJ_BODY,
                       name1=ee_body_name, name2=cube_body_name, active=0)


def _weld_quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _weld_quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def activate_weld(model, data, eq_id: int, body1_id: int, body2_id: int) -> None:
    """Activate a previously-added, currently-INACTIVE mjEQ_WELD equality
    (`eq_id`, from add_cube_weld), "magnet"-style: captures body2's CURRENT
    pose relative to body1 as the weld's target (writing model.eq_data),
    then sets data.eq_active[eq_id]=1 -- so the weld pins whatever relative
    pose body2 happened to be in at the instant of activation (no snap/jump).

    eq_data layout for mjEQ_WELD (mjNEQDATA=11), verified empirically (see
    diff notes -- MuJoCo's XML/header docs don't spell this out explicitly):
        eq_data[0:3]  = anchor, in body2's local frame (0 => body2's own origin)
        eq_data[3:6]  = relpos: target position of that anchor, in body1's frame
        eq_data[6:10] = relquat: target orientation, body1_quat * relquat == body2_quat
        eq_data[10]   = torquescale (1.0 = full)
    i.e. body2 is held such that its origin sits at
    body1_xpos + body1_xmat @ relpos, with orientation body1_xquat * relquat.
    """
    R1 = data.xmat[body1_id].reshape(3, 3)
    relpos = R1.T @ (data.xpos[body2_id] - data.xpos[body1_id])
    relquat = _weld_quat_mul(_weld_quat_conj(data.xquat[body1_id]), data.xquat[body2_id])
    model.eq_data[eq_id, 0:3] = 0.0
    model.eq_data[eq_id, 3:6] = relpos
    model.eq_data[eq_id, 6:10] = relquat
    model.eq_data[eq_id, 10] = 1.0
    data.eq_active[eq_id] = 1


def deactivate_weld(data, eq_id: int) -> None:
    """Deactivate the weld (magnet "drop") -- the welded body is released to
    real (unconstrained) physics from whatever state it was in."""
    data.eq_active[eq_id] = 0


def add_keep_zone_marker(spec, center_xy, table_top_z: float,
                          radius: float = KEEP_ZONE_RADIUS_DEFAULT_M,
                          name: str = KEEP_ZONE_MARKER_NAME) -> None:
    """Add a thin, translucent, NON-COLLIDING disk geom (contype=0,
    conaffinity=0 -- visual only, no physics) to `spec` (an MjSpec, not yet
    compiled), centered at `center_xy` and sitting just above the table
    surface (world z = table_top_z + a fraction of a mm, to avoid z-fighting
    with table_top's own visual) -- a reference circle around the cube
    collection/"keep" zone, for the render.

    A cylinder (type=mjGEOM_CYLINDER, size=[radius, half_height]) makes a
    flat disk -- the simplest "circle on the table" primitive MuJoCo has;
    half_height is kept tiny (1mm) so it reads as a flat ring/disk rather
    than a puck. Added to `spec.worldbody` (fixed in the world, not
    attached to any moving body) since the keep zone itself never moves
    during an episode. Must be called BEFORE `spec.compile()`, same
    convention as add_cube_weld above."""
    spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[float(center_xy[0]), float(center_xy[1]), float(table_top_z) + 0.0011],
        size=[float(radius), 0.001, 0.0],
        rgba=[0.05, 0.85, 0.15, 0.30],   # translucent green
        contype=0,
        conaffinity=0,
        group=2,   # same "visual, not a tuned collider" group convention this
                   # scene already uses for non-colliding visual mesh geoms
                   # (see _restore_finger_object_collision's docstring above).
    )


def load_yor_cup_collect_model(xml_path: str = YOR_CUP_COLLECT_XML,
                               restore_finger_object_collision: bool = False,
                               weld_cube_name: str | None = None,
                               weld_cube_names: "list[str] | None" = None,
                               weld_ee_body_name: str = "right_arm_gripper_base",
                               keep_zone_marker: bool = True,
                               keep_zone_center_xy: "tuple[float, float] | None" = None,
                               keep_zone_radius_m: float = KEEP_ZONE_RADIUS_DEFAULT_M):
    """Return (model, data) for scenes/scene_yor_cup_collect.xml with:
      1. the arm-arm collision contype/conaffinity fix applied in-memory
         (sim_common._patch_real_collision_geoms, same as
         scripts/cup_task_utils.py's load_yor_cup_model), so left/right arm
         real collision geometry can contact each other;
      2. the real ZED workspace camera (`zed_cam`, scripts/zed_camera.py's
         add_zed_camera) attached, mounted on Lift_Top, framing the tabletop
         + both arms -- identical mount/pose to load_yor_cup_model;
      3. the finger/object collision restore fix (see
         `_restore_finger_object_collision` above), needed on top of (1) so
         the gripper fingers can still pinch the new cubes;
      4. IF weld_cube_name is given (single-cube, back-compat): an mjEQ_WELD
         "magnet" equality between `weld_ee_body_name` and the named cube body
         (see add_cube_weld), added but INACTIVE (data.eq_active=0) -- the
         caller activates it at runtime via activate_weld/deactivate_weld.
         Look up its id post-compile with `mujoco.mj_name2id(model,
         mujoco.mjtObj.mjOBJ_EQUALITY, cup_collect_task_utils.CUBE_WELD_EQ_NAME)`.
      5. IF weld_cube_names is given (multi-cube; takes precedence over
         weld_cube_name when both are given): one mjEQ_WELD per cube body in
         the list, each also INACTIVE, named f"{CUBE_WELD_EQ_NAME}_{i}" for
         i, name in enumerate(weld_cube_names) -- look each one up
         post-compile the same way, with that per-index eq name.
      6. IF keep_zone_marker (default True): a thin, translucent, NON-
         COLLIDING disk geom (see add_keep_zone_marker above) at world xy
         `keep_zone_center_xy` -- or, when that is None (default), the CUP
         BODY's own rest ground xy (`spec.body(CUP_BODY_NAME).pos[:2]`,
         since the cup sits directly under worldbody in this scene's XML,
         so its spec `.pos` already IS its world position) -- at the table
         surface height. This is only an approximate stand-in for the
         actual pick/place cluster center (which is computed later, from a
         compiled model + IK solve, by make_cup_collect_safety_demo.py's
         build_episode -- not available yet at this MjSpec-patch stage) but
         it sits at the same place the cluster is built around, so it is a
         reasonable visual reference regardless. Look up its geom id
         post-compile via `mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
         cup_collect_task_utils.KEEP_ZONE_MARKER_NAME)`.
    scenes/scene_yor_cup_collect.xml is never edited on disk."""
    import sim_common  # noqa: E402  (experiments/ab_smooth_vs_bangbang, read-only import)
    from scripts.zed_camera import add_zed_camera  # noqa: E402

    spec = mujoco.MjSpec.from_file(xml_path)
    add_zed_camera(spec)
    sim_common._patch_real_collision_geoms(spec)
    # Default OFF: the demo holds the cup via kinematic/rigid attach
    # (reattach_cup_rigidly), not a real finger pinch, so finger<->object
    # contact is unneeded -- and enabling it makes the closing pads disturb
    # the cup's balls during the zero-gravity grasp-init, emptying the cup
    # at t=0 (root cause of the "lift-spill"). Arm-arm collision incl. the
    # finger pads still works via _patch_real_collision_geoms regardless.
    # Only enable for a genuine finger-contact pinch (and grasp-init with it off).
    if restore_finger_object_collision:
        _restore_finger_object_collision(spec)
    if weld_cube_names is not None:
        for i, cube_name in enumerate(weld_cube_names):
            add_cube_weld(spec, weld_ee_body_name, cube_name,
                           eq_name=f"{CUBE_WELD_EQ_NAME}_{i}")
    elif weld_cube_name is not None:
        add_cube_weld(spec, weld_ee_body_name, weld_cube_name)
    if keep_zone_marker:
        table_geom = spec.geom(TABLE_TOP_GEOM_NAME)
        table_top_z = float(table_geom.pos[2] + table_geom.size[2])
        center_xy = keep_zone_center_xy
        if center_xy is None:
            cup_body_spec = spec.body(CUP_BODY_NAME)
            # match the demo's place_center offset (cup_ground + [-0.02, -0.06]) so the
            # reference circle is centered on where cubes are ACTUALLY placed, not on the
            # cup ground xy (they differ by ~6.6cm; keep in sync with build_episode's p_place).
            center_xy = (float(cup_body_spec.pos[0]) - 0.02, float(cup_body_spec.pos[1]) - 0.06)
        add_keep_zone_marker(spec, center_xy, table_top_z, radius=keep_zone_radius_m)
    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data


def cube_body_ids(model, cube_prefix: str = CUBE_PREFIX):
    """Body ids of the scattered cubes (cube_0..cube_5), in name order."""
    ids = []
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if name and name.startswith(cube_prefix):
            ids.append(bid)
    return ids


def count_deposited_cubes(model, data, cup_radius: float = CUP_RADIUS,
                           cup_height: float = CUP_HEIGHT,
                           cup_body_name: str = CUP_BODY_NAME,
                           cube_prefix: str = CUBE_PREFIX,
                           height_margin: float = 0.002) -> int:
    """Ground-truth "deposited" detector -- structured exactly like
    scripts/cup_task_utils.py's count_spilled, but INVERTED: a cube counts
    as deposited if its position, expressed in the cup body's own
    bottom-centered local frame (translated by the cup's world position and
    rotated by the cup's current world orientation -- correct even while the
    cup is held/tilted), falls INSIDE the upright cylinder approximating the
    cup's interior:
        radial = hypot(local_x, local_y)   (distance from the cup's central axis)
        height = local_z
        deposited if radial < cup_radius AND 0 <= height < cup_height

    (count_spilled's analogous ball condition is
    `radial > cup_radius or height < -height_margin or height > cup_height`;
    this is its complement, restricted to height>=0 with `height_margin`
    only relaxing the boundary at height==0 the same way count_spilled's
    lower bound does, so a cube resting exactly on the cup's bottom disc --
    height~0 -- is not spuriously excluded by float error.)

    Returns: int, number of cubes currently inside the cup interior.
    """
    cup_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cup_body_name)
    if cup_bid < 0:
        raise ValueError(f"body '{cup_body_name}' not found in model")
    cup_pos = data.xpos[cup_bid].copy()
    cup_mat = data.xmat[cup_bid].reshape(3, 3).copy()

    deposited = 0
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not name or not name.startswith(cube_prefix):
            continue
        cube_pos = data.xpos[bid]
        local = cup_mat.T @ (cube_pos - cup_pos)
        radial = float(np.hypot(local[0], local[1]))
        height = float(local[2])
        if radial < cup_radius and -height_margin <= height < cup_height:
            deposited += 1
    return deposited


def count_placed_cubes(model, data, center_xy, table_z: float,
                        cube_prefix: str = CUBE_PREFIX,
                        cube_half: float = CUBE_HALF,
                        radius: float = 0.14, z_tol: float = 0.05,
                        vel_tol: float = 0.05) -> int:
    """Ground-truth "cube placed on the table near `center_xy`" detector, for
    the multi-cube pick-and-place task (make_cup_collect_safety_demo.py's
    "crossing" scenario). A cube counts as PLACED if all three hold at the
    current `data`:
      - xy proximity: hypot(cube_x - center_x, cube_y - center_y) < radius
        (the cube is within the target drop-zone footprint, not still out at
        its original scatter position or mid-carry);
      - resting height: |cube_z - (table_z + cube_half)| < z_tol (the cube's
        CENTER sits at the table surface + its own half-extent, i.e. it is
        actually resting ON the table, not mid-air/falling or still gripped
        well above it);
      - settled: the free-joint's linear speed < vel_tol [m/s] (not still
        bouncing/sliding right after a drop).

    Cube linear velocity is read directly off data.qvel at the cube body's
    own free-joint dof address (qvel[dofadr:dofadr+3] = linear velocity in
    the WORLD frame for a MuJoCo free joint) -- cheaper and exactly as
    correct as data.cvel for this purpose, no extra kinematics needed.

    Returns: int, number of cubes currently counted as placed.
    """
    cx, cy = float(center_xy[0]), float(center_xy[1])
    placed = 0
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not name or not name.startswith(cube_prefix):
            continue
        pos = data.xpos[bid]
        if np.hypot(float(pos[0]) - cx, float(pos[1]) - cy) > radius:
            continue
        if abs(float(pos[2]) - (table_z + cube_half)) > z_tol:
            continue
        dofadr = model.body_dofadr[bid]
        lin_vel = data.qvel[dofadr:dofadr + 3]
        if float(np.linalg.norm(lin_vel)) > vel_tol:
            continue
        placed += 1
    return placed
