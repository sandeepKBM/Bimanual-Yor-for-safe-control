"""Domain randomization for the bimanual sim (WM/HJ generalization now,
sim2real later).

Scope + design (mirrors the seeded-family pattern in
collection/perturbations.py, read-only reference -- not imported here):
  - One entry point, `randomize(spec_or_model, rng, config)`, dispatched by
    input type:
      * `mujoco.MjSpec`  -> STRUCTURAL edits only (currently: `scene`
        family's distractor objects, which must be added before compile()).
        Returns the same spec, mutated in place. Caller must then call
        `spec.compile()` to obtain an `MjModel`.
      * `mujoco.MjModel` -> NUMERIC in-place edits on the compiled model's
        arrays (`visual`, `dynamics`, and the `scene` family's object-pose /
        table-pose / ball-count jitter, via `model.qpos0` and
        `model.body_pos`). Returns the same model, mutated in place. Caller
        must reset data from the model (`mujoco.mj_resetData` +
        `mujoco.mj_forward`, or `MjData(model)` fresh) to see the effect,
        since `qpos0` only takes effect on reset.
  - Every family draws from an independent child `random.Random` spawned
    from the caller-supplied `rng`, seeded by one integer drawn from `rng`
    itself. This decouples families' internal draw sequences from each
    other, so toggling one family's `enabled` flag on/off does not change
    the numbers drawn by the others (same reasoning as
    collection/perturbations.py's per-family seeding, generalized to
    "call order also independent of enable flags").
  - This module is READ-ONLY with respect to the rest of the repo: it
    never edits scene XML files or `collection/`. It only mutates
    in-memory MjSpec/MjModel objects handed to it by the caller.

Typical call site (see randomization/README.md for the full collection +
eval recipe):

    from randomization.domain_randomization import RandomizationConfig, randomize
    import random, mujoco

    config = RandomizationConfig()  # defaults, or override per-family ranges
    spec = mujoco.MjSpec.from_file("scenes/scene_yor_cup.xml")
    rng = random.Random(episode_seed)
    randomize(spec, rng, config)          # structural (distractors)
    model = spec.compile()
    randomize(model, rng, config)         # numeric (visual/dynamics/scene)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)      # picks up randomized qpos0
    mujoco.mj_forward(model, data)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, pi, sin
from random import Random
from typing import Optional

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "randomization.domain_randomization requires the 'mujoco' package "
        "(run under the safe_env conda env)."
    ) from exc


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass
class VisualConfig:
    """Lighting / material / camera / sensor-noise randomization ranges."""

    enabled: bool = True

    # Light (scene has 1 light; loop covers however many the scene defines).
    light_pos_jitter_m: float = 0.4          # +/- per-axis uniform jitter
    light_intensity_scale_range: tuple[float, float] = (0.5, 1.6)  # diffuse+specular scale
    light_color_jitter: float = 0.15         # per-channel additive, clipped to [0, 1]

    # Materials (table/cup/balls/robot) + bare geom rgba (geoms with no material).
    material_color_jitter: float = 0.20      # per-channel additive on rgb, clipped [0, 1]
    material_reflectance_range: tuple[float, float] = (0.0, 0.4)

    # Small camera-pose jitter (all cameras in the compiled model, including
    # the ZED-standin wrist cams left/right_arm_egocentric).
    camera_pos_jitter_m: float = 0.01
    camera_angle_jitter_deg: float = 2.0

    # ZED depth/RGB noise hooks -- applied post-render by the caller via
    # `apply_sensor_noise`, not by `randomize()` itself (no rendered image
    # exists at randomize()-call time).
    rgb_noise_std_8bit: float = 6.0          # gaussian std, 0-255 scale
    depth_noise_std_m: float = 0.003         # gaussian std, meters
    depth_dropout_prob: float = 0.01         # fraction of pixels zeroed (missing-return sim)


@dataclass
class SceneConfig:
    """Cup/ball/table pose + distractor-object randomization ranges."""

    enabled: bool = True

    # Cup + balls are jittered TOGETHER (same dx/dy) so they stay stacked
    # on the table regardless of table jitter -- see module docstring.
    cup_xy_radius_m: float = 0.03            # radial offset, within reachable shared region
    cup_yaw_jitter_deg: float = 15.0         # rotate cup (and its balls) about world z
    ball_pos_jitter_m: float = 0.004         # small independent per-ball xyz jitter (on top of the shared offset)

    # Ball "count": scaffold technique -- park a random subset of balls far
    # below the floor (out of camera/contact range) instead of deleting
    # them, since removing a body is a structural (spec-level) edit and the
    # cup/ball scene's ball bodies are fixed in scene_yor_cup.xml. See
    # randomization/README.md for the follow-up if true count-varying is
    # needed (would add spec-level geom deletion).
    ball_count_range: tuple[int, int] = (4, 8)   # inclusive; scene has 8 balls total
    ball_park_z_m: float = -1.0

    # Table pose jitter (small; the whole table body + cup/balls are shifted
    # together by the same dx/dy so nothing falls off the edge).
    table_xy_jitter_m: float = 0.015

    # Distractor objects -- STRUCTURAL, added only in the MjSpec branch.
    max_distractors: int = 2
    distractor_size_range_m: tuple[float, float] = (0.01, 0.025)
    distractor_xy_radius_m: float = 0.12     # scattered around the cup, on the table


@dataclass
class DynamicsConfig:
    """Friction / mass / damping / actuator-gain randomization ranges (all multiplicative)."""

    enabled: bool = True

    friction_scale_range: tuple[float, float] = (0.7, 1.3)   # applied to all 3 geom_friction cols
    mass_scale_range: tuple[float, float] = (0.8, 1.2)       # body_mass, skips massless/world bodies
    damping_scale_range: tuple[float, float] = (0.8, 1.2)    # dof_damping
    gain_scale_range: tuple[float, float] = (0.85, 1.15)     # actuator kp (gainprm[0], biasprm[1])


@dataclass
class RandomizationConfig:
    visual: VisualConfig = field(default_factory=VisualConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _child_rng(rng: Random) -> Random:
    """Spawn an independent child RNG seeded by one draw from `rng`."""
    return Random(rng.randrange(2**31))


def _sample_xy_offset(rng: Random, radius: float) -> tuple[float, float]:
    """Uniform-in-disc xy offset (same recipe as collection/perturbations.py)."""
    if radius <= 0.0:
        return 0.0, 0.0
    angle = rng.uniform(0.0, 2.0 * pi)
    magnitude = rng.uniform(0.0, radius)
    return magnitude * cos(angle), magnitude * sin(angle)


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


# --------------------------------------------------------------------------
# MjSpec branch (structural edits, pre-compile)
# --------------------------------------------------------------------------


def _randomize_spec_distractors(spec: "mujoco.MjSpec", rng: Random, cfg: SceneConfig) -> None:
    if cfg.max_distractors <= 0:
        return
    n = rng.randint(0, cfg.max_distractors)
    for i in range(n):
        dx, dy = _sample_xy_offset(rng, cfg.distractor_xy_radius_m)
        # Cup sits at (0.42, -0.21, 0.55) with rim at z=0.63 in scene_yor_cup.xml;
        # place distractors directly on the table top, offset from the cup.
        pos = [0.42 + dx, -0.21 + dy, 0.545]
        size = rng.uniform(*cfg.distractor_size_range_m)
        rgba = [rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), 1.0]
        shape = rng.choice(["box", "sphere", "cylinder"])
        geom_type = {
            "box": mujoco.mjtGeom.mjGEOM_BOX,
            "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
            "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
        }[shape]
        geom_size = [size, size, size] if shape != "sphere" else [size, 0, 0]
        body = spec.worldbody.add_body(name=f"dr_distractor_{i}", pos=pos)
        body.add_geom(
            name=f"dr_distractor_{i}_geom",
            type=geom_type,
            size=geom_size,
            rgba=rgba,
            contype=1,
            conaffinity=1,
        )


def _randomize_spec(spec: "mujoco.MjSpec", rng: Random, config: RandomizationConfig) -> "mujoco.MjSpec":
    if config.scene.enabled:
        _randomize_spec_distractors(spec, _child_rng(rng), config.scene)
    # visual/dynamics families need a compiled MjModel's flat arrays; they
    # are applied in the MjModel branch below (after the caller compiles).
    return spec


# --------------------------------------------------------------------------
# MjModel branch (numeric edits, post-compile)
# --------------------------------------------------------------------------


def _randomize_visual(model: "mujoco.MjModel", rng: Random, cfg: VisualConfig) -> None:
    if not cfg.enabled:
        return

    for i in range(model.nlight):
        jitter = np.array([rng.uniform(-cfg.light_pos_jitter_m, cfg.light_pos_jitter_m) for _ in range(3)])
        model.light_pos[i] = model.light_pos[i] + jitter
        scale = rng.uniform(*cfg.light_intensity_scale_range)
        color_jit = np.array([rng.uniform(-cfg.light_color_jitter, cfg.light_color_jitter) for _ in range(3)])
        model.light_diffuse[i] = _clip01(model.light_diffuse[i] * scale + color_jit)
        model.light_specular[i] = _clip01(model.light_specular[i] * scale)

    for i in range(model.nmat):
        jit = np.array([rng.uniform(-cfg.material_color_jitter, cfg.material_color_jitter) for _ in range(3)])
        model.mat_rgba[i, :3] = _clip01(model.mat_rgba[i, :3] + jit)
        model.mat_reflectance[i] = rng.uniform(*cfg.material_reflectance_range)

    # Bare geoms (no material assigned, geom_matid == -1) still carry their
    # own rgba -- jitter those too so un-materialed geoms vary as well.
    for i in range(model.ngeom):
        if model.geom_matid[i] == -1:
            jit = np.array([rng.uniform(-cfg.material_color_jitter, cfg.material_color_jitter) for _ in range(3)])
            model.geom_rgba[i, :3] = _clip01(model.geom_rgba[i, :3] + jit)

    for i in range(model.ncam):
        pos_jit = np.array([rng.uniform(-cfg.camera_pos_jitter_m, cfg.camera_pos_jitter_m) for _ in range(3)])
        model.cam_pos[i] = model.cam_pos[i] + pos_jit
        # Small random-axis rotation composed onto the existing cam_quat.
        axis = np.array([rng.uniform(-1, 1) for _ in range(3)])
        norm = np.linalg.norm(axis)
        if norm > 1e-8:
            axis = axis / norm
            angle = np.deg2rad(rng.uniform(-cfg.camera_angle_jitter_deg, cfg.camera_angle_jitter_deg))
            dq = np.zeros(4)
            mujoco.mju_axisAngle2Quat(dq, axis, angle)
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, model.cam_quat[i], dq)
            model.cam_quat[i] = q


def _randomize_dynamics(model: "mujoco.MjModel", rng: Random, cfg: DynamicsConfig) -> None:
    if not cfg.enabled:
        return

    for i in range(model.ngeom):
        scale = rng.uniform(*cfg.friction_scale_range)
        model.geom_friction[i] = model.geom_friction[i] * scale

    for i in range(model.nbody):
        if model.body_mass[i] <= 0.0:
            continue
        model.body_mass[i] *= rng.uniform(*cfg.mass_scale_range)

    for i in range(model.nv):
        model.dof_damping[i] *= rng.uniform(*cfg.damping_scale_range)

    for i in range(model.nu):
        scale = rng.uniform(*cfg.gain_scale_range)
        model.actuator_gainprm[i, 0] *= scale
        # Standard MuJoCo position actuator: biasprm[1] == -gainprm[0] (kp).
        # Scale in lockstep to keep that identity (sign-preserving, since
        # biasprm[1] is already negative).
        model.actuator_biasprm[i, 1] *= scale


def _joint_qpos_slice(model: "mujoco.MjModel", joint_name: str) -> Optional[slice]:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid == -1:
        return None
    adr = model.jnt_qposadr[jid]
    return slice(adr, adr + 7)  # freejoint: [x, y, z, qw, qx, qy, qz]


def _randomize_scene_poses(model: "mujoco.MjModel", rng: Random, cfg: SceneConfig) -> None:
    if not cfg.enabled:
        return

    cup_slice = _joint_qpos_slice(model, "cup_free")
    ball_slices = [_joint_qpos_slice(model, f"ball_{i}_free") for i in range(8)]
    table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    if cup_slice is None or table_id == -1:
        return  # scene doesn't match scene_yor_cup.xml's layout; no-op

    # Shared xy offset applied to table + cup + all balls, so relative
    # layout (cup centered on table, balls inside cup) is preserved.
    shared_dx, shared_dy = _sample_xy_offset(rng, cfg.table_xy_jitter_m)
    model.body_pos[table_id, 0] += shared_dx
    model.body_pos[table_id, 1] += shared_dy

    # Cup gets an additional independent offset (within the reachable
    # shared region) plus a yaw rotation about world z.
    cup_dx, cup_dy = _sample_xy_offset(rng, cfg.cup_xy_radius_m)
    yaw = np.deg2rad(rng.uniform(-cfg.cup_yaw_jitter_deg, cfg.cup_yaw_jitter_deg))
    cup_quat = np.array([cos(yaw / 2), 0.0, 0.0, sin(yaw / 2)])

    total_dx = shared_dx + cup_dx
    total_dy = shared_dy + cup_dy
    model.qpos0[cup_slice.start] += total_dx
    model.qpos0[cup_slice.start + 1] += total_dy
    model.qpos0[cup_slice.start + 3:cup_slice.start + 7] = cup_quat

    # Ball count: park a random subset far below the floor.
    lo, hi = cfg.ball_count_range
    n_active = rng.randint(int(lo), int(hi))
    n_active = max(0, min(8, n_active))
    order = list(range(8))
    rng.shuffle(order)
    parked = set(order[n_active:])

    for i, sl in enumerate(ball_slices):
        if sl is None:
            continue
        jit_dx, jit_dy = _sample_xy_offset(rng, cfg.ball_pos_jitter_m)
        model.qpos0[sl.start] += total_dx + jit_dx
        model.qpos0[sl.start + 1] += total_dy + jit_dy
        if i in parked:
            model.qpos0[sl.start + 2] = cfg.ball_park_z_m + i * 0.05  # spread so parked balls don't overlap each other


def _randomize_model(model: "mujoco.MjModel", rng: Random, config: RandomizationConfig) -> "mujoco.MjModel":
    _randomize_visual(model, _child_rng(rng), config.visual)
    _randomize_dynamics(model, _child_rng(rng), config.dynamics)
    _randomize_scene_poses(model, _child_rng(rng), config.scene)
    return model


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def randomize(spec_or_model, rng: Random, config: Optional[RandomizationConfig] = None):
    """Apply domain randomization in place; returns the same object.

    `spec_or_model` is either an `mujoco.MjSpec` (structural edits only --
    caller must `.compile()` afterwards) or an `mujoco.MjModel` (numeric
    edits -- caller must reset+forward `MjData` afterwards to see effect).
    `rng` is a `random.Random` (NOT numpy) seeded per-episode by the caller,
    e.g. `random.Random(episode_seed)`.
    """
    if config is None:
        config = RandomizationConfig()

    if isinstance(spec_or_model, mujoco.MjSpec):
        return _randomize_spec(spec_or_model, rng, config)
    if isinstance(spec_or_model, mujoco.MjModel):
        return _randomize_model(spec_or_model, rng, config)
    raise TypeError(
        f"randomize() expects mujoco.MjSpec or mujoco.MjModel, got {type(spec_or_model)!r}"
    )


def apply_sensor_noise(
    rgb: Optional[np.ndarray],
    depth: Optional[np.ndarray],
    rng: Random,
    config: Optional[RandomizationConfig] = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """ZED-style post-render noise hook. Call after rendering, not inside
    `randomize()` (no image exists at randomize()-call time).

    `rgb`: uint8 HxWx3 or None. `depth`: float32 HxW (meters) or None.
    Returns (rgb_noisy, depth_noisy), same dtypes/shapes as input.
    """
    if config is None:
        config = RandomizationConfig()
    cfg = config.visual
    if not cfg.enabled:
        return rgb, depth

    out_rgb = rgb
    if rgb is not None:
        noise = np.array(
            [rng.gauss(0.0, cfg.rgb_noise_std_8bit) for _ in range(rgb.size)]
        ).reshape(rgb.shape)
        out_rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    out_depth = depth
    if depth is not None:
        noise = np.array(
            [rng.gauss(0.0, cfg.depth_noise_std_m) for _ in range(depth.size)]
        ).reshape(depth.shape)
        out_depth = depth.astype(np.float32) + noise
        dropout_mask = np.array(
            [rng.random() < cfg.depth_dropout_prob for _ in range(depth.size)]
        ).reshape(depth.shape)
        out_depth = np.where(dropout_mask, 0.0, out_depth)

    return out_rgb, out_depth


__all__ = [
    "VisualConfig",
    "SceneConfig",
    "DynamicsConfig",
    "RandomizationConfig",
    "randomize",
    "apply_sensor_noise",
]
