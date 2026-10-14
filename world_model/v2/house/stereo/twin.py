"""stereo/twin.py -- the textured PERCEPTION TWIN of a teleop world.  SIMULATION ONLY.

The teleop model has flat colours (no textures), so block matching has nothing to match.  The twin is the SAME model (same bodies, joints, mocap, equalities:
state layout asserted equal to the sim's) compiled from a spec whose visual geoms carry a procedural speckle texture, plus the two ZED cameras.  It is built by
running teleop.scene.build_rig unchanged with one hook: sim_common._patch_real_collision_geoms (called by build_rig on the FULL spec right before
compile) is temporarily wrapped to texturise the spec and add the cameras.  No file of teleop/ or zed_demo/ is edited.

tex modes
  'speckle'  every visual geom textured (the best case; what zed_demo / the flagship perception twin do)
  'objects'  everything textured EXCEPT walls and floors (room_*, wall_*): textured furniture / robot / people against textureless walls
  'none'     flat colours as in the sim: the honest textureless case (stereo then sees only shading gradients + sensor noise)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import mujoco

from .cfg import StereoCfg, head_axes

OUT = Path(os.environ.get("STEREO_OUT", "/data/local/ss5772/molmospaces/percept/stereo"))
FLAT_PREFIX = ("wall_", "room_")


def _walk(body, fn):
    for g in body.geoms:
        fn(body, g)
    for c in body.bodies:
        _walk(c, fn)


def texturize(spec: mujoco.MjSpec, mode: str, tex_dir: Path):
    """give the spec's visual geoms a speckle texture according to `mode` (see module doc)."""
    if mode == "none":
        return
    from world_model.v2.house.zed_demo import rig as ZR
    ZR.apply_speckle(spec, Path(tex_dir))
    _person_texture(spec)
    if mode == "objects":
        flat = {}

        def fix(b, g):
            if g.group in (3, 4, 5):
                return
            if not (b.name or "").startswith(FLAT_PREFIX):
                return
            old = g.material
            key = old or "_white"
            if key not in flat:
                rgba = [1, 1, 1, 1]
                if old:
                    rgba = list(spec.material(old).rgba)
                nm = f"flat_{len(flat)}"
                spec.add_material(name=nm, rgba=rgba)           # no texture -> flat colour (shaded by the lights only)
                flat[key] = nm
            g.material = flat[key]
        _walk(spec.worldbody, fix)


def _person_texture(spec: mujoco.MjSpec):
    """people are capsules / spheres: a 2-D image texture maps to vertical STRIPES on a capsule (no horizontal structure -> ambiguous matching along the rows).
    Use a random-dot CUBE texture (box-mapped in the geom frame) on the person geoms instead: real clothing / skin has texture in both directions."""
    tex = spec.add_texture(name="zed_person_dots", type=mujoco.mjtTexture.mjTEXTURE_CUBE, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                           mark=mujoco.mjtMark.mjMARK_RANDOM, rgb1=[0.35, 0.35, 0.35], rgb2=[0.35, 0.35, 0.35], markrgb=[1, 1, 1], random=0.35, width=96, height=96)
    mat = spec.add_material(name="zed_person_mat", rgba=[1, 1, 1, 1])
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "zed_person_dots"

    def fix(b, g):
        if (b.name or "").startswith("p_") and g.group not in (3, 4, 5):
            g.material = "zed_person_mat"
    _walk(spec.worldbody, fix)


def add_stereo_cams(spec: mujoco.MjSpec, cfg: StereoCfg):
    """two fixed world cameras (poses are overwritten each frame from the robot root; only the intrinsics matter here)."""
    for nm in ("zed_left", "zed_right"):
        spec.worldbody.add_camera(name=nm, pos=[0, 0, 0], fovy=cfg.fovy)


def build_twin_model(sim, cfg: StereoCfg | None = None) -> mujoco.MjModel:
    """compile the perception twin of a running TeleopSim (same rig arguments), assert the state layout equals the sim's."""
    import sim_common
    from world_model.v2.house.teleop import scene as SC
    cfg = cfg or StereoCfg()
    tex_dir = OUT / "tex" / f"p{os.getpid()}"
    tex_dir.mkdir(parents=True, exist_ok=True)
    orig = sim_common._patch_real_collision_geoms

    def hooked(spec):
        orig(spec)
        texturize(spec, cfg.tex, tex_dir)
        add_stereo_cams(spec, cfg)

    m0 = sim.m
    has = lambda n: mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, n) >= 0
    info = sim.rig.info
    counter_at = desk_at = None
    # the slabs are placed at the hold pose of the base pose they were built for: recover it from the body names (TeleopSim passes pick / wp lab_desk_place)
    wp = sim.scene.waypoints
    if has("tp_counter"):
        counter_at = wp["cafe_counter_pick"]
    if has("tp_desk"):
        desk_at = wp["lab_desk_place"]
    sim_common._patch_real_collision_geoms = hooked
    try:
        rig = SC.build_rig(sim.scene, info["spawn"][:2], info["yaw"], dt=info["dt"], with_people=sim.with_people, counter_at=counter_at, desk_at=desk_at,
                           cup=info["has_cup"], dynamic_base=info["dynamic_base"], finger_contact=info["finger_contact"])
    finally:
        sim_common._patch_real_collision_geoms = orig
    m = rig.m
    for k in ("nq", "nv", "nmocap", "neq", "nbody", "ngeom"):
        assert getattr(m, k) == getattr(m0, k), (k, getattr(m, k), getattr(m0, k))
    return m


def build_house_twin(scene, cfg: StereoCfg | None = None) -> mujoco.MjModel:
    """house-only static twin (no robot) for the mapping pass: the camera pose is set freely by the caller."""
    from world_model.v2.house import build as B
    cfg = cfg or StereoCfg()
    tex_dir = OUT / "tex" / f"p{os.getpid()}"
    tex_dir.mkdir(parents=True, exist_ok=True)
    spec = scene.spec_loader() if scene.spec_loader else B.load_house_spec(scene.house, True)
    texturize(spec, cfg.tex, tex_dir)
    add_stereo_cams(spec, cfg)
    return spec.compile()


def save_model(m: mujoco.MjModel, path) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 720
    mujoco.mj_saveModel(m, str(path), None)
    return str(path)
