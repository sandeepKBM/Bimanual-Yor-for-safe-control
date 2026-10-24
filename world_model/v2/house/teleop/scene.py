"""teleop/scene.py -- scene composition for the teleoperated coffee run.  SIMULATION ONLY (offline MuJoCo; no robot code).

`build_rig(scene, ...)` composes: house (or office) static world + YOR under a MOCAP base (as phase2) with the lift joints
(copied from house/lift.py 'locked' mode; lift.py is not imported because another agent edits it) + tray (two weld equalities to
the gripper bases) + a CUP WITH BEADS standing on the tray + optional counter / desk slabs + scripted people (mocap capsules).

Why not bead_rep's cup rig: bead_rep is a cup-only rig with a KINEMATICALLY pinned cup.  Here the cup is a free body standing on
the tray (its stability under base motion is the point), so only bead_rep's cup GEOMETRY / bead layout / spill predicate
(world_model.v2.data.contents: cup body xml, layout(), Oracle.out_mask) are reused, unchanged.

Scene registry
  SCENES["h5"]     ProcTHOR val house 5, Bedroom ("lab", start) -> double door -> Bathroom ("cafeteria"), route polyline copied from
                   the phase2 plan (p2/plan_h5_a0.25_v0.3.json, 4.45 m, one door).
  SCENES["office"] HOOK for the custom office scene (house/office/ is built by another agent): needs `load_office_spec()` and
                   `office_meta()`; see `_office_scene`.  Not exercised here (the package does not exist yet).
"""
from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import mujoco

from .. import build as B
from .. import phase1 as P1            # puts experiments/ab_smooth_vs_bangbang on sys.path (sim_common, mpc_fl)

REPO = B.REPO

# ---------------------------------------------------------------------------------------------- physical parameters
MU_TRUE = 0.25            # cup-base / tray friction used in the SIM (wet paper cup on a plastic tray; both geoms -> max-combine = 0.25)
TRAY_HOLD_Z = 0.85        # tray surface height at lift 0 (build.set_hold_state tray_local z)
TRAY_LOCAL = (0.45, -0.21, TRAY_HOLD_Z)
COUNTER_TOP_Z = TRAY_HOLD_Z - 0.006     # == tray underside: the tray rests on the counter at the hold height
PERSON_R = 0.22           # person torso capsule radius [m]
H_MAX = 0.416
HANDLE_W, HANDLE_T = 0.040, 0.006   # ASSUMED tray handle strips (friction rig): width along tray x, thickness under the plate
PAD_BIT = 64              # private collision bit: finger pads <-> tray only (opt-in friction-grasp rig)
PAD_SOLIMP = (0.995, 0.9999, 0.0003, 0.5, 2.0)
FINGER_ARMATURE = 0.3   # kg, reflected drive inertia of the finger slide joints (friction-grasp rig)
FINGER_KP_GRASP = 2000.0  # N/m, finger position gain in the friction-grasp rig (stock 200)
BASE_FOLLOW_TC = 0.01     # solref time constant of the root-follows-target weld (dynamic_base)
# cup position in the TRAY frame.  Verified by the lift/perception agent: at the Phase-1 hold pose the left arm reaches over the tray; only
# tray-frame x in [-0.07,-0.02], |y|<=0.07 is >=5 mm from both arms (the tray centre is 9 mm from the left arm).  The cup is placed at the
# middle of that region, and cup-vs-arm clearance is checked explicitly in set_hold (info['cup_arm_clear_m']).
CUP_TRAY_XY = (-0.045, 0.0)


# ---------------------------------------------------------------------------------------------- scene registry
@dataclass
class PersonTrack:
    """piecewise-linear xy track: rows (t, x, y); holds the first/last point outside the time range."""
    name: str
    pts: np.ndarray
    dynamic: bool = True      # False: the person is a static body already in the scene spec (office mannequins); only geometry is used

    def at(self, t):
        p = self.pts
        x = np.interp(t, p[:, 0], p[:, 1])
        y = np.interp(t, p[:, 0], p[:, 2])
        i = int(np.clip(np.searchsorted(p[:, 0], t) - 1, 0, len(p) - 2))
        dt = p[i + 1, 0] - p[i, 0]
        if t <= p[0, 0] or t >= p[-1, 0] or dt <= 0:
            return np.array([x, y]), np.zeros(2)
        return np.array([x, y]), (p[i + 1, 1:3] - p[i, 1:3]) / dt


@dataclass
class SceneSpec:
    name: str
    house: int | None
    waypoints: dict                      # name -> (x, y, yaw); lab_start, cafe_counter_pick, lab_desk_place, door1, door2
    out_route: list                      # xy polyline lab -> cafe (operator's nominal route)
    people: list = field(default_factory=list)       # list[PersonTrack]
    occ_loader: object = None            # callable -> (Grid-like object with .res .x0 .y0 .occ .edt) or None
    spec_loader: object = None           # callable -> mujoco.MjSpec (static world) ; default = build.load_house_spec(house)
    back_route: list = None              # operator's return polyline (None: reversed out_route, driven in reverse)
    back_mode: str = "reverse"           # 'reverse' (back along the out route) | 'backout' (reverse 0.7 m, then forward on back_route)
    slab_size: tuple = (0.60, 0.50)      # tray hand-off shelf (pick counter / place desk) size [m]
    note: str = ""


def _h5():
    way = dict(lab_start=(3.1, 6.09, -2.33121118), door1=(2.13449, 5.075, -1.5708), door2=(2.13449, 4.491, -1.5708),
               cafe_counter_pick=(2.135, 2.0, -1.5698), lab_desk_place=(3.1, 6.09, -2.33121118))
    route = [[3.1, 6.09], [2.13449, 5.075], [2.13449, 4.491], [2.13449, 2.5], [2.135, 2.0]]
    people = [
        # one person crossing the route (t=2..10 s) and one standing at the route's left edge (operator ignores both)
        PersonTrack("p_walker", np.array([[0.0, 3.6, 3.8], [2.0, 3.6, 3.8], [10.0, 0.9, 3.8], [1e4, 0.9, 3.8]])),
        PersonTrack("p_stand", np.array([[0.0, 1.10, 3.5], [1e4, 1.10, 3.5]])),
    ]
    return SceneSpec("h5", 5, way, route, people,
                     note="ProcTHOR val_5; route copied from phase2 plan h5 (a=0.25, v=0.3); people/counter/desk are teleop additions")


SCENES = {"h5": _h5}


def freeze_people(scene: SceneSpec) -> SceneSpec:
    """every person stands still at the first point of its track (people option `static`).  In place on a freshly built SceneSpec (get_scene returns a new one per call)."""
    for p in scene.people:
        x, y = float(p.pts[0, 1]), float(p.pts[0, 2])
        p.pts = np.array([[0.0, x, y], [1e4, x, y]])
    return scene


def _resample(x, y, step=0.5):
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    k = np.unique(np.concatenate([np.arange(0, s[-1], step), [s[-1]]]))
    return [[float(np.interp(v, s, x)), float(np.interp(v, s, y))] for v in k]


def _office_scene():
    """Custom office scene (house/office/, built by another agent; imported read-only): lab -> door1 -> corridor -> door2 -> cafeteria.
    Waypoints / mannequins from office_meta(); occupancy from nav_grid.build_office_grid(); nominal route = the cached
    office/route_traj.npz of route.plan_mission (A: lab -> counter, B: counter back-off -> desk) or a fresh plan_mission.
    The office counter (0.90 m) / desk (0.734 m) are not at the tray hold height, so (as for the houses) a small hand-off shelf at the
    hold height is added at the stand-off pose (documented simplification; grasp is a weld activation anyway)."""
    from ..office import scene as OS, nav_grid as NG
    meta = OS.office_meta()
    wp = {k: (float(v["xy"][0]), float(v["xy"][1]), float(v["heading"])) for k, v in meta["waypoints"].items()}
    npz = Path(os.environ.get("OFFICE_OUT", "/data/local/ss5772/molmospaces/office")) / "route_traj.npz"
    if npz.exists():
        z = np.load(npz, allow_pickle=True)
        x, y, mk = z["x"], z["y"], z["marks"]
    else:
        from ..office import route as OR
        res = OR.plan_mission(meta=meta, verbose=False)
        tr = res["traj"]
        x, y, mk = tr.x, tr.y, np.array([[0, i0, i1] for (_n, i0, i1) in res["marks"]])
    out_route = _resample(x[mk[0][1]:mk[0][2]], y[mk[0][1]:mk[0][2]])
    back_route = _resample(x[mk[3][1]:mk[3][2]], y[mk[3][1]:mk[3][2]])
    people = [PersonTrack(p["name"], np.array([[0.0, p["xy"][0], p["xy"][1]], [1e4, p["xy"][0], p["xy"][1]]]), dynamic=False)
              for p in meta["persons"]]
    # one walker in the corridor, heading toward the lab, meets the robot between the doors
    people.append(PersonTrack("p_walker", np.array([[0.0, 17.0, 3.55], [18.0, 9.5, 3.55], [1e4, 9.5, 3.55]])))
    return SceneSpec("office", None, wp, out_route, people, spec_loader=OS.load_office_spec, occ_loader=NG.build_office_grid,
                     back_route=back_route, back_mode="backout", slab_size=(0.34, 0.26), note="custom office scene (house/office)")


def get_scene(name: str) -> SceneSpec:
    if name == "office":
        return _office_scene()
    return SCENES[name]()


# ---------------------------------------------------------------------------------------------- occupancy (read-only copy of nav.Grid)
@dataclass
class OccGrid:
    res: float
    x0: float
    y0: float
    occ: np.ndarray
    edt: np.ndarray

    def lookup(self, x, y):
        """EDT [m] (distance to nearest occupied cell centre) at world points; 0 outside the grid."""
        ix = np.floor((np.asarray(x) - self.x0) / self.res).astype(int)
        iy = np.floor((np.asarray(y) - self.y0) / self.res).astype(int)
        ny, nx = self.occ.shape
        oob = (ix < 0) | (ix >= nx) | (iy < 0) | (iy >= ny)
        v = self.edt[np.clip(iy, 0, ny - 1), np.clip(ix, 0, nx - 1)]
        return np.where(oob, 0.0, v)


def load_occupancy(scene: SceneSpec) -> OccGrid:
    if scene.occ_loader is not None:
        g = scene.occ_loader()
        return OccGrid(g.res, g.x0, g.y0, g.occ, g.edt)
    from ..assets import HOUSE_DATA
    from scipy import ndimage
    f = HOUSE_DATA.parent / "nav_cache" / f"occ_{scene.house}_0.010.npz"
    if not f.exists():
        from .. import nav as N                       # slow (~1 min); writes the cache
        g = N.build_occupancy(scene.house)
        return OccGrid(g.res, g.x0, g.y0, g.occ, g.edt)
    z = np.load(f)
    occ = z["occ"]
    return OccGrid(float(z["res"]), float(z["x0"]), float(z["y0"]), occ, ndimage.distance_transform_edt(~occ) * float(z["res"]))


def load_perceived_occupancy(scene: SceneSpec, occ_map=None) -> OccGrid:
    """PERCEIVED static-obstacle grid (house/loc mapping pass with the head ZED depth, in the estimated-pose frame) instead of the true house
    grid.  occ_map: None -> built once per scene with the default loc.mapper config and cached on disk; a path (npz saved by
    loc.mapper.PerceivedMap.save); a PerceivedMap; or an OccGrid.  Returns an OccGrid (same interface as load_occupancy)."""
    from ..loc import mapper as MP
    return MP.resolve_occgrid(scene, occ_map)


# ---------------------------------------------------------------------------------------------- cup with beads xml
def _cup_xml(cup_pos, contents=None):
    """cup body + beads xml (bead_rep / contents.py geometry verbatim; base friction -> MU_TRUE; materials -> rgba)."""
    from world_model.v2.data import contents as C
    c = contents if contents is not None else C.Contents(**C.NOMINAL)
    txt = C.SCENE.read_text()
    cup = C._extract_body(txt, "cup")
    cup = re.sub(r'<body name="cup" pos="[^"]+"', '<body name="cup" pos="%.5f %.5f %.5f"' % tuple(cup_pos), cup, count=1)
    cup = cup.replace('material="cup_mat"', 'rgba="0.85 0.15 0.15 0.95"')
    # ALL cup geoms touch the tray (the 12 wall boxes start at z=0 like the base disc): set every cup friction to MU_TRUE, else the
    # 0.6 of the walls wins the max-combine and the cup-tray mu is 0.6, not MU_TRUE.  (bead-wall pairs keep the bead's 0.5.)
    cup = re.sub(r'friction="0\.6 0\.01 0\.0001"', 'friction="%.3f 0.005 0.0001"' % MU_TRUE, cup)
    cup = re.sub(r'(name="cup_bottom"[^>]*?)friction="[^"]+"', r'\1friction="%.3f 0.005 0.0001"' % MU_TRUE, cup)
    s = c.cup_mass / 0.03
    cup = re.sub(r'<inertial pos="0 0 0.03" mass="0.03" diaginertia="2.5e-5 2.5e-5 1.5e-5" />',
                 f'<inertial pos="0 0 0.03" mass="{c.cup_mass:.5f}" diaginertia="{2.5e-5*s:.3e} {2.5e-5*s:.3e} {1.5e-5*s:.3e}" />', cup)
    lay = C.layout(c)
    balls = "\n".join(C._ball_xml(i, np.asarray(cup_pos) + p, c).replace('material="ball_mat"', 'rgba="0.15 0.55 0.9 1"')
                      for i, p in enumerate(lay))
    xml = f'<mujoco model="cup_beads"><worldbody>\n{cup}\n{balls}\n</worldbody></mujoco>'
    return xml, c


# ---------------------------------------------------------------------------------------------- lift patch (copy of lift.py, locked mode)
_STAGE_RANGE = (0.0, 0.208)
_TELESCOPE = dict(damping=50.0, stiffness=0.02, armature=0.01, frictionloss=1.5)


def _patch_lift(robot: mujoco.MjSpec, house: mujoco.MjSpec):
    for body, name in (("Lift_Middle", "Slider_1"), ("Lift_Top", "Slider_2")):
        robot.body(body).add_joint(name=name, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=list(_STAGE_RANGE),
                                   limited=mujoco.mjtLimited.mjLIMITED_TRUE, **_TELESCOPE)
    for i in (1, 2):
        eq = house.add_equality(name=f"lift_lock_{i}", type=mujoco.mjtEq.mjEQ_JOINT, name1=f"Slider_{i}", active=1)
        eq.data[:5] = 0.0
        eq.solref = [0.005, 1.0]


def _add_slab(house, name, xy, yaw, size_xy, top_z, thick=0.03, rgba=(0.55, 0.42, 0.30, 1)):
    """table: top slab (collides with tray/cup only) + 4 visual legs."""
    b = house.worldbody.add_body(name=name, pos=[xy[0], xy[1], 0.0], quat=B.yaw_quat(yaw).tolist())
    b.add_geom(name=f"{name}_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[size_xy[0] / 2, size_xy[1] / 2, thick / 2],
               pos=[0, 0, top_z - thick / 2], rgba=list(rgba), contype=1, conaffinity=1, friction=[0.6, 0.005, 0.0001])
    for sx in (-1, 1):
        for sy in (-1, 1):
            b.add_geom(name=f"{name}_leg{sx}{sy}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, (top_z - thick) / 2],
                       pos=[sx * (size_xy[0] / 2 - 0.03), sy * (size_xy[1] / 2 - 0.03), (top_z - thick) / 2],
                       rgba=list(rgba), contype=0, conaffinity=0)


def _add_person(house, name):
    b = house.worldbody.add_body(name=name, mocap=True, pos=[0.0, 0.0, -5.0])
    b.add_geom(name=f"{name}_torso", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[PERSON_R, 0.45, 0], fromto=[0, 0, 0.30, 0, 0, 1.45],
               rgba=[0.2, 0.5, 0.3, 1], contype=0, conaffinity=0)
    b.add_geom(name=f"{name}_head", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.11, 0, 0], pos=[0, 0, 1.66],
               rgba=[0.85, 0.7, 0.55, 1], contype=0, conaffinity=0)


# ---------------------------------------------------------------------------------------------- rig
@dataclass
class Rig:
    m: mujoco.MjModel
    d: mujoco.MjData
    info: dict
    scene: SceneSpec
    n_beads: int = 8
    person_mocap: list = field(default_factory=list)


def build_rig(scene: SceneSpec, spawn_xy, yaw, dt=0.005, with_people=True, counter_at=None, desk_at=None, cup=True,
              contents=None, dynamic_base=False, finger_contact=False, pick_props=None) -> Rig:
    """pick_props (OPT-IN, default None = scene unchanged): add the pick table + pickable props of house/pick/props.py in front of the spawn pose
    (True | prop-set name | list of prop names) and a fixed world camera `pick_head` (pose set by the pick module from the head-camera mount + lift).
    counter_at / desk_at: (x, y, yaw) base poses whose hold-pose tray position gets a table slab (pick / place spots).
    dynamic_base (OPT-IN, default off: the weld-mode rig is bit-identical): the robot root is a free body that follows the kinematic
    base target (mocap `yor_target`, mocap index 0) through a stiff weld, so the base's velocity/acceleration is visible to the contact
    solver -- required for a FRICTION grasp (a mocap root gives the solver zero base velocity: a tray held by pad friction would be
    left behind).  finger_contact (OPT-IN): the four finger pads collide with the tray (private bit PAD_BIT; they still do not collide
    with the counter slab / cup / house)."""
    import sim_common as sc
    house = scene.spec_loader() if scene.spec_loader else B.load_house_spec(scene.house, True)
    robot = mujoco.MjSpec.from_string(B.YOR_MJCF.read_text())
    from world_model.v2.testbeds.b_tray_ball import tray_env as TE
    tray_spec = mujoco.MjSpec.from_string(TE.tray_xml(TE.BallParams(mu=MU_TRUE), lip=False, home=(0.0, 0.0, 5.0)))
    if finger_contact:
        # ASSUMED handle strips (friction-grasp rig only): a 6 mm plate cannot be pinched robustly (measured: 10-20 mm creep, lost grips), so the
        # two short edges carry 6 mm thick strips UNDER the plate (plate + strip = 12 mm pinch, top surface still flat at z = 0), each 40 mm
        # wide x full edge length: the pads' footprint.  The tray top, mass (0.3 kg), inertia and the handle positions are unchanged.
        tb = tray_spec.body("tray")
        for sx in (-1, 1):
            tb.add_geom(name=f"handle_{'p' if sx > 0 else 'n'}", type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[HANDLE_W / 2, TE.HY, HANDLE_T / 2], pos=[sx * (TE.HX - HANDLE_W / 2), 0.0, -2 * TE.TRAY_T - HANDLE_T / 2],
                        friction=[MU_TRUE, 0.005, 0.0001], solref=[0.01, 1.0], condim=6, rgba=[0.55, 0.55, 0.6, 1])
    spawn_xy = np.asarray(spawn_xy, float)
    if dynamic_base:
        tgt = house.worldbody.add_body(name="yor_target", mocap=True, pos=[spawn_xy[0], spawn_xy[1], 0.0], quat=B.yaw_quat(yaw).tolist())
        root = house.worldbody.add_body(name="yor_root", pos=[spawn_xy[0], spawn_xy[1], 0.0], quat=B.yaw_quat(yaw).tolist())
        root.add_freejoint(name="yor_free")
        eqb = house.add_equality(name="base_follow", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                                 name1="yor_root", name2="yor_target", active=1)
        eqb.data[:] = 0.0
        eqb.data[6] = 1.0
        eqb.data[10] = 1.0
        eqb.solref = [BASE_FOLLOW_TC, 1.0]
        eqb.solimp = [0.999, 0.9999, 0.001, 0.5, 2.0]
    else:
        root = house.worldbody.add_body(name="yor_root", mocap=True, pos=[spawn_xy[0], spawn_xy[1], 0.0], quat=B.yaw_quat(yaw).tolist())
    _patch_lift(robot, house)
    root.add_frame().attach_body(robot.body("base"), "", "")
    house.worldbody.add_frame(pos=[spawn_xy[0], spawn_xy[1], 5.0]).attach_body(tray_spec.body("tray"), "", "")
    cup_xml, cont = (None, None)
    if cup:
        cup_xml, cont = _cup_xml((spawn_xy[0], spawn_xy[1], 6.0), contents)
        cspec = mujoco.MjSpec.from_string(cup_xml)
        # attach the cup AND beads (all top-level bodies) at the same place; bodies keep their names
        fr = house.worldbody.add_frame(pos=[0, 0, 0])
        for bd in list(cspec.bodies)[1:]:
            if bd.parent.name == "world":
                fr.attach_body(bd, "", "")
    for nm in ("left_arm_gripper_base", "right_arm_gripper_base"):
        house.add_equality(name=f"hold_{nm.split('_')[0]}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                           name1=nm, name2="tray", active=0)
    extra_own = ["cup", "tray"]
    for nm, pose in (("tp_counter", counter_at), ("tp_desk", desk_at)):
        if pose is not None:
            c, s_ = math.cos(pose[2]), math.sin(pose[2])
            tc = np.array(pose[:2]) + np.array([c * TRAY_LOCAL[0] - s_ * TRAY_LOCAL[1], s_ * TRAY_LOCAL[0] + c * TRAY_LOCAL[1]])
            _add_slab(house, nm, tc, pose[2] + math.pi / 2, scene.slab_size, COUNTER_TOP_Z - (HANDLE_T if finger_contact else 0.0))
            extra_own.append(nm)
    pm = []
    if with_people:
        for p in scene.people:
            if p.dynamic:
                _add_person(house, p.name)
            extra_own.append(p.name)
    pick_names = []
    if pick_props:
        from ..pick import props as PP
        pick_names = PP.add_pick_props(house, spawn_xy, yaw, pick_props)
        house.worldbody.add_camera(name="pick_head", pos=[0, 0, 0], fovy=70.0)
    sc._patch_real_collision_geoms(house)
    m = house.compile()
    own = B._robot_tray_bodies(m)
    for b in range(m.nbody):                      # cup, beads, tables, people are NOT house geoms: no arm bits
        nm = m.body(b).name or ""
        if nm in extra_own or re.fullmatch(r"ball_\d+", nm):
            own.add(b)
    n_or = 0
    for gi in range(m.ngeom):
        if int(m.geom_bodyid[gi]) in own:
            continue
        if (m.geom_contype[gi] or m.geom_conaffinity[gi]) and (m.geom_conaffinity[gi] & B.ARM_BITS) != B.ARM_BITS:
            m.geom_conaffinity[gi] |= B.ARM_BITS
            n_or += 1
    if finger_contact:
        for side in ("left", "right"):
            for k in (7, 8):
                g = m.geom(f"{side}_arm_link{k}_pad").id
                m.geom_contype[g] |= PAD_BIT
                m.geom_conaffinity[g] |= PAD_BIT
                m.geom_solimp[g] = PAD_SOLIMP                         # stiff pad contact: with the stock (0.95, 0.99) a 10 N pinch sinks 4 mm into the 6 mm plate
                m.body_contype[m.geom_bodyid[g]] |= PAD_BIT           # MuJoCo >= 3 culls body pairs by the aggregated body_contype/conaffinity
                m.body_conaffinity[m.geom_bodyid[g]] |= PAD_BIT
        for gt in range(m.ngeom):
            if m.body(m.geom_bodyid[gt]).name == "tray":
                m.geom_contype[gt] |= PAD_BIT
                m.geom_conaffinity[gt] |= PAD_BIT
                m.body_contype[m.geom_bodyid[gt]] |= PAD_BIT
                m.body_conaffinity[m.geom_bodyid[gt]] |= PAD_BIT
        # finger servo gain: the stock kp = 200 N/m with the command clipped to the joint range [0, 0.035] can push a 6 mm plate
        # (3 mm per finger) with kp * 3 mm = 0.6 N only.  Friction grasp needs ~10 N: raise the POSITION GAIN (forcerange +-20 N and the
        # command range are untouched, so no limit of the robot model changes).
        for side in ("left", "right"):
            for k in (7, 8):
                a = m.actuator(f"{side}_arm_joint{k}_pos").id
                m.actuator_gainprm[a, 0] = FINGER_KP_GRASP
                m.actuator_biasprm[a, 1] = -FINGER_KP_GRASP
                # reflected drive inertia: a 25 g link alone makes the constraint-space mass of a pad contact tiny, and MuJoCo's soft contact
                # stiffness scales with that mass (measured: 4 mm pad sink at 6 N, 6 deg tray tilt in the pinch).  A geared gripper drive reflects
                # far more inertia than the link; FINGER_ARMATURE [kg] on the slide joint stiffens the contact ~12x (0.05 mm sink at 12 N).
                m.dof_armature[m.jnt_dofadr[m.joint(f"{side}_arm_joint{k}").id]] = FINGER_ARMATURE
    m.opt.timestep = dt
    m.opt.gravity[:] = B.CANON_GRAVITY
    d = mujoco.MjData(m)
    for p in ([q for q in scene.people if q.dynamic] if with_people else []):
        pm.append(int(m.body_mocapid[m.body(p.name).id]))
    info = dict(scene=scene.name, spawn=spawn_xy.tolist(), yaw=yaw, dt=dt, n_conaff_or=n_or, nbody=m.nbody, ngeom=m.ngeom, nv=m.nv,
                nmocap=m.nmocap, model_hash=B.model_hash(m), has_tray=True, lift_h=0.0, lift_mode="locked", has_cup=cup,
                mu_true=MU_TRUE, dynamic_base=dynamic_base, finger_contact=finger_contact)
    if pick_names:
        info["pick_props"] = pick_names
    rig = Rig(m, d, info, scene, n_beads=(cont.n_balls if cont else 0), person_mocap=pm)
    return rig


def cup_arm_clearance(rig, dmax=0.2):
    """min mj_geomDistance between the cup geoms and every arm geom (>0: clear)."""
    m, d = rig.m, rig.d
    ft = np.zeros(6)
    cup_g = [g for g in range(m.ngeom) if (m.body(m.geom_bodyid[g]).name == "cup") and (m.geom_contype[g] or m.geom_conaffinity[g])]
    arm_g = [g for g in range(m.ngeom) if (m.body(m.geom_bodyid[g]).name or "").startswith(("left_arm_", "right_arm_"))
             and (m.geom_contype[g] or m.geom_conaffinity[g])]
    best = dmax
    for a in cup_g:
        for b in arm_g:
            best = min(best, mujoco.mj_geomDistance(m, d, a, b, dmax, ft))
    return float(best)


def _arm_contacts(m, d):
    """contacts that involve neither the cup nor a bead (initial bead/cup-floor overlap of the scene layout is not an IK failure)."""
    n = 0
    for i in range(d.ncon):
        c = d.contact[i]
        nm = [(m.body(m.geom_bodyid[g]).name or "") for g in (c.geom1, c.geom2)]
        if any(x == "cup" or x.startswith("ball_") for x in nm):
            continue
        n += 1
    return n


# ---------------------------------------------------------------------------------------------- hold pose (build.set_hold_state minus ball)
def set_hold(rig: Rig, grasped: bool = True, tray_xy_yaw=None):
    """IK the arms to the tray hold pose at the spawn base pose; put the tray (and cup+beads on it) at the hold position
    (or at `tray_xy_yaw`=(x,y,yaw,z) for a tray resting on a counter elsewhere); capture the two weld relposes (nominal) and
    activate them iff `grasped`.  Mirrors build.set_hold_state (copied: that function requires the tray_ball 'ball' body)."""
    m, d = rig.m, rig.d
    sp = np.array(rig.info["spawn"]); yaw = rig.info["yaw"]
    R = B._rotz(yaw)
    base = np.array([sp[0], sp[1], 0.0])
    d.mocap_pos[0] = base
    d.mocap_quat[0] = B.yaw_quat(yaw)
    if rig.info.get("dynamic_base"):
        ra_ = int(m.jnt_qposadr[m.joint("yor_free").id])
        d.qpos[ra_:ra_ + 3] = base
        d.qpos[ra_ + 3:ra_ + 7] = B.yaw_quat(yaw)
    c, hl, hr, al, ar, jw = B.tray_hold_targets(TRAY_LOCAL, "side", across=True)
    solsL = B.solve_hold_ik(hl, al, jw, "left", collect=8, seeds=200)
    solsR = B.solve_hold_ik(hr, ar, jw, "right", collect=8, seeds=200)
    la, _, _ = B._jaddr(m, B.LEFT_J)
    ra, _, _ = B._jaddr(m, B.RIGHT_J)
    ql = qr = None
    for (q1, e1) in solsL:
        for (q2, e2) in solsR:
            d.qpos[la] = q1; d.qpos[ra] = q2
            mujoco.mj_forward(m, d)
            if _arm_contacts(m, d) == 0:
                ql, el, qr, er = q1, e1, q2, e2
                break
        if ql is not None:
            break
    if ql is None:
        raise RuntimeError("no contact-free IK pair for the hold pose")
    d.qpos[la] = ql; d.qpos[ra] = qr
    fin = 0.0175
    for side in ("left", "right"):
        d.qpos[m.jnt_qposadr[m.joint(f"{side}_arm_joint7").id]] = fin
        d.qpos[m.jnt_qposadr[m.joint(f"{side}_arm_joint8").id]] = -fin
    tray_c = base + R @ c
    tq = B.yaw_quat(yaw + math.pi / 2)
    ta = m.jnt_qposadr[m.joint("tray_free").id]
    d.qpos[ta:ta + 3] = tray_c
    d.qpos[ta + 3:ta + 7] = tq
    cupnames = (["cup"] + [f"ball_{i}" for i in range(rig.n_beads)]) if rig.info["has_cup"] else []
    for nm in cupnames:          # cup + beads were spawned at (spawn_xy, z=6): move onto the tray (cup base 0.5 mm above the surface)
        q = m.jnt_qposadr[m.joint(f"{nm}_free").id]
        off = B._rotz(yaw + math.pi / 2) @ np.array([CUP_TRAY_XY[0], CUP_TRAY_XY[1], 0.0])
        d.qpos[q:q + 3] += tray_c + off + np.array([0.0, 0.0, 0.0005]) - np.array([sp[0], sp[1], 6.0])
    mujoco.mj_forward(m, d)
    # weld relposes: nominal hold geometry (tray at its hold pose relative to the grippers)
    for side in ("left", "right"):
        B.capture_weld(m, d, f"hold_{side}", f"{side}_arm_gripper_base", "tray")
    nominal = {s: m.eq_data[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"hold_{s}")].copy() for s in ("left", "right")}
    if tray_xy_yaw is not None:      # tray (+cup+beads) resting elsewhere (counter): rigid map about the hold-tray centre
        x, y, ty, z = tray_xy_yaw
        dyaw = ty - (yaw + math.pi / 2)
        Rz = B._rotz(dyaw)
        tgt = np.array([x, y, z])
        for nm in ["tray"] + cupnames:
            q = m.jnt_qposadr[m.joint(f"{nm}_free").id]
            d.qpos[q:q + 3] = tgt + Rz @ (d.qpos[q:q + 3] - tray_c)
            if nm == "tray":
                d.qpos[q + 3:q + 7] = B.yaw_quat(ty)
        mujoco.mj_forward(m, d)
    if not grasped:
        for s in ("left", "right"):
            d.eq_active[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"hold_{s}")] = 0
    mujoco.mj_forward(m, d)
    rig.info.update(hold_q=dict(left=ql.tolist(), right=qr.tolist()), finger=fin, hold_mode="side", tray_across=True,
                    tray_local=list(TRAY_LOCAL), ik_err={k: {a: float(b) for a, b in v.items()} for k, v in dict(left=el, right=er).items()})
    if rig.info["has_cup"]:
        rig.info["cup_arm_clear_m"] = cup_arm_clearance(rig)
    rig.info["weld_nominal"] = {k: v.tolist() for k, v in nominal.items()}
    return rig


def new_rig(scene_name="h5", spawn=None, grasped=True, dt=0.005, with_people=True, counter_at=None, desk_at=None, tray_at=None,
            cup=True, dynamic_base=False, finger_contact=False, pick_props=None) -> Rig:
    scene = get_scene(scene_name)
    if spawn is None:
        spawn = scene.waypoints["lab_start"]
    rig = build_rig(scene, spawn[:2], spawn[2], dt=dt, with_people=with_people, counter_at=counter_at, desk_at=desk_at, cup=cup,
                    dynamic_base=dynamic_base, finger_contact=finger_contact, pick_props=pick_props)
    set_hold(rig, grasped=grasped, tray_xy_yaw=tray_at)
    B.disable_arm_servos(rig.m)
    fin = rig.info["finger"]
    for side in ("left", "right"):
        rig.d.ctrl[rig.m.actuator(f"{side}_arm_joint7_pos").id] = fin
        rig.d.ctrl[rig.m.actuator(f"{side}_arm_joint8_pos").id] = -fin
    mujoco.mj_forward(rig.m, rig.d)
    return rig
