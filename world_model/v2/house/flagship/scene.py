"""flagship/scene.py -- the flagship coffee-run world: office + desk clutter + counter objects + serving ledge + placement stand.  SIMULATION ONLY.

Reuses (read-only imports): office.scene/thor/compose, zed_demo.rig (clutter, speckle textures, ZED cameras), teleop.scene/sim.
The MAIN sim model has flat colours (looks good in the video); the PERCEPTION TWIN (build_twin) is the same spec + speckle textures + ZED stereo
cameras (block matching needs texture) -- the twin is posed at the main sim's base pose when the robot "captures".

Lift / ledge:  the office counter is 0.90 m high, the arms' hold height is 0.85 m at lift 0.  The tray + cup wait on a SERVING LEDGE (hand-off slab,
top = 0.90 m = counter height, flush with the counter top) just in front of the counter edge; the robot raises its lift by H_PICK = 0.056 m to put the
grippers at the handles.  Patch of teleop.scene module constants (TRAY_HOLD_Z, COUNTER_TOP_Z) is applied ONLY while the TeleopSim is constructed.
Desk: the delivery desk (0.734 m) is below the lowest hold height (0.85 m, lift cannot descend) -> the tray is lowered onto a small PLACEMENT STAND
(static body `tp_stand`, repositioned via model.body_pos) moved under the chosen spot (documented simplification).
"""
from __future__ import annotations

import contextlib
import math
import os
from pathlib import Path

import numpy as np
import mujoco

from world_model.v2.house.office import scene as OS
from world_model.v2.house.teleop import scene as SC
from world_model.v2.house.teleop.sim import TeleopSim

OUT = Path(os.environ.get("FLAGSHIP_OUT", "/data/local/ss5772/molmospaces/flagship"))
COUNTER_TOP = OS.COUNTER_TOP                       # 0.90
TRAY_UNDERSIDE_LIFT0 = SC.TRAY_HOLD_Z - 0.006      # 0.844
H_PICK = COUNTER_TOP - TRAY_UNDERSIDE_LIFT0        # 0.056 m lift to bring the tray underside to the ledge
LIFT_CARRY_EXTRA = 0.06                            # lift above H_PICK to clear the ledge (operator LIFT phase)
STAND_HALF = (0.075, 0.055)                        # placement stand half extents (tray-frame x,y after yaw) [m]; smaller than the tray (0.30 x 0.20)
STAND_PARK = (0.0, 0.0, -6.0)

_walker_cache = {}


def counter_extras(spec: mujoco.MjSpec):
    """more counter clutter for the stereo demo (static; collision twins like the office primitives): milk jug, sugar jar, napkin box, tall cup stack.
    Returns {name: (centre, half)} GT (EVAL only)."""
    W = spec.worldbody
    from world_model.v2.house.zed_demo.rig import _cyl
    gt = {}
    _cyl(W, "milk_jug", (23.98, 6.66), 0.050, COUNTER_TOP, 0.22, (0.92, 0.92, 0.95, 1))
    gt["milk_jug"] = dict(centre=[23.98, 6.66, COUNTER_TOP + 0.11], half=[0.05, 0.05, 0.11])
    _cyl(W, "sugar_jar", (23.30, 6.78), 0.045, COUNTER_TOP, 0.13, (0.85, 0.55, 0.20, 1))
    gt["sugar_jar"] = dict(centre=[23.30, 6.78, COUNTER_TOP + 0.065], half=[0.045, 0.045, 0.065])
    OS.add_rect(W, "napkin_box", 22.28, 22.46, 6.58, 6.76, COUNTER_TOP, COUNTER_TOP + 0.14, (0.80, 0.80, 0.35, 1))
    gt["napkin_box"] = dict(centre=[22.37, 6.67, COUNTER_TOP + 0.07], half=[0.09, 0.09, 0.07])
    _cyl(W, "cup_stack", (24.95, 6.55), 0.040, COUNTER_TOP, 0.24, (0.95, 0.95, 0.95, 1))
    gt["cup_stack"] = dict(centre=[24.95, 6.55, COUNTER_TOP + 0.12], half=[0.04, 0.04, 0.12])
    return gt


def add_ledge_visual(spec: mujoco.MjSpec, pad_x=22.9, edge_y=6.3447):
    """decorative serving ledge plate: flush with the counter top, bridges the gap between the counter front edge and the tray slab (visual only; the
    collision surface the tray rests on is the `tp_counter` slab TeleopSim adds).  Main model only (the perception twin does not carry it: it is the
    pick target surface, not an obstacle)."""
    W = spec.worldbody
    OS.add_rect(W, "ledge_vis", pad_x - 0.17, pad_x + 0.17, edge_y - 0.20, edge_y, COUNTER_TOP - 0.030, COUNTER_TOP - 0.004, (0.55, 0.42, 0.30, 1),
                collide=False)


def add_stand(spec: mujoco.MjSpec):
    """placement stand: STATIC body parked under the floor (NOT mocap: mocap index 0 must stay yor_root); moved under the chosen desk spot at placement
    time by editing model.body_pos / body_quat (set_stand).  Collides with tray/cup only."""
    b = spec.worldbody.add_body(name="tp_stand", pos=list(STAND_PARK))
    b.add_geom(name="tp_stand_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[STAND_HALF[0], STAND_HALF[1], 0.02], pos=[0, 0, -0.02],
               rgba=[0.30, 0.32, 0.36, 1], contype=1, conaffinity=1, friction=[0.6, 0.005, 0.0001])
    b.add_geom(name="tp_stand_post", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, 0.07], pos=[0, 0, -0.11], rgba=[0.30, 0.32, 0.36, 1],
               contype=0, conaffinity=0)


def load_spec(ledge_visual=True):
    """office + desk clutter + counter extras + ledge visual (main model only) + placement stand.  Fresh spec each call."""
    from world_model.v2.house.zed_demo import rig as ZR
    spec = OS.load_office_spec()
    meta = OS.office_meta()
    gt = ZR.add_clutter(spec, meta["desk_top_height"])
    gt.update(counter_extras(spec))
    if ledge_visual:
        add_ledge_visual(spec)
    add_stand(spec)
    load_spec.gt = gt
    return spec


def launch_fix(track):
    """teleop.sim.set_people only advances a person's track clock when the CURRENT velocity is non-zero, and PersonTrack.at() returns zero velocity at
    t <= first time -> a track that starts at t=0 never starts (walkers stay at their first point forever).  Prepend a point at t=-1 one second 'upstream'
    so the velocity at t=0 is non-zero.  (teleop/ is read-only; the fix lives here.)"""
    p = track.pts
    if p[0, 0] >= 0.0 and len(p) > 1 and p[1, 0] > p[0, 0]:
        v = (p[1, 1:3] - p[0, 1:3]) / (p[1, 0] - p[0, 0])
        if np.linalg.norm(v) > 0:
            track.pts = np.vstack([[p[0, 0] - 1.0, *(p[0, 1:3] - v * 1.0)], p])
    return track


def walker_tracks():
    """OPTIONAL (default off, see README known issues) extra cafeteria walker pacing north-south across the robot's route (x = 21.55, y 2.3 .. 4.3), 0.8 m/s with 1.5 s pauses at the ends."""
    from world_model.v2.house.teleop.scene import PersonTrack
    a, b = np.array([21.55, 2.3]), np.array([21.55, 4.3])
    pts, t = [[0.0, *a]], 0.0
    L = float(np.linalg.norm(b - a)); v = 0.8
    for k in range(60):
        src, dst = (a, b) if k % 2 == 0 else (b, a)
        t += L / v
        pts.append([t, *dst])
        t += 1.5
        pts.append([t, *dst])
    return PersonTrack("p_cafe", np.array(pts))


def make_scene(cafe_walker=False):
    sc = SC._office_scene()
    sc.name = "flagship"
    sc.spec_loader = load_spec
    if cafe_walker:
        sc.people = list(sc.people) + [walker_tracks()]
    for p in sc.people:
        if p.name == "p_walker":
            # the office walker ends as a permanent obstacle in the 2 m corridor (set_people freezes it at its last point); let it walk on into the lab and
            # stop in an empty corner (6.2, 4.4), away from both routes, so it is a moving person for both passes and not a roadblock on the return.
            p.pts = np.array([[0.0, 17.0, 3.55], [26.0, 8.2, 3.55], [30.5, 6.2, 4.4], [1e4, 6.2, 4.4]])
    sc.people = [launch_fix(p) if p.dynamic else p for p in sc.people]
    return sc


@contextlib.contextmanager
def lift_heights():
    """tray hand-off at counter height while TeleopSim builds its rig."""
    old = (SC.TRAY_HOLD_Z, SC.COUNTER_TOP_Z)
    SC.TRAY_HOLD_Z, SC.COUNTER_TOP_Z = COUNTER_TOP + 0.006, COUNTER_TOP
    try:
        yield
    finally:
        SC.TRAY_HOLD_Z, SC.COUNTER_TOP_Z = old


def register(cafe_walker=False):
    SC.SCENES["flagship"] = lambda: make_scene(cafe_walker)
    _orig = SC.get_scene

    def get_scene(name):
        if name == "flagship":
            return make_scene(cafe_walker)
        return _orig(name)
    SC.get_scene = get_scene


def new_sim(seed=0, cafe_walker=False, spawn=None, grasped=False, counter=True):
    """TeleopSim in the flagship world.  Default: robot at lab_start, EMPTY hands, tray + cup on the serving ledge at the counter.
    (spawn / grasped / counter are for debugging and tests: e.g. spawn=desk stand-off with grasped=True, counter=False = tray + cup in the hands.)"""
    register(cafe_walker)
    with lift_heights():
        sim = TeleopSim("flagship", grasped=grasped, seed=seed, with_people=True, counter=counter, desk=False, spawn=spawn)
    m = sim.m
    # the stand / ledge slab are environment, not 'house' bumps for the robot's tray
    for g in range(m.ngeom):
        bn = m.body(m.geom_bodyid[g]).name
        if bn in ("tp_stand",):
            sim.house_geom[g] = False
    return sim


def build_twin(pitch_deg=30.0, dt=0.005, tex_dir=None):
    """perception twin: same spec + speckle textures + ZED stereo cameras; robot with a held tray at the hold pose. Returns Composed."""
    from world_model.v2.house.office import compose as CP
    from world_model.v2.house.zed_demo import rig as ZR
    spec = load_spec(ledge_visual=False)
    ZR.apply_speckle(spec, Path(tex_dir or OUT / "tex"))
    old = CP._head_cam
    CP._head_cam = ZR._zed_cams
    try:
        wp = OS.office_meta()["waypoints"]["lab_desk_place"]
        C = CP.new_hold_office(spawn_xy=wp["xy"], yaw=wp["heading"], dt=dt, lift_h=0.0, servos_off=True, head_cam=True,
                               head_pitch_deg=pitch_deg, head_fovy=ZR.ZED_FOVY, spec=spec)
    finally:
        CP._head_cam = old
    return C


def set_stand(m, xy=None, z=None, yaw=0.0):
    """move the placement stand (static body) to (xy, z_top) or park it (xy=None)."""
    b = m.body("tp_stand").id
    if xy is None:
        m.body_pos[b] = STAND_PARK
        m.body_quat[b] = (1, 0, 0, 0)
    else:
        m.body_pos[b] = (xy[0], xy[1], z)
        m.body_quat[b] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
