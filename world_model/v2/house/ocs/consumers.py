"""ocs/consumers.py -- what the control path reads FROM a WorldState.  SIMULATION ONLY.

    to_env_state(ws)        -> tray_spec.EnvState (the shield IRMonitor's input: tray pose + cup base position, world frame)
    to_people_state(ws)     -> [(xy (2,), v (2,))]   the shield's `sim.people_state` (+ person_info: names, per-person inflation from `unc`)
    to_obstacle_zonos(ws)   -> [(oid, Zonotope)]     inflated world-frame zonotopes (zono rows / base filter / ClearanceMap)
    to_occ(ws)              -> teleop.scene.OccGrid  the shield's wall grid (+ localisation margin: `loc_margin(ws)`)
    to_ir_entities(ws)      -> {name: ir.Entity}     frame/unc closures READ the WorldState (or a rollout list) -> ir specs evaluate on it directly
    ShieldView(sim, ws)     -> proxy of the TeleopSim whose reads of tray / cup / people / base pose come from the WorldState (like
                               people_track's SimProxy; teleop/shield.py is not edited)
"""
from __future__ import annotations

import math

import numpy as np

from world_model.v2.tray_rep.sets import ir as IR
from world_model.v2.tray_rep.sets import tray_spec as TS
from world_model.v2.tray_rep.sets.ir import Pose, Unc, Entity, Point, prism_xy
from world_model.v2.tray_rep.sets.zono import Zonotope

from .state import Kind, WorldState, FreeSpace, ObjectState
from . import util as U


# ====================================================================================== uncertainty through the parent chain
def world_unc(ws: WorldState, oid: str) -> Unc:
    """position set (world frame) and rotation bound of object `oid`, composed along its parent chain:
        pos_w = pos_parent_w (+) R_parent * own.pos (+) ball(rot_parent * |p_child|)        rot_w = rot_parent + own.rot
    (parent "robot" -> RobotState.base_unc; parent None -> own.unc is already a world-frame set)."""
    o = ws.objects[oid]
    if o.parent is None:
        return Unc(o.unc.pos, float(o.unc.rot))
    if o.parent == "robot":
        Pp = ws.robot.base
        pu = Unc(ws.robot.base_unc.pos, float(ws.robot.base_unc.rot))
    else:
        Pp = ws.world_pose(o.parent)
        pu = world_unc(ws, o.parent)
    Rp = np.asarray(Pp.R)
    pos = pu.pos
    if o.unc.pos is not None:
        own = o.unc.pos.linear(Rp)
        pos = own if pos is None else pos.minkowski(own)
    # `rot` = bound on the error of the body AXES (tilt).  The base is planar: its (yaw) angle bound swings child positions (lever term below) but does
    # not change any child's tilt, so it is not added to the rotation bound.
    rot = (0.0 if o.parent == "robot" else float(pu.rot)) + float(o.unc.rot)
    if pu.rot > 0:                                                  # parent's angle error swings the child's position
        lever = float(pu.rot) * float(np.linalg.norm(o.pose.p))
        if lever > 0:
            ball = U.ball_zono(lever)
            pos = ball if pos is None else pos.minkowski(ball)
    return Unc(pos, rot)


# ====================================================================================== shield inputs
def _world_quat(o: ObjectState, Pw: Pose):
    q = o.extra.get("quat")
    return np.asarray(q, float) if (q is not None and o.parent is None) else U.R_to_quat(Pw.R)


def to_env_state(ws: WorldState, tray: str = "tray", cup: str = "cup") -> TS.EnvState:
    """EnvState(tray_pos, tray_quat, cup_world) from the OCS (exact for a TruthProducer state: quat and positions are passed through unchanged)."""
    ot, oc = ws.objects[tray], ws.objects[cup]
    Pt, Pc = ws.world_pose(tray), ws.world_pose(cup)
    return TS.EnvState(np.asarray(Pt.p), _world_quat(ot, Pt), np.asarray(Pc.p))


def person_ids(ws: WorldState):
    return [o.oid for o in ws.objects.values() if o.kind == Kind.PERSON]


def to_people_state(ws: WorldState):
    """[(xy, v)] in the order of the PERSON objects (the order of `scene.people` for a TruthProducer state)."""
    out = []
    for oid in person_ids(ws):
        o = ws.objects[oid]
        p = np.asarray(ws.world_pose(oid).p)[:2].copy()
        v = np.zeros(2) if o.vel is None else np.asarray(o.vel, float)[:2].copy()
        out.append((p, v))
    return out


def person_info(ws: WorldState):
    """per person: dict(oid, name, infl) with `infl` = horizontal support radius of the world-frame position set (isotropic bound) -- the
    extra keep-out radius the shield should add (people_track.PerceivedShield semantics)."""
    out = []
    ang = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    for oid in person_ids(ws):
        u = world_unc(ws, oid)
        infl = 0.0 if u.pos is None else max(U.unc_support(u, (math.cos(a), math.sin(a), 0.0)) for a in ang)
        out.append(dict(oid=oid, name=ws.objects[oid].name or oid, infl=float(infl)))
    return out


def to_obstacle_zonos(ws: WorldState, kinds=(Kind.OBSTACLE, Kind.SURFACE), inflate: bool = True):
    """[(oid, Zonotope)] world-frame obstacle sets.  inflate: Minkowski-add the (chain-composed) position uncertainty set and the rotation-lever
    ball, i.e. exactly what a robust IR clearance atom / the zonotope rows should keep out of."""
    out = []
    for o in ws.objects.values():
        if o.kind not in kinds:
            continue
        Pw = ws.world_pose(o.oid)
        Z = o.shape.zonotope().linear(np.asarray(Pw.R), np.asarray(Pw.p))
        if inflate:
            u = world_unc(ws, o.oid)
            if u.pos is not None:
                Z = Z.minkowski(u.pos)
            if u.rot > 0:
                Z = Z.minkowski(U.ball_zono(float(u.rot) * o.shape.max_norm()))
        out.append((o.oid, Z))
    return out


def loc_margin(ws: WorldState, k: float = 1.0) -> float:
    """localisation margin [m] for the shield's wall buffer: horizontal support of the base position set (k scales it)."""
    u = ws.robot.base_unc
    if u.pos is None:
        return 0.0
    ang = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    return float(k * max(U.unc_support(u, (math.cos(a), math.sin(a), 0.0)) for a in ang))


_OCC_CACHE: dict = {}


def to_occ(ws: WorldState):
    """OccGrid (res, x0, y0, occ, edt) from ws.free: occupied = log-odds > 0; edt = distance_transform_edt(~occ) * res (the formula of
    teleop.scene.load_occupancy, so a TruthProducer grid round-trips exactly).  Cached on the FreeSpace object."""
    f: FreeSpace = ws.free
    if f is None:
        return None
    c = getattr(f, "_occ_grid", None)
    if c is None:
        from scipy import ndimage
        from world_model.v2.house.teleop.scene import OccGrid
        occ = np.asarray(f.logodds) > 0.0
        c = OccGrid(float(f.res), float(f.origin[0]), float(f.origin[1]), occ, ndimage.distance_transform_edt(~occ) * float(f.res))
        f._occ_grid = c
    return c


# ====================================================================================== robot-frame view
def robot_frame(ws: WorldState) -> WorldState:
    """the same state expressed in the ROBOT frame (base = identity, base_unc = 0): relative quantities (cup-to-person clearance) do not carry the
    base localisation error twice.  Objects that were measured relative to the robot (parent "robot" chains: tray, cup, tracked people) keep their own
    sets; map-registered objects (parent None: stereo obstacles, truth objects) keep their pose transformed by the ESTIMATED base and ADD the base set
    (a map object's position relative to the robot is only as good as the localisation)."""
    Pb = ws.robot.base
    Rb, pb = np.asarray(Pb.R), np.asarray(Pb.p)
    bu = ws.robot.base_unc
    from dataclasses import replace as _rep
    out = WorldState(t=ws.t, robot=_rep(ws.robot, base=Pose(np.eye(3), np.zeros(3)), base_unc=Unc(), yaw=0.0), objects={}, relations=ws.relations, free=None,
                     meta=ws.meta)
    for oid, o in ws.objects.items():
        if o.parent is not None:
            out.objects[oid] = o
            continue
        P = o.pose
        p_rel = Rb.T @ (np.asarray(P.p) - pb)
        pos = o.unc.pos.linear(Rb.T) if o.unc.pos is not None else None
        if bu.pos is not None:
            b = bu.pos.linear(Rb.T)
            pos = b if pos is None else pos.minkowski(b)
        if bu.rot > 0:
            ball = U.ball_zono(float(bu.rot) * float(np.linalg.norm(p_rel)))
            pos = ball if pos is None else pos.minkowski(ball)
        out.objects[oid] = _rep(o, pose=Pose(Rb.T @ np.asarray(P.R), p_rel), unc=Unc(pos, float(o.unc.rot)))
    return out


# ====================================================================================== IR entities
def _as_seq(ws):
    return list(ws) if isinstance(ws, (list, tuple)) else None


def _stack_unc(us):
    rot = float(max(float(u.rot) for u in us))                    # ir.Inside tests `if rot:` -> a scalar; the max over the window is conservative
    if all(u.pos is None for u in us):
        return Unc(None, rot)
    m = max((u.pos.G.shape[-1] for u in us if u.pos is not None), default=0)
    C = np.zeros((len(us), 3))
    G = np.zeros((len(us), 3, m))
    for i, u in enumerate(us):
        if u.pos is not None:
            C[i] = u.pos.c
            G[i, :, :u.pos.G.shape[-1]] = u.pos.G
    return Unc(Zonotope(C, G), rot)


def _entity(name: str, oid: str, shape, kind_pose=None):
    """Entity whose frame/unc read the state: a WorldState (K=1, unbatched) or a list of WorldStates (leading time axis)."""
    def frame(s):
        seq = _as_seq(s)
        if seq is None:
            P = s.world_pose(oid)
            return Pose(np.asarray(P.R), np.asarray(P.p))
        Ps = [x.world_pose(oid) for x in seq]
        return Pose(np.stack([np.asarray(P.R) for P in Ps]), np.stack([np.asarray(P.p) for P in Ps]))

    def unc(s):
        seq = _as_seq(s)
        if seq is None:
            return world_unc(s, oid)
        return _stack_unc([world_unc(x, oid) for x in seq])

    return Entity(name, shape, frame, unc, None)


def _rel_entity(name: str, oid: str, parent_oid: str, shape):
    """the object expressed in `parent_oid`'s frame (Entity.parent = "tray_top" so ir.Inside takes the pose as already relative): the pose and the
    set are the object's OWN (parent-frame) ones -- the common-mode error of the chain (base, tray FK) cancels, as it does for the real cup estimate,
    which is measured relative to the estimated tray."""
    def rel(s, o_):
        if o_.parent == parent_oid:
            return o_.pose, o_.unc
        Pw, Pp = s.world_pose(oid), s.world_pose(parent_oid)
        R = np.asarray(Pp.R)
        return Pose(R.T @ np.asarray(Pw.R), R.T @ (np.asarray(Pw.p) - np.asarray(Pp.p))), Unc()

    def frame(s):
        seq = _as_seq(s)
        if seq is None:
            P, _ = rel(s, s.objects[oid])
            return Pose(np.asarray(P.R), np.asarray(P.p))
        Ps = [rel(x, x.objects[oid])[0] for x in seq]
        return Pose(np.stack([np.asarray(P.R) for P in Ps]), np.stack([np.asarray(P.p) for P in Ps]))

    def unc(s):
        seq = _as_seq(s)
        if seq is None:
            return rel(s, s.objects[oid])[1]
        return _stack_unc([rel(x, x.objects[oid])[1] for x in seq])

    return Entity(name, shape, frame, unc, parent_oid if parent_oid != "tray" else "tray_top")


def to_ir_entities(ws, tray: str = "tray", cup: str = "cup", hx: float = U.HX, hy: float = U.HY, frame: str = "world"):
    """{name: ir.Entity} over a WorldState or a rollout (list of WorldStates).  Keys:
        "tray", "tray_top"  (prism over the tray rectangle, pose of the tray object)
        "cup"               (cup BODY zonotope, pose of the cup object)       "ball" = the cup base POINT (the shield's TRAY_SPEC_FULL semantics)
        every other object under its oid and, if set, its name (e.g. "laptop", "person"), shape = its ObjectState.shape."""
    if frame == "robot":                                              # all entities in the robot frame (relative statements: no doubled localisation error)
        ws = [robot_frame(x) for x in ws] if _as_seq(ws) else robot_frame(ws)
    s0 = _as_seq(ws)[0] if _as_seq(ws) else ws
    ents = {}
    ents["tray"] = _entity("tray", tray, prism_xy(hx, hy))
    ents["tray_top"] = _entity("tray_top", tray, prism_xy(hx, hy))
    if cup in s0.objects:
        ents["cup"] = _entity("cup", cup, s0.objects[cup].shape)
        ents["ball"] = _entity("ball", cup, Point())
        ents["cup_rel"] = _rel_entity("cup_rel", cup, tray, s0.objects[cup].shape)
    for o in s0.objects.values():
        if o.oid in (tray, cup):
            continue
        e = _entity(o.name or o.oid, o.oid, o.shape)
        ents[o.oid] = e
        if o.name:
            ents[o.name] = e
    return ents


# ====================================================================================== shield proxy
class _DataProxy:
    """MuJoCo-data stand-in: xpos / xmat / xquat of the tray and cup rows come from the OCS, everything else from the real data."""

    def __init__(self, d, over: dict):
        self._d = d
        self.xpos, self.xmat, self.xquat = d.xpos.copy(), d.xmat.copy(), d.xquat.copy()
        for bid, (p, R, q) in over.items():
            self.xpos[bid] = p
            self.xmat[bid] = np.asarray(R).reshape(9)
            self.xquat[bid] = q

    def __getattr__(self, n):
        return getattr(self._d, n)


class _BaseProxy:
    def __init__(self, base, x, y, yaw, v):
        self._b = base
        self.x, self.y, self.yaw, self.v = x, y, yaw, v

    def __getattr__(self, n):
        return getattr(self._b, n)


class _Named:
    def __init__(self, name, dynamic=True):
        self.name, self.dynamic = name, dynamic


class _ScenePeople:
    def __init__(self, scene, names):
        self._s = scene
        self.people = [_Named(n) for n in names]

    def __getattr__(self, n):
        return getattr(self._s, n)


class ShieldView:
    """`sim` as the shield should see it when it consumes the OCS: delegates everything to the real TeleopSim except
         d.xpos/xmat/xquat[tray, cup]   <- ws tray / cup world poses          people_state, scene.people   <- ws PERSON objects
         base.x/y/yaw/v, pose(), ctrl_pose()  <- ws.robot.base / base_vel   (ctrl_pose only when `base_from_ws`)
    The rows of the shield that use the *true* base pose for relative geometry (cup rows via sim.base.*) then read the OCS base."""

    def __init__(self, sim, ws: WorldState, tray: str = "tray", cup: str = "cup", base_from_ws: bool = True):
        self._sim = sim
        over = {}
        for oid, bid in ((tray, sim.tray), (cup, sim.cup)):
            o = ws.objects[oid]
            P = ws.world_pose(oid)
            over[bid] = (np.asarray(P.p), np.asarray(P.R), _world_quat(o, P))
        self.d = _DataProxy(sim.d, over)
        ps = to_people_state(ws)
        self.people_state = ps
        self.scene = _ScenePeople(sim.scene, [ws.objects[i].name or i for i in person_ids(ws)])
        if base_from_ws:
            Pb = ws.robot.base
            yaw = ws.robot.yaw if ws.robot.yaw is not None else U.yaw_of(Pb.R)
            v = np.zeros(3) if ws.robot.base_vel is None else np.asarray(ws.robot.base_vel, float)
            self.base = _BaseProxy(sim.base, float(Pb.p[0]), float(Pb.p[1]), yaw, v)
            self._pose = np.array([float(Pb.p[0]), float(Pb.p[1]), yaw])
        else:
            self.base, self._pose = sim.base, None

    def pose(self):
        return self._sim.pose() if self._pose is None else self._pose.copy()

    def ctrl_pose(self):
        return self.pose()

    def __getattr__(self, n):
        return getattr(self._sim, n)
