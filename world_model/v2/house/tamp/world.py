"""tamp/world.py -- the planner's world model (layer-2 input), built from a TeleopSim.  SIMULATION ONLY.

World state source today: sim TRUTH for the item / surface poses (the perception / object-centric-state path, house/ocs, replaces it later).
Uncertainty enters as SETS, never points:
  * item mass box   = model mass x (1 -+ MASS_FRAC)        ASSUMPTION (stands in for a VLM / prior interval)
  * grasp friction  = grasp3 defaults [0.4, 0.8]            ASSUMPTION (grasp3's own)
  * payload (cup on tray) friction = ShieldCfg [mu_lo, mu_hi] = [0.18, 0.35]
  * base pose box   = the gatekeeper belief's e_xy (0 with the truth pose source), or `sigma_star` if larger (T3 H0's sigma*)
People are NOT part of the plan-time world: the template belief has no people; the runtime gatekeeper certifies against the live people belief (T1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..checkers import grasp3 as G3
from ..gatekeeper.belief import Belief, GKConfig, build_belief
from ..teleop import geom as GM
from ..teleop import shield as SH
from .types import Conf, Item, Placement, Surface, Cert

MASS_FRAC = 0.20                    # ASSUMPTION: item mass box half-width as a fraction of the model mass
FREE_CUP_MARGIN = 1e3               # hands empty: the cup row is inactive (a_cup_max made huge -> only the hardware slew box limits acceleration)
# (surface name, slab top geom, scene waypoint of its dock conf)
SURFACES = (("counter", "tp_counter_top", "cafe_counter_pick"), ("desk", "tp_desk_top", "lab_desk_place"))
# grasps the EXECUTOR can realize, per item (C3 may admit more; a plan must not use a grasp no skill implements).
# sim tray grasp = two hands on the short rims, held by welds (TeleopSim grasp_mode="weld") == grasp3 "bimanual_squeeze_a".
SKILLS = {"tray": ("bimanual_squeeze_a",)}
# the gatekeeper config the executor runs.  cup_row="attached" (10-22): cup row only while the tray is on the robot (the old "always" applied it to the
# cup on the counter, lever arm ~3.75 m, which stalled empty-handed turns near people); run_tamp --cup-row always reproduces the pre-10-22 runs.
RUNTIME_GK = dict(person_model="accel", candidates="set", keep_rule="monitor", cup_chord=True, cup_row="attached")


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def rel_pose(base, p):
    """pose p (x, y, yaw) expressed in the frame of `base` (x, y, yaw)."""
    c, s = math.cos(base[2]), math.sin(base[2])
    dx, dy = p[0] - base[0], p[1] - base[1]
    return (c * dx + s * dy, -s * dx + c * dy, wrap(p[2] - base[2]))


def compose(base, rel):
    c, s = math.cos(base[2]), math.sin(base[2])
    return (base[0] + c * rel[0] - s * rel[1], base[1] + s * rel[0] + c * rel[1], wrap(base[2] + rel[2]))


def _yaw(xmat):
    return math.atan2(float(xmat[3]), float(xmat[0]))


@dataclass
class World:
    scene: object
    q0: Conf
    surfaces: dict                  # name -> Surface
    items: dict                     # name -> Item
    hold_rel: dict                  # item -> (dx, dy, dyaw): item pose in the base frame while held
    cup_rel: tuple                  # cup xy in the base frame while the tray is held (payload row anchor)
    gk: GKConfig
    shield_cfg: object
    bel0: Belief                    # template: no people; re-anchored per pose by belief_at()
    routes: list                    # [(name, pts)] scene routes (xy polylines)
    back_mode: str
    sigma_star: float = 0.0
    people_obs: tuple = ()          # people seen when the world was built (used by the NEAR part of route certificates, see Streams.certify_route)
    stats: dict = field(default_factory=dict)
    bel_people: Belief | None = None    # the belief WITH the current people (bel0 = same, people stripped); None -> no near-people certification
    cup_world: tuple | None = None      # cup xy in the MAP frame when the world was built (the runtime gatekeeper's cup row is anchored on it even with empty hands)

    def count(self, key, ok):
        s = self.stats.setdefault(key, [0, 0])
        s[0] += 1
        s[1] += 0 if ok else 1

    def belief_at(self, pose, holding: bool, item: str | None = None, people: bool = False) -> Belief:
        """the template belief re-anchored at map pose `pose`; holding=True activates the payload (cup) rows with the item's payload friction set."""
        e_xy = max(float(self.bel0.e_xy), float(self.sigma_star))
        kw = dict(pose_wall=np.asarray(pose, float).copy(), pose_people=np.asarray(pose, float).copy(), e_xy=e_xy, e_th=self.gk.th_over_xy * e_xy)
        if holding:
            it = self.items[item] if item else None
            mu = it.extra.get("payload_mu", (self.shield_cfg.mu_lo, self.shield_cfg.mu_hi)) if it else (self.shield_cfg.mu_lo, self.shield_cfg.mu_hi)
            kw.update(r_cup=np.asarray(self.cup_rel, float).copy(), cup_margin=self.gk.cup_margin, mu_lo=float(mu[0]), mu_hi=float(mu[1]))
        else:
            kw.update(r_cup=np.zeros(2), cup_margin=FREE_CUP_MARGIN)
        base = self.bel_people if (people and self.bel_people is not None) else self.bel0
        return base.with_(**kw)


def runtime_belief_at(w: World, pose, holding: bool, item: str | None = None, people: bool = False) -> Belief:
    """belief_at() as the RUNTIME gatekeeper builds it: the cup row is ALWAYS active.  Holding: the payload cup in the base frame (cup_rel).  Hands empty: the cup
    stays where it is in the map (w.cup_world) -> r_cup = R(-yaw)(cup - base xy), cup_margin = gk.cup_margin (gate.build_belief does this whether or not the tray is
    grasped; the legacy from-rest stream used FREE_CUP_MARGIN instead).  Falls back to belief_at() if the world has no cup position."""
    if holding or w.cup_world is None or getattr(w.gk, "cup_row", "always") == "attached":
        return w.belief_at(pose, holding, item, people)              # "attached": empty hands -> no cup row, as the runtime gatekeeper now does
    b = w.belief_at(pose, True, item or "tray", people)
    c, s = math.cos(pose[2]), math.sin(pose[2])
    dx, dy = w.cup_world[0] - pose[0], w.cup_world[1] - pose[1]
    return b.with_(r_cup=np.array([c * dx + s * dy, -s * dx + c * dy]))


def world_from_sim(sim, gk: GKConfig | None = None, shield_cfg=None, mass_frac: float = MASS_FRAC, sigma_star: float = 0.0,
                   held: tuple | None = None, robot_pose=None) -> World:
    """planner world from a TeleopSim built with counter=True, desk=True (the h5 'coffee' fetch world).  Reads sim truth for poses (see module doc).
    held: (item_name, Grasp) when the item is in hand (the executor knows it from the executed pick's args, which came from a C3-certified plan).
    The Grasp (with its Cert) is stored on the item (extra["held_grasp"]) and planner.init_facts turns it into Holding.
    robot_pose: optional (x, y, yaw) overriding the robot's start conf q0 (plan-prediction replan from the pose the committed trajectory predicts); None = sim.pose()."""
    import dataclasses
    gk = gk or dataclasses.replace(GKConfig(), **RUNTIME_GK)
    shield_cfg = shield_cfg or SH.ShieldCfg()
    m, d = sim.m, sim.d
    sc = sim.scene
    wp = sc.waypoints
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    surfaces = {}
    for name, geom, dock_wp in SURFACES:
        gid = next((g for g in range(m.ngeom) if gname(g) == geom), None)
        if gid is None:
            continue
        x, y, z = (float(v) for v in d.geom_xpos[gid])
        q = wp[dock_wp]
        surfaces[name] = Surface(name, x, y, _yaw(d.geom_xmat[gid]), (2 * float(m.geom_size[gid][0]), 2 * float(m.geom_size[gid][1])),
                                 z + float(m.geom_size[gid][2]), Conf(f"q_{name}", float(q[0]), float(q[1]), float(q[2]), surface=name))
    # the tray (+ cup + contents): geometry, mass box, current pose and the surface it rests on
    tray_g = next(g for g in range(m.ngeom) if gname(g) == "tray_top")
    ext = tuple(sorted((2 * float(m.geom_size[tray_g][0]), 2 * float(m.geom_size[tray_g][1])), reverse=True)) + (2 * float(m.geom_size[tray_g][2]),)
    bname = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, j) or ""
    m_cont = sum(float(m.body_mass[j]) for j in range(m.nbody) if bname(j).startswith(("ball", "bead")))
    m_tot = float(m.body_subtreemass[sim.tray]) + float(m.body_subtreemass[sim.cup]) + m_cont
    tp = (float(d.xpos[sim.tray][0]), float(d.xpos[sim.tray][1]), _yaw(d.xmat[sim.tray]))
    on = None
    for s in surfaces.values():
        r = rel_pose((s.x, s.y, s.yaw), tp)
        if abs(r[0]) <= s.size[0] / 2 and abs(r[1]) <= s.size[1] / 2:
            on = s.name
    tray = Item("tray", ext, (m_tot * (1 - mass_frac), m_tot * (1 + mass_frac)), (G3.MU_LO_DEFAULT, G3.MU_HI_DEFAULT), category="tray",
                pose=Placement("tray", on, tp[0], tp[1], float(d.xpos[sim.tray][2]), tp[2], Cert("world_state_truth", 0.0)) if on else None,
                extra=dict(model_mass=m_tot, payload_mu=(shield_cfg.mu_lo, shield_cfg.mu_hi), grasped=bool(sim.observe()["grasped"])))
    # hold offset: the scene puts each slab at the hold-pose tray position of its dock (scene.build_rig counter_at / desk_at)
    rels = {s.name: rel_pose(s.dock.xyth(), (s.x, s.y, s.yaw)) for s in surfaces.values()}
    if on is not None:
        hr = rel_pose(surfaces[on].dock.xyth(), tp)
        cp = (float(d.xpos[sim.cup][0]), float(d.xpos[sim.cup][1]), 0.0)
        cup_rel = rel_pose(surfaces[on].dock.xyth(), cp)[:2]
    else:
        hr = next(iter(rels.values()))
        cup_rel = (0.0, 0.0)
    hold_rel = {"tray": (hr[0], hr[1], hr[2])}
    pose = sim.pose() if robot_pose is None else robot_pose
    q0 = Conf("q_start", float(pose[0]), float(pose[1]), float(pose[2]), surface=None)
    docks = [(s.dock.x, s.dock.y) for s in surfaces.values()]
    bel = build_belief(sim, shield_cfg, gk, fp=GM.FP, docks=docks)
    bel0 = bel.with_(people=(), people_vel=(), people_e_p=(), people_e_v=(), people_stale=())
    routes = [("out_route", tuple(tuple(map(float, p)) for p in sc.out_route))]
    if sc.back_route is not None:
        routes.append(("back_route", tuple(tuple(map(float, p)) for p in sc.back_route)))
    if held is not None:
        ho, hg = held
        if ho != "tray":
            raise ValueError(f"held item {ho!r} is not in the world")
        if not sim.observe()["grasped"]:
            raise ValueError("held grasp given but the sim reports nothing grasped (dropped?)")
        tray.extra["held_grasp"] = hg
        tray.pose = None                     # a held item has no resting placement
    w = World(sc, q0, surfaces, {"tray": tray}, hold_rel, tuple(cup_rel), gk, shield_cfg, bel0, routes, sc.back_mode, sigma_star,
              people_obs=tuple(tuple(map(float, xy)) for xy, _ in (sim.people_state or [])), bel_people=bel,
              cup_world=(float(d.xpos[sim.cup][0]), float(d.xpos[sim.cup][1])))
    w.stats["hold_rel_by_surface"] = {k: tuple(round(x, 4) for x in v) for k, v in rels.items()}
    return w
