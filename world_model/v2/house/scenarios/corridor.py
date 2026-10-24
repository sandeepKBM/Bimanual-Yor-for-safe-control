"""scenarios/corridor.py -- Scenario 1: ONE standing person beside the carry route at a controlled lateral offset d (the livelock curve).  SIMULATION ONLY.

h5 geometry (teleop/scene.py `_h5` route + the 1 cm occupancy grid, see README): the out route runs lab_start (3.1, 6.09) -> door1 (2.13, 5.08) -> south along
x = 2.134 to the cafe counter (2.135, 2.0).  Between y = 4.2 and 3.3 it crosses a 1.86 m wide wall-bounded passage (walls x = 0.94 / 2.80; the route is
0.67 m from the east wall, 1.15 m from the west wall) and the doorway plane y = 3.3.  The robot footprint (teleop/geom.FP) is asymmetric about the base
origin that follows the route: 0.64 m to the RIGHT of travel (west; tray + arms), 0.13 m to the left (east).  So the person is put on the RIGHT (west)
by default: torso-vs-footprint contact starts at d ~= 0.86 and the d = 0.4..1.4 sweep goes from "person inside the carry swath" (the planner must
re-route east / hold) to "free".  Default arc s = 3.6 (y = 2.88, 0.6 m past the doorway plane) keeps the whole d range inside free space (the person
must stand on free floor: max valid d is 2.4 there; in the passage itself only <= 0.91 on the right / 0.40 on the left is free) -- `arc_s`, `side` ('right' |
'left') are parameters; an invalid placement raises ScenarioError.

The person is a normal scene person (mocap capsule, r = 0.22 m) with a constant 2-row track (zero velocity forever), so every people consumer
(sim.people_state, shield, gatekeeper, head-camera tracker) sees a standing person.  Other scene people (p_walker, p_stand) are removed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from ..teleop import scene as SC
from . import ScenarioError, scene_key, register, route_point, dist_to_route

NAME = "p_corr"


@dataclass(frozen=True)
class CorridorSpec:
    d: float = 0.8              # m, distance from the person's centre to the route centreline (perpendicular to the route at arc_s)
    arc_s: float = 3.6          # m along sc.out_route (route length 4.476 m)
    side: str = "right"         # 'right' = right of travel direction (west on the southbound segment, the footprint-heavy side) | 'left'
    clear_margin: float = 0.02  # m, extra free space required around the torso when validating against the occupancy grid


def person_xy(spec, route=None):
    route = route if route is not None else SC.get_scene("h5").out_route
    xy, t = route_point(route, spec.arc_s)
    left = np.array([-t[1], t[0]])
    if spec.side not in ("left", "right"):
        raise ScenarioError(f"side must be 'left' or 'right', got {spec.side!r}")
    return xy + (left if spec.side == "left" else -left) * spec.d


def max_valid_d(spec, d_hi=3.0, step=0.01):
    """largest d (on spec's arc / side) up to which the whole torso circle stays on free floor (contiguously from d=0)."""
    g = SC.load_occupancy(SC.get_scene("h5"))
    best = 0.0
    for d in np.arange(step, d_hi, step):
        xy = person_xy(CorridorSpec(float(d), spec.arc_s, spec.side))
        if float(g.lookup(*xy)) >= SC.PERSON_R + spec.clear_margin:
            best = float(d)
        else:
            break
    return best


def validate(spec, scene=None):
    scene = scene or SC.get_scene("h5")
    xy = person_xy(spec, scene.out_route)
    g = SC.load_occupancy(scene)
    free = float(g.lookup(*xy))
    if free < SC.PERSON_R + spec.clear_margin:
        raise ScenarioError(f"corridor person at d={spec.d} arc_s={spec.arc_s} side={spec.side} ({xy.round(3).tolist()}) is inside a wall/obstacle "
                            f"(EDT {free:.2f} m < torso radius {SC.PERSON_R}); max valid d here is {max_valid_d(spec):.2f} m")
    return xy


def build_scene(spec):
    """fresh h5 SceneSpec whose only person is the standing one (deterministic, seed-independent)."""
    sc = SC.get_scene("h5")
    xy = validate(spec, sc)
    sc.people = [SC.PersonTrack(NAME, np.array([[0.0, xy[0], xy[1]], [1e4, xy[0], xy[1]]]))]
    sc.note = f"h5 corridor scenario: standing person d={spec.d} arc_s={spec.arc_s} side={spec.side}"
    return sc


def install(spec):
    """register the scenario; returns the scene key to pass as `scene=` to run_episode / run_one."""
    validate(spec)
    key = scene_key("corridor", **asdict(spec))
    register(key, lambda: build_scene(spec), policy=None, watch=False)
    return key
