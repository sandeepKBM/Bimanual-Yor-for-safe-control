"""scenarios/occlusion.py -- Scenario 2: a person walks out from behind a wall corner into the carry route.  SIMULATION ONLY.

h5 geometry (see README): after the doorway plane (y = 3.3) the route runs south through the lower room.  A wall (y = 3.3, x in [0, 0.94], ending at the
west doorway post) separates the north passage from the part of the lower room west of x ~ 1.0.  The person starts STANDING at `hide` = (0.45, 2.9) behind
that wall (outside the head camera's view from every pose of the north approach, checked in tests), walks east past the corner `corner` = (0.95, 2.9) (the moment it leaves cover),
crosses the route centreline at `cross` = (2.134, 2.9) and leaves at `exit` = (3.0, 2.9).  Timeline (all sim s):

    t_app = t_appear + jitter_s * U(-1, 1)            seeded: rng(seed), same seed => same t_app
    stand at `hide` until t_app - |hide->corner| / speed, then walk hide -> corner -> cross -> exit at `speed` (<= 1.2 m/s)

i.e. `t_appear` is when the person passes the corner (leaves cover); it reaches the route (cross) |corner->cross| / speed later (1.18 s at 1 m/s).
With no people the nominal operator reaches the doorway plane at ~t = 5 s (the route point (2.13, 2.9) at ~5.5 s), so t_appear ~ 4..5 s is a crossing
conflict.  If t_app is so small that the person would have to start before t = 0 the start point is moved along the path (it then starts walking at t = 0).

The scene person is a normal mocap capsule (r = 0.22); with ScenSim (people_clock="time") the clock runs with sim time.  `oblivious=True` makes it never
yield to the robot (PersonPolicy.yield_m = None); default = the standard policy (stops while the robot footprint is < 0.15 m from the torso).
`ScenSim` logs the head-camera visibility of the person (`vis_log`, people_track Visibility: frustum + line of sight) -> `first_visible_t`.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from ..teleop import scene as SC
from ..teleop.sim import PersonPolicy
from . import ScenarioError, scene_key, register

NAME = "p_occ"
V_MAX = 1.2


@dataclass(frozen=True)
class OcclusionSpec:
    t_appear: float = 4.5
    speed: float = 1.0
    jitter_s: float = 0.5
    seed: int = 0
    hide: tuple = (0.45, 2.9)
    corner: tuple = (0.95, 2.9)
    cross: tuple = (2.134, 2.9)
    exit: tuple = (3.0, 2.9)
    oblivious: bool = False


def effective_t_appear(spec):
    rng = np.random.default_rng(7_000_000 + int(spec.seed))
    return float(spec.t_appear + spec.jitter_s * (2.0 * rng.random() - 1.0))


def track_points(spec):
    """rows (t, x, y) of the person's track (deterministic in the spec; the seed enters only through the timing jitter)."""
    if not (0.0 < spec.speed <= V_MAX):
        raise ScenarioError(f"speed must be in (0, {V_MAX}] m/s, got {spec.speed}")
    H, K, C, E = (np.asarray(p, float) for p in (spec.hide, spec.corner, spec.cross, spec.exit))
    t_app = max(effective_t_appear(spec), 0.0)
    lead = float(np.linalg.norm(K - H))
    t_start = t_app - lead / spec.speed
    rows = []
    if t_start > 1e-9:
        rows += [[0.0, *H], [t_start, *H]]
    else:                                                  # would have to start before t = 0: start walking at t = 0 from a point closer to the corner
        H = K - (K - H) / max(lead, 1e-9) * (spec.speed * t_app) if lead > 1e-9 else K
        rows += [[0.0, *H]]
    t = max(t_start, 0.0)
    prev = H
    for P in (K, C, E):
        dt = float(np.linalg.norm(P - prev)) / spec.speed
        t += dt
        rows.append([t, *P])
        prev = P
    rows.append([1e4, *E])
    a = np.array(rows, float)
    keep = np.concatenate([[True], np.diff(a[:, 0]) > 1e-9])       # strictly increasing time (a zero-length leg is dropped)
    return a[keep]


def build_scene(spec):
    sc = SC.get_scene("h5")
    g = SC.load_occupancy(sc)
    pts = track_points(spec)
    for _, x, y in pts[:: max(1, len(pts) - 2)]:
        if float(g.lookup(x, y)) < SC.PERSON_R:
            raise ScenarioError(f"occlusion track point ({x:.2f}, {y:.2f}) is inside an obstacle (EDT {float(g.lookup(x, y)):.2f} m)")
    sc.people = [SC.PersonTrack(NAME, pts)]
    sc.note = f"h5 occlusion scenario: person emerges at t={effective_t_appear(spec):.2f} s, v={spec.speed} m/s"
    return sc


def install(spec):
    key = scene_key("occlusion", **asdict(spec))
    policy = PersonPolicy(yield_m=None) if spec.oblivious else None
    register(key, lambda: build_scene(spec), policy=policy, watch=True)
    return key
