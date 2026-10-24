"""tamp/types.py -- the objects a PDDLStream plan carries between the planner (planner.py / streams.py) and the executor (executor.py).  SIMULATION ONLY.

All are frozen dataclasses of plain floats / tuples / strings: PDDLStream needs hashable values, and plans are logged as JSON (`to_json`).
Frames: map frame (x, y in m, yaw in rad), as TeleopSim.pose() / SceneSpec.waypoints.

Plan = list of (action_name, args) with these argument types (domain pddl/fetch_domain.pddl):
    ("move_free",    (q1: Conf, q2: Conf, r: Route))                  base drives route r with empty hands; ends docked at q2 if q2.surface is set
    ("move_holding", (o: str, g: Grasp, q1: Conf, q2: Conf, r: Route)) same, carrying o (cup-acceleration rows active)
    ("pick",         (o: str, p: Placement, g: Grasp, q: Conf))        at dock conf q: align hands, grasp o, lift it off its support
    ("place",        (o: str, p: Placement, g: Grasp, q: Conf, s: str)) at dock conf q: lower o onto surface s, release
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields


@dataclass(frozen=True)
class Conf:
    """base configuration.  surface: name of the surface this conf docks at (None = free conf, e.g. the start pose)."""
    name: str
    x: float
    y: float
    yaw: float
    surface: str | None = None

    def xyth(self):
        return (self.x, self.y, self.yaw)

    def __repr__(self):
        return f"Conf({self.name})"


@dataclass(frozen=True)
class Cert:
    """certificate summary attached to a stream output: which checker admitted it, under which belief, with what worst margin.
    ok is always True for an output the planner sees (rejected samples are never yielded); `detail` is a tuple of (key, value) pairs."""
    checker: str                    # "C1C2_static_route" | "C2_dock_static" | "C3_grasp_corners" | "support_footprint"
    margin: float                   # worst margin over the belief (>= 0)
    detail: tuple = ()

    def to_json(self):
        return dict(checker=self.checker, margin=self.margin, **{k: v for k, v in self.detail})


@dataclass(frozen=True)
class Route:
    """a base route from conf q1 to conf q2: xy polyline pts (first ~ q1, last ~ q2).  reverse=True: the base drives BACKWARDS along it
    (heading = opposite of the travel direction, e.g. h5 cafe -> lab).  kind: "free" (hands empty) | "holding" (payload rows active)."""
    q1: str
    q2: str
    pts: tuple                      # ((x, y), ...)
    reverse: bool
    kind: str
    cert: Cert | None = None
    source: str = "scene"           # "scene" (SceneSpec route) | "nav" (planned)
    # --- cert_mode="segments" only (None in the legacy from-rest mode; see tamp/segments.py) ---
    v_max: float | None = None      # planner-chosen speed cap [m/s]
    s_stop: float | None = None     # arclength [m] at which the segment-driven part ends (the rest is the executor's final approach)
    segments: tuple | None = field(default=None, compare=False, repr=False)      # tuple[RouteSegment] (numpy inside -> excluded from eq / hash)

    @property
    def length(self):
        import math
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(self.pts[:-1], self.pts[1:]))

    def __repr__(self):
        return f"Route({self.q1}->{self.q2},{self.kind}{',rev' if self.reverse else ''},{self.length:.1f}m)"


@dataclass
class RouteSegment:
    """one certified timed segment of a route, in the gatekeeper's Traj format: V (N+1,3) body-twist samples on the grid h = nominal part (t_sw s, starts at the
    PREDICTED moving state at the end of the previous segment) followed by the cup-safe backup to rest.  pose_start: predicted map pose at the segment start;
    s_start / s_end: arclength along the route polyline at the start / at the end of the nominal part; t_start: time since the route start."""
    V: object
    h: float
    t_sw: float
    T_S: float
    pose_start: tuple
    s_start: float
    s_end: float
    t_start: float = 0.0
    margin: float = float("nan")
    binding: str = ""
    offset: float = 0.0                 # m, lateral route offset of this segment (left normal +; 0 = on the route)
    ready: int = -1                    # C2-ready score of the chosen candidate (-1 = ranking off)

    def traj(self, t0: float = 0.0):
        from ..gatekeeper.traj import Traj
        import numpy as np
        return Traj(np.array(self.V, float).copy(), self.h, t0, self.t_sw, self.T_S)

    def summary(self):
        d = dict(s_start=round(self.s_start, 3), s_end=round(self.s_end, 3), t_start=round(self.t_start, 2), t_sw=round(self.t_sw, 2),
                 margin=None if self.margin != self.margin else round(float(self.margin), 4), binding=self.binding)
        if self.offset != 0.0:                       # opt-in fields appear only when used (default JSON keys unchanged)
            d["offset"] = round(float(self.offset), 3)
        if self.ready >= 0:
            d["ready"] = int(self.ready)
        return d


@dataclass(frozen=True)
class Grasp:
    """a grasp of object obj admitted by Checker 3 for EVERY (mass, CoM, mu, accel) corner of the object's parameter box."""
    obj: str
    name: str                       # grasp3 candidate name (e.g. "bimanual_squeeze", "rim", "handle", "top")
    arms: int
    mass_range: tuple               # (lo, hi) kg the certificate covers
    mu_range: tuple
    cert: Cert | None = None

    def __repr__(self):
        return f"Grasp({self.obj},{self.name},{self.arms}arm)"


@dataclass(frozen=True)
class Placement:
    """pose of object obj resting on surface `surface` (centre x, y, z of the object; yaw of its long axis)."""
    obj: str
    surface: str
    x: float
    y: float
    z: float
    yaw: float
    cert: Cert | None = None

    def __repr__(self):
        return f"Placement({self.obj}@{self.surface})"


@dataclass
class Surface:
    """a support surface (world state, not a PDDL object value): slab centre, yaw, size (x along its own axis), top height, and the dock conf to reach it."""
    name: str
    x: float
    y: float
    yaw: float
    size: tuple                     # (sx, sy) m
    top_z: float
    dock: Conf


@dataclass
class Item:
    """a movable object (world state): its parameter BOX (what the checkers must hold for), geometry and current pose."""
    name: str
    extent: tuple                   # (a, b, h) m, a >= b
    mass_range: tuple               # (lo, hi) kg  (ASSUMPTION box around the model value)
    mu_range: tuple                 # payload friction set (cup on tray: shield mu_lo..mu_hi)
    category: str | None = None
    pose: Placement | None = None
    extra: dict = field(default_factory=dict)


def to_json(x):
    """JSON-safe view of a plan argument (dataclass -> dict, recursively)."""
    if isinstance(x, Route):                     # segments hold numpy profiles: log a summary, not the arrays
        d = {f.name: to_json(getattr(x, f.name)) for f in fields(x) if f.name != "segments"}
        d["segments"] = None if x.segments is None else [sg.summary() for sg in x.segments]
        d["_type"] = "Route"
        return d
    if hasattr(x, "__dataclass_fields__"):
        d = asdict(x)
        d["_type"] = type(x).__name__
        return d
    if isinstance(x, (list, tuple)):
        return [to_json(v) for v in x]
    return x


def plan_to_json(plan):
    return [dict(action=name, args=[to_json(a) for a in args]) for name, args in (plan or [])]
