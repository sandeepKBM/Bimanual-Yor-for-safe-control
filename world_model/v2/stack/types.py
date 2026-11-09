"""stack/types.py -- the shared interfaces of the skills + planner stack (USER-approved plan 2026-11-03,
/common/home/ss5772/.claude/plans/zesty-skipping-swing.md).  SIMULATION ONLY.

Layers (top -> bottom):  Proposer -> Planner -> Skills -> Checkers -> Executor/Tracker -> WorldState <- Producers -> Universe -> MuJoCo.
Every module talks to its neighbours ONLY through the types below.  The world state is house/ocs WorldState (re-exported here): each field
carries its Source, so "the planner never read sim truth" is checkable (WorldState.assert_honest).

Bounds rule (AGENTS.md): every number a checker/planner/certificate uses is tagged MEASURED / MODEL / ASSUMED (`Bound`).
Guarantee rule: every admission names its type: proof | sampled | refuted (`Verdict.guarantee`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, Sequence, runtime_checkable

import numpy as np

from world_model.v2.house.ocs.state import (Belief, Kind, ObjectState, Producer, RobotState, Source,  # noqa: F401  (re-export)
                                            WorldState)

Tag = Literal["MEASURED", "MODEL", "ASSUMED"]
Guarantee = Literal["proof", "sampled", "refuted", "missing"]
SIDES = ("left", "right")


# ----------------------------------------------------------------------------------------------------------------------- bounds / worlds
@dataclass(frozen=True)
class Bound:
    """one labelled interval.  `src` = file + n + statistic (MEASURED), formula/function (MODEL) or who chose it (ASSUMED)."""
    lo: float
    hi: float
    tag: Tag
    src: str = ""
    unit: str = ""


@dataclass(frozen=True)
class ThetaBox:
    """the prior parameter box of a universe: name -> Bound.  Worlds are points in it; certificates are stated on it."""
    fields: dict[str, Bound]

    def assumptions(self) -> dict:
        return {k: dict(lo=b.lo, hi=b.hi, tag=b.tag, src=b.src, unit=b.unit) for k, b in self.fields.items()}


@dataclass(frozen=True)
class WorldSpec:
    """one world = one point of a universe's ThetaBox + the seed of everything random in it.  (universe, world_id, seed) is the key
    that makes two methods' results comparable (AGENTS.md rule 8: same world set)."""
    universe: str
    world_id: str                 # stable, e.g. "box/cell(3.5,4.0)/i0" or a pan tag
    theta: dict                   # universe-native parameters (JSON-able)
    seed: int
    native: Any = None            # the universe's own world object (TrayWorld / BoxWorld / pan dict / DemoCParams / ...), not serialized


@dataclass
class Sim:
    """a built world: MuJoCo model + data + named handles the universe exposes (body/site/equality ids, actuator ids)."""
    m: Any
    d: Any
    world: WorldSpec
    handles: dict = field(default_factory=dict)
    episode: Any = None           # legacy episode object (CarryEpisode / pan Episode) while the migration is in progress


@dataclass(frozen=True)
class TaskSpec:
    """what to achieve, never how: object, goal region, constraints.  Goals are poses/regions in the world frame."""
    name: str                                  # e.g. "carry", "serve", "turnover"
    obj: str                                   # object id in WorldState
    goal: dict                                 # e.g. {"pose": [x,y,z,qw,qx,qy,qz], "tol_m": 0.03} or {"region": ...}
    limits: dict = field(default_factory=dict) # name -> Bound (tilt, speed, torque, ...)


# ----------------------------------------------------------------------------------------------------------------------- plans
Mode = Literal["approach", "pregrasp", "grasp", "lift", "carry", "place", "release", "retreat", "probe", "observe", "backup", "hold"]


@dataclass
class Segment:
    """one mode of a plan.  q_ref(t) -> {side: (q, qd, qdd)} for t in [t0, t1] (seconds from the segment start);
    x_obj(t) -> object pose (7,) for carry-type modes; tau_ff(t) optional feedforward; weld = which sides are welded during it."""
    mode: Mode
    t0: float
    t1: float
    q_ref: Callable[[float], dict] | None = None
    x_obj: Callable[[float], np.ndarray] | None = None
    tau_ff: Callable[[float], dict] | None = None
    weld: dict = field(default_factory=lambda: {s: False for s in SIDES})
    gripper: dict = field(default_factory=dict)           # side -> opening fraction target (gripper.py convention)
    meta: dict = field(default_factory=dict)               # planner-specific (legacy phases list, IK errors, margins)


@dataclass
class BimanualPlan:
    segments: list[Segment]
    planner: str                                           # registry name
    provenance: dict = field(default_factory=dict)         # planner params, proposer model + reasoning level, seeds
    assumptions: dict = field(default_factory=dict)        # name -> Bound-like dict (labelled)


@dataclass
class Verdict:
    """a checker's answer on a plan."""
    checker: str
    admit: bool
    guarantee: Guarantee
    margins: dict = field(default_factory=dict)            # name -> float (>0 = inside)
    assumptions: dict = field(default_factory=dict)
    why: str = ""


@dataclass
class SkillCall:
    """one node of a task graph (what a proposer emits): skill name + goal-level arguments.  No waypoints."""
    skill: str
    args: dict = field(default_factory=dict)
    sides: tuple = SIDES


# ----------------------------------------------------------------------------------------------------------------------- protocols
@runtime_checkable
class Universe(Protocol):
    name: str

    def theta_box(self, task: str | None = None) -> ThetaBox: ...      # per-task box when a universe holds several tasks (contents)
    def worlds(self, set_name: str, n: int | None = None, seed: int = 0) -> list[WorldSpec]: ...
    def build(self, world: WorldSpec) -> Sim: ...
    def tasks(self, world: WorldSpec) -> list[TaskSpec]: ...
    def producers(self, mode: str = "truth") -> list[Producer]: ...
    def success(self, sim: Sim, task: TaskSpec) -> dict: ...          # {"ok": bool, ...metrics}
    def assumptions(self) -> dict: ...


@runtime_checkable
class Skill(Protocol):
    name: str

    def pre(self, ws: WorldState, call: SkillCall) -> tuple[bool, str]: ...
    def plan(self, ws: WorldState, call: SkillCall, planner: "Planner", sim: Sim) -> BimanualPlan: ...
    def post(self, ws: WorldState, call: SkillCall) -> tuple[bool, str]: ...


@runtime_checkable
class Planner(Protocol):
    name: str

    def plan(self, ws: WorldState, task: TaskSpec | SkillCall, sim: Sim, theta: ThetaBox | None = None) -> BimanualPlan: ...


@runtime_checkable
class Checker(Protocol):
    name: str

    def check(self, plan: BimanualPlan, ws: WorldState, sim: Sim, theta: ThetaBox | None = None) -> Verdict: ...


@runtime_checkable
class Executor(Protocol):
    def run(self, plan: BimanualPlan, sim: Sim, monitors: Sequence[Any] = ()) -> dict: ...   # -> record (segments, viol, truth_end, ...)
