"""ocs/state.py -- the OBJECT-CENTRIC STATE (OCS): the single world estimate every control-path consumer reads.  SIMULATION ONLY.

One WorldState per control tick = robot proprioception/localisation + a set of ObjectState + relations + free space.  Producers fill it
(from sensors, or from ground truth for ablations); consumers (shield, zonotope rows / base filter, planner, VLM-authored IR specs) read
ONLY it.  Every field carries its provenance, so "fully perception-based" is a checkable property (WorldState.privileged() == []).

Uncertainty has two forms on purpose:
  * cov   -- Gaussian (estimators: KF / odometry / tracker); optional.
  * unc   -- a SET (tray_rep.sets.ir.Unc: position zonotope + rotation-angle bound) -- what the safety layer uses (robust IR, zonotope
             rows).  It should be CALIBRATED (e.g. conformal on residuals), not just k*sigma; experiments measure its coverage.
Design notes and the experiment plan: ocs/README.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

from world_model.v2.tray_rep.sets.ir import Pose, Unc, Shape


class Source(str, Enum):
    TRUTH = "truth"          # simulator ground truth: PRIVILEGED (allowed for eval / ablation only)
    STEREO = "stereo"        # head ZED stereo -> segments -> zonotopes
    TRACKER = "tracker"      # head-camera detector + Kalman tracker (people)
    WRIST = "wrist"          # wrist-camera estimate (grasp)
    FK = "fk"                # arm forward kinematics from encoders (+ a grasp offset)
    KF = "kf"                # other filtered estimate (ball KF, payload estimate, ...)
    ODOM = "odom"            # wheel odometry / VIO / localisation
    ENCODER = "encoder"      # proprioception (joint / finger / lift encoders, motor current)
    VLM = "vlm"              # semantic label / prior from the language model
    PRIOR = "prior"          # a fixed engineering prior (documented constant)
    MAP = "map"              # a perceived map (SLAM / mapping pass)
    EXEC = "exec"            # ADDITIVE (10-22): facts the EXECUTOR knows (what it grasped / released: weld, grasp-assist state).  Not flagged by privileged():
                             # a real executor knows its own grasp state; a sim TruthProducer writes it from sim.grasped / sim.assist.


class Kind(str, Enum):
    PAYLOAD = "payload"              # carried by the robot (tray)
    CONTENT = "content"              # sits on / in a payload (cup, ball)
    OBSTACLE = "obstacle"            # static object (bottle, laptop, furniture part)
    MOVABLE = "movable"              # object that can move but is not a person
    PERSON = "person"
    SURFACE = "surface"              # support surfaces (desk top, counter, placement stand)
    UNKNOWN = "unknown"


@dataclass
class Belief:
    """a physical-parameter belief: an interval the safety layer can trust, plus optional point estimate / std."""
    lo: float
    hi: float
    mean: float | None = None
    std: float | None = None
    source: Source = Source.PRIOR
    note: str = ""


@dataclass
class ObjectState:
    oid: str                                   # stable id (track id); never reused within an episode
    kind: Kind
    shape: Shape                               # local convex shape (ir Shape: ZonoShape / BoxShape / Sphere / Point / Region)
    pose: Pose                                 # in the frame of `parent` (None = world/map frame)
    unc: Unc = field(default_factory=Unc)      # SET uncertainty of `pose` (calibrated); empty = exact
    vel: np.ndarray | None = None              # (6,) twist [v, w] in the world frame, if estimated
    cov: np.ndarray | None = None              # (6,6) or (3,3) Gaussian covariance of pose (and/or vel), estimator-side
    parent: str | None = None                  # oid of the supporting / carrying object (cup -> tray, tray -> "robot")
    name: str | None = None                    # semantic label (VLM), e.g. "laptop"
    name_conf: float | None = None
    params: dict[str, Belief] = field(default_factory=dict)   # e.g. {"mu": Belief(0.18, 0.35), "mass": ...}
    dynamics: str = "static"                   # key into ocs.dynamics registry (static, const_vel, attached, slip_on_support, ...)
    source: Source = Source.TRUTH              # where the POSE came from (name/params carry their own sources)
    t_meas: float = 0.0                        # time of the measurement the pose is based on [s]
    visible: bool = True                       # observed at t_meas (False = predicted through occlusion)
    age: float = 0.0                           # time since first observed [s]
    extra: dict = field(default_factory=dict)
    name_source: Source | None = None          # ADDITIVE (ocs round 1): provenance of `name` (None = no name); TRUTH => flagged by privileged()
    # ---- ADDITIVE (10-22, world-facts interface).  Person fields feed the gatekeeper Belief (people_e_p / people_e_v / people_stale): None = not an
    #      estimate (truth source) -> only the global gk.e_p / gk.e_v apply.  `vel` (above) = the velocity estimate v_hat (world frame, [vx, vy, 0, 0, 0, 0]).
    pos_err: float | None = None               # m, per-person ADDITIONAL position error (tracker e_p)
    vel_err: float | None = None               # m/s, per-person ADDITIONAL velocity-estimate error (tracker e_v)
    stale: float | None = None                 # s, time since the track's last update
    affordances: tuple = ()                    # e.g. ("graspable", "support", "payload"); minimal for now, the planner migration extends it


@dataclass
class RobotState:
    base: Pose                                 # base pose in the world/map frame (estimate)
    base_unc: Unc = field(default_factory=Unc)
    base_vel: np.ndarray | None = None         # (3,) body twist (vx, vy, wz)
    base_source: Source = Source.TRUTH
    lift_h: float = 0.0
    q: dict[str, np.ndarray] = field(default_factory=dict)       # {"left": (6,), "right": (6,)} encoder positions
    qd: dict[str, np.ndarray] = field(default_factory=dict)
    gripper_gap: dict[str, float] = field(default_factory=dict)
    gripper_effort: dict[str, float] = field(default_factory=dict)
    q_source: Source = Source.ENCODER
    yaw: float | None = None                   # ADDITIVE (ocs round 1): the base yaw as the estimator reports it (exact, unwrapped); `base.R` carries the same rotation


@dataclass
class FreeSpace:
    """2-D occupancy in the world/map frame: log-odds grid + last-observed time per cell (unknown = never observed)."""
    origin: np.ndarray                         # (2,) world xy of cell (0, 0)
    res: float                                 # m / cell
    logodds: np.ndarray                        # (H, W)
    last_seen: np.ndarray                      # (H, W) time [s], -inf = never
    source: Source = Source.TRUTH


@dataclass
class Relation:
    a: str
    rel: str                                   # "on" | "in" | "held_by" | "near"
    b: str
    conf: float = 1.0
    source: Source = Source.TRUTH
    status: str = ""                           # ADDITIVE (10-22): for rel == "held_by": "closing" | "held" | "releasing" (b == "robot")


@dataclass
class WorldState:
    t: float
    robot: RobotState
    objects: dict[str, ObjectState] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)
    free: FreeSpace | None = None
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ queries
    def by_kind(self, *kinds: Kind) -> list[ObjectState]:
        return [o for o in self.objects.values() if o.kind in kinds]

    def by_name(self, name: str) -> ObjectState | None:
        for o in self.objects.values():
            if o.name == name or o.oid == name:
                return o
        return None

    def attached(self, oid: str, default: bool = False) -> bool:
        """True iff `oid` is on the robot: a Relation(oid, "held_by", "robot", status) in ANY status (closing / held / releasing = the conservative rule
        of teleop.geom.cup_attached).  A producer that KNOWS the object is free lists it in meta["attachment_known"] (-> False); if no producer said
        anything about `oid` the answer is `default` (pass True for the conservative 'unknown = attached')."""
        for r in self.relations:
            if r.a == oid and r.rel == "held_by" and r.b == "robot":
                return True
        return False if oid in self.meta.get("attachment_known", ()) else bool(default)

    def world_pose(self, oid: str) -> Pose:
        """compose parent chain -> world pose (parent "robot" = the base frame)."""
        o = self.objects[oid]
        R, p = np.asarray(o.pose.R), np.asarray(o.pose.p)
        if o.parent is None:
            return Pose(R, p)
        P = self.robot.base if o.parent == "robot" else self.world_pose(o.parent)   # recursion resolves the rest of the chain
        Rb, pb = np.asarray(P.R), np.asarray(P.p)
        return Pose(Rb @ R, Rb @ p + pb)

    # ------------------------------------------------------------------ honesty audit
    def privileged(self) -> list[str]:
        """every field filled from simulator ground truth. Fully perception-based <=> this is empty."""
        out = []
        if self.robot.base_source == Source.TRUTH:
            out.append("robot.base")
        if self.robot.q_source == Source.TRUTH:
            out.append("robot.q")
        if self.free is not None and self.free.source == Source.TRUTH:
            out.append("free")
        for o in self.objects.values():
            if o.source == Source.TRUTH:
                out.append(f"{o.oid}.pose")
            if o.name is not None and o.name_source == Source.TRUTH:
                out.append(f"{o.oid}.name")
            for k, b in o.params.items():
                if b.source == Source.TRUTH:
                    out.append(f"{o.oid}.params.{k}")
        for r in self.relations:
            if r.source == Source.TRUTH:
                out.append(f"rel:{r.a}-{r.rel}-{r.b}")
        return out

    def assert_honest(self, allow: tuple[str, ...] = ()):
        bad = [f for f in self.privileged() if not any(f.startswith(a) for a in allow)]
        if bad:
            raise AssertionError(f"privileged fields in a perception-only WorldState: {bad}")


# ---------------------------------------------------------------------- producer / consumer protocols
class Producer:
    """fills / updates part of a WorldState from one sensor stream (or from truth, for ablations).  Stateful (trackers, filters)."""
    name = "producer"

    def reset(self, sim) -> None: ...

    def update(self, ws: WorldState, sim, t: float) -> None:
        """read ONLY this producer's sensor from `sim` (render a camera, read encoders, ...) and write into `ws`."""
        raise NotImplementedError


Consumer = Callable[[WorldState], object]       # e.g. to_ir_entities, to_people_state, to_obstacle_zonos, to_occ
