"""stack/universes/pot.py -- PotUniverse: the pot (two side handles) / pan (one handle) on the stove, YOR-NERO + NYU grippers.  SIMULATION ONLY.

Parameter box (theta_box): pan/theta.py THETA, copied with its labels (user-approved values, NOT changed here).  Worlds are points of the box ->
theta.world_from_theta -> the pan Episode's world dict -> pan/covered.build_spec (pot = handle2_len > 0; pan = 0).
Sets (worlds(set_name, n, seed)):  "nominal" (1 world), "corners" (n seeded-random vertices of the box: SAMPLED, the full 2^9 are not run),
"random" (n uniform draws, rng = default_rng(seed + i)), "all" (nominal + corners + random).
The nominal point is NOT in THETA: NOMINAL below (every value ASSUMED, picked by the code author 11-04, needs user OK).
"""
from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from world_model.v2.house.ocs.state import Belief, Kind, ObjectState, Producer, RobotState, Source, WorldState
from world_model.v2.tray_rep.sets.ir import BoxShape, Pose
from ..types import Bound, Sim, TaskSpec, ThetaBox, WorldSpec, SIDES

SETS = ("nominal", "corners", "random", "all")
# ASSUMED (11-04, the code author's pick, needs user OK): the nominal point.  Each value sits inside its THETA range: unscaled friction (1.0), the puck
# at the centre, the covered.py hidden_mass default, and handle_len = the THETA midpoint 0.11.  NOT 0.13 (scene default): with the NYU gripper the right
# claw has no passing pre-grasp on handle2 at 0.12-0.13 (IK reach + palm-on-handle, checked 11-04), passes at 0.09-0.11.
NOMINAL = dict(dx=0.0, dy=0.0, yaw_deg=0.0, mu_scale=1.0, mass=None, c_r=0.0, c_phi=0.0, lid_mu_scale=1.0, handle_len=0.11)
_PRIOR_PARAMS = ("mass", "mu_scale", "lid_mu_scale", "handle_len")      # object parameters the producer exposes as Beliefs over the PRIOR box


def _tag(label: str) -> str:
    return "MODEL" if label.startswith("MODEL") else "ASSUMED"


def _box() -> ThetaBox:
    from ...nero.pan.theta import THETA
    return ThetaBox({k: Bound(lo, hi, _tag(lab), lab) for k, (lo, hi, lab) in THETA.items()})


def nominal_theta() -> dict:
    from ...nero.pan import covered as CV
    th = dict(NOMINAL)
    th["mass"] = float(CV.ASSUMED_COVERED["hidden_mass"])
    return th


def build_model(world: dict, obj: str = "pot"):
    """compiled model + data for a world dict (theta.world_from_theta): covered pan/pot scene, weld equalities (inactive), friction scaling, settled,
    pose perturbation applied.  Mirrors segments.Episode.__init__ (pan/pot branch, grasp='weld')."""
    from ...nero.pan import covered as CV, scene as SC
    from ...nero.pan.rim_geometry import _perturb_pose
    from ...nero.pan.segments import POT_HANDLE2_LEN
    cp = CV.CoveredParams(placement=world["placement"])
    if world.get("hidden_mass") is not None:
        cp = dataclasses.replace(cp, hidden_mass=world["hidden_mass"], overload_mass=world["hidden_mass"])
    hl = float(world.get("handle_len") or cp.pan.handle_len)
    cp = dataclasses.replace(cp, pan=dataclasses.replace(cp.pan, handle_len=hl if obj == "pot" else cp.pan.handle_len,
                                                         handle2_len=(hl if world.get("handle_len") else POT_HANDLE2_LEN) if obj == "pot" else 0.0))
    spec = CV.build_spec(cp)
    for s in SIDES:
        spec.add_equality(name=f"weld_{s}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                          name1=f"{s}_arm_gripper_base", name2="pan", active=False)
    m = SC.apply_sim_defaults(spec.compile())
    pt = dict(world.get("perturb") or {})
    if pt.get("mu_scale"):
        m.geom_friction[:, 0] *= pt["mu_scale"]
    if pt.get("lid_mu_scale"):
        sel = m.geom_bodyid == m.body("lid").id
        m.geom_friction[sel, 0] *= pt["lid_mu_scale"]
        m.geom_priority[sel] = 1
    d = mujoco.MjData(m)
    SC.settle(m, d, 1.0)
    if any(pt.get(k) for k in ("dx", "dy", "yaw_deg")):
        _perturb_pose(m, d, cp.pan.pan_centre_xy, pt.get("dx", 0.0), pt.get("dy", 0.0), pt.get("yaw_deg", 0.0))
        SC.settle(m, d, 0.5)
    return m, d, cp


class PotTruthProducer(Producer):
    """pot ObjectState straight from the simulator (Source.TRUTH => WorldState.privileged() lists 'pot.pose'); robot q from the encoders;
    object parameters = Beliefs over the PRIOR box (never the sampled world's values)."""
    name = "truth"

    def __init__(self, box: ThetaBox | None = None):
        self._box = box

    def reset(self, sim=None) -> None:
        pass

    def update(self, ws: WorldState, sim, t: float) -> None:
        m, d = sim.m, sim.d
        from ...nero import adapters as A
        q = {s: np.asarray(d.qpos[A.arm_addrs(m, s)[0]], float).copy() for s in SIDES}
        qd = {s: np.asarray(d.qvel[A.arm_addrs(m, s)[1]], float).copy() for s in SIDES}
        ws.t = float(t)
        ws.robot = RobotState(base=Pose(np.eye(3), np.zeros(3)), base_source=Source.PRIOR, q=q, qd=qd, q_source=Source.ENCODER)
        p = sim.handles["params"]
        box = self._box or _box()
        params = {k: Belief(b.lo, b.hi, None, None, Source.PRIOR, f"{b.tag}: {b.src}") for k, b in box.fields.items() if k in _PRIOR_PARAMS}
        b = m.body("pan").id
        reach = p.pan_radius + p.pan_wall_t + max(p.handle_len, p.handle2_len)     # bounding box about the body origin (the underside centre)
        half = np.array([p.pan_radius + p.pan_wall_t, reach, 0.5 * (p.pan_bottom_t + p.pan_wall_h)])
        name = "pot" if p.handle2_len > 0 else "pan"
        ws.objects = {name: ObjectState(name, Kind.MOVABLE, BoxShape(half),
                                        Pose(np.asarray(d.xmat[b], float).reshape(3, 3).copy(), np.asarray(d.xpos[b], float).copy()),
                                        name=name, name_source=Source.PRIOR, dynamics="static", source=Source.TRUTH, t_meas=float(t),
                                        params=params, affordances=("graspable",))}
        ws.meta["producers"] = ws.meta.get("producers", []) + ["pot_truth"]


class PotUniverse:
    name = "pot"

    def __init__(self, obj: str = "pot"):
        assert obj in ("pot", "pan"), obj
        self.obj = obj          # "pot": two handles | "pan": one handle (handle2_len = 0)

    def theta_box(self, task: str | None = None) -> ThetaBox:
        return _box()

    def nominal_theta(self) -> dict:
        th = nominal_theta()
        assert set(th) == set(self.theta_box().fields)
        return th

    def _spec(self, wid, th, seed):
        from ...nero.pan.theta import world_from_theta
        return WorldSpec(self.name, wid, th, seed, world_from_theta(th))

    def worlds(self, set_name: str, n: int | None = None, seed: int = 0) -> list[WorldSpec]:
        if set_name not in SETS:
            raise ValueError(f"unknown world set {set_name!r}; have {SETS}")
        fl = self.theta_box().fields
        keys = list(fl)
        out = []
        if set_name in ("nominal", "all"):
            out.append(self._spec("pot/nominal", self.nominal_theta(), seed))
        if set_name in ("corners", "all"):
            rng = np.random.default_rng(seed)
            for i in range(int(n if n is not None else 16)):
                bits = rng.integers(0, 2, len(keys))
                th = {k: (fl[k].hi if b else fl[k].lo) for k, b in zip(keys, bits)}
                out.append(self._spec(f"pot/corner{i}", th, seed + i))
        if set_name in ("random", "all"):
            for i in range(int(n if n is not None else 16)):
                rng = np.random.default_rng(seed + i)
                th = {k: float(rng.uniform(fl[k].lo, fl[k].hi)) for k in keys}
                out.append(self._spec(f"pot/rand{i}", th, seed + i))
        return out

    def build(self, world: WorldSpec) -> Sim:
        m, d, cp = build_model(world.native, self.obj)
        mujoco.mj_forward(m, d)
        geoms = {"pot": "pan_handle"}
        return Sim(m, d, world, handles=dict(obj_geoms=geoms, obj_joint={"pot": "pan_free"}, obj_attached_joints={"pot": ("lid_free",)},
                                             benign_non_robot="base", params=cp.pan, covered=cp,
                                             handle_geoms=("pan_handle", "pan_handle2") if self.obj == "pot" else ("pan_handle",)))

    def object_id(self) -> str:
        return "pot" if self.obj == "pot" else "pan"

    def tasks(self, world: WorldSpec) -> list[TaskSpec]:
        return [TaskSpec("pregrasp", self.object_id(), goal=dict(note="both hands in a contact-free pre-grasp with the jaws straddling the handle(s)"
                                                                  if self.obj == "pot" else "left hand pre-grasp on the pan handle"), limits={})]

    def producers(self, mode: str = "truth"):
        if mode != "truth":
            raise NotImplementedError(f"pot producers: only 'truth' exists; got {mode!r}")
        return [PotTruthProducer()]

    def initial_state(self, sim: Sim) -> WorldState:
        ws = WorldState(0.0, RobotState(base=Pose(np.eye(3), np.zeros(3))))
        for pr in self.producers("truth"):
            pr.update(ws, sim, 0.0)
        return ws

    def grasp_features(self, sim: Sim) -> dict:
        """PreGrasp `features` arg for this scene: the free-end face of each handle (skills.pregrasp.handle_features), from the scene's PanParams.
        ASSUMED: the handle geometry (length, section) is known to the planner (a prior / perception of the handle tip), not read from sim truth poses."""
        from ..skills import pregrasp as PG
        return PG.handle_features(sim.handles["params"])

    def pregrasp_call(self, sim: Sim, constraints: dict | None = None):
        from ..types import SkillCall
        sides = SIDES if self.obj == "pot" else ("left",)
        feats = self.grasp_features(sim)
        return SkillCall("pregrasp", dict(obj=self.object_id(), features=feats, constraints=dict(constraints or {})), tuple(sides))

    @staticmethod
    def apply_pregrasp(sim: Sim, plan) -> None:
        """put the sim's arms/fingers at the plan's pre-grasp (jaws open), so success() can be evaluated."""
        from ...nero import adapters as A, gripper as G
        seg = plan.segments[0]
        for s, q in seg.meta["q_target"].items():
            sim.d.qpos[A.arm_addrs(sim.m, s)[0]] = q
            G.set_opening(sim.m, sim.d, s, seg.gripper[s])
        mujoco.mj_forward(sim.m, sim.d)

    def success(self, sim: Sim, task: TaskSpec) -> dict:
        """pregrasp: with the arms where they are, (a) no robot contact with the jaws as they are, (b) closing the gripper(s) of the planned side(s)
        makes BOTH jaws touch that side's handle.  Sides = those whose arm is away from home (q != home) -> here: the sides of the handles present.
        Evaluated on a copy of the data; the sim is not modified."""
        from ...nero import gripper as G
        from ..skills.pregrasp import jaws_straddle
        m = sim.m
        d = mujoco.MjData(m)
        d.qpos[:] = sim.d.qpos
        mujoco.mj_forward(m, d)
        robot = m.body("base").id
        bad = [(m.geom(c.geom1).name, m.geom(c.geom2).name) for c in d.contact[:d.ncon]
               if (m.body_rootid[m.geom_bodyid[c.geom1]] == robot) or (m.body_rootid[m.geom_bodyid[c.geom2]] == robot)]
        sides = SIDES if self.obj == "pot" else ("left",)
        for s in sides:
            G.set_opening(m, d, s, 0.0)
        mujoco.mj_forward(m, d)
        gid = {"left": m.geom("pan_handle").id, "right": m.geom("pan_handle2").id if self.obj == "pot" else None}
        ok, touch = jaws_straddle(m, d, {s: gid[s] for s in sides}, sides)
        return dict(ok=bool(not bad and ok), contact_free=bool(not bad), straddles=bool(ok), touch=touch)

    def assumptions(self) -> dict:
        from ..skills import pregrasp as PG
        return dict(theta=self.theta_box().assumptions(), nominal={k: ("covered.ASSUMED_COVERED.hidden_mass" if v is None else v) for k, v in NOMINAL.items()},
                    nominal_tag="ASSUMED (11-04, code author's pick inside each THETA range; needs user OK)",
                    pregrasp={k: v.__dict__ for k, v in PG.SEARCH_ASSUMPTIONS.items()},
                    worlds="nominal | corners = seeded-random vertices (SAMPLED) | random = uniform draws",
                    grasp="ASSUMED rigid weld (user 11-03); pre-grasp is a kinematic/contact check, no forces",
                    handle_geometry="ASSUMED known to the planner (handle_features from the scene's PanParams)")
