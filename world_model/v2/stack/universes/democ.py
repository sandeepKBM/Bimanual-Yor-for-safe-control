"""stack/universes/democ.py -- DemoCUniverse: Demo C (YOR-NERO + table/pedestal + free slab box) as a stack Universe.  SIMULATION ONLY.

Parameter box (theta_box): PROPOSED by agent C 2026-11-03 -- every field is ASSUMED and needs user OK.  Nominals are democ/scene.ASSUMED
(file values); the ranges are new.  Worlds are points of the box -> scene.DemoCParams -> scene.compose.
Sets (worlds(set_name, n, seed)):  "nominal" (1 world), "corners" (n seeded-random vertices of the box: SAMPLED, the full 2^11 are not run),
"random" (n uniform draws, rng = default_rng(seed + i)), "all" (nominal + corners + random).
"""
from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from world_model.v2.house.ocs.state import Belief, Kind, ObjectState, Producer, RobotState, Source, WorldState
from world_model.v2.tray_rep.sets.ir import BoxShape, Pose
from ..types import Bound, Sim, TaskSpec, ThetaBox, WorldSpec, SIDES

SETS = ("nominal", "corners", "random", "all")
_PROP = "proposed by agent C 2026-11-03, needs user OK"


def _box() -> ThetaBox:
    from ...nero.democ import scene as S
    A = S.ASSUMED
    f = {}

    def add(k, nom, lo, hi, unit, why):
        f[k] = Bound(lo, hi, "ASSUMED", f"{_PROP}; nominal {nom} = democ/scene.ASSUMED; {why}", unit)
    add("box_mass", A["box_mass"], 0.3, 1.2, "kg", "range: 0.5x-2x nominal; NERO payload datasheet 3 kg/arm")
    for i, ax in enumerate("xyz"):
        add(f"com_{ax}", 0.0, -0.02 if ax != "z" else -0.01, 0.02 if ax != "z" else 0.01, "m", "COM offset from the geometric centre, hidden contents/uneven fill")
    add("box_mu", A["box_mu"], 0.6, 1.4, "-", "sliding friction coefficient of the box geom (cardboard-to-rubber range guess)")
    for i, ax in enumerate("xyz"):
        j = 0.005 if ax == "z" else 0.01      # z (jaw-axis thickness) kept +-5 mm: thickness 80 mm + 2x2 mm clearance exceeds the 82.2 mm open jaw gap
        add(f"half_{ax}", A["box_half"][i], A["box_half"][i] - j, A["box_half"][i] + j, "m", f"box half extent, +-{j*1e3:.0f} mm size jitter")
    add("centre_x", A["box_centre"][0], A["box_centre"][0] - 0.02, A["box_centre"][0] + 0.02, "m", "box centre (pedestal follows x), +-2 cm placement")
    add("centre_y", A["box_centre"][1], A["box_centre"][1] - 0.02, A["box_centre"][1] + 0.02, "m", "box centre y, +-2 cm placement")
    return ThetaBox(f)


def params_from_theta(th: dict):
    from ...nero.democ import scene as S
    return S.DemoCParams(box_half=(th["half_x"], th["half_y"], th["half_z"]), box_mass=th["box_mass"], box_com_offset=(th["com_x"], th["com_y"], th["com_z"]),
                         box_mu=th["box_mu"], box_centre=(th["centre_x"], th["centre_y"], S.ASSUMED["box_centre"][2]))


class DemoCTruthProducer(Producer):
    """box ObjectState straight from the simulator (Source.TRUTH => WorldState.privileged() lists 'box.pose'); robot q from the encoders."""
    name = "truth"

    def reset(self, sim=None) -> None:
        pass

    def update(self, ws: WorldState, sim, t: float) -> None:
        m, d = sim.m, sim.d
        from ...nero import adapters as A
        q = {s: np.asarray(d.qpos[A.arm_addrs(m, s)[0]], float).copy() for s in SIDES}
        qd = {s: np.asarray(d.qvel[A.arm_addrs(m, s)[1]], float).copy() for s in SIDES}
        ws.t = float(t)
        ws.robot = RobotState(base=Pose(np.eye(3), np.zeros(3)), base_source=Source.PRIOR, q=q, qd=qd, q_source=Source.ENCODER)
        b = m.body("democ_box").id
        half = m.geom_size[m.geom("democ_box_geom").id].copy()
        ws.objects = {"box": ObjectState("box", Kind.MOVABLE, BoxShape(half),
                                         Pose(np.asarray(d.xmat[b], float).reshape(3, 3).copy(), np.asarray(d.xpos[b], float).copy()),
                                         name="box", name_source=Source.PRIOR, dynamics="static", source=Source.TRUTH, t_meas=float(t),
                                         affordances=("graspable",))}
        ws.meta["producers"] = ws.meta.get("producers", []) + ["democ_truth"]


class DemoCUniverse:
    name = "democ"

    def theta_box(self) -> ThetaBox:
        return _box()

    def nominal_theta(self) -> dict:
        b = self.theta_box().fields
        from ...nero.democ import scene as S
        A = S.ASSUMED
        nom = dict(box_mass=A["box_mass"], box_mu=A["box_mu"], com_x=0.0, com_y=0.0, com_z=0.0, centre_x=A["box_centre"][0], centre_y=A["box_centre"][1],
                   half_x=A["box_half"][0], half_y=A["box_half"][1], half_z=A["box_half"][2])
        assert set(nom) == set(b)
        return nom

    def _spec(self, wid, th, seed):
        return WorldSpec(self.name, wid, th, seed, params_from_theta(th))

    def worlds(self, set_name: str, n: int | None = None, seed: int = 0) -> list[WorldSpec]:
        if set_name not in SETS:
            raise ValueError(f"unknown world set {set_name!r}; have {SETS}")
        fl = self.theta_box().fields
        keys = list(fl)
        out = []
        if set_name in ("nominal", "all"):
            out.append(self._spec("democ/nominal", self.nominal_theta(), seed))
        if set_name in ("corners", "all"):
            rng = np.random.default_rng(seed)
            for i in range(int(n if n is not None else 16)):
                bits = rng.integers(0, 2, len(keys))
                th = {k: (fl[k].hi if b else fl[k].lo) for k, b in zip(keys, bits)}
                out.append(self._spec(f"democ/corner{i}", th, seed + i))
        if set_name in ("random", "all"):
            for i in range(int(n if n is not None else 16)):
                rng = np.random.default_rng(seed + i)
                th = {k: float(rng.uniform(fl[k].lo, fl[k].hi)) for k in keys}
                out.append(self._spec(f"democ/rand{i}", th, seed + i))
        return out

    def build(self, world: WorldSpec) -> Sim:
        from ...nero import adapters as A
        from ...nero.democ import scene as S
        p = world.native if world.native is not None else params_from_theta(world.theta)
        m = S.compose(p)
        d = mujoco.MjData(m)
        ba = m.jnt_qposadr[m.joint("democ_box_free").id]
        d.qpos[ba:ba + 3] = p.box_centre
        d.qpos[ba + 3:ba + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        from ...nero.democ import kin as K
        return Sim(m, d, world, handles=dict(obj_geoms={"box": "democ_box_geom"}, obj_joint={"box": "democ_box_free"},
                                             benign_geoms=tuple(K.NAMES_OK), params=p))

    def tasks(self, world: WorldSpec) -> list[TaskSpec]:
        p = world.native
        return [TaskSpec("pregrasp", "box", goal=dict(note="both hands in a contact-free pre-grasp with the jaws straddling the box ends",
                                                       box_centre=list(p.box_centre)), limits={})]

    def producers(self, mode: str = "truth"):
        if mode != "truth":
            raise NotImplementedError(f"democ producers: only 'truth' exists; got {mode!r}")
        return [DemoCTruthProducer()]

    def initial_state(self, sim: Sim) -> WorldState:
        ws = WorldState(0.0, RobotState(base=Pose(np.eye(3), np.zeros(3))))
        for pr in self.producers("truth"):
            pr.update(ws, sim, 0.0)
        return ws

    @staticmethod
    def apply_pregrasp(sim: Sim, plan) -> None:
        """put the sim's arms/fingers at the plan's pre-grasp (open jaws), so success() can be evaluated."""
        from ...nero import adapters as A, gripper as G
        seg = plan.segments[0]
        for s, q in seg.meta["q_target"].items():
            sim.d.qpos[A.arm_addrs(sim.m, s)[0]] = q
            G.set_opening(sim.m, sim.d, s, seg.gripper[s])
        mujoco.mj_forward(sim.m, sim.d)

    def success(self, sim: Sim, task: TaskSpec) -> dict:
        """pregrasp: with the arms where they are, (a) no robot contact with the jaws as they are, (b) closing the jaws of both hands makes BOTH jaws
        of each hand touch the box.  (Evaluated on a copy of the data; the sim is not modified.)"""
        from ...nero import gripper as G
        from ...nero.democ import kin as K
        from ..skills.pregrasp import jaws_straddle
        m = sim.m
        d = mujoco.MjData(m)
        d.qpos[:] = sim.d.qpos
        mujoco.mj_forward(m, d)
        free = K.bad_contacts(m, d) == []
        for s in SIDES:
            G.set_opening(m, d, s, 0.0)
        mujoco.mj_forward(m, d)
        ok, touch = jaws_straddle(m, d, m.geom("democ_box_geom").id, SIDES)
        return dict(ok=bool(free and ok), contact_free=bool(free), straddles=bool(ok), touch=touch)

    def assumptions(self) -> dict:
        from ...nero.democ import scene as S
        from ..skills import pregrasp as PG
        return dict(theta=self.theta_box().assumptions(), scene_ASSUMED={k: v for k, v in S.ASSUMED.items()},
                    pregrasp={k: v.__dict__ for k, v in PG.SEARCH_ASSUMPTIONS.items()},
                    worlds="nominal | corners = seeded-random vertices (SAMPLED) | random = uniform draws",
                    grasp="ASSUMED rigid weld (user 11-03); pre-grasp is a kinematic/contact check, no forces")
