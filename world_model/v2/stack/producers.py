"""stack/producers.py -- ContentsTruthProducer: fills a house/ocs WorldState from the contents (carry) sim.  SIMULATION ONLY.

robot q/qd per side: Source.ENCODER (proprioception; the real arms have encoders).
carried object pose: Source.TRUTH (simulator ground truth; WorldState.privileged() flags it -- the contents sim has no perception yet).
object params (contents mass, per-mug fill, ...): Belief over the universe's PRIOR box (Source.PRIOR), never the sampled world's value.
Attachment: Relation(obj held_by robot, Source.EXEC) from the weld equalities' active flags.
"""
from __future__ import annotations

import numpy as np

from world_model.v2.house.ocs.state import Belief, Kind, ObjectState, Producer, Relation, RobotState, Source, WorldState
from world_model.v2.tray_rep.sets.ir import BoxShape, Pose

SIDES = ("left", "right")


class ContentsTruthProducer(Producer):
    name = "truth"

    def __init__(self, box=None):
        self._box = box            # ThetaBox; default = the universe's box for the sim's task

    def reset(self, sim=None) -> None:
        pass

    def _theta_box(self, sim):
        if self._box is not None:
            return self._box
        from .universes.contents import ContentsUniverse
        return ContentsUniverse().theta_box(sim.world.theta.get("task"))

    def update(self, ws: WorldState, sim, t: float) -> None:
        ep, m, d = sim.episode, sim.m, sim.d
        ws.t = float(t)
        ws.robot = RobotState(
            base=Pose(np.eye(3), np.zeros(3)), base_source=Source.PRIOR,        # fixed base in the carry scenes
            q={s: np.asarray(d.qpos[ep.addr[s]], float).copy() for s in SIDES},
            qd={s: np.asarray(d.qvel[ep.dofs[s]], float).copy() for s in SIDES},
            q_source=Source.ENCODER)
        task = sim.world.theta.get("task", ep.task)
        tb = self._theta_box(sim)
        params = {k: Belief(b.lo, b.hi, None, None, Source.PRIOR, f"{b.tag}: {b.src}") for k, b in tb.fields.items()
                  if k in ("fill_kg", "contents_kg", "contents_x", "contents_y", "cup_x", "cup_y")}
        if task.startswith("tray"):
            from ..nero.contents import objects as OB
            half = np.array([OB.TRAY["lx"] / 2, OB.TRAY["ly"] / 2, 0.004])
        else:
            from ..nero.contents import objects as OB
            half = np.array([OB.BOX["lx"] / 2, OB.BOX["ly"] / 2, OB.BOX["lz"] / 2])
        ws.objects = {"obj": ObjectState(
            "obj", Kind.PAYLOAD, BoxShape(half), Pose(np.asarray(d.xmat[ep.obj], float).reshape(3, 3).copy(), np.asarray(d.xpos[ep.obj], float).copy()),
            parent=None, name="tray" if task.startswith("tray") else "box", dynamics="static", source=Source.TRUTH, t_meas=float(t),
            params=params, name_source=Source.PRIOR, affordances=("graspable", "payload"))}
        held = all(bool(d.eq_active[m.equality(f"weld_{s}").id]) for s in SIDES)
        ws.relations = [Relation("obj", "held_by", "robot", 1.0, Source.EXEC, "held")] if held else []
        if not held:
            ws.meta.setdefault("attachment_known", []).append("obj")
        ws.meta["producers"] = ws.meta.get("producers", []) + ["truth"]
