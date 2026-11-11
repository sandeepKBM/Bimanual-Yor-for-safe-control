"""adapters/payload_envelope.py -- builds the nero/contents/envelope.Envelope backends that stack/checks/payload, probe_rung and admit are given.
SIMULATION ONLY."""
from __future__ import annotations

import functools
import json

import numpy as np

from world_model.v2.adapters import contents_specs as CS
from world_model.v2.nero.contents import envelope as E
from world_model.v2.nero.contents import objects as OB
from world_model.v2.nero.contents import scenes as SC
from world_model.v2.nero.contents import thresholds as TH
from world_model.v2.stack import tasks as TK
from world_model.v2.stack.checks.admit import PathSamples, path_offsets


@functools.lru_cache(maxsize=1)
def caps():
    return json.loads(TH.CAP_FILE.read_text())


def o_ref():
    tl, tr = np.asarray(caps()["pot"]["tcp"]["W0_liftoff"]["left"]), np.asarray(caps()["pot"]["tcp"]["W0_liftoff"]["right"])
    return 0.5 * (tl + tr) - np.array([0, 0, TH.GRASP_DZ])


class EnvelopeBackend:
    """stack.checks.admit.Backend: Envelope of a spec on captured arm configurations."""

    def o_ref(self):
        return o_ref()

    def make_envelope(self, spec, cap, offs):
        return E.Envelope(spec, E.Arms(cap), o_ref(), np.eye(3), ["left", "right"], grip_mass=caps()["pot"]["gripper_mass"], offsets=offs)

    def support_extent(self, world, name):
        top = SC.support_tops(world)[name]
        return top["x"], top["y"]


def scripted_path(task: str) -> PathSamples:
    """the legacy scripted plan's samples (TaskDef.moves, N_PER_MOVE per move) with the continuation-IK arm configurations."""
    t = TK.task(task)
    offs, per_seg = path_offsets(t)
    disp = np.array([sum(d[i] for _, d in t.moves) for i in range(3)], float)
    return PathSamples(offs, per_seg, TH.capture_path(caps()["pot"], offs), disp)


class RestBackend:
    """stack.checks.payload.Rest: u at the start pose (path key p0, a = 0) on the full contents prior of the task."""

    def __init__(self, task: str):
        spec = CS.spec_from_total(task, *CS.prior_total(task))
        offs = {"p0": np.zeros(3)}
        self.env = EnvelopeBackend().make_envelope(spec, TH.capture_path(caps()["pot"], offs), offs)
        self.P = spec.contents.P

    def u(self, c: float) -> float:
        u, _ = self.env.worst(OB.Contents(c, c, self.P), ["p0"], 0.0, "two_lever")
        return float(u)


@functools.lru_cache(maxsize=None)
def rest_backend(task: str) -> RestBackend:
    return RestBackend(task)
