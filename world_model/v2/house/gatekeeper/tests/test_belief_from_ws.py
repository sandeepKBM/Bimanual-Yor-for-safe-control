"""build_belief (TruthProducer-style facts -> WorldState -> belief_from_ws) is BIT-IDENTICAL to the original sim-reading implementation
(_build_belief_legacy), and belief_from_ws needs no simulator.  SIMULATION ONLY."""
import dataclasses
import itertools
import types

import numpy as np
import pytest

from world_model.v2.house.teleop import geom as GM, shield as SH
from world_model.v2.house.gatekeeper import GKConfig
from world_model.v2.house.gatekeeper import belief as BL
from world_model.v2.house.gatekeeper.tests.test_gatekeeper import FakeSim
from world_model.v2.house.ocs import producers as PR
from world_model.v2.house.ocs.state import WorldState, Source

CFG = SH.ShieldCfg()
DOCKS = ((1.0, 2.0), (3.5, 4.0))


def _eq(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, (tuple, list)):
        return type(a) == type(b) and len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_eq(a[k], b[k]) for k in a)
    if dataclasses.is_dataclass(a) or hasattr(a, "edt"):
        return a is b
    return a == b


def assert_same(b0, b1):
    for f in dataclasses.fields(BL.Belief):
        assert _eq(getattr(b0, f.name), getattr(b1, f.name)), f"Belief.{f.name}: {getattr(b0, f.name)!r} != {getattr(b1, f.name)!r}"


def _both(sim, gk, **kw):
    b0 = BL._build_belief_legacy(sim, CFG, gk, fp=dict(GM.FP), docks=DOCKS, **kw)
    b1 = BL.build_belief(sim, CFG, gk, fp=dict(GM.FP), docks=DOCKS, **kw)
    assert_same(b0, b1)
    return b1


def _tracks():
    return [dict(xy=np.array([3.0, 5.0]), v=np.array([-0.4, 0.1]), e_p=0.1, e_v=0.4, stale=0.12),
            dict(xy=np.array([2.2, 6.1]), v=np.array([0.0, 0.3]), e_p=0.05, e_v=0.2, stale=0.0)]


GKS = [GKConfig(cup_row=cr, person_model=pm) for cr, pm in itertools.product(("always", "attached"), ("speed", "accel"))]
GKS.append(GKConfig(person_model="accel", vel_source="fd", e_p=0.05, latency_s=0.1, k_pose=2.0))


class Stub(FakeSim):
    def __init__(self, pose, people=(), **attrs):
        super().__init__(pose, people)
        for k, v in attrs.items():
            setattr(self, k, v)


@pytest.fixture(scope="module", params=[dict(grasped=True), dict(grasped=False, counter=True, desk=True), dict(grasped=True, pose_source="est")],
                ids=["grasped", "free", "grasped_est_pose"])
def real_sim(request):
    from world_model.v2.house.teleop.sim import TeleopSim
    return TeleopSim("h5", seed=0, **request.param)


@pytest.mark.parametrize("with_people", [True, False])
@pytest.mark.parametrize("gk", GKS)
def test_real_sim(real_sim, gk, with_people):
    sim = real_sim
    saved = sim.people_state
    if not with_people:
        sim.people_state = []
    try:
        assert (len(sim.people_state) > 0) == with_people
        b = _both(sim, gk)
        assert len(b.people) == len(sim.people_state)
        _both(sim, gk, people_vel=(np.array([0.1, -0.2]),) * len(sim.people_state), e_v=0.3)
        _both(sim, gk, tracks=_tracks(), e_v=None)
        _both(sim, gk, tracks=_tracks(), e_v=0.25)
        _both(sim, gk, tracks=[])
    finally:
        sim.people_state = saved


@pytest.mark.parametrize("gk", GKS)
@pytest.mark.parametrize("people", [(), ((3.0, 5.0),), ((3.0, 5.0), (2.0, 6.5))])
def test_fakesim_and_stubs(gk, people):
    for sim in (FakeSim((1.5, 5.0, 0.0), people), FakeSim((2.0, 4.0, 1.1), people), Stub((1.5, 5.0, 0.0), people, grasped=True),
                Stub((1.5, 5.0, 0.0), people, grasped=False), Stub((1.5, 5.0, 0.3), people, grasped=False, assist=types.SimpleNamespace(state="IDLE"))):
        sim.people_state = [(xy, np.array([-0.7, 0.1])) for xy, _ in sim.people_state]
        _both(sim, gk)
        _both(sim, gk, tracks=_tracks())
    # friction assist stub, every state
    for st in ("PRE", "APPROACH", "CLOSE", "VERIFY", "HOLD", "PLACE_LOWER", "PLACE_OPEN", "PLACE_RETREAT", "PLACE_SETTLE", "FAILED", "ABORT", "IDLE"):
        sim = Stub((1.5, 5.0, 0.0), people, grasped=False, assist=types.SimpleNamespace(state=st))
        sim.d.xpos[0] = np.array([6.0, 7.0, 0.0])                 # cup far away so the attached/free branch changes r_cup
        b = _both(sim, gk)
        assert (PR.attachment_status(sim) is not None) == GM.cup_attached(sim)


def test_loc_sigma_path():
    sim = Stub((1.5, 5.0, 0.4), ((3.0, 5.0),), loc=types.SimpleNamespace(sigma_xy=0.07))
    for gk in GKS:
        _both(sim, gk)


def test_belief_from_ws_needs_no_sim():
    sim = Stub((1.5, 5.0, 0.3), ((3.0, 5.0),), grasped=False)
    ws = WorldState(t=0.0, robot=None)
    PR.SimFactsProducer().update(ws, sim, 0.0)
    robot = BL.robot_info_from_sim(sim)
    del sim                                                         # nothing below may touch it
    gk = GKConfig(cup_row="attached", person_model="accel")
    b = BL.belief_from_ws(ws, robot, CFG, gk, fp=dict(GM.FP), docks=DOCKS)
    assert np.allclose(b.r_cup, 0.0) and b.cup_margin == BL.CUP_FREE_MARGIN and len(b.people) == 1
    # a VLM/perception-style producer: just write the facts (attachment, people) into a WorldState with another Source
    ws.relations = [r for r in ws.relations if r.rel != "held_by"]
    ws.relations.append(PR.Relation("cup", "held_by", "robot", 0.9, Source.VLM, "held"))
    assert ws.attached("cup")
    b2 = BL.belief_from_ws(ws, robot, CFG, gk, fp=dict(GM.FP), docks=DOCKS)
    assert np.linalg.norm(b2.r_cup) > 0.0 and b2.cup_margin == gk.cup_margin


def test_attached_query_semantics():
    ws = WorldState(t=0.0, robot=None)
    assert not ws.attached("cup") and ws.attached("cup", default=True)              # unknown -> default
    PR.write_attachment(ws, types.SimpleNamespace(grasped=False), oids=("cup",))
    assert not ws.attached("cup", default=True) and ws.meta["attachment_known"] == {"cup"}
    PR.write_attachment(ws, types.SimpleNamespace(grasped=True), oids=("cup",))
    assert ws.attached("cup") and ws.relations[0].status == "held" and ws.relations[0].source == Source.EXEC
    PR.write_attachment(ws, types.SimpleNamespace(grasped=False, assist=types.SimpleNamespace(state="CLOSE")), oids=("cup",))
    assert ws.attached("cup") and [r.status for r in ws.relations] == ["closing"]
