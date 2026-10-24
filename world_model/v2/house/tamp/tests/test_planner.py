"""tamp planner tests: PDDL goal -> PDDLStream plan, every continuous value certified by our checkers; and the checkers really GATE the plan.
SIMULATION ONLY (builds one h5 TeleopSim; no stepping, no rendering, no network)."""
from __future__ import annotations

import dataclasses
import os

import pytest

from world_model.v2.house.tamp import planner as TP
from world_model.v2.house.tamp import world as W

pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(TP.PDDLSTREAM_ROOT, "downward", "builds", "release")),
                                reason="PDDLStream / Fast Downward not built at PDDLSTREAM_ROOT")


@pytest.fixture(scope="module")
def world():
    from world_model.v2.house.teleop.sim import TeleopSim
    sim = TeleopSim("h5", grasped=False, seed=0, with_people=True, counter=True, desk=True)
    return W.world_from_sim(sim)


def test_parse_goal():
    assert TP.parse_goal("(On tray desk)") == ("On", "tray", "desk")
    assert TP.parse_goal("(AND (On tray desk) (HandsEmpty))") == ("and", ("On", "tray", "desk"), ("HandsEmpty",))
    for bad in ("On tray desk", "(On tray", "(On tray desk))"):
        with pytest.raises(ValueError):
            TP.parse_goal(bad)


def test_world_model(world):
    assert set(world.surfaces) == {"counter", "desk"}
    tray = world.items["tray"]
    assert tray.pose is not None and tray.pose.surface == "counter"
    lo, hi = tray.mass_range
    assert lo < tray.extra["model_mass"] < hi
    hr = world.stats["hold_rel_by_surface"]
    assert max(abs(a - b) for a, b in zip(hr["counter"], hr["desk"])) < 0.01       # both slabs sit at the same hold pose of their dock


def test_fetch_plan_certified(world):
    res = TP.solve_task(world, "(On tray desk)", max_time=60)
    assert res["solved"], res["pddlstream_out"]
    names = [a for a, _ in res["plan"]]
    assert names == ["move_free", "pick", "move_holding", "place"]
    _, (q1, q2, r1) = res["plan"][0]
    assert q2.surface == "counter" and r1.kind == "free" and r1.cert.checker == "C1C2_static_route" and r1.cert.margin >= 0
    _, (o, g, q1b, q2b, r2) = res["plan"][2]
    assert g.arms == 2 and g.cert.checker == "C3_grasp_corners" and g.cert.margin >= 0
    assert r2.kind == "holding" and r2.reverse and dict(r2.cert.detail)["holding"] is True
    log = res["stream_log"]
    assert any(e["stream"] == "s-grasp" and not e["ok"] and e["reason"].startswith("C3") for e in log)      # C3 rejected the single-arm grasps


def test_pick_only_goal(world):
    res = TP.solve_task(world, "(HoldingAny tray)", max_time=60)
    assert res["solved"] and [a for a, _ in res["plan"]] == ["move_free", "pick"]


def test_too_heavy_is_rejected_by_C3(world):
    tray = world.items["tray"]
    world.items["tray"] = dataclasses.replace(tray, mass_range=(1.6, 2.0))          # above 2 x the 0.75 kg per-arm payload
    try:
        res = TP.solve_task(world, "(On tray desk)", max_time=20)
    finally:
        world.items["tray"] = tray
    assert not res["solved"]
    assert all(not e["ok"] for e in res["stream_log"] if e["stream"] == "s-grasp")


def test_uncertifiable_route_blocks_plan(world, monkeypatch):
    # In h5 the dock check (C2 at rest) already fails from sigma* >= 0.25 m, before any route is tried, and no sigma* makes
    # only the route fail (sweep 2026-10-21).  So accept docks unchecked here, to test that the ROUTE certificate blocks.
    import world_model.v2.house.tamp.streams as ST

    def dock_unchecked(self, s):
        self._rec("s-dock", (s,), True, "unchecked (test)")
        yield (self.w.surfaces[s].dock,)

    monkeypatch.setattr(ST.Streams, "dock_gen", dock_unchecked)
    old = world.sigma_star
    world.sigma_star = 0.6                                                         # pose box larger than the corridor: H0 fails
    try:
        res = TP.solve_task(world, "(On tray desk)", max_time=20)
    finally:
        world.sigma_star = old
    assert not res["solved"]
    assert any(e["stream"].startswith("s-route") and not e["ok"] and "no admissible candidate" in e["reason"] for e in res["stream_log"])


def test_no_executor_skill_no_plan(world, monkeypatch):
    monkeypatch.setitem(W.SKILLS, "tray", ())
    import world_model.v2.house.tamp.streams as ST
    monkeypatch.setitem(ST.SKILLS, "tray", ())
    res = TP.solve_task(world, "(On tray desk)", max_time=20)
    assert not res["solved"]
    assert any(e["reason"] == "no executor skill for this grasp" for e in res["stream_log"])


def test_unknown_goal_name(world):
    with pytest.raises(ValueError):
        TP.solve_task(world, "(On mug desk)")
