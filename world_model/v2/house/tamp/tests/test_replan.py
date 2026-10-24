"""replanning enablers: planning while holding (G1) and partial-route candidates from a mid-route pose (G2).  SIMULATION ONLY, no stepping."""
from __future__ import annotations

import os

import numpy as np
import pytest

from world_model.v2.house.tamp import planner as TP
from world_model.v2.house.tamp import streams as ST
from world_model.v2.house.tamp import world as W
from world_model.v2.house.tamp.types import Cert, Conf, Grasp

pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(TP.PDDLSTREAM_ROOT, "downward", "builds", "release")),
                                reason="PDDLStream / Fast Downward not built at PDDLSTREAM_ROOT")


@pytest.fixture(scope="module")
def sim():
    from world_model.v2.house.teleop.sim import TeleopSim
    return TeleopSim("h5", grasped=False, seed=0, with_people=False, counter=True, desk=True)


@pytest.fixture(scope="module")
def world(sim):
    return W.world_from_sim(sim)


def _held_world(sim, world):
    """world with the tray 'in hand' (sim.observe patched: this test never steps the sim)."""
    lo, hi = world.items["tray"].mass_range
    g = Grasp("tray", "bimanual_squeeze_a", 2, (lo, hi), world.items["tray"].mu_range, Cert("C3_grasp_corners", 0.1))
    real = sim.observe
    sim.observe = lambda: dict(real(), grasped=True)
    try:
        w = W.world_from_sim(sim, held=("tray", g))
    finally:
        sim.observe = real
    return w, g


def test_init_facts_holding(sim, world):
    w, g = _held_world(sim, world)
    init = TP.init_facts(w)
    assert ("Holding", "tray", g) in init and ("Grasp", "tray", g) in init
    assert ("HandsEmpty",) not in init and not any(f[0] in ("AtPose", "Pose") for f in init)
    w2 = W.world_from_sim(sim)
    w2.items["tray"].extra["grasped"] = True
    with pytest.raises(ValueError):
        TP.init_facts(w2)                                        # grasped but no certified grasp given


def test_solve_from_holding_at_cafe_dock(sim, world):
    w, g = _held_world(sim, world)
    cafe = w.surfaces["counter"].dock
    w.q0 = Conf("q_start", cafe.x, cafe.y, cafe.yaw, surface=None)      # sim sits at the lab start; the replan start is the cafe dock
    res = TP.solve_task(w, "(On tray desk)", max_time=60)
    assert res["solved"], res["pddlstream_out"]
    assert [a for a, _ in res["plan"]] == ["move_holding", "place"]


def test_suffix_route_certifies(world):
    st = ST.Streams(world)
    P = np.asarray(world.routes[0][1], float)
    cum = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    k = int(np.searchsorted(cum, cum[-1] / 2))
    mid = P[k - 1] + (P[k] - P[k - 1]) * 0.5
    q1 = Conf("q_mid", float(mid[0]), float(mid[1]), 0.0)
    for q2, tag in ((Conf("q_end", *map(float, P[-1]), 0.0), "scene:out_route:suffix"), (Conf("q_beg", *map(float, P[0]), 0.0), "scene:out_route:back:suffix")):
        cands = [c for c in st._candidate_routes(q1, q2) if c[2].endswith(":suffix")]
        assert cands and cands[0][2] == tag, (tag, [c[2] for c in st._candidate_routes(q1, q2)])
        pts, rev, src = cands[0]
        assert pts[0] == (q1.x, q1.y)
        cert, why = st.certify_route(pts, rev, holding=False)
        assert cert is not None, why
        assert cert.margin >= 0
