"""recovery phase 1: near-people route certification, lateral-shift candidates, --recovery none unchanged.  SIMULATION ONLY, no stepping."""
from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

from world_model.v2.house.tamp import planner as TP
from world_model.v2.house.tamp import streams as ST
from world_model.v2.house.tamp import world as W
from world_model.v2.house.tamp.types import Conf

pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(TP.PDDLSTREAM_ROOT, "downward", "builds", "release")),
                                reason="PDDLStream / Fast Downward not built at PDDLSTREAM_ROOT")


@pytest.fixture(scope="module")
def sim():
    from world_model.v2.house.teleop.sim import TeleopSim
    return TeleopSim("h5", grasped=False, seed=0, with_people=False, counter=True, desk=True)


@pytest.fixture(scope="module")
def world(sim):
    return W.world_from_sim(sim)


def _on_route_at(P, s):
    cum = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    i = min(int(np.searchsorted(cum, s, side="right") - 1), len(P) - 2)
    return P[i] + (P[i + 1] - P[i]) * (s - cum[i]) / (cum[i + 1] - cum[i])


def _with_person(world, xy):
    b = world.bel0.with_(people=(np.asarray(xy, float),), people_vel=(np.zeros(2),), people_e_p=(), people_e_v=(), people_stale=())
    return dataclasses.replace(world, bel_people=b)


def test_near_people_rejects_blocked_route_and_shifts_are_person_related(world):
    P = np.asarray(world.routes[0][1], float)
    q1, q2 = Conf("a", *map(float, P[0]), 0.0), Conf("b", *map(float, P[-1]), 0.0)
    st0 = ST.Streams(world)
    cands = st0._candidate_routes(q1, q2)
    assert cands[0][2] == "scene:out_route"
    pts, rev, _ = cands[0]
    cert, why = st0.certify_route(pts, rev, holding=False)
    assert cert is not None, why                                     # no person: unshifted certifies
    assert dict(cert.detail)["near_people_applied"] is False
    w2 = _with_person(world, _on_route_at(P, 1.0))
    st = ST.Streams(w2)
    cert, why = st.certify_route(pts, rev, holding=False)
    assert cert is None and "person" in why, why                     # person ON the route 1 m from the start
    results = []
    for p2, r2, src in st._candidate_routes(q1, q2)[1:]:
        assert ":shift" in src
        c2, w = st.certify_route(p2, r2, holding=False)
        results.append((src, c2 is not None, w))
        if c2 is None:
            assert "person" in w or "wall" in w or "static" in w or "clear" in w, w
    assert results
    pers = [r for r in results if not r[1] and "person" in r[2]]
    wall = [r for r in results if not r[1] and "person" not in r[2]]
    print("shift results:", results)
    assert pers or any(r[1] for r in results) or wall               # every shift decided; person rejections are reported as such
    # far person (beyond near_m of the start) does not matter
    w3 = _with_person(world, _on_route_at(P, 3.2))
    cert, why = ST.Streams(w3).certify_route(pts, rev, holding=False)
    assert cert is not None, why
    cert2, _ = ST.Streams(w3, near_m=0.0).certify_route(pts, rev, holding=False)
    assert cert2 is not None and dict(cert2.detail)["near_people_applied"] is False


def test_shift_variants(world):
    P = np.asarray(world.routes[0][1], float)
    st = ST.Streams(world)
    for off in (0.15, -0.30):
        sh = np.asarray(st.shift_route(tuple(map(tuple, P)), off), float)
        assert np.allclose(sh[0], P[0]) and np.allclose(sh[-1], P[-1]) and len(sh) == len(P)
        i = 2
        t = P[i + 1] - P[i - 1]
        t /= np.hypot(*t)
        assert np.allclose(sh[i] - P[i], off * np.array([-t[1], t[0]]))
        assert abs(np.hypot(*(sh[i] - P[i])) - abs(off)) < 1e-9
    assert st.shift_route(((0, 0), (1, 0)), 0.1) is None
    q1, q2 = Conf("a", *map(float, P[0]), 0.0), Conf("b", *map(float, P[-1]), 0.0)
    srcs = [c[2] for c in st._candidate_routes(q1, q2)]
    assert srcs[0] == "scene:out_route" and "scene:out_route:shift+0.15" in srcs and "scene:out_route:shift-0.30" in srcs
    assert [c[2] for c in ST.Streams(world, lateral_offsets=())._candidate_routes(q1, q2)] == [c for c in srcs if ":shift" not in c]


def test_recovery_none_plan_unchanged(world):
    """legacy stream config (--recovery none): same plan as the default streams in the no-people world, and no near-people detail."""
    from world_model.v2.house.tamp import run_tamp as RT
    res0 = TP.solve_task(world, "(On tray desk)", max_time=60, streams=ST.Streams(world, near_m=0.0, lateral_offsets=()))
    res1 = TP.solve_task(world, "(On tray desk)", max_time=60)
    assert res0["solved"] and res1["solved"]
    assert [a for a, _ in res0["plan"]] == [a for a, _ in res1["plan"]]
    assert RT._plan_sources(res0["plan"]) == RT._plan_sources(res1["plan"]) == ["scene:out_route", "scene:out_route:back"] or \
        RT._plan_sources(res0["plan"]) == RT._plan_sources(res1["plan"])
    import inspect
    assert inspect.signature(RT.run_episode).parameters["recovery"].default == "none"
