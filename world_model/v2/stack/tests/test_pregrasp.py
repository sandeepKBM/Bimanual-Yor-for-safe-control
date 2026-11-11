"""PreGrasp skill + DemoCUniverse tests.  SIMULATION ONLY, CPU.   PYTHONPATH=. python -m pytest -q world_model/v2/stack/tests/test_pregrasp.py"""
import time

import numpy as np
import pytest

from world_model.v2.stack.types import SkillCall
from world_model.v2.stack.skills import pregrasp as PG
from world_model.v2.stack.universes.democ import DemoCUniverse

FAST = dict(grid=dict(inset=(0.01,), tilt_deg=(10.0,)), refine_rounds=0)     # one cell (the lead's best grid point), ~5-15 s


@pytest.fixture(scope="module")
def nominal():
    u = DemoCUniverse()
    w = u.worlds("nominal")[0]
    sim = u.build(w)
    ws = u.initial_state(sim)
    return u, sim, ws


def test_box_pose_comes_from_state_not_sim(nominal):
    """shifting the ESTIMATED pose changes the targets even though the sim box did not move."""
    u, sim, ws = nominal
    sk = PG.PreGrasp()
    ok, why = sk.pre(ws, SkillCall("pregrasp", {"obj": "box"}))
    assert ok, why
    assert not sk.pre(ws, SkillCall("pregrasp", {"obj": "nope"}))[0]
    assert "box.pose" in ws.privileged()                       # truth producer is flagged
    f = PG.box_features(ws.objects["box"].shape.half)["+y"]
    cell = dict(inset=0.01, tilt_deg=10.0, roll_deg=0.0, dz=0.0)
    from world_model.v2.nero.democ import pregrasp_probe as PP
    p = u.build(u.worlds("nominal")[0]).handles["params"]
    R = np.asarray(ws.world_pose("box").R); pos = np.asarray(ws.world_pose("box").p)
    for side, key in (("left", "+y"), ("right", "-y")):
        a, Ra = PG.grasp_target(R, pos, PG.box_features(ws.objects["box"].shape.half)[key], cell, 1.0)
        b, Rb = PP.targets(p, dict(inset=0.01, tilt_deg=10.0, roll_deg=0.0, dz=0.0), (1.0, 1.0))[side]
        assert np.allclose(a, b, atol=1e-9) and np.allclose(Ra, Rb, atol=1e-9)          # same targets as the lead's probe


def test_nominal_world_finds_passing_candidate(nominal):
    u, sim, ws = nominal
    sk = PG.PreGrasp()
    call = SkillCall("pregrasp", {"obj": "box", "constraints": FAST})
    t0 = time.time()
    plan = sk.plan(ws, call, None, sim)
    wall = time.time() - t0
    print(f"\nsearch wall {wall:.1f} s, cells {plan.provenance['n_cells']}, margin {plan.segments[0].meta['margins']}")
    assert plan.provenance["ok"], plan.provenance["reason"]
    seg = plan.segments[0]
    assert seg.mode == "pregrasp" and set(seg.gripper) == {"left", "right"}
    v = seg.meta["verdict"]
    assert v.admit and v.guarantee == "sampled" and v.margins["joint_limit_rad"] > 0 and v.margins["jaw_gap_over_thickness_m"] > 0
    assert not sk.post(ws, call)[0]                          # arms still at home (ws.robot.q = executed state): jaws do not straddle
    # universe success: arms at the plan -> contact-free + straddle; post() on the re-read state
    u.apply_pregrasp(sim, plan)
    assert sk.post(u.initial_state(sim), call)[0]
    assert u.success(sim, u.tasks(sim.world)[0])["ok"]


def test_impossible_request_returns_none_with_reason(nominal):
    u, sim, ws = nominal
    sk = PG.PreGrasp()
    thick = 2 * ws.objects["box"].shape.half[2]
    cap = 0.0                                               # fully closed: gap << 60 mm box
    assert PG.gap_at_frac(cap) < thick
    plan = sk.plan(ws, SkillCall("pregrasp", {"obj": "box", "constraints": {"open_cap": cap}}), None, sim)
    assert plan.segments == [] and not plan.provenance["ok"]
    assert "cannot straddle" in plan.provenance["reason"]
    # a face that no side may use
    plan = sk.plan(ws, SkillCall("pregrasp", {"obj": "box", "faces": ["+y"]}), None, sim)
    assert plan.segments == [] and "no allowed grasp face" in plan.provenance["reason"]


def test_universe_worlds_and_box():
    u = DemoCUniverse()
    tb = u.theta_box()
    assert all(b.tag == "ASSUMED" and "needs user OK" in b.src for b in tb.fields.values())
    ws1, ws2 = u.worlds("all", 4, seed=3), u.worlds("all", 4, seed=3)
    assert [w.theta for w in ws1] == [w.theta for w in ws2] and len(ws1) == 9
    for w in ws1:
        for k, v in w.theta.items():
            assert tb.fields[k].lo - 1e-12 <= v <= tb.fields[k].hi + 1e-12
    sim = u.build(ws1[1])
    assert sim.m.nbody > 30
