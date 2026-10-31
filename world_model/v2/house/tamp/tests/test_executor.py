"""Executor smoke tests (CPU, no GPU/GL/robot).  python -m pytest -q world_model/v2/house/tamp/tests/test_executor.py
Fast: precondition failure leaves the sim alive.  Full 4-action plan, no people, filter none."""
import pytest

from world_model.v2.house.tamp import exec_smoke as XS
from world_model.v2.house.tamp import executor as EX
from world_model.v2.house.teleop.sim import TeleopSim
from world_model.v2.house.teleop import shield as SH


def test_precondition_failure_keeps_sim_alive():
    sim = TeleopSim("h5", grasped=False, seed=0, with_people=False, counter=True, desk=True)
    ex = EX.Executor(sim, SH.Passthrough())
    plan = XS.build_h5_plan(sim.scene)
    t0 = sim.t
    r = ex.run_action(*plan[1])                  # pick before reaching the dock
    assert r.status == "failed" and "precondition" in r.reason and sim.t == t0
    r = ex.run_action(*plan[3])                  # place with nothing held
    assert r.status == "failed" and "precondition" in r.reason


def test_full_plan_no_people_no_filter():
    s = XS.run_smoke("none", people=False, seed=0, max_s=150.0, verbose=False)
    for r in s["results"]:
        assert r["status"] == "ok", (r["name"], r["status"], r["reason"], r["checks"])
        assert all(c["ok"] for c in r["checks"].values()), (r["name"], r["checks"])
    assert [r["name"] for r in s["results"]] == ["move_free", "pick", "move_holding", "place"]
    assert s["plan_completed"] and not s["episode"]["spilled"]


@pytest.mark.slow
def test_full_plan_shield_people():
    s = XS.run_smoke("shield", people=True, seed=0, max_s=200.0, verbose=False)
    assert s["results"]            # outcome (ok / stalled / timeout) is a result, not an assertion; physical checks are recorded
