"""Runtime watcher gate (owner-written, 11-16): adapters/z3_watcher_contents.py on box 1.0-1.5, closed_chain default plan.  SIMULATION ONLY.

Pins (the agent's smoke, re-run independently):
  nominal  run through the obj_world U (continue_through_u): no quantity fires; every quantity has a recorded peak below its tolerance; cost recorded
  stop     without continue_through_u the executed default plan stops on the object-world U (the D9 box finding) -- pinned so a change shows up
  load     +1.0 kg at t = 2.0 s: claw_load_total fires AFTER the injection (not before), the suggestion implies a total above the cell's upper bound
           (1.75 kg) and carries a block_exact Learned for the plan.  Latency is reported, not bounded: the moving-carry load margin (1.6 kg) hides
           1 kg until the object is at rest (finding 11-16).
  shift    a 10 mm object shift in the grasp fires nothing: the watcher reads only encoders + joint torques (robot-fidelity rule), which do not see it
"""
from __future__ import annotations

import os

import pytest

W = pytest.importorskip("world_model.v2.adapters.z3_watcher_contents")
CELL = (1.0, 1.5)


@pytest.fixture(scope="module", autouse=True)
def _closed_chain_planner():
    """BIMANUAL_PLANNER=closed_chain for THIS module only, restored after.  (Was a module-level setdefault: the ROS launch_testing pytest plugin in this env imports
    every test module at collection, so it leaked into every later test file: test_carry's scripted-backup test failed in ilab job 1301045.)"""
    old = os.environ.get("BIMANUAL_PLANNER")
    os.environ["BIMANUAL_PLANNER"] = "closed_chain"
    yield
    if old is None:
        os.environ.pop("BIMANUAL_PLANNER", None)
    else:
        os.environ["BIMANUAL_PLANNER"] = old


@pytest.fixture(scope="module")
def nominal(_closed_chain_planner):
    return W.run_watched("box", CELL, seed=0, continue_through_u=True)


def test_nominal_fires_nothing(nominal):
    assert not nominal["fired"], nominal["fired"]
    assert set(nominal["peak"]) >= {"joint_q", "obj_pos", "obj_rot", "claw_load_total", "clearance"}, sorted(nominal["peak"])
    assert "cost_ms" in nominal


def test_default_plan_stops_on_obj_world():
    rec = W.run_watched("box", CELL, seed=0, continue_through_u=False)
    viol = [v for _, v in rec["segments_viol"] if v]
    assert viol and any("obj_world" in str(v) for v in viol), rec["segments_viol"]


def test_extra_load_fires_after_injection():
    rec = W.run_watched("box", CELL, seed=0, inject=dict(kind="extra_load", kg=1.0, t=2.0), continue_through_u=True)
    assert "claw_load_total" in rec["fired"], rec["fired"]
    t_fire = rec["inject"]["t_fire"]
    assert all(f["first"]["t"] >= t_fire - 1e-9 for f in rec["fired"].values()), (t_fire, rec["fired"])
    print("first fire steps after injection:", rec["first_fire_steps_after_inject"])
    sug = [s for s in rec["suggestions"] if s.quantity == "claw_load_total"]
    assert sug and sug[0].implied_total_kg > 0.25 + CELL[1]          # box empty 0.25 kg + contents hi
    assert sug[0].learned.kind == "block_exact" and sug[0].learned.plan


def test_grasp_shift_is_invisible_to_encoders():
    rec = W.run_watched("box", CELL, seed=0, inject=dict(kind="shift_obj", dx=0.01, t=2.0), continue_through_u=True)
    assert not rec["fired"], rec["fired"]
