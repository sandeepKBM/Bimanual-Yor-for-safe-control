"""BIMANUAL_CHECKER switch (step K2): carry_cert.static_cell under "legacy" and "unified" (stack/checks/admit via adapters/static_admission) returns EQUAL dicts.
LIVE vs LIVE, exact floats.  Offline: plans come from the closed_chain on-disk plan cache (or are planned locally), no proposer / VLM."""
import numpy as np
import pytest

from world_model.v2.nero.base import smoke as SM
from world_model.v2.nero.contents import carry_cert as CC
from world_model.v2.nero.contents import probe as P
from world_model.v2.stack.planners import closed_chain as CCP


def _both(monkeypatch, *a, **k):
    monkeypatch.setenv("BIMANUAL_CHECKER", "legacy")
    leg = CC.static_cell(*a, **k)
    monkeypatch.setenv("BIMANUAL_CHECKER", "unified")
    uni = CC.static_cell(*a, **k)
    return leg, uni


def _equal(a, b, path=""):
    """first difference as a readable string, '' when equal (exact; np arrays element-wise)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return f"{path}: keys legacy-only {sorted(set(a) - set(b))} unified-only {sorted(set(b) - set(a))}"
        for k in a:
            d = _equal(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return ""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return f"{path}: len {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _equal(x, y, f"{path}[{i}]")
            if d:
                return d
        return ""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return "" if np.array_equal(np.asarray(a), np.asarray(b)) else f"{path}: legacy {a} unified {b}"
    return "" if (a == b or (a != a and b != b)) else f"{path}: legacy {a!r} unified {b!r}"


def _check(monkeypatch, *a, **k):
    leg, uni = _both(monkeypatch, *a, **k)
    assert _equal(leg, uni) == ""
    return leg


def _three(task):
    cs = [c[0] for c in CC.cells(task)]
    return [cs[0], cs[len(cs) // 2], cs[-1]]


def test_default_is_unified(monkeypatch):   # default flipped 11-16 after the owner gate (test_checker_gate_k2.py)
    monkeypatch.delenv("BIMANUAL_CHECKER", raising=False)
    assert CC.checker_impl() == "unified"
    monkeypatch.setenv("BIMANUAL_CHECKER", "unified")
    assert CC.checker_impl() == "unified"
    monkeypatch.setenv("BIMANUAL_CHECKER", "bogus")
    with pytest.raises(ValueError):
        CC.checker_impl()


@pytest.mark.parametrize("task", ["box", "tray4"])
@pytest.mark.parametrize("planner", ["closed_chain", "scripted"])
def test_fixed_base_cells(monkeypatch, task, planner):
    w = CC.world_default(task)
    for cell in (_three(task) if planner == "closed_chain" else _three(task)[1:2]):      # scripted: the middle cell only (slow continuation IK; ~2 min each when the machine is loaded)
        _check(monkeypatch, task, cell, w, planner=planner)


@pytest.mark.parametrize("task", ["box_behind", "tray6_behind"])
def test_behind_cells(monkeypatch, task):
    bt = task[:-len("_behind")]
    w = SM.task_world(bt)
    lo, hi = P.prior_contents(bt)
    for c in (lo, 0.5 * (lo + hi), hi):
        c = round(float(c), 4)
        _check(monkeypatch, task, (c, c), w, spec=P.interval_spec(bt, c, c), wp=CCP.default_waypoint(task, w))


def test_behind_box_com_restricted(monkeypatch):
    w = SM.task_world("box")
    lo, hi = P.prior_contents("box")
    c = round(0.5 * (lo + hi), 4)
    _check(monkeypatch, "box_behind", (c, c), w, spec=P.interval_spec("box", c, c, com=[0.05, -0.03, 0.10]), wp=CCP.default_waypoint("box_behind", w))


def test_proposer_waypoint_plan(monkeypatch):
    """a WaypointPlan as the proposer writes it (box fixture shape: lift, carry over the shelf, place)."""
    wp = CCP.WaypointPlan(("s0_lift", "s1_move", "s2_place"), ((0.0, 0.0, 0.18), (-0.20, 0.0, 0.0), (0.0, 0.0, -0.03)), ("set_down", "set_down", "hold"), True)
    w = CC.world_default("box")
    leg = _check(monkeypatch, "box", (1.0, 1.5), w, wp=wp)
    assert not leg["action"].startswith("refuse (unreachable")
    assert leg["wp"] == wp.key_data() and leg["planner"] == "closed_chain"


def test_refused_unreachable(monkeypatch):
    wp = CCP.WaypointPlan(("s0_lift", "s1_move", "s2_place"), ((0.0, 0.0, 0.18), (-3.0, 0.0, 0.0), (0.0, 0.0, -0.18)), ("set_down", "set_down", "hold"), True)
    leg = _check(monkeypatch, "box", (1.0, 1.5), CC.world_default("box"), wp=wp)
    assert leg["action"] == "refuse (unreachable)"


def test_tcap_stays_legacy(monkeypatch):
    from world_model.v2.nero.contents import envelope as E
    w = CC.world_default("box")
    monkeypatch.setenv("BIMANUAL_CHECKER", "legacy")
    leg = CC.static_cell("box", (0.5, 1.0), w, T_cap=E.T_CAP, planner="scripted")
    monkeypatch.setenv("BIMANUAL_CHECKER", "unified")
    called = []
    from world_model.v2.adapters import static_admission as SA
    monkeypatch.setattr(SA, "admit_static", lambda *a, **k: called.append(1))
    uni = CC.static_cell("box", (0.5, 1.0), w, T_cap=E.T_CAP, planner="scripted")
    assert not called and _equal(leg, uni) == "" and uni["T_cap"] == E.T_CAP
