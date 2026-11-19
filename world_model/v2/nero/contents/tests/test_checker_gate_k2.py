"""K2 gate (owner-written, 11-16): the live static admission gives the SAME verdict dict under BIMANUAL_CHECKER=legacy and =unified.
SIMULATION ONLY.  Live code against live code (design 5.3: cached RESULT files were made on an older object position).

Wider than the agent's test: EVERY cell of box / tray2 / tray4 / tray6 on both planners, point and interval specs of the table-behind tasks
across the prior on the default waypoint plan, two COM-restricted specs, and the T_cap rule (must stay legacy).
Equality is exact (json of the dict with sorted keys), so a float that differs in the last bit fails and names the key.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from world_model.v2.nero.contents import carry_cert as CC
from world_model.v2.nero.contents import envelope as E
from world_model.v2.nero.contents import probe as P
from world_model.v2.nero.base import smoke as SM
from world_model.v2.stack.planners import closed_chain as CCP


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def _both(monkeypatch, *a, **k):
    out = {}
    for impl in ("legacy", "unified"):
        monkeypatch.setenv("BIMANUAL_CHECKER", impl)
        out[impl] = _plain(CC.static_cell(*a, **k))
    return out["legacy"], out["unified"]


TOL_M = 1e-6          # USER 11-17 "Compare with tolerance": the unified support row reads the plan's end pose (static_admission.plan_end_xy), 5.4e-8 m
                      # from scene.place_pose_xy; numbers equal within TOL_M; which of two tied corners is named in support_dest.worst is ignored


def _num(x):
    return isinstance(x, (int, float, np.number)) and not isinstance(x, (bool, np.bool_))


def _diff(a, b, path=""):
    if _num(a) and _num(b):
        fa, fb = float(a), float(b)
        return [] if (fa == fb or abs(fa - fb) <= TOL_M or (np.isnan(fa) and np.isnan(fb))) else [f"{path}: {a!r} vs {b!r}"]     # fa == fb first: inf == inf
    if path.endswith("support_dest.worst.c"):
        return []
    if type(a) is not type(b):
        return [f"{path}: type {type(a).__name__} vs {type(b).__name__}"]
    if isinstance(a, dict):
        d = [f"{path}.{k}: missing in {'unified' if k in a else 'legacy'}" for k in set(a) ^ set(b)]
        for k in set(a) & set(b):
            d += _diff(a[k], b[k], f"{path}.{k}")
        return d
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: len {len(a)} vs {len(b)}"]
        return [x for i, (u, v) in enumerate(zip(a, b)) for x in _diff(u, v, f"{path}[{i}]")]
    return [] if (a == b or (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b))) else [f"{path}: {a!r} vs {b!r}"]


def _assert_same(leg, uni, label):
    d = _diff(leg, uni)
    assert not d, f"{label}: {len(d)} differences, first: {d[:5]}"


@pytest.mark.parametrize("task", ["box", "tray2", "tray4", "tray6"])
@pytest.mark.parametrize("planner", ["scripted", "closed_chain"])
def test_every_cell_equal(monkeypatch, task, planner):
    w = CC.world_default(task)
    for cell, *_ in CC.cells(task):
        leg, uni = _both(monkeypatch, task, tuple(cell), w, planner=planner)
        _assert_same(leg, uni, f"{task}/{planner}/{cell}")


@pytest.mark.parametrize("task", ["box_behind", "tray6_behind"])
def test_behind_point_and_interval_specs_equal(monkeypatch, task):
    bt = task.replace("_behind", "")
    w = SM.task_world(bt)
    wp = CCP.default_waypoint(task, w)
    lo, hi = P.prior_contents(bt)
    for c in np.linspace(lo, hi, 5):
        c = round(float(c), 4)
        leg, uni = _both(monkeypatch, task, (c, c), w, spec=P.interval_spec(bt, c, c), wp=wp)
        _assert_same(leg, uni, f"{task}/point {c}")
    leg, uni = _both(monkeypatch, task, (round(lo, 4), round(hi, 4)), w, spec=P.interval_spec(bt, lo, hi), wp=wp)
    _assert_same(leg, uni, f"{task}/prior interval")


@pytest.mark.parametrize("com", [[0.0, 0.0, 0.05], [0.08, -0.05, 0.12]])
def test_box_behind_com_restricted_equal(monkeypatch, com):
    w = SM.task_world("box")
    wp = CCP.default_waypoint("box_behind", w)
    for c in (1.0, 2.5, 4.0):
        leg, uni = _both(monkeypatch, "box_behind", (c, c), w, spec=P.interval_spec("box", c, c, com=com), wp=wp)
        _assert_same(leg, uni, f"box_behind/com {com}/{c}")


def test_default_is_unified_and_tcap_stays_legacy(monkeypatch):   # default flipped 11-16 after this gate passed 13/13
    monkeypatch.delenv("BIMANUAL_CHECKER", raising=False)
    assert CC.checker_impl() == "unified"
    w = CC.world_default("box")
    cell = tuple(CC.cells("box")[2][0])
    monkeypatch.setenv("BIMANUAL_CHECKER", "legacy")
    leg = _plain(CC.static_cell("box", cell, w, T_cap=E.T_CAP))
    monkeypatch.setenv("BIMANUAL_CHECKER", "unified")
    uni = _plain(CC.static_cell("box", cell, w, T_cap=E.T_CAP))
    _assert_same(leg, uni, "box/T_cap")
    assert json.dumps(leg, sort_keys=True) == json.dumps(uni, sort_keys=True)
