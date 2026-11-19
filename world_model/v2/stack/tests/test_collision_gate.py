"""K5 gate (owner-written, 11-16): swept collision row (stack/checks/collision.py) + its adapter (adapters/collision_rows.py).  SIMULATION ONLY.

Design K5 test: the swept lower bound is never above the dense-sampled distance.  Checked (1) on analytic 1-D motions with a known exact minimum, and
(2) on the real closed-chain plans, choosing the intervals where the bound is weakest: the FASTEST intervals and the CLOSEST ones of each class.
"""
from __future__ import annotations

import numpy as np
import pytest

from world_model.v2.stack.checks import collision as K


def test_swept_bound_analytic():
    rng = np.random.default_rng(3)
    for _ in range(200):
        A, w, W, dt = rng.uniform(0.01, 0.2), rng.uniform(1, 30), rng.uniform(0.0, 0.3), rng.choice([0.005, 0.02, 0.05])
        t = np.arange(0.0, 1.0, dt)
        d = A * np.sin(w * t) + W                      # distance to a wall; |d'| <= A w
        v = np.full(len(t) - 1, A * w)
        _, lb = K.swept_lower_bound(d, v, t)
        tf = np.linspace(0, t[-1], 20001)
        df = A * np.sin(w * tf) + W
        k = np.minimum((tf / dt).astype(int), len(t) - 2)
        exact = np.full(len(t) - 1, np.inf)
        np.minimum.at(exact, k, df)
        assert np.all(lb <= exact + 1e-12)


def test_row_refutes_a_dip_between_samples():
    t = np.arange(0.0, 1.0, 0.05)
    d = np.full(len(t), 0.012)                          # every sample 12 mm clear
    v = np.full(len(t), 0.4)                            # 0.4 m/s closing: the bound dips 10 mm between samples -> 2 mm < 5 mm
    r = K.swept_clearance_row({"arm_arm": d}, {"arm_arm": v}, t, {"arm_arm": 0.005})
    assert not r.ok and r.guarantee == "refuted" and r.worst["cls"] == "arm_arm"
    r2 = K.swept_clearance_row({"arm_arm": d}, {"arm_arm": np.zeros(len(t))}, t, {"arm_arm": 0.005})
    assert r2.ok and r2.guarantee == "proof" and r2.margin == pytest.approx(0.007)
    d3 = d.copy(); d3[7] = 0.003
    r3 = K.swept_clearance_row({"obj_world": d3}, {"obj_world": np.zeros(len(t))}, t, {"obj_world": 0.005})
    assert not r3.ok and r3.worst["index"] in (6, 7)


CASES = [("box", "place"), ("tray4", "lift"), ("box_behind", "s2_place")]


@pytest.fixture(scope="module")
def plans():
    from world_model.v2.stack.planners import closed_chain as CCP
    return {task: CCP.plan_for(CCP._nominal_inputs(task, None), 0.5) for task in {c[0] for c in CASES}}


@pytest.mark.parametrize("task,sg", CASES)
def test_swept_bound_below_dense_on_plans(plans, task, sg):
    from world_model.v2.adapters import collision_rows as CR
    plan = plans[task]
    sc = CR._Scene(plan)
    t, res = CR.piece_distances(plan, sg, sc)
    n_checked = 0
    for cls, r in res.items():
        d, v = r["d"], r["v"][:-1]
        if not np.isfinite(d).any() or np.all(d >= sc.cap[cls] - 1e-12):
            continue
        _, lb = K.swept_lower_bound(d, v, t)
        pick = set(np.argsort(-v)[:4]) | set(np.argsort(lb)[:4])        # fastest + closest intervals
        for k in sorted(pick):
            dense = CR.dense_min_distance(plan, sg, cls, int(k), upsample=20, scene=sc)
            assert lb[k] <= dense + 1e-9, (task, sg, cls, int(k), float(lb[k]), dense, float(v[k]))
            n_checked += 1
    assert n_checked >= 8


@pytest.mark.skip(reason="RETIRED (USER 11-19): fixed-base task, out of scope; with the measured load tilt its nominal plan is refused at lift (4.4 mm < 5 mm, n=4 tilt). Listed in docs/status/checker_measured_set_plan_2026-11-19.md Part 3")
def test_collision_rows_admit_nominal_plans(plans):
    """the tilt-aware row (USER 11-17) admits the nominal plans with the load tilt MEASURED per plan (n = 4 ASSUMED test choice); its guarantee is "sampled"
    (the tilt is a max over n worlds, not a proof).  Without a tilt measurement the row is MISSING (fail closed): adapters/tests/test_tilt_collision_gate.py."""
    from world_model.v2.adapters import collision_rows as CR, contents_specs as CS
    from world_model.v2.adapters.tests.tilt_input import measured_tilt
    for task, plan in plans.items():
        lo, hi = CS.prior_total(task)
        r = CR.collision_rows(plan, tilt=measured_tilt(task, (lo, min(hi, lo + 0.5))))      # the lightest 0.5 kg cell of the prior
        assert r.ok and r.guarantee == "sampled", (task, r.guarantee, r.why)
        assert set(r.worst["by_class"]) == set(K.CLASSES), (task, r.worst["by_class"].keys())
