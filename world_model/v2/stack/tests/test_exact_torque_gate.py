"""K10 gate (owner-written, 11-16): exact inverse-dynamics torque row (adapters/exact_torque.py, OPT-IN, not in admission).  SIMULATION ONLY.

  vertex = exact    (no rotation) the box row equals the max over its vertices evaluated one point at a time
  soundness         random contents (m in [m_lo, m_hi], COM a random point of hull(P)) never exceed the box row
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from world_model.v2.nero.contents.objects import Contents

TASKS = ("box", "tray4")


@pytest.fixture(scope="module")
def setup():
    from world_model.v2.stack.planners import closed_chain as CCP
    from world_model.v2.adapters import contents_specs as CS
    out = {}
    for task in TASKS:
        plan = CCP.plan_for(CCP._nominal_inputs(task, None), 0.5)
        spec = CS.spec_from_total(task, *CS.prior_total(task))
        out[task] = (plan, spec)
    return out


def _point(spec, m, c):
    return dataclasses.replace(spec, contents=Contents(m, m, np.asarray([c], float), spec.contents.labels))


@pytest.mark.parametrize("task", TASKS)
def test_vertex_max_is_exact_and_random_contents_never_exceed(setup, task):
    from world_model.v2.adapters import exact_torque as X
    plan, spec = setup[task]
    sg = list(plan.pieces)[0]
    u_box = X.exact_torque_row(plan, spec, pieces=[sg]).worst["u_max"]
    c = spec.contents
    u_vert = max(X.exact_torque_row(plan, _point(spec, m, v), pieces=[sg]).worst["u_max"] for m in sorted({c.m_lo, c.m_hi}) for v in c.P)
    assert u_vert == pytest.approx(u_box, rel=1e-9), (task, u_vert, u_box)
    rng = np.random.default_rng(31)
    P = np.asarray(c.P, float)
    for _ in range(25):
        w = rng.dirichlet(np.ones(len(P)))
        m = float(rng.uniform(c.m_lo, c.m_hi))
        u = X.exact_torque_row(plan, _point(spec, m, w @ P), pieces=[sg]).worst["u_max"]
        assert u <= u_box + 1e-12, (task, m, w @ P, u, u_box)


def test_second_moment_box_contains_sampled_points():
    from world_model.v2.adapters import exact_torque as X
    rng = np.random.default_rng(32)
    for _ in range(20):
        P = rng.uniform(-0.2, 0.2, (int(rng.integers(4, 12)), 3))
        lo, hi = X.second_moment_bounds(P)
        for _ in range(500):
            x = rng.dirichlet(np.ones(len(P))) @ P
            s = np.array([x[i] * x[j] for i, j in X._SIJ])
            assert np.all(s >= lo - 1e-12) and np.all(s <= hi + 1e-12), (P, x, s, lo, hi)
