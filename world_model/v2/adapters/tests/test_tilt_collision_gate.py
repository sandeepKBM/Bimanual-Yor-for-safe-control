"""Owner gate (11-17): the tilt-aware load-world collision row (USER 11-17 "Checker uses measured tilt"; both ASSUMED values approved 11-17).
SIMULATION ONLY, CPU, no network.  PYTHONPATH=. python -m pytest -q world_model/v2/adapters/tests/test_tilt_collision_gate.py

Checks, independent of the agent's own scripts:
  1. swing_twist_lb (MODEL, per vertex): for every rotation R = R_swing(horizontal axis, <= sw) R_twist(z, <= tw), n.(R u) >= the bound
     (brute force, random + boundary draws);
  2. displacement_bound >= |R u - u| over the same set;
  3. a zero cover gives the untilted value (n.u) exactly;
  4. collision_rows without a tilt measurement is MISSING (fail closed); BIMANUAL_LEGACY_UNTILTED_COLLISION=1 restores the untilted row;
  5. TiltSpec: a piece no episode reached, or an unknown piece, raises TiltMissing (never a zero cover)."""
import math

import numpy as np
import pytest

from world_model.v2.adapters import collision_rows as RCOL


def _rot(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def _draws(rng, sw, tw, n):
    out = []
    for i in range(n):
        th = rng.uniform(0, 2 * np.pi)
        phi = sw if i % 2 == 0 else rng.uniform(0, sw)          # half on the swing boundary
        psi = rng.choice([-tw, tw]) if i % 3 == 0 else rng.uniform(-tw, tw)
        out.append(_rot([math.cos(th), math.sin(th), 0.0], phi) @ _rot([0, 0, 1], psi))
    return out


@pytest.mark.parametrize("seed", range(6))
def test_swing_twist_lb_is_a_lower_bound(seed):
    rng = np.random.default_rng(1117 + seed)
    worst = np.inf
    for _ in range(40):
        U = rng.normal(size=(8, 3)) * rng.uniform(0.02, 0.3)
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        sw, tw = math.radians(rng.uniform(0, 12)), math.radians(rng.uniform(0, 3))
        lb = RCOL.swing_twist_lb(n, U, sw, tw)
        for R in _draws(rng, sw, tw, 60):
            val = (U @ R.T) @ n
            worst = min(worst, float((val - lb).min()))
    assert worst >= -1e-12, worst


def test_displacement_bound_covers_every_rotation():
    rng = np.random.default_rng(7)
    for _ in range(40):
        U = rng.normal(size=(8, 3)) * 0.2
        sw, tw = math.radians(rng.uniform(0, 12)), math.radians(rng.uniform(0, 3))
        db = RCOL.displacement_bound(U, sw, tw)
        for R in _draws(rng, sw, tw, 60):
            assert np.all(np.linalg.norm(U @ R.T - U, axis=1) <= db + 1e-12)


def test_zero_cover_is_untilted():
    rng = np.random.default_rng(3)
    U = rng.normal(size=(8, 3))
    n = rng.normal(size=3)
    n /= np.linalg.norm(n)
    np.testing.assert_allclose(RCOL.swing_twist_lb(n, U, 0.0, 0.0), U @ n, atol=1e-12)
    assert np.all(RCOL.displacement_bound(U, 0.0, 0.0) == 0.0)


def test_no_measurement_is_missing_and_legacy_restores(monkeypatch):
    from world_model.v2.stack.planners import closed_chain as CCP
    plan = CCP.plan_for(CCP._nominal_inputs("box", None), 0.5)
    monkeypatch.delenv("BIMANUAL_LEGACY_UNTILTED_COLLISION", raising=False)
    r = RCOL.collision_rows(plan)
    assert not r.ok and r.guarantee == "missing", (r.ok, r.guarantee, r.why)
    monkeypatch.setenv("BIMANUAL_LEGACY_UNTILTED_COLLISION", "1")
    r0 = RCOL.collision_rows(plan)
    assert r0.guarantee != "missing", r0.why


def test_tiltspec_never_defaults_to_zero():
    sp = RCOL.TiltSpec(per_piece=(("s0_lift", 0.05, 0.003, 0.001, 20), ("s1_move", 0.02, 0.002, 0.002, 0)),
                       pivot_obj=(0.0, 0.0, 0.0366), x0=(0.0, 0.0, 0.0), R0=tuple(np.eye(3).ravel()), n=20)
    assert sp.piece("s0_lift") == (0.05, 0.003, 0.001)
    with pytest.raises(RCOL.TiltMissing):
        sp.piece("s1_move")                                       # no episode reached it
    with pytest.raises(RCOL.TiltMissing):
        sp.piece("s2_place")                                      # not measured
    assert sp.envelope() == (0.05, 0.003, 0.002)
