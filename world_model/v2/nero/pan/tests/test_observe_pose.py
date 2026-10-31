"""observe_pose: robust fits on synthetic points, the aim convention, and a clean render (nominal + one known shift).  SIM ONLY."""
import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")

from world_model.v2.nero.pan import observe_pose as O


def test_circle_fit_robust_to_outliers():
    rng = np.random.default_rng(0)
    a = rng.uniform(np.pi / 2, 3 * np.pi / 2, 800)                      # a half arc, like the visible wall
    xy = np.c_[0.3 + 0.0925 * np.cos(a), -0.2 + 0.0925 * np.sin(a)] + rng.normal(0, 3e-4, (800, 2))
    xy[:80] = rng.uniform([0.2, -0.3], [0.4, -0.1], (80, 2))           # 10 % junk
    c, r, rms = O._circle(xy, np.array([0.31, -0.19]), 0.0925)
    assert np.linalg.norm(c - [0.3, -0.2]) < 5e-4 and rms < 1e-3


def test_aim_transform_matches_perturb_convention():
    f, Rz = O.aim_transform(dict(dx=0.01, dy=-0.005, yaw_deg=90.0), (0.0, 0.0))
    assert np.allclose(f([1.0, 0.0, 0.5]), [0.01, 0.995, 0.5])
    assert np.allclose(Rz @ [1.0, 0, 0], [0, 1.0, 0], atol=1e-12)


@pytest.fixture(scope="module")
def nominal():
    import mujoco
    from world_model.v2.nero.pan import covered as CV, scene as S
    cp = CV.CoveredParams(placement="none")
    m = CV.compose(cp)
    d = mujoco.MjData(m)
    S.settle(m, d, 1.0)
    return m, d, cp.pan


def test_clean_estimate_nominal_and_shift(nominal):
    from world_model.v2.nero.pan import under_rim as U, scene as S
    m, d, p = nominal
    tm = O.template(m, d, p)
    e = O.estimate(m, d, p, tm, "clean")
    assert e["fit_ok"] and abs(e["dx"]) < 1e-4 and abs(e["dy"]) < 1e-4 and abs(e["yaw_deg"]) < 0.2
    pos0, R0 = d.xpos[m.body("pan").id].copy(), d.xmat[m.body("pan").id].reshape(3, 3).copy()
    U._perturb_pose(m, d, p.pan_centre_xy, 0.01, -0.008, 5.0)
    S.settle(m, d, 0.5)
    tru = O.true_pose(m, d, p, pos0, R0)
    e = O.estimate(m, d, p, tm, "clean")
    assert abs(e["dx"] - tru["dx"]) < 5e-4 and abs(e["dy"] - tru["dy"]) < 5e-4 and abs(e["yaw_deg"] - tru["yaw_deg"]) < 0.5
