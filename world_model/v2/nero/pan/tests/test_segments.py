"""tests for pan/segments.py and pan/segcert.py (simulation only, CPU)."""
import numpy as np
import pytest

from world_model.v2.nero.pan import segments as G
from world_model.v2.nero.pan import segcert as SC

CORNER = dict(placement="far_left", hidden_mass=0.4, perturb=dict(dx=0.015, dy=-0.015, yaw_deg=8.0, mu_scale=0.6))


@pytest.fixture(scope="module")
def ep():
    return G.Episode(CORNER, observe=None)


def test_snapshot_restore_is_exact(ep):
    ep.run_phases(ep.segment_phases("start"))
    sn = ep.snapshot()
    ep.run_phases([("hold", 0.1)])
    a = ep.d.qpos.copy()
    ep.restore(sn)
    ep.run_phases([("hold", 0.1)])
    assert np.abs(ep.d.qpos - a).max() == 0.0


def test_untouched_corner_pan_is_task_ready(ep):
    ok, mg, st = ep.in_R(reach=False)
    assert ok, mg
    assert mg["rim"] > 0 and mg["yaw"] > 0


def test_unsafe_tilt_and_lid(ep):
    sn = ep.snapshot()
    a = ep.pan_qadr
    q = ep.d.qpos[a + 3:a + 7].copy()
    import mujoco
    mujoco.mju_mulQuat(ep.d.qpos[a + 3:a + 7], np.array([np.cos(np.radians(10)), np.sin(np.radians(10)), 0, 0]), q)   # 20 deg roll
    mujoco.mj_forward(ep.m, ep.d)
    assert ep.pan_state()["tilt"] > G.U["tilt_deg"]
    ep.restore(sn)
    la = ep.m.jnt_qposadr[ep.m.body(ep.lid).jntadr[0]]
    ep.d.qpos[la] += 0.02
    mujoco.mj_forward(ep.m, ep.d)
    assert ep.pan_state()["lid_mm"] > G.U["lid_mm"]
    ep.restore(sn)


def test_cma_minimises_quadratic():
    es = SC.CMA(np.zeros(3), 0.5, 8, seed=1)
    for _ in range(40):
        X = es.ask()
        es.tell(X, [float(np.sum((x - 0.3) ** 2)) for x in X])
    assert np.abs(es.m - 0.3).max() < 0.02


def test_clopper_pearson():
    assert abs(SC.cp_upper(0, 100) - (1 - 0.05 ** (1 / 100))) < 1e-6
    assert SC.cp_upper(5, 5) == 1.0
