"""tests for contents/coop_imp.py: grasp matrix algebra, static object wrench."""
import numpy as np

from world_model.v2.nero.contents import coop_imp as CI


def _W():
    return CI.grasp_matrix([np.array([0.0, 0.2, 0.1]), np.array([0.0, -0.2, 0.1])], np.array([0.05, 0.0, 0.0]))


def test_W_pinv_roundtrip_and_internal_part_zero():
    W = _W()
    h_o = np.array([1.0, -2.0, 30.0, 0.5, 4.0, -1.0])
    h = np.linalg.pinv(W) @ h_o
    assert np.allclose(W @ h, h_o)
    Ni = np.eye(12) - np.linalg.pinv(W) @ W
    assert np.allclose(Ni @ h, 0, atol=1e-9)
    # squeeze along the grasp line is internal
    e = np.array([0, 0.4, 0]) / 0.4
    hs = np.concatenate([-5 * e, np.zeros(3), 5 * e, np.zeros(3)])
    assert np.allclose(W @ hs, 0, atol=1e-9)


def test_zero_error_object_wrench_is_static_wrench():
    g = np.array([0, 0, -9.81])
    M, S = 6.0, 6.0 * np.array([0.02, -0.1, 0.03])
    acc = np.zeros(3) - g
    z = np.zeros(3)
    h = CI.object_wrench(M, S, acc, CI.GAINS, z, z, z, z)
    assert np.allclose(h[:3], [0, 0, M * 9.81]) and np.allclose(h[3:], np.cross(S, acc))
