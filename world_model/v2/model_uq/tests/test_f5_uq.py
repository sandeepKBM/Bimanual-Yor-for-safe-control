"""const eps in UQReachModel == ReachModel(eps_a) exactly (centre, hw, J)."""
import numpy as np
from world_model.v2.tray_rep.sets import reach as RC
from world_model.v2.model_uq.f5_uq import UQReachModel, ConstEps


def _case(rng):
    x = np.array([rng.normal(0, .03), rng.normal(0, .03), rng.normal(0, .1), rng.normal(0, .1)])
    P = np.diag([1e-5, 1e-5, 1e-3, 1e-3])
    c0 = np.array([rng.normal(0, .05), rng.normal(0, .05), 1.0]); c0 /= np.linalg.norm(c0)
    vt0 = rng.normal(0, .2, 3); w0 = rng.normal(0, .2, 3) * np.array([1, 1, 0])
    u0 = np.concatenate([rng.normal(0, .5, 3), rng.normal(0, .3, 3)])
    return x, P, c0, vt0, w0, u0


def test_const_equiv():
    rng = np.random.default_rng(0)
    for kd, kp in [(3.0, 2.0), (0.0, 0.0)]:
        p = RC.ReachParams(eps_a=0.08, kd=kd, kp=kp, a_max=1.5, w_max=1.0, alpha=10.0)
        m0, m1 = RC.ReachModel(p), UQReachModel(RC.ReachParams(**p.__dict__), ConstEps(0.08))
        for _ in range(5):
            a = _case(rng)
            r0, r1 = m0.predict(*a), m1.predict(*a)
            assert np.allclose(r0.hw, r1.hw, atol=1e-12), np.abs(r0.hw - r1.hw).max()
            assert np.allclose(r0.centre, r1.centre) and np.allclose(r0.J, r1.J)


if __name__ == "__main__":
    test_const_equiv(); print("ok")
