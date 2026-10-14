"""reach.py: the ball model against the repo's reference QS model and against MuJoCo; Jacobians; set properties."""
import numpy as np
import pytest

from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep.sets import reach as RC
from world_model.v2.testbeds.b_tray_ball import analytic as AN


def run_model(x0, c0, vt0, w0, U, mu_eff):
    """roll the ball model under held commands U (K,6), no backup (all ticks scripted). Returns S (K+1,4)."""
    c, vt, w = np.array(c0, float), np.array(vt0, float), np.array(w0, float)
    s = np.array(x0, float)
    out = [s.copy()]
    for u in U:
        c2, vt2, w2 = RC.tray_step(c, vt, w, u[:3].copy(), u[3:].copy(), 10.0)
        s, _, _ = RC.ball_step(s, c, c2, u[:3].copy(), mu_eff, RC.K_IN, RC.R_BALL)
        c, vt, w = c2, vt2, w2
        out.append(s.copy())
    return np.array(out)


def test_ball_step_matches_qs_reference_without_friction():
    """analytic.qs_rollout (mu_roll = 0) is the reference. Tray: tilt about x then y, plus a lateral acceleration."""
    K = 25
    U = np.zeros((K, 6))
    U[:10, 3] = 0.4
    U[5:20, 4] = -0.3
    U[:15, 0] = 0.3
    S = run_model([0.01, -0.01, 0.0, 0.0], [0, 0, 1.0], [0, 0, 0], [0, 0, 0], U, 0.0)
    pos = np.zeros(3)
    v = np.zeros(3)
    quat = np.array([1.0, 0, 0, 0])
    # the reference takes the slew-limited rates; reproduce them
    w = np.zeros(3)
    Ur = U.copy()
    for k in range(K):
        w = w + np.clip(U[k, 3:] - w, -10.0 * RC.DT, 10.0 * RC.DT)
        Ur[k, 3:] = w
    xs, ys = AN.qs_rollout(0.01, -0.01, 0.0, 0.0, pos, v, quat, Ur, RC.R_BALL, RC.K_IN, 0.0, 4, 1.5, RC.DT / 4)
    err = np.hypot(S[1:, 0] - xs, S[1:, 1] - ys)
    assert err.max() < 2.5e-3, err.max()


def test_ball_model_vs_mujoco_hold_and_tilt():
    """a short tilt-and-hold in MuJoCo (hidden mu_roll known to the test only) vs the model at the calibrated resistance."""
    errs = []
    for mu in (7e-4, 2e-3, 5e-3):
        renv = E.TrayRepEnv("full", 0.3, False, w_scale=0.5, alpha=10.0)
        renv.reset(ball_xy=(0.0, 0.0), mu_roll=mu)
        K = 30
        U = np.zeros((K, 6))
        U[:8, 3] = 0.5
        U[8:16, 3] = -0.5
        S = run_model([0, 0, 0, 0], [0, 0, 1.0], [0, 0, 0], [0, 0, 0], U, RC.MU_EFF(mu))
        tr = [renv.current()["priv"][:2].astype(float)]
        for u in U:
            tr.append(renv.tick(u)["priv"][:2].astype(float))
        errs.append(np.abs(np.array(tr) - S[:, :2]).max())
    assert max(errs) < 8e-3, errs


def test_jacobian_matches_finite_difference():
    m = RC.ReachModel(RC.ReachParams(H=30))
    x = np.array([0.02, -0.01, 0.05, 0.02])
    P = np.diag([1e-6, 1e-6, 1e-3, 1e-3])
    c0, vt0, w0 = np.array([0.02, 0.01, 1.0]), np.array([0.3, 0.1, 0.0]), np.array([0.0, 0.1, 0.0])
    c0 /= np.linalg.norm(c0)
    u0 = np.array([0.2, -0.1, 0.0, 0.05, 0.1, 0.0])
    pr = m.predict(x, P, c0, vt0, w0, u0, jac=True)
    rng = np.random.default_rng(0)
    for _ in range(5):
        du = rng.normal(size=6) * np.array([0.05, 0.05, 0.02, 0.02, 0.02, 0.0])
        a = m.predict(x, P, c0, vt0, w0, u0 + du, jac=False).S_mid[:, :2]
        lin = pr.S_mid[:, :2] + np.einsum("kij,j->ki", pr.J, du)
        assert np.abs(a - lin).max() < 3e-3 + 0.05 * np.abs(a - pr.S_mid[:, :2]).max()


def test_set_properties():
    p = RC.ReachParams(H=40, eps_a=0.05)
    m = RC.ReachModel(p)
    x = np.array([0.0, 0.0, 0.1, 0.0])
    c0 = np.array([0, 0, 1.0])
    P1 = np.diag([1e-6, 1e-6, 1e-4, 1e-4])
    P2 = np.diag([9e-6, 9e-6, 4e-4, 4e-4])
    a = m.predict(x, P1, c0, np.zeros(3), np.zeros(3), np.zeros(6), jac=False)
    b = m.predict(x, P2, c0, np.zeros(3), np.zeros(3), np.zeros(6), jac=False)
    assert np.all(b.hw >= a.hw - 1e-12) and b.hw[0, 0] > a.hw[0, 0]            # bigger covariance -> bigger set
    # wider rolling-resistance set -> bigger set; the centre rollout is the same
    m2 = RC.ReachModel(RC.ReachParams(H=40, eps_a=0.05, mu_lo=5e-4, mu_hi=6e-3))
    m3 = RC.ReachModel(RC.ReachParams(H=40, eps_a=0.05, mu_lo=1.8e-3, mu_hi=2.2e-3))
    w, n = m2.predict(x, P1, c0, np.zeros(3), np.zeros(3), np.zeros(6), jac=False), m3.predict(x, P1, c0, np.zeros(3), np.zeros(3), np.zeros(6), jac=False)
    assert np.all(w.hw >= n.hw - 1e-12) and w.hw[-1, 0] > n.hw[-1, 0]
    # the mu band contains the rollouts at intermediate resistances
    for mu in (8e-4, 1.5e-3, 3e-3, 5e-3):
        S, _ = m2.roll(x, c0, np.zeros(3), np.zeros(3), np.zeros(6), RC.MU_EFF(mu), 1)
        pos = S[:, :2].copy()
        assert np.all(np.abs(pos - w.centre) <= w.hw + 1e-9)
    # a narrower set after a probe (BOED hook)
    m2.set_mu(1.8e-3, 2.2e-3)
    n2 = m2.predict(x, P1, c0, np.zeros(3), np.zeros(3), np.zeros(6), jac=False)
    assert np.allclose(n2.hw, n.hw)


def test_stop_certificate_bounds_the_slide():
    """no ball feedback, level tray at rest: a slow ball is stopped by friction, so the set stops growing (the stopping inflation)."""
    p = RC.ReachParams(H=60, kd=0.0, kp=0.0, mu_lo=3e-3, mu_hi=6e-3, eps_a=0.0)
    m = RC.ReachModel(p)
    x = np.array([0.0, 0.0, 0.03, 0.0])
    P = np.diag([1e-6, 1e-6, 1e-4, 1e-4])
    pr = m.predict(x, P, np.array([0, 0, 1.0]), np.zeros(3), np.zeros(3), np.zeros(6), jac=False)
    assert pr.k_stuck > 0
    assert np.allclose(pr.hw[-1], pr.hw[pr.k_stuck:], atol=1e-12) or np.ptp(pr.hw[pr.k_stuck:, 0]) < 1e-9
    # (the MuJoCo check of the sets is the coverage test in validate_reach.py)
