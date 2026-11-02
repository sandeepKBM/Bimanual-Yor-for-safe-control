"""tests for contents/envelope.py: sign convention vs MuJoCo inverse dynamics, corner-check exactness, LP optimality."""
import numpy as np
import mujoco
import pytest

from world_model.v2.nero import adapters as A
from world_model.v2.nero.ik import ArmIK
from world_model.v2.nero.pan import covered as CV
from world_model.v2.nero.contents import envelope as E, objects as OB

Q = dict(left=[0.67, -0.87, 1.97, 1.66, -0.14, -0.18, 0.82], right=[-1.03, -0.8, -1.69, 1.36, 0.0, 0.26, 1.39])   # pot pre-grasp (probe)


@pytest.fixture(scope="module")
def setup():
    m = CV.compose()
    d = mujoco.MjData(m)
    cap = dict(sides=["left", "right"], qpos_rest=d.qpos.tolist(), q={wp: dict(Q) for wp in E.WAYPOINTS})
    arms = E.Arms(cap, model=m)
    tl, tr = arms.terms[("W0_liftoff", "left")]["p"], arms.terms[("W0_liftoff", "right")]["p"]
    o0 = 0.5 * (tl + tr) - np.array([0, 0, 0.03])
    spec = OB.pan_like("pot", 0.3, np.array([0, 0, 0.01]))
    env = E.Envelope(spec, arms, o0, np.eye(3), ["left", "right"], grip_mass=0.5)
    return m, arms, env, spec


def test_torque_sign_matches_mj_inverse(setup):
    m, arms, env, _ = setup
    t = arms.terms[("W0_liftoff", "left")]
    w = np.array([1.5, -2.0, 6.0, 0.2, -0.1, 0.3])                 # wrench the arm applies to the object, about p
    tau_pred = t["G"] + t["J"].T @ w
    gc, fl = m.body_gravcomp.copy(), m.opt.disableflags
    m.body_gravcomp[:] = 0
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT     # the envelope is free-space arm + object wrench (no contacts)
    d = mujoco.MjData(m)
    for s in ("left", "right"):
        d.qpos[A.arm_addrs(m, s)[0]] = Q[s]
    mujoco.mj_forward(m, d)
    b = m.site_bodyid[m.site("left_arm_ee_grasp").id]
    p = d.site_xpos[m.site("left_arm_ee_grasp").id]
    F, T = -w[:3], -w[3:]                                           # the object pushes back on the arm
    d.xfrc_applied[b, :3] = F
    d.xfrc_applied[b, 3:] = T + np.cross(p - d.xipos[b], F)         # moved from p to the body COM
    d.qacc[:] = 0
    mujoco.mj_inverse(m, d)                                         # qfrc_inverse = actuator torque + J^T xfrc (MuJoCo docs)
    ext = np.zeros(m.nv)
    mujoco.mj_applyFT(m, d, F, -w[3:], p, b, ext)                   # J^T of the object's push on the arm, applied at p
    m.body_gravcomp[:] = gc
    m.opt.disableflags = fl
    da = A.arm_addrs(m, "left")[1]
    assert np.allclose(d.qfrc_inverse[da] - ext[da], tau_pred, atol=1e-6)


def test_corner_check_is_exact_on_random_interior(setup):
    _, _, env, spec = setup
    cont = spec.contents.scaled(1.0)
    for md in ("one", "two"):
        top, _ = env.worst(cont, ("W0_liftoff",), 0.3, md)
        rng = np.random.default_rng(0)
        V = cont.P
        for _ in range(40):
            mm = rng.uniform(cont.m_lo, cont.m_hi)
            lam = rng.dirichlet(np.ones(len(V)))
            c = lam @ V
            a = rng.uniform(-0.3, 0.3, 3)
            val, _ = env.u("W0_liftoff", np.concatenate([[mm], mm * c]), a, md)
            assert val <= top + 1e-7


def test_lp_beats_fixed_splits(setup):
    _, arms, env, spec = setup
    pi = np.array([0.8, 0.8 * 0.03, 0.0, 0.8 * 0.02])
    val, _ = env.u("W0_liftoff", pi, (0, 0, 0), "two")
    o = env.o0
    F, T = env.wrench(pi, (0, 0, 0), o)
    tl, tr = arms.terms[("W0_liftoff", "left")], arms.terms[("W0_liftoff", "right")]
    for alpha in np.linspace(0, 1, 11):                             # force split alpha / 1-alpha, moments balanced on the left
        FL, FR = alpha * F, (1 - alpha) * F
        TR = np.zeros(3)
        TL = T - np.cross(tl["p"] - o, FL) - np.cross(tr["p"] - o, FR) - TR
        r = max(np.max(np.abs(tl["G"] + tl["J"].T @ np.r_[FL, TL]) / env.tau_lim),
                np.max(np.abs(tr["G"] + tr["J"].T @ np.r_[FR, TR]) / env.tau_lim), abs(FL[2]) / env.F_cap, abs(FR[2]) / env.F_cap)
        assert val <= r + 1e-7


def test_contents_vertices_cover_mass_range():
    c = OB.Contents(0.2, 1.0, OB.prism(OB.circ_polygon(0.05, 8), 0.0, 0.04))
    V = c.pi_vertices()
    assert set(np.round(V[:, 0], 6)) == {0.2, 1.0} and len(V) == 32
    assert OB.cup_spill_limit(0.08, 0.095, 0.085) == pytest.approx(9.81 * 2 * 0.01 / 0.08)


def test_controller_split_balances_and_is_no_better_than_lp(setup):
    _, arms, env, spec = setup
    pi = np.array([0.6, 0.6 * 0.04, -0.6 * 0.02, 0.6 * 0.02])
    u_share, det = env.u("W0_liftoff", pi, (0.2, 0, 0), "two_share")
    u_lp, _ = env.u("W0_liftoff", pi, (0.2, 0, 0), "two")
    assert u_lp <= u_share + 1e-9                                     # the LP split is the best split
    F, _ = env.wrench(pi, (0.2, 0, 0), env.o0)
    assert np.isclose(det["Fz"]["left"] + det["Fz"]["right"], F[2], atol=1e-3)


def test_twist_polytope_inside_ball_and_tight_on_axes():
    A, b = E.TWIST_A, E.TWIST_B
    rng = np.random.default_rng(1)
    for _ in range(2000):                                           # every point of the polytope is inside the unit ball
        x = rng.normal(size=3)
        x /= np.linalg.norm(x)
        s = np.min(b / np.maximum(A @ x, 1e-12))                    # largest scale with A (s x) <= b
        assert s <= 1.0 + 1e-9
    for ax in np.eye(3):                                            # full capacity along each axis
        assert np.all(A @ ax <= b + 1e-9)


def test_lever_split_balances_and_lp_is_no_worse(setup):
    _, arms, env, spec = setup
    pl, pr = arms.terms[("W0_liftoff", "left")]["p"], arms.terms[("W0_liftoff", "right")]["p"]
    o = env.o0
    pi = np.array([0.7, 0.7 * 0.03, -0.7 * 0.05, 0.7 * 0.02])
    acc = np.array([0.2, -0.1, 0.3]) - E.G_VEC
    Mtot = spec.empty_mass + pi[0]
    S = spec.empty_mass * env.c0 + pi[1:4]
    FL, TL, FR, TR = E.lever_split(Mtot, acc, S, o, pl, pr)
    assert np.allclose(FL + FR, Mtot * acc)
    assert np.allclose(TL + TR + np.cross(pl - o, FL) + np.cross(pr - o, FR), np.cross(S, acc))
    u_lp, _ = env.u("W0_liftoff", pi, (0.2, -0.1, 0.3), "two")       # no twist row: the LP is optimal over ALL splits
    for mode in ("two_lever", "two_share"):
        assert u_lp <= env.u("W0_liftoff", pi, (0.2, -0.1, 0.3), mode)[0] + 1e-9
    env2 = E.Envelope(spec, arms, env.o0, env.R0, ["left", "right"], grip_mass=0.5, T_cap=E.T_CAP)
    u_lv, det = env2.u("W0_liftoff", pi, (0.0, 0.0, 0.0), "two_lever")
    u_sh, det_sh = env2.u("W0_liftoff", pi, (0.0, 0.0, 0.0), "two_share")
    assert max(det["twist"].values()) < max(det_sh["twist"].values())  # the lever split commands less claw twist
