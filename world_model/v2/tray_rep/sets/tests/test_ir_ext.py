"""IR extensions of 2026-10-26 (exact Clear, Ball, Unc.ball, ClearEgress): the existing atoms are unchanged by default, the new ones match closed forms / brute force."""
import math

import numpy as np
import pytest
from scipy.optimize import minimize

from world_model.v2.tray_rep.sets import ir
from world_model.v2.tray_rep.sets.ir import Entity, Pose, Unc, Point, Ball, BoxShape, Sphere, prism_xy
from world_model.v2.tray_rep.sets.zono import Zonotope, sd_origin


class S:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def ent(name, shape, unc=False, parent=None):
    return Entity(name, shape, lambda s, n=name: Pose(*getattr(s, n)), (lambda s, n=name: getattr(s, n + "_unc")) if unc else None, parent)


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def box_point_sd(half, centre, R, p, q):
    """closed form: signed distance of the point q to the box (half, centre) posed (R, p)."""
    x = R.T @ (q - p) - centre
    s = half - np.abs(x)
    if s.min() >= 0:
        return -s.min()
    return float(np.linalg.norm(np.maximum(-s, 0.0)))


# ------------------------------------------------------------------ exact signed distance
def test_sd_exact_equals_the_box_point_closed_form():
    rng = np.random.default_rng(0)
    for _ in range(400):
        half = rng.uniform(0.05, 0.6, 3)
        cen = rng.uniform(-0.3, 0.3, 3)
        R = rotz(rng.uniform(-math.pi, math.pi))
        p, q = rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3)
        Z = Zonotope(cen, np.diag(half)).linear(R, p).minkowski(Zonotope(q).neg())
        assert abs(ir.sd_exact(Z.c, Z.G) - box_point_sd(half, cen, R, p, q)) < 1e-9


def test_sd_exact_vs_brute_force_on_random_zonotopes():
    rng = np.random.default_rng(1)
    n_edge = 0
    for _ in range(120):
        m = int(rng.integers(3, 9))
        G = rng.normal(size=(3, m)) * rng.uniform(0.05, 0.4)
        c = rng.normal(size=3) * 1.2
        s = ir.sd_exact(c, G)
        # independent: multi-start L-BFGS-B on |c + G l|^2, l in [-1, 1]^m (convex: any local minimum is global)
        best = min(minimize(lambda l: float(np.sum((c + G @ l) ** 2)), rng.uniform(-1, 1, m), jac=lambda l: 2 * G.T @ (c + G @ l),
                            bounds=[(-1, 1)] * m, method="L-BFGS-B", options=dict(ftol=1e-15, gtol=1e-12, maxiter=500)).fun for _ in range(3))
        if s > 0:
            assert abs(s - math.sqrt(best)) < 1e-6
        else:                                                       # inside: the minimum is 0; the depth is positive and below every facet distance
            assert best < 1e-9
            n_edge += 1
            assert s >= -np.abs(G).sum() - 1e-9
        assert sd_origin(Zonotope(c, G)) <= s + 1e-9                  # the grid version is a lower bound
    assert n_edge > 0


def test_sd_exact_flat_and_segment_cases():
    a, b = np.array([1.0, 0.5, 0.0]), np.array([2.0, -0.5, 0.0])
    Z = Zonotope((a + b) / 2, ((b - a) / 2)[:, None])               # a segment (rank 1)
    t = np.clip(-(a @ (b - a)) / ((b - a) @ (b - a)), 0, 1)
    assert abs(ir.sd_exact(Z.c, Z.G) - np.linalg.norm(a + t * (b - a))) < 1e-9
    P = Zonotope(np.array([0.0, 0.0, 0.3]), np.diag([1.0, 1.0, 0.0]))   # a flat square 0.3 above the origin
    assert abs(ir.sd_exact(P.c, P.G) - 0.3) < 1e-9


def test_sd_exact_far_returns_a_lower_bound():
    Z = Zonotope(np.array([[10.0, 0.0, 0.0], [0.5, 0.0, 0.0]]), np.broadcast_to(np.eye(3) * 0.1, (2, 3, 3)))
    far = ir.sd_exact_batch(Z, far=3.0)
    assert far[0] >= 3.0 and far[0] <= 9.9 + 1e-9 and abs(far[1] - 0.4) < 1e-9


# ------------------------------------------------------------------ Clear: default unchanged, exact, Ball, Unc.ball
def test_default_clear_is_unchanged():
    a = ent("a", BoxShape([0.3, 0.2, 0.1]))
    b = ent("b", Sphere(0.1))
    st = S(a=(rotz(0.4), np.array([0.0, 0.0, 0.0])), b=(np.eye(3), np.array([0.8, 0.3, 0.0])))
    ZA = BoxShape([0.3, 0.2, 0.1]).zonotope().linear(rotz(0.4), np.zeros(3))
    ZB = Sphere(0.1).zonotope().linear(np.eye(3), np.array([0.8, 0.3, 0.0]))
    want = sd_origin(ZA.minkowski(ZB.neg())) - 0.05
    assert ir.Clear(a, b, 0.05).eval(st)[0] == want and ir.margin(ir.Clear(a, b, 0.05), st) == want


def test_clear_exact_ball_equals_box_distance_minus_radius():
    half, cen = np.array([0.37, 0.39, 0.5]), np.array([0.2, -0.25, 0.65])
    fp = ent("fp", BoxShape(half, cen))
    pe = ent("pe", Ball(0.3))
    rng = np.random.default_rng(2)
    K = 60
    th = rng.uniform(-math.pi, math.pi, K)
    R = np.stack([rotz(t) for t in th])
    p = np.concatenate([rng.uniform(-1, 1, (K, 2)), np.zeros((K, 1))], 1)
    q = np.concatenate([rng.uniform(-1, 1.5, (K, 2)), np.full((K, 1), 0.65)], 1)
    st = S(fp=(R, p), pe=(np.broadcast_to(np.eye(3), (K, 3, 3)), q))
    got = ir.Clear(fp, pe, 0.1, exact=True).eval(st)
    want = np.array([box_point_sd(half, cen, R[k], p[k], q[k]) for k in range(K)]) - 0.3 - 0.1
    assert np.abs(got - want).max() < 1e-9


def test_unc_ball_enters_only_the_robust_margin_and_exactly():
    a, b = ent("a", BoxShape([0.1, 0.1, 0.1]), unc=True), ent("b", Ball(0.2), unc=True)
    st = S(a=(np.eye(3), np.zeros(3)), b=(np.eye(3), np.array([1.0, 0.0, 0.0])), a_unc=Unc(None, 0.0, 0.05), b_unc=Unc(None, 0.0, np.array(0.1)))
    cl = ir.Clear(a, b, 0.0, exact=True)
    assert abs(cl.eval(st, False)[0] - (1.0 - 0.1 - 0.2)) < 1e-12
    assert abs(cl.eval(st, True)[0] - (1.0 - 0.1 - 0.2 - 0.05 - 0.1)) < 1e-12
    cg = ir.Clear(a, b, 0.0)                                           # the default (grid) path honours Ball / Unc.ball too
    assert abs(cg.eval(st, True)[0] - (1.0 - 0.1 - 0.2 - 0.05 - 0.1)) < 1e-2
    assert cg.eval(st, True)[0] <= 1.0 - 0.1 - 0.2 - 0.05 - 0.1 + 1e-12      # lower bound


def test_unc_ball_in_inside_and_support():
    tray = ent("tray", prism_xy(0.15, 0.10))
    ball = ent("ball", Point(), unc=True)
    mk = lambda r: S(tray=(np.eye(3), np.zeros(3)), ball=(np.eye(3), np.array([0.05, 0.0, 0.0])), ball_unc=Unc(None, 0.0, r))
    sp = ir.Inside(ball, tray, 0.0)
    nom = ir.margin(sp, mk(0.03), False)
    assert abs(ir.margin(sp, mk(0.03), True) - (nom - 0.03)) < 1e-12 and abs(ir.margin(sp, mk(0.0), True) - nom) < 1e-12
    # K axis: per-time radius
    K = 3
    st = S(tray=(np.broadcast_to(np.eye(3), (K, 3, 3)), np.zeros((K, 3))), ball=(np.broadcast_to(np.eye(3), (K, 3, 3)), np.tile([0.05, 0, 0], (K, 1))),
           ball_unc=Unc(None, 0.0, np.array([0.0, 0.02, 0.04])))
    assert np.allclose(sp.eval(st, True), nom - np.array([0.0, 0.02, 0.04]))


def test_clear_rows_subtract_the_ball_radius():
    a, b = ent("a", BoxShape([0.1, 0.1, 0.1]), unc=True), ent("b", Ball(0.2))
    st = S(a=(np.eye(3), np.array([1.0, 0.0, 0.0])), b=(np.eye(3), np.zeros(3)), a_unc=Unc(None, 0.0, 0.05))
    jac = {"u0": np.zeros(1), "a": {"p": np.array([[1.0], [0.0], [0.0]])}}
    A0, b0, _ = ir.rows(ir.Clear(a, ent("b0", Point()), 0.0), S(a=st.a, b0=(np.eye(3), np.zeros(3)), a_unc=Unc(None, 0.0, 0.0)), jac, robust=True)
    A1, b1, _ = ir.rows(ir.Clear(a, b, 0.0), st, jac, robust=True)
    assert np.allclose(A0, A1) and np.allclose(b0 - b1, 0.2 + 0.05)


# ------------------------------------------------------------------ ClearEgress
def _traj_state(xs, vel_c=None):
    """box a (half 0.1) moving along x at xs[k]; fixed ball b of radius 0.2 at x = 1.0 (a 'person')."""
    K = len(xs)
    p = np.stack([np.asarray(xs, float), np.zeros(K), np.zeros(K)], 1)
    return S(a=(np.broadcast_to(np.eye(3), (K, 3, 3)), p), b=(np.broadcast_to(np.eye(3), (K, 3, 3)), np.tile([1.0, 0.0, 0.0], (K, 1))))


def _eg(d=0.1):
    return ir.ClearEgress(ent("a", BoxShape([0.1, 0.1, 0.1])), ent("b", Ball(0.2)), d)


def test_clear_egress_equals_clear_when_the_start_is_clear():
    st = _traj_state([0.0, 0.05, 0.1, 0.2, 0.3])
    eg, cl = _eg(), ir.Clear(ent("a", BoxShape([0.1, 0.1, 0.1])), ent("b", Ball(0.2)), 0.1, exact=True)
    assert np.allclose(eg.eval(st, True), cl.eval(st, True)) and eg.parts(st)["exit"] is None


def test_clear_egress_start_inside_keepout():
        # a (half 0.1) at x, the person ball (r 0.2) at x = 1.0, d = 0.1:  m(x) = (1.0 - x - 0.1) - 0.2 - 0.1 = 0.6 - x
    away = _traj_state([0.65, 0.58, 0.50, 0.40, 0.30])               # m_0 = -0.05 < 0, leaves the keep-out at k = 1 moving away
    eg = _eg()
    p = eg.parts(away)
    assert p["m"][0] < 0 and p["exit"] > 0 and p["clear"] > 0
    assert eg.eval(away, True).min() > 0 and ir.margin(ir.G(eg, 0, None), away, True) > 0
    toward = _traj_state([0.65, 0.70, 0.75, 0.80, 0.85])             # approaches from inside the keep-out
    p = eg.parts(toward)
    assert p["exit"] < 0 and eg.eval(toward, True).min() < 0
    still = _traj_state([0.65] * 4)                                   # not approaching, never leaves: exit = 0 (admissible), clear = BIG
    p = eg.parts(still)
    assert abs(p["exit"]) < 1e-12 and p["clear"] == ir.BIG and eg.eval(still, True).min() >= -1e-12
    # leaves the keep-out and re-enters: the plain row binds after k1
    reenter = _traj_state([0.65, 0.55, 0.45, 0.55, 0.65])
    p = eg.parts(reenter)
    assert p["exit"] > 0 and p["clear"] < 0 and eg.eval(reenter, True).min() < 0
    one = _traj_state([0.65])                                          # single sample: exit placeholder 0
    assert eg.parts(one)["exit"] == 0.0


def test_clear_egress_dict_roundtrip_and_no_rows():
    a, b = ent("a", BoxShape([0.1, 0.1, 0.1])), ent("b", Ball(0.2))
    d = {"G": {"lo": 0, "hi": None, "of": {"clear_egress": {"a": "a", "b": "b", "d": "dp", "far": 1e9, "tag": "x"}}}}
    sp = ir.from_dict(d, {"a": a, "b": b}, {"dp": 0.1})
    assert isinstance(sp.c, ir.ClearEgress) and sp.c.exact and sp.c.far == 1e9 and sp.c.d == 0.1
    st = _traj_state([0.65, 0.6, 0.5, 0.4])
    assert np.allclose(sp.eval(st, True), ir.G(_eg(), 0, None).eval(st, True))
    with pytest.raises(ir.NotConvex):
        sp.rows(st, {"u0": np.zeros(1)})
    c2 = ir.from_dict({"clear": {"a": "a", "b": "b", "d": 0.0, "exact": True}}, {"a": a, "b": b}, {})
    assert isinstance(c2, ir.Clear) and c2.exact and not ir.from_dict({"clear": {"a": "a", "b": "b"}}, {"a": a, "b": b}, {}).exact
