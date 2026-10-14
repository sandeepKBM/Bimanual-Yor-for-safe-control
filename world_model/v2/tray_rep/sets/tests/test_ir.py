"""IR tests: relation semantics, robust <= nominal, three-valued status, logic and temporal operators, compiled rows vs margin,
and the OBJECT-AGNOSTIC demo: a 2-box stack spec that uses support() and inside() with no new relation code."""
import numpy as np
import pytest

from world_model.v2.tray_rep.sets import ir
from world_model.v2.tray_rep.sets.ir import (Entity, Pose, Unc, Point, Sphere, BoxShape, Region, prism_xy, shrink)
from world_model.v2.tray_rep.sets.zono import Zonotope, Polytope
from world_model.v2.tray_rep.sets import tray_spec as TS


class S:
    """a trivial state: dict of poses/uncertainties per entity name."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def ent(name, shape, parent=None, unc=None):
    return Entity(name, shape, lambda s, n=name: Pose(*getattr(s, n)), (lambda s, n=name: getattr(s, n + "_unc")) if unc else None, parent)


def I3():
    return np.eye(3)


def rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


# ------------------------------------------------------------------ inside
def test_inside_point_in_box_signed_distance():
    tray = ent("tray", prism_xy(0.15, 0.10))
    ball = ent("ball", Point())
    sp = ir.Inside(ball, tray, 0.0)
    for loc, expect in [((0.0, 0.0, 0.03), 0.10), ((0.12, 0.0, 0.03), 0.03), ((0.0, 0.09, 0.0), 0.01), ((0.16, 0.0, 0), -0.01),
                        ((0.16, 0.11, 0), -np.hypot(0.01, 0.01))]:
        st = S(tray=(I3(), np.zeros(3)), ball=(I3(), np.array(loc)))
        assert abs(ir.margin(sp, st) - expect) < 1e-12


def test_inside_sphere_equals_shrunk_point_and_margin_param():
    tray = ent("tray", prism_xy(0.15, 0.10))
    st = S(tray=(rotx(0.1), np.array([0.0, 0.0, 1.0])), ball=(I3(), rotx(0.1) @ np.array([0.05, 0.02, 0.03]) + np.array([0, 0, 1.0])))
    a = ir.Inside(ent("ball", Point()), shrink(tray, 0.03), 0.0)
    b = ir.Inside(ent("ball", Sphere(0.03)), tray, 0.0)
    c = ir.Inside(ent("ball", Point()), tray, 0.03)
    assert abs(ir.margin(a, st) - ir.margin(b, st)) < 1e-12 and abs(ir.margin(a, st) - ir.margin(c, st)) < 1e-12


def test_robust_margin_is_smaller_by_the_uncertainty_support():
    tray = ent("tray", prism_xy(0.15, 0.10))
    unc = Unc(Zonotope.from_box([-0.01, -0.02, 0], [0.01, 0.02, 0]), 0.0)
    ball = ent("ball", Point(), unc=True)
    sp = ir.Inside(ball, tray, 0.0)
    st = S(tray=(I3(), np.zeros(3)), ball=(I3(), np.array([0.05, 0.0, 0.0])), ball_unc=unc)
    nom, rob = ir.margin(sp, st, False), ir.margin(sp, st, True)
    assert abs(nom - 0.10) < 1e-12                     # y facet: 0.10 - 0
    assert abs(rob - 0.08) < 1e-12                     # y half-width 0.02 eats 2 cm; x facet gives 0.15-0.05-0.01 = 0.09
    assert rob <= nom


def test_rotation_uncertainty_inflates_by_lever_arm():
    tray = ent("tray", prism_xy(0.15, 0.10))
    ball = ent("ball", Sphere(0.03), unc=True)
    st = S(tray=(I3(), np.zeros(3)), ball=(I3(), np.array([0.05, 0.0, 0.0])), ball_unc=Unc(None, 0.1))
    sp = ir.Inside(ball, tray, 0.0)
    nom, rob = ir.margin(sp, st, False), ir.margin(sp, st, True)
    assert nom - rob > 0.1 * 0.03 - 1e-12


# ------------------------------------------------------------------ status
def test_three_valued_status():
    tray = ent("tray", prism_xy(0.15, 0.10))
    ball = ent("ball", Point(), unc=True)
    sp = ir.Inside(ball, tray, 0.0)
    unc = Unc(Zonotope.from_box([-0.03, -0.03, 0], [0.03, 0.03, 0]), 0.0)
    mk = lambda x: S(tray=(I3(), np.zeros(3)), ball=(I3(), np.array([x, 0.0, 0.0])), ball_unc=unc)
    assert ir.status(sp, mk(0.0)) == "true"
    assert ir.status(sp, mk(0.13)) == "unknown"        # centre inside by 2 cm, set sticks out by 1 cm
    assert ir.status(sp, mk(0.2)) == "false"


# ------------------------------------------------------------------ upright
def test_upright():
    tray = ent("tray", prism_xy(0.15, 0.10))
    sp = ir.Upright(tray, 0.2)
    st = S(tray=(rotx(0.15), np.zeros(3)))
    assert abs(ir.margin(sp, st) - 0.05) < 1e-12
    st = S(tray=(rotx(0.25), np.zeros(3)))
    assert abs(ir.margin(sp, st) + 0.05) < 1e-12


# ------------------------------------------------------------------ clear
def test_clear_distance_between_boxes():
    a = ent("a", BoxShape([0.05, 0.05, 0.05]))
    b = ent("b", BoxShape([0.05, 0.05, 0.05]))
    sp = ir.Clear(a, b, 0.0)
    st = S(a=(I3(), np.array([0.0, 0.0, 0.0])), b=(I3(), np.array([0.30, 0.0, 0.0])))
    assert abs(ir.margin(sp, st) - 0.20) < 5e-3
    st = S(a=(I3(), np.array([0.0, 0.0, 0.0])), b=(I3(), np.array([0.08, 0.0, 0.0])))
    assert ir.margin(sp, st) < 0                       # overlapping by 2 cm


# ------------------------------------------------------------------ logic and time
def test_logic_and_temporal_operators():
    tray = ent("tray", prism_xy(0.15, 0.10))
    ball = ent("ball", Point())
    xs = np.array([0.0, 0.05, 0.10, 0.13, 0.16, 0.05])
    R = np.broadcast_to(np.eye(3), (len(xs), 3, 3))
    p = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], 1)
    st = S(tray=(R, np.zeros((len(xs), 3))), ball=(R, p))
    base = ir.Inside(ball, tray, 0.0)
    v = ir.Inside(ball, tray, 0.0).eval(st)
    assert np.allclose(v, np.minimum(0.15 - np.abs(xs), 0.10)) or np.allclose(v[:5], np.minimum(0.15 - xs[:5], 0.10))
    g = ir.G(base, 0, None).eval(st)
    assert abs(g[0] - v.min()) < 1e-12 and abs(g[5] - v[5]) < 1e-12
    f = ir.F(base, 0, 2).eval(st)
    assert abs(f[0] - v[:3].max()) < 1e-12
    g12 = ir.G(base, 1, 2).eval(st)
    assert abs(g12[0] - v[1:3].min()) < 1e-12
    both = ir.And(base, ir.Upright(tray, 0.1))
    assert np.allclose(both.eval(st), np.minimum(v, 0.1))
    either = ir.Or(base, ir.Upright(tray, 0.5))
    assert np.allclose(either.eval(st), np.maximum(v, 0.5))
    assert np.allclose(ir.Not(base).eval(st), -v)


# ------------------------------------------------------------------ rows
def test_rows_match_margin_for_linear_motion():
    """ball position linear in u: p_k(u) = p0_k + J_k (u - u0). Rows A u <= b must hold exactly when the margin is >= 0."""
    rng = np.random.default_rng(0)
    K, nu = 5, 3
    p0 = np.stack([rng.uniform(-0.05, 0.05, K), rng.uniform(-0.04, 0.04, K), np.full(K, 0.03)], 1)
    J = np.zeros((K, 3, nu))
    J[:, :2, :] = rng.normal(size=(K, 2, nu)) * 0.05
    u0 = rng.normal(size=nu) * 0.1
    zhw = 0.01
    unc = lambda s: Unc(Zonotope(np.zeros((K, 3)), np.broadcast_to(np.diag([zhw, zhw, 0.0]), (K, 3, 3))), 0.0)
    tray = Entity("tray_top", prism_xy(0.15, 0.10), lambda s: Pose(np.broadcast_to(np.eye(3), (K, 3, 3)), np.zeros((K, 3))))
    ball = Entity("ball", Point(), lambda s: Pose(np.broadcast_to(np.eye(3), (K, 3, 3)), s.p), unc, parent="tray_top")
    sp = ir.G(ir.Inside(ball, tray, 0.001), 0, None)
    st = S(p=p0)
    A, b, tags = ir.rows(sp, st, {"u0": u0, "ball": {"p": J}})
    assert A.shape == (K * 4, nu)
    # rows hold iff the robust margin (which already includes the 1 mm inside margin) is >= 0
    for _ in range(300):
        u = u0 + rng.normal(size=nu) * 0.5
        pk = p0 + np.einsum("kij,j->ki", J, u - u0)
        m = ir.margin(ir.G(ir.Inside(ball, tray, 0.001), 0, None), S(p=pk), True)
        assert bool(np.all(A @ u <= b + 1e-10)) == (m >= -1e-10)


def test_rows_refuse_nonconvex():
    tray = ent("tray_top", prism_xy(0.15, 0.10))
    ball = Entity("ball", Point(), lambda s: Pose(np.eye(3), np.zeros(3)), parent="tray_top")
    with pytest.raises(ir.NotConvex):
        ir.rows(ir.Or(ir.Inside(ball, tray), ir.Inside(ball, tray)), None, {"u0": np.zeros(2), "ball": {"p": np.zeros((3, 2))}})
    with pytest.raises(ir.NotConvex):
        ir.rows(ir.F(ir.Inside(ball, tray)), None, {"u0": np.zeros(2), "ball": {"p": np.zeros((3, 2))}})


def test_declarative_dict_equals_python_tree():
    tray = ent("tray", prism_xy(0.15, 0.10))
    ents = {"tray": tray, "tray_top": tray, "ball": ent("ball", Point())}
    sp = TS.build_spec(TS.TRAY_SPEC_FULL, ents, H=3, r_shrink=0.01, theta_max=0.2, lo=0)
    K = 4
    R = np.broadcast_to(rotx(0.05), (K, 3, 3))
    p = np.stack([np.linspace(0, 0.1, K), np.zeros(K), np.zeros(K)], 1)
    st = S(tray=(R, np.zeros((K, 3))), ball=(np.broadcast_to(np.eye(3), (K, 3, 3)), p))
    ref = ir.G(ir.And(ir.Inside(ents["ball"], shrink(tray, 0.01), 0.0), ir.Upright(tray, 0.2)), 0, 3)
    assert abs(ir.margin(sp, st) - ir.margin(ref, st)) < 1e-14


# ------------------------------------------------------------------ the IR is not tray-specific: a 2-box stack, no new relation code
def test_two_box_stack_spec_uses_only_support_and_inside():
    """Spec for 'top box rests stably on the bottom box, bottom box on the table':
         support(top, bottom)  AND  inside(bottom, table_region)  AND  upright(top, 0.05)
    Entities are plain boxes with world poses; nothing here is specific to a tray or a ball."""
    table = ent("table", Region(Polytope.box([0.5, 0.5, 1.0], axes=[0, 1])))
    bottom = ent("bottom", BoxShape([0.05, 0.05, 0.05]), unc=True)
    top = ent("top", BoxShape([0.05, 0.05, 0.05]), unc=True)
    spec = ir.from_dict({"and": [{"support": {"a": "top", "b": "bottom"}},
                                 {"inside": {"a": "bottom", "b": "table"}},
                                 {"upright": {"a": "top", "theta_max": 0.05}}]},
                        {"table": table, "bottom": bottom, "top": top}, {})
    zero = np.zeros(3)
    mk = lambda dx, tilt=0.0, u=0.0: S(table=(I3(), zero), bottom=(I3(), np.array([0.1, 0.0, 0.05])), top=(rotx(tilt), np.array([0.1 + dx, 0.0, 0.15])),
                                       bottom_unc=Unc(Zonotope.from_box([-u, -u, 0], [u, u, 0]), 0.0),
                                       top_unc=Unc(Zonotope.from_box([-u, -u, 0], [u, u, 0]), 0.0))
    assert ir.margin(spec, mk(0.0)) > 0                                       # aligned: stable
    assert abs(ir.margin(ir.Support(top, bottom), mk(0.03)) - 0.02) < 1e-12  # COM 3 cm off-centre of a 5 cm half-width face: 2 cm left
    assert ir.margin(ir.Support(top, bottom), mk(0.07)) < 0                   # COM past the edge: topples
    assert ir.status(spec, mk(0.0, u=0.0)) == "true"
    assert ir.status(spec, mk(0.04, u=0.02)) == "unknown"                     # 1 cm margin, 2 cm position uncertainty
    assert ir.status(spec, mk(0.07)) == "false"
    # rows exist for the stack too (decision variable = a 2-D offset of the top box; Jacobian = identity on x, y)
    J = np.zeros((3, 2)); J[0, 0] = J[1, 1] = 1.0
    A, b, _ = ir.rows(ir.Support(top, bottom), mk(0.0), {"u0": np.zeros(2), "top": {"p": J}})
    assert np.all(A @ np.array([0.02, 0.0]) <= b) and not np.all(A @ np.array([0.06, 0.0]) <= b)
