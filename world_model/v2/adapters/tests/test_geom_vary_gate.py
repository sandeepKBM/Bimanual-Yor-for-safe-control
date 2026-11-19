"""Owner gate (11-19): varied scene geometry is the default world list of the behind tasks (nero/base/geom_prior.py), and every geometry reader
asks one source (the base backend, adapters/base_sweep.py).  SIMULATION ONLY, CPU, no network.
PYTHONPATH=. python -m pytest -q world_model/v2/adapters/tests/test_geom_vary_gate.py

Checks, with truth read here from the compiled model (not from the backend):
  1. nominal=True gives the dataclass defaults (no draw applied); varied worlds differ from each other in geometry; the contents draw is the same
     with and without the variation;
  2. the hidden contents lie inside the DRAWN crate's interior (crate frame, model geoms), worlds 0-49;
  3. the backend goal: the crate centre lands inside the l1 floor board (oriented, model geom), the crate footprint at the goal yaw overlaps no
     side / back board, and the crate top at the place height stays below the l1 roof (bookcase_mid underside), worlds 0-9;
  4. the default route (closed_chain.default_waypoint): <= 4 segments; composed HERE (line = along the base yaw, turn = in place) from the start
     pose it ends at the backend's required end pose, and the welded crate ends at pre_xy, with pre_xy + insert * heading = goal_xy; a world may
     instead refuse with a reason (fail closed), worlds 0-9 + nominal 2;
  5. nominal world 2: the route equals the TaskDef route and the goal equals obj_xy + goal_dxy (today's numbers);
  6. the plan-cache key changes with the geometry (varied 0 vs 1) and is stable for one world."""
import math

import mujoco
import numpy as np
import pytest

from world_model.v2.nero.base import geom_prior as GPR
from world_model.v2.nero.base import scene as BS
from world_model.v2.nero.base import smoke as SM
from world_model.v2.stack import tasks as TK
from world_model.v2.stack.planners import base_route as BR
from world_model.v2.stack.planners import closed_chain as CC

TASK = "box_bookcase"
TOL = 1e-6


def _world(i, nominal=False):
    return SM.behind_world(TASK, i, nominal=nominal)[1]


def _model(w):
    m = BS.build_spec(w).compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def _rect(m, d, name):
    g = m.geom(name).id
    R = d.geom_xmat[g].reshape(3, 3)
    return d.geom_xpos[g].copy(), R, m.geom_size[g].copy()


def _corners(c, yaw, hx, hy):
    u, v = np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])
    return np.array([c + sx * hx * u + sy * hy * v for sx in (-1, 1) for sy in (-1, 1)])


def _overlap(P, Q):
    """2D separating-axis test of two convex polygons (corner arrays)."""
    for poly in (P, Q):
        n = len(poly)
        order = np.argsort(np.arctan2(*(poly - poly.mean(0)).T[::-1]))
        poly = poly[order]
        for k in range(n):
            e = poly[(k + 1) % n] - poly[k]
            ax = np.array([-e[1], e[0]])
            a, b = P @ ax, Q @ ax
            if a.max() <= b.min() or b.max() <= a.min():
                return False
    return True


def test_nominal_and_draw_separation():
    w0, wn = _world(0), _world(0, nominal=True)
    from world_model.v2.nero.contents import scenes as SC
    dflt = SC.BoxWorld()
    for f in ("box_outer", "obj_yaw", "bc_dxy", "bc_yaw", "bc_l1_h"):
        assert getattr(wn, f) == getattr(dflt, f), f
    assert w0.box_outer != wn.box_outer or w0.bc_dxy != wn.bc_dxy
    assert _world(1).box_outer != w0.box_outer
    for i in range(5):                                                  # the contents draw does not depend on the geometry draw (mass, and the draw before clipping)
        a, b = _world(i), _world(i, nominal=True)
        assert a.contents_kg == b.contents_kg, i
    assert GPR.draw(TASK, 3) == GPR.draw(TASK, 3)


@pytest.mark.parametrize("i", range(50))
def test_contents_inside_drawn_crate(i):
    w = _world(i)
    m, d = _model(w)
    b = m.body("obj").id
    R, p = d.xmat[b].reshape(3, 3), d.xpos[b]
    inner = np.array(w.box_outer[:2]) / 2 - w.wall_t
    g = [g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name == "contents"]
    assert g
    for k in g:
        c = R.T @ (d.geom_xpos[k] - p)
        e = np.abs(R.T @ d.geom_xmat[k].reshape(3, 3)) @ m.geom_size[k]
        assert np.all(np.abs(c[:2]) + e[:2] <= inner + 1e-9), (i, c[:2], e[:2], inner)


@pytest.mark.parametrize("i", range(10))
def test_goal_fits_compartment(i):
    w = _world(i)
    m, d = _model(w)
    be = BR.default_backend()
    g = be.goal(w)
    gxy = np.array(g["goal_xy"])
    c, R, s = _rect(m, d, "bookcase_l1_floor")
    q = R.T @ (np.r_[gxy, c[2]] - c)
    assert abs(q[0]) <= s[0] and abs(q[1]) <= s[1], (i, q, s)          # crate centre over the l1 floor board
    yaw_goal = float(w.obj_yaw) + (g["heading"] - be.start_pose(w)[2])   # welded crate turns with the base
    hx, hy, hz = (v / 2 for v in w.box_outer)
    crate = _corners(gxy, yaw_goal, hx, hy)
    for nm in ("bookcase_side_left", "bookcase_side_right", "bookcase_back"):
        cb, Rb, sb = _rect(m, d, nm)
        board = _corners(cb[:2], math.atan2(Rb[1, 0], Rb[0, 0]), sb[0], sb[1])
        assert not _overlap(crate, board), (i, nm)
    floor_top = c[2] + s[2]
    cm, Rm, sm = _rect(m, d, "bookcase_mid")
    assert floor_top + 2 * hz < cm[2] - sm[2], (i, floor_top + 2 * hz, cm[2] - sm[2])


def _compose(p0, route):
    x, y, yaw = p0
    for k, v in route:
        if k == "line":
            x, y = x + v * math.cos(yaw), y + v * math.sin(yaw)
        elif k == "turn":
            yaw += math.radians(v)
        else:
            raise AssertionError(f"unexpected segment {k}")
    return x, y, yaw


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


@pytest.mark.parametrize("i,nominal", [(i, False) for i in range(10)] + [(2, True)])
def test_default_route_reaches_goal(i, nominal):
    w = _world(i, nominal)
    be = BR.default_backend()
    try:
        wp = CC.default_waypoint(TASK, w, be)
    except CC.PlanRefused as e:
        assert str(e).strip(), i                                         # a refusal must carry its reason
        pytest.skip(f"world {i} refused (fail closed): {e}")
    (_, route, _), = wp.base
    route = BR.normalize_route(route)
    assert len(route) <= BR.MAX_SEGMENTS
    p0 = be.start_pose(w)
    pe = _compose(p0, route)
    want = be.required_end_pose(w)
    assert abs(pe[0] - want[0]) < 1e-6 and abs(pe[1] - want[1]) < 1e-6 and abs(_wrap(pe[2] - want[2])) < 1e-6, (i, pe, want)
    dl = pe[2] - p0[2]
    Rz = np.array([[math.cos(dl), -math.sin(dl)], [math.sin(dl), math.cos(dl)]])
    obj_end = np.array(pe[:2]) + Rz @ (np.array(w.obj_xy[:2]) - np.array(p0[:2]))
    g = be.goal(w)
    assert np.allclose(obj_end, g["pre_xy"], atol=1e-6), (i, obj_end, g["pre_xy"])
    h = g["heading"]
    assert np.allclose(np.array(g["pre_xy"]) + g["insert_m"] * np.array([math.cos(h), math.sin(h)]), g["goal_xy"], atol=1e-9)
    ins = wp.deltas[wp.names.index("s1_insert")][0]
    assert ins == pytest.approx(g["insert_m"])


def test_nominal_numbers_unchanged():
    w = _world(2, nominal=True)
    td = TK.task(TASK)
    wp = CC.default_waypoint(TASK, w, BR.default_backend())
    (_, route, _), = wp.base
    assert BR.normalize_route(route) == BR.normalize_route(td.routes[0][1])
    g = BR.default_backend().goal(w)
    assert np.allclose(g["goal_xy"], np.array(w.obj_xy[:2]) + np.array(td.goal_dxy), atol=1e-9), (g["goal_xy"], td.goal_dxy)


def test_plan_key_follows_geometry():
    k = [CC.inputs_for_world(TASK, _world(i)).key() for i in (2, 3)]   # two worlds that plan (owner scan 11-19: of the grasp-reachable worlds, 2 3 9 19 25 35 49 plan; others refuse, fail closed)
    assert k[0] != k[1]
    assert CC.inputs_for_world(TASK, _world(2)).key() == k[0]


# ---- 11-19 grasp follows the drawn crate (USER 11-19: an unreachable grasp is a refusal) ----

@pytest.mark.parametrize("i", range(10))
def test_tabs_follow_the_drawn_crate(i):
    """each tab's inner end touches the crate's end wall (obj frame, model geoms); the grasp site sits at box_y/2 + 0.036 (nominal offsets kept, MODEL)."""
    w = _world(i)
    m, d = _model(w)
    hy = w.box_outer[1] / 2
    for nm, sgn in (("tab_left", 1), ("tab_right", -1)):
        g = m.geom(nm).id
        yc, yh = float(m.geom_pos[g][1]), float(m.geom_size[g][1])
        assert abs(sgn * yc - yh - hy) < 1e-9, (i, nm, sgn * yc - yh, hy)          # inner end at the wall
        assert abs(sgn * yc + yh - (hy + 0.06)) < 1e-9, (i, nm)
        s = m.site(f"obj_grasp_{nm.split('_')[1]}").id
        assert abs(sgn * float(m.site_pos[s][1]) - (hy + 0.036)) < 1e-9, (i, nm)


def test_unreachable_grasp_is_a_plan_refusal():
    """world 13 (crate yaw 5.9 deg): no hold pose -> default_waypoint raises PlanRefused with reason grasp_unreachable (fail closed, never runs)."""
    with pytest.raises(CC.PlanRefused) as e:
        CC.default_waypoint(TASK, _world(13), BR.default_backend())
    assert (getattr(e.value, "info", None) or {}).get("reason") == "grasp_unreachable", e.value


def test_refused_world_geometry_queries_are_undisturbed():
    """base_sweep builds its geometry episode with the captured arm pose when the grasp is unreachable: the base start pose must equal a holding
    world's (the base start does not depend on the crate) and the bookcase boards must match the compiled scene."""
    be = BR.default_backend()
    p_ref, p13 = be.start_pose(_world(1)), be.start_pose(_world(13))
    assert np.allclose(p_ref, p13, atol=1e-9), (p_ref, p13)
    w = _world(13)
    m, d = _model(w)
    c, _, s = _rect(m, d, "bookcase_l1_floor")
    ob = __import__("world_model.v2.adapters.base_sweep", fromlist=["obstacles"]).obstacles(w)["bookcase_l1_floor"]
    assert abs(np.mean(ob["x"]) - c[0]) < 1e-4 and abs(np.mean(ob["y"]) - c[1]) < 1e-4, (ob, c)        # obstacles() rounds to 0.1 mm
