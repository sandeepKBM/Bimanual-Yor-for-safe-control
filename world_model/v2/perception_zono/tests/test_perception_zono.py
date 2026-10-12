"""Unit tests. Run:  PYTHONPATH=. python -m pytest -q world_model/v2/perception_zono/tests   (about 30 s). SIMULATION ONLY."""
import numpy as np
import pytest

from world_model.v2.perception_zono import geometry as G, sensing as S, segment as SG, fit as F, zonoutil as ZU
from world_model.v2.perception_zono.score import Truth, score_fit
from world_model.v2.tray_rep.sets.zono import Zonotope

RNG = np.random.default_rng(0)


def test_sdf_matches_primitive_surface():
    for k in ["ball", "box", "bottle", "mug", "L", "cup", "dish"]:
        o = G.proto(k).moved(0.02, 0.01, 0.7)
        s = o.surface(2000, RNG)
        assert np.abs(o.sdf(s)).max() < 1e-4, k                   # boundary points have sdf ~ 0 (union boundary)
        assert (o.sdf(s) > -1e-4).all()


def test_scene_geom_labels_and_raycast_consistent_with_truth():
    o = G.proto("box").moved(0.03, 0.0, 0.4)
    sc = G.Scene([o])
    cl = S.capture(sc, "head_aim+wrist", "ideal", 0)
    P = cl.pts[cl.label == 1]
    assert len(P) > 500
    assert np.abs(o.sdf(P)).max() < 1e-3                          # ideal sensor: hits lie on the true surface (1 mm tol)


def test_halfspaces_and_volume_exact():
    Z = Zonotope(RNG.normal(size=3), RNG.normal(size=(3, 6)))
    A, b = ZU.halfspaces(Z)
    assert ((Z.sample(2000, RNG) @ A.T - b) <= 1e-7).all()
    lo, hi = Z.interval_hull()
    assert abs(ZU.volume(Z) - ZU.ZUnion([Z]).volume_mc(lo, hi, RNG, 400000)) < 0.02 * ZU.volume(Z)


def test_hull_zono_contains_hull_all_orders():
    o = G.proto("bottle").moved(0.0, 0.0, 0.5)
    P = o.surface(3000, RNG)
    for order in (2, 3, 4, None):
        Z = F.fit_hull_zono(P, order)
        assert ZU.ZUnion([Z]).contains(P).all(), order
    assert F.fit_hull_zono(P, 4).n_gen <= 12 and F.fit_hull_zono(P, 2).n_gen <= 6


def test_box_contains_points():
    P = G.proto("L").moved(0, 0, 1.0).surface(2000, RNG)
    for fr in ("pca", "up"):
        assert ZU.ZUnion([F.fit_box(P, fr)]).contains(P).all()


def test_pa_detects_jump():
    f = np.concatenate([np.ones(10), 0.3 * np.ones(22)])
    L = F.pa_jump(f, 2)
    assert np.argmax(np.abs(L)) == 10 and abs(L[10]) > 0.5
    assert np.abs(np.delete(L, 10)).max() < 1e-9


def test_split_L_full_surface_gives_two_parts_and_stays_contained():
    P = G.proto("L").moved(0.0, 0.0, 0.3).surface(4000, RNG)
    zs, info = F.fit_split(P, "pa")
    U = ZU.ZUnion(zs)
    assert len(zs) == 2 and U.contains(P).mean() > 0.995
    assert U.volume_sum() < 0.6 * ZU.volume(F.fit_hull_zono(P, 2))


def test_sd_points_exact_vs_optimizer():
    from scipy.optimize import minimize
    Z = Zonotope(np.zeros(3), RNG.normal(size=(3, 5)))
    q = np.array([[5.0, 1.0, -2.0]])
    r = minimize(lambda s: np.sum((Z.c + Z.G @ s - q[0]) ** 2), np.zeros(5), bounds=[(-1, 1)] * 5, method="L-BFGS-B")
    assert abs(ZU.ZUnion([Z]).sd_points(q)[0] - np.sqrt(r.fun)) < 1e-5


def test_dbscan_two_blobs_and_outlier():
    X = np.concatenate([RNG.normal(0, 0.002, (200, 3)), RNG.normal(0, 0.002, (200, 3)) + [0.1, 0, 0], [[1.0, 1.0, 1.0]]])
    lab = SG.dbscan(X, 0.009, 6)
    assert lab[:200].min() == lab[:200].max() >= 0 and lab[200:400].min() == lab[200:400].max() >= 0
    assert lab[0] != lab[300] and lab[-1] == -1


def test_score_fit_perfect_box_ideal():
    o = G.proto("box").moved(0.0, 0.0, 0.0)
    T = Truth(o, RNG)
    Zt = Zonotope(o.prims[0].pos, o.prims[0].R * np.array(o.prims[0].size)[None, :])
    m = score_fit(ZU.ZUnion([Zt.scale(1.0)]).inflate(1e-4), T, RNG)
    assert m["contain"] > 0.999 and 0.95 < m["oc_true"] < 1.1 and abs(m["clear_mean_mm"]) < 1.5
