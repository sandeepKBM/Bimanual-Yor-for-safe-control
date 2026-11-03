"""cert/belief.py + cert/model_check.py: set update soundness, vertex exactness, widen-only, consistency events."""
import numpy as np
import pytest

from world_model.v2.nero.pan.cert import belief as B
from world_model.v2.nero.pan.cert import model_check as MC


def _sample_inside(c, n, rng):
    """rejection samples from the contents polytope (bounding box from its vertices)."""
    V = c.vertices()
    lo, hi = V.min(0), V.max(0)
    out = []
    while len(out) < n:
        x = rng.uniform(lo, hi)
        if c.contains(x):
            out.append(x)
    return np.array(out)


def test_prior_vertices():
    c = B.Contents.from_prior()
    V = c.vertices()
    assert len(V) == 1 + B.N_GON                      # apex at m = 0 plus the 8-gon at m_hi
    assert np.allclose(V[np.argmin(V[:, 0])], 0.0)
    top = V[V[:, 0] > 0.3]
    r = np.linalg.norm(top[:, 1:] / top[:, :1], axis=1)
    assert np.allclose(r, B.C_MAX / np.cos(np.pi / B.N_GON))   # circumscribed: corners outside the disc


def test_true_theta_stays_in_belief():
    """soundness of the update: with |noise| <= eps the true contents are never excluded (1000 random readings)."""
    rng = np.random.default_rng(0)
    prior = B.Belief()
    for trial in range(20):
        truth = _sample_inside(prior.contents, 1, rng)[0]
        b = prior
        for k in range(50):
            a = rng.normal(size=3)
            eps = 0.05 + rng.uniform(0, 0.2)
            y = a @ truth + rng.uniform(-eps, eps)
            b, ok = b.update(a, y, eps)
            assert ok, "a reading consistent with the truth was flagged inconsistent"
            assert b.contents.contains(truth), "the update excluded the true contents"


def test_interval_exact_at_vertices():
    """a linear function's range over the set equals its range over the vertices (and dense samples stay inside it)."""
    rng = np.random.default_rng(1)
    b, _ = B.Belief().update([[9.81 * 0.09, 9.81, 0.0]], [0.2], [0.05])
    for _ in range(10):
        f = rng.normal(size=3)
        lo, hi = b.contents.interval(f)
        S = _sample_inside(b.contents, 400, rng) @ f
        assert S.min() >= lo - 1e-9 and S.max() <= hi + 1e-9
        assert hi - lo > 0 and (S.max() - S.min()) > 0.5 * (hi - lo)


def test_widen_only_grows():
    rng = np.random.default_rng(2)
    b = B.Belief()
    pts = _sample_inside(b.contents, 200, rng)
    w = b.widen(m=(0.0, 0.8), c=0.07, mu=(0.5, 1.0), dxy=0.01, yaw=10.0)
    assert all(w.contents.contains(p) for p in pts)
    assert w.mu == (0.5, 1.2) and w.dxy == b.dxy and w.yaw == 10.0
    nb, _ = b.update([[1, 0, 0]], [0.2], [0.05])
    with pytest.raises(ValueError):
        nb.widen(m=(0, 1))


def test_inconsistent_rows_count_an_event():
    b = B.Belief()
    nb, ok = b.update([[1.0, 0, 0]], [2.0], [0.1])     # mass 2 kg: outside the 0-0.4 kg prior
    assert not ok and nb.events == 1 and nb.contents == b.contents


def test_contents_world_clips_corner_pucks():
    w = B.contents_world([0.4, 0.4 * 0.07, 0.4 * 0.0])
    assert w["clipped"] and abs(np.linalg.norm(w["placement"]) - B.C_MAX) < 1e-9
    assert B.contents_world([0.0, 0, 0])["placement"] == "none"


def test_support_polygon_and_exactness():
    """centred pan: contact area = the disc; pushed 7 cm along -y: area is cut by the stove edge.  The vertex margin is the
    minimum over dense contents samples (linear check, exact at the vertices)."""
    stove, half = (0.38, -0.30), (0.12, 0.12)
    P, planes = MC.support_polygon(stove, 0.0925, stove, half)
    assert len(P) == MC.DISC_N
    centre = (0.38, -0.37)
    P, planes = MC.support_polygon(centre, 0.0925, stove, half)
    assert P[:, 1].min() >= -0.30 - 0.12 - 1e-9
    b = B.Belief()
    r = MC.support_at_rest(b.contents_vertices(), 0.5, centre, centre, (0, 1), planes)
    rng = np.random.default_rng(3)
    u, w = np.array([0, 1.0]), np.array([-1.0, 0])
    worst = np.inf
    for m, su, sw in _sample_inside(b.contents, 400, rng):
        com = (0.5 * np.array(centre) + m * np.array(centre) + su * u + sw * w) / (0.5 + m)
        worst = min(worst, min(h - n @ com for n, h in planes))
    assert r["margin_m"] <= worst + 1e-6


def test_planned_accel_smoothstep():
    peak, per = MC.planned_accel([("move", {"left": np.array([0, 0, 0.05])}, 1.5), ("hold", 0.3)])
    assert abs(peak - 6 * 0.05 / 1.5 ** 2) < 1e-12 and len(per) == 1
