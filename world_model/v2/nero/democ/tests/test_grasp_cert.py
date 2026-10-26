"""grasp_cert tests (math only, CPU).   PYTHONPATH=. python -m pytest -q world_model/v2/nero/democ/tests/test_grasp_cert.py"""
import numpy as np
import pytest

from world_model.v2.nero.democ import grasp_cert as GC
from world_model.v2.nero.democ.grasp_cert import Contact, Params, certify_hold

MG = 0.6 * GC.G


def flat_pad(a=0.02, b=0.015, F=10.0, mu=0.5, z=0.0):
    """pad in the xy plane at height z, normal +z (pushes the box up), half sizes a (x), b (y), one group limited to F."""
    return [Contact([sx * a, sy * b, z], [0, 0, 1], mu, mu, f_max=F, group="pad", hand="left", active=True)
            for sx in (-1, 1) for sy in (-1, 1)]


def P(m=0.6, r=(0, 0, 0), **kw):
    return Params(m_lo=m, m_hi=m, com_lo=r, com_hi=r, **kw)


def test_single_pad_holds_weight_iff_force_limit_allows():
    for F, expect in ((5.5, False), (MG - 1e-3, False), (MG + 1e-3, True), (10.0, True)):
        ok, eps, det = certify_hold(flat_pad(F=F), P())
        assert ok == expect, (F, eps)
    ok, eps, _ = certify_hold(flat_pad(F=10.0), P())
    # eps* is the min over 12 directions: the binding one is the twist torque about z from friction at the corner contacts,
    # mu * m g * corner_radius / L  (pyramid edge discretisation makes it a hair smaller)
    assert 0.9 * 0.5 * MG * np.hypot(0.02, 0.015) / GC.L_SCALE < eps <= 0.5 * MG * np.hypot(0.02, 0.015) / GC.L_SCALE + 1e-9


def test_single_pad_torque_limit_matches_closed_form():
    a = 0.02
    for ax in (0, 1):                                             # COM offset along x (half-length a) or y (half-width b)
        half = (a, 0.015)[ax]
        for frac, expect in ((0.98, True), (0.999, True), (1.01, False), (1.5, False)):
            r = [0, 0, 0]; r[ax] = frac * half
            ok, eps, _ = certify_hold(flat_pad(a=a, b=0.015, F=10.0), P(r=tuple(r)))
            assert ok == expect, (ax, frac, eps)
            r[ax] = -frac * half
            assert certify_hold(flat_pad(a=a, b=0.015, F=10.0), P(r=tuple(r)))[0] == expect


def test_friction_hand_only_vertical_pad():
    # pad on a vertical face (normal +x): weight carried by friction only: holds iff mu*F >= m g (edge along -z is ON the cone)
    def pad(mu, F):          # two opposing pads (faces x=+-0.03), shared squeeze budget F (a lone one-sided pad can never balance)
        return [Contact([sx * 0.03, sy * 0.02, sz * 0.02], [-sx, 0, 0], mu, mu, f_max=F, group="p") for sx in (-1, 1)
                for sy in (-1, 1) for sz in (-1, 1)]
    assert not certify_hold(pad(0.5, 20.0)[:4], P())[0]            # one-sided pad: net normal force cannot cancel
    assert certify_hold(pad(0.5, 20.0), P())[0]                   # 10 N friction > 5.89
    assert not certify_hold(pad(0.5, 11.0), P())[0]               # 5.5 N < 5.89
    assert certify_hold(pad(0.5, 11.9), P())[0]                   # 5.95 N
    ok, eps, _ = certify_hold(pad(0.5, 20.0), P(), edge_scale="spec")
    assert ok and eps > 0
    assert not certify_hold(pad(0.5, 12.0), P(), edge_scale="spec")[0]   # cos(pi/8) loss: 5.54 < 5.89



def _two_pad_hand(mu, F=10.0, jaw_z=True):
    # jaw axis z: pads on the top (normal -z) and bottom (normal +z) faces, extending along y in [0.09, 0.14], x in +-0.014
    cs = []
    for side, nz in ((+1, -1), (-1, +1)):
        for x in (-0.014, 0.014):
            for y in (0.09, 0.128):
                cs.append(Contact([x, y, side * 0.03], [0, 0, nz], mu, mu, f_max=F, group="h", hand="left", active=True))
    return cs


def test_monotone_in_mu():
    # friction-critical set: vertical-face pad with a large COM offset so friction + torque both matter
    def pad(mu):
        return [Contact([sx * 0.03, sy * 0.02, sz * 0.02], [-sx, 0, 0], mu, mu, f_max=20.0, group="p") for sx in (-1, 1)
                for sy in (-1, 1) for sz in (-1, 1)]
    prev = -np.inf
    for mu in (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2):
        ok, eps, _ = certify_hold(pad(mu), P(r=(0, 0.01, 0.0)))
        assert eps >= prev - 1e-9
        assert not (ok is False and prev >= 0)
        prev = eps
    # same for the two-pad vertical-jaw hand with an offset COM
    prev = -np.inf
    for mu in (0.05, 0.2, 0.4, 0.8):
        eps = certify_hold(_two_pad_hand(mu), P(r=(0, 0.05, 0)))[1]
        assert eps >= prev - 1e-9
        prev = eps


def test_vertex_sufficiency_vs_random_samples():
    rng = np.random.default_rng(0)
    # a set that passes at the vertices but is near the limit, and one that fails
    cons = flat_pad(a=0.03, b=0.02, F=10.0)
    for r_hi, expect in ((0.02, True), (0.04, False)):
        Pm = Params(m_lo=0.5, m_hi=0.7, com_lo=(-r_hi, -0.01, 0.0), com_hi=(r_hi, 0.01, 0.0))
        ok, eps, det = certify_hold(cons, Pm)
        assert ok == expect, eps
        nbad = 0
        for _ in range(2000):
            m = rng.uniform(0.5, 0.7); r = rng.uniform(Pm.com_lo, Pm.com_hi)
            if not certify_hold(cons, P(m=m, r=tuple(r)), compute_margin=False)[0]:
                nbad += 1
        if ok:
            assert nbad == 0                                      # all vertices pass => no interior violation
        else:
            assert nbad > 0


def test_exact_mr_polytope_not_outer_box():
    # m in [0.5,0.7], |r_x| <= 0.025 on a +-0.03 pad: every real (m, r) is balanceable; the owner's outer (m, m*r) box would reject it
    cons = flat_pad(a=0.03, b=0.02, F=10.0)
    Pm = Params(m_lo=0.5, m_hi=0.7, com_lo=(-0.025, 0, 0), com_hi=(0.025, 0, 0))
    assert certify_hold(cons, Pm)[0]
    assert not certify_hold(cons, P(m=0.5, r=(0.035, 0, 0)))[0]      # the outer-box corner (m_lo, s_hi) = r 0.035 is NOT a real value


def test_vertex_with_uncertain_acceleration():
    cons = flat_pad(a=0.03, b=0.02, F=10.0)
    Pm = Params(m_lo=0.5, m_hi=0.7, com_lo=(-0.01, 0, 0), com_hi=(0.01, 0, 0), a_lo=(-0.5, -0.5, 0), a_hi=(0.5, 0.5, 2.0))
    ok, eps, det = certify_hold(cons, Pm)
    assert len(det["vertex_vars"]) >= 5
    rng = np.random.default_rng(1)
    if ok:
        for _ in range(100):
            m = rng.uniform(0.5, 0.7); r = rng.uniform(Pm.com_lo, Pm.com_hi); a = rng.uniform(Pm.a_lo, Pm.a_hi)
            g = np.array(Pm.g)
            w_f = m * (g - a)
            # contact force needed = -w_f; with a flat pad it is balanceable iff fz in [0,F] and horizontal friction ok -> brute check
            assert certify_hold(cons, P(m=m, r=tuple(r), a_lo=tuple(a), a_hi=tuple(a)), compute_margin=False)[0]


def test_inscribed_pyramid_inside_true_cone():
    rng = np.random.default_rng(2)
    for mu in (0.2, 0.5, 1.0):
        for scale in ("vertex", "spec"):
            c = Contact([0, 0, 0], [0.3, -0.5, 0.8], mu, mu)
            _, Fe = GC.contact_matrix([c], edge_scale=scale)
            lam = rng.dirichlet(np.ones(len(Fe)), 3000) * rng.uniform(0.1, 5, (3000, 1))
            f = lam @ Fe
            fn = f @ c.n; ft = np.linalg.norm(f - np.outer(fn, c.n), axis=1)
            assert np.all(ft <= mu * fn + 1e-12)                  # pyramid is INSIDE the cone
    # and not vacuous: cone forces exist that the pyramid cannot produce (so it is strictly conservative)
    c = Contact([0, 0, 0], [0, 0, 1], 0.5, 0.5)
    G6, _ = GC.contact_matrix([c])
    ang = np.pi / 8                                               # between two edges: cone boundary, pyramid chord is shorter
    f = np.array([0.5 * np.cos(ang), 0.5 * np.sin(ang), 1.0])
    ok = certify_hold([Contact([0, 0, 0], [0, 0, 1], 0.5, 0.5), Contact([0.01, 0, 0], [0, 0, 1], 0.5, 0.5)],
                      Params(m_lo=1.0, m_hi=1.0, g=(-f[0] * 1, -f[1] * 1, -f[2] * 1)))[0]
    assert ok is False


def test_two_hand_optimistic_flag():
    left = _two_pad_hand(0.5)
    right = [Contact(c.p * [1, -1, 1], c.n, 0.5, 0.5, f_max=10.0, group="hr", hand="right", active=True) for c in left]
    sup = GC.support_contacts([[0.09, 0.07, -0.03], [-0.09, 0.07, -0.03], [-0.09, -0.07, -0.03], [0.09, -0.07, -0.03]], [0, 0, 1], 0.5)
    assert certify_hold(left + right, P())[2]["optimistic_internal_force"] is True
    assert certify_hold(left, P())[2]["optimistic_internal_force"] is False
    assert certify_hold(left + sup, P())[2]["optimistic_internal_force"] is False
    assert certify_hold(sup, P())[0]                              # resting on the pedestal holds
    assert not certify_hold([], P())[0]


def test_pose_uncertainty_is_sound_and_monotone():
    # support patch + one hand; large shift must eventually fail, tiny shift must keep a passing set passing; sampled shifts of an
    # accepted set must stay balanceable
    rng = np.random.default_rng(3)
    cons = flat_pad(a=0.03, b=0.02, F=10.0)
    r = (0.02, 0, 0)
    base = certify_hold(cons, P(r=r))
    assert base[0]
    res = [certify_hold(cons, P(r=r, pose_delta=(d, d, d)))[0] for d in (0.0, 0.0005, 0.002, 0.005, 0.02)]
    assert res[0] and res[1] is True and res[-1] is False
    assert all(not res[i + 1] or res[i] for i in range(len(res) - 1))            # monotone
    for d in (0.0005, 0.002):
        if certify_hold(cons, P(r=r, pose_delta=(d, d, d)))[0]:
            for _ in range(20):
                sh = rng.uniform(-d, d, (len(cons), 3)); sh[:, 2] = 0
                moved = [Contact(c.p + s, c.n, c.mu_lo, c.mu_hi, c.f_max, c.group, c.hand, c.active) for c, s in zip(cons, sh)]
                assert certify_hold(moved, P(r=r), compute_margin=False)[0]


def test_support_patch_from_pedestal_geometry():
    h = np.array([0.1, 0.15, 0.03])
    cs = GC.support_patch_from_pedestal(np.eye(3), [0.45, -0.21, 0.85], h, [0.45, -0.21, 0.79], [0.09, 0.07, 0.03], 0.4)
    assert len(cs) == 4
    pts = np.array([c.p for c in cs])
    assert np.allclose(sorted(set(np.round(np.abs(pts[:, 0]), 6))), [0.09]) and np.allclose(sorted(set(np.round(np.abs(pts[:, 1]), 6))), [0.07])
    assert np.allclose(pts[:, 2], -0.03) and np.allclose(cs[0].n, [0, 0, 1])
    assert certify_hold(cs, P())[0]
