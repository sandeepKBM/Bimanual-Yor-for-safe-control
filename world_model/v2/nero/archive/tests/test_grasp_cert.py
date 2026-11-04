"""grasp_cert tests (math only, CPU).   PYTHONPATH=. python -m pytest -q world_model/v2/nero/democ/tests/test_grasp_cert.py"""
import numpy as np
import pytest

from functools import partial

from world_model.v2.nero.archive.democ import grasp_cert as GC
from world_model.v2.nero.archive.democ.grasp_cert import Contact, Params

certify_hold = partial(GC.certify_hold, mode="exists")      # the original existence-only tests below run in mode='exists'

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


# =============================================================================== compliant (linear-elastic) mode
def hand_pair(Fg=9.0, mu=0.5, y=(0.11, 0.15), x=(-0.014, 0.014), hand="left"):
    """one hand: jaw along box z (pads on the z = +-0.03 faces), 4 corners per pad, shared squeeze budget Fg"""
    return [Contact([xx, yy, sz * 0.03], [0, 0, -sz], mu, mu, f_max=Fg, group="h" + hand, hand=hand, active=True)
            for sz in (1, -1) for xx in x for yy in y]


def ped(mu=0.5, hx=0.09, hy=0.07):
    return GC.support_contacts([[hx, hy, -0.03], [hx, -hy, -0.03], [-hx, -hy, -0.03], [-hx, hy, -0.03]], [0, 0, 1], mu)


def test_exists_default_changed_and_modes_dispatch():
    cs = hand_pair() + ped()
    assert GC.certify_hold(cs, P())[2]["mode"] == "compliant"
    assert "mode" not in GC.certify_hold(cs, P(), mode="exists")[2]
    with pytest.raises(ValueError):
        GC.certify_hold(cs, P(), mode="nope")


@pytest.mark.parametrize("tm", ["spring", "creep"])
def test_equilibrium_and_jaw_balance(tm):
    cs = hand_pair(Fg=9.0) + ped()
    M = GC.CompliantModel(cs, GC.Stiffness(rho=0.7, kappa=0.3), tangential=tm)
    assert not M.singular
    w = np.array([0.3, -0.2, -MG, 0.04, 0.02, -0.01])
    S = M.solve(w[None], np.array([[9.0]]))
    fn, ft = S["fn"][0], S["ft"][0]
    wr = np.zeros(6)
    for i, c in enumerate(cs):
        f = fn[i] * M.n[i] + ft[i, 0] * M.t1[i] + ft[i, 1] * M.t2[i]
        wr += np.concatenate([f, np.cross(c.p, f)])
    assert np.allclose(wr, -w, atol=1e-8)                              # box equilibrium
    assert abs(fn[:8].sum() - 9.0) < 1e-8                              # jaw force balance: sum of the two pad normals = F_g


def test_affine_in_parameters_vertices_are_exact():
    """forces at interior parameter points == the convex combination of the vertex solutions (same weights as the parameters)"""
    from scipy.optimize import linprog
    rng = np.random.default_rng(5)
    cs = hand_pair(Fg=9.0) + ped()
    Pm = Params(m_lo=0.5, m_hi=0.7, com_lo=(-0.02, -0.01, 0.0), com_hi=(0.02, 0.03, 0.0))
    w0, base, cols, los, his, names = GC.wrench_vertices(Pm)
    assert cols.shape[1] == 0
    Wv = np.array([wb for _, wb in base]); Fv_lo, Fv_hi = 7.0, 10.0
    for tm in ("spring", "creep"):
        M = GC.CompliantModel(cs, GC.Stiffness(rho=1.3, kappa=0.4), tangential=tm)
        Sv = {F: M.solve(Wv, np.full((len(Wv), 1), F)) for F in (Fv_lo, Fv_hi)}
        for _ in range(20):
            m = rng.uniform(0.5, 0.7); r = rng.uniform(Pm.com_lo, Pm.com_hi); F = rng.uniform(Fv_lo, Fv_hi)
            g = np.array(Pm.g); w = np.concatenate([m * g, np.cross(m * r, g)])
            # convex weights over (vertex wrench) x (F endpoint): lam >= 0, sum = 1, sum lam w_v = w, sum lam F_v = F
            pts = [(Wv[i], f) for i in range(len(Wv)) for f in (Fv_lo, Fv_hi)]
            Aeq = np.vstack([np.array([p[0] for p in pts]).T, [p[1] for p in pts], np.ones(len(pts))])
            res = linprog(np.zeros(len(pts)), A_eq=Aeq, b_eq=np.concatenate([w, [F, 1.0]]), bounds=(0, None), method="highs")
            assert res.status == 0
            lam = res.x
            fn_cb = sum(lam[k] * Sv[pts[k][1]]["fn"][k // 2] for k in range(len(pts)))
            ft_cb = sum(lam[k] * Sv[pts[k][1]]["ft"][k // 2] for k in range(len(pts)))
            S = M.solve(w[None], np.array([[F]]))
            assert np.allclose(S["fn"][0], fn_cb, atol=1e-8) and np.allclose(S["ft"][0], ft_cb, atol=1e-8)


def test_single_patch_liftoff_matches_closed_form():
    """Rectangular support patch (corners +-a, +-b), vertical load P at offset (rx, ry), 4 normal springs: pressure is bilinear,
    f_i = P/4 (1 + s_x rx/a + s_y ry/b)  -> a corner unloads when |M_y|/a + |M_x|/b > P (M = P r): the KERN is the DIAMOND |rx|/a + |ry|/b <= 1,
    not the rectangle that the existence LP allows.  With margin n_min the bound is |rx|/a + |ry|/b <= 1 - 4 n_min / P."""
    a, b, Pz = 0.09, 0.07, MG
    patch = lambda: GC.support_contacts([[a, b, 0], [a, -b, 0], [-a, -b, 0], [-a, b, 0]], [0, 0, 1], 0.8)
    for tm in ("spring", "creep"):
        M = GC.CompliantModel(patch(), GC.Stiffness(), tangential=tm)
        for rx, ry in ((0.02, 0.0), (0.0, 0.03), (0.04, 0.02), (-0.05, 0.01)):
            w = np.concatenate([[0, 0, -Pz], np.cross([rx, ry, 0], [0, 0, -Pz])])
            fn = M.solve(w[None], np.zeros((1, 0)))["fn"][0]
            cf = [Pz / 4 * (1 + sx * rx / a + sy * ry / b) for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
            assert np.allclose(fn, cf, atol=1e-9)
    def ok(rx, ry, n_min=0.0, mode="compliant"):
        w = P(r=(rx, ry, 0)); return GC.certify_hold(patch(), w, n_min=n_min, gap_tol=0.0)[0]
    assert ok(0.99 * a, 0.0) and not ok(1.01 * a, 0.0)                  # 1-D load: edge of the patch
    assert ok(0.5 * a, 0.45 * b) and not ok(0.5 * a, 0.55 * b)           # diamond, |rx|/a + |ry|/b = 0.95 vs 1.05
    assert not ok(0.8 * a, 0.5 * b)                                      # inside the rectangle (existence LP says yes) but outside the diamond
    assert certify_hold(patch(), P(r=(0.8 * a, 0.5 * b, 0)))[0]          # mode='exists' admits it
    n = 0.5; lim = 1 - 4 * n / Pz                                        # n_min shrinks the diamond
    assert ok(0.99 * lim * a, 0.0, n_min=n) and not ok(1.01 * lim * a, 0.0, n_min=n)


def test_monotone_in_mu_for_fixed_stiffness():
    cs = lambda mu: hand_pair(Fg=9.0, mu=mu) + ped(mu=mu)
    prev = -np.inf
    for mu in (0.05, 0.1, 0.2, 0.4, 0.8, 1.5):
        ok, eps, _ = GC.certify_hold(cs(mu), P(r=(0, 0.05, 0)), gap_tol=0.0)
        assert eps >= prev - 1e-9
        assert not (prev >= 0 and not ok)
        prev = eps


def test_gap_sensitivity_matches_finite_difference_and_robust_margin_is_sound():
    cs = hand_pair(Fg=9.0) + ped()
    rng = np.random.default_rng(7)
    for tm in ("spring", "creep"):
        M = GC.CompliantModel(cs, GC.Stiffness(rho=0.8, kappa=0.5), tangential=tm)
        G = M.gap_sensitivity()
        w = np.array([0, 0, -MG, 0.02, 0.0, 0.0])[None]; F = np.array([[9.0]])
        S0 = M.solve(w, F)
        for _ in range(5):
            g = rng.uniform(-1e-5, 1e-5, M.nc)
            S = M.solve(w, F, gap=g)
            assert np.allclose(S["fn"][0] - S0["fn"][0], G["dfn"] @ g, atol=1e-8)
            assert np.allclose(S["ft"][0] - S0["ft"][0], np.einsum("ikj,j->ik", G["dft"], g), atol=1e-8)
            assert np.allclose(S["u"][0] - S0["u"][0], G["du"] @ g, atol=1e-12)
            wr = np.zeros(6)                                                     # still an equilibrium
            for i, c in enumerate(cs):
                f = S["fn"][0][i] * M.n[i] + S["ft"][0][i, 0] * M.t1[i] + S["ft"][0][i, 1] * M.t2[i]
                wr += np.concatenate([f, np.cross(c.p, f)])
            assert np.allclose(wr, -w[0], atol=1e-8)
        # robust bound: min fn over random gaps never below the certificate's worst-case estimate
        gt = 1e-5
        lo = S0["fn"][0] - gt * np.abs(G["dfn"]).sum(1)
        for _ in range(200):
            g = rng.uniform(-gt, gt, M.nc)
            assert np.all(M.solve(w, F, gap=g)["fn"][0] >= lo - 1e-9)
    # gap_tol only ever makes the certificate more conservative
    e0 = GC.certify_hold(cs, P(r=(0, 0.05, 0)), gap_tol=0.0)[1]
    e1 = GC.certify_hold(cs, P(r=(0, 0.05, 0)), gap_tol=2e-5)[1]
    assert e1 <= e0 + 1e-12


def test_arm_stiffness_series_compliance():
    cs = hand_pair(Fg=9.0)[:0] or (hand_pair(Fg=9.0) + ped())
    st = GC.Stiffness()
    w = np.array([0, 0, -MG, 0.0, 0.0, 0.0])[None]; F = np.array([[9.0]])
    S0 = GC.CompliantModel(cs, st).solve(w, F)
    Kbig = np.eye(6) * 1e12
    Mb = GC.CompliantModel(cs, st, arm={"left": (Kbig, np.zeros(6))})
    Sb = Mb.solve(w, F)
    assert np.allclose(Sb["fn"], S0["fn"], atol=1e-4)                 # rigid arm == fixed base
    Ks = np.diag([800.0] * 3 + [60.0] * 3)
    M2 = GC.CompliantModel(cs, st, arm={"left": (Ks, np.zeros(6))})
    S2 = M2.solve(w, F)
    b = S2["x"][0, M2.ib["left"]]
    hand_w = sum(M2.Wc[i] for i in range(M2.nc) if M2.hidx[i] >= 0) @ S2["x"][0]      # wrench the box receives from the hand
    assert np.allclose(Ks @ b, -hand_w, atol=1e-8)                    # arm spring balances the reaction (series compliance)
    assert np.linalg.norm(b) > 1e-9 and not np.allclose(S2["fn"], S0["fn"], atol=1e-3)   # a soft arm moves the base AND re-splits the load (hand + support is hyperstatic)


def test_regression_scenario_A_false_safes_are_rejected():
    """the 3 releases (hand alone, left kept, COM +10 cm, mu 0.2/0.4/0.8) where mode='exists' said SAFE and the box fell 92 mm / 25 deg in MuJoCo"""
    pts = [[-0.0132, 0.1103, 0.03], [-0.0132, 0.15, 0.03], [0.0148, 0.1103, 0.03], [0.0148, 0.15, 0.03],
           [-0.0132, 0.1103, -0.03], [-0.0132, 0.15, -0.03], [0.0148, 0.1103, -0.03], [0.0148, 0.15, -0.03]]
    for mu in (0.2, 0.4, 0.8):
        cs = [Contact(p, [0, 0, -np.sign(p[2])], mu, mu, f_max=10.0, group="hl", hand="left", active=True) for p in pts]
        Pm = P(r=(0, 0.1, 0), g=(0.0, 0.0, -GC.G))
        assert certify_hold(cs, Pm)[0]                                # exists mode: admitted (the bug)
        for gt in (0.0, 1e-5):
            ok, eps, det = GC.certify_hold(cs, Pm, gap_tol=gt)
            assert not ok and eps < 0, (mu, gt)


def test_differential_jaw_mode_limits():
    def fingers(Fg=9.0):
        cs = hand_pair(Fg=Fg)
        for c in cs:
            c.finger = 1 if c.p[2] > 0 else 2
        return cs + ped()
    cs = fingers()
    w = np.array([0, 0, -MG, 0.0, 0.0, 0.0])[None]; F = np.array([[9.0]])
    base = GC.CompliantModel(cs, GC.Stiffness(), tangential="creep").solve(w, F)["fn"]
    rigid = GC.CompliantModel(cs, GC.Stiffness(kd=1e12), tangential="creep").solve(w, F)["fn"]
    soft = GC.CompliantModel(cs, GC.Stiffness(kd=2e3), tangential="creep")
    S = soft.solve(w, F)
    assert np.allclose(base, rigid, atol=1e-4)                        # kd -> inf == rigid coupling
    assert not np.allclose(base, S["fn"], atol=1e-2)                  # a finite kd re-splits top/bottom
    assert abs(S["fn"][0, :8].sum() - 9.0) < 1e-8                     # jaw force balance still exact
    G = soft.gap_sensitivity(); g = np.random.default_rng(1).uniform(-1e-5, 1e-5, soft.nc)
    assert np.allclose(soft.solve(w, F, gap=g)["fn"][0] - S["fn"][0], G["dfn"] @ g, atol=1e-8)


def test_cartesian_stiffness_from_jacobian():
    # planar-like check: identity Jacobian -> K_x = Kq ; series of two equal springs halves the stiffness
    Kq = np.array([100.0, 200.0, 300.0, 40.0, 50.0, 60.0])
    assert np.allclose(GC.cartesian_stiffness_from_jacobian(np.eye(6), Kq), np.diag(Kq))
    J = np.hstack([np.eye(6), np.eye(6)])                             # redundant: two joints per DoF in parallel-compliance sense: C = 2/K
    assert np.allclose(GC.cartesian_stiffness_from_jacobian(J, np.concatenate([Kq, Kq])), np.diag(Kq / 2))
    # moving the reference point: a pure translation stiffness seen from an offset origin picks up a rotation coupling, and stays symmetric PD
    K1 = GC.cartesian_stiffness_from_jacobian(np.eye(6), Kq, R_box=np.eye(3), x_hand=[0, 0, 0], origin=[0.1, 0, 0])
    assert np.allclose(K1, K1.T) and np.all(np.linalg.eigvalsh(K1) > 0)
