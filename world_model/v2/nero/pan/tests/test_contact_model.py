"""cert/contact_model.py: the conservative hold LP matches analytic cases, has zero torsion, extracts flush/tilted pad contacts,
and its admission is exact at the load vertices (convexity)."""
import numpy as np

from world_model.v2.nero.pan.cert import contact_model as CM

O = np.zeros(3)


def _pads(sep=0.02, half=0.01, n_pts=4, mu=0.6):
    """two pads squeezing a block along x (normals into the block), corners in the y-z plane."""
    if n_pts == 4:
        yz = [(-half, -half), (half, -half), (half, half), (-half, half)]
    else:
        yz = [(0.0, 0.0)]
    c = []
    for sgn in (1, -1):
        p = np.array([[-sgn * sep, y, z] for y, z in yz])
        c.append(dict(body="pan", hand="left", p=p, n=np.array([sgn, 0, 0.0]), mu=mu))
    return c


NO_RIM = dict(p=np.zeros((0, 3)), n=np.array([0, 0, 1.0]))
LID0 = np.zeros(6)


def _lid_support():
    """a lid with no load needs no forces: give the lid body zero load and no contacts (its 6 rows read 0 = 0)."""
    return LID0


def test_pinch_capacity_matches_coulomb():
    F, mu, m = 10.0, 0.6, 0.3
    H = CM.Hold(_pads(mu=mu), NO_RIM, 0.36, O, F)
    w = np.r_[0, 0, -m * CM.G, 0, 0, 0]
    k = H.load_factor(w, _lid_support())
    assert abs(k - mu * F / (m * CM.G)) < 1e-6


def test_point_contacts_have_no_torsion():
    H = CM.Hold(_pads(n_pts=1), NO_RIM, 0.36, O, 10.0)
    w = np.r_[0, 0, 0, 0.01, 0, 0]                 # pure moment about the jaw axis x
    assert H.load_factor(w, LID0) < 1e-9
    H4 = CM.Hold(_pads(n_pts=4), NO_RIM, 0.36, O, 10.0)
    assert H4.load_factor(w, LID0) > 1.0           # pads (corner contacts) do resist it


def test_flush_pad_gives_corners_tilted_pad_gives_edge():
    box = np.array([[x, y, z] for x in (-0.01, 0.01) for y in (-0.01, 0.01) for z in (0.0, 0.01)])
    surf = dict(p=np.zeros(3), n=np.array([0, 0, 1.0]),
                poly=np.array([[-0.05, -0.05, 0], [0.05, -0.05, 0], [0.05, 0.05, 0], [-0.05, 0.05, 0]]))
    pts, gap, _ = CM.contact_points(box, surf)
    assert len(pts) >= 4 and abs(gap) < 1e-9 and np.ptp(pts[:, 0]) > 0.0195 and np.ptp(pts[:, 1]) > 0.0195
    a = np.radians(2.0)                            # tilt about y: the -x edge goes down; strip width ~ FLUSH_H / tan(2 deg) = 2.9 mm
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    pts, _, _ = CM.contact_points(box @ R.T, surf)
    assert np.ptp(pts[:, 0]) <= CM.FLUSH_H / np.tan(a) + CM.GRID and np.ptp(pts[:, 1]) > 0.0195
    off = dict(surf, poly=surf["poly"] * 0.1 + np.array([0.04, 0, 0]))   # region away from the pad: the pad misses it
    pts, _, _ = CM.contact_points(box, off)
    assert len(pts) == 0


def test_lid_slides_past_friction_angle():
    """lid on a ring of rim points; the pan body is held by an unlimited support.  Inner 8-pyramid: tangential capacity mu*cos(pi/8)
    between edges, mu along an edge."""
    mu = 0.36
    ring = np.array([[0.09 * np.cos(t), 0.09 * np.sin(t), 0.04] for t in np.linspace(0, 2 * np.pi, 48, endpoint=False)])
    sup = [dict(body="pan", hand="support", p=ring * [1, 1, 0], n=np.array([0, 0, 1.0]), mu=5.0)]
    H = CM.Hold(sup, dict(p=ring, n=np.array([0, 0, 1.0])), mu, O, 10.0)
    m = 0.1
    for tan_t, ok in ((0.95 * mu * np.cos(np.pi / 8), True), (1.05 * mu, False)):
        g = CM.G * np.array([tan_t, 0, -1.0]) / np.sqrt(1 + tan_t ** 2)
        w_lid = np.r_[m * g, np.cross([0, 0, 0.045], m * g)]
        k = H.load_factor(np.zeros(6), w_lid)
        assert (k >= 1.0) == ok, (tan_t, k)


def test_vertex_check_is_exact_by_convexity():
    rng = np.random.default_rng(0)
    H = CM.Hold(_pads(), NO_RIM, 0.36, O, 10.0)
    V = [np.r_[rng.normal(0, 1.5, 3), rng.normal(0, 0.02, 3)] for _ in range(6)]
    kmin = min(H.load_factor(v, LID0) for v in V)
    for _ in range(30):
        lam = rng.dirichlet(np.ones(len(V)))
        assert H.load_factor(sum(l * v for l, v in zip(lam, V)), LID0) >= kmin - 1e-7


def test_g_eff_box_covers_tilt_cone_and_accel():
    gs = np.array(CM.g_eff_corners(0.5, 3.0))
    lo, hi = gs.min(axis=0), gs.max(axis=0)
    rng = np.random.default_rng(1)
    for _ in range(200):
        th, ph = np.radians(3.0) * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        g = -CM.G * np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]) - rng.uniform(-0.5, 0.5, 3)
        assert np.all(g >= lo - 1e-12) and np.all(g <= hi + 1e-12)


def _tag(cs, squeezed=True):
    out = []
    for i, c in enumerate(cs):
        for p in c["p"]:
            out.append(dict(c, p=p[None, :], finger=1 + (i % 2), squeezed=squeezed))
    return out


def test_compliant_pinch_holds_below_coulomb_limit_only():
    F, mu = 10.0, 0.6
    cs = _tag(_pads(mu=mu))
    for frac, gap, ok in ((0.9, 0.0, True), (1.1, 0.0, False), (0.3, CM.GAP_TOL_C, True), (1.1, CM.GAP_TOL_C, False)):
        m = frac * mu * F / CM.G
        C = CM.compliant(cs, O, np.array([[0, 0, -m * CM.G, 0, 0, 0]]), np.array([[F]]), gap_tol=gap)
        assert (C["margin"] >= 0) == ok, (frac, gap, C["margin"], C["why"])
        if gap == 0.0:
            assert abs(C["margin"] - (1 - frac)) < 1e-9          # creep: every corner carries W/8 on F/8: ratio = W / (mu F)


def test_compliant_rejects_the_side_push_the_existence_lp_admits():
    """V10 geometry in miniature: the left claw pinches a handle (vertical jaw) 0.17 m from the COM; the right claw's fingers only
    touch the far wall SIDE-ON (normal +x, perpendicular to their closing direction).  The existence LP may let the arm push the
    wall and lean on friction; the compliant model gives that contact no pre-load, so it cannot carry weight."""
    mu, F = 0.6, 10.0
    left = []
    for sgn in (1, -1):                                    # pads above / below the handle, 4 corners, normals -+z into the handle
        p = np.array([[0.17 + dx, dy, 0.03 - sgn * 0.007] for dx, dy in ((-0.012, -0.01), (0.012, -0.01), (0.012, 0.01), (-0.012, 0.01))])
        left.append(dict(body="pan", hand="left", p=p, n=np.array([0, 0, -sgn * 1.0]), mu=mu))
    right = [dict(body="pan", hand="right", p=np.array([[-0.093, 0.0, z]]), n=np.array([1.0, 0, 0]), mu=mu, finger=f, squeezed=False)
             for f, z in ((1, 0.035), (2, 0.028))]
    cs = _tag(left) + right
    w = np.r_[0, 0, -0.3 * CM.G, 0, 0, 0]                  # 0.3 kg at the origin
    H = CM.Hold(cs, NO_RIM, 0.36, O, {"left": 10.0, "right": 3.0})
    assert H.load_factor(w, LID0) >= 1.0                   # existence: "holds"
    C = CM.compliant(cs, O, w[None, :], np.array([[F]]))
    assert C["margin"] < 0                                 # compliant: no
