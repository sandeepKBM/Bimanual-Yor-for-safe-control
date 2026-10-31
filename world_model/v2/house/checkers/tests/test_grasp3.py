"""tests for checkers/grasp3.py (pure numpy, no sim)."""
import itertools
import numpy as np
import pytest

from world_model.v2.house.checkers import grasp3 as g
from world_model.v2.house.pick import assign as A


def names(rep):
    return {r.name for r in rep.feasible}


def test_constants_reused_from_assign():
    assert g.GRIPPER_OPEN_M == A.GRIPPER_OPEN_M == 0.07
    assert g.SINGLE_MAX_KG == A.SINGLE_MAX_KG and g.MIN_HAND_SEP_M == A.MIN_HAND_SEP_M and g.MAX_SPAN_M == A.MAX_SPAN_M
    assert g.JAW_MARGIN_M == A.JAW_MARGIN_M and g.FINGER_FORCE_N == A.FINGER_FORCE_N


def test_candidate_set_geometry():
    c = {x.name for x in g.generate_candidates((0.28, 0.20, 0.029), "Book")}
    assert c == {"topdown_pinch", "edge_pinch", "bimanual_squeeze_a"}          # b=20 cm < 25 cm -> only the a axis
    c = {x.name for x in g.generate_candidates((0.09, 0.09, 0.10), "Mug")}
    assert {"rim_pinch", "handle_loop_pinch", "topdown_pinch"} <= c and "edge_pinch" not in c
    c = {x.name for x in g.generate_candidates((0.3, 0.3, 0.1), "Box")}
    assert {"bimanual_squeeze_a", "bimanual_squeeze_b"} <= c and "rim_pinch" not in c
    assert "handle_shank_pinch" in {x.name for x in g.generate_candidates((0.4, 0.25, 0.1), "Pan")}
    # explicit hints override the category
    assert "rim_pinch" in {x.name for x in g.generate_candidates((0.1, 0.1, 0.1), None, has_rim=True)}


def test_thin_light_book_edge_pinch_single():
    d = g.decide((0.20, 0.14, 0.02), (0.1, 0.3), category="Book")
    assert d["mode"] == "single" and d["grasp"] == "edge_pinch"
    assert A.assign((0.2, 0.14, 0.02), (0.5, 0, 0.9), mass_vlm=0.3).mode != "single"       # the old top-down-only rule says no


def test_heavy_wide_book_bimanual_not_single():
    d = g.decide((0.28, 0.20, 0.029), (0.65, 1.3), category="Book")
    assert d["mode"] == "bimanual" and d["grasp"].startswith("bimanual")
    assert not [r for r in d["report"].feasible if r.arms == 1]


def test_mug_single_via_rim_or_handle_and_old_rule_agrees_pinch_for_pen():
    d = g.decide((0.09, 0.09, 0.10), (0.2, 0.5), category="Mug")
    assert d["mode"] == "single" and d["grasp"] in ("handle_loop_pinch", "rim_pinch")
    assert g.decide((0.14, 0.01, 0.01), (0.005, 0.03), category="Pen")["mode"] == "single"


def test_bowl_wide_single_via_rim_only_when_light():
    assert g.decide((0.20, 0.20, 0.07), (0.1, 0.25), category="Bowl")["grasp"] == "rim_pinch"      # light bowl: rim pinch holds
    assert g.decide((0.20, 0.20, 0.07), (0.15, 0.8), category="Bowl")["mode"] == "infeasible"       # 0.8 kg: payload + rim pitch torque


def test_infeasible_goes_to_probe():
    d = g.decide((0.3, 0.3, 0.4), (0.3, 5.0), category="Statue")
    assert d["mode"] == "infeasible" and "probe" in d["reason"] and d["arms"] == 0 and d["fallback_arms"] in (1, 2)


def test_payload_enforced():
    # a mass the grip could hold but the arm payload cannot
    r = g.check_grasp((0.04, 0.03, 0.03), (0.8, 0.9)).candidates[0]
    assert not r.ok and r.binding["binding"] == "payload"


def test_width_gate_matches_old_rule_for_topdown():
    for b in (0.05, 0.059, 0.061, 0.08):
        r = {x.name: x for x in g.check_grasp((0.3, b, 0.05), (0.01, 0.05)).candidates}["topdown_pinch"]
        assert (r.ok) == (b <= A.SINGLE_MAX_WIDTH_M)


def _G(cand, m, d, a, mu):
    return float(g.grip_gauge(cand, [m], np.array([d]), np.array([a]), mu)[0][0])


def test_corners_are_exact_max_over_box():
    """random interior points never exceed the best corner (multi-affine w + convex gauge => vertex max)."""
    rng = np.random.default_rng(0)
    ext = (0.28, 0.20, 0.029)
    for cand in g.generate_candidates(ext, "Book") + g.generate_candidates((0.09, 0.09, 0.1), "Mug"):
        half = np.array([0.012, 0.01, 0.02]); mlo, mhi = 0.2, 1.1
        acc = [(-1, 1), (-1, 1), (-1, 1.5)]
        M = np.array([mlo, mhi]); D = g._corners(-half, half); AC = g._corners([x[0] for x in acc], [x[1] for x in acc])
        gc = max(_G(cand, m, d, a, 0.4) for m in M for d in D for a in AC)
        for _ in range(400):
            m = rng.uniform(mlo, mhi); d = rng.uniform(-half, half); a = np.array([rng.uniform(*x) for x in acc])
            assert _G(cand, m, d, a, 0.4) <= gc + 1e-9


def test_report_margin_equals_corner_max():
    ext = (0.20, 0.14, 0.02)
    rep = g.check_grasp(ext, (0.1, 0.3), com_box=(0.01, 0.01, 0.002), category="Book")
    r = [x for x in rep.candidates if x.name == "edge_pinch"][0]
    cand = [c for c in g.generate_candidates(ext, "Book") if c.name == "edge_pinch"][0]
    M = [0.1, 0.3]; D = g._corners([-0.01, -0.01, -0.002], [0.01, 0.01, 0.002]); AC = g._corners([-1, -1, -1], [1, 1, 1.5])
    gmax = max(_G(cand, m, d, a, g.MU_LO_DEFAULT) for m in M for d in D for a in AC)
    assert abs((1 - gmax) - min(1 - gmax, r.binding["payload_margin"])) < 1e-9 or r.binding["binding"] == "payload"
    assert abs(r.binding["gauge"] - gmax) < 1e-9


EXTS = [((0.28, 0.20, 0.029), "Book"), ((0.20, 0.14, 0.02), "Book"), ((0.09, 0.09, 0.10), "Mug"), ((0.20, 0.20, 0.07), "Bowl"), ((0.3, 0.3, 0.1), "Box"),
        ((0.4, 0.25, 0.15), "Pan"), ((0.33, 0.23, 0.02), "Laptop"), ((0.07, 0.07, 0.2), "Vase")]


def test_monotonicity_property():
    """enlarging mass_range, com_box, accel_box or lowering mu_lo can only remove feasible candidates, and margins never increase."""
    rng = np.random.default_rng(1)
    for _ in range(60):
        ext, cat = EXTS[rng.integers(len(EXTS))]
        m_lo = rng.uniform(0.02, 0.8); m_hi = m_lo + rng.uniform(0, 0.8)
        com = rng.uniform(0, 0.03, 3); mu = rng.uniform(0.2, 0.9)
        acc = [(-rng.uniform(0, 1), rng.uniform(0, 1)), (-rng.uniform(0, 1), rng.uniform(0, 1)), (-rng.uniform(0, 1), rng.uniform(0, 2))]
        base = {r.name: r for r in g.check_grasp(ext, (m_lo, m_hi), com, (mu, 1.0), cat, accel_box=acc).candidates}
        k = rng.integers(4)
        m2, c2, mu2, a2 = (m_lo, m_hi), com, mu, acc
        if k == 0:
            m2 = (m_lo * rng.uniform(0.3, 1), m_hi * rng.uniform(1, 2))
        elif k == 1:
            c2 = com + rng.uniform(0, 0.03, 3)
        elif k == 2:
            mu2 = mu * rng.uniform(0.3, 1)
        else:
            a2 = [(lo - rng.uniform(0, 1), hi + rng.uniform(0, 1)) for lo, hi in acc]
        new = {r.name: r for r in g.check_grasp(ext, m2, c2, (mu2, 1.0), cat, accel_box=a2).candidates}
        assert set(base) == set(new)
        for n, r in new.items():
            assert r.margin <= base[n].margin + 1e-9, (n, k)
            assert (not r.ok) or base[n].ok


def test_friction_lowers_capacity_only_at_mu_lo():
    r1 = g.check_grasp((0.28, 0.2, 0.029), (0.6, 1.0), mu_range=(0.4, 0.4), category="Book")
    r2 = g.check_grasp((0.28, 0.2, 0.029), (0.6, 1.0), mu_range=(0.4, 0.9), category="Book")
    assert [x.margin for x in r1.candidates] == [x.margin for x in r2.candidates]          # mu_hi is only reported
    assert all(x.margin_mu_hi >= x.margin - 1e-12 or np.isnan(x.margin_mu_hi) for x in r2.candidates)


def test_decide_prefers_single_and_ranks():
    d = g.decide((0.09, 0.09, 0.10), (0.2, 0.5), category="Mug", centroid_y=0.1)
    assert d["arm"] == "left"
    ms = [r.margin for r in d["report"].feasible]
    assert ms == sorted(ms, reverse=True)
