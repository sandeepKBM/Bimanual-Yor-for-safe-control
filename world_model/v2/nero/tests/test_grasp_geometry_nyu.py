"""11-03 grasp-geometry fix for the NYU gripper: TCP / jaw opening rule, welded-contact rule, hold-pose cache."""
import numpy as np
import pytest

from world_model.v2.nero import config as C, gripper as G


def test_tcp_is_40_percent_of_pivot_to_tip():
    from world_model.v2.nero import nyu_gripper as N
    assert abs(G.tcp_offset() - (0.1225 + 0.4 * (0.2110 - 0.1225))) < 1e-9


def test_jaw_gap_matches_sampled_mesh_within_3mm():
    import trimesh
    from world_model.v2.nero import nyu_gripper as N
    if not (N.OUT_DIR / "finger_a.stl").exists():
        pytest.skip("jaw meshes not split")
    np.random.seed(0)
    P = trimesh.load(str(N.OUT_DIR / "finger_a.stl")).sample(120000)
    B = np.array([N.mesh_to_base(p) for p in P])
    piv = np.array(N.JAW_PIVOT_YZ_M)
    for th, z in ((0.0, 0.17), (0.4, 0.17), (0.75, 0.17)):
        v = B[:, [1, 2]] - piv
        c, s = np.cos(th), np.sin(th)
        y, zz = piv[0] + c * v[:, 0] - s * v[:, 1], piv[1] + s * v[:, 0] + c * v[:, 1]
        sel = (np.abs(zz - z) < 0.0015) & (np.abs(B[:, 0]) < 0.008) & (B[:, 2] > 0.13)
        assert abs(2 * np.percentile(y[sel], 1) - G.jaw_gap_m(th, z)) < 0.003


def test_opening_for_width_hits_the_target_gap_and_flags_infeasible():
    o = G.opening_for_width(0.020)
    assert o["feasible"] and abs(o["gap"] - 0.024) < 1e-6 and 0.0 < o["frac"] < 0.2
    assert not G.opening_for_width(0.014)["feasible"]          # 14 mm + 4 mm < 22.2 mm (fully closed gap at the TCP)
    assert G.opening_for_width(0.014)["gap_min"] > 0.022


def test_welded_contact_rule_toggles_only_that_gripper_object_pairs():
    import mujoco
    from world_model.v2.nero.contents import carry as CR, scenes as SC
    ep = CR.CarryEpisode(SC.TrayWorld())
    wc, m = ep.wc, ep.m
    assert wc.active and wc.pairs_off(ep.d) == {"left": 0, "right": 0}          # both welded at t = 0: no gripper<->object pair
    ep.weld("left", False)
    ep.wc.tick(ep.d)
    off = wc.pairs_off(ep.d)
    assert off["right"] == 0 and (off["left"] > 0 or "left" in wc.pending)       # left restored (or deferred while the jaws overlap the tray)
    # table / counter pairs are untouched: a gripper geom still pairs with the counter
    g = wc.grip_geoms["left"][0]
    cn = m.geom("counter").id
    assert (m.geom_contype[cn] & m.geom_conaffinity[g]) or (m.geom_contype[g] & m.geom_conaffinity[cn])


def test_nyu_hold_cache_has_5mm_clearance():
    import json
    from world_model.v2.nero.contents import envelope as E
    pv = json.loads(E.CAP_FILE.read_text())["pot"]["provenance"]
    assert pv["clearance_at_hold"]["min"] >= 0.005 and pv["clearance_at_lift"]["min"] >= 0.005
