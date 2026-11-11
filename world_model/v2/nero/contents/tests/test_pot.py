"""pot task (11-05): scene, weld at rest, closed_chain plan, one carry segment + stored backup, one probe rung.  SIMULATION ONLY, CPU.
    PYTHONPATH=. python -m pytest -q world_model/v2/nero/contents/tests/test_pot.py
The first closed_chain plan of the pot is computed (~40 s) and cached in stack/planners/plan_cache/."""
import mujoco
import numpy as np
import pytest

from world_model.v2.nero.contents import carry as CR, carry_cert as CC, probe as PR, scenes as SC


def _pot(**kw):
    return SC.PotWorld(puck_kg=kw.pop("puck_kg", 0.5), puck_uw=kw.pop("puck_uw", (0.03, 0.02)), **kw)


def test_scene_builds_and_settles():
    w = _pot()
    m = SC.compose(w); d = mujoco.MjData(m)
    SC.settle(m, d, 1.0)
    ob, lid, pk = m.body("obj").id, m.body("lid").id, m.body("contents").id
    assert abs(d.xpos[ob][2] - w.counter_top) < 2e-3
    R = d.xmat[ob].reshape(3, 3)
    assert np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))) < 1.0
    rel = d.xpos[lid] - d.xpos[ob]
    assert np.linalg.norm(rel[:2]) < 1e-3 and abs(rel[2] - 0.0437) < 1e-3          # loose lid sits on the rim
    prel = R.T @ (d.xpos[pk] - d.xpos[ob])
    assert prel[:2] == pytest.approx(w.contents_xy, abs=2e-3)                       # puck at the placement (u, w) -> (x, y) = (-w, u)
    t = SC.object_truth(m, d)
    assert t["contents_mass"] == pytest.approx(0.5)
    m_e, c_e = SC.pot_empty_props(w)
    assert t["empty_mass"] == pytest.approx(m_e, rel=1e-6)                           # pot + lid, MODEL (volume x density) = compiled mass
    assert np.allclose(t["empty_com"], c_e, atol=2e-3)
    for s in ("obj_ref", "obj_grasp_left", "obj_grasp_right"):                      # the tab grasp points of the other tasks exist
        assert m.site(s).id >= 0


def test_handle_length_fixed_at_nominal_and_lid_friction_scaled():
    w = _pot(lid_mu_scale=0.5)
    assert w.handle_len == SC.POT_HANDLE_LEN == 0.11
    m = SC.compose(w)
    assert m.geom("pan_handle").size[0] * 2 == pytest.approx(0.11)
    assert m.geom("pan_handle2").size[0] * 2 == pytest.approx(0.11)
    assert m.geom("lid_disk").friction[0] == pytest.approx(0.6 * 0.5)


@pytest.fixture(scope="module")
def ep_cc():
    ep = CR.CarryEpisode(_pot(), planner="closed_chain")
    return ep


def test_weld_holds_at_rest(ep_cc):
    ep = ep_cc
    assert ep.task == "pot" and ep.lid is not None
    ep.run_phases([("hold", 0.6)])
    assert ep.track["viol"] is None
    assert ep.track["lid_slide_mm"] < 1.0 and ep.track["obj_tilt_max"] < 3.0
    Fz = sum(ep.claw_wrench(s)[0][2] for s in CR.SIDES)
    assert Fz > 0.0
    assert ep.m_nom == pytest.approx(SC.object_truth(ep.m, ep.d)["empty_mass"] + ep.m_hat)


def test_plan_reaches_the_counter_goal(ep_cc):
    from world_model.v2.stack import tasks as TK
    cc = ep_cc.closed_chain()
    geo = cc.geo
    for s in CR.SIDES:
        assert geo.fk_pos[s].max() < 1e-6 and geo.margin[s].min() > 0.0
    gx, gy = TK.GOAL_DXY["pot"]
    assert np.allclose(geo.offs[-1], [gx, gy, 0.0])                                  # back down on the counter, 10 cm toward the robot
    assert [k for k in CR.plan_moves("pot")][0][0] == "lift"


def test_lift_segment_and_stored_backup_run(ep_cc):
    ep = ep_cc
    ep.reset_track()
    ep.store("lift")
    ep.run_phases(ep.segment_phases("lift"))
    assert ep.track["viol"] is None and ep.d.xpos[ep.obj][2] - ep.z0 > 0.04          # the pot is 5 cm up, lid still on
    ep.reset_track()
    ep.run_phases(ep.backup_phases("lift"))                                          # stop, lower back to the lift start, let go in place
    okR, why = ep.in_R()
    assert ep.d.xpos[ep.obj][2] - ep.z0 < 0.01
    assert ep.track["viol"] is None, ep.track["viol"]


def test_lid_off_is_in_U():
    ep = CR.CarryEpisode(_pot())
    ep.d.qpos[ep.m.jnt_qposadr[ep.m.joint("lid_free").id] + 2] += 0.03               # lift the lid 3 cm off the rim
    mujoco.mj_forward(ep.m, ep.d)
    ep._check(0.0, 0.0)
    assert ep.track["viol"]["kind"] == "lid_off"
    assert "lid" in ep.in_R()[1]


def test_cert_cells_cover_the_theta_mass_range_and_static_rows():
    cs = CC.cells("pot")
    assert cs[0][0][0] == 0.02 and cs[-1][0][1] == 1.5
    sp = CC.cell_spec("pot", (0.02, 0.25))
    assert sp.open_limits[0].name == "lid_slide"                                     # solid contents: no spill row
    w = CC.sample_world("pot", (0.5, 0.75), np.random.default_rng(1))
    assert isinstance(w, SC.PotWorld) and 0.5 <= w.puck_kg <= 0.75 and 0.5 <= w.lid_mu_scale <= 1.0
    assert np.hypot(*w.puck_uw) <= 0.06 + 1e-9                                       # inside C_R
    st = CC.static_cell("pot", (0.02, 0.25), SC.PotWorld(), planner="closed_chain")
    assert "support_dest" in st["rows"]["release"] and st["rows"]["release"]["support_dest"]["ok"]
    assert st["rows"]["lift"]["open_content"]["ok"]


def test_probe_ladder_runs_one_rung():
    assert __import__("world_model.v2.stack.tasks", fromlist=["x"]).prior_box("pot") == (0.02, 1.5) and PR.prior_contents("pot") == (0.02, 1.5)
    assert PR.empty_mass("pot") == pytest.approx(SC.pot_empty_props()[0])
    sp = PR.interval_spec("pot", 0.1, 0.4)
    assert sp.contents.m_lo == 0.1 and sp.contents.m_hi == 0.4
    w = CC.sample_world("pot", (0.3, 0.3001), np.random.default_rng(2))
    ep = PR._episode(w)
    pr = PR.make_probe(ep, "pot", eps_lo=0.05, eps_free=0.05, noise=False, method="ladder")
    ok, rows = pr.admissible(pr.M_lo)
    assert ok
    ep.ctl.mass = 0.0
    ep.run_phases([("hold", 0.3)])
    L, nb, rise = pr.rung(pr.M_lo)                                                   # first rung: feedforward of the lightest object
    assert np.isfinite(L) and nb > 0
    assert ep.track["viol"] is None
