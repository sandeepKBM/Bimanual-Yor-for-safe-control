"""CarryEpisode(base="swerve", goal_support="counter2"): constructor rules, snapshot/restore of the base, base_move + world-frame arm moves after a turn
(tray6 light, route R1 at L1; ~1 min).  The fixed-base default stays bit-identical (tests/test_carry.py + the A/B run in the build report)."""
import numpy as np
import pytest

from world_model.v2.nero.contents import carry as CR, scenes as SC


def test_default_has_no_base():
    ep = CR.CarryEpisode(SC.TrayWorld(n_cups=2))
    assert ep.base == "" and ep.sb is None and ep.driver is None and ep.step_hooks == [] and ep.goal_support == ""
    with pytest.raises(ValueError):
        ep.base_move(dict(route=[["line", -0.1]], level="L1"))


def test_constructor_rules():
    with pytest.raises(ValueError):
        CR.CarryEpisode(SC.TrayWorld(n_cups=2), goal_support="counter2")      # needs the swerve scene
    with pytest.raises(ValueError):
        CR.CarryEpisode(SC.TrayWorld(n_cups=2), base="mecanum")


@pytest.fixture(scope="module")
def ep():
    w = SC.TrayWorld(n_cups=6)
    e = CR.CarryEpisode(w, base="swerve", goal_support="counter2")
    e.set_belief(float(e.truth0["contents_mass"]))
    e.run_phases([("hold", 0.4), ("move", (0, 0, 0.05), 1.0), ("hold", 0.5)])
    return e


def test_snapshot_restore_covers_the_base(ep):
    snaps = []
    ep.run_phases([("base_move", dict(route=[["line", -0.1]], level="L2"))], branch=lambda e, t: snaps.append((t, e.snapshot())) if len(snaps) < 3 else None)
    assert len(snaps) >= 2
    t, sn = snaps[1]
    assert sn["base"]["active"] and sn["base"]["sb"]["yaw_nom"] is not None
    k0, pose_end = ep.k, ep.sb.pose().copy()
    ep.restore(sn)
    assert ep.k == sn["k"] < k0 and ep.driver.active and np.allclose(ep.sb._v, sn["base"]["sb"]["v"]) and np.array_equal(ep.sb._wi, sn["base"]["sb"]["wi"])
    ep.run_phases([("hold", 4.0)])                                          # the route clock continues from the snapshot (still active until its queue ends)
    assert np.linalg.norm(ep.sb.pose()[:2] - pose_end[:2]) < 5e-3             # same end (the nominal route end), not a stalled or doubled route
    ep.driver.active = False
    ep.run_phases([("move", (0, 0, 0), 0.7)])


def test_base_move_then_world_frame_place(ep):
    """lift 0.05 (done) -> R1 at L1 -> down 0.05 -> release (retreat outward in the ROBOT frame): the tray lands on counter2 at the goal, no U."""
    ep.driver.active = False
    w = ep.world
    r = ep.base_move(dict(route=[["line", -0.4], ["turn", 180.0], ["line", 0.5]], level="L1"))      # the previous test already moved 0.1 m back: total back 0.5, advance 0.5 = R1
    assert r["err_mm"] < 20.0 and abs(r["err_deg"]) < 2.0
    assert ep.track["viol"] is None
    ep.run_phases([("move", (0, 0, -0.05), 1.0), ("hold", 0.7), ("weld", False), ("retreat", CR.RETREAT, 0.8), ("hold", 1.0)])
    goal = np.array(w.obj_xy) + np.array([-1.76, 0.0209])
    assert ep.track["viol"] is None, ep.track["viol"]
    assert np.linalg.norm(ep.d.xpos[ep.obj][:2] - goal) < 0.02
    assert abs(ep.d.xpos[ep.obj][2] - ep.z0) < 0.005
    ok, why = ep.in_R()
    assert ok, why
