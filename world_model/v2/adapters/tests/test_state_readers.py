"""state readers against the old backends' own monitors: tray4 hold with forced first violations, a pot hold with the lid nudged, a pan hold (R v1 and v2
against segments.Episode.in_R), the coffee-tray detail reader.  Each fixture is a few seconds of sim."""
import mujoco
import pytest

from world_model.v2.adapters import contents_specs as CS
from world_model.v2.adapters.state_carry import from_carry
from world_model.v2.adapters.state_embodichain import from_embodichain
from world_model.v2.adapters.state_pan import from_pan
from world_model.v2.nero.contents import carry as CR
from world_model.v2.nero.pan import segments as S
from world_model.v2.stack import goal as GL
from world_model.v2.stack import tasks as TK
from world_model.v2.stack.checks import monitors as MON
from world_model.v2.stack.checks.row import inside


@pytest.fixture(scope="module")
def tray4():
    ep = CR.CarryEpisode(CS.build_world("tray4"), planner="scripted")
    ep.run_phases([("hold", 0.4)])
    return ep


@pytest.fixture(scope="module")
def pot():
    ep = CR.CarryEpisode(CS.build_world("pot"), planner="scripted")
    ep.run_phases([("hold", 0.4)])
    return ep


@pytest.mark.parametrize("kind", ["tau_sat", "obj_tilt", "obj_drop", "cup_tipped", "spill", "cup_off_tray", "collision"])
def test_tray4_forced_first_violation(tray4, kind):
    t = TK.task("tray4")
    ep = tray4
    assert ep.track["viol"] is None
    s0 = from_carry(ep)
    assert MON.u_row(s0, t).ok and MON.unsafe(s0, t) is False and MON.recoverable(s0, t).ok == ep.in_R()[0]
    ep.track["viol"] = dict(kind=kind, t=1.0)
    try:
        s = from_carry(ep)
        r = MON.u_row(s, t)
        assert not r.ok and f"U entered: {kind}" in r.why                    # the backend's time-ordered first, not the table order
        assert MON.unsafe(s, t) is True
        assert f"U:{kind}" in "U:" + kind and not MON.recoverable(s, t).ok and (not ep.in_R()[0]) and ep.in_R()[1][-1] == f"U:{kind}"
        # the flag / just-past-the-bound value the reader gives for the three first-only kinds also trips the TABLE
        tab = MON.first_violation(s, t)[0]
        assert tab == {"collision": "collision_flag", "cup_off_tray": "cup_off_tray_flag", "obj_drop": "drop_m"}.get(kind)
    finally:
        ep.track["viol"] = None


def test_pot_lid_nudge(pot):
    t = TK.task("pot")
    ep = pot
    assert MON.recoverable(from_carry(ep), t).ok == ep.in_R()[0]
    sn = (ep.d.qpos.copy(), ep.d.qvel.copy())
    la = ep.m.jnt_qposadr[ep.m.body(ep.lid).jntadr[0]]
    ep.d.qpos[la + 2] += 0.02                                              # 2 cm up: off its seat (LID_OFF_Z 1 cm)
    mujoco.mj_forward(ep.m, ep.d)
    try:
        okR, why = ep.in_R()
        s = from_carry(ep)
        r = MON.recoverable(s, t)
        assert not okR and "lid" in why and not r.ok and "lid_dz_m" in r.why
        ep._check(1.0, 0.0)                                                # the old monitor records lid_off
        assert ep.track["viol"] and ep.track["viol"]["kind"] == "lid_off"
        s = from_carry(ep)
        assert not MON.u_row(s, t).ok and "lid_off" in MON.u_row(s, t).why and s.worst["lid_dz_m"] > 0.01
    finally:
        ep.track["viol"] = None
        ep.d.qpos[:], ep.d.qvel[:] = sn
        mujoco.mj_forward(ep.m, ep.d)


PAN_Q = {"tilt": "obj_tilt_deg", "dz": "abs(dz_mm)", "lid": "lid_mm", "spill": "spill_flag", "rim": "rim_in_m", "support": "support_frac", "pan_v": "pan_speed",
         "arm_v": "arm_speed", "yaw": "abs(yaw_deg)", "released": "released", "reach": "reach", "stable": "stable_margin_m"}


@pytest.fixture(scope="module")
def pan_ep():
    ep = S.Episode(dict(placement="far_left", hidden_mass=0.4, perturb=dict(dx=0.015, dy=-0.015, yaw_deg=8.0, mu_scale=0.6)), observe=None)
    ep.run_phases(ep.segment_phases("start"))
    return ep


def _compare_R(ep):
    t = TK.task("pan_lift")
    for version in ("v1", "v2"):
        ok, mg, _ = ep.in_R(version=version)                               # reach=True: the same find_pregrasp call as the reader
        s = from_pan(ep, version=version)
        tab = dict(t.R_table(version))
        r = MON.recoverable(s, t, version=version)                         # pan: R does not include U (TaskDef.r_includes_U False)
        assert r.ok == ok, (version, r.why, mg)
        assert set(PAN_Q[k] for k in mg) == set(r.worst), (version, sorted(mg), sorted(r.worst))
        for k, v in mg.items():
            assert inside(r.worst[PAN_Q[k]], tab[PAN_Q[k]]) == (v > 0), (version, k, v, r.worst[PAN_Q[k]])   # old flags are +-1, new flag margins 0 when inside


def test_pan_hold_R_v1_and_v2_match_old_in_R(pan_ep):
    _compare_R(pan_ep)
    s = from_pan(pan_ep)
    assert MON.u_row(s, TK.task("pan_lift")).ok == (pan_ep.track["viol"] is None)
    assert from_pan(pan_ep, reach=False).now.get("reach") is None
    assert MON.recoverable(from_pan(pan_ep, reach=False), TK.task("pan_lift")).guarantee in ("missing", "runtime")    # an unmeasured reach is never a pass


def test_pan_lid_nudge_R_flips_like_old(pan_ep):
    ep = pan_ep
    sn = ep.snapshot()
    la = ep.m.jnt_qposadr[ep.m.body(ep.lid).jntadr[0]]
    ep.d.qpos[la] += 0.02
    mujoco.mj_forward(ep.m, ep.d)
    try:
        assert not ep.in_R(reach=False)[0]
        _compare_R(ep)
        assert not MON.recoverable(from_pan(ep, reach=False), TK.task("pan_lift")).ok
    finally:
        ep.restore(sn)


def test_embodichain_unrounded_xy_err():
    d = dict(tray_dxy=[0.1, 0.0], capsule_rel_tray=[0.0, 0.0, 0.01], tray_level=True, tray_xy_err=0.0300004)
    s = from_embodichain(d)
    assert s.now["xy_err"] == 0.0300004                                    # the sim's unrounded value, not hypot of rounded positions
    assert not GL.check_goal(GL.goal_for("coffee_tray"), s).ok              # old: norm <= 0.03 -> 0.0300004 fails
    assert GL.check_goal(GL.goal_for("coffee_tray"), from_embodichain(dict(d, tray_xy_err=0.03))).ok


def test_local_tilt_and_dt_equal_carry():
    import numpy as np
    from world_model.v2.adapters import state_carry as SCY
    from world_model.v2.nero.contents import carry as CR
    assert SCY.DT == CR.DT
    R = np.array([[1, 0, 0], [0, 0.8, -0.6], [0, 0.6, 0.8]])
    assert SCY._tilt_deg(R) == CR._tilt_deg(R)
