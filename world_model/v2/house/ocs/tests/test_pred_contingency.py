"""E4 predictive contingency set ('hold' vs 'hold+brake'): brake profile, the brake branch spreading the cup set, the conservative combination
rule, and the default path staying bit-identical.  SIMULATION ONLY.   python -m pytest world_model/v2/house/ocs/tests/test_pred_contingency.py -q"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from world_model.v2.house.ocs import dynamics as D, evalcore as EC, util as U
from world_model.v2.house.ocs.tests.test_ocs import _tray_ws
from world_model.v2.house.teleop import base as BS


def _moving_ws(v=(0.9, 0.0, 0.0)):
    ws = _tray_ws(yaw=0.0)
    ws.robot.base_vel = np.array(v, float)
    return ws


def _cup_half_x(traj, k=-1):
    """x half-width of the cup's own position set (tray frame; the set is anchored at the nominal centre)."""
    u = traj[k].objects["cup"].unc.pos
    return float(np.abs(u.G[0]).sum()) if u is not None else 0.0


# ------------------------------------------------------------------------------------------------ brake profile
def test_brake_plan_is_the_shield_hardware_slew_to_zero():
    v0 = np.array([0.9, -0.2, 0.5])
    plan = D.brake_plan(v0, 10, 0.1)
    tw = np.asarray(plan.twists)
    assert tw.shape == (10, 3)
    # end-of-step twists: per-axis slew at A_HW = [4, 4, 8]; an axis stops at exactly zero and stays there (never reverses)
    assert np.allclose(tw[0], [0.9 - 0.4, 0.0, 0.0])
    assert np.allclose(tw[1], [0.9 - 0.8, 0.0, 0.0])
    assert np.allclose(tw[2:], 0.0)
    # same law as the shield's emergency fallback  u = v - sign(v) min(|v|, a_hw dt)  applied tick by tick
    v = v0.copy()
    for _ in range(5):                                           # 5 x 20 ms = the first 0.1 s step
        v = v - np.sign(v) * np.minimum(np.abs(v), BS.A_HW * 0.02)
    assert np.allclose(v, tw[0])
    # at rest: brake == hold
    assert np.allclose(D.brake_plan(np.zeros(3), 3, 0.1).twists, 0.0)


# ------------------------------------------------------------------------------------------------ the brake branch moves / spreads the cup set
def test_brake_branch_moves_and_spreads_the_cup_set():
    ws = _moving_ws()
    hold = D.rollout(ws, 1.0, 0.1, robot_plan=None)
    brake = D.rollout(ws, 1.0, 0.1, robot_plan=D.brake_plan(ws.robot.base_vel, 10, 0.1))
    # hold: constant twist -> no tray acceleration -> the cup does not slide
    assert np.allclose(hold[-1].objects["cup"].pose.p, hold[0].objects["cup"].pose.p, atol=1e-9)
    # brake: 4 m/s^2 > mu_hi g (3.43) -> the nominal (mu_mid) cup slides FORWARD in the tray frame ...
    d_nom = brake[-1].objects["cup"].pose.p[0] - brake[0].objects["cup"].pose.p[0]
    assert d_nom > 0.03, d_nom
    # ... and the set widens beyond the hold set: the mu_lo / mu_hi hypotheses slide different distances (dynamics._support_traj rebuilds them from
    # params['mu'] = the tracker's (lo, hi) prior; it does not read the anchored tracker's hypothesis states, only pose + extra['v_rel'])
    assert _cup_half_x(brake) > _cup_half_x(hold) + 0.02, (_cup_half_x(brake), _cup_half_x(hold))
    # the lo / hi slides really differ under the brake accel (this is what the set half-width is built from)
    ctx_dt = 0.1
    R = [np.eye(3)] * 11
    a = [np.array([-4.0, 0.0, 0.0])] * 10
    P_lo, _ = D._slide_traj(np.zeros(2), np.zeros(2), R, a, 0.18, 0.0, 1.0, 10, ctx_dt)
    P_hi, _ = D._slide_traj(np.zeros(2), np.zeros(2), R, a, 0.35, 0.0, 1.0, 10, ctx_dt)
    assert abs(P_lo[-1, 0] - P_hi[-1, 0]) > 0.02
    # the base itself stops: shorter travel than hold
    d_hold = np.linalg.norm(hold[-1].robot.base.p - hold[0].robot.base.p)
    d_brake = np.linalg.norm(brake[-1].robot.base.p - brake[0].robot.base.p)
    assert d_brake < d_hold - 0.1
    # a robot at rest: the brake branch cannot move the cup
    ws0 = _moving_ws((0.0, 0.0, 0.0))
    b0 = D.rollout(ws0, 1.0, 0.1, robot_plan=D.brake_plan(ws0.robot.base_vel, 10, 0.1))
    assert np.allclose(b0[-1].objects["cup"].pose.p, b0[0].objects["cup"].pose.p, atol=1e-9)


# ------------------------------------------------------------------------------------------------ combination rule
def test_combine_status_rule():
    c = EC.combine_status
    assert c(["true", "true"]) == "true"
    assert c(["false", "false"]) == "false"
    assert c(["true", "false"]) == "unknown"
    assert c(["true", "unknown"]) == "unknown"
    assert c(["false", "unknown"]) == "unknown"
    assert c(["unknown", "unknown"]) == "unknown"
    assert c(["true"]) == "true" and c(["false"]) == "false"


def test_combine_pred_per_atom_and_margins():
    nan = float("nan")
    per = {"hold": (("true", 0.10, 0.02), {"inside": "true", "upright": "true"}),
           "brake": (("false", -0.03, -0.05), {"inside": "false", "upright": "true"})}
    st, ast = EC._combine_pred(per)
    assert st[0] == "unknown" and st[1] == -0.03 and st[2] == -0.05            # margins: the minimum over contingencies
    assert ast == {"inside": "unknown", "upright": "true"}
    per = {"hold": (("true", 0.1, 0.0), {"inside": "true"}), "brake": (("true", 0.2, 0.1), {"inside": "true"})}
    assert EC._combine_pred(per)[0][0] == "true"
    per = {"hold": (("unknown", nan, nan), None), "brake": (("unknown", nan, nan), None)}
    st, ast = EC._combine_pred(per)
    assert st[0] == "unknown" and ast is None


# ------------------------------------------------------------------------------------------------ e4_records plumbing
def _fake_rec(n=70):
    ws = _moving_ws()
    ws.objects["wall"].name, ws.objects["trk0"].name = "p_stand", "p_walker"          # the h5 spec mapping (laptop -> p_stand, person -> p_walker)
    rec = SimpleNamespace(meta=dict(scene="h5"), n=n)
    return rec, [ws] * n, [ws] * n


def test_default_path_unchanged_and_hold_branch_identical():
    rec, wsp, wst = _fake_rec()
    base = EC.e4_records(rec, wsp, wst, stride=20, pred_stride=5)
    hold = EC.e4_records(rec, wsp, wst, stride=20, pred_stride=5, pred_contingency="hold")
    assert _same(base, hold)
    assert all("perc_cont" not in r for r in base["pred"])
    # reference computed the old way: perception rollout with robot_plan=None
    from world_model.v2.house.ocs import consumers as C
    tr = D.rollout(wsp[0], 1.0, 0.1, robot_plan=None)
    ref, ref_ast = EC._pred_perc(tr, base["mapping"], 10)
    assert _same(base["pred"][0]["perc"], ref) and base["pred"][0]["perc_astat"] == ref_ast
    assert not np.isnan(ref[1])                                    # the fixture is a COMPLETE spec (finite margins)
    # hold+brake: the hold branch reproduces the default row, the instantaneous rows are untouched
    both = EC.e4_records(rec, wsp, wst, stride=20, pred_stride=5, pred_contingency="hold+brake")
    assert _same(both["inst"], base["inst"])
    for rb, r0 in zip(both["pred"], base["pred"]):
        assert rb["truth"] == r0["truth"] and rb["truth_astat"] == r0["truth_astat"]            # truth side unchanged
        assert rb["perc_cont"]["hold"]["status"] == r0["perc"][0]
        assert rb["perc_cont"]["hold"]["astat"] == r0["perc_astat"]
        # combination rule applied per atom
        for nm, s in rb["perc_astat"].items():
            assert s == EC.combine_status([rb["perc_cont"][c]["astat"][nm] for c in ("hold", "brake")])
        assert rb["perc"][0] == EC.combine_status([rb["perc_cont"][c]["status"] for c in ("hold", "brake")])
    # conservative: nothing that was 'false'/'unknown' under hold becomes 'true'
    for rb, r0 in zip(both["pred"], base["pred"]):
        if r0["perc"][0] != "true":
            assert rb["perc"][0] != "true"


def _same(a, b):
    """equality tolerant to NaN (the margins of rows with a missing atom are nan)."""
    import pickle
    return pickle.dumps(a) == pickle.dumps(b)
