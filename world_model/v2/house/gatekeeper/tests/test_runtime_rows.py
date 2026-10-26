"""Opt-in "planner owns safety" additions (user decisions 2026-10-24..26): GKConfig.runtime_rows ("all" | "cup") and person_model="pred" with the plan-time
separating-motion exit rule.  CPU only, no MuJoCo.  python -m pytest -q world_model/v2/house/gatekeeper/tests/test_runtime_rows.py"""
import dataclasses

import numpy as np
import pytest

from world_model.v2.house.teleop import geom as GM, shield as SH
from world_model.v2.house.gatekeeper import GKConfig, Gatekeeper, plan_candidate, check1, check2
from world_model.v2.house.gatekeeper import gate as GATE
from world_model.v2.house.gatekeeper.monitor import falsification_monitor
from world_model.v2.house.gatekeeper.tests.test_gatekeeper import FakeSim, bel, GK

CUP = dataclasses.replace(GK, runtime_rows="cup")
FX, FY = GM.FP["x"], GM.FP["y"]


# ------------------------------------------------------------------------------------------------ defaults are bit-identical
# (name, belief kwargs, v0, u, T_S, person_model) and the check1/check2 outputs of the code BEFORE this change (computed from a pristine copy of the old module)
FIXTURES = [
    ("free_hold", dict(), np.zeros(3), np.array([0.9, 0, 0]), 1.0, "speed"),
    ("wall_near", dict(pose_wall=np.array([4.2, 5.0, 0.0]), pose_people=np.array([4.2, 5.0, 0.0])), np.array([0.9, 0, 0]), np.array([0.9, 0, 0]), 0.3, "speed"),
    ("person_speed", dict(people=(np.array([3.4, 5.6]),)), np.zeros(3), np.array([0.6, 0.1, 0.2]), 0.6, "speed"),
    ("person_accel", dict(people=(np.array([3.4, 5.6]),), people_vel=(np.array([-0.3, 0.1]),)), np.zeros(3), np.array([0.6, 0.1, 0.2]), 0.6, "accel"),
]
GOLDEN = {
    "free_hold": ((True, 0.0, 'speed_cap', {'hw_slew': 2.7667652799999995, 'speed_cap': 0.0, 'cup': 0.03814127999999961, 'wall': 1.1619169132}),
                  (True, 1e-06, 'rest_speed', {'rest_speed': 1e-06, 'wall_at_rest': 1.9249999999999998})),
    "wall_near": ((False, -0.1375, 'wall', {'hw_slew': 2.7667652799999995, 'speed_cap': 0.0, 'cup': 0.03814127999999961, 'wall': -0.1375}),
                  (False, -0.115, 'wall_at_rest', {'rest_speed': 1e-06, 'wall_at_rest': -0.115})),
    "person_speed": ((False, -0.9061079411749613, 'person', {'hw_slew': 2.882594125164983, 'speed_cap': 0.3606601433138415, 'cup': 0.03814127999999961, 'wall': 1.1606353611375908, 'person': -0.9061079411749613}),
                     (False, -0.5610304014956814, 'passive_person', {'rest_speed': 1e-06, 'wall_at_rest': 1.445, 'passive_person': -0.5610304014956814})),
    "person_accel": ((False, -0.9061079411749613, 'person', {'hw_slew': 2.882594125164983, 'speed_cap': 0.3606601433138415, 'cup': 0.03814127999999961, 'wall': 1.1606353611375908, 'person': -0.9061079411749613}),
                     (False, -0.42265828579486786, 'passive_person', {'rest_speed': 1e-06, 'wall_at_rest': 1.445, 'passive_person': -0.42265828579486786})),
}


def _build(kw, v0, u, T_S, model):
    b = bel(person_model=model, **kw)
    gk = dataclasses.replace(GK, person_model=model)
    return plan_candidate(v0, u, T_S, b, gk), b, gk


def test_new_config_fields_default_to_the_old_behaviour():
    c = GKConfig()
    assert c.runtime_rows == "all" and c.plan_cert == "off" and c.person_model == "speed" and (c.e_v, c.a_p) == (0.2, 2.0)
    assert Gatekeeper().gk.runtime_rows == "all"
    with pytest.raises(AssertionError):
        Gatekeeper(gk=GKConfig(runtime_rows="walls"))


@pytest.mark.parametrize("name,kw,v0,u,T_S,model", FIXTURES, ids=[f[0] for f in FIXTURES])
def test_default_checks_equal_the_pre_change_golden_values_and_explicit_all(name, kw, v0, u, T_S, model):
    tr, b, gk = _build(kw, v0, u, T_S, model)
    g1, g2 = GOLDEN[name]
    assert check1(tr, b, gk) == g1 and check2(tr, b, gk) == g2
    gk_all = dataclasses.replace(gk, runtime_rows="all")
    assert check1(tr, b, gk_all) == g1 and check2(tr, b, gk_all) == g2


# ------------------------------------------------------------------------------------------------ cup mode: rows
@pytest.mark.parametrize("name,kw,v0,u,T_S,model", FIXTURES, ids=[f[0] for f in FIXTURES])
def test_cup_mode_evaluates_only_cup_and_hw_rows(name, kw, v0, u, T_S, model):
    tr, b, gk = _build(kw, v0, u, T_S, model)
    gk = dataclasses.replace(gk, runtime_rows="cup")
    o1, m1, r1, g1 = check1(tr, b, gk)
    o2, m2, r2, g2 = check2(tr, b, gk)
    assert set(g1) == {"hw_slew", "speed_cap", "cup"} and set(g2) == {"rest_speed"}
    assert o1 and o2                                                       # a cup-safe plan: walls / people do not matter here (wall_near and person_* FAIL in "all")
    full = GOLDEN[name]
    for k in g1:                                                           # the surviving rows are the same numbers as in "all"
        assert g1[k] == full[0][3][k]


def test_cup_mode_still_enforces_the_cup_row_and_the_rest_condition():
    tr = plan_candidate(np.zeros(3), np.array([0.9, 0, 0]), 1.0, bel(mu_lo=0.35), GK)
    o, m, why, _ = check1(tr, bel(mu_lo=0.05), CUP)                         # planned for a grippy cup, run on a slippery one
    assert (not o) and why == "cup" and m < 0
    from world_model.v2.house.gatekeeper.traj import Traj
    moving = Traj(np.array([[0.0, 0, 0], [0.2, 0, 0], [0.2, 0, 0]]), 0.05)
    o2, _, why2, _ = check2(moving, bel(), CUP)
    assert (not o2) and why2 == "no_rest"


# ------------------------------------------------------------------------------------------------ cup mode: no Shield hand-off
def _no_shield_base(monkeypatch):
    calls = []

    def boom(self, sim, u_op):
        calls.append(1)
        raise AssertionError("internal Shield.filter_base entered")
    monkeypatch.setattr(SH.Shield, "filter_base", boom)
    return calls


def _nothing_passes(monkeypatch):
    """model falsified situation: no candidate (and no kept trajectory) is certified -> the branch that used to enter regime 2."""
    monkeypatch.setattr(GATE, "check1", lambda tr, b, gk, tol_geo=None: (False, -1.0, "cup", {"cup": -1.0}))
    monkeypatch.setattr(GATE, "check2", lambda tr, b, gk, tol_geo=None: (False, -1.0, "rest_speed", {"rest_speed": -1.0}))


@pytest.mark.parametrize("keep_rule", ["monitor", "recertify"])
def test_cup_mode_never_enters_regime2_and_commits_the_cup_safe_brake(monkeypatch, keep_rule):
    g = Gatekeeper(cfg=SH.ShieldCfg(use_ir_veto=False), gk=GKConfig(runtime_rows="cup", keep_rule=keep_rule))
    sim = FakeSim((1.5, 5.0, 0.0))
    u = np.array([0.9, 0, 0])
    g.gk_step(sim, u, 0.0)                                                 # a real commit first
    assert g.mode == "commit"
    for _ in range(5):
        sim.step(g.traj.u_at(0.1))
    calls = _no_shield_base(monkeypatch)
    _nothing_passes(monkeypatch)
    sim.base.x += 0.3 if keep_rule == "monitor" else 0.0                   # monitor rule: pose falsification; recertify rule: the kept trajectory fails the (patched) checks
    g.gk_step(sim, u, 0.1)
    assert g.mode == "cup_brake" and g.n_falsified == 1
    assert g.falsify_reasons == ({"pose_xy": 1} if keep_rule == "monitor" else {"no_commit_recertify": 1})
    assert g.traj.t_sw <= 0.0 and g.traj.i_rest is not None                # pure cup-safe backup (traj.backup_from) from the current twist
    for k in range(60):                                                    # drive it: the base comes to rest, regime 2 is never ticked
        v, lab = g.filter_base(sim, u)
        g.n_calls += 1
        sim.step(v)
    assert lab.startswith("gk_") and g.n_regime2_ticks == 0 and not calls
    assert float(np.abs(sim.base.v).max()) < 1e-3
    assert g.counters()["n_regime2_ticks"] == 0


def test_all_mode_still_enters_regime2_in_the_same_situation(monkeypatch):
    g = Gatekeeper(cfg=SH.ShieldCfg(use_ir_veto=False), gk=GKConfig(keep_rule="monitor"))
    sim = FakeSim((1.5, 5.0, 0.0))
    u = np.array([0.9, 0, 0])
    g.gk_step(sim, u, 0.0)
    for _ in range(5):
        sim.step(g.traj.u_at(0.1))
    _nothing_passes(monkeypatch)
    sim.base.x += 0.3
    g.gk_step(sim, u, 0.1)
    assert g.mode == "regime2" and g.traj is None and g.n_falsified == 1


def test_cup_mode_monitor_has_no_people_part_but_keeps_the_pose_part():
    res = {}
    for rr in ("all", "cup"):
        g = Gatekeeper(cfg=SH.ShieldCfg(use_ir_veto=False), gk=GKConfig(keep_rule="monitor", runtime_rows=rr))
        sim = FakeSim((1.5, 5.0, 0.0), people=[(6.0, 9.0)])
        g.gk_step(sim, np.array([0.9, 0, 0]), 0.0)
        for _ in range(3):
            sim.step(g.traj.u_at(0.1))
        rem = g.traj.shifted(0.1)
        near = ([np.array([sim.base.x + 1.2, sim.base.y])], [0.0])         # a person that appears in front of the moving robot
        res[rr] = falsification_monitor(g._cb[0], g._cb[1], g._cb[2], rem, g.last_bel, near, sim.pose(), 0.1, g.gk)
        sim.base.x += 0.3
        res[rr + "_pose"] = falsification_monitor(g._cb[0], g._cb[1], g._cb[2], rem, g.last_bel, ([], []), sim.pose(), 0.1, g.gk)
    assert res["all"] == (False, "person_unmatched_near") and res["cup"] == (True, "ok")
    assert res["all_pose"] == res["cup_pose"] == (False, "pose_xy")


# ------------------------------------------------------------------------------------------------ person_model="pred"
def _pred_case(person, u, T_S, ev=0.0, ap=0.0, pose=(1.5, 5.0, 0.0), vel=(0.0, 0.0), model="pred"):
    b = bel(people=(np.asarray(person, float),), people_vel=(np.asarray(vel, float),), person_model=model, e_v=ev, a_p=ap,
            pose_wall=np.array(pose), pose_people=np.array(pose))
    tr = plan_candidate(np.zeros(3), np.asarray(u, float), T_S, b, GK)
    o1, m1, r1, g1 = check1(tr, b, GK)
    o2, m2, r2, g2 = check2(tr, b, GK)
    return o1 and o2, {**g1, **g2}, (r1, r2)


P_INSIDE = (1.5 + FX[1] + 0.15, 5.0)           # stationary person 0.15 m in front of the footprint: inside the 0.3 + 0.1 keep-out


@pytest.mark.parametrize("ev,ap", [(0.0, 0.0), (0.2, 2.0)])
def test_pred_exit_rule_moving_away_from_a_stationary_person_inside_the_keepout_passes(ev, ap):
    ok, mg, _ = _pred_case(P_INSIDE, (-0.3, 0, 0), 1.0, ev, ap)
    assert ok and mg["person_exit"] >= 0.0 and mg["rest_person"] >= 0.0
    # the SAME plan under the default (speed) model is rejected: it asks for 1.5 m/s reachable discs
    ok_s, mg_s, _ = _pred_case(P_INSIDE, (-0.3, 0, 0), 1.0, model="speed")
    assert not ok_s and mg_s["person"] < 0 and "person_exit" not in mg_s


@pytest.mark.parametrize("ev,ap", [(0.0, 0.0), (0.2, 2.0)])
def test_pred_exit_rule_moving_toward_the_person_fails(ev, ap):
    ok, mg, reasons = _pred_case(P_INSIDE, (0.3, 0, 0), 1.0, ev, ap)
    assert not ok and mg["person_exit"] < 0.0 and mg["rest_person"] < 0.0
    assert "person_exit" in reasons


def test_pred_exit_rule_exit_margin_is_the_smallest_distance_step():
    ok, mg, _ = _pred_case(P_INSIDE, (-0.3, 0, 0), 1.0)
    b = bel(people=(np.asarray(P_INSIDE, float),), people_vel=(np.zeros(2),), person_model="pred", e_v=0.0, a_p=0.0)
    tr = plan_candidate(np.zeros(3), np.array([-0.3, 0, 0]), 1.0, b, GK)
    from world_model.v2.house.gatekeeper.checks import _pred_series, world_poses, _slack, _rect_far
    n_mov = tr.i_rest
    Pp = world_poses(tr.poses()[:n_mov], b.pose_people)
    vfar, _ = _slack(tr, _rect_far(b.fp))
    m, d = _pred_series(b, GK, 0, np.asarray(P_INSIDE, float), Pp, np.arange(n_mov) * tr.h, vfar[:n_mov], tr.h)
    assert m[0] < 0 and (m >= 0).any()                                  # starts inside, clears during the retreat
    k1 = int(np.argmax(m >= 0))
    assert mg["person_exit"] == pytest.approx(float(np.diff(d[:k1 + 1]).min()), abs=1e-12)
    assert (np.diff(d) >= -1e-12).all() and ok                          # a straight retreat never decreases D


def test_pred_passing_a_person_inside_the_keepout_fails_and_outside_passes():
    ok_near, mg_near, _ = _pred_case((1.5 + 2.0, 5.0 + FY[1] + 0.2), (0.3, 0, 0), 4.0)        # starts clear, drives past at 0.2 m lateral gap < 0.4
    ok_far, mg_far, _ = _pred_case((1.5 + 2.0, 5.0 + FY[1] + 0.9), (0.3, 0, 0), 4.0)
    assert not ok_near and mg_near["person"] < 0 and "person_exit" not in mg_near            # started clear -> the plain row, no exit rule
    assert ok_far and mg_far["person"] > 0.4


def test_pred_row_is_the_prediction_disc_only_not_the_speed_disc_and_not_max():
    """a person walking AWAY at 0.5 m/s: pred (c + v t) is far; the speed disc (1.5 m/s) would reject; 'accel' = max(speed, accel) accepts too, pred == accel's second disc."""
    kw = dict(person=(1.5 + FX[1] + 0.9, 5.0), u=(0.3, 0, 0), T_S=1.0, vel=(0.5, 0.0))
    ok_p, mg_p, _ = _pred_case(model="pred", **kw)
    ok_a, mg_a, _ = _pred_case(model="accel", **kw)
    ok_s, mg_s, _ = _pred_case(model="speed", **kw)
    assert ok_p and not ok_s and mg_p["person"] > 0 > mg_s["person"]
    assert mg_p["person"] == mg_a["person"]                                # the accel max picks the prediction disc here, pred is exactly that disc
    # stationary person, big growth: pred is the strictest (no speed-disc safety net is needed -- it is the model); accel >= pred
    kw2 = dict(person=(1.5 + FX[1] + 1.0, 5.0), u=(0.3, 0, 0), T_S=1.0)
    _, mp, _ = _pred_case(model="pred", ev=0.2, ap=2.0, **kw2)
    _, ma, _ = _pred_case(model="accel", ev=0.2, ap=2.0, **kw2)
    assert ma["person"] >= mp["person"]


def test_pred_growth_uses_e_v_a_p_and_the_stale_time():
    base = dict(person=(1.5 + FX[1] + 1.2, 5.0), u=(0.3, 0, 0), T_S=1.0)
    m0 = _pred_case(ev=0.0, ap=0.0, **base)[1]["person"]
    m1 = _pred_case(ev=0.2, ap=0.0, **base)[1]["person"]
    m2 = _pred_case(ev=0.2, ap=2.0, **base)[1]["person"]
    assert m0 > m1 > m2                                                    # monotone in e_v and a_p
    b = bel(people=(np.array([1.5 + FX[1] + 3.0, 5.0]),), people_vel=(np.zeros(2),), person_model="pred", e_v=0.2, a_p=0.5)
    tr = plan_candidate(np.zeros(3), np.array([0.3, 0, 0]), 1.0, b, GK)
    m_now = check1(tr, b, GK)[3]["person"]
    m_late = check1(tr, b.with_(people_stale=(0.5,)), GK)[3]["person"]     # the observation is 0.5 s old at the trajectory start: the disc is wider
    assert 0 < m_late < m_now


def test_pred_cup_mode_has_no_person_keys_and_pred_defaults_leave_other_models_untouched():
    b = bel(people=(np.asarray(P_INSIDE, float),), person_model="pred")
    tr = plan_candidate(np.zeros(3), np.array([0.3, 0, 0]), 1.0, b, GK)
    mg = {**check1(tr, b, CUP)[3], **check2(tr, b, CUP)[3]}
    assert "person" not in mg and "person_exit" not in mg and "rest_person" not in mg and "wall" not in mg
