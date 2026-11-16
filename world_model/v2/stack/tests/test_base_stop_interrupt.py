"""F2 (11-13): base_stop recovery, execute-and-continue (executor interrupt), continuation route.  SIMULATION ONLY.  Pure tests first; the MuJoCo ones
build the swerve scene (tray6) and run a ~2 s base line."""
import copy
import math

import numpy as np
import pytest

from world_model.v2.nero.contents import carry as CR, scenes as SC
from world_model.v2.stack import executor as EX
from world_model.v2.stack.planners import base_route as BR
from world_model.v2.stack.planners import closed_chain as CCP
from world_model.v2.stack.types import BimanualPlan, Segment


def test_continuation_route_pure():
    assert BR.continuation_route((0, 0, 0), (0, 0, 0)) == ()
    r = BR.continuation_route((0.0, 0.0, 0.0), (-0.3, 0.0, math.pi))                      # straight behind: drive backwards, then turn 180
    assert [k for k, _ in r] == ["line", "turn"] and r[0][1] == pytest.approx(-0.3) and abs(r[1][1]) == pytest.approx(180.0)
    end = BR.route_end((0.0, 0.0, 0.0), r)
    assert np.allclose(end[:2], (-0.3, 0.0), atol=1e-9) and abs(math.remainder(end[2] - math.pi, 2 * math.pi)) < 1e-9
    r = BR.continuation_route((1.0, 1.0, 0.3), (0.2, 0.4, -2.0))                          # far: chunks <= 1 m, <= 4 segments
    assert all(abs(v) <= 1.0 + 1e-9 for k, v in r if k == "line") and len(r) <= BR.MAX_SEGMENTS
    end = BR.route_end((1.0, 1.0, 0.3), r)
    assert np.allclose(end[:2], (0.2, 0.4), atol=1e-9) and abs(math.remainder(end[2] + 2.0, 2 * math.pi)) < 1e-9


def test_flag_off_base_segment_has_no_backup(monkeypatch):
    monkeypatch.delenv("BIMANUAL_BASE_BACKUP", raising=False)
    inp = CCP._nominal_inputs("tray6_behind")
    pl = CCP.ClosedChainPlanner().plan(None, CCP.SkillCall("plan", {"task": "tray6_behind", "a_plan": 0.5}), inputs=inp)
    base_off = pl.segments[1].meta["backup_phases"]
    others_off = [s.meta["backup_phases"] for s in pl.segments if s.meta.get("base_ref") is None]
    assert base_off is None
    monkeypatch.setenv("BIMANUAL_BASE_BACKUP", "1")
    pl2 = CCP.ClosedChainPlanner().plan(None, CCP.SkillCall("plan", {"task": "tray6_behind", "a_plan": 0.5}), inputs=inp)
    assert pl2.segments[1].meta["backup_phases"] == [("base_stop", CCP.BASE_STOP_MAX_S), ("hold", 0.5)]
    assert [s.meta["backup_phases"] for s in pl2.segments if s.meta.get("base_ref") is None] == others_off      # nothing else changes
    assert [s.meta["phases"] for s in pl.segments] == [s.meta["phases"] for s in pl2.segments]


def _ep():
    e = CR.CarryEpisode(SC.TrayWorld(n_cups=6), base="swerve", goal_support="counter2")
    e.set_belief(float(e.truth0["contents_mass"]))
    e.run_phases([("hold", 0.4), ("move", (0, 0, 0.05), 1.0), ("hold", 0.5)])
    return e


def _plan(backup=True):
    ref = dict(route=[["line", -0.6]], level="L2")
    meta = dict(phases=[("base_move", ref)], base_ref=ref, accel=dict(source="test"), backup_phases=[("base_stop", 4.0), ("hold", 0.3)] if backup else None,
                end_pose=None)
    return BimanualPlan(segments=[Segment(mode="carry", t0=0.0, t1=3.0, meta=meta)], planner="test", provenance={}, assumptions={})


def _nom(rec):
    return copy.deepcopy(rec["segments"][0]["nominal"]), copy.deepcopy(rec["segments"][0]["family"])


def test_base_stop_lands_at_rest_and_interrupt_leaves_nominal_unchanged():
    e1 = _ep()
    rec_off = EX.run_segments(e1, _plan(), dict(segments=[]))
    assert "interrupt" not in rec_off
    e2 = _ep()
    p0 = e2.sb.pose().copy()
    rec_on = EX.run_segments(e2, _plan(), dict(segments=[]), interrupt=dict(seg=0, t=0.75))
    assert _nom(rec_on) == _nom(rec_off)                                                    # the interrupt runs AFTER the nominal record
    it = rec_on["interrupt"]
    assert it["t_branch"] == pytest.approx(0.75) and it["backup"] == ["base_stop", "hold"] and "resume" not in it
    ln = it["landing"]
    assert ln["recovery_s"] < 4.5 and ln["viol"] is None or ln["viol"]["kind"] != "base_stop_timeout"
    assert e2.base_stop_log[-1]["at_rest"] and e2.base_stop_log[-1]["v_mps"] < CR.CarryEpisode.BASE_STOP_TOL_MPS
    moved = float(np.linalg.norm(np.array(ln["base_pose"][:2]) - p0[:2]))
    assert 0.0 < moved < 0.55                                                               # stopped well short of the 0.6 m route
    assert ln["base_pose"] != rec_off["segments"][0]["nominal"]  # (landing is a dict of its own)
    assert set(ln) >= {"in_R", "R_fail", "base_pose", "obj_truth", "viol_all", "recovery_s"}


def test_interrupt_without_backup_is_recorded_not_run():
    rec = EX.run_segments(_ep(), _plan(backup=False), dict(segments=[]), interrupt=dict(seg="carry", t=0.5))
    assert "error" in rec["interrupt"] and "landing" not in rec["interrupt"]


def test_resume_refused_without_nominal_end_pose():
    rec = EX.run_segments(_ep(), _plan(), dict(segments=[]), interrupt=dict(seg=0, t=0.75, resume=True))
    assert rec["interrupt"]["resume"]["done"] is False and len(rec["segments"]) == 1
