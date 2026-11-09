"""ClosedChainPlanner tests.  SIMULATION ONLY, CPU.   PYTHONPATH=. python -m pytest -q world_model/v2/stack/tests/test_closed_chain.py
The first run plans the geometry (~40 s per task, SRS IK + DP) and caches it in stack/planners/plan_cache/ (BIMANUAL_PLAN_CACHE)."""
import numpy as np
import pytest

from world_model.v2.nero.contents import carry as CR
from world_model.v2.nero.contents import scenes as SC
from world_model.v2.stack.planners import closed_chain as CC
from world_model.v2.stack.planners.scripted import ScriptedPlanner
from world_model.v2.stack.types import SIDES, SkillCall

A = 0.5            # object acceleration cap used in the tests (envelope.A_NOM, ASSUMED)


@pytest.fixture(scope="module", params=["box", "tray6"])
def planned(request):
    inp = CC._nominal_inputs(request.param)
    geo = CC.plan_geometry(inp)
    return request.param, inp, geo, CC.ClosedChainPlan(geo, inp, A)


def test_reaches_goal_with_margin(planned):
    task, inp, geo, cc = planned
    for s in SIDES:
        assert geo.fk_pos[s].max() < 1e-6 and geo.fk_rot[s].max() < 1e-6          # IK miss 0, FK-validated at every sample
        assert np.all(geo.q[s] >= inp.arms[s].lo) and np.all(geo.q[s] <= inp.arms[s].hi)
        assert geo.margin[s].min() > 0.0
        assert np.allclose(geo.q[s][0], inp.q0[s], atol=1e-12)                      # starts exactly at the hold pose
        assert np.abs(np.diff(geo.q[s], axis=0)).max() <= CC.DQ_MAX + 1e-12          # continuous
    w = inp.world
    from world_model.v2.stack import tasks as TK
    gx, gy = TK.GOAL_DXY[inp.task]
    top_goal = SC.support_tops(w)["shelf" if task == "box" else "counter"]["top"]
    top_start = SC.support_tops(w)["counter"]["top"]
    assert np.allclose(geo.offs[-1], [gx, gy, top_goal - top_start])                # object ends on the goal support
    goal_xy = (w.obj_xy[0] + gx, w.obj_xy[1] + gy)
    from world_model.v2.stack import goal as GL                                       # tray (0.28,-0.22), box (0.18,-0.22)
    assert np.allclose(goal_xy, GL.goal_xy(task, w))
    if task == "box":                                                                # shelf clearance: above the shelf top + CLEAR_M before leaving x >= shelf edge
        assert geo.nodes[1][2] + SC.support_tops(w)["counter"]["top"] >= SC.support_tops(w)["shelf"]["top"] + CC.CLEAR_M["box"] - 1e-9
        assert CC._collides(inp, geo.nodes) is None
    # the old plan misses here (11-05): continuation IK error > 1e-3 m for the box; the planned path has none


def test_unreachable_goal_refused(monkeypatch):
    monkeypatch.setitem(CC.CLEAR_M, "box", 0.40)                                     # a 55 cm lift: no arm can hold the object level there
    inp = CC._nominal_inputs("box")
    with pytest.raises(CC.PlanRefused) as e:
        CC.plan_geometry(inp, use_cache=False)
    assert e.value.info.get("reason") in ("unreachable", "no in-limit solution at the sample", "disconnected", "collision")
    assert "rad outside a joint limit" in str(e.value) or e.value.info.get("reason") == "collision"      # the arm and the miss are named
    assert e.value.info.get("arm") in SIDES or e.value.info.get("reason") == "collision"


def test_toppra_respects_limits(planned):
    task, inp, geo, cc = planned
    v = CC.vlim()
    for sg, pc in cc.pieces.items():
        assert pc.qd_ratio <= 1.0 + 1e-6                                              # datasheet joint speeds
        assert np.abs(pc.pdd).max() <= A + 1e-3                                       # object acceleration cap, per axis, max over the trajectory
        for s in SIDES:
            assert np.all(np.abs(pc.qd[s]) <= v + 1e-6)
            assert np.allclose(pc.qd[s][0], 0, atol=1e-9) and np.allclose(pc.qd[s][-1], 0, atol=1e-9)    # rest to rest
        # finite-difference object acceleration of the sampled trajectory (5 ms) agrees with the TOPP-RA one
        a_fd = np.abs(np.diff(pc.p, 2, axis=0) / CC.DT ** 2).max()
        assert a_fd <= A + 0.05
        i0, i1 = geo.bounds[sg]
        assert np.allclose(pc.q["left"][0], geo.q["left"][i0], atol=1e-9) and np.allclose(pc.q["right"][-1], geo.q["right"][i1], atol=1e-9)


def test_retrace_returns_to_start_by_the_planned_path(planned):
    task, inp, geo, cc = planned
    pc = cc.pieces["lift"]
    k = pc.n // 2
    q_now = {s: pc.q[s][k] for s in SIDES}
    pieces = cc.retrace(q_now, ("lift",), CR.A_BACKUP)
    assert pieces
    for p in pieces:
        assert p.qd_ratio <= 1.0 + 1e-6 and np.abs(p.pdd).max() <= CR.A_BACKUP + 1e-3
    for s in SIDES:
        assert np.allclose(pieces[-1].q[s][-1], inp.q0[s], atol=1e-9)                # back at the hold pose
        assert np.allclose(pieces[0].q[s][0], q_now[s], atol=1e-6)


@pytest.mark.parametrize("seg,frac", [("carry", 0.3), ("carry", 1.0), ("place", 0.5), ("lift", 1.0), ("lift", 0.0)])
def test_retrace_from_any_segment(planned, seg, frac):
    task, inp, geo, cc = planned
    back = {"lift": ("lift",), "carry": ("carry", "lift"), "place": ("place", "carry", "lift")}[seg]
    pc = cc.pieces[seg]
    k = int(round(frac * pc.n))
    q_now = {s: pc.q[s][k] + 1e-4 for s in SIDES}                                   # a commanded q slightly off the planned path
    pieces = cc.retrace(q_now, back, CR.A_BACKUP)
    for p in pieces:
        assert p.qd_ratio <= 1.0 + 1e-6 and np.abs(p.pdd).max() <= CR.A_BACKUP + 1e-3
    for a, b in zip(pieces[:-1], pieces[1:]):                                        # consecutive pieces join (each is rest to rest)
        for s in SIDES:
            assert np.allclose(a.q[s][-1], b.q[s][0], atol=1e-6)
    last = pieces[-1].q if pieces else q_now
    for s in SIDES:
        assert np.allclose(last[s][-1] if pieces else last[s], inp.q0[s], atol=2e-4)


def test_retreat_planned(planned):
    task, inp, geo, cc = planned
    for where in ("goal", "start"):
        r = cc.retreat(where)
        assert r["qd_ratio"] <= 1.0 + 1e-6 and all(0 < r["d"][s] <= CC.RETREAT_M + 1e-9 for s in SIDES)
        i = -1 if where == "goal" else 0
        for s in SIDES:
            assert np.allclose(r["q"][s][0], geo.q[s][i], atol=1e-9)


def test_plan_structure(planned):
    task, inp, geo, cc = planned
    plan = CC.ClosedChainPlanner().plan(None, SkillCall("carry_task", dict(task=task, a_plan=A)), None, inputs=inp)
    assert [s.mode for s in plan.segments] == ["lift", "carry", "place", "release"]
    assert plan.planner == "closed_chain" and plan.provenance["legacy"] is False
    assert plan.segments[0].meta["backup_phases"][1] == ("retrace", ("lift",))
    assert plan.segments[3].meta["backup_phases"] is None
    q, qd, qdd = plan.segments[0].q_ref(0.0)["left"]
    assert q.shape == (7,) and np.allclose(qd, 0)
    assert "joint_accel" in plan.assumptions and plan.assumptions["joint_accel"]["tag"] == "OPEN"


def test_static_path_has_no_ik_miss(planned):
    task, inp, geo, cc = planned
    offs, per_seg, cap = CC.static_path(task)
    assert list(per_seg) == ["lift", "carry", "place"] and set(sum(per_seg.values(), [])) == set(offs)
    assert all(not any(k.endswith("_ik_err") for k in v) for v in cap["q"].values())
    assert np.allclose(offs[per_seg["lift"][0]], 0)


def test_default_path_unchanged(monkeypatch):
    monkeypatch.delenv("BIMANUAL_PLANNER", raising=False)
    assert CR.resolve_planner() == "scripted" and CR.resolve_planner("closed_chain") == "closed_chain"
    with pytest.raises(ValueError):
        CR.resolve_planner("nope")
    ep = CR.CarryEpisode(SC.TrayWorld(n_cups=2))
    assert ep.planner == "scripted" and ep.cc is None
    from world_model.v2.stack.planners import scripted as SP
    for seg in CR.SEGS:
        assert ep.segment_phases(seg) == SP.segment_phases("tray", seg, ep.a_plan)[0]
        assert ep.backup_phases(seg) == SP.backup_phases(seg)
    plan = ScriptedPlanner().plan(None, SkillCall("carry_task", dict(task="tray", a_plan=0.5)), None)
    assert plan.segments[0].meta["phases"][0][0] == "move" and "cc" not in plan.segments[0].meta


def test_episode_tracks_planned_lift_without_ik():
    ep = CR.CarryEpisode(SC.TrayWorld(n_cups=2), m_hat=0.0, planner="closed_chain")
    assert ep.planner == "closed_chain"
    ep.run_phases([("hold", 0.4)])
    ep.reset_track()
    ep.run_phases(ep.segment_phases("lift"))
    tr = ep.track
    assert tr["viol"] is None and tr["path_unreachable"] == 0 and tr["path_ik_err_max"] < 1e-6 and tr["path_q_margin_min"] > 0.0
    assert all(e.get("planned") for e in ep.path_log)
    assert ep.d.xpos[ep.obj][2] - ep.z0 == pytest.approx(0.05, abs=0.01)             # lifted 5 cm (tray)
