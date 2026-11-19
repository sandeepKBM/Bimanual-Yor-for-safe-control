"""Stop-finish gate (owner-written, 11-16): "a stop in the last T seconds of a move lets the move finish".  SIMULATION ONLY.

Why: the backup's stop phase is a Hermite curve q -> q + 0.5 v T (constant deceleration v / T, T = 0.3 s).  Every move is rest to rest, so a stop in
the last ~T seconds of a move ended PAST the move's end knot (worst excess a T^2 / 8: 5.6 mm at a = 0.5, measured on the box closed-chain plan, every
segment, every a).  Rule 1 modelled that overshoot as MISSING past the last sample, which refused the box place step in every cell, and as "continuing
along the next segment" at corners, where the executor actually continues straight in joint space.

What is checked here:
  1. the executor (stack/executor.py "stop") finishes the move when its remaining time <= T, and keeps the Hermite stop otherwise (fake episode, no physics);
  2. the Rule 1 path model marks exactly the branches in the last T seconds of each segment as finish, ending at the segment's last sample, at the
     plan's acceleration, and no branch is MISSING any more on the box plan;
  3. the executor and the Rule 1 model take the SAME decision at every Rule 1 branch time of the box plan (the executor run on the planned joint
     references of each segment).
"""
from __future__ import annotations

import numpy as np
import pytest

from world_model.v2.nero.contents import carry as CR
from world_model.v2.stack import executor as EX

DT = CR.DT
T_STOP = 0.3
SIDES = ("left", "right")


class FakeEp:
    """the parts of CarryEpisode the executor's stop phase touches: the reference paths, _cmd / _trim (the real methods), k, _step."""
    _cmd = CR.CarryEpisode._cmd
    _trim = CR.CarryEpisode._trim

    def __init__(self, paths, t):
        self.q0 = {s: np.zeros_like(np.asarray(paths[s][0][2][0], float)) for s in SIDES}
        self.paths = {s: list(paths[s]) for s in SIDES}
        self.k = int(round(t / DT))

    def _step(self):
        self.k += 1


def smoothstep_move(Q, D, n=401):
    tau = np.linspace(0, 1, n)
    return [Q * (3 * x ** 2 - 2 * x ** 3) for x in tau]


def ep_with_move(t_stop, D=1.0):
    Q = np.array([0.4, -0.2, 0.1, 0.0, 0.3, -0.1, 0.2])
    wp = smoothstep_move(Q, D)
    return FakeEp({s: [(0.0, D, wp)] for s in SIDES}, t_stop), Q


def test_stop_in_last_T_finishes_the_move(monkeypatch):
    ep, Q = ep_with_move(0.8)                                   # 0.2 s left <= T
    before = {s: list(ep.paths[s]) for s in SIDES}
    EX.run_phases(ep, [("stop", T_STOP)])
    assert ep.stop_log[-1]["mode"] == "finish", ep.stop_log
    for s in SIDES:
        assert ep.paths[s] == before[s]                         # nothing trimmed, nothing appended
        assert np.allclose(ep._cmd(s, ep.k * DT), Q)            # at rest on the move's end knot
        ts = np.arange(0.8, ep.k * DT + 1e-9, DT)
        prog = [float(ep._cmd(s, t) @ Q / (Q @ Q)) for t in ts]
        assert max(prog) <= 1.0 + 1e-12                         # never past the end
    assert ep.k == int(round(0.8 / DT)) + int(round(T_STOP / DT))   # the phase still lasts T


def test_stop_earlier_keeps_the_hermite_stop(monkeypatch):
    ep, Q = ep_with_move(0.4)                                   # 0.6 s left > T
    q = ep._cmd("left", 0.4)
    v = (ep._cmd("left", 0.4 + 1e-3) - q) / 1e-3
    EX.run_phases(ep, [("stop", T_STOP)])
    assert ep.stop_log[-1]["mode"] == "hermite", ep.stop_log
    end = ep._cmd("left", ep.k * DT)
    assert np.allclose(end, q + 0.5 * v * T_STOP, atol=1e-9)
    assert float(end @ Q / (Q @ Q)) < 1.0                       # a smoothstep move: the Hermite end stays before the knot when > T is left


# ----------------------------------------------------------------------------------------------------------------------- Rule 1 model on the box plan
def _box_model(cell=(1.0, 1.5)):
    from world_model.v2.adapters import rule1_contents as RC
    ctx = RC.Ctx("box", "closed_chain")                         # fresh: models are cached per Ctx
    lo, hi = RC.cell_total("box", cell)
    adm, model = RC.plan_admission(ctx, cell, lo, hi)
    assert model is not None, adm.action
    return ctx, model


def test_rule1_model_finishes_exactly_the_last_T_branches(monkeypatch):
    ctx, model = _box_model()
    g = model.geo
    n_finish = {}
    for seg in model.names:
        Tseg, sT, (i0, i1) = model.T[seg], ctx._stop_T(seg), g.bounds[seg]
        n_finish[seg] = 0
        for br in model.branches[seg]:
            assert br.ahead is not None, (seg, br.t)            # no MISSING branch on the box plan any more
            in_zone = br.v > 1e-9 and (Tseg - br.t) <= sT + 1e-9
            assert bool(getattr(br, "finish", False)) == in_zone, (seg, br.t, Tseg)
            if in_zone:
                n_finish[seg] += 1
                assert br.idx_e == i1, (seg, br.t, br.idx_e, i1)                  # ends on the segment's last sample
                assert br.a_stop == pytest.approx(model.a)                         # at the plan's own acceleration
                assert i0 <= br.idx_b <= i1
    assert n_finish["place"] >= 1, n_finish


def test_executor_and_rule1_take_the_same_decision_on_the_box_plan(monkeypatch):
    """the executor's stop decision, run on each segment's PLANNED joint reference at every Rule 1 branch time, equals the Rule 1 model's finish flag."""
    ctx, model = _box_model()
    for seg in model.names:
        pc = model.cc.segment(seg)
        dur = pc.n * DT
        sT = ctx._stop_T(seg)
        for br in model.branches[seg]:
            if br.v <= 1e-9:
                continue
            ep = FakeEp({s: [(0.0, dur, list(np.asarray(pc.q[s])))] for s in SIDES}, br.t)
            EX.run_phases(ep, [("stop", sT)])
            assert (ep.stop_log[-1]["mode"] == "finish") == bool(br.finish), (seg, br.t, dur, ep.stop_log[-1])


def test_one_stop_time_source_is_read_at_call_time(monkeypatch):
    """STOP_T_S (stack/types.py, ASSUMED 0.3 s) is the one source: a patch reaches both planners' backup programs, Rule 1 and Rule 4 (the sweep relies on it)."""
    from world_model.v2.adapters import rule1_contents as RC
    from world_model.v2.adapters import rule4_contents as R4
    from world_model.v2.stack import types as TY
    from world_model.v2.stack.planners import closed_chain as CCP
    from world_model.v2.stack.planners import scripted as SP
    assert TY.STOP_T_S == 0.3
    monkeypatch.setattr(TY, "STOP_T_S", 0.45)
    for seg in ("lift", "carry", "place"):
        assert CCP.backup_phases_for(seg)[0] == ("stop", 0.45), seg
        assert SP.backup_phases(seg)[0] == ("stop", 0.45), seg
    assert RC.MotionRef("stop_hold", "lift").stop_T == 0.45
    assert R4._stop_t0() == 0.45
    ctx = RC.Ctx("box", "closed_chain")
    assert all(ctx._stop_T(s) == 0.45 for s in ("lift", "carry", "place"))
