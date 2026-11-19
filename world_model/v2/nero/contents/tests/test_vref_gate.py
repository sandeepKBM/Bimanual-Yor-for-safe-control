"""Owner gate (11-17): USER "Velocity reference in tracker" -- FL-MPC gets d/dt of the planned joint reference over the horizon (rad/s), not zeros.
SIMULATION ONLY, CPU, no network.  PYTHONPATH=. python -m pytest -q world_model/v2/nero/contents/tests/test_vref_gate.py
Checks: at a hold the velocity reference is exactly 0; during a planned move it equals np.gradient(reference, FDT) and is non-zero; the vector passed
to fl.step is that array; BIMANUAL_LEGACY_ZERO_VREF=1 passes zeros.  (Bit-identity of the legacy flag vs the pre-edit carry.py was checked once
by the owner on 11-17: lift, box cell 2.0-2.5 world 0, 339 steps, same qpos SHA256 2ea4cad9...)"""
from types import SimpleNamespace

import numpy as np

from world_model.v2.nero.contents import carry as CR
from world_model.v2.nero.contents import carry_cert as CC
from world_model.v2.stack.planners.closed_chain import ClosedChainPlanner
from world_model.v2.stack.types import SkillCall


def _ep(monkeypatch, legacy):
    monkeypatch.setenv("BIMANUAL_NO_NETWORK", "1")
    monkeypatch.setenv("BIMANUAL_PLANNER", "closed_chain")
    if legacy:
        monkeypatch.setenv("BIMANUAL_LEGACY_ZERO_VREF", "1")
    else:
        monkeypatch.delenv("BIMANUAL_LEGACY_ZERO_VREF", raising=False)
    w = CC.sample_world("box", (2.0, 2.5), np.random.default_rng(1117200000))
    ep = CR.CarryEpisode(w, a_plan=0.5, planner="closed_chain")
    ep.set_belief(2.25, a_plan=0.5)
    seen = []
    real = ep.ctl.fl.step

    def spy(m, d, lq, ld, rq, rd, qe_ref, qe_v, qi_ref, qi_v, *a, **k):
        seen.append((np.array(qe_ref), np.array(qe_v), np.array(qi_ref), np.array(qi_v)))
        return real(m, d, lq, ld, rq, rd, qe_ref, qe_v, qi_ref, qi_v, *a, **k)
    ep.ctl.fl.step = spy
    return ep, seen


def _lift(ep):
    plan = ClosedChainPlanner().plan(None, SkillCall("carry_task", dict(task="box", a_plan=ep.a_plan)), SimpleNamespace(episode=ep))
    sg = plan.segments[0]
    ep.store(sg.mode)
    ep.reset_track()
    if "cc" in sg.meta:
        ep.cc = sg.meta["cc"]
    ep.last_accel = sg.meta["accel"]
    ep.run_phases(sg.meta["phases"])


def test_hold_zero_and_move_is_gradient(monkeypatch):
    ep, seen = _ep(monkeypatch, legacy=False)
    ep.run_phases([("hold", 0.1)])
    assert seen and all(np.all(v == 0.0) and np.all(w == 0.0) for _, v, _, w in seen), "hold must give an exact zero velocity reference"
    n0 = len(seen)
    _lift(ep)
    moved = 0
    for qe, ve, qi, vi in seen[n0:]:
        np.testing.assert_allclose(ve, np.gradient(qe, ep.FDT, axis=0), rtol=0, atol=1e-12)
        np.testing.assert_allclose(vi, np.gradient(qi, ep.FDT, axis=0), rtol=0, atol=1e-12)
        moved += int(np.abs(ve).max() > 1e-6)
    assert moved > 0, "a planned move must give a non-zero velocity reference"


def test_legacy_passes_zeros(monkeypatch):
    ep, seen = _ep(monkeypatch, legacy=True)
    ep.run_phases([("hold", 0.05)])
    _lift(ep)
    assert seen and all(np.all(v == 0.0) and np.all(w == 0.0) for _, v, _, w in seen)
