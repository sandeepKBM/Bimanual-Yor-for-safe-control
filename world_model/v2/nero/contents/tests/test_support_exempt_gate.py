"""Owner gate (11-17): USER "Release: exempt the support" -- once BOTH welds are off, the world geoms the load touches at that moment (the landing
support) are skipped by the obj_world U row; every other world geom still counts.  SIMULATION ONLY, CPU, no network.
PYTHONPATH=. python -m pytest -q world_model/v2/nero/contents/tests/test_support_exempt_gate.py

Checks: latched only at the LAST weld-off (from contacts); cleared on re-grasp; saved / restored with the snapshot; pairs with other world geoms
are still evaluated; BIMANUAL_LEGACY_NO_SUPPORT_EXEMPT=1 never latches."""
import numpy as np
import pytest

from world_model.v2.nero.contents import carry as CR
from world_model.v2.nero.contents import carry_cert as CC


def _ep(monkeypatch, legacy=False):
    monkeypatch.setenv("BIMANUAL_NO_NETWORK", "1")
    if legacy:
        monkeypatch.setenv("BIMANUAL_LEGACY_NO_SUPPORT_EXEMPT", "1")
    else:
        monkeypatch.delenv("BIMANUAL_LEGACY_NO_SUPPORT_EXEMPT", raising=False)
    w = CC.sample_world("box", (0.5, 1.0), np.random.default_rng(1117050000))
    ep = CR.CarryEpisode(w, a_plan=0.5)
    ep.set_belief(0.75, a_plan=0.5)
    ep.run_phases([("hold", 0.3)])                                   # box resting on the counter, both welds on
    return ep


def _name(ep, g):
    return ep.m.geom(g).name


def test_latch_at_last_weld_off_and_clear_on_regrasp(monkeypatch):
    ep = _ep(monkeypatch)
    assert ep._ow_support == frozenset()
    s0, s1 = CR.SIDES
    ep.weld(s0, False)
    assert ep._ow_support == frozenset(), "one weld still on: not released"
    ep.weld(s1, False)
    names = {_name(ep, g) for g in ep._ow_support}
    assert names == {"counter"}, names                               # the box rests on the counter at the start
    assert ep.support_exempt == ["counter"]
    ep.weld(s0, True)
    assert ep._ow_support == frozenset()


def test_snapshot_restore_carries_the_latch(monkeypatch):
    ep = _ep(monkeypatch)
    sn_held = ep.snapshot()
    for s in CR.SIDES:
        ep.weld(s, False)
    sn_released = ep.snapshot()
    assert ep._ow_support
    ep.restore(sn_held)
    assert ep._ow_support == frozenset()
    ep.restore(sn_released)
    assert {_name(ep, g) for g in ep._ow_support} == {"counter"}


def test_other_world_geoms_still_checked(monkeypatch):
    ep = _ep(monkeypatch)
    for s in CR.SIDES:
        ep.weld(s, False)
    sup = set(ep._ow_support)
    seen = []
    real = CR.mujoco.mj_geomDistance

    def spy(m, d, g1, g2, dmax, ft):
        seen.append(int(g1))
        return real(m, d, g1, g2, dmax, ft)
    monkeypatch.setattr(CR.mujoco, "mj_geomDistance", spy)
    monkeypatch.setattr(CR, "_OBJ_WORLD_DMAX", 10.0)                 # evaluate every pair (no bounding-sphere skip)
    ep._obj_world_check(ep.d.time)
    assert seen and not (set(seen) & sup), "support pairs must be skipped"
    world = set(int(w) for w in ep._ow_w) - sup
    assert set(seen) == world, (sorted(world - set(seen)))


def test_legacy_never_latches(monkeypatch):
    ep = _ep(monkeypatch, legacy=True)
    for s in CR.SIDES:
        ep.weld(s, False)
    assert ep._ow_support == frozenset() and getattr(ep, "support_exempt", None) is None
