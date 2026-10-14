"""people-safety v2 unit tests (flag-gated features; defaults must stay the legacy behaviour).  python -m pytest -q world_model/v2/house/teleop/tests/test_people_v2.py"""
from types import SimpleNamespace

import numpy as np

from world_model.v2.house.teleop import shield as SH, scene as SC
from world_model.v2.house.teleop.sim import TeleopSim, PersonPolicy, Cmd, TICK


def _walker_x(clock, n=160, **kw):
    sim = TeleopSim("h5", grasped=True, seed=0, with_people=True, people_clock=clock, **kw)
    names = [p.name for p in sim.scene.people]
    for _ in range(n):
        sim.tick(Cmd())
    return sim.people_state[names.index("p_walker")][0].copy()


def test_walker_clock_legacy_frozen_time_moves():
    x0 = np.array([3.6, 3.8])
    assert np.allclose(_walker_x("legacy"), x0)                    # legacy bug: a track starting with a stationary segment never moves
    assert _walker_x("time")[0] < 3.6 - 0.1                        # time clock: walks (starts at t = 2 s, 0.34 m/s)


def test_person_policy_defaults_and_oblivious():
    assert PersonPolicy().yield_m == 0.15 and PersonPolicy().giveup_s is None
    assert PersonPolicy(yield_m=None).yield_m is None


def _fake(pos, vel, pose=(0.0, 0.0, 0.0)):
    sim = SimpleNamespace()
    sim.pose = lambda: np.array(pose)
    sim.people_state = [(np.asarray(pos, float), np.asarray(vel, float))]
    sim.scene = SimpleNamespace(people=[SimpleNamespace(name="p")])
    return sim


def test_pred_rows_off_by_default_and_moving_only():
    sh = SH.Shield(None)
    assert sh.cfg.pred_taus == () and sh.cfg.deadlock_s is None and not sh.cfg.relax_tier
    cfg = SH.ShieldCfg(pred_taus=(1.0, 2.0))
    sh = SH.Shield(cfg)
    A, b = sh._pred_rows(_fake([1.5, -0.3], [0.0, 0.0]))
    assert len(b) == 0                                              # static person: no predictive row
    A, b = sh._pred_rows(_fake([2.5, -0.3], [-1.0, 0.0]))          # walks into the footprint front within 2 s
    assert A.shape == (2, 3) and b.min() < 0                       # demands retreat (negative rhs)


def test_relaxed_person_rows_are_looser():
    sh = SH.Shield(SH.ShieldCfg(d_person_relax=0.0))
    sim = _fake([1.0, -0.3], [0.0, 0.0])
    r0, b0, _ = sh._person_rows(sim)
    r1, b1, _ = sh._person_rows(sim, relax=True)
    assert b1[0] > b0[0]
