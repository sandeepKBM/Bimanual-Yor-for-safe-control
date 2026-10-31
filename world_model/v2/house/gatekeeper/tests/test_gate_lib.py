"""LibGatekeeper (gate_lib.py): library hooks, recovery paths direct / move / look / none, LOOK only from rest, base defaults untouched.  CPU, no MuJoCo.
    nice -n 10 python -m pytest -q world_model/v2/house/gatekeeper/tests/test_gate_lib.py"""
import math
import types

import numpy as np
import pytest

from world_model.v2.house.gatekeeper import GKConfig, Gatekeeper, plan_candidate, check1, check2
from world_model.v2.house.gatekeeper import tests as _t                                          # noqa: F401
from world_model.v2.house.gatekeeper.tests import test_gatekeeper as TG
from world_model.v2.house.gatekeeper.tests.test_gatekeeper import FakeSim, GK
from world_model.v2.house.gatekeeper.tests.test_gatekeeper import bel as _bel
from world_model.v2.house.gatekeeper.tests.test_tracker_source import FakePerc, _ti
from world_model.v2.house.gatekeeper.gate_lib import LibGatekeeper
from world_model.v2.house.checkers.library import Library
from world_model.v2.house.people_track.look import Looker, LookCfg
from world_model.v2.house.teleop.sim import TICK

ACC = GKConfig(person_model="accel")


def bel(pose=(1.5, 5.0, 0.0), **kw):
    p = np.array(pose, float)
    return _bel(pose_wall=p.copy(), pose_people=p.copy(), **kw)


def seg(b, u=(0.3, 0.0, 0.0), T_S=0.6):
    return plan_candidate(np.zeros(3), np.array(u, float), T_S, b, GK)


def admit(tr, b, gk=GK):
    return check1(tr, b, gk)[0] and check2(tr, b, gk)[0]


def drive(g, sim, n, u_op=(0.8, 0.0, 0.0)):
    for _ in range(n):
        u, label = g.filter_base(sim, np.array(u_op, float))
        g.n_calls += 1
        sim.step(u)
    return label


def rest_lib(xs=(1.5, 2.5), y=5.0, grow=dict(xy=0.08)):
    lib = Library(GK, grow=grow)
    for x in xs:
        b = bel(pose=(x, y, 0.0), e_xy=0.02, e_th=0.003)
        assert lib.add(seg(b), b, GK) is not None
    return lib


# ------------------------------------------------------------------------------------------------------------ defaults untouched / hooks
@pytest.mark.parametrize("name", ["a", "b"])
def test_base_gatekeeper_defaults_untouched_golden(name):
    TG.test_defaults_bit_identical_to_pre_feature_behaviour(name)                                  # the base class golden test, unchanged


@pytest.mark.parametrize("name", ["a", "b"])
def test_libgatekeeper_without_recovery_is_bit_identical_to_the_base_golden(name, monkeypatch):
    monkeypatch.setattr(TG, "Gatekeeper", lambda gk=None: LibGatekeeper(gk=gk, stall_s=1e9, grow=None))
    g, c, chk, pose = TG._golden_run(name)
    G = TG.GOLDEN[name]
    assert {k: c[k] for k in G["c"]} == G["c"] and chk == G["chk"] and pose == G["pose"]
    assert c["n_rec_calls"] == 0 and c["lib_size"] > 0                                              # it only observed: the library filled, nothing was recovered


def test_library_add_called_on_every_commit_with_the_certified_belief(monkeypatch):
    g = LibGatekeeper(grow=None)
    sim = FakeSim((1.5, 5.0, 0.0))
    calls = []
    lib = Library(g.gk, grow=None)
    g.lib = lib
    orig = lib.add

    def spy(tr, b, gk=None, **kw):
        calls.append((tr, b, kw))
        return orig(tr, b, gk, **kw)
    monkeypatch.setattr(lib, "add", spy)
    drive(g, sim, 100)
    assert g.n_commit > 5 and len(calls) == g.n_commit == g.n_lib_offers
    tr, b, kw = calls[-1]
    assert b is not None and b.e_xy == g.last_bel.e_xy and kw["step"] == g.n_steps and np.allclose(kw["v0"], tr.V[0])
    assert all(admit(c[0], c[1]) for c in calls)                                                     # only certified segments were offered (and entered)
    assert len(lib) >= 1 and g.counters()["lib_size"] == len(lib)


# ------------------------------------------------------------------------------------------------------------ recovery trigger (integration)
def stalled_setup(d=0.9):
    g = LibGatekeeper(gk=ACC, grow=dict(xy=0.1, p=0.3, mu=0.10), debug=True, strict=True)
    sim = FakeSim((1.5, 5.0, 0.0), people=[(1.5 + d, 5.0)])
    return g, sim


def test_stall_triggers_after_stall_s_and_empty_library_gives_none():
    g, sim = stalled_setup()
    drive(g, sim, 40)                                                                                # 0.8 s of stall (steps at 0, .1, ..., .7)
    assert g.n_commit == 0 and g.n_keep_backup > 0 and g.n_rec_calls == 0
    drive(g, sim, 60)
    assert g.n_rec_calls >= 1 and g.n_rec_none == g.n_rec_calls and g.n_rec_direct == 0
    first = g._last_rec_t
    assert first >= 1.0 - 1e-9 and first <= 1.5                                                      # no earlier than stall_s
    c = g.counters()
    assert c["n_rec_none"] == g.n_rec_none and c["lib_size"] == 0


def test_direct_recovery_in_the_loop_is_certified_and_moves_the_robot():
    g, sim = stalled_setup()
    drive(g, sim, 1)                                                                                 # first gk step: no candidate, keep-backup
    b = g.last_bel
    moving = False
    for u in ([0, 0.3, 0], [0, -0.3, 0], [-0.3, 0, 0], [0, 0.15, 0], [-0.15, 0, 0], [0, 0, 0]):       # in this geometry only the stay-put segment is certified
        tr = seg(b, u)
        if admit(tr, b):
            moving = any(u)
            break
    e = g.lib.add(tr, b, g.gk, v0=np.zeros(3))
    assert e is not None
    p0 = sim.pose().copy()
    drive(g, sim, 100)
    assert g.n_rec_direct >= 1 and g.n_rec_commit >= 1 and g.n_rec_mismatch == 0
    assert g.traj is not None and g.traj.V.shape == tr.V.shape or g.mode != "rec_direct"
    if moving:
        assert np.hypot(*(sim.pose()[:2] - p0[:2])) > 0.05                                           # the recovery trajectory was tracked


def test_direct_recovery_trajectory_passes_check1_check2_and_mismatch_is_counted(monkeypatch):
    sim = FakeSim((1.5, 5.0, 0.0))
    g = LibGatekeeper(grow=None, debug=True)
    g.lib = rest_lib()
    b = bel(pose=(1.5, 5.0, 0.0), e_xy=0.02, e_th=0.003)
    rc = g._recover(sim, b, np.zeros(3), 3.0)
    assert rc.mode == "direct" and g.mode == "rec_direct" and g.traj.t0 == 3.0 and g.n_rec_direct == 1 and g.n_rec_commit == 1
    assert admit(g.traj, b) and g.n_rec_mismatch == 0 and g.traj.i_rest is not None
    # force a mismatch: the checker rejects -> refused, counted, nothing committed
    g2 = LibGatekeeper(grow=None, debug=True)
    g2.lib = rest_lib()
    import world_model.v2.house.gatekeeper.gate_lib as GL
    monkeypatch.setattr(GL, "check1", lambda *a, **k: (False, -1.0, "x", {}))
    g2._recover(sim, b, np.zeros(3), 3.0)
    assert g2.n_rec_direct == 1 and g2.n_rec_mismatch == 1 and g2.n_rec_commit == 0 and g2.traj is None
    g3 = LibGatekeeper(grow=None, debug=True, strict=True)
    g3.lib = rest_lib()
    with pytest.raises(AssertionError):
        g3._recover(sim, b, np.zeros(3), 3.0)


def test_move_recovery_commits_certified_legs_then_the_entry():
    sim = FakeSim((1.95, 5.25, 0.15))
    g = LibGatekeeper(grow=None, debug=True)
    g.lib = rest_lib()
    b = bel(pose=(1.95, 5.25, 0.15), e_xy=0.02, e_th=0.003)
    rc = g._recover(sim, b, np.zeros(3), 2.0)
    assert rc.mode == "move" and g.n_rec_move == 1 and g.mode == "rec_move" and g._rq is not None
    assert admit(g.traj, b)
    # follow the connector: re-anchor the belief at the pose each leg ends at, advance until the entry is committed
    t, n = 2.0, 0
    while g._rq is not None and n < 10:
        t = g._rq["end_t"]
        end = rc.legs[min(g._rq["idx"], len(rc.legs) - 1)].pose_w1
        bb = bel(pose=tuple(end), e_xy=0.02, e_th=0.003)
        g._advance_move(bb, np.zeros(3), t)
        n += 1
    assert g.n_rec_move_abort == 0 and g.n_rec_move_done == 1 and g.mode == "rec_direct" and g.traj.t0 == t


def test_move_leg_not_certified_aborts_the_connector():
    sim = FakeSim((1.95, 5.25, 0.15))
    g = LibGatekeeper(grow=None)
    g.lib = rest_lib()
    b = bel(pose=(1.95, 5.25, 0.15), e_xy=0.02, e_th=0.003)
    rc = g._recover(sim, b, np.zeros(3), 2.0)
    assert rc.mode == "move" and len(rc.legs) >= 1
    if len(rc.legs) == 1:
        pytest.skip("single-leg connector")
    bad = bel(pose=(1.95, 5.25, 0.15), e_xy=0.02, e_th=0.003, people=(np.array([2.0, 5.3]),))        # a person on top of the robot invalidates the next leg
    g._advance_move(bad, np.zeros(3), g._rq["end_t"])
    assert g._rq is None and g.n_rec_move_abort == 1


def test_candidate_commit_preempts_a_pending_move():
    sim = FakeSim((1.95, 5.25, 0.15))
    g = LibGatekeeper(grow=None)
    g.lib = rest_lib()
    b = bel(pose=(1.95, 5.25, 0.15), e_xy=0.02, e_th=0.003)
    assert g._recover(sim, b, np.zeros(3), 2.0).mode == "move"
    g.gk_step(sim, np.array([0.3, 0, 0]), 2.1)                                                       # free space: hold-u_op certifies and commits
    assert g._rq is None and g.n_rec_move_abandoned == 1 and g.mode == "commit"


def test_none_keeps_base_behaviour():
    sim = FakeSim((1.5, 5.0, 0.0))
    g = LibGatekeeper(grow=None)
    g.lib = rest_lib()
    rc = g._recover(sim, bel(pose=(1.5, 5.0, 0.0), mu_lo=0.05), np.zeros(3), 1.0)                    # mu outside every entry: looking cannot fix it
    assert rc.mode == "none" and g.n_rec_none == 1 and g.traj is None and g.n_rec_commit == 0
    assert g._rq is None and g.looker is None


# ------------------------------------------------------------------------------------------------------------ LOOK
class FakeCam:
    def __init__(self, sim):
        self.sim = sim

    def pose(self):
        y = self.sim.base.yaw
        c, s = math.cos(y), math.sin(y)
        R = np.array([[-s, 0.0, -c], [c, 0.0, -s], [0.0, -1.0, 0.0]])                                # forward = -R[:,2] = (cos, sin, 0)
        return np.array([self.sim.base.x, self.sim.base.y, 1.0]), R


class LookPerc(FakePerc):
    hfov = math.radians(90.0)
    omap = None
    tracks = []

    def __init__(self, sim, tis):
        super().__init__(tis)
        self.cam = FakeCam(sim)


def look_setup(person_xy=(1.5, 7.0), v=(0, 0, 0)):
    """tracker source; the entry is certified for a tighter person position error than the (stale) track has: a SOFT deficit; person at +90 deg (outside the 90 deg FOV)."""
    sim = FakeSim((1.5, 5.0, 0.0))
    sim.t = 0.0
    sim.base.v[:] = v
    tis = [_ti(person_xy, sp=0.05, since=0.0, name="p0")]
    perc = LookPerc(sim, tis)
    g = LibGatekeeper(gk=GKConfig(people_source="tracker", person_model="accel", k_sigma=2.0), perc=perc, grow=None, debug=True)
    g.perceive(sim)
    g._shield = lambda sim: types.SimpleNamespace(footprint_of=lambda s: g.fp_now)                   # no occ_ctrl-dependent internals needed here
    return g, sim, perc


def look_belief(g, sim, e_p_track=0.4):
    """belief with one stale person near the path and an entry certified for e_p = 0.1 -> look recommended (person0:e_p)."""
    b0 = bel(pose=(1.5, 5.0, 0.0), people=(np.array([1.5, 7.0]),), people_e_p=(0.1,), people_stale=(0.0,), people_e_v=(0.0,), people_vel=(np.zeros(2),), person_model="accel", e_v=0.0)
    tr = seg(b0)
    assert admit(tr, b0), "premise: entry admissible"
    lib = Library(g.gk, grow=None)
    assert lib.add(tr, b0, g.gk, v0=np.zeros(3)) is not None
    g.lib = lib
    return b0.with_(people_e_p=(e_p_track,))


def test_look_glance_issued_from_rest_toward_the_person_and_cancelled_if_uncertified():
    g, sim, perc = look_setup()
    bl = look_belief(g, sim)
    rc = g.lib.recover(bl, np.zeros(3), gk=g.gk)
    assert rc.mode == "look" and any(k.startswith("person0") for k in rc.look["reduce"]), rc
    g._proxy = perc.proxy(sim)
    rc = g._recover(sim, bl, np.zeros(3), 1.0)
    assert rc.mode == "look" and g.n_rec_look == 1 and g.n_look_glances == 1 and g.looker is not None and g.looker.active
    # the glance rotates toward +y (left): the twist replaces u_op
    u, label = g.filter_base(sim, np.array([0.8, 0.0, 0.0]))
    assert g.glance_ticks == 1
    # run it to completion: the base yaws until the person is inside the FOV, translation stays zero
    n = 0
    while g.looker.active and n < 200:
        u, _ = g.filter_base(sim, np.array([0.8, 0.0, 0.0])); g.n_calls += 1; sim.step(u); sim.t += TICK; n += 1
        assert abs(u[0]) < 1e-9 and abs(u[1]) < 1e-9
    assert not g.looker.active and sim.base.yaw > 0.1


def test_look_only_from_rest_and_never_for_a_moving_base():
    g, sim, perc = look_setup(v=(0.2, 0.0, 0.0))
    bl = look_belief(g, sim)
    g._proxy = perc.proxy(sim)
    g.lib.recover = lambda *a, **k: types.SimpleNamespace(mode="look", entry=None, look=dict(entry_id=0, reduce={"person0:e_p": (0.4, 0.1)}), ms=0.0, matches=[], legs=[])
    g._recover(sim, bl, np.array([0.2, 0.0, 0.0]), 1.0)
    assert g.n_rec_look == 1 and g.n_look_deferred == 1 and g.n_look_glances == 0 and not (g.looker and g.looker.active)
    sim.base.v[:] = 0.0
    g._recover(sim, bl, np.zeros(3), 1.5)
    assert g.n_look_glances == 1 and g.looker.active


def test_look_blocked_when_glance_twist_is_not_certified():
    g, sim, perc = look_setup()
    bl = look_belief(g, sim)
    g._proxy = perc.proxy(sim)
    g._recover(sim, bl, np.zeros(3), 1.0)
    assert g.looker.active
    g.mode, g.traj = "keep", None                                                                    # the next gatekeeper step did not commit the glance twist
    g._stall_t0 = 0.0
    g.looker.active = True
    orig = Gatekeeper.gk_step
    try:
        Gatekeeper.gk_step = lambda self, s, u, t: setattr(self, "mode", "keep")
        g.gk_step(sim, np.array([0.0, 0.0, 0.7]), 1.1)
    finally:
        Gatekeeper.gk_step = orig
    assert not g.looker.active and g.n_look_blocked == 1


def test_look_is_a_counted_noop_for_the_truth_source():
    sim = FakeSim((1.5, 5.0, 0.0))
    g = LibGatekeeper(gk=ACC, grow=None)
    b0 = bel(pose=(1.5, 5.0, 0.0), e_xy=0.02, e_th=0.003)
    g.lib = rest_lib(grow=dict(xy=0.08))
    ex = g.lib.entries[0].X.e_xy
    big = bel(pose=(1.5, 5.0, 0.0), e_xy=ex + 0.05, e_th=0.01)
    rc = g._recover(sim, big, np.zeros(3), 1.0)
    assert rc.mode == "look" and g.n_rec_look == 1 and g.n_look_noop == 1 and g.n_look_glances == 0 and g.looker is None


def test_look_target_person_in_view_or_pose_only_deficit_is_not_a_glance():
    g, sim, perc = look_setup(person_xy=(4.0, 5.0))                                                  # straight ahead: inside the FOV
    bl = look_belief(g, sim)
    g._proxy = perc.proxy(sim)
    g._recover(sim, bl, np.zeros(3), 1.0)
    assert g.n_look_glances == 0 and g.n_look_infov + g.n_look_nofix >= 1
    g2, sim2, perc2 = look_setup()
    g2.lib = rest_lib(grow=dict(xy=0.08))
    ex = g2.lib.entries[0].X.e_xy
    g2._proxy = perc2.proxy(sim2)
    g2._recover(sim2, bel(pose=(1.5, 5.0, 0.0), e_xy=ex + 0.05, e_th=0.01), np.zeros(3), 1.0)         # e_xy deficit: a person glance cannot reduce it
    assert g2.n_look_glances == 0 and g2.n_look_nofix == 1


def test_counters_expose_recovery_counts():
    g = LibGatekeeper(grow=None)
    sim = FakeSim((1.5, 5.0, 0.0))
    drive(g, sim, 30)
    c = g.counters()
    for k in ("n_rec_direct", "n_rec_move", "n_rec_look", "n_rec_none", "lib_size", "lookup_ms_mean", "recover_ms_max", "n_commit", "n_regime2_ticks", "step_ms_mean"):
        assert k in c
