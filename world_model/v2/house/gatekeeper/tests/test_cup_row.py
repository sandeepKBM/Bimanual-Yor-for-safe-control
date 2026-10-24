"""GKConfig.cup_row: the cup row applies only while the cup rides on the robot ("attached"); "always" (default) is the original behaviour.

Regression for the 10-22 bug: with empty hands the old belief anchored the cup row on the cup sitting on the counter (~3.75 m lever arm), so braking a
turn took 0.8 s instead of 0.15 s and the person disc reached the robot.  SIMULATION ONLY (no sim, FakeSim stand-in).
"""
import types

import numpy as np
import pytest

from world_model.v2.house.gatekeeper import GKConfig, Gatekeeper, backup_from, check1
from world_model.v2.house.gatekeeper.belief import CUP_FREE_MARGIN, cup_attached
from world_model.v2.house.gatekeeper.traj import Traj
from world_model.v2.house.gatekeeper.tests.test_gatekeeper import FakeSim, build_belief_for

CUP_ON_COUNTER = np.array([1.5 + 3.3, 5.0 + 1.78, 0.0])        # ~3.75 m from the robot (the h5 failing state)
V_TURN = np.array([0.264, 0.0, 0.258])                         # the moving, turning state of the failing h5 segment


class EmptyHands(FakeSim):
    """robot NOT holding the tray; the cup stays on a counter in the world."""
    grasped = False

    def _sync(self):
        self.d.xpos[0] = CUP_ON_COUNTER.copy()


class Carrying(FakeSim):
    grasped = True


def _belief(mode, sim, **gk_kw):
    gk = GKConfig(cup_row=mode, **gk_kw)
    return build_belief_for(Gatekeeper(gk=gk), sim), gk


def _turn_then_backup(b, gk):
    bk = backup_from(V_TURN, b, gk)
    n = 2
    return Traj(np.concatenate([np.tile(V_TURN, (n + 1, 1)), bk.V[1:]]), gk.h, 0.0, n * gk.h, n * gk.h), bk


def test_default_is_always():
    assert GKConfig().cup_row == "always"


def test_always_keeps_the_old_behaviour_with_empty_hands():
    b, gk = _belief("always", EmptyHands((1.5, 5.0, 0.0)))
    assert np.linalg.norm(b.r_cup) == pytest.approx(3.75, abs=0.01)
    assert b.cup_margin == gk.cup_margin


def test_attached_drops_the_cup_row_with_empty_hands():
    b, _ = _belief("attached", EmptyHands((1.5, 5.0, 0.0)))
    assert np.allclose(b.r_cup, 0.0)
    assert b.cup_margin == CUP_FREE_MARGIN and b.a_cup_max > 100.0


@pytest.mark.parametrize("pose", [(1.5, 5.0, 0.0), (2.0, 4.0, 1.1)])
def test_attached_is_identical_while_carrying(pose):
    ba, _ = _belief("always", Carrying(pose))
    bb, _ = _belief("attached", Carrying(pose))
    assert np.array_equal(ba.r_cup, bb.r_cup) and ba.cup_margin == bb.cup_margin and ba.a_cup_max == bb.a_cup_max


def test_cup_attached_rule():
    ns = types.SimpleNamespace
    assert cup_attached(ns())                                              # unknown -> conservative (attached)
    assert cup_attached(ns(grasped=True))
    assert not cup_attached(ns(grasped=False))
    assert not cup_attached(ns(grasped=False, assist=ns(state="IDLE")))
    for st in ("HOLD", "CLOSE", "FAILED", "ABORT", "PLACE"):               # friction grasp: any non-IDLE state counts as carrying
        assert cup_attached(ns(grasped=False, assist=ns(state=st)))


def test_empty_hands_turn_near_a_person_is_admitted_only_with_attached():
    """person 1.8 m ahead, robot turning with empty hands: the old row makes the backup 0.8 s and the person disc reaches the robot."""
    sim_kw = dict(people=[(1.5 + 1.8, 5.0)])
    b_old, gk_old = _belief("always", EmptyHands((1.5, 5.0, 0.0), **sim_kw), person_model="accel")
    tr_old, bk_old = _turn_then_backup(b_old, gk_old)
    ok_old, _, row_old, _ = check1(tr_old, b_old, gk_old)
    b_new, gk_new = _belief("attached", EmptyHands((1.5, 5.0, 0.0), **sim_kw), person_model="accel")
    tr_new, bk_new = _turn_then_backup(b_new, gk_new)
    ok_new, _, _, g_new = check1(tr_new, b_new, gk_new)
    assert len(bk_old.V) * gk_old.h > 0.6 and len(bk_new.V) * gk_new.h < 0.3
    assert not ok_old and row_old == "person"
    assert ok_new and g_new["person"] > 0.5


# ---------------------------------------------------------------------------------------------- Shield (teleop/shield.py) uses the same rule
def _shield_out(mode, sim, u_op):
    from world_model.v2.house.teleop import shield as SH
    sh = SH.Shield(SH.ShieldCfg(cup_row=mode, use_ir_veto=False), occ=sim.occ_ctrl, a_hw=sim.a_hw)
    sim.base.v = V_TURN.copy()
    u, _ = sh.filter_base(sim, np.asarray(u_op, float))
    return np.asarray(u, float)


def test_shield_default_is_always():
    from world_model.v2.house.teleop.shield import ShieldCfg
    assert ShieldCfg().cup_row == "always"


def test_shield_attached_drops_cup_rows_with_empty_hands():
    """empty hands, hard stop out of a turn: the old rows limit the turn braking (lever arm 3.75 m); attached only applies the hardware slew."""
    u_op = np.zeros(3)
    u_old = _shield_out("always", EmptyHands((1.5, 5.0, 0.0)), u_op)
    u_new = _shield_out("attached", EmptyHands((1.5, 5.0, 0.0)), u_op)
    assert abs(u_new[2]) < abs(u_old[2]) - 1e-3                           # brakes the rotation harder without the phantom cup row


def test_shield_attached_is_identical_while_carrying():
    u_op = np.array([0.6, 0.1, -0.4])
    assert np.array_equal(_shield_out("always", Carrying((1.5, 5.0, 0.0)), u_op), _shield_out("attached", Carrying((1.5, 5.0, 0.0)), u_op))


def test_gatekeeper_internal_shield_inherits_cup_row():
    g = Gatekeeper(gk=GKConfig(cup_row="attached"))
    assert g.cfg.cup_row == "attached" and Gatekeeper().cfg.cup_row == "always"
