"""11-13 F1: stored backup of a move AFTER the base move (table-behind): new default phase list, legacy list, retreat('start') guard.  Pure (no MuJoCo)."""
import pytest

from world_model.v2.stack.planners import closed_chain as CCP

R1 = (("line", -0.5), ("turn", 180.0), ("line", 0.5))
NAMES = ("s0_lift", "s1_move", "s2_place")


def _wp(base=True, backups=("set_down",) * 3):
    return CCP.WaypointPlan(names=NAMES, deltas=((0, 0, .15), (0, .1, 0), (0, 0, -.15)), backups=backups, base=((2, R1, "L1"),) if base else ())


def test_new_default_after_base(monkeypatch):
    monkeypatch.delenv("BIMANUAL_LEGACY_BEHIND_BACKUP", raising=False)
    wp = _wp()
    assert CCP.backup_phases_for("s2_place", wp) == [("stop", 0.3), ("advance", ("s2_place",)), ("hold", 0.3), ("weld", False), ("jretreat", "goal"), ("hold", 0.6)]
    assert CCP.backup_phases_for("s0_lift", wp) == [("stop", 0.3), ("retrace", ("s0_lift",)), ("hold", 0.3), ("weld", False), ("jretreat", "start"), ("hold", 0.6)]


def test_legacy_list_bit_identical(monkeypatch):
    monkeypatch.setenv("BIMANUAL_LEGACY_BEHIND_BACKUP", "1")
    assert CCP.backup_phases_for("s2_place", _wp()) == [("stop", 0.3), ("retrace", ("s2_place",)), ("hold", 0.3), ("weld", False), ("jretreat", "start"), ("hold", 0.6)]


def test_fixed_base_unchanged(monkeypatch):
    for v in (None, "1"):
        if v:
            monkeypatch.setenv("BIMANUAL_LEGACY_BEHIND_BACKUP", v)
        else:
            monkeypatch.delenv("BIMANUAL_LEGACY_BEHIND_BACKUP", raising=False)
        assert CCP.backup_phases_for("s2_place", _wp(base=False)) == [("stop", 0.3), ("retrace", ("s2_place", "s1_move", "s0_lift")), ("hold", 0.3), ("weld", False),
                                                                      ("jretreat", "start"), ("hold", 0.6)]
        assert CCP.backup_phases_for("place") == [("stop", 0.3), ("retrace", ("place", "carry", "lift")), ("hold", 0.3), ("weld", False), ("jretreat", "start"), ("hold", 0.6)]


def test_guard_rejects_start_retreat_not_knot0(monkeypatch):
    monkeypatch.delenv("BIMANUAL_LEGACY_BEHIND_BACKUP", raising=False)
    wp = _wp()
    bad = [("stop", 0.3), ("retrace", ("s2_place",)), ("hold", 0.3), ("weld", False), ("jretreat", "start"), ("hold", 0.6)]
    with pytest.raises(ValueError, match="not knot 0"):
        CCP.check_start_retreat(bad, wp)
    CCP.check_start_retreat([("stop", 0.3), ("retrace", ("s2_place", "s1_move", "s0_lift")), ("jretreat", "start")], wp)       # ends at names[0]: fine
    monkeypatch.setenv("BIMANUAL_LEGACY_BEHIND_BACKUP", "1")
    CCP.check_start_retreat(bad, wp)                                                                                            # legacy: not checked
