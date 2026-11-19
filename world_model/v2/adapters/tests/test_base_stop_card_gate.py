"""Owner gate (11-17): the MEASURED base-stop card is read only when it is valid for the level (USER 11-17: MISSING unless the level matches;
the card is a property of the base per speed level, no plan match).  No physics run.
PYTHONPATH=. python -m pytest -q world_model/v2/adapters/tests/test_base_stop_card_gate.py"""
import json

import pytest

from world_model.v2.adapters import rule1_contents as RC

GOOD = dict(n=10, levels=["L2"], extra_overshoot_m=0.006, extra_overshoot_rad=0.009, settle_s=0.31, n_not_at_rest=0, U_kinds=[],
            extra_lateral_m=0.002, extra_yaw_rad=0.001)


def _card(tmp_path, **kw):
    c = dict(GOOD, **kw)
    c = {k: v for k, v in c.items() if v is not ...}
    p = tmp_path / "CARD.json"
    p.write_text(json.dumps(c))
    return p


def test_valid_card_for_its_level(tmp_path):
    c = RC.base_stop_card(_card(tmp_path), level="L2")
    assert c is not None and c["card_level"] == "L2" and c["extra_overshoot"] == dict(line=0.006, turn=0.009)
    assert c["extra_lateral_m"] == 0.002 and c["extra_yaw_rad"] == 0.001


@pytest.mark.parametrize("bad", [dict(levels=["L1"]), dict(n_not_at_rest=1), dict(U_kinds=["wheel_liftoff"]), dict(extra_overshoot_m=float("nan")),
                                 dict(settle_s=-0.1), dict(extra_lateral_m=-0.01), dict(extra_overshoot_rad=...)])
def test_invalid_card_is_missing(tmp_path, bad):
    assert RC.base_stop_card(_card(tmp_path, **bad), level="L2") is None


def test_no_level_or_no_file_is_missing(tmp_path):
    assert RC.base_stop_card(_card(tmp_path), level=None) is None
    assert RC.base_stop_card(tmp_path / "nope.json", level="L2") is None


def test_old_card_without_drift_gives_none_drift(tmp_path):
    c = RC.base_stop_card(_card(tmp_path, extra_lateral_m=..., extra_yaw_rad=...), level="L2")
    assert c is not None and c["extra_lateral_m"] is None and c["extra_yaw_rad"] is None


def test_per_level_values_are_used(tmp_path):
    per = dict(L1=dict(GOOD, extra_lateral_m=0.0028), L2=dict(GOOD, extra_lateral_m=0.0052))
    p = _card(tmp_path, levels=["L1", "L2"], extra_lateral_m=0.0052, per_level=per)
    assert RC.base_stop_card(p, level="L1")["extra_lateral_m"] == 0.0028
    assert RC.base_stop_card(p, level="L2")["extra_lateral_m"] == 0.0052
    bad = dict(per, L1=dict(GOOD, n_not_at_rest=2))                       # (overwrites the same CARD.json)
    assert RC.base_stop_card(_card(tmp_path, levels=["L1", "L2"], per_level=bad), level="L1") is None
