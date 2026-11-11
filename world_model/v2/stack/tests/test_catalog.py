"""stack/skills/catalog.py tests (probe-selection build step 1).  SIMULATION ONLY, CPU.   PYTHONPATH=. python -m pytest -q world_model/v2/stack/tests/test_catalog.py"""
import json
import math

import pytest

from world_model.v2.stack import registry as R
from world_model.v2.stack import tasks as TK
from world_model.v2.stack.skills import catalog as CAT
from world_model.v2.stack.types import Bound

CARRY = [n for n in TK.names("carry")]


def test_every_number_is_a_tagged_bound_with_a_source():
    bs = CAT.all_bounds()
    assert bs
    for owner, b in bs:
        assert isinstance(b, Bound) and b.tag in ("MEASURED", "MODEL", "ASSUMED"), owner
        assert b.src.strip(), f"{owner}: empty src"
        assert b.lo <= b.hi
    w = CAT.SKILLS["weigh"]
    assert w.noise["residual"].tag == "MEASURED" and "CALIB.json" in w.noise["residual"].src and "n = 20" in w.noise["residual"].src
    assert w.noise["read"].tag == "ASSUMED"
    lim = dict(w.own_limits)
    assert lim["rung_utilisation"].hi == 1.0 and lim["rung_utilisation"].tag == "MODEL" and lim["rung_overshoot_kg"].tag == "ASSUMED"
    assert math.isinf(lim["rung_utilisation"].lo)


def test_fallbacks_carry_tag_and_status():
    for (kind, ph), fb in CAT.FALLBACK.items():
        assert kind in ("box", "tray", "pot") and ph in CAT.PHENOMENA
        assert fb.tag in ("MEASURED", "MODEL", "ASSUMED") and fb.src and fb.status in ("approved", "out_of_scope")
        assert (fb.status == "out_of_scope") == (ph in CAT.OUT_OF_SCOPE)
    assert CAT.fallback("tray6_behind", "PH7").kind == "fill_max" and CAT.fallback("box_behind", "PH2").kind == "interior"
    assert CAT.fallback("pot", "PH8").tag == "ASSUMED" and CAT.fallback("box", "PH1").kind == "prior_box"


@pytest.mark.parametrize("task", CARRY)
def test_phi_ids_exist_and_every_phi_has_an_approved_fallback(task):
    for pid in CAT.phi(task):
        assert pid in CAT.PHENOMENA and CAT.PHENOMENA[pid].in_scope and pid not in CAT.OUT_OF_SCOPE
        assert CAT.kind_of(task) in CAT.PHENOMENA[pid].objects
        assert CAT.fallback(task, pid).status == "approved", (task, pid)


def test_phi_values():
    assert CAT.phi("box") == ("PH1", "PH2", "PH3") and CAT.phi("box_behind") == CAT.phi("box")
    assert CAT.phi("tray6") == CAT.phi("tray6_behind") == ("PH1", "PH2", "PH3", "PH7") == CAT.phi("tray2") == CAT.phi("tray4")
    assert CAT.phi("pot") == ("PH1", "PH2", "PH3", "PH8")
    assert CAT.phi("pan_lift") == () and CAT.phi("coffee_tray") == ()
    assert set(CAT.OUT_OF_SCOPE) == {i for i, p in CAT.PHENOMENA.items() if not p.in_scope}


def test_phenomenon_rows_exist_in_the_registry():
    for p in CAT.PHENOMENA.values():
        for row in p.rows:
            assert row in R.CHECKERS, (p.id, row)
        assert p.theta_fields and p.objects


def test_design_entries_are_not_selectable_and_have_no_card():
    assert set(CAT.built()) == {"weigh"}
    for sid in ("hold_moment", "shake_listen", "watch_item"):
        assert CAT.SKILLS[sid].status == "design" and sid not in CAT.built() and CAT.SKILLS[sid].card == {}
        with pytest.raises(NotImplementedError):
            R.get("measure", sid)
    assert CAT.skill_for("PH1") == ["weigh"] and CAT.skill_for("PH2") == [] and CAT.skill_for("PH5") == []
    assert R.MEASURE["weigh"] == CAT.SKILLS["weigh"].code and set(R.MEASURE) == set(CAT.SKILLS)
    for s in CAT.SKILLS.values():
        assert set(s.measures) <= set(CAT.PHENOMENA)


@pytest.mark.parametrize("task", CARRY)
def test_cards_are_deterministic_and_free_of_prior_numbers(task):
    c1, c2 = CAT.cards(task), CAT.cards(task)
    assert json.dumps(c1, sort_keys=True) == json.dumps(c2, sort_keys=True)
    assert [c["id"] for c in c1["skills"]] == ["weigh"]
    kind = CAT.kind_of(task)
    assert [c["id"] for c in c1["phenomena"]] == sorted(p.id for p in CAT.PHENOMENA.values() if kind in p.objects)
    txt = json.dumps(c1)
    lo, hi = TK.prior_box(task)
    for bad in ("sanity", "prior", "contents_kg", "fill_total", f"{hi} kg", f"[{lo}, {hi}]"):
        assert bad not in txt, bad
    for c in c1["phenomena"]:
        assert set(c) == {"id", "name", "what"}


def test_order_errors_are_reported_never_repaired():
    assert CAT.order_errors(["weigh"]) == [] and CAT.order_errors([]) == []
    import dataclasses
    # a made-up skill that needs weigh first: checked through a catalog copy (order_errors reads CAT.SKILLS)
    CAT.SKILLS["_t"] = dataclasses.replace(CAT.WEIGH, id="_t", preconditions=("weigh",))
    try:
        assert CAT.order_errors(["weigh", "_t"]) == []
        e = CAT.order_errors(["_t", "weigh"])
        assert len(e) == 1 and "needs 'weigh'" in e[0]
        assert "not in probes" in CAT.order_errors(["_t"])[0]
    finally:
        del CAT.SKILLS["_t"]
