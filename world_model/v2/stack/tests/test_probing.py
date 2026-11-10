"""stack/probing.py tests with fake skills (probe-selection build step 4).  SIMULATION ONLY, CPU."""
import dataclasses
import math

import pytest

from world_model.v2.stack import probing as PB
from world_model.v2.stack.skills import catalog as CAT

INF = math.inf


@dataclasses.dataclass
class FRow:
    ok: bool
    name: str = "rung_utilisation"
    why: str = ""
    margin: float = 0.0


class Fake:
    def __init__(self, sid, reading=None, rows=None):
        self.id, self._r, self._rows, self.ran, self.rows_seen = sid, reading, rows or [], 0, []

    def limit_rows(self, theta_t):
        self.rows_seen.append(dict(theta_t))
        return list(self._rows)

    def run(self, theta_t):
        self.ran += 1
        return self._r(theta_t) if callable(self._r) else self._r


def entry(sid, measures=("PH1",), fields=("contents_mass_kg",), pre=()):
    return dataclasses.replace(CAT.WEIGH, id=sid, measures=measures, theta_fields=fields, preconditions=pre)


def cat2():
    return {"a": entry("a"), "b": entry("b", ("PH2",), ("contents_com_xy",), pre=("a",)), "c": entry("c", ("PH3",), ("contents_com_z",))}


T0 = {"contents_mass_kg": (0.0, 5.0), "contents_com_xy": (-0.1, 0.1), "contents_com_z": (0.0, 0.2)}


def test_real_catalog_weigh_shrinks_only_its_field():
    sk = Fake("weigh", PB.Reading("weigh", {"contents_mass_kg": (1.0, 2.0)}))
    r = PB.run_probes(["weigh"], {"weigh": sk}, T0)
    assert r.theta_t["contents_mass_kg"] == (1.0, 2.0) and r.theta_t["contents_com_xy"] == T0["contents_com_xy"] and r.measured == ["PH1"] and r.skipped == []
    assert r.events[0]["kind"] == "ran" and r.counts() == {"ran": 1} and r.stopped is None
    assert T0["contents_mass_kg"] == (0.0, 5.0)                                     # the input is not mutated


def test_own_limit_fails_probe_skipped_and_recorded_theta_unchanged():
    sk = Fake("weigh", PB.Reading("weigh", {"contents_mass_kg": (1.0, 2.0)}), rows=[FRow(True), FRow(False, "rung_overshoot_kg", "overshoot 1.2 kg > 1.0", -0.2)])
    r = PB.run_probes(["weigh"], {"weigh": sk}, T0)
    assert sk.ran == 0 and r.theta_t == {k: tuple(v) for k, v in T0.items()} and r.measured == [] and r.skipped == ["PH1"]
    e = r.events[0]
    assert e["kind"] == "skipped" and e["failed_rows"][0]["name"] == "rung_overshoot_kg" and "overshoot" in e["failed_rows"][0]["why"]


def test_limit_rows_see_the_current_theta_t_not_the_prior():
    a = Fake("a", PB.Reading("a", {"contents_mass_kg": (1.0, 2.0)}))
    b = Fake("b", PB.Reading("b", {"contents_com_xy": (0.0, 0.05)}))
    r = PB.run_probes(["a", "b"], {"a": a, "b": b}, T0, skill_catalog=cat2())
    assert b.rows_seen[0]["contents_mass_kg"] == (1.0, 2.0) and a.rows_seen[0]["contents_mass_kg"] == (0.0, 5.0)
    assert r.measured == ["PH1", "PH2"] and r.theta_t["contents_com_xy"] == (0.0, 0.05)


def test_theta_only_shrinks_under_wider_readings():
    a = Fake("a", PB.Reading("a", {"contents_mass_kg": (1.0, 2.0)}, "interval"))
    c = Fake("c", PB.Reading("c", {"contents_com_z": (-1.0, 9.0)}))                       # a reading wider than Theta_t cannot widen it
    r = PB.run_probes(["a", "c"], {"a": a, "c": c}, T0, skill_catalog=cat2())
    assert r.theta_t["contents_com_z"] == (0.0, 0.2) and r.theta_t["contents_mass_kg"] == (1.0, 2.0)
    for k, (lo, hi) in r.theta_t.items():
        assert lo >= T0[k][0] and hi <= T0[k][1]


def test_one_sided_reading_intersects():
    a = Fake("a", PB.Reading("a", {"contents_mass_kg": (3.0, INF)}, "lower_bound"))
    r = PB.run_probes(["a"], {"a": a}, T0, skill_catalog=cat2())
    assert r.theta_t["contents_mass_kg"] == (3.0, 5.0) and r.events[0]["output"] == "lower_bound"


def test_t3_contradiction_stops_and_is_counted():
    a = Fake("a", PB.Reading("a", {"contents_mass_kg": (1.0, 2.0)}))
    a2 = Fake("a2", PB.Reading("a2", {"contents_mass_kg": (3.0, 4.0)}))                    # disjoint from the sound set [1, 2]
    c = Fake("c", PB.Reading("c", {"contents_com_z": (0.0, 0.1)}))
    cat = {**cat2(), "a2": entry("a2")}
    r = PB.run_probes(["a", "a2", "c"], {"a": a, "a2": a2, "c": c}, T0, skill_catalog=cat)
    assert r.stopped == "T3" and r.counts() == {"ran": 1, "T3": 1, "not_run": 1}
    t3 = [e for e in r.events if e["kind"] == "T3"][0]
    assert t3["contradictions"][0]["previous"] == [1.0, 2.0] and t3["contradictions"][0]["reading"] == [3.0, 4.0]
    assert r.theta_t["contents_mass_kg"] == (1.0, 2.0) and c.ran == 0 and r.skipped == ["PH3"] and r.measured == ["PH1"]


def test_touching_reading_within_tol_is_not_a_contradiction():
    a = Fake("a", PB.Reading("a", {"contents_mass_kg": (1.0, 2.0)}))
    a2 = Fake("a2", PB.Reading("a2", {"contents_mass_kg": (2.0 + 1e-12, 3.0)}))
    r = PB.run_probes(["a", "a2"], {"a": a, "a2": a2}, T0, skill_catalog={**cat2(), "a2": entry("a2")})
    assert r.stopped is None and r.theta_t["contents_mass_kg"][0] == r.theta_t["contents_mass_kg"][1]


def test_order_is_checked_never_repaired():
    b = Fake("b", PB.Reading("b", {"contents_com_xy": (0.0, 0.05)}))
    a = Fake("a", PB.Reading("a", {"contents_mass_kg": (1.0, 2.0)}))
    with pytest.raises(PB.ProbeOrderError) as e:
        PB.run_probes(["b", "a"], {"a": a, "b": b}, T0, skill_catalog=cat2())
    assert "needs 'a'" in str(e.value) and a.ran == 0 and b.ran == 0                      # nothing ran, nothing was reordered
    with pytest.raises(PB.ProbeOrderError):
        PB.run_probes(["weigh", "nope"], {"weigh": a}, T0)


def test_no_probe_needed_when_static_verdict_is_the_same():
    sk = Fake("weigh", PB.Reading("weigh", {"contents_mass_kg": (1.0, 2.0)}))
    r = PB.run_probes(["weigh"], {"weigh": sk}, T0, static_same=lambda th: True)
    assert sk.ran == 0 and r.events[0]["kind"] == "no_probe_needed" and r.skipped == ["PH1"] and r.theta_t["contents_mass_kg"] == (0.0, 5.0)


def test_reading_on_a_foreign_field_is_an_error():
    sk = Fake("weigh", PB.Reading("weigh", {"contents_com_z": (0.0, 0.1)}))
    with pytest.raises(ValueError, match="not one of the skill's theta_fields"):
        PB.run_probes(["weigh"], {"weigh": sk}, T0)
    sk = Fake("weigh", PB.Reading("other", {"contents_mass_kg": (1.0, 2.0)}))
    with pytest.raises(ValueError, match="reading.skill"):
        PB.run_probes(["weigh"], {"weigh": sk}, T0)


def test_empty_probe_list_is_a_no_op():
    r = PB.run_probes([], {}, T0)
    assert r.events == [] and r.measured == [] and r.theta_t == {k: tuple(v) for k, v in T0.items()}
