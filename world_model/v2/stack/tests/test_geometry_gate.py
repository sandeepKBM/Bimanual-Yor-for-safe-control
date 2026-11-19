"""Owner gate (11-19): the interval geometry structure (stack/geometry.py; USER 11-19: Codex establishes the scene geometry itself, priors then
probes; intervals first; no factor graph).  Pure python, CPU, no network.
PYTHONPATH=. python -m pytest -q world_model/v2/stack/tests/test_geometry_gate.py

Checks:
  1. a prior never narrows anything: require() refuses a field with only a prior (fail closed), and the set ignores the prior;
  2. the set is the intersection of every reading (brute force over random readings: a value lies in the set iff it lies in every reading);
  3. inconsistent readings -> contradiction, require() refuses; a reading outside the prior -> set follows the reading, the clash is recorded;
  4. source tag: MODEL only when every reading is MODEL;
  5. JSON round trip exact; codex_view carries priors, sets, status and readings;
  6. parse_priors rejects bad replies (unknown field, lo > hi, NaN, missing kind)."""
import json
import math

import numpy as np
import pytest

from world_model.v2.stack import geometry as G


def _rd(v, e, src="MEASURED", probe="measure_box"):
    return G.Reading(probe=probe, value=float(v), err=float(e), guarantee="sampled", noise_model="clean sim depth", n=200, source=src, note="")


def _gs():
    g = G.GeometrySet()
    g.add_entity("crate", "object", roles=("carried",))
    g.add_entity("shelf", "support", roles=("goal_support",))
    return g


def test_prior_never_narrows():
    g = _gs()
    g.set_prior("crate", "sx", 0.10, 0.12, "looks small")
    with pytest.raises(G.GeometryMissing):
        g.require("crate", "sx")                                       # prior only: MISSING
    g.add_reading("crate", "sx", _rd(0.20, 0.005))
    iv = g.require("crate", "sx")
    assert (iv.lo, iv.hi) == pytest.approx((0.195, 0.205))             # the reading, not the prior
    with pytest.raises(G.GeometryMissing):
        g.box_hull("crate")                                            # other fields have no reading


@pytest.mark.parametrize("seed", range(5))
def test_set_is_intersection_of_readings(seed):
    rng = np.random.default_rng(seed)
    for _ in range(50):
        g = _gs()
        truth = rng.uniform(-1, 1)
        rds = [(truth + rng.uniform(-e, e), e) for e in rng.uniform(0.01, 0.1, size=rng.integers(1, 6))]
        for v, e in rds:
            g.add_reading("crate", "cx", _rd(v, e))
        iv = g.require("crate", "cx")
        for x in rng.uniform(-1.5, 1.5, 400):
            assert iv.contains(x) == all(v - e <= x <= v + e for v, e in rds)
        assert iv.contains(truth)


def test_contradiction_fails_closed_and_prior_clash_recorded():
    g = _gs()
    g.add_reading("crate", "cy", _rd(0.0, 0.01))
    g.add_reading("crate", "cy", _rd(0.1, 0.01))                       # disjoint
    with pytest.raises(G.GeometryMissing):
        g.require("crate", "cy")
    assert g.entities["crate"].fields["cy"].status == "contradiction"
    g.set_prior("shelf", "cz", 0.5, 0.6, "guess")
    g.add_reading("shelf", "cz", _rd(0.8, 0.01))
    assert g.require("shelf", "cz").contains(0.8)
    assert len(g.prior_contradictions) == 1 and g.prior_contradictions[0]["entity"] == "shelf"
    g.add_reading("crate", "sz", _rd(0.30, 0.01))
    g.set_prior("crate", "sz", 0.10, 0.12, "late guess")              # a prior given AFTER the reading is compared too
    assert len(g.prior_contradictions) == 2 and g.require("crate", "sz").contains(0.30)
    g2 = G.GeometrySet.from_json(json.loads(json.dumps(g.to_json())))
    assert len(g2.prior_contradictions) == 2 and len(g2.contradictions) == 1
    assert len(g.codex_view()["prior_contradictions"]) == 2


def test_source_tag():
    g = _gs()
    g.add_reading("crate", "sz", _rd(0.2, 0.001, src="MODEL", probe="model"))
    assert g.entities["crate"].fields["sz"].source == "MODEL"
    g.add_reading("crate", "sz", _rd(0.2, 0.002))
    assert g.entities["crate"].fields["sz"].source == "MEASURED"


def test_json_round_trip_and_view():
    g = _gs()
    g.set_prior("crate", "yaw", -0.1, 0.1, "looks square to the table")
    for f in G.FIELDS:
        g.add_reading("crate", f, _rd(0.3, 0.01))
    g2 = G.GeometrySet.from_json(json.loads(json.dumps(g.to_json())))
    assert g2.to_json() == g.to_json()
    h = g2.box_hull("crate")
    assert set(h) == set(G.FIELDS)
    v = json.dumps(g.codex_view())
    for k in ("prior", "set", "status", "measure_box"):
        assert k in v, k
    assert g.assumptions()
    sch = G.schema_for_prompt().lower()
    assert "never" in sch and all(f in sch for f in G.FIELDS)


def test_parse_priors_rejects_bad():
    ok = {"entities": [{"name": "crate", "kind": "object", "roles": ["carried"], "priors": {"sx": [0.2, 0.3]}, "why": {"sx": "photo"}}]}
    g = G.parse_priors(ok)
    assert g.entities["crate"].fields["sx"].prior.hi == pytest.approx(0.3)
    bad = [
        {"entities": [{"name": "c", "kind": "object", "priors": {"width": [0, 1]}}]},
        {"entities": [{"name": "c", "kind": "object", "priors": {"sx": [0.3, 0.2]}}]},
        {"entities": [{"name": "c", "kind": "object", "priors": {"sx": [math.nan, 0.2]}}]},
        {"entities": [{"name": "c", "priors": {"sx": [0.1, 0.2]}}]},
    ]
    for b in bad:
        with pytest.raises(ValueError):
            G.parse_priors(b)
