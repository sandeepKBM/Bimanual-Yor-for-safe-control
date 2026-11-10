"""probe-selection build steps 3 + 6 (sim): Weigh == today's probe; the v2 per-world run (coverage, Rule 2 T1/T2/T3, re-ask, fixed point).  SIMULATION ONLY, CPU, no network, mock proposer only.
PYTHONPATH=. python -m pytest -q world_model/v2/nero/contents/tests/test_probe_select.py        (each test = one or two simulated worlds: minutes)
Fake skills stand in for the weigh where a reading must be forced (a lower bound above the admissible mass, a contradiction); the Weigh object itself is still built on the episode."""
import argparse
import math

import pytest

from world_model.v2.adapters import skills_contents as SK
from world_model.v2.nero.contents import carry_cert as CC
from world_model.v2.nero.contents import probe as P
from world_model.v2.nero.contents import probe_select as PS
from world_model.v2.nero.contents import same_worlds as SW
from world_model.v2.stack import probing as PB
from world_model.v2.stack.proposer import propose as PP

EPS = (0.0766, 0.0316)               # eps_lo, eps_free: the poslift CALIB's EPS_LO / EPS_FREE (catalog.WEIGH, MEASURED n = 20); fixed so the test does not read the file


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("BIMANUAL_LEGACY_PROPOSER_V1", raising=False)
    monkeypatch.setenv("BIMANUAL_NO_NETWORK", "1")
    monkeypatch.setattr(P, "PROBE_METHOD", "poslift")


def ns(cache, spec="mock:fixture"):
    return argparse.Namespace(plan_spec=spec, plan_cache=str(cache), offline=False, outside_b="refuse")


def plan_for(task, tmp_path, spec="mock:fixture"):
    return SW._proposer_plans([task], ns(tmp_path / "cache", spec), tmp_path)[task]["plan"]


class LowerBound:
    """fake weigh that reads only a lower bound (contents mass >= lo): forces Rule 1 to refuse the heavy end -> T2 (the real weigh is built but not run)."""
    id = "weigh"

    def __init__(self, weigh, lo=4.9):
        self.w, self.lo = weigh, lo

    def limit_rows(self, th):
        return []

    def run(self, th):
        self.w.why = "rung_not_admissible"
        return PB.Reading("weigh", {"contents_mass_kg": (self.lo, math.inf)}, "lower_bound", dict(why="rung_not_admissible", shows=["PH1"]))


class Contradict(LowerBound):
    def run(self, th):
        return PB.Reading("weigh", {"contents_mass_kg": (50.0, 60.0)}, "interval", dict(why="free", shows=["PH1"]))


class Shows(LowerBound):
    """reads the whole prior but shows PH2 (T1: a phenomenon in PHI the plan did not name)."""

    def run(self, th):
        return PB.Reading("weigh", {"contents_mass_kg": tuple(th["contents_mass_kg"])}, "interval", dict(why="free", shows=["PH1", "PH2"]))


def test_weigh_limit_row_is_the_probe_rung_row():
    from world_model.v2.nero.contents import carry as CR
    w = P._world("box", P._seed("e2e", "box", 0))
    ep = P._episode(w)
    wg = SK.Weigh(ep, "box", *EPS, method="poslift")
    c0 = P.prior_contents("box")
    rows = wg.limit_rows({"contents_mass_kg": (float(c0[0]), float(c0[1]))})
    assert [r.name for r in rows] == ["probe_rung"] and rows[0].ok and rows[0].worst["overshoot"] == 0.0 and rows[0].guarantee == "proof"
    assert wg.id == "weigh" and wg.probe.M_lo == P.empty_mass("box") + c0[0]


@pytest.mark.parametrize("task,i", [("box", 0), ("box", 1)])
def test_weigh_theta_and_why_equal_todays_e2e_job(task, i):
    old = P._e2e_job((task, i, *EPS), method="poslift")
    assert "error" not in old, old.get("tb")
    seed = P._seed("e2e", task, i)
    import numpy as np
    w = P._world(task, seed)
    ep = P._episode(w)
    wg = SK.Weigh(ep, task, *EPS, rng=np.random.default_rng(seed), method="poslift")
    cmax = P.c_admit_max(task)
    wg.m_stop = None if cmax is None else P.empty_mass(task) + cmax
    c0 = P.prior_contents(task)
    th = {"contents_mass_kg": (float(c0[0]), float(c0[1]))}
    if old["why"] == "no_probe_needed":
        pytest.skip("this world needs no probe")
    run = PB.run_probes(["weigh"], {"weigh": wg}, th)
    assert wg.why == old["why"] and [wg.probe.M_lo, wg.probe.M_hi] == old["Theta"] and len(wg.probe.rungs) == len(old["rungs"])      # bit-identical
    assert wg.reading.detail["M_lo"] == old["Theta"][0] and wg.reading.detail["M_hi"] == old["Theta"][1]
    assert run.measured == ["PH1"] and run.events[0]["kind"] == "ran" and run.events[0]["output"] == ("interval" if old["why"] == "free" else "lower_bound")
    if old["decision"] is not None:
        e = P.empty_mass(task)
        lo, hi = run.theta_t["contents_mass_kg"]
        assert abs(lo - old["decision"]["contents"][0]) < 1e-12 and abs(hi - old["decision"]["contents"][1]) < 1e-12 and abs(lo + e - old["Theta"][0]) < 1e-12


def test_t2_lower_bound_reask_fixed_point_counted(tmp_path):
    """a lower-bound-only reading above the admissible mass: Rule 1 refuses -> T2 -> re-ask (mock:fixture answers the SAME plan) -> fixed point, counted, refused by static."""
    plan = plan_for("box", tmp_path)
    r = PS.fixed_job(("box", 0, *EPS, plan), method="poslift", make_skills=lambda w: {"weigh": LowerBound(w)})
    assert "error" not in r, r.get("tb")
    v = r["v2"]
    assert r["outcome"].startswith("refused") and r["refused_by"] == "static" and "carry" not in r
    assert v["n_reasks"] == 1 and v["reask_end"] == "reask_fixed_point" and v["t_counts"] == {"T2": 1} and v["t_events"][0]["lower_bound_only"] is True
    assert v["readings"][0]["output"] == "lower_bound" and v["readings"][0]["fields"]["contents_mass_kg"][1] is None and v["probes_run"] == ["weigh"]
    assert v["coverage"]["covered"] and {x["phi"]: x["status"] for x in v["coverage"]["rows"]}["PH1"] == "measured" and "PH4" in v["out_of_scope"]
    assert v["attempts"][0]["decision"]["refused_by"] == "static" and v["reasks"][0]["trigger"] == "T2"


def test_t2_reask_then_admit_with_the_mock_variant(tmp_path):
    """mock:t2_reask: the first plan names no probe (PH1 on its fallback, the whole prior is refused) -> T2 -> the re-ask returns the fixture plan with the weigh -> it runs on the re-asked plan."""
    plan = plan_for("box", tmp_path, spec="mock:t2_reask")
    assert plan["plan_dict"]["probes"] == [] and plan["proposer"]["spec"] == "mock:t2_reask"
    r = PS.fixed_job(("box", 1, *EPS, plan), method="poslift")
    assert "error" not in r, r.get("tb")
    v = r["v2"]
    assert v["n_reasks"] >= 1 and v["t_counts"].get("T2", 0) >= 1 and v["first_plan_id"] != v["plan_id"] and v["probes_run"] == ["weigh"] and v["reasks"][0]["trigger"] == "T2"      # a lower-bound weigh is a 2nd T2 (-> fixed point with this mock)
    assert v["attempts"][0]["probes"] == [] and v["attempts"][1]["probes"] == ["weigh"] and v["attempts"][0]["decision"]["refused_by"] == "static"
    assert {x["phi"]: x["status"] for x in v["attempts"][0]["coverage"]["rows"]}["PH1"] == "fallback"
    assert r["why"] in ("free", "decided_refuse", "rung_not_admissible", "load_guard", "z_cap", "no_probe_needed")
    assert r["theta_sound"] is True


def test_t3_contradiction_stops_and_is_counted(tmp_path):
    plan = plan_for("box", tmp_path)
    r = PS.fixed_job(("box", 0, *EPS, plan), method="poslift", make_skills=lambda w: {"weigh": Contradict(w)})
    assert "error" not in r, r.get("tb")
    v = r["v2"]
    assert r["refused_by"] == "T3" and v["stopped"] == "T3" and v["t_counts"] == {"T3": 1} and v["n_reasks"] == 0 and "carry" not in r and r["decision"] is None
    assert v["t_events"][0]["contradictions"][0]["field"] == "contents_mass_kg" and v["theta_t"]["contents_mass_kg"] == [pytest.approx(P.prior_contents("box")[0]), pytest.approx(P.prior_contents("box")[1])]


def test_t1_phenomenon_shown_and_not_named_reasks(tmp_path):
    """a reading that shows PH2 while the plan never named it (mock fixture names only the kind's phenomena it lists... PH2 IS named there, so drop it from the plan first)."""
    plan = plan_for("box", tmp_path)
    pd = plan["plan_dict"]
    pd2 = dict(pd, phenomena=[p for p in pd["phenomena"] if p["id"] != "PH2"], plan_id="not-the-fixture")
    plan = dict(plan, plan_dict=pd2)
    r = PS.fixed_job(("box", 0, *EPS, plan), method="poslift", make_skills=lambda w: {"weigh": Shows(w)})
    assert "error" not in r, r.get("tb")
    v = r["v2"]
    assert v["t_counts"].get("T1") == 1 and v["reasks"][0]["trigger"] == "T1" and v["attempts"][0]["t1_shown"] == ["PH2"]
    assert v["n_reasks"] >= 1
