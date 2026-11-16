"""K6 gate (owner-written, 11-16): Rule 1 (stack/checks/rule1.py) -- exists ONE recovery program viable for EVERY theta in Theta_t from EVERY branch point.
SIMULATION ONLY.  Pure part: synthetic physics callables with a known truth, checked against a brute-force point oracle.  Physics part: a small
consistency check of adapters/rule1_contents.py against the K2 static admission on box.

Synthetic model: theta = {"m"}.  Each plan has a GOOD set G(plan) (a ThetaSet); check_motion(plan, S, t) passes iff S is inside G(plan) (and the plan is not
blocked at branch time t).  landing_ok passes iff the landing is in a set of good landings.
"""
from __future__ import annotations

import numpy as np
import pytest

from world_model.v2.stack import theta as TH
from world_model.v2.stack.checks import rule1 as R1
from world_model.v2.stack.checks.row import Row

TIMES = [0.0, 0.25, 0.5, 0.6]


def box(lo, hi):
    return TH.ThetaSet.box(m=(lo, hi))


def make_callables(good, blocked=(), landings_ok=("set_down",), sampled=()):
    """good: {plan: ThetaSet}; blocked: {(plan, t)} that fail at that branch time; sampled: plans whose row is 'sampled' instead of 'proof'."""
    calls = []

    def check_motion(plan, S, t):
        calls.append((plan, t))
        ok = good[plan].contains(S) and (plan, t) not in blocked
        g = "sampled" if (ok and plan in sampled) else ("proof" if ok else "refuted")
        return Row(f"motion[{plan}]", ok, margin=1.0 if ok else -1.0, guarantee=g, n=100 if g == "sampled" else None)

    def landing_ok(landing, S):
        if landing in landings_ok:
            return Row(f"landing[{landing}]", True, margin=1.0, guarantee="proof")
        return Row(f"landing[{landing}]", False, guarantee="missing", why="no R_K witness")

    return check_motion, landing_ok, calls


def spec(noise=0.05):
    return TH.ReadingSpec("claw_load", "vertical load [kg]", {"m": 1.0}, 0.0, noise, "MEASURED", "test")


def test_union_of_two_per_cell_plans_is_not_admitted():
    good = {"A": box(1.0, 2.0), "B": box(2.0, 3.0)}
    cm, lo, _ = make_callables(good)
    progs = [TH.RecoveryProgram.fixed("A", "A", "set_down"), TH.RecoveryProgram.fixed("B", "B", "set_down")]
    for cell, expect in ((box(1.0, 2.0), "A"), (box(2.0, 3.0), "B")):
        adm, _ = R1.admit_step(0, cell, progs, TIMES, cm, lo)
        assert adm is not None and adm.recovery == expect
    adm, per = R1.admit_step(0, box(1.0, 3.0), progs, TIMES, cm, lo)   # A covers half, B the other half: no single program covers all
    assert adm is None and all(R1.failing(rs) for rs in per.values())


def test_reading_program_bins_admit_the_union_only_when_every_bin_passes():
    # the same two plans, chosen AFTER a reading: allowed (a program), as long as the noisy bins stay inside each plan's good set
    good = {"A": box(0.9, 2.1), "B": box(1.9, 3.1)}
    cm, lo, _ = make_callables(good)
    prog = TH.RecoveryProgram("read", None, spec(0.05), ((TH.Interval(0.0, 2.0), "A"), (TH.Interval(2.0, 4.0), "B")), "set_down")
    adm, _ = R1.admit_step(0, box(1.0, 3.0), [prog], TIMES, cm, lo)
    assert adm is not None and adm.recovery == "read"
    # noise 0.2 > the 0.1 overlap: the bin "A" now contains worlds up to 2.2 kg, outside A's good set -> one failing bin -> refused
    prog2 = TH.RecoveryProgram("read", None, spec(0.2), prog.bins, "set_down")
    adm2, per = R1.admit_step(0, box(1.0, 3.0), [prog2], TIMES, cm, lo)
    assert adm2 is None and any("bin0" in r.name for r in R1.failing(per["read"]))


def test_uncovered_reading_refuses():
    good = {"A": box(0.0, 5.0)}
    cm, lo, _ = make_callables(good)
    prog = TH.RecoveryProgram("gap", None, spec(0.01), ((TH.Interval(0.0, 1.5), "A"), (TH.Interval(1.7, 5.0), "A")), "set_down")
    adm, per = R1.admit_step(0, box(1.0, 3.0), [prog], TIMES, cm, lo)
    assert adm is None and any(r.name.endswith(":coverage") and not r.ok for r in per["gap"])


def test_missing_landing_never_admits():
    good = {"A": box(0.0, 5.0)}
    cm, lo, _ = make_callables(good, landings_ok=("set_down",))
    adm, per = R1.admit_step(0, box(1.0, 2.0), [TH.RecoveryProgram.fixed("A", "A", "let_go:carry")], TIMES, cm, lo)
    assert adm is None and any(r.guarantee == "missing" for r in per["A"])


def test_every_branch_point_and_preference_order_and_table():
    good = {"A": box(0.0, 5.0), "B": box(0.0, 5.0)}
    cm, lo, calls = make_callables(good, blocked={("A", 0.5)})
    progs = [TH.RecoveryProgram.fixed("A", "A", "set_down"), TH.RecoveryProgram.fixed("B", "B", "set_down")]
    adm, _ = R1.admit_step(0, box(1.0, 2.0), progs, TIMES, cm, lo)
    assert adm.recovery == "B"                                        # A fails at ONE branch point -> not viable; B wins
    assert {t for p, t in calls if p == "A"} == set(TIMES)            # every branch point was checked
    assert [n for n, _ in adm.table[0.5]] == ["B"] and [n for n, _ in adm.table[0.0]] == ["A", "B"]
    cm2, lo2, _ = make_callables(good)
    assert R1.admit_step(0, box(1.0, 2.0), progs, TIMES, cm2, lo2)[0].recovery == "A"   # both viable: the first in preference order


def test_guarantee_is_the_weakest_row_and_fallback_is_recorded_not_required():
    good = {"A": box(0.0, 5.0), "F": box(0.0, 2.0)}
    cm, lo, _ = make_callables(good, sampled={"A"})
    theta_0 = TH.ThetaSet((TH.Cell({"m": TH.Interval(0.0, 2.0)}), TH.Cell({"m": TH.Interval(2.0, 5.0)})))
    adm, _ = R1.admit_step(0, box(2.5, 3.0), [TH.RecoveryProgram.fixed("A", "A", "set_down")], TIMES, cm, lo,
                           fallback=TH.RecoveryProgram.fixed("F", "F", "set_down"), theta_0=theta_0)
    assert adm is not None                                            # admitted although the fallback fails on part of Theta_0 (D1: recorded only)
    g = next(a["value"] for a in adm.assumptions if a["name"] == "guarantee")
    assert g == "sampled"
    assert adm.fallback_ok_on is not None and adm.fallback_ok_on.contains(box(0.0, 2.0)) and not adm.fallback_ok_on.contains(box(2.0, 5.0))
    assert {a["name"] for a in adm.assumptions} >= {"K", "NODE_CAP"}
    assert all(a["tag"] == "ASSUMED" for a in adm.assumptions if a["name"] in ("K", "NODE_CAP"))


def test_random_soundness_vs_point_oracle():
    """admitted => for sampled worlds theta in Theta_t and any noisy reading y it can produce, the winning program's bin for y has a plan good at theta."""
    rng = np.random.default_rng(21)
    n_adm = n_ref = 0
    for trial in range(150):
        cuts = np.sort(rng.uniform(0.0, 5.0, 2))
        good = {"A": box(0.0, float(cuts[1])), "B": box(float(cuts[0]), 5.0)}
        cm, lo, _ = make_callables(good)
        noise = float(rng.uniform(0.0, 0.3))
        edge = float(rng.uniform(0.5, 4.5))
        progs = [TH.RecoveryProgram.fixed("A", "A", "set_down"), TH.RecoveryProgram.fixed("B", "B", "set_down"),
                 TH.RecoveryProgram("read", None, spec(noise), ((TH.Interval(-1.0, edge), "A"), (TH.Interval(edge, 7.0), "B")), "set_down")]
        a, b = np.sort(rng.uniform(0.0, 5.0, 2))
        th = box(float(a), float(b) + 1e-3)
        adm, _ = R1.admit_step(0, th, progs, TIMES, cm, lo)
        if adm is None:
            n_ref += 1
            continue
        n_adm += 1
        win = next(p for p in progs if p.name == adm.recovery)
        for _ in range(200):
            m = float(rng.uniform(a, b + 1e-3))
            y = m + float(rng.uniform(-noise if win.reading else 0.0, noise if win.reading else 0.0))
            plans = [pl for iv, pl in win.bins if (win.reading is None or iv.lo <= y <= iv.hi)]
            assert plans, (trial, m, y)
            assert any(good[pl].contains_point({"m": m}) for pl in plans), (trial, adm.recovery, m, y)
    assert n_adm > 20 and n_ref > 5


# ---------------------------------------------------------------------------------------------------------- physics consistency (box, closed_chain)
@pytest.fixture(scope="module")
def box_records(tmp_path_factory):
    # planner passed explicitly; no os.environ change (a module-scoped setdefault leaked BIMANUAL_PLANNER into later test files: ilab job 1301045)
    from world_model.v2.adapters import rule1_contents as RC
    out = tmp_path_factory.mktemp("r1")
    return {cell: RC.run_cell("box", cell, out, n_sampled=0, planner="closed_chain", fallback=False) for cell in ((1.0, 1.5), (3.0, 3.5))}


def test_rule1_admits_only_inside_the_static_admission(box_records):
    from world_model.v2.nero.contents import carry_cert as CC
    w = CC.world_default("box")
    for cell, recs in box_records.items():
        static = CC.static_cell("box", cell, w, planner="closed_chain")
        refused = static["action"].startswith("refuse")
        assert recs, cell
        for r in recs:
            if refused:
                assert not r["admitted"], (cell, r["seg"])           # Rule 1 never admits a step of a plan the static admission refuses
            if r["admitted"]:
                assert r["guarantee"] in ("proof", "sampled")
                assert r["admission"]["recovery"] == r["program"]
    light = {r["seg"]: r for r in box_records[(1.0, 1.5)]}
    assert light["lift"]["admitted"] and light["carry"]["admitted"], [(s, r.get("refusal")) for s, r in light.items()]
    # 11-16 stop-finish fix: no place branch is MISSING any more (a stop in the last T s finishes the move); place is still refused, now ONLY by the
    # D9 object-world row: box_wall_e vs shelf, sample distance 7.1 mm, swept bound 2.1 mm < d_min 5 mm (U_OBJ_CLEAR_M, ASSUMED).  Pinned so a silent
    # change shows up; re-decide when the place pose / D9 rule changes.
    assert not light["place"]["admitted"]
    fails = [f for p in light["place"]["per_program"].values() for f in p["failing"]]
    assert fails and all(f["guarantee"] == "refuted" and "obj_world" in (f.get("why") or "") for f in fails), [(f["name"], f.get("why")) for f in fails]
    for r in box_records[(1.0, 1.5)]:
        if r["admitted"]:
            assert r.get("executable") is True and any(a["name"] == "between_samples" or "between_samples" in str(a) for a in r["admission"]["assumptions"])
    assert not any(r["admitted"] for r in box_records[(3.0, 3.5)])


def test_empty_theta_is_never_admitted():
    """independent review 11-16: contradictory evidence (empty Theta_t) must not admit every step vacuously."""
    cm, lo, _ = make_callables({"A": box(0.0, 5.0)})
    adm, per = R1.admit_step(0, TH.ThetaSet(()), [TH.RecoveryProgram.fixed("A", "A", "set_down")], TIMES, cm, lo)
    assert adm is None and any(not r.ok for r in per["A"])
