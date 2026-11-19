"""K9 gate (owner-written, 11-16): Rule 4 (stack/checks/rule4.py) on synthetic callables.  SIMULATION ONLY (pure).

  accepted only if the transfer AND the at-target rest (AND the next step, for an in-air target) pass on all of Theta_t
  a missing landing / missing phase is never accepted; no candidate is checked twice; refusal after max_rounds keeps the log
  trap: accepted only when a set-down target is offered
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from world_model.v2.stack import theta as TH
from world_model.v2.stack.checks import rule4 as R4
from world_model.v2.stack.checks.row import Row

TIMES = [0.0, 0.25, 0.5]
THETA = TH.ThetaSet.box(m=(2.5, 3.0))


@dataclass(frozen=True)
class Tg:
    name: str
    landing: str
    requires_next: bool


def cm(plan, S, t):
    ok = plan in ("slow_retrace", "rest", "next_ok")
    return Row(f"m[{plan}]", ok, margin=1.0 if ok else -1.0, guarantee="proof" if ok else "refuted")


def lo(landing, S):
    if landing in ("set_down:start", "hold:lift"):
        return Row(f"l[{landing}]", True, margin=1.0, guarantee="proof")
    return Row(f"l[{landing}]", False, guarantee="missing", why="no witness")


def programs_at(tg, phase):
    if phase == "transfer":
        plan = "slow_retrace" if tg.name != "bad_transfer" else "fast"
        return [TH.RecoveryProgram.fixed(f"tr:{tg.name}", plan, tg.landing)]
    if phase == "here":
        return [TH.RecoveryProgram.fixed(f"rest:{tg.name}", "rest", tg.landing)]
    if phase == "next":
        if tg.name == "hold_ok_next":
            return [TH.RecoveryProgram.fixed("n", "next_ok", "set_down:start")], TIMES, cm, lo
        if tg.name == "hold_no_next":
            return None
        return [TH.RecoveryProgram.fixed("n", "fast", "set_down:start")], TIMES, cm, lo
    raise ValueError(phase)


def check(tg):
    return R4.check_target(tg, None, THETA, programs_at, TIMES, cm, lo)


def test_check_target_parts():
    assert check(Tg("set_down", "set_down:start", False))[0]
    assert not check(Tg("bad_transfer", "set_down:start", False))[0]          # transfer fails
    assert not check(Tg("let_go", "let_go:carry", False))[0]                  # landing missing
    assert check(Tg("hold_ok_next", "hold:lift", True))[0]                    # in-air with a viable next step
    ok, rows, _ = check(Tg("hold_no_next", "hold:lift", True))
    assert not ok and any(r.guarantee == "missing" for r in rows)
    assert not check(Tg("hold_bad_next", "hold:lift", True))[0]
    ok, rows, adm = check(Tg("set_down", "set_down:start", False))
    assert adm["guarantee"] == "proof" and {a["name"] for a in adm["assumptions"]} >= {"MAX_ROUNDS", "K", "NODE_CAP"}


def test_loop_trap_and_no_repeats():
    seen = []

    def chk(tg):
        seen.append(tg.name)
        return check(tg)

    def propose_with_setdown(state, th, fails):
        return [Tg("hold_bad_next", "hold:lift", True), Tg("let_go", "let_go:carry", False)] if len(seen) < 2 else \
               [Tg("hold_bad_next", "hold:lift", True), Tg("set_down", "set_down:start", False)]
    acc, log = R4.loop(None, THETA, propose_with_setdown, chk, max_rounds=3)
    assert acc is not None and acc.name == "set_down"
    assert seen.count("hold_bad_next") == 1                                    # never checked twice

    seen.clear()

    def propose_no_setdown(state, th, fails):
        return [Tg("hold_bad_next", "hold:lift", True), Tg("let_go", "let_go:carry", False)]
    acc, log = R4.loop(None, THETA, propose_no_setdown, chk, max_rounds=3)
    assert acc is None and len(log) >= 1 and log[-1].get("note") == "the proposer offered no new target"

    seen.clear()
    rounds = iter(range(100))

    def propose_endless(state, th, fails):
        return [Tg(f"bad{next(rounds)}", "let_go:x", False)]
    acc, log = R4.loop(None, THETA, propose_endless, chk, max_rounds=3)
    assert acc is None and len(log) == 3


# ---------------------------------------------------------------------------------------------------------- physics (box trap, closed_chain), slow
@pytest.mark.skip(reason="RETIRED (USER 11-19): fixed-base task, out of scope; with the measured load tilt its nominal plan is refused at lift (4.4 mm < 5 mm, n=4 tilt). Listed in docs/status/checker_measured_set_plan_2026-11-19.md Part 3")
def test_box_trap_offgrid_set_down_is_a_safe_park(monkeypatch):
    """box 2.5-3.0 kg mid-carry at t = 0.8 s (off the 0.25 s grid on purpose: no snapping).  Holding in the air is refused (the next step fails Rule 1);
    a set-down at the start is accepted, checked from EVERY branch point of the transfer (0.25 s grid + its end), and labelled a safe park
    (Rule 1 still refuses continuing: the 11-16 livelock finding; the policy is the user's call)."""
    if "BIMANUAL_PLANNER" not in __import__("os").environ:
        monkeypatch.setenv("BIMANUAL_PLANNER", "closed_chain")          # restored after the test (was a leaking setdefault)
    from world_model.v2.adapters.tests.tilt_input import measured_tilt      # the recovery collision rows need the load tilt: MEASURED for the cell (n = 4 ASSUMED test choice)
    tilt = measured_tilt("box", (2.5, 3.0))
    from world_model.v2.adapters import rule4_contents as R4C
    from world_model.v2.stack.checks import rule1 as R1
    acc, log, tr = R4C.run_trap(True, t=0.8, tilt=tilt)
    assert tr["state"].t == 0.8
    assert acc is not None and acc.name == "set_down_start", [(e["round"], e["verdicts"]) for e in log]
    rejected = {v["target"] for e in log for v in e["verdicts"] if not v["ok"]}
    assert {"hold_here", "hold_lower"} <= rejected
    ad = tr["admission"]
    assert ad["continuation"].startswith("static_only")
    t_ = ad["transfer"]
    assert list(t_["branch_times"]) == list(R1.branch_times(t_["duration"])), t_["branch_times"]
    assert len(t_["branch_times"]) >= int(t_["duration"] / R1.BRANCH_DT)
    # review fixes 11-16: the admitted in-flight recovery is recorded per branch time; the stop-ladder values are carried as ASSUMED
    pb = t_["per_branch"]
    assert [round(e["t"], 9) for e in pb] == [round(x, 9) for x in t_["branch_times"]], ([e["t"] for e in pb], t_["branch_times"])
    assert all(e["T"] is not None and e["end_idx"] is not None and e["mode"] for e in pb if e["mode"] != "continue_decel_to_target"), pb
    names = {x["name"]: x for x in ad["assumptions"]}
    assert {"STOP_K_MAX", "RETRACE_HALVINGS", "STOP_T0"} <= set(names) and all(names[n]["tag"] == "ASSUMED" for n in ("STOP_K_MAX", "RETRACE_HALVINGS", "STOP_T0"))
    acc2, log2, _ = R4C.run_trap(False, t=0.8, tilt=tilt)
    assert acc2 is None and log2[-1].get("note") == "the proposer offered no new target"
