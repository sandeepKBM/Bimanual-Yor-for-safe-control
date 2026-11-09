"""stack/outcome.py -- the ONE success rule and the outcome record.  SIMULATION ONLY.

success = finished AND goal reached AND never in U (including before the hold) AND at rest in R      (USER 11-09, every method)
A refusal is not a success.  Any ingredient that is UNKNOWN (finished / unsafe / pre_hold_unsafe None, or a missing Row) gives success None unless
something else already failed (then False).  Unknown is never read as True or False.
load_legacy(record) READS old result dicts (ours probe e2e records, theirs execute records, pan cert / runtime records); it never writes them.
"""
from __future__ import annotations

from dataclasses import dataclass

from world_model.v2.stack.checks.row import Row, missing


@dataclass
class Outcome:
    finished: bool | None
    goal: Row
    unsafe: bool | None
    pre_hold_unsafe: bool | None
    in_R: Row
    refused: bool = False
    reason: str = ""
    success: bool | None = None


def _known(r: Row) -> bool:
    return r.guarantee != "missing"


def _b(x):
    return None if x is None else bool(x)


def classify(finished, goal: Row, unsafe, pre_hold_unsafe, in_R: Row, refused=False, reason="") -> Outcome:
    """finished / unsafe / pre_hold_unsafe: True | False | None (None = unknown)."""
    fails = (finished is False) or refused or (unsafe is True) or (pre_hold_unsafe is True) or (_known(goal) and not goal.ok) or (_known(in_R) and not in_R.ok)
    unknown = finished is None or unsafe is None or pre_hold_unsafe is None or not (_known(goal) and _known(in_R))
    success = False if fails else (None if unknown else True)
    return Outcome(_b(finished), goal, _b(unsafe), _b(pre_hold_unsafe), in_R, bool(refused), reason, success)


def _flag_row(name, ok, why=""):
    return Row(name, bool(ok), guarantee="runtime", why=why)


def load_legacy(rec: dict) -> Outcome:
    """old result dict -> Outcome.  Recognised shapes:
      theirs  (agentchord execute.run_world): 'goal' {xy_err, z_err, ...}, 'end_in_R', 'R_fail', 'pre_hold_viol', 'track'.viol, 'outcome' 'completed'.
              The goal row is re-evaluated against `goal_tol_m` (default 0.03, USER 11-03) -- NOT the record's own ac_success.  A record with no `track`
              has unsafe unknown.
      ours    (probe e2e): 'outcome' carried | carry_failed | refused*, 'carry' {end_in_R, segments[].nominal.viol, pre_hold_viol}; no goal in the record -> goal
              missing.  pre_hold_viol is written by nero/contents/probe.py (a U entry during the load-transfer hold, carry_cert.py:331); an absent key = unknown.
      pan     (cert / runtime): 'segments' with 'nominal' {viol, success, status}; goal = the last segment's success flag.  Pre-hold is not recorded
              (unknown); a segment with no `status` has unknown finish."""
    if "pre_hold_viol" in rec and "goal" in rec:                                   # theirs
        gc = rec["goal"]
        tol = rec.get("goal_tol_m", 0.03)
        m = {"xy_err": tol - gc["xy_err"], "z_err": tol - gc["z_err"]}
        goal = Row("goal", all(v >= 0 for v in m.values()), margin=min(m.values()), unit="m", guarantee="runtime", worst=m)
        trk = rec.get("track")
        unsafe = None if trk is None else (trk.get("viol") is not None)
        rr = rec.get("end_in_R")
        in_R = _flag_row("R", rr, ",".join(rec.get("R_fail") or [])) if rr is not None else missing("R", "record has no end_in_R")
        fin = None if rec.get("outcome") is None else rec.get("outcome") == "completed"
        return classify(fin, goal, unsafe, rec.get("pre_hold_viol") is not None, in_R,
                        refused=str(rec.get("outcome", "")).startswith("plan_rejected"), reason=str(rec.get("reason") or ""))
    if "segments" in rec and "outcome" not in rec and not rec.get("carry"):          # pan
        segs = rec["segments"]
        unsafe = any((s.get("nominal") or {}).get("viol") for s in segs)
        sts = [(s.get("nominal") or {}).get("status") for s in segs]
        if not segs:
            done = None
        elif any(x is not None and x != "done" for x in sts):
            done = False
        else:
            done = None if any(x is None for x in sts) else True                    # a missing status is unknown, never "done"
        last = (segs[-1].get("nominal") or {}) if segs else {}
        goal = _flag_row("goal", last.get("success"), "last segment success flag") if last.get("success") is not None else missing("goal", "no success flag")
        return classify(done, goal, unsafe, None, missing("R", "pan records carry the backup's R per branch, not an end R"), refused=bool(rec.get("refused")))
    oc = rec.get("outcome")                                                           # ours
    carry = rec.get("carry") or {}
    unsafe = any((s.get("nominal") or {}).get("viol") for s in carry.get("segments", []))
    er = carry.get("end_in_R")
    rr = er[0] if isinstance(er, (list, tuple)) else er
    in_R = _flag_row("R", rr) if rr is not None else missing("R", "record has no end_in_R")
    pre = (carry.get("pre_hold_viol") is not None) if "pre_hold_viol" in carry else None
    oc_s = str(oc or "")
    fin = None if oc is None else oc in ("carried", "carry_failed")
    return classify(fin, missing("goal", "ours e2e records have no goal check"), unsafe if carry else None, pre, in_R, refused=oc_s.startswith("refused"), reason=oc_s)
