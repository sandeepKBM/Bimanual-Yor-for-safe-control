"""stack/outcome.py -- the ONE success rule and the outcome record.  SIMULATION ONLY.

USER 11-09 (R1): success = AgentChord's own rule, for every method: finished AND goal reached (goal terms of the task: tasks.py).  Safety is reported SEPARATELY:
  success       finished AND goal reached                         (True | False | None = unknown)
  unsafe        any entry into U, including before the hold      (True | False | None)
  in_R          the end state is at rest in R                    (a Row; in_R_ok in the record)
  safe_success  success AND not unsafe
A refusal is not a success.  Any ingredient that is UNKNOWN (finished / unsafe / pre_hold_unsafe None, or a missing goal Row) gives None unless something else already
failed (then False).  Unknown is never read as True or False.
load_legacy(record) READS old result dicts (ours probe e2e records, theirs execute records, pan cert / runtime records); it never writes them.
"""
from __future__ import annotations

from dataclasses import dataclass

from world_model.v2.stack.checks.row import Row, missing


@dataclass
class Outcome:
    finished: bool | None
    goal: Row
    unsafe: bool | None             # any U entry, including the pre-hold phase
    pre_hold_unsafe: bool | None
    in_R: Row
    refused: bool = False
    reason: str = ""
    success: bool | None = None
    safe_success: bool | None = None

    def record(self) -> dict:
        """flat JSON-able form stored in result records under `result`."""
        return dict(finished=self.finished, success=self.success, unsafe=self.unsafe, pre_hold_unsafe=self.pre_hold_unsafe, in_R=self.in_R.ok if self.in_R.guarantee != "missing" else None,
                    safe_success=self.safe_success, refused=self.refused, reason=self.reason, goal_ok=self.goal.ok if self.goal.guarantee != "missing" else None)


def _known(r: Row) -> bool:
    return r.guarantee != "missing"


def _b(x):
    return None if x is None else bool(x)


def classify(finished, goal: Row, unsafe, pre_hold_unsafe, in_R: Row, refused=False, reason="") -> Outcome:
    """finished / unsafe / pre_hold_unsafe: True | False | None (None = unknown)."""
    s_fail = (finished is False) or bool(refused) or (_known(goal) and not goal.ok)
    s_unknown = finished is None or not _known(goal)
    success = False if s_fail else (None if s_unknown else True)
    u_any = True if (unsafe is True or pre_hold_unsafe is True) else (None if (unsafe is None or pre_hold_unsafe is None) else False)
    ss = False if (success is False or u_any is True) else (None if (success is None or u_any is None) else True)
    return Outcome(_b(finished), goal, u_any, _b(pre_hold_unsafe), in_R, bool(refused), reason, success, ss)


def _flag_row(name, ok, why=""):
    return Row(name, bool(ok), guarantee="runtime", why=why)


def _legacy_goal(rec: dict) -> Row:
    """goal Row of an old theirs record: the measured quantities against the task's goal terms (tasks.py)."""
    from world_model.v2.stack import goal as GL, tasks as TK
    gc = rec["goal"]
    t = TK.task(rec["task"]) if rec.get("task") in TK.TASKS else TK.task("box")
    g = GL.Goal(t.name, tuple(GL.GoalTerm(q, b) for q, b in t.goal_terms))
    now = dict(xy_err=gc["xy_err"], z_err=gc["z_err"], obj_tilt_deg=gc.get("tilt_deg", 0.0), cup_tilt_deg=gc.get("mug_tilt_deg", 0.0))
    return GL.check_goal(g, GL.StepState(0.0, now, now))


def load_legacy(rec: dict) -> Outcome:
    """old result dict -> Outcome.  Recognised shapes:
      theirs  (agentchord execute.run_world): 'goal' {xy_err, z_err, ...}, 'end_in_R', 'R_fail', 'pre_hold_viol', 'track'.viol, 'outcome' 'completed'.
              The goal row is re-evaluated with the task's CURRENT goal terms (AgentChord's rule, USER 11-09: xy <= 0.05, z <= 0.02, tilts < 45 deg) from the
              record's xy_err / z_err / tilt_deg / mug_tilt_deg -- NOT the record's own ac_success / success_ours.  A record with no `track` has unsafe unknown.
      ours    (probe e2e): 'outcome' carried | carry_failed | refused*, 'carry' {end_in_R, segments[].nominal.viol, pre_hold_viol}; no goal in the record -> goal
              missing.  pre_hold_viol is written by nero/contents/probe.py (a U entry during the load-transfer hold, carry_cert.py:331); an absent key = unknown.
      pan     (cert / runtime): 'segments' with 'nominal' {viol, success, status}; goal = the last segment's success flag.  Pre-hold is not recorded
              (unknown); a segment with no `status` has unknown finish."""
    if "pre_hold_viol" in rec and isinstance(rec.get("goal"), dict) and "xy_err" in rec["goal"]:      # theirs (old shape: goal = the raw measured dict)
        goal = _legacy_goal(rec)
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
    return classify(fin, missing("goal", "ours e2e records have no goal check"), unsafe if carry else None, pre, in_R, refused=oc_s.startswith(("refused", "plan_rejected")), reason=oc_s)


def result_of(rec: dict) -> dict:
    """the flat outcome record of a result record: rec['result'] when the record carries one, else read it from the old keys (load_legacy)."""
    return rec["result"] if isinstance(rec.get("result"), dict) else load_legacy(rec).record()


def tally(recs) -> dict:
    """the ONE tally of a list of result records (both methods): counts of each outcome field; None (unknown) is counted apart, never as False."""
    rs = [result_of(r) for r in recs if "error" not in r]
    c = lambda k, v: sum(r.get(k) is v for r in rs)
    return dict(n=len(rs), success=c("success", True), success_unknown=c("success", None), unsafe=c("unsafe", True), unsafe_unknown=c("unsafe", None),
                safe_success=c("safe_success", True), safe_success_unknown=c("safe_success", None), in_R=c("in_R", True), in_R_unknown=c("in_R", None),
                refused=sum(bool(r.get("refused")) for r in rs))
