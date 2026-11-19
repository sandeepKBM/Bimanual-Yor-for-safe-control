"""stack/outcome.py -- the ONE success rule and the outcome record.  SIMULATION ONLY.

USER 11-09 (R1): success = AgentChord's own rule, for every method: finished AND goal reached (goal terms of the task: tasks.py).  Safety is reported SEPARATELY:
  success       finished AND goal reached                         (True | False | None = unknown)
  unsafe        any entry into U, including before the hold      (True | False | None)
  in_R          the end state is at rest in R                    (a Row; in_R_ok in the record)
  safe_success  success AND not unsafe
A refusal is not a success.  Any ingredient that is UNKNOWN (finished / unsafe / pre_hold_unsafe None, or a missing goal Row) gives None unless something else already
failed (then False).  Unknown is never read as True or False.
"""
from __future__ import annotations

from dataclasses import dataclass

from world_model.v2.stack.checks.row import Row


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


def result_of(rec: dict) -> dict:
    """the flat outcome record of a result record: rec['result'].  A record without a "result" dict is refused (records of the pre-11-09 shapes are no longer read)."""
    if not isinstance(rec.get("result"), dict):
        raise ValueError("result record has no 'result' outcome dict (old record shape; the legacy loader was removed 11-19): re-run to get a current record")
    return rec["result"]


def tally(recs) -> dict:
    """the ONE tally of a list of result records (both methods): counts of each outcome field; None (unknown) is counted apart, never as False."""
    rs = [result_of(r) for r in recs if "error" not in r]
    c = lambda k, v: sum(r.get(k) is v for r in rs)
    return dict(n=len(rs), success=c("success", True), success_unknown=c("success", None), unsafe=c("unsafe", True), unsafe_unknown=c("unsafe", None),
                safe_success=c("safe_success", True), safe_success_unknown=c("safe_success", None), in_R=c("in_R", True), in_R_unknown=c("in_R", None),
                refused=sum(bool(r.get("refused")) for r in rs))
