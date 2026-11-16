"""checks/rule4.py -- Rule 4: no recovery is viable on Theta_t -> ask the proposer for a TARGET state to move to first
(checker design 2026-11-05 section 4.5; user rulings 2026-11-16: D2 R = R_K).  SIMULATION ONLY.  PURE: no physics (the callables are passed in).

check_target(target, transfer, theta_t, programs_at, branch_times, check_motion, landing_ok) -> (ok, rows, admission | None)
  (a) the TRANSFER (current state -> target) is admitted by Rule 1 applied to the transfer as a step:
      rule1.admit_step(0, theta_t, programs_at(target, "transfer"), branch_times, check_motion, landing_ok)
      = one program (the transfer motion + its landing at the target) passes on ALL of theta_t from every branch point of the transfer.
  (b) at the target a recovery is viable on theta_t:  rule1.admit_step(.., programs_at(target, "here"), ..) -- the programs that start AT the target
      (a rest program: no motion, landing = the target's landing), whose landing_ok row is S (rest set) and R_K (planner witness within depth K,
      node cap NODE_CAP = the landing's own rows, as in Rule 1).
  (c) only when `target.requires_next` (an in-air target: the arms still carry the object): Rule 1 holds on the NEXT step from the target
      (programs_at(target, "next") -> (programs, branch_times, check_motion, landing_ok)).  A target that rests the object on a support does not
      need it: the retry of the step is the orchestrator's next Rule 1 call (design 4.5 item 4), not part of check_target.
programs_at(target, phase): phase "transfer" | "here" -> [RecoveryProgram]; phase "next" -> (programs, times, check_motion, landing_ok) | None.
A target with no program for a required phase is a MISSING row (not viable; "not found" = not shown).
Guarantee of an admission = the weakest row of (a)+(b)+(c) (row.combine).

loop(state, theta_t, propose, check, max_rounds=3) -> (accepted target | None, log)
  propose(state, theta_t, failing_rows) -> list of candidate targets (failing_rows: the rows that refused the step in round 0, then the rows of the
  candidates refused in the previous round); check(target) -> (ok, rows, admission | None).  No candidate is checked twice from the same (state, theta_t)
  (design 4.5 item 4).  Every round is logged.  None after max_rounds = a certified refusal with the per-row reasons.
"""
from __future__ import annotations

from world_model.v2.stack.checks import rule1 as R1
from world_model.v2.stack.checks.row import Row, combine, missing

MAX_ROUNDS = 3        # ASSUMED (to be raised to the user): propose-check rounds before the refusal (design: N_prop / K attempts)


def _name(t):
    return getattr(t, "name", None) or str(t)


def _tag(rows, prefix):
    import dataclasses
    return [dataclasses.replace(r, name=f"{prefix}/{r.name}") for r in rows]


def _viable(step, theta_t, programs, times, cm, lo, prefix):
    """-> (ok, rows tagged, Admission | None).  The rows of the winning program, or those of every program when none is viable."""
    if not programs:
        return False, [missing(f"{prefix}/no_program", f"no recovery program is built for phase {prefix!r}")], None
    adm, per = R1.admit_step(step, theta_t, list(programs), list(times), cm, lo)
    if adm is not None:
        win = adm.recovery
        return True, _tag(per[win], prefix), adm
    rows = []
    for n, rs in per.items():
        rows += _tag(rs, f"{prefix}[{n}]")
    return False, rows, None


def check_target(target, transfer, theta_t, programs_at, branch_times, check_motion, landing_ok):
    """see the module doc.  `transfer` is the opaque transfer motion (carried into the admission record; programs_at builds its program)."""
    rows, parts, ok = [], {}, True
    ok_a, ra, adm_a = _viable(0, theta_t, programs_at(target, "transfer"), branch_times, check_motion, landing_ok, "transfer")
    rows += ra
    parts["transfer"] = adm_a
    ok &= ok_a
    ok_b, rb, adm_b = _viable(1, theta_t, programs_at(target, "here"), [0.0], check_motion, landing_ok, "at_target")
    rows += rb
    parts["at_target"] = adm_b
    ok &= ok_b
    if getattr(target, "requires_next", False):
        nxt = programs_at(target, "next")
        if nxt is None:
            ok_c, rc, adm_c = False, [missing("next_step/not_built", "no next-step programs for this target")], None
        else:
            progs, times, cm, lo = nxt
            ok_c, rc, adm_c = _viable(2, theta_t, progs, times, cm, lo, "next_step")
        rows += rc
        parts["next_step"] = adm_c
        ok &= ok_c
    if not ok:
        return False, rows, None
    verdict = combine(rows, "rule4")
    adm = dict(target=_name(target), transfer=transfer if isinstance(transfer, (str, int, float, dict, list, type(None))) else repr(transfer),
               guarantee=verdict.guarantee, parts={k: (None if v is None else v.to_json()) for k, v in parts.items()},
               assumptions=[dict(name="MAX_ROUNDS", value=MAX_ROUNDS, tag="ASSUMED", src="rule4.loop propose-check rounds (to be raised to the user)"),
                            dict(name="K", value=R1.K, tag="ASSUMED", src="rule1.K"), dict(name="NODE_CAP", value=R1.NODE_CAP, tag="ASSUMED", src="rule1.NODE_CAP")])
    return True, rows, adm


def loop(state, theta_t, propose, check, max_rounds=MAX_ROUNDS, failing_rows=()):
    """-> (accepted target | None, log).  log = [dict(round, failing_in, candidates, verdicts=[dict(target, ok, guarantee, failing)], accepted)]."""
    log, seen, fails = [], set(), list(failing_rows)
    for rnd in range(int(max_rounds)):
        cands = [c for c in propose(state, theta_t, fails) if _name(c) not in seen]
        ent = dict(round=rnd, failing_in=[_row_str(r) for r in fails], candidates=[_name(c) for c in cands], verdicts=[], accepted=None)
        log.append(ent)
        if not cands:
            ent["note"] = "the proposer offered no new target"
            break
        new_fails = []
        for c in cands:
            seen.add(_name(c))
            ok, rows, adm = check(c)
            bad = [r for r in rows if not r.ok]
            ent["verdicts"].append(dict(target=_name(c), ok=bool(ok), guarantee=None if adm is None else adm["guarantee"], failing=[_row_str(r) for r in bad]))
            if ok:
                ent["accepted"] = _name(c)
                return c, log
            new_fails += bad
        fails = new_fails
    return None, log


def _row_str(r) -> str:
    return f"{r.name}: {r.why or r.guarantee}"[:260]
