"""checks/rule1.py -- Rule 1: admit a step only if ONE recovery program is viable for EVERY theta in Theta_t from EVERY branch point of the step
(checker design 2026-11-05 section 4.3; user rulings 2026-11-16: exists-b-for-all-theta, D4 programs first, D2 R = R_K, D1 F = Theta_0 recorded only).
SIMULATION ONLY.  PURE: no physics.  The two physics callables are passed in (adapters/rule1_contents.py builds them for the contents carry):

  check_motion(plan_ref, theta_subset, t_branch) -> Row    a static (proof under MODEL) and/or sampled verdict for that motion on that set from that branch
  landing_ok(landing, theta_subset)              -> Row    the landing is in S (rest set) and in R_K (planner witness within depth K, node cap NODE_CAP)

A program (stack/theta.RecoveryProgram; a fixed plan is the one-bin program) is viable on Theta_t iff
  (i)   covers_readings(program, Theta_t): every reading any theta can produce falls in some bin        (one coverage row)
  (ii)  the prefix passes on ALL of Theta_t, from every branch point                                    (one row per branch time)
  (iii) for each bin j its plan passes on Theta_j (theta.bin_sets) from every branch point               (one row per bin and branch time)
  (iv)  the program's landing is in S and in R_K on every Theta_j                                         (one row per bin)
A row that cannot be evaluated is a MISSING row (ok False): a program with a missing row is NOT viable ("not found" = not shown, never a pass).
Guarantee of an admission = the weakest row (row.combine): proof if every row is proof, sampled if any row is sampled; refuted if any row fails.
"""
from __future__ import annotations

import dataclasses
import math

from world_model.v2.stack import theta as TH
from world_model.v2.stack.checks import payload as PAY
from world_model.v2.stack.checks.row import Row, combine

K = 3                  # ASSUMED (to be raised to the user): R_K depth -- the planner must find a sequence to the next subgoal of at most K admitted steps
NODE_CAP = 50          # ASSUMED (to be raised to the user): node cap of that search
BRANCH_DT = 0.25       # s, reused: executor.run_segments stores a branch snapshot every 0.25 s (design: carry_cert / executor.py:99)


def branch_times(duration: float, dt: float = BRANCH_DT) -> list:
    """the branch points of a step of this duration: 0, dt, 2 dt, ... (< duration), then the end of the step (the executor's `i > 0 and i % 50 == 0` loop,
    plus the step's start and its end state)."""
    out = [0.0]
    k = 1
    while k * dt < duration - 1e-9:
        out.append(round(k * dt, 9))
        k += 1
    out.append(float(duration))
    return out


# ----------------------------------------------------------------------------------------------------------------------- rows
def row_to_dict(r: Row) -> dict:
    """a JSON-safe dict of a Row (numpy scalars / arrays and nested dataclasses turned into plain python)."""
    return _clean(dataclasses.asdict(r))


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_clean(v) for v in x]
    if hasattr(x, "tolist"):
        return _clean(x.tolist())
    if isinstance(x, float) and not math.isfinite(x):
        return repr(x)
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return repr(x)


def _tagged(name, row: Row, what: str, t) -> Row:
    return dataclasses.replace(row, name=f"{name}:{what}" + ("" if t is None else f"@t={t:.2f}") + f"/{row.name}")


EMPTY_WHY = "theta_t is empty: contradictory evidence, no world is certified"


def _empty_row(name: str) -> Row:
    return Row(f"{name}:theta_t", False, margin=None, unit="", guarantee="refuted", worst=dict(cells=0), why=EMPTY_WHY,
               assumptions=dict(model="an empty belief set certifies nothing (never a vacuous admission)"))


def _rows(program: TH.RecoveryProgram, theta: TH.ThetaSet, times, check_motion, landing_ok) -> list:
    """[(t | None, Row)]: the rows of one program on theta.  t = the branch time a row belongs to (None = a row that holds for every branch time).
    An EMPTY theta gives one failing row (never a vacuous pass)."""
    if theta.is_empty():
        return [(None, _empty_row(program.name))]
    out = []
    ok, gaps = TH.covers_readings(program, theta)
    out.append((None, Row(f"{program.name}:coverage", bool(ok), margin=None, unit="kg", guarantee="proof" if ok else "refuted",
                          worst=dict(uncovered=[g.to_json() for g in gaps]),
                          assumptions=dict(model="reading bins cover every reading any theta in the set can produce (interval arithmetic, affine reading)"),
                          why="" if ok else f"reading intervals {[g.to_json() for g in gaps]} fall in no bin")))
    sets = TH.bin_sets(program, theta)
    for t in times:
        if program.prefix is not None:
            out.append((t, _tagged(program.name, check_motion(program.prefix, theta, t), "prefix", t)))
        for j, th_j, exact in sets:
            if th_j.is_empty():
                continue
            iv, plan = program.bins[j]
            what = "motion" if program.reading is None else f"bin{j}[{iv.lo:.3f},{iv.hi:.3f}]"
            r = _tagged(program.name, check_motion(plan, th_j, t), what, t)
            if not exact:
                r.assumptions = dict(r.assumptions, bin_set="a sound SUPERSET of Theta_j (reading depends on >= 2 fields)")
            out.append((t, r))
    for j, th_j, exact in sets:
        if th_j.is_empty():
            continue
        out.append((None, _tagged(program.name, landing_ok(program.landing, th_j), f"landing[bin{j}]", None)))
    return out


def program_viable(program: TH.RecoveryProgram, theta: TH.ThetaSet, branch_times, check_motion, landing_ok) -> list:
    """all rows of `program` on theta over the given branch times (see the module doc).  Viable iff every row is ok (a missing row is not ok)."""
    return [r for _, r in _rows(program, theta, branch_times, check_motion, landing_ok)]


def failing(rows) -> list:
    return [r for r in rows if not r.ok]


# ----------------------------------------------------------------------------------------------------------------------- admission
def _assumption(name, value, tag, src) -> dict:
    return dict(name=name, value=value, tag=tag, src=src)


def admit_step(step: int, theta_t: TH.ThetaSet, programs, branch_times, check_motion, landing_ok, fallback=None, theta_0=None, assumptions_extra=()):
    """Rule 1 for one step.  programs: recovery programs in the order of preference (first viable wins).  branch_times: the step's branch points.
    fallback (a RecoveryProgram) is evaluated on every cell of theta_0 with the same callables; `fallback_ok_on` records the union of the cells where
    it passes (D1: F = prior box; recorded, admission does NOT require it -- an open sub-point for K8).
    -> (theta.Admission | None, per_program_rows {name: [Row]}).  None = no program viable on theta_t: failing(per_program_rows[name]) are the Rule 4 input."""
    per, viable, times = {}, {}, list(branch_times)
    if theta_t.is_empty():
        er = {p.name: [_empty_row(p.name)] for p in programs} or {"theta_t": [_empty_row("rule1")]}
        return None, er
    for p in programs:
        tagged = _rows(p, theta_t, times, check_motion, landing_ok)
        per[p.name] = [r for _, r in tagged]
        viable[p.name] = tagged
    ok_names = [p.name for p in programs if per[p.name] and all(r.ok for r in per[p.name])]
    if not ok_names:
        return None, per
    win = next(p for p in programs if p.name == ok_names[0])
    table = {}
    for t in times:
        lst = []
        for p in programs:
            tr = viable[p.name]
            if all(r.ok for tt, r in tr if tt is None or tt == t):          # the rows of this branch time + the rows that hold for every time
                lst.append((p.name, theta_t))
        table[float(t)] = lst
    fb_ok = None
    if fallback is not None and theta_0 is not None and not theta_0.is_empty():
        good = []
        for c in theta_0.cells:
            rows = program_viable(fallback, TH.ThetaSet((c,)), times, check_motion, landing_ok)
            if all(r.ok for r in rows):
                good.append(c)
        fb_ok = TH.ThetaSet(tuple(good)) if good else None
    verdict = combine(per[win.name], "rule1")
    ass = [_assumption("K", K, "ASSUMED", "R_K depth (to be raised to the user)"),
           _assumption("NODE_CAP", NODE_CAP, "ASSUMED", "R_K node cap (to be raised to the user)"),
           _assumption("branch_dt", BRANCH_DT, "MODEL", "stack/executor.py run_segments: branch snapshot every 0.25 s"),
           _assumption("U_ADMIT", PAY.U_ADMIT, "MODEL", "checks/payload.U_ADMIT (rated limits, USER 11-09)"),
           _assumption("guarantee", verdict.guarantee, "MODEL", f"row.combine over {len(per[win.name])} rows of program {win.name!r} (weakest row)"),
           _assumption("fallback_F", "prior box Theta_0 (recorded, not required)", "ASSUMED", "user ruling 11-16 D1")]
    if win.reading is not None:
        ass.append(_assumption("reading_noise", win.reading.noise, win.reading.noise_tag, win.reading.noise_src))
    ass.extend(assumptions_extra)
    adm = TH.Admission(int(step), win.name, theta_t, table, fb_ok, [row_to_dict(r) for r in per[win.name]], ass)
    return adm, per
