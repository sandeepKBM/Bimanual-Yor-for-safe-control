"""stack/probing.py -- the probe runner: runs the probes the plan selected, in order, on the current parameter set Theta_t.  SIMULATION ONLY.   (11-09, build step 4)

Core module: no nero / adapters / agentchord import.  Skills are passed in (duck-typed, see `ProbeSkill`); the adapter layer supplies the real ones (`weigh`, step 3).

  1. ORDER      skills/catalog.order_errors(probes): a skill whose precondition skill is not earlier in the list is REJECTED (ProbeOrderError); nothing is reordered or repaired.
  2. OWN LIMITS before each probe the skill's own limit rows are evaluated on the CURRENT Theta_t (skill.limit_rows(theta_t) -> rows with .ok / .name / .why, e.g. checks/probe_rung
     rows).  A row that fails -> the probe is SKIPPED and recorded (event "skipped" with the failing rows); its phenomena stay on their fallback.
  3. NO NEED    optional static_same(theta_t) -> True: the static verdict is the same on all of Theta_t, so a selected probe is not run (event "no_probe_needed").
  4. READING    skill.run(theta_t) -> Reading(fields {theta field: (lo, hi)}, output kind, detail).  Only the skill's declared theta_fields may appear in it.  The reading is
                INTERSECTED with Theta_t on those fields only (the prior box is never shrunk to model sensing: bounds rule 4).  Theta_t only shrinks.
  5. T3         a reading disjoint from the previous sound set (on any field, beyond TOL) = a CONTRADICTION: Theta_t is left as it was, the runner STOPS (no later probe runs) and
                records event "T3" (a counted event).  The reading is kept in the record.
Result: ProbeRun(theta_t, readings, events, measured (phenomena with a reading), skipped (phenomena of skipped / unrun probes), stopped).  Coverage (build step 5) reads
`measured`; everything else sits on its fallback.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from world_model.v2.stack.skills import catalog as CAT

TOL = 1e-9          # [same unit as the field] slack when testing a reading against the previous set (a numerical tolerance, not a physical bound)
OUTPUTS = ("interval", "lower_bound", "presence", "monitor_event")


class ProbeOrderError(ValueError):
    """the selected probe order violates a skill's preconditions (never repaired)."""

    def __init__(self, reasons):
        self.reasons = list(reasons)
        super().__init__("; ".join(self.reasons))


@dataclass
class Reading:
    skill: str
    fields: dict                  # {theta field: (lo, hi)}; one-sided: (lo, inf) or (-inf, hi)
    output: str = "interval"      # one of OUTPUTS
    detail: dict = field(default_factory=dict)


class ProbeSkill(Protocol):
    id: str
    def limit_rows(self, theta_t: dict) -> list: ...          # rows with .ok .name .why ; evaluated on Theta_t BEFORE run
    def run(self, theta_t: dict) -> Reading: ...


@dataclass
class ProbeRun:
    theta_t: dict                                    # {field: (lo, hi)} after the probes
    readings: list = field(default_factory=list)     # Reading, in run order
    events: list = field(default_factory=list)       # dicts: kind in {"ran", "skipped", "no_probe_needed", "T3", "not_run"}, skill, phenomena, ...
    measured: list = field(default_factory=list)     # phenomenon ids with a reading
    skipped: list = field(default_factory=list)      # phenomenon ids whose probe was skipped, not needed or never run (they go to their fallback)
    stopped: str | None = None                       # "T3" when a contradiction stopped the run

    def counts(self):
        c = {}
        for e in self.events:
            c[e["kind"]] = c.get(e["kind"], 0) + 1
        return c


def intersect(cur, new):
    """(lo, hi) intersection, or None when disjoint beyond TOL."""
    lo, hi = max(cur[0], new[0]), min(cur[1], new[1])
    if lo > hi + TOL:
        return None
    return (lo, max(lo, hi))


def _check_theta(theta):
    for k, v in theta.items():
        if not (len(v) == 2 and v[0] <= v[1]):
            raise ValueError(f"theta field {k!r}: need (lo, hi) with lo <= hi, got {v!r}")


def run_probes(probes, skills, theta_t, static_same=None, skill_catalog=None):
    """run the selected `probes` (skill ids, in the plan's order) on `theta_t` ({field: (lo, hi)}).  skills = {id: ProbeSkill}.  -> ProbeRun.
    static_same(theta_t) -> bool: True = the static verdict is the same on all of Theta_t (the probe is not run).  skill_catalog = {id: SkillEntry} (default: catalog.SKILLS)."""
    cat = skill_catalog if skill_catalog is not None else CAT.SKILLS
    errs = CAT.order_errors(list(probes)) if skill_catalog is None else _order_errors(probes, cat)
    for sid in probes:
        if sid not in cat or sid not in skills:
            errs.append(f"probe {sid!r}: not in the catalog / not supplied as a skill")
    if errs:
        raise ProbeOrderError(errs)
    theta = {k: (float(v[0]), float(v[1])) for k, v in theta_t.items()}
    _check_theta(theta)
    run = ProbeRun(theta_t=dict(theta))
    for idx, sid in enumerate(probes):
        entry, sk = cat[sid], skills[sid]
        ph = list(entry.measures)
        if run.stopped:
            run.events.append(dict(kind="not_run", skill=sid, phenomena=ph, why=f"stopped after {run.stopped}"))
            run.skipped += ph
            continue
        if static_same is not None and static_same(dict(run.theta_t)):
            run.events.append(dict(kind="no_probe_needed", skill=sid, phenomena=ph))
            run.skipped += ph
            continue
        bad = [r for r in (sk.limit_rows(dict(run.theta_t)) or []) if not r.ok]
        if bad:
            run.events.append(dict(kind="skipped", skill=sid, phenomena=ph, failed_rows=[dict(name=getattr(r, "name", ""), why=getattr(r, "why", ""), margin=getattr(r, "margin", None)) for r in bad]))
            run.skipped += ph
            continue
        rd = sk.run(dict(run.theta_t))
        err = _reading_errors(rd, sid, entry)
        if err:
            raise ValueError(f"skill {sid!r} returned an invalid reading: " + "; ".join(err))
        new, contra = dict(run.theta_t), []
        for f, iv in rd.fields.items():
            cur = run.theta_t.get(f)
            if cur is None:
                new[f] = (float(iv[0]), float(iv[1]))                 # a field Theta_t did not constrain yet: the reading is its first set
                continue
            x = intersect(cur, iv)
            if x is None:
                contra.append(dict(field=f, previous=list(cur), reading=[float(iv[0]), float(iv[1])]))
            else:
                new[f] = x
        run.readings.append(rd)
        if contra:                                                    # T3: stop, counted event; Theta_t unchanged
            run.events.append(dict(kind="T3", skill=sid, phenomena=ph, contradictions=contra, output=rd.output))
            run.stopped = "T3"
            continue
        for f in new:                                                 # Theta_t only shrinks
            c0 = run.theta_t.get(f)
            assert c0 is None or (new[f][0] >= c0[0] - TOL and new[f][1] <= c0[1] + TOL), f"Theta_t grew on {f}"
        run.theta_t = new
        run.measured += ph
        run.events.append(dict(kind="ran", skill=sid, phenomena=ph, output=rd.output, fields={f: list(v) for f, v in rd.fields.items()}, order=idx))
    return run


def _order_errors(probes, cat):
    seen, errs = [], []
    for i, sid in enumerate(probes):
        for pre in (cat[sid].preconditions if sid in cat else ()):
            if not pre.startswith("fact:") and pre not in seen:
                errs.append(f"probes[{i}]: skill {sid!r} needs {pre!r} to run before it")
        seen.append(sid)
    return errs


def _reading_errors(rd, sid, entry):
    errs = []
    if rd.skill != sid:
        errs.append(f"reading.skill {rd.skill!r} != {sid!r}")
    if rd.output not in OUTPUTS:
        errs.append(f"unknown output {rd.output!r}")
    for f, iv in rd.fields.items():
        if f not in entry.theta_fields:
            errs.append(f"field {f!r} is not one of the skill's theta_fields {list(entry.theta_fields)}")
        if not (len(iv) == 2 and iv[0] <= iv[1] and not any(isinstance(x, float) and math.isnan(x) for x in iv)):
            errs.append(f"field {f!r}: need (lo, hi) with lo <= hi, got {iv!r}")
    return errs
