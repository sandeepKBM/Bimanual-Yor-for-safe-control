"""checks/coverage.py -- the coverage row: every phenomenon in PHI(task) is measured (a skill ran) or on its approved fallback, else UNCOVERED -> refuse.  SIMULATION ONLY.   (11-09, build step 5)

Build spec 2026-11-09 s.2 item 6, ruling 11-09 Q1 (enforce mode):
  PHI(task)      the checker's OWN list (TaskDef.phenomena via skills/catalog.phi).  The LLM's phenomena list does NOT enter this check: it only changes which probes ran.
                 A phenomenon the LLM omits sits on its fallback; a phenomenon it names that is not in PHI is logged only (U1 default), never a row.
  measured       phi in ProbeRun.measured (a built skill ran and its reading was intersected into Theta_t).
  fallback       catalog.fallback(task, phi) with status "approved" (already in code, tagged there).  Status "out_of_scope" or no entry = NOT a fallback.
  uncovered      neither.  enforce mode -> the plan is refused ("refuse (uncovered)"); report mode only lists it.
  out of scope   PH4 slosh, PH5 hidden loose item, PH6 visible loose item (ruling 11-09 Q1 = option c) are kept out of PHI; every result carries catalog.OUT_OF_SCOPE_NOTE.
Theta_t = the probe run's parameter set (contents_mass_kg shrunk by the weigh; the other fields stay in the prior / on their fallback sets: bounds rule 4).
Pure module (stack_pure rule: it imports no skills/catalog): the catalog data comes in as arguments; `skills.catalog.coverage(task, run, ...)` is the convenience wrapper that supplies them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from world_model.v2.stack.checks.row import Row

MODES = ("enforce", "report")


@dataclass
class PhiRow:
    phi: str
    status: str                     # "measured" | "fallback" | "uncovered"
    by: str                         # skill id | fallback kind | "-"
    tag: str = ""                   # the fallback's MEASURED/MODEL/ASSUMED tag ("" for measured)
    note: str = ""


@dataclass
class Coverage:
    task: str
    mode: str
    rows: list = field(default_factory=list)            # list[PhiRow], in PHI order
    theta_t: dict = field(default_factory=dict)         # {field: (lo, hi)} the set the admission runs on
    named: list | None = None                           # the LLM's phenomenon ids (recorded, never used for the verdict)
    new_names: list = field(default_factory=list)       # NEW / non-PHI names the LLM gave (U1 default: logged only)
    out_of_scope: str = ""                              # catalog.OUT_OF_SCOPE_NOTE (supplied by the wrapper)

    @property
    def uncovered(self):
        return [r.phi for r in self.rows if r.status == "uncovered"]

    @property
    def covered(self):
        return not self.uncovered

    @property
    def refuses(self):
        """enforce mode and something uncovered."""
        return self.mode == "enforce" and not self.covered

    def to_dict(self):
        return dict(task=self.task, mode=self.mode, covered=self.covered, uncovered=self.uncovered,
                    rows=[dict(phi=r.phi, status=r.status, by=r.by, tag=r.tag, note=r.note) for r in self.rows],
                    theta_t={k: [float(v[0]), float(v[1])] for k, v in self.theta_t.items()}, named_by_llm=self.named, new_names=self.new_names,
                    out_of_scope=self.out_of_scope)


def check(task, run=None, theta0=None, *, phi, fallback, measures, mode="enforce", named=None, new_names=None, out_of_scope="") -> Coverage:
    """coverage of `task` after the probes.  run = a probing.ProbeRun (or None: nothing ran); theta0 = Theta_0 when there is no run; phi = PHI(task) (ids); fallback(phi) -> a
    Fallback (.kind .what .tag .status) or None; measures(phi) -> id of a built skill that measures it, or "-".  `named` is RECORDED only: the verdict is a function of
    (phi, fallback, run) alone."""
    if mode not in MODES:
        raise ValueError(f"coverage mode {mode!r} not in {MODES}")
    measured = set(run.measured) if run is not None else set()
    theta = dict(run.theta_t) if run is not None else dict(theta0 or {})
    rows = []
    for p in tuple(phi):
        if p in measured:
            rows.append(PhiRow(p, "measured", measures(p)))
            continue
        fb = fallback(p)
        if fb is not None and fb.status == "approved":
            rows.append(PhiRow(p, "fallback", fb.kind, fb.tag, fb.what))
        else:
            rows.append(PhiRow(p, "uncovered", "-", "", "no skill ran and no approved fallback" + ("" if fb is None else f" (fallback status {fb.status!r})")))
    return Coverage(task, mode, rows, theta, None if named is None else list(named), list(new_names or []), out_of_scope)


def row(cov: Coverage) -> Row:
    """the coverage row: ok iff nothing is uncovered; an uncovered phi is a MISSING row (never admits)."""
    if cov.covered:
        return Row("coverage", True, margin=None, unit="phenomena", guarantee="runtime", worst=dict(rows=[(r.phi, r.status) for r in cov.rows]),
                   assumptions=dict(out_of_scope=cov.out_of_scope), why="")
    return Row("coverage", False, margin=None, unit="phenomena", guarantee="missing", worst=dict(uncovered=cov.uncovered),
               assumptions=dict(out_of_scope=cov.out_of_scope), why=f"uncovered phenomena {cov.uncovered}: no skill ran and no approved fallback")


def contents_mass(cov: Coverage, field_name="contents_mass_kg"):
    """the contents-mass interval [kg] of the coverage set (the one Theta_t field the contents admission reads), or None if absent."""
    v = cov.theta_t.get(field_name)
    return None if v is None else (float(v[0]), float(v[1]))
