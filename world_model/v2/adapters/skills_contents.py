"""adapters/skills_contents.py -- the measuring skills of the contents carry as stack ProbeSkills.  SIMULATION ONLY.   (11-09, probe-selection build step 3)

`Weigh` wraps today's probe (nero/contents/probe.make_probe; default method "poslift", env BIMANUAL_PROBE) without changing a number of it:
  limit_rows(theta_t)  the skill's own limit row BEFORE it runs: stack/checks/probe_rung.rung_row on the LIGHTEST total mass of Theta_t (the first push may not exceed it:
                       overshoot 0 by construction; payload at the rest pose u <= U_ADMIT, MODEL).  The probe's internal per-rung admission (Probe.admissible, rung_torque_u)
                       stays as it is.
  run(theta_t)         runs the probe (Probe.run(M_stop)) and returns probing.Reading("weigh", {"contents_mass_kg": (lo, hi)}, "interval" | "lower_bound", detail):
                         interval     the probe stopped "free": the object hung free, both bounds read (contents mass = total M - the known empty mass).
                         lower_bound  any other stop (decided_refuse, rung_not_admissible, load_guard, z_cap ...): only the lower bound is read; hi = inf (one-sided, the previous
                                      Theta_t hi stays); a lower-bound-only reading is Rule 2 T2 when the admission then fails (probe_select).
                       detail carries the probe's `why`, the total mass interval [M_lo, M_hi] exactly as the probe holds it (bit-identical to today's record Theta), and shows=["PH1"].
The skill measures the TOTAL mass only (PH1, ruling 11-09): no moments (PH2), no height (PH3), no split (PH7).  The episode `ep` is the carry episode the carry continues in
(the probe leaves the claws holding at the start pose with the feed-forward `probe.m_ff`; the caller ramps it).
Bounds: EPS_LO / EPS_FREE come from the poslift CALIB (MEASURED, nero/results/probe_calib_poslift_adaptive_2026-11-09; n = 20 per task); TAU_NOISE 0.05 Nm ASSUMED (probe.py);
see stack/skills/catalog.WEIGH for the tagged entry.
"""
from __future__ import annotations

import math

from world_model.v2.nero.contents import probe as P
from world_model.v2.stack.checks import probe_rung as PRG
from world_model.v2.stack.probing import Reading

FIELD = "contents_mass_kg"


class _Rest:
    """payload.Rest backend: u at the rest pose carrying contents mass c (COM anywhere in the interior) = probe.rung_torque_u (MODEL)."""

    def __init__(self, task):
        self.task, self.e = task, P.empty_mass(task)

    def u(self, c):
        return P.rung_torque_u(self.task, self.e + float(c))


class Weigh:
    id = "weigh"

    def __init__(self, ep, task, eps_lo, eps_free, rng=None, method=None, adaptive=True, noise=True, m_stop=None):
        """task = the probe task key ("box" | "tray6" | "pot": a table-behind task passes its base task).  m_stop = empty mass + the heaviest admissible contents mass of the plan
        (Probe.run stops "decided_refuse" above it); settable before run (the plan may change at a re-ask)."""
        self.ep, self.task = ep, task
        self.e = P.empty_mass(task)
        self.probe = P.make_probe(ep, task, eps_lo, eps_free, rng=rng, noise=noise, method=method, adaptive=adaptive)
        self.m_stop = m_stop
        self.why = None
        self.reading = None

    def limit_rows(self, theta_t):
        lo = float(theta_t[FIELD][0])
        return [PRG.rung_row(_Rest(self.task), self.e, self.e + lo, self.e + lo)]

    def run(self, theta_t):
        cur = theta_t[FIELD]
        pr, e = self.probe, self.e
        pr.M_lo, pr.M_hi = e + float(cur[0]), e + float(cur[1])        # the same expression as Probe.__init__ for the prior: bit-identical when Theta_t is the prior
        m_lo0, m_hi0 = pr.M_lo, pr.M_hi
        self.why = why = pr.run(M_stop=self.m_stop)
        free = why == "free"
        c_lo = float(cur[0]) if pr.M_lo == m_lo0 else pr.M_lo - e       # untouched bound: keep Theta_t's own value (no M - e + e round-off)
        c_hi = float(cur[1]) if pr.M_hi == m_hi0 else pr.M_hi - e
        detail = dict(why=why, M_lo=pr.M_lo, M_hi=pr.M_hi, n_rungs=len(pr.rungs), shows=["PH1"], probe_method=P.method_of(pr))
        if c_lo > c_hi:                                                 # internally inconsistent read (lower bound above upper bound): cannot be a Reading; the job counts it as T3
            detail["inconsistent"] = [c_lo, c_hi]
            self.reading = Reading("weigh", {}, "interval", detail)
        else:
            self.reading = Reading("weigh", {FIELD: (c_lo, c_hi if free else math.inf)}, "interval" if free else "lower_bound", detail)
        return self.reading
