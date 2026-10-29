"""Stage-A certificate for a candidate pan plan.  SIMULATION ONLY.

check(plan, db) -> Verdict.  Rules, in order (first rejection wins; each names its evidence class):
  R1 lid      (model)     a lifting arm holds the lid, and the lid is a separate free body in the MJCF -> lifting it does not
                          lift the pan.  Counterexample: the nominal theta (holds for every theta).
  R2 one-arm  (model + empirical capacity)  one claw on the handle: the pitch moment the claw must supply,
                          W*(u_grasp - u_com) from the compiled MJCF masses with the hidden mass anywhere in the pan, against the
                          measured claw capacity at the squeeze (0.05/0.14/0.22/0.30 Nm at 2.5/5/7.5/10 N; linear fit).
                          Counterexample: the theta of LEAST need if even that fails, else the theta of most need.
  R3 recorded (empirical) a recorded sim run with the same physical signature failed at a theta inside the plan's belief box.
  R4 closed world          a skeleton with no rule above and no Stage-B executor is rejected ("cannot check").
Admission by Stage A means "no known counterexample"; Stage B (simcheck) searches for new ones.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np

from . import domain as D
from . import evidence as E

CAPACITY_PTS = ((2.5, 0.05), (5.0, 0.14), (7.5, 0.22), (10.0, 0.30))     # MEASURED 10-28 (docs/status/pan_grasp_two_arm_2026-10-28.md;
                                    # empty pan, pinch 0.6, one run per squeeze): (squeeze N, claw pitch capacity Nm). A linear fit, NOT a bound.
_cf = np.polyfit([p[0] for p in CAPACITY_PTS], [p[1] for p in CAPACITY_PTS], 1)
PUCK_R, PAN_R_IN = 0.025, 0.09        # MODEL: hidden puck radius and pan inner radius in the MJCF
U_HIDDEN = PAN_R_IN - PUCK_R             # hidden puck centre can sit anywhere |u| <= 0.065 m from the pan centre

EXECUTABLE = {(("L", "handle"), ("R", "rim"))}      # skeletons Stage B can run (pan/under_rim.py)


def claw_capacity(F):
    return float(np.polyval(_cf, F))


@dataclass
class Verdict:
    ok: bool
    rule: str = ""
    evidence: str = ""
    detail: str = ""
    counterexample: dict | None = None
    support: dict = field(default_factory=dict)       # recorded runs inside the box: held / total
    assumptions: tuple = ()


@functools.lru_cache(maxsize=None)
def _pan_lid_statics(frac):
    """(moment of pan + lid about the grasp [Nm], u of the grasp from the pan centre [m]) from the compiled MJCF."""
    import dataclasses
    import mujoco
    from .. import covered as CV
    from .. import scene as S
    cp = CV.CoveredParams(placement="none")
    cp = dataclasses.replace(cp, pan=dataclasses.replace(cp.pan, handle_grasp_frac=frac))
    m = CV.compose(cp)
    d = mujoco.MjData(m)
    S.settle(m, d, 0.3)
    st = CV.probe_statics(m, d, cp.pan)
    u_g = st["lever_grasp_to_pivot_m"] - (cp.pan.pan_radius + cp.pan.pan_wall_t)
    return st["lift_handle_moment_Nm"], u_g


def one_arm_need(frac, mass_lo, mass_hi):
    """(min, max) claw moment needed over the hidden mass interval and position, and the arg thetas."""
    base, u_g = _pan_lid_statics(frac)
    g = 9.81
    need = lambda mh, uh: base + mh * g * (u_g - uh)
    lo = (need(mass_lo, U_HIDDEN), dict(hidden_mass=mass_lo, hidden_u=U_HIDDEN))
    hi = (need(mass_hi, -U_HIDDEN), dict(hidden_mass=mass_hi, hidden_u=-U_HIDDEN))
    return lo, hi


def lid_is_free_body():
    import mujoco
    from .. import covered as CV
    m = CV.compose(CV.CoveredParams(placement="none"))
    b = m.body("lid").id
    return any(m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]))


_LID_FREE = None


def check(plan, db):
    global _LID_FREE
    h = plan.holders()
    sig = plan.signature()
    box = plan.aim_belief()
    asm = box.assumptions
    nominal = dict(dx=0.0, dy=0.0, yaw_deg=0.0, mu_scale=1.0, hidden_mass=box.mass[0])
    if "lid" in h.values():
        if _LID_FREE is None:
            _LID_FREE = lid_is_free_body()
        if _LID_FREE:
            return Verdict(False, "R1_lid", "model", "lid is a separate free body (MJCF freejoint): lifting it does not lift the pan",
                           nominal, assumptions=asm)
    if len(h) == 1 and "handle" in h.values():
        _, pl = plan.act("pinch", part="handle")
        (n_lo, th_lo), (n_hi, th_hi) = one_arm_need(pl.frac if pl.arm == "L" else D.R_HANDLE_FRAC, *box.mass)
        cap = claw_capacity(D.GRIP_N)
        if n_lo > cap or n_hi > cap:
            th = th_lo if n_lo > cap else th_hi
            return Verdict(False, "R2_one_arm", "model+empirical",
                           f"claw must supply {n_lo:.2f}-{n_hi:.2f} Nm; capacity at {D.GRIP_N:.0f} N squeeze {cap:.2f} Nm",
                           dict(nominal, **th), assumptions=asm)
    fails = E.match(db, sig, box, failed=True) + E.match_pregrasp(db, sig, box)
    held = E.match(db, sig, box, failed=False)
    sup = dict(held=len(held), total=len(held) + len(fails))
    if fails:
        e = max(fails, key=lambda e: _depth(e["theta"], box))     # the deepest-inside failure: most robust counterexample
        return Verdict(False, "R3_recorded", "empirical",
                       f"{len(fails)} recorded failure(s) inside the box ({sup['held']}/{sup['total']} held); e.g. {e['cat']} in {e['src'].split('/')[-1]}",
                       dict(e["theta"], cat=e["cat"], src=e["src"]), sup, asm)
    if tuple(sorted(h.items())) not in EXECUTABLE:
        return Verdict(False, "R4_closed_world", "-", f"no rule and no executor for holders {h}", None, sup, asm)
    return Verdict(True, "admit_stageA", "empirical" if sup["total"] else "-",
                   f"no known counterexample; {sup['held']}/{sup['total']} recorded runs inside the box held", None, sup, asm)


def _depth(th, b):
    """how far inside the box a theta lies (min normalised slack); larger = more robust counterexample."""
    s = [1 - abs(th["dx"]) / max(b.dxy, 1e-9), 1 - abs(th["dy"]) / max(b.dxy, 1e-9), 1 - abs(th["yaw_deg"]) / max(b.yaw, 1e-9)]
    return min(s)
