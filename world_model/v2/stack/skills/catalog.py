"""stack/skills/catalog.py -- the phenomenon catalog, the measuring-skill entries, the fallback table and the LLM-facing cards.  SIMULATION ONLY.   (11-09)

Pure data + a few pure functions; imports only stack.types / stack.tasks / the two unified check modules for their constants (no nero, no adapters).
Build spec: docs/status/probe_selection_build_spec_2026-11-09.md (steps 1-2, 4).  The 11-08 ruling: the LLM names phenomena, skills supply every number.

  PHENOMENA   PH1..PH8 (design page ids).  PH4 slosh, PH5 hidden loose item, PH6 visible loose item are OUT OF SCOPE (ruling 11-09 Q1 = option c): no built skill measures them,
              no row covers them; they are kept OUT of PHI(task) (TaskDef.phenomena) and stated as a limit in every result (OUT_OF_SCOPE).
  SKILLS      SkillEntry per measuring skill.  Only status "built" (today: `weigh`, PH1) gets a card and can be selected.  Design entries are listed for the record.
  FALLBACK    what the checker assumes for a phenomenon no skill measured.  Every entry is already in code (no new assumption); status "approved" | "out_of_scope".
  cards(task) what the prompt shows: phenomenon cards (the object's kind) + built skill cards (robot facts: measure, output, resolution; never a contents range).

Every number is a `stack.types.Bound` with its tag (MEASURED / MODEL / ASSUMED) and src (AGENTS.md bounds rule).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from world_model.v2.stack import tasks as TK
from world_model.v2.stack.checks import payload as PAY
from world_model.v2.stack.checks import probe_rung as PRG
from world_model.v2.stack.types import Bound

INF = math.inf
NEW = "NEW"                       # the id an LLM gives to a phenomenon outside the catalog
OUT_OF_SCOPE = ("PH4", "PH5", "PH6")
OUT_OF_SCOPE_NOTE = ("OUT OF SCOPE (ruling 11-09 Q1, option c): PH4 liquid slosh, PH5 hidden loose item and PH6 visible loose item are not measured by any built skill and not "
                     "covered by any checker row; they are not in PHI(task), so a result says nothing about them.")


# ----------------------------------------------------------------------------------------------------------------------- phenomena
@dataclass(frozen=True)
class Phenomenon:
    id: str
    name: str
    sentence: str                  # shown to the LLM (a physical effect, no number)
    theta_fields: tuple            # names of the hidden-parameter fields it lives in (generic names; adapters map them to a universe's own theta)
    rows: tuple                    # checker rows (registry CHECKERS names) that bind on it
    objects: tuple                 # task kinds it can occur in ("box" | "tray" | "pot")
    in_scope: bool = True


PHENOMENA = {p.id: p for p in (
    Phenomenon("PH1", "total hidden mass", "The hidden contents add mass: the arms carry more than the empty object weighs.",
               ("contents_mass_kg",), ("payload", "payload_rest", "support", "probe_rung"), ("box", "tray", "pot")),
    Phenomenon("PH2", "horizontal centre of mass", "The contents sit off-centre, so the load is uneven between the two arms and puts a torque on the object about its grasp axis.",
               ("contents_com_xy",), ("payload", "support", "tilt"), ("box", "tray", "pot")),
    Phenomenon("PH3", "centre-of-mass height", "How high the contents sit changes how strongly accelerations and tilts load the grasp and tip the object.",
               ("contents_com_z",), ("tilt", "dynamic"), ("box", "tray", "pot")),
    Phenomenon("PH4", "liquid slosh", "Liquid in the mugs moves when the tray accelerates or tilts; the surface rises on one side.",
               ("liquid_slosh",), ("open_content",), ("tray",), in_scope=False),
    Phenomenon("PH5", "hidden loose item", "A loose, unseen item inside the object can slide or roll when the object moves.",
               ("loose_item_hidden",), (), ("box", "pot"), in_scope=False),
    Phenomenon("PH6", "visible loose item", "A loose, visible item (a mug, a capsule) rests on the object by friction only and can slide off.",
               ("loose_item_visible",), ("open_content",), ("tray",), in_scope=False),
    Phenomenon("PH7", "per-mug fill", "Each mug has its own unknown fill, so the mugs differ in weight and in how much can spill.",
               ("fill_each_kg",), ("open_content", "tilt"), ("tray",)),
    Phenomenon("PH8", "loose lid", "The lid rests on the pot by friction only and can slide off when the pot is accelerated or tilted.",
               ("lid_mu_scale",), ("dynamic",), ("pot",)),
)}


# ----------------------------------------------------------------------------------------------------------------------- skills
@dataclass(frozen=True)
class SkillEntry:
    id: str
    status: str                    # "built" | "design"
    measures: tuple                # phenomenon ids
    theta_fields: tuple            # the fields a reading intersects (and only these)
    output: str                    # "interval" | "lower_bound" | "presence" | "monitor_event"  (a skill may return the first two; the weigh does)
    range: Bound | None            # what the skill can read (robot fact)
    noise: dict                    # {"read": Bound, "residual": Bound, "propagation": str}
    guarantee: str
    preconditions: tuple           # skill ids that must run earlier, or state facts ("fact:...")
    own_limits: tuple              # ((row name, Bound), ...) checked by the checker on the current Theta_t BEFORE the skill runs
    needs_checker: bool
    one_sided: bool
    tracker: str
    cost: str
    calib: str
    code: str                      # "module:attr", resolved through stack.registry (MEASURE)
    card: dict = field(default_factory=dict)   # what the LLM sees (no contents prior, no hidden truth)


_CALIB = "world_model/v2/nero/results/probe_calib_poslift_2026-11-09/CALIB.json"
_WEIGH_TRACKER = "carry.LeverCtl (FL-MPC, pre-11-07): the poslift calibration was run under it (design D8); the numbers are conditional on it"

_EPS_LO = Bound(0.0, 0.0766, "MEASURED", f"{_CALIB} eps.EPS_LO: max over n = 20 worlds per task, noise-free reads; lower-bound residual after lift-off", "kg")
_EPS_FREE = Bound(0.0, 0.0316, "MEASURED", f"{_CALIB} eps.EPS_FREE: max over n = 20 worlds per task, noise-free reads; residual of a free interval", "kg")

WEIGH = SkillEntry(
    id="weigh", status="built", measures=("PH1",), theta_fields=("contents_mass_kg",), output="interval",
    range=Bound(0.0, 6.0, "MODEL", "2 x per-arm payload 3.0 kg (nero/config.py MANUAL['payload_kg'], datasheet); the per-rung admission row (checks/probe_rung) decides what a rung may push", "kg"),
    noise=dict(
        read=Bound(0.0, 0.05, "ASSUMED", "nero/contents/probe.py TAU_NOISE: uniform +-0.05 Nm per joint and sample (the NERO torque read itself is an open hardware fact)", "Nm"),
        residual=_EPS_LO, residual_free=_EPS_FREE,
        propagation="worst case of the per-sample torque noise through the arm Jacobian into the mass bound (MODEL, probe.py); added to the measured residuals"),
    guarantee="sampled: the interval is sound for the calibration worlds (n = 20 per task, uniform on the full prior); the model of the read is MODEL, the read noise ASSUMED",
    preconditions=("fact:object_grasped_on_start_support",),
    own_limits=(("rung_utilisation", Bound(-INF, PAY.U_ADMIT, "MODEL", "checks/payload.py U_ADMIT: the rated joint-torque / payload limits themselves, at the rest pose with the pushed mass", "ratio")),
                ("rung_overshoot_kg", Bound(-INF, PRG.O_MAX, "ASSUMED", "checks/probe_rung.py O_MAX: largest push above the current lower bound", "kg"))),
    needs_checker=True, one_sided=False, tracker=_WEIGH_TRACKER,
    cost="a few seconds of simulated time at the start pose, before any motion toward the goal",
    calib=_CALIB, code="world_model.v2.adapters.skills_contents:Weigh",
    card=dict(
        id="weigh", measures=["PH1"],
        does=("the two arms push the held object up a few millimetres in small steps and read the joint torques; this gives the TOTAL mass of the object. "
              "It reads no moments, so it does not give the position of the contents, their height, or how they are split."),
        output=("an interval on the total hidden mass, or only a lower bound when a larger push would exceed the arms' limits"),
        resolution=f"measured calibration residual (max over n = 20 worlds per task): {_EPS_LO.hi} kg on a lower bound, {_EPS_FREE.hi} kg on a free interval",
        limit_check="the checker tests every push against the arms' joint-torque limit before it runs and skips the weigh when a push is not admissible"))

DESIGN = (
    SkillEntry(id="hold_moment", status="design", measures=("PH2",), theta_fields=("contents_com_xy",), output="interval", range=None,
               noise=dict(read=None, residual=None, propagation="not built"), guarantee="missing", preconditions=("weigh",), own_limits=(),
               needs_checker=True, one_sided=False, tracker="", cost="not built", calib="", code=""),
    SkillEntry(id="shake_listen", status="design", measures=("PH5",), theta_fields=("loose_item_hidden",), output="presence", range=None,
               noise=dict(read=None, residual=None, propagation="not built"), guarantee="missing", preconditions=(), own_limits=(),
               needs_checker=True, one_sided=True, tracker="", cost="not built", calib="", code=""),
    SkillEntry(id="watch_item", status="design", measures=("PH6", "PH8"), theta_fields=("loose_item_visible", "lid_mu_scale"), output="monitor_event", range=None,
               noise=dict(read=None, residual=None, propagation="not built"), guarantee="missing", preconditions=(), own_limits=(),
               needs_checker=False, one_sided=False, tracker="", cost="not built (runtime monitor)", calib="", code=""),
)
SKILLS = {s.id: s for s in (WEIGH,) + DESIGN}


def built():
    """the skills an LLM may select."""
    return {k: s for k, s in SKILLS.items() if s.status == "built"}


def skill_for(phenomenon_id):
    """ids of the BUILT skills that measure `phenomenon_id`."""
    return [k for k, s in built().items() if phenomenon_id in s.measures]


# ----------------------------------------------------------------------------------------------------------------------- fallbacks
@dataclass(frozen=True)
class Fallback:
    kind: str                      # "prior_box" | "interior" | "fill_max" | "lid_row" | "none"
    what: str                      # the set or row the checker uses
    tag: str                       # MEASURED | MODEL | ASSUMED
    src: str
    status: str                    # "approved" (already in code) | "out_of_scope" (ruling 11-09 Q1 c)


_INT = Fallback("interior", "the contents prism P: the container interior (the COM may be anywhere inside)", "MODEL", "nero/contents/probe.py interval_spec (container interior)", "approved")
_PRIOR = Fallback("prior_box", "the task's contents prior box, tasks.prior_box(task)", "ASSUMED", "stack/tasks.py TaskDef.prior (one copy; user-approved 11-02)", "approved")
_OOS = Fallback("none", "no assumption and no row: out of scope (stated as a limit in the result)", "ASSUMED", "ruling 11-09 Q1 option c", "out_of_scope")


def _fb_table():
    t = {}
    for kind in ("box", "tray", "pot"):
        t[(kind, "PH1")] = _PRIOR
        t[(kind, "PH2")] = _INT
        t[(kind, "PH3")] = _INT
    t[("tray", "PH7")] = Fallback("fill_max", "each mug anywhere in [0, fill_max]; the total capped at min(fill_max, total fill hi)", "MODEL", "nero/contents/probe.py interval_spec; objects.CUP fill_max_kg", "approved")
    t[("pot", "PH8")] = Fallback("lid_row", "the lid-slide row with mu_lid = 0.5 x the dry value", "ASSUMED", "existing lid-slide row (mu_lid 0.5 ASSUMED)", "approved")
    for kind, ids in (("box", ("PH5",)), ("tray", ("PH4", "PH6")), ("pot", ("PH5",))):
        for i in ids:
            t[(kind, i)] = _OOS
    return t


FALLBACK = _fb_table()


def fallback(task, phenomenon_id):
    """Fallback for (task, phenomenon); the key is the task's KIND (behind tasks use their base object's kind).  None = none defined."""
    return FALLBACK.get((TK.task(task).kind, phenomenon_id))


# ----------------------------------------------------------------------------------------------------------------------- per task
def phi(task):
    """PHI(task): the checker's own coverage list (TaskDef.phenomena)."""
    return tuple(TK.task(task).phenomena)


def kind_of(task):
    return TK.task(task).kind


def order_errors(probes, selected=None):
    """probe order vs the skills' preconditions: a skill id in `preconditions` must appear EARLIER in `probes`.  -> list of reasons (never reorders or repairs).
    Unknown or design skills are the caller's error (schema); state facts ("fact:...") are checked by the executor, not here."""
    errs, seen = [], []
    for i, sid in enumerate(probes):
        s = SKILLS.get(sid)
        if s is not None:
            for pre in s.preconditions:
                if pre.startswith("fact:"):
                    continue
                if pre not in seen:
                    errs.append(f"probes[{i}]: skill {sid!r} needs {pre!r} to run before it" + ("" if pre in probes else f" ({pre!r} is not in probes)"))
        seen.append(sid)
    return errs


# ----------------------------------------------------------------------------------------------------------------------- cards (LLM-facing)
def phenomenon_card(p):
    return dict(id=p.id, name=p.name, what=p.sentence)


def cards(task):
    """what the prompt shows for `task`: one card per phenomenon that can occur in the task's object kind (in-scope or not: the model may name any), one per BUILT skill.
    Deterministic (sorted by id).  Holds no contents prior, no sanity range, no hidden truth."""
    k = kind_of(task)
    return dict(phenomena=[phenomenon_card(p) for _, p in sorted(PHENOMENA.items()) if k in p.objects],
                skills=[dict(s.card) for _, s in sorted(built().items())])


# ----------------------------------------------------------------------------------------------------------------------- coverage (checks/coverage.py is pure: it gets the catalog's data here)
def coverage(task, run=None, theta0=None, phi_override=None, fallbacks=None, mode="enforce", named=None, new_names=None):
    """checks.coverage.check on PHI(task) (override: tests) with this catalog's fallbacks (override {phi: Fallback}) and built skills."""
    from world_model.v2.stack.checks import coverage as COV
    fb = (lambda p: fallbacks.get(p)) if fallbacks is not None else (lambda p: fallback(task, p))
    return COV.check(task, run, theta0, phi=phi(task) if phi_override is None else tuple(phi_override), fallback=fb,
                     measures=lambda p: next((s for s, e in built().items() if p in e.measures), "-"), mode=mode, named=named, new_names=new_names, out_of_scope=OUT_OF_SCOPE_NOTE)


# ----------------------------------------------------------------------------------------------------------------------- audit helpers (tests)
def all_bounds():
    """(owner, Bound) for every number in the catalog."""
    out = []
    for s in SKILLS.values():
        if s.range is not None:
            out.append((f"{s.id}.range", s.range))
        for k in ("read", "residual", "residual_free"):
            if s.noise.get(k) is not None:
                out.append((f"{s.id}.noise.{k}", s.noise[k]))
        out += [(f"{s.id}.own_limits.{n}", b) for n, b in s.own_limits]
    return out
