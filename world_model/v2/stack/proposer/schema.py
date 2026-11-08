"""stack/proposer/schema.py -- the JSON the proposer model must return, its skill vocabulary and a STRICT validator.  SIMULATION ONLY.   (11-05)

USER 11-05: "The plan to be done when running should be authored by the codex gpt ones, not us."  The model writes, per task:
  belief      B = {param: [lo, hi]} over the task's hidden parameters (PARAMS[task])
  steps       [{skill, args, why}]            the high-level plan, a sequence of SEGMENTS (one object-level motion each) ending in place + release
  recoveries  {"<step index>": [{skill, args, covers?}]}   for every motion step: what to do if the step has to be abandoned (recovery skills below)
  fallback    [{skill, args}]                 what to do when no plan segment is admitted (nothing has moved yet)
Anything that does not validate is `plan_rejected` with the reason(s) (PlanRejected).  A reply is NEVER repaired: no clipping, no defaults, no dropped keys.

Skill vocabulary = what the low-level planner (stack/planners/closed_chain.py WaypointPlan + nero/contents/carry.py backups) can execute today.  Object-level,
the object stays level (both claws are welded to it, ASSUMED rigid grasp), straight-line translations in the world frame:
  lift(dz)            object up by dz              dz in [0.02, 0.20] m
  move(dx, dy, dz)    object translation           |dx|, |dy| <= 0.30 m, |dz| <= 0.20 m
  place(support)      lower straight down onto a support ("counter" | "shelf" where the scene has it) until it rests
  release()           let go and retreat the claws (last step; letting go is not undoable)
Recovery skills (no arguments; run from the commanded state inside the step):
  hold()              stop and keep holding
  retrace()           stop, go back along the executed path to the START of this step, keep holding
  set_down()          stop, retrace this step and all earlier steps to the start of the first, let go there (the object is back on its start support)
  let_go()            stop and let go in place
Argument ranges (labels, AGENTS.md bounds rule):  the dz / dx / dy ranges are ASSUMED outer ranges for ONE motion (MODEL anchors: the counter is 0.40 x 0.30 m,
scenes.Common; the box lift of 0.18 m reaches and 0.20 m does not, scenes.BoxWorld.shelf_top comment, 11-02); whether a plan is reachable is decided per plan by
the planner (PlanRefused), not by these ranges.  PARAMS ranges are SANITY limits of the hidden parameters (MODEL from the scene geometry where stated, else ASSUMED),
NOT the prior box: the prompt does not contain the prior.
Not offered (the planner has no such input): a speed / acceleration request (the admission chooses a_admit), rotations, per-arm motions, regrasp, push, base motion.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from world_model.v2.stack.types import SkillCall

TASKS = ("box", "tray2", "tray4", "tray6", "pot")
MAX_STEPS = 8                  # ASSUMED: longest plan accepted (steps incl. place + release)
MAX_WHY = 400                  # ASSUMED: characters of the free-text reason per step

# ---- skills --------------------------------------------------------------------------------------------------------------------
STEP_SKILLS = {                # name -> {arg: (lo, hi)}; "place": {"support": enum}; tags in the module doc
    "lift": {"dz": (0.02, 0.20)},
    "move": {"dx": (-0.30, 0.30), "dy": (-0.30, 0.30), "dz": (-0.20, 0.20)},
    "place": {"support": ("counter", "shelf")},
    "release": {},
}
MOTION_SKILLS = ("lift", "move", "place")
RECOVERY_SKILLS = ("hold", "retrace", "set_down", "let_go")
RANGE_TAG = "ASSUMED outer range per motion; reach is decided by the planner (PlanRefused)"


def vocabulary(supports=None):
    """JSON-able description of the vocabulary (what the prompt shows).  supports = the support names of the scene (default: all enum values)."""
    steps = {k: {n: (list(supports) if (k == "place" and supports is not None) else list(r)) for n, r in a.items()} for k, a in STEP_SKILLS.items()}
    return dict(steps=steps, recoveries=list(RECOVERY_SKILLS), tags=dict(ranges=RANGE_TAG))


# ---- hidden parameters (belief) ---------------------------------------------------------------------------------------------------
def params(task):
    """{name: dict(range=(lo, hi), unit, doc, probed, tag)} -- sanity range of each hidden parameter of the task (NOT the prior box)."""
    from world_model.v2.nero.contents import objects as OB, scenes as SC
    if task == "box":
        w = SC.BoxWorld()
        hx, hy = w.box_outer[0] / 2 - w.wall_t - w.contents_half[0], w.box_outer[1] / 2 - w.wall_t - w.contents_half[1]
        return {"contents_kg": dict(range=(0.0, 20.0), unit="kg", probed=True, tag="ASSUMED sanity range",
                                    doc="mass of the hidden contents inside the box (the empty crate is known)"),
                "contents_x": dict(range=(-hx, hx), unit="m", probed=False, tag="MODEL: crate interior minus the contents block",
                                   doc="x of the contents centre in the crate frame (crate centre = 0)"),
                "contents_y": dict(range=(-hy, hy), unit="m", probed=False, tag="MODEL: crate interior minus the contents block",
                                   doc="y of the contents centre in the crate frame (crate centre = 0)")}
    if task.startswith("tray"):
        n = int(task[4:])
        return {"fill_total_kg": dict(range=(0.0, n * OB.CUP["fill_max_kg"]), unit="kg", probed=True, tag="MODEL: n mugs x mug capacity (objects.CUP fill_max_kg)",
                                      doc=f"total liquid in the {n} mugs together (1 L of liquid = 1 kg); the empty mugs and tray are known")}
    if task == "pot":
        return {"puck_kg": dict(range=(0.0, 10.0), unit="kg", probed=True, tag="ASSUMED sanity range", doc="mass of the hidden contents inside the pot (the empty pot with lid is known)"),
                "puck_u": dict(range=(-0.06, 0.06), unit="m", probed=False, tag="MODEL: contents disc radius C_R = 0.06 m (pan/theta.py)", doc="contents centre offset along the handle axis (y)"),
                "puck_w": dict(range=(-0.06, 0.06), unit="m", probed=False, tag="MODEL: contents disc radius C_R = 0.06 m (pan/theta.py)", doc="contents centre offset across the handles (x)"),
                "lid_mu_scale": dict(range=(0.1, 1.5), unit="-", probed=False, tag="ASSUMED sanity range", doc="friction of the loose lid on the pot, as a multiple of the dry value (1.0 = dry)")}
    raise ValueError(f"unknown task {task!r}; have {TASKS}")


def probe_offset(task):
    """contents-mass interval of the probe (total contents mass c, nero/contents/probe.py) -> value of the probed belief parameter: (param name, offset subtracted)."""
    from world_model.v2.nero.contents import objects as OB
    if task == "box":
        return "contents_kg", 0.0
    if task == "pot":
        return "puck_kg", 0.0
    return "fill_total_kg", int(task[4:]) * OB.CUP["empty"]


# ---- result types ------------------------------------------------------------------------------------------------------------------
class PlanRejected(Exception):
    """a reply / plan that cannot be used.  stage = parse | schema | compile | planner; reasons = list of strings (never empty)."""

    def __init__(self, stage, reasons):
        self.stage = stage
        self.reasons = [str(r) for r in ([reasons] if isinstance(reasons, str) else reasons)]
        super().__init__(f"plan_rejected[{stage}]: " + "; ".join(self.reasons))

    def to_dict(self):
        return dict(status="plan_rejected", stage=self.stage, reasons=self.reasons)


@dataclass
class RecoveryCall:
    skill: str
    args: dict = field(default_factory=dict)
    covers: dict | None = None                       # {param: [lo, hi]} the part of B it is written for (the proposer's CLAIM; nobody verified it yet)


@dataclass
class Step:
    call: SkillCall
    why: str
    recoveries: list = field(default_factory=list)   # list[RecoveryCall] (alternatives, first = the one the executor runs today)


@dataclass
class TaskPlan:
    task: str
    belief: dict                                     # {param: [lo, hi]}
    steps: list                                      # list[Step]
    fallback: list                                   # list[RecoveryCall]
    proposer: dict = field(default_factory=dict)     # model, reasoning level, caveats (bounds rule 9); filled by propose()

    def to_dict(self):
        return dict(status="ok", task=self.task, belief=self.belief,
                    steps=[dict(skill=s.call.skill, args=s.call.args, why=s.why) for s in self.steps],
                    recoveries={str(i): [dict(skill=r.skill, args=r.args, **({"covers": r.covers} if r.covers else {})) for r in s.recoveries]
                                for i, s in enumerate(self.steps) if s.recoveries},
                    fallback=[dict(skill=r.skill, args=r.args) for r in self.fallback])


# ---- parsing -------------------------------------------------------------------------------------------------------------------------
def parse_reply(text):
    """the LAST top-level JSON object in the reply text (AGENTS.md: reasoning replies may contain earlier objects).  No repair: a reply without a decodable
    JSON object raises PlanRejected('parse')."""
    if not isinstance(text, str) or not text.strip():
        raise PlanRejected("parse", "empty reply")
    dec, i, last = json.JSONDecoder(), 0, None
    while i < len(text):
        if text[i] == "{":
            try:
                obj, end = dec.raw_decode(text, i)
                if isinstance(obj, dict):
                    last = obj
                i = end
                continue
            except ValueError:
                pass
        i += 1
    if last is None:
        raise PlanRejected("parse", "no JSON object in the reply")
    return last


# ---- validation --------------------------------------------------------------------------------------------------------------------
def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _interval(v, lo_lim, hi_lim, what, errs):
    if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(_num(x) for x in v)):
        errs.append(f"{what}: must be [lo, hi] with two finite numbers, got {v!r}")
        return None
    lo, hi = float(v[0]), float(v[1])
    if lo > hi:
        errs.append(f"{what}: lo {lo} > hi {hi}")
        return None
    if lo < lo_lim - 1e-12 or hi > hi_lim + 1e-12:
        errs.append(f"{what}: [{lo}, {hi}] outside the sanity range [{lo_lim:.4g}, {hi_lim:.4g}]")
        return None
    return [lo, hi]


def _keys(d, required, optional, what, errs):
    if not isinstance(d, dict):
        errs.append(f"{what}: must be an object, got {type(d).__name__}")
        return False
    miss, extra = [k for k in required if k not in d], [k for k in d if k not in required and k not in optional]
    if miss:
        errs.append(f"{what}: missing keys {miss}")
    if extra:
        errs.append(f"{what}: unknown keys {extra}")
    return not miss and not extra


def validate(d, task, supports=("counter", "shelf")):
    """strict validation of the parsed reply `d` for `task` -> TaskPlan, or raises PlanRejected('schema') listing EVERY problem found.  supports = the support names
    of the scene (place(support) must name one)."""
    if task not in TASKS:
        raise PlanRejected("schema", f"unknown task {task!r}")
    errs = []
    if not _keys(d, ("belief", "steps", "recoveries", "fallback"), (), "reply", errs):
        raise PlanRejected("schema", errs)
    # belief
    P = params(task)
    belief = {}
    if _keys(d["belief"], tuple(P), (), "belief", errs):
        for k, spec in P.items():
            iv = _interval(d["belief"][k], *spec["range"], f"belief.{k}", errs)
            if iv is not None:
                belief[k] = iv
    # steps
    steps, raw = [], d["steps"]
    if not isinstance(raw, list) or not raw:
        errs.append("steps: must be a non-empty list")
        raw = []
    if len(raw) > MAX_STEPS:
        errs.append(f"steps: {len(raw)} steps, at most {MAX_STEPS}")
    for i, s in enumerate(raw[:MAX_STEPS]):
        w = f"steps[{i}]"
        if not _keys(s, ("skill", "args", "why"), (), w, errs):
            continue
        sk = s["skill"]
        if sk not in STEP_SKILLS:
            errs.append(f"{w}.skill: unknown skill {sk!r}; have {sorted(STEP_SKILLS)}")
            continue
        if not (isinstance(s["why"], str) and s["why"].strip() and len(s["why"]) <= MAX_WHY):
            errs.append(f"{w}.why: must be a non-empty string of at most {MAX_WHY} characters")
        spec, args = STEP_SKILLS[sk], s["args"]
        if not _keys(args, tuple(spec), (), f"{w}.args", errs):
            continue
        ok = True
        for a, rng in spec.items():
            if sk == "place":
                if args[a] not in supports:
                    errs.append(f"{w}.args.support: {args[a]!r} is not a support of this scene; have {list(supports)}")
                    ok = False
            elif not _num(args[a]):
                errs.append(f"{w}.args.{a}: must be a finite number, got {args[a]!r}")
                ok = False
            elif not rng[0] - 1e-12 <= args[a] <= rng[1] + 1e-12:
                errs.append(f"{w}.args.{a}: {args[a]} outside [{rng[0]}, {rng[1]}] m")
                ok = False
        if ok:
            steps.append(Step(SkillCall(sk, {k: (float(v) if _num(v) else v) for k, v in args.items()}), s["why"].strip()))
    if len(steps) == len(raw) and raw:                       # shape rules only when every step parsed
        sk = [s.call.skill for s in steps]
        if sk[-1] != "release":
            errs.append("steps: the last step must be release()")
        if sk.count("release") != 1:
            errs.append(f"steps: exactly one release() allowed, found {sk.count('release')}")
        if len(sk) < 2 or sk[-2] != "place":
            errs.append("steps: the step before release() must be place(support)")
        if sk.count("place") != 1:
            errs.append(f"steps: exactly one place() allowed, found {sk.count('place')}")
    # recoveries
    rec = d["recoveries"]
    if not isinstance(rec, dict):
        errs.append("recoveries: must be an object {step index: [recovery, ...]}")
        rec = {}
    for k in rec:
        if not (isinstance(k, str) and k.isdigit() and int(k) < len(raw) and isinstance(raw[int(k)], dict) and raw[int(k)].get("skill") in MOTION_SKILLS):
            errs.append(f"recoveries.{k}: key must be the index of a motion step (lift / move / place), not {k!r}")
    for i, s in enumerate(steps):
        if s.call.skill in MOTION_SKILLS:
            lst = rec.get(str(i))
            if not isinstance(lst, list) or not lst:
                errs.append(f"recoveries.{i}: motion step {i} ({s.call.skill}) needs at least one recovery")
                continue
            for j, r in enumerate(lst):
                s.recoveries.append(_recovery(r, f"recoveries.{i}[{j}]", belief, errs, with_covers=True))
    # fallback
    fb, fbl = d["fallback"], []
    if not isinstance(fb, list) or not fb or len(fb) > 3:
        errs.append("fallback: must be a list of 1..3 recovery calls")
    else:
        for j, r in enumerate(fb):
            fbl.append(_recovery(r, f"fallback[{j}]", belief, errs, with_covers=False))
    if errs:
        raise PlanRejected("schema", errs)
    return TaskPlan(task, belief, steps, [r for r in fbl if r is not None])


def _recovery(r, what, belief, errs, with_covers):
    if not _keys(r, ("skill", "args"), ("covers",) if with_covers else (), what, errs):
        return None
    if r["skill"] not in RECOVERY_SKILLS:
        errs.append(f"{what}.skill: unknown recovery skill {r['skill']!r}; have {list(RECOVERY_SKILLS)}")
        return None
    if r["args"] != {}:
        errs.append(f"{what}.args: recovery skills take no arguments, got {r['args']!r}")
        return None
    covers = None
    if with_covers and "covers" in r:
        c = r["covers"]
        if not isinstance(c, dict) or not c:
            errs.append(f"{what}.covers: must be a non-empty object {{param: [lo, hi]}}")
            return None
        covers = {}
        for k, v in c.items():
            if k not in belief:
                errs.append(f"{what}.covers.{k}: not a (valid) belief parameter")
                continue
            iv = _interval(v, *belief[k], f"{what}.covers.{k}", errs)         # inside B
            if iv is not None:
                covers[k] = iv
    return RecoveryCall(r["skill"], {}, covers)


def validate_reply(text, task, supports=("counter", "shelf")):
    """parse + validate a raw reply text -> TaskPlan (raises PlanRejected)."""
    return validate(parse_reply(text), task, supports)


def plan_id(plan_dict):
    import hashlib
    return hashlib.sha256(json.dumps(plan_dict, sort_keys=True).encode()).hexdigest()[:12]

