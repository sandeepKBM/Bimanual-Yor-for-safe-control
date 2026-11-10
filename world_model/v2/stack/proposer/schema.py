"""stack/proposer/schema.py -- the JSON the proposer model must return, its skill vocabulary and a STRICT validator.  SIMULATION ONLY.   (11-05)

SCHEMA v2 (11-09, probe-selection build spec; default): the reply is {phenomena, probes, steps, recoveries, fallback} -- the model NAMES the hidden-contents phenomena
(skills/catalog.py PH1..PH8 or "NEW") with a handle each ({"measure": skill id} | {"unmeasurable": reason}) and lists the probes to run, in order; it states NO number about the contents
(no belief, no covers; 11-08: the skills supply the numbers).  The v1 format below (belief + covers) is replayable with BIMANUAL_LEGACY_PROPOSER_V1=1.

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
Table-behind tasks (tray6_behind, box_behind; BASE_TASKS) also offer
  base_move(route)    drive the whole robot: route = 1-4 segments {"line": metres along the robot's current +x} | {"turn": degrees CCW, in place}; |line| <= 1.0 m,
                      |turn| <= 180 deg (ASSUMED outer ranges, design 2026-11-09 A3); the arms hold still, the object rides along; at most one per plan, after the lift.
                      A motion step (needs a recovery: hold = stop the base, retrace = drive the route back, set_down = retrace everything; let_go is accepted here and
                      refused by the checker).  No speed argument: the admission picks the base level.  Fixed-base tasks do not see it.
Not offered (the planner has no such input): a speed / acceleration request (the admission chooses a_admit), rotations, per-arm motions, regrasp, push.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

from world_model.v2.stack.types import SkillCall

V1_KEYS = ("belief", "steps", "recoveries", "fallback")                       # the 11-05..11-09 reply
V2_KEYS = ("phenomena", "probes", "steps", "recoveries", "fallback")           # schema v2 (11-09): the model names phenomena and probes, no number about the contents


def legacy_v1():
    """BIMANUAL_LEGACY_PROPOSER_V1=1: the 11-05..11-09 format (belief + covers) -- prompts, validator and mock replies -- so the old response caches still replay."""
    return os.environ.get("BIMANUAL_LEGACY_PROPOSER_V1") == "1"


def reply_keys():
    return V1_KEYS if legacy_v1() else V2_KEYS


def reply_of(plan):
    """the reply-format part of a plan dict (propose() output): only the keys of the current format."""
    d = {k: plan[k] for k in reply_keys() if k in plan}
    if isinstance(d.get("probes"), list):                     # a plan dict (TaskPlan.to_dict) lists probe skill ids; the reply format is [{"skill": id}] (11-09: the round trip was missing)
        d["probes"] = [{"skill": q} if isinstance(q, str) else q for q in d["probes"]]
    return d


TASKS = ("box", "tray2", "tray4", "tray6", "pot", "tray6_behind", "box_behind")
BASE_TASKS = ("tray6_behind", "box_behind")      # TaskDef.base == "swerve": the object goes to the counter BEHIND the robot; the base can move (base_move)
MAX_STEPS = 8                  # ASSUMED: longest plan accepted (steps incl. place + release)
MAX_WHY = 400                  # ASSUMED: characters of the free-text reason per step

# ---- skills --------------------------------------------------------------------------------------------------------------------
STEP_SKILLS = {                # name -> {arg: (lo, hi)}; "place": {"support": enum}; tags in the module doc
    "lift": {"dz": (0.02, 0.20)},
    "move": {"dx": (-0.30, 0.30), "dy": (-0.30, 0.30), "dz": (-0.20, 0.20)},
    "place": {"support": ("counter", "shelf")},
    "release": {},
}
BASE_STEP_SKILLS = {           # only offered for BASE_TASKS (fixed-base plans / prompts are unchanged).  The arg is a list, validated by _route_errors
    "base_move": {"route": dict(segments="1..4", line_m=[-1.0, 1.0], turn_deg=[-180.0, 180.0])},
}
BASE_TAG = "ASSUMED outer ranges (design 2026-11-09 A3: <= 4 segments, |line| <= 1.0 m, |turn| <= 180 deg); the sweep clearance and reach are decided by the planner (PlanRefused)"
MOTION_SKILLS = ("lift", "move", "place", "base_move")
RECOVERY_SKILLS = ("hold", "retrace", "set_down", "let_go")
RANGE_TAG = "ASSUMED outer range per motion; reach is decided by the planner (PlanRefused)"


def step_skills(task=None):
    """the step skills offered for `task`: STEP_SKILLS, plus base_move for the table-behind tasks."""
    return {**STEP_SKILLS, **BASE_STEP_SKILLS} if task in BASE_TASKS else dict(STEP_SKILLS)


def vocabulary(supports=None, base=False):
    """JSON-able description of the vocabulary (what the prompt shows).  supports = the support names of the scene (default: all enum values).  base=True adds base_move."""
    steps = {k: {n: (list(supports) if (k == "place" and supports is not None) else list(r)) for n, r in a.items()} for k, a in STEP_SKILLS.items()}
    tags = dict(ranges=RANGE_TAG)
    if base:
        steps.update({k: {n: dict(r) for n, r in a.items()} for k, a in BASE_STEP_SKILLS.items()})
        tags["base_move"] = BASE_TAG
    return dict(steps=steps, recoveries=list(RECOVERY_SKILLS), tags=tags)


def _fixed(task):
    """the fixed-base task a table-behind task is built on ("tray6_behind" -> "tray6"): same hidden parameters."""
    return task[:-len("_behind")] if task in BASE_TASKS else task


# ---- hidden parameters (belief) ---------------------------------------------------------------------------------------------------
def params(task):
    """{name: dict(range=(lo, hi), unit, doc, probed, tag)} -- sanity range of each hidden parameter of the task (NOT the prior box)."""
    from world_model.v2.nero.contents import objects as OB, scenes as SC
    task = _fixed(task)
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
    task = _fixed(task)
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
    phenomena: list = field(default_factory=list)    # v2: [{id, why, handle: {measure: skill} | {unmeasurable: reason}, name? (NEW)}]
    probes: list = field(default_factory=list)       # v2: skill ids in the order the model wants them run
    schema: str = "v1"                               # "v1" (belief) | "v2" (phenomena / probes)

    def to_dict(self):
        if self.schema == "v2":
            return dict(status="ok", task=self.task, phenomena=self.phenomena, probes=self.probes,
                        steps=[dict(skill=s.call.skill, args=s.call.args, why=s.why) for s in self.steps],
                        recoveries={str(i): [dict(skill=r.skill, args=r.args) for r in s.recoveries] for i, s in enumerate(self.steps) if s.recoveries},
                        fallback=[dict(skill=r.skill, args=r.args) for r in self.fallback])
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
    of the scene (place(support) must name one).  Schema v2 (phenomena / probes, no belief) unless BIMANUAL_LEGACY_PROPOSER_V1=1 (the 11-05..11-09 format)."""
    return validate_v1(d, task, supports) if legacy_v1() else validate_v2(d, task, supports)


def validate_v1(d, task, supports=("counter", "shelf")):
    """the 11-05..11-09 format: belief + steps + recoveries (with optional covers) + fallback."""
    if task not in TASKS:
        raise PlanRejected("schema", f"unknown task {task!r}")
    errs = []
    if not _keys(d, V1_KEYS, (), "reply", errs):
        raise PlanRejected("schema", errs)
    # belief
    P = params(task)
    belief = {}
    if _keys(d["belief"], tuple(P), (), "belief", errs):
        for k, spec in P.items():
            iv = _interval(d["belief"][k], *spec["range"], f"belief.{k}", errs)
            if iv is not None:
                belief[k] = iv
    steps, fbl = _plan_parts(d, task, supports, belief, True, errs)
    if errs:
        raise PlanRejected("schema", errs)
    return TaskPlan(task, belief, steps, [r for r in fbl if r is not None])


def validate_v2(d, task, supports=("counter", "shelf")):
    """schema v2 (11-09 probe-selection build spec s.3): phenomena + probes + steps + recoveries + fallback.  NO number about the contents (no belief, no covers): the
    skills supply the numbers.  Every problem is listed; nothing is repaired, reordered or defaulted."""
    if task not in TASKS:
        raise PlanRejected("schema", f"unknown task {task!r}")
    errs = []
    if isinstance(d, dict) and "belief" in d:
        errs.append("belief: not part of this format (no contents numbers from the model: name phenomena and probes instead)")
    if not _keys({k: v for k, v in d.items() if k != "belief"} if isinstance(d, dict) else d, V2_KEYS, (), "reply", errs):
        raise PlanRejected("schema", errs)
    phen, probes = _phenomena_probes(d["phenomena"], d["probes"], errs)
    steps, fbl = _plan_parts(d, task, supports, {}, False, errs)
    for sec in ("phenomena", "probes", "recoveries", "fallback"):          # numbers only in step args
        errs += [f"{sec}: a number ({x!r}) at {w}; numbers are allowed only in step args" for w, x in _numbers(d[sec], sec)]
    if errs:
        raise PlanRejected("schema", errs)
    tp = TaskPlan(task, {}, steps, [r for r in fbl if r is not None])
    tp.phenomena, tp.probes, tp.schema = phen, probes, "v2"
    return tp


def _numbers(x, path):
    """(path, value) of every JSON number (not bool) in x."""
    if isinstance(x, bool):
        return []
    if isinstance(x, (int, float)):
        return [(path, x)]
    if isinstance(x, dict):
        return [r for k, v in x.items() for r in _numbers(v, f"{path}.{k}")]
    if isinstance(x, list):
        return [r for i, v in enumerate(x) for r in _numbers(v, f"{path}[{i}]")]
    return []


def _st(v):
    return v.strip() if isinstance(v, str) else v


def _text_ok(v):
    return isinstance(v, str) and bool(v.strip()) and len(v) <= MAX_WHY


def _phenomena_probes(raw_ph, raw_pr, errs):
    """validate the `phenomena` and `probes` sections -> (phenomena list of normalized dicts, probe skill ids)."""
    from world_model.v2.stack.skills import catalog as CAT
    phen, ids, names, measured = [], set(), set(), {}
    if not isinstance(raw_ph, list):
        errs.append("phenomena: must be a list")
        raw_ph = []
    for i, p in enumerate(raw_ph):
        w = f"phenomena[{i}]"
        pid = p.get("id") if isinstance(p, dict) else None
        is_new = pid == CAT.NEW
        if not _keys(p, ("id", "why", "handle") + (("name",) if is_new else ()), (), w, errs):
            continue
        if not is_new and not (isinstance(pid, str) and pid in CAT.PHENOMENA):
            errs.append(f"{w}.id: unknown phenomenon {pid!r}; have {sorted(CAT.PHENOMENA)} or \"{CAT.NEW}\"")
            continue
        if not is_new and pid in ids:
            errs.append(f"{w}.id: duplicate phenomenon {pid!r}")
        if is_new:
            if not _text_ok(p["name"]):
                errs.append(f"{w}.name: must be a non-empty string of at most {MAX_WHY} characters")
            elif str(_st(p["name"])).lower() in names:
                errs.append(f"{w}.name: duplicate NEW phenomenon name {p['name']!r}")
            else:
                names.add(str(_st(p["name"])).lower())
        ids.add(pid)
        if not _text_ok(p["why"]):
            errs.append(f"{w}.why: must be a non-empty string of at most {MAX_WHY} characters")
        h = p["handle"]
        if not _keys(h, (), ("measure", "unmeasurable"), f"{w}.handle", errs):
            continue
        if len(h) != 1:
            errs.append(f"{w}.handle: must have exactly one of \"measure\" or \"unmeasurable\", got {sorted(h)}")
            continue
        if "unmeasurable" in h:
            if not _text_ok(h["unmeasurable"]):
                errs.append(f"{w}.handle.unmeasurable: must be a non-empty reason string of at most {MAX_WHY} characters")
            phen.append(dict(id=pid, why=_st(p["why"]), handle=dict(unmeasurable=_st(h["unmeasurable"])), **({"name": _st(p["name"])} if is_new else {})))
            continue
        sk = h["measure"]
        s = CAT.SKILLS.get(sk) if isinstance(sk, str) else None
        if s is None:
            errs.append(f"{w}.handle.measure: unknown skill {sk!r}; have the built skills {sorted(CAT.built())}")
        elif s.status != "built":
            errs.append(f"{w}.handle.measure: skill {sk!r} is not built (design only); have {sorted(CAT.built())}")
        elif pid not in s.measures:
            errs.append(f"{w}.handle.measure: skill {sk!r} does not measure {pid!r} (it measures {list(s.measures)})")
        else:
            measured.setdefault(sk, []).append(pid)
        phen.append(dict(id=pid, why=_st(p["why"]), handle=dict(measure=sk), **({"name": _st(p["name"])} if is_new else {})))
    probes = []
    if not isinstance(raw_pr, list):
        errs.append("probes: must be a list")
        raw_pr = []
    for i, q in enumerate(raw_pr):
        w = f"probes[{i}]"
        if not _keys(q, ("skill",), (), w, errs):
            continue
        sk = q["skill"]
        if not (isinstance(sk, str) and sk in CAT.SKILLS and CAT.SKILLS[sk].status == "built"):
            errs.append(f"{w}.skill: {sk!r} is not a built measuring skill; have {sorted(CAT.built())}")
            continue
        if sk in probes:
            errs.append(f"{w}.skill: {sk!r} listed twice")
            continue
        probes.append(sk)
        if sk not in measured:
            errs.append(f"{w}.skill: {sk!r} is in probes but no phenomenon names it in handle.measure")
    for sk, pids in measured.items():
        if sk not in probes:
            errs.append(f"probes: skill {sk!r} is the handle of {pids} but is not in probes")
    if len(probes) == len(raw_pr):
        errs += CAT.order_errors(probes)
    return phen, probes


def _plan_parts(d, task, supports, belief, with_covers, errs):
    """steps / recoveries / fallback validation shared by both formats.  -> (steps, fallback recovery calls); appends to errs."""
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
        SK = step_skills(task)
        if sk not in SK:
            errs.append(f"{w}.skill: unknown skill {sk!r}; have {sorted(SK)}")
            continue
        if not (isinstance(s["why"], str) and s["why"].strip() and len(s["why"]) <= MAX_WHY):
            errs.append(f"{w}.why: must be a non-empty string of at most {MAX_WHY} characters")
        spec, args = SK[sk], s["args"]
        if not _keys(args, tuple(spec), (), f"{w}.args", errs):
            continue
        if sk == "base_move":
            rerr = _route_errors(args["route"], f"{w}.args.route")
            errs.extend(rerr)
            if not rerr:
                steps.append(Step(SkillCall(sk, dict(route=[{k: float(v)} for seg in args["route"] for k, v in seg.items()])), s["why"].strip()))
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
        if sk.count("base_move") > 1:
            errs.append(f"steps: at most one base_move() allowed, found {sk.count('base_move')}")
    # recoveries
    rec = d["recoveries"]
    if not isinstance(rec, dict):
        errs.append("recoveries: must be an object {step index: [recovery, ...]}")
        rec = {}
    for k in rec:
        if not (isinstance(k, str) and k.isdigit() and int(k) < len(raw) and isinstance(raw[int(k)], dict) and raw[int(k)].get("skill") in MOTION_SKILLS):
            errs.append(f"recoveries.{k}: key must be the index of a motion step ({' / '.join(s_ for s_ in MOTION_SKILLS if s_ != 'base_move' or task in BASE_TASKS)}), not {k!r}")
    for i, s in enumerate(steps):
        if s.call.skill in MOTION_SKILLS:
            lst = rec.get(str(i))
            if not isinstance(lst, list) or not lst:
                errs.append(f"recoveries.{i}: motion step {i} ({s.call.skill}) needs at least one recovery")
                continue
            for j, r in enumerate(lst):
                s.recoveries.append(_recovery(r, f"recoveries.{i}[{j}]", belief, errs, with_covers=with_covers))
    # fallback
    fb, fbl = d["fallback"], []
    if not isinstance(fb, list) or not fb or len(fb) > 3:
        errs.append("fallback: must be a list of 1..3 recovery calls")
    else:
        for j, r in enumerate(fb):
            fbl.append(_recovery(r, f"fallback[{j}]", belief, errs, with_covers=False))
    return steps, fbl


def _route_errors(route, what):
    """strict check of a base_move route: a list of 1..4 objects {"line": metres} | {"turn": degrees}, inside the ASSUMED outer ranges."""
    from world_model.v2.stack.planners import base_route as BR
    if not isinstance(route, list) or not route:
        return [f"{what}: must be a non-empty list of {{\"line\": m}} / {{\"turn\": deg}} objects"]
    errs, ok = [], True
    for i, seg in enumerate(route):
        if not (isinstance(seg, dict) and len(seg) == 1 and next(iter(seg)) in ("line", "turn")):
            errs.append(f"{what}[{i}]: must be an object with exactly one key, \"line\" (metres) or \"turn\" (degrees), got {seg!r}")
            ok = False
        elif not _num(next(iter(seg.values()))):
            errs.append(f"{what}[{i}].{next(iter(seg))}: must be a finite number, got {next(iter(seg.values()))!r}")
            ok = False
    if ok:
        errs += [what + e[len("route"):] for e in BR.check_ranges(BR.normalize_route(route))]
    return errs


def _recovery(r, what, belief, errs, with_covers):
    if isinstance(r, dict) and "covers" in r and not with_covers:
        errs.append(f"{what}.covers: not part of this format (no contents ranges from the model; the checker tries every recovery on the whole contents set)")
        r = {k: v for k, v in r.items() if k != "covers"}
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

