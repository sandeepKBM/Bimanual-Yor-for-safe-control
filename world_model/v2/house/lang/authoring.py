"""lang/authoring.py -- instruction + grounded entities -> VLM-authored constraint spec in the set IR (tray_rep/sets/ir.py).  SIMULATION / SAVED DATA ONLY.

    cat = build_catalog(grounding)                          # named zonotope groups (VLM names -> perceived zonotopes)
    r = author(backend, "bring the coffee to my desk without spilling and keep it away from the monitor", cat)
    r.spec            the dict the shield / rows use (VLM spec, or TRAY_SPEC_FULL on fallback)
    r.status          'vlm' | 'repaired' | 'fallback'      r.reasons  every decision, in order

Pipeline: VLM writes {"spec": <IR dict>, "unresolved": [...]} -> extract_json -> expand_groups (an object with k perceived fragments becomes an AND of k
clear atoms) -> ir.from_dict (compile) -> checks:
    eval      margin(nominal) finite on every probe state
    sem       nominal margin > 0; must-violate probes < 0: ball_off, tilt60, and 'overlap_<obj>' (tray on top of the object) for every object the spec constrains
    sanity    theta_max in [0.02, 0.6] rad, clear d in [0, 0.6] m, inside margin / shrink in [-0.005, 0.1] m (a VLM cannot silently switch the constraint off)
    coverage  an instruction with a keep-away phrase must yield a clear atom for every named object that exists in the catalog (unresolved ones are reported)
    distance  a metric distance in the instruction ('5 cm') must match the clear d within 25 %
On failure: ONE repair round (the checks' error text is sent back); still failing -> fall back to TRAY_SPEC_FULL and say why.
"""
from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass, field

import numpy as np

from world_model.v2.tray_rep.sets import ir as IR, tray_spec as TS
from world_model.v2.tray_rep.sets.ir import Entity, Pose, Point, BoxShape, ZonoShape, prism_xy
from world_model.v2.tray_rep.sets.zono import Zonotope
from world_model.v2.testbeds.b_tray_ball.tray_env import HX, HY
from world_model.v2.vlm import prompts as VP
from world_model.v2.vlm.spec_eval import extract_json, atoms

CARRIED = {"tray", "tray_top", "cup", "ball"}
PARAMS = dict(H=0, lo=0, r_shrink=0.03, theta_max=0.10, r_ball=0.03)        # the shield IR monitor's parameters (ShieldCfg.r_cup, theta_tray_ir)
PAYLOAD_H = 0.12                                                            # tray plate + the cup standing on it [m]
BOUNDS = dict(theta_max=(0.02, 0.6), d=(0.0, 0.6), margin=(-0.005, 0.1), shrink=(0.0, 0.1))
KEEP_AWAY = re.compile(r"away from|avoid|clear of|stay (?:off|clear|away)|keep (?:it |the \w+ )?(?:\w+ )?(?:from|off|away)|don'?t (?:hit|touch|bump|knock|go near)|"
                       r"not (?:hit|touch|bump|knock)|distance|near|close to|no closer", re.I)


# ------------------------------------------------------------------------------------------------------------------ catalog
@dataclass
class Catalog:
    groups: dict                  # name -> [Proposal]  (fragments of one named object)
    conf: dict = field(default_factory=dict)

    @staticmethod
    def member(name, j):
        return f"{name}#{j}"

    def descriptions(self):
        d = {"tray": "the carried tray plus the cup standing on it (a solid 30 x 20 cm box, 12 cm tall): use it in clear atoms and upright",
             "tray_top": "the tray's top-face region (a container region: use it only as the B of inside)",
             "cup": "the centre point of the cup (the carried drink): use it as the A of inside"}
        for nm, ps in self.groups.items():
            d[nm] = f"scene object '{nm}' as named by a vision model" + (f" ({len(ps)} perceived fragments, handled automatically)" if len(ps) > 1 else "")
        return d

    def entities(self, inflated=True):
        """IR entities: the carried tray / cup (world-frame pose from the EnvState adapter) + every perceived fragment as a world-frame zonotope."""
        I = np.eye(3)
        pose_t = lambda s: Pose(s.R_t, s.p_t)
        ents = {"tray": Entity("tray", BoxShape([HX, HY, PAYLOAD_H / 2], centre=[0, 0, PAYLOAD_H / 2]), pose_t),
                "tray_top": Entity("tray_top", prism_xy(HX, HY), pose_t),
                "cup": Entity("cup", Point(), lambda s: Pose(I, s.p_b))}
        ents["ball"] = ents["cup"]
        for nm, ps in self.groups.items():
            for j, p in enumerate(ps):
                Z = Zonotope(p.ci, p.Gi) if inflated else Zonotope(p.c, p.G)
                e = Entity(self.member(nm, j), ZonoShape(Z), lambda s: Pose(I, np.zeros(3)))
                ents[self.member(nm, j)] = e
                if len(ps) == 1:
                    ents[nm] = e
        return ents

    def zono_ids(self, name):
        return [p.id for p in self.groups.get(name, [])]


def build_catalog(grounding, min_conf=0.0):
    groups, conf = {}, {}
    pid = {p.id: p for p in grounding.props}
    for nm, ids in grounding.groups.items():
        ps = [pid[i] for i in ids if grounding.names[i][1] >= min_conf]
        if ps:
            groups[nm] = ps
            conf[nm] = float(np.mean([grounding.names[p.id][1] for p in ps]))
    return Catalog(groups, conf)


# ------------------------------------------------------------------------------------------------------------------ spec dict utilities
def expand_groups(spec, cat: Catalog):
    """clear atoms that name a multi-fragment object -> AND over its fragments (<name>#j).  Returns a new dict."""
    multi = {nm for nm, ps in cat.groups.items() if len(ps) > 1}

    def rec(d):
        if not isinstance(d, dict):
            return d
        out = {}
        for k, v in d.items():
            if k in ("and", "or"):
                out[k] = [rec(c) for c in v]
            elif k == "not":
                out[k] = rec(v)
            elif k in ("G", "F"):
                out[k] = dict(v, of=rec(v.get("of")))
            elif k == "clear" and isinstance(v, dict) and (v.get("a") in multi or v.get("b") in multi):
                names_a = [Catalog.member(v["a"], j) for j in range(len(cat.groups[v["a"]]))] if v.get("a") in multi else [v.get("a")]
                names_b = [Catalog.member(v["b"], j) for j in range(len(cat.groups[v["b"]]))] if v.get("b") in multi else [v.get("b")]
                out["and"] = [{"clear": dict(v, a=a, b=b)} for a in names_a for b in names_b]
                if len(out["and"]) == 1:
                    out = out["and"][0]
            else:
                out[k] = v
        return out
    return rec(copy.deepcopy(spec))


def collect_atoms(spec, ctx="pos", out=None):
    """[(kind, a, b, params, ctx)] over the spec; ctx 'pos' (holds unconditionally) | 'cond' (under or/not/F)."""
    out = [] if out is None else out
    if not isinstance(spec, dict):
        return out
    for k, v in spec.items():
        if k == "and":
            for c in v:
                collect_atoms(c, ctx, out)
        elif k == "or":
            for c in v:
                collect_atoms(c, "cond", out)
        elif k == "not":
            collect_atoms(v, "cond", out)
        elif k == "G":
            collect_atoms(v.get("of"), ctx, out)
        elif k == "F":
            collect_atoms(v.get("of"), "cond", out)
        elif k in ("inside", "upright", "clear", "support") and isinstance(v, dict):
            out.append((k, v.get("a"), v.get("b"), v, ctx))
    return out


def _nm(x):
    if isinstance(x, str):
        return x
    if isinstance(x, dict) and "shrink" in x:
        return _nm(x["shrink"]["of"])
    return "?"


def clearances(spec, cat: Catalog):
    """{object group name: required clearance d [m]} from the unconditional clear atoms that involve a scene object (max over atoms)."""
    out = {}
    for kind, a, b, v, ctx in collect_atoms(spec):
        if kind != "clear" or ctx != "pos":
            continue
        a, b = _nm(a), _nm(b)
        for nm in (a, b):
            base = nm.split("#")[0]
            if base in cat.groups and (a in CARRIED or b in CARRIED or a.split("#")[0] != b.split("#")[0]):
                try:
                    d = float(v.get("d", 0.0))
                except Exception:      # noqa: BLE001 (param-name strings are not resolved here)
                    continue
                out[base] = max(out.get(base, 0.0), d)
    return out


def pretty(spec, indent=0):
    """one human-readable line per atom (HUD / logs)."""
    pad = "  " * indent
    if not isinstance(spec, dict) or not spec:
        return pad + str(spec)
    (k, v), = spec.items()
    ent = lambda x: (f"shrink({_nm(x)}, {x['shrink']['by']})" if isinstance(x, dict) and "shrink" in x else str(x))
    if k in ("and", "or"):
        return pad + k.upper() + ":\n" + "\n".join(pretty(c, indent + 1) for c in v)
    if k == "not":
        return pad + "NOT:\n" + pretty(v, indent + 1)
    if k in ("G", "F"):
        w = f"[{v.get('lo', 0)}, {v.get('hi', 'H')}]"
        return pad + ("ALWAYS" if k == "G" else "EVENTUALLY") + f" {w}:\n" + pretty(v.get("of"), indent + 1)
    if k == "inside":
        return pad + f"{ent(v.get('a'))} inside {ent(v.get('b'))} (margin {v.get('margin', 0)})"
    if k == "upright":
        return pad + f"{ent(v.get('a'))} upright, tilt <= {v.get('theta_max')} rad"
    if k == "clear":
        return pad + f"{ent(v.get('a'))} at least {v.get('d')} m from {ent(v.get('b'))}"
    if k == "support":
        return pad + f"{ent(v.get('a'))} supported by {ent(v.get('b'))}"
    return pad + json.dumps(spec)


# ------------------------------------------------------------------------------------------------------------------ states and checks
class _S:
    """EnvState-compatible pose table (tray R_t, p_t; cup point p_b)."""

    def __init__(self, p_t, R_t, p_b):
        self.p_t, self.R_t, self.p_b = np.asarray(p_t, float), np.asarray(R_t, float), np.asarray(p_b, float)


def _Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _box_dist(p, lo, hi):
    return float(np.linalg.norm(np.maximum(np.maximum(lo - p, p - hi), 0.0)))


def free_pose(cat: Catalog):
    """a tray position far from every perceived object (maximin of the AABB distance over a coarse grid around the scene)."""
    boxes = []
    for ps in cat.groups.values():
        for p in ps:
            Z = Zonotope(p.ci, p.Gi)
            lo, hi = Z.interval_hull()
            boxes.append((lo, hi))
    if not boxes:
        return np.array([0.0, 0.0, 1.0])
    allc = np.array([(l + h) / 2 for l, h in boxes])
    c0 = allc.mean(0)
    best, bp = -1.0, c0
    for dx in np.arange(-12, 12.1, 1.0):
        for dy in np.arange(-12, 12.1, 1.0):
            p = np.array([c0[0] + dx, c0[1] + dy, c0[2]])
            dm = min(_box_dist(p, l, h) for l, h in boxes)
            if dm > best:
                best, bp = dm, p
    return bp


def probe_states(cat: Catalog):
    """name -> (state, must_violate).  nominal: tray in free space, cup on it."""
    p0 = free_pose(cat)
    I = np.eye(3)
    cup_off = np.array([-0.045, 0.0, 0.06])
    out = {"nominal": (_S(p0, I, p0 + cup_off), False),
           "ball_off": (_S(p0, I, p0 + cup_off + [0.30, 0.0, 0.0]), True),
           "tilt60": (_S(p0, _Rx(np.pi / 3), p0 + _Rx(np.pi / 3) @ cup_off), True)}
    for nm, ps in cat.groups.items():
        for j, p in enumerate(ps):
            pt = np.array(p.ci, float)
            out[f"overlap_{nm}#{j}"] = (_S(pt, I, pt + cup_off), True)
    return out


def _numbers_ok(spec):
    bad = []
    for kind, a, b, v, ctx in collect_atoms(spec):
        try:
            if kind == "upright":
                x = float(v["theta_max"])
                if not BOUNDS["theta_max"][0] <= x <= BOUNDS["theta_max"][1]:
                    bad.append(f"upright theta_max={x} outside [{BOUNDS['theta_max'][0]}, {BOUNDS['theta_max'][1]}] rad")
            elif kind == "clear":
                x = float(v["d"])
                if not BOUNDS["d"][0] <= x <= BOUNDS["d"][1]:
                    bad.append(f"clear d={x} outside [{BOUNDS['d'][0]}, {BOUNDS['d'][1]}] m")
            elif kind == "inside":
                x = float(v.get("margin", 0.0))
                if not BOUNDS["margin"][0] <= x <= BOUNDS["margin"][1]:
                    bad.append(f"inside margin={x} outside [{BOUNDS['margin'][0]}, {BOUNDS['margin'][1]}] m")
                for e in (v.get("a"), v.get("b")):
                    if isinstance(e, dict) and "shrink" in e:
                        s = float(e["shrink"]["by"])
                        if not BOUNDS["shrink"][0] <= s <= BOUNDS["shrink"][1]:
                            bad.append(f"shrink by {s} outside [{BOUNDS['shrink'][0]}, {BOUNDS['shrink'][1]}] m")
        except (KeyError, TypeError, ValueError):
            continue                      # a param-name string ("theta_max") is fine: resolved by from_dict
    return bad


_DIST = re.compile(r"(\d+(?:\.\d+)?)\s*(cm|centimet\w*|mm|millimet\w*|m|met\w*)\b", re.I)


def instruction_distances(ins):
    out = []
    for v, u in _DIST.findall(ins):
        u = u.lower()
        out.append(float(v) * (0.01 if u.startswith("c") else 0.001 if u.startswith("mi") or u == "mm" else 1.0))
    return out


_CUT = re.compile(r"[.;]|\bthen\b|,?\s+(?:and\s+)?(?:then\s+)?(?:bring|put|place|carry|take|deliver|go|drive|move|serve|set)\b", re.I)


def keepaway_text(ins):
    """the part(s) of the instruction that state a keep-away requirement: from each KEEP_AWAY match to the next sentence / new-verb boundary."""
    out = []
    for m in KEEP_AWAY.finditer(ins):
        rest = ins[m.start():]
        cut = _CUT.search(rest, 1)
        out.append(rest[:cut.start()] if cut else rest)
    return " ; ".join(out)


def instruction_mentions(ins, cat: Catalog):
    low = " " + re.sub(r"[^a-z0-9 ]+", " ", ins.lower()) + " "
    out = []
    for nm in cat.groups:
        toks = [t for t in nm.split("_") if t]
        if any(f" {t} " in low or f" {t}s " in low for t in toks if len(t) > 2) or f" {' '.join(toks)} " in low:
            out.append(nm)
    return out


def check_spec(spec, cat: Catalog, instruction="", mode="replace", unresolved=()):
    """compile + evaluate + semantic checks.  Returns dict(compile, eval, sem, ok, errors[], margins{}, clearances{}, covered[], uncovered[])."""
    r = dict(compile=False, eval=False, sem=False, ok=False, errors=[], margins={}, probes_caught={}, clearances={}, covered=[], uncovered=[], expanded=None, node=None)
    try:
        spec_x = expand_groups(spec, cat)
        full = {"and": [TS.TRAY_SPEC_FULL, spec_x]} if mode == "augment" else spec_x
        r["expanded"] = full
        ents = cat.entities()
        node = IR.from_dict(full, ents, PARAMS)
        r["compile"], r["node"] = True, node
    except KeyError as e:
        r["errors"].append(f"unknown entity or parameter {e}; available entities: {sorted(k for k in cat.descriptions())}")
        return r
    except Exception as e:      # noqa: BLE001
        r["errors"].append(f"compile error {type(e).__name__}: {str(e)[:160]}")
        return r
    try:
        sts = probe_states(cat)
        for n, (s, mv) in sts.items():
            r["margins"][n] = float(IR.margin(node, s, False))
        r["eval"] = bool(all(np.isfinite(v) for v in r["margins"].values()))
        if not r["eval"]:
            r["errors"].append("margin is not finite")
    except Exception as e:      # noqa: BLE001
        r["errors"].append(f"evaluation error {type(e).__name__}: {str(e)[:160]}")
        return r
    errs = []
    if r["margins"]["nominal"] <= 0:
        errs.append(f"the nominal state (tray in free space, cup on it, tilt 0) violates your spec (margin {r['margins']['nominal']:.3f}): write only safety constraints that "
                    "hold during a normal carry (not the goal, no F/eventually of the destination)")
    for n, (s, mv) in sts.items():
        if mv:
            r["probes_caught"][n] = bool(r["margins"][n] < 0)
    if not r["probes_caught"].get("ball_off", False):
        errs.append("the spec does not forbid the cup leaving the tray (inside(cup, shrink(tray_top, ...)))")
    if not r["probes_caught"].get("tilt60", False):
        errs.append("the spec does not forbid tilting the tray (upright(tray, theta_max))")
    cl = clearances(spec, cat)
    r["clearances"] = cl
    for nm in cl:
        miss = [n for n in r["probes_caught"] if n.startswith(f"overlap_{nm}#") and not r["probes_caught"][n]]
        if miss:
            errs.append(f"the clear atom for '{nm}' is not enforced when the tray overlaps it (check the logic: it must be inside the always-AND)")
    errs += _numbers_ok(spec)
    mentioned = instruction_mentions(keepaway_text(instruction), cat) if instruction else []
    if instruction and KEEP_AWAY.search(instruction):
        r["covered"] = [m for m in mentioned if m in cl]
        r["uncovered"] = [m for m in mentioned if m not in cl]
        if r["uncovered"]:
            errs.append("the instruction mentions " + ", ".join(r["uncovered"]) + " and a keep-away requirement, but the spec has no clear(tray, <object>, d) atom for it")
    if unresolved and cl:                      # the VLM says an object of the instruction is not in the scene: a clear atom on an object the user did not name is a substitution
        stray = [nm for nm in cl if nm not in instruction_mentions(instruction, cat)]
        if stray:
            errs.append("you listed " + ", ".join(map(str, unresolved)) + " as unresolved, but the spec adds a clear atom for " + ", ".join(stray) +
                        " which the instruction does not mention: do NOT substitute another object, leave the missing one out")
    dists = instruction_distances(instruction) if instruction else []
    if dists and cl:
        for nm, d in cl.items():
            if min(abs(d - x) / x for x in dists) > 0.25:
                errs.append(f"the instruction gives the distance(s) {dists} m but clear d for '{nm}' is {d}")
    r["errors"] += errs
    r["sem"] = bool(not errs)
    r["ok"] = r["compile"] and r["eval"] and r["sem"]
    return r


# ------------------------------------------------------------------------------------------------------------------ prompts
RULES = """
SCENE RULES (this robot carries a cup standing on a tray with both arms through cluttered rooms):
- Write only SAFETY constraints that must hold the WHOLE carry: wrap everything in one {"G": {"lo": 0, "hi": "H", "of": ...}}. Do NOT encode the destination or any goal (no F, no support).
- ALWAYS include: the cup stays on the tray {"inside": {"a": "cup", "b": {"shrink": {"of": "tray_top", "by": 0.03}}, "margin": 0.0}} and the tray stays level
  {"upright": {"a": "tray", "theta_max": 0.1}} (use a smaller theta_max, 0.05 to 0.08, if the user asks for gentle / careful / smooth carrying).
- For every scene object the user wants kept away from (or not hit / not touched / not near), add {"clear": {"a": "tray", "b": "<object name>", "d": <metres>}} with the
  distance from the instruction (cm -> metres; default 0.10 if none is given). Use the object names EXACTLY as listed in Entities; never invent names.
- If the instruction names an object that is NOT in Entities, do not invent it: leave it out and list its name in "unresolved".
- Output ONLY one JSON object on a SINGLE LINE (compact: no indentation, no line breaks, no code fence): {"spec": <the spec>, "unresolved": [<names>]}."""

EXAMPLE_LANG = {"instruction": "Carry it to the table gently and stay 8 cm away from the vase.",
                "entities": {"tray": "...", "tray_top": "...", "cup": "...", "vase": "scene object"},
                "out": {"spec": {"G": {"lo": 0, "hi": "H", "of": {"and": [
                    {"inside": {"a": "cup", "b": {"shrink": {"of": "tray_top", "by": 0.03}}, "margin": 0.0}},
                    {"upright": {"a": "tray", "theta_max": 0.06}},
                    {"clear": {"a": "tray", "b": "vase", "d": 0.08}}]}}}, "unresolved": []}}


def system_prompt():
    ex = EXAMPLE_LANG
    return (VP.GRAMMAR.replace("Output ONLY the JSON object, nothing else.", "") + RULES + "\n\nEXAMPLE\nEntities: " + json.dumps(ex["entities"]) + "\nInstruction: " +
            ex["instruction"] + "\nAnswer: " + json.dumps(ex["out"]))


def user_prompt(instruction, cat: Catalog, errors=None, prev=None):
    s = ("Look at the image of the scene (numbered boxes mark perceived objects).\nEntities: " + json.dumps(cat.descriptions()) + "\nInstruction: " + instruction +
         "\nWrite the constraint spec as one JSON object {\"spec\": ..., \"unresolved\": [...]}.")
    if errors:
        s += ("\n\nYour previous answer was:\n" + str(prev)[:1500] + "\nIt failed these checks:\n- " + "\n- ".join(errors) +
              "\nFix every problem and answer again with ONLY the JSON object. If an object of the instruction is not in Entities, leave it out and "
              "list it in \"unresolved\" (never substitute another object for it).")
    return s


# ------------------------------------------------------------------------------------------------------------------ author
@dataclass
class SpecResult:
    spec: dict                      # as authored (or the fallback)
    spec_expanded: dict             # group-expanded (what from_dict compiles)
    status: str                     # 'vlm' | 'repaired' | 'fallback'
    reasons: list
    unresolved: list
    clearances: dict                # object name -> d
    check: dict                     # final check summary (no node)
    attempts: list
    seconds: float = 0.0            # sum of model seconds
    wall_s: float = 0.0
    instruction: str = ""
    mode: str = "replace"

    def pretty(self):
        return pretty(self.spec)


FORMS = ("and", "or", "not", "G", "F", "inside", "upright", "clear", "support")


def _strip_extras(d, extras):
    """a VLM sometimes puts 'unresolved' / 'notes' INSIDE the spec dict next to the one grammar key: lift them out (recursively)."""
    if isinstance(d, dict):
        if any(k in FORMS for k in d):
            for k in [k for k in d if k not in FORMS]:
                v = d.pop(k)
                if k == "unresolved" and isinstance(v, list):
                    extras.extend(str(u) for u in v)
        for v in list(d.values()):
            _strip_extras(v, extras)
    elif isinstance(d, list):
        for v in d:
            _strip_extras(v, extras)
    return d


def parse_answer(text):
    """-> (spec dict | None, unresolved list)."""
    j = extract_json(text)
    if not isinstance(j, dict):
        return None, []
    extras = []
    if "spec" in j and isinstance(j["spec"], dict):
        un = j.get("unresolved") or []
        un = [str(u) for u in un] if isinstance(un, list) else []
        spec = _strip_extras(j["spec"], extras)
        return spec, list(dict.fromkeys(un + extras))
    spec = _strip_extras(j, extras)
    return (spec if len(spec) == 1 else None), list(dict.fromkeys(extras))


def _summ(c):
    return {k: v for k, v in c.items() if k not in ("node", "expanded")}


def author(backend, instruction, cat: Catalog, images=(), mode="replace", max_repair=1, max_new_tokens=500, log=print):
    """see the module docstring.  `images`: PNG paths shown to the VLM (e.g. the marked head-camera image)."""
    attempts, reasons, unresolved = [], [], []
    errors, prev = None, None
    sec = wall = 0.0
    spec = None
    chk = None
    for it in range(max_repair + 1):
        o = backend.ask(user_prompt(instruction, cat, errors, prev), images=list(images), system=system_prompt(), max_new_tokens=max_new_tokens, temperature=0.0)
        txt = o["texts"][0] if o["texts"] else ""
        sec += float(o.get("seconds") or 0.0)
        wall += float(o.get("wall_s") or 0.0)
        spec, un = parse_answer(txt)
        att = dict(round=it, text=txt, seconds=o.get("seconds"), wall_s=o.get("wall_s"), cached=o.get("cached", False), parsed=spec is not None)
        if spec is None:
            errors, prev = ["the answer did not contain a JSON object of the required form"], txt
            att["errors"] = errors
            attempts.append(att)
            reasons.append(f"round {it}: no parsable JSON spec")
            continue
        unresolved = un
        chk = check_spec(spec, cat, instruction, mode, un)
        att.update(spec=spec, check=_summ(chk), unresolved=un)
        attempts.append(att)
        if chk["ok"]:
            st = "vlm" if it == 0 else "repaired"
            reasons.append(f"round {it}: compiled, sem ok" + (f"; unresolved by the VLM: {un}" if un else ""))
            return SpecResult(spec, chk["expanded"], st, reasons, un, chk["clearances"], _summ(chk), attempts, sec, wall, instruction, mode)
        errors, prev = chk["errors"], json.dumps(spec)
        reasons.append(f"round {it}: failed ({'; '.join(chk['errors'])[:300]})")
        log(f"[lang] spec round {it} failed: {chk['errors']}")
    # fallback: the hand-written tray spec (kept honest in the log)
    reasons.append("FALLBACK to TRAY_SPEC_FULL: " + (reasons[-1] if reasons else ""))
    fb = copy.deepcopy(TS.TRAY_SPEC_FULL)
    fchk = check_spec(fb, cat, "", "replace")
    return SpecResult(fb, fchk["expanded"] or fb, "fallback", reasons, unresolved, {}, _summ(fchk), attempts, sec, wall, instruction, "replace")
