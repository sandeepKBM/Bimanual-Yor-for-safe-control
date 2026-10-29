"""Proposer: FACTS + CONSTRAINTS in a fixed JSON grammar (VIZ-COAST style: the model prunes, it never plans).  SIMULATION ONLY.

Any house/lang/backend.Backend works: MockVLM (scripted responses below) now; LiveVLM (open-weight Qwen file-queue daemon,
never started from here) wrapped in CachedVLM later.  Parsing takes the LAST JSON object in the reply (AGENTS.md), drops
anything outside the grammar (logged), and lets facts only WIDEN the belief box.

Grammar
  {"facts": {"lid_attached": bool, "contents": "empty"|"unknown"|str,
             "contents_mass_kg": [lo, hi], "pose_error_m": x, "yaw_error_deg": x},
   "constraints": [ {"forbid": {"skill": S, ...param: value}, "why": str},
                    {"require": {"skill": S, "atom": A}, "why": str} ]}
  forbid  = no ground action with that name and those params;  require = S only when atom A holds (did_<skill> atoms exist
  for every skill).  Narrowing facts are ignored.
"""
from __future__ import annotations

import json
import re

from . import domain as D

SYSTEM = (
    "You are the constraint writer for a two-arm robot planner. You never write plans. You look at the scene and write FACTS "
    "about the objects and hard CONSTRAINTS that rule out actions a careful person would never take. A separate physics "
    "checker decides what is safe; your constraints only prune the search. Write a constraint only when you are confident, "
    "give a one-line reason for each, and never invent numbers you cannot see.")

PROMPT = """Task: lift the covered pan off the stove without losing the lid or the contents.
Scene: a pan with a long handle sits on a stove block; the handle points toward the robot. A lid rests on the pan. The pan
may contain food of unknown mass. Arms: L (on the handle side), R (can reach the far side). Each arm has a two-finger claw.

Skills (name(params): precondition -> effect):
{skills}

State atoms you may use in "require": {atoms}, and did_<skill> for every skill.
Parts: {parts}.

Reply with ONE JSON object and nothing after it:
{{"facts": {{"lid_attached": true|false, "contents": "...", "contents_mass_kg": [lo, hi] (optional)}},
  "constraints": [{{"forbid": {{"skill": "...", "<param>": "<value>"}}, "why": "..."}},
                  {{"require": {{"skill": "...", "atom": "..."}}, "why": "..."}}]}}"""


def prompt():
    sk = "\n".join(f"  {sig}: {pre} -> {eff}" for sig, pre, eff in D.SKILLS.values())
    return PROMPT.format(skills=sk, atoms=", ".join(D.ATOMS), parts=", ".join(D.PARTS))


def last_json(text):
    """last top-level {...} object that parses (reasoning models may think aloud first)."""
    text = re.sub(r"```(?:json)?", "", text)
    best = None
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                try:
                    best = json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    pass
    return best


_PARAM_KEYS = {"arm", "part", "frac", "push"}


def parse(text, belief):
    """-> (constraints, belief', log).  constraints: list of ('forbid', name, params) | ('require', name, atom)."""
    log, cons = [], []
    obj = last_json(text or "")
    if not isinstance(obj, dict):
        return cons, belief, ["no JSON object in the reply: no constraints, belief unchanged"]
    for c in obj.get("constraints") or []:
        if not isinstance(c, dict):
            log.append(f"dropped (not an object): {c!r}")
            continue
        if isinstance(c.get("forbid"), dict):
            f = dict(c["forbid"])
            sk = f.pop("skill", None)
            bad = set(f) - _PARAM_KEYS
            if sk not in D.SKILLS or bad:
                log.append(f"dropped forbid (unknown skill/param): {c['forbid']}")
                continue
            cons.append(("forbid", sk, tuple(sorted(f.items()))))
        elif isinstance(c.get("require"), dict):
            r = c["require"]
            sk, at = r.get("skill"), r.get("atom")
            ok_atom = at in D.ATOMS or (isinstance(at, str) and at.startswith("did_") and at[4:] in D.SKILLS)
            if sk not in D.SKILLS or not ok_atom:
                log.append(f"dropped require (unknown skill/atom): {r}")
                continue
            cons.append(("require", sk, at))
        else:
            log.append(f"dropped (no forbid/require): {c}")
    facts = obj.get("facts") or {}
    b = belief
    mk = facts.get("contents_mass_kg")
    if isinstance(mk, (list, tuple)) and len(mk) == 2:
        try:
            lo, hi = float(mk[0]), float(mk[1])
            nb = b.widen(mass=(lo, hi))
            if lo > b.mass[0] or hi < b.mass[1]:
                log.append(f"contents_mass_kg {mk}: narrowing part ignored (the proposer may only widen the box)")
            b = nb
        except (TypeError, ValueError):
            log.append(f"dropped contents_mass_kg {mk!r}")
    for key, attr in (("pose_error_m", "dxy"), ("yaw_error_deg", "yaw")):
        if key in facts:
            try:
                v = float(facts[key])
                if v < getattr(b, attr):
                    log.append(f"{key}={v}: narrowing ignored")
                b = b.widen(**{attr: v})
            except (TypeError, ValueError):
                log.append(f"dropped {key} {facts[key]!r}")
    if facts.get("lid_attached") is True:
        log.append("fact lid_attached=true recorded but not trusted: the certificate reads attachment from the scene model")
    return cons, b, log


def allowed(act, state, cons):
    for kind, sk, x in cons:
        if act.name != sk:
            continue
        if kind == "forbid" and all(getattr(act, k) == v or str(getattr(act, k)) == str(v) for k, v in x):
            return False
        if kind == "require" and x not in state:
            return False
    return True


def produce(backend, belief, image=None):
    """ask the backend once; returns dict(constraints, belief, log, raw, backend info)."""
    imgs = [image] if image else []
    out = backend.ask(prompt(), images=imgs, system=SYSTEM, max_new_tokens=800, temperature=0.0)
    raw = out["texts"][0] if out.get("texts") else ""
    cons, b, log = parse(raw, belief)
    return dict(constraints=cons, belief=b, log=log, raw=raw, backend=out.get("backend"), seconds=out.get("seconds"))


# --- scripted proposers for tests / ablations (MockVLM rules) ---------------------------------------------------------
GOOD = json.dumps(dict(
    facts=dict(lid_attached=False, contents="unknown food", contents_mass_kg=[0.0, 0.4]),
    constraints=[dict(forbid=dict(skill="pinch", part="lid"), why="the lid is loose; lifting it does not lift the pan"),
                 dict(require=dict(skill="lift", atom="lid_secured"), why="a loose lid can slide off during the lift")]))
WRONG = json.dumps(dict(                     # confidently wrong: says the lid is attached, narrows the mass, adds nothing useful
    facts=dict(lid_attached=True, contents="empty", contents_mass_kg=[0.0, 0.05], pose_error_m=0.002),
    constraints=[]))
OVERPRUNE = json.dumps(dict(                 # forbids the only working skeleton
    facts=dict(lid_attached=False, contents="unknown"),
    constraints=[dict(forbid=dict(skill="clamp_rim"), why="do not touch the hot rim"),
                 dict(require=dict(skill="lift", atom="lid_secured"), why="keep the lid on")]))
GARBAGE = "Let me think... the pan looks heavy. {not json} I would lift it with one hand."
CANNED = dict(good=GOOD, wrong=WRONG, overprune=OVERPRUNE, garbage=GARBAGE, none="{}")


def mock_backend(kind):
    from world_model.v2.house.lang.backend import MockVLM
    return MockVLM(rules=[(r"constraint writer", CANNED[kind])], use_default=False)
