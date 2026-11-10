"""stack/proposer/mock.py -- scripted proposer replies for tests and dry runs (NOT a model).  SIMULATION ONLY.   (11-05)

spec "mock:<kind>":
  fixture        the legacy hand-written plan of the task (carry.PLANS shapes: tray / pot lift 5 cm, 10 cm toward the robot, place; box lift 18 cm, 20 cm, place on the
                 shelf) written as a proposer reply, belief = the task's sanity ranges (wide), recovery = set_down on every motion step, fallback = let_go
  malformed      prose around a broken JSON object
  unknown_skill  the fixture plan with one step "teleport"
  narrow         the fixture plan with a narrow belief on the probed parameter (to test Rule 2: probe set outside B)
  Schema v2 variants (11-09; the default format -- BIMANUAL_LEGACY_PROPOSER_V1=1 gives the v1 replies above):
  omit_ph1       the fixture plan that does not name PH1 and runs no probe (PH1 sits on its fallback)
  wrong_skill    the fixture plan that claims `weigh` measures PH2 (a skill that does not measure it)
  new_phenomenon the fixture plan plus a phenomenon "NEW" (named, unmeasurable)
  numeric        the fixture plan plus a numeric contents belief and a `covers` range (v1 fields): rejected in v2
  t2_reask       first reply = the omit_ph1 plan (no probe, PH1 on its fallback: the checker refuses it on the whole prior -> Rule 2 T2); the Rule 2 re-ask ("RULE 2 RE-ASK") is
                 answered with the fixture plan (PH1 -> weigh), which the checker then admits or refuses on the weigh's reading
fixture / narrow follow the format of the mode (narrow is always the v1 belief reply).
A v1 re-ask prompt ("NEW MEASUREMENT") is answered with the fixture plan whose belief is the sanity ranges; a v2 Rule 2 re-ask ("RULE 2 RE-ASK") with the fixture v2 plan.
"""
from __future__ import annotations

import json
import re

from world_model.v2.house.lang.backend import Backend
from world_model.v2.stack.proposer import schema as S

KINDS = ("fixture", "malformed", "unknown_skill", "narrow", "omit_ph1", "wrong_skill", "new_phenomenon", "numeric", "t2_reask")


def _motion(task):
    """(steps, recoveries) of the fixture plan: the legacy carry.PLANS shapes (ASSUMED, user-confirmed 11-02); table-behind: lift, the known-good route R1, place on counter 2."""
    if task in S.BASE_TASKS:                  # route R1 = tasks.py TaskDef.routes, ASSUMED
        steps = [dict(skill="lift", args=dict(dz=0.05), why="raise the object clear of counter 1"),
                 dict(skill="base_move", args=dict(route=[dict(line=-0.5), dict(turn=180.0), dict(line=0.5)]), why="route R1: back, turn around, forward to counter 2"),
                 dict(skill="place", args=dict(support="counter2"), why="lower it onto counter 2"),
                 dict(skill="release", args={}, why="let go")]
        return steps, {"0": [dict(skill="set_down", args={})], "1": [dict(skill="retrace", args={})], "2": [dict(skill="hold", args={})]}
    if task == "box":
        steps = [dict(skill="lift", args=dict(dz=0.18), why="raise the crate clear of the counter and above the shelf lip"),
                 dict(skill="move", args=dict(dx=-0.20, dy=0.0, dz=0.0), why="carry it over the shelf"),
                 dict(skill="place", args=dict(support="shelf"), why="lower it onto the shelf"),
                 dict(skill="release", args={}, why="let go")]
    else:
        steps = [dict(skill="lift", args=dict(dz=0.05), why="raise the object off the counter"),
                 dict(skill="move", args=dict(dx=-0.10, dy=0.0, dz=0.0), why="carry it to the goal"),
                 dict(skill="place", args=dict(support="counter"), why="set it down"),
                 dict(skill="release", args={}, why="let go")]
    return steps, {str(i): [dict(skill="set_down", args={})] for i in range(3)}


def fixture_plan_v2(task, variant="fixture"):
    """schema v2 reply: one phenomenon entry per catalog phenomenon of the object kind (PH1 -> weigh, the others unmeasurable), probes [weigh], the fixture motion."""
    from world_model.v2.stack.skills import catalog as CAT
    k = CAT.kind_of(task)
    ph = []
    for pid, p in sorted(CAT.PHENOMENA.items()):
        if k not in p.objects:
            continue
        if pid == "PH1" and variant != "omit_ph1":
            ph.append(dict(id=pid, why="the hidden contents add mass the arms must carry", handle=dict(measure="weigh")))
        elif pid == "PH1":
            continue
        else:
            ph.append(dict(id=pid, why=f"{p.name} can change the carry", handle=dict(unmeasurable="no measuring skill lists this phenomenon")))
    probes = [] if variant == "omit_ph1" else [dict(skill="weigh")]
    if variant == "wrong_skill":
        for e in ph:
            if e["id"] == "PH2":
                e["handle"] = dict(measure="weigh")
    if variant == "new_phenomenon":
        ph.append(dict(id="NEW", name="contents shift", why="the contents may move during the carry", handle=dict(unmeasurable="no skill watches the contents move")))
    steps, rec = _motion(task)
    d = dict(phenomena=ph, probes=probes, steps=steps, recoveries=rec, fallback=[dict(skill="let_go", args={})])
    if variant == "numeric":
        d["belief"] = {"contents_kg": [0.5, 5.0]}
        d["recoveries"]["0"][0]["covers"] = {"contents_kg": [0.5, 2.0]}
    return d


def fixture_plan(task, narrow=False):
    """the fixture plan as the reply dict, in the format of the mode: schema v2 (default) or, under BIMANUAL_LEGACY_PROPOSER_V1=1 or narrow=True, the v1 plan
    (belief = the task's sanity ranges).  Same motion shapes as carry.PLANS (ASSUMED, user-confirmed 11-02)."""
    if not narrow and not S.legacy_v1():
        return fixture_plan_v2(task)
    return fixture_plan_v1(task, narrow)


def fixture_plan_v1(task, narrow=False):
    """the 11-05..11-09 reply: belief + steps + recoveries + fallback."""
    P = S.params(task)
    belief = {k: [round(float(v["range"][0]), 4), round(float(v["range"][1]), 4)] for k, v in P.items()}
    if narrow:
        k = next(k for k, v in P.items() if v["probed"])
        lo, hi = belief[k]
        belief[k] = [lo, lo + 0.1 * (hi - lo)]
    steps, rec = _motion(task)
    return dict(belief=belief, steps=steps, recoveries=rec, fallback=[dict(skill="let_go", args={})])


class MockProposer(Backend):
    name = "mock-proposer"

    def __init__(self, kind="fixture"):
        if kind not in KINDS:
            raise ValueError(f"unknown mock kind {kind!r}; have {KINDS}")
        self.kind, self.model = kind, f"mock-{kind}"
        self.calls = 0

    def ask(self, text, images=(), system=None, max_new_tokens=512, temperature=0.0, n=1):
        self.calls += 1
        m = re.search(r"TASK_ID: (\w+)", text)
        task = m.group(1)
        if "NEW MEASUREMENT" in text or "RULE 2 RE-ASK" in text:
            reply = json.dumps(fixture_plan(task))
        elif self.kind == "t2_reask":
            reply = json.dumps(fixture_plan_v2(task, "omit_ph1"))
        elif self.kind == "malformed":
            reply = "Sure, here is my plan: {\"belief\": {\"contents_kg\": [0, 5]}, \"steps\": [ lift the thing"
        elif self.kind in ("omit_ph1", "wrong_skill", "new_phenomenon", "numeric"):
            reply = json.dumps(fixture_plan_v2(task, self.kind))
        elif self.kind == "unknown_skill":
            d = fixture_plan(task)
            d["steps"][1] = dict(skill="teleport", args=dict(x=0.0), why="faster")
            reply = json.dumps(d)
        else:
            reply = json.dumps(fixture_plan(task, narrow=(self.kind == "narrow")))
        return dict(texts=[reply] * n, seconds=0.0, wall_s=0.0, cached=False, backend=self.name)
