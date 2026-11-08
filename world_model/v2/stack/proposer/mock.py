"""stack/proposer/mock.py -- scripted proposer replies for tests and dry runs (NOT a model).  SIMULATION ONLY.   (11-05)

spec "mock:<kind>":
  fixture        the legacy hand-written plan of the task (carry.PLANS shapes: tray / pot lift 5 cm, 10 cm toward the robot, place; box lift 18 cm, 20 cm, place on the
                 shelf) written as a proposer reply, belief = the task's sanity ranges (wide), recovery = set_down on every motion step, fallback = let_go
  malformed      prose around a broken JSON object
  unknown_skill  the fixture plan with one step "teleport"
  narrow         the fixture plan with a narrow belief on the probed parameter (to test Rule 2: probe set outside B)
A re-ask prompt ("NEW MEASUREMENT") is answered with the fixture plan whose belief is the sanity ranges.
"""
from __future__ import annotations

import json
import re

from world_model.v2.house.lang.backend import Backend
from world_model.v2.stack.proposer import schema as S

KINDS = ("fixture", "malformed", "unknown_skill", "narrow")


def fixture_plan(task, narrow=False):
    """the legacy plan as the reply dict.  Same shapes as carry.PLANS (ASSUMED, user-confirmed 11-02)."""
    P = S.params(task)
    belief = {k: [round(float(v["range"][0]), 4), round(float(v["range"][1]), 4)] for k, v in P.items()}
    if narrow:
        k = next(k for k, v in P.items() if v["probed"])
        lo, hi = belief[k]
        belief[k] = [lo, lo + 0.1 * (hi - lo)]
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
    rec = {str(i): [dict(skill="set_down", args={})] for i in range(3)}
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
        if "NEW MEASUREMENT" in text:
            reply = json.dumps(fixture_plan(task))
        elif self.kind == "malformed":
            reply = "Sure, here is my plan: {\"belief\": {\"contents_kg\": [0, 5]}, \"steps\": [ lift the thing"
        elif self.kind == "unknown_skill":
            d = fixture_plan(task)
            d["steps"][1] = dict(skill="teleport", args=dict(x=0.0), why="faster")
            reply = json.dumps(d)
        else:
            reply = json.dumps(fixture_plan(task, narrow=(self.kind == "narrow")))
        return dict(texts=[reply] * n, seconds=0.0, wall_s=0.0, cached=False, backend=self.name)
