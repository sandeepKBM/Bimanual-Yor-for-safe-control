"""VLM Reasoning Module (VRM), as in VIZ-COAST: inputs = PDDL domain + problem, scene image, text geometric representation,
grasp specification, example compliant output; output = (1) refinement issues in words, (2) a Python program against the Z3
constraint API.  Re-prompts: revise (after K refinement failures, with the feedback), relax (no plan), fix (program error).
Backends: any house/lang/backend.Backend.  For now a scripted MockVLM (profiles below); the paper used Gemini 3 Flash (medium).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from . import satplan as SP

PDDL = Path(__file__).with_name("pan.pddl")

SYSTEM = ("You are the reasoning module of a task and motion planner. You do not write plans. You read a PDDL domain and "
          "problem, look at the scene, and write task-level constraints that prevent task plans which cannot be refined into "
          "feasible robot motions or which would fail in the real scene.")

API_DOC = "\n".join(f"  api.{n}{_sig}: {getattr(SP.ConstraintAPI, n).__doc__}" for n, _sig in
                    (("steps", "()"), ("action", "(name, *args, t)"), ("actions", "(name=None, t=None)"), ("holds", "(pred, *args, t)"),
                     ("disallow", "(name, *args, when=None)"), ("add", "(*exprs)")))

EXAMPLE = '''Example (a different domain: items in containers with lids):
Issues: 1. An item cannot be picked from or placed into a container whose lid is closed; the lid must be removed first.
```python
def constraints(api):
    for c in ("box1", "box2"):
        for o in ("cup", "spoon"):
            api.disallow("pick", o, c, when=lambda t, c=c: api.holds("closed", c, t=t))
            api.disallow("place", o, c, when=lambda t, c=c: api.holds("closed", c, t=t))
```'''

GRASP_SPEC = ("Each arm has a parallel two-finger claw: max opening 100 mm, fingers 80 mm long, squeeze up to 10 N. The left claw "
              "takes the handle from its end (approach along -y, fingers above and below the handle). The right claw can reach the "
              "far side of the pan (approach along +y) or the handle from the side or tilted.")

PROMPT = """KIND: {kind}
PDDL domain and problem:
{pddl}

Scene geometry (world frame, metres; x forward, y left, z up):
{geom}

Grasp specification: {grasp}

Constraint API (Z3 expressions; And, Or, Not, Implies, If, AtMost, AtLeast are available; no imports):
{api}

{example}

{task}
Step 1: list the issues that would make a task plan fail when refined into motions or executed (one line each).
Step 2: write ONE Python code block defining constraints(api) that encodes them. Reply with the issues and then the code block."""


def geometry_text(scene):
    m, d, p = scene.m, scene.d, scene.p
    c = d.xpos[m.body("pan").id]
    u = p.handle_axis()
    h0 = c + u * (p.pan_radius + p.pan_wall_t)
    h1 = h0 + u * p.handle_len
    lid = d.xpos[m.body("lid").id]
    from .. import scene as S
    rows = [f"stove block: top centre ({p.pan_centre_xy[0]:.3f}, {p.pan_centre_xy[1]:.3f}), half size {list(S.ASSUMED['stove_half'])}",
            f"pan: centre {np.round(c, 3).tolist()}, inner radius {p.pan_radius}, wall height {p.pan_wall_h}, wall {p.pan_wall_t}",
            f"handle: from {np.round(h0, 3).tolist()} to {np.round(h1, 3).tolist()}, width 0.024, thickness {p.handle_t}",
            f"lid: centre {np.round(lid, 3).tolist()}, resting on the pan (not fixed), knob on top",
            f"left arm base {np.round(d.xpos[m.body('left_arm_base_link').id], 3).tolist() if _has(m, 'left_arm_base_link') else 'n/a'}, "
            f"right arm base {np.round(d.xpos[m.body('right_arm_base_link').id], 3).tolist() if _has(m, 'right_arm_base_link') else 'n/a'}"]
    return "\n".join(rows)


def _has(m, name):
    import mujoco
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name) >= 0


def extract_code(text):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text or "", re.S)
    return blocks[-1] if blocks else None


class VRM:
    def __init__(self, backend, scene, image=None):
        self.b, self.image = backend, image
        self.geom = geometry_text(scene)
        self.pddl = PDDL.read_text()
        self.log = []

    def _ask(self, kind, task):
        txt = PROMPT.format(kind=kind, pddl=self.pddl, geom=self.geom, grasp=GRASP_SPEC, api=API_DOC, example=EXAMPLE, task=task)
        out = self.b.ask(txt, images=[self.image] if self.image else [], system=SYSTEM, max_new_tokens=2000, temperature=0.0)
        raw = out["texts"][0] if out.get("texts") else ""
        code = extract_code(raw)
        self.log.append(dict(kind=kind, raw=raw[:3000], has_code=code is not None, seconds=out.get("seconds")))
        return code

    def initial(self):
        return self._ask("initial", "Task: lift the covered pan off the stove.")

    def revise(self, code, feedback):
        fb = "\n".join(f"- plan {f['plan']}: refinement failed at {f['action']}: {f['reason']}" for f in feedback)
        return self._ask("revise", f"Your previous constraints:\n```python\n{code}\n```\nRefinement feedback:\n{fb}\nRevise the constraints.")

    def relax(self, code, horizon):
        return self._ask("relax", f"Your previous constraints:\n```python\n{code}\n```\nNo task plan exists up to horizon {horizon}. "
                                  "Make the constraints less restrictive.")

    def fix(self, code, err):
        return self._ask("error", f"Your previous constraints:\n```python\n{code}\n```\nApplying them raised:\n{err}\nFix the program.")


# --- scripted profiles (what a VLM might write; NOT measured model behaviour) -------------------------------------------
P_NONE = "Issues: none.\n```python\ndef constraints(api):\n    pass\n```"
P_LID = '''Issues:
1. The lid rests loose on the pan; lifting the pan while the lid is not secured can drop the lid.
```python
def constraints(api):
    for name, args, t, v in api.actions("lift-one") + api.actions("lift-two"):
        api.add(Implies(v, api.holds("lid-secured", t=t)))
```'''
P_TWO = '''Issues:
1. The lid rests loose on the pan; secure it before lifting.
2. A covered pan with contents is heavy and long; one claw on the handle cannot hold it level. Lift with two arms.
```python
def constraints(api):
    for name, args, t, v in api.actions("lift-one") + api.actions("lift-two"):
        api.add(Implies(v, api.holds("lid-secured", t=t)))
    for name, args, t, v in api.actions("lift-one"):
        api.add(Not(v))
```'''
P_BUGGY = '''Issues: 1. secure the lid.
```python
def constraints(api):
    for name, args, t, v in api.actions("lift-two"):
        api.add(Implies(v, api.holds("lid-secured", t=t)))
    api.add(Not(api.actions("lift-one")))
```'''
PROFILES = dict(none=dict(initial=P_NONE), lid=dict(initial=P_LID), two_arm=dict(initial=P_TWO),
                buggy=dict(initial=P_BUGGY, error=P_LID))


def mock_backend(profile):
    from world_model.v2.house.lang.backend import MockVLM
    prof = PROFILES[profile]
    rules = [(rf"KIND: {k}", v) for k, v in prof.items()]
    rules.append((r"KIND: ", prof["initial"]))           # revise / relax: the mock keeps its program
    return MockVLM(rules=rules, use_default=False)
