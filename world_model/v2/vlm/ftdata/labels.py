"""labels.py -- prompts + labels of the pick-proposer fine-tune data.  Pure python (no mujoco / GL): imports only pick.planner / pick.assign.

The prompts are the planner's own (planner.SYSTEM / planner.USER, formatted exactly as GraspPlanner._ask does).  The assistant answer follows the planner's JSON schema.
LABEL RULES (authoritative = pick/assign.py, exact geometry):
  arms/arm/grasp : assign.assign((a, b, h), centroid, yaw, mass_vlm = real-range HI) on the NOISY extent (Gaussian depth noise sigma 5 mm on a, b, h).  Nothing else.
  mass_kg        : the REAL-WORLD range of the category (priors.mass_range), never the sim mass.
  contents       : the prior table's value ("empty": sim objects are empty).   fragile / handle: the prior table.
  why            : short template from the rule's reason (<= 15-20 words).
  infeasible     : assign() says "infeasible" -> policy "least_bad" (default) answers with the less-violated option and a 'No grasp fits' why; "drop" skips the sample.
                   Either way meta.rule_mode == "infeasible" so a consumer can filter.  (The veto replaces such a plan by the rule's infeasible plan at inference anyway.)
"""
from __future__ import annotations

import json
import math

import numpy as np

from world_model.v2.house.pick import assign as A
from world_model.v2.house.pick import planner as PL

from . import priors as P

NOISE_SIGMA_M = 0.005
HEAD_X, HEAD_Y, HEAD_Z = 0.05, -0.20955, 1.15        # = ocs/fp_render.HEAD_POS (robot frame, lift 0); a test asserts equality
NAME_SYSTEM = ("You see a robot head-camera image with ONE object boxed in yellow. Answer with the name of the boxed object only, in one or two words, lower case, no punctuation.")
NAME_USER = "Name the boxed object in one or two words."
INFEASIBLE_POLICIES = ("least_bad", "drop")


def system_prompt():
    return PL.SYSTEM.format(open_cm=A.SINGLE_MAX_WIDTH_M * 100, pay=A.SINGLE_MAX_KG, minsep=A.MIN_HAND_SEP_M * 100, maxspan=A.MAX_SPAN_M * 100)


def user_prompt(label, a, b, h, x, y):
    """exactly GraspPlanner._ask: side = left iff y > CENTRE_Y."""
    return PL.USER.format(label=label, a_cm=a * 100, b_cm=b * 100, h_cm=h * 100, x=float(x), y=float(y), side="left" if y > A.CENTRE_Y else "right")


def noisy_extent(rng, a, b, h, sigma=NOISE_SIGMA_M):
    """depth-noise on the true OBB extent -> (a >= b, h), rounded to 0.1 mm (the stored numbers are what the rule is applied to)."""
    v = np.maximum(np.array([a, b, h]) + rng.normal(0.0, sigma, 3), 0.005)
    a2, b2 = max(v[0], v[1]), min(v[0], v[1])
    return round(float(a2), 4), round(float(b2), 4), round(float(v[2]), 4)


def _violation_single(b, mass):
    return max(b / A.SINGLE_MAX_WIDTH_M, mass / A.SINGLE_MAX_KG)


def _violation_bimanual(a, b, mass):
    wv = min(max(A.MIN_HAND_SEP_M / w, w / A.MAX_SPAN_M) for w in (a, b))
    return max(1.0, wv, mass / (2 * A.SINGLE_MAX_KG))


def _why(plan, a, b, mass, policy_pick=None):
    b_cm, a_cm = b * 100, a * 100
    wide, heavy = b > A.SINGLE_MAX_WIDTH_M, mass > A.SINGLE_MAX_KG
    r1 = "too wide and heavy" if wide and heavy else "too wide" if wide else "too heavy"
    if plan.mode == "single":
        return f"Narrow side {b_cm:.1f} cm fits the jaws; est. {mass:.2f} kg is within one arm's limit."
    if plan.mode == "bimanual":
        return f"{r1.capitalize()} for one arm; two hands squeeze across {plan.width * 100:.0f} cm."
    w_ok = A.bimanual_axis((a, b), 0.0)[0] is not None
    r2 = f"over the {2 * A.SINGLE_MAX_KG:.1f} kg two-arm limit" if w_ok else f"no {A.MIN_HAND_SEP_M * 100:.0f}-{A.MAX_SPAN_M * 100:.0f} cm axis for two hands"
    return f"No grasp fits: {r1} for one arm ({b_cm:.1f} cm, {mass:.2f} kg); {r2}."


def rule_label(a, b, h, centroid, yaw, mass_hi, policy="least_bad"):
    """-> dict(arms, arm, grasp, why, rule_mode, plan) or None (infeasible + policy 'drop').  `plan` is the assign.Plan."""
    plan = A.assign((a, b, h), centroid, yaw, mass_vlm=mass_hi)
    mass = plan.mass_est
    if plan.mode == "single":
        arms, arm, grasp = 1, plan.arm, "pinch"
    elif plan.mode == "bimanual":
        arms, arm, grasp = 2, "both", "squeeze"
    else:
        if policy == "drop":
            return None
        if _violation_single(b, mass) <= _violation_bimanual(a, b, mass):
            arms, arm, grasp = 1, A.arm_for(centroid[1]), "pinch"
        else:
            arms, arm, grasp = 2, "both", "squeeze"
    return dict(arms=arms, arm=arm, grasp=grasp, why=_why(plan, a, b, mass), rule_mode=plan.mode, plan=plan)


def answer_obj(rl, mass_range, prior):
    return {"arms": rl["arms"], "arm": rl["arm"], "grasp": rl["grasp"], "mass_kg": list(mass_range), "contents": prior.contents, "fragile": bool(prior.fragile),
            "handle": bool(prior.has_handle), "why": rl["why"]}


def pick_label(rng, category, table=None, p_generic=0.04, p_wrong=0.06):
    """label shown to the VLM: usually the true display name; sometimes 'object' (unknown) or a WRONG category name (the detector may be wrong -> trust the image).
    -> (label, kind in {'true','generic','wrong'})."""
    table = table or P.load()
    u = rng.random()
    if u < p_generic:
        return "object", "generic"
    if u < p_generic + p_wrong:
        others = sorted({p.display for k, p in table.items() if k != category and p.display != table[category].display})
        return str(others[int(rng.integers(len(others)))]), "wrong"
    return table[category].display, "true"


def arms_record(image_path, label, ext_noisy, centroid, answer, meta):
    a, b, h = ext_noisy
    return {"messages": [{"role": "system", "content": system_prompt()},
                         {"role": "user", "content": [{"type": "image", "image": str(image_path)},
                                                      {"type": "text", "text": user_prompt(label, a, b, h, centroid[0], centroid[1])}]},
                         {"role": "assistant", "content": json.dumps(answer)}],
            "meta": meta}


def name_record(image_path, display, meta):
    return {"messages": [{"role": "system", "content": NAME_SYSTEM},
                         {"role": "user", "content": [{"type": "image", "image": str(image_path)}, {"type": "text", "text": NAME_USER}]},
                         {"role": "assistant", "content": display}],
            "meta": meta}
