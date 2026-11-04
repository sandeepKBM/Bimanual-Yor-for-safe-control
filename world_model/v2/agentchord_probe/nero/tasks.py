"""agentchord_probe/nero/tasks.py -- the task variants, their prompt text (task description + the SAME facts text our VLM gets), the world lists
(same seeds as carry_cert / baseline_ungated) and the scene image.  SIMULATION ONLY."""
from __future__ import annotations

import hashlib

import numpy as np

TASKS = ("tray2", "tray4", "tray6", "box")
PRIOR = {"tray": (0.0, 0.25), "box": (0.5, 5.0)}            # ASSUMED prior boxes (contents_eval.PRIOR_FILL / PRIOR_BOX)

TRAY_TEXT = (
    "Use the robot's two arms together to carry the coffee tray (object name 'tray') on the counter (object name 'counter', top surface at "
    "z = {counter_top:.2f} m). The tray's reference point is now at x = {x0:.2f} m, y = {y0:.2f} m. Carry it {dx:.2f} m toward the robot "
    "(x decreases by {dx:.2f} m, y unchanged) and set it down on the counter at x = {x1:.2f} m, y = {y1:.2f} m, then let go of it. "
    "Positions are world coordinates in metres (x forward, y left, z up).\n"
    "Known facts about the load: {facts}")
BOX_TEXT = (
    "Use the robot's two arms together to carry the closed box (object name 'box') from the counter (object name 'counter', top surface at "
    "z = {counter_top:.2f} m) onto the shelf (object name 'shelf', top surface at z = {shelf_top:.2f} m, spanning x = {sx0:.2f}..{sx1:.2f} m, "
    "y = {sy0:.2f}..{sy1:.2f} m). The box's reference point (centre of its underside) is now at x = {x0:.2f} m, y = {y0:.2f} m. Set it down on the "
    "shelf with its reference point at x = {x1:.2f} m, y = {y1:.2f} m, then let go of it. "
    "Positions are world coordinates in metres (x forward, y left, z up).\n"
    "Known facts about the load: {facts}")


def kind(task):
    return "tray" if task.startswith("tray") else "box"


def goal_xy(task, world):
    from .sim_imports import CR
    d = np.sum([np.asarray(dl, float) for _, dl in CR.PLANS[kind(task)]], axis=0)
    return (world.obj_xy[0] + d[0], world.obj_xy[1] + d[1])


def cell_list(task):
    from world_model.v2.nero.contents import carry_cert as CC
    return [c for (c,) in CC.cells(task)]


def world_for(task, cell, i, seeds="cert"):
    """the world of (task, cell, i).  seeds='cert': carry_cert.cell_seed (the worlds the certificate samples); 'baseline': baseline_ungated.seed_of."""
    from world_model.v2.nero.contents import baseline_ungated as BU, carry_cert as CC
    seed = CC.cell_seed(task, cell, i) if seeds == "cert" else BU.seed_of(task, cell, i)
    return seed, CC.sample_world(task, cell, np.random.default_rng(seed))


def rep_world(task):
    """representative world for the planner's image / facts: seed 0 over the FULL prior box (the facts text does not depend on the hidden fill)."""
    from world_model.v2.nero.contents import carry_cert as CC
    return CC.sample_world(task, PRIOR[kind(task)], np.random.default_rng(0))


def task_prompt(task, world=None):
    """-> (text, facts).  facts = vlm.contents_eval.facts_and_truth(...)[0]  (the same text our VLM gets; truth/prior are NOT put in the prompt)."""
    from world_model.v2.vlm import contents_eval as CE
    w = world or rep_world(task)
    facts = CE.facts_and_truth(task, w)[0]
    x1, y1 = goal_xy(task, w)
    common = dict(counter_top=w.counter_top, x0=w.obj_xy[0], y0=w.obj_xy[1], x1=x1, y1=y1, facts=facts)
    if kind(task) == "tray":
        return TRAY_TEXT.format(dx=w.obj_xy[0] - x1, **common), facts
    sy = (w.shelf_yc - w.shelf_yhalf, w.shelf_yc + w.shelf_yhalf)
    return BOX_TEXT.format(shelf_top=w.shelf_top, sx0=w.shelf_x[0], sx1=w.shelf_x[1], sy0=sy[0], sy1=sy[1], **common), facts


def render_rgb(world, view="torso"):
    """MuJoCo torso view of the scene at rest (contents_eval.VIEWS; arms at their initial pose, as in the VLM runs).  -> uint8 (480, 640, 3) RGB."""
    import os
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import mujoco
    from world_model.v2.nero.contents import scenes as SC
    from world_model.v2.nero.pan import scene as PS
    from world_model.v2.vlm import contents_eval as CE
    m = PS.apply_sim_defaults(SC.build_spec(world).compile())
    d = mujoco.MjData(m)
    SC.settle(m, d, 0.5)
    r = mujoco.Renderer(m, 480, 640)
    try:
        r.update_scene(d, camera=CE._cam(CE.VIEWS[view]))
        return np.ascontiguousarray(r.render())
    finally:
        r.close()


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]
