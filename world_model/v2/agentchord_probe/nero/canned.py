"""agentchord_probe/nero/canned.py -- hand-written plans in THEIR schema (task graph + recovery spec) for tests / dry runs (`plan --backend canned`).
NOT model output.  Replies list = [task_graph_json, recovery_spec_json]."""
from __future__ import annotations

import json

L, R = "left_arm", "right_arm"


def _both(fn, **kw):
    return dict(left_arm_action=dict(fn=fn, kwargs=dict(robot_name=L, **kw)), right_arm_action=dict(fn=fn, kwargs=dict(robot_name=R, **kw)))


def graph(task, edges):
    """edges: list of (id_suffix, left_action|None, right_action|None)."""
    n = len(edges)
    nodes = [dict(id="v0_start", semantic="start")] + [dict(id=f"v{i + 1}_s{i + 1}", semantic=f"after edge {i + 1}") for i in range(n)]
    es = []
    for i, (name, la, ra) in enumerate(edges):
        es.append(dict(id=f"e{i + 1:02d}_{name}", source=nodes[i]["id"], target=nodes[i + 1]["id"], left_arm_action=la, right_arm_action=ra))
    nodes[-1]["id"] = "v%d_done" % n
    es[-1]["target"] = nodes[-1]["id"]
    return dict(task=task, start="v0_start", goal=nodes[-1]["id"], nodes=nodes, edges=es)


def act(fn, arm, **kw):
    return dict(fn=fn, kwargs=dict(robot_name=arm, **kw))


def two_arm(task):
    """grasp, lift, carry, place + release (the PLANS of carry.py as AgentChord atoms)."""
    box = task == "box"
    obj = "box" if box else "tray"
    lift, carry_dx = (0.18, -0.20) if box else (0.05, -0.10)
    x_t, pre = (0.18, 0.03) if box else (0.28, 0.05)
    mk = lambda fn, **kw: (lambda b: (b["left_arm_action"], b["right_arm_action"]))(_both(fn, **kw))
    e = [("grasp", *mk("grasp", obj_name=obj, pre_grasp_dis=0.1)),
         ("lift", *mk("move_by_relative_offset", dx=0.0, dy=0.0, dz=lift, mode="extrinsic")),
         ("carry", *mk("move_by_relative_offset", dx=carry_dx, dy=0.0, dz=0.0, mode="extrinsic")),
         ("place", *mk("place_on_table", obj_name=obj, x=x_t, y=-0.30, pre_place_dis=pre))]
    return graph(f"carry_{obj}", e)


def one_arm(task):
    g = two_arm(task)
    g["edges"][2]["right_arm_action"] = None                  # carry with the left arm only
    return g


def rotate(task):
    g = two_arm(task)
    g["edges"][2]["left_arm_action"] = act("rotate_eef", L, degree=-30)
    g["edges"][2]["right_arm_action"] = act("rotate_eef", R, degree=30)
    return g


def empty_recovery(g):
    return dict(task=g["task"], recovery_bindings=[])


def loop_recovery(g, thr=0.001):
    """object_moved(thr) on the lift edge + retry: fires on normal motion when thr is below the per-sample displacement -> recovery runs."""
    return dict(task=g["task"], recovery_bindings=[dict(edge_id=g["edges"][1]["id"], failure_name="moved", monitors=[
        dict(type="object_moved", objects=["tray" if "tray" in g["task"] else "box"], threshold=thr)],
        recovery=[dict(type="retry_failed_edge")], merge="target", repeat_until_success=False)])


def replies(kind, task="tray4", **kw):
    g = dict(two_arm=two_arm, one_arm=one_arm, rotate=rotate, loop=two_arm)[kind](task)
    rec = loop_recovery(g, **kw) if kind == "loop" else empty_recovery(g)
    return [json.dumps(g), json.dumps(rec)]
