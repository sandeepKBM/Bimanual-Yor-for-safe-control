"""path_reach.py -- is every NOMINAL carry path reachable for each arm?  (11-03; SIMULATION ONLY)

Replays carry.CarryEpisode._add_path's continuation IK (same PoseIK, same smoothstep, 60 waypoints, orientation held at the start of each move) for the
move lists below, from a hold-pose q, and records per move: the largest pose-IK miss over all waypoints [m; rad norm] and the smallest joint-limit distance [rad].
Move lists (TCP deltas of one arm; the left arm retreats +y, the right -y, as carry.run_phases):
  plan    carry.PLANS[task] (ASSUMED, user-confirmed 11-02) + release retreat carry.RETREAT
  ac_box  AgentChord cached plan_box moves: up 0.25, x -0.2, down 0.1 (the graph says dz -0.1), release retreat 0.05, withdraw 0.1 outward, home (back to the hold TCP)
  ac_tray AgentChord place_on_table of the tray (x -0.10, y 0 [goal_xy of tasks.py], +pre 0.05, then -pre), release retreat 0.05, home
Thresholds: carry.PATH_IK_TOL (ASSUMED 1e-4) and carry.PATH_MARGIN_MIN (ASSUMED 0.05 rad).
"""
from __future__ import annotations

import numpy as np

from ..democ import kin as K
from . import carry as CR

SIGN = {"left": 1.0, "right": -1.0}


def move_lists(task):
    """-> {name: [("d", delta_left_frame_signs applied later) ...]}; deltas are (dx, dy, dz) with y given for the LEFT arm (right arm flips y)."""
    plan = [("d", d) for _, d in CR.PLANS[task]] + [("d", (0, CR.RETREAT, 0))]
    if task == "box":
        ac = [("d", (0, 0, 0.25)), ("d", (-0.2, 0, 0)), ("d", (0, 0, -0.1)), ("d", (0, 0.05, 0)), ("d", (0, 0.1, 0)), ("home", None)]
        return dict(plan=plan, ac=ac)
    ac = [("d", (-0.10, 0, 0.05)), ("d", (0, 0, -0.05)), ("d", (0, 0.05, 0)), ("home", None)]
    return dict(plan=plan, ac=ac)


def eval_moves(m, side, q_hold, moves, p_hold=None, stop_on_bad=False):
    """-> list of dict(i, err_max, margin_min, margin_j) per move, continuing from the end of the previous move."""
    pik = K.PoseIK(m, side)
    q = np.asarray(q_hold, float).copy()
    p_home = pik.ik.pose(q)[0] if p_hold is None else np.asarray(p_hold, float)
    out = []
    for i, (kind, dl) in enumerate(moves):
        p0, R0 = pik.ik.pose(q)
        if kind == "home":
            delta = p_home - p0
        else:
            delta = np.asarray(dl, float) * np.array([1.0, SIGN[side], 1.0])
        q_post, e_max = q.copy(), 0.0
        mg, mj = pik.margin(q)
        for k in range(1, 61):
            a = k / 60
            q, err = pik.solve(p0 + delta * a * a * (3 - 2 * a), R0, q, q_post=q_post, k_post=0.1, iters=60, tol=1e-9)
            e_max = max(e_max, float(err))
            g, j = pik.margin(q)
            if g < mg:
                mg, mj = g, j
        out.append(dict(i=i, kind=kind, err_max=e_max, margin_min=mg, margin_j=mj))
        if stop_on_bad and (e_max > CR.PATH_IK_TOL or mg < CR.PATH_MARGIN_MIN):
            break
    return out


def summarize(m, side, q_hold, task, p_hold=None, stop_on_bad=False):
    """-> {list name: dict(err_max, margin_min, bad_moves)} for the arm."""
    res = {}
    for name, moves in move_lists(task).items():
        r = eval_moves(m, side, q_hold, moves, p_hold, stop_on_bad)
        res[name] = dict(err_max=max(x["err_max"] for x in r), margin_min=min(x["margin_min"] for x in r),
                         bad=[x["i"] for x in r if x["err_max"] > CR.PATH_IK_TOL or x["margin_min"] < CR.PATH_MARGIN_MIN], moves=r)
    return res


def ok(res):
    return all(not v["bad"] for v in res.values())
