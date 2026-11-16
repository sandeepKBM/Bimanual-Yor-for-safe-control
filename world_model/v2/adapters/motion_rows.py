"""adapters/motion_rows.py -- motion rows (checks/motion.py) built from a planned ClosedChainPlan (physics: MuJoCo FK through SRSArm).

Per plan piece (a rest-to-rest segment, arrays at the control period): joint position (URDF limits, margin = carry.PATH_MARGIN_MIN, ASSUMED), joint speed
(datasheet config.MANUAL, MODEL), reach (FK at the planner geometry knots vs the planned tool pose, pos and rot tolerance separately; tolerance carry.PATH_IK_TOL, ASSUMED), closed chain (object pose implied by
each arm).  The four rows are concatenated over the pieces (the worst piece binds; `worst` names it).

Closed-chain frames (GUESS, stated): the planner holds each tool pose rigidly fixed to the object, T_tcp_s(k) = T0_s with translation + p_k.  The object pose
implied by arm s is  T_obj_s(k) = T_fk_s(q_k) @ inv(T0_s) @ T_obj0,  with T_obj0 = [I | midpoint of the two hold tool positions] (the object's world pose is not held by
the plan; this gives the grasp transform T_grasp_s = inv(T0_s) @ T_obj0 as the fixed tool->object transform, rotation identity assumed).  With T_fk_s = FK of the
joint sample, any FK miss of either arm shows up as a disagreement of the two implied object poses.
"""
from __future__ import annotations

import numpy as np

from world_model.v2.nero import config as C
from world_model.v2.stack.checks import motion as M
from world_model.v2.stack.planners import closed_chain as CC

SIDES = CC.SIDES


def _limits():
    from world_model.v2.nero.contents import carry
    lo, hi = C.URDF_LIMITS[:, 0], C.URDF_LIMITS[:, 1]
    qd = np.deg2rad(np.asarray(C.MANUAL["joint_speed_deg_s"], float))
    return lo, hi, qd, carry.PATH_MARGIN_MIN, carry.PATH_IK_TOL


def piece_arrays(plan, sg):
    """-> t (n+1,), q {side}, qd {side}, p (n+1,3) of one piece."""
    pc = plan.pieces[sg]
    t = np.arange(pc.p.shape[0]) * CC.DT
    return t, {s: np.asarray(pc.q[s], float) for s in SIDES}, {s: np.asarray(pc.qd[s], float) for s in SIDES}, np.asarray(pc.p, float)


def _fk_arrays(plan, q, p):
    """FK of every sample -> (miss_pos, miss_rot, T_obj_from_side {side: (n,4,4)})."""
    inp = plan.inp
    mid = 0.5 * (inp.T0["left"][:3, 3] + inp.T0["right"][:3, 3])
    T_obj0 = np.eye(4)
    T_obj0[:3, 3] = mid
    pe = {s: np.zeros(len(p)) for s in SIDES}
    re = {s: np.zeros(len(p)) for s in SIDES}
    To = {s: np.zeros((len(p), 4, 4)) for s in SIDES}
    for s in SIDES:
        arm, T0 = inp.arms[s], inp.T0[s]
        T0inv = np.linalg.inv(T0)
        for i in range(len(p)):
            T = arm.fk(q[s][i])
            Tt = CC._tcp_pose(T0, p[i])
            pe[s][i] = np.linalg.norm(T[:3, 3] - Tt[:3, 3])
            re[s][i] = float(M.rot_angle(Tt[:3, :3].T @ T[:3, :3]))
            To[s][i] = T @ T0inv @ T_obj0
    return np.maximum(pe["left"], pe["right"]), np.maximum(re["left"], re["right"]), To


def _knot_miss(plan, sg):
    """reach at the planner's GEOMETRY knots of segment sg (design 3.3): FK of the chosen configuration vs the target pose, worst of the two arms.
    -> (pos miss [m] (k,), rot miss [rad] (k,), knot index (k,))  (position and rotation stay separate; reach_row tests each against its own tolerance)."""
    g, inp = plan.geo, plan.inp
    i0, i1 = g.bounds[sg]
    pe, re = np.zeros(i1 - i0 + 1), np.zeros(i1 - i0 + 1)
    for s in SIDES:
        a, b = CC._fk_check(inp.arms[s], inp.T0[s], g.offs[i0:i1 + 1], g.q[s][i0:i1 + 1])
        pe, re = np.maximum(pe, a), np.maximum(re, b)
    return pe, re, np.arange(i0, i1 + 1)


def motion_rows(plan, model=None, pieces=None):
    """plan: closed_chain.ClosedChainPlan (model is unused: the plan's own SRSArm/MuJoCo model does the FK; kept for the registry signature).
    pieces: optional subset of segment names.  -> [joint_pos, joint_speed, reach, closed_chain] rows over all pieces."""
    lo, hi, qd_max, margin, ik_tol = _limits()
    names = list(pieces) if pieces else list(plan.pieces)
    per = {"joint_pos": [], "joint_speed": [], "reach": [], "closed_chain": []}
    for sg in names:
        t, q, qd, p = piece_arrays(plan, sg)
        _, _, To = _fk_arrays(plan, q, p)
        mp, mr, tk = _knot_miss(plan, sg)
        per["joint_pos"].append((sg, M.joint_pos_row(q, t, lo, hi, margin)))
        per["joint_speed"].append((sg, M.joint_speed_row(qd, t, qd_max)))
        per["reach"].append((sg, M.reach_row(mp, mr, ik_tol, ik_tol, t=tk)))
        per["closed_chain"].append((sg, M.closed_chain_row(To["left"], To["right"], t=t)))
    rows = []
    for name, lst in per.items():
        sg, r = min(lst, key=lambda x: x[1].margin)               # binding piece
        r.worst = dict(piece=sg, **(r.worst or {}))
        if any(not x.ok for _, x in lst):
            r = next(x for _, x in lst if not x.ok) if r.ok else r
        r.why = ("[" + sg + "] " + r.why) if r.why else r.why
        r.name = name
        rows.append(r)
    return rows


def scripted_rows(task):
    """legacy scripted path: NOT built.  The ScriptedPlanner (carry.PLANS) exposes object offsets and a per-waypoint IK inside CarryEpisode.run (carry.py,
    pose-IK miss recorded at ~l.526), but no per-sample joint arrays at the control period outside a live episode; deriving them would re-run the controller."""
    raise NotImplementedError("scripted_rows: the scripted planner has no joint arrays outside an episode (see docstring)")
