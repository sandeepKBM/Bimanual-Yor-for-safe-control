"""gatekeeper/ready.py -- C2-ready: a RANKING term over already-admitted segment candidates.  SIMULATION ONLY.

C2 (checks.check2) certifies that the cup-safe backup ends in SAFE REST (rest pose clears walls, robot stopped before any person disc can touch).  It does not
say the robot can CONTINUE from that rest pose: a wide side of the footprint facing a person who then stands still can leave no certified next move.
C2-ready asks: from the rest pose, with the people RE-OBSERVED at the rest time, does at least one library move pass check1 + check2 and make progress
>= gk.c2_ready_dmin?  Two hypotheses about the people after re-observation: "walk" (keep v_hat) and "stop" (stand still).
Ranking only: it never admits or rejects a candidate, so the T1 safety argument is unchanged.  The re-observed belief is the NOMINAL prediction
(ASSUMPTION; it concerns progress, not safety).  Opt-in: GKConfig.c2_ready = "rank" (default "off").
"""
from __future__ import annotations

import numpy as np

from .checks import world_poses

HYPS = ("walk", "stop")


def rest_poses(traj, bel):
    """-> (pose_wall_rest, pose_people_rest, t_rest) of the backup's rest pose (world frame, anchored as the belief's two poses), or None if no rest."""
    i = traj.i_rest
    if i is None:
        return None
    Pr = traj.poses()
    pw = world_poses(Pr[i:i + 1], bel.pose_wall)[0]
    pp = world_poses(Pr[i:i + 1], bel.pose_people)[0]
    return pw, pp, i * traj.h


def belief_at_rest(bel, pose_wall_r, pose_people_r, t_rest, hyp):
    """the belief re-observed at the rest time: every person advanced by v_hat * t_rest (both hypotheses; nominal prediction until the robot re-observes), then
    velocities v_hat ("walk") or zero ("stop"); observation age reset; poses re-anchored at the rest pose.  Other fields unchanged."""
    vel = tuple(bel.people_vel)
    people = []
    for i, c in enumerate(bel.people):
        v = np.asarray(vel[i], float)[:2] if vel else np.zeros(2)
        people.append(np.asarray(c, float)[:2] + v * float(t_rest))
    if vel:
        new_vel = tuple(np.asarray(v, float).copy() for v in vel) if hyp == "walk" else tuple(np.zeros(2) for _ in vel)
    else:
        new_vel = ()
    stale = tuple(0.0 for _ in bel.people_stale) if len(bel.people_stale) else ()
    return bel.with_(people=tuple(people), people_vel=new_vel, people_stale=stale,
                     pose_wall=np.asarray(pose_wall_r, float).copy(), pose_people=np.asarray(pose_people_r, float).copy())


def ready_score(traj, bel, gk, moves_fn, hyps=HYPS):
    """-> (score, n, detail): score = number of hypotheses under which moves_fn(b_r) returns a certified progress move (a string) instead of None."""
    rp = rest_poses(traj, bel)
    if rp is None:
        return 0, len(hyps), ["no_rest"]
    pw, pp, t_rest = rp
    score, detail = 0, []
    for h in hyps:
        mv = moves_fn(belief_at_rest(bel, pw, pp, t_rest, h))
        score += mv is not None
        detail.append(f"{h}:{mv if mv is not None else 'none'}")
    return int(score), len(hyps), detail
