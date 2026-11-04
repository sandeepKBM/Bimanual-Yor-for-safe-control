"""pan/skills/pan_skills.py -- the covered-pan skills written in the v2 action DSL (skills/actions.py).  SIMULATION ONLY.

Same task chain and the same plan parameters as v1 (segments.segment_phases), so skills_check can compare v1 and v2 on the same
worlds; the differences are in how each skill moves:
    pinch : Close L, Hold                                         (as v1)
    push  : MoveRel L by -push * u_aim (mink path, clearance-checked), Hold
    clamp : MoveRel R in by APPROACH_IN (guarded: stops on claw contact force > CLAMP_GUARD), Close R, Hold
    lift  : payload model on, MoveObject +LIFT (both claws carry ONE rigid pan-frame motion), Hold
Marks (push_start, clamp_start, lift_start) are recorded as v1 does, so the cert/ backups still find them.
"""
from __future__ import annotations

import numpy as np

from . import actions as A

CLAMP_GUARD = None    # design: None = the right approach runs to its target as v1 (guarded variant: set a force [N])


LIFT_DEFAULT = "v2"   # "v2": MoveObject from the arms' reference (10-30); "v3" (11-01): Relax onto the measured joints, then MoveObject
                      # anchored at the MEASURED claw poses (the claws' real grasp), so the two arms stop pushing on each other;
                      # "v4" (11-01) = v3 + the right arm FOLLOWS the measured left claw (actions.Follower, reference level only)


def run(ep, seg, monitor=None, lift=None):
    """run skill `seg` on episode ep.  -> (status, info) like Episode.run_phases ("done" | "abort" | "plan_fail")."""
    from .. import segments as S
    from ..rim_geometry import APPROACH_IN, RIGHT_APPROACH
    ex = A.Executor(ep, monitor=monitor)
    if seg == "start":
        ok = ex.run(both=[A.Hold(dur=0.5)])
    elif seg == "pinch":
        ok = ex.run(left=[A.Close("left", 0.7), A.Hold("left", 0.3)])
    elif seg == "push":
        ep.run_phases([("mark", "push_start")])
        ok = ex.run(left=[A.MoveRel("left", tuple(-ep.plan["push"] * ep.u_aim), dur=1.5, label="push"), A.Hold("left", 0.3)])
    elif seg == "clamp":
        ep.run_phases([("mark", "clamp_start")])
        app = ep.d.xmat[ep.gb["right"]].reshape(3, 3) @ ep.RgR0.T @ RIGHT_APPROACH     # the claw's own approach axis (FK)
        ok = ex.run(right=[A.MoveRel("right", tuple(APPROACH_IN * app), dur=1.0, guard=CLAMP_GUARD, label="clamp in"),
                           A.Close("right", 0.7), A.Hold("right", 0.3)])
    elif seg == "lift":
        ep.run_phases([("mark", "lift_start"), ("payload", True)])
        if (lift or LIFT_DEFAULT) in ("v3", "v4"):
            fol = "right" if (lift or LIFT_DEFAULT) == "v4" else None      # v4: the right (rim-clamp) arm follows the left claw
            ok = ex.run(both=[A.Relax(dur=0.4, label="relax"),
                              A.MoveObject(delta=(0.0, 0.0, S.LIFT), dur=1.5, label="lift", anchor="measured", follow=fol),
                              A.Hold(dur=2.0)])
            if ex.follower is not None:
                ep.sense["follower"] = dict(updates=ex.follower.n_upd, failed=ex.follower.n_fail,
                                            max_ik_err=(round(1e3 * ex.follower.max_err[0], 3), round(float(np.degrees(ex.follower.max_err[1])), 3)))
        else:
            ok = ex.run(both=[A.MoveObject(delta=(0.0, 0.0, S.LIFT), dur=1.5, label="lift"), A.Hold(dur=2.0)])
    else:
        raise ValueError(seg)
    info = dict(log=[r.__dict__ for r in ex.log])
    if not ok:
        bad = next((r for r in ex.log if not r.ok or (r.trigger and not str(r.trigger).startswith("guard"))), None)
        if bad is not None and not bad.ok:
            return "plan_fail", dict(info, why=bad.why)
        return "abort", dict(info, why=bad.trigger if bad else "")
    return "done", info
