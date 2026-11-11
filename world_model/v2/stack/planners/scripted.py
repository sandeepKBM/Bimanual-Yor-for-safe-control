"""stack/planners/scripted.py -- ScriptedPlanner: the LEGACY contents-carry plan (carry.PLANS fixed TCP deltas) as a types.BimanualPlan.
SIMULATION ONLY.  Extracted 2026-11-03 from nero/contents/carry.py CarryEpisode (step 1 of the stack plan); no behaviour change.

Segment.meta carries exactly what CarryEpisode.segment_phases / backup_phases returned:
  phases         list of legacy phases ("move", delta, T) | ("hold", T) | ("weld", on) | ("retreat", r, T)   (stop / to are backups only)
  backup_phases  legacy backup list (or None for release)
  accel          carry.commanded_accel record (None for release)
The per-arm IK ("_add_path", continuation IK from the last commanded q) is NOT run here: it runs at execution time (executor.py), on the
episode's own command history, as before.  provenance["legacy"] = True.
ASSUMED: PLANS deltas (user-confirmed 11-02), see carry.PLANS.
"""
from __future__ import annotations

import numpy as np

from world_model.v2.nero.contents import carry as CR
from world_model.v2.stack.types import SIDES, BimanualPlan, Segment, SkillCall, TaskSpec

NAME = "scripted"


def segment_phases(task, seg, a_plan):
    """-> (phases, accel).  Same numbers as the former CarryEpisode.segment_phases(seg)."""
    plan = dict(CR.plan_moves(task))
    if seg == "release":
        return [("hold", 0.2), ("weld", False), ("retreat", CR.RETREAT, 0.8), ("hold", 0.5)], None
    dl = np.asarray(plan[seg], float)
    accel = CR.commanded_accel(float(np.linalg.norm(dl)), a_plan)           # 11-03 item 10: recorded next to a_plan
    return [("move", tuple(dl), CR.move_time(float(np.linalg.norm(dl)), a_plan)), ("hold", 0.3)], accel


def backup_phases(seg):
    """-> legacy backup phase list or None.  Same as the former CarryEpisode.backup_phases(seg)."""
    back = {"lift": ["lift"], "carry": ["carry", "lift"], "place": ["place", "carry", "lift"]}.get(seg)
    if back is None:
        return None
    # let go in place (11-02): back on the counter, release the weld and retreat, so the object settles flat on its own
    # (held welded, wrist sag of up to ~4 deg under load left it resting on one edge: smoke box 2 kg, 14/16 backups not in R)
    return ([("stop", 0.3)] + [("to", f"start_{s}", None) for s in back]
            + [("hold", 0.3), ("weld", False), ("retreat", CR.RETREAT, 0.8), ("hold", 0.6)])


def _duration(phases):
    t = 0.0
    for ph in phases:
        if ph[0] in ("move", "retreat"):
            t += float(ph[2])
        elif ph[0] == "hold":
            t += float(ph[1])
    return t


class ScriptedPlanner:
    name = NAME

    def plan(self, ws, task, sim=None, theta=None, a_plan=None):
        """task: SkillCall(args={"task": "tray"|"box", "a_plan": float}) or TaskSpec(name="tray"|"box"); falls back to sim.episode."""
        ep = getattr(sim, "episode", None)
        args = task.args if isinstance(task, SkillCall) else {}
        name = args.get("task") or (task.name if isinstance(task, TaskSpec) else None) or getattr(ep, "task", None)
        if name not in CR.PLANS and name not in CR.PLANS_EXTRA:
            raise ValueError(f"ScriptedPlanner: no legacy plan for task {name!r} (have {sorted(CR.PLANS)})")
        a = args.get("a_plan", a_plan)
        a = float(a if a is not None else getattr(ep, "a_plan", CR.E.A_NOM))
        segs, t = [], 0.0
        for seg in CR.SEGS:
            ph, acc = segment_phases(name, seg, a)
            dur = _duration(ph)
            segs.append(Segment(mode=seg, t0=t, t1=t + dur, weld={s: seg != "release" for s in SIDES},
                                meta=dict(phases=ph, backup_phases=backup_phases(seg), accel=acc, legacy=True)))
            t += dur
        return BimanualPlan(segments=segs, planner=NAME,
                            provenance=dict(legacy=True, task=name, a_plan=a, source="carry.PLANS"),
                            assumptions=dict(PLANS=dict(tag="ASSUMED", src="carry.PLANS, user-confirmed 11-02"),
                                             A_BACKUP=dict(tag="ASSUMED", value=CR.A_BACKUP)))
