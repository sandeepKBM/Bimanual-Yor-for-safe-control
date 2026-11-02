"""pan/cert/backups.py -- stored "let go in place" backups (plan 2026-10-29).  SIMULATION ONLY.

A Backup is fixed when the segment is admitted: a phase tuple with no online choices (directions may come from the arm's own
FK -- state feedback -- but every distance and duration is fixed), plus the parameter it was certified with.  The family is
indexed by the retreat distance d; admission searches the family for ONE d that works for every certification sample
(exists b, for all theta).

Undo is gone: no backup drags the pan back (that popped the lid, 10-29).  The pan stays where it is; per the user decision R_V2
it is task-ready if at rest, stable and reachable.

    pinch: stop, release L, retreat L by d
    push:  stop, release L, retreat L by d              (pan left part-pushed)
release = open each finger RELEASE_OPEN past its encoder reading, not fully: the claw's jaw axis is vertical, so a full open
(0.03 m per finger) drove the lower finger into the stove pad once the push had moved the rim over it (smokeB 10-29).
    clamp: stop, open R, R back to clamp_start, then the push backup
    lift:  stop, lower both to lift_start (slow), payload model off, then the clamp backup
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import segments as S

FAMILY = (0.06, 0.04, 0.03, 0.02, 0.01)   # design: candidate retreat distances [m], longest first
STOP_T = 0.3      # design: smooth stop before any backup [s]
OPEN_T = 0.5      # design: claw opening [s]
RELEASE_OPEN = 0.006  # design: left release opening past the closed reading [m per finger]; lower finger drops this much
SETTLE_T = 0.5    # design: hold at the end, so R is judged at rest [s]
LOWER_T = 3.0     # design: lowering duration for the full LIFT (5 cm): peak commanded acceleration 6*0.05/9 = 0.033 m/s^2
RIGHT_BACK_T = 2.0  # design: right claw back to clamp_start over APPROACH_IN (8 cm): peak 6*0.08/4 = 0.12 m/s^2


@dataclass(frozen=True)
class Backup:
    seg: str
    retreat: float
    phases: tuple

    def describe(self):
        return dict(seg=self.seg, retreat_m=self.retreat, phases=[p[0] for p in self.phases])


def phases(ep, seg, retreat, first=True):
    """the stored phase list.  move_to targets read the segment-start marks the robot recorded (its own FK), fixed per run."""
    ph = [("stop", STOP_T)] if first else []
    if seg == "lift":
        ph += [("move_to", {s: (lambda s=s: ep.mem["lift_start"][s]) for s in S.SIDES}, LOWER_T, S.LIFT), ("hold", 0.5),
               ("payload", False)]
        return ph + phases(ep, "clamp", retreat, first=False)
    if seg == "clamp":
        ph += [("open", "right", OPEN_T), ("move_to", {"right": lambda: ep.mem["clamp_start"]["right"]}, RIGHT_BACK_T, S.APPROACH_IN),
               ("hold", 0.2)]
        return ph + phases(ep, "push", retreat, first=False)
    if seg in ("push", "pinch"):
        return ph + [("open", "left", OPEN_T, RELEASE_OPEN), ("retreat_fixed", "left", float(retreat)), ("hold", SETTLE_T)]
    raise ValueError(seg)


def make(ep, seg, retreat):
    return Backup(seg=seg, retreat=float(retreat), phases=tuple(phases(ep, seg, retreat)))


def has_online_choice(bk):
    """True if a phase picks something at trigger time (the v1 'retreat' kind chose its distance from a clearance search)."""
    return any(p[0] == "retreat" for p in bk.phases)
