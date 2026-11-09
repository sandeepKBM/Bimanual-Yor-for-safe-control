"""adapters/state_embodichain.py -- AgentChord-sim `check_detail()` dict (agentchord_probe/coffee_tray/env.py) -> stack.state.EndState.
Positions (obj_x/y, capsule) are the record's 4-decimal rounded values (the old check used the unrounded; <= 5e-5 m); the goal xy_err is the unrounded tray_xy_err.  Positions are relative to the tray's start (goal_frame "start").  The record carries only the boolean `tray_level` (tilt < 10 deg), not the angle:
without `tray_tilt_deg` the tilt is 0.0 when level and +inf when not (meta['tilt_exact'] = False; the tilt margin is then not meaningful).
No U monitors exist in that sim.  Pure dict reader: no sim import."""
from __future__ import annotations

import math

from world_model.v2.stack.state import EndState


def from_embodichain(detail: dict, tray_tilt_deg: float | None = None, finished=None) -> EndState:
    dx, dy = detail["tray_dxy"]
    cx, cy, cz = detail["capsule_rel_tray"]
    exact = tray_tilt_deg is not None
    tilt = float(tray_tilt_deg) if exact else (0.0 if detail["tray_level"] else math.inf)
    now = dict(obj_x=float(dx), obj_y=float(dy), cap_rel_x=float(cx), cap_rel_y=float(cy), cap_rel_z=float(cz), obj_tilt_deg=tilt)
    if detail.get("tray_xy_err") is not None:         # the sim's UNROUNDED |tray_xy - target| (env.py:31-37); tray_dxy / capsule_rel_tray are rounded to 1e-4 m
        now["xy_err"] = float(detail["tray_xy_err"])
    return EndState(t=0.0, now=now, worst=dict(now), finished=finished,
                    meta=dict(backend="embodichain", tilt_exact=exact, tray_xy_err=detail.get("tray_xy_err")))
