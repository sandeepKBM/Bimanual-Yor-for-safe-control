"""adapters/state_carry.py -- CarryEpisode (nero/contents/carry.py) -> stack.state.EndState.  U quantities come from the episode's running track
(maxima since reset_track); the track keeps only the FIRST violation for collision / cup_off_tray / drop, so those give a flag / a just-past-the-bound value."""
from __future__ import annotations

import math

import mujoco
import numpy as np

from world_model.v2.stack import tasks as TK
from world_model.v2.stack.state import EndState


DT = 0.005          # = nero/contents/carry.DT (adapters/tests/test_state_readers.py checks it); not imported: carry reaches flmpc (dependency rules)


def _tilt_deg(R):   # = carry._tilt_deg (checked by the same test)
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))


_kind_task = TK.of_kind          # a TaskDef of this CarryEpisode.task kind ("tray" | "box" | "pot"); the drop bound is the same for every task of a kind


def _tops(world, goal_support=""):
    out = {k: float(getattr(world, f"{k}_top")) for k in ("counter", "shelf") if hasattr(world, f"{k}_top")}
    if goal_support == "counter2":                 # table-behind (11-09): counter2 = same height as the counter (MODEL, nero/base/scene.support_tops_behind)
        from world_model.v2.nero.base import scene as BS
        out["counter2"] = float(BS.support_tops_behind(world)["counter2"]["top"])
    return out


def from_carry(ep, finished=None, pre_hold_viol=None) -> EndState:
    m, d = ep.m, ep.d
    R = d.xmat[ep.obj].reshape(3, 3)
    p = d.xpos[ep.obj]
    v = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, ep.obj, v, 0)
    gs = getattr(ep, "goal_support", "")           # the episode's goal support (table-behind tasks: "counter2"; the TaskDef.goal_support of the row it runs); "" = legacy rule
    sup = {m.geom(gs).id} if gs else ({m.geom("counter").id} | ({m.geom("shelf").id} if getattr(ep.world, "shelf", False) and _kind_task(ep.task).goal_support == "shelf" else set()))
    touching = any((int(c.geom1) in sup and m.geom_bodyid[c.geom2] == ep.obj) or (int(c.geom2) in sup and m.geom_bodyid[c.geom1] == ep.obj)
                   for c in d.contact[:d.ncon])
    tr = ep.track
    now = dict(obj_x=float(p[0]), obj_y=float(p[1]), obj_z=float(p[2]), obj_z0=float(ep.z0), rise_m=float(p[2] - ep.z0), drop_m=float(ep.z0 - p[2]),
               obj_tilt_deg=_tilt_deg(R), obj_speed=float(np.linalg.norm(v[3:])), support_contact=1.0 if touching else 0.0,
               cup_tilt_deg=max([_tilt_deg(d.xmat[b].reshape(3, 3)) for b in ep.cups] or [0.0]), tau_ratio=float(tr["tau_ratio"]))
    if getattr(ep, "cap", None) is not None:          # coffee tray (nero/contents/coffee.py): the capsule's position in the tray frame
        crel = R.T @ (d.xpos[ep.cap] - p)
        now.update(cap_rel_x=float(crel[0]), cap_rel_y=float(crel[1]), cap_rel_z=float(crel[2]))
    if ep.lid is not None:
        rel, ltilt = ep._lid_rel()
        r0 = ep.lid_rest[0]
        now.update(lid_slide_m=float(np.linalg.norm(rel[:2] - r0[:2])), lid_dz_m=float(rel[2] - r0[2]), lid_tilt_deg=float(ltilt))
    worst = dict(now)
    worst.update(obj_tilt_deg=max(now["obj_tilt_deg"], float(tr["obj_tilt_max"])), cup_tilt_deg=max(now["cup_tilt_deg"], float(tr["cup_tilt_max"])),
                 spill_ratio=float(tr["spill_ratio"]), collision_flag=0.0, cup_off_tray_flag=0.0)
    if ep.lid is not None:
        worst.update(lid_slide_m=max(now["lid_slide_m"], tr["lid_slide_mm"] / 1e3), lid_dz_m=max(now["lid_dz_m"], tr["lid_dz_mm"] / 1e3),
                     lid_tilt_deg=max(now["lid_tilt_deg"], float(tr["lid_tilt_deg"])))
    fv = dict(tr["viol"]) if tr["viol"] else None
    if fv:
        k = fv["kind"]
        if k == "collision":
            worst["collision_flag"] = 1.0
        elif k == "cup_off_tray":
            worst["cup_off_tray_flag"] = 1.0
        elif k == "obj_drop":
            worst["drop_m"] = max(worst["drop_m"], math.nextafter(_kind_task(ep.task).U_dict()["drop_m"].hi, math.inf))   # just past the TaskDef drop bound
    return EndState(t=float(ep.k * DT), now=now, worst=worst, first_viol=fv, finished=finished, pre_hold_viol=pre_hold_viol,
                    meta=dict(backend="carry", task=ep.task, goal_support_tops=_tops(ep.world, getattr(ep, "goal_support", ""))))
