"""adapters/state_pan.py -- pan segments.Episode (nero/pan/segments.py) -> stack.state.EndState.  U 'worst' = the instantaneous values except
tau_ratio (running max) and the backend's own first violation."""
from __future__ import annotations

import math

import numpy as np

from world_model.v2.nero.pan import observe_pose as O
from world_model.v2.nero.pan import segments as S
from world_model.v2.stack import tasks as TK
from world_model.v2.stack.state import EndState

_T = TK.task("pan_lift")          # the pan tasks share U / R / reader thresholds


def _reach(ep, version) -> float:
    """1.0 iff find_pregrasp finds a contact-free pre-grasp at the handle: the SAME call as segments.Episode.in_R (n_sol=4, seeds=24; v1 fingers FINGER_OPEN,
    v2 REGRASP_OPEN)."""
    ht = S.handle_point(ep.m, ep.d, ep.p, ep.plan["grasp_frac"])
    Rz = ep.d.xmat[ep.pan].reshape(3, 3) @ ep.R0.T
    try:
        S.find_pregrasp(ep.m, ep.d, "left", ht, Rz @ S.LEFT_APPROACH, Rz @ S.LEFT_JAW, n_sol=4, seeds=24, open_w=S.FINGER_OPEN if version == "v1" else S.REGRASP_OPEN)
        return 1.0
    except RuntimeError:
        return 0.0


def from_pan(ep, finished=None, reach=True, version="v1") -> EndState:
    """reach=True / version="v1" are Episode.in_R's defaults.  reach=False leaves `reach` out of the state: R then reports it as not measured (missing)."""
    ps, cs = ep.pan_state(), ep.contacts()
    t = cs["touch"]
    jm = min(float(np.min(np.minimum(ep.d.qpos[ep.addr[s]] - ep.jrange[s][:, 0], ep.jrange[s][:, 1] - ep.d.qpos[ep.addr[s]]))) for s in S.SIDES)
    arm_v = max(float(np.max(np.abs(ep.d.qvel[ep.dofs[s]]))) for s in S.SIDES)
    p = ep.d.xpos[ep.pan]
    lifted = ps["dz_mm"] > _T.reader_dict()["lifted_dz_mm"].lo       # strict (`>`), the bound is the limit
    now = dict(obj_x=float(p[0]), obj_y=float(p[1]), obj_z=float(p[2]), dz_mm=ps["dz_mm"], rise_mm=ps["dz_mm"], drop_mm=-ps["dz_mm"],
               obj_tilt_deg=ps["tilt"], lid_mm=ps["lid_mm"], spill_flag=1.0 if ps["spill"] else 0.0,
               com_in_m=math.inf if lifted else ps["com_in"], pan_speed=ps["v"], arm_speed=arm_v, joint_margin_rad=jm,
               collision_flag=1.0 if cs["coll"] else 0.0, support_force_N=float(cs["support"]),
               support_frac=float((np.mean(ep._sup) if ep._sup else cs["support"]) / ep.W),
               yaw_deg=float(O.true_pose(ep.m, ep.d, ep.p, ep.pos0, ep.R0)["yaw_deg"]),
               released=1.0 if all(v < 1e-6 for v in cs["grip_obj"].values()) else 0.0,
               touch_left_1_n=float("pan" in t[("left", 1)]), touch_left_2_n=float("pan" in t[("left", 2)]),
               touch_left_pan_n=float("pan" in (t[("left", 1)] | t[("left", 2)])), touch_right_1_n=float(len(t[("right", 1)])),
               touch_right_2_n=float(len(t[("right", 2)])), touch_right_pan_n=float("pan" in (t[("right", 1)] | t[("right", 2)])),
               tau_ratio=float(ep.track["tau_ratio"]), rim_in_m=ps["rim_in"])
    if version == "v2":
        now["stable_margin_m"] = float(ep.stable_margin())
    if reach:
        now["reach"] = _reach(ep, version)
    if "pan_at_push_start" in ep.sense and ep.plan.get("push"):
        moved = float(-(ep.d.xpos[ep.pan] - ep.sense["pan_at_push_start"]) @ ep.u_true)
        now["push_frac"] = moved / float(ep.plan["push"])
    fv = dict(ep.track["viol"]) if ep.track["viol"] else None
    return EndState(t=float(ep.k * S.DT), now=now, worst=dict(now), first_viol=fv, finished=finished, meta=dict(backend="pan"))
