"""flagship_place.py -- the flagship's `--place-backend zono_rows` (SIMULATION ONLY; default OFF, selected by that flag only).

The zed_demo backend PLANS a collision-free route (A* on a payload-vs-zonotope clearance map).  This backend does not plan: the operator's naive straight
line to the desk centre is kept as the INTENT and the perceived zonotopes act as a safety filter on it (base_filter.py: zonotope clearance CBF on the base twist):

  flagship_backend(result, obs)   rolls the CBF forward along the naive line on the perceived zonotopes and the robot's own payload geometry
                                  (rigid with the base) -> dict(ok, spot_t, path_t, ...): the route the filtered base takes and the place SPOT where it stops
                                  (so the placement stand goes under the tray where it will actually be, not 106 mm inside the bottle).
                                  None when `result` carries no `payload` (old cached perception) -> the caller keeps the zed_demo planner.
  PlaceGuard                      the LIVE filter during PLACE: arm keypoints + tracked tray + chassis footprint vs the same zonotopes (BaseFilter), applied to
                                  the mission's body-frame twist every tick.  The arm side stays the flagship's own controller (this repo cannot edit
                                  teleop/sim.py), so only the base is filtered in the flagship.
"""
from __future__ import annotations

import math
import types

import numpy as np

from . import base_filter as BF
from . import rows as R

NAIVE_T = np.array([0.0, 0.405])                     # = flagship.mission.NAIVE_T


def _cg(z):
    z = z.get("Zi", z)
    return np.asarray(z["c"], float), np.asarray(z["G"], float)


def obstacle_zonos(result):
    """(ZonoSet, names) of the perceived zonotopes (inflated Zi) of a flagship perception result."""
    zl = [_cg(z) for z in result["zonos"]]
    names = [f"Z{z['id']}:{z['gt_key']}" for z in result["zonos"]]
    return R.ZonoSet(zl, names), names


def payload_points(payload, max_per_zono=48):
    """hull vertices of the robot's own payload zonotopes [(name, c, G)] -> (K,3) world points."""
    from scipy.spatial import ConvexHull
    from world_model.v2.tray_rep.sets.zono import Zonotope
    from world_model.v2.house.zed_demo import perceive as PC
    out = []
    for _n, c, G in payload:
        V = PC.zono_vertices(Zonotope(np.asarray(c, float), np.asarray(G, float)))
        try:
            V = V[ConvexHull(V).vertices]
        except Exception:
            pass
        if len(V) > max_per_zono:
            V = V[np.linspace(0, len(V) - 1, max_per_zono).astype(int)]
        out.append(V)
    return np.vstack(out)


def rollout_naive(pts0, zs, base0, yaw0, naive_t=NAIVE_T, cfg=None, dt=0.02, t_max=30.0, stall_v=2e-3, stall_s=0.6):
    """filtered base route along the naive line.  pts0 (K,3) payload points in the world at the CURRENT base pose, zs ZonoSet of obstacles.
    Returns dict(path_t (n,2) base offsets from base0, spot_t (2,), progress_frac, stopped (bool: the filter held the base short), h_min, n_steps)."""
    from world_model.v2.house.zed_demo import execute as EX
    cfg = cfg or BF.BaseCfg(use_footprint=False)
    path = np.stack([np.linspace(0, naive_t[0], 60), np.linspace(0, naive_t[1], 60)], 1)
    traj = EX.make_traj(path, np.asarray(base0, float), yaw0)
    ref = BF.PathRef(traj, EX.DT)
    xy = np.array(base0, float)
    u_prev = np.zeros(3)
    w = 1.0 / np.array(cfg.w_scale)
    vmax, asl = np.array(cfg.v_max), np.array(cfg.a_slew)
    hist, hmin, slow_t = [xy.copy()], np.inf, 0.0
    r0 = np.zeros(len(pts0))
    va = np.zeros((len(pts0), 3))
    n = int(t_max / dt)
    for k in range(n):
        u_nom, _i = ref.nominal(xy, True)
        p = pts0 + np.array([xy[0] - base0[0], xy[1] - base0[1], 0.0])
        A, b, meta = BF.base_rows_from_points(p, r0, va, zs, xy, cfg, u_ref=u_nom)
        lo = np.maximum(-vmax, u_prev - asl * dt); hi = np.minimum(vmax, u_prev + asl * dt)
        u, _info = BF.solve_base(A, b, u_nom, lo, np.maximum(hi, lo), w, cfg, u_prev)
        u[2] = 0.0                                                                    # holonomic: the rollout keeps the yaw
        u_prev = u
        xy = xy + u[:2] * dt
        hist.append(xy.copy())
        if len(meta["h"]):
            hmin = min(hmin, float(meta["h"].min()))
        end = ref.seg[-1] - ref.seg[ref.idx] < 2e-3
        slow_t = slow_t + dt if np.linalg.norm(u[:2]) < stall_v else 0.0
        if end or slow_t > stall_s:
            break
    H = np.array(hist) - np.asarray(base0, float)
    keep = [0]
    for i in range(1, len(H) - 1):
        if np.linalg.norm(H[i] - H[keep[-1]]) >= 4e-3:
            keep.append(i)
    keep.append(len(H) - 1)
    P = H[keep]
    return dict(path_t=P, spot_t=P[-1].copy(), progress_frac=float(ref.progress()), stopped=bool(ref.seg[-1] - ref.seg[ref.idx] > 2e-2), h_min=float(hmin),
                n_steps=int(k + 1), naive_len_m=float(ref.seg[-1]))


def flagship_backend(result, obs, cfg=None, min_progress_m=0.05, log=print):
    """see the module docstring.  result = flagship perception result (kind 'desk'), obs = mission observation (pose, tray_pos, ...)."""
    payload = result.get("payload")
    if not payload:
        log("[zono_rows] perception result has no `payload` (old cache): using the zed_demo planner")
        return None
    zs, names = obstacle_zonos(result)
    pts0 = payload_points(payload)
    base0 = np.array(obs["pose"][:2], float)
    ro = rollout_naive(pts0, zs, base0, float(obs["pose"][2]), cfg=cfg)
    prog_m = float(np.linalg.norm(ro["spot_t"]))
    ok = prog_m >= min_progress_m
    log(f"[zono_rows] naive line filtered by the zonotope CBF: spot_t={np.round(ro['spot_t'], 3).tolist()} ({100 * ro['progress_frac']:.0f}% of the naive path), "
        f"stopped_short={ro['stopped']}, h_min={ro['h_min']:.3f}, ok={ok}")
    tray0 = np.asarray(obs.get("tray_pos", result.get("tray0")), float)
    return dict(ok=bool(ok), spot_t=ro["spot_t"], path_t=ro["path_t"], spot_xy=(tray0[:2] + ro["spot_t"]).tolist(), backend="zono_rows",
                progress_frac=ro["progress_frac"], stopped_short=ro["stopped"], rollout_h_min=ro["h_min"], naive_len_m=ro["naive_len_m"],
                guard_zonos=[_cg(z) for z in result["zonos"]], guard_names=names)


class PlaceGuard:
    """live base-twist filter for the flagship's PLACE phase (BaseFilter on the sim's own state).  filter(sim, twist_body) -> body twist."""

    def __init__(self, sim, zonos, names=None, cfg=None):
        m = sim.m
        ctl = sim.ctl
        zs = R.ZonoSet(zonos, names)
        ns = types.SimpleNamespace(m=m, kps={s: R.arm_keypoints(m, s) for s in ("left", "right")}, tracked=R.tracked_keypoints(m, ["tray"]),
                                   gb={"left": m.body("left_arm_gripper_base").id, "right": m.body("right_arm_gripper_base").id}, ld=ctl.ld, rd=ctl.rd, zonos=zs)
        self.bf = BF.BaseFilter(ns, cfg or BF.BaseCfg(), origin_xy=(sim.base.x, sim.base.y), yaw=float(sim.base.yaw), data=sim.d)
        self.n_calls = 0
        self.interv_max = 0.0

    def filter(self, sim, twist_body, dt):
        import mujoco
        yaw = float(sim.base.yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        u_nom = np.array([c * twist_body[0] - s * twist_body[1], s * twist_body[0] + c * twist_body[1], twist_body[2]])
        mujoco.mj_forward(sim.m, sim.d)
        u, info = self.bf.step(sim.d, u_nom, (sim.base.x, sim.base.y), yaw, dt)
        self.n_calls += 1
        self.interv_max = max(self.interv_max, float(np.linalg.norm(u - u_nom)))
        return np.array([c * u[0] + s * u[1], -s * u[0] + c * u[1], u[2]])

    def summary(self):
        out = dict(self.bf.summary())                    # base_filter.summary() also has an 'intervention_max' key: keep both, distinctly named
        out.update(n_calls=self.n_calls, guard_intervention_max=self.interv_max)
        return out
