"""ocs/dynamics.py -- per-object motion models + `rollout(ws, horizon, dt, robot_plan)` with GROWING SET uncertainty.  SIMULATION ONLY.

Registry (ObjectState.dynamics key):
  static             pose constant; set unchanged (the perception set only).
  attached           rigid to the parent: pose in the parent frame constant; the child's world set = parent's set (+) attachment set (consumers.world_unc).
  slip_on_support    cup on the tray: Coulomb sliding block in the tray frame (reach.ball_step with k_in = 0, r = 1, mu_eff = mu): it slides
                     when the in-plane effective gravity |g_in(a_tray)| exceeds mu*N/m.  Centre = rollout at mu_mid under the planned tray
                     acceleration; SET = hull of the rollouts at mu_lo / mu_hi under the plan +- a_dist (8 directions), cumulative maximum (monotone).
  rolling_on_support the same with the rolling ball (k_in = 0.4, r = R_BALL, mu_eff = params["mu_roll"]).
  const_vel          person / movable: p(t) = p0 + v t; set = unc0 (+) disc(h(tau)),  h(tau) = sigma_v tau + 0.5 a_max tau^2  (sigma_v from the
                     velocity covariance / params, a_max from params["a_max"].hi or the kind default).  tau counts from the MEASUREMENT time, so an
                     occluded object (visible=False, old t_meas) keeps growing.

The robot base follows the plan (default: constant body twist from RobotState.base_vel); its set grows as unc0 (+) cube(e_v tau + 0.5 e_a tau^2),
rotation bound + e_w tau (e_a smaller when a plan is given).  Every set is a pure Minkowski sum of the t=0 set with a radius that is non-decreasing
in tau, so the support function of each set is non-decreasing along the rollout (tests/test_ocs.py::test_rollout_monotone).

IR atoms accept a leading time axis: `consumers.to_ir_entities(rollout_list)` -> `ir.margin / status` evaluate G / F windows on the rollout directly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from world_model.v2.tray_rep.sets import reach as RE
from world_model.v2.tray_rep.sets.ir import Pose, Unc
from world_model.v2.tray_rep.sets.zono import Zonotope

from .state import Kind, ObjectState, RobotState, WorldState, Belief
from . import util as U

G_ACC = 9.81
R_BALL = RE.R_BALL

REGISTRY: dict[str, Callable] = {}


def register(key):
    def deco(fn):
        REGISTRY[key] = fn
        return fn
    return deco


@dataclass
class DynCfg:
    base_v_err: float = 0.05            # m/s   velocity error of the base twist
    base_a_unplanned: float = 1.0       # m/s^2 unmodelled base acceleration when there is no plan (operator may change the command)
    base_a_planned: float = 0.3         # m/s^2 tracking error against a given plan
    base_w_err: float = 0.05            # rad/s yaw-rate error
    person_sigma_v: float = 0.3         # m/s   fallback velocity std of a person
    person_a_max: float = 1.5           # m/s^2 a person can change velocity this fast
    movable_sigma_v: float = 0.1
    movable_a_max: float = 0.5
    a_dist_unplanned: float = 0.6       # m/s^2 tray acceleration disturbance set (slip set) with no plan
    a_dist_planned: float = 0.2
    mu_default: tuple = (0.18, 0.35)    # used when the cup has no params["mu"]


@dataclass
class RobotPlan:
    """body twists (vx, vy, wz) applied on rollout step k (k = 0..K-1); shorter plans hold their last twist."""
    twists: np.ndarray

    def twist(self, k, fallback):
        t = np.asarray(self.twists, float)
        if t.size == 0:
            return np.asarray(fallback, float)
        return t[min(k, len(t) - 1)]


def hold_plan(twist, K=1):
    return RobotPlan(np.tile(np.asarray(twist, float), (K, 1)))


def brake_plan(v0, K, dt, a_hw=None):
    """the shield's brake contingency as a RobotPlan: from the current body twist v0 every axis slews to zero at the hardware limit a_hw
    ([4, 4, 8] m/s^2 | rad/s^2 = teleop.base.A_HW), i.e. the shield's emergency fallback  u = v - sign(v) min(|v|, a_hw dt)  (shield.py filter_base)
    applied every tick, which is also what teleop release / u_op = 0 gives at best (sim.base.advance slews at a_hw).  twists[k] = the twist at the END
    of rollout step k, so the step acceleration (v_k - v_{k-1}) / dt that drives the cup is the true brake decel (the position integral is slightly short)."""
    if a_hw is None:
        from world_model.v2.house.teleop import base as BS
        a_hw = BS.A_HW
    v0 = np.asarray(v0, float)
    a_hw = np.asarray(a_hw, float)
    tw = [v0 - np.sign(v0) * np.minimum(np.abs(v0), a_hw * dt * (k + 1)) for k in range(max(int(K), 1))]
    return RobotPlan(np.array(tw))


# ====================================================================================== helpers
def _clone_ws(ws: WorldState, t: float) -> WorldState:
    return WorldState(t=t, robot=replace(ws.robot), objects={k: replace(o) for k, o in ws.objects.items()}, relations=list(ws.relations),
                      free=ws.free, meta=dict(ws.meta))


def _topo(ws: WorldState):
    order, seen = [], set()

    def visit(oid):
        if oid in seen:
            return
        seen.add(oid)
        p = ws.objects[oid].parent
        if p is not None and p != "robot" and p in ws.objects:
            visit(p)
        order.append(oid)
    for oid in ws.objects:
        visit(oid)
    return order


def _belief(o: ObjectState, key, default=None):
    b = o.params.get(key)
    return default if b is None else b


def _yaw(rs: RobotState) -> float:
    return rs.yaw if rs.yaw is not None else U.yaw_of(rs.base.R)


def _grow(tau: float, e_v: float, e_a: float) -> float:
    return e_v * tau + 0.5 * e_a * tau * tau


def _add_set(u: Unc, pos_extra: Zonotope | None, rot_extra: float = 0.0) -> Unc:
    pos = u.pos
    if pos_extra is not None:
        pos = pos_extra if pos is None else pos.minkowski(pos_extra)
    return Unc(pos, float(u.rot) + float(rot_extra))


# ====================================================================================== robot
def _robot_traj(ws: WorldState, K: int, dt: float, plan: RobotPlan | None, cfg: DynCfg):
    """-> (list[RobotState] K+1, list of body twists K, list of body accelerations K)."""
    rs0 = ws.robot
    v0 = np.zeros(3) if rs0.base_vel is None else np.asarray(rs0.base_vel, float)
    e_a = cfg.base_a_planned if plan is not None else cfg.base_a_unplanned
    out = [rs0]
    tw, ac = [], []
    x, y, yaw = float(rs0.base.p[0]), float(rs0.base.p[1]), _yaw(rs0)
    v_prev = v0
    for k in range(K):
        v = plan.twist(k, v0) if plan is not None else v0
        tw.append(v)
        ac.append((v - v_prev) / dt)
        ym = yaw + 0.5 * v[2] * dt
        c, s = math.cos(ym), math.sin(ym)
        x += (c * v[0] - s * v[1]) * dt
        y += (s * v[0] + c * v[1]) * dt
        yaw += v[2] * dt
        tau = (k + 1) * dt
        h = _grow(tau, cfg.base_v_err, e_a)
        unc = _add_set(rs0.base_unc, U.ball_zono(h) if h > 0 else None, cfg.base_w_err * tau)
        rs = replace(rs0, base=U.base_pose(x, y, yaw), base_unc=unc, base_vel=v.copy(), yaw=yaw)
        out.append(rs)
        v_prev = v
    return out, tw, ac


# ====================================================================================== registry entries
@register("static")
def _static(oid, wss, ctx):
    o0 = wss[0].objects[oid]
    return [replace(o0) for _ in wss]


@register("attached")
def _attached(oid, wss, ctx):
    o0 = wss[0].objects[oid]
    return [replace(o0) for _ in wss]


@register("const_vel")
def _const_vel(oid, wss, ctx):
    o0 = wss[0].objects[oid]
    cfg: DynCfg = ctx["cfg"]
    v = np.zeros(3) if o0.vel is None else np.asarray(o0.vel, float)[:3]
    person = o0.kind == Kind.PERSON
    sig = o0.extra.get("sigma_v", cfg.person_sigma_v if person else cfg.movable_sigma_v)
    a_b = _belief(o0, "a_max")
    a_max = a_b.hi if a_b is not None else (cfg.person_a_max if person else cfg.movable_a_max)
    tau_age = max(wss[0].t - o0.t_meas, 0.0)          # time already elapsed since the measurement (the set at k=0 already includes it)
    h0 = _grow(tau_age, sig, a_max)
    out = []
    pw0 = np.asarray(wss[0].world_pose(oid).p, float)                 # world position at k = 0 (the motion model lives in the world frame)
    Rw0 = np.asarray(wss[0].world_pose(oid).R, float)
    for k, ws in enumerate(wss):
        tau = k * ctx["dt"]
        h = _grow(tau_age + tau, sig, a_max) - h0
        pos = o0.unc.pos
        if h > 0:
            ex = U.disc_zono(h) if person else U.ball_zono(h)
            pos = ex if pos is None else pos.minkowski(ex)
        pw = pw0 + v * tau
        if o0.parent == "robot":                                       # the person stays on its world track while the robot moves: re-express in the new base frame
            Pb = ws.robot.base
            Rb, pb = np.asarray(Pb.R), np.asarray(Pb.p)
            pose = Pose(Rb.T @ Rw0, Rb.T @ (pw - pb))
        elif o0.parent is None:
            pose = Pose(np.asarray(o0.pose.R), pw)
        else:
            raise NotImplementedError("const_vel: parent must be the world or 'robot'")
        out.append(replace(o0, pose=pose, unc=Unc(pos, float(o0.unc.rot)), t_meas=o0.t_meas))
    return out


def _tray_axes_inplane(R):
    """tray frame axes (world) from a rotation matrix + the reach frame (minimal rotation z -> c) and the 2x2 rotation Q between them."""
    R = np.asarray(R, float)
    ex_t, ey_t, c = R[:, 0], R[:, 1], np.ascontiguousarray(R[:, 2])
    ex_r, ey_r = RE._frame(c)
    Q = np.array([[ex_t @ ex_r, ex_t @ ey_r], [ey_t @ ex_r, ey_t @ ey_r]])
    return c, Q


def _slide_traj(p0, v0, tray_R, a_world, mu, k_in, r, K, dt=None):
    """slide the support-contact body in the TRAY frame for K steps of `dt` (a multiple of reach.DT = 20 ms; None = one reach tick).  p0 (2,), v0 (2,)
    tray-frame; tray_R: list of K+1 world rotations of the tray; a_world: list of K tray accelerations (3,) (world).
    Returns (K+1, 2) positions, (K+1, 2) velocities."""
    nsub = 1 if dt is None else int(round(dt / RE.DT))
    assert nsub >= 1 and (dt is None or abs(nsub * RE.DT - dt) < 1e-9), f"support dynamics need dt a multiple of {RE.DT} s (got {dt})"
    P = np.zeros((K + 1, 2))
    V = np.zeros((K + 1, 2))
    P[0], V[0] = p0, v0
    for k in range(K):
        ca, Qa = _tray_axes_inplane(tray_R[k])
        cb, Qb = _tray_axes_inplane(tray_R[k + 1])
        Q = 0.5 * (Qa + Qb)
        s_r = np.concatenate([Q.T @ P[k], Q.T @ V[k]])                     # tray frame -> reach frame
        a_k = np.ascontiguousarray(a_world[k], dtype=float)
        for j in range(nsub):                                              # reach.ball_step is one 20 ms tick; the axis is interpolated over the step
            c0 = ca + (cb - ca) * (j / nsub)
            c1 = ca + (cb - ca) * ((j + 1) / nsub)
            s_r, _, _ = RE.ball_step(s_r, np.ascontiguousarray(c0 / np.linalg.norm(c0)), np.ascontiguousarray(c1 / np.linalg.norm(c1)), a_k, float(mu), float(k_in), float(r))
        P[k + 1] = Q @ s_r[:2]
        V[k + 1] = Q @ s_r[2:]
    return P, V


def _support_traj(oid, wss, ctx, k_in, r, mu_key):
    o0 = wss[0].objects[oid]
    cfg: DynCfg = ctx["cfg"]
    K, dt = len(wss) - 1, ctx["dt"]
    parent = o0.parent
    assert parent is not None and parent != "robot", f"{oid}: support dynamics need a parent object"
    mu_b = _belief(o0, mu_key)
    lo, hi = (mu_b.lo, mu_b.hi) if mu_b is not None else cfg.mu_default
    mid = mu_b.mean if (mu_b is not None and mu_b.mean is not None) else 0.5 * (lo + hi)
    trayR = [np.asarray(ws.world_pose(parent).R) for ws in wss]
    a_pl = ctx["tray_acc"](parent)                                          # K x (3,) world tray acceleration from the plan
    plan_given = ctx["plan_given"]
    a_d = cfg.a_dist_planned if plan_given else cfg.a_dist_unplanned
    p0 = np.asarray(o0.pose.p, float)
    v0 = np.asarray(o0.extra.get("v_rel", np.zeros(2)), float)
    Pc, Vc = _slide_traj(p0[:2], v0, trayR, a_pl, mid, k_in, r, K, dt)
    half = np.zeros((K + 1, 2))
    dirs = [np.zeros(3)] + [np.array([math.cos(a), math.sin(a), 0.0]) * a_d for a in np.arange(8) * math.pi / 4]
    for mu in (lo, hi):
        for dv in dirs:
            Ps, _ = _slide_traj(p0[:2], v0, trayR, [a + dv for a in a_pl], mu, k_in, r, K, dt)
            half = np.maximum(half, np.abs(Ps - Pc))
    half = np.maximum.accumulate(half, axis=0)                              # monotone
    out = []
    for k in range(K + 1):
        pos = o0.unc.pos
        if k > 0:
            ex = Zonotope(np.zeros(3), np.diag([half[k, 0], half[k, 1], 0.0]))
            pos = ex if pos is None else pos.minkowski(ex)
        p = np.array([Pc[k, 0], Pc[k, 1], p0[2]])
        out.append(replace(o0, pose=Pose(np.asarray(o0.pose.R), p), unc=Unc(pos, float(o0.unc.rot)),
                           extra={**o0.extra, "v_rel": Vc[k].copy()}))
    return out


@register("slip_on_support")
def _slip(oid, wss, ctx):
    return _support_traj(oid, wss, ctx, 0.0, 1.0, "mu")


@register("rolling_on_support")
def _roll(oid, wss, ctx):
    return _support_traj(oid, wss, ctx, RE.K_IN, R_BALL, "mu_roll")


# ====================================================================================== rollout
def rollout(ws: WorldState, horizon: float, dt: float = 0.1, robot_plan: RobotPlan | None = None, cfg: DynCfg | None = None) -> list[WorldState]:
    """predict the OCS `horizon` seconds ahead in steps of `dt`.  Returns [ws_0 (a copy of `ws`), ..., ws_K] with K = round(horizon / dt)."""
    cfg = cfg or DynCfg()
    K = max(int(round(horizon / dt)), 0)
    rts, tw, ac = _robot_traj(ws, K, dt, robot_plan, cfg)
    wss = []
    for k in range(K + 1):
        w = _clone_ws(ws, ws.t + k * dt)
        w.robot = rts[k]
        wss.append(w)

    def tray_acc(tray_oid):
        """world acceleration of the tray (3,) on step k: the acceleration of the base-fixed point at the tray position (cup_accel_body)."""
        from world_model.v2.house.teleop import base as BS
        o = wss[0].objects[tray_oid]
        if o.parent == "robot":
            rxy = np.asarray(o.pose.p, float)[:2]
        else:
            P = wss[0].world_pose(tray_oid)
            Rb = np.asarray(wss[0].robot.base.R)
            rxy = (Rb.T @ (np.asarray(P.p) - np.asarray(wss[0].robot.base.p)))[:2]
        v_prev = np.zeros(3) if ws.robot.base_vel is None else np.asarray(ws.robot.base_vel, float)
        out = []
        for k in range(K):
            v_mid = 0.5 * (v_prev + tw[k])
            a_b = BS.cup_accel_body(v_mid, ac[k], rxy)
            Rw = np.asarray(rts[k].base.R)[:2, :2]
            a_w = Rw @ a_b
            out.append(np.array([a_w[0], a_w[1], 0.0]))
            v_prev = tw[k]
        return out

    ctx = dict(cfg=cfg, dt=dt, tray_acc=tray_acc, plan_given=robot_plan is not None)
    for oid in _topo(ws):
        key = ws.objects[oid].dynamics
        if key not in REGISTRY:
            raise KeyError(f"unknown dynamics {key!r} for {oid}")
        traj = REGISTRY[key](oid, wss, ctx)
        for k, o in enumerate(traj):
            wss[k].objects[oid] = o
    return wss


def set_radius(ws: WorldState, oid: str) -> float:
    """horizontal support radius (max over 16 directions) of the object's world-frame position set -- a scalar to track growth."""
    from .consumers import world_unc
    u = world_unc(ws, oid)
    if u.pos is None:
        return 0.0
    ang = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    return float(max(U.unc_support(u, (math.cos(a), math.sin(a), 0.0)) for a in ang))
