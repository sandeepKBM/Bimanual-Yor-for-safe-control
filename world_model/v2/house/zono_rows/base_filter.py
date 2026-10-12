"""base_filter.py -- zonotope clearance CBF on the kinematic BASE twist (vx, vy, wz).  SIMULATION ONLY (pure numpy + the Hildreth projection QP).

The arm-only zonotope rows (rows.py) cannot stop the base: J qd only sees the arm joints and the base is a kinematic mocap path.  Here the base
twist u = (vx, vy, wz) (WORLD-frame translation of the base origin o, yaw rate about o) is a decision variable of a small projection QP:

    min_u  || (u - u_nom) / V ||^2        s.t.   hard box   |u| <= u_max,  |u - u_prev| <= a_slew * dt
                                                 one CBF row per (point, zonotope active facet) pair

Points: every arm-link keypoint of both arms, the carried tray / ball points (tracked or fixed-offset, whatever ZonoCtl uses), and the robot
FOOTPRINT (oriented-bounding-box corners + face centres of every collidable non-arm YOR geom, rigid with the base).  For a point p with sphere radius r
and the active facet (n, d) of a perceived zonotope  (same sd_lb logic as rows.py):

    h     = n.p - d - r - margin
    hdot  = n.(v_arm + v_base(u)),    v_base = v + wz e_z x (p - o)  ->  n.v_base = g.u,   g = [nx, ny, -nx (py-oy) + ny (px-ox)]
    CBF   hdot >= -beta(h),  beta = min(gamma h, sqrt(2 a_brk h)) for h >= 0 (first-order class-K AND a braking-distance bound: with a slew-limited
          base a pure hdot + gamma h >= 0 cannot stop in time at gamma ~ 10), beta = max(gamma h, -retreat_cap) for h < 0 (retreat)
  <=>  -g.u <= n.v_arm + beta.         v_arm = J qd of the arm joints (rigid-point model, as rows.py); the facet normal is constant (static obstacle).

SELECTIVE handling of hopeless rows (the base is a 3-variable problem, so this is a cheap screen + peel instead of an elastic QP):
  * a row that cannot be met even at the best corner of the hard box (b < min_box A u) is lifted to 10% of the way from that best corner to the
    worst one (it then asks for ~maximum effort instead of making the whole QP infeasible);
  * if the QP is still infeasible the most violated row is dropped and the QP re-solved (<= max_peel times, every drop is counted);
  * last resort: brake at the slew limit (counted).
The hard box is never relaxed.  Reported residuals are against the ORIGINAL rows (before any lift/peel), so they stay honest.

Limits (not privileged): the filter reads the same simulated tracker the rows already use (true tray / ball body pose in MjData) and the true zonotope
H-reps; base acceleration is bounded by the slew limit but not fed to the arm rows.
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass

import numpy as np
import mujoco

from . import rows as R
from . import zono_fast as ZF
from .rows import ZonoSet

for _p in (R.REPO,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from world_model.v2.tray_rep.sets import qp as QP  # noqa: E402


@dataclass
class BaseCfg:
    gamma: float = 1.5                   # 1/s    class-K gain (gamma * v_max <= a_slew, else the slew limit cannot brake in time)
    a_brk: float = 0.4                   # m/s^2  braking decel assumed by the stopping-distance bound (< a_slew)
    margin: float = 0.03                 # m      extra clearance beyond the point's radius (positive = tighter)
    d_act: float = 0.40                  # m      activation distance (>= v^2 / (2 a_brk) + margin)
    retreat_cap: float = 0.15            # m/s    max demanded retreat speed when h < 0
    v_max: tuple = (0.30, 0.30, 0.30)    # m/s, m/s, rad/s
    a_slew: tuple = (0.50, 0.50, 1.00)   # m/s^2, rad/s^2   |u - u_prev| <= a_slew dt
    w_scale: tuple = (0.30, 0.30, 0.10)  # metric: deviation of this size costs 1 (yaw is expensive: the path is holonomic)
    max_rows: int = 48
    per_facet: int = 3
    max_peel: int = 6
    hopeless_frac: float = 0.10          # lifted rows sit this fraction of the way from the best to the worst box corner
    use_footprint: bool = True
    every: int = 1                       # re-solve every N-th call, hold the (slew-clipped) twist in between


# --------------------------------------------------------------------------------------------------------------- pure row math
def twist_row(n, p, origin):
    """g (R,3) with  n.v_base(u) = g.u  for world normals n (R,3) at world points p (R,3), base origin `origin` (2,)."""
    return np.stack([n[:, 0], n[:, 1], -n[:, 0] * (p[:, 1] - origin[1]) + n[:, 1] * (p[:, 0] - origin[0])], 1)


def beta_of(h, gamma, a_brk, retreat_cap):
    """allowed approach speed: hdot >= -beta."""
    hp = np.maximum(h, 0.0)
    return np.where(h >= 0.0, np.minimum(gamma * hp, np.sqrt(2.0 * a_brk * hp)), np.maximum(gamma * h, -retreat_cap))


def facet_pairs_fidx(p, r, zs: ZonoSet, margin, d_act):
    """facet_pairs + the active facet index: (kk, nrm, h, zii, fidx).  Numba candidate search with the grid-hash cull (zono_fast), same pairs / order / values
    as the dense facet_pairs_ref to ~1e-15."""
    kk, zii, fidx, nrm, h = ZF.facet_pairs(p, r, zs.packed(), d_act, margin)
    return kk, nrm, h, zii, fidx


def facet_pairs(p, r, zs: ZonoSet, margin, d_act):
    """(kk, nrm, h, zii) for every (point, zonotope) pair with sd_lb - r < d_act (active facet), exactly the rows.py logic (h includes margin)."""
    if R.FAST:
        return facet_pairs_fidx(p, r, zs, margin, d_act)[:4]
    return facet_pairs_ref(p, r, zs, margin, d_act)


def facet_pairs_ref(p, r, zs: ZonoSet, margin, d_act):
    """the dense numpy original (kept for the equivalence tests / ZONO_ROWS_REF=1)."""
    gap = np.maximum(np.maximum(zs.lo[None] - p[:, None], p[:, None] - zs.hi[None]), 0.0)
    near = np.linalg.norm(gap, axis=2) - r[:, None] < d_act
    kk_l, nn_l, h_l, zi_l = [], [], [], []
    for zi in np.flatnonzero(near.any(0)):
        H = zs.h[zi]
        ks = np.flatnonzero(near[:, zi])
        S = p[ks] @ H.N.T - H.d
        a = S.argmax(1)
        sd = S[np.arange(len(ks)), a]
        act = sd - r[ks] < d_act
        if act.any():
            kk_l.append(ks[act]); nn_l.append(H.N[a[act]]); h_l.append(sd[act] - r[ks[act]] - margin); zi_l.append(np.full(int(act.sum()), zi))
    if not kk_l:
        return np.zeros(0, int), np.zeros((0, 3)), np.zeros(0), np.zeros(0, int)
    return np.concatenate(kk_l), np.concatenate(nn_l), np.concatenate(h_l), np.concatenate(zi_l)


def base_rows_from_points(p, r, v_arm, zs: ZonoSet, origin, cfg: BaseCfg, u_ref=None):
    """CBF rows  A u <= b  on the base twist from points p (K,3), radii r (K,), arm-induced point velocity v_arm (K,3).
    Returns (A (R,3), b (R,), meta dict(h, kk, zii, nrm)).  Rows are capped at cfg.max_rows (most urgent at u_ref first, <= per_facet per (zonotope, facet))."""
    if R.FAST:
        kk, nrm, h, zii, fidx = facet_pairs_fidx(p, r, zs, cfg.margin, cfg.d_act)
        if not len(h):
            return np.zeros((0, 3)), np.zeros(0), dict(h=h, kk=kk, zii=zii, nrm=nrm, dropped=0)
        u0 = np.zeros(3) if u_ref is None else np.asarray(u_ref, float)
        keep, A, b = ZF.bf_rows(kk, nrm, h, p, v_arm, origin, cfg.gamma, cfg.a_brk, cfg.retreat_cap, u0, zs.packed().gid1000[fidx], cfg.max_rows, cfg.per_facet)
        dropped = len(h) - len(keep)
        if dropped:
            h, kk, nrm, zii = h[keep], kk[keep], nrm[keep], zii[keep]
        return A, b, dict(h=h, kk=kk, zii=zii, nrm=nrm, dropped=dropped)
    kk, nrm, h, zii = facet_pairs_ref(p, r, zs, cfg.margin, cfg.d_act)
    if not len(h):
        return np.zeros((0, 3)), np.zeros(0), dict(h=h, kk=kk, zii=zii, nrm=nrm, dropped=0)
    g = twist_row(nrm, p[kk], origin)
    b = np.einsum("ri,ri->r", nrm, v_arm[kk]) + beta_of(h, cfg.gamma, cfg.a_brk, cfg.retreat_cap)
    A = -g
    dropped = 0
    if len(h) > cfg.max_rows:
        u0 = np.zeros(3) if u_ref is None else np.asarray(u_ref, float)
        order = np.argsort(-(A @ u0 - b), kind="stable")                         # most violated at the nominal first
        key = np.concatenate([zii[:, None], np.round(nrm * 1000).astype(np.int64)], 1)
        _u, gid = np.unique(key, axis=0, return_inverse=True)
        gid = np.asarray(gid).reshape(-1)
        og = gid[order]
        srt = np.argsort(og, kind="stable")                                      # group-major, urgency order kept inside a group
        start = np.concatenate([[0], np.flatnonzero(np.diff(og[srt])) + 1])
        rank_sorted = np.arange(len(srt)) - np.repeat(start, np.diff(np.concatenate([start, [len(srt)]])))
        rank = np.empty(len(srt), int); rank[srt] = rank_sorted
        keep = order[rank < cfg.per_facet][: cfg.max_rows]
        dropped = len(h) - len(keep)
        A, b, h, kk, nrm, zii = A[keep], b[keep], h[keep], kk[keep], nrm[keep], zii[keep]
    return A, b, dict(h=h, kk=kk, zii=zii, nrm=nrm, dropped=dropped)


def lift_hopeless(A, b, lo, hi, frac):
    """rows A u <= b that no u in the box [lo, hi] can satisfy are lifted to bmin + frac (bmax - bmin).  Returns (b', n_lifted)."""
    if not len(b):
        return b, 0
    bmin = np.where(A > 0, A * lo, A * hi).sum(1)
    bmax = np.where(A > 0, A * hi, A * lo).sum(1)
    bad = b < bmin
    return np.where(bad, bmin + frac * (bmax - bmin), b), int(bad.sum())


def solve_base(A, b, u_nom, lo, hi, w, cfg: BaseCfg, u_prev):
    """projection of u_nom onto {A u <= b} ∩ box.  Returns (u, info).  See the module docstring for the lift / peel / brake policy."""
    b2, n_lift = lift_hopeless(A, b, lo, hi, cfg.hopeless_frac)
    E = np.eye(3)
    A_box = np.vstack([E, -E])
    b_box = np.concatenate([hi, -lo])
    Ak, bk = A.copy(), b2.copy()
    n_peel, ok = 0, False
    u = np.clip(u_nom, lo, hi)
    for _ in range(cfg.max_peel + 1):
        AA = np.vstack([A_box, Ak]) if len(bk) else A_box
        bb = np.concatenate([b_box, bk]) if len(bk) else b_box
        u, ok, nsw, viol = QP.project(AA, bb, u_nom, w)
        if ok:
            break
        if not len(bk):
            break
        res = Ak @ u - bk
        j = int(np.argmax(res))
        if res[j] <= 1e-4:
            break
        Ak, bk = np.delete(Ak, j, 0), np.delete(bk, j)
        n_peel += 1
    braked = False
    if not ok:                                                                  # last resort: stop at the slew limit
        braked = True
        u = np.clip(np.zeros(3), lo, hi)                                          # [lo, hi] already encodes the slew limit around u_prev
    return np.clip(u, lo, hi), dict(ok=ok, n_lift=n_lift, n_peel=n_peel, braked=braked)


# --------------------------------------------------------------------------------------------------------------- footprint
def footprint_points(model, origin_xy, yaw, data=None, cls=None):
    """local-frame footprint points (K,3) + zero radii: the 8 corners + 6 face centres of the oriented bounding box of every collidable geom of
    the YOR that is neither an arm link nor the tray / ball.  Stored relative to the mocap origin in the base frame (rigid with the base)."""
    from world_model.v2.house import phase1 as P1
    cls = P1.geom_class(model) if cls is None else cls
    d = mujoco.MjData(model)
    if data is not None:
        mujoco.mj_copyData(d, model, data)
    mujoco.mj_forward(model, d)
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    o = np.array([origin_xy[0], origin_xy[1], 0.0])
    sgn = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
    faces = np.vstack([np.eye(3), -np.eye(3)])
    pts = []
    for g in range(model.ngeom):
        if cls[g] != "B" or not (model.geom_contype[g] or model.geom_conaffinity[g]) or model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        ctr, half = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
        Rg = d.geom_xmat[g].reshape(3, 3)
        loc = np.vstack([ctr + sgn * half, ctr + faces * half])
        pw = d.geom_xpos[g] + loc @ Rg.T
        pts.append(((pw - o) @ Rz))                                              # R_z^T (p - o)
    P = np.vstack(pts) if pts else np.zeros((0, 3))
    return P


class BaseFilter:
    """per-step base-twist filter.  `ctl` = the ZonoCtl whose perceived zonotopes / keypoints / tracked tray are reused."""

    def __init__(self, ctl, cfg: BaseCfg | None = None, origin_xy=(0.0, 0.0), yaw=0.0, data=None):
        self.ctl, self.cfg = ctl, cfg or BaseCfg()
        self.m = ctl.m
        self.u_prev = np.zeros(3)
        self.foot_loc = footprint_points(self.m, origin_xy, yaw, data=data) if self.cfg.use_footprint else np.zeros((0, 3))
        ks_r = ctl.kps["right"]
        self.right_arm_idx = np.array([i for i, nm in enumerate(ks_r.names) if ":" not in nm], int)        # tray / ball offsets live in `left` already
        self.ks_right_arm = R.KeypointSet(ks_r.bodies[self.right_arm_idx], ks_r.offs[self.right_arm_idx], ks_r.radii[self.right_arm_idx],
                                          [ks_r.names[i] for i in self.right_arm_idx])
        self._names_arm = list(ctl.kps["left"].names) + list(self.ks_right_arm.names)
        self._radii_arm = np.concatenate([ctl.kps["left"].radii, self.ks_right_arm.radii])
        self.n_calls = 0
        self.binding = []
        self.stats = []        # (ms, n_rows, n_lift, n_peel, braked, h_min, resid_max, intervention)

    # ---- point set
    def points(self, d, origin, yaw):
        """(p (K,3), r (K,), v_arm (K,3), n_footprint) at the current (forwarded) state."""
        ctl = self.ctl
        pl, _Jl, _Jdl, vl, _ = R.kp_states(self.m, d, ctl.kps["left"], ctl.ld)
        pr, _Jr, _Jdr, vr, _ = R.kp_states(self.m, d, self.ks_right_arm, ctl.rd)
        P = [pl, pr]; V = [vl, vr]; Rr = [self._radii_arm]
        self.names = self._names_arm
        if ctl.tracked is not None and len(ctl.tracked):
            if R.FAST:                                                       # J qd only: no measured-velocity Jacobians
                K_ = len(ctl.tracked)
                pt = d.xpos[ctl.tracked.bodies] + np.einsum("kij,kj->ki", d.xmat[ctl.tracked.bodies].reshape(K_, 3, 3), ctl.tracked.offs)
                va = 0.5 * (R.tracked_arm_velocity(self.m, d, ctl.tracked, ctl.ld, ctl.gb["left"], pt) + R.tracked_arm_velocity(self.m, d, ctl.tracked, ctl.rd, ctl.gb["right"], pt))
            else:
                pt, Jl, _a, _b, qdl = R.tracked_states(self.m, d, ctl.tracked, ctl.ld, ctl.gb["left"])
                _p2, Jr_, _c, _e, qdr = R.tracked_states(self.m, d, ctl.tracked, ctl.rd, ctl.gb["right"])
                va = 0.5 * (np.einsum("kij,j->ki", Jl, qdl) + np.einsum("kij,j->ki", Jr_, qdr))
            P.append(pt); V.append(va); Rr.append(ctl.tracked.radii); self.names = self.names + list(ctl.tracked.names)
        nf = len(self.foot_loc)
        if nf:
            c, s = math.cos(yaw), math.sin(yaw)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
            P.append(np.array([origin[0], origin[1], 0.0]) + self.foot_loc @ Rz.T); V.append(np.zeros((nf, 3))); Rr.append(np.zeros(nf))
            self.names = self.names + ["footprint"] * nf
        return np.vstack(P), np.concatenate(Rr), np.vstack(V), nf

    # ---- one step
    def step(self, d, u_nom, origin, yaw, dt):
        """d: forwarded MjData at the CURRENT state.  u_nom (3,) world-frame twist wanted by the path.  Returns (u_safe (3,), info)."""
        t0 = time.perf_counter()
        cfg = self.cfg
        u_nom = np.asarray(u_nom, float)
        vmax, aslew = np.array(cfg.v_max), np.array(cfg.a_slew)
        lo = np.maximum(-vmax, self.u_prev - aslew * dt)
        hi = np.minimum(vmax, self.u_prev + aslew * dt)
        lo, hi = np.minimum(lo, hi), hi
        self.n_calls += 1
        if cfg.every > 1 and (self.n_calls - 1) % cfg.every:                        # held tick: keep the last safe twist (slew-limited: unchanged)
            u = np.clip(self.u_prev, lo, hi)
            self.stats.append(((time.perf_counter() - t0) * 1e3, *self.stats[-1][1:]) if self.stats else (0.0, 0, 0, 0, 0, np.inf, 0.0, 0.0))
            return u, dict(held=True)
        zs = self.ctl.zonos
        if zs is None or not len(zs):
            u = np.clip(u_nom, lo, hi)
            self.u_prev = u
            self.stats.append(((time.perf_counter() - t0) * 1e3, 0, 0, 0, 0, np.inf, 0.0, float(np.linalg.norm((u_nom - u) / cfg.w_scale))))
            return u, dict(n_rows=0)
        p, r, v_arm, nf = self.points(d, origin, yaw)
        A, b, meta = base_rows_from_points(p, r, v_arm, zs, origin, cfg, u_ref=u_nom)
        u, sinfo = solve_base(A, b, u_nom, lo, hi, 1.0 / np.array(cfg.w_scale), cfg, self.u_prev)
        resid = float(np.max(A @ u - b, initial=-np.inf)) if len(b) else -np.inf
        hmin = float(meta["h"].min()) if len(meta["h"]) else np.inf
        self.u_prev = u.copy()
        if len(b) and self.n_calls % 100 == 1:                                  # which row binds (most active at the solution)
            j = int(np.argmax(A @ u - b))
            self.binding.append(dict(call=self.n_calls, point=self.names[int(meta["kk"][j])], zono=zs.names[int(meta["zii"][j])], h=float(meta["h"][j]),
                                     slack=float(b[j] - A[j] @ u), n=meta["nrm"][j].round(2).tolist(), u=u.round(4).tolist()))
        interv = float(np.linalg.norm((u_nom - u) / np.array(cfg.w_scale)))
        self.stats.append(((time.perf_counter() - t0) * 1e3, len(b), sinfo["n_lift"], sinfo["n_peel"], int(sinfo["braked"]), hmin, max(resid, 0.0), interv))
        return u, dict(n_rows=len(b), h_min=hmin, resid=resid, **sinfo)

    def summary(self):
        s = np.array(self.stats, float)
        if not len(s):
            return {}
        act = s[:, 1] > 0
        return dict(binding_samples=self.binding[-12:], steps=int(len(s)), ms_median=float(np.median(s[:, 0])), ms_p95=float(np.percentile(s[:, 0], 95)), ms_max=float(s[:, 0].max()),
                    rows_mean=float(s[:, 1].mean()), rows_max=int(s[:, 1].max()), steps_with_rows=int(act.sum()), steps_lifted=int((s[:, 2] > 0).sum()),
                    rows_peeled=int(s[:, 3].sum()), steps_peeled=int((s[:, 3] > 0).sum()), steps_braked=int(s[:, 4].sum()),
                    h_min_base_rows=float(s[act, 5].min()) if act.any() else float("inf"), row_resid_max=float(s[:, 6].max()),
                    steps_intervening=int((s[:, 7] > 1e-3).sum()), intervention_max=float(s[:, 7].max()))


# --------------------------------------------------------------------------------------------------------------- path reference
class PathRef:
    """progress-based nominal twist along a time-parameterised base path (traj.x / traj.y [/ traj.yaw], step dt).  The reference index advances only as
    far as the (filtered) base actually progressed along the path (nearest point searched forward from the previous index), so a base held back by the
    filter does not wind up: u_nom = path velocity at that index + k_p * (path point - base)."""

    def __init__(self, traj, dt, kp=1.0, window=600):
        self.xy = np.stack([np.asarray(traj.x, float), np.asarray(traj.y, float)], 1)
        self.v = np.zeros_like(self.xy)
        self.v[:-1] = (self.xy[1:] - self.xy[:-1]) / dt
        self.kp, self.window, self.idx = kp, window, 0
        self.seg = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.xy, axis=0), axis=1))])

    def nominal(self, xy_base, started=True):
        if not started:
            return np.zeros(3), 0
        i0 = self.idx
        i1 = min(i0 + self.window, len(self.xy) - 1)
        dd = np.linalg.norm(self.xy[i0:i1 + 1] - xy_base[None], axis=1)
        self.idx = i0 + int(np.flatnonzero(dd <= dd.min() + 1e-7)[-1])            # ties (dwell points) -> the LAST one, so a stationary start is escaped
        e = self.xy[self.idx] - xy_base
        vv = self.v[self.idx] + self.kp * e
        return np.array([vv[0], vv[1], 0.0]), self.idx

    def progress(self):
        return float(self.seg[self.idx] / max(self.seg[-1], 1e-9))
