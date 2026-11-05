"""srs_ik.py -- analytic arm-angle (SRS) inverse kinematics for one NERO arm, constants read from the compiled MuJoCo model.

SRS = spherical shoulder (joint axes 1,2,3 meet at S), revolute elbow (joint 4 at E), spherical wrist (axes 5,6,7 meet at W).
Shimizu et al. 2008 style: for a tool pose and an arm angle psi (rotation of the elbow about the S->W line) the joints follow in closed form.

Everything is derived from the model at q=0 (product of exponentials, base-body frame): axis a_i, point p_i, tool home pose M.
Nothing is hand-typed.  The geometry is CHECKED (MODEL): shoulder axes meet at one point, wrist axes meet at one point (build raises if not).

Conventions
  * T_tool: 4x4 WORLD pose of the site `<side>_arm_ee_grasp`.  The arm base (parent body of joint1's body) is assumed static
    (it is in the scenes used so far); call `refresh_base(d)` to re-read its pose from an MjData.
  * psi = signed angle (about v=W-S) from the reference direction `ref` (default arm-base +z, projected perpendicular to v)
    to the elbow offset perpendicular to v.   psi is undefined when the elbow is straight (nan from arm_angle).
  * branch = (s4, s_sh, s_wr), each 0/1: elbow root, shoulder subproblem root, wrist subproblem root.  8 branches.
  * Joint angles are wrapped to (-pi, pi]; all MJCF limits span < 2*pi, so no 2*pi alternative can be in limits.
"""
from __future__ import annotations

import itertools

import mujoco
import numpy as np

BRANCHES = tuple(itertools.product((0, 1), repeat=3))


def _rot(a, t):
    c, s = np.cos(t), np.sin(t)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def _ang(a, b, ax):
    """signed angle about unit ax rotating a into b (both perpendicular-projected)."""
    return np.arctan2(ax @ np.cross(a, b), a @ b - (a @ ax) * (b @ ax))


def _euler3(R, a1, a2, a3):
    """all (q1,q2,q3) with R(a1,q1)R(a2,q2)R(a3,q3)=R (fixed home axes, a2 not parallel a1 or a3).  Paden-Kahan 2 on k=R a3."""
    k = R @ a3
    c = a1 @ a2
    det = 1 - c * c
    al = ((a1 @ k) - c * (a2 @ a3)) / det
    be = ((a2 @ a3) - c * (a1 @ k)) / det
    n = np.cross(a1, a2)
    g2 = ((a3 @ a3) - (al * a1 + be * a2) @ (al * a1 + be * a2)) / (n @ n)
    if g2 < -1e-9:
        return []
    g = np.sqrt(max(g2, 0.0))
    out = []
    for sg in (+1, -1):
        z = al * a1 + be * a2 + sg * g * n
        q2 = _ang(a3, z, a2)
        q1 = _ang(z, k, a1)
        R12 = _rot(a1, q1) @ _rot(a2, q2)
        R3 = R12.T @ R
        # rotation angle about a3 of R3 (R3 is a pure rotation about a3):  vee part along a3, cos from trace
        vee = np.array([R3[2, 1] - R3[1, 2], R3[0, 2] - R3[2, 0], R3[1, 0] - R3[0, 1]]) / 2.0
        q3 = np.arctan2(a3 @ vee, (np.trace(R3) - 1.0) / 2.0)
        out.append((_wrap(q1), _wrap(q2), _wrap(q3)))
    return out


class SRSArm:
    PSI_STEP = np.deg2rad(0.5)      # psi_intervals sampling step (edges then bisected to 1e-7 rad)

    def __init__(self, m, side, ref=(0.0, 0.0, 1.0)):
        self.m, self.side = m, side
        self.d = mujoco.MjData(m)
        jn = [m.joint(f"{side}_arm_joint{i}") for i in range(1, 8)]
        self.qadr = np.array([j.qposadr[0] for j in jn])
        self.lo = np.array([j.range[0] for j in jn])
        self.hi = np.array([j.range[1] for j in jn])
        self.site = m.site(f"{side}_arm_ee_grasp").id
        self.base = int(m.body_parentid[jn[0].bodyid[0]])
        self.ref = np.asarray(ref, float) / np.linalg.norm(ref)
        # ---- home-pose screw data in the base-body frame (MODEL: read from the compiled model)
        self.d.qpos[self.qadr] = 0.0
        mujoco.mj_kinematics(m, self.d)
        self.refresh_base(self.d)
        Rb, pb = self.Tb[:3, :3], self.Tb[:3, 3]
        self.a = np.array([Rb.T @ self.d.xaxis[j.id] for j in jn])
        self.p = np.array([Rb.T @ (self.d.xanchor[j.id] - pb) for j in jn])
        self.p_M = Rb.T @ (self.d.site_xpos[self.site] - pb)
        self.R_M = Rb.T @ self.d.site_xmat[self.site].reshape(3, 3)
        a, p = self.a, self.p
        self.S = p[0]
        self.W0 = p[4]
        self.shoulder_gap = max(self._line_dist(p[0], a[0], p[i], a[i]) for i in (1, 2))      # axes 1,2,3 concurrent?
        self.wrist_gap = max(self._line_dist(p[4], a[4], p[i], a[i]) for i in (5, 6))         # axes 5,6,7 concurrent?
        if self.shoulder_gap > 1e-6 or self.wrist_gap > 1e-6:
            raise ValueError(f"{side}: arm is not SRS (shoulder gap {self.shoulder_gap:.2e} m, wrist gap {self.wrist_gap:.2e} m)")
        self.u0 = p[3] - self.S                 # shoulder -> elbow joint (home)
        self.w0 = self.W0 - p[3]                # elbow joint -> wrist (home)
        self.a4 = a[3]

    @staticmethod
    def _line_dist(p1, a1, p2, a2):
        """distance between lines (p1,a1) and (p2,a2); concurrent <=> 0 (point-to-line if parallel)."""
        n = np.cross(a1, a2)
        if np.linalg.norm(n) < 1e-9:
            r = p2 - p1
            return np.linalg.norm(r - (r @ a1) * a1)
        return abs((p2 - p1) @ n) / np.linalg.norm(n)

    # ------------------------------------------------------------------ frames / FK
    def refresh_base(self, d):
        T = np.eye(4)
        T[:3, :3] = d.xmat[self.base].reshape(3, 3)
        T[:3, 3] = d.xpos[self.base]
        self.Tb = T

    def fk(self, q):
        """MuJoCo FK of the ee_grasp site: 4x4 world pose."""
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        T = np.eye(4)
        T[:3, :3] = self.d.site_xmat[self.site].reshape(3, 3)
        T[:3, 3] = self.d.site_xpos[self.site]
        return T

    def _fk_poe(self, q):
        """product-of-exponentials FK in the base frame: (R_tool, p_tool, R123, R4, W)."""
        Rs = [_rot(self.a[i], q[i]) for i in range(7)]
        R123 = Rs[0] @ Rs[1] @ Rs[2]
        R14 = R123 @ Rs[3]
        Rq = R14 @ Rs[4] @ Rs[5] @ Rs[6]
        S = self.S
        W = S + R123 @ (self.u0 + Rs[3] @ self.w0)
        p = W + Rq @ (self.p_M - self.W0)
        return Rq @ self.R_M, p, R123, W

    def _nref(self, v):
        v = v / np.linalg.norm(v)
        for r in (self.ref, np.array([1.0, 0, 0])):
            n = r - (r @ v) * v
            if np.linalg.norm(n) > 1e-6:
                return v, n / np.linalg.norm(n)
        raise ValueError("no reference direction")

    def arm_angle(self, q):
        """psi of configuration q (nan if the elbow is straight or S-W is parallel to every reference)."""
        _, _, R123, W = self._fk_poe(q)
        v, n = self._nref(W - self.S)
        e = R123 @ self.u0
        e = e - (e @ v) * v
        if np.linalg.norm(e) < 1e-9:
            return float("nan")
        e /= np.linalg.norm(e)
        return float(np.arctan2(v @ np.cross(n, e), n @ e))

    # ------------------------------------------------------------------ IK
    def _to_base(self, T_tool):
        Tl = np.linalg.inv(self.Tb) @ T_tool
        return Tl[:3, :3], Tl[:3, 3]

    def _prep(self, T_tool):
        R, p = self._to_base(T_tool)
        Rq = R @ self.R_M.T
        W = p - Rq @ (self.p_M - self.W0)
        v = W - self.S
        d2 = v @ v
        a, u, w = self.a4, self.u0, self.w0
        wpar = (w @ a) * a
        wperp = w - wpar
        A = u @ wperp
        B = u @ np.cross(a, w)
        C = (d2 - u @ u - w @ w) / 2.0 - (u @ a) * (w @ a)
        r = np.hypot(A, B)
        if r < 1e-12 or abs(C) > r * (1 + 1e-12):
            return None
        th = np.arccos(np.clip(C / r, -1, 1))
        q4s = [_wrap(np.arctan2(B, A) + s * th) for s in (+1, -1)]
        vh, n = self._nref(v)
        return dict(Rq=Rq, v=vh, n=n, q4s=q4s)

    def _solve(self, pr, psi, s4):
        """the 4 (shoulder, wrist) solutions for elbow root s4 -> dict {(s4,ssh,swr): q}"""
        q4 = pr["q4s"][s4]
        R4 = _rot(self.a4, q4)
        g = self.u0 + R4 @ self.w0
        gh = g / np.linalg.norm(g)
        uperp = self.u0 - (self.u0 @ gh) * gh
        v, n = pr["v"], pr["n"]
        if np.linalg.norm(uperp) < 1e-9:                       # elbow straight: psi irrelevant
            R123s = [self._align_axis(gh, v)]
        else:
            uh = uperp / np.linalg.norm(uperp)
            ep = np.cos(psi) * n + np.sin(psi) * np.cross(v, n)
            R123s = [np.column_stack([v, ep, np.cross(v, ep)]) @ np.column_stack([gh, uh, np.cross(gh, uh)]).T]
        out = {}
        R123 = R123s[0]
        sh = _euler3(R123, self.a[0], self.a[1], self.a[2])
        Rw = (R123 @ R4).T @ pr["Rq"]
        wr = _euler3(Rw, self.a[4], self.a[5], self.a[6])
        for i, (q1, q2, q3) in enumerate(sh):
            for j, (q5, q6, q7) in enumerate(wr):
                out[(s4, i, j)] = np.array([q1, q2, q3, q4, q5, q6, q7])
        return out

    @staticmethod
    def _align_axis(g, v):
        c = g @ v
        ax = np.cross(g, v)
        s = np.linalg.norm(ax)
        if s < 1e-12:
            return np.eye(3) if c > 0 else -np.eye(3) * 0 + np.diag([1, -1, -1])
        return _rot(ax / s, np.arctan2(s, c))

    def in_limits(self, q, tol=0.0):
        return bool(np.all(q >= self.lo - tol) and np.all(q <= self.hi + tol))

    def ik_full(self, T_tool, psi):
        """list of dict(branch, q, pos_err, rot_err, in_limits) over the (up to 8) branches; errors from MuJoCo FK."""
        pr = self._prep(T_tool)
        if pr is None:
            return []
        res = []
        for s4 in (0, 1):
            for br, q in self._solve(pr, psi, s4).items():
                T = self.fk(q)
                dR = T_tool[:3, :3].T @ T[:3, :3]
                res.append(dict(branch=br, q=q, pos_err=float(np.linalg.norm(T[:3, 3] - T_tool[:3, 3])),
                                rot_err=float(np.arctan2(np.linalg.norm([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / 2, (np.trace(dR) - 1) / 2)),
                                in_limits=self.in_limits(q)))
        return res

    def ik(self, T_tool, psi, limits_only=False, tol_pos=1e-6, tol_rot=1e-6):
        """list of q (7,) over the branches whose MuJoCo-FK residual is below tol; limits_only drops out-of-limit ones."""
        return [r["q"] for r in self.ik_full(T_tool, psi)
                if r["pos_err"] <= tol_pos and r["rot_err"] <= tol_rot and (r["in_limits"] or not limits_only)]

    # ------------------------------------------------------------------ feasible arm angles
    def psi_intervals(self, T_tool, branch):
        """list of (lo, hi) in (-pi, pi] [a feasible run crossing +-pi is merged as (lo, hi+2pi) with hi>pi] where branch's q is inside the
        MJCF limits.  Method: dense sampling with step PSI_STEP (0.5 deg), run edges bisected to 1e-7 rad.  A feasible run shorter than
        the step and wholly between two samples is missed (so sampled, not proof)."""
        pr = self._prep(T_tool)
        if pr is None:
            return []

        def ok(psi):
            q = self._solve(pr, psi, branch[0]).get(tuple(branch))
            return q is not None and self.in_limits(q)

        n = int(np.ceil(2 * np.pi / self.PSI_STEP))
        grid = -np.pi + np.arange(n + 1) * (2 * np.pi / n)
        f = np.array([ok(x) for x in grid])
        if f.all():
            return [(-np.pi, np.pi)]
        if not f.any():
            return []

        def edge(a, b):                    # a infeasible/feasible mix: bisect to the boundary
            fa = ok(a)
            for _ in range(40):
                mid = 0.5 * (a + b)
                if ok(mid) == fa:
                    a = mid
                else:
                    b = mid
            return 0.5 * (a + b)

        h = 2 * np.pi / n
        f = f[:n]
        s0 = int(np.argmin(f))                 # start at an infeasible sample
        runs, cur = [], None
        for k in range(n):
            i, j = (s0 + k) % n, (s0 + k + 1) % n
            x0 = grid[s0] + k * h
            if not f[i] and f[j]:
                cur = edge(x0, x0 + h)
            elif f[i] and not f[j]:
                runs.append((cur, edge(x0, x0 + h)))
                cur = None
        out = []
        for lo, hi in runs:
            while lo > np.pi:
                lo -= 2 * np.pi; hi -= 2 * np.pi
            while lo <= -np.pi:
                lo += 2 * np.pi; hi += 2 * np.pi
            out.append((lo, hi))
        return out
