"""pan/skills/kin.py -- mink kinematics for the v2 skills.  SIMULATION ONLY.

mink 1.2.0 (Apache-2.0) from bimanual_data/pylibs/mink120 (pip --target; 1.3 needs MuJoCo 3.10, we stay on 3.9.0).

ClearanceLimit replaces mink.CollisionAvoidanceLimit: on this model mj_geomDistance reads <= 0 for 25 of 159 separated
arm/env configurations (exact clearance > 3 mm; same with native CCD off, measured 10-30), so a limit built on it stalls on
false contacts.  Here: exact signed distance from the arm's convex-hull vertices to the environment boxes and the floor (the
same hulls MuJoCo collides), one row per (geom, box) for the K nearest vertices within DETECT:
    -n^T J_v dq <= GAIN * max(dist - DMIN, 0)   (displacement form; mink 1.2.0's own limit divides by dt, 1/dt too loose)
Box-edge vs hull-face: box-edge samples every EDGE_STEP against the hull face planes (the judge's lower bound), row on the
argmax face with the point's projection on it as the moving point.  Every path is still re-checked with segments._arm_env_clearance (vertex + edge
samples, never over-reads), which stays the judge.
"""
from __future__ import annotations

import sys

MINK_PATH = "/common/users/ss5772/bimanual_data/pylibs/mink120"
if MINK_PATH not in sys.path:
    sys.path.insert(0, MINK_PATH)

import mink                      # noqa: E402
import mujoco                    # noqa: E402
import numpy as np               # noqa: E402

DMIN = 0.005      # = segments.CLAMP["v2"]["clear"] (ASSUMED 5 mm)
DETECT = 0.03     # design: rows only for vertices closer than this [m]
GAIN = 0.5        # design: fraction of the remaining clearance one step may use
K_NEAR = 3        # design: rows per (geom, box)
ENV = ("pan_table_top", "stove_pad", "counter_near")


def _hulls(m, side):
    from scipy.spatial import ConvexHull
    out = []
    for g in range(m.ngeom):
        if ((m.body(m.geom_bodyid[g]).name or "").startswith(f"{side}_arm") and (m.geom_contype[g] or m.geom_conaffinity[g])
                and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH):
            did = m.geom_dataid[g]
            V = m.mesh_vert[m.mesh_vertadr[did]:m.mesh_vertadr[did] + m.mesh_vertnum[did]].astype(float)
            H = ConvexHull(V)
            Vh = V[H.vertices].copy()
            c = Vh.mean(axis=0)
            out.append((g, Vh, H.equations.copy(), c, float(np.linalg.norm(Vh - c, axis=1).max())))
    return out


_SGN = np.array([[(k >> 2) & 1, (k >> 1) & 1, k & 1] for k in range(8)]) * 2.0 - 1.0
_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if np.sum(_SGN[a] != _SGN[b]) == 1]
EDGE_STEP = 0.002   # = segments.EDGE_STEP


def box_sdf(x, c, R, h):
    """signed distance + outward unit normal from points x (N,3) to a box (centre c, rotation R, half sizes h)."""
    p = (x - c) @ R
    q = np.abs(p) - h
    out = np.linalg.norm(np.maximum(q, 0), axis=1)
    inside = np.minimum(q.max(axis=1), 0)
    dist = out + inside
    n_loc = np.where((q > 0), np.sign(p) * np.maximum(q, 0), 0.0)
    nrm = np.linalg.norm(n_loc, axis=1)
    k = np.argmax(q, axis=1)
    ins = nrm < 1e-12
    n_loc[ins] = 0.0
    n_loc[ins, k[ins]] = np.sign(p[ins, k[ins]])
    n_loc /= np.maximum(np.linalg.norm(n_loc, axis=1, keepdims=True), 1e-12)
    return dist, n_loc @ R.T


class ClearanceLimit(mink.Limit):
    def __init__(self, m, sides, env=ENV, dmin=DMIN, detect=DETECT, gain=GAIN, k=K_NEAR, floor=True):
        self.m, self.dmin, self.detect, self.gain, self.k = m, dmin, detect, gain, k
        self.hulls = [h for s in sides for h in _hulls(m, s)]
        self.env = [m.geom(n).id for n in env]
        self.floor = m.geom("floor").id if floor else None
        self._jp = np.zeros((3, m.nv))

    def rows(self, d):
        """-> list of (dist, normal, body, point) for the near vertex/box pairs."""
        m = self.m
        out = []
        for g, V, Eq, c, rad in self.hulls:
            Rg, pg = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
            X = pg + V @ Rg.T
            cw = pg + c @ Rg.T
            for e in self.env:
                Re, pe, hh = d.geom_xmat[e].reshape(3, 3), d.geom_xpos[e], m.geom_size[e]
                dist, nrm = box_sdf(X, pe, Re, hh)
                for i in np.argsort(dist)[:self.k]:
                    if dist[i] < self.detect:              # hull vertex vs box: moving point = the vertex, normal = box outward
                        out.append((float(dist[i]), nrm[i], m.geom_bodyid[g], X[i]))
                C = (_SGN * hh) @ Re.T + pe
                reach = rad + self.detect
                pts = []
                for a, b in _EDGES:
                    u = C[b] - C[a]
                    L = float(np.linalg.norm(u))
                    u = u / L
                    t0 = float((cw - C[a]) @ u)
                    perp2 = float(np.sum((cw - C[a] - t0 * u) ** 2))
                    if perp2 > reach * reach:
                        continue
                    half = np.sqrt(reach * reach - perp2)
                    lo, hi = max(0.0, t0 - half), min(L, t0 + half)
                    if hi >= lo:
                        ts = np.arange(lo, hi + EDGE_STEP, EDGE_STEP)
                        pts.append(C[a] + np.minimum(ts, hi)[:, None] * u)
                if pts:
                    P = np.vstack(pts)
                    F = (P - pg) @ Rg @ Eq[:, :3].T + Eq[:, 3]       # per point, per face: signed plane distance (mesh frame)
                    fi = F.argmax(axis=1)
                    lb = F[np.arange(len(P)), fi]
                    for i in np.argsort(lb)[:self.k]:
                        if lb[i] < self.detect:            # box edge vs hull face: moving point = projection on the face,
                            nf = Rg @ Eq[fi[i], :3]        # normal = -(hull face outward), so the row reads like the vertex one
                            out.append((float(lb[i]), -nf, m.geom_bodyid[g], P[i] - lb[i] * nf))
            if self.floor is not None:
                zf = d.geom_xpos[self.floor][2]
                for i in np.argsort(X[:, 2])[:self.k]:
                    if X[i, 2] - zf < self.detect:
                        out.append((float(X[i, 2] - zf), np.array([0, 0, 1.0]), m.geom_bodyid[g], X[i]))
        return out

    def compute_qp_inequalities(self, configuration, dt):
        d = configuration.data
        rs = self.rows(d)
        if not rs:
            return mink.Constraint()
        G = np.zeros((len(rs), self.m.nv))
        h = np.zeros(len(rs))
        for j, (dist, n, b, x) in enumerate(rs):
            mujoco.mj_jac(self.m, d, self._jp, None, x, b)
            G[j] = -n @ self._jp
            h[j] = self.gain * max(dist - self.dmin, 0.0)   # below DMIN: no closer (h = 0), not "jump out" (QP stays feasible)
        return mink.Constraint(G=G, h=h)


class ArmJointLimit(mink.Limit):
    """joint-range limit on the SOLVED arm joints only.  mink.ConfigurationLimit covers every limited joint in the model; a
    frozen finger joint pressed past its range by a closed claw then conflicts with "frozen" and the QP is infeasible (V10
    lift, 10-30).  Rows: lo - q <= dq <= hi - q, scaled by GAIN, never asking for more than "no further out"."""

    def __init__(self, m, dofs, gain=0.95):
        self.m, self.gain = m, gain
        self.dofs = list(dofs)
        jn = [int(np.where(m.jnt_dofadr == v)[0][0]) for v in self.dofs]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jn])
        self.lo, self.hi = m.jnt_range[jn, 0].copy(), m.jnt_range[jn, 1].copy()

    def compute_qp_inequalities(self, configuration, dt):
        q = configuration.q[self.qadr]
        n, nv = len(self.dofs), self.m.nv
        G = np.zeros((2 * n, nv))
        G[np.arange(n), self.dofs] = 1.0
        G[n + np.arange(n), self.dofs] = -1.0
        h = np.concatenate([self.gain * np.maximum(self.hi - q, 0.0), self.gain * np.maximum(q - self.lo, 0.0)])
        return mink.Constraint(G=G, h=h)


class Kin:
    """one mink Configuration on a COPY of the model state; solves for the given arms only (all other DoFs frozen)."""

    def __init__(self, m, d, dofs, sides, clearance=True):
        self.m = m
        self.cfg = mink.Configuration(m, d.qpos.copy())
        free = [i for s in sides for i in dofs[s]]
        self.freeze = mink.DofFreezingTask(m, dof_indices=[i for i in range(m.nv) if i not in free])
        self.limits = [ArmJointLimit(m, free)] + ([ClearanceLimit(m, sides)] if clearance else [])
        self.sides = sides

    def solve(self, targets, iters=60, tol=(1e-4, 1e-3), dt=0.02, posture=None):
        """targets: {site_name: SE3}.  Integrates mink until every frame error is below tol (pos m, rot rad)."""
        tasks = []
        for site, T in targets.items():
            t = mink.FrameTask(site, "site", position_cost=1.0, orientation_cost=0.5)
            t.set_target(T)
            tasks.append(t)
        if posture is not None:
            pt = mink.PostureTask(self.m, cost=1e-3)
            pt.set_target(posture)
            tasks.append(pt)
        err = np.inf
        for _ in range(iters):
            try:
                v = mink.solve_ik(self.cfg, tasks, dt, "daqp", damping=1e-3, limits=self.limits, constraints=[self.freeze])
            except mink.exceptions.NoSolutionFound:
                return self.cfg.q.copy(), (np.inf, np.inf)          # the planner reports it as a failed plan
            self.cfg.integrate_inplace(v, dt)
            errs = [t.compute_error(self.cfg) for t in tasks[:len(targets)]]
            ep_, er_ = max(np.linalg.norm(e[:3]) for e in errs), max(np.linalg.norm(e[3:]) for e in errs)
            err = (ep_, er_)
            if ep_ < tol[0] and er_ < tol[1]:
                break
        return self.cfg.q.copy(), err

    def site_pose(self, site):
        return self.cfg.get_transform_frame_to_world(site, "site")


def se3(R, p):
    return mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(np.asarray(R)), np.asarray(p, float))
