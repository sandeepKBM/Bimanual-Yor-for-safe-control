"""pan/cert/belief.py -- the robot's set-valued belief Theta_t.  SIMULATION ONLY.

Theta = (pose prior box) x (aim residual box) x (friction interval) x (contents polytope).

* Pose prior (dx, dy, yaw): the WORLD set from the prior box; it is never shrunk by sensing (AGENTS.md rule 4).
* Aim residual (pos, yaw): how far the camera estimate may be from the true pose after observe_pose.  Sensing enters here.
* Friction scale mu: an interval; torque readings do not narrow it (proposal: friction shows only when slipping).
* Contents pi = (m, s_u, s_w), with m the hidden mass and s = m * c its first moment about the pan centre, in the pan frame
  (u = handle axis, pointing to the handle; w = z x u).  Loads are linear in pi, so the set is an H-polytope
  {pi : A pi <= b} and every linear check is exact at its vertices.  Torque rows |a . pi - y| <= eps narrow it (proposal,
  "Torque update").  If the rows leave the set empty, an assumption has failed (consistency event).

Widening (the proposer may only widen, Lemma 1) is defined on the prior, before any rows are added.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace

import numpy as np

# --- bounds (AGENTS.md: MEASURED / MODEL / ASSUMED) ------------------------------------------------------------------
PRIOR_DXY = 0.015      # ASSUMED prior, pre-registered (robust/PREREG.txt): |dx|, |dy| <= 1.5 cm -- a WORLD bound
PRIOR_YAW = 8.0        # ASSUMED prior: |yaw| <= 8 deg -- a WORLD bound
PRIOR_MU = (0.6, 1.2)  # ASSUMED prior: friction scale (all geoms)
PRIOR_M = (0.0, 0.4)   # ASSUMED prior: contents mass [kg]
C_MAX = 0.09 - 0.025   # MODEL: puck centre stays within the pan's inner radius (0.09) minus the puck radius (0.025) [m]
N_GON = 8              # design: the disc |c| <= C_MAX is over-approximated by the circumscribed regular 8-gon (a superset, so
                       # linear model checks stay conservative).  Its corners lie 8 % outside the disc; for SIMULATION a corner
                       # puck is pulled radially back onto the disc (contents_world, flagged 'clipped') -- sim is the falsifier.
AIM_POS = 0.0034       # ASSUMED bound, set from MEASURED max over n = 200 (results/pan_observe_pose_2026-10-29, zed noise):
AIM_YAW = 9.42         #   position residual p95 2.3 / max 3.4 mm; yaw residual p95 4.7 / max 9.4 deg.  Max over n is not a bound.
TOL = 1e-9


@dataclass(frozen=True)
class Contents:
    """H-polytope {pi : A pi <= b} over pi = (m, s_u, s_w).  `prior` holds the prior box it was built from (for widen)."""
    A: tuple
    b: tuple
    prior: tuple = (PRIOR_M, C_MAX)
    n_rows: int = 0                      # torque rows added since the prior

    @classmethod
    def from_prior(cls, m=PRIOR_M, c=C_MAX):
        lo, hi = float(m[0]), float(m[1])
        A, b = [(-1, 0, 0), (1, 0, 0)], [-lo, hi]
        for k in range(N_GON):                       # cos(a) s_u + sin(a) s_w <= c m   (a cone in (m, s): c inside the 8-gon)
            a = 2 * np.pi * k / N_GON
            A.append((-c, np.cos(a), np.sin(a)))
            b.append(0.0)
        return cls(A=tuple(map(tuple, np.asarray(A, float))), b=tuple(map(float, b)), prior=((lo, hi), float(c)))

    @property
    def Am(self):
        return np.asarray(self.A, float)

    @property
    def bv(self):
        return np.asarray(self.b, float)

    def contains(self, pi, tol=1e-7):
        return bool(np.all(self.Am @ np.asarray(pi, float) <= self.bv + tol))

    def chebyshev(self):
        """(centre, radius) of the largest ball inside the set (LP); radius <= 0 means empty or flat."""
        from scipy.optimize import linprog
        A, b = self.Am, self.bv
        nrm = np.linalg.norm(A, axis=1)
        res = linprog(c=[0, 0, 0, -1.0], A_ub=np.hstack([A, nrm[:, None]]), b_ub=b,
                      bounds=[(None, None)] * 3 + [(0, None)], method="highs")
        if res.status != 0:
            return None, -1.0
        return res.x[:3], float(res.x[3])

    def vertices(self):
        """exact vertices via halfspace intersection about the Chebyshev centre.  A set thinner than ~1e-9 counts as empty
        (rows with eps > 0 never make it flat unless the readings contradict the prior)."""
        from scipy.spatial import HalfspaceIntersection
        x0, r = self.chebyshev()
        if x0 is None or r <= 1e-9:
            return np.zeros((0, 3))
        A, b = self.Am, self.bv
        hs = HalfspaceIntersection(np.hstack([A, -b[:, None]]), x0)
        out = []
        for x in hs.intersections:
            if np.all(np.isfinite(x)) and not any(np.allclose(x, y, atol=1e-10) for y in out):
                out.append(x)
        return np.array(out).reshape(-1, 3)

    def empty(self):
        return len(self.vertices()) == 0

    def pruned(self):
        """drop rows that touch no vertex (redundant); keeps the set unchanged and the row count small."""
        V = self.vertices()
        if len(V) == 0:
            return self
        A, b = self.Am, self.bv
        keep = [i for i in range(len(b)) if np.min(b[i] - V @ A[i]) <= 1e-9 * (1 + abs(b[i]))]
        return replace(self, A=tuple(map(tuple, A[keep])), b=tuple(map(float, b[keep])))

    def interval(self, f):
        """[min, max] of the linear function f . pi over the set (exact at the vertices); None if empty."""
        V = self.vertices()
        if len(V) == 0:
            return None
        v = V @ np.asarray(f, float)
        return float(v.min()), float(v.max())

    def add_rows(self, a, y, eps):
        """intersect with |a_i . pi - y_i| <= eps_i (one torque reading per row; eps = model + sensor error bound)."""
        a = np.atleast_2d(np.asarray(a, float))
        y, eps = np.atleast_1d(np.asarray(y, float)), np.atleast_1d(np.asarray(eps, float))
        A = list(self.A) + [tuple(r) for r in a] + [tuple(-r) for r in a]
        b = list(self.b) + list(y + eps) + list(-y + eps)
        return replace(self, A=tuple(A), b=tuple(map(float, b)), n_rows=self.n_rows + len(y))

    def widen(self, m=None, c=None):
        """union with a proposed prior (may only grow).  Defined only before torque rows: the proposer widens the prior."""
        if self.n_rows:
            raise ValueError("widen is defined on the prior only (no torque rows yet)")
        (lo, hi), c0 = self.prior
        if m is not None:
            lo, hi = min(lo, float(m[0])), max(hi, float(m[1]))
        if c is not None:
            c0 = max(c0, float(c))
        return Contents.from_prior((lo, hi), c0)


@dataclass(frozen=True)
class Belief:
    """the robot's belief.  Pose prior = world set; aim residual = sensing error; contents polytope narrowed by torque rows."""
    dxy: float = PRIOR_DXY
    yaw: float = PRIOR_YAW
    mu: tuple = PRIOR_MU
    contents: Contents = field(default_factory=Contents.from_prior)
    aim_pos: float = AIM_POS
    aim_yaw: float = AIM_YAW
    observed: str | None = None          # observe_pose noise level, or None before observing
    events: int = 0                      # consistency events counted so far

    # --- vertices ------------------------------------------------------------------------------------------------
    def contents_vertices(self):
        return self.contents.vertices()

    def vertices(self, residual=True, pose_prior=False):
        """the corner thetas a certificate must cover.  residual: aim-residual corners (dx, dy, dyaw signs; the pan is
        reconstructed at the estimate plus this residual); pose_prior: world-box corners instead (before observing)."""
        if pose_prior:
            poses = [dict(dx=sx * self.dxy, dy=sy * self.dxy, yaw_deg=sw * self.yaw)
                     for sx, sy, sw in itertools.product((-1, 1), repeat=3)]
        elif residual:
            poses = [dict(dx=sx * self.aim_pos, dy=sy * self.aim_pos, yaw_deg=sw * self.aim_yaw)
                     for sx, sy, sw in itertools.product((-1, 1), repeat=3)]
        else:
            poses = [dict(dx=0.0, dy=0.0, yaw_deg=0.0)]
        out = []
        for pose, mu, pi in itertools.product(poses, self.mu, self.contents_vertices()):
            out.append(dict(pose, mu_scale=float(mu), **contents_world(pi)))
        return out

    # --- update / consistency ------------------------------------------------------------------------------------
    def update(self, a, y, eps):
        """torque rows -> (new belief, consistent).  Inconsistent: no theta explains the readings (count the event; the
        caller runs the guard backup and resets to a guard range).  The returned belief is then the input belief."""
        c = self.contents.add_rows(a, y, eps)
        if c.empty():
            return replace(self, events=self.events + 1), False
        return replace(self, contents=c.pruned()), True

    def widen(self, dxy=None, yaw=None, mu=None, m=None, c=None):
        b = self
        if dxy is not None:
            b = replace(b, dxy=max(b.dxy, float(dxy)))
        if yaw is not None:
            b = replace(b, yaw=max(b.yaw, float(yaw)))
        if mu is not None:
            b = replace(b, mu=(min(b.mu[0], float(mu[0])), max(b.mu[1], float(mu[1]))))
        if m is not None or c is not None:
            b = replace(b, contents=b.contents.widen(m=m, c=c))
        return b

    def split(self, fn, **kw):
        """evaluate a verdict fn(theta) -> bool over the vertices: 'all' | 'none' | 'mixed' (probe only when mixed)."""
        v = [bool(fn(th)) for th in self.vertices(**kw)]
        return "all" if all(v) else ("none" if not any(v) else "mixed")

    def mass_interval(self):
        return self.contents.interval((1, 0, 0))

    def describe(self):
        V = self.contents_vertices()
        return dict(prior_pose=dict(dxy_m=self.dxy, yaw_deg=self.yaw, label="ASSUMED (world)"),
                    aim_residual=dict(pos_m=self.aim_pos, yaw_deg=self.aim_yaw, label="ASSUMED from MEASURED max over n=200"),
                    mu_scale=list(self.mu), mass_kg=self.mass_interval(), n_contents_vertices=len(V),
                    torque_rows=self.contents.n_rows, consistency_events=self.events, observed=self.observed)


def contents_world(pi):
    """contents vertex pi = (m, s_u, s_w) -> the world-dict keys Episode understands (puck at c = s / m, or none)."""
    m, su, sw = (float(x) for x in pi)
    if m <= 1e-6:
        return dict(hidden_mass=None, placement="none", contents=(0.0, 0.0, 0.0), clipped=False)
    c = np.array([su / m, sw / m])
    r = float(np.linalg.norm(c))
    clipped = r > C_MAX + 1e-9
    if clipped:
        c = c * (C_MAX / r)                          # physical puck position (the 8-gon corner lies outside the disc)
    return dict(hidden_mass=m, placement=[float(c[0]), float(c[1])], contents=(m, su, sw), clipped=bool(clipped))


def world_dict(theta, est_offset=None):
    """theta (a vertex) -> Episode world dict.  est_offset: the robot's pose estimate offsets (dx, dy, yaw_deg) that the
    residual is added to (reconstruction); None = theta's pose fields are the world pose itself."""
    dx, dy, yaw = theta["dx"], theta["dy"], theta["yaw_deg"]
    if est_offset is not None:
        dx, dy, yaw = dx + est_offset[0], dy + est_offset[1], yaw + est_offset[2]
    return dict(placement=theta["placement"], hidden_mass=theta["hidden_mass"],
                perturb=dict(dx=float(dx), dy=float(dy), yaw_deg=float(yaw), mu_scale=float(theta["mu_scale"])))
