"""constraints/ir_ext.py -- smallest extensions of the frozen set IR (tray_rep/sets/ir.py, zono.py), as NEW atoms. SIMULATION ONLY.

Nothing in ir.py / zono.py is edited: the new atoms subclass ir.Node (eval / rows with the same contracts) and are registered in a
dict-form builder `from_dict` that also understands every v0 form (it recurses itself, so new atoms can sit inside and / G / ...).

NEW SHAPES   Cylinder(r, z0, z1)     solid cylinder about the local z axis (cup volume).

NEW ATOMS (dict key)                                   margin (signed distance, > 0 satisfied)                    rows A u <= b
  InsideCyl      "inside_cyl"  {a, b, margin}          exact SDF of A to the solid cylinder B (inside: min slack,    inscribed 16-gon side + exact caps
                                                       outside: Euclidean, like Polytope.orthogonal)                 (inner approx of the disc)
  ContactKept    "contact"     {a, b, d}               d - sd(A (+) -B)   ==  Not(Clear(A, B, d)) but CONVEX in the   facets of the zonotope A(+)-B (+) d, fixed
                                                       relative position (v0 raises NotConvex on Not(Clear))         at u0; relative Jacobian J_A - J_B (B may move)
  ClearX         "clear_x"     {a, b, d}               v0 Clear with an exact-inside signed distance (v0 over-estimates   v0 Clear rows (fixed direction, inner approx)
                                                       penetration for rotated boxes)
  Between        "between"     {a, b, c, margin}       A inside hull(B u C): min facet slack of the hull H-rep       fixed hull normals at u0, active branch of
                                                                                                                     max(h_B, h_C) (inner approx; normals ignore motion)
  Over           "over"        {a, b, margin}          macro: Inside(A, gravity-aligned prism of B) AND A above     composition of v0 Inside rows (A.parent == B)
                                                       the top face of B (z >= top)
  SupportRel     "support_rel" {a, b}                  v0 Support margin                                             v0 rows with the relative Jacobian J_A - J_B
  TiltToward     "tilt_toward" {a, dir, theta}         asin(c . e) - theta : axis of A tilted at least theta         one exact linear row in the axis (halfspace)
                                                       towards the horizontal direction e (convex; plain Not(Upright) is not)
  Until          "until"       {phi, psi, lo, hi, at}  STL until: max_t' min(psi(t'), min_{t<=t'} phi(t))            rows for a CHOSEN switch time `at`
                                                                                                                     (default = argmax at u0): G[0,at] phi AND psi@at

Rows conventions are those of ir.rows: `jac = {"u0": (nu,), name: {"p": (K,3,nu), "axis": (K,2,nu)}}`; entities missing from jac are
treated as not moving with u (zero Jacobian).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull

from world_model.v2.tray_rep.sets import ir
from world_model.v2.tray_rep.sets.ir import (Entity, Pose, Unc, Shape, Region, Node, NotConvex, Inside, Clear, And,
                                              _rel, _t, _bat, shrink)
from world_model.v2.tray_rep.sets.zono import Zonotope, Polytope, sd_origin


# ====================================================================================== shapes
class Cylinder(Shape):
    """solid cylinder |xy| <= r, z0 <= z <= z1 in the local frame (the cup volume)."""

    def __init__(self, r, z0, z1):
        self.r, self.z0, self.z1 = float(r), float(z0), float(z1)

    def support(self, d):
        d = np.asarray(d)
        zc = np.where(d[..., 2] >= 0, self.z1, self.z0)
        return self.r * np.linalg.norm(d[..., :2], axis=-1) + d[..., 2] * zc

    def max_norm(self):
        return float(np.hypot(self.r, max(abs(self.z0), abs(self.z1))))

    def com(self):
        return np.array([0.0, 0.0, 0.5 * (self.z0 + self.z1)])

    def zonotope(self, m=8):
        """OUTER zonotope: zonogon prism with m generator directions, scaled so its support covers the disc."""
        th = np.arange(m) * np.pi / m
        phi = (np.arange(4 * m) + 0.5) * np.pi / (2 * m)
        hmin = min(np.abs(np.cos(phi_ - th)).sum() for phi_ in phi)
        ell = self.r / hmin
        G = np.zeros((3, m + 1))
        G[0, :m], G[1, :m] = ell * np.cos(th), ell * np.sin(th)
        G[2, m] = 0.5 * (self.z1 - self.z0)
        return Zonotope(np.array([0, 0, 0.5 * (self.z1 + self.z0)]), G)


# ====================================================================================== helpers
def _yaw_only(R):
    R = _bat(R)
    yaw = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    Ry = np.zeros(R.shape)
    Ry[..., 0, 0], Ry[..., 0, 1], Ry[..., 1, 0], Ry[..., 1, 1], Ry[..., 2, 2] = c, -s, s, c, 1.0
    return Ry


def _jp(jac, name, K, nu):
    """position Jacobian (K, 3, nu) of entity `name`; zeros if the entity does not move with u."""
    if name in jac and "p" in jac[name]:
        J = np.asarray(jac[name]["p"], np.float64)
        return np.broadcast_to(J, (K, 3, nu)) if J.ndim == 2 else J
    return np.zeros((K, 3, nu))


def _world_zono(e: Entity, state):
    P = e.pose(state)
    return e.shape.zonotope().linear(_bat(P.R), _bat(P.p))


def zono3_facets(Z: Zonotope, tol=1e-9, dedupe=True):
    """exact H-rep of a 3-D zonotope (unbatched): unit normals n_i (m,3) and offsets h_i = h_Z0(n_i) - n_i.c of the CENTRED set
    (so Z = {x: n_i.(x - c) <= h_i}). Facet normals are the cross products of generator pairs (needs rank 3)."""
    G = Z.G[:, np.linalg.norm(Z.G, axis=0) > tol]
    m = G.shape[1]
    ii, jj = np.triu_indices(m, 1)
    n = np.cross(G[:, ii].T, G[:, jj].T)
    nn = np.linalg.norm(n, axis=1)
    ok = nn > tol * (1 + np.linalg.norm(G[:, ii], axis=0) * np.linalg.norm(G[:, jj], axis=0))
    if not ok.any():
        raise ValueError("zono3_facets: generators do not span R^3")
    N = n[ok] / nn[ok, None]
    N = np.concatenate([N, -N], 0)
    if dedupe:
        keep = []
        for i, nv in enumerate(N):
            if not keep or not (N[keep] @ nv > 1 - 1e-9).any():
                keep.append(i)
        N = N[keep]
    h = np.abs(N @ G).sum(1)
    return N, h


def sep_dir(Z: Zonotope):
    """unit direction n from B to A that realises the signed distance of the origin to the (unbatched) zonotope Z = A (+) -B:
    outside -> towards the closest point of Z; inside -> minus the outward normal of the least-penetrated facet. n.c - h_G(n) == sd(Z)."""
    G0 = Zonotope(np.zeros(3), Z.G)
    try:
        N, h = zono3_facets(G0, dedupe=False)
    except ValueError:
        z = Z.c / max(np.linalg.norm(Z.c), 1e-12)
        return z
    slack = h + N @ Z.c
    i = int(np.argmin(slack))
    if slack[i] >= 0:
        return -N[i]
    from scipy.optimize import lsq_linear
    r = lsq_linear(Z.G, -Z.c, bounds=(-1.0, 1.0), method="bvls", tol=1e-12)
    zs = Z.G @ r.x + Z.c
    return zs / max(np.linalg.norm(zs), 1e-12)


def sd_exact(Z: Zonotope):
    """signed distance of the origin to an UNBATCHED 3-D zonotope Z (> 0 outside). v0 sd_origin is a direction-grid bound; for a point INSIDE it
    returns minus the best grid direction's support, i.e. over-estimates the penetration (up to size * grid angle, centimetres for 10 cm boxes
    rotated off the grid). Here: inside -> exact min facet slack of the exact H-rep; outside -> exact distance by a bounded least-squares
    (min |c + G xi|, |xi|<=1)."""
    try:
        N, h = zono3_facets(Zonotope(np.zeros(3), Z.G), dedupe=False)
    except ValueError:
        return sd_origin(Z)
    slack = h - N @ (-Z.c)                       # facets of the centred set evaluated at the origin relative to the centre
    smin = slack.min()
    if smin >= 0:
        return float(-smin)
    from scipy.optimize import lsq_linear                       # exact distance: min |c + G xi|, |xi| <= 1 (BVLS)
    r = lsq_linear(Z.G, -Z.c, bounds=(-1.0, 1.0), method="bvls", tol=1e-12)
    return float(np.linalg.norm(Z.G @ r.x + Z.c))


# ====================================================================================== InsideCyl
@dataclass
class InsideCyl(Node):
    """A (Point / Sphere / small convex body) is inside the solid cylinder B (shape Cylinder), shrunk by `margin`."""
    A: Entity
    B: Entity
    margin: float = 0.0
    n_facets: int = 16
    tag: str = ""

    def _a_pts(self):
        sh = self.A.shape
        if isinstance(sh, ir.Point):
            return np.zeros((1, 3)) + sh.at, 0.0
        if isinstance(sh, ir.Sphere):
            return sh.centre[None], sh.r
        Z = sh.zonotope()
        return Z.vertices(), 0.0

    def _extent(self, state, robust):
        """-> (rad_max, z_lo, z_hi, ...) of A (+ uncertainty) in B's frame at every K. Conservative in robust mode."""
        R, p, upos, rot = _rel(self.A, self.B, state)
        pts, rs = self._a_pts()
        P = np.einsum("...ij,vj->...vi", R, pts) + p[..., None, :]                 # (..., V, 3)
        rad = np.linalg.norm(P[..., :2], axis=-1).max(-1) + rs
        zlo, zhi = P[..., 2].min(-1) - rs, P[..., 2].max(-1) + rs
        if robust:
            if upos is not None:
                rad = rad + np.linalg.norm(upos.G[..., :2, :], axis=-2).sum(-1)       # |sum xi_i g_i| <= sum |g_i|
                zlo = zlo - np.abs(upos.G[..., 2, :]).sum(-1)
                zhi = zhi + np.abs(upos.G[..., 2, :]).sum(-1)
            if rot:
                e = rot * (self.A.shape.max_norm() + np.linalg.norm(p, axis=-1))
                rad, zlo, zhi = rad + e, zlo - e, zhi + e
        return rad, zlo, zhi

    def eval(self, state, robust=False):
        sh = self.B.shape
        rad, zlo, zhi = self._extent(state, robust)
        vr = rad - (sh.r - self.margin)
        vz = np.maximum(zhi - (sh.z1 - self.margin), (sh.z0 + self.margin) - zlo)
        slack = np.minimum(-vr, -vz)
        out = -np.hypot(np.maximum(vr, 0.0), np.maximum(vz, 0.0))
        return _t(np.where(slack >= 0, slack, out))

    def rows(self, state, jac, robust=True):
        if self.A.parent != self.B.name:
            raise ValueError("InsideCyl.rows: A must be expressed in B's frame (A.parent == B.name)")
        if not isinstance(self.A.shape, (ir.Point, ir.Sphere)):
            raise NotImplementedError("InsideCyl.rows: Point / Sphere A only")
        sh = self.B.shape
        R, p, upos, rot = _rel(self.A, self.B, state)
        pts, rs = self._a_pts()
        p = np.atleast_2d(p)
        K = len(p)
        Rr = np.broadcast_to(_bat(R), (K, 3, 3))
        c = p + np.einsum("kij,j->ki", Rr, pts[0])                          # centre of A in B's frame
        u0 = np.asarray(jac["u0"], np.float64)
        nu = len(u0)
        Jp = _jp(jac, self.A.name, K, nu)
        ang = (np.arange(self.n_facets) + 0.5) * 2 * np.pi / self.n_facets
        N = np.concatenate([np.stack([np.cos(ang), np.sin(ang), 0 * ang], 1), [[0, 0, 1.0], [0, 0, -1.0]]], 0)
        lim_r = (sh.r - self.margin - rs) * np.cos(np.pi / self.n_facets)
        bN = np.concatenate([np.full(self.n_facets, lim_r), [sh.z1 - self.margin - rs, -(sh.z0 + self.margin + rs)]])
        extra = np.zeros((K, len(N)))
        if robust:
            if upos is not None:
                extra = extra + np.broadcast_to(upos._support_many(N), (K, len(N)))
            if rot:
                extra = extra + rot * (self.A.shape.max_norm() + np.linalg.norm(p, axis=-1))[:, None]
        nJ = np.einsum("mi,kiu->kmu", N, Jp)
        b = bN[None] - extra - c @ N.T + np.einsum("kmu,u->km", nJ, u0)
        Kk, m = b.shape
        tags = [(self.tag or "inside_cyl", k, i) for k in range(Kk) for i in range(m)]
        return [(nJ.reshape(Kk * m, -1), b.reshape(-1), tags)]


# ====================================================================================== ContactKept
@dataclass
class ContactKept(Node):
    """dist(A, B) <= d  (sets stay in contact within d).  margin = d - sd(A (+) -B)  ==  Not(Clear(A, B, d)).eval, but its rows exist:
    the set of relative positions  {p : dist(p + A0, B0) <= d}  is convex.  Rows use the (outer) polytope  facets(A0 (+) -B0) offset by d,
    fixed at u0 (corner regions of the d-neighbourhood are slightly over-covered by <= d (sqrt2 - 1) -- a sound-near-edges caveat)."""
    A: Entity
    B: Entity
    d: float = 0.0
    tag: str = ""

    def _sets(self, state):
        PA, PB = self.A.pose(state), self.B.pose(state)
        ZA = self.A.shape.zonotope().linear(_bat(PA.R), None)             # centred
        ZB = self.B.shape.zonotope().linear(_bat(PB.R), None)
        cA, cB = _bat(PA.p), _bat(PB.p)
        return ZA, ZB, cA, cB

    def eval(self, state, robust=False):
        ZA, ZB, cA, cB = self._sets(state)
        Z0 = ZA.minkowski(ZB.neg())                                      # centred, possibly batched
        rel = cA + ZA.c - (cB + ZB.c)
        unc = [self.A.uncertainty(state).pos, self.B.uncertainty(state).pos]
        U = None
        if robust:
            U = unc[0]
            if unc[1] is not None:
                U = unc[1].neg() if U is None else U.minkowski(unc[1].neg())
        Zc = Zonotope(rel + Z0.c, Z0.G)
        out = []
        n = rel.shape[0] if rel.ndim > 1 else 1
        for k in range(n):
            Zk = Zonotope(Zc.c[k], Zc.G[k]) if Zc.batch else Zc
            if U is None:
                out.append(sd_exact(Zk))
            else:
                Uk = Zonotope(U.c[k], U.G[k]) if U.batch else U
                best = -np.inf
                for v in Uk.vertices():
                    best = max(best, sd_exact(Zonotope(Zk.c + v, Zk.G)))
                out.append(best)
        return self.d - np.array(out)

    def rows(self, state, jac, robust=True):
        ZA, ZB, cA, cB = self._sets(state)
        Z0 = ZA.minkowski(ZB.neg())
        rel = np.atleast_2d(cA + ZA.c - (cB + ZB.c))
        u0 = np.asarray(jac["u0"], np.float64)
        nu, K = len(u0), len(rel)
        Jr = _jp(jac, self.A.name, K, nu) - _jp(jac, self.B.name, K, nu)
        unc = [self.A.uncertainty(state).pos, self.B.uncertainty(state).pos]
        U = None
        if robust:
            U = unc[0]
            if unc[1] is not None:
                U = unc[1].neg() if U is None else U.minkowski(unc[1].neg())
        A_all, b_all, tags = [], [], []
        for k in range(K):
            Zk = Zonotope(Z0.c[k] if Z0.batch else Z0.c, Z0.G[k] if Z0.batch else Z0.G)
            N, h = zono3_facets(Zk)
            hb = h + self.d
            if U is not None:
                Uk = Zonotope(U.c[k], U.G[k]) if U.batch else U
                hb = hb - Uk._support_many(N) - 0.0
            nJ = N @ Jr[k]
            for i in range(len(N)):          # n.(rel(u)) <= hb  ->  n.Jr u <= hb - n.rel + n.Jr u0
                A_all.append(nJ[i])
                b_all.append(hb[i] - N[i] @ rel[k] + nJ[i] @ u0)
                tags.append((self.tag or "contact", k, i))
        return [(np.array(A_all), np.array(b_all), tags)]


@dataclass
class ClearX(Clear):
    """v0 Clear with the exact-inside signed distance (sd_exact): same rows."""

    def eval(self, state, robust=False):
        ZA, ZB = self._sets(state, robust)
        Z = ZA.minkowski(ZB.neg())
        if Z.batch:
            return np.array([sd_exact(Zonotope(Z.c[k], Z.G[k])) for k in range(Z.c.shape[0])]) - self.d
        return _t(sd_exact(Z) - self.d)

    def rows(self, state, jac, robust=True):
        """as v0 Clear.rows (a valid separating halfspace fixed at u0, inner approximation) but along the TRUE separating direction
        (closest point / least-penetrated facet) instead of the centre-to-centre direction: tight at u0, exact gradient."""
        ZA, ZB = self._sets(state, robust)
        Z = ZA.minkowski(ZB.neg())
        c = np.atleast_2d(Z.c)
        G = Z.G if Z.batch else Z.G[None]
        u0 = np.asarray(jac["u0"], np.float64)
        K, nu = len(c), len(u0)
        Jp = _jp(jac, self.A.name, K, nu) - _jp(jac, self.B.name, K, nu)
        Arows, brow, tags = [], [], []
        for k in range(K):
            n = sep_dir(Zonotope(c[k], G[k]))
            hz = float(np.abs(G[k].T @ n).sum())
            nJ = n @ Jp[k]
            Arows.append(-nJ)
            brow.append(float(n @ c[k] - nJ @ u0 - self.d - hz))
            tags.append((self.tag or "clear_x", k, 0))
        return [(np.array(Arows), np.array(brow), tags)]


# ====================================================================================== Between
@dataclass
class Between(Node):
    """A inside the convex hull of B and C (e.g. the box centre between two gripper pads). All three in a common frame.
    margin = min over the hull facets of (offset - h_A(n))  (inside: signed distance; outside: a lower bound of -distance)."""
    A: Entity
    B: Entity
    C: Entity
    margin: float = 0.0
    tag: str = ""

    def _hull(self, state, k):
        pts = []
        for e in (self.B, self.C):
            P = e.pose(state)
            R, p = np.broadcast_to(_bat(P.R), np.broadcast_shapes(_bat(P.R).shape, (3, 3)))[..., :, :], _bat(P.p)
            if R.ndim == 3:
                R, p = R[k], p[k]
            pts.append(e.shape.zonotope().linear(R, p).vertices())
        pts = np.concatenate(pts, 0)
        H = ConvexHull(pts)
        eq = np.unique(np.round(H.equations, 9), axis=0)               # coplanar triangles give duplicate facets
        return eq[:, :3], -eq[:, 3]                                    # n.x <= b, unit normals

    def _h_A(self, state, N, k):
        P = self.A.pose(state)
        R, p = _bat(P.R), _bat(P.p)
        if R.ndim == 3:
            R, p = R[k], p[k]
        return np.array([self.A.shape.support(R.T @ n) + n @ p for n in N])

    def _K(self, state):
        p = _bat(self.A.pose(state).p)
        return p.shape[0] if p.ndim > 1 else 1

    def eval(self, state, robust=False):
        out = []
        for k in range(self._K(state)):
            N, b = self._hull(state, k)
            h = self._h_A(state, N, k)
            if robust:
                u = self.A.uncertainty(state)
                if u.pos is not None:
                    z = u.pos if not u.pos.batch else Zonotope(u.pos.c[k], u.pos.G[k])
                    h = h + z._support_many(N)
            out.append(float((b - self.margin - h).min()))
        return np.array(out)

    def rows(self, state, jac, robust=True):
        u0 = np.asarray(jac["u0"], np.float64)
        nu, K = len(u0), self._K(state)
        JA, JB, JC = (_jp(jac, e.name, K, nu) for e in (self.A, self.B, self.C))
        A_all, b_all, tags = [], [], []
        for k in range(K):
            N, b = self._hull(state, k)
            h = self._h_A(state, N, k)
            pA = _bat(self.A.pose(state).p)
            pA = pA[k] if pA.ndim > 1 else pA
            hB = np.array([self._h_set(self.B, state, n, k) for n in N])
            hC = np.array([self._h_set(self.C, state, n, k) for n in N])
            ext = np.zeros(len(N))
            if robust:
                u = self.A.uncertainty(state)
                if u.pos is not None:
                    z = u.pos if not u.pos.batch else Zonotope(u.pos.c[k], u.pos.G[k])
                    ext = z._support_many(N)
            for i, n in enumerate(N):
                useB = hB[i] >= hC[i]                            # active branch of max(h_B, h_C) at u0 (inner approximation)
                Jbr = (JB if useB else JC)[k]
                hs = hB[i] if useB else hC[i]
                row = n @ (JA[k] - Jbr)                           # n.pA(u) + h_A0 + ext <= hs + n.pBr(u)
                rhs = hs - h[i] - ext[i] - self.margin + row @ u0     # n.(JA - Jbr)(u - u0) <= hs - h_A - ext - margin
                A_all.append(row)
                b_all.append(rhs)
                tags.append((self.tag or "between", k, i))
        return [(np.array(A_all), np.array(b_all), tags)]

    def _h_set(self, e, state, n, k):
        P = e.pose(state)
        R, p = _bat(P.R), _bat(P.p)
        if R.ndim == 3:
            R, p = R[k], p[k]
        return float(e.shape.zonotope().linear(R, p).support(n))


# ====================================================================================== Over (macro)
class Over(And):
    """A over B (gravity direction): the xy position of A is inside B's footprint (yaw only; B's x/y half extents, from a BoxShape)
    AND A is above B's top face.  Pure composition of the v0 Inside atom (so rows and robust semantics are v0's).
    A.parent must be None or B.name for rows (as Inside)."""

    def __init__(self, A: Entity, B: Entity, margin=0.0, tag="", above=0.0):
        hx, hy, hz = B.shape.half
        if A.parent == B.name:            # A given in B's frame (rows available): footprint prism and the half-space above the top face, in B's frame
            prism = Entity(B.name, ir.prism_xy(hx, hy), B.frame, B.unc, B.parent)
            top = Entity(B.name, Region(Polytope(np.array([[0.0, 0, -1.0]]), np.array([-(hz + above)]))), B.frame, None, B.parent)
        else:                             # A in the world frame: gravity-aligned (yaw-only) prism of B; margin only (v0 Inside.rows needs parent)
            prism = Entity(B.name + "__prism", ir.prism_xy(hx, hy), (lambda s, f=B.frame: Pose(_yaw_only(f(s).R), _bat(f(s).p))), B.unc, B.parent)
            top = Entity(B.name + "__top", Region(Polytope(np.array([[0.0, 0, -1.0]]), np.array([0.0]))),
                         (lambda s, f=B.frame, hz=hz: Pose(np.broadcast_to(np.eye(3), _bat(f(s).p).shape[:-1] + (3, 3)),
                                                           _bat(f(s).p) + np.array([0, 0, hz + above]))), None, None)
        super().__init__(Inside(A, prism, margin, tag=tag or "over_xy"), Inside(A, top, margin, tag=(tag or "over") + "_z"))
        self.A, self.B = A, B


# ====================================================================================== TiltToward
@dataclass
class TiltToward(Node):
    """axis of A tilted at least `theta` towards the horizontal unit direction e:   asin(c . e) - theta  (rad).  Convex (a halfspace in
    the axis space); the v0 spelling Not(Upright(A, theta)) is the non-convex 'outside the disc'."""
    A: Entity
    e: np.ndarray
    theta: float
    tag: str = ""

    def _ce(self, state, robust):
        z = _bat(self.A.pose(state).R)[..., :, 2]
        e = np.asarray(self.e, np.float64)
        e = e / np.linalg.norm(e)
        s = np.clip(z @ e, -1, 1)
        v = np.arcsin(s)
        if robust:
            v = v - self.A.uncertainty(state).rot
        return _t(v)

    def eval(self, state, robust=False):
        return self._ce(state, robust) - self.theta

    def rows(self, state, jac, robust=True):
        e = np.asarray(self.e, np.float64)
        e = e / np.linalg.norm(e)
        rot = self.A.uncertainty(state).rot if robust else 0.0
        z = np.atleast_2d(_bat(self.A.pose(state).R)[..., :, 2])
        Jc = np.asarray(jac[self.A.name]["axis"], np.float64)
        Jc = Jc[None] if Jc.ndim == 2 else Jc
        u0 = np.asarray(jac["u0"], np.float64)
        lim = np.sin(self.theta + rot)
        eJ = np.einsum("i,kiu->ku", e[:2], Jc)                         # d(c.e)/du
        # c(u) = c0 + Jc (u - u0);  c.e >= lim  <=>  -eJ u <= c0.e - lim - eJ u0
        b = z[:, :2] @ e[:2] - lim - np.einsum("ku,u->k", eJ, u0)
        # rescale each row by 1/cos(tilt0) (positive: same feasible set) so that the row slack is in RADIANS to first order, like the margin
        sc = 1.0 / np.maximum(np.sqrt(np.maximum(1.0 - (z[:, :2] @ e[:2]) ** 2, 0.0)), 0.05)
        return [(-eJ * sc[:, None], b * sc, [(self.tag or "tilt_toward", k, 0) for k in range(len(z))])]


# ====================================================================================== Until
class Until(Node):
    """phi U[lo,hi] psi (STL robustness): at k, max over t' in [k+lo, k+hi] of min(psi(t'), min over t in [k, t'] of phi(t)).
    NON-CONVEX in general (a disjunction over the switch time). rows(): with the switch time `at` fixed (or taken as the argmax at u0),
    phi must hold on [0, at] and psi at `at` -- convex when phi, psi rows are."""

    def __init__(self, phi: Node, psi: Node, lo=0, hi=None, at=None):
        self.phi, self.psi, self.lo, self.hi, self.at = phi, psi, int(lo), (None if hi is None else int(hi)), at

    def eval(self, state, robust=False):
        a, b = _t(self.phi.eval(state, robust)), _t(self.psi.eval(state, robust))
        K = len(a)
        out = np.empty(K)
        for k in range(K):
            hi = K - 1 if self.hi is None else min(k + self.hi, K - 1)
            best = -np.inf
            run = np.inf
            for t in range(k, hi + 1):
                run = min(run, a[t])
                if t >= k + self.lo:
                    best = max(best, min(b[t], run))
            out[k] = best
        return out

    def switch_time(self, state, robust=False):
        a, b = _t(self.phi.eval(state, robust)), _t(self.psi.eval(state, robust))
        K = len(a)
        best, arg, run = -np.inf, 0, np.inf
        for t in range(K):
            run = min(run, a[t])
            if t >= self.lo and min(b[t], run) > best:
                best, arg = min(b[t], run), t
        return arg

    def rows(self, state, jac, robust=True):
        at = self.at if self.at is not None else self.switch_time(state, False)
        out = []
        for A, b, tags in self.phi.rows(state, jac, robust):
            keep = np.array([t[1] <= at for t in tags])
            out.append((A[keep], b[keep], [t for t, k in zip(tags, keep) if k]))
        for A, b, tags in self.psi.rows(state, jac, robust):
            keep = np.array([t[1] == at for t in tags])
            out.append((A[keep], b[keep], [t for t, k in zip(tags, keep) if k]))
        return out


# ====================================================================================== SupportRel
@dataclass
class SupportRel(ir.Support):
    """v0 Support whose rows use the RELATIVE Jacobian J_A - J_B (v0 Support.rows treats the supporting body B as fixed under u, which is
    wrong for a stack whose lower blocks also move). B's rotation is still frozen at u0."""

    def rows(self, state, jac, robust=True):
        Zf, com = self._prep(state)
        u0 = np.asarray(jac["u0"], np.float64)
        K, nu = len(com), len(u0)
        Jp = (_jp(jac, self.A.name, K, nu) - _jp(jac, self.B.name, K, nu))
        unc = self.A.uncertainty(state)
        A_all, b_all, tags = [], [], []
        for k in range(K):
            P = Zonotope(Zf.c[k][:2], Zf.G[k][:2]).to_polytope_2d()
            nJ = P.N @ Jp[k][:2]
            extra = (self._unc_support(unc, P.N, k, K) + unc.rot * self.A.shape.max_norm()) if robust else np.zeros(len(P.N))
            for i in range(len(P.N)):
                A_all.append(nJ[i])
                b_all.append(P.b[i] - extra[i] - P.N[i] @ com[k][:2] + nJ[i] @ u0)
                tags.append((self.tag or "support", k, i))
        return [(np.array(A_all), np.array(b_all), tags)]


# ====================================================================================== dict form
def _builders():
    def b_inside_cyl(v, E, V, rec):
        return InsideCyl(E(v["a"]), E(v["b"]), float(V(v.get("margin", 0.0))), tag=v.get("tag", ""))

    def b_clear_x(v, E, V, rec):
        return ClearX(E(v["a"]), E(v["b"]), float(V(v.get("d", 0.0))), tag=v.get("tag", ""))

    def b_contact(v, E, V, rec):
        return ContactKept(E(v["a"]), E(v["b"]), float(V(v.get("d", 0.0))), tag=v.get("tag", ""))

    def b_between(v, E, V, rec):
        return Between(E(v["a"]), E(v["b"]), E(v["c"]), float(V(v.get("margin", 0.0))), tag=v.get("tag", ""))

    def b_over(v, E, V, rec):
        return Over(E(v["a"]), E(v["b"]), float(V(v.get("margin", 0.0))), tag=v.get("tag", ""))

    def b_support_rel(v, E, V, rec):
        return SupportRel(E(v["a"]), E(v["b"]), tag=v.get("tag", ""))

    def b_tilt(v, E, V, rec):
        return TiltToward(E(v["a"]), np.asarray(V(v["dir"]), float), float(V(v["theta"])), tag=v.get("tag", ""))

    def b_until(v, E, V, rec):
        return Until(rec(v["phi"]), rec(v["psi"]), V(v.get("lo", 0)), V(v.get("hi")), V(v.get("at")))

    return {"inside_cyl": b_inside_cyl, "contact": b_contact, "clear_x": b_clear_x, "between": b_between, "over": b_over, "support_rel": b_support_rel, "tilt_toward": b_tilt,
            "until": b_until}


REGISTRY = _builders()


def register(name, builder):
    """builder(v, ent, val, rec) -> Node   (ent: entity resolver, val: parameter resolver, rec: recursive builder)."""
    REGISTRY[name] = builder


def _subst(x, var, name):
    """replace the placeholder string "$var" by `name` everywhere in a nested dict / list."""
    if isinstance(x, str):
        return name if x == "$" + var else x
    if isinstance(x, dict):
        return {k: _subst(v, var, name) for k, v in x.items()}
    if isinstance(x, list):
        return [_subst(v, var, name) for v in x]
    return x


def from_dict(d, ents: dict, params: dict) -> Node:
    """v0 dict forms + the extension atoms. Strings naming an entity are entities, strings naming a parameter are replaced by params."""

    def val(x):
        return params[x] if isinstance(x, str) else x

    def ent(x):
        if isinstance(x, str):
            return ents[x]
        if "shrink" in x:
            return shrink(ent(x["shrink"]["of"]), float(val(x["shrink"]["by"])))
        raise ValueError(x)

    def rec(s):
        return from_dict(s, ents, params)

    (k, v), = d.items()
    if k == "forall":              # {"forall": {"var": "b", "in": ["bead0", ...], "of": spec with "$b" placeholders}} -> And over the entities
        return And(*[rec(_subst(v["of"], v["var"], name)) for name in v["in"]])
    if k in REGISTRY:
        return REGISTRY[k](v, ent, val, rec)
    if k == "and":
        return And(*[rec(c) for c in v])
    if k == "or":
        return ir.Or(*[rec(c) for c in v])
    if k == "not":
        return ir.Not(rec(v))
    if k in ("G", "F"):
        cls = ir.G if k == "G" else ir.F
        return cls(rec(v["of"]), val(v.get("lo", 0)), val(v.get("hi")))
    return ir.from_dict(d, ents, params)          # v0 leaf atoms: inside / upright / clear / support
