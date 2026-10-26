"""sets/ir.py -- constraint IR v0: entities (sets with poses) + relations with signed-distance semantics + STL-style logic,
compiled to (a) a scalar margin, (b) linearised rows  A u <= b, (c) a three-valued status.   SIMULATION / SAVED DATA ONLY.

GRAMMAR (python objects; `from_dict` builds the same tree from a plain dict / yaml, see tray_spec.py)

    spec    := atom | And(spec, ...) | Or(spec, ...) | Not(spec) | G(spec, lo, hi) | F(spec, lo, hi)
    atom    := Inside(A, B, margin)        A subset of B shrunk by `margin`          B: polytope region
             | Upright(A, theta_max)       tilt of A's z axis from the up vector <= theta_max   [rad]
             | Clear(A, B, d)              distance between A and B >= d
             | Support(A, B)               the centre of mass of A lies in the support polygon (top face footprint) of B
    A, B    := Entity  or  shrink(Entity, r)        (an entity with its polytope eroded by r)

    Entity  := name, shape (local convex set), frame(state) -> Pose, unc(state) -> Unc, parent (None = world)
    Shape   := Point | Sphere(r) | BoxShape(half) | ZonoShape(Z) | Region(Polytope)
    Pose    := (R, p) in the parent frame        Unc := (pos: Zonotope offset of p in the parent frame or None, rot: angle bound [rad])

SEMANTICS. Every atom returns a SIGNED DISTANCE (metres; radians for Upright): > 0 satisfied, < 0 violated.
    Inside   min over the facets i of B of (b_i - margin - h_A(n_i)): the support-function test; if B is box-like (`orthogonal`)
             the outside value is the exact Euclidean distance (so it equals the rectangle SDF).
    Upright  theta_max - angle(R_A z, up)
    Clear    sd(A (+) -B) - d  with sd the signed distance of the origin to the Minkowski difference (zonotopes)
    Support  Inside(COM(A), footprint(B))
    And = min, Or = max, Not = -x; G[lo,hi] = min over the window, F[lo,hi] = max over the window (STL robustness).
ROBUST version (robust=True): every set of A is replaced by its Minkowski sum with the uncertainty set (position zonotope, plus a ball
of radius rot * (reach of A + distance to B) for the pose-angle bound). The robust margin is <= the nominal one.

STATE and TIME. `state` is whatever the providers accept. If the providers return arrays with a leading time axis K (a predicted
trajectory), every atom returns a length-K array and G / F reduce over that axis; with no leading axis K = 1. `margin(state)` is the
robustness at time 0.

EXTENSIONS (2026-10-26, for tamp/spec_cert.py; with every default the existing behaviour is bit-identical)
    Unc.ball       float or (K,) array: radius of a EUCLIDEAN ball added to the position uncertainty (robust sets = A (+) ball). Clear subtracts it exactly
                   (sd(Z (+) B_r) = sd(Z) - r in both signs); Inside / Support add it to the support values. 0 = unchanged.
    Ball(r)        a Sphere that Clear treats exactly (centre point + radius r instead of the outer ball zonotope of Sphere).
    Clear(exact=True, far=)   exact signed distance of the origin to the Minkowski difference (facet slack when inside; foot-point test, else bounded least squares,
                   when outside) instead of the direction-grid sd_origin (a lower bound whose error grows with the SIZE of the sets: ~3 cm on a 0.7 m box).
                   A value above `far` [m] is returned as a cheap lower bound (>= far): far-away obstacles cost O(1).
    ClearEgress    Clear that tolerates a violated start when the motion SEPARATES (class docstring): G[0,T] Clear with an exit clause.

COMPILE TARGETS
    margin(spec, state, robust)           -> float
    status(spec, state)                   -> 'true' (robust margin >= 0) | 'false' (nominal margin < 0) | 'unknown' (otherwise)
    rows(spec, state, jac, robust=True)   -> (A (m, nu), b (m,), tags)   linear rows  A u <= b  (robust, around the linearisation u0)
        jac = {"u0": (nu,), entity_name: {"p": (K,3,nu), "axis": (K,2,nu)}}  Jacobians of the entity's position (in its parent frame)
        and of its axis x, y components w.r.t. the decision vector u. Rows exist for And / G of atoms; Or / F / Not are not convex
        and raise NotConvex.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from world_model.v2.tray_rep.sets.zono import Zonotope, Polytope, sd_origin

UP = np.array([0.0, 0.0, 1.0])


class NotConvex(Exception):
    pass


# ====================================================================================== shapes (local frame)
class Shape:
    def support(self, d):               # d (..., 3) -> (...)
        raise NotImplementedError

    def max_norm(self):
        raise NotImplementedError

    def com(self):
        return np.zeros(3)

    def zonotope(self):
        raise NotImplementedError(type(self).__name__)


class Point(Shape):
    def __init__(self, at=(0.0, 0.0, 0.0)):
        self.at = np.asarray(at, np.float64)

    def support(self, d):
        return np.asarray(d) @ self.at

    def max_norm(self):
        return float(np.linalg.norm(self.at))

    def com(self):
        return self.at

    def zonotope(self):
        return Zonotope(self.at)


class Sphere(Shape):
    def __init__(self, r, centre=(0.0, 0.0, 0.0)):
        self.r, self.centre = float(r), np.asarray(centre, np.float64)

    def support(self, d):
        d = np.asarray(d)
        return d @ self.centre + self.r * np.linalg.norm(d, axis=-1)

    def max_norm(self):
        return float(np.linalg.norm(self.centre)) + self.r

    def com(self):
        return self.centre

    def zonotope(self):
        from world_model.v2.tray_rep.sets.zono import ball_outer
        return ball_outer(self.centre, self.r)


class Ball(Sphere):
    """a Euclidean ball that Clear handles EXACTLY: its zonotope() is the centre POINT and the radius r is subtracted from the signed distance
    (Sphere.zonotope() is an outer polygon/zonotope, ~2-5 % too large in some directions).  Everywhere else (support, max_norm) it is a Sphere."""

    def zonotope(self):
        return Zonotope(self.centre)


class BoxShape(Shape):
    """axis-aligned box in the local frame: centre + [-half, half]; also a Polytope region (6 facets)."""

    def __init__(self, half, centre=(0.0, 0.0, 0.0)):
        self.half, self.centre = np.asarray(half, np.float64), np.asarray(centre, np.float64)
        self.poly = Polytope(np.concatenate([np.eye(3), -np.eye(3)]), np.concatenate([self.centre + self.half, -self.centre + self.half]),
                             orthogonal=True)

    def support(self, d):
        d = np.asarray(d)
        return d @ self.centre + np.abs(d) @ self.half

    def max_norm(self):
        return float(np.linalg.norm(np.abs(self.centre) + self.half))

    def com(self):
        return self.centre

    def zonotope(self):
        return Zonotope(self.centre, np.diag(self.half))

    def polytope(self):
        return self.poly

    def top_face(self):
        """top face (local +z) as a 3-D zonotope (flat)."""
        return Zonotope(self.centre + np.array([0, 0, self.half[2]]), np.diag(self.half * np.array([1, 1, 0])))


class ZonoShape(Shape):
    def __init__(self, Z: Zonotope):
        self.Z = Z

    def support(self, d):
        return self.Z.support(d)

    def max_norm(self):
        return float(np.abs(self.Z.c).max() * 0 + np.linalg.norm(self.Z.c) + np.linalg.norm(self.Z.G, axis=0).sum())

    def com(self):
        return self.Z.c

    def zonotope(self):
        return self.Z


class Region(Shape):
    """a (possibly unbounded) polytope in H-rep, e.g. the footprint prism of the tray top."""

    def __init__(self, poly: Polytope, com=(0.0, 0.0, 0.0)):
        self.poly, self._com = poly, np.asarray(com, np.float64)

    def support(self, d):
        raise NotImplementedError("Region is a container: use it as B in Inside / Support")

    def max_norm(self):
        return np.inf

    def com(self):
        return self._com

    def polytope(self):
        return self.poly

    def shrunk(self, r):
        return Region(self.poly.shrink(r), self._com)


def prism_xy(hx, hy):
    """infinite prism over the rectangle |x| <= hx, |y| <= hy (local), e.g. a tray top; normals in the local x and y only."""
    return Region(Polytope.box([hx, hy, 1.0], axes=[0, 1]))


# ====================================================================================== entities
@dataclass
class Pose:
    R: np.ndarray            # (..., 3, 3)
    p: np.ndarray            # (..., 3)


@dataclass
class Unc:
    pos: Zonotope | None = None      # offset set of the position, in the parent frame (3-D)
    rot: float = 0.0                 # bound on the pose-angle error [rad]
    ball: object = 0.0               # radius [m] of an extra Euclidean ball on the position (float, or (K,) per time index); exact in Clear (module docstring)


@dataclass
class Entity:
    name: str
    shape: Shape
    frame: Callable                  # state -> Pose  (in the parent frame; parent None = world)
    unc: Callable | None = None      # state -> Unc
    parent: str | None = None

    def pose(self, state):
        return self.frame(state)

    def uncertainty(self, state):
        return self.unc(state) if self.unc is not None else Unc()


def shrink(e: Entity, r: float) -> Entity:
    """the entity with its polytope eroded by r (same frame, same uncertainty provider)."""
    sh = e.shape
    if isinstance(sh, BoxShape):
        sh = Region(sh.poly.shrink(r), sh.centre)
    elif isinstance(sh, Region):
        sh = sh.shrunk(r)
    else:
        raise TypeError(f"shrink: {type(sh).__name__} has no H-rep")
    return Entity(e.name, sh, e.frame, e.unc, e.parent)


def _bat(x):
    return np.asarray(x, np.float64)


def _rel(A: Entity, B: Entity, state):
    """pose of A in B's frame, with the uncertainty set of the relative position (a Zonotope in B's frame, or None) and the pose angle bound."""
    PA = A.pose(state)
    uA = A.uncertainty(state)
    if A.parent == B.name:
        return _bat(PA.R), _bat(PA.p), uA.pos, uA.rot
    PB = B.pose(state)
    uB = B.uncertainty(state)
    RB = _bat(PB.R)
    RBt = np.swapaxes(RB, -1, -2)
    R = RBt @ _bat(PA.R)
    p = np.einsum("...ij,...j->...i", RBt, _bat(PA.p) - _bat(PB.p))
    pos = None
    if uA.pos is not None:
        pos = uA.pos.linear(RBt)
    if uB.pos is not None:
        z = uB.pos.neg().linear(RBt)
        pos = z if pos is None else pos.minkowski(z)
    return R, p, pos, uA.rot + uB.rot


def _ball_rel(A: Entity, B: Entity, state):
    """sum of the Unc.ball radii that enter the relative set of A in B's frame (A's, plus B's unless A is expressed in B's frame, as _rel)."""
    b = A.uncertainty(state).ball
    if A.parent != B.name:
        b = b + B.uncertainty(state).ball
    return b


def _ball_col(b):
    """ball radius -> broadcastable against the support values (..., m): scalar stays, a (K,) array gets a trailing axis."""
    b = np.asarray(b, np.float64)
    return b[..., None] if b.ndim else b


def _expand_dirs(N, R):
    """facet normals N (m,3) in B's frame; A's local direction = R^T n.  R (..., 3, 3) -> (..., m, 3)."""
    return np.einsum("...ji,mj->...mi", R, N)


def _h_A(A: Entity, N, R, p, upos, rot, robust):
    """support values h_A(n_i) of A (placed at pose (R, p) in B's frame) for each facet normal n_i: (..., m)."""
    dl = _expand_dirs(N, R)
    sh = A.shape
    if isinstance(sh, Region):
        raise TypeError("A must be a convex body (Point, Sphere, BoxShape, ZonoShape)")
    base = sh.support(dl) + np.einsum("...i,mi->...m", p, N)
    if robust:
        if upos is not None:
            base = base + upos._support_many(N)
        if rot:
            base = base + rot * (sh.max_norm() + np.linalg.norm(p, axis=-1))[..., None]
    return base


# ====================================================================================== exact signed distance of the origin to a 3-D zonotope
_PAIRS = {}


def _pairs(m):
    if m not in _PAIRS:
        _PAIRS[m] = np.triu_indices(m, 1)
    return _PAIRS[m]


def _dist_bvls(c, G):
    """distance of the origin to c + G [-1, 1]^m: min |c + G lam| over the box, active-set (BVLS) -- exact."""
    from scipy.optimize import lsq_linear
    res = lsq_linear(G, -c, bounds=(-1.0, 1.0), method="bvls", tol=1e-13, lsq_solver="exact")
    return float(np.linalg.norm(G @ res.x + c))


def sd_exact(c, G, tol=1e-9, far=np.inf):
    """EXACT signed distance of the origin to the 3-D zonotope Z = c + G [-1,1]^m (> 0 outside: the distance; < 0 inside: minus the distance to the boundary).
    Full-dimensional Z: the facets of Z are among the planes n_ij = g_i x g_j / |.|, so with h(n) = n.c + sum_l |n.g_l| (support function) the origin is inside iff
    min over +-n_ij of h >= 0 and then sd = -min h (the depth to the nearest facet).  Outside, v = -min h is the largest facet violation (a lower bound); if the foot point
    q = -v n of the origin on that facet plane lies in Z (all h(n_j) - n_j.q >= -tol) the distance IS v, otherwise (the nearest feature is an edge or vertex) the exact
    distance comes from the bounded least squares.  Flat / thin Z (rank < 3): bounded least squares (sd >= 0).  far: a largest facet violation >= far is returned as is
    (a lower bound, no least squares)."""
    c = np.asarray(c, np.float64)
    G = np.asarray(G, np.float64)
    G = G[:, np.abs(G).max(0) > 1e-14]
    m = G.shape[1]
    if m == 0:
        return float(np.linalg.norm(c))
    if m < 3:
        return _dist_bvls(c, G)
    S = G @ G.T
    if np.linalg.det(S) <= 1e-12 * (np.trace(S) / 3.0) ** 3:           # rank < 3 (flat / thin)
        return _dist_bvls(c, G)
    i, j = _pairs(m)
    a, b = G[:, i], G[:, j]
    Nn = np.stack([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]], 1)
    nn = np.sqrt((Nn * Nn).sum(1))
    keep = nn > 1e-10 * max(np.trace(S), 1e-12)
    Nn = Nn[keep] / nn[keep][:, None]
    N = np.concatenate([Nn, -Nn])
    h = N @ c + np.abs(N @ G).sum(1)
    k = int(np.argmin(h))
    if h[k] >= 0.0:
        return float(-h[k])
    v = float(-h[k])
    if v >= far:
        return v                                                        # lower bound >= far (largest facet violation)
    q = -v * N[k]
    if np.all(N @ q <= h + tol):
        return v
    return _dist_bvls(c, G)


def sd_exact_batch(Z: Zonotope, far: float = 3.0):
    """sd_exact for a (possibly batched) zonotope -> (K,) array.  Samples whose distance from the origin to the AXIS-ALIGNED interval hull of Z (a lower bound of the
    distance) is already >= far are returned as that lower bound; the others get sd_exact(far=far) (also a lower bound where it is >= far)."""
    c = np.atleast_2d(Z.c)
    G = Z.G if Z.batch else Z.G[None]
    G = np.broadcast_to(G, c.shape[:1] + G.shape[-2:])
    gap = np.maximum(np.abs(c) - np.abs(G).sum(2), 0.0)
    lb = np.sqrt((gap * gap).sum(1))
    out = lb.copy()
    for k in np.nonzero(lb < far)[0]:
        out[k] = sd_exact(c[k], G[k], far=far)
    return out


# ====================================================================================== spec nodes
class Node:
    def eval(self, state, robust=False):           # -> (K,) array
        raise NotImplementedError

    def rows(self, state, jac, robust=True):       # -> list of (A (m,nu), b (m,), tags)
        raise NotImplementedError


def _t(x):
    x = np.asarray(x, np.float64)
    return x if x.ndim else x[None]


@dataclass
class Inside(Node):
    A: Entity
    B: Entity
    margin: float = 0.0
    tag: str = ""

    def _poly(self):
        sh = self.B.shape
        if hasattr(sh, "polytope"):
            return sh.polytope()
        raise TypeError("Inside: B needs an H-rep (BoxShape or Region)")

    def eval(self, state, robust=False):
        P = self._poly()
        R, p, upos, rot = _rel(self.A, self.B, state)
        h = _h_A(self.A, P.N, R, p, upos, rot, robust)
        if robust:
            h = h + _ball_col(_ball_rel(self.A, self.B, state))
        s = P.b - self.margin - h
        return _t(P.margin_from_slacks(s))

    def rows(self, state, jac, robust=True):
        P = self._poly()
        R, p, upos, rot = _rel(self.A, self.B, state)
        if self.A.parent != self.B.name:
            raise ValueError("Inside.rows: A must be expressed in B's frame (A.parent == B.name)")
        Jp = np.asarray(jac[self.A.name]["p"], np.float64)                    # (K, 3, nu) or (3, nu)
        u0 = np.asarray(jac["u0"], np.float64)
        p = np.atleast_2d(p)
        Jp = Jp[None] if Jp.ndim == 2 else Jp
        h_rest = _h_A(self.A, P.N, R, np.zeros_like(p), upos, rot, robust)       # support without the translation
        if robust:
            h_rest = h_rest + _ball_col(_ball_rel(self.A, self.B, state))
        h_rest = h_rest.reshape(len(p), -1)
        nJ = np.einsum("mi,kiu->kmu", P.N, Jp)                                 # (K, m, nu)
        n_p = p @ P.N.T                                                        # (K, m)
        b = P.b[None] - self.margin - n_p - h_rest + np.einsum("kmu,u->km", nJ, u0)
        K, m = b.shape
        tags = [(self.tag or "inside", k, i) for k in range(K) for i in range(m)]
        return [(nJ.reshape(K * m, -1), b.reshape(-1), tags)]


@dataclass
class Upright(Node):
    A: Entity
    theta_max: float
    up: np.ndarray = field(default_factory=lambda: UP.copy())
    n_facets: int = 16
    tag: str = ""

    def _tilt(self, state, robust):
        PA = self.A.pose(state)
        z = _bat(PA.R)[..., :, 2]
        cos = np.clip(z @ self.up, -1.0, 1.0)
        tilt = np.arccos(cos)
        if robust:
            tilt = tilt + self.A.uncertainty(state).rot
        return _t(tilt)

    def eval(self, state, robust=False):
        return self.theta_max - self._tilt(state, robust)

    def rows(self, state, jac, robust=True):
        """inner polygon of the disc |axis_xy| <= sin(theta_max - rot): linear in the axis x, y, hence in u via jac[name]['axis']."""
        if not np.allclose(self.up, UP):
            raise NotImplementedError("Upright.rows: up = +z only")
        rot = self.A.uncertainty(state).rot if robust else 0.0
        z = _bat(self.A.pose(state).R)[..., :, 2]
        z = np.atleast_2d(z)
        Jc = np.asarray(jac[self.A.name]["axis"], np.float64)
        Jc = Jc[None] if Jc.ndim == 2 else Jc
        u0 = np.asarray(jac["u0"], np.float64)
        ang = (np.arange(self.n_facets) + 0.5) * 2 * np.pi / self.n_facets
        N = np.stack([np.cos(ang), np.sin(ang)], 1)
        lim = np.sin(max(self.theta_max - rot, 0.0)) * np.cos(np.pi / self.n_facets)
        nJ = np.einsum("mi,kiu->kmu", N, Jc)
        b = lim - (z[:, :2] @ N.T) + np.einsum("kmu,u->km", nJ, u0)
        K, m = b.shape
        return [(nJ.reshape(K * m, -1), b.reshape(-1), [(self.tag or "upright", k, i) for k in range(K) for i in range(m)])]


@dataclass
class Clear(Node):
    """distance(A, B) >= d.  Default: sd of the origin to the zonotope A (+) -B by the direction-grid sd_origin (unchanged).  exact=True: the exact signed distance
    (sd_exact below; values above `far` are a lower bound).  Ball shapes and Unc.ball are exact in both modes (radius subtracted from the signed distance)."""
    A: Entity
    B: Entity
    d: float = 0.0
    tag: str = ""
    exact: bool = False
    far: float = 3.0

    def _sets(self, state, robust):
        """A and B as zonotopes in the common (world or shared parent) frame."""
        PA, PB = self.A.pose(state), self.B.pose(state)
        ZA = self.A.shape.zonotope().linear(_bat(PA.R), _bat(PA.p))
        ZB = self.B.shape.zonotope().linear(_bat(PB.R), _bat(PB.p))
        if robust:
            uA, uB = self.A.uncertainty(state), self.B.uncertainty(state)
            if uA.pos is not None:
                ZA = ZA.minkowski(uA.pos)
            if uB.pos is not None:
                ZB = ZB.minkowski(uB.pos)
        return ZA, ZB

    def _rho(self, state, robust):
        """total radius of the Ball shapes (+ Unc.ball when robust): float, or (K,) array; 0.0 when there is none."""
        r = 0.0
        for e in (self.A, self.B):
            if isinstance(e.shape, Ball):
                r = r + e.shape.r
            if robust:
                r = r + np.asarray(e.uncertainty(state).ball, np.float64)
        return r

    def _sd(self, Z, far=None):
        if self.exact:
            return sd_exact_batch(Z, self.far if far is None else far)
        if Z.batch:
            return np.array([sd_origin(Zonotope(Z.c[k], Z.G[k])) for k in range(Z.c.shape[0])])
        return _t(sd_origin(Z))

    def _margin(self, state, robust, d, far=None):
        ZA, ZB = self._sets(state, robust)
        out = self._sd(ZA.minkowski(ZB.neg()), far) - d
        rho = self._rho(state, robust)
        if np.any(rho):
            out = out - rho
        return out

    def eval(self, state, robust=False):
        return self._margin(state, robust, self.d)

    def rows(self, state, jac, robust=True):
        """fixed-direction linearisation: for the unit direction n pointing from B to A at u0,
        n . (c_A(u) - c_B) >= d + h_{Z - c}(n)  is a valid separation for every u (an inner approximation of Clear)."""
        ZA, ZB = self._sets(state, robust)
        Z = ZA.minkowski(ZB.neg())
        rho = np.broadcast_to(np.asarray(self._rho(state, robust), np.float64), np.atleast_2d(Z.c).shape[:1])
        c = np.atleast_2d(Z.c)
        G = Z.G if Z.batch else Z.G[None]
        Jp = np.asarray(jac[self.A.name]["p"], np.float64)
        Jp = Jp[None] if Jp.ndim == 2 else Jp
        u0 = np.asarray(jac["u0"], np.float64)
        Arows, brow, tags = [], [], []
        for k in range(len(c)):
            n = c[k] / max(np.linalg.norm(c[k]), 1e-12)
            hz = float(np.abs(G[k].T @ n).sum())
            nJ = n @ Jp[k]
            Arows.append(-nJ)
            brow.append(float(n @ c[k] - nJ @ u0 - self.d - hz - rho[k]))
            tags.append((self.tag or "clear", k, 0))
        return [(np.array(Arows), np.array(brow), tags)]


BIG = 1e3          # "no constraint" margin (checks.py uses the same 1e3)


@dataclass
class ClearEgress(Clear):
    """G[0,T] Clear(A, B, d) with an EXIT CLAUSE for a start inside the keep-out (user decision 2026-10-26; gatekeeper/checks.py "person_exit").
    Let m_k be the (robust) Clear margin and D_k the NOMINAL distance (no uncertainty, Ball radii excluded: only its steps are used) at sample k, k = 0..K-1.
      m_0 >= 0 :  x_k = m_k                                                       (plain Clear)
      m_0 <  0 :  k1 = first k with m_k >= 0 (K if none), up = min(k1, K-1);
                  x_k = D_{k+1} - D_k  for k < up   (A must move AWAY from B at every step until the keep-out is left, the step into k1 included)
                  x_k = m_k            for k >= k1  (from there on the plain row holds)
                  x_{K-1} = BIG        when the keep-out is never left (only the separation is required; K = 1: x_0 = 0).
    Margin = min_k x_k: >= 0 <=> admissible; the units mix m (metres) and metres per sample step (as checks.py's person / person_exit keys). eval() returns x,
    so G[0,T](ClearEgress) is its min.  parts() splits it: {"clear": min m over the plain part, "exit": min step or None}.  Not convex: no rows."""

    def parts(self, state, robust=True):
        m = _t(self._margin(state, robust, self.d))
        K = len(m)
        if m[0] >= 0:
            return dict(m=m, x=m.copy(), clear=float(m.min()), exit=None)
        D = _t(self._margin(state, False, 0.0, far=np.inf))                 # steps of D need the EXACT distance (a `far` lower bound is not smooth)
        nz = np.nonzero(m >= 0)[0]
        k1 = int(nz[0]) if len(nz) else K
        up = min(k1, K - 1)
        x = np.empty(K)
        if up < 1:
            x[:] = BIG
            x[0] = 0.0
            ex = 0.0
        else:
            st = np.diff(D[:up + 1])
            x[:up] = st
            ex = float(st.min())
        x[k1:] = m[k1:]
        if k1 >= K:
            x[K - 1] = BIG if up >= 1 else 0.0
        return dict(m=m, x=x, clear=float(m[k1:].min()) if k1 < K else BIG, exit=ex)

    def eval(self, state, robust=False):
        return self.parts(state, robust)["x"]

    def rows(self, state, jac, robust=True):
        raise NotConvex("ClearEgress: the exit clause is a disjunction over the start sample")


@dataclass
class Support(Node):
    """the centre of mass of A lies inside the support polygon of B (the horizontal footprint of B's top face). B must be a BoxShape."""
    A: Entity
    B: Entity
    tag: str = ""

    def _poly_world(self, state):
        PB = self.B.pose(state)
        R, p = _bat(PB.R), _bat(PB.p)
        face = self.B.shape.top_face()
        Zf = face.linear(R, p)                                  # world top face (flat 3-D zonotope), possibly batched
        return Zf

    def _prep(self, state):
        Zf = self._poly_world(state)
        Zf = Zonotope(np.atleast_2d(Zf.c), Zf.G if Zf.batch else Zf.G[None])
        PA = self.A.pose(state)
        com = np.atleast_2d(np.einsum("...ij,j->...i", _bat(PA.R), self.A.shape.com()) + _bat(PA.p))
        return Zf, com

    def _unc_support(self, unc, N2, k, K):
        """support of A's position-uncertainty set along the 2-D normals N2 (m, 2), at time index k."""
        ball = np.asarray(unc.ball, np.float64)
        ball = float(ball[k]) if ball.ndim else float(ball)
        if unc.pos is None:
            return np.zeros(len(N2)) + ball
        z = unc.pos
        if z.batch:
            z = Zonotope(z.c[k], z.G[k])
        N3 = np.concatenate([N2, np.zeros((len(N2), 1))], 1)
        return z._support_many(N3) + ball

    def eval(self, state, robust=False):
        Zf, com = self._prep(state)
        unc = self.A.uncertainty(state)
        out = []
        for k in range(len(com)):
            P = Zonotope(Zf.c[k][:2], Zf.G[k][:2]).to_polytope_2d()
            s = P.b - P.N @ com[k][:2]
            if robust:
                s = s - self._unc_support(unc, P.N, k, len(com)) - unc.rot * self.A.shape.max_norm()
            out.append(float(s.min()))
        return np.array(out)

    def rows(self, state, jac, robust=True):
        Zf, com = self._prep(state)
        Jp = np.asarray(jac[self.A.name]["p"], np.float64)
        Jp = Jp[None] if Jp.ndim == 2 else Jp
        u0 = np.asarray(jac["u0"], np.float64)
        unc = self.A.uncertainty(state)
        A_all, b_all, tags = [], [], []
        for k in range(len(com)):
            P = Zonotope(Zf.c[k][:2], Zf.G[k][:2]).to_polytope_2d()
            nJ = P.N @ Jp[k][:2]
            extra = (self._unc_support(unc, P.N, k, len(com)) + unc.rot * self.A.shape.max_norm()) if robust else np.zeros(len(P.N))
            for i in range(len(P.N)):
                A_all.append(nJ[i])
                b_all.append(P.b[i] - extra[i] - P.N[i] @ com[k][:2] + nJ[i] @ u0)
                tags.append((self.tag or "support", k, i))
        return [(np.array(A_all), np.array(b_all), tags)]


# ---- logic
class And(Node):
    def __init__(self, *c):
        self.c = list(c)

    def eval(self, state, robust=False):
        return np.min(np.stack([_t(x.eval(state, robust)) for x in self.c]), axis=0)

    def rows(self, state, jac, robust=True):
        return [r for x in self.c for r in x.rows(state, jac, robust)]


class Or(Node):
    def __init__(self, *c):
        self.c = list(c)

    def eval(self, state, robust=False):
        return np.max(np.stack([_t(x.eval(state, robust)) for x in self.c]), axis=0)

    def rows(self, state, jac, robust=True):
        raise NotConvex("Or: the feasible set is a union; choose a disjunct (or branch)")


class Not(Node):
    def __init__(self, c):
        self.c = c

    def eval(self, state, robust=False):
        return -_t(self.c.eval(state, robust))

    def rows(self, state, jac, robust=True):
        raise NotConvex("Not")


class _Temporal(Node):
    reduce = None

    def __init__(self, c, lo=0, hi=None):
        self.c, self.lo, self.hi = c, int(lo), (None if hi is None else int(hi))

    def eval(self, state, robust=False):
        x = _t(self.c.eval(state, robust))
        K = len(x)
        if self.hi is None or self.hi >= K - 1:                 # window runs to the end: one accumulate
            acc = np.minimum.accumulate if self.is_min else np.maximum.accumulate
            suf = acc(x[::-1])[::-1]
            return suf[np.minimum(np.arange(K) + self.lo, K - 1)]
        out = np.empty(K)
        for k in range(K):
            a = min(k + self.lo, K - 1)
            b = min(k + self.hi, K - 1)
            out[k] = self.reduce(x[a:b + 1])
        return out


class G(_Temporal):
    is_min = True
    reduce = staticmethod(np.min)

    def rows(self, state, jac, robust=True):
        """always over the window [lo, hi] (from time 0): rows of the child at every time index in the window."""
        return _window_rows(self.c, state, jac, robust, self.lo, self.hi)


class F(_Temporal):
    is_min = False
    reduce = staticmethod(np.max)

    def rows(self, state, jac, robust=True):
        raise NotConvex("F (eventually) is a disjunction over time")


def _window_rows(child, state, jac, robust, lo, hi):
    rs = child.rows(state, jac, robust)
    out = []
    for A, b, tags in rs:
        keep = np.array([lo <= t[1] <= (10 ** 9 if hi is None else hi) for t in tags])
        out.append((A[keep], b[keep], [t for t, kk in zip(tags, keep) if kk]))
    return out


# ====================================================================================== compile targets
def margin(spec: Node, state, robust=False) -> float:
    """robustness at time 0."""
    return float(_t(spec.eval(state, robust))[0])


def status(spec: Node, state) -> str:
    """'true' | 'false' | 'unknown'. unknown = the nominal margin is positive but the robust margin is negative."""
    nom = margin(spec, state, False)
    if nom < 0:
        return "false"
    rob = margin(spec, state, True)
    return "true" if rob >= 0 else "unknown"


def rows(spec: Node, state, jac, robust=True):
    """-> (A (m, nu), b (m,), tags): rows  A u <= b."""
    parts = spec.rows(state, jac, robust)
    A = np.concatenate([p[0] for p in parts], 0) if parts else np.zeros((0, len(jac["u0"])))
    b = np.concatenate([p[1] for p in parts], 0) if parts else np.zeros(0)
    tags = [t for p in parts for t in p[2]]
    return A, b, tags


# ====================================================================================== declarative form
def from_dict(d, ents: dict, params: dict) -> Node:
    """build a spec from a plain dict (yaml-compatible). Strings that name an entity are entities; strings that name a parameter
    are replaced by params[name]. Forms:
        {"and": [s, ...]} {"or": [...]} {"not": s} {"G": {"lo": 0, "hi": "H", "of": s}} {"F": {...}}
        {"inside": {"a": E, "b": E, "margin": m}} {"upright": {"a": E, "theta_max": x}} {"clear": {"a": E, "b": E, "d": x}}
        {"support": {"a": E, "b": E}}
        {"clear": {"a": E, "b": E, "d": x, "exact": bool, "far": m}} {"clear_egress": {...same...}}     (exact defaults to false for clear, true for clear_egress)
        E := "name" | {"shrink": {"of": E, "by": r}}"""

    def val(x):
        return params[x] if isinstance(x, str) else x

    def ent(x):
        if isinstance(x, str):
            return ents[x]
        if "shrink" in x:
            return shrink(ent(x["shrink"]["of"]), float(val(x["shrink"]["by"])))
        raise ValueError(x)

    (k, v), = d.items()
    if k == "and":
        return And(*[from_dict(c, ents, params) for c in v])
    if k == "or":
        return Or(*[from_dict(c, ents, params) for c in v])
    if k == "not":
        return Not(from_dict(v, ents, params))
    if k in ("G", "F"):
        cls = G if k == "G" else F
        return cls(from_dict(v["of"], ents, params), val(v.get("lo", 0)), val(v.get("hi")))
    if k == "inside":
        return Inside(ent(v["a"]), ent(v["b"]), float(val(v.get("margin", 0.0))), tag=v.get("tag", ""))
    if k == "upright":
        return Upright(ent(v["a"]), float(val(v["theta_max"])), tag=v.get("tag", ""))
    if k in ("clear", "clear_egress"):
        cls = Clear if k == "clear" else ClearEgress
        return cls(ent(v["a"]), ent(v["b"]), float(val(v.get("d", 0.0))), tag=v.get("tag", ""), exact=bool(v.get("exact", k == "clear_egress")),
                   far=float(val(v.get("far", 3.0))))
    if k == "support":
        return Support(ent(v["a"]), ent(v["b"]), tag=v.get("tag", ""))
    raise ValueError(f"unknown spec form {k!r}")
