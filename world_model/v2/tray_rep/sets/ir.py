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
    A: Entity
    B: Entity
    d: float = 0.0
    tag: str = ""

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

    def eval(self, state, robust=False):
        ZA, ZB = self._sets(state, robust)
        Z = ZA.minkowski(ZB.neg())
        if Z.batch:
            return np.array([sd_origin(Zonotope(Z.c[k], Z.G[k])) for k in range(Z.c.shape[0])]) - self.d
        return _t(sd_origin(Z) - self.d)

    def rows(self, state, jac, robust=True):
        """fixed-direction linearisation: for the unit direction n pointing from B to A at u0,
        n . (c_A(u) - c_B) >= d + h_{Z - c}(n)  is a valid separation for every u (an inner approximation of Clear)."""
        ZA, ZB = self._sets(state, robust)
        Z = ZA.minkowski(ZB.neg())
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
            brow.append(float(n @ c[k] - nJ @ u0 - self.d - hz))
            tags.append((self.tag or "clear", k, 0))
        return [(np.array(Arows), np.array(brow), tags)]


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
        if unc.pos is None:
            return np.zeros(len(N2))
        z = unc.pos
        if z.batch:
            z = Zonotope(z.c[k], z.G[k])
        N3 = np.concatenate([N2, np.zeros((len(N2), 1))], 1)
        return z._support_many(N3)

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
        {"support": {"a": E, "b": E}}        E := "name" | {"shrink": {"of": E, "by": r}}"""

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
    if k == "clear":
        return Clear(ent(v["a"]), ent(v["b"]), float(val(v.get("d", 0.0))), tag=v.get("tag", ""))
    if k == "support":
        return Support(ent(v["a"]), ent(v["b"]), tag=v.get("tag", ""))
    raise ValueError(f"unknown spec form {k!r}")
