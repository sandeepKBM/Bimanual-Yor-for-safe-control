"""checks/base_tip.py -- base tip-over row (table-behind design 3e).  SIMULATION ONLY; pure math, no sim imports.

The zero-moment point (ZMP: where the resultant floor force acts) must stay inside the wheel contact polygon.  Point-mass model in the base
frame B (origin = rotation centre on the floor, x forward, z up); the arms hold, so every mass moves with the base:
    a_i = a_b + alpha z x r_i - w^2 r_i                       (r_i = horizontal position of mass i; flat floor, no vertical acceleration)
    ZMP = c_xy - (1 / (M g)) sum_i m_i z_i a_i,xy              (M = total mass, c = total COM)
        = c_xy - (S_z a_b + alpha z x S_zr - w^2 S_zr) / (M g),  S_z = sum m_i z_i,  S_zr = sum m_i z_i r_i
Segment kinds (route R1):  line: |a_b| <= a (any direction: worst per edge in closed form), alpha = w = 0;
                           turn: a_b = 0 (centre held), alpha in [-alpha_max, alpha_max], w^2 in [0, w_max^2] (4 corners, affine).
Contents: the ObjectSpec contents set (masses {m_lo, m_hi} x COM polytope vertices, object frame).  M * margin is affine in m and multilinear
in the COM coordinates over a prism, so the SIGN of the margin is exact at those vertices (the reported margin value is the minimum over them).
Guarantee: the ZMP test is exact under MODEL (rigid bodies as point masses at their COMs (the bodies' own rotational inertia is left out), flat level floor,
all wheels on the floor at the start, arms rigid at the hold pose, the acceleration box of the level).  Which acceleration box: the row records
its source ("command" = the level caps; "measured" = the achieved peaks of the base card, MEASURED with the card's n and filter).
Row guarantee (AGENTS.md rules 6-7): "measured" -> "sampled" (the acceleration box is a max over n runs, not a bound; n in Row.n);
"command" -> "proof" only of the model with the commanded box, whose premise the base card CONTRADICTS (achieved > command at every level):
never admit on it; it is reported for comparison.
Margin unit: m (signed distance of the ZMP to the polygon edge, > 0 inside).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from world_model.v2.stack.checks.row import Row

G = 9.81
GUARANTEE = ("proof under MODEL: point masses at the body COMs (body rotational inertia left out), flat level floor, all wheels down at the "
             "start, arms rigid at the hold pose, the level's acceleration box; transients beyond the box are NOT covered")


@dataclass
class Polygon:
    """axis-aligned contact rectangle in B [m] (MODEL: the inner wheel contact points over the steer angles of the segment kind)."""
    x_lo: float
    x_hi: float
    y_lo: float
    y_hi: float

    def edges(self):
        """(outward normal, offset d): inside iff n . p <= d."""
        return ((np.array([1.0, 0.0]), self.x_hi), (np.array([-1.0, 0.0]), -self.x_lo),
                (np.array([0.0, 1.0]), self.y_hi), (np.array([0.0, -1.0]), -self.y_lo))


@dataclass
class Accel:
    """acceleration box of one level for one segment kind."""
    a_lin: float = 0.0          # m/s^2, bound on |a_b| (line)
    alpha: float = 0.0          # rad/s^2 (turn)
    w: float = 0.0              # rad/s (turn)
    source: str = "command"     # "command" | "measured"
    src: str = ""
    n: int | None = None        # measured: number of runs behind the max


@dataclass
class MassModel:
    """known masses in B: (m, xyz) rows (robot bodies + the empty object), and the object frame pose in B (contents are added per vertex)."""
    m: np.ndarray
    p: np.ndarray               # (k, 3)
    obj_R: np.ndarray           # (3, 3) object frame -> B
    obj_p: np.ndarray           # (3,)
    labels: dict = field(default_factory=dict)


def _sums(m, p):
    M = float(m.sum())
    c = (m[:, None] * p).sum(0) / M
    Sz = float((m * p[:, 2]).sum())
    Szr = (m[:, None] * p[:, 2:3] * p[:, :2]).sum(0)
    return M, c, Sz, Szr


def zmp_margin(m, p, poly: Polygon, kind: str, acc: Accel):
    """(margin [m], worst case dict) of the ZMP over the acceleration box of `kind`."""
    M, c, Sz, Szr = _sums(m, p)
    best, arg = np.inf, None
    for n, d in poly.edges():
        base = d - float(n @ c[:2])
        if kind == "line":
            mg = base - Sz * acc.a_lin / (M * G)                   # worst a_b = -a n (closed form over the disc)
            cases = [(mg, dict(a_b=(-acc.a_lin * n).round(3).tolist()))]
        elif kind == "turn":
            cases = []
            for al in (-acc.alpha, acc.alpha):
                for w2 in (0.0, acc.w ** 2):
                    zx = np.array([-Szr[1], Szr[0]])               # z x S_zr
                    shift = -(al * zx - w2 * Szr) / (M * G)        # ZMP - c
                    cases.append((base - float(n @ shift), dict(alpha=al, w2=round(w2, 4))))
        else:
            raise ValueError(kind)
        for mg, info in cases:
            if mg < best:
                best, arg = float(mg), dict(info, edge=n.astype(int).tolist(), com_xy=c[:2].round(4).tolist(), com_z=round(float(c[2]), 4), M=round(M, 3))
    return best, arg


def tip_row(model: MassModel, contents, polys: dict, accs: dict, name="base_tip") -> Row:
    """polys / accs: {kind: Polygon / Accel} for the segment kinds of the route.  contents: objects.Contents (m_lo, m_hi, P in the object frame)."""
    worst, wv = np.inf, None
    for mm in sorted({contents.m_lo, contents.m_hi}):
        for v in contents.P:
            pc = model.obj_R @ np.asarray(v, float) + model.obj_p
            m = np.append(model.m, mm)
            p = np.vstack([model.p, pc])
            for kind, poly in polys.items():
                mg, info = zmp_margin(m, p, poly, kind, accs[kind])
                if mg < worst:
                    worst, wv = mg, dict(info, kind=kind, contents_m=round(float(mm), 3), contents_com_obj=np.round(v, 4).tolist())
    ok = bool(worst >= 0.0)
    srcs = {k: dict(a_lin=a.a_lin, alpha=a.alpha, w=a.w, source=a.source, src=a.src) for k, a in accs.items()}
    meas = [a for a in accs.values() if a.source == "measured"]
    g = "sampled" if meas else "proof"
    n = min(a.n for a in meas if a.n is not None) if meas and any(a.n is not None for a in meas) else None
    return Row(name, ok, margin=float(worst), unit="m", guarantee=g, worst=wv, n=n,
               assumptions=dict(model=GUARANTEE, accel=srcs, polygons={k: vars(p) for k, p in polys.items()}, **model.labels),
               why="" if ok else f"ZMP leaves the wheel polygon by {-worst * 1e3:.1f} mm ({wv['kind']}, contents {wv['contents_m']} kg)")


def a_tip(model: MassModel, contents_m: float, contents_obj_xyz, poly: Polygon) -> float:
    """largest |a_b| (line, worst direction) with the ZMP inside: (min edge distance of the COM) * M g / S_z.  For reports."""
    pc = model.obj_R @ np.asarray(contents_obj_xyz, float) + model.obj_p
    M, c, Sz, _ = _sums(np.append(model.m, contents_m), np.vstack([model.p, pc]))
    base = min(d - float(n @ c[:2]) for n, d in poly.edges())
    return float(base * M * G / Sz)


def pick_level(model: MassModel, contents, polys: dict, accs_by_level: dict):
    """(highest level whose tip row passes, {level: Row}); levels in increasing order of accs_by_level.  None = no level passes -> refuse.
    The row is monotone in the level when every acceleration bound grows with it (it does for the LEVELS grid and the measured card)."""
    rows, best = {}, None
    for L, accs in accs_by_level.items():
        rows[L] = tip_row(model, contents, polys, accs)
        if rows[L].ok:
            best = L
    return best, rows
