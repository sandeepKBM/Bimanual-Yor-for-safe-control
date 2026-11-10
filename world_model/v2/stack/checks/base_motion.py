"""checks/base_motion.py -- rows for a MOVING base (table-behind design 3a, 3c).  SIMULATION ONLY; pure math, no sim imports.

Base kinematics (as checks/base_tip): the arms hold their joints, so every point at r (horizontal, from the rotation centre) moves with the base:
    a(r) = a_b + alpha z x r - w^2 r          (flat floor, no vertical acceleration; Coriolis = 0 because the arms hold)
Segment kinds of route R1 (same Accel boxes as base_tip):  line: |a_b| <= a (disc), alpha = w = 0;  turn: a_b = 0, alpha in +-alpha_max, w^2 in [0, w_max^2].
Acceleration source: Accel.source "measured" (achieved peaks of the base card, max over n) -> row guarantee "sampled"; "command" (the level caps; the
card contradicts that premise) -> "proof" of the model only: reported, never used to admit (as base_tip).

Row `base_torque` (3a).  Per arm: tau = ID_arm(base motion; arm qd = qdd = 0) + J^T h, h = the LEVER split of the object wrench with acc = a - g.  Everything
is AFFINE in mu = (a_bx, a_by, alpha, w^2) for a fixed contents vertex (rne is affine in the base acceleration and in w^2; the object wrench too), so the
backend is only asked for loads at mu = 0 and the 4 unit steps; the maximum of |tau_j| over the line disc is then exact (|t0_j| + a |g_j|) and over the turn box
exact at its 4 corners.  Contents set: vertices (m_lo / m_hi x COM polytope), as checks/payload.  u = max(max_j |tau_j| / tau_lim_j, |F_claw,z| / F_cap)
over both arms; margin = U_ADMIT - u (the payload u definition, unchanged).  The bodies' own spread about their lumped point enters as an upper bound
(`spread`, an |extra torque| per joint added in the worst direction).  NOT covered: base pitch/bounce and the tracker's feedback torque (MEASURED excess
belongs on the base card), joint friction, contact between segments.

Row `base_open_content` (3c, tray of mugs).  Specific force at each mug position (the corners given): f = a - g, |f_xy| / f_z = |a_xy| / g.
Spill |f_xy|/f_z <= 2 (H - h) / D at the fullest fill (open_content.spill_limit / g), slide <= mu_cup (ASSUMED), tip <= D / (2 z_c) with z_c the highest mug COM
over the fill interval.  Quasi-static: slosh, jerk, chatter and creep are NOT covered.  margin = min over the three of (limit - ratio) * g  [m/s^2].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from world_model.v2.stack.checks.base_tip import Accel
from world_model.v2.stack.checks.open_content import spill_limit
from world_model.v2.stack.checks.payload import U_ADMIT
from world_model.v2.stack.checks.row import Row

G = 9.81          # MODEL

GUARANTEE_TORQUE = ("proof under MODEL: rigid weld ASSUMED, quasi-static arms (joint velocities 0 at the hold pose), floating-model inverse dynamics of the arm "
                    "chains under the base motion, lever split of the object wrench (point masses at the lumped COMs + a spread bound), flat level floor, "
                    "no claw rows (user 11-02), the level's acceleration box; base pitch/bounce and the tracker's feedback torque are NOT covered")
GUARANTEE_OPEN = ("proof under MODEL, quasi-static: specific force at the listed positions from the base motion; liquid surface tilt (spill), Coulomb slide, rigid "
                  "mug tip; slosh, jerk, chatter and creep NOT covered")


class BaseLoads(Protocol):
    """physics backend of the torque row (adapters/base_envelope.BaseEnvelope)."""
    arms: tuple
    tau_lim: np.ndarray                 # (7,) per joint [Nm]
    F_cap: float                        # per-claw vertical load cap [N]

    def loads(self, vertex, a_b, alpha, w2) -> dict: ...         # vertex = (m, c object frame) -> {arm: (tau (7,), fz)}
    def spread(self, vertex, alpha, w2) -> dict: ...             # {arm: (7,) extra |tau| bound}, alpha >= 0 magnitude


def _affine(env: BaseLoads, vertex):
    """{arm: (t0 (7,), Gt (7,4), fz0, Gf (4,))} from mu = 0 and the four unit steps (exact: the map is affine)."""
    base = env.loads(vertex, (0.0, 0.0), 0.0, 0.0)
    steps = [env.loads(vertex, (1.0, 0.0), 0.0, 0.0), env.loads(vertex, (0.0, 1.0), 0.0, 0.0),
             env.loads(vertex, (0.0, 0.0), 1.0, 0.0), env.loads(vertex, (0.0, 0.0), 0.0, 1.0)]
    out = {}
    for arm, (t0, f0) in base.items():
        out[arm] = (np.asarray(t0, float),
                    np.stack([np.asarray(s[arm][0], float) - t0 for s in steps], 1),
                    float(f0), np.array([s[arm][1] - f0 for s in steps], float))
    return out


def _vertices(contents):
    """(m, c) pairs of the contents set: masses {m_lo, m_hi} x COM polytope vertices (exact for the affine map, sign of the margin)."""
    return [(float(mm), np.asarray(v, float)) for mm in sorted({contents.m_lo, contents.m_hi}) for v in contents.P]


def torque_row(env: BaseLoads, contents, accs: dict, name="base_torque") -> Row:
    """accs: {"line": Accel, "turn": Accel}.  Worst utilisation over contents vertices x segment kinds x the acceleration box."""
    lim = np.asarray(env.tau_lim, float)
    best, wv = -1.0, None
    for vert in _vertices(contents):
        aff = _affine(env, vert)
        for kind, acc in accs.items():
            if kind == "line":
                cases = [(None, 0.0, 0.0)]
            elif kind == "turn":
                cases = [(None, al, w2) for al in (-acc.alpha, acc.alpha) for w2 in (0.0, acc.w ** 2)]
            else:
                raise ValueError(kind)
            sp = env.spread(vert, 0.0, 0.0) if kind == "line" else None
            for _, al, w2 in cases:
                spr = sp if kind == "line" else env.spread(vert, abs(al), w2)
                for arm, (t0, Gt, f0, Gf) in aff.items():
                    if kind == "line":
                        g = np.hypot(Gt[:, 0], Gt[:, 1])
                        tau = np.abs(t0) + acc.a_lin * g                                  # max over the |a_b| <= a disc
                        fz = abs(f0) + acc.a_lin * float(np.hypot(Gf[0], Gf[1]))
                    else:
                        tau = np.abs(t0 + Gt[:, 2] * al + Gt[:, 3] * w2)
                        fz = abs(f0 + Gf[2] * al + Gf[3] * w2)
                    tau = tau + np.asarray(spr[arm], float)
                    r_t = tau / lim
                    j = int(np.argmax(r_t))
                    u_t, u_f = float(r_t[j]), float(fz / env.F_cap)
                    u = max(u_t, u_f)
                    if u > best:
                        info = dict(kind=kind, arm=arm, binding="torque" if u_t >= u_f else "claw_Fz", joint=j + 1, u_tau=round(u_t, 4),
                                    u_Fz=round(u_f, 4), contents_m=round(vert[0], 3), contents_com_obj=np.round(vert[1], 4).tolist())
                        if kind == "line":
                            gj = Gt[j, :2]
                            sgn = 1.0 if t0[j] >= 0 else -1.0
                            info["a_b"] = (sgn * acc.a_lin * gj / (np.linalg.norm(gj) + 1e-12)).round(3).tolist()
                        else:
                            info.update(alpha=al, w2=round(w2, 4))
                        best, wv = u, info
    ok = bool(best <= U_ADMIT)
    meas = [a for a in accs.values() if a.source == "measured"]
    g = "sampled" if meas else "proof"
    n = min(a.n for a in meas if a.n is not None) if meas and any(a.n is not None for a in meas) else None
    srcs = {k: dict(a_lin=a.a_lin, alpha=a.alpha, w=a.w, source=a.source, src=a.src) for k, a in accs.items()}
    return Row(name, ok, margin=float(U_ADMIT - best), unit="utilisation", guarantee=g, worst=wv, n=n,
               assumptions=dict(model=GUARANTEE_TORQUE, accel=srcs, u_admit=f"MODEL {U_ADMIT} (payload.U_ADMIT)"),
               why="" if ok else f"utilisation {best:.3f} > {U_ADMIT} ({wv['kind']}, arm {wv['arm']}, {wv['binding']}, contents {wv['contents_m']} kg)")


# ---------------------------------------------------------------------------------------------------------------------------
@dataclass
class MugGeom:
    D: float                    # mug diameter [m]
    H: float                    # mug height [m]
    fill_h_max: float           # fullest fill height [m] (spill limit)
    mu: float                   # mug-on-tray friction (ASSUMED, the caller tags it)
    zc_max: float               # highest mug COM above its base over the fill interval [m] (tip limit)
    labels: dict = None


def ratio_limits(mug: MugGeom) -> dict:
    """limits on |f_xy| / f_z."""
    return dict(spill=spill_limit(mug.D, mug.H, mug.fill_h_max) / G, slide=float(mug.mu), tip=mug.D / (2.0 * mug.zc_max))


def planar_accel_max(corners, kind: str, acc: Accel):
    """(max |a_xy| over the acceleration box and the corner positions [m/s^2], binding corner index).  corners: (k, 2) positions from the rotation centre."""
    corners = np.asarray(corners, float)
    if kind == "line":
        return float(acc.a_lin), 0
    best, arg = -1.0, 0
    for i, r in enumerate(corners):
        for al in (-acc.alpha, acc.alpha):
            for w2 in (0.0, acc.w ** 2):
                a = al * np.array([-r[1], r[0]]) - w2 * r
                if np.linalg.norm(a) > best:
                    best, arg = float(np.linalg.norm(a)), i
    return best, arg


def open_content_base_row(mug: MugGeom, corners, accs: dict, name="base_open_content") -> Row:
    lim = ratio_limits(mug)
    worst, wv = np.inf, None
    for kind, acc in accs.items():
        a_h, ci = planar_accel_max(corners, kind, acc)
        ratio = a_h / G                                         # f_z = g: no vertical acceleration (flat floor)
        for term, l in lim.items():
            mg = (l - ratio) * G
            if mg < worst:
                worst, wv = float(mg), dict(kind=kind, term=term, a_xy=round(a_h, 4), ratio=round(ratio, 4), limit=round(l, 4), corner=ci,
                                            limits={k: round(v, 4) for k, v in lim.items()})
    ok = bool(worst >= 0.0)
    meas = [a for a in accs.values() if a.source == "measured"]
    g = "sampled" if meas else "proof"
    n = min(a.n for a in meas if a.n is not None) if meas and any(a.n is not None for a in meas) else None
    srcs = {k: dict(a_lin=a.a_lin, alpha=a.alpha, w=a.w, source=a.source, src=a.src) for k, a in accs.items()}
    return Row(name, ok, margin=float(worst), unit="m/s^2", guarantee=g, worst=wv, n=n,
               assumptions=dict(model=GUARANTEE_OPEN, accel=srcs, corners=np.asarray(corners, float).round(4).tolist(),
                                mug=dict(D=mug.D, H=mug.H, fill_h_max=mug.fill_h_max, mu=mug.mu, zc_max=mug.zc_max), **(mug.labels or {})),
               why="" if ok else f"{wv['term']} limit exceeded ({wv['kind']}): |a_xy|/g {wv['ratio']:.3f} > {wv['limit']:.3f}")
