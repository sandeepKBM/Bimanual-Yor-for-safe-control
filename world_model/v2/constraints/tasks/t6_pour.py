"""Task 6: pouring. A cup (rim radius r = 0.04, height H = 0.10) tilts towards +x; its spout (the rim point at local (r, 0, H)) must stay over
the target container's opening (0.12 x 0.12 m, top at z = 0.05; margin m = 1 cm, and above the rim by m) UNTIL the pour is done (cup tilt
towards +x >= TH_DONE). K = 30 ticks of 0.1 s. Everything in the target frame (target at the origin).
spec:  until( over(spout, target, m),  tilt_toward(cup, +x, TH_DONE) )
Over is a macro of v0 atoms (Inside prism AND Inside above-top); Until and TiltToward are NEW.  With only v0 atoms: Over is expressible, 'done' needs
Not(Upright) (non-convex, no rows) and there is no until (F/G nesting does not express 'phi holds up to the first psi').
hand-coded reference: spout position by closed-form kinematics R_y(theta) [r, 0, H] + cup position; footprint SDF of the rectangle; the STL until
by an explicit double loop."""
from __future__ import annotations

import numpy as np

from world_model.v2.tray_rep.sets.ir import BoxShape, Point
from world_model.v2.tray_rep.sets.zono import Zonotope
from world_model.v2.tray_rep.sets.ir import Unc
from world_model.v2.constraints import ir_ext
from world_model.v2.constraints.common import PoseState, Task, ent, min_rot

R_CUP, H_CUP = 0.04, 0.10
HX_T, HY_T, HZ_T = 0.06, 0.06, 0.05
M_OVER, TH_DONE = 0.01, 1.0
K = 30
DT = 0.1
SPEC = {"until": {"phi": {"over": {"a": "spout", "b": "target", "margin": "m", "tag": "over"}},
                  "psi": {"tilt_toward": {"a": "cup", "dir": "e_dir", "theta": "th_done", "tag": "done"}}}}
SPEC_OVER_ONLY = {"G": {"lo": 0, "hi": None, "of": {"over": {"a": "spout", "b": "target", "margin": "m"}}}}


def build_with(spec):
    def build():
        ents = {"target": ent("target", BoxShape((HX_T, HY_T, HZ_T)), with_unc=False),
                "spout": ent("spout", Point(), parent="target"), "cup": ent("cup", Point(), parent="target")}
        return ir_ext.from_dict(spec, ents, dict(m=M_OVER, th_done=TH_DONE, e_dir=[1.0, 0.0, 0.0])), ents
    return build


def Ry(th):
    c, s = np.cos(th), np.sin(th)
    R = np.zeros(th.shape + (3, 3))
    R[..., 0, 0], R[..., 0, 2], R[..., 1, 1], R[..., 2, 0], R[..., 2, 2] = c, s, 1.0, -s, c
    return R


def sample(n, rng):
    S = []
    t = np.arange(K) / (K - 1)
    for _ in range(n):
        th_f = rng.uniform(0.95, 1.5)
        th = th_f * (3 * t ** 2 - 2 * t ** 3)                                # smooth ramp
        s0 = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), rng.uniform(0.075, 0.13)])
        vel = rng.normal(0, 0.015, 3)
        spout = s0 + t[:, None] * vel
        R = Ry(th)
        off = np.array([R_CUP, 0.0, H_CUP])
        pcup = spout - R @ off
        unc = Unc(Zonotope(np.zeros((K, 3)), np.broadcast_to(np.diag([0.003, 0.003, 0.003]), (K, 3, 3)).copy()), 0.0)
        S.append(PoseState({"target": (np.broadcast_to(np.eye(3), (K, 3, 3)).copy(), np.zeros((K, 3))),
                            "spout": (np.broadcast_to(np.eye(3), (K, 3, 3)).copy(), spout), "cup": (R, pcup)}, unc={"spout": unc},
                           extra=dict(theta=th, off=off)))
    return S


def rect_sdf(x, y, hx, hy):
    qx, qy = abs(x) - hx, abs(y) - hy
    return -(np.hypot(max(qx, 0), max(qy, 0))) if (qx > 0 or qy > 0) else -max(qx, qy)


def over_margin(p):
    """footprint SDF (shrunk by m) and height above the top (minus m)."""
    xy = rect_sdf(p[0], p[1], HX_T - M_OVER, HY_T - M_OVER)
    return min(xy, p[2] - HZ_T - M_OVER)


def phi_psi(s):
    th = s.extra["theta"]
    spout = np.array([R_CUP * np.cos(th) + H_CUP * np.sin(th), np.zeros_like(th), -R_CUP * np.sin(th) + H_CUP * np.cos(th)]).T
    spout = spout + (s.poses["cup"][1])                      # cup position + closed-form lip offset
    phi = np.array([over_margin(p) for p in spout])
    psi = np.arcsin(np.sin(th)) - TH_DONE
    return phi, psi


def ref(s):
    phi, psi = phi_psi(s)
    best, run = -np.inf, np.inf
    for t in range(K):
        run = min(run, phi[t])
        best = max(best, min(psi[t], run))
    return float(best)


def ref_phi_only(s):
    return float(phi_psi(s)[0].min())


def toy(s, rng):
    """u = [a(3), w(3)] (cup acceleration, tilt rate about y): numeric-kinematics Jacobians of the cup position / tilt, spout position."""
    th = s.extra["theta"]
    t_ = (np.arange(K) + 1) * DT
    Jp_c = np.zeros((K, 3, 6))
    for i in range(3):
        Jp_c[:, i, i] = 0.5 * t_ ** 2
    dth = np.zeros((K, 6))
    dth[:, 4] = t_
    dR = np.stack([-R_CUP * np.sin(th) + H_CUP * np.cos(th), np.zeros(K), -R_CUP * np.cos(th) - H_CUP * np.sin(th)], 1)    # d(R off)/dth
    Jp_s = Jp_c + dR[:, :, None] * dth[:, None, :]
    Jc = np.zeros((K, 2, 6))
    Jc[:, 0, :] = np.cos(th)[:, None] * dth                          # c = (sin th, 0, cos th)
    return {"u0": rng.normal(size=6) * 0.3, "spout": {"p": Jp_s}, "cup": {"p": Jp_c, "axis": Jc}}


def make(spec=SPEC):
    return Task("pour_until" if spec is SPEC else "pour_over_only", spec, build_with(spec), sample, ref if spec is SPEC else ref_phi_only, toy,
                notes=dict(atoms=["over(macro)", "until(new)", "tilt_toward(new)"], nonconvex=["until: disjunction over switch time", "done = outside-disc (v0) / halfspace (TiltToward)"]))
