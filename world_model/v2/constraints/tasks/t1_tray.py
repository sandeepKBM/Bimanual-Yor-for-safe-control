"""Task 1: tray + ball (the v0 worked task), re-expressed on PoseState (ball in the tray frame). Uses ONLY v0 atoms.
spec: G[0,H] ( inside(ball, tray_top) AND upright(tray, theta_max) )     (K = 1: H = 0; K-version: H = K-1)
hand-coded reference: min( sdf_rect(ball xy in tray frame)  [tray_env.sdf_rect],  theta_max - arccos(tray axis z) )."""
from __future__ import annotations

import numpy as np

from world_model.v2.testbeds.b_tray_ball.tray_env import HX, HY, sdf_rect
from world_model.v2.tray_rep.sets import ir
from world_model.v2.tray_rep.sets.ir import Point, prism_xy
from world_model.v2.constraints import ir_ext
from world_model.v2.constraints.common import PoseState, Task, ent, min_rot

THETA = 0.3
SPEC = {"G": {"lo": 0, "hi": "H", "of": {"and": [
    {"inside": {"a": "ball", "b": "tray_top", "margin": 0.0, "tag": "edge"}},
    {"upright": {"a": "tray", "theta_max": "theta_max", "tag": "tilt"}}]}}}
DT = 0.1


def make_build(K):
    def build():
        ents = {"tray": ent("tray", prism_xy(HX, HY)), "tray_top": ent("tray_top", prism_xy(HX, HY)),
                "ball": ent("ball", Point(), parent="tray_top")}
        return ir_ext.from_dict(SPEC, ents, dict(H=K - 1, theta_max=THETA)), ents
    return build


def sample_one(rng, K=1):
    ang = rng.uniform(0, 2 * np.pi)
    tilt = abs(rng.normal(0, 0.25))
    c = np.array([np.sin(tilt) * np.cos(ang), np.sin(tilt) * np.sin(ang), np.cos(tilt)])
    R = min_rot(np.array([0, 0, 1.0]), c)
    loc = np.array([rng.uniform(-HX - 0.06, HX + 0.06), rng.uniform(-HY - 0.06, HY + 0.06), 0.03])
    if K > 1:
        R = np.broadcast_to(R, (K, 3, 3)).copy()
        loc = loc + np.arange(K)[:, None] * rng.normal(0, 0.01, 3)[None] * np.array([1, 1, 0])
        p_tray = np.zeros((K, 3))
    else:
        p_tray = np.zeros(3)
    return PoseState({"tray": (R, p_tray), "tray_top": (R, p_tray), "ball": (np.broadcast_to(np.eye(3), R.shape).copy(), loc)})


def ref(s):
    R, _ = s.poses["tray"]
    _, loc = s.poses["ball"]
    R, loc = np.atleast_3d(R) if R.ndim == 3 else R[None], np.atleast_2d(loc)
    d = np.array([sdf_rect(l[0], l[1], False) for l in loc])
    tilt = THETA - np.arccos(np.clip(R[:, 2, 2], -1, 1))
    return float(np.min(np.minimum(d, tilt)))                    # G over all K ticks, margin at t = 0 = min over the window


def toy(s, rng):
    """u = [a(3), w(3)]: ball slides opposite to a (tau^2/2 gain 0.045), tray axis rate (c' = w x c: dcx = w_y dt, dcy = -w_x dt)."""
    R, p = s.poses["ball"]
    K = p.shape[0] if p.ndim > 1 else 1
    f = np.arange(1, K + 1) / K if K > 1 else np.ones(1)
    Jp = np.zeros((K, 3, 6))
    Jp[:, 0, 0], Jp[:, 1, 1] = -0.045 * f, -0.045 * f
    Jp[:, 0, 4], Jp[:, 1, 3] = 0.01 * f, -0.01 * f
    Jc = np.zeros((K, 2, 6))
    Jc[:, 0, 4], Jc[:, 1, 3] = DT * f, -DT * f
    if K == 1:
        Jp, Jc = Jp[0], Jc[0]
    return {"u0": rng.normal(size=6) * 0.3, "ball": {"p": Jp}, "tray": {"axis": Jc}, "tray_top": {"axis": Jc}}


def make(K=1):
    return Task("tray_ball" + ("" if K == 1 else f"_K{K}"), SPEC, make_build(K), lambda n, rng: [sample_one(rng, K) for _ in range(n)], ref, toy,
                notes=dict(atoms=["inside", "upright", "G", "and"], nonconvex=[], K=K))
