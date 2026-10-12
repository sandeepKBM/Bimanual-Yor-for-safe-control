"""sets/tray_spec.py -- the tray + ball spec in the IR, and the two state adapters that feed it. SIMULATION / SAVED DATA ONLY.

SPEC (declarative; a plain dict, yaml-compatible):
    G[lo, H] ( inside(ball, shrink(tray_top, r_shrink)) AND upright(tray, theta_max) )
`TRAY_SPEC_FULL` is exactly this. `TRAY_SPEC_PAPER` is the literal form from the task text,
    G[0, H] ( inside(ball, shrink(tray_top, r_ball)) )  AND  upright(tray, theta_max).
The ball is a POINT (its centre) when the failure rule is "the ball CENTRE leaves the tray" (the env's drop rule: sdf < 0);
`ball_shape="body"` makes it a sphere of radius r_ball, so that  inside(ball_body, tray_top) == inside(ball_centre, shrink(tray_top, r_ball)).

Two adapters:
    EnvState        a MuJoCo state: tray pose (pos, quat) and the ball's WORLD position; frames are in the world frame.
                    Used for the equivalence check against the hand-coded margin l(x) (tests/test_equiv.py).
    PredState       a predicted trajectory from reach.Prediction: the tray is a world-frame pose per tick (from the plan's axis),
                    the ball is expressed in the TRAY frame (parent = tray_top) with a position-zonotope uncertainty per tick.
"""
from __future__ import annotations

import numpy as np

from world_model.v2.tray_rep import common as K          # noqa: F401
from world_model.v2.testbeds.b_tray_ball.tray_env import HX, HY, M_LO, M_HI
from world_model.v2.tray_rep.sets import ir
from world_model.v2.tray_rep.sets.ir import Entity, Pose, Unc, Point, Sphere, prism_xy
from world_model.v2.tray_rep.sets.zono import Zonotope

R_BALL = 0.03

TRAY_SPEC_FULL = {"G": {"lo": "lo", "hi": "H", "of": {"and": [
    {"inside": {"a": "ball", "b": {"shrink": {"of": "tray_top", "by": "r_shrink"}}, "margin": 0.0, "tag": "edge"}},
    {"upright": {"a": "tray", "theta_max": "theta_max", "tag": "tilt"}}]}}}

TRAY_SPEC_PAPER = {"and": [
    {"G": {"lo": 0, "hi": "H", "of": {"inside": {"a": "ball", "b": {"shrink": {"of": "tray_top", "by": "r_ball"}}, "margin": 0.0}}}},
    {"upright": {"a": "tray", "theta_max": "theta_max"}}]}


def l_from_margin(d):
    """the env's scaled margin from a signed distance: l = clip(d, M_LO, M_HI) / M_HI   (tray_env.margin_from_d)."""
    return np.clip(d, M_LO, M_HI) / M_HI


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rot_from_axis(c):
    """(..., 3) unit axes -> (..., 3, 3): minimal rotation taking z to c (yaw ignored)."""
    c = np.asarray(c, np.float64)
    Kx = np.zeros(c.shape[:-1] + (3, 3))
    Kx[..., 0, 2], Kx[..., 1, 2], Kx[..., 2, 0], Kx[..., 2, 1] = c[..., 0], c[..., 1], -c[..., 0], -c[..., 1]
    return np.eye(3) + Kx + (Kx @ Kx) / (1.0 + c[..., 2])[..., None, None]


def make_entities(frames: dict, ball_shape="centre", r_ball=R_BALL, hx=HX, hy=HY):
    """frames: {"tray": (state->Pose, state->Unc|None), "ball": (...), "ball_parent": None | "tray_top"}.
    The tray and tray_top entities share one pose provider."""
    shape_ball = Point() if ball_shape == "centre" else Sphere(r_ball)
    tray = Entity("tray", prism_xy(hx, hy), frames["tray"][0], frames["tray"][1])
    tray_top = Entity("tray_top", prism_xy(hx, hy), frames["tray"][0], frames["tray"][1])
    ball = Entity("ball", shape_ball, frames["ball"][0], frames["ball"][1], parent=frames.get("ball_parent"))
    return {"tray": tray, "tray_top": tray_top, "ball": ball}


# ====================================================================================== adapter 1: a MuJoCo state
class EnvState:
    def __init__(self, tray_pos, tray_quat, ball_world, unc_ball=None):
        self.p_t, self.R_t = np.asarray(tray_pos, np.float64), quat_to_R(np.asarray(tray_quat, np.float64))
        self.p_b = np.asarray(ball_world, np.float64)
        self.unc_ball = unc_ball


def env_entities(ball_shape="centre"):
    fr = {"tray": (lambda s: Pose(s.R_t, s.p_t), None),
          "ball": (lambda s: Pose(np.eye(3), s.p_b), lambda s: Unc(s.unc_ball, 0.0))}
    return make_entities(fr, ball_shape)


def env_state_from(renv):
    """EnvState of a TrayRepEnv's current MuJoCo state."""
    e = renv.env
    return EnvState(e.pos.copy(), e.quat.copy(), e.d.qpos[e.bq:e.bq + 3].copy())


# ====================================================================================== adapter 2: a predicted trajectory
class PredState:
    """K = H + 1 ticks. tray: world-frame pose from the planned axis. ball: tray-frame position (cx, cy, r_ball) with the
    reachable zonotope as the uncertainty set."""

    def __init__(self, pred, r_ball=R_BALL):
        K_ = len(pred.centre)
        self.R_t = rot_from_axis(pred.C)
        self.p_t = np.zeros((K_, 3))
        self.p_b = np.concatenate([pred.centre, np.full((K_, 1), r_ball)], 1)
        Z = np.zeros((K_, 3, 2))
        Z[:, 0, 0], Z[:, 1, 1] = pred.hw[:, 0], pred.hw[:, 1]
        self.unc_ball = Zonotope(np.zeros((K_, 3)), Z)
        self.pred = pred

    def jac(self):
        """jacobians for ir.rows: ball position (K,3,nu) and tray axis x, y (K,2,nu) w.r.t. u0."""
        pr = self.pred
        K_ = len(pr.centre)
        Jp = np.zeros((K_, 3, 6))
        Jp[:, :2, :] = pr.J
        return {"u0": pr.u0, "ball": {"p": Jp}, "tray": {"axis": pr.Jc}}


def pred_entities(ball_shape="centre"):
    fr = {"tray": (lambda s: Pose(s.R_t, s.p_t), None),
          "ball": (lambda s: Pose(np.broadcast_to(np.eye(3), s.p_b.shape[:-1] + (3, 3)), s.p_b), lambda s: Unc(s.unc_ball, 0.0)),
          "ball_parent": "tray_top"}
    return make_entities(fr, ball_shape)


def build_spec(spec_dict, ents, H, r_shrink=0.0, theta_max=0.3, lo=1, r_ball=R_BALL):
    return ir.from_dict(spec_dict, ents, dict(H=H, r_shrink=r_shrink, theta_max=theta_max, lo=lo, r_ball=r_ball))
