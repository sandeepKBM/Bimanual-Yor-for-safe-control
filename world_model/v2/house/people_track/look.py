"""people_track/look.py -- look-before-move: an in-place yaw "glance" so the forward ZED sees the space the base is about to enter.  SIMULATION ONLY.

Real sensors only: ONE forward head/torso camera on the fixed mount (no rear / side cameras).  The camera rides the base root, so the only way to look
sideways is to yaw the base.  The holonomic base can yaw while the tray is held; the glance is a body-twist command (vx = vy = 0, wz = +-wz_glance)
that goes THROUGH the shield's QP like every other command, so the cup slip / jerk budget (and the wall / person keep-out) still hold.

Two triggers (both need the target bearing within max_glance_deg of the camera axis but outside the FOV; a bearing behind the robot gets the speed cap only):
  (a) UNSEEN NEAR FIELD the operator commands translation (|v| > v_min).  Free cells of the ObservedMap within r_max of the base are grouped into
                        sector_deg-wide sectors of the camera's front half-plane.  A sector OUTSIDE the current FOV that holds >= min_pts stale free samples
                        (not inside the camera view for t_look s) is glanced at (the sector with the highest proximity-weighted stale score; sectors within
                        +-60 deg of the travel direction weigh 1.5x; only sectors within travel_deg of the travel direction; none near a dock).  A person walking in from the side is only ever seen if the camera has looked there.
  (b) LOST TRACK        a confirmed track that is coasting (no update for > t_lost) and whose constant-velocity prediction (t_pred s ahead) is within r_cross
                        of the base and outside the FOV: glance toward the predicted position (a person we saw crossing and lost is probably still there).
A glance runs until the bearing is inside (FOV half-angle - margin) or max_glance_s, then `cooldown_s` without a new glance.  While glancing, translation
is zero.  Everything is OFF unless LookCfg.enabled (PerceivedShield.look is None by default -> bit-identical).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from world_model.v2.house.teleop import geom as GM
from world_model.v2.house.teleop.sim import Cmd, TICK


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class LookCfg:
    enabled: bool = True
    t_look: float = 4.0              # s, a free cell not inside the view for longer than this is 'unseen'
    v_min: float = 0.2               # m/s commanded translation below which no scan glance is considered
    travel_deg: float = 75.0         # only sectors within this angle of the travel direction (the space the base is about to enter)
    dock_m: float = 1.2              # no scan glance within this distance of a dock pose (manipulation stand-off: the operator is trimming, not travelling)
    sector_deg: float = 20.0
    r_min: float = 0.8
    r_max: float = 3.0
    n_r: int = 8
    min_pts: int = 4                 # stale free samples (of n_r) a sector needs
    travel_w: float = 1.5            # weight of sectors within +-60 deg of the travel direction
    wz_glance: float = 0.7           # rad/s (stick max 1.2); the shield QP enforces the cup budget on top
    fov_margin_deg: float = 8.0
    max_glance_deg: float = 100.0
    max_glance_s: float = 1.6
    cooldown_s: float = 1.0          # s, minimum pause between glances
    duty: float = 0.25               # max fraction of time spent glancing: pause >= glance_duration * (1 / duty - 1)
    # trigger (b)
    lost_track: bool = True
    t_lost: float = 0.3              # s without a measurement update
    t_pred: float = 1.0              # s
    r_cross: float = 3.0             # m
    corridor: bool = True            # trigger (a) (named 'corridor' for the stats)


def ages(om, pts, t):
    """age [s] of the last time each point was inside the camera view (nan: occupied / outside the map -> not counted)."""
    edt = om.occ.lookup(pts[:, 0], pts[:, 1])
    free = edt > 0.15
    i, j = om._ij(pts[:, 0], pts[:, 1])
    ok = free & (i >= 0) & (i < om.ny) & (j >= 0) & (j < om.nx)
    a = np.full(len(pts), np.nan)
    a[ok] = t - om.t[i[ok], j[ok]]
    return a


class Looker:
    def __init__(self, perc, cfg: LookCfg | None = None):
        self.perc = perc
        self.cfg = cfg or LookCfg()
        self.active = False
        self.target = None                 # world bearing of the glance target [rad]
        self.t0 = 0.0
        self.cool_until = -1.0
        self.n_glances = 0
        self.glance_ticks = 0
        self.n_trig_corridor = 0
        self.n_trig_lost = 0
        self.last_trigger = None
        self.last_side = 0                 # sign of the last glance direction (+1 left / -1 right)
        self.side_bonus = 1.5

    # ------------------------------------------------------------------ geometry
    def _cam(self):
        o, R = self.perc.cam.pose()
        yaw_cam = math.atan2(-R[1, 2], -R[0, 2])
        return o, yaw_cam

    def _half(self):
        return 0.5 * self.perc.hfov - math.radians(self.cfg.fov_margin_deg)

    def _bearing(self, o, p):
        return math.atan2(p[1] - o[1], p[0] - o[0])

    # ------------------------------------------------------------------ triggers
    def _corridor_target(self, sim, twist, o, yaw_cam, docks=None):
        c = self.cfg
        om = self.perc.omap
        vxy = np.asarray(twist[:2], float)
        sp = float(np.linalg.norm(vxy))
        if om is None or sp < c.v_min:
            return None
        pose = sim.ctrl_pose() if getattr(sim, "loc", None) is not None else sim.pose()
        if any(math.hypot(pose[0] - x, pose[1] - y) < c.dock_m for x, y in (docks or ())):
            return None
        Rm = GM.rot(pose[2])
        dir_w = Rm @ (vxy / sp)
        th_travel = math.atan2(dir_w[1], dir_w[0])
        half = self._half()
        centers = np.radians(np.arange(-c.max_glance_deg, c.max_glance_deg + 1e-9, c.sector_deg))
        rr = np.linspace(c.r_min, c.r_max, c.n_r)
        best, best_s = None, 0.0
        for dlt in centers:
            if abs(dlt) <= half:                 # already inside the FOV: the next frame stamps it
                continue
            ang = yaw_cam + dlt
            if abs(wrap(ang - th_travel)) > math.radians(c.travel_deg):
                continue
            sub = ang + np.radians(np.array([-0.3, 0.0, 0.3]) * c.sector_deg)       # 3 rays per sector, 8 radii each
            pts = np.stack([o[0] + np.outer(np.cos(sub), rr), o[1] + np.outer(np.sin(sub), rr)], -1).reshape(-1, 2)
            a = ages(om, pts, sim.t)
            ok = np.isfinite(a)
            stale = ok & (a > c.t_look)
            if stale.sum() < c.min_pts * 3 // 2:
                continue
            w = np.tile(1.0 / (0.5 + rr), 3)
            sc = float((w * stale).sum()) * (c.travel_w if abs(wrap(ang - th_travel)) < math.radians(60) else 1.0)
            if self.last_side != 0 and np.sign(dlt) == -self.last_side:       # alternate sides: after a left glance prefer the right one
                sc *= self.side_bonus
            if sc > best_s:
                best, best_s = ang, sc
        return best

    def _lost_target(self, sim, o):
        c = self.cfg
        k = sim.k
        pose = sim.ctrl_pose() if getattr(sim, "loc", None) is not None else sim.pose()
        best = None
        for t in self.perc.tracks:
            if (k - t.last_upd) * TICK <= c.t_lost:
                continue
            pp = t.x[:2] + c.t_pred * t.x[2:]
            dd, _, _ = GM.rect_dist_point(pp, pose)
            if dd > c.r_cross:
                continue
            if best is None or dd < best[0]:
                best = (dd, pp)
        return None if best is None else self._bearing(o, best[1])

    # ------------------------------------------------------------------ one tick
    def apply(self, sim, cmd: Cmd, docks=None) -> Cmd:
        c = self.cfg
        if not c.enabled:
            return cmd
        o, yaw_cam = self._cam()
        half = self._half()
        t = sim.t
        if self.active:
            delta = wrap(self.target - yaw_cam)
            if abs(delta) <= half or t - self.t0 >= c.max_glance_s:
                self.active = False
                self.cool_until = t + max(c.cooldown_s, (t - self.t0) * (1.0 / c.duty - 1.0))
        if not self.active and t >= self.cool_until:
            tgt, why = None, None
            if c.lost_track:
                tgt = self._lost_target(sim, o)
                why = "lost"
            if tgt is None and c.corridor:
                tgt = self._corridor_target(sim, cmd.twist, o, yaw_cam, docks)
                why = "corridor"
            if tgt is not None:
                delta = wrap(tgt - yaw_cam)
                if half < abs(delta) <= math.radians(c.max_glance_deg):
                    self.active, self.target, self.t0, self.last_trigger = True, tgt, t, why
                    self.last_side = 1 if delta > 0 else -1
                    self.n_glances += 1
                    if why == "lost":
                        self.n_trig_lost += 1
                    else:
                        self.n_trig_corridor += 1
        if not self.active:
            return cmd
        self.glance_ticks += 1
        delta = wrap(self.target - yaw_cam)
        tw = np.array([0.0, 0.0, math.copysign(c.wz_glance, delta)])
        return Cmd(twist=tw, lift_v=cmd.lift_v, jog=cmd.jog, grasp=cmd.grasp, release=cmd.release)

    def stats(self):
        return dict(n_glances=self.n_glances, glance_s=self.glance_ticks * TICK, n_trig_corridor=self.n_trig_corridor, n_trig_lost=self.n_trig_lost)
