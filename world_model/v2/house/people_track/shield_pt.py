"""people_track/shield_pt.py -- the teleop Shield fed by the tracker, with an uncertainty-inflated person keep-out.  SIMULATION ONLY.

PerceivedShield(Shield) -- a subclass in THIS package; teleop/shield.py is not edited.  `filter(sim, cmd)` has the Shield signature, so it drops in
wherever `Shield` is used (teleop/run.py builds its own Shield; people_track/run_pt.py is the loop that builds this one).

  1. perception.step(sim): detections -> tracks (see perception.py)
  2. unseen-space speed cap (optional, `cfg.unseen_cap`): see perception.py
  3. Shield.filter on a SimProxy (tracker-derived people_state / scene.people; every other attribute is the real sim)
  4. keep-out rows (override of Shield._keep_rows): walls unchanged (super() on the people-free proxy); each TRACK gets its own disc radius
         r_i = PERSON_CLEAR_R + d_buffer + k_sigma * h_i(n),     (d_buffer = 0: the original fixed d_person = 0.10 m stood in for tracking error and is replaced by h_i)     h_i(n) = sqrt( n' Sigma_T n ),   n = unit vector rectangle -> person,
         Sigma_T = position block of  [I, T_h I] P [I, T_h I]'   (the track's position covariance propagated over the reaction horizon T_h, i.e.
         P_pp + T_h (P_pv + P_vp) + T_h^2 P_vv ),     r_i capped at `max_infl` above the nominal radius.
     DOCUMENTED CHOICE (disc vs zonotope): the shield's person row is the support function of (footprint-rectangle  (+)  person set) along n.  For
     the Minkowski sum  disc(r) (+) k*ellipse(Sigma_T)  the support function is exactly  r + k sqrt(n' Sigma_T n)  (support functions add under
     Minkowski sum), so inflating the disc radius by the ellipse support h_i(n) in the row direction IS the Minkowski-sum constraint, with no
     zonotope outer approximation (a zonotope enclosing the ellipse -- tray_rep/sets/zono.py -- would only loosen it).  The mean velocity of the
     track enters the row exactly as in the original (relative-velocity term n.v_person).
     The row is otherwise the original (CBF + braking-distance bound), so with k_sigma = 0 and a truth-like track it reduces to the original.
Reported in info: n_tracks, max inflation, unseen-cap flag.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from world_model.v2.house.teleop import geom as GM, base as BS
from world_model.v2.house.teleop.shield import Shield, ShieldCfg
from world_model.v2.house.teleop.sim import Cmd, TICK


@dataclass
class UncertaintyCfg:
    k_sigma: float = 2.0             # confidence multiplier on the horizon-propagated position std (~95% in 1-D)
    t_horizon: float = 0.5           # s, velocity-uncertainty horizon (the shield's CBF + braking bound already budgets the stop; this prices a wrong velocity)
    d_buffer: float = 0.0            # m, fixed extra buffer (the original shield's d_person = 0.10 stood in for tracking error; the measured sigma replaces it)
    max_infl: float = 0.5            # m, cap on the extra radius
    unseen_cap: bool = True
    v_unseen: float = 0.25           # m/s cap on the translational speed toward unobserved space
    t_stale: float = 3.0             # s
    frac_unseen: float = 0.4
    include_tentative: bool = False
    soft_floor: float | None = None  # m: cap the inflation so the row's margin h >= soft_floor while the robot is outside the nominal disc (an uncertainty margin may slow the
                                     # approach but never creates an instantaneous violation -> no uncertainty-induced emergency brake at the hardware limit / cup slip)
    conformal: object = None         # ConformalMargin: replaces k_sigma * sqrt(n' Sigma n) by the calibrated (1 - alpha) horizon-error quantile of the track's (visibility, age) bin


class PerceivedShield(Shield):
    def __init__(self, perception, cfg: ShieldCfg | None = None, occ=None, a_hw=BS.A_HW, ucfg: UncertaintyCfg | None = None):
        super().__init__(cfg, occ=occ, a_hw=a_hw)
        self.perc = perception
        self.ucfg = ucfg or UncertaintyCfg()
        self.last_infl = 0.0
        self.max_infl_seen = 0.0
        self.n_unseen_cap = 0
        self.look = None                 # look.Looker (look-before-move glance); None = off (bit-identical)

    # ------------------------------------------------------------------ keep-out with per-track inflation
    def _person_rows(self, sim, relax=False):
        cfg, u = self.cfg, self.ucfg
        rows, b, tags = [], [], []
        pose = sim.pose()
        R = GM.rot(pose[2])
        a_b = cfg.a_brk_frac * cfg.a_lim_flat
        T = u.t_horizon
        Fh = np.hstack([np.eye(2), T * np.eye(2)])
        if not relax:
            self.last_infl = 0.0
        for ti in sim.track_info:
            xy, vel, P = ti["xy"], ti["v"], ti["P"]
            dist, qn, nw = GM.rect_dist_point(xy, pose)
            Sig = Fh @ P @ Fh.T
            if u.conformal is not None:
                infl = u.conformal.radius(ti.get("since_upd_s", 0.0), ti.get("age_s", 0.0), ti.get("range_m"))
            else:
                infl = min(u.k_sigma * math.sqrt(max(float(nw @ Sig @ nw), 0.0)), u.max_infl)
            if u.soft_floor is not None:
                infl = min(infl, max(0.0, dist - (GM.PERSON_CLEAR_R + u.d_buffer) - u.soft_floor))
            if relax:                                              # shield cfg.relax_tier: no fixed buffer, half the covariance inflation
                infl = 0.5 * infl
            hi = dist - (GM.PERSON_CLEAR_R + (0.0 if relax else u.d_buffer) + infl)
            if dist - (GM.PERSON_CLEAR_R + cfg.d_person) > 2.0:        # same range cut as the original row
                continue
            if not relax:
                self.last_infl = max(self.last_infl, infl)
            nb = R.T @ nw
            row = np.array([nb[0], nb[1], -qn[1] * nb[0] + qn[0] * nb[1]])
            beta = min(cfg.gamma * hi, math.sqrt(2 * a_b * hi)) if hi >= 0 else max(cfg.gamma * hi, -0.3)
            rows.append(row); b.append(beta + float(nw @ vel)); tags.append(f"person:{ti['name']}")
        if not relax:
            self.max_infl_seen = max(self.max_infl_seen, self.last_infl)
        return rows, b, tags

    def _keep_rows(self, sim, v, relax=False):
        if not hasattr(sim, "track_info"):                       # plain sim: original behaviour
            return super()._keep_rows(sim, v, relax)
        A, bb, tags = super()._keep_rows(sim.without_people(), v, relax)        # walls only
        pr, pb, pt = self._person_rows(sim, relax)
        if pr:
            A = np.vstack([A.reshape(-1, 3), np.array(pr)])
            bb = np.concatenate([bb, np.array(pb)])
            tags = list(tags) + pt
        return A.reshape(-1, 3), bb, tags

    # ------------------------------------------------------------------ unseen-space speed cap
    def _unseen_cap(self, sim, twist):
        u = self.ucfg
        om = self.perc.omap
        if not u.unseen_cap or om is None:
            return twist, False
        vxy = np.asarray(twist[:2], float)
        sp = float(np.linalg.norm(vxy))
        if sp < 1e-3:
            return twist, False
        pose = sim.ctrl_pose() if getattr(sim, "loc", None) is not None else sim.pose()       # est pose_source: the unseen-space map lives in the estimated frame
        Rm = GM.rot(pose[2])
        dir_w = Rm @ (vxy / sp)
        c = pose[:2] + Rm @ np.array([0.2, -0.25])               # footprint centre
        ss = np.linspace(0.5, 0.6 + sp, 10)
        pts = c[None, :] + ss[:, None] * dir_w[None, :]
        if om.stale_fraction(pts, sim.t, u.t_stale) >= u.frac_unseen and sp > u.v_unseen:
            tw = np.asarray(twist, float).copy()
            tw[:2] *= u.v_unseen / sp
            return tw, True
        return twist, False

    # ------------------------------------------------------------------ one call
    def filter(self, sim, cmd: Cmd):
        self.perc.step(sim)
        cmd_op, glanced = cmd, False
        if self.look is not None:
            cmd = self.look.apply(sim, cmd, self.docks)
            glanced = cmd is not cmd_op
        tw, capped = self._unseen_cap(sim, cmd.twist)
        if capped:
            self.n_unseen_cap += 1
            cmd2 = Cmd(twist=tw, lift_v=cmd.lift_v, jog=cmd.jog, grasp=cmd.grasp, release=cmd.release)
        else:
            cmd2 = cmd
        safe, info = super().filter(self.perc.proxy(sim), cmd2)
        if capped or glanced:                                    # intervention w.r.t. what the OPERATOR asked
            dv = cmd_op.twist - safe.twist
            info["intervention"] = float(np.linalg.norm(dv))
            info["intervention_norm"] = float(np.linalg.norm(dv / BS.V_STICK))
            if glanced and info["label"] in ("none", "other"):
                info["label"] = "look_glance"
            elif info["label"] in ("none", "other"):
                info["label"] = "unseen_cap"
        info.update(n_tracks=len(self.perc.tracks), person_infl=self.last_infl, unseen_capped=bool(capped), glancing=bool(glanced))
        return safe, info
