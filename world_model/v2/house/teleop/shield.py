"""teleop/shield.py -- shared-control safety layer: pass the operator's command when safe, minimally modify it otherwise.  SIMULATION ONLY.

Operator command u_op = (vx, vy, wz) body twist target (+ lift rate + tray-frame jog).  The shield returns u_safe, the point of a convex
polytope closest to u_op in the stick-normalised metric  ||(u - u_op) / V_STICK||  (min-norm QP, Hildreth, tray_rep/sets/qp.py), where the
polytope is the intersection of

 (a) CUP SET (reuses the F5 set math).  The cup stands on the tray, so what it feels is the tray-frame effective gravity
        g_in(a) = (e_x.(g - a), e_y.(g - a)),   N/m = c.(a + g z)                      -- reach.gin_of(c, a), c = tray axis (measured)
     and the cup keeps its contact (no slip, no spill) iff  |g_in| <= r * N/m  with
        r = min( kappa * mu_lo ,  tan(theta_spill) )
     mu in [mu_lo, mu_hi] = [0.18, 0.35] (sim truth 0.25), kappa = 0.8, theta_spill = 0.30 rad (the F5 default tilt set theta_max; the bead
     cup has 46 mm of freeboard, so the friction limit binds first).  Rows: 16-gon of the disc |g_in| <= r N/m, exact in a (g_in is linear in a
     for fixed c).  a = the cup's inertial acceleration from the base twist change over one tick (base.cup_accel_body: a_b + alpha x r + w x (v + w x r)).
     A jerk row set |a_k - a_{k-1}| <= J_MAX dt (12-gon) damps slosh excitation.  The tray tilt enters through c (measured), so a tilted tray
     shrinks the admissible acceleration: the base governor and the FL-MPC tilt cone share one budget.
 (b) KEEP-OUT (CBF-style speed scaling).  For every footprint boundary point q with clearance d(q) (1 cm EDT of the occupancy grid) and outward
     gradient n:  n . v_pt >= -min(gamma h, sqrt(2 a_b h)),  h = d - d_wall,  v_pt = R v_b + w z x (R q)   [class-K CBF + braking-distance bound,
     a_b = 0.6 a_lim; for h < 0 the row demands retreat at gamma |h| (capped)].  People: the footprint rectangle vs a disc of radius 0.30 + margin,
     relative velocity (the person's velocity is known to the shield: simulated tracker), same row form.  Cross-checked by ir.Clear.
 (c) HARDWARE: slew |u - v| <= A_HW dt, |u| <= V_STICK.
 (d) FL-MPC arm constraints are NOT touched: the arm side runs the unmodified FL-MPC.  The shield only sees the tray through the measured axis.
 Priority if the polytope is empty: drop jerk rows, then the cup rows (keep-out + hardware stay hard), else brake at the hardware limit.
 IR monitor (tray_rep/sets/ir.py atoms): margin of  inside(cup, shrink(tray_top, r_cup)) AND upright(tray, theta_max)  each tick; a negative margin
 (cup sliding toward the rim / tray over-tilted) VETOES the command (u_op := 0 -> brake inside the same polytope).
Tray jog governor: tilt channel rate-limited and clamped so the commanded tray tilt stays <= THETA_TRAY (the FL-MPC cone is 0.10); lift rate
and its slew limited.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from world_model.v2.tray_rep.sets import reach as RE, ir as IR, tray_spec as TS, qp as QP

from . import base as BS
from . import geom as GM
from .sim import Cmd, TICK
from .arm import _log_R, _rotvec_to_R

G = 9.81


@dataclass
class ShieldCfg:
    mu_lo: float = 0.18
    mu_hi: float = 0.35
    kappa: float = 0.80
    theta_spill: float = 0.30          # rad, F5 tilt set (no-spill)
    theta_tray: float = 0.05           # rad, commanded tray tilt bound (jog governor); FL-MPC cone is 0.10
    jerk_max: float = 15.0             # m/s^3 on the cup inertial acceleration
    d_wall: float = 0.10               # m, footprint clearance buffer to static obstacles
    d_person: float = 0.10             # m, extra buffer beyond the person disc (GM.PERSON_CLEAR_R)
    gamma: float = 1.5                 # 1/s class-K gain
    a_brk_frac: float = 0.6            # braking decel assumed by the braking-distance bound, as a fraction of the cup limit
    n_keep: int = 28
    lift_v_max: float = 0.06
    lift_a_max: float = 0.30
    jog_v_max: float = 0.08
    jog_w_max: float = 0.15
    jog_a_max: float = 0.6             # m/s^2
    jog_alpha_max: float = 2.0         # rad/s^2
    use_cup: bool = True
    cup_row: str = "always"            # OPT-IN (10-22): "always" (default, bit-identical) | "attached": cup rows + cup lever arm only while the cup rides on the robot (geom.cup_attached)
    use_keepout: bool = True
    use_ir_veto: bool = True
    r_cup: float = 0.03
    dock_radius: float = 0.8           # within this distance of a dock pose (pick / place stand-off) the wall buffer shrinks to d_wall_dock
    d_wall_dock: float = 0.03
    loc_margin: float = 0.0            # m, extra wall buffer for localization/map error (house/loc); 0 = off
    loc_margin_k: float = 0.0           # x the PoseEstimator 1-sigma proxy (sim.loc.sigma_xy) added to the wall buffer; 0 = off
    # ---- people v2 (ALL default OFF -> bit-identical to the original shield) -----------------------------------------------------------
    pred_taus: tuple = ()              # velocity-obstacle style rows: keep-out against the person's CONSTANT-VELOCITY position at t+tau (moving people only)
    pred_min_speed: float = 0.05       # m/s, a person slower than this gets no predictive row (static people keep the original CBF row only)
    pred_buffer: float = 0.10          # m, buffer beyond PERSON_CLEAR_R in the predictive rows
    relax_tier: bool = False           # before the emergency brake try the keep-out with relaxed buffers (wall d_wall_relax, person d_person_relax), no cup rows
    fallback: str = "brake"            # "escape": before the emergency brake, try the keep-out rows with each ALREADY-VIOLATED row's demand capped at what the hardware slew
                                       # can deliver this tick (A_i u >= min over the slew box), no cup rows.  The brake ignores the operator and holds the robot inside a
                                       # violated buffer forever (h < 0 needs gamma*|h| > a_hw*dt outward speed from rest, which the brake never builds up); escape moves out.
    d_wall_relax: float = 0.02
    d_person_relax: float = 0.0
    deadlock_s: float | None = None    # social yield: operator pushing + person row binding + robot (almost) stopped for this long -> retreat, then hold
    retreat_speed: float = 0.35        # m/s
    retreat_clear: float = 0.30        # m, retreat until every person keep-out margin h >= this (or retreat_max_s)
    retreat_max_s: float = 5.0
    hold_s: float = 3.0                # stationary hold after a retreat (lets the other party pass) before control returns to the operator
    squeeze_s: float | None = None     # social yield also fires when the predictive rows (pred_taus) demand intrusion (b<0) but cannot be met for this long
    blk_near: float = 0.25             # m, a person within this keep-out margin h counts as 'blocking'
    # ---- live footprint / doorway buffer (default "fixed" / off -> bit-identical to before) -------------------------------------------------
    footprint: str = "fixed"           # "fixed" = geom.FP (phase-1 hold-pose AABB, conservative) | "live" = base-frame xy box of the robot's CURRENT geoms (arms, tray, cup) + live_pad
    live_pad: float = 0.02             # m, pad added on every side of the live box (hull is of the collision geoms; pad covers the <= 5 ms kinematics lag + tracking error)
    doorways: tuple = ()               # [(x, y)] map-frame doorway centres: within door_radius the wall buffer is d_wall_door (opt-in; needs d_wall_door)
    d_wall_door: float | None = None   # m, wall buffer inside a doorway (None = off: d_wall everywhere)
    door_radius: float = 0.5           # m, distance from the footprint rectangle to the doorway centre under which d_wall_door applies

    @property
    def ratio(self):
        return min(self.kappa * self.mu_lo, math.tan(self.theta_spill))

    @property
    def a_lim_flat(self):
        return self.ratio * G


PEOPLE_V2 = dict(pred_taus=(0.5, 1.0, 1.5, 2.0), relax_tier=True, deadlock_s=3.0, squeeze_s=0.4)        # ShieldCfg overrides: the people-safety v2 shield


def _poly_normals(n, offset=0.5):
    ang = (np.arange(n) + offset) * 2 * math.pi / n
    return np.stack([np.cos(ang), np.sin(ang)], 1)


class IRMonitor:
    """ir.py atoms on the live MuJoCo state: inside(cup, shrink(tray_top, r_cup)) AND upright(tray, theta_max); clear(footprint, person)."""

    def __init__(self, cfg: ShieldCfg, lang=None):
        if lang is None:                                    # default: the hand-written TRAY_SPEC_FULL (bit-identical to before the language layer)
            ents = TS.env_entities("centre")
            self.spec = TS.build_spec(TS.TRAY_SPEC_FULL, ents, H=0, r_shrink=cfg.r_cup, theta_max=cfg.theta_tray_ir, lo=0)
        else:                                               # house/lang LangResult: VLM-authored spec over the carried entities + named perceived objects
            self.spec = lang.monitor_spec(cfg)
        self.lang = lang
        self.cfg = cfg

    def tray_margin(self, sim):
        d = sim.d
        st = TS.EnvState(d.xpos[sim.tray].copy(), d.xquat[sim.tray].copy(), d.xpos[sim.cup].copy())
        return float(IR.margin(self.spec, st))

    def person_margin(self, sim, fp=None):
        """min over people of ir.Clear(footprint box, person disc) [m]; footprint = AABB zonotope of the robot, person = outer ball zonotope."""
        if not sim.people_state:
            return np.inf
        pose = sim.pose()
        fp = GM.FP if fp is None else fp
        half = np.array([(fp["x"][1] - fp["x"][0]) / 2, (fp["y"][1] - fp["y"][0]) / 2, 0.5])
        cen_local = np.array([(fp["x"][1] + fp["x"][0]) / 2, (fp["y"][1] + fp["y"][0]) / 2, 0.5])
        R3 = np.eye(3); R3[:2, :2] = GM.rot(pose[2])
        pA = np.array([pose[0], pose[1], 0.0]) + R3 @ cen_local
        # box centred on its own origin: use centre=0 and place the origin at pA
        A = IR.Entity("fp", IR.BoxShape(half), lambda s: IR.Pose(R3, pA))
        best = np.inf
        for xy, _v in sim.people_state:
            B = IR.Entity("pp", IR.Sphere(GM.PERSON_CLEAR_R), lambda s, xy=xy: IR.Pose(np.eye(3), np.array([xy[0], xy[1], 0.5])))
            best = min(best, float(IR.Clear(A, B, 0.0).eval(None, False)))
        return best


ShieldCfg.theta_tray_ir = 0.10     # IR upright bound = FL-MPC tilt cone


class Shield:
    def __init__(self, cfg: ShieldCfg | None = None, occ=None, a_hw=BS.A_HW, lang=None):
        self.cfg = cfg or ShieldCfg()
        self.occ = occ
        self.a_hw = np.asarray(a_hw)
        self._mu0 = (self.cfg.mu_lo, self.cfg.mu_hi)          # hard-coded mu set (restored by set_lang(None))
        self.ir_mon = IRMonitor(self.cfg)
        self.lang = None
        self.a_prev_w = np.zeros(2)
        self.jog_prev = np.zeros(6)
        self.lift_prev = 0.0
        self.n_fallback = 0
        self.n_veto = 0
        self.n_calls = 0
        self._pts = GM.boundary_pts(step=0.05)
        self._cup_N = _poly_normals(16)
        self._jerk_N = _poly_normals(12)
        self.last = {}
        self.soc = dict(mode="free", t=0.0, hist=[], sq=0.0)
        self.trail = []
        self._pred_fail = False
        self.n_deadlock = 0
        self.n_pred = 0
        self.n_squeeze = 0
        self._live = None                  # geom.LiveFootprint (built lazily for cfg.footprint == "live")
        self._live_m = None
        self.fp_now = GM.FP                # footprint used by the current tick's rows (dict x/y)
        self.fp_ms = [0.0, 0]              # [sum ms, n] cost of the live footprint
        self.docks = []                    # [(x, y)] stand-off poses: the manipulation phase needs to get within `standoff_clear` (0.07 m) of the counter
        if lang is not None:
            self.set_lang(lang)

    def set_lang(self, lang):
        """language layer (house/lang, flag-gated; default never called): mu set and the IR-monitor spec from a LangResult; None restores the hard-coded ones.
        Mutates cfg in place (callers hold references to shield.cfg).  The new monitor is built first, so a failing spec leaves the old one active."""
        if lang is None:
            self.cfg.mu_lo, self.cfg.mu_hi = self._mu0
            self.ir_mon = IRMonitor(self.cfg)
            self.lang = None
            return
        mon = IRMonitor(self.cfg, lang)
        ov = lang.mu_overrides()
        self.cfg.mu_lo, self.cfg.mu_hi = ov["mu_lo"], ov["mu_hi"]
        self.ir_mon = mon
        self.lang = lang

    # ---------------------------------------------------------------------------------------------- base governor
    def _cup_rows(self, sim, c, v, r_c, w_mid):
        cfg = self.cfg
        yaw = sim.base.yaw
        Rw = GM.rot(yaw)
        dt = TICK
        M = np.array([[1.0, 0.0, -r_c[1]], [0.0, 1.0, r_c[0]]])
        Au = M / dt                                        # a_body = Au u + a0
        # centripetal term at the mid-tick yaw rate
        w = w_mid
        wv = np.array([v[0] - w * r_c[1], v[1] + w * r_c[0]])
        a_c0 = np.array([-w * wv[1], w * wv[0]])
        a0 = a_c0 - (M @ v) / dt
        # f_psi(a) = N_psi . g_in(a) - r * N/m  (affine in a=(ax,ay,0) world)
        def gin(a2):
            gx, gy, nm = RE.gin_of(c, np.array([a2[0], a2[1], 0.0]))
            return np.array([gx, gy]), nm
        g0, nm0 = gin(np.zeros(2))
        g1, nm1 = gin(np.array([1.0, 0.0])); g2, nm2 = gin(np.array([0.0, 1.0]))
        Jg = np.stack([g1 - g0, g2 - g0], 1)               # (2,2): d g_in / d a_world
        Jn = np.array([nm1 - nm0, nm2 - nm0])              # d N/m / d a_world
        r = cfg.ratio * math.cos(math.pi / 16)             # inner polygon of the disc
        N = self._cup_N
        # row_psi(a) = N_psi.(g0 + Jg a) - r (nm0 + Jn.a) <= 0
        coef_a = N @ Jg - r * Jn[None, :]                  # (16,2) on a_world
        const = N @ g0 - r * nm0
        Aw = coef_a @ Rw @ Au                              # (16,3) on u
        bw = -(const + coef_a @ Rw @ a0)
        # jerk rows
        rj = cfg.jerk_max * dt * math.cos(math.pi / 12)
        NJ = self._jerk_N
        Aj = NJ @ Rw @ Au
        bj = rj - NJ @ (Rw @ a0) + NJ @ self.a_prev_w
        return (Aw, bw), (Aj, bj), (Rw, Au, a0)

    @staticmethod
    def _wall_pose(sim):
        """pose the wall keep-out is evaluated at: sim.ctrl_pose() (estimate or truth).  The grid (self.occ) lives in the same map frame.
        People rows / cup rows use RELATIVE geometry (people tracker / tray sensing are in the robot frame) -> they keep the true pose."""
        f = getattr(sim, "ctrl_pose", None)
        return sim.pose() if f is None else f()

    def _loc_margin(self, sim):
        """extra wall buffer [m] for localization / map error:  cfg.loc_margin + cfg.loc_margin_k * (estimator 1-sigma proxy)."""
        cfg = self.cfg
        if cfg.loc_margin == 0.0 and cfg.loc_margin_k == 0.0:
            return 0.0
        loc = getattr(sim, "loc", None)
        return cfg.loc_margin + (cfg.loc_margin_k * loc.sigma_xy if (loc is not None and cfg.loc_margin_k) else 0.0)

    def footprint_of(self, sim):
        """footprint rectangle (robot frame) the rows of THIS tick use: geom.FP ("fixed", default) or the live box of the current configuration ("live")."""
        if self.cfg.footprint != "live":
            return GM.FP
        import time as _t
        if self._live is None or self._live_m is not sim.m:          # one-time precompute of the hull point sets (not counted in fp_ms)
            self._live, self._live_m = GM.LiveFootprint(sim.m, pad=self.cfg.live_pad), sim.m
        t0 = _t.perf_counter()
        fp = self._live.box(sim.d, sim.pose(), pad=self.cfg.live_pad)
        self.fp_ms[0] += (_t.perf_counter() - t0) * 1e3
        self.fp_ms[1] += 1
        return fp

    def _keep_rows(self, sim, v, relax=False):
        cfg = self.cfg
        rows, b, tags = [], [], []
        pose = self._wall_pose(sim)                        # MAP-frame pose (= truth by default; the localization estimate with pose_source="est")
        d_wall = cfg.d_wall_dock if any(np.hypot(pose[0] - x, pose[1] - y) < cfg.dock_radius for x, y in self.docks) else cfg.d_wall
        if cfg.d_wall_door is not None and any(GM.rect_dist_point(np.array([x, y]), pose, self.fp_now)[0] < cfg.door_radius for x, y in cfg.doorways):
            d_wall = min(d_wall, cfg.d_wall_door)           # opt-in doorway buffer: while the FOOTPRINT is within door_radius of the doorway centre
        if relax:
            d_wall = min(d_wall, cfg.d_wall_relax)
        d_wall = d_wall + self._loc_margin(sim)            # localization-error margin (0 by default)
        R = GM.rot(pose[2])
        a_b = cfg.a_brk_frac * cfg.a_lim_flat
        pts = self._pts if cfg.footprint != "live" else GM.boundary_pts(self.fp_now, step=0.05)
        if self.occ is not None:
            w = pts @ R.T + pose[:2]
            d, gx, gy = GM.grid_lookup_grad(self.occ, w[:, 0], w[:, 1])
            h = d - d_wall
            idx = np.argsort(h)[:cfg.n_keep]
            idx = idx[h[idx] < 1.3]
            for i in idx:
                n = np.array([gx[i], gy[i]])
                if not n.any():
                    n = (pose[:2] - w[i]) / (np.linalg.norm(pose[:2] - w[i]) + 1e-9)
                nb = R.T @ n
                q = pts[i]
                row = np.array([nb[0], nb[1], -q[1] * nb[0] + q[0] * nb[1]])
                hi = float(h[i])
                beta = min(cfg.gamma * hi, math.sqrt(2 * a_b * hi)) if hi >= 0 else max(cfg.gamma * hi, -0.3)
                rows.append(-row); b.append(beta); tags.append("wall")
        pr, pb, pt = self._person_rows(sim, relax)
        rows += pr; b += pb; tags += pt
        return (np.array(rows).reshape(-1, 3), np.array(b), tags)

    def _person_rows(self, sim, relax=False):
        """original person keep-out rows (footprint rectangle vs a disc, relative velocity).  relax: buffer d_person -> d_person_relax."""
        cfg = self.cfg
        rows, b, tags = [], [], []
        pose = sim.pose()
        R = GM.rot(pose[2])
        a_b = cfg.a_brk_frac * cfg.a_lim_flat
        d_p = cfg.d_person_relax if relax else cfg.d_person
        for (xy, vel), p in zip(sim.people_state, sim.scene.people):
            dist, qn, nw = GM.rect_dist_point(xy, pose, self.fp_now)
            hi = dist - (GM.PERSON_CLEAR_R + d_p)
            if hi > 2.0:
                continue
            nb = R.T @ nw                                  # unit vector from the rectangle to the person (robot frame)
            row = np.array([nb[0], nb[1], -qn[1] * nb[0] + qn[0] * nb[1]])   # n . v_pt per unit u
            beta = min(cfg.gamma * hi, math.sqrt(2 * a_b * hi)) if hi >= 0 else max(cfg.gamma * hi, -0.3)
            rows.append(row); b.append(beta + float(nw @ vel)); tags.append(f"person:{p.name}")
        return rows, b, tags

    def _pred_rows(self, sim):
        """velocity-obstacle style rows (cfg.pred_taus): if the base keeps twist u for tau, the footprint must not enter the person's keep-out disc
        at the person's constant-velocity position p + w tau:   n_tau . v_pt <= h_tau / tau   (n_tau, h_tau = rectangle->p_tau direction, margin).
        Only for MOVING people.  Soft tier: dropped first when the QP is infeasible (the original CBF rows stay hard)."""
        cfg = self.cfg
        pose = sim.pose()
        R = GM.rot(pose[2])
        rows, b = [], []
        for (xy, vel), p in zip(sim.people_state, sim.scene.people):
            if float(np.hypot(*vel)) < cfg.pred_min_speed:
                continue
            for tau in cfg.pred_taus:
                dist, qn, nw = GM.rect_dist_point(np.asarray(xy) + np.asarray(vel) * tau, pose, self.fp_now)
                h = dist - (GM.PERSON_CLEAR_R + cfg.pred_buffer)
                rhs = h / tau
                if rhs > 1.2:
                    continue
                nb = R.T @ nw
                rows.append(np.array([nb[0], nb[1], -qn[1] * nb[0] + qn[0] * nb[1]])); b.append(max(rhs, -0.6))
        return np.array(rows).reshape(-1, 3), np.array(b)

    def _min_margin(self, sim):
        """smallest person keep-out margin h (footprint-rectangle distance - (disc + d_person)); (inf, None) with nobody around."""
        pose = sim.pose()
        best, who = np.inf, None
        for (xy, vel), p in zip(sim.people_state, sim.scene.people):
            dist, qn, nw = GM.rect_dist_point(xy, pose, self.fp_now)
            h = dist - (GM.PERSON_CLEAR_R + self.cfg.d_person)
            if h < best:
                best, who = h, (p.name, nw)
        return best, who

    def _breadcrumb_dir(self, pose, back_m=0.9):
        """world unit vector from the base to the trail point ~back_m of path length behind it (None if the trail is shorter than 0.15 m)."""
        tr = self.trail
        pos = np.asarray(pose[:2])
        acc, prev = 0.0, pos
        for q in reversed(tr):
            acc += float(np.linalg.norm(q - prev))
            prev = q
            if acc >= back_m:
                break
        d = prev - pos
        n = float(np.linalg.norm(d))
        return None if (acc < 0.15 or n < 0.05) else d / n

    def _social(self, sim, u_req, v):
        """deadlock resolution (cfg.deadlock_s): returns (u_req', tag or None).  Mutual blocking = the operator pushes (|u_op|_norm > 0.15), the
        robot is (almost) stopped, and a person keep-out margin is < blk_near.  After deadlock_s: RETREAT from the nearby people at retreat_speed
        (QP-projected through all rows, so walls still hold) until margin >= retreat_clear or retreat_max_s, then HOLD still for hold_s."""
        cfg = self.cfg
        st = self.soc
        if st["mode"] != "retreat":                    # the trail is the path driven BEFORE the retreat starts
            q = np.array([sim.base.x, sim.base.y])
            if not self.trail or float(np.linalg.norm(q - self.trail[-1])) > 0.02:
                self.trail.append(q)
            if len(self.trail) > 600:
                self.trail.pop(0)
        if st["mode"] == "retreat":
            st["t"] += TICK
            hmin, _ = self._min_margin(sim)
            if (hmin >= cfg.retreat_clear and not self._pred_fail) or st["t"] >= cfg.retreat_max_s:
                st.update(mode="hold", t=0.0)
            else:
                pose = sim.pose()
                R = GM.rot(pose[2])
                dirw = self._breadcrumb_dir(pose)               # back along the path we came by (known free); else away from the people
                if dirw is None:
                    dirw = np.zeros(2)
                    for (xy, vel), p in zip(sim.people_state, sim.scene.people):
                        dist, qn, nw = GM.rect_dist_point(xy, pose, self.fp_now)
                        if dist - (GM.PERSON_CLEAR_R + cfg.d_person) < 1.0:
                            dirw -= nw / max(dist, 0.05)
                db = R.T @ dirw
                if np.linalg.norm(db) < 1e-6:
                    db = np.array([-1.0, 0.0])
                db = db / np.linalg.norm(db)
                u = np.array([db[0] * cfg.retreat_speed, db[1] * cfg.retreat_speed, 0.0])
                return u, "yield_retreat"
        if st["mode"] == "hold":
            st["t"] += TICK
            if st["t"] >= cfg.hold_s:
                st.update(mode="free", t=0.0, hist=[])
            else:
                return np.zeros(3), "yield_hold"
        hmin, _ = self._min_margin(sim)
        pushing = float(np.linalg.norm(u_req / BS.V_STICK)) > 0.15
        pos = np.array([sim.base.x, sim.base.y])
        hist = st["hist"]
        hist.append((pos, pushing and hmin < cfg.blk_near))
        n_win = int(round(cfg.deadlock_s / TICK))
        if len(hist) > n_win:
            hist.pop(0)
        # mutual blocking = for the whole window: operator pushing AND a person margin < blk_near AND the base went nowhere (< 0.15 m net)
        if len(hist) == n_win and sum(h[1] for h in hist) >= 0.9 * n_win and \
                max(float(np.linalg.norm(h[0] - hist[0][0])) for h in hist) < 0.15:
            st.update(mode="retreat", t=0.0, hist=[])
            self.n_deadlock += 1
        elif cfg.squeeze_s is not None:
            st["sq"] = st["sq"] + TICK if self._pred_fail else 0.0
            if st["sq"] >= cfg.squeeze_s:
                st.update(mode="retreat", t=0.0, hist=[], sq=0.0)
                self.n_squeeze += 1
        return u_req, None

    def filter_base(self, sim, u_op):
        cfg = self.cfg
        self.fp_now = self.footprint_of(sim)               # one footprint for all rows of this tick (GM.FP unless cfg.footprint == "live")
        d = sim.d
        v = sim.base.v.copy()
        c = d.xmat[sim.tray].reshape(3, 3)[:, 2].copy()
        R = GM.rot(sim.base.yaw)
        cup_xy = d.xpos[sim.cup][:2]
        r_c = R.T @ (cup_xy - np.array([sim.base.x, sim.base.y]))
        use_cup = cfg.use_cup
        if cfg.cup_row == "attached" and not GM.cup_attached(sim):
            r_c, use_cup = np.zeros(2), False               # the cup is not on the robot: no cup rows, no lever arm (it sits on a surface)
        vs = BS.V_STICK
        wgt = 1.0 / vs
        dt = TICK
        u_req = np.asarray(u_op, float).copy()
        social = None
        if cfg.deadlock_s is not None:
            u_req, social = self._social(sim, u_req, v)
        veto = False
        ir_m = self.ir_mon.tray_margin(sim) if cfg.use_ir_veto else 1.0
        if cfg.use_ir_veto and ir_m < 0:
            u_req[:] = 0.0
            veto = True
            self.n_veto += 1
        # hardware / speed rows (hard)
        A_hw, b_hw, t_hw = [], [], []
        for i in range(3):
            e = np.zeros(3); e[i] = 1.0
            A_hw += [e, -e]; b_hw += [v[i] + self.a_hw[i] * dt, -v[i] + self.a_hw[i] * dt]; t_hw += ["hw_slew"] * 2
            A_hw += [e, -e]; b_hw += [vs[i], vs[i]]; t_hw += ["speed_cap"] * 2
        A_hw, b_hw = np.array(A_hw), np.array(b_hw)
        A_keep, b_keep, t_keep = self._keep_rows(sim, v) if cfg.use_keepout else (np.zeros((0, 3)), np.zeros(0), [])
        A_pred, b_pred = self._pred_rows(sim) if (cfg.use_keepout and cfg.pred_taus) else (np.zeros((0, 3)), np.zeros(0))
        relax_cache = []

        def relaxed():
            if not relax_cache:
                relax_cache.append(self._keep_rows(sim, v, relax=True))
            return relax_cache[0]
        u = np.clip(u_req, v - self.a_hw * dt, v + self.a_hw * dt)
        sol_ok, used = True, "all"
        for it in range(2):                                # fixed point on the mid-tick yaw rate
            w_mid = 0.5 * (v[2] + u[2])
            if use_cup:
                (Aw, bw), (Aj, bj), (Rw, Au, a0) = self._cup_rows(sim, c, v, r_c, w_mid)
            else:
                Aw = Aj = np.zeros((0, 3)); bw = bj = np.zeros(0)
                Rw = GM.rot(sim.base.yaw); M = np.array([[1.0, 0.0, -r_c[1]], [0.0, 1.0, r_c[0]]]); Au = M / dt
                a0 = -(M @ v) / dt + np.array([-w_mid * (v[1] + w_mid * r_c[0]), w_mid * (v[0] - w_mid * r_c[1])])
            groups = [("all", [(A_hw, b_hw), (A_keep, b_keep), (Aw, bw), (Aj, bj)]),
                      ("no_jerk", [(A_hw, b_hw), (A_keep, b_keep), (Aw, bw)]),
                      ("no_cup", [(A_hw, b_hw), (A_keep, b_keep)])]
            if len(b_pred):
                groups.insert(0, ("all+pred", [(A_hw, b_hw), (A_keep, b_keep), (Aw, bw), (Aj, bj), (A_pred, b_pred)]))
            if cfg.relax_tier and cfg.use_keepout:
                groups.append(("relax_keep", [(A_hw, b_hw), "relaxed"]))
            for name, parts in groups:
                if "relaxed" in parts:
                    parts = [parts[0], relaxed()[:2]]
                A = np.vstack([p[0] for p in parts if len(p[1])])
                bb = np.concatenate([p[1] for p in parts if len(p[1])])
                us, ok, nsw, viol = QP.project(A, bb, u_req, wgt)
                self.dbg = getattr(self, 'dbg', {}); self.dbg[name] = (bool(ok), float(viol), int(nsw))
                if ok:
                    self._pred_fail = bool(len(b_pred) and name != "all+pred" and float(b_pred.min()) < 0.0)
                    used = "all" if name == "all+pred" else name
                    if name == "all+pred":
                        self.n_pred += 1
                    u = us
                    break
            else:
                if cfg.fallback == "escape" and cfg.use_keepout and len(b_keep):
                    # slew-capped keep-out: each row asks for at most theta x the improvement the hardware slew box allows this tick,
                    # b_i(theta) = max(b_i, A_i v - theta |A_i| a_hw dt).  theta = 0 is always feasible (u = v holds every row at its current value,
                    # so nothing gets worse); bisect for the largest theta at which all rows are jointly feasible (two walls in a corner each
                    # asking for their own full-slew escape can be jointly infeasible), then project the operator's command onto that set.
                    Av, step = A_keep @ v, np.abs(A_keep) @ (self.a_hw * dt)
                    A_e = np.vstack([A_hw, A_keep])
                    best = None
                    lo_t, hi_t = 0.0, 1.0
                    for th in (1.0, *([None] * 7)):
                        th = hi_t if th is not None else 0.5 * (lo_t + hi_t)
                        us, ok, nsw, viol = QP.project(A_e, np.concatenate([b_hw, np.maximum(b_keep, Av - th * step + 1e-6)]), u_req, wgt)
                        if ok:
                            best, lo_t = (us, th), th
                            if th == hi_t:
                                break
                        else:
                            hi_t = th
                    self.dbg["escape"] = (best is not None, float(best[1]) if best else -1.0, 0)
                    if best is not None:
                        self._pred_fail = bool(len(b_pred) and float(b_pred.min()) < 0.0)
                        u, used = best[0], "escape"
                        break
                # emergency brake at the hardware limit
                self._pred_fail = bool(len(b_pred) and float(b_pred.min()) < 0.0)
                u = v - np.sign(v) * np.minimum(np.abs(v), self.a_hw * dt)
                used = "brake"
                sol_ok = False
                break
        if used != "all":
            self.n_fallback += 1
        # bookkeeping: which row is active / dominant
        if used == "relax_keep":
            A_keep, b_keep, t_keep = relaxed()
        label = "none"
        du = u_req - u
        if np.linalg.norm(du * wgt) > 1e-4:
            cand = []
            for nm, (A, bb, tg) in (("hw", (A_hw, b_hw, t_hw)), ("keep", (A_keep, b_keep, t_keep)),
                                    ("cup", (Aw, bw, ["cup_slip"] * len(bw))), ("jerk", (Aj, bj, ["cup_jerk"] * len(bj)))):
                if len(bb) == 0:
                    continue
                slack = bb - A @ u
                for i in np.where(slack < 2e-3)[0]:
                    nrm = np.linalg.norm(A[i] * wgt) + 1e-12
                    cand.append((float(A[i] @ du) / nrm, tg[i] if isinstance(tg, list) else tg))
            if cand:
                label = max(cand)[1]
            if label == "none":
                label = "veto" if veto else "other"
        if veto:
            label = "ir_veto"
        if used == "brake":
            label = "brake_fallback"
        elif used == "escape":
            label = "escape_fallback"
        if social and used != "brake":
            label = social
        a_body = Au @ u + a0
        a_w = Rw @ a_body
        self.a_prev_w = a_w
        self.last = dict(qp=used, ir_margin=ir_m, a_cup=float(np.linalg.norm(a_w)), veto=veto)
        if social:
            self.last['social'] = social
        return u, label

    # ---------------------------------------------------------------------------------------------- jog / lift governor
    def filter_jog(self, sim, jog):
        cfg = self.cfg
        out = np.asarray(jog, float).copy()
        out[:3] = np.clip(out[:3], -cfg.jog_v_max, cfg.jog_v_max)
        out[3:] = np.clip(out[3:], -cfg.jog_w_max, cfg.jog_w_max)
        # tilt: keep the commanded tray tilt (roll/pitch of the object frame) inside theta_tray; restoring rate beyond it
        rv = _log_R(sim.of.dR)
        nxt = rv[:2] + out[3:5] * TICK
        n = np.linalg.norm(nxt)
        if n > cfg.theta_tray:
            out[3:5] = -(rv[:2] / (np.linalg.norm(rv[:2]) + 1e-9)) * min(0.1, (np.linalg.norm(rv[:2]) - 0.9 * cfg.theta_tray) / TICK) \
                if np.linalg.norm(rv[:2]) > cfg.theta_tray else 0.0
        # slew
        dv = np.clip(out[:3] - self.jog_prev[:3], -cfg.jog_a_max * TICK, cfg.jog_a_max * TICK)
        dw = np.clip(out[3:] - self.jog_prev[3:], -cfg.jog_alpha_max * TICK, cfg.jog_alpha_max * TICK)
        out = np.concatenate([self.jog_prev[:3] + dv, self.jog_prev[3:] + dw])
        self.jog_prev = out.copy()
        return out

    def filter_lift(self, lift_v):
        cfg = self.cfg
        x = float(np.clip(lift_v, -cfg.lift_v_max, cfg.lift_v_max))
        x = self.lift_prev + float(np.clip(x - self.lift_prev, -cfg.lift_a_max * TICK, cfg.lift_a_max * TICK))
        self.lift_prev = x
        return x

    # ---------------------------------------------------------------------------------------------- one call
    def filter(self, sim, cmd: Cmd):
        """returns (safe Cmd, info dict(intervention, intervention_norm, label, ...))."""
        self.n_calls += 1
        u_safe, label = self.filter_base(sim, cmd.twist)
        jog_s = self.filter_jog(sim, cmd.jog)
        lift_s = self.filter_lift(cmd.lift_v)
        safe = Cmd(twist=u_safe, lift_v=lift_s, jog=jog_s, grasp=cmd.grasp, release=cmd.release)
        dv = (cmd.twist - u_safe)
        interv = float(np.linalg.norm(dv))
        interv_n = float(np.linalg.norm(dv / BS.V_STICK))
        jl = float(np.linalg.norm(cmd.jog - jog_s)) + abs(cmd.lift_v - lift_s)
        info = dict(intervention=interv, intervention_norm=interv_n, intervention_aux=jl, label=label, **self.last)
        return safe, info


class Passthrough:
    """shield OFF: the operator command goes straight to the (hardware-slew-limited) base."""

    n_fallback = 0
    n_veto = 0

    def filter(self, sim, cmd: Cmd):
        return cmd, dict(intervention=0.0, intervention_norm=0.0, intervention_aux=0.0, label="off", qp="off", ir_margin=np.nan,
                         a_cup=np.nan, veto=False)
