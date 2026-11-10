"""nero/base/drive.py -- the swerve base driver of a CarryEpisode (SIMULATION ONLY).  Promoted from smoke.BaseDrive (11-09).

A step hook of the episode: while a route runs it drives `SwerveBase` from a `PathFollower` (one trapezoid profile per segment at a speed LEVEL,
P hold on the segment goal in the rest gap after each); while idle it holds the steers and lets the wheel command decay (`SwerveBase.stop`).
The arms are held by the carry controller (joint references), they are not touched here (ASSUMED A4: no concurrent arm and base motion).

  ep.base_move(ref, branch=None)   (CarryEpisode method, calls BaseDriver.run)
      ref = {"route": [["line", d_m] | ["turn", deg], ...], "level": "L1" | "L2" | "L3"}   line d along the robot's CURRENT +x, turn deg CCW.
      Steps the episode for route time + END_SETTLE_S; branch(ep, t_rel) every 0.25 s (like moves).  -> dict(start, end, nominal_end, err_mm, err_deg, T_s).

Bounds: GAP_S (rest after each segment) = the G0 smoke gap, ASSUMED; END_SETTLE_S = the extra hold after the last gap of the G0 smoke, ASSUMED;
ROUTE_TOL_M / ROUTE_TOL_DEG (route_unfinished, ASSUMED) live in monitors.py.
"""
from __future__ import annotations

import copy

import numpy as np

from . import monitors as MON
from . import swerve as SW

DT = 0.005                 # = carry.DT (checked by the base tests)
GAP_S = 0.5                # ASSUMED s: rest after each route segment (G0 smoke)
END_SETTLE_S = 0.5         # ASSUMED s: extra hold after the last rest gap (G0 smoke ran route + 0.5 s)
BRANCH_PERIOD_S = 0.25     # same as the arm moves


def route_to_segments(route):
    """[["line", d_m] | ["turn", deg], ...] -> [("line", d_m) | ("turn", rad), ...]; validated."""
    out = []
    for seg in route:
        kind, a = seg[0], float(seg[1])
        if kind == "line":
            out.append(("line", a))
        elif kind == "turn":
            out.append(("turn", float(np.radians(a))))
        else:
            raise ValueError(f"route segment must be line or turn, got {kind!r}")
    return out


def nominal_end(p0, route):
    """nominal end pose (x, y, yaw) of `route` from the base pose p0 = (x, y, yaw): lines along the current heading, turns add to it."""
    x, y, yaw = float(p0[0]), float(p0[1]), float(p0[2])
    for kind, a in route_to_segments(route):
        if kind == "line":
            x, y = x + a * np.cos(yaw), y + a * np.sin(yaw)
        else:
            yaw += a
    return np.array([x, y, yaw])


class BaseDriver:
    """step hook; registered by CarryEpisode(base="swerve") at the END of its constructor (so it runs after the carry step, like smoke.BaseDrive)."""

    def __init__(self, ep, sb):
        self.ep, self.sb = ep, sb
        self.reset()
        ep.step_hooks.append(self)

    def reset(self):
        self.queue, self.pf, self.gap_left, self.seg_err = [], None, 0.0, []
        self.seg_idx, self.seg_kind, self.last, self._pend, self.gap = -1, "", None, False, GAP_S
        self.active = False
        self.slip_peak = 0.0

    # ---- the route clock (same logic as smoke.BaseDrive) -------------------------------------------------------------------
    def start_route(self, route, gap=GAP_S):
        """route = [("line", m) | ("turn", rad)]; -> total time of the segments + rest gaps [s]."""
        self.queue, self.gap, self.gap_left, self.pf, self.seg_err, self.last, self._pend = list(route), gap, 0.0, None, [], None, False
        self.seg_idx, self.seg_kind = -1, ""
        T = 0.0
        L = self.sb.lim
        for kind, a in route:
            T += (SW.trap_profile(a, L.v_max, L.a_max)[0] if kind == "line" else SW.trap_profile(a, L.w_max, L.alpha_max)[0]) + gap
        self.active = True
        return T

    def _advance(self):
        if self.pf is not None and self.pf.done():
            self.last, self.pf, self.gap_left, self._pend = self.pf, None, self.gap, True
        if self.pf is None and self.gap_left <= 0.0 and self._pend:
            self.seg_err.append(self.last.error())          # segment error AFTER the rest gap (P hold on the segment's goal)
            self._pend = False
        if self.pf is None and self.gap_left <= 0.0 and self.queue:
            kind, a = self.queue.pop(0)
            self.seg_idx, self.seg_kind = self.seg_idx + 1, kind
            self.pf = SW.PathFollower(self.sb)
            (self.pf.drive_line if kind == "line" else self.pf.turn_in_place)(a)

    def abort(self):
        """drop the rest of the route; while still active, __call__ rate-limits the twist to 0 at the level's a / alpha (PathFollower._limit).
        The next route references the actual heading (yaw_nom cleared), not the aborted segment's goal."""
        self.queue, self.pf, self.last, self.gap_left, self._pend = [], None, None, 0.0, False
        self.sb.yaw_nom = None

    def phase(self):
        if not self.active or self.seg_idx < 0:
            return "pre" if self.active else "post"
        return f"seg{self.seg_idx}_{self.seg_kind}" if self.pf is not None else f"rest{self.seg_idx}"

    def moving(self):
        return self.active

    def __call__(self, ep):
        if self.active:
            self._advance()
            if self.pf is not None:
                tw = self.pf.step(DT)
            else:
                self.gap_left -= DT
                tw = self.last.hold_twist(DT) if self.last is not None else SW.PathFollower(self.sb)._limit(np.zeros(3), DT)      # rest gap: P on the last goal
            self.sb.apply(tw, DT)
        else:
            self.sb.stop()
        self.slip_peak = max(self.slip_peak, float(self.sb.wheel_slip().max()))

    # ---- state for CarryEpisode.snapshot / restore -------------------------------------------------------------------------
    def state(self):
        """everything the route clock needs (the PathFollowers are shallow-copied: their arrays are never modified in place)."""
        sb = self.sb
        return dict(queue=list(self.queue), gap_left=self.gap_left, gap=self.gap, seg_err=list(self.seg_err), seg_idx=self.seg_idx, seg_kind=self.seg_kind,
                    pend=self._pend, active=self.active, slip=self.slip_peak,
                    pf=None if self.pf is None else copy.copy(self.pf), last=None if self.last is None else copy.copy(self.last),
                    sb=dict(th_cmd=sb.th_cmd.copy(), have_cmd=sb._have_cmd, sgn=sb._sgn.copy(), n_rev=sb.n_reversal, wi=sb._wi.copy(), yaw_off=sb.yaw_off,
                            v=sb._v.copy(), yaw_nom=sb.yaw_nom, lim=sb.lim))

    def set_state(self, s):
        self.queue, self.gap_left, self.gap, self.seg_err = list(s["queue"]), s["gap_left"], s["gap"], list(s["seg_err"])
        self.seg_idx, self.seg_kind, self._pend, self.active, self.slip_peak = s["seg_idx"], s["seg_kind"], s["pend"], s["active"], s["slip"]
        self.pf = None if s["pf"] is None else copy.copy(s["pf"])
        self.last = None if s["last"] is None else copy.copy(s["last"])
        b, sb = s["sb"], self.sb
        sb.th_cmd, sb._have_cmd, sb._sgn, sb.n_reversal, sb._wi = b["th_cmd"].copy(), b["have_cmd"], b["sgn"].copy(), b["n_rev"], b["wi"].copy()
        sb.yaw_off, sb._v, sb.yaw_nom, sb.lim = b["yaw_off"], b["v"].copy(), b["yaw_nom"], b["lim"]

    # ---- one route --------------------------------------------------------------------------------------------------------
    def run(self, ref, branch=None):
        """run the route of `ref` at its level while the episode keeps the arms (steps the episode).  Raises ValueError on a bad level / segment."""
        from . import smoke as SM                         # LEVELS / level_limits live in smoke (the design's one level table); imported late (smoke imports carry)
        ep, sb = self.ep, self.sb
        level = ref.get("level", "L1")
        if level not in SM.LEVELS:
            raise ValueError(f"level must be one of {sorted(SM.LEVELS)}, got {level!r}")
        sb.lim = SM.level_limits(level)
        route = route_to_segments(ref["route"])
        p0 = sb.pose()
        nom = nominal_end(p0, ref["route"])
        T = self.start_route(route)
        n = int(round((T + END_SETTLE_S) / DT))
        step = int(round(BRANCH_PERIOD_S / DT))
        for i in range(n):
            if branch is not None and i > 0 and i % step == 0:
                branch(ep, i * DT)
            ep._step()
        self.active = False
        pb = sb.pose()
        dyaw = float((pb[2] - nom[2] + np.pi) % (2 * np.pi) - np.pi)
        e_m, e_deg = float(np.linalg.norm(pb[:2] - nom[:2])), float(np.degrees(dyaw))
        if e_m > MON.ROUTE_TOL_M or abs(e_deg) > MON.ROUTE_TOL_DEG:
            ep.base_flag("route_unfinished", ep.k * DT, err_mm=1e3 * e_m, yaw_err_deg=e_deg, tol=f"{1e3 * MON.ROUTE_TOL_M:.0f} mm / {MON.ROUTE_TOL_DEG} deg ASSUMED")
        return dict(level=level, T_s=float(T), start=p0.tolist(), end=pb.tolist(), nominal_end=nom.tolist(), err_mm=e_m, err_deg=e_deg,
                    seg_pos_err_mm=[1e3 * e[0] for e in self.seg_err], seg_yaw_err_deg=[float(np.degrees(e[1])) for e in self.seg_err],
                    slip_peak_mps=self.slip_peak)
