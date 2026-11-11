"""nero/base/smoke.py -- smokes of the swerve base (SIMULATION ONLY, CPU).

  python -m world_model.v2.nero.base.smoke nl       # (a) no load: rest drift, turn 180 deg, drive 0.5 m
  python -m world_model.v2.nero.base.smoke light|heavy [seed] [--task tray6|box] [--level L1|L2|L3] [--g0 [--video out.mp4]]   # G0: loaded route R1 + monitor
      --task box = the crate of task box_behind (BoxWorld(shelf=False), stack/tasks.py); light = prior low end, heavy = heaviest of 40 e2e seeds
  python -m world_model.v2.nero.base.smoke loaded   # (b) tray6 welded, lifted 2 cm, base turns 180 deg  (+ (c) fixed-vs-floating hold comparison)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import mujoco

from .. import config as C
from ..contents import scenes as SC
from ..pan import scene as PS
from ..contents import carry as CA
from ..contents import probe as PR
from . import scene as BS
from . import swerve as SW

DT = 0.005

# G0 speed levels of the table-behind design (docs/status/table_behind_design_2026-11-09.md section 8, A1): (v m/s, a m/s^2, w rad/s, alpha rad/s^2).
# ASSUMED, pending user ruling.  L1 = today's swerve.Limits defaults; L3 = the YOR caps (swerve.YOR_MAX_VEL / YOR_MAX_ACCEL, MEASURED from YOR source;
# using them as a cap is our choice); L2 = ASSUMED midpoint.
LEVELS = dict(L1=(0.25, 0.25, 0.5, 0.5), L2=(0.5, 0.5, 1.0, 1.0), L3=(1.0, 1.0, 1.57, 1.57))

# G0 base U flags (USER 11-09 accepted the proposal "U flags for base tilt and unfinished route"):
#   wheel_liftoff      MODEL: the two wheels of one support EDGE (same axle or same side) have no floor contact for LIFTOFF_STEPS consecutive steps:
#                      the base pivots about the opposite edge (tip onset).  Not flagged: one wheel off (a rigid 4-wheel base on a plane often stands
#                      on 3 wheels) and a diagonal pair off (rocking / hop; seen 11-09 with 2 wheels off at 0.01-0.04 deg base tilt).  The base angle
#                      is reported, not thresholded.
#   base_world_contact MODEL: a base/torso geom (robot, not an arm) penetrates a world geom other than the floor
#   load_world_contact MODEL: the carried object or a mug penetrates a world geom (after the lift, nothing may touch it during the route)
#   route_unfinished   ASSUMED tolerance: base end pose after the last rest gap off the nominal end of the route by more than ROUTE_TOL_M / ROUTE_TOL_DEG
from .monitors import BaseMonitors, EDGES, LIFTOFF_STEPS, ROUTE_TOL_M, ROUTE_TOL_DEG   # 11-09: the base U flags live in monitors.py (same values)


def level_limits(level="L1"):
    v, a, w, al = LEVELS[level]
    return SW.Limits(v_max=v, a_max=a, w_max=w, alpha_max=al)


def arm_geoms(m):
    return {g for g in range(m.ngeom) if (m.body(m.geom_bodyid[g]).name or "").startswith(("left_arm", "right_arm"))}


def bad_contacts(m, d, ag, obj=None):
    """names of penetrating (dist < -1 mm) arm contacts: arm-world / arm-other-arm / arm-base (kinematic neighbours are excluded in the MJCF);
    `obj` = body id of the carried object whose contacts with the gripper bodies are the grasp (skipped)."""
    out = set()
    for c in d.contact[:d.ncon]:
        if c.dist > -1e-3:
            continue
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in ag) or (g2 in ag):
            if obj is not None and (m.geom_bodyid[g1] == obj or m.geom_bodyid[g2] == obj):
                continue
            out.add((m.geom(g1).name or m.body(m.geom_bodyid[g1]).name, m.geom(g2).name or m.body(m.geom_bodyid[g2]).name))
    return sorted(out)


def make_ep(world, base="swerve", second_table=True, strict_claw=True):
    """strict_claw=True (default since 11-09): the per-claw payload row |Fz| > (3 kg - gripper) g is in U, as in same_worlds (both methods)."""
    with BS.use_swerve_scene(second_table=second_table, base=base):
        ep = CA.CarryEpisode(world, strict_claw=strict_claw)
    ep.set_belief(float(ep.truth0["contents_mass"]))          # the payload feed-forward uses the TRUE contents mass (smoke of the base, not of the belief)
    return ep


class BaseDrive:
    """step hook: drives the swerve base from a PathFollower while the carry controller holds the arms; records per-step peaks."""

    def __init__(self, ep, sb):
        self.ep, self.sb, self.pf = ep, sb, None
        self.active = False
        self.reset()
        ep.step_hooks.append(self)

    def pf_zero(self):
        return SW.PathFollower(self.sb)._limit(np.zeros(3), DT)

    def reset(self):
        self.queue, self.pf, self.gap_left, self.seg_err = [], None, 0.0, []
        self.seg_idx, self.seg_kind = -1, ""
        self.peak = dict(Fz_left=0.0, Fz_right=0.0, slip=0.0, tau_ratio=0.0, bad=set())
        self.n = 0

    def start_route(self, route, gap=0.5):
        """route = [("line", dist) | ("turn", dyaw), ...]; a rest `gap` [s] after each segment.  Returns the total duration [s]."""
        self.queue, self.gap, self.gap_left, self.pf, self.seg_err, self.last, self._pend = list(route), gap, 0.0, None, [], None, False
        T = 0.0
        for kind, a in route:
            T += (SW.trap_profile(a, self.sb.lim.v_max, self.sb.lim.a_max)[0] if kind == "line" else SW.trap_profile(a, self.sb.lim.w_max, self.sb.lim.alpha_max)[0]) + gap
        self.active = True
        return T

    def _advance(self):
        if self.pf is not None and self.pf.done():
            self.last, self.pf, self.gap_left, self._pend = self.pf, None, self.gap, True
        if self.pf is None and self.gap_left <= 0.0 and getattr(self, "_pend", False):
            self.seg_err.append(self.last.error())          # segment error AFTER the rest gap (P hold on the segment's goal)
            self._pend = False
        if self.pf is None and self.gap_left <= 0.0 and self.queue:
            kind, a = self.queue.pop(0)
            self.seg_idx, self.seg_kind = self.seg_idx + 1, kind
            self.pf = SW.PathFollower(self.sb)
            (self.pf.drive_line if kind == "line" else self.pf.turn_in_place)(a)

    def phase(self):
        """route phase label for the G0 monitor: seg<i>_<kind> while a segment runs, rest<i> in the gap after it."""
        if not self.active or self.seg_idx < 0:
            return "pre" if self.active else "post"
        return f"seg{self.seg_idx}_{self.seg_kind}" if self.pf is not None else f"rest{self.seg_idx}"

    def __call__(self, ep):
        m, d = ep.m, ep.d
        if self.active:
            self._advance()
            if self.pf is not None:
                tw = self.pf.step(DT)
            else:
                self.gap_left -= DT
                tw = self.last.hold_twist(DT) if getattr(self, "last", None) is not None else self.pf_zero()      # rest gap: P on the last goal
            self.sb.apply(tw, DT)
        else:
            self.sb.stop()
        pk = self.peak
        for s in C.SIDES:
            F, _ = ep.claw_wrench(s)
            pk[f"Fz_{s}"] = max(pk[f"Fz_{s}"], abs(float(F[2])))
        pk["slip"] = max(pk["slip"], float(self.sb.wheel_slip().max()))
        self.n += 1
        if self.n % 10 == 0:
            pk["bad"].update(bad_contacts(m, d, ep._ag, ep.obj if ep._loaded else None))


class G0Monitor:
    """G0 step hook (smoke-local, read-only): records per-step series during the base route and reduces them to MEASURED maxima with the time and the
    route phase of each maximum.  Registered AFTER BaseDrive, so each sample is the state after the step.  Series are post-processed in `finalize`.
    Acceleration filter: 0.05 s moving average (10 steps) of the finite-differenced velocity; the raw 5 ms value is reported as well."""
    SMOOTH = 10

    def __init__(self, ep, sb, drv, world, video=None):
        self.ep, self.sb, self.drv, self.world = ep, sb, drv, world
        m = ep.m
        self.base_id = m.body("base").id
        self.arm_g = ep._ag
        self.load_b = {ep.obj, *ep.cups}
        self.bm = BaseMonitors(m, sb, self.arm_g, self.load_b)          # base U flags (moved to monitors.py 11-09)
        self.robot_g, self.load_g, self.world_g, self.floor_g = self.bm.robot_g, self.bm.load_g, self.bm.world_g, self.bm.floor_g
        self.flagged = set()                               # base U flags already emitted (one entry per kind)
        self.R0 = ep.d.xmat[self.base_id].reshape(3, 3).copy()
        self.on = False
        self.rows, self.phases, self.contact_events, self.viol_log = [], [], {}, []
        self.rel0 = None
        self.tau_seen, self.tau_t, self.tau_ph = 0.0, 0.0, ""
        self._phase_ix = {}
        orig = ep._viol
        def _viol(kind, t, **kw):
            if self.on:
                self.viol_log.append(dict(kind=kind, t=round(float(t), 3), phase=self.drv.phase(), **{k: (round(float(v), 4) if isinstance(v, (int, float, np.number)) else v) for k, v in kw.items()}))
            return orig(kind, t, **kw)
        ep._viol = _viol
        self.video = video
        if video:
            self.rend = mujoco.Renderer(m, 480, 640)
            self.cam = mujoco.MjvCamera()
            p0 = sb.pose()
            self.cam.lookat[:] = [p0[0] - 0.5 * np.cos(p0[2]) , p0[1] - 0.5 * np.sin(p0[2]), 0.7]
            self.cam.distance, self.cam.azimuth, self.cam.elevation = 3.6, 135.0, -22.0
            self.frames, self.next_frame_t = [], 0.0
        ep.step_hooks.append(self)

    def start(self):
        ep, d, m = self.ep, self.ep.d, self.ep.m
        self.on = True
        self.t0 = ep.k * DT
        ref = d.site_xpos[m.site("obj_ref").id]
        self.rel0 = [(d.xmat[ep.obj].reshape(3, 3).T @ (d.xpos[b] - ref)) for b in ep.cups]

    def _flag(self, kind, t, **kw):
        """a base U flag: emitted once per kind through ep._viol (so it is in U_first / U_all like every carry monitor)."""
        if kind not in self.flagged:
            self.flagged.add(kind)
            self.ep._viol(kind, t, **kw)

    def _pid(self, name):
        return self._phase_ix.setdefault(name, len(self._phase_ix))

    def __call__(self, ep):
        if not self.on:
            return
        m, d = ep.m, ep.d
        t = ep.k * DT
        ref = d.site_xpos[m.site("obj_ref").id]
        Ro = d.xmat[ep.obj].reshape(3, 3)
        rel = [Ro.T @ (d.xpos[b] - ref) for b in ep.cups]
        slide = [1e3 * float(np.linalg.norm((r - r0)[:2])) for r, r0 in zip(rel, self.rel0)] or [0.0]
        off = sum(abs(r[0]) > self.world.tray_lx / 2 or abs(r[1]) > self.world.tray_ly / 2 or r[2] < -0.005 for r in rel)
        cup_tilt = [CA._tilt_deg(d.xmat[b].reshape(3, 3)) for b in ep.cups] or [0.0]
        Fz = [abs(float(ep.claw_wrench(s)[0][2])) for s in C.SIDES]
        v = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, ep.obj, v, 0)
        pose = self.sb.pose()
        Rb = d.xmat[self.base_id].reshape(3, 3) @ self.R0.T
        zc = Rb[:, 2]
        cy, sy = np.cos(pose[2]), np.sin(pose[2])
        fwd, lat = zc[0] * cy + zc[1] * sy, -zc[0] * sy + zc[1] * cy         # tilt of the base z axis along / across the robot x axis [rad]
        ph = self.drv.phase()
        tr = ep.track["tau_ratio"]
        if tr > self.tau_seen + 1e-12:
            self.tau_seen, self.tau_t, self.tau_ph = tr, t - self.t0, ph
        self.rows.append([t - self.t0, self._pid(ph), *pose, *self.sb._v, *v[3:], CA._tilt_deg(Ro), max(cup_tilt), max(slide), off, max(Fz), np.degrees(fwd), np.degrees(lat), np.degrees(np.arccos(np.clip(Rb[2, 2], -1, 1)))])
        self.bm.step(d, t, self._flag, base_tilt_deg=float(self.rows[-1][-1]))      # wheel_liftoff / base_world_contact / load_world_contact
        for c in d.contact[:d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if c.dist > 1e-4:
                continue
            for a, b in ((g1, g2), (g2, g1)):
                if b in self.world_g and b not in self.floor_g:
                    kind = "tray_mug_world" if a in self.load_g else ("arm_world" if a in self.arm_g else ("base_world" if a in self.robot_g else None))
                elif b in self.floor_g:
                    kind = "tray_mug_floor" if a in self.load_g else ("arm_floor" if a in self.arm_g else None)
                elif a in self.robot_g and b in self.load_g and not (a in self.arm_g) and not (m.geom_bodyid[b] == ep.obj):
                    kind = "base_mug"
                elif a in self.arm_g and b in self.load_g and m.geom_bodyid[b] in set(ep.cups):
                    kind = "arm_mug"
                else:
                    kind = None
                if kind:
                    key = (kind, m.geom(a).name or m.body(m.geom_bodyid[a]).name, m.geom(b).name or m.body(m.geom_bodyid[b]).name)
                    ev = self.contact_events.setdefault(key, dict(first_t=round(t - self.t0, 3), phase=ph, n=0, min_dist_mm=1e3 * float(c.dist)))
                    ev["n"] += 1
                    ev["min_dist_mm"] = min(ev["min_dist_mm"], 1e3 * float(c.dist))
                break
        if self.video and t - self.t0 >= self.next_frame_t:
            self.next_frame_t += 1.0 / 30.0
            self.rend.update_scene(d, self.cam)
            self.frames.append(self.rend.render().copy())

    def finalize(self, lim, F_cap, tau_limit_note):
        A = np.array(self.rows)
        names = {v: k for k, v in self._phase_ix.items()}
        t, ph = A[:, 0], A[:, 1].astype(int)
        pose, cmd, tray_v = A[:, 2:5], A[:, 5:8], A[:, 8:11]
        k = self.SMOOTH
        ker = np.ones(k) / k

        def sm(x):
            return np.stack([np.convolve(x[:, i], ker, mode="same") for i in range(x.shape[1])], 1)

        def peak(x, thr=None):
            i = int(np.argmax(x))
            return dict(max=float(x[i]), t=round(float(t[i]), 3), phase=names[ph[i]])

        yaw = np.unwrap(pose[:, 2])
        v_w = np.diff(pose[:, :2], axis=0) / DT                      # base-centre world velocity
        w = np.diff(yaw) / DT
        a_raw = np.linalg.norm(np.diff(v_w, axis=0) / DT, axis=1)
        al_raw = np.abs(np.diff(w) / DT)
        vs, ws = sm(v_w), sm(w[:, None])[:, 0]
        a_s = np.linalg.norm(np.diff(vs, axis=0) / DT, axis=1)
        al_s = np.abs(np.diff(ws) / DT)
        # commanded: the rate-limited body twist command (sb._v, body frame) -> world frame with the measured yaw
        cw = np.stack([cmd[:, 0] * np.cos(yaw) - cmd[:, 1] * np.sin(yaw), cmd[:, 0] * np.sin(yaw) + cmd[:, 1] * np.cos(yaw)], 1)
        ca = np.linalg.norm(np.diff(cw, axis=0) / DT, axis=1)
        cal = np.abs(np.diff(cmd[:, 2]) / DT)
        def pk2(x, off):                                                # series shorter by `off` samples
            i = int(np.argmax(x)); return dict(max=float(x[i]), t=round(float(t[i + off]), 3), phase=names[ph[i + off]])
        jerk_base = np.abs(np.diff(sm(np.diff(vs, axis=0) / DT), axis=0) / DT)
        jb = np.linalg.norm(jerk_base, axis=1)
        a_tray = np.diff(tray_v, axis=0) / DT
        a_tray_s = sm(a_tray)
        j_tray = np.linalg.norm(np.diff(a_tray_s, axis=0) / DT, axis=1)
        out = dict(
            limits=dict(v=lim.v_max, a=lim.a_max, w=lim.w_max, alpha=lim.alpha_max),
            Fz_over_Fcap=dict(peak(A[:, 15] / F_cap), F_cap=F_cap),
            tau_ratio=dict(max=self.tau_seen, t=round(self.tau_t, 3), phase=self.tau_ph, note=tau_limit_note),
            tray_tilt_deg=peak(A[:, 11]), mug_tilt_max_deg=peak(A[:, 12]), mug_slide_max_mm=peak(A[:, 13]),
            mugs_off_tray_max=int(A[:, 14].max()), mugs_off_tray_end=int(A[-1, 14]),
            base_acc_cmd_lin=pk2(ca, 1), base_acc_cmd_ang=pk2(cal, 1),
            base_acc_ach_lin_smooth=pk2(a_s, 2), base_acc_ach_ang_smooth=pk2(al_s, 2),
            base_acc_ach_lin_raw5ms=pk2(a_raw, 2), base_acc_ach_ang_raw5ms=pk2(al_raw, 2),
            base_jerk_lin_smooth=pk2(jb, 3), tray_acc_smooth=pk2(np.linalg.norm(a_tray_s, axis=1), 1), tray_jerk_smooth=pk2(j_tray, 2),
            base_pitch_deg_maxabs=peak(np.abs(A[:, 16])), base_roll_deg_maxabs=peak(np.abs(A[:, 17])), base_tilt_deg=peak(A[:, 18]),
            contacts={"|".join(k): v for k, v in self.contact_events.items()},
            U_all=self.viol_log, n_steps=len(A), t_route_s=float(t[-1]),
            filter="achieved = 0.05 s moving average of finite-differenced velocity of the module-pivot centre; commanded = finite difference of the rate-limited twist command (swerve._v); jerk = 0.05 s smoothed d(acc)/dt")
        return out


def arm_dev(ep, q_ref):
    return max(float(np.abs(ep.d.qpos[ep.addr[s]] - q_ref[s]).max()) for s in C.SIDES)


def _prep(ep, sb, loaded):
    ep._ag = arm_geoms(ep.m)
    ep._loaded = loaded
    drv = BaseDrive(ep, sb)
    return drv


def _noload_ep(dz):
    w = SC.TrayWorld(n_cups=6)
    ep = make_ep(w)
    m, d = ep.m, ep.d
    for s in C.SIDES:                         # no load: let go and move the tray + mugs out of the way (arms stay at the hold pose, FL-MPC holds q)
        ep.weld(s, False)
    ep.ctl.on = False
    for b in [ep.obj] + ep.cups:
        j = m.body_jntadr[b]
        d.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 3] = [0.0, 2.5, 0.05 + 0.1 * b]
    mujoco.mj_forward(m, d)
    sb = SW.SwerveBase(m, d)
    drv = _prep(ep, sb, False)
    ep.run_phases([("move", (0, 0, dz), 1.0), ("hold", 0.5)])      # same lift as the loaded smoke
    return ep, sb, drv


def smoke_noload(dz=0.02):
    out = {}
    t0 = time.time()
    for name, route in (("rest5s", []), ("route", [("line", -BS.RETREAT_M), ("turn", np.pi), ("line", BS.ADVANCE_M)]),
                        ("back0.5", [("line", -0.5)]), ("turn_at_pick_pose", [("turn", np.pi)])):
        ep, sb, drv = _noload_ep(dz)
        m, d = ep.m, ep.d
        q_ref = {s: d.qpos[ep.addr[s]].copy() for s in C.SIDES}
        pa = sb.pose()
        drv.reset(); ep.reset_track()
        T = drv.start_route(route) if route else 5.0
        if not route:
            drv.active = False
        ep.run_phases([("hold", T + 0.5)])
        drv.active = False
        pb = sb.pose()
        r = dict(t_total=T + 0.5, end_pose=pb.round(4).tolist(), start_pose=pa.round(4).tolist(),
                 moved_mm=1e3 * float(np.linalg.norm(pb[:2] - pa[:2])), dyaw_deg=float(np.degrees(pb[2] - pa[2])),
                 peak_slip_mps=drv.peak["slip"], contacts=sorted(drv.peak["bad"]), arm_q_dev_rad=arm_dev(ep, q_ref),
                 tau_ratio_max=ep.track["tau_ratio"])
        if route:
            r.update(seg_pos_err_mm=[round(1e3 * e[0], 2) for e in drv.seg_err], seg_yaw_err_deg=[round(float(np.degrees(e[1])), 3) for e in drv.seg_err])
        out[name] = r
    out["wall_s"] = time.time() - t0
    return out


def run_loaded(world, base="swerve", turn=True, T_lift=1.0, dz=0.02, level="L1", g0=False, video=None):
    """tray welded at the hold pose, lifted 2 cm (T_lift), 0.5 s hold; then (turn) the base turns 180 deg + 1 s settle (swerve) -- or the same hold time
    with the base still (base='fixed', or turn=False).  Returns the metrics of the post-lift window."""
    ep = make_ep(world, base=base)
    sb = SW.SwerveBase(ep.m, ep.d, level_limits(level)) if base == "swerve" else None
    out = dict(F_cap=ep.F_cap, mass_total=ep.truth0["total_mass"], fills_kg=list(map(float, world.layout()[1])) if hasattr(world, "layout") else None, contents_kg=_contents_kg(world))
    if sb is not None:
        drv = _prep(ep, sb, True)
    ep.run_phases([("move", (0, 0, dz), T_lift), ("hold", 0.5)])
    out["lift_viol"] = ep.track["viol"]
    ep.reset_track()
    if sb is not None:
        drv.reset()
    q_ref = {s: ep.d.qpos[ep.addr[s]].copy() for s in C.SIDES}
    F_mean = {s: [] for s in C.SIDES}
    ep.step_hooks.append(lambda e: [F_mean[s].append(e.claw_wrench(s)[0].copy()) for s in C.SIDES])
    p0 = sb.pose() if sb is not None else None
    mon = None
    if g0 and sb is not None:
        mon = G0Monitor(ep, sb, drv, world, video=video)
        mon.start()
    if sb is not None and turn:
        T = drv.start_route([("line", -BS.RETREAT_M), ("turn", np.pi), ("line", BS.ADVANCE_M)])
        ep.run_phases([("hold", T + 0.5)])
        drv.active = False
        out.update(T=T, seg_pos_err_mm=[1e3 * e[0] for e in drv.seg_err], seg_yaw_err_deg=[float(np.degrees(e[1])) for e in drv.seg_err])
    else:
        ep.run_phases([("hold", 2.0)])
    tr = ep.track
    pk = drv.peak if sb is not None else {}
    d, m = ep.d, ep.m
    last = slice(-100, None)
    out.update(
        Fz_peak=dict(left=pk.get("Fz_left", float("nan")), right=pk.get("Fz_right", float("nan"))),
        Fz_over_Fcap=max(pk.get("Fz_left", 0.0), pk.get("Fz_right", 0.0)) / ep.F_cap if pk else float("nan"),
        Fz_track_max=tr["Fz_max"], Fz_over_Fcap_track=tr["Fz_max"] / ep.F_cap,
        F_end_mean={s: np.mean(F_mean[s][last], axis=0).round(4).tolist() for s in C.SIDES},
        tau_ratio_max=tr["tau_ratio"], n_sat=tr["n_sat"], obj_tilt_max_deg=tr["obj_tilt_max"], cup_tilt_max_deg=tr["cup_tilt_max"],
        spill_ratio=tr["spill_ratio"], viol=tr["viol"], twist_max=tr["twist_max"], F_int_max=tr["F_int_max"],
        arm_q_dev_rad=arm_dev(ep, q_ref), contacts=sorted(pk.get("bad", [])), peak_slip_mps=pk.get("slip"))
    if sb is not None:
        pb = sb.pose()
        out["base_end_pose"] = pb.round(4).tolist(); out["base_start_pose"] = p0.round(4).tolist()
    rel = [d.xmat[ep.obj].reshape(3, 3).T @ (d.xpos[b] - d.site_xpos[m.site("obj_ref").id]) for b in ep.cups]
    out["mugs_off_tray"] = int(sum(abs(r[0]) > world.tray_lx / 2 or abs(r[1]) > world.tray_ly / 2 or r[2] < -0.005 for r in rel)) if ep.cups else None
    out["obj_end_xyz"] = d.xpos[ep.obj].round(4).tolist()
    if mon is not None:
        n = BS.RETREAT_M + BS.ADVANCE_M          # nominal end of R1: start + R(yaw0) [-(retreat+advance), 0], yaw0 + pi
        nom = np.array([p0[0] - n * np.cos(p0[2]), p0[1] - n * np.sin(p0[2]), p0[2] + np.pi])
        dyaw = (pb[2] - nom[2] + np.pi) % (2 * np.pi) - np.pi
        e_m, e_deg = float(np.linalg.norm(pb[:2] - nom[:2])), float(np.degrees(dyaw))
        if e_m > ROUTE_TOL_M or abs(e_deg) > ROUTE_TOL_DEG:
            mon._flag("route_unfinished", ep.k * DT, err_mm=1e3 * e_m, yaw_err_deg=e_deg, tol=f"{1e3 * ROUTE_TOL_M:.0f} mm / {ROUTE_TOL_DEG} deg ASSUMED")
        out["g0"] = dict(level=level, route_end_err_mm=1e3 * e_m, route_end_yaw_err_deg=e_deg, strict_claw=ep.strict_claw,
                         U_first=tr["viol"], U_base_flags=sorted(mon.flagged), wheels_off_max=mon.bm.max_off, **mon.finalize(sb.lim, ep.F_cap, "ep.track tau_ratio = commanded joint torque / config.tau_limit() (100 Nm per joint, USER 11-05)"))
        if video:
            import imageio
            imageio.mimwrite(video, mon.frames, fps=30, macro_block_size=1)
    return out


def _contents_kg(w):
    return float(sum(w.layout()[1])) if hasattr(w, "n_cups") else float(w.contents_kg)


def task_world(task, seed=None):
    """smoke world of a table-behind task: tray6 -> TrayWorld (6 mugs), box -> BoxWorld(shelf=False) (task box_behind: no wall shelf).
    seed None = light (tray: default fills; box: the prior low end); else probe._world(task, seed) (the e2e world list)."""
    from dataclasses import replace
    if task == "tray6":
        return SC.TrayWorld(n_cups=6) if seed is None else PR._world("tray6", seed)
    if task == "box":
        from ...stack import tasks as TK
        w = SC.BoxWorld(shelf=False, contents_kg=TK.prior_box("box")[0]) if seed is None else PR._world("box", seed)
        return replace(w, **dict(TK.task("box_behind").world_kw))
    raise ValueError(task)


def behind_world(task, i):
    """THE world list of the table-behind tasks, for BOTH methods (same_worlds ours + theirs): (seed, world) of world i of `task` (tray6_behind | box_behind) =
    probe._seed("e2e", base task, i) and task_world(base task, seed) (full prior box; box_behind: shelf=False)."""
    bt = task[:-len("_behind")] if task.endswith("_behind") else task
    seed = PR._seed("e2e", bt, i)
    return seed, task_world(bt, seed)


def heaviest_seed(n=40, task="tray6"):
    best = (-1.0, None)
    for sd in range(n):
        w = PR._world(task, sd)
        f = _contents_kg(w)
        if f > best[0]:
            best = (f, sd)
    return best


DZ = float(os.environ.get("SMOKE_DZ", "0.02"))     # lift height [m] before the turn

def _opt(name, default=None):
    """pop `--name value` from sys.argv"""
    if name in sys.argv:
        i = sys.argv.index(name)
        v = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return v
    return default


if __name__ == "__main__":
    LEVEL = _opt("--level", "L1")                      # {L1, L2, L3}; default L1 = today's limits
    TASK = _opt("--task", "tray6")                     # tray6 | box
    assert LEVEL in LEVELS, LEVEL
    G0 = "--g0" in sys.argv
    if G0:
        sys.argv.remove("--g0")
    VIDEO = _opt("--video")
    which = sys.argv[1] if len(sys.argv) > 1 else "nl"
    if which == "nl":
        print(json.dumps(smoke_noload(DZ), indent=1, default=float))
    elif which == "seeds":
        print(heaviest_seed(task=TASK))
    elif which in ("light", "heavy", "heavy_still", "light_still", "heavy_fixed", "light_fixed"):
        kind = which.split("_")[0]
        mode = which.split("_")[1] if "_" in which else "turn"
        w = task_world(TASK, None if kind == "light" else (int(sys.argv[2]) if len(sys.argv) > 2 else heaviest_seed(task=TASK)[1]))
        print(json.dumps(run_loaded(w, base=("fixed" if mode == "fixed" else "swerve"), turn=(mode == "turn"), dz=DZ, level=LEVEL, g0=G0, video=VIDEO), indent=1, default=float))
