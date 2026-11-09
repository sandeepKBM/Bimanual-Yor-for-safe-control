"""nero/base/smoke.py -- smokes of the swerve base (SIMULATION ONLY, CPU).

  python -m world_model.v2.nero.base.smoke nl       # (a) no load: rest drift, turn 180 deg, drive 0.5 m
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


def make_ep(world, base="swerve", second_table=True):
    with BS.use_swerve_scene(second_table=second_table, base=base):
        ep = CA.CarryEpisode(world)
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
            self.pf = SW.PathFollower(self.sb)
            (self.pf.drive_line if kind == "line" else self.pf.turn_in_place)(a)

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


def run_loaded(world, base="swerve", turn=True, T_lift=1.0, dz=0.02):
    """tray welded at the hold pose, lifted 2 cm (T_lift), 0.5 s hold; then (turn) the base turns 180 deg + 1 s settle (swerve) -- or the same hold time
    with the base still (base='fixed', or turn=False).  Returns the metrics of the post-lift window."""
    ep = make_ep(world, base=base)
    sb = SW.SwerveBase(ep.m, ep.d) if base == "swerve" else None
    out = dict(F_cap=ep.F_cap, mass_total=ep.truth0["total_mass"], fills_kg=list(map(float, world.layout()[1])))
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
    out["mugs_off_tray"] = int(sum(abs(r[0]) > world.tray_lx / 2 or abs(r[1]) > world.tray_ly / 2 or r[2] < -0.005 for r in rel))
    out["obj_end_xyz"] = d.xpos[ep.obj].round(4).tolist()
    return out


def heaviest_seed(n=40):
    best = (-1.0, None)
    for sd in range(n):
        w = PR._world("tray6", sd)
        f = float(sum(w.layout()[1]))
        if f > best[0]:
            best = (f, sd)
    return best


DZ = float(os.environ.get("SMOKE_DZ", "0.02"))     # lift height [m] before the turn

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "nl"
    if which == "nl":
        print(json.dumps(smoke_noload(DZ), indent=1, default=float))
    elif which == "seeds":
        print(heaviest_seed())
    elif which in ("light", "heavy", "heavy_still", "light_still", "heavy_fixed", "light_fixed"):
        kind = which.split("_")[0]
        mode = which.split("_")[1] if "_" in which else "turn"
        if kind == "light":
            w = SC.TrayWorld(n_cups=6)
        else:
            w = PR._world("tray6", int(sys.argv[2]) if len(sys.argv) > 2 else heaviest_seed()[1])
        print(json.dumps(run_loaded(w, base=("fixed" if mode == "fixed" else "swerve"), turn=(mode == "turn"), dz=DZ), indent=1, default=float))
