"""tamp/seg_gate.py -- SegmentGatekeeper: execute the planner's timed route segments under the gatekeeper's unchanged safety layer.  SIMULATION ONLY.

Subclass of gatekeeper.gate.Gatekeeper.  While a route is active (begin_route) the operator command is IGNORED for the base twist:
  * at each segment boundary (first tick, then LEAD_S before the committed nominal ends) the next segment is REBUILT from the ACTUAL state (measured twist, ctrl
    pose, arclength projection) with the same function the planner used (segments.make_segment), re-checked with check1 + check2 under the CURRENT belief
    (all people, set-valued), and COMMITTED (Gatekeeper._commit).  The base twist is the committed Traj's profile (Gatekeeper.filter_base, unchanged).
  * every gatekeeper step (k_ticks) between boundaries runs the T1 KEEP rule (falsification monitor, as gate.py keep_rule="monitor"): if the model is falsified the
    internal Shield takes over (regime 2), exactly as in the plain gatekeeper.  No new safety logic: segments only replace WHICH candidate is committed.
  * a boundary re-check that fails REFUSES the segment: the committed trajectory (its nominal remainder and cup-safe backup) keeps braking, the base gatekeeper takes
    over with a zero command (hold-zero = backup to rest), and seg_state() == "refused" so the executor can return status "segment_refused" to the run_tamp ladder.
After the last segment (the driven part ends at rest, end_skip m before the goal) seg_state() == "done" and the executor's own final approach runs through the
plain gatekeeper.  Optional `downshift`: on a refusal try the slower SPEEDS before refusing (default off: the refusal is the plan/execution mismatch we measure).
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..gatekeeper.belief import GKConfig, build_belief
from ..gatekeeper.gate import Gatekeeper
from ..gatekeeper.monitor import falsification_monitor, observation_discs
from ..teleop.sim import TICK
from . import segments as SEG
from ..gatekeeper.checks import _ev, _pe
from ..gatekeeper.ready import rest_poses
from ..gatekeeper.traj import backup_from
from ..teleop import geom as GM
from . import plan_monitor as PMON
from .plan_monitor import PlanMonitor
from . import hop as HOP

REST_V = 0.03


def make_seg_filter(**kw):
    """the gatekeeper config the executor runs (world.RUNTIME_GK) as a SegmentGatekeeper."""
    from .world import RUNTIME_GK
    import dataclasses
    return SegmentGatekeeper(gk=dataclasses.replace(GKConfig(), **RUNTIME_GK), **kw)


class SegmentGatekeeper(Gatekeeper):
    def __init__(self, cfg=None, gk=None, perc=None, *, downshift: bool = False, speeds=SEG.SPEEDS):
        super().__init__(cfg, gk, perc)
        self.downshift, self.speeds = bool(downshift), tuple(speeds)
        self.sr = None                      # active route state (dict) or None
        self.seg_log = []                   # one dict per finished route
        self.n_seg_boundaries = 0
        # opt-in plan-prediction monitor (gk.plan_monitor == "on"); None = off = no code path below is reached
        g = self.gk
        self.pm = PlanMonitor(g.plan_eps_p, g.plan_eps_v, g.plan_r_rel, g.plan_confirm) if g.plan_monitor == "on" else None
        self.pm_L, self.pm_margin = 2.0, float(g.plan_conf_margin)       # judgement latency L [s] the timing test assumes (the ladder sets it) and t_margin
        self._coast = False                 # coast mode: follow ONLY the committed trajectory (no new commits of any kind)
        self.plan_flag_log = []             # one dict per flag raised
        self.hop_log = []                   # opt-in hop primitive (tamp/hop.py): one dict per hop commit attempt (empty = never used)
        self.tube_log = []                  # opt-in tracked-tube ranking term (gk.c2_ready_tube_h > 0): one dict per boundary commit (empty = term off)

    # ------------------------------------------------------------------------------------------ plan-prediction monitor API (opt-in)
    def set_plan_prediction(self, people, t_obs):
        """hand the monitor the people prediction the just-solved plan was certified under: [(c, v_hat), ...] observed at sim time t_obs.  No-op when the monitor is off."""
        if self.pm is not None:
            self.pm.set_prediction(people, t_obs)

    def set_plan_latency(self, L):
        """judgement latency L [s] used by the timing test at the next flag."""
        self.pm_L = float(L)

    @property
    def plan_continue(self):
        """a flag was raised and the old route keeps being driven until t_ready ("continue" decision)."""
        return self.sr is not None and self.sr.get("pf") == "continue"

    def set_t_ready(self, t_ready):
        if self.sr is not None:
            self.sr["t_ready"] = float(t_ready)

    def _chain(self, sim, bel, t_now, horizon):
        """predicted OLD-route motion r(t) from the ACTUAL state: the committed nominal remainder, then segments chained with next_certified_segment (the function the
        boundaries run) at the route's current v_max under `bel`, nominal parts only, until the end of the route / a refusal / `horizon` s; then the cup-safe brake and rest.
        Read-only (commits nothing).  -> [(t, pose, v, phase)] with phase in {nominal, brake, rest}."""
        sr, gk = self.sr, self.gk
        h = gk.h
        pose = tuple(float(x) for x in np.asarray(bel.pose_wall, float))
        v = np.asarray(sim.base.v, float).copy()
        t = float(t_now)
        S = [(t, pose, v.copy(), "nominal")]

        def push(Vs, phase):
            nonlocal pose, v, t
            for k in range(1, len(Vs)):
                pose = SEG._integrate(pose, Vs[k - 1], Vs[k], h)
                v = np.asarray(Vs[k], float).copy()
                t += h
                S.append((t, pose, v.copy(), phase))

        if self.traj is not None:
            rem = self.traj.shifted(t_now)
            n0 = min(rem.N, max(0, int(round(max(rem.t_sw, 0.0) / h))))
            if n0 > 0:
                push(rem.V[:n0 + 1], "nominal")
        s = sr["path"].project(np.asarray(pose[:2], float), sr["s"])
        for _ in range(400):
            if t - t_now >= horizon or (sr["s_stop"] - s <= SEG.STOP_EPS + 0.02 and float(np.abs(v).max()) <= REST_V):
                break
            tr, res, _m, _row, _T, _why = SEG.next_certified_segment(sr["path"], sr["reverse"], sr["v_max"], pose, v, s, bel, gk, sr["s_stop"], offset=sr["offset"])
            if tr is None:
                break
            n = max(1, int(round(res["t_sw"] / h)))
            push(res["V"][:n + 1], "nominal")
            s = sr["path"].project(np.asarray(pose[:2], float), s)
            if res["final"]:
                break
        if float(np.abs(v).max()) > 1e-9:
            bk = backup_from(np.array(v), bel, gk)
            push(bk.V, "brake")
        while t - t_now < horizon:
            t += 0.25
            S.append((t, pose, np.zeros(3), "rest"))
        return S

    def _plan_decide(self, sim, bel, t_now, L, t_ready=None):
        """the timing test: continue driving the old route until t_ready = t_now + L iff  t_ready + t_stop + margin < t_conf.
        t_conf: earliest time the TRACKED tube (all people in the belief: conservative superset of the relevant ones) reaches within d_person of the footprint along the predicted
        old-route motion; t_stop: duration of the cup-safe brake (gatekeeper.traj.backup_from) from the predicted state at t_ready."""
        gk = self.gk
        t_ready = t_now + L if t_ready is None else float(t_ready)
        S = self._chain(sim, bel, t_now, max(10.0, t_ready - t_now + 3.0))
        pose, v, phase = PMON.state_at(S, t_ready)
        bk = backup_from(np.array(v, float), bel, gk)
        t_stop = 0.0 if float(np.abs(v).max()) <= 1e-9 else float(bk.t_rest if bk.t_rest is not None else bk.N * bk.h)
        n = len(bel.people)
        vel = bel.people_vel if len(bel.people_vel) == n else tuple(np.zeros(2) for _ in range(n))
        t_conf = PMON.conflict_time(S, bel.people, vel, [GM.PERSON_CLEAR_R + _pe(bel, i) for i in range(n)], [_ev(bel, i) for i in range(n)], t_now, bel.d_person, bel.fp)
        ok = (t_ready + t_stop + self.pm_margin) < t_conf
        return dict(decision="continue" if ok else "brake", t_conf=t_conf, t_ready=t_ready, t_stop=t_stop, margin=self.pm_margin, L=float(t_ready - t_now),
                    start_kind=phase, start_pose=[float(x) for x in pose], start_v=[float(x) for x in v], t_stop_source="gatekeeper.traj.backup_from(v at t_ready).t_rest")

    def predict_old_route(self, sim, dt):
        """state the old route (chained certified segments, current belief) reaches dt s from the next filter call -> dict(pose, v, kind, dt); the "continue" start state."""
        t_now = self.n_calls * TICK
        bel = self.last_bel
        S = self._chain(sim, bel, t_now, max(10.0, dt + 3.0))
        pose, v, ph = PMON.state_at(S, t_now + dt)
        return dict(pose=tuple(pose), v=np.asarray(v, float), kind=ph, dt=float(dt))

    def coast(self, on: bool):
        """coast mode on: gk_step only runs the T1 KEEP rule on the committed trajectory (no plain-mode candidates, no segment boundaries, no commits)."""
        self._coast = bool(on)

    # ------------------------------------------------------------------------------------------ hop primitive (opt-in, tamp/hop.py)
    def hop_plan(self, sim, kinds=HOP.HOP_DIRS, durations=HOP.HOP_DURATIONS, min_m=HOP.HOP_MIN_M, route_path=None, back_target=None, compass=False):
        """choose the hop (first admissible direction, longest admissible duration) from the ACTUAL state under the CURRENT belief.  Read-only: commits nothing.
        -> HOP.choose_hop dict (+ "bel") or None.  Needs the base at rest and no active route."""
        if self.sr is not None or float(np.abs(sim.base.v).max()) > REST_V:
            return None
        bel = self._bel_now(sim, self.n_calls * TICK)
        r = HOP.choose_hop(bel, self.gk, kinds, durations, min_m, route_path, back_target, compass)
        return None if r is None else dict(r, bel=bel)

    def hop_commit(self, sim, d, t_h, kind="hop", min_m=HOP.HOP_MIN_M):
        """T1 execution of a hop: rebuild it from the ACTUAL state (rest, current yaw), re-run check1 + check2 under a FRESH belief and COMMIT it (Gatekeeper._commit);
        coast mode is switched on so only the committed trajectory (hop, then its rest) is followed -- the caller runs ticks until rest and calls coast(False).
        A failed re-check commits nothing and the base does not move.  -> record dict (ok, why, ...)."""
        t = self.n_calls * TICK
        rec = dict(t=float(t), kind=kind, t_h=float(t_h), d=[float(x) for x in d], ok=False, why="")
        self.hop_log.append(rec)
        if self.sr is not None or float(np.abs(sim.base.v).max()) > REST_V:
            rec["why"] = "not_at_rest"
            return rec
        bel = self._bel_now(sim, t)
        yaw = float(np.asarray(bel.pose_wall, float)[2])
        a_r = HOP.hop_a_r(bel, self.gk, HOP.body_dir(d, yaw), t_h)
        tr = HOP.hop_traj(d, yaw, t_h, a_r, self.gk.h, t)
        disp = HOP.displacement(tr)
        rec.update(a_r=a_r, disp_planned=disp, pose_before=[float(x) for x in sim.pose()])
        if disp < min_m:
            rec["why"] = "below_min_displacement"
            return rec
        ok, info = HOP.certify(tr, bel, self.gk)
        rec.update(c1=info["c1"], c2=info["c2"], margin=info["margin"])
        if not ok:
            rec["why"] = info["why"]
            return rec
        self._commit(tr, tr.T_S, 0.0, bel, t)
        self.coast(True)
        rec.update(ok=True, n_people=len(bel.people))
        return rec

    def predict_state(self, sim, dt):
        """state the COMMITTED trajectory reaches dt seconds from now (the next filter call), replicating filter_base tick by tick (u_at(t + TICK), slew-clipped to a_hw).
        -> dict(pose (x, y, yaw), v (body twist), kind in {nominal, brake, rest}, dt).  Needs self.traj (the committed trajectory)."""
        t_now = self.n_calls * TICK
        rem = self.traj.shifted(t_now)
        pose = tuple(float(x) for x in sim.pose())
        v = np.asarray(sim.base.v, float).copy()
        m = int(round(float(dt) / TICK))
        for j in range(1, m + 1):
            u = np.clip(rem.u_at(t_now + j * TICK), v - sim.a_hw * TICK, v + sim.a_hw * TICK)
            pose = SEG._integrate(pose, v, u, TICK)
            v = u
        el = m * TICK
        tr = rem.t_rest
        if el <= rem.t_sw + 1e-9:
            kind = "nominal"
        elif tr is None or el < tr - 1e-9:
            kind = "brake"
        else:
            kind = "rest"
        return dict(pose=tuple(float(x) for x in pose), v=np.asarray(v, float), kind=kind, dt=el)

    # ------------------------------------------------------------------------------------------ route API (executor)
    def begin_route(self, route):
        """start driving `route` (a Route with segments).  The first segment is built on the next filter call."""
        planned = route.segments or ()
        if self.pm is not None:
            self.pm.reset_debounce()
        self.sr = dict(path=SEG.Path(route.pts), reverse=bool(route.reverse), v_max=float(route.v_max), v_plan=float(route.v_max),
                       s_stop=float(route.s_stop), s=0.0, k=0, status="active", t_next=-math.inf, t_end=None, final=False, n_planned=len(planned),
                       n_admitted=0, n_refused=0, reason="", source=route.source, tube_hist={}, speeds=[float(route.v_max)], planned_margin=[float(sg.margin) for sg in planned],
                       exec_margin=[], t_begin=None, offset=0.0, ready_hist={}, n_offset_changes=0, offsets_used=[0.0])

    def seg_state(self):
        """"idle" (no route) | "active" | "done" | "refused" | "plan_falsified" (plan-prediction monitor flagged this route; also while a "continue" decision keeps driving it)."""
        if self.sr is None:
            return "idle"
        return "plan_falsified" if self.sr.get("pf") is not None else self.sr["status"]

    def end_route(self, outcome=None):
        """finish bookkeeping for the active route (idempotent); appends to seg_log."""
        sr = self.sr
        if sr is None:
            return None
        st = outcome or sr["status"]
        rec = dict(source=sr["source"], v_planned=sr["v_plan"], speeds_used=sr["speeds"], n_segments_planned=sr["n_planned"], n_segments_admitted=sr["n_admitted"],
                   n_refused=sr["n_refused"], status=st, reason=sr["reason"], s_reached=round(float(sr["s"]), 3), s_stop=round(sr["s_stop"], 3),
                   planned_margin_min=min(sr["planned_margin"]) if sr["planned_margin"] else None,
                   exec_margin_min=min(sr["exec_margin"]) if sr["exec_margin"] else None,
                   ready_hist=dict(sr["ready_hist"]), n_offset_changes=sr["n_offset_changes"], offsets_used=list(sr["offsets_used"]))
        if self.gk.c2_ready_tube_h > 0.0:
            rec["tube_hist"] = dict(sr["tube_hist"])
        self.seg_log.append(rec)
        self.sr = None
        return rec

    # ------------------------------------------------------------------------------------------ belief (as gate.gk_step)
    def _bel_now(self, sim, t):
        shield = self._shield(sim)
        fp = shield.footprint_of(sim)
        self.fp_now = fp
        if self.tracker:
            if self._proxy is None:
                self.perceive(sim)
            tracks = self._track_list()
            self._audit(sim, tracks, t)
            bel = build_belief(sim, self.cfg, self.gk, fp=fp, docks=self.docks, tracks=tracks)
        else:
            pv, ev = self._people_vel(sim, t)
            bel = build_belief(sim, self.cfg, self.gk, fp=fp, docks=self.docks, people_vel=pv, e_v=ev)
        self._obs_now = observation_discs(bel, self.gk)
        if self._pf is not None:
            bel = self._filter_belief(bel, tracks if self.tracker else None, t)
        self.last_bel = bel
        return bel

    # ------------------------------------------------------------------------------------------ boundaries
    def _boundary(self, sim, t):
        sr = self.sr
        t0 = time.perf_counter()
        bel = self._bel_now(sim, t)
        pose = np.asarray(bel.pose_wall, float)
        v0 = np.asarray(sim.base.v, float).copy()
        s0 = sr["path"].project(pose[:2], sr["s"])
        sr["s"] = s0
        if sr["t_begin"] is None:
            sr["t_begin"] = t
        if sr["s_stop"] - s0 <= SEG.STOP_EPS + 0.02 and float(np.abs(v0).max()) <= REST_V:
            sr["status"], sr["t_next"] = "done", math.inf
            return
        if sr.get("pf") == "continue" and self._reeval(sim, bel, t, sr):
            return
        self.n_seg_boundaries += 1
        speeds = [sr["v_max"]] + ([v for v in self.speeds if v < sr["v_max"] - 1e-9] if self.downshift else [])
        why_all = []
        for v in speeds:
            # SAME function as the planner: T_S in gk.t_switch (longest first), build from the ACTUAL state, check1 + check2 under the CURRENT belief
            tr, res, m, row, T, why = SEG.next_certified_segment(sr["path"], sr["reverse"], v, pose, v0, s0, bel, self.gk, sr["s_stop"], offset=sr["offset"], site="gate")
            if tr is None:
                why_all.append(f"v={v}: {why}")
                continue
            tr.t0 = t
            sr["exec_margin"].append(float(m))
            if v != sr["v_max"]:
                sr["v_max"] = v
                sr["speeds"].append(float(v))
            self._commit(tr, T, 0.0, bel, t)
            o_new = float(res["offset"])
            if o_new != sr["offset"]:
                sr["n_offset_changes"] += 1
                sr["offsets_used"].append(o_new)
            sr["offset"] = o_new
            key = "none" if res.get("ready") is None else str(res["ready"])
            sr["ready_hist"][key] = sr["ready_hist"].get(key, 0) + 1
            if "tube_off" in res:                                           # tube term active (gk.c2_ready_tube_h > 0)
                k2 = "off_tube" if res["tube_off"] else "on_tube"
                sr["tube_hist"][k2] = sr["tube_hist"].get(k2, 0) + 1
                sr["tube_hist"]["changed"] = sr["tube_hist"].get("changed", 0) + int(bool(res.get("tube_changed")))
                rp = rest_poses(tr, bel)
                self.tube_log.append(dict(t=float(t), T_S=float(T), tube_off=bool(res["tube_off"]), changed=bool(res.get("tube_changed")),
                                          rest_pose=None if rp is None else [float(x) for x in rp[0]], t_rest=None if rp is None else float(rp[2]),
                                          n_people=len(bel.people)))
            sr["k"] += 1
            sr["n_admitted"] += 1
            sr["final"] = bool(res["final"])
            lead = min(SEG.LEAD_S, 0.5 * res["t_sw"])
            sr["t_next"] = math.inf if res["final"] else t + max(res["t_sw"] - lead, TICK)
            sr["t_end"] = t + res["t_sw"]
            self.step_ms.append((time.perf_counter() - t0) * 1e3)
            return
        sr["n_refused"] += 1
        sr["status"], sr["t_next"] = "refused", math.inf
        sr["reason"] = f"segment {sr['k']} at s={s0:.2f} m refused: " + " | ".join(why_all)

    # ------------------------------------------------------------------------------------------ gatekeeper hooks
    def filter_base(self, sim, u_op):
        sr = self.sr
        if sr is not None and sr["status"] == "active":
            t = self.n_calls * TICK
            if t >= sr["t_next"] - 1e-9:
                self._boundary(sim, t)
            elif sr["final"] and t >= sr["t_end"] and float(np.abs(sim.base.v).max()) <= REST_V:
                sr["status"] = "done"
        return super().filter_base(sim, u_op)

    def _coast_step(self, sim, t):
        """coast mode: T1 KEEP rule on the committed trajectory only (falsification monitor); nothing is ever committed.  If the trajectory is gone (regime 2) the internal
        Shield keeps running exactly as in the plain gatekeeper's regime 2, but no candidate is searched."""
        if self.traj is None or self._cb is None:
            return
        t0 = time.perf_counter()
        bel = self._bel_now(sim, t)
        self.n_steps += 1
        rem = self.traj.shifted(t)
        pose_now = np.asarray(sim.ctrl_pose() if hasattr(sim, "ctrl_pose") else sim.pose(), float)
        cb, trj, tj = self._cb
        ok, why = falsification_monitor(cb, trj, tj, rem, bel, self._obs_now, pose_now, t, self.gk)
        self.last_why = ("monitor", why)
        if ok:
            self.traj, self.mode = rem, "keep"
            self.n_keep_backup += 1
        elif self.gk.runtime_rows == "cup":                                 # no regime 2: commit the cup-safe brake from the current twist
            self._cup_brake(np.asarray(sim.base.v, float).copy(), bel, t, why)
        else:
            self.traj, self.mode = None, "regime2"
            self.n_falsified += 1
            self.falsify_reasons[why] = self.falsify_reasons.get(why, 0) + 1
        self.step_ms.append((time.perf_counter() - t0) * 1e3)

    def gk_step(self, sim, u_op, t):
        sr = self.sr
        if self._coast:
            return self._coast_step(sim, t)
        if sr is None or sr["status"] != "active" or self.traj is None or self._cb is None:
            return super().gk_step(sim, u_op, t)
        # segment following: T1 KEEP rule only (no re-planning of the operator command) -- the committed segment stands unless the monitor falsifies it
        t0 = time.perf_counter()
        bel = self._bel_now(sim, t)
        self.n_steps += 1
        rem = self.traj.shifted(t)
        pose_now = np.asarray(sim.ctrl_pose() if hasattr(sim, "ctrl_pose") else sim.pose(), float)
        cb, trj, tj = self._cb
        ok, why = falsification_monitor(cb, trj, tj, rem, bel, self._obs_now, pose_now, t, self.gk)
        self.last_why = ("monitor", why)
        if ok:
            self.traj, self.mode = rem, "keep"
            self.n_keep_backup += 1
            if self.pm is not None and self.pm.active:
                self._plan_check(sim, bel, t, sr)
        else:
            if self.gk.runtime_rows == "cup":                               # no regime 2: commit the cup-safe brake, refuse the route (the ladder replans)
                self._cup_brake(np.asarray(sim.base.v, float).copy(), bel, t, why)
            else:
                self.traj, self.mode = None, "regime2"
                self.n_falsified += 1
                self.falsify_reasons[why] = self.falsify_reasons.get(why, 0) + 1
            sr["status"], sr["t_next"], sr["reason"] = "refused", math.inf, f"falsified: {why}"
            sr["n_refused"] += 1
        self.step_ms.append((time.perf_counter() - t0) * 1e3)

    def _plan_check(self, sim, bel, t, sr):
        """plan-prediction monitor step (opt-in): compare the belief's people with the plan's prediction; on a flag stop committing segments of the old route
        (status "plan_falsified") and enter coast mode, so only the already-committed certified trajectory (nominal remainder, then its brake) is followed."""
        s_cur = sr["path"].project(np.asarray(bel.pose_wall, float)[:2], sr["s"])
        fl = self.pm.check(bel.people, bel.people_vel, float(getattr(sim, "t", t)), sr["path"], s_cur, sr["s_stop"])
        if fl is None:
            return
        fl = dict(fl, gk_t=float(t), s=float(s_cur), pose=[float(x) for x in np.asarray(bel.pose_wall, float)], route=sr["source"])
        self.plan_flag_log.append(fl)
        sr["reason"] = f"plan falsified: person {fl['person']} {fl['kind']} (pos_err {fl['pos_err']:.2f} m, vel_err {fl['vel_err']:.2f} m/s)"
        if self.gk.plan_decision == "timing":
            dec = self._plan_decide(sim, bel, t, self.pm_L)
            fl.update(dec)
            fl["decision_initial"] = dec["decision"]
        else:
            dec = dict(decision="brake")
            fl.update(decision="brake", decision_initial="brake", t_conf=None, t_ready=t + self.pm_L, t_stop=None, margin=None, L=self.pm_L)
        if dec["decision"] == "continue":
            sr["pf"], sr["t_ready"] = "continue", dec["t_ready"]            # keep driving the old route (normal boundary certification) until t_ready
            return
        sr["pf"] = "brake"
        sr["status"], sr["t_next"] = "plan_falsified", math.inf
        self._coast = True

    def _reeval(self, sim, bel, t, sr):
        """at a segment boundary during a "continue": redo the timing test with the CURRENT belief; if it fails, switch to brake NOW (commit no further old-route segment;
        the committed trajectory's brake is the fallback).  -> True if switched."""
        dec = self._plan_decide(sim, bel, t, sr["t_ready"] - t, t_ready=sr["t_ready"])
        if dec["decision"] == "continue":
            return False
        fl = self.plan_flag_log[-1]
        fl.update(decision="switched_to_brake", t_switch=float(t), t_conf_switch=dec["t_conf"], t_stop_switch=dec["t_stop"])
        sr["pf"] = "switched_to_brake"
        sr["status"], sr["t_next"] = "plan_falsified", math.inf
        self._coast = True
        return True

    def counters(self):
        c = super().counters()
        c.update(seg_boundaries=self.n_seg_boundaries, seg_routes=list(self.seg_log))
        if self.pm is not None:
            c.update(plan_flags=list(self.plan_flag_log))
        if self.hop_log:
            c.update(hops=list(self.hop_log))
        if self.tube_log:
            c.update(tube_log=list(self.tube_log))
        return c
