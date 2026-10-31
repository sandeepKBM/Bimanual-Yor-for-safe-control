"""gatekeeper/gate.py -- the gatekeeper (Agrawal & Panagou, T-RO 2024 pattern) over set-valued beliefs.  SIMULATION ONLY.

Drop-in for teleop.shield.Shield in run.run_episode(..., filt=Gatekeeper()):  filter(sim, cmd) -> (Cmd, info).
Every k_ticks ticks (default 5 = 0.1 s):
  1. for T_S in gk.t_switch (descending): plan nominal(u_op held T_S) + backup, run Checker 1 and 2 on the CURRENT belief; commit the first that passes
     (n_commit);
  2. none passes -> keep executing the previously committed trajectory (its remaining nominal part and backup) if it is still certified under the current
     belief (n_keep_backup);
  3. it is not (model falsified: person faster than v_p_max, unseen person, pose error beyond the box, start inside a violated buffer, ...) -> n_falsified,
     REGIME 2: the internal teleop Shield (cfg.fallback = gk.regime2_fallback, default "escape") produces the command until a candidate passes again.
Between steps the committed twist profile is tracked: u(t + TICK).  Lift / jog go through the Shield's filter_lift / filter_jog.
OPT-IN gk.runtime_rows == "cup" (user decision 10-24..26: the planner plans safely, the runtime layer keeps only cup + actuator rows): step 3 does NOT enter regime 2.  The
model-falsified event (pose part of the monitor, or the kept trajectory failing the cup/hw recertification) commits the cup-safe brake (traj.backup_from from the current
twist), mode "cup_brake", and is counted in n_falsified / falsify_reasons.  The internal Shield object still exists as a HELPER (footprint, jog / lift governors) but its
base QP (Shield.filter_base) is never called.  Default "all" == the behaviour above, bit-identical.
"""
from __future__ import annotations

import dataclasses
import math
import time

import numpy as np

from ..teleop import base as BS
from ..teleop import geom as GM
from ..teleop import shield as SH
from ..teleop.sim import Cmd, TICK
from .belief import GKConfig, build_belief
from .checks import check1, check2
from .monitor import PersonSetFilter, falsification_monitor, observation_discs
from .traj import plan_candidate, backup_from, Traj


class Gatekeeper:
    def __init__(self, cfg: SH.ShieldCfg | None = None, gk: GKConfig | None = None, perc=None):
        """perc: optional injected people_track PerceptionStack-like object (step(sim), proxy(sim)) for gk.people_source == "tracker" (tests); default: built lazily."""
        self.gk = gk or GKConfig()
        assert self.gk.people_source in ("truth", "tracker"), self.gk.people_source
        assert self.gk.keep_rule in ("recertify", "monitor"), self.gk.keep_rule
        assert self.gk.belief_filter in ("none", "intersect"), self.gk.belief_filter
        assert self.gk.runtime_rows in ("all", "cup"), self.gk.runtime_rows
        self.tracker = self.gk.people_source == "tracker"
        self.perc = perc                      # tracker source: the PerceptionStack (stepped once per sim tick in filter())
        self._proxy = None                    # tracker source: SimProxy of the current tick (tracker-derived people_state / track_info)
        self._ucfg = None
        self._conf = None
        if self.tracker:
            from ..people_track.source import load_conformal
            self._conf = load_conformal(self.gk.conformal)           # path / ConformalMargin / None (-> k_sigma * sqrt(lam_max(P_pp)))
        self.n_audit_steps = 0                # tracker source, METRICS ONLY (truth is never used for a decision): gatekeeper steps audited
        self.n_audit_person_steps = 0         # true-person x gk-step pairs audited
        self.n_audit_matched = 0              # ... with a track within audit_match_r
        self.n_belief_miss = 0                # ... matched, but |true - c| > e_p of the matched track (the belief set at t=0 does NOT contain the person)
        self.n_untracked_near = 0             # ... no track within audit_match_r and the person is within audit_near_r of the footprint (unseen / occluded / dropped)
        self.max_miss_excess_m = 0.0          # max over misses of |true - c| - e_p
        self.max_untracked_gap_s = 0.0
        self._untracked_since = None
        self.cfg = dataclasses.replace(cfg) if cfg is not None else SH.ShieldCfg()
        self.cfg.fallback = self.gk.regime2_fallback
        if getattr(self.gk, "cup_row", "always") != "always":
            self.cfg.cup_row = self.gk.cup_row        # the internal regime-2 Shield uses the same cup-row rule as the gatekeeper belief
        self.docks = []
        self._sh = None                       # internal Shield (regime 2 + lift/jog), built on the first call (needs sim.occ_ctrl)
        self.traj: Traj | None = None         # committed trajectory
        self.mode = "init"
        self.n_calls = 0
        self.n_steps = 0
        self.n_commit = 0
        self.n_keep_backup = 0
        self.n_falsified = 0
        self.falsify_reasons = {}             # keep_rule="monitor": reason -> count of n_falsified events
        self._cb = None                       # (belief, committed Traj, t) of the last commit (monitor keep rule)
        self._obs_now = None                  # raw observation discs of the current gk step (monitor)
        self._pf = PersonSetFilter(self.gk) if self.gk.belief_filter == "intersect" else None
        self.n_regime2_ticks = 0
        self.n_deviate = 0                    # committed a candidate != u_op (candidates="set"); n_deviate <= n_commit
        self.dev_hist = []                    # stick-normalised deviation ||(u_cand - u_op)/V_STICK|| of every committed gk step (0 for hold-u_op)
        self.dev_commit = []                  # same, only the deviating commits
        self.n_cand_checks = 0
        self.cand_checks_max = 0
        self.last_bel = None
        self._pp_prev = None                  # (t, positions) for the finite-difference velocity estimate
        self.T_S_hist = {}
        self.step_ms = []
        self.last = {}
        self.last_why = None
        self.fp_now = GM.FP

    # shield-compatible counters (run.summarize): the internal shield's fallbacks / vetoes are regime-2 events
    @property
    def n_fallback(self):
        return self._sh.n_fallback if self._sh is not None else 0

    @property
    def n_veto(self):
        return self._sh.n_veto if self._sh is not None else 0

    # ------------------------------------------------------------------------------------------ tracker people source
    def _ensure_perc(self, sim):
        """build (once) the PerceptionStack the same way people_track.source.make_shield does (condition parser, conformal, seed); CPU-only conditions."""
        if self.perc is not None:
            return self.perc
        from dataclasses import replace
        from ..people_track import perception as PC
        from ..people_track.run_pt import parse_cond
        pcfg, u = parse_cond(self.gk.perception)
        assert pcfg is not None, "perception='truth' is the truth source; use people_source='truth'"
        assert pcfg.detector in ("depth", "seg", "param"), f"{pcfg.detector}: needs est pose / a GPU detector server; the gatekeeper sweeps are CPU-only"
        if self._conf is not None:
            u = replace(u, conformal=self._conf, soft_floor=0.05 if u.soft_floor is None else u.soft_floor)
        self._ucfg = u
        self.perc = PC.PerceptionStack(sim, replace(pcfg, seed=self.gk.seed))
        return self.perc

    def perceive(self, sim):
        """step the tracker once (call once per sim tick BEFORE filter_base; filter() does it) and snapshot the tracker-derived sim proxy."""
        self._ensure_perc(sim)
        self.perc.step(sim)
        self._proxy = self.perc.proxy(sim)
        return self._proxy

    def _track_list(self):
        """tracks of the current proxy -> belief inputs: c = mean xy, v_hat = mean velocity, e_p = conformal radius(since_upd, age, range) if a conformal margin is
        loaded else k_sigma*sqrt(lam_max(P_pp)), e_v = k_sigma*sqrt(lam_max(P_vv)), stale = time since the track's last measurement update."""
        k = self.gk.k_sigma
        out = []
        for ti in self._proxy.track_info:
            P = np.asarray(ti["P"], float)
            lp = float(np.linalg.eigvalsh(0.5 * (P[:2, :2] + P[:2, :2].T))[-1])
            lv = float(np.linalg.eigvalsh(0.5 * (P[2:4, 2:4] + P[2:4, 2:4].T))[-1])
            su = float(ti.get("since_upd_s", 0.0))
            if self._conf is not None:
                e_p = float(self._conf.radius(su, float(ti.get("age_s", 0.0)), ti.get("range_m")))
            else:
                e_p = k * math.sqrt(max(lp, 0.0))
            out.append(dict(xy=np.asarray(ti["xy"], float), v=np.asarray(ti["v"], float), e_p=e_p, e_v=k * math.sqrt(max(lv, 0.0)), stale=su, name=ti.get("name")))
        return out

    def _audit(self, sim, tracks, t):
        """ASSUMPTION AUDIT (metrics only): is every TRUE person inside the belief's set for its matched track at time 0 (|true - c| <= e_p), and is any true person
        within audit_near_r of the footprint but UNTRACKED (no track within audit_match_r)?  Truth is read from sim.people_state and never fed back."""
        gk = self.gk
        self.n_audit_steps += 1
        pose = np.asarray(sim.pose(), float)
        untracked_near = False
        for xy, _v in (sim.people_state or []):
            xy = np.asarray(xy, float)
            self.n_audit_person_steps += 1
            j, dj = -1, math.inf
            for i, tr in enumerate(tracks):
                dd = float(np.hypot(*(xy - tr["xy"])))
                if dd < dj:
                    j, dj = i, dd
            if j >= 0 and dj <= gk.audit_match_r:
                self.n_audit_matched += 1
                ex = dj - (tracks[j]["e_p"] + gk.e_p)
                if ex > 1e-9:
                    self.n_belief_miss += 1
                    self.max_miss_excess_m = max(self.max_miss_excess_m, ex)
            elif GM.rect_dist_point(xy, pose)[0] <= gk.audit_near_r:
                self.n_untracked_near += 1
                untracked_near = True
        if untracked_near:
            self._untracked_since = t if self._untracked_since is None else self._untracked_since
            self.max_untracked_gap_s = max(self.max_untracked_gap_s, t - self._untracked_since + 0.1)
        else:
            self._untracked_since = None

    def _shield(self, sim):
        if self._sh is None and self.tracker:
            from ..people_track import shield_pt as SPT
            self._ensure_perc(sim)                                              # internal shield (regime 2) sees the SAME tracks (it is only used via filter_base on the proxy)
            self._sh = SPT.PerceivedShield(self.perc, self.cfg, occ=sim.occ_ctrl, a_hw=sim.a_hw, ucfg=self._ucfg)
        if self._sh is None:
            self._sh = SH.Shield(self.cfg, occ=sim.occ_ctrl, a_hw=sim.a_hw)
        self._sh.docks = self.docks
        return self._sh

    def counters(self):
        ms = np.array(self.step_ms) if self.step_ms else np.zeros(1)
        return dict(n_commit=self.n_commit, n_keep_backup=self.n_keep_backup, n_falsified=self.n_falsified, n_regime2_ticks=self.n_regime2_ticks,
                    n_gk_steps=self.n_steps, n_fallback=self.n_fallback, n_veto=self.n_veto, T_S_hist=dict(self.T_S_hist),
                    **(dict(n_deviate=self.n_deviate, mean_deviation=float(np.mean(self.dev_hist)) if self.dev_hist else 0.0,
                            mean_deviation_when_deviating=float(np.mean(self.dev_commit)) if self.dev_commit else 0.0,
                            n_cand_checks=self.n_cand_checks, cand_checks_max_per_step=self.cand_checks_max) if self.gk.candidates == "set" else {}),
                    **(dict(n_audit_steps=self.n_audit_steps, n_audit_person_steps=self.n_audit_person_steps, n_audit_matched=self.n_audit_matched,
                            n_belief_miss=self.n_belief_miss, n_untracked_near=self.n_untracked_near, max_miss_excess_m=self.max_miss_excess_m,
                            max_untracked_gap_s=self.max_untracked_gap_s) if self.tracker else {}),
                    **(dict(falsify_reasons=dict(self.falsify_reasons)) if self.gk.keep_rule == "monitor" else {}),
                    **(self._pf.stats() if self._pf is not None else {}),
                    step_ms_mean=float(ms.mean()), step_ms_max=float(ms.max()), step_ms_p95=float(np.percentile(ms, 95)))

    # ------------------------------------------------------------------------------------------ one gatekeeper step
    def gk_step(self, sim, u_op, t):
        t0 = time.perf_counter()
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
        v0 = np.asarray(sim.base.v, float).copy()
        self.n_steps += 1
        for T_S in self.gk.t_switch:
            if self._certify(v0, u_op, T_S, bel, t):
                self._commit(self._last_tr, T_S, 0.0, bel, t)
                self.step_ms.append((time.perf_counter() - t0) * 1e3)
                return
        if self.gk.candidates == "set":
            hit = self._candidate_search(sim, v0, u_op, bel, t, t0)
            if hit is not None:
                self._commit(hit[0], hit[1], hit[2], bel, t)
                self.n_deviate += 1
                self.dev_commit.append(hit[2])
                self.step_ms.append((time.perf_counter() - t0) * 1e3)
                return
        if self.gk.keep_rule == "monitor" and self.traj is not None and self._cb is not None:
            # T1 KEEP rule: no re-certification; the commit-time certificate stands unless the falsification monitor fires (monitor.py)
            rem = self.traj.shifted(t)
            pose_now = np.asarray(sim.ctrl_pose() if hasattr(sim, "ctrl_pose") else sim.pose(), float)
            cb, trj, tj = self._cb
            ok, why = falsification_monitor(cb, trj, tj, rem, bel, self._obs_now, pose_now, t, self.gk)
            self.last_why = ("monitor", why)
            if ok:
                self.traj, self.mode = rem, "keep"
                self.n_keep_backup += 1
            elif self.gk.runtime_rows == "cup":
                self._cup_brake(v0, bel, t, why)
            else:
                self.traj, self.mode = None, "regime2"
                self.n_falsified += 1
                self.falsify_reasons[why] = self.falsify_reasons.get(why, 0) + 1
            self.step_ms.append((time.perf_counter() - t0) * 1e3)
            return
        # nothing passes: keep the committed trajectory if it is still certified under the current belief
        rem = self._remaining(t, v0, bel)
        tolg = self.gk.falsify_tol
        ok = False
        if rem is not None:
            o1, m1, r1, _ = check1(rem, bel, self.gk, tol_geo=tolg)
            o2, m2, r2, _ = check2(rem, bel, self.gk, tol_geo=tolg)
            ok = o1 and o2
            self.last_why = ("keep", r1 if not o1 else r2, m1 if not o1 else m2)
        if ok:
            self.traj, self.mode = rem, "keep"
            self.n_keep_backup += 1
        elif self.gk.runtime_rows == "cup":
            self._cup_brake(v0, bel, t, "no_commit_recertify")
        else:
            self.traj, self.mode = None, "regime2"
            self.n_falsified += 1
            if self.gk.keep_rule == "monitor":
                self.falsify_reasons["no_commit_recertify"] = self.falsify_reasons.get("no_commit_recertify", 0) + 1
        self.step_ms.append((time.perf_counter() - t0) * 1e3)

    # ------------------------------------------------------------------------------------------ helpers
    def _cup_brake(self, v0, bel, t, why):
        """runtime_rows="cup": the model is falsified -> NO regime 2 / Shield hand-off.  Commit the cup-safe brake (traj.backup_from) from the current twist and count the event;
        the committed brake is the new reference of the pose monitor."""
        tr = backup_from(np.asarray(v0, float), bel, self.gk, t0=t)
        self.traj, self.mode = tr, "cup_brake"
        self._cb = (bel, tr, t)
        self.n_falsified += 1
        self.falsify_reasons[why] = self.falsify_reasons.get(why, 0) + 1

    def _certify(self, v0, u, T_S, bel, t):
        """plan nominal(u held T_S)+backup and run Checkers 1 and 2; sets self._last_tr / self.last_why; -> bool."""
        tr = self._last_tr = plan_candidate(v0, u, T_S, bel, self.gk, t0=t)
        ok1, m1, r1, _ = check1(tr, bel, self.gk)
        if not ok1:
            self.last_why = ("c1", T_S, r1, m1)
            return False
        ok2, m2, r2, _ = check2(tr, bel, self.gk)
        if not ok2:
            self.last_why = ("c2", T_S, r2, m2)
            return False
        return True

    def _filter_belief(self, bel, tracks, t):
        """belief_filter="intersect": replace each person's (centre, own position error) by the set-membership filtered disc (monitor.PersonSetFilter)."""
        n = len(bel.people)
        if n == 0:
            self._pf.state = {}
            return bel
        keys = [(tr.get("name") if tr.get("name") is not None else ("i", i)) for i, tr in enumerate(tracks)] if tracks is not None else list(range(n))
        own = list(bel.people_e_p) if len(bel.people_e_p) == n else [0.0] * n
        vel = list(bel.people_vel) if len(bel.people_vel) == n else [None] * n
        evs = [bel.e_v + (bel.people_e_v[i] if len(bel.people_e_v) == n else 0.0) for i in range(n)]
        cs, rs = self._pf.update(keys, bel.people, own, vel, evs, t)
        kw = dict(people=tuple(np.asarray(c, float).copy() for c in cs))
        if len(bel.people_e_p) == n or any(r != 0.0 for r in rs):
            kw["people_e_p"] = tuple(float(r) for r in rs)
        return bel.with_(**kw)

    def _commit(self, tr, T_S, dev, bel=None, t=None):
        if bel is not None:
            self._cb = (bel, tr, t)
        self.traj, self.mode = tr, "commit"
        self.n_commit += 1
        self.T_S_hist[T_S] = self.T_S_hist.get(T_S, 0) + 1
        self.dev_hist.append(dev)

    def _people_vel(self, sim, t):
        """-> (people_vel or None, e_v override or None).  accel model only.  vel_source="state": sim.people_state velocities (truth source);
        "fd" (or a source without velocity): finite difference of the observed positions over the previous gk step, e_v inflated by a_p*dt (acceleration
        within the window; first step: v_hat = 0 and e_v = v_p_max, i.e. nothing known -> the speed disc governs)."""
        gk = self.gk
        if gk.person_model not in ("accel", "pred"):
            return None, None
        ps = sim.people_state or []
        pos = [np.asarray(xy, float).copy() for xy, _ in ps]
        if gk.vel_source == "state" and all(v is not None for _, v in ps):
            return None, None                                                   # build_belief reads people_state velocities
        prev, self._pp_prev = self._pp_prev, (t, pos)
        if prev is None or len(prev[1]) != len(pos) or t - prev[0] <= 1e-9:
            return tuple(np.zeros(2) for _ in pos), gk.v_p_max
        dt = t - prev[0]
        vel = tuple((p - q) / dt for p, q in zip(pos, prev[1]))
        return vel, gk.e_v + gk.a_p * dt + (2.0 * gk.e_p / dt)

    def _candidate_twists(self, sim, u_op, bel):
        """deviation candidates around u_op (body twists), sorted by stick-normalised deviation (ties: generation order), duplicates / u_op removed."""
        gk = self.gk
        vc = bel.v_cap
        u = np.asarray(u_op, float)
        c = []
        for sc in gk.cand_scales:
            c.append(sc * u)
        for dg in gk.cand_rot_deg:
            for sg in (1.0, -1.0):
                a = math.radians(sg * dg)
                ca, sa = math.cos(a), math.sin(a)
                c.append(np.array([ca * u[0] - sa * u[1], sa * u[0] + ca * u[1], u[2]]))
        for sg in (1.0, -1.0):
            c.append(u + np.array([0.0, sg * gk.cand_lat_frac * vc[1], 0.0]))
        for sg in (1.0, -1.0):
            c.append(np.array([0.0, 0.0, sg * gk.cand_spin_frac * vc[2]]))
        bo = self._backoff_dir(sim, bel)
        if bo is not None:
            c.append(np.array([bo[0] * gk.cand_backoff_frac * vc[0], bo[1] * gk.cand_backoff_frac * vc[1], 0.0]))
        out, seen = [], set()
        for x in c:
            x = np.clip(x, -vc, vc)
            dev = float(np.linalg.norm((x - u) / vc))
            key = tuple(np.round(x, 6))
            if dev < 1e-9 or key in seen:
                continue
            seen.add(key)
            out.append((dev, x))
        out.sort(key=lambda p: p[0])
        return out

    def _backoff_dir(self, sim, bel):
        """unit direction (BODY frame) pointing away from the binding hazard of the last failed hold attempt (reason person -> from the nearest person;
        wall -> along +grad EDT at the robot); None for other reasons."""
        why = self.last_why
        if not why or why[0] not in ("c1", "c2"):
            return None
        reason = why[2]
        pw = np.asarray(bel.pose_wall, float)
        if reason in ("person", "passive_person") and len(bel.people):
            pp = np.asarray(bel.pose_people, float)
            near = min(bel.people, key=lambda c: float(np.hypot(c[0] - pp[0], c[1] - pp[1])))
            rel = pp[:2] - np.asarray(near, float)[:2]                           # person -> robot (world)
        elif reason in ("wall", "wall_at_rest") and bel.grid is not None:
            e = 0.05
            g = np.array([bel.grid.lookup(np.array([pw[0] + e]), np.array([pw[1]]))[0] - bel.grid.lookup(np.array([pw[0] - e]), np.array([pw[1]]))[0],
                          bel.grid.lookup(np.array([pw[0]]), np.array([pw[1] + e]))[0] - bel.grid.lookup(np.array([pw[0]]), np.array([pw[1] - e]))[0]])
            rel = g
        else:
            return None
        n = float(np.hypot(*rel))
        if n < 1e-9:
            return None
        rel = rel / n
        th = pw[2]
        c, s = math.cos(th), math.sin(th)
        return np.array([c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1]])            # world -> body

    def _candidate_search(self, sim, v0, u_op, bel, t, t_start):
        """try deviation candidates in order of increasing deviation, T_S in gk.cand_t_switch (descending); at most cand_max_checks (candidate, T_S) attempts.
        First certified candidate wins; equal-deviation candidates after it are still evaluated and the larger T_S wins the tie.
        -> (traj, T_S, dev) or None."""
        gk = self.gk
        cands = self._candidate_twists(sim, u_op, bel)
        n = 0
        best = None
        for dev, x in cands:
            if best is not None and dev > best[2] + 1e-9:
                break
            if n >= gk.cand_max_checks:
                break
            if gk.cand_time_budget_ms > 0 and (time.perf_counter() - t_start) * 1e3 > gk.cand_time_budget_ms:
                break
            for T_S in gk.cand_t_switch:
                if n >= gk.cand_max_checks:
                    break
                n += 1
                if self._certify(v0, x, T_S, bel, t):
                    if best is None or T_S > best[1]:
                        best = (self._last_tr, T_S, dev)
                    break
        self.n_cand_checks += n
        self.cand_checks_max = max(self.cand_checks_max, n)
        return best

    def _remaining(self, t, v0, bel):
        """committed trajectory re-based at t; with no commitment yet: the pure backup from the current twist (the 'stay safe' default)."""
        if self.traj is None:
            if self.mode == "regime2":
                return None
            return plan_candidate(v0, np.zeros(3), 0.0, bel, self.gk, t0=t)
        return self.traj.shifted(t)

    # ------------------------------------------------------------------------------------------ per tick
    def filter_base(self, sim, u_op):
        """-> (u_safe twist, label)."""
        t = self.n_calls * TICK
        if (self.n_calls % self.gk.k_ticks) == 0 or (self.mode == "init"):
            self.gk_step(sim, np.asarray(u_op, float), t)
        v = np.asarray(sim.base.v, float)
        if self.mode == "regime2":
            self.n_regime2_ticks += 1
            sh = self._shield(sim)
            u, label = sh.filter_base(self._proxy if self.tracker else sim, u_op)          # tracker: the internal shield sees the tracks, never the truth
            label = "gk_regime2:" + label
        else:
            u = self.traj.u_at(t + TICK)
            u = np.clip(u, v - sim.a_hw * TICK, v + sim.a_hw * TICK)
            label = "gk_" + self.mode
            if self.mode != "regime2" and self._sh is not None:                 # keep the Shield's jerk-row memory consistent for a hand-over
                R = GM.rot(sim.base.yaw)
                r_c = R.T @ (sim.d.xpos[sim.cup][:2] - np.array([sim.base.x, sim.base.y]))
                self._sh.a_prev_w = R @ np.asarray(BS.cup_accel_body(v, (u - v) / TICK, r_c))
        return u, label

    def filter(self, sim, cmd: Cmd):
        if self.tracker:
            self.perceive(sim)                                                   # same per-tick stepping as PerceivedShield.filter
        sh = self._shield(sim)
        u_safe, label = self.filter_base(sim, cmd.twist)
        self.n_calls += 1
        jog_s = sh.filter_jog(sim, cmd.jog)
        lift_s = sh.filter_lift(cmd.lift_v)
        safe = Cmd(twist=u_safe, lift_v=lift_s, jog=jog_s, grasp=cmd.grasp, release=cmd.release)
        v = np.asarray(sim.base.v, float)
        R = GM.rot(sim.base.yaw)
        r_c = R.T @ (sim.d.xpos[sim.cup][:2] - np.array([sim.base.x, sim.base.y]))
        a_cup = float(np.linalg.norm(BS.cup_accel_body(v, (u_safe - v) / TICK, r_c)))
        dv = cmd.twist - u_safe
        jl = float(np.linalg.norm(cmd.jog - jog_s)) + abs(cmd.lift_v - lift_s)
        info = dict(intervention=float(np.linalg.norm(dv)), intervention_norm=float(np.linalg.norm(dv / BS.V_STICK)), intervention_aux=jl, label=label,
                    qp=self.mode, ir_margin=float("nan"), a_cup=a_cup, veto=False)
        return safe, info
