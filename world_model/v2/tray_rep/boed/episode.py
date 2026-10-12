"""boed/episode.py -- one carry under one policy (P1..P6 or a table cell). SIMULATION / SAVED DATA ONLY (MuJoCo CPU + torch CPU for F3).

Policies (the F3 filter, Rc3k value with the bridge's tau, is on in every carry phase)
  P1  conservative   intensity from the table at the WORST mu of the prior (the lowest). No probe.
  P2  reactive       the stored (nominal) carry, F3 filter, no probe. = bridge F3@Rc3k.
  P3  probe-at-pickup  certify, probe at t = 0, update the belief, intensity from the table under the posterior, carry.
  P4  probe-on-trigger the stored carry until F3's first intervention and then 6 quiet ticks (F3 has saved the ball); then hold (anti-roll stop, F3 on), certify, probe, update, re-pick the
                       intensity for the rest of the path (constant hazard along the remaining path) and continue.
  P5  oracle-mu      intensity from the table with the TRUE mu (upper bound).
  P6  population     intensity from the table under the prior (extra arm: the best single intensity with no information).
  FIX:I=<I_rel>,j=<node>   a table cell: intensity I_rel, mu_roll forced to NODES[j] (VAL; builds the predicted-drop table).
All commands pass through project_box and the env slew limit. F3 sees only the carry phase it is running in (its latent is restarted
after a probe), never the probe.
Time: one tick = 20 ms. total_ticks = probe + hold + carry. t_complete_s = time to reach the goal, or, if the plan ends short of it,
the pace-extrapolated time total / progress (the stored carry covers only ~30% of its path); NaN if the ball dropped.
"""
from __future__ import annotations

import time

import numpy as np

from world_model.v2.tray_rep import common as K          # noqa: F401
from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep import oracle as OR
from world_model.v2.tray_rep.store import Store
from world_model.v2.bead_rep import bead_env as B
from world_model.v2.tray_rep.bridge import filters as F
from world_model.v2.tray_rep.bridge.filters import project_box
from world_model.v2.tray_rep.bridge.metrics import PathProgress
from world_model.v2.tray_rep.bridge.episode import TRK_SEED_BASE, TRK_SEED_STRIDE
from world_model.v2.tray_rep.boed import carry as CA
from world_model.v2.tray_rep.boed import model as MD
from world_model.v2.tray_rep.boed import policy as PO
from world_model.v2.tray_rep.boed import probe as PB
from world_model.v2.tray_rep.boed import p7 as P7

DT = E.DT_WM
NMAX = 9000
POLICIES = ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P7s")      # P8: p7.run_p8 (stub)


class Ctx:
    """per-process objects: env, shadow env, the F3 filter, the table, settings."""

    def __init__(self, setup, vseed, split, data=None, runs_root=None, tau=0.1, arm="Rc3k", eps=PO.EPS, table=None, spec=None,
                 n_nodes=PO.N_NODES, cert=True, p7=None):
        from pathlib import Path
        self.setup, self.vseed, self.split, self.eps, self.cert = setup, int(vseed), split, float(eps), bool(cert)
        data = Path(data) if data else K.ROOT / f"data_{setup}_full"
        self.store = Store(data, split)
        self.meta = self.store.meta
        self.renv = E.env_from_meta(self.meta)
        self.shadow = E.env_from_meta(self.meta)
        self.noisefree = setup == "piper"
        self.trk = None if self.noisefree else MD.TRK_REAL
        self.lat = 0 if self.noisefree else int(MD.TRK_REAL["obs_latency_ticks"])
        self.sigma = MD.sigma_total(setup)
        self.spec = spec or PB.ProbeSpec()
        self.tau, self.arm = float(tau), arm
        self.filt = F.make_filter(f"F3:arm={arm},tau={tau:g}", self.renv, seed=int(vseed),
                                  runs_root=str(runs_root or K.ROOT / "runs"), model_setup=setup)
        self.table = table
        self.p7 = p7 or P7.P7Spec()                          # P7 / P7s settings (set certificate, decision quantile)
        self.quiet = 10                                      # P4: F3 quiet for this many ticks after its first intervention, then hold
        self.stop_decel = 0.5                                # fraction of a_max used to stop the tray in the P4 hold
        self.nodes = np.exp(np.linspace(np.log(E.MU_LO), np.log(E.MU_HI), n_nodes))


class Run:
    def __init__(self, ctx: Ctx, seed, mu_override=None):
        self.c = ctx
        r = ctx.renv
        mu, bxy, cr = CA.draw_episode(seed, ctx.meta["intensity"])
        self.seed, self.cr = int(seed), cr
        self.mu = float(mu if mu_override is None else mu_override)
        self.I_feas = CA.feasible_speed(cr, r.a_max)
        rec = r.reset(ball_xy=bxy, mu_roll=self.mu, track_seed=int(seed) + TRK_SEED_BASE + TRK_SEED_STRIDE * ctx.vseed)
        self.obs = np.zeros((NMAX, r.obs_dim), np.float32)
        self.act = np.zeros((NMAX, 6), np.float32)
        self.obs[0] = rec["obs"]
        self.n = 0                                           # ticks run = index of the last state row
        self.start_xy = r.env.pos[:2].copy()
        ref = CA.Plan(cr, r.a_scale, r.w_scale, 1.0, home_xy=self.start_xy)
        self.prog = PathProgress(self.start_xy, [s[2] for s in ref.ref])
        self.dropped = self.tunnel = False
        self.drop_tick = -1
        self.kf = None
        self.belief = MD.Belief()
        self.x = dict(n_interv=0, n_carry=0, n_probe=0, n_hold=0, fb=0)
        self.info = dict(probe_run=False, probe_drop=False, probe_skipped="", cert_ok=None, cert_dmin=None, cert_label=None,
                         cert_n=0, p4_trigger_tick=-1)
        self.I_hist = []
        self.extra = {}                                      # extra record fields (P7 / P7s); empty for P1..P6 so their records are unchanged
        self.dlog = None                                     # list: when not None, tick() appends the true edge distance (P7 diagnostics)

    # ------------------------------------------------------------------ primitives
    def tick(self, u):
        r = self.c.renv
        u = project_box(u, r.a_max, r.w_max)
        out = r.tick(u)
        self.obs[self.n + 1] = out["obs"]
        self.act[self.n] = out["act"]
        self.n += 1
        if self.dlog is not None:
            self.dlog.append(float(out["d"]))
        self.prog.update(r.env.pos[:2], self.n)
        if self.kf is not None:
            self.kf.feed(out["obs"], self.n)
        if out["failed"] or out["tunnel"]:
            self.dropped, self.drop_tick = True, self.n
            self.tunnel |= bool(out["tunnel"])
        return out

    def carry_phase(self, plan, quiet=None):
        """run `plan` with the F3 filter until it ends or the ball drops. With `quiet` = q: stop after F3's first intervention, once F3 has
        been quiet (u == u_nom) for q ticks in a row (the danger has passed and the ball is supported again). -> 'end'|'drop'|'trigger'."""
        c, f = self.c, self.c.filt
        f.begin(dict(renv=c.renv, hid=self.mu, seed=self.seed, T=plan.n_ticks))
        cs = self.n
        i = -1
        first = last = -1
        while True:
            i += 1
            if i >= plan.n_ticks and not plan.touchup(i * DT, c.renv.env.pos[:2]):
                break
            u_nom = plan.u(i * DT, c.renv.env.axis(), c.renv.env.pos[:2])
            u = project_box(f(self.obs[cs:self.n + 1], self.act[cs:self.n], u_nom, i), c.renv.a_max, c.renv.w_max)
            interv = bool(np.any(np.abs(u - u_nom) > 1e-9))
            self.x["n_interv"] += int(interv)
            self.x["n_carry"] += 1
            self.tick(u)
            if self.dropped:
                break
            if interv:
                last = i
                if first < 0:
                    first = i
                    if self.info["p4_trigger_tick"] < 0:
                        self.info["p4_trigger_tick"] = self.n
            if quiet is not None and first >= 0 and i - last >= quiet:
                self._collect(f)
                self.info["p4_hold_tick"] = self.n
                return "trigger", (i + 1) * DT
            if plan.waypoint_mode and plan.arrived(i * DT, c.renv.env.pos[:2], c.renv.env.v[:2]):
                break                                         # the goal is reached and the tray has stopped: the carry is done
        self._collect(f)
        return ("drop" if self.dropped else "end"), plan.T_end

    def _collect(self, f):
        self.x["fb"] += int(f.stats().get("x_fallback_ticks", 0))

    def stop_command(self, o, kappa, tau):
        """'anti-roll stop': decelerate the tray gently to rest while the tilt loop of the operator law keeps the ball supported
        (c* = normalize([kappa a_x, kappa a_y, g]), w = (c x c*) / tau, no tilt noise). Plain braking (bridge brake_action) levels the tray
        but does not support the ball while the tray decelerates: the ball shoots forward (seen in the smoke: P4 dropped carries P2 saved)."""
        c = self.c
        v = o[3:5].astype(np.float64)
        sp = float(np.hypot(*v))
        a = np.zeros(3)
        if sp > 1e-9:
            a[:2] = -v / sp * min(c.stop_decel * c.renv.a_max, sp / DT)
        ct = np.array([kappa * a[0], kappa * a[1], 9.81]); ct /= np.linalg.norm(ct)
        w = np.cross(o[0:3].astype(np.float64), ct) / tau
        w[2] = 0.0
        return np.concatenate([a, w])

    def hold_phase(self, max_ticks=80):
        """stop the tray (anti-roll stop as the nominal command), F3 filtering on top, until the tray is still and level."""
        c, f = self.c, self.c.filt
        f.begin(dict(renv=c.renv, hid=self.mu, seed=self.seed, T=max_ticks))
        cs = self.n
        n0 = self.n
        kappa, tau = float(self.cr["kappa"]), float(self.cr["tau"])
        for i in range(max_ticks):
            o = self.obs[self.n]
            sp = float(np.hypot(o[3], o[4]))
            tilt = float(np.arccos(np.clip(o[2], -1, 1)))
            if sp < 0.02 and tilt < 0.03 and i >= 2:
                break
            u_nom = self.stop_command(o, kappa, tau)
            u = project_box(f(self.obs[cs:self.n + 1], self.act[cs:self.n], u_nom, i), c.renv.a_max, c.renv.w_max)
            self.tick(u)
            if self.dropped:
                break
        self.x["n_hold"] += self.n - n0
        self._collect(f)

    def settle_phase(self, max_ticks=90):
        """tray at rest: bring the ball to the tray centre and stop it (the probe's balance law as the nominal command, F3 on top).
        -> True if the tracked ball is still and centred."""
        c, f = self.c, self.c.filt
        f.begin(dict(renv=c.renv, hid=self.mu, seed=self.seed, T=max_ticks))
        cs, n0 = self.n, self.n
        ctrl = PB.ProbeController(c.spec, self.kf, c.renv.w_max, balance_first=True)
        settled = False
        for i in range(max_ticks):
            u_nom = ctrl.step(c.renv.env.axis(), self.n)
            if ctrl.phase != "settle":                    # the controller has seen the ball still and centred
                settled = True
                break
            u = project_box(f(self.obs[cs:self.n + 1], self.act[cs:self.n], u_nom, i), c.renv.a_max, c.renv.w_max)
            self.tick(u)
            if self.dropped:
                break
        self.x["n_hold"] += self.n - n0
        self._collect(f)
        return settled

    # ------------------------------------------------------------------ probe, certificate, belief
    def certify(self, balance_first):
        """shadow run of the whole probe at the worst mu of the belief; must stay on the tray with the edge margin and end hold_ok/savable."""
        c = self.c
        snap = c.renv.snapshot()
        mu_w = self.belief.lower_mu(0.01)
        res = PB.run_shadow(c.shadow, snap, mu_w, c.spec, self.kf.clone(), self.n, balance_first, c.renv.w_max)
        label = OR.UNCERT
        if not res["dropped"]:
            label = OR.certify(c.shadow, c.shadow.snapshot(), (OR.D_HOLD,), K=OR.K_H)["label"]
        ok = (not res["dropped"]) and res["dmin"] >= c.spec.d_margin and label != OR.UNCERT
        self.info["cert_n"] += 1
        self.info.update(cert_ok=bool(ok), cert_dmin=res["dmin"], cert_label=int(label), cert_mu_worst=float(mu_w))
        return ok

    def probe(self, balance_first):
        c = self.c
        ctrl = PB.ProbeController(c.spec, self.kf, c.renv.w_max, balance_first)
        n0 = self.n
        guard = c.spec.n_settle_max + c.spec.n_pre + c.spec.n_hold_max + c.spec.n_bal_max + 5
        for _ in range(guard):
            self.tick(ctrl.step(c.renv.env.axis(), self.n))
            if self.dropped or ctrl.done:
                break
        self.x["n_probe"] += self.n - n0
        self.info["probe_run"] = True
        self.info["probe_drop"] = bool(self.dropped)
        if not self.dropped and ctrl.t_hold_end > 0:
            t0, t1 = ctrl.window()
            ll = MD.probe_loglik(self.obs, t0, t1, c.sigma, c.lat, ctrl.prior, c.noisefree)
            self.belief.update(ll)
            self.info.update(probe_hold=int(ctrl.n_hold), probe_reason=ctrl.reason, probe_settled=bool(ctrl.settled_end),
                             probe_end_off=float(np.hypot(*c.renv.env.local()[:2])))
        return ctrl

    def pick(self, frac=1.0):
        c = self.c
        i = c.table.choose(self.belief.w, c.eps, frac)
        return float(c.table.igrid[i])

    def plan_for(self, I_rel):
        c = self.c
        I_abs = float(I_rel) * self.I_feas
        self.I_hist.append(float(I_rel))
        return CA.Plan(self.cr, c.renv.a_scale, c.renv.w_scale, I_abs, home_xy=self.start_xy)

    # ------------------------------------------------------------------ policies
    def run(self, policy):
        c = self.c
        kind, _, rest = policy.partition(":")
        kw = {k: v for k, v in (kv.split("=") for kv in rest.split(",") if kv)} if rest else {}
        if kind == "P2":
            self.carry_phase(CA.Plan(self.cr, c.renv.a_scale, c.renv.w_scale, 1.0, home_xy=self.start_xy))
        elif kind == "FIX":
            self.carry_phase(self.plan_for(float(kw["I"])))
        elif kind == "P1":
            self.carry_phase(self.plan_for(float(c.table.igrid[c.table.choose(PO.w_worst(), c.eps)])))
        elif kind == "P5":
            self.carry_phase(self.plan_for(float(c.table.igrid[c.table.choose(PO.w_true(self.mu), c.eps)])))
        elif kind == "P6":
            self.carry_phase(self.plan_for(float(c.table.igrid[c.table.choose(PO.w_prior(), c.eps)])))
        elif kind == "P3":
            self._p3()
        elif kind == "P4":
            self._p4()
        elif kind in ("P7", "P7s"):
            P7.run_policy(self, kind)
        elif kind == "P8":
            P7.run_p8(self)
        else:
            raise ValueError(policy)
        return self.record(policy)

    def _p3(self):
        c = self.c
        self.kf = PB.make_kf(c.trk, c.noisefree)
        self.kf.reset_prior(sx=0.006, sv=0.01)
        self.kf.feed(self.obs[0], 0)
        ok = self.certify(False) if c.cert else True
        if ok:
            self.probe(False)
        else:
            self.info["probe_skipped"] = "cert_failed"
        if self.dropped:
            return
        self.carry_phase(self.plan_for(self.pick()))

    def _p4(self):
        c = self.c
        self.kf = PB.make_kf(c.trk, c.noisefree)
        self.kf.feed(self.obs[0], 0)
        plan = CA.Plan(self.cr, c.renv.a_scale, c.renv.w_scale, 1.0, home_xy=self.start_xy)
        code, tp = self.carry_phase(plan, quiet=c.quiet)
        if code != "trigger":
            return
        self.hold_phase()
        if self.dropped:
            return
        settled = self.settle_phase()
        if self.dropped:
            return
        ok = False
        if settled:
            ok = self.certify(False) if c.cert else True
            if ok:
                self.probe(False)
                if self.dropped:
                    return
        self.info["probe_skipped"] = "" if ok else ("not_settled" if not settled else "cert_failed")
        frac = max(0.05, 1.0 - self.prog.done / max(self.prog.len_nom, 1e-9))
        I_rel = self.pick(frac)
        plan.replan_from(tp, c.renv.env.pos[:2], I_rel * self.I_feas)
        self.I_hist.append(float(I_rel))
        self.carry_phase(plan)

    # ------------------------------------------------------------------ record
    def record(self, policy):
        c = self.c
        pg = self.prog.record()
        n = self.n
        reached = pg["goal_reached"] and not self.dropped
        if self.dropped:
            t_complete = float("nan")
        elif reached:
            t_complete = pg["goal_tick"] * DT
        else:
            t_complete = n * DT / max(pg["progress"], 0.05)
        out = dict(seed=self.seed, policy=policy, setup=c.setup, vseed=c.vseed, mu=self.mu, I_feas=self.I_feas,
                   dropped=bool(self.dropped), drop_tick=int(self.drop_tick), tunnel=bool(self.tunnel), n_ticks=int(n),
                   total_s=n * DT, probe_ticks=int(self.x["n_probe"]), hold_ticks=int(self.x["n_hold"]), carry_ticks=int(self.x["n_carry"]),
                   n_interv=int(self.x["n_interv"]), interv_rate=float(self.x["n_interv"] / max(self.x["n_carry"], 1)),
                   x_fallback_ticks=int(self.x["fb"]), t_complete_s=float(t_complete),
                   I_rel=(self.I_hist[-1] if self.I_hist else None), I_hist=self.I_hist, **pg, **self.info)
        out["probe_drop"] = bool(self.info["probe_drop"])
        if self.belief.n_updates:
            b = self.belief
            lo, hi = b.interval(0.9)
            out.update(bel_mean_log=b.mean_log(), bel_sd_log=b.sd_log(), bel_cdf_true=b.cdf(self.mu), bel_cover90=b.covers(self.mu, 0.9),
                       bel_cover50=b.covers(self.mu, 0.5), bel_lo90=lo, bel_hi90=hi, bel_err_log=b.mean_log() - float(np.log(self.mu)))
        out.update(self.extra)
        return out


def run_carry(ctx: Ctx, policy, ep, mu_override=None):
    info = ctx.store.ep_info(ep)
    t0 = time.perf_counter()
    r = Run(ctx, info["seed"], mu_override)
    rec = r.run(policy)
    rec["ep"] = int(ep)
    rec["onset_stored"] = int(info["onset"])
    rec["wall_s"] = time.perf_counter() - t0
    return rec
