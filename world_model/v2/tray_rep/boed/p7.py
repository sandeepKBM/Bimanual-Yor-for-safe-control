"""boed/p7.py -- P7 / P7s: probe at pickup with a WORST-CASE speed rule; P7 with a SET-BASED (robot-computable) certificate.
SIMULATION / SAVED DATA ONLY. CPU. No hardware, no network. (Imports only tray_rep modules; nothing under YOR/robot or YOR/extra.)

P7   = P3's machinery (F3 on in every phase, same probe, same belief, same frozen VAL table, same eps) with two changes.
  (a) decision rule  mu_q = 2.5% quantile of the posterior (the lower end of its central 95% credible set).
      I = the fastest grid intensity whose table drop rate AT mu_q is <= eps  (Table.drop_at(mu_q)); the slowest if none.
      Lower mu = the ball rolls more freely = the dangerous direction; the table fit is monotone DECREASING in mu (policy.smooth_table),
      so the drop at mu_q bounds the drop at every mu >= mu_q (checked in p7_checks.py).
      P3 used the posterior-EXPECTED drop, which picks full speed for a mid-mu ball whose posterior still has a rolling tail (RESULT_BOED.md).
  (b) certificate = an analytic SET rollout (no MuJoCo state, no shadow env; only what the robot has at pickup):
        - ball estimate: the probe's own filter (boed.estimator.BallKF) at probe start: mean (x, v) and covariance P (pos, vel) per axis, in
          the tray frame; tray state (axis c, velocity vt, realised tilt rate w) read from the observation row (joint encoders);
        - mu = the belief's 1% quantile (before the probe: the population prior's lowest value, 5e-4), mapped through the MuJoCo calibration;
        - the planned command profile of the probe, rolled with the sets/reach.py kernels (ball_step, tray_step) in closed loop on the
          realised axis: `n_pre` level ticks (w = c x z / tau_p), then `h` ticks of tilt about y by +-theta (w = c x c_des / tau_p, clipped to
          w_max, then the arm's slew limit alpha), for EVERY hold length h the probe can end with (the probe ends its hold at the first tick where the
          tracked ball has moved 2 cm, or at 28 ticks = 0.56 s; `feasible_stops` takes the lengths whose trigger is possible for some ball of the
          predicted set. trigger_aware=0 covers every length 4..28 including 'lowest mu and the full 0.56 s hold', which the controller never
          does and which no mu <= 0.0005-ish start survives: the pass rate is then ~0),
          followed by F5's BACKUP law (reach.backup_cmd) for `h_after` more ticks. With backup=1 (default) its gains are the probe's balance gains
          (kd 10, kp 25, tilt <= 0.12, rate gain 1/tau_p): with the tray at rest reach.backup_cmd IS probe._balance (c_des = -(1+k)(kd v + kp x)/g).
          With F5's own gentle gains (backup=0: kd 3, kp 2) the ball cannot be caught inside the tray and the certificate fails on every carry;
        - the error set around that centre trajectory is reach._axis_gens (k_sigma Kalman sigmas of the initial estimate, the eps_a model-error
          box, ball-feedback damping of the backup only after the profile + fb_delay ticks);
        - tray_spec: G inside(ball, shrink(tray_top, d_margin)) evaluated with ir.margin(robust=True) on the predicted sets. PASS iff the
          robust margin >= 0 for every stop length, i.e. every point of the predicted ball set keeps >= d_margin (2 cm) from the tray edge at
          every tick.
      Fails -> NO probe; carry at P6's intensity (the population choice).
P7s  = P7's decision rule with the ORIGINAL shadow certificate (Run.certify, true MuJoCo state): separates rule change from certificate change.
      Fails -> no probe, P6's intensity (same fallback as P7).

Known differences between the certificate and the real probe (honest): the real balance law adds a stiction-break tilt for a slow ball off the
centre and ends when the ball is still; the certificate keeps the feedback on to the end of the horizon. p7_checks.py `certvalid` measures the
true edge distance of the real probe on carries where the certificate passes. The tray yaw is ignored (as in F5); the tracker/model error is the F5 default.

P8 hook (below): no probe; an ONLINE belief over mu from the carry's own motion re-picks the intensity at every transport boundary.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import numba as nb

from world_model.v2.tray_rep import common as K          # noqa: F401
from world_model.v2.tray_rep.sets import ir
from world_model.v2.tray_rep.sets import reach as RC
from world_model.v2.tray_rep.sets import tray_spec as TS
from world_model.v2.tray_rep.boed import policy as PO

Q_DECISION = 0.025            # decision quantile of the posterior over mu (lower end of the central 95% credible set)
Q_CERT = 0.01                 # certificate quantile (as in Run.certify)

TRAY_SPEC_EDGE = {"G": {"lo": "lo", "hi": "H", "of": {
    "inside": {"a": "ball", "b": {"shrink": {"of": "tray_top", "by": "r_shrink"}}, "margin": 0.0, "tag": "edge"}}}}


# ====================================================================================== decision rule
def pick_worst_case(table, mu_q, eps=PO.EPS, frac=1.0):
    """index (into table.igrid) of the fastest intensity whose table drop rate AT mu_q is <= eps; the slowest if none.
    frac < 1: a carry that still has `frac` of its path to run (constant hazard, 1 - (1 - d)^frac), as P4 / P8 scale the table."""
    d = table.drop_at(mu_q)
    if frac < 1.0:
        d = 1.0 - (1.0 - d) ** frac
    ok = np.flatnonzero(d <= eps)
    return int(ok[0]) if len(ok) else len(table.igrid) - 1


def decide(table, belief, eps=PO.EPS, q=Q_DECISION):
    """-> (mu_q, index): the worst-case rule on a belief."""
    mu_q = belief.lower_mu(q)
    return mu_q, pick_worst_case(table, mu_q, eps)


# ====================================================================================== set certificate
@dataclass
class P7Spec:
    d_margin: float = 0.02          # [m] every point of the predicted ball set keeps this edge distance
    h_after: int = 60               # backup ticks after the probe profile (F5's horizon)
    stop_min: int = 4               # hold lengths covered: stop_min, stop_min + stop_step, ..., n_hold_max (and n_hold_max)
    stop_step: int = 2
    k_sigma: float = 2.5            # F5 defaults
    eps_a: float = 0.04             # [m/s^2] model-error box. F5's 0.08 is for carries (moving tray, yaw); the probe stands still. VAL probes (200 per setup, mu 5e-4..6e-3):
    #                                 predicted set edge distance <= true min edge distance in 100% for eps_a >= 0.02 (min slack 0.3 cm); 0.04 = one grid step above (min slack 0.6 cm)
    eps_p0: float = RC.ReachParams().eps_p0
    backup: int = 1                 # 1: F5's backup law (reach.backup_cmd) with the PROBE's balance gains (kd, kp, tilt limit, 1/tau_p), which is the
    #                                 same feedback law as probe._balance (tray at rest: the brake term is 0); 0: F5's own gains (kd 3, kp 2, tilt 0.2, kappa 6)
    kd: float = 3.0                 # used when backup = 0
    kp: float = 2.0
    kappa_t: float = 6.0
    tilt_max: float = 0.2
    fb_delay: int = 5
    tau_b: float = 0.5
    a_brake_frac: float = 0.5
    trigger_aware: int = 1          # 1: only hold lengths the probe's 2 cm trigger can produce (see SetCertificate.feasible_stops); 0: every length 4..28
    trig_early: int = 3             # ticks: hold lengths h* - trig_early .. h* + trig_late are certified (h* = the nominal ball's trigger step)
    trig_late: int = 1              # VAL: the real hold (tracker noise) ended at most 1 tick after h* in 200 probes (piper_trk) and 0 in 200 (piper); earlier in the rest
    q_cert: float = Q_CERT
    q_decision: float = Q_DECISION

    @staticmethod
    def parse(s):
        sp = P7Spec()
        for kv in (s or "").split(","):
            if kv:
                k, v = kv.split("=")
                setattr(sp, k, int(float(v)) if isinstance(getattr(sp, k), int) else float(v))
        return sp


@nb.njit(cache=True)
def _clip(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


@nb.njit(cache=True)
def probe_rollout(s0, c0, vt0, w0, n_pre, n_prof, sth, cth, tau_p, H, mu_eff, par, S, C):
    """centre rollout of the probe's planned profile: ticks < n_pre level, ticks < n_prof tilt toward (sth, 0, cth) (open loop in the command, closed
    loop on the realised axis, exactly the probe.level_command / tilt_to laws), then the F5 BACKUP law until H. Fills S (H+1, 4), C (H+1, 3)."""
    k_in, r, alpha, w_max = par[9], par[10], par[2], par[1]
    s = s0.copy()
    c = c0.copy()
    vt = vt0.copy()
    w = w0.copy()
    S[0] = s
    C[0] = c
    for k in range(H):
        if k < n_prof:
            cd = np.zeros(3)
            if k < n_pre:
                cd[2] = 1.0
            else:
                cd[0] = sth
                cd[2] = cth
            wc = np.zeros(3)
            wc[0] = _clip((c[1] * cd[2] - c[2] * cd[1]) / tau_p, -w_max, w_max)
            wc[1] = _clip((c[2] * cd[0] - c[0] * cd[2]) / tau_p, -w_max, w_max)
            a = np.zeros(3)
        else:
            a, wc = RC.backup_cmd(c, vt, s, par)
        c2, vt2, w2 = RC.tray_step(c, vt, w, a, wc, alpha)
        s, g, f = RC.ball_step(s, c, c2, a, mu_eff, k_in, r)
        c, vt, w = c2, vt2, w2
        S[k + 1] = s
        C[k + 1] = c


class SetCertificate:
    """analytic set certificate of the probe (see module docstring). One per process; `check` allocates nothing persistent."""

    def __init__(self, renv, probe_spec, p7: P7Spec | None = None):
        self.sp, self.ps = p7 or P7Spec(), probe_spec
        self.renv = renv
        self.ents = TS.pred_entities("centre")
        self.poly = RC._poly(8)
        self._par_cache = {}

    def params(self, mu_w):
        """ReachParams of one certificate: mu_lo = mu_hi = the belief's worst-case mu (a single rolling resistance)."""
        sp, r, ps = self.sp, self.renv, self.ps
        kd, kp, kap, tmax = (ps.kd, ps.kp, 1.0 / ps.tau_p, ps.theta_bal) if sp.backup else (sp.kd, sp.kp, sp.kappa_t, sp.tilt_max)
        return RC.ReachParams(H=sp.h_after, mu_lo=float(mu_w), mu_hi=float(mu_w), k_sigma=sp.k_sigma, eps_a=sp.eps_a, eps_p0=sp.eps_p0,
                              kd=kd, kp=kp, kappa_t=kap, tilt_max=tmax, fb_delay=sp.fb_delay, tau_b=sp.tau_b, a_brake_frac=sp.a_brake_frac, calib=True,
                              a_max=float(r.a_max), w_max=float(r.w_max), alpha=float(r.alpha or 1e3))

    def stops(self):
        ps = self.ps
        s = list(range(self.sp.stop_min, ps.n_hold_max + 1, self.sp.stop_step))
        if s[-1] != ps.n_hold_max:
            s.append(ps.n_hold_max)
        return s

    def predict(self, x, v, Pn, obs0, direction, mu_w, n_hold):
        """the Prediction (tray frame, ticks 0..H) of the probe profile with a hold of `n_hold` ticks followed by the backup, mu = mu_w."""
        ps, sp = self.ps, self.sp
        rp = self.params(mu_w)
        par = rp.par()
        n_pre, n_prof = int(ps.n_pre), int(ps.n_pre) + int(n_hold)
        H = n_prof + sp.h_after
        S, C = np.zeros((H + 1, 4)), np.zeros((H + 1, 3))
        th = ps.theta * direction
        s0 = np.array([x[0], x[1], v[0], v[1]], np.float64)
        probe_rollout(s0, np.asarray(obs0[0:3], np.float64), np.asarray(obs0[3:6], np.float64), np.asarray(obs0[6:9], np.float64),
                      n_pre, n_prof, float(np.sin(th)), float(np.cos(th)), float(ps.tau_p), H, float(RC.MU_EFF(mu_w)), par, S, C)
        hw = np.zeros((H + 1, 2))
        Pa = np.asarray(Pn, np.float64)                      # per-axis (pos, vel) covariance of the probe filter (same for both axes)
        for a in range(2):
            r_p, _ = RC._axis_gens(Pa, rp.k_sigma, self.poly, H, rp.eps_a, rp.eps_p0, rp.kd, rp.kp, n_prof + rp.fb_delay, 0, 0.0)
            hw[:, a] = r_p
        return RC.Prediction(centre=S[:, :2].copy(), hw=hw, S_mid=S, C=C, J=None, Jc=None, u0=np.zeros(6)), H

    def margin(self, pred, H):
        """robust margin [m] of 'ball set stays inside the tray shrunk by d_margin' over ticks 0..H (>= 0 = pass)."""
        st = TS.PredState(pred)
        spec = TS.build_spec(TRAY_SPEC_EDGE, self.ents, H, r_shrink=self.sp.d_margin, theta_max=0.3, lo=0)
        return ir.margin(spec, st, True)

    def feasible_stops(self, x, v, Pn, obs0, direction, mu_w):
        """hold lengths h (tilt commands) the probe can end with. The probe ends the hold at the first hold step i >= 4 at which the filtered ball has
        moved d_trig along the tilt direction, or at n_hold_max. A ball that rolls freely (low mu) therefore ends its hold early; asking the
        certificate to survive 'lowest mu AND the full 0.56 s hold' asks for something the controller never does (trigger_aware=0 does ask it).
        trigger_aware=1: h* = the trigger step of the NOMINAL ball (centre rollout at mu_w; n_hold_max if it never fires); the certificate covers
        h in [h* - trig_early, h* + trig_late] (clipped to [stop_min, n_hold_max]); the set around each of those rollouts carries the position
        uncertainty after the stop. The window is calibrated on VAL probes (the tracked ball fires earlier than the nominal ball, at most 1 tick later),
        not a proof: a ball that is slower fires later but also arrives slower. The soundness of the pass decision is MEASURED against MuJoCo (p7_checks certvalid).
        (An interval version, 'some member of the k_sigma / eps_a set has not fired yet', was tried: the set is 2-3 cm wide at the trigger time, as wide
        as the trigger itself, so it keeps every length up to 0.56 s feasible and is the same as trigger_aware=0.)"""
        ps, sp = self.ps, self.sp
        if not sp.trigger_aware:
            return self.stops(), None
        pred, H = self.predict(x, v, Pn, obs0, direction, mu_w, ps.n_hold_max)
        disp = (pred.centre[:, 0] - x[0]) * direction
        hs = ps.n_hold_max
        for i in range(sp.stop_min, ps.n_hold_max):
            if disp[ps.n_pre + i] >= ps.d_trig:
                hs = i
                break
        lo, hi = max(sp.stop_min, hs - sp.trig_early), min(ps.n_hold_max, hs + sp.trig_late)
        return list(range(lo, hi + 1)), (pred, H)

    def check(self, kf, obs0, mu_w):
        """-> dict(ok, dmin (predicted min edge distance of the set, m), margin, stop_worst, ms, direction, mu_w). kf: the probe's ball filter at probe start."""
        t0 = time.perf_counter()
        x, v, Pn = kf.now(with_cov=True)
        direction = -1.0 if x[0] > 0 else 1.0                # the probe's own direction rule (ProbeController._begin_pre)
        worst, h_worst = 1e9, -1
        stops, cached = self.feasible_stops(x, v, Pn, obs0, direction, mu_w)
        for h in stops:
            if cached is not None and h == self.ps.n_hold_max:
                pred, H = cached
            else:
                pred, H = self.predict(x, v, Pn, obs0, direction, mu_w, h)
            m = self.margin(pred, H)
            if m < worst:
                worst, h_worst = m, h
        return dict(ok=bool(worst >= 0.0), margin=float(worst), dmin=float(worst + self.sp.d_margin), stop_worst=int(h_worst), stops=(min(stops), max(stops)),
                    ms=1e3 * (time.perf_counter() - t0), direction=float(direction), mu_w=float(mu_w))


# ====================================================================================== P7 / P7s
def run_policy(run, kind):
    """P7 (kind 'P7', set certificate) or P7s (kind 'P7s', shadow certificate) on a `boed.episode.Run`. Duck-typed on run / run.c (the Ctx)."""
    from world_model.v2.tray_rep.boed import probe as PB
    c = run.c
    table = c.table
    run.kf = PB.make_kf(c.trk, c.noisefree)
    run.kf.reset_prior(sx=0.006, sv=0.01)
    run.kf.feed(run.obs[0], 0)
    mu_w = run.belief.lower_mu(c.p7.q_cert)
    if not c.cert:
        ok, ckind = True, "none"
    elif kind == "P7":
        ckind = "set"
        if getattr(c, "_setcert", None) is None:
            c._setcert = SetCertificate(c.renv, c.spec, c.p7)
        res = c._setcert.check(run.kf, run.obs[0], mu_w)
        ok = res["ok"]
        run.info["cert_n"] += 1
        run.info.update(cert_ok=bool(ok), cert_dmin=res["dmin"], cert_label=None, cert_mu_worst=float(mu_w))
        run.extra.update(cert_stop_worst=res["stop_worst"])
    elif kind == "P7s":
        ckind = "shadow"
        ok = run.certify(False)
    else:
        raise ValueError(kind)
    run.extra.update(cert_kind=ckind, cert_pass=bool(ok))
    if ok:
        run.dlog = []
        run.probe(False)
        run.extra["probe_dmin_true"] = float(min(run.dlog)) if run.dlog else None
        run.dlog = None
    else:
        run.info["probe_skipped"] = "cert_failed"
    mu_q = run.belief.lower_mu(c.p7.q_decision)
    run.extra["mu_q"] = float(mu_q)
    if run.dropped:
        run.extra["I_chosen"] = None
        return
    if ok:
        idx = pick_worst_case(table, mu_q, c.eps)
        run.extra["rule"] = "worstcase_q"
    else:
        idx = table.choose(PO.w_prior(), c.eps)               # P6's population choice
        run.extra["rule"] = "p6_fallback"
    I = float(table.igrid[idx])
    run.extra["I_chosen"] = I
    run.carry_phase(run.plan_for(I))


# ====================================================================================== P8 hook
def run_p8(run):
    """P8 (NOT IMPLEMENTED): no probe. An ONLINE belief over mu_roll, fed by the carry's own motion, re-picks the intensity at every transport boundary
    with the same worst-case rule as P7 and the table scaled by the remaining path fraction (as P4: constant hazard, 1 - (1 - d)^frac).

    Wiring plan for the owner (the estimator is built separately: sets/vlmprior/online_mu.py):
        est = OnlineMu(prior=..., ...)                     # interface: est.feed(obs_row, act_row)  and  est.quantile(q) -> mu
        plan = run.plan_for(run.c.table.igrid[<P6 population index>])        # start at P6's intensity (the prior's worst-case rule)
        loop over the three transports of the plan (CA.Plan transport boundaries = plan.ref[i] ends):
            run each transport with run.carry_phase-like stepping, calling est.feed(run.obs[run.n], run.act[run.n - 1]) after every run.tick
            at each boundary:   frac = max(0.05, 1 - run.prog.done / run.prog.len_nom)
                                mu_q = est.quantile(PO-level Q_DECISION)          # 0.025
                                idx  = pick_worst_case(run.c.table, mu_q, run.c.eps, frac)       # defined above, handles `frac`
                                plan.replan_from(t_boundary, run.c.renv.env.pos[:2], table.igrid[idx] * run.I_feas)     # as Run._p4 does
                                run.I_hist.append(table.igrid[idx])
        record extras (run.extra): mu_q_hist, I_chosen_hist, est_kind.
    Needs a carry-phase variant that stops at the transport boundaries (Run.carry_phase(plan, quiet=...) only stops on an F3 trigger), i.e. a
    `stop_at` list of plan times. F3 stays on in every phase.
    """
    raise NotImplementedError("P8: wire sets/vlmprior/online_mu.py (feed(obs_row, act_row), quantile(q)) as described in the docstring")
