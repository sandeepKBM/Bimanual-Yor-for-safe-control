"""al/policies.py -- SAFE ACTIVE-LEARNING policies for the tray-and-ball carry (AL-1). SIMULATION ONLY. CPU. No hardware, no network.

Krause-style split: OPTIMISTIC information gathering (a probe / the online belief is used to NARROW the mu set), PESSIMISTIC safety (F5 always guards with the whole set;
the set is only ever an interval the policy believes covers the truth at >= 95%; the probe is run only if the robot-computable SetCertificate passes at the set's
WORST-CASE mu), TRANSDUCTIVE (probe only where the information changes the decision: expected carry-time saving > probe time).

A policy is a small state machine driven by the runner (run_al.py) while the hidden mu is poisoned:
    begin_episode(prior, l_oracle)      prior = (m, sigma, confident) from the synthetic VLM prior (drawn BEFORE the carry from the truth by the prior simulator, as
                                         F5P does); l_oracle is passed ONLY to the oracle policy (None for every other policy; asserted)
    pre_carry(run) -> I_rel             pickup: certificate / probe / decision; returns the carry speed
    carry_begin(renv)                   reset the online belief
    on_tick(t, obs_row, act_row) -> (lo_l, hi_l) | None   feed the belief; every `upd` ticks return the mu set (in l = ln mu) for F5
    finish(obs_row, act_row)            last observation row
Spec string  "AL_<kind>[:key=val,...]"  kinds: static online online0 probe transductive oracle unif point
    common   calreg (1: lam/sa of the regime-specific closed-loop recalibration instead of the pooled one) rho (prior error rate, default 0.2) pseed upd (5 = the 'every k ticks' of the set refresh) lam sa (online belief; default = recalibrated, see CAL_FILE)
    static   conformal prior set (F5P arm=conf, online=0): [m -+ q sigma] intersected with the population range
    ivmode   'hpd' (default for online / probe / transductive; central for the reference arms and online0; hull of the highest-density cells holding 95%: a flat belief gives the whole range) | 'central' (equal tails = F5P)
    conflict thr>0 (online, cum=1): if post_mass(cur)/prior_mass(cur) < thr the current set is dropped for the online hull (prior-data conflict; default off)
    lvl      level of the applied hull (default 0.99 for AL_online with cum=1, else 0.95)
    online   S0 as static, then cumulative intersection  cur <- cur ∩ (lvl hull of the online belief)  every upd ticks (empty -> the interval); cum=0: S0 ∩ interval
             (= F5P arm=conf with the recalibrated belief)
    online0  online with cum=0
    probe    S0; at pickup certify at mu_w = exp(S0.lo) and probe with the LARGEST certified tilt of `menu` (default 0.12/0.06; `theta=x` fixes one tilt) if any passes; posterior (uniform prior on S0 x probe LL^lam_p) ->
             set = 95% interval ∩ S0 for the whole carry (online=1: the online LL is multiplied in and the set keeps shrinking during the carry)
    transductive  as probe, but the probe (tilt from `menu`, default 0.06/0.12, '/'-separated) is run only if
             E[ time(I*(S1), S1) ] + probe_s  <  time(I*(S0), S0) - margin   (value.ValueModel, probe_voi.ProbeVOI), and the carry speed is I*(set) = value.choose_I
    oracle   [mu, mu]            unif  population range ∩ online 95% interval (online=0: the plain population range)
    point    Phys2Real-style inverse-variance fusion of the prior N(m, sigma) with the online Gaussian approximation -> point set
Fixed speed I = `I` (default 0.99, the rebaseline value) unless ichoose=1 (transductive: default 1).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep.al import belief_v2 as BV
from world_model.v2.tray_rep.al import probe_v2 as PV
from world_model.v2.tray_rep.sets.vlmprior import conformal as CF
from world_model.v2.tray_rep.sets.vlmprior import online_mu as OM

DT = 0.02
CAL_FILE = K.ROOT / "al" / "online_calib_cl.json"     # written by calibrate_lam.py (closed-loop recalibration); fallback = rebaseline_v2 values
LAM_DEFAULT, SA_DEFAULT = 0.1, 0.04                    # rebaseline_v2/vlmprior_cal/online_calib.json (open-loop carries)
KINDS = ("static", "online", "online0", "probe", "transductive", "oracle", "unif", "point")
CHECKS = {"t0": 0, "t1s": 50, "t2s": 100, "t4s": 200}


def load_cal(regime=None, per_regime=False):
    """(lam, sigma_a) of the closed-loop recalibration (per regime key if present), else the rebaseline defaults."""
    try:
        d = json.loads(CAL_FILE.read_text())["chosen"]
        d = (d.get(regime) or d.get("all")) if per_regime else (d.get("all") or d.get(regime))
        return float(d["lam"]), float(d["sigma_a"])
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
        return LAM_DEFAULT, SA_DEFAULT


def parse(spec):
    name, _, rest = spec.partition(":")
    assert name.startswith("AL_"), spec
    kind = name[3:]
    assert kind in KINDS, kind
    kw = {}
    for kv in rest.split(","):
        if kv:
            k, v = kv.split("=")
            kw[k] = v
    return kind, kw


class ALPolicy:
    def __init__(self, spec, renv, regime="normal", value=None, voi=None, rho=None, pseed=None, log_ll=False):
        self.spec = spec
        self.kind, kw = parse(spec)
        self.kw = kw
        g = lambda k, d, t=float: t(kw[k]) if k in kw else d
        self.rho = g("rho", 0.2 if rho is None else rho)
        self.pseed = g("pseed", 0 if pseed is None else pseed, int)
        self.upd = max(g("upd", 5, int), 1)
        lam_c, sa_c = load_cal(regime, bool(g("calreg", 0, int)))
        self.lam, self.sa = g("lam", lam_c), g("sa", sa_c)
        self.cum = bool(g("cum", 0 if self.kind == "online0" else 1, int))
        self.I_fixed = g("I", 0.99)
        self.theta = g("theta", 0.12)
        self.menu = [float(x) for x in kw["menu"].split("/")] if "menu" in kw else [0.06, 0.12]
        self.lam_p = g("lam_p", 0.5)
        self.online = bool(g("online", {"static": 0, "oracle": 0, "probe": 0, "transductive": 0}.get(self.kind, 1), int))
        self.ichoose = bool(g("ichoose", 1 if self.kind == "transductive" else 0, int))
        self.lvl = g("lvl", 0.99 if (self.cum and self.kind == "online") else 0.95)         # level of the hull applied/intersected every update (cum=1: 0.99, union-bound style safety)
        self.ivmode = kw.get("ivmode", "central" if self.kind in ("unif", "online0", "point", "static", "oracle") else "hpd")
        self.conflict = g("conflict", 0.0)
        self.eps = g("eps", 0.0)
        self.margin = g("margin", 0.0)
        self.drop_pen = g("drop_pen", 30.0)
        self.probe_always = bool(g("always", 0, int))        # transductive test hook: skip the value test (not used in experiments)
        self.value, self.voi = value, voi
        self.reads_truth = self.kind == "oracle"
        self.renv = renv
        self.nf = renv.tracker is None
        self.q = CF.load_q(self.rho) if self.kind in ("static", "online", "online0", "probe", "transductive") else None
        self.log_ll = bool(log_ll)
        self.online_bel = BV.OnlineLL(renv, self.lam, self.sa) if (self.online or self.kind in ("point", "unif", "online", "online0")) else None
        self.bank = [BV.OnlineLL(renv, 1.0, s) for s in (0.04, 0.06, 0.08)] if log_ll else []
        self.use_online_feed = self.online_bel is not None and not (self.kind == "unif" and not self.online)
        self.reset()

    def reset(self):
        self.S0 = None
        self.cur = None
        self.prior = None
        self.l_oracle = None
        self.LL_probe = None
        self.base = None               # interval the online intersection starts from
        self.log = dict(probe_considered=False, cert_ok={}, probe_run=False, probe_skip="", probe_theta=None, probe_ticks=0, probe_drop=False, voi=None, I_rel=None,
                        n_empty=0, n_upd=0, S0=None, S_post=None, probe_s=0.0)
        self.hist = []                  # (t, lo_l, hi_l) change points of the applied set (carry ticks)
        self.ll_log = {}
        self._t_last = -1

    # ------------------------------------------------------------ episode
    def begin_episode(self, prior, l_oracle):
        self.reset()
        if self.online_bel is not None:
            self.online_bel.reset()
        for b in self.bank:
            b.reset()
        if self.reads_truth:
            assert l_oracle is not None
        else:
            assert l_oracle is None, "only the oracle policy may be handed the truth"
        self.prior, self.l_oracle = prior, l_oracle
        m, sg, _ = prior
        pop = BV.L_POP
        if self.kind == "oracle":
            self.S0 = (l_oracle, l_oracle)
        elif self.kind == "unif":
            self.S0 = pop
        elif self.kind == "point":
            self.S0 = pop
        else:
            lo, hi = m - self.q * sg, m + self.q * sg
            lo, hi = max(lo, pop[0]), min(hi, pop[1])
            self.S0 = (lo, hi) if lo <= hi else pop           # = F5P arm=conf with the population range as the uninformative interval
        self.cur = self.S0
        self.base = self.S0
        self.log["S0"] = [float(self.S0[0]), float(self.S0[1])]

    # ------------------------------------------------------------ pickup
    def _iv(self, w):
        return BV.interval(w, self.lvl, self.ivmode)

    def _speed(self, S):
        if not self.ichoose:
            return self.I_fixed
        return float(self.value.choose_I(S[1] - S[0], self.eps))

    def pre_carry(self, run):
        L = self.log
        if self.kind in ("probe", "transductive"):
            L["probe_considered"] = True
            PV.prepare(run)
            mu_w = float(np.exp(self.S0[0]))
            thetas = ([self.theta] if "theta" in self.kw else sorted(self.menu, reverse=True)) if self.kind == "probe" else list(self.menu)
            ok = {}
            for th in thetas:
                ok[th] = PV.certify(run, th, mu_w)
            L["cert_ok"] = {f"{th:g}": bool(v["ok"]) for th, v in ok.items()}
            L["cert_dmin"] = {f"{th:g}": v["dmin"] for th, v in ok.items()}
            cands = [th for th in thetas if ok[th]["ok"]]
            run.info["cert_n"] += len(thetas)
            run.info.update(cert_ok=bool(cands), cert_dmin=float(max(v["dmin"] for v in ok.values())), cert_mu_worst=mu_w)
            th_run = None
            if not cands:
                L["probe_skip"] = "cert_failed"
            elif self.kind == "probe":
                th_run = cands[0]
            else:
                th_run = self._decide(cands)
                if th_run is None:
                    L["probe_skip"] = "no_value"
            if th_run is not None:
                res = PV.probe(run, th_run)
                L.update(probe_run=True, probe_theta=float(th_run), probe_ticks=res["ticks"], probe_drop=res["dropped"], probe_s=res["ticks"] * DT)
                if res["ll"] is None and not res["dropped"]:
                    L["probe_skip"] = "no_ll"
                self.LL_probe = res["ll"]
            else:
                run.info["probe_skipped"] = L["probe_skip"]
            run.kf = None
            if self.LL_probe is not None:
                self._after_probe()
        I = self._speed(self.cur)
        L["I_rel"] = I
        return I

    def _after_probe(self):
        w = BV.posterior([(self.LL_probe, self.lam_p)], prior=BV.set_prior(*self.S0))
        iv = self._iv(w)
        s = BV.intersect(iv, self.S0) or iv
        self.cur = self.base = s
        self.log["S_post"] = [float(s[0]), float(s[1])]
        self.post_w = w

    def _decide(self, cands):
        """transductive value of information of each certified tilt; returns the tilt to run or None."""
        L = self.log
        S0 = self.S0
        w0 = BV.set_prior(*S0)
        T0 = self.value.time(self.value.choose_I(S0[1] - S0[0], self.eps), S0[1] - S0[0])
        I0 = self.value.choose_I(S0[1] - S0[0], self.eps)
        best, rows = None, {}
        for th in cands:
            ent = self.voi.entries(th, w0) if self.voi is not None else None
            if not ent:
                continue
            ts, wc, ws, pchg, pen = [], [], 0.0, 0.0, 0.0
            for (wt, ll, ticks, dropped) in ent:
                if ll is None or dropped:
                    ts.append((wt, T0 + ticks * DT + self.drop_pen)); continue
                post = BV.posterior([(ll, self.lam_p)], prior=w0)
                iv = self._iv(post)
                S1 = BV.intersect(iv, S0) or iv
                I1 = self.value.choose_I(S1[1] - S1[0], self.eps)
                ts.append((wt, self.value.time(I1, S1[1] - S1[0]) + ticks * DT))
                ws += wt * (S1[1] - S1[0]); pchg += wt * float(I1 != I0)
            Tm = float(sum(wt * t for wt, t in ts) / sum(wt for wt, _ in ts))
            rows[f"{th:g}"] = dict(T=Tm, w1=ws / max(sum(wt for wt, _ in ts), 1e-12), p_change_I=pchg / max(sum(wt for wt, _ in ts), 1e-12), n=len(ent))
            if best is None or Tm < best[1]:
                best = (th, Tm)
        L["voi"] = dict(T0=float(T0), I0=float(I0), w0=float(S0[1] - S0[0]), options=rows, margin=self.margin, value=self.value.describe())
        if best is None:
            return None
        L["voi"]["best"] = dict(theta=best[0], T=best[1], saving=float(T0 - best[1]))
        return best[0] if (self.probe_always or best[1] + self.margin < T0) else None

    # ------------------------------------------------------------ carry
    def carry_begin(self):
        if self.online_bel is not None:
            self.online_bel.reset()
        for b in self.bank:
            b.reset()
        self._t_last = -1

    def _online_parts(self):
        parts = []
        if self.LL_probe is not None:
            parts.append((self.LL_probe, self.lam_p))
        if self.online_bel is not None and self.use_online_feed:
            parts.append((self.online_bel.LL, self.lam))
        return parts

    def _iv_online(self):
        w = BV.posterior(self._online_parts(), prior=(BV.set_prior(*self.S0) if self.kind in ("probe", "transductive") else None))
        return self._iv(w), w

    def on_tick(self, t, obs_row, act_row):
        self._t_last = t
        if self.use_online_feed:
            self.online_bel.feed(obs_row, act_row)
        for b in self.bank:
            b.feed(obs_row, act_row)
        if self.bank and t % 25 == 0:
            self.ll_log[str(t)] = [b.LL.tolist() for b in self.bank]
        if t % self.upd != 0:
            return None
        k = self.kind
        if k == "oracle" or k == "static":
            s = self.S0
        elif k == "unif":
            if not self.online:
                s = BV.L_POP
            else:
                iv, _ = self._iv_online()
                s = BV.intersect(BV.L_POP, iv) or iv
        elif k == "point":
            mo, so = self.online_bel.mean_sd_log()
            m, sg, _ = self.prior
            from world_model.v2.tray_rep.sets.vlmprior.filter import fuse
            lf, _ = fuse(m, sg, mo, so)
            s = (lf, lf)
        elif k in ("probe", "transductive") and not self.online:
            s = self.cur
        else:                                            # online / online0 / probe+online
            iv, w = self._iv_online()
            self.log["n_upd"] += 1
            if self.conflict > 0 and self.cum:               # prior-data conflict: the data put much less mass on the current set than the flat prior did -> drop the set
                ov = np.clip(np.minimum(BV.OM.EDGES[1:], self.cur[1]) - np.maximum(BV.OM.EDGES[:-1], self.cur[0]), 0.0, None)
                post = float((w * ov / BV.OM.WIDTH).sum())
                prior_m = float(max(self.cur[1] - self.cur[0], 1e-9) / (BV.L_POP[1] - BV.L_POP[0]))
                if post / prior_m < self.conflict:
                    self.log["n_conflict"] = self.log.get("n_conflict", 0) + 1
                    self.cur = iv
            if self.cum:
                s = BV.intersect(self.cur, iv)
                if s is None:
                    self.log["n_empty"] += 1
                    s = iv
            else:
                s = BV.intersect(self.base, iv)
                if s is None:
                    self.log["n_empty"] += 1
                    s = iv
            self.cur = s
        self.cur = s
        return s

    def finish(self, obs_row, act_row, t_end):
        """feed the last observation row (the belief at the END of the carry has seen it); t_end = number of carry ticks run."""
        if self.use_online_feed and self.online_bel.t < t_end:
            self.online_bel.feed(obs_row, act_row)
        for b in self.bank:
            if b.t < t_end:
                b.feed(obs_row, act_row)
        if self.bank:
            self.ll_log["end"] = [b.LL.tolist() for b in self.bank]
            self.ll_log["t_end"] = int(t_end)

    # ------------------------------------------------------------ logging (truth allowed: called after the carry)
    def end_stats(self, l_true):
        out = {}
        L = self.log
        out["alp_kind"], out["alp_spec"] = self.kind, self.spec
        out["alp_lam"], out["alp_sa"], out["alp_cum"], out["alp_rho"], out["alp_upd"] = self.lam, self.sa, self.cum, self.rho, self.upd
        for k, v in L.items():
            out[f"alp_{k}"] = v
        out["alp_S0_cov"] = None if self.S0 is None else bool(self.S0[0] - 1e-9 <= l_true <= self.S0[1] + 1e-9)
        out["alp_S0_lw"] = None if self.S0 is None else float(self.S0[1] - self.S0[0])
        if self.LL_probe is not None:
            out["alp_probe_post"] = BV.cover_stats(self.post_w, l_true)
            out["alp_probe_S_cov"] = bool(self.log["S_post"][0] - 1e-9 <= l_true <= self.log["S_post"][1] + 1e-9)
        if self.use_online_feed and self.online_bel.t >= 0:
            w = BV.posterior([(self.online_bel.LL, self.lam)])
            out["alp_online_end"] = BV.cover_stats(w, l_true)
            out["alp_n_samples"] = self.online_bel.n_samples
            if self.online or self.kind in ("online", "online0"):
                w2 = BV.posterior(self._online_parts(), prior=(BV.set_prior(*self.S0) if self.kind in ("probe", "transductive") else None))
                out["alp_comb_end"] = BV.cover_stats(w2, l_true)
        if self.ll_log:
            out["alp_ll"] = self.ll_log
        return out
