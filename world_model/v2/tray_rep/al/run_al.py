"""al/run_al.py -- runner of the safe active-learning policies (al/policies.py) on closed-loop tray-and-ball carries. SIMULATION / SAVED DATA ONLY. CPU. No hardware, no network.

One call = one policy x regime x seed x shard on VAL or TEST carries (data_piper_trk_full_v2, FIXED physics, calib_v2 map, F5 eps_a = 0).
Reuses (imports, no edits): boed.run_cl.RunCL (closed-loop carry plan), boed.plan_cl, boed.episode.Ctx/Run (record), boed.p8.HiddenMu (poison object), wrongpoint.run_cell.mu_for
(low-mu population) and the run_cl_wp convention for arm authority (a_scale, w_scale x f on a NEW TrayRepEnv built before the filter), sets.filter.F5SetFilter.

HIDDEN-MU GUARANTEE (every policy except oracle): while the pickup/probe/carry decision path runs, Run.mu, renv.mu_roll and renv.env.p.mu_roll are replaced by a poison object
(any use raises; MuJoCo reads its own copy in geom_friction). The synthetic VLM prior is drawn from the truth BEFORE the guard (it simulates an outside source, as F5P does) and
the truth is read again only after the carry, for LOGGING (alp_* / x_p_cov_* fields). The oracle policy alone is handed the true l.

Regimes (--regime):  normal    auth 1,    full mu population [5e-4, 6e-3],        prior error rate rho 0.2
                     stressed  auth 0.25, low-mu population mu_for in [5e-4, 1.5e-3], rho 0.4        (wrongpoint 'lowI' cell)
Record = run_cl fields (Run.record + cl_*) + filter stats (x_f5_*, x_p_* set checkpoints/coverage) + alp_* (policy log: set history, probe decision / certificate / ticks, belief
coverage at the end of the carry, probe posterior) + x_alp_* (applied-set log-width).
  python -m world_model.v2.tray_rep.al.run_al --policy AL_online --regime normal --split val --n-eps 20 --workers 2 --out <dir>
Output: <out>/<regime>/<policy tag>/AL__s<seed>[__p<i>of<n>].jsonl   (rerun = resume by episode)
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep.al import belief_v2 as BV
from world_model.v2.tray_rep.al import policies as POL
from world_model.v2.tray_rep.al import value as VAL
from world_model.v2.tray_rep.boed import episode as EP
from world_model.v2.tray_rep.boed import plan_cl as PC
from world_model.v2.tray_rep.boed import policy as PO
from world_model.v2.tray_rep.boed import run as BR
from world_model.v2.tray_rep.boed import run_cl as RCL
from world_model.v2.tray_rep.boed.p8 import HiddenMu
from world_model.v2.tray_rep.bridge.filters import project_box
from world_model.v2.tray_rep.sets.filter import F5SetFilter
from world_model.v2.tray_rep.sets.vlmprior import prior as PR
from world_model.v2.tray_rep.store import Store
from world_model.v2.tray_rep.wrongpoint.run_cell import mu_for

DT = EP.DT
DATA = str(K.ROOT / "data_piper_trk_full_v2")
OUT = K.ROOT / "al" / "runs"
REGIMES = {"normal": dict(auth=1.0, mu_range=None, rho=0.2), "stressed": dict(auth=0.25, mu_range=(5e-4, 1.5e-3), rho=0.4)}
POPS = {"normal": "full", "stressed": "low"}
CK = {"t0": 0, "t1s": 50, "t2s": 100, "t4s": 200}
G = {}


# ====================================================================================== filter
class ALFilter(F5SetFilter):
    """F5 whose mu set is handed over by the policy every `upd` ticks (policy.on_tick). Never reads the hidden mu (begin ignores ctx['hid'])."""
    name = "AL"

    def __init__(self, renv, **f5kw):
        f5kw.setdefault("calib_file", "v2")
        f5kw.setdefault("eps_a", 0.0)
        super().__init__(renv, **f5kw)
        self.pol = None
        self._al_reset()

    def _al_reset(self):
        self._cur = (BV.OM.MU_LO, BV.OM.MU_HI)
        self.hist = []
        self._ck = {}
        self._lw_sum, self._n = 0.0, 0

    def begin(self, ctx):
        super().begin(ctx)
        self._al_reset()
        self.set_mu(*self._cur)

    def __call__(self, obs_hist, act_hist, u_nom, t):
        s = self.pol.on_tick(t, obs_hist[-1], act_hist[-1] if len(act_hist) else None)
        if s is not None:
            lo, hi = BV.clip_phys(*s)
            new = (BV.to_mu(lo), BV.to_mu(hi))
            self._cur = new
            self.set_mu(*new)
            if not self.hist or (abs(self.hist[-1][1] - lo) > 1e-12 or abs(self.hist[-1][2] - hi) > 1e-12):
                self.hist.append((int(t), float(lo), float(hi)))
        for name, tk in CK.items():
            if t == tk:
                self._ck[name] = self._cur
        self._lw_sum += float(np.log(self._cur[1] / self._cur[0]))
        self._n += 1
        return super().__call__(obs_hist, act_hist, u_nom, t)

    def stats(self):
        s = super().stats()
        s["x_alp_logw_mean"] = self._lw_sum / max(self._n, 1)
        s["x_alp_n_set"] = len(self.hist)
        s["x_alp_hist"] = [[t, float(np.exp(lo)), float(np.exp(hi))] for t, lo, hi in self.hist]
        s["_ck"] = dict(self._ck, end=self._cur if self._n else None)
        return s


# ====================================================================================== guard
@contextmanager
def hidden_guard(run):
    """poison every python-side copy of the hidden mu_roll for the duration of the decision path."""
    r = run.c.renv
    saved = (run.mu, r.mu_roll, r.env.p.mu_roll)
    run.mu = r.mu_roll = r.env.p.mu_roll = HiddenMu()
    try:
        yield
    finally:
        run.mu, r.mu_roll, r.env.p.mu_roll = saved


# ====================================================================================== run
class ALRun(RCL.RunCL):
    def run_al(self, pol, filt, cfg):
        c = self.c
        mu_true = self.mu
        l_true = float(np.log(mu_true))
        prior = PR.draw_prior(l_true, self.seed, pol.rho, pol.pseed)           # outside source (reads the truth by construction), BEFORE the guard
        pol.begin_episode(prior, l_true if pol.reads_truth else None)
        filt.pol = pol
        c.filt = filt
        with hidden_guard(self):
            I_rel = pol.pre_carry(self)
            if not self.dropped:
                plan = self.plan_for(I_rel)
                self._carry(plan, pol, filt)
        return l_true

    def _carry(self, plan, pol, filt):
        c = self.c
        filt.begin(dict(renv=c.renv, hid=None, seed=self.seed, T=plan.n_ticks))
        pol.carry_begin()
        cs = self.n
        t_arr = float("nan")
        n_done = 0
        for i in range(plan.n_ticks):
            e = c.renv.env
            u_nom = plan.u(i * DT, e.axis(), e.pos[:2], e.v[:2])
            u = project_box(filt(self.obs[cs:self.n + 1], self.act[cs:self.n], u_nom, i), c.renv.a_max, c.renv.w_max)
            interv = bool(np.any(np.abs(u - u_nom) > 1e-9))
            self.x["n_interv"] += int(interv)
            self.x["n_carry"] += 1
            self.tick(u)
            n_done += 1
            if self.dropped:
                break
            if plan.arrived(e.pos[:2], e.v[:2]):
                t_arr = self.n * DT
                break
        pol.finish(self.obs[self.n], self.act[self.n - 1], n_done)
        self._collect(filt)
        self.extra.update(plan.stats())
        self.extra.update(cl_arrived=bool(np.isfinite(t_arr)), cl_t_arrive_s=float(t_arr), ctrl="cl")


def make_ctx(cfg):
    PR.WRONG_LO, PR.WRONG_HI = cfg.get("wrong", (1.0, 2.0))
    ctx = EP.Ctx(cfg["setup"], cfg["seed"], cfg["split"], cfg["data"] or None, None, BR.default_tau(cfg["setup"]), "Rc3k", PO.EPS, None)
    if cfg["auth"] != 1.0:
        m = dict(ctx.meta)
        m["a_scale"] = ctx.meta["a_scale"] * cfg["auth"]
        m["w_scale"] = ctx.meta["w_scale"] * cfg["auth"]
        ctx.renv = E.env_from_meta(m)
        ctx.shadow = E.env_from_meta(m)
    return ctx


def make_value(cfg):
    return VAL.load_value(cfg.get("value") or "placeholder", pop=POPS.get(cfg["regime"], "full"), auth=cfg["auth"])


def make_voi(cfg):
    if not cfg.get("voi"):
        return None
    from world_model.v2.tray_rep.al.probe_voi import ProbeVOI
    return ProbeVOI.load(cfg["voi"])


def _init(cfg):
    G["cfg"] = cfg
    ctx = make_ctx(cfg)
    ctx.cert = True
    G["ctx"] = ctx
    G["filt"] = ALFilter(ctx.renv)
    G["pol"] = POL.ALPolicy(cfg["policy"], ctx.renv, regime=cfg["regime"], value=make_value(cfg), voi=make_voi(cfg), rho=cfg["rho"], pseed=cfg["seed"], log_ll=cfg["log_ll"])


def carry_record(ctx, filt, pol, cfg, ep):
    t0 = time.perf_counter()
    info = ctx.store.ep_info(ep)
    mu = None if cfg["mu_range"] is None else mu_for(info["seed"], *cfg["mu_range"])
    ALRun.gains = dict(omega=cfg["omega"], e_lo=cfg["e_lo"], e_hi=cfg["e_hi"])
    r = ALRun(ctx, info["seed"], mu)
    l_true = r.run_al(pol, filt, cfg)
    rec = r.record(f"AL:{cfg['policy']}")
    rec["ep"] = int(ep)
    rec["onset_stored"] = int(info["onset"])
    rec["wall_s"] = time.perf_counter() - t0
    rec["filt"] = "AL"
    rec["regime"], rec["auth"] = cfg["regime"], cfg["auth"]
    rec.update(cl_omega=cfg["omega"], cl_e_lo=cfg["e_lo"], cl_e_hi=cfg["e_hi"], pseed=cfg["seed"], rho=pol.rho)
    st = filt.stats()
    ck = st.pop("_ck")
    for k, v in st.items():
        rec.setdefault(k, v)
    # ---- LOGGING ONLY (truth allowed after the carry): coverage of the applied sets
    mu_t = float(rec["mu"])
    for name in list(CK) + ["end"]:
        b = ck.get(name)
        rec[f"x_p_lo_{name}"] = None if b is None else b[0]
        rec[f"x_p_hi_{name}"] = None if b is None else b[1]
        rec[f"x_p_cov_{name}"] = None if b is None else bool(b[0] * (1 - 1e-9) <= mu_t <= b[1] * (1 + 1e-9))
        rec[f"x_p_lerr_{name}"] = None if b is None else float(0.5 * np.log(b[0] * b[1]) - l_true)
    h = filt.hist
    rec["alp_set_ever_miss"] = bool(any(not (lo - 1e-9 <= l_true <= hi + 1e-9) for _, lo, hi in h))
    if h:
        ts = [t for t, _, _ in h] + [filt._n]
        miss = sum(ts[i + 1] - ts[i] for i, (_, lo, hi) in enumerate(h) if not (lo - 1e-9 <= l_true <= hi + 1e-9))
        rec["alp_set_miss_frac"] = float(miss / max(filt._n, 1))
    rec.update(pol.end_stats(l_true))
    rec["probe_skipped"] = pol.log["probe_skip"] if pol.log["probe_considered"] else rec.get("probe_skipped", "")
    return rec


def _work(ep):
    return carry_record(G["ctx"], G["filt"], G["pol"], G["cfg"], ep)


def policy_tag(spec):
    return spec.replace(":", "__").replace(",", "_").replace("=", "-").replace("/", "+")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="AL_<kind>[:k=v,...] (al/policies.py)")
    ap.add_argument("--regime", default="normal", choices=list(REGIMES))
    ap.add_argument("--auth", type=float, default=None)
    ap.add_argument("--mu-range", default=None, help="lo,hi (overrides the regime)")
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--setup", default="piper_trk", choices=["piper", "piper_trk"])
    ap.add_argument("--seed", type=int, default=0, help="tracker-noise seed AND prior seed")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--n-eps", type=int, default=None)
    ap.add_argument("--ep-start", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--value", default="placeholder", help="placeholder[:drop=proxy] | cells:<vimap analysis/cells.csv>[,pop=low][,auth=0.25]")
    ap.add_argument("--voi", default="", help="probe VOI table (build_voi.py) for AL_transductive")
    ap.add_argument("--log-ll", action="store_true", help="log raw online LL banks (sigma_a 0.04/0.06/0.08) every 25 ticks (calibrate_lam.py)")
    ap.add_argument("--omega", type=float, default=PC.OMEGA)
    ap.add_argument("--e-lo", type=float, default=PC.E_LO)
    ap.add_argument("--e-hi", type=float, default=PC.E_HI)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    assert a.workers <= 8
    R = REGIMES[a.regime]
    auth = R["auth"] if a.auth is None else a.auth
    mur = R["mu_range"] if a.mu_range is None else tuple(float(x) for x in a.mu_range.split(","))
    rho = R["rho"] if a.rho is None else a.rho
    cfg = dict(policy=a.policy, regime=a.regime, auth=auth, mu_range=mur, rho=rho, setup=a.setup, seed=a.seed, split=a.split, data=a.data, value=a.value, voi=a.voi,
               log_ll=bool(a.log_ll), omega=a.omega, e_lo=a.e_lo, e_hi=a.e_hi, wrong=(1.0, 2.0))
    d = Path(a.out) / a.regime / policy_tag(a.policy)
    d.mkdir(parents=True, exist_ok=True)
    i, n = (int(x) for x in a.shard.split("/"))
    pj = d / (f"AL__s{a.seed}" + (f"__p{i}of{n}" if n > 1 else "") + ".jsonl")
    st = Store(Path(a.data), a.split)
    n_all = st.n_eps if a.n_eps is None else min(st.n_eps, a.ep_start + a.n_eps)
    eps = list(range(a.ep_start, n_all))[i::n]
    done = {r["ep"] for r in BR.read_jsonl(pj)}
    todo = [e for e in eps if e not in done]
    BR.log(f"[run_al] {a.policy} {a.regime} auth={auth} mu={mur} rho={rho} {a.split} s{a.seed} n={len(eps)} done={len(done)} todo={len(todo)} workers={a.workers} -> {pj}")
    t0 = time.perf_counter()
    if todo:
        with open(pj, "a") as f:
            if a.workers <= 1:
                _init(cfg)
                for ep in todo:
                    f.write(json.dumps(_work(ep), default=float) + "\n"); f.flush()
            else:
                with Pool(a.workers, initializer=_init, initargs=(cfg,)) as P:
                    for r in P.imap_unordered(_work, todo, chunksize=1):
                        f.write(json.dumps(r, default=float) + "\n"); f.flush()
    recs = [r for r in BR.read_jsonl(pj) if r["ep"] in set(eps)]
    s = BR.summarize(recs)
    if s.get("n"):
        arr = [r["cl_t_arrive_s"] for r in recs if r.get("cl_arrived")]
        BR.log(f"[run_al] {a.policy} {a.regime}: drop {s['drop']['k']}/{s['drop']['n']} t_arrive {np.mean(arr) if arr else float('nan'):.2f}s interv {s['interv_rate']:.3f} "
               f"probes run {sum(bool(r.get('alp_probe_run')) for r in recs)} [{time.perf_counter() - t0:.0f}s]")


if __name__ == "__main__":
    main()
