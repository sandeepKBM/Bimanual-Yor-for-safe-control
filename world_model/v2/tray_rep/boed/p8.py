"""boed/p8.py -- P8: NO probe. An ONLINE belief over mu_roll (sets/vlmprior/online_mu.OnlineMu, the VAL-calibrated lam / sigma_a of the setup),
fed EVERY carry tick from the obs stream, re-picks the carry intensity at each transport boundary. SIMULATION / SAVED DATA ONLY. CPU. No hardware,
no network. (Imports only tray_rep modules; nothing under YOR/robot or YOR/extra.)

New file. boed/{episode,run,p7,carry,policy,...}.py are NOT changed (frozen pre-registrations hash them). The runner builds the BOED Ctx as boed/run.py does
(F3 Rc3k at the bridge's tau; the F3 filter is ON for the whole carry, exactly as P1..P7) and runs its own carry loop (a copy of Run.carry_phase that has
transport-boundary hooks instead of the P4 `quiet` stop).

Policy
  start     the intensity of P6: Table.choose(PO.w_prior(), eps) on the frozen table (read only).
  every tick  belief.feed(obs[t], act[t-1])   (obs/act stream only; the belief never sees the hidden mu).
  boundary   the start of transport 2 and of transport 3 of the 3-transport plan (plan time = the start of the transport, read from the live plan):
               mu_q = belief 2.5% quantile;  frac = max(0.05, 1 - path_done / path_len_nom);
               idx  = p7.pick_worst_case(table, mu_q, eps, frac)   = the fastest grid intensity whose table drop AT mu_q, scaled by the remaining
                      path (constant hazard 1 - (1-d)^frac, as P4 via Table.choose), is <= eps; the slowest one if none. Any grid intensity is allowed
                      (it may slow down as well as speed up).
             if the intensity changed: plan.replan_from(tp, tray xy, I*I_feas) (as Run._p4 does); plan time restarts at 0. If it did not change, NOTHING
             is touched (no re-plan), so a belief that never changes the intensity reproduces P6 exactly.
  F3 is not restarted at a re-plan (the filter and its latent run through the whole carry).
Hidden mu: Run.mu is replaced by a poison object for the duration of the carry (any read raises); the true value is read back only in `_log_truth` for
the coverage fields (LOGGING only).

Record: the standard Run.record fields with policy "P8" (I_hist = intensity per transport, 3 entries when the carry reaches transport 3), plus
  p8_mu_q_hist [mu_q at boundary 2, boundary 3] (null = boundary not reached), p8_frac_hist, p8_replan [bool, bool], p8_belief ('online'|'null'), p8_n_samples,
  bel_mean_log, bel_sd_log, bel_cdf_true, bel_cover90, bel_cover50, bel_lo90, bel_hi90, bel_err_log (belief at the END of the carry; boed/report.row reads them).

  python -m world_model.v2.tray_rep.boed.p8 --setup piper_trk --seed 0 --split test --shard 0/4 --workers 2 --out <bimanual_data/tray_rep/boed_p8> \
         --table-dir <bimanual_data/tray_rep/boed/main> --n-nodes 8 --eps 0.02
Output: <out>/main/<split>/<setup>/P8__s<seed>[__p<i>of<n>].jsonl  (boed.report.load(<out>, "main", split) picks it up). Rerun = resume (by episode).
--belief null: a never-informative belief (mu_q = 5e-4 always); only for the plumbing check against P6 (p8_checks.py).
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep.boed import episode as EP
from world_model.v2.tray_rep.boed import policy as PO
from world_model.v2.tray_rep.boed import run as BR
from world_model.v2.tray_rep.boed.p7 import pick_worst_case, Q_DECISION
from world_model.v2.tray_rep.sets.vlmprior import online_mu as OM
from world_model.v2.tray_rep.sets.vlmprior.filter import load_online_defaults

DT = EP.DT
OUT = K.ROOT / "boed_p8"
FROZEN_DIRS = ("boed", "boed_p7", "boed_f5", "vlmprior", "sets", "bridge")
G = {}


# ====================================================================================== hidden-mu poison
class HiddenMu:
    """stands in for Run.mu while the carry runs: any use raises. The decision path must never read the hidden mu_roll."""

    def _boom(self, *a, **k):
        raise RuntimeError("P8: the hidden mu_roll was read in the decision path")

    __float__ = __int__ = __index__ = __array__ = __bool__ = __format__ = _boom
    __lt__ = __le__ = __gt__ = __ge__ = __eq__ = __ne__ = _boom
    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __truediv__ = __rtruediv__ = __pow__ = __neg__ = _boom
    __hash__ = object.__hash__

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        self._boom()


# ====================================================================================== beliefs
class OnlineBelief:
    """OnlineMu with the VAL-calibrated lam / sigma_a of the setup (as sets/vlmprior/filter.F5PFilter builds it). Reset per carry."""
    kind = "online"

    def __init__(self, renv):
        nf = renv.tracker is None
        lam, sa = load_online_defaults("piper" if nf else "piper_trk")
        okw = dict(noise_free=nf, lam=lam, sigma_a=sa)
        if not nf:
            okw.update(sigma_meas=renv.tracker.noise, lat=renv.tracker.lat)
        self.lam, self.sa = lam, sa
        self.om = OM.OnlineMu(**okw)

    def reset(self):
        self.om.reset()

    def feed(self, obs_row, act_row=None):
        self.om.feed(obs_row, act_row)

    @property
    def t(self):
        return self.om.t

    @property
    def n_samples(self):
        return int(self.om.n_samples)

    def mu_quantile(self, q):
        return float(np.exp(self.om.quantile(q)))

    def end_stats(self, l_true):
        """LOGGING ONLY (l_true = ln of the hidden mu)."""
        om = self.om
        w = om.weights()
        lo90, hi90 = om.interval_log(0.90)
        lo50, hi50 = om.interval_log(0.50)
        m, sd = om.mean_sd_log()
        return dict(bel_mean_log=float(m), bel_sd_log=float(sd), bel_cdf_true=float(OM.cell_cdf(w, l_true)),
                    bel_cover90=bool(lo90 <= l_true <= hi90), bel_cover50=bool(lo50 <= l_true <= hi50),
                    bel_lo90=float(np.exp(lo90)), bel_hi90=float(np.exp(hi90)), bel_err_log=float(m - l_true))


class NullBelief(OnlineBelief):
    """never informative: mu_q = 5e-4 (the lowest mu of the population) always. Plumbing check only."""
    kind = "null"

    def __init__(self, renv):
        self.om = OM.OnlineMu(noise_free=True)          # only to give end_stats a (prior) belief to report
        self._t = -1

    def reset(self):
        self.om.reset()
        self._t = -1

    def feed(self, obs_row, act_row=None):
        self._t += 1

    @property
    def t(self):
        return self._t

    @property
    def n_samples(self):
        return 0

    def mu_quantile(self, q):
        return OM.MU_LO


# ====================================================================================== the policy
def run_p8(run, bel, n_boundaries=2):
    """P8 on a `boed.episode.Run` (duck-typed on run / run.c). Leaves the results in run.* and run.extra; call run.record("P8") afterwards."""
    c = run.c
    table, eps = c.table, c.eps
    f = c.filt
    mu_true = run.mu
    run.mu = HiddenMu()                                   # structural guarantee: nothing below can read the hidden mu
    try:
        bel.reset()
        I_cur = float(table.igrid[table.choose(PO.w_prior(), eps)])           # P6's intensity
        plan = run.plan_for(I_cur)                         # appends I_hist[0]
        mu_q_hist, frac_hist, replan_hist = [], [], []
        f.begin(dict(renv=c.renv, hid=None, seed=run.seed, T=plan.n_ticks))
        cs = run.n
        b, j0, i = 1, 0, -1                                # b: next boundary = absolute index of the transport that starts; j0: global tick of plan time 0
        while True:
            i += 1
            j = i - j0
            pos = c.renv.env.pos[:2]
            if j >= plan.n_ticks and not plan.touchup(j * DT, pos):
                break
            bel.feed(run.obs[run.n], run.act[run.n - 1] if run.n > 0 else None)
            tp = j * DT
            k = b - plan.i_first
            if b <= n_boundaries and 0 <= k < len(plan.segs) and tp + 1e-12 >= plan.segs[k][0]:
                frac = max(0.05, 1.0 - run.prog.done / max(run.prog.len_nom, 1e-9))
                mu_q = bel.mu_quantile(Q_DECISION)
                I_new = float(table.igrid[pick_worst_case(table, mu_q, eps, frac)])
                mu_q_hist.append(float(mu_q)); frac_hist.append(float(frac))
                changed = abs(I_new - I_cur) > 1e-12
                replan_hist.append(bool(changed))
                if changed:                                # an unchanged intensity is NOT re-planned (no-op)
                    plan.replan_from(tp, pos, I_new * run.I_feas)
                    j0, tp = i, 0.0
                    I_cur = I_new
                run.I_hist.append(I_cur)
                b += 1
            u_nom = plan.u(tp, c.renv.env.axis(), pos)
            u = EP.project_box(f(run.obs[cs:run.n + 1], run.act[cs:run.n], u_nom, i), c.renv.a_max, c.renv.w_max)
            interv = bool(np.any(np.abs(u - u_nom) > 1e-9))
            run.x["n_interv"] += int(interv)
            run.x["n_carry"] += 1
            run.tick(u)
            if run.dropped:
                break
            if interv and run.info["p4_trigger_tick"] < 0:
                run.info["p4_trigger_tick"] = run.n
            if plan.waypoint_mode and plan.arrived(tp, c.renv.env.pos[:2], c.renv.env.v[:2]):
                break
        run._collect(f)
        if bel.t < run.n:                                  # the belief at the END of the carry has seen the last row too
            bel.feed(run.obs[run.n], run.act[run.n - 1])
    finally:
        run.mu = mu_true
    n_b = len(mu_q_hist)
    run.extra.update(p8_mu_q_hist=mu_q_hist + [None] * (n_boundaries - n_b), p8_frac_hist=frac_hist + [None] * (n_boundaries - n_b),
                     p8_replan=replan_hist + [None] * (n_boundaries - n_b), p8_belief=bel.kind, p8_n_samples=bel.n_samples,
                     p8_lam=getattr(bel, "lam", None), p8_sigma_a=getattr(bel, "sa", None))
    run.extra.update(_log_truth(bel, mu_true))


def _log_truth(bel, mu_true):
    """LOGGING ONLY: coverage of the end-of-carry belief of the hidden mu."""
    return bel.end_stats(float(np.log(mu_true)))


# ====================================================================================== runner
def _init(cfg):
    G["cfg"] = cfg
    table = PO.Table.load(Path(cfg["table_dir"]) / f"table_{cfg['setup']}.json")
    ctx = EP.Ctx(cfg["setup"], cfg["seed"], cfg["split"], None, None, BR.default_tau(cfg["setup"]), "Rc3k", cfg["eps"], table, n_nodes=cfg["n_nodes"])
    G["ctx"] = ctx
    G["bel"] = (NullBelief if cfg["belief"] == "null" else OnlineBelief)(ctx.renv)


def carry_record(ctx, bel, ep):
    info = ctx.store.ep_info(ep)
    t0 = time.perf_counter()
    r = EP.Run(ctx, info["seed"], None)
    run_p8(r, bel)
    rec = r.record("P8")
    rec["ep"] = int(ep)
    rec["onset_stored"] = int(info["onset"])
    rec["wall_s"] = time.perf_counter() - t0
    return rec


def _work(ep):
    return carry_record(G["ctx"], G["bel"], ep)


def paths(out, split, setup, seed, shard, tag="main"):
    d, pj, _ = BR.run_paths(out, tag, split, setup, "P8", seed, shard)
    return d, pj


def check_out(out):
    p = Path(out).resolve()
    root = K.ROOT.resolve()
    for n in FROZEN_DIRS:
        q = root / n
        if p == q or q in p.parents:
            raise SystemExit(f"refusing to write into the frozen folder {q}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", required=True, choices=["piper", "piper_trk"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n-eps", type=int, default=None)
    ap.add_argument("--ep-start", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--tag", default="main")
    ap.add_argument("--table-dir", required=True, help="folder with the frozen table_<setup>.json (read only), e.g. .../boed/main")
    ap.add_argument("--n-nodes", type=int, default=PO.N_NODES)
    ap.add_argument("--eps", type=float, default=PO.EPS)
    ap.add_argument("--belief", default="online", choices=["online", "null"])
    a = ap.parse_args()
    check_out(a.out)
    d, pj = paths(a.out, a.split, a.setup, a.seed, a.shard, a.tag)
    d.mkdir(parents=True, exist_ok=True)
    from world_model.v2.tray_rep.store import Store
    st = Store(K.ROOT / f"data_{a.setup}_full", a.split)
    si, sn = (int(x) for x in a.shard.split("/"))
    n_all = st.n_eps if a.n_eps is None else min(st.n_eps, a.ep_start + a.n_eps)
    eps_l = list(range(a.ep_start, n_all))[si::sn]
    done = {r["ep"] for r in BR.read_jsonl(pj)}
    todo = [e for e in eps_l if e not in done]
    BR.log(f"[p8] {a.belief} {a.setup} s{a.seed} {a.split} n={len(eps_l)} done={len(done)} todo={len(todo)} workers={a.workers} -> {pj}")
    cfg = dict(setup=a.setup, seed=a.seed, split=a.split, table_dir=a.table_dir, n_nodes=a.n_nodes, eps=a.eps, belief=a.belief)
    t0 = time.perf_counter()
    if todo:
        with open(pj, "a") as f:
            if a.workers <= 1:
                _init(cfg)
                for i, ep in enumerate(todo):
                    f.write(json.dumps(_work(ep), default=float) + "\n"); f.flush()
                    if (i + 1) % 10 == 0:
                        BR.log(f"  {i + 1}/{len(todo)}")
            else:
                with Pool(a.workers, initializer=_init, initargs=(cfg,)) as P:
                    for i, r in enumerate(P.imap_unordered(_work, todo, chunksize=1)):
                        f.write(json.dumps(r, default=float) + "\n"); f.flush()
                        if (i + 1) % 20 == 0:
                            BR.log(f"  {i + 1}/{len(todo)}")
    recs = [r for r in BR.read_jsonl(pj) if r["ep"] in set(eps_l)]
    s = BR.summarize(recs)
    if s.get("n"):
        BR.log(f"[p8] {a.setup} s{a.seed} {a.split}: drop {s['drop']['k']}/{s['drop']['n']} goal {s['goal']['rate']:.2f} "
               f"t_complete {s['t_complete_s_ok']:.2f}s interv {s['interv_rate']:.3f} [{time.perf_counter() - t0:.0f}s]")


if __name__ == "__main__":
    main()
