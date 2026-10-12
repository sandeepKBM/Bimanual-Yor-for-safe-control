"""wrongpoint/run_cl_wp.py -- NEW wrapper around boed/run_cl.py (NOT edited) that adds the wrongpoint knobs the closed-loop runner lacks. SIMULATION ONLY. CPU.

Same carry, Ctx, RunCL (closed-loop planner), record and filter-stats merge as run_cl; ADDED (all default OFF = run_cl behaviour):
  --mu-range lo,hi   the carry's TRUE mu is redrawn log-uniform in [lo,hi] (deterministic per carry seed; wrongpoint.run_cl.mu_for); passed as RunCL(mu_override)
  --auth f           arm authority: a_scale, w_scale of the dataset meta multiplied by f (new TrayRepEnv for ctx.renv/ctx.shadow, built BEFORE the filter,
                     so F5's box/slew limits and CLPlan's a_max/w_max follow)
  --wrong lo,hi      |error| range (ln mu) of the confidently wrong prior: sets sets.vlmprior.prior.WRONG_LO/HI in each worker (module globals read at call time)
  --arm-key/--cell   labels written to each record; prior_m/prior_err/prior_conf recomputed per record for the *@rho arms
Output: <out>/<cell>/<armkey>/CL__I-<I>__s<seed>__p<i>of<n>.jsonl  (rerun = resume by episode)
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep.boed import episode as EP
from world_model.v2.tray_rep.boed import plan_cl as PC
from world_model.v2.tray_rep.boed import policy as PO
from world_model.v2.tray_rep.boed import run as BR
from world_model.v2.tray_rep.boed import run_cl as RCL
from world_model.v2.tray_rep.bridge import filters as F
from world_model.v2.tray_rep.sets.vlmprior import prior as PR
from world_model.v2.tray_rep.store import Store
from world_model.v2.tray_rep.wrongpoint.run_cell import mu_for

G = {}


def _init(cfg):
    G["cfg"] = cfg
    PR.WRONG_LO, PR.WRONG_HI = cfg["wrong"]
    ctx = EP.Ctx(cfg["setup"], cfg["seed"], cfg["split"], cfg["data"] or None, None, BR.default_tau(cfg["setup"]), "Rc3k", PO.EPS, None)
    if cfg["auth"] != 1.0:
        m = dict(ctx.meta)
        m["a_scale"] = ctx.meta["a_scale"] * cfg["auth"]
        m["w_scale"] = ctx.meta["w_scale"] * cfg["auth"]
        ctx.renv = E.env_from_meta(m)
        ctx.shadow = E.env_from_meta(m)
    if cfg["filt"] != "F3":
        ctx.filt = F.make_filter(cfg["filt"], ctx.renv, seed=cfg["seed"], runs_root=str(K.ROOT / "runs"), model_setup=cfg["setup"])
    G["ctx"] = ctx


def _work(ep):
    cfg, ctx = G["cfg"], G["ctx"]
    t0 = time.perf_counter()
    info = ctx.store.ep_info(ep)
    mu = None if cfg["mu_range"] is None else mu_for(info["seed"], *cfg["mu_range"])
    RCL.RunCL.gains = dict(omega=cfg["omega"], e_lo=cfg["e_lo"], e_hi=cfg["e_hi"])
    r = RCL.RunCL(ctx, info["seed"], mu)
    rec = r.run(f"FIX:I={cfg['I']:g}")
    rec.update(cl_omega=cfg["omega"], cl_e_lo=cfg["e_lo"], cl_e_hi=cfg["e_hi"])
    rec["ep"] = int(ep)
    rec["onset_stored"] = int(info["onset"])
    rec["wall_s"] = time.perf_counter() - t0
    rec["filt"] = cfg["filt"]
    rec["cell"], rec["arm"] = cfg["cell"], cfg["arm_key"]
    for k, v in (ctx.filt.stats() or {}).items():
        rec.setdefault(k, v)
    if cfg["rho"] is not None:
        m, sg, cf = PR.draw_prior(float(np.log(rec["mu"])), info["seed"], cfg["rho"], cfg["pseed"])
        rec.update(prior_m=m, prior_sigma=sg, prior_conf=bool(cf), prior_err=float(m - np.log(rec["mu"])), rho=cfg["rho"], pseed=cfg["pseed"])
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--arm-key", required=True, help="label, e.g. point@0.4__ps1 | oracle | F0")
    ap.add_argument("--filt", required=True)
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--pseed", type=int, default=0)
    ap.add_argument("--I", type=float, default=0.99)
    ap.add_argument("--setup", default="piper_trk", choices=["piper", "piper_trk"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n-eps", type=int, default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--mu-range", default=None)
    ap.add_argument("--auth", type=float, default=1.0)
    ap.add_argument("--wrong", default="1,2")
    ap.add_argument("--omega", type=float, default=PC.OMEGA)
    ap.add_argument("--e-lo", type=float, default=PC.E_LO)
    ap.add_argument("--e-hi", type=float, default=PC.E_HI)
    ap.add_argument("--data", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert a.workers <= 12
    d = Path(a.out) / a.cell / a.arm_key
    d.mkdir(parents=True, exist_ok=True)
    i, n = (int(x) for x in a.shard.split("/"))
    pj = d / (f"CL__I-{a.I:g}__s{a.seed}" + (f"__p{i}of{n}" if n > 1 else "") + ".jsonl")
    st = Store(Path(a.data) if a.data else K.ROOT / f"data_{a.setup}_full", a.split)
    eps = list(range(st.n_eps if a.n_eps is None else min(st.n_eps, a.n_eps)))[i::n]
    done = {r["ep"] for r in BR.read_jsonl(pj)}
    todo = [e for e in eps if e not in done]
    cfg = dict(filt=a.filt, I=a.I, setup=a.setup, seed=a.seed, split=a.split, omega=a.omega, e_lo=a.e_lo, e_hi=a.e_hi, data=a.data, cell=a.cell,
               arm_key=a.arm_key, rho=a.rho, pseed=a.pseed, mu_range=None if a.mu_range is None else tuple(float(x) for x in a.mu_range.split(",")),
               auth=a.auth, wrong=tuple(float(x) for x in a.wrong.split(",")))
    BR.log(f"[run_cl_wp] {a.cell}/{a.arm_key} filt={a.filt} I={a.I:g} {a.split} n={len(eps)} done={len(done)} todo={len(todo)} auth={a.auth} mu={a.mu_range} wrong={a.wrong} -> {pj}")
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
    BR.log(f"[run_cl_wp] {a.cell}/{a.arm_key}: drop {s['drop']['k']}/{s['drop']['n']} goal {s['goal']['rate']:.2f} interv {s['interv_rate']:.3f} "
           f"progress {s['progress']:.4f} [{time.perf_counter() - t0:.0f}s]")


if __name__ == "__main__":
    main()
