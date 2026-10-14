"""al/run_vi.py -- value-of-information map runner (AL-0). SIMULATION / SAVED DATA ONLY. CPU. No hardware, no network.

One call = one (arm authority, population, mu-set mode, speed I) cell on VAL carries (closed-loop planner boed.plan_cl, F5 = al.filt.F5SetAL).
Reuses (imports, no edits): boed.run_cl.RunCL (closed-loop carry), wrongpoint.run_cl_wp helpers' conventions (auth: a_scale/w_scale x f on a NEW TrayRepEnv built
before the filter; low-mu population: true mu redrawn log-uniform per carry seed by wrongpoint.run_cell.mu_for), boed.episode.Ctx.
ADDED vs run_cl: per-carry deviation dev_norm_mean = mean over carry ticks of |D (u - u_nom)|_2, D = diag(1/a_max x3, 1/w_max x3) (filter deviation from the nominal,
normalised by the arm limits); the per-carry mu set (filt.F5SetAL) and its coverage in the record.

  python -m world_model.v2.tray_rep.al.run_vi --auth 0.25 --pop low --set oracle --I 0.99 --n-eps 8 --workers 2 --out <dir>
Output: <out>/auth<a>_pop<p>/<set>/CL__I-<I>__s<seed>__p<i>of<n>.jsonl   (rerun = resume by episode)
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
from world_model.v2.tray_rep.al.filt import F5SetAL, MODES
from world_model.v2.tray_rep.boed import episode as EP
from world_model.v2.tray_rep.boed import plan_cl as PC
from world_model.v2.tray_rep.boed import policy as PO
from world_model.v2.tray_rep.boed import run as BR
from world_model.v2.tray_rep.boed import run_cl as RCL
from world_model.v2.tray_rep.bridge.filters import project_box
from world_model.v2.tray_rep.store import Store
from world_model.v2.tray_rep.wrongpoint.run_cell import mu_for

DT = EP.DT
DATA = "/common/users/ss5772/bimanual_data/tray_rep/data_piper_trk_full_v2"
POPS = {"full": (5e-4, 6e-3), "low": (5e-4, 1.5e-3)}
G = {}


class VIRun(RCL.RunCL):
    """RunCL.carry_phase verbatim + dev_norm accumulation."""

    def carry_phase(self, plan, quiet=None):
        assert quiet is None
        c, f = self.c, self.c.filt
        f.begin(dict(renv=c.renv, hid=self.mu, seed=self.seed, T=plan.n_ticks))
        D = np.array([1.0 / c.renv.a_max] * 3 + [1.0 / c.renv.w_max] * 3)
        dev = 0.0
        cs = self.n
        t_arr = float("nan")
        for i in range(plan.n_ticks):
            e = c.renv.env
            u_nom = plan.u(i * DT, e.axis(), e.pos[:2], e.v[:2])
            u = project_box(f(self.obs[cs:self.n + 1], self.act[cs:self.n], u_nom, i), c.renv.a_max, c.renv.w_max)
            interv = bool(np.any(np.abs(u - u_nom) > 1e-9))
            self.x["n_interv"] += int(interv)
            self.x["n_carry"] += 1
            dev += float(np.linalg.norm(D * (u - u_nom)))
            self.tick(u)
            if self.dropped:
                break
            if plan.arrived(e.pos[:2], e.v[:2]):
                t_arr = self.n * DT
                break
        self._collect(f)
        self.extra.update(plan.stats())
        self.extra.update(cl_arrived=bool(np.isfinite(t_arr)), cl_t_arrive_s=float(t_arr), ctrl="cl",
                          dev_norm_sum=dev, dev_norm_mean=dev / max(self.x["n_carry"], 1))
        return ("drop" if self.dropped else "end"), t_arr


def _init(cfg):
    G["cfg"] = cfg
    ctx = EP.Ctx(cfg["setup"], cfg["seed"], cfg["split"], cfg["data"], None, BR.default_tau(cfg["setup"]), "Rc3k", PO.EPS, None)
    if cfg["auth"] != 1.0:
        m = dict(ctx.meta)
        m["a_scale"] = ctx.meta["a_scale"] * cfg["auth"]
        m["w_scale"] = ctx.meta["w_scale"] * cfg["auth"]
        ctx.renv = E.env_from_meta(m)
        ctx.shadow = E.env_from_meta(m)
    plo, phi = POPS[cfg["pop"]]
    ctx.filt = F5SetAL(ctx.renv, mode=cfg["set"], pop_lo=plo, pop_hi=phi, calib_file="v2", eps_a=0.0)
    G["ctx"] = ctx


def _work(ep):
    cfg, ctx = G["cfg"], G["ctx"]
    t0 = time.perf_counter()
    info = ctx.store.ep_info(ep)
    mu = None if cfg["pop"] == "full" else mu_for(info["seed"], *POPS[cfg["pop"]])
    VIRun.gains = dict(omega=PC.OMEGA, e_lo=PC.E_LO, e_hi=PC.E_HI)
    r = VIRun(ctx, info["seed"], mu)
    rec = r.run(f"FIX:I={cfg['I']:g}")
    rec.update(cl_omega=PC.OMEGA, cl_e_lo=PC.E_LO, cl_e_hi=PC.E_HI)
    rec["ep"] = int(ep)
    rec["wall_s"] = time.perf_counter() - t0
    rec["cell"], rec["set"], rec["auth"], rec["pop"], rec["I"] = cfg["cell"], cfg["set"], cfg["auth"], cfg["pop"], cfg["I"]
    rec["a_max"], rec["w_max"] = float(ctx.renv.a_max), float(ctx.renv.w_max)
    for k, v in (ctx.filt.stats() or {}).items():
        rec.setdefault(k, v)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--auth", type=float, required=True)
    ap.add_argument("--pop", required=True, choices=list(POPS))
    ap.add_argument("--set", required=True, choices=list(MODES))
    ap.add_argument("--I", type=float, required=True)
    ap.add_argument("--setup", default="piper_trk", choices=["piper", "piper_trk"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--n-eps", type=int, default=None)
    ap.add_argument("--ep0", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert a.workers <= 8
    cell = f"auth{a.auth:g}_pop{a.pop}"
    d = Path(a.out) / cell / a.set
    d.mkdir(parents=True, exist_ok=True)
    i, n = (int(x) for x in a.shard.split("/"))
    pj = d / (f"CL__I-{a.I:g}__s{a.seed}" + (f"__p{i}of{n}" if n > 1 else "") + ".jsonl")
    st = Store(Path(a.data), a.split)
    eps = list(range(a.ep0, st.n_eps if a.n_eps is None else min(st.n_eps, a.ep0 + a.n_eps)))[i::n]
    done = {r["ep"] for r in BR.read_jsonl(pj)}
    todo = [e for e in eps if e not in done]
    cfg = dict(auth=a.auth, pop=a.pop, set=a.set, I=a.I, setup=a.setup, seed=a.seed, split=a.split, data=a.data, cell=cell)
    BR.log(f"[run_vi] {cell}/{a.set} I={a.I:g} {a.split} n={len(eps)} done={len(done)} todo={len(todo)} -> {pj}")
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
    arr = [r["cl_t_arrive_s"] for r in recs if np.isfinite(r["cl_t_arrive_s"])]
    BR.log(f"[run_vi] {cell}/{a.set} I={a.I:g}: drop {s['drop']['k']}/{s['drop']['n']} goal {s['goal']['rate']:.2f} t_complete {s['t_complete_s_ok']:.2f}s "
           f"t_arrive {np.mean(arr) if arr else float('nan'):.2f}s ({len(arr)}) interv {s['interv_rate']:.3f} cov {np.mean([r['x_al_cov'] for r in recs]):.2f} [{time.perf_counter() - t0:.0f}s]")


if __name__ == "__main__":
    main()
