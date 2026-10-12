"""boed/run_cl.py -- fixed-intensity BOED carries with the CLOSED-LOOP nominal (plan_cl.CLPlan) and ANY bridge filter. SIMULATION / SAVED DATA ONLY. CPU.

Modelled on boed/run_fxi.py (which is NOT changed; neither are carry.py / episode.py / run.py). Same Ctx (F3 Rc3k at the bridge's tau, replaced by the
requested filter), same carry (mu = the carry's own), same record (boed.episode.Run.record) plus
  ctrl ('cl'|'ol'), cl_arrived, cl_t_arrive_s (first tick the arrival rule holds: ref played out, <=1.5 cm, <=3 cm/s), cl_s_final (reference clock),
  cl_ref_done_s, cl_pause_ticks, cl_slow_ticks, cl_max_err, cl_T_ref, cl_omega/e_lo/e_hi.
--ctrl ol re-runs the EXISTING open-loop plan through the same code path (identical to run_fxi's records) for paired comparisons.
The closed-loop carry ends on arrival, ball drop, or the timeout 2.5 * T_ref + 8 s. t_complete_s = goal_tick * 20 ms when the tray ever came within 2 cm
of the goal (PathProgress, as in boed records; may be before it has stopped), else pace-extrapolated; NaN if dropped. cl_t_arrive_s is the stricter time.

  python -m world_model.v2.tray_rep.boed.run_cl --filt F5 --I 0.99 --setup piper_trk --seed 0 --split val --n-eps 40 --workers 4
Output: <out>/<spec_tag(filt)>/<split>/<setup>/CL__I-<I>__s<seed>[__p<i>of<n>].jsonl  (OL__ for --ctrl ol). Rerun = resume (by episode).
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
from world_model.v2.tray_rep.boed import plan_cl as PC
from world_model.v2.tray_rep.boed import policy as PO
from world_model.v2.tray_rep.boed import run as BR
from world_model.v2.tray_rep.bridge import filters as F
from world_model.v2.tray_rep.bridge.filters import project_box
from world_model.v2.tray_rep.store import Store

OUT = K.ROOT / "plan_cl"
DT = EP.DT
G = {}


class RunCL(EP.Run):
    """Run with a closed-loop carry phase. plan_for() returns a CLPlan; carry_phase is the base loop with the CL command and arrival rule."""
    gains = {}

    def plan_for(self, I_rel):
        c = self.c
        self.I_hist.append(float(I_rel))
        return PC.CLPlan(self.cr, c.renv.a_scale, c.renv.w_scale, float(I_rel) * self.I_feas, home_xy=self.start_xy, **self.gains)

    def carry_phase(self, plan, quiet=None):
        assert quiet is None
        c, f = self.c, self.c.filt
        f.begin(dict(renv=c.renv, hid=self.mu, seed=self.seed, T=plan.n_ticks))
        cs = self.n
        t_arr = float("nan")
        for i in range(plan.n_ticks):
            e = c.renv.env
            u_nom = plan.u(i * DT, e.axis(), e.pos[:2], e.v[:2])
            u = project_box(f(self.obs[cs:self.n + 1], self.act[cs:self.n], u_nom, i), c.renv.a_max, c.renv.w_max)
            interv = bool(np.any(np.abs(u - u_nom) > 1e-9))
            self.x["n_interv"] += int(interv)
            self.x["n_carry"] += 1
            self.tick(u)
            if self.dropped:
                break
            if plan.arrived(e.pos[:2], e.v[:2]):
                t_arr = self.n * DT
                break
        self._collect(f)
        self.extra.update(plan.stats())
        self.extra.update(cl_arrived=bool(np.isfinite(t_arr)), cl_t_arrive_s=float(t_arr), ctrl="cl")
        return ("drop" if self.dropped else "end"), t_arr


def _init(cfg):
    G["cfg"] = cfg
    ctx = EP.Ctx(cfg["setup"], cfg["seed"], cfg["split"], cfg.get("data") or None, None, BR.default_tau(cfg["setup"]), "Rc3k", PO.EPS, None)
    if cfg["filt"] != "F3":
        ctx.filt = F.make_filter(cfg["filt"], ctx.renv, seed=cfg["seed"], runs_root=str(K.ROOT / "runs"), model_setup=cfg["setup"])
    G["ctx"] = ctx


def _work(ep):
    cfg, ctx = G["cfg"], G["ctx"]
    t0 = time.perf_counter()
    info = ctx.store.ep_info(ep)
    if cfg["ctrl"] == "cl":
        RunCL.gains = dict(omega=cfg["omega"], e_lo=cfg["e_lo"], e_hi=cfg["e_hi"])
        r = RunCL(ctx, info["seed"], None)
    else:
        r = EP.Run(ctx, info["seed"], None)
    rec = r.run(f"FIX:I={cfg['I']:g}")
    if cfg["ctrl"] == "ol":
        rec["ctrl"] = "ol"
    else:
        rec.update(cl_omega=cfg["omega"], cl_e_lo=cfg["e_lo"], cl_e_hi=cfg["e_hi"])
    rec["ep"] = int(ep)
    rec["onset_stored"] = int(info["onset"])
    rec["wall_s"] = time.perf_counter() - t0
    rec["filt"] = cfg["filt"]
    for k, v in (ctx.filt.stats() or {}).items():      # filter stats of this carry (x_f5_*, x_p_*): added fields only, never overwrite
        rec.setdefault(k, v)
    return rec


def paths(out, ctrl, filt, split, setup, I, seed, shard):
    d = Path(out) / F.spec_tag(filt) / split / setup
    i, n = (int(x) for x in str(shard).split("/"))
    stem = f"{ctrl.upper()}__I-{I:g}__s{int(seed)}" + (f"__p{i}of{n}" if n > 1 else "")
    return d, d / f"{stem}.jsonl"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filt", required=True)
    ap.add_argument("--I", type=float, required=True)
    ap.add_argument("--setup", required=True, choices=["piper", "piper_trk"])
    ap.add_argument("--ctrl", default="cl", choices=["cl", "ol"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n-eps", type=int, default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--omega", type=float, default=PC.OMEGA)
    ap.add_argument("--e-lo", type=float, default=PC.E_LO)
    ap.add_argument("--e-hi", type=float, default=PC.E_HI)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--data", default="", help="dataset dir (default <root>/data_<setup>_full); e.g. data_piper_trk_full_v2")
    a = ap.parse_args()
    d, pj = paths(a.out, a.ctrl, a.filt, a.split, a.setup, a.I, a.seed, a.shard)
    d.mkdir(parents=True, exist_ok=True)
    st = Store(Path(a.data) if a.data else K.ROOT / f"data_{a.setup}_full", a.split)
    si, sn = (int(x) for x in a.shard.split("/"))
    eps = list(range(st.n_eps if a.n_eps is None else min(st.n_eps, a.n_eps)))[si::sn]
    done = {r["ep"] for r in BR.read_jsonl(pj)}
    todo = [e for e in eps if e not in done]
    BR.log(f"[run_cl] {a.ctrl} filt={a.filt} I={a.I:g} {a.setup} s{a.seed} {a.split} n={len(eps)} done={len(done)} todo={len(todo)} -> {pj}")
    cfg = dict(filt=a.filt, I=a.I, setup=a.setup, seed=a.seed, split=a.split, ctrl=a.ctrl, omega=a.omega, e_lo=a.e_lo, e_hi=a.e_hi, data=a.data)
    t0 = time.perf_counter()
    if todo:
        with open(pj, "a") as f:
            if a.workers <= 1:
                _init(cfg)
                for ep in todo:
                    f.write(json.dumps(_work(ep), default=float) + "\n"); f.flush()
            else:
                with Pool(a.workers, initializer=_init, initargs=(cfg,)) as P:
                    for i, r in enumerate(P.imap_unordered(_work, todo, chunksize=1)):
                        f.write(json.dumps(r, default=float) + "\n"); f.flush()
    recs = [r for r in BR.read_jsonl(pj) if r["ep"] in set(eps)]
    s = BR.summarize(recs)
    arr = [r["cl_arrived"] for r in recs if "cl_arrived" in r]
    BR.log(f"[run_cl] {a.ctrl} {a.filt} I={a.I:g} {a.setup} s{a.seed}: drop {s['drop']['k']}/{s['drop']['n']} goal {s['goal']['rate']:.2f} "
           f"t_complete {s['t_complete_s_ok']:.2f}s interv {s['interv_rate']:.3f}" + (f" arrived {sum(arr)}/{len(arr)}" if arr else "")
           + f" [{time.perf_counter() - t0:.0f}s]")


if __name__ == "__main__":
    main()
