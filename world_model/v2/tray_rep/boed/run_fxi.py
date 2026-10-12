"""boed/run_fxi.py -- Experiment C-F5 (extension of C): fixed-intensity carries of the BOED harness at each carry's OWN hidden mu, with ANY
bridge filter as the carry filter (F3 as in BOED, F5, or an F5P arm of sets/vlmprior). SIMULATION / SAVED DATA ONLY. CPU. No hardware.

New file. boed/run.py and boed/episode.py are NOT changed (the frozen Experiment C pre-registration hashes them): this runner builds the
BOED Ctx as boed/run.py does (F3 Rc3k at the bridge's tau), replaces ctx.filt by the requested filter, and runs the policy "FIX:I=<I_rel>"
(boed.episode.Run.run: carry_phase(plan_for(I))) with mu_override=None, i.e. the carry's own mu. Records = boed.episode.Run.record.

  python -m world_model.v2.tray_rep.boed.run_fxi --filt F5 --I 1.0 --setup piper_trk --seed 0 --split test --shard 0/4 --workers 2
Output: <out>/<spec_tag(filt)>/<split>/<setup>/FIX__I-<I>__s<seed>[__p<i>of<n>].jsonl. Rerun = resume (by episode).
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep.boed import episode as EP
from world_model.v2.tray_rep.boed import policy as PO
from world_model.v2.tray_rep.boed import run as BR
from world_model.v2.tray_rep.bridge import filters as F
from world_model.v2.tray_rep.store import Store

OUT = K.ROOT / "boed_f5"
G = {}


def _init(cfg):
    G["cfg"] = cfg
    ctx = EP.Ctx(cfg["setup"], cfg["seed"], cfg["split"], cfg.get("data") or None, None, BR.default_tau(cfg["setup"]), "Rc3k", PO.EPS, None)
    if cfg["filt"] != "F3":
        ctx.filt = F.make_filter(cfg["filt"], ctx.renv, seed=cfg["seed"], runs_root=str(K.ROOT / "runs"), model_setup=cfg["setup"])
    G["ctx"] = ctx


def _work(ep):
    r = EP.run_carry(G["ctx"], f"FIX:I={G['cfg']['I']:g}", ep, None)
    r["filt"] = G["cfg"]["filt"]
    return r


def paths(out, filt, split, setup, I, seed, shard):
    d = Path(out) / F.spec_tag(filt) / split / setup
    i, n = (int(x) for x in str(shard).split("/"))
    stem = f"FIX__I-{I:g}__s{int(seed)}" + (f"__p{i}of{n}" if n > 1 else "")
    return d, d / f"{stem}.jsonl"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filt", required=True)
    ap.add_argument("--I", type=float, required=True)
    ap.add_argument("--setup", required=True, choices=["piper", "piper_trk"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n-eps", type=int, default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--data", default="", help="dataset dir (default <root>/data_<setup>_full); e.g. data_piper_full_v2")
    a = ap.parse_args()
    d, pj = paths(a.out, a.filt, a.split, a.setup, a.I, a.seed, a.shard)
    d.mkdir(parents=True, exist_ok=True)
    st = Store(Path(a.data) if a.data else K.ROOT / f"data_{a.setup}_full", a.split)
    si, sn = (int(x) for x in a.shard.split("/"))
    eps = list(range(st.n_eps if a.n_eps is None else min(st.n_eps, a.n_eps)))[si::sn]
    done = {r["ep"] for r in BR.read_jsonl(pj)}
    todo = [e for e in eps if e not in done]
    BR.log(f"[run_fxi] filt={a.filt} I={a.I:g} {a.setup} s{a.seed} {a.split} n={len(eps)} done={len(done)} todo={len(todo)} -> {pj}")
    cfg = dict(filt=a.filt, I=a.I, setup=a.setup, seed=a.seed, split=a.split, data=a.data)
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
                        if (i + 1) % 20 == 0:
                            BR.log(f"  {i + 1}/{len(todo)}")
    recs = [r for r in BR.read_jsonl(pj) if r["ep"] in set(eps)]
    s = BR.summarize(recs)
    BR.log(f"[run_fxi] {a.filt} I={a.I:g} {a.setup} s{a.seed}: drop {s['drop']['k']}/{s['drop']['n']} goal {s['goal']['rate']:.2f} "
           f"t_complete {s['t_complete_s_ok']:.2f}s interv {s['interv_rate']:.3f} [{time.perf_counter() - t0:.0f}s]")


if __name__ == "__main__":
    main()
