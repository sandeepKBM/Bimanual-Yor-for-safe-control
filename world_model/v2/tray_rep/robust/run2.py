"""robust/run2.py -- ONE robust filter x setup x condition x seed over a split of nominal carries. Resumable.
SIMULATION / SAVED DATA ONLY (MuJoCo CPU + torch CPU; no GPU, no hardware, no network).

  python -m world_model.v2.tray_rep.robust.run2 --filter "rob:arm=Rc3k,thr=0.1,kappa=1" --setup piper_trk --cond trkB10f --seed 0 --split test --workers 2
  python -m world_model.v2.tray_rep.robust.run2 --filter "plain:arm=Rc3k,thr=-1e9" --setup piper_trk --cond trk --split val --n-eps 60 --record --tag profile

--setup   piper | piper_trk : which TRAINED models (runs/<setup>/...) and which dataset (carry seeds, physics) are used.
          The ball observation is rebuilt from the TRUE ball by the corruption condition in BOTH setups, so `piper_trk` models
          with `--cond off` and `piper` models with `--cond trk` are the two cross cells.
--cond    robust.corrupt condition (off | trk | trkB10f | lat3 | trklat5 | ...)
--seed    the model seed (epi uses seeds 0-2 whatever this is). It also shifts the corruption seed, so filters run with the same
          seed see the same corrupted stream on the same carry (paired comparison).
Output    <out>/test2/<tag>/<split>/<setup>/<cond>/<spec_tag>__s<seed>.{jsonl,json}
"""
from __future__ import annotations

import argparse
import json
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K          # noqa: F401
from world_model.v2.tray_rep.store import Store
from world_model.v2.tray_rep.bridge import filters as BF
from world_model.v2.tray_rep.bridge import metrics as MT
from world_model.v2.tray_rep.robust import corrupt as CR
from world_model.v2.tray_rep.robust import episode as EPR
from world_model.v2.tray_rep.robust import rfilters as RF
from world_model.v2.tray_rep.robust import util as RU

G = {}


def spec_tag(spec):
    kind, p = BF.parse_spec(spec)
    items = []
    for k, v in sorted(p.items()):
        if k == "calib":
            v = "calib"
        items.append(f"{k}-{v:g}" if isinstance(v, float) else f"{k}-{v}")
    return kind + "__" + "_".join(items)


def paths(out_root, tag, split, setup, cond, spec, seed):
    d = Path(out_root) / "test2" / tag / split / setup / cond
    stem = f"{spec_tag(spec)}__s{int(seed)}"
    return d, d / f"{stem}.jsonl", d / f"{stem}.json"


def _init(cfg):
    G["cfg"] = cfg
    G["st"] = Store(cfg["data"], cfg["split"])
    G["meta"] = G["st"].meta
    G["renv"] = RU.make_clean_env(G["meta"])
    G["cond"] = CR.parse_cond(cfg["cond"])
    G["filt"] = RF.make_rfilter(cfg["spec"], G["renv"], cfg["model_setup"], cfg["seed"], G["cond"], cfg["runs_root"],
                                record=cfg["record"])


def _work(ep):
    cfg, st = G["cfg"], G["st"]
    info = st.ep_info(ep)
    t0 = time.perf_counter()
    r = EPR.run_carry_c(G["renv"], G["meta"], info["seed"], G["filt"], G["cond"], run_seed=cfg["seed"],
                        onset_stored=info["onset"])
    r["ep"] = int(ep)
    r["wall_s"] = time.perf_counter() - t0
    return r


def summarize(recs):
    """bridge summary + phase splits. Drops and interventions by phase; carries split by class and by nominal outcome."""
    s = MT.summarize(recs)
    g = lambda k: np.array([r[k] for r in recs])
    dropped, nom, cls = g("dropped").astype(bool), g("nom_drop").astype(bool), g("cls")
    tph, iph = np.array([r["ticks_ph"] for r in recs]).sum(0), np.array([r["interv_ph"] for r in recs]).sum(0)
    dph = g("drop_phase")
    s["phase"] = {nm: dict(ticks=int(tph[i]), interv=int(iph[i]), interv_rate=float(iph[i] / max(tph[i], 1)),
                           drops_at_phase=int(((dph == i) & dropped).sum()),
                           carries_cls=int((cls == i).sum()),
                           drop_cls=RU.rate(int((dropped & (cls == i)).sum()), int((cls == i).sum())),
                           drop_cls_nomfail=RU.rate(int((dropped & nom & (cls == i)).sum()), int((nom & (cls == i)).sum())))
                  for i, nm in enumerate(CR.PHASES)}
    return s


def read_summary(pj):
    return RU.read_jsonl(pj)


def run(spec, setup, seed, cond, split, n_eps=None, workers=1, tag="main", out_root=RU.OUT, model_setup=None, record=False,
        ep_start=0, quiet=False):
    out_root = Path(out_root)
    model_setup = model_setup or setup
    data = RU.data_dir(setup)
    d, pj, ps = paths(out_root, tag, split, setup, cond, spec, seed)
    d.mkdir(parents=True, exist_ok=True)
    st = Store(data, split)
    eps = list(range(ep_start, st.n_eps if n_eps is None else min(st.n_eps, ep_start + n_eps)))
    done = {r["ep"] for r in RU.read_jsonl(pj)}
    todo = [e for e in eps if e not in done]
    if not quiet:
        RU.log(f"[run2] {spec} {setup} cond={cond} s{seed} {split} n={len(eps)} done={len(done)} todo={len(todo)} -> {pj.name}")
    cfg = dict(data=str(data), split=split, spec=spec, seed=int(seed), runs_root=RU.runs_root(), model_setup=model_setup,
               cond=cond, record=bool(record))
    t0 = time.perf_counter()
    if todo:
        with open(pj, "a") as f:
            if workers <= 1:
                _init(cfg)
                for i, e in enumerate(todo):
                    f.write(json.dumps(_work(e)) + "\n"); f.flush()
                    if not quiet and (i + 1) % 10 == 0:
                        RU.log(f"  {i + 1}/{len(todo)}")
            else:
                with Pool(workers, initializer=_init, initargs=(cfg,)) as P:
                    for i, r in enumerate(P.imap_unordered(_work, todo, chunksize=1)):
                        f.write(json.dumps(r) + "\n"); f.flush()
                        if not quiet and (i + 1) % 20 == 0:
                            RU.log(f"  {i + 1}/{len(todo)}")
    keep = set(eps)
    recs = sorted((r for r in RU.read_jsonl(pj) if r["ep"] in keep), key=lambda r: r["ep"])
    summ = summarize(recs)
    summ.update(spec=spec, setup=setup, seed=int(seed), split=split, cond=cond, model_setup=model_setup,
                wall_s_this_call=time.perf_counter() - t0,
                cpu_s_per_ep=float(np.mean([r["wall_s"] for r in recs])) if recs else None)
    ps.write_text(json.dumps(summ, indent=1, default=float))
    if not quiet:
        dr = summ["drop"]
        RU.log(f"[run2] {spec} {setup} {cond} s{seed}: drop {dr['k']}/{dr['n']} = {dr['rate']:.3f} interv {summ['interv_rate']:.4f} "
               f"[{summ['cpu_s_per_ep']:.2f} s/ep]")
    return summ, recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filter", required=True)
    ap.add_argument("--setup", required=True, choices=RU.SETUPS)
    ap.add_argument("--cond", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n-eps", type=int, default=None)
    ap.add_argument("--ep-start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", default=str(RU.OUT))
    ap.add_argument("--model-setup", default="")
    ap.add_argument("--record", action="store_true", help="store per-tick scores and residuals in the records (profile runs)")
    a = ap.parse_args()
    run(a.filter, a.setup, a.seed, a.cond, a.split, a.n_eps, a.workers, a.tag, a.out, a.model_setup or None, a.record, a.ep_start)


if __name__ == "__main__":
    main()
