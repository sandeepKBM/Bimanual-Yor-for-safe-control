"""bridge/run.py -- ONE filter x setup x seed over a split of nominal carries. Resumable. SIMULATION / SAVED DATA ONLY
(MuJoCo CPU + torch CPU for F3; no GPU, no hardware, no network).

  python -m world_model.v2.tray_rep.bridge.run --filter "F3:arm=Rc3k,tau=0.0" --setup piper_trk --seed 0 --split test --workers 8
  python -m world_model.v2.tray_rep.bridge.run --filter "F1:l_align=3,l_prox=30" --setup piper --split val --n-eps 200 --tag main

--setup   piper (noise-free full obs, data_piper_full) | piper_trk (full obs + realistic tracker, data_piper_trk_full).
          Data = <root>/data_<setup>_full unless --data. Models (F3) = <runs-root>/<model-setup>/<ARM>_full_b<budget>_s<seed>/.
--seed    F3: the model seed (N0_full_b0_s<seed>). All filters: shifts the tracker noise stream (0 = the stored stream).
          F0, F1, F2 have no model, so in the noise-free setup the seed changes nothing (run seed 0 only).
--filter  spec string (filters.py): F0 | F1:l_align=..,l_prox=.. | F2:theta_max=..,a_cap=.. | F3:arm=N0|Rc3k,tau=.. | F4:check_every=5
          presets: --filter "@F1@Rc3k" | "@F3@N0" | "@F2" reads the VAL-selected spec from <out>/<tag>/chosen_params.json
          (written by `report select`).
Output    <out>/<tag>/<split>/<setup>/<spec_tag>__s<seed>.jsonl  (one record per carry, appended as they finish; rerun = resume)
          and  .json  (summary). Rerunning with a finished file only rewrites the summary.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep.store import Store
from world_model.v2.tray_rep.bridge import filters as F
from world_model.v2.tray_rep.bridge import episode as EP
from world_model.v2.tray_rep.bridge import metrics as MT

OUT = K.ROOT / "bridge"
G = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def resolve_spec(spec, out_root, tag, setup):
    """'@F1@Rc3k' -> the spec string stored in <out>/<tag>/chosen_params.json under "<setup>/F1@Rc3k"; other strings pass through."""
    if not spec.startswith("@"):
        return spec
    key = f"{setup}/{spec[1:]}"
    p = Path(out_root) / tag / "chosen_params.json"
    ch = json.loads(p.read_text())
    if key not in ch:
        raise KeyError(f"{key} not in {p} (keys: {sorted(ch)})")
    return ch[key]["spec"]


def paths(out_root, tag, split, setup, spec, seed):
    d = Path(out_root) / tag / split / setup
    stem = f"{F.spec_tag(spec)}__s{int(seed)}"
    return d, d / f"{stem}.jsonl", d / f"{stem}.json"


def _init(cfg):
    G["cfg"] = cfg
    G["store"] = Store(cfg["data"], cfg["split"])
    G["renv"] = E.env_from_meta(G["store"].meta)
    G["meta"] = G["store"].meta
    G["filt"] = F.make_filter(cfg["spec"], G["renv"], seed=cfg["seed"], runs_root=cfg["runs_root"],
                              model_setup=cfg["model_setup"])


def _work(ep):
    cfg, st = G["cfg"], G["store"]
    info = st.ep_info(ep)
    logged = st.rows("act", ep, 0, info["lens"]) if cfg["nominal"] == "replay" else None
    t0 = time.perf_counter()
    r = EP.run_carry(G["renv"], G["meta"], info["seed"], G["filt"], run_seed=cfg["seed"], nominal=cfg["nominal"],
                     logged_act=logged, onset_stored=info["onset"])
    r["ep"] = int(ep)
    r["wall_s"] = time.perf_counter() - t0
    return r


def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    out = []
    for ln in p.read_text().splitlines():
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:                  # a half-written last line from a killed run
            pass
    return out


def run(spec, setup, seed, split, n_eps=None, workers=1, tag="main", out_root=OUT, data=None, runs_root=None,
        model_setup=None, nominal="closedloop", ep_start=0):
    out_root = Path(out_root)
    spec = resolve_spec(spec, out_root, tag, setup)
    data = Path(data) if data else K.ROOT / f"data_{setup}_full"
    runs_root = str(runs_root or K.ROOT / "runs")
    model_setup = model_setup or setup
    d, pj, ps = paths(out_root, tag, split, setup, spec, seed)
    d.mkdir(parents=True, exist_ok=True)
    st = Store(data, split)
    eps = list(range(ep_start, st.n_eps if n_eps is None else min(st.n_eps, ep_start + n_eps)))
    done = {r["ep"] for r in read_jsonl(pj)}
    todo = [e for e in eps if e not in done]
    log(f"[bridge.run] spec={spec} setup={setup} seed={seed} split={split} n={len(eps)} done={len(done)} todo={len(todo)} "
        f"workers={workers} nominal={nominal} -> {pj}")
    cfg = dict(data=str(data), split=split, spec=spec, seed=int(seed), runs_root=runs_root, model_setup=model_setup,
               nominal=nominal)
    t0 = time.perf_counter()
    if todo:
        with open(pj, "a") as f:
            if workers <= 1:
                _init(cfg)
                it = map(_work, todo)
                for i, r in enumerate(it):
                    f.write(json.dumps(r) + "\n"); f.flush()
                    if (i + 1) % 10 == 0:
                        log(f"  {i + 1}/{len(todo)}")
            else:
                with Pool(workers, initializer=_init, initargs=(cfg,)) as P:
                    for i, r in enumerate(P.imap_unordered(_work, todo, chunksize=1)):
                        f.write(json.dumps(r) + "\n"); f.flush()
                        if (i + 1) % 20 == 0:
                            log(f"  {i + 1}/{len(todo)}")
    recs = sorted((r for r in read_jsonl(pj) if r["ep"] in set(eps)), key=lambda r: r["ep"])
    summ = MT.summarize(recs)
    summ.update(spec=spec, setup=setup, seed=int(seed), split=split, nominal=nominal, data=str(data), model_setup=model_setup,
                runs_root=runs_root, wall_s_this_call=time.perf_counter() - t0,
                cpu_s_per_ep=float(np.mean([r["wall_s"] for r in recs])) if recs else None)
    ps.write_text(json.dumps(summ, indent=1, default=float))
    dr = summ["drop"]
    log(f"[bridge.run] {spec} {setup} s{seed} {split}: drop {dr['k']}/{dr['n']} = {dr['rate']:.3f} ci95 "
        f"({dr['ci95'][0]:.3f},{dr['ci95'][1]:.3f}) interv {summ['interv_rate']:.3f} dev_norm {summ['mean_dev_norm']:.4f} "
        f"compl {summ['completion']:.3f} saved {summ['saved']} caused {summ['caused']} "
        f"[{summ['cpu_s_per_ep']:.2f} s/ep]")
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filter", required=True)
    ap.add_argument("--setup", required=True, choices=["piper", "piper_trk"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n-eps", type=int, default=None, help="first N carries of the split (default all)")
    ap.add_argument("--ep-start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="main", help="sub-directory of --out (main | smoke | ...)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--data", default="")
    ap.add_argument("--runs-root", default="")
    ap.add_argument("--model-setup", default="")
    ap.add_argument("--nominal", default="closedloop", choices=["closedloop", "replay"])
    a = ap.parse_args()
    run(a.filter, a.setup, a.seed, a.split, a.n_eps, a.workers, a.tag, a.out, a.data or None, a.runs_root or None,
        a.model_setup or None, a.nominal, a.ep_start)


if __name__ == "__main__":
    main()
