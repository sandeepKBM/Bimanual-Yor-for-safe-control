"""boed/run.py -- ONE policy x setup x value-seed over a split of carries. Resumable. SIMULATION / SAVED DATA ONLY
(MuJoCo CPU + torch CPU for F3; no GPU, no hardware, no network).

  python -m world_model.v2.tray_rep.boed.run --policy P3 --setup piper_trk --seed 0 --split test --workers 4 --tag main
  python -m world_model.v2.tray_rep.boed.run --policy "FIX:I=0.55,j=2" --setup piper_trk --seed 0 --split val --n-eps 40 --tag main

--policy  P1..P6, P7, P7s (P8 = stub) (need the table <out>/<tag>/table_<setup>.json from `table assemble`) | FIX:I=<I_rel>,j=<mu node> (VAL cells for the table)
--seed    value seed = F3 model seed (Rc_full_b3000_s<seed>) = tracker-noise stream shift (0 = the stored stream), as in the bridge.
Output    <out>/<tag>/<split>/<setup>/<policy_tag>__s<seed>.{jsonl,json}   one json line per carry, appended as it finishes (rerun = resume).
The tau of F3 is the bridge's VAL choice (bridge/main/chosen_params.json, key "<setup>/F3@Rc3k"; piper has none: it takes piper_trk's).
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep.bridge import metrics as MT
from world_model.v2.tray_rep.bridge.filters import parse_spec, spec_tag
from world_model.v2.tray_rep.boed import episode as EP
from world_model.v2.tray_rep.boed import policy as PO
from world_model.v2.tray_rep.boed import p7 as P7

OUT = K.ROOT / "boed"
BRIDGE_PARAMS = K.ROOT / "bridge" / "main" / "chosen_params.json"
G = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def default_tau(setup):
    ch = json.loads(BRIDGE_PARAMS.read_text())
    for key in (f"{setup}/F3@Rc3k", "piper_trk/F3@Rc3k"):
        if key in ch:
            return float(parse_spec(ch[key]["spec"])[1]["tau"])
    return 0.1


def run_paths(out_root, tag, split, setup, policy, seed, shard="0/1"):
    """-> (dir, jsonl, json). A shard i/n (Slurm array pieces of one run) gets its own files: <stem>__p<i>of<n>."""
    d = Path(out_root) / tag / split / setup
    i, n = (int(x) for x in str(shard).split("/"))
    stem = f"{spec_tag(policy)}__s{int(seed)}" + (f"__p{i}of{n}" if n > 1 else "")
    return d, d / f"{stem}.jsonl", d / f"{stem}.json"


def read_all(out_root, tag, split, setup, policy, seed):
    """records of a run, whole or in shards (every <stem>.jsonl and <stem>__p*.jsonl), de-duplicated by episode."""
    d = Path(out_root) / tag / split / setup
    stem = f"{spec_tag(policy)}__s{int(seed)}"
    recs = {}
    for p in sorted([d / f"{stem}.jsonl"] + list(d.glob(f"{stem}__p*of*.jsonl"))):
        for r in read_jsonl(p):
            recs[r["ep"]] = r
    return [recs[k] for k in sorted(recs)]


def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    out = []
    for ln in p.read_text().splitlines():
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    return out


def _init(cfg):
    G["cfg"] = cfg
    table = PO.Table.load(cfg["table"]) if cfg.get("table") else None
    G["ctx"] = EP.Ctx(cfg["setup"], cfg["seed"], cfg["split"], cfg.get("data"), cfg.get("runs_root"), cfg["tau"], "Rc3k", cfg["eps"], table,
                      n_nodes=cfg["n_nodes"], cert=cfg["cert"], p7=P7.P7Spec.parse(cfg.get("p7_opts", "")))


def _work(ep):
    cfg, ctx = G["cfg"], G["ctx"]
    pol = cfg["policy"]
    mu = None
    if pol.startswith("FIX"):
        mu = float(ctx.nodes[int(parse_spec(pol)[1]["j"])])
    return EP.run_carry(ctx, pol, ep, mu)


def summarize(recs):
    n = len(recs)
    if n == 0:
        return dict(n=0)
    g = lambda k: np.array([r[k] for r in recs], float)
    dr = g("dropped").astype(bool)
    ok = ~dr
    nan_mean = lambda a: float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")
    out = dict(n=n, drop=MT._rate(int(dr.sum()), n), progress=float(g("progress").mean()), goal=MT._rate(int(g("goal_reached").sum()), n),
               t_complete_s_ok=nan_mean(g("t_complete_s")), total_s=float(g("total_s").mean()), interv_rate=float(g("interv_rate").mean()),
               probe_run=int(sum(bool(r.get("probe_run")) for r in recs)), probe_drop=int(sum(bool(r.get("probe_drop")) for r in recs)),
               probe_skipped=int(sum(bool(r.get("probe_skipped")) for r in recs)), p4_triggered=int(sum(r.get("p4_trigger_tick", -1) >= 0 for r in recs)),
               I_rel_mean=float(np.nanmean([r["I_rel"] if r.get("I_rel") is not None else np.nan for r in recs])) if any(r.get("I_rel") is not None for r in recs) else None,
               wall_s_per_ep=float(g("wall_s").mean()))
    if any("bel_cover90" in r for r in recs):
        b = [r for r in recs if "bel_cover90" in r]
        out.update(bel_n=len(b), bel_cover90=float(np.mean([r["bel_cover90"] for r in b])), bel_cover50=float(np.mean([r["bel_cover50"] for r in b])),
                   bel_sd_log=float(np.mean([r["bel_sd_log"] for r in b])), bel_rmse_log=float(np.sqrt(np.mean([r["bel_err_log"] ** 2 for r in b]))))
    return out


def run(policy, setup, seed, split, n_eps=None, workers=1, tag="main", out_root=OUT, table=None, tau=None, eps=PO.EPS, n_nodes=PO.N_NODES,
        data=None, runs_root=None, cert=True, ep_start=0, shard="0/1", table_dir=None, p7_opts=""):
    from world_model.v2.tray_rep.store import Store
    out_root = Path(out_root)
    tau = default_tau(setup) if tau is None else float(tau)
    if table is None and not policy.startswith(("FIX", "P2")):
        table = Path(table_dir or out_root / tag) / f"table_{setup}.json"       # table_dir: read the (frozen) table from another run's folder
    d, pj, ps = run_paths(out_root, tag, split, setup, policy, seed, shard)
    d.mkdir(parents=True, exist_ok=True)
    st = Store(Path(data) if data else K.ROOT / f"data_{setup}_full", split)
    si, sn = (int(x) for x in str(shard).split("/"))
    eps_l = list(range(ep_start, st.n_eps if n_eps is None else min(st.n_eps, ep_start + n_eps)))[si::sn]
    done = {r["ep"] for r in read_jsonl(pj)}
    todo = [e for e in eps_l if e not in done]
    log(f"[boed.run] {policy} {setup} s{seed} {split} n={len(eps_l)} done={len(done)} todo={len(todo)} workers={workers} tau={tau} -> {pj}")
    cfg = dict(policy=policy, setup=setup, seed=int(seed), split=split, table=str(table) if table else None, tau=tau, eps=eps, n_nodes=n_nodes,
               data=str(data) if data else None, runs_root=str(runs_root) if runs_root else None, cert=cert, p7_opts=p7_opts)
    t0 = time.perf_counter()
    if todo:
        with open(pj, "a") as f:
            if workers <= 1:
                _init(cfg)
                for i, ep in enumerate(todo):
                    f.write(json.dumps(_work(ep), default=float) + "\n"); f.flush()
                    if (i + 1) % 10 == 0:
                        log(f"  {i + 1}/{len(todo)}")
            else:
                with Pool(workers, initializer=_init, initargs=(cfg,)) as P:
                    for i, r in enumerate(P.imap_unordered(_work, todo, chunksize=1)):
                        f.write(json.dumps(r, default=float) + "\n"); f.flush()
                        if (i + 1) % 20 == 0:
                            log(f"  {i + 1}/{len(todo)}")
    recs = sorted((r for r in read_jsonl(pj) if r["ep"] in set(eps_l)), key=lambda r: r["ep"])
    summ = summarize(recs)
    summ.update(policy=policy, setup=setup, seed=int(seed), split=split, tau=tau, eps=eps, table=str(table) if table else None,
                wall_s_this_call=time.perf_counter() - t0)
    ps.write_text(json.dumps(summ, indent=1, default=float))
    dr = summ["drop"]
    log(f"[boed.run] {policy} {setup} s{seed} {split}: drop {dr['k']}/{dr['n']} = {dr['rate']:.3f} progress {summ['progress']:.3f} "
        f"goal {summ['goal']['rate']:.2f} t_complete {summ['t_complete_s_ok']:.1f}s probes {summ['probe_run']} probe_drops {summ['probe_drop']} "
        f"[{summ['wall_s_per_ep']:.2f} s/ep]")
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--setup", required=True, choices=["piper", "piper_trk"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n-eps", type=int, default=None)
    ap.add_argument("--ep-start", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n: this task takes every n-th carry (Slurm array piece); own output files")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--table", default="")
    ap.add_argument("--table-dir", default="", help="folder holding table_<setup>.json (default <out>/<tag>); e.g. the frozen .../boed/main (read only)")
    ap.add_argument("--p7-opts", default="", help="P7 certificate settings, k=v,...: d_margin,h_after,stop_min,stop_step,k_sigma,eps_a,eps_p0,kd,kp,fb_delay,q_cert,q_decision")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--eps", type=float, default=PO.EPS)
    ap.add_argument("--n-nodes", type=int, default=PO.N_NODES)
    ap.add_argument("--data", default="")
    ap.add_argument("--runs-root", default="")
    ap.add_argument("--no-cert", dest="cert", action="store_false", help="skip the probe certificate (ablation)")
    a = ap.parse_args()
    run(a.policy, a.setup, a.seed, a.split, a.n_eps, a.workers, a.tag, a.out, a.table or None, a.tau, a.eps, a.n_nodes, a.data or None,
        a.runs_root or None, a.cert, a.ep_start, a.shard, a.table_dir or None, a.p7_opts)


if __name__ == "__main__":
    main()
