"""al/build_voi.py -- build the probe pre-posterior table (probe_voi.py) for one regime on VAL carries. SIMULATION ONLY. CPU.
  python -m world_model.v2.tray_rep.al.build_voi --regime normal --out <json> [--thetas 0.06,0.12 --n-levels 12 --reps 6 --workers 4]
For each (theta, level) it runs `reps` different VAL episodes with the hidden mu forced to the level (Run(mu_override)), the real probe (no certificate gate: the table also
records what the certificate says at the regime's lowest mu, for information), and stores the v2-map probe log-likelihood. Episodes 0..reps-1 of VAL are used for every cell.
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep.al import probe_v2 as PV
from world_model.v2.tray_rep.al import run_al as RA
from world_model.v2.tray_rep.boed import episode as EP

G = {}


def _init(cfg):
    G["cfg"] = cfg
    G["ctx"] = RA.make_ctx(cfg)


def _work(task):
    th, j, mu, ep = task
    ctx = G["ctx"]
    info = ctx.store.ep_info(ep)
    r = EP.Run(ctx, info["seed"], mu)
    PV.prepare(r)
    cert = PV.certify(r, th, G["cfg"]["mu_cert"])
    res = PV.probe(r, th)
    return dict(theta=th, j=j, mu=mu, ep=ep, ll=None if res["ll"] is None else res["ll"].tolist(), ticks=res["ticks"], dropped=res["dropped"], n_hold=res["n_hold"],
                reason=res["reason"], cert_ok=cert["ok"], cert_dmin=cert["dmin"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regime", default="normal", choices=list(RA.REGIMES))
    ap.add_argument("--thetas", default="0.06,0.12")
    ap.add_argument("--n-levels", type=int, default=12)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=RA.DATA)
    a = ap.parse_args()
    R = RA.REGIMES[a.regime]
    lo, hi = (5e-4, 6e-3) if R["mu_range"] is None else R["mu_range"]
    levels = np.exp(np.linspace(np.log(lo), np.log(hi), a.n_levels)).tolist()
    cfg = dict(setup="piper_trk", seed=0, split="val", data=a.data, auth=R["auth"], wrong=(1.0, 2.0), mu_cert=float(lo))
    thetas = [float(x) for x in a.thetas.split(",")]
    tasks = [(th, j, levels[j], ep) for th in thetas for j in range(a.n_levels) for ep in range(a.reps)]
    t0 = time.perf_counter()
    with Pool(a.workers, initializer=_init, initargs=(cfg,)) as P:
        res = list(P.imap_unordered(_work, tasks, chunksize=1))
    ent = {}
    for r in res:
        ent.setdefault(f"{r['theta']:g}", {}).setdefault(str(r["j"]), []).append(dict(ll=r["ll"], ticks=r["ticks"], dropped=r["dropped"], ep=r["ep"], n_hold=r["n_hold"],
                                                                                   reason=r["reason"], cert_ok=r["cert_ok"], cert_dmin=r["cert_dmin"]))
    for t in ent.values():
        for v in t.values():
            v.sort(key=lambda x: x["ep"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(regime=a.regime, auth=R["auth"], mu_levels=levels, reps=a.reps, thetas=thetas, entries=ent), default=float))
    nd = sum(r["dropped"] for r in res)
    print(f"[build_voi] {a.regime}: {len(res)} probes, {nd} dropped, cert ok {sum(r['cert_ok'] for r in res)}/{len(res)} at mu={lo:g} [{time.perf_counter() - t0:.0f}s] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
