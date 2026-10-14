"""wrongpoint/run_cell.py -- C4: where does a confidently wrong POINT mu prior cause drops? SIMULATION ONLY (MuJoCo CPU). No hardware, no network.

One CELL = (carry intensity I, F5 eps_scale, mu population, authority scale, wrong-prior magnitude). For each carry of the chosen split and each
arm key it runs the BOED fixed-intensity carry (boed.episode.Run, F5P / F0 as the carry filter) exactly as boed/run_fxi.py does, EXCEPT that this
runner can change (all optional, defaults = run_fxi):
  --mu-range lo,hi   the carry's TRUE mu is redrawn log-uniform in [lo,hi] (deterministic per carry seed); F5/F5P keep the population set 5e-4..6e-3
  --auth f           multiplies a_scale and w_scale of the arm (piper: 0.3, 0.5); alpha (tilt slew) multiplied by f too with --auth-alpha
  --wrong lo,hi      |error| range (in ln mu) of the confidently wrong prior (default 1,2 = prior.py)
  --eps-scale s      F5 model-error box multiplier (F5 key eps_scale)
Arm keys: point@rho, conf@rho, oracle, F0 (no filter reference), F5 (plain, population set).
Output: <out>/<cell>/<arm>.jsonl  (one record per carry; rerun = resume).
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
from world_model.v2.tray_rep.boed import carry as CA
from world_model.v2.tray_rep.boed import episode as EP
from world_model.v2.tray_rep.boed import policy as PO
from world_model.v2.tray_rep.boed import run as BR
from world_model.v2.tray_rep.bridge import filters as F
from world_model.v2.tray_rep.sets.vlmprior import prior as PR
from world_model.v2.tray_rep.store import Store

G = {}


def spec_of(arm, eps_scale, pseed):
    if arm == "F0":
        return "F0"
    if arm == "F5":
        return f"F5:eps_scale={eps_scale:g}"
    if arm == "oracle":
        return f"F5P:arm=oracle,eps_scale={eps_scale:g}"
    a, rho = arm.split("@")
    return f"F5P:arm={a},rho={rho},pseed={pseed},eps_scale={eps_scale:g}"


def mu_for(seed, lo, hi):
    rng = np.random.default_rng([0x4D55, int(seed)])
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _init(cfg):
    G["cfg"] = cfg
    PR.WRONG_LO, PR.WRONG_HI = cfg["wrong"]
    ctx = EP.Ctx(cfg["setup"], 0, cfg["split"], None, None, BR.default_tau(cfg["setup"]), "Rc3k", PO.EPS, None)
    if cfg["auth"] != 1.0 or cfg["auth_alpha"] != 1.0:
        m = dict(ctx.meta)
        m["a_scale"] = ctx.meta["a_scale"] * cfg["auth"]
        m["w_scale"] = ctx.meta["w_scale"] * cfg["auth"]
        m["alpha"] = ctx.meta["alpha"] * cfg["auth_alpha"]
        ctx.renv = E.env_from_meta(m)
        ctx.shadow = E.env_from_meta(m)
    G["ctx"] = ctx
    G["filters"] = {}


def _filt(arm):
    f = G["filters"].get(arm)
    if f is None:
        cfg = G["cfg"]
        f = F.make_filter(spec_of(arm, cfg["eps_scale"], cfg["pseed"]), G["ctx"].renv, seed=0, runs_root=str(K.ROOT / "runs"),
                          model_setup=cfg["setup"])
        G["filters"][arm] = f
    return f


def _work(task):
    arm, ep = task
    cfg, ctx = G["cfg"], G["ctx"]
    ctx.filt = _filt(arm)
    info = ctx.store.ep_info(ep)
    mu = None if cfg["mu_range"] is None else mu_for(info["seed"], *cfg["mu_range"])
    t0 = time.perf_counter()
    r = EP.Run(ctx, info["seed"], mu)
    rec = r.run(f"FIX:I={cfg['I']:g}")
    rec["ep"] = int(ep)
    rec["wall_s"] = time.perf_counter() - t0
    rec["arm"] = arm
    if "@" in arm:
        a, rho = arm.split("@")
        m, sg, cf = PR.draw_prior(float(np.log(rec["mu"])), info["seed"], float(rho), cfg["pseed"])
        rec.update(prior_m=m, prior_sigma=sg, prior_conf=bool(cf), prior_err=float(m - np.log(rec["mu"])))
    return rec


def read(p):
    return BR.read_jsonl(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--arms", default="point@0.2,conf@0.2,point@0.4,conf@0.4,oracle,F0")
    ap.add_argument("--I", type=float, default=0.99)
    ap.add_argument("--eps-scale", type=float, default=1.0)
    ap.add_argument("--mu-range", default=None)
    ap.add_argument("--auth", type=float, default=1.0)
    ap.add_argument("--auth-alpha", type=float, default=1.0)
    ap.add_argument("--wrong", default="1,2")
    ap.add_argument("--pseed", type=int, default=0)
    ap.add_argument("--setup", default="piper_trk")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-eps", type=int, default=100)
    ap.add_argument("--ep-start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=str(K.ROOT / "wrongpoint" / "runs"))
    a = ap.parse_args()
    assert a.workers <= 12
    cfg = dict(I=a.I, eps_scale=a.eps_scale, mu_range=None if a.mu_range is None else tuple(float(x) for x in a.mu_range.split(",")),
               auth=a.auth, auth_alpha=a.auth_alpha, wrong=tuple(float(x) for x in a.wrong.split(",")), pseed=a.pseed, setup=a.setup, split=a.split)
    d = Path(a.out) / a.cell
    d.mkdir(parents=True, exist_ok=True)
    (d / "cfg.json").write_text(json.dumps(dict(cfg, arms=a.arms, n_eps=a.n_eps, ep_start=a.ep_start), default=float))
    st = Store(K.ROOT / f"data_{a.setup}_full", a.split)
    eps = list(range(a.ep_start, min(st.n_eps, a.ep_start + a.n_eps)))
    arms = a.arms.split(",")
    todo = []
    for arm in arms:
        done = {r["ep"] for r in read(d / f"{arm}.jsonl")}
        todo += [(arm, e) for e in eps if e not in done]
    BR.log(f"[wrongpoint] cell={a.cell} cfg={cfg} arms={arms} eps={len(eps)} todo={len(todo)}")
    t0 = time.perf_counter()
    if todo:
        with Pool(a.workers, initializer=_init, initargs=(cfg,)) as P:
            fh = {arm: open(d / f"{arm}.jsonl", "a") for arm in arms}
            for i, r in enumerate(P.imap_unordered(_work, todo, chunksize=1)):
                fh[r["arm"]].write(json.dumps(r, default=float) + "\n"); fh[r["arm"]].flush()
                if (i + 1) % 100 == 0:
                    BR.log(f"  {i + 1}/{len(todo)} [{time.perf_counter() - t0:.0f}s]")
            for f in fh.values():
                f.close()
    for arm in arms:
        rs = [r for r in read(d / f"{arm}.jsonl") if r["ep"] in set(eps)]
        BR.log(f"  {a.cell} {arm}: n={len(rs)} drop={sum(r['dropped'] for r in rs)} interv={np.mean([r['interv_rate'] for r in rs]):.3f} "
               f"prog={np.mean([r['progress'] for r in rs]):.4f}")
    BR.log(f"[wrongpoint] {a.cell} done [{time.perf_counter() - t0:.0f}s]")


if __name__ == "__main__":
    main()
