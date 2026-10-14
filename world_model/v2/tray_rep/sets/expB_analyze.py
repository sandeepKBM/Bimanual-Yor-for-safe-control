"""sets/expB_analyze.py -- verdict of Experiment B (bimanual_data/tray_rep/vlmprior/expB/PREREG_B_FROZEN.md). SAVED DATA ONLY. CPU.

Does a calibrated SET built from a (possibly confidently wrong) synthetic VLM prior beat a fused POINT estimate (Phys2Real-style)?
All arms run F5 (sets/filter.py); only the rolling-resistance set handed to F5 differs (sets/vlmprior/filter.py, F5PFilter).

  PRIMARY (piper_trk, rho = 0.2):
    KB1 "sets beat points" FAILS (KILL) unless the 95% CI of drop(point) - drop(conf) lies entirely above 0.
    KB2 "the prior is dead weight" if |progress(conf) - progress(unif)| / progress(unif) < 0.02 (point estimates).
        Reported with the oracle headroom H = (progress(oracle) - progress(unif)) / progress(unif); if H < 0.02 the KB2 outcome is
        additionally labelled "no headroom" (knowing mu exactly does not move progress either).
  Secondary: the same at rho = 0, 0.1, 0.4; band vs conf; intervention rate; coverage of the set at t = 0 / 1 / 2 s / end;
  piper (noise-free, seed 0); unif (online shrink only) vs plain F5 of Experiment A (same carries and seeds).

Bootstrap: sets.expA_analyze.boot_diff (hierarchical paired over 600 carries x 3 seeds, B = 10000), rng 20261009.
  python -m world_model.v2.tray_rep.sets.expB_analyze [--root R]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep.bridge import metrics as M
from world_model.v2.tray_rep.sets import expA_analyze as A

ROOT = Path("/common/users/ss5772/bimanual_data/tray_rep/vlmprior/expB")
EXPA = Path("/common/users/ss5772/bimanual_data/tray_rep/sets/expA")
RHOS = (0.0, 0.1, 0.2, 0.4)
PRIMARY_RHO = 0.2
N = 600
SEEDS = {"piper_trk": (0, 1, 2), "piper": (0,)}
CK = ("t0", "t1s", "t2s", "end")


def spec(arm, rho=None):
    return f"F5P:arm={arm}" if rho is None else f"F5P:arm={arm},rho={rho:g},pseed=0"


def arms_for(rho):
    return {"point": spec("point", rho), "band": spec("band", rho), "conf": spec("conf", rho), "unif": spec("unif"), "oracle": spec("oracle")}


def cov(recs, name):
    v = [r.get(f"x_p_cov_{name}") for s in recs for r in recs[s]]
    v = [x for x in v if x is not None]
    return (float(np.mean(v)) if v else float("nan")), len(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args()
    root = Path(a.root)
    rng = np.random.default_rng(20261009)
    res, L = {}, []
    for setup in ("piper_trk", "piper"):
        seeds = SEEDS[setup]
        res[setup] = {}
        for rho in RHOS:
            arms = arms_for(rho)
            data, rs = {}, {}
            for arm, sp in arms.items():
                recs = A.load(root, setup, sp, seeds, N)
                if recs is None:
                    continue
                data[arm] = recs
                sm = M.summarize([r for s in seeds for r in recs[s]])
                rs[arm] = dict(spec=sp, drop=sm["drop"]["rate"], interv=sm["interv_rate"], dev=sm["mean_dev_norm"],
                               progress=sm.get("progress"), cov={c: cov(recs, c) for c in CK},
                               logw=float(np.mean([r.get("x_p_logw_mean", np.nan) for s in seeds for r in recs[s]])),
                               mu_bins=A.mu_bins(recs))
            con = {}
            for x, y, k in [("point", "conf", "dropped"), ("band", "conf", "dropped"), ("point", "oracle", "dropped"),
                            ("conf", "unif", "dropped"), ("conf", "unif", "progress"), ("oracle", "unif", "progress"),
                            ("conf", "unif", "n_interv"), ("point", "conf", "progress")]:
                if x in data and y in data:
                    con[f"{x}-{y}:{k}"] = A.boot_diff(A.mat(data[x], k), A.mat(data[y], k), rng)
            out = dict(arms=rs, contrasts=con)
            kb1 = kb2 = head = None
            if "point-conf:dropped" in con:
                kb1 = "PASS" if con["point-conf:dropped"][1] > 0 else "KILL"
            if all(k in rs and rs[k]["progress"] for k in ("conf", "unif", "oracle")):
                pu = rs["unif"]["progress"]
                rel = (rs["conf"]["progress"] - pu) / pu
                head = (rs["oracle"]["progress"] - pu) / pu
                kb2 = ("DEAD WEIGHT" if abs(rel) < 0.02 else "prior adds progress") + (" (no headroom)" if head < 0.02 else "")
                out["kb2_rel"] = rel
            out.update(KB1=kb1, KB2=kb2, headroom=head)
            res[setup][f"rho={rho:g}"] = out
            tag = "PRIMARY" if (setup == "piper_trk" and rho == PRIMARY_RHO) else "secondary"
            L += [f"## {setup}, rho = {rho:g} ({tag})", "",
                  "| arm | drop | interv | dev | progress | mean ln(hi/lo) | coverage t0 / 1 s / 2 s / end |", "|---|---|---|---|---|---|---|"]
            for arm, r in rs.items():
                L.append(f"| {arm} | {100*r['drop']:.1f}% | {100*r['interv']:.1f}% | {r['dev']:.3f} | {r['progress']:.4f} | {r['logw']:.2f} | "
                         + " / ".join(f"{c[0]:.2f}" for c in r["cov"].values()) + " |")
            L += ["", "| contrast | point | 95% CI |", "|---|---|---|"]
            for k, (p, lo, hi) in con.items():
                sc = 100 if k.endswith("dropped") else 1
                L.append(f"| {k} | {sc*p:+.4f}{' pp' if sc == 100 else ''} | [{sc*lo:+.4f}, {sc*hi:+.4f}] |")
            L += ["", f"**KB1 (sets beat points): {kb1}; KB2: {kb2}; oracle headroom {head if head is None else round(100*head, 2)}%**", ""]
        # online shrink alone: unif (F5P, population range cut by the online interval) vs plain F5 (Experiment A, same carries/seeds)
        u = A.load(root, setup, spec("unif"), seeds, N)
        f5 = A.load(EXPA, setup, "F5", seeds, N)
        if u is not None and f5 is not None:
            c = {k: A.boot_diff(A.mat(u, k), A.mat(f5, k), rng) for k in ("dropped", "progress", "n_interv")}
            res[setup]["unif_vs_F5"] = c
            L += [f"Online shrink alone ({setup}): unif - plain F5: " + "; ".join(f"{k} {p:+.4f} [{lo:+.4f}, {hi:+.4f}]" for k, (p, lo, hi) in c.items()), ""]
    md = "\n".join(L) + "\n"
    (root / "analysis").mkdir(parents=True, exist_ok=True)
    (root / "analysis" / "tables_B.md").write_text(md)
    (root / "analysis" / "summary_B.json").write_text(json.dumps(res, indent=1, default=float))
    print(md, flush=True)


if __name__ == "__main__":
    main()
