"""sets/vlmprior/calibrate.py -- conformal q table per rho, from VAL carries only (no simulation: the prior draw needs only l_true and the carry seed).

  python -m world_model.v2.tray_rep.sets.vlmprior.calibrate [--rhos 0 0.1 0.2 0.4] [--alpha 0.1]

l_true of a VAL carry = ln of the first draw of default_rng(carry seed) (bridge/episode.Nominal). The prior draws use prior_seed PSEED_CAL, which
no evaluation run uses. n_cal = number of VAL carries. Writes bimanual_data/tray_rep/vlmprior/conformal_q.json (merges with existing rhos).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep.store import Store
from world_model.v2.tray_rep.sets.vlmprior import conformal as CF


def val_truth(data=None):
    st = Store(data or K.ROOT / "data_piper_trk_full", "val")
    seeds = np.array([int(st.ep_info(i)["seed"]) for i in range(st.n_eps)])
    l_true = np.array([np.log(E.sample_mu_roll(np.random.default_rng(int(s)))) for s in seeds])
    return seeds, l_true


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.4])
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--out", default=str(CF.QFILE))
    a = ap.parse_args()
    seeds, l_true = val_truth()
    n = len(seeds)
    try:
        d = json.loads(open(a.out).read())
        if abs(d.get("alpha", a.alpha) - a.alpha) > 1e-12:
            d = {}
    except (FileNotFoundError, json.JSONDecodeError):
        d = {}
    d.setdefault("q", {}); d.setdefault("info", {})
    d.update(alpha=a.alpha, n_cal=n, split="val", pseed_cal=CF.PSEED_CAL,
             note="q = ceil((n+1)(1-alpha))/n quantile of |l_true - m|/sigma over VAL carries; interval = m +- q*sigma (in l = ln mu_roll)")
    for rho in a.rhos:
        sc, cf = CF.val_scores(l_true, seeds, rho)
        q = CF.conformal_q(sc, a.alpha)
        key = f"{rho:g}"
        d["q"][key] = q
        # half-width in l at the two reported sigmas, vs the population range (ln 6e-3 - ln 5e-4 = 2.485)
        d["info"][key] = dict(frac_confident=float(cf.mean()), halfwidth_honest=q * 0.40, halfwidth_conf=q * 0.15,
                              width_honest_vs_pop=2 * q * 0.40 / np.log(E.MU_HI / E.MU_LO),
                              width_conf_vs_pop=2 * q * 0.15 / np.log(E.MU_HI / E.MU_LO))
        print(f"rho={rho:g}  n_cal={n}  confident share {cf.mean():.3f}  q={q:.3f}  half-width in l: honest-sigma {q * 0.40:.2f}, "
              f"conf-sigma {q * 0.15:.2f}  (population range width 2.485)", flush=True)
    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(json.dumps(d, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
