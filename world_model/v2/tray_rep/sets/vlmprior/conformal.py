"""sets/vlmprior/conformal.py -- split-conformal quantile of the prior's normalised error, from VAL carries only. Pure numpy.
Score of a carry:  s = |l_true - m| / sigma   (sigma = the REPORTED sigma of the draw: 0.40 honest, 0.15 confident).
q = the ceil((n+1)(1-alpha))/n empirical quantile of the n calibration scores (finite-sample split-conformal; inf if the rank exceeds n).
The prior interval for a new carry is [m - q sigma, m + q sigma].
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep.sets.vlmprior import prior as PR

QFILE = K.ROOT / "vlmprior" / "conformal_q.json"
PSEED_CAL = 7_000_001            # calibration draws use their own prior_seed (evaluation uses 0, 1, 2 ...)


def conformal_q(scores, alpha=0.10):
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    k = math.ceil((n + 1) * (1.0 - alpha))
    return float("inf") if k > n else float(s[k - 1])


def val_scores(l_true, seeds, rho, pseed=PSEED_CAL):
    m, sg, cf = PR.draw_many(l_true, seeds, rho, pseed)
    return np.abs(np.asarray(l_true) - m) / sg, cf


def load_q(rho, path=QFILE):
    d = json.loads(Path(path).read_text())
    key = f"{float(rho):g}"
    if key not in d["q"]:
        raise KeyError(f"conformal q for rho={key} not in {path} (have {sorted(d['q'])}); run sets/vlmprior/calibrate.py --rhos ...")
    return float(d["q"][key])
