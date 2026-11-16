#!/usr/bin/env python
"""F2 step 1: validate the TRAINED value net against the analytic clearance
l(x)=h on states from real crossing episodes recorded by f2_harness.py
(--record). Sign/scale, correlation, false-safe rate near the boundary,
gradient-direction agreement, and anticipation lead time. SIMULATION ONLY.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import glob
import json
import sys
from pathlib import Path

import numpy as np

OUT_DIR = Path("/common/home/ss5772/.claude/jobs/f1580deb/tmp/f2_eval")
sys.path.insert(0, "/common/users/ss5772/bimanual/hj_reachability")


def auroc(score, pos):
    from scipy.stats import rankdata
    r = rankdata(score)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cos(a, b):
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    ok = (na > 1e-12) & (nb > 1e-12)
    c = np.full(len(a), np.nan)
    c[ok] = np.sum(a[ok] * b[ok], axis=1) / (na[ok] * nb[ok])
    return c


def main():
    from scipy.stats import spearmanr
    files = sorted(glob.glob(str(OUT_DIR / "states_*.npz")))
    res = {}
    allD = {k: [] for k in ("h", "V", "logit", "gV", "gh", "ql", "qr")}
    lead = []
    for f in files:
        d = np.load(f)
        ic = int(d["idx_cross"])
        sl = slice(ic, None)  # crossing phase only
        for k, kk in (("h", "h"), ("V", "V"), ("logit", "logit"), ("gV", "gV"), ("gh", "gh"),
                      ("ql", "q_left"), ("qr", "q_right")):
            allD[k].append(d[kk][sl])
        # anticipation: first unsafe (h<=0) vs first V>0 / V>-0.5 in the crossing phase
        h, V, t = d["h"][sl], d["V"][sl], d["t"][sl]
        tu = t[np.argmax(h <= 0)] if np.any(h <= 0) else None
        tv0 = t[np.argmax(V > 0)] if np.any(V > 0) else None
        tv5 = t[np.argmax(V > -0.5)] if np.any(V > -0.5) else None
        th5 = t[np.argmax(h < 0.05)] if np.any(h < 0.05) else None
        lead.append(dict(file=Path(f).name, t_first_h_le0=tu, t_first_V_gt0=tv0,
                         t_first_V_gt_m05=tv5, t_first_h_lt_5cm=th5,
                         h_min=float(h.min()), V_max=float(V.max())))
    D = {k: np.concatenate(v) for k, v in allD.items()}
    h, V, lg = D["h"], D["V"], D["logit"]
    unsafe = h <= 0
    near = np.abs(h) < 0.05
    res["n_states"] = int(len(h))
    res["n_unsafe"] = int(unsafe.sum())
    res["n_near_5cm"] = int(near.sum())
    res["pearson_V_vs_negh"] = float(np.corrcoef(V, -h)[0, 1])
    res["pearson_logit_vs_negh"] = float(np.corrcoef(lg, -h)[0, 1])
    res["spearman_V_vs_negh"] = float(spearmanr(V, -h).correlation)
    res["auroc_V_unsafe"] = auroc(V, unsafe)
    res["auroc_V_h_lt_5cm"] = auroc(V, h < 0.05)
    res["false_safe_rate(V<=0|h<=0)"] = float(np.mean(V[unsafe] <= 0)) if unsafe.any() else float("nan")
    res["false_safe_rate(V<=-0.5|h<=0)"] = float(np.mean(V[unsafe] <= -0.5)) if unsafe.any() else float("nan")
    res["near_acc_sign"] = float(np.mean((V[near] > 0) == (h[near] <= 0))) if near.any() else float("nan")
    res["false_alarm(V>0|h>0.05)"] = float(np.mean(V[h > 0.05] > 0))
    res["V_range"] = [float(V.min()), float(V.max())]
    # calibration: median V per h bin
    bins = [-0.1, -0.02, 0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0]
    cal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (h >= lo) & (h < hi)
        if m.any():
            cal.append(dict(h_bin=[lo, hi], n=int(m.sum()), V_med=float(np.median(V[m])),
                            V_p10=float(np.percentile(V[m], 10)), V_p90=float(np.percentile(V[m], 90)),
                            logit_med=float(np.median(lg[m]))))
    res["calibration"] = cal
    # gradients: trained value decreases where clearance increases => -gV ~ +gh
    gV, gh = D["gV"], D["gh"]
    close = h < 0.10
    cr = cos(-gV[:, 6:], gh[:, 6:])
    cf = cos(-gV, gh)
    for name, c in (("right_arm", cr), ("full12", cf)):
        cc = c[close & np.isfinite(c)]
        res[f"gradcos_{name}_h<10cm"] = dict(
            n=int(len(cc)), median=float(np.median(cc)) if len(cc) else float("nan"),
            mean=float(np.mean(cc)) if len(cc) else float("nan"),
            frac_pos=float(np.mean(cc > 0)) if len(cc) else float("nan"),
            frac_gt_0p5=float(np.mean(cc > 0.5)) if len(cc) else float("nan"))
    gnV = np.linalg.norm(gV[:, 6:], axis=1)
    res["gradnorm_V_right_pct(h<10cm)"] = [float(x) for x in np.percentile(gnV[close], [5, 50, 95])] if close.any() else []
    res["gradnorm_V_right_pct(h>20cm)"] = [float(x) for x in np.percentile(gnV[h > 0.2], [5, 50, 95])] if (h > 0.2).any() else []
    # oracle consistency: recorded sim h vs mujoco_fk oracle l(q_left, q_right)
    try:
        import mujoco_fk
        orc = mujoco_fk.get_oracle()
        idx = np.linspace(0, len(h) - 1, 300).astype(int)
        ho = np.array([orc.boundary_fn(D["ql"][i], D["qr"][i]) for i in idx])
        res["oracle_vs_sim_h_absdiff_max"] = float(np.max(np.abs(ho - h[idx])))
        res["oracle_vs_sim_h_absdiff_med"] = float(np.median(np.abs(ho - h[idx])))
    except Exception as e:  # noqa: BLE001
        res["oracle_check_err"] = str(e)[:200]
    res["lead"] = lead
    (OUT_DIR / "validation.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
