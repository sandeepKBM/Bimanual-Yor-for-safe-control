#!/usr/bin/env python
"""evaluate.py -- shared metrics for per-step spill scores (E,T).

  causal     : score(t) uses kinematics up to t only (monitor setting)
  plan-aware : max_{tau in [t, t+H]} score(tau) -- what an MPC constraint sees
               when it rolls the model over the KNOWN planned cup trajectory
  episode    : max score over the valid window (pre-onset) -> does enforcing
               score <= thr along a trajectory separate spill / no-spill?
  lead/recall: thr set on TRAIN negatives at a 10 % episode false-alarm rate;
               lead = onset - first crossing (s), over detected test positives
  calibration: Platt (1-D logistic) fit on TRAIN valid steps -> Brier, ECE on test
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d

import slosh_models as sm
from data_prep import labels

DT = sm.DT


def plan_aware(score, H):
    return maximum_filter1d(score, H + 1, axis=-1, origin=-(H // 2), mode="nearest")


def episode_scores(score, valid):
    s = np.where(valid, score, -np.inf)
    return s.max(-1)


def evaluate(score, D, H=50, fpr=0.10, sub=2, platt_H=50, mask=None):
    """score (E,T) over ALL episodes; D from data_prep.load_rig. Returns dict of
    test metrics (+ thresholds fitted on train). mask (E,) bool restricts the
    TEST episodes scored (thresholds / Platt always fit on all train)."""
    tr, te = ~D["test"], D["test"].copy()
    if mask is not None:
        te &= mask
    y, valid = labels(D["onset"], D["k0"], D["T"], H)
    out = {}
    st = score[te][:, ::sub]; yt = y[te][:, ::sub]; vt = valid[te][:, ::sub]
    out["auroc"] = sm.auroc(st[vt], yt[vt]); out["auprc"] = sm.auprc(st[vt], yt[vt])
    pa = plan_aware(score, H)
    pat = pa[te][:, ::sub]
    out["auroc_plan"] = sm.auroc(pat[vt], yt[vt]); out["auprc_plan"] = sm.auprc(pat[vt], yt[vt])
    out["base_rate"] = float(yt[vt].mean())
    # episode level
    es = episode_scores(score, valid); pos = D["onset"] >= 0
    out["ep_auroc"] = sm.auroc(es[te], pos[te]); out["ep_auprc"] = sm.auprc(es[te], pos[te])
    thr = float(np.quantile(es[tr & ~pos], 1 - fpr))
    out["thr"] = thr
    det, leads = 0, []
    for i in np.where(te & pos)[0]:
        on = D["onset"][i]; w = np.where(score[i, D["k0"][i]:on] >= thr)[0]
        if len(w):
            det += 1; leads.append((on - (w[0] + D["k0"][i])) * DT)
    out["recall@10fa"] = det / max((te & pos).sum(), 1)
    out["fa_test"] = float(np.mean(es[te & ~pos] >= thr))
    out["lead_med_s"] = float(np.median(leads)) if leads else float("nan")
    out["lead_p10_s"] = float(np.percentile(leads, 10)) if leads else float("nan")
    # calibration (Platt on train steps)
    yP, vP = labels(D["onset"], D["k0"], D["T"], platt_H)
    s_tr = score[tr][:, ::sub * 2]; m_tr = vP[tr][:, ::sub * 2]; y_tr = yP[tr][:, ::sub * 2]
    cal = sm.platt_fit(s_tr[m_tr], y_tr[m_tr])
    p = cal(st[vt]); yy = yt[vt].astype(float)
    out["brier"] = float(np.mean((p - yy) ** 2)); out["ece"] = sm.ece(p, yy)
    return out


def fmt_row(name, m):
    return (f"| {name} | {m['auroc']:.3f} | {m['auprc']:.3f} | {m['auroc_plan']:.3f} | {m['auprc_plan']:.3f} | "
            f"{m['ep_auroc']:.3f} | {m['ep_auprc']:.3f} | {m['recall@10fa']:.2f} | {m['fa_test']:.2f} | "
            f"{m['lead_med_s']:.2f} | {m['ece']:.3f} |")


HEADER = ("| model | AUROC causal | AUPRC causal | AUROC plan | AUPRC plan | ep-AUROC | ep-AUPRC | "
          "recall@10%FA | FA(test) | lead med (s) | ECE |\n|---|---|---|---|---|---|---|---|---|---|---|")
