"""world_model/v2/eval/harness.py -- one evaluation for ANY spill predictor.

Predictor contract (duck-typed; see `Predictor`):
    name: str
    horizons_s: tuple[float, ...] | None   # columns of predict()'s output; None = 1 score
    probabilistic: bool                    # True -> calibration is reported
    predict(episode: dict) -> np.ndarray   # (T,) or (T, len(horizons_s)); CAUSAL:
                                           # row t may only use obs/actions <= t
                                           # (plan-aware predictors document it)

Episode contract (the minimum the harness reads):
    hz (float), spill_count (T,) and/or onset (int, -1 = never), optional
    t_start (int), optional T, optional meta (dict, for OOD split selection).

    report = evaluate(predictor, val_eps, {"test": test_eps, "ood_light": ood})
    print(format_report(report))

The event threshold is chosen ONCE on `val_eps` (<= fa_per_min false alarms per
minute of calm motion) and frozen for every test/OOD set.
"""
from __future__ import annotations

import json
import time
from typing import Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

from . import metrics as M


class Predictor(Protocol):
    name: str
    horizons_s: tuple | None
    probabilistic: bool

    def predict(self, episode: dict) -> np.ndarray: ...


def _alarm_col(pred_h, eval_h, alarm_horizon_s):
    """Column of the eval-horizon matrix used as the event (alarm) score."""
    target = alarm_horizon_s if alarm_horizon_s is not None else max(eval_h)
    return int(np.argmin([abs(h - target) for h in eval_h]))


def _align(pred: np.ndarray, pred_h, eval_h) -> np.ndarray:
    """Map predictor columns to eval horizons (nearest predictor horizon)."""
    p = np.asarray(pred, float)
    if p.ndim == 1 or pred_h is None:
        return M.as_matrix(p if p.ndim == 1 else p[:, 0], len(eval_h))
    idx = [int(np.argmin([abs(ph - h) for ph in pred_h])) for h in eval_h]
    return p[:, idx]


def run_predictor(predictor, eps: Sequence[dict], eval_h) -> tuple[list, float]:
    t0 = time.perf_counter(); out = []; nsteps = 0
    for ep in eps:
        p = _align(predictor.predict(ep), getattr(predictor, "horizons_s", None), eval_h)
        T = M.episode_len(ep)
        if len(p) != T:
            raise ValueError(f"{predictor.name}: predict() returned {len(p)} rows for T={T}")
        out.append(p); nsteps += T
    return out, (time.perf_counter() - t0) / max(nsteps, 1)


def score_set(preds, eps, thr, eval_h, alarm_col, credit_window_s, probabilistic) -> dict:
    res = dict(n_eps=len(eps), n_spill_eps=int(sum(M.episode_onset(e) >= 0 for e in eps)))
    per_h = {}
    for j, H in enumerate(eval_h):
        s, y, (sd, yd) = M.stack_valid(preds, eps, H, j, len(eval_h))
        d = dict(n_valid=int(len(y)), n_pos=int(y.sum()), base_rate=float(y.mean()) if len(y) else float("nan"),
                 auroc_pre_onset=M.auroc(s, y), auprc_pre_onset=M.auprc(s, y),
                 detect_auroc_INCLUDES_POST_ONSET=M.auroc(sd, yd))
        if probabilistic:
            d["calibration"] = M.calibration(s, y)
        per_h[f"{H:g}s"] = d
    res["per_horizon"] = per_h
    tick = min(1.0 / float(e["hz"]) for e in eps) if eps else None
    ev = M.event_stats([p[:, alarm_col] for p in preds], eps, thr, credit_window_s)
    res["event"] = ev.summary(tick_s=tick)
    return res


def evaluate(predictor, val_eps: Sequence[dict], test_sets: Mapping[str, Sequence[dict]],
             horizons_s=M.DEFAULT_HORIZONS_S, fa_per_min: float = M.DEFAULT_FA_PER_MIN,
             credit_window_s: float = M.DEFAULT_CREDIT_WINDOW_S, alarm_horizon_s: float | None = None,
             thr: float | None = None) -> dict:
    eval_h = tuple(float(h) for h in horizons_s)
    col = _alarm_col(getattr(predictor, "horizons_s", None), eval_h, alarm_horizon_s)
    prob = bool(getattr(predictor, "probabilistic", False))
    val_preds, sec_val = run_predictor(predictor, val_eps, eval_h)
    if thr is None:
        thr = M.select_threshold([p[:, col] for p in val_preds], val_eps, fa_per_min, credit_window_s)
    rep = dict(predictor=predictor.name, horizons_s=list(eval_h), alarm_horizon_s=eval_h[col],
               fa_budget_per_min=fa_per_min, credit_window_s=credit_window_s, threshold=float(thr),
               threshold_chosen_on="val", sets={})
    rep["sets"]["val"] = score_set(val_preds, val_eps, thr, eval_h, col, credit_window_s, prob)
    secs = [sec_val]
    for name, eps in test_sets.items():
        preds, sec = run_predictor(predictor, eps, eval_h); secs.append(sec)
        rep["sets"][name] = score_set(preds, eps, thr, eval_h, col, credit_window_s, prob)
    rep["offline_sec_per_step"] = float(np.mean(secs))
    return rep


# --------------------------------------------------------------------------
# OOD split helpers
# --------------------------------------------------------------------------
def meta_get(ep: dict, dotted: str, default=None):
    cur = ep.get("meta", {})
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def split_by(eps: Iterable[dict], pred: Callable[[dict], bool]):
    a, b = [], []
    for e in eps:
        (a if pred(e) else b).append(e)
    return a, b


def split_by_meta_quantile(eps, dotted: str, q: float = 0.8):
    """In-dist = meta value <= q-quantile; OOD = above (e.g. held-out contents/lighting)."""
    vals = np.array([float(meta_get(e, dotted, np.nan)) for e in eps])
    cut = np.nanquantile(vals, q)
    return split_by(eps, lambda e: float(meta_get(e, dotted, np.nan)) <= cut)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def format_report(rep: dict) -> str:
    L = [f"== {rep['predictor']}  thr={rep['threshold']:.4g} (val, <= {rep['fa_budget_per_min']:g} FA/min calm)"
         f"  alarm-H={rep['alarm_horizon_s']:g}s  credit={rep['credit_window_s']:g}s"]
    for sname, s in rep["sets"].items():
        e = s["event"]
        L.append(f"  [{sname}] eps={s['n_eps']} spill_eps={s['n_spill_eps']}")
        for h, d in s["per_horizon"].items():
            cal = d.get("calibration")
            cs = f"  ECE={cal['ece']:.3f} Brier={cal['brier']:.4f}" if cal else ""
            L.append(f"    H={h:>5s} pre-onset AUROC={d['auroc_pre_onset']:.3f} AUPRC={d['auprc_pre_onset']:.3f}"
                     f" (base {d['base_rate']:.3f}, n={d['n_valid']})"
                     f" | detect-AUROC(post-onset incl., NOT prediction)={d['detect_auroc_INCLUDES_POST_ONSET']:.3f}{cs}")
        L.append(f"    event: det={e['detection_rate']:.2f} lead med={e['lead_med_s']:.3f}s p10={e['lead_p10_s']:.3f}s"
                 f" p25={e['lead_p25_s']:.3f}s | det@>=0.25s={e['det@lead>=0.25s']:.2f} det@>=0.5s={e['det@lead>=0.5s']:.2f}"
                 f" (chance {e['chance_detection_rate']:.2f}, lift {e['alarm_lift_window_vs_calm']:.1f}x)"
                 f" | FA/min={e['fa_per_min']:.2f} (calm {e['calm_min']:.1f} min) duty={e['calm_alarm_frac']:.3f}"
                 f" negEpFA={e['neg_ep_fa_frac']:.2f}")
    L.append(f"  offline sec/step (batched predict, not the streaming latency) = {rep['offline_sec_per_step']*1e3:.3f} ms")
    return "\n".join(L)


def save_report(rep: dict, path) -> None:
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(rep, f, indent=1, default=conv)
