"""world_model/v2/eval/metrics.py -- PREDICTION-framed spill metrics.

Everything here is pure numpy (no sklearn) and predictor-agnostic: it consumes
per-step scores (higher = more spill risk) plus the episode's spill ONSET.

Why a new module (audit 2026-09-24): the v1 monitor head reported 0.999 AUROC,
but that label was the per-step CUMULATIVE spill flag, so every post-onset step
counted as a positive -> it measured DETECTION of an already-spilled cup. At
0.2 s ahead the same head scored 0.891 and its median lead was 65 ms (< 1 vision
tick). The definitions below make detection-vs-prediction impossible to confuse.

DEFINITIONS (per episode; hz = the episode's step rate; t = step index)
  t_start     first step eligible for scoring (e.g. motion start / ball release);
              steps before it are never scored.
  onset       first step t >= t_start with spill_count[t] > spill_count[t_start]
              ("first rise above the settled baseline"), -1 if never. A
              provided ep["onset"] takes precedence.
  valid(t)    t_start <= t < onset (or < T if never spilled). POST-ONSET STEPS ARE
              EXCLUDED from every step-level metric: the event has happened, a
              high score there is detection, not prediction.
  y_H(t)      1 iff t < onset <= t + H_steps  ("spill within horizon H"),
              H_steps = round(H_s * hz).
  (a) pre-onset AUROC / AUPRC at H: ranking of score over valid steps vs y_H.
  (b) event level at threshold thr (alarm = score >= thr):
      credit window  [onset - W, onset), W = round(credit_window_s * hz). An
                     alarm in it is a PREDICTION; lead = (onset - t_first)/hz,
                     t_first = first alarmed step in the window. Lead >= 1 step
                     by construction (onset itself is excluded).
      calm region    valid steps that are NOT in any credit window (all valid
                     steps of never-spill episodes). Rising edges of the alarm
                     inside calm = FALSE ALARMS; fa_per_min = #FA / calm minutes.
                     An alarm already ON at the start of the calm region counts
                     as an edge. calm_alarm_frac = duty cycle of the alarm in calm.
      threshold      chosen on VALIDATION episodes as the lowest threshold whose
                     fa_per_min <= budget (default 1 FA / min of calm motion),
                     then FROZEN and applied to test / OOD sets.
      reported       detection rate, lead median/p10/p25/p75, detection rate at
                     lead >= {1 tick, 0.25, 0.5, 1.0} s, realised test FA/min,
                     calm_alarm_frac, fraction of never-spill episodes with any FA.
  (c) OOD: same metrics on held-out splits with the in-distribution threshold.
  (d) calibration (probabilistic predictors only): per-H ECE (10 equal-width
      bins, count-weighted), Brier, reliability table, mean-pred vs base rate,
      all over VALID steps.
  Contrast number (clearly labelled): detect_auroc_INCLUDES_POST_ONSET uses the
  old cumulative label over ALL steps >= t_start; shown only so the gap to (a)
  is visible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

DEFAULT_HORIZONS_S = (0.25, 0.5, 1.0)
DEFAULT_CREDIT_WINDOW_S = 2.0
DEFAULT_FA_PER_MIN = 1.0


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------
def onset_from_counts(spill_count, t_start: int = 0) -> int:
    c = np.asarray(spill_count)
    if len(c) == 0 or t_start >= len(c):
        return -1
    idx = np.nonzero(c[t_start:] > c[t_start])[0]
    return int(idx[0] + t_start) if len(idx) else -1


def episode_onset(ep: dict) -> int:
    if ep.get("onset") is not None:
        return int(ep["onset"])
    return onset_from_counts(ep["spill_count"], int(ep.get("t_start", 0)))


def episode_len(ep: dict) -> int:
    if ep.get("T") is not None:
        return int(ep["T"])
    return len(ep["spill_count"])


def horizon_steps(H_s: float, hz: float) -> int:
    return max(1, int(round(H_s * hz)))


def labels(onset: int, T: int, t_start: int, H_steps: int):
    """(y, valid) boolean arrays of length T -- see module docstring."""
    t = np.arange(T)
    on = onset if onset >= 0 else 10**12
    valid = (t >= t_start) & (t < min(on, T))
    y = (t < on) & (on <= t + H_steps) & valid
    return y, valid


# --------------------------------------------------------------------------
# ranking metrics (tie-aware)
# --------------------------------------------------------------------------
def auroc(score, y) -> float:
    s = np.asarray(score, float); y = np.asarray(y, bool)
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    # average rank of each unique value
    cum = np.cumsum(cnt); avg_rank = cum - (cnt - 1) / 2.0
    r = avg_rank[inv]
    return float((r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def auprc(score, y) -> float:
    """Average precision (step function, ties grouped)."""
    s = np.asarray(score, float); y = np.asarray(y, bool)
    npos = int(y.sum())
    if npos == 0 or npos == len(y):
        return float("nan")
    o = np.argsort(-s, kind="mergesort"); s, y = s[o], y[o]
    tp = np.cumsum(y); fp = np.cumsum(~y)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]       # end of each tie group
    tp, fp = tp[last], fp[last]
    prec = tp / (tp + fp); rec = tp / npos
    drec = np.diff(np.r_[0.0, rec])
    return float((prec * drec).sum())


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
def calibration(p, y, bins: int = 10) -> dict:
    p = np.clip(np.asarray(p, float), 0.0, 1.0); y = np.asarray(y, float)
    if len(p) == 0:
        return dict(ece=float("nan"), brier=float("nan"), table=[])
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0; table = []
    for i in range(bins):
        m = b == i
        if not m.any():
            continue
        conf, acc, n = float(p[m].mean()), float(y[m].mean()), int(m.sum())
        ece += n / len(p) * abs(conf - acc)
        table.append(dict(lo=float(edges[i]), hi=float(edges[i + 1]), n=n, mean_pred=conf, frac_pos=acc))
    return dict(ece=float(ece), brier=float(np.mean((p - y) ** 2)),
                mean_pred=float(p.mean()), base_rate=float(y.mean()), table=table)


# --------------------------------------------------------------------------
# event-level
# --------------------------------------------------------------------------
@dataclass
class EventStats:
    thr: float
    n_pos_eps: int = 0
    n_neg_eps: int = 0
    n_detected: int = 0
    leads_s: list = field(default_factory=list)          # detected positives only
    n_fa: int = 0
    calm_s: float = 0.0
    calm_alarm_steps: int = 0
    calm_steps: int = 0
    neg_eps_with_fa: int = 0
    win_alarm_steps: int = 0
    win_steps: int = 0
    win_s_sum: float = 0.0

    @property
    def fa_per_min(self) -> float:
        return self.n_fa / max(self.calm_s / 60.0, 1e-9)

    def summary(self, tick_s: float | None = None) -> dict:
        L = np.asarray(self.leads_s, float)
        pct = (lambda q: float(np.percentile(L, q)) if len(L) else float("nan"))
        out = dict(
            thr=float(self.thr), n_pos_eps=self.n_pos_eps, n_neg_eps=self.n_neg_eps,
            detection_rate=self.n_detected / max(self.n_pos_eps, 1) if self.n_pos_eps else float("nan"),
            lead_med_s=pct(50), lead_p10_s=pct(10), lead_p25_s=pct(25), lead_p75_s=pct(75),
            fa_per_min=float(self.fa_per_min), n_fa=self.n_fa, calm_min=self.calm_s / 60.0,
            calm_alarm_frac=self.calm_alarm_steps / max(self.calm_steps, 1),
            win_alarm_frac=self.win_alarm_steps / max(self.win_steps, 1),
            neg_ep_fa_frac=self.neg_eps_with_fa / max(self.n_neg_eps, 1) if self.n_neg_eps else float("nan"),
        )
        # CHANCE: an alarm process with the calm duty cycle + calm FA rate, placed
        # independently of the spill, hits a credit window of mean length Wbar
        # with prob 1-(1-duty)exp(-rate*Wbar). Detection near this is NOT prediction.
        duty = out["calm_alarm_frac"]; rate_s = self.n_fa / max(self.calm_s, 1e-9)
        wbar = self.win_s_sum / max(self.n_pos_eps, 1)
        out["chance_detection_rate"] = float(1 - (1 - duty) * np.exp(-rate_s * wbar)) if self.n_pos_eps else float("nan")
        out["alarm_lift_window_vs_calm"] = float(out["win_alarm_frac"] / duty) if duty > 0 else float("inf") if out["win_alarm_frac"] > 0 else float("nan")
        for x in ([tick_s] if tick_s else []) + [0.25, 0.5, 1.0]:
            key = "det@lead>=tick" if (tick_s and x == tick_s) else f"det@lead>={x:g}s"
            out[key] = float((L >= x - 1e-9).sum() / max(self.n_pos_eps, 1)) if self.n_pos_eps else float("nan")
        return out


def _rising_edges(above: np.ndarray) -> np.ndarray:
    prev = np.r_[False, above[:-1]]
    return np.nonzero(above & ~prev)[0]


def event_stats(scores: Sequence[np.ndarray], eps: Sequence[dict], thr: float,
                credit_window_s: float = DEFAULT_CREDIT_WINDOW_S) -> EventStats:
    st = EventStats(thr=float(thr))
    for s, ep in zip(scores, eps):
        s = np.asarray(s, float); T = episode_len(ep); hz = float(ep["hz"])
        t0 = int(ep.get("t_start", 0)); onset = episode_onset(ep)
        above = s >= thr
        if onset >= 0:
            if onset <= t0:            # spilled before scoring starts: unusable
                continue
            W = horizon_steps(credit_window_s, hz)
            w0 = max(t0, onset - W)
            hit = np.nonzero(above[w0:onset])[0]
            st.n_pos_eps += 1
            st.win_alarm_steps += int(above[w0:onset].sum()); st.win_steps += onset - w0
            st.win_s_sum += (onset - w0) / hz
            if len(hit):
                st.n_detected += 1
                st.leads_s.append((onset - (w0 + hit[0])) / hz)
            calm = slice(t0, w0)
        else:
            st.n_neg_eps += 1
            calm = slice(t0, T)
        a = above[calm]
        if len(a):
            nfa = len(_rising_edges(a))
            st.n_fa += nfa
            st.calm_alarm_steps += int(a.sum())
            st.calm_steps += len(a)
            st.calm_s += len(a) / hz
            if onset < 0 and nfa > 0:
                st.neg_eps_with_fa += 1
    return st


def candidate_thresholds(scores: Sequence[np.ndarray], eps: Sequence[dict], n: int = 400) -> np.ndarray:
    vals = []
    for s, ep in zip(scores, eps):
        T = episode_len(ep); onset = episode_onset(ep)
        _, valid = labels(onset, T, int(ep.get("t_start", 0)), 1)
        vals.append(np.asarray(s, float)[valid])
    v = np.concatenate(vals) if vals else np.zeros(1)
    v = v[np.isfinite(v)]
    q = np.unique(np.quantile(v, np.linspace(0, 1, n))) if len(v) else np.zeros(1)
    return np.r_[q, (q.max() + 1e-6) if len(q) else 1.0]


def select_threshold(scores, eps, fa_per_min: float = DEFAULT_FA_PER_MIN,
                     credit_window_s: float = DEFAULT_CREDIT_WINDOW_S, n: int = 400) -> float:
    """Lowest threshold whose realised FA/min on (validation) eps is <= budget."""
    best = None
    for thr in candidate_thresholds(scores, eps, n):
        if event_stats(scores, eps, thr, credit_window_s).fa_per_min <= fa_per_min:
            best = thr if best is None else min(best, thr)
    return float(best if best is not None else np.inf)


# --------------------------------------------------------------------------
# step-level collection
# --------------------------------------------------------------------------
def as_matrix(pred, n_h: int) -> np.ndarray:
    """(T,) or (T,k) -> (T,n_h). 1-D scores are broadcast to every horizon."""
    p = np.asarray(pred, float)
    if p.ndim == 1:
        return np.repeat(p[:, None], n_h, 1)
    if p.shape[1] != n_h:
        raise ValueError(f"prediction has {p.shape[1]} columns, expected {n_h}")
    return p


def stack_valid(preds: Sequence[np.ndarray], eps: Sequence[dict], H_s: float, col: int, n_h: int):
    S, Y, D = [], [], []
    for p, ep in zip(preds, eps):
        T = episode_len(ep); onset = episode_onset(ep); t0 = int(ep.get("t_start", 0))
        y, valid = labels(onset, T, t0, horizon_steps(H_s, float(ep["hz"])))
        P = as_matrix(p, n_h)[:T, col]
        S.append(P[valid]); Y.append(y[valid])
        # detection-framing contrast: cumulative label over ALL steps >= t_start
        t = np.arange(T); m = t >= t0
        D.append((P[m], (onset >= 0) & (t[m] >= onset)))
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)
    return cat(S), cat(Y).astype(bool), (cat([d[0] for d in D]), cat([d[1] for d in D]).astype(bool))
