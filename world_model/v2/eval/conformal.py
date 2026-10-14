"""world_model/v2/eval/conformal.py -- threshold hooks for the spill shield.

  CalmConformalThreshold  split conformal on CALM steps (valid steps outside every
                          credit window, from calibration episodes): thr = the
                          ceil((n+1)(1-alpha))-th smallest calm score, so a fresh
                          exchangeable calm step alarms w.p. <= alpha (marginal).
  EventBudgetThreshold    the harness rule (<= k false alarms / min of calm motion).
  AdaptiveThreshold       online adaptive conformal (Gibbs & Candes 2021, ACI) with
                          DELAYED feedback: a step is only known to be calm once
                          `confirm_s` seconds pass without a spill, then
                          alpha_t <- alpha_t + gamma (alpha* - err). Tracks drift
                          (new contents / lighting) without retraining.
All operate on any scalar risk score (hazard at the alarm horizon, -Vmin, an
ensemble upper bound mean+k*std, ...).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from . import metrics as M


def calm_scores(scores, eps, credit_window_s=M.DEFAULT_CREDIT_WINDOW_S):
    out = []
    for s, ep in zip(scores, eps):
        s = np.asarray(s, float); T = M.episode_len(ep); on = M.episode_onset(ep)
        t0 = int(ep.get("t_start", 0)); hz = float(ep["hz"])
        end = T if on < 0 else max(t0, on - M.horizon_steps(credit_window_s, hz))
        out.append(s[t0:end])
    return np.concatenate(out) if out else np.zeros(0)


class CalmConformalThreshold:
    def __init__(self, alpha=0.01):
        self.alpha = alpha; self.thr = None

    def fit(self, scores, eps, credit_window_s=M.DEFAULT_CREDIT_WINDOW_S):
        c = np.sort(calm_scores(scores, eps, credit_window_s)); n = len(c)
        k = int(np.ceil((n + 1) * (1 - self.alpha)))
        self.thr = float(c[k - 1]) if k <= n else float("inf")
        return self.thr


class EventBudgetThreshold:
    def __init__(self, fa_per_min=M.DEFAULT_FA_PER_MIN, credit_window_s=M.DEFAULT_CREDIT_WINDOW_S):
        self.fa, self.W = fa_per_min, credit_window_s; self.thr = None

    def fit(self, scores, eps):
        self.thr = M.select_threshold(scores, eps, self.fa, self.W)
        return self.thr


class AdaptiveThreshold:
    def __init__(self, calib_calm_scores, alpha=0.01, gamma=0.005, hz=15.0, confirm_s=2.0, window=5000):
        self.target = alpha; self.alpha_t = alpha; self.gamma = gamma
        self.buf = deque(np.asarray(calib_calm_scores, float)[-window:], maxlen=window)
        self.pending = deque(); self.confirm = int(round(confirm_s * hz))

    @property
    def thr(self):
        a = float(np.clip(self.alpha_t, 1e-4, 0.5))
        return float(np.quantile(np.asarray(self.buf), 1 - a))

    def update(self, score: float, spilled_now: bool = False) -> float:
        """Call once per tick; returns the threshold to use at this tick."""
        thr = self.thr
        self.pending.append((score, score >= thr))
        if spilled_now:            # the recent steps were NOT calm: discard them
            self.pending.clear()
        while len(self.pending) > self.confirm:
            s, alarmed = self.pending.popleft()
            self.buf.append(s)
            self.alpha_t += self.gamma * (self.target - float(alarmed))
        return thr
