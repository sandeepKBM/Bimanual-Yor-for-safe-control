"""bridge/metrics.py -- per-episode records -> summary numbers. Pure numpy. SIMULATION / SAVED DATA ONLY.

One record per carry (a dict, one JSON line):
  ep, seed, L (nominal ticks), onset_stored (first tick with l<0 in the stored nominal data, -1 = none),
  dropped, drop_tick, tunnel, n_ticks (ticks run), n_interv (ticks with u != u_nom),
  dev_raw_sum (sum of |u-u_nom|, 6-vector, mixed units), dev_norm_sum (same, a/a_max and w/w_max),
  completion (fraction of the nominal path done before a drop or the end), filt_s, sim_s, x_* (filter extras),
  and, since the path-progress metric (optional: old files lack them): path_len_nom [m], path_len_done [m], progress (in [0,1]),
  goal_tick (first tick within 2 cm of the end of the nominal path, -1 = never), goal_reached.
"""
from __future__ import annotations

import numpy as np

GOAL_TOL = 0.02          # [m] the tray origin is "at the goal" when its xy is this close to the end of the nominal path


class PathProgress:
    """Path progress of the carried tray. NEW (boed work): the old `completion` is only time survived.
    nominal path = the polyline  start -> start + D1 -> start + D1 + D2 ...  (xy; the planned transports of the carry).
    progress     = min(1, arc length the tray origin has actually travelled / nominal arc length). The carry paths are straight
                   segments and the tray never backtracks, so travelled arc length = distance covered along the path.
    goal_tick    = first tick at which the tray origin is within GOAL_TOL of the END of the nominal path (-1 = never).
    The authority-limited operator clips its acceleration, so a plan that is too fast covers LESS than its nominal path in the
    time it has: progress < 1 at the end of the episode is normal for the stored (intensity 3) carries."""

    def __init__(self, start_xy, seg_vectors, tol=GOAL_TOL):
        self.start = np.asarray(start_xy, float)[:2].copy()
        D = [np.asarray(d, float)[:2] for d in seg_vectors]
        self.len_nom = float(sum(np.linalg.norm(d) for d in D))
        self.goal = self.start + (np.sum(D, axis=0) if D else np.zeros(2))
        self.tol, self.last, self.done, self.goal_tick = float(tol), self.start.copy(), 0.0, -1

    def update(self, pos_xy, tick):
        p = np.asarray(pos_xy, float)[:2]
        self.done += float(np.linalg.norm(p - self.last))
        self.last = p.copy()
        if self.goal_tick < 0 and float(np.linalg.norm(p - self.goal)) <= self.tol:
            self.goal_tick = int(tick)

    def record(self):
        return dict(path_len_nom=self.len_nom, path_len_done=float(self.done),
                    progress=float(min(1.0, self.done / self.len_nom)) if self.len_nom > 0 else 1.0,
                    goal_tick=int(self.goal_tick), goal_reached=bool(self.goal_tick >= 0))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def _rate(k, n):
    return dict(k=int(k), n=int(n), rate=(float(k) / n if n else float("nan")), ci95=wilson(int(k), int(n)))


def summarize(recs, nom_drop=None):
    """recs: list of record dicts. nom_drop: optional bool array (same order) = the nominal carry drops the ball.
    Default = the stored nominal data (onset_stored >= 0)."""
    n = len(recs)
    if n == 0:
        return dict(n=0)
    g = lambda k: np.array([r[k] for r in recs])
    dropped = g("dropped").astype(bool)
    ticks = max(float(g("n_ticks").sum()), 1.0)
    if nom_drop is None:
        nom_drop = g("onset_stored") >= 0
    nom_drop = np.asarray(nom_drop, bool)
    out = dict(n=n, drop=_rate(dropped.sum(), n),
               interv_rate=float(g("n_interv").sum() / ticks),
               mean_dev_raw=float(g("dev_raw_sum").sum() / ticks),
               mean_dev_norm=float(g("dev_norm_sum").sum() / ticks),
               completion=float(g("completion").mean()),
               drop_given_nom_drop=_rate(dropped[nom_drop].sum(), nom_drop.sum()),     # does the filter SAVE them?
               drop_given_nom_ok=_rate(dropped[~nom_drop].sum(), (~nom_drop).sum()),   # does it CAUSE drops?
               saved=int((nom_drop & ~dropped).sum()), caused=int((~nom_drop & dropped).sum()),
               n_tunnel=int(g("tunnel").sum()),
               filt_s_per_ep=float(g("filt_s").mean()), sim_s_per_ep=float(g("sim_s").mean()))
    if all("progress" in r for r in recs):                 # records written after the path-progress metric was added
        prog = g("progress").astype(float)
        reached = g("goal_reached").astype(bool)
        gt = g("goal_tick").astype(float)
        out.update(progress=float(prog.mean()), progress_dropped=float(prog[dropped].mean()) if dropped.any() else float("nan"),
                   progress_ok=float(prog[~dropped].mean()) if (~dropped).any() else float("nan"),
                   goal=_rate(reached.sum(), n),
                   goal_time_s=float(gt[reached].mean() * 0.02) if reached.any() else float("nan"))
    return out
