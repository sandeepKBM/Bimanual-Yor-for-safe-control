"""boed/policy.py -- the predicted-drop table (drop rate by carry intensity and mu_roll) and the intensity choice rules.
SIMULATION / SAVED DATA ONLY. Pure numpy.

Intensity grid (speed factor relative to the fastest transport the arm can do, per carry): I_rel in IGRID. I_rel = 1 runs every
transport at the arm's acceleration clip (the plan's peak acceleration = 5 * a_scale = 1.5 m/s^2); I_rel < 1 divides the speed
(the peak acceleration falls as I_rel^2, the time grows as 1/I_rel). Every plan with I_rel <= 1 is feasible: the tray reaches the goal.
The stored carry (bridge, P2) is NOT on this grid: it is a plan about 7x more aggressive than the clip, so the tray covers only
about 30% of the path (the bridge path-progress metric).

Table: drop[i, j] = probability that the F3-filtered carry at IGRID[i] drops the ball when mu_roll = NODES[j]. Built on VAL with MuJoCo (table.py).
The raw counts are smoothed by a monotone fit along mu_roll (drop falls as mu_roll rises), weighted by the cell sizes, because cells
with a few dozen carries cannot resolve a 2% rate. Along speed the fit is left free (a slow carry is exposed to the tilt noise for longer).

Rules (eps = target drop rate, default 2%). "fastest" = the largest I_rel in the grid.
  choose(table, w)   fastest I_rel with  sum_g w_g drop(I_rel, mu_g) <= eps ; the slowest I_rel if none meets it.
    P1 conservative : w = all mass on the lowest mu (the worst case in the prior)
    P3 / P4         : w = the belief after the probe
    P5 oracle       : w = all mass on the TRUE mu
    P6 population   : w = the prior (log-uniform)      [extra arm: the best single intensity without any information]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep.boed import model as MD

IGRID = (1.0, 0.75, 0.55, 0.40)
N_NODES = 8
NODES = np.exp(np.linspace(np.log(E.MU_LO), np.log(E.MU_HI), N_NODES))
EPS = 0.02


def pav(y, w, increasing=True):
    """weighted pool-adjacent-violators on a 1-D array."""
    y = np.asarray(y, float).copy(); w = np.asarray(w, float).copy() + 1e-9
    if not increasing:
        return pav(y[::-1], w[::-1], True)[::-1]
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            tw = wts[-2] + wts[-1]
            vals[-2:] = [(vals[-2] * wts[-2] + vals[-1] * wts[-1]) / tw]
            wts[-2:] = [tw]
            cnt[-2:] = [cnt[-2] + cnt[-1]]
    return np.repeat(vals, cnt)


def smooth_table(k, n, monotone_I=False, iters=6):
    """monotone fit of drop = k / n over (I rows, mu columns): decreasing along mu (axis 1, always). Along I (axis 0, speed) it is
    NOT forced to increase unless asked: a slower carry is exposed to the tilt noise for longer, and the exploration runs showed drop
    rates that do not fall with speed for small mu_roll."""
    p = np.where(n > 0, k / np.maximum(n, 1), 0.0)
    q = p.copy()
    for _ in range(iters if monotone_I else 1):
        if monotone_I:
            for j in range(q.shape[1]):
                q[:, j] = pav(q[:, j], n[:, j], True)
        for i in range(q.shape[0]):
            q[i, :] = pav(q[i, :], n[i, :], False)
    return q


class Table:
    def __init__(self, igrid, nodes, k, n, extra=None, meta=None):
        self.igrid = np.asarray(igrid, float)
        self.nodes = np.asarray(nodes, float)
        self.k, self.n = np.asarray(k, float), np.asarray(n, float)
        self.raw = np.where(self.n > 0, self.k / np.maximum(self.n, 1), np.nan)
        self.drop = smooth_table(self.k, self.n)
        self.extra, self.meta = extra or {}, meta or {}
        order = np.argsort(self.igrid)[::-1]                          # fastest first
        assert np.all(order == np.arange(len(self.igrid))), "igrid must be sorted fastest (largest) first"

    # ---- drop per I over the 40-point belief grid (log-linear interpolation in mu)
    def drop_on_grid(self):
        return np.stack([np.interp(MD.LOG_MU, np.log(self.nodes), self.drop[i]) for i in range(len(self.igrid))])      # (nI, 40)

    def drop_at(self, mu):
        return np.array([np.interp(np.log(mu), np.log(self.nodes), self.drop[i]) for i in range(len(self.igrid))])

    def expected(self, w, frac=1.0):
        """expected drop probability of each intensity under weights w over the belief grid. frac < 1: a carry that still has `frac` of
        its path to run; constant hazard along the path, 1 - (1 - d)^frac."""
        d = self.drop_on_grid() @ np.asarray(w, float)
        return 1.0 - (1.0 - d) ** frac if frac < 1.0 else d

    def choose(self, w, eps=EPS, frac=1.0):
        """index of the fastest intensity with expected drop <= eps (the slowest if none)."""
        e = self.expected(w, frac)
        ok = np.flatnonzero(e <= eps)
        return int(ok[0]) if len(ok) else len(self.igrid) - 1

    def to_json(self):
        return dict(igrid=self.igrid.tolist(), nodes=self.nodes.tolist(), k=self.k.tolist(), n=self.n.tolist(), drop_smooth=self.drop.tolist(),
                    drop_raw=np.where(np.isnan(self.raw), None, self.raw).tolist(), extra=self.extra, meta=self.meta)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_json(), indent=1, default=float))

    @staticmethod
    def load(path):
        d = json.loads(Path(path).read_text())
        return Table(d["igrid"], d["nodes"], d["k"], d["n"], d.get("extra"), d.get("meta"))

    def text(self):
        out = ["| I_rel \\ mu_roll | " + " | ".join(f"{m:.4f}" for m in self.nodes) + " |", "|---|" + "---|" * len(self.nodes)]
        for i, I in enumerate(self.igrid):
            out.append(f"| {I:g} | " + " | ".join((f"{self.drop[i, j]:.3f} ({int(self.k[i, j])}/{int(self.n[i, j])})") for j in range(len(self.nodes))) + " |")
        return "\n".join(out)


# ---- weights
def w_worst():
    w = np.zeros(MD.N_GRID); w[0] = 1.0
    return w


def w_prior():
    return np.full(MD.N_GRID, 1.0 / MD.N_GRID)


def w_true(mu):
    w = np.zeros(MD.N_GRID)
    w[int(np.argmin(np.abs(MD.LOG_MU - np.log(mu))))] = 1.0
    return w
