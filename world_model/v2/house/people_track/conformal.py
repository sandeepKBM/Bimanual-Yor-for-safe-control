"""people_track/conformal.py -- split-conformal person-keep-out margin (replaces the fixed k_sigma * sqrt(n' Sigma n)).  SIMULATION ONLY.

Score (one per tick per ground-truth person that has a nearby track):
        e_T = | (p_hat - p) + T (v_hat - v) |          [m]   the tracker's position error propagated over the reaction horizon T
(p, v = ground truth, hat = the track; T = UncertaintyCfg.t_horizon).  Scores are binned by the TRACK STATE the shield can observe at run time:
        since_upd : seconds since the track's last measurement update (visibility state)   -> 'upd' < 0.15 s, 'coast_s' < 0.6 s, 'coast_l' >= 0.6 s
        age       : seconds since the track was born                                       -> 'young' < 0.5 s, 'mid' < 2 s, 'old' >= 2 s
        range     : horizontal distance camera -> track estimate (close = partial body view)  -> 'r<1.0', 'r<1.7', 'r<2.4', 'far' [m]
The margin of a bin is the split-conformal (1 - alpha) quantile: the ceil((n + 1)(1 - alpha))-th smallest score of the CALIBRATION seeds.  Bins with
fewer than `n_min` samples fall back (upd x age x range) -> (upd x range) -> (upd) -> global; without a range: (upd x age) -> (upd) -> global.  The shield inflates the person disc by this
radius (isotropic: the support function of a disc is its radius in every direction), capped at `cap`.
Coverage is exchangeable across calibration ticks only approximately (ticks within an episode are correlated); it is MEASURED on held-out seeds
(`coverage`) rather than assumed.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

UPD_EDGES = (0.15, 0.6)               # s
AGE_EDGES = (0.5, 2.0)                # s
UPD_NAMES = ("upd", "coast_s", "coast_l")
AGE_NAMES = ("young", "mid", "old")
RNG_EDGES = (1.0, 1.7, 2.4)           # m
RNG_NAMES = ("r<1.0", "r<1.7", "r<2.4", "far")


def bin_of(since_upd_s, age_s):
    return int(np.searchsorted(UPD_EDGES, since_upd_s, side="right")), int(np.searchsorted(AGE_EDGES, age_s, side="right"))


def rbin(range_m):
    return int(np.searchsorted(RNG_EDGES, range_m, side="right"))


def key_str(iu, ia, ir=None):
    return f"{UPD_NAMES[iu]}/{AGE_NAMES[ia]}" + ("" if ir is None else f"/{RNG_NAMES[ir]}")


def key_ur(iu, ir):
    return f"{UPD_NAMES[iu]}|{RNG_NAMES[ir]}"


def cq(scores, alpha):
    """split-conformal quantile: ceil((n+1)(1-alpha))-th order statistic (inf if n is too small)."""
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    return float(s[k - 1]) if 1 <= k <= n else float("inf")


@dataclass
class ConformalMargin:
    alpha: float = 0.05
    t_horizon: float = 0.5
    n_min: int = 40
    cap: float = 0.6
    floor: float = 0.0
    cell: dict = field(default_factory=dict)         # "upd/age" -> (n, q)
    cell3: dict = field(default_factory=dict)        # "upd/age/range" -> (n, q)
    ur: dict = field(default_factory=dict)           # "upd|range" -> (n, q)
    vis: dict = field(default_factory=dict)          # "upd" -> (n, q)
    glob: tuple = (0, 0.0)

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, S, alpha=0.05, t_horizon=0.5, n_min=40, cap=0.6, floor=0.0):
        """S: (N, 3 or 4) columns [since_upd_s, age_s, e_T, (range_m)]."""
        S = np.asarray(S, float)
        iu = np.searchsorted(UPD_EDGES, S[:, 0], side="right")
        ia = np.searchsorted(AGE_EDGES, S[:, 1], side="right")
        ir = np.searchsorted(RNG_EDGES, S[:, 3], side="right") if S.shape[1] > 3 else None
        m = cls(alpha=alpha, t_horizon=t_horizon, n_min=n_min, cap=cap, floor=floor)
        for u in range(3):
            sel = iu == u
            if sel.sum() >= 1:
                m.vis[UPD_NAMES[u]] = (int(sel.sum()), cq(S[sel, 2], alpha))
            for a in range(3):
                sel2 = sel & (ia == a)
                if sel2.sum() >= 1:
                    m.cell[key_str(u, a)] = (int(sel2.sum()), cq(S[sel2, 2], alpha))
                if ir is not None:
                    for r in range(4):
                        sel3 = sel2 & (ir == r)
                        if sel3.sum() >= 1:
                            m.cell3[key_str(u, a, r)] = (int(sel3.sum()), cq(S[sel3, 2], alpha))
            if ir is not None:
                for r in range(4):
                    sel4 = sel & (ir == r)
                    if sel4.sum() >= 1:
                        m.ur[key_ur(u, r)] = (int(sel4.sum()), cq(S[sel4, 2], alpha))
        m.glob = (len(S), cq(S[:, 2], alpha))
        return m

    # ------------------------------------------------------------------ query
    def _ok(self, nq):
        return nq is not None and nq[0] >= self.n_min and np.isfinite(nq[1])

    def radius(self, since_upd_s, age_s, range_m=None):
        u, a = bin_of(since_upd_s, age_s)
        chain = []
        if range_m is not None and self.cell3:
            r = rbin(range_m)
            chain = [self.cell3.get(key_str(u, a, r)), self.ur.get(key_ur(u, r))]
        else:
            chain = [self.cell.get(key_str(u, a))]
        chain += [self.vis.get(UPD_NAMES[u]), self.glob]
        q = self.cap
        for nq in chain:
            if self._ok(nq):
                q = nq[1]
                break
        return float(min(max(q, self.floor), self.cap))

    # ------------------------------------------------------------------ evaluation
    def coverage(self, S):
        """held-out coverage: fraction of samples with e_T <= radius(bin) (overall and per cell)."""
        S = np.asarray(S, float)
        has_r = S.shape[1] > 3
        r = np.array([self.radius(s[0], s[1], s[3] if has_r else None) for s in S])
        ok = S[:, 2] <= r + 1e-12
        out = dict(n=int(len(S)), overall=float(ok.mean()) if len(S) else None, target=1.0 - self.alpha, mean_radius=float(r.mean()) if len(S) else None, cells={})
        iu = np.searchsorted(UPD_EDGES, S[:, 0], side="right")
        ia = np.searchsorted(AGE_EDGES, S[:, 1], side="right")
        for u in range(3):
            for a in range(3):
                sel = (iu == u) & (ia == a)
                if sel.any():
                    out["cells"][key_str(u, a)] = dict(n=int(sel.sum()), cov=float(ok[sel].mean()), mean_r=float(r[sel].mean()))
        if has_r:
            ir = np.searchsorted(RNG_EDGES, S[:, 3], side="right")
            out["by_upd_range"] = {}
            for u in range(3):
                for rr in range(4):
                    sel = (iu == u) & (ir == rr)
                    if sel.any():
                        out["by_upd_range"][key_ur(u, rr)] = dict(n=int(sel.sum()), cov=float(ok[sel].mean()), mean_r=float(r[sel].mean()))
        return out

    # ------------------------------------------------------------------ io
    def to_json(self):
        return dict(alpha=self.alpha, t_horizon=self.t_horizon, n_min=self.n_min, cap=self.cap, floor=self.floor, cell=self.cell, cell3=self.cell3, ur=self.ur,
                    vis=self.vis, glob=list(self.glob))

    @classmethod
    def from_json(cls, j):
        t = lambda d: {k: tuple(v) for k, v in d.items()}
        return cls(alpha=j["alpha"], t_horizon=j["t_horizon"], n_min=j["n_min"], cap=j["cap"], floor=j.get("floor", 0.0),
                   cell=t(j["cell"]), cell3=t(j.get("cell3", {})), ur=t(j.get("ur", {})), vis=t(j["vis"]), glob=tuple(j["glob"]))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_json(), indent=1))

    @classmethod
    def load(cls, path):
        return cls.from_json(json.loads(Path(path).read_text()))
