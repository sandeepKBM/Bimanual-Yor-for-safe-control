"""model_uq/c2_partition.py -- workstream C2: regime partition by discontinuity detection, then per-regime conformal bounds.
SIMULATION / SAVED DATA ONLY (pure numpy / torch-CPU; no robot, no network).

* pa_edges_1d      polynomial-annihilation edge detector (Archibald, Gelb, Yoon 2005), minmod-enhanced, on a nonuniform 1-D grid.
* PAPartition      robust conditional-statistic profile (per-bin 90th pct of y) along each regime coordinate -> PA edges -> cell grid
                   (cross product of the intervals) -> per-cell split conformal (global-conformal fallback for cells with < 19 CAL pts).
* TreePartition    baseline: CART on log(y) -> Mondrian (per-leaf) split conformal.
* CQRBaseline      baseline: conformalised quantile regression (Romano et al. 2019), small MLP, pinball loss.

All follow the protocol API:  fit_calibrate(Xf, yf, Xc, yc, alpha, rng) -> self ;  bound(X) -> (m,) ;  .name
Edges / cells are chosen on FIT only; CAL only sets the per-cell quantiles (cells never depend on CAL labels), so split-conformal validity holds.
"""
from __future__ import annotations

import math
import time

import numpy as np

from world_model.v2.model_uq import protocol as P

ALPHA = P.ALPHA


# ----------------------------------------------------------------------------------------------------------------- polynomial annihilation
def _minmod(rows):
    """rows (M, N) -> elementwise minmod: min |.| if all signs agree (and nonzero), else 0."""
    s = np.sign(rows)
    agree = (s == s[0]).all(0) & (s[0] != 0)
    return np.where(agree, np.sign(rows[0]) * np.abs(rows).min(0), 0.0)


def pa_jump(xg, fg, order=3, minmod=True, return_all=False):
    """Jump function by polynomial annihilation on a (nonuniform) grid.

    Entry j (j=1..N) approximates the jump of f in the cell (x_{j-1}, x_j]; entry 0 is 0.
    For order m: stencil S_j = {x_{j-ml..j+mr}}, ml=ceil(m/2), mr=floor(m/2) (m+1 points, shifted inside the grid at the boundaries),
    c_i = m!/prod_{l!=i}(x_i-x_l), q_m(x_j) = sum_{x_i in S_j, x_i>=x_j} c_i, L_m f(x_j) = sum_i c_i f(x_i) / q_m(x_j).
    With `minmod`, returns minmod_{m=1..order} L_m f (Archibald et al. 2005, sec. 3).
    """
    x = np.asarray(xg, np.float64)
    f = np.asarray(fg, np.float64)
    n = len(x)
    Ls = []
    for m in range(1, order + 1):
        L = np.zeros(n)
        if n >= m + 1:
            ml = (m + 1) // 2
            fm = math.factorial(m)
            for j in range(1, n):
                lo = min(max(j - ml, 0), n - m - 1)
                idx = np.arange(lo, lo + m + 1)
                xs = x[idx]
                d = xs[:, None] - xs[None, :]
                np.fill_diagonal(d, 1.0)
                c = fm / d.prod(1)
                right = xs >= x[j]
                q = c[right].sum()
                if (~right).all() or right.all() or abs(q) < 1e-9 * np.abs(c).max():
                    continue
                L[j] = float((c * f[idx]).sum() / q)
        Ls.append(L)
    Ls = np.array(Ls)
    out = _minmod(Ls) if minmod else Ls[-1]
    return (out, Ls) if return_all else out


def _noise_scale(Lraw, trim=0.05):
    """robust noise level of the jump function: median of |L_m| pooled over orders, excluding the top `trim` fraction."""
    a = np.abs(Lraw[:, 1:]).ravel()
    if a.size == 0:
        return 0.0
    cut = np.quantile(a, 1 - trim)
    a = a[a <= cut]
    return float(np.median(a) / 0.6745) if a.size else 0.0


def pa_edges_1d(xg, fg, order=3, thresh=8.0, min_sep=3, min_jump=0.0, return_strength=False):
    """Edge locations (midpoints of grid cells) of a piecewise-smooth f sampled at xg (sorted). Edge at cell j if
    |minmod L f(x_j)| > thresh * noise_scale (and > min_jump), after non-maximum suppression with separation >= min_sep cells."""
    x = np.asarray(xg, np.float64)
    f = np.asarray(fg, np.float64)
    if len(x) < order + 2:
        return (np.zeros(0), np.zeros(0)) if return_strength else np.zeros(0)
    Lmm, Lraw = pa_jump(x, f, order, True, True)
    thr = max(thresh * _noise_scale(Lraw), min_jump)
    a = np.abs(Lmm)
    cand = np.flatnonzero(a > thr)
    keep = []
    for j in cand[np.argsort(-a[cand], kind="stable")]:
        if all(abs(int(j) - k) >= min_sep for k in keep):
            keep.append(int(j))
    keep.sort()
    ed = np.array([0.5 * (x[j - 1] + x[j]) for j in keep])
    st = np.array([a[j] for j in keep])
    return (ed, st) if return_strength else ed


# ----------------------------------------------------------------------------------------------------------------- helpers
def _fallback_global(yc, alpha):
    return P.conformal_quantile(yc, alpha)


def _min_pts(alpha):
    return int(math.ceil(1.0 / alpha)) - 1


def _cell_bounds(cell_c, yc, ncell, alpha, glob):
    """per-cell conformal quantile of yc; global fallback if < ceil(1/alpha)-1 points. -> (bounds, counts)"""
    mn = _min_pts(alpha)
    b = np.full(ncell, glob)
    cnt = np.bincount(cell_c, minlength=ncell)
    for k in np.flatnonzero(cnt >= mn):
        b[k] = P.conformal_quantile(yc[cell_c == k], alpha)
    return b, cnt


# ----------------------------------------------------------------------------------------------------------------- PA partition
class PAPartition:
    """Edges of the robust conditional profile (PA, dimension-by-dimension) -> cell grid -> per-cell split conformal."""

    def __init__(self, cols_edge=(0,), stat="q90", q=0.9, order=3, thresh=4.0, min_sep=3, rel_jump=0.0, k_bins=None, name=None):
        self.cols_edge = tuple(cols_edge)
        self.stat, self.q, self.order, self.thresh, self.min_sep, self.rel_jump, self.k_bins = stat, q, order, thresh, min_sep, rel_jump, k_bins
        self.name = name or f"pa[{','.join(map(str, self.cols_edge))}]"
        self.edges, self.edges_raw, self.cells, self.fit_time = {}, {}, {}, 0.0

    def _profile(self, x, y):
        n = len(x)
        K = self.k_bins or max(12, min(64, n // 125))   # ~125 pts/bin (synthetic sweep: n//8-sized bins are too noisy for q90 edges)
        qs = np.unique(np.quantile(x, np.linspace(0, 1, K + 1)))
        b = np.clip(np.searchsorted(qs, x, side="right") - 1, 0, len(qs) - 2)
        xc, fv = [], []
        for k in range(len(qs) - 1):
            m = b == k
            if m.sum() == 0:
                continue
            xc.append(np.median(x[m]))
            fv.append(np.quantile(y[m], self.q) if self.stat.startswith("q") else y[m].mean())
        return np.array(xc), np.array(fv)

    def _cell_ids(self, X, edges):
        ids = np.zeros(len(X), np.int64)
        radix = 1
        for c in self.cols_edge:
            e = np.asarray(edges[c])
            ids += radix * np.searchsorted(e, X[:, c], side="right")
            radix *= len(e) + 1
        return ids

    def fit_calibrate(self, Xf, yf, Xc, yc, alpha=ALPHA, rng=None):
        t0 = time.time()
        Xf, yf, Xc, yc = (np.asarray(a, np.float64) for a in (Xf, yf, Xc, yc))
        strength = {}
        for c in self.cols_edge:
            xg, fg = self._profile(Xf[:, c], yf)
            rng_f = max(float(fg.max() - fg.min()), 1e-12) if len(fg) else 1.0
            ed, st = pa_edges_1d(xg, fg, self.order, self.thresh, self.min_sep, self.rel_jump * rng_f, return_strength=True)
            self.edges_raw[c] = [float(e) for e in ed]
            strength[c] = {float(e): float(s) for e, s in zip(ed, st)}
        edges = {c: sorted(strength[c]) for c in self.cols_edge}
        # cap #cells so that CAL averages >= ceil(1/alpha)-1 pts per cell: drop the weakest edge (= merge along it) until satisfied
        mn = _min_pts(alpha)
        cap = max(1, len(yc) // mn)
        while int(np.prod([len(edges[c]) + 1 for c in self.cols_edge])) > cap:
            allr = [(strength[c][e], c, e) for c in self.cols_edge for e in edges[c]]
            _, c, e = min(allr)
            edges[c].remove(e)
        self.edges = {c: list(v) for c, v in edges.items()}
        self._edges = edges
        ncell = int(np.prod([len(edges[c]) + 1 for c in self.cols_edge]))
        self._glob = _fallback_global(yc, alpha)
        cell_c = self._cell_ids(Xc, edges)
        self._b, cnt = _cell_bounds(cell_c, yc, ncell, alpha, self._glob)
        self.cells = dict(n_cells=ncell, n_cal=cnt.tolist(), bounds=self._b.tolist(), n_fallback=int((cnt < mn).sum()), cap=cap)
        self.fit_time = time.time() - t0
        return self

    def bound(self, X):
        return self._b[self._cell_ids(np.asarray(X, np.float64), self._edges)]


# ----------------------------------------------------------------------------------------------------------------- CART baseline
class _Tree:
    def __init__(self, max_depth, min_leaf, n_cand=32):
        self.max_depth, self.min_leaf, self.n_cand = max_depth, min_leaf, n_cand
        self.nodes = []        # (feat, thr, left, right); leaf: (-1, leaf_id, -1, -1)
        self.n_leaves = 0

    def fit(self, X, z):
        self.nodes, self.n_leaves = [], 0
        self._grow(X, z, 0)
        return self

    def _leaf(self):
        i = len(self.nodes)
        self.nodes.append((-1, self.n_leaves, -1, -1))
        self.n_leaves += 1
        return i

    def _best_split(self, X, z):
        n = len(z)
        best = (0.0, -1, 0.0)
        tot, tot2 = z.sum(), (z * z).sum()
        base = tot2 - tot * tot / n
        for f in range(X.shape[1]):
            xs = X[:, f]
            thr = np.unique(np.quantile(xs, np.linspace(0, 1, self.n_cand + 2)[1:-1]))
            o = np.argsort(xs, kind="stable")
            xo, zo = xs[o], z[o]
            cs = np.cumsum(zo)
            for t in thr:
                nl = int(np.searchsorted(xo, t, side="right"))
                nr = n - nl
                if nl < self.min_leaf or nr < self.min_leaf:
                    continue
                sl = cs[nl - 1]
                sr = tot - sl
                gain = base - ((zo[:nl] - sl / nl) ** 2).sum() - ((zo[nl:] - sr / nr) ** 2).sum()
                if gain > best[0] + 1e-12:
                    best = (gain, f, float(t))
        return best

    def _grow(self, X, z, depth):
        if depth >= self.max_depth or len(z) < 2 * self.min_leaf:
            return self._leaf()
        gain, f, t = self._best_split(X, z)
        if f < 0:
            return self._leaf()
        i = len(self.nodes)
        self.nodes.append(None)
        m = X[:, f] <= t
        l = self._grow(X[m], z[m], depth + 1)
        r = self._grow(X[~m], z[~m], depth + 1)
        self.nodes[i] = (f, t, l, r)
        return i

    def apply(self, X):
        out = np.zeros(len(X), np.int64)
        stack = [(0, np.arange(len(X)))]
        while stack:
            i, idx = stack.pop()
            f, t, l, r = self.nodes[i]
            if f < 0:
                out[idx] = int(t)
                continue
            m = X[idx, f] <= t
            stack.append((l, idx[m]))
            stack.append((r, idx[~m]))
        return out


class TreePartition:
    """CART (variance reduction on log(y+1e-3)) -> Mondrian (per-leaf) split conformal, global fallback for sparse leaves."""
    name = "tree"

    def __init__(self, max_depth=4, min_leaf=None, name=None):
        self.max_depth, self.min_leaf_arg = max_depth, min_leaf
        if name:
            self.name = name
        self.fit_time = 0.0
        self.cells = {}

    def fit_calibrate(self, Xf, yf, Xc, yc, alpha=ALPHA, rng=None):
        t0 = time.time()
        Xf, yf, Xc, yc = (np.asarray(a, np.float64) for a in (Xf, yf, Xc, yc))
        ml = self.min_leaf_arg or max(10, len(yf) // 20)
        self.tree = _Tree(self.max_depth, ml).fit(Xf, np.log(yf + 1e-3))
        glob = _fallback_global(yc, alpha)
        leaf_c = self.tree.apply(Xc)
        self._b, cnt = _cell_bounds(leaf_c, yc, self.tree.n_leaves, alpha, glob)
        self.cells = dict(n_cells=self.tree.n_leaves, n_cal=cnt.tolist(), bounds=self._b.tolist(), n_fallback=int((cnt < _min_pts(alpha)).sum()))
        self.fit_time = time.time() - t0
        return self

    def bound(self, X):
        return self._b[self.tree.apply(np.asarray(X, np.float64))]


# ----------------------------------------------------------------------------------------------------------------- CQR baseline
class CQRBaseline:
    """Conformalised quantile regression (Romano, Patterson, Candes 2019): MLP 2x32 pinball loss at tau=1-alpha on log(y+1e-3); CAL correction."""
    name = "cqr"

    def __init__(self, epochs=300, hidden=32, lr=1e-2, name=None):
        self.epochs, self.hidden, self.lr = epochs, hidden, lr
        if name:
            self.name = name
        self.fit_time = 0.0

    def _q(self, X):
        import torch
        with torch.no_grad():
            return self.net(torch.as_tensor((np.asarray(X, np.float64) - self.mu) / self.sd, dtype=torch.float32)).squeeze(1).numpy().astype(np.float64)

    def fit_calibrate(self, Xf, yf, Xc, yc, alpha=ALPHA, rng=None):
        import torch
        t0 = time.time()
        torch.set_num_threads(1)
        rng = rng if rng is not None else np.random.default_rng(0)
        torch.manual_seed(int(rng.integers(2**31 - 1)))
        Xf, yf, Xc, yc = (np.asarray(a, np.float64) for a in (Xf, yf, Xc, yc))
        self.mu, self.sd = Xf.mean(0), Xf.std(0) + 1e-9
        h = self.hidden
        self.net = torch.nn.Sequential(torch.nn.Linear(Xf.shape[1], h), torch.nn.ReLU(), torch.nn.Linear(h, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))
        xt = torch.as_tensor((Xf - self.mu) / self.sd, dtype=torch.float32)
        zt = torch.as_tensor(np.log(yf + 1e-3), dtype=torch.float32)
        # start the output near the empirical (1-alpha) quantile of the target
        with torch.no_grad():
            self.net[-1].bias.fill_(float(np.quantile(np.log(yf + 1e-3), 1 - alpha)))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        tau = 1 - alpha
        for _ in range(self.epochs):
            opt.zero_grad()
            d = zt - self.net(xt).squeeze(1)
            loss = torch.maximum(tau * d, (tau - 1) * d).mean()
            loss.backward()
            opt.step()
        zc = np.log(yc + 1e-3)
        self.Q = P.conformal_quantile(zc - self._q(Xc), alpha)
        self.fit_time = time.time() - t0
        return self

    def bound(self, X):
        with np.errstate(over="ignore"):
            return np.exp(np.minimum(self._q(X) + self.Q, 50.0)) - 1e-3 if np.isfinite(self.Q) else np.full(len(X), np.inf)
