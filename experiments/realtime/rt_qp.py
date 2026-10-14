"""Bounded-time dense QP for the real-time FL-MPC: Goldfarb-Idnani (GI) dual
active-set method, numba (nogil) with preallocated workspaces. SIMULATION /
OFFLINE math only.

Problem (the SAME one FLMPC hands to solve_constrained_box_qp, incl. its
0.5(H+H^T) + 1e-8*I Tikhonov term):

    min_y  0.5 y'G y + g'y
    s.t.   A_i x - [s_{sidx_i}] <= b_i        general rows (x = y[:n]); a row may carry
                                              ONE slack column (the relaxed fallback)
           lo <= y <= hi                      simple bounds (x box, 0 <= s <= S_MAX)

y = (x, s), nt = n + ns.  ns = 0 for the hard FL-MPC QP.

Why GI (see rt_flmpc.py / the report for the full argument):
  * EXACT for strictly convex QPs (G = H + 1e-8 I > 0 here) -> solves the same
    optimisation problem to machine precision, and returns a KKT certificate
    (primal feasibility, u >= 0, stationarity) instead of an approximate
    dual-coordinate-ascent point;
  * infeasibility is detected EXACTLY (a violated row that is linearly dependent
    on the active set with no droppable multiplier), not by a multiplier-size
    heuristic;
  * every iteration (one constraint added or dropped) has a FIXED worst-case cost
    O(m*n + nt^2) (violation scan + d = J'c + Givens updates of J and R), and the
    iteration count is CAPPED -> a deterministic worst-case time
    t_max = t_setup + cap * t_iter_max. G's factor J = L^{-T} depends only on the
    cost weights, not on the state, so it is cached (no per-step factorisation).
  * dual method: every iterate is optimal for a SUBSET of the constraints, so a
    capped iterate is NOT certified feasible -> the caller treats a cap exactly
    like infeasibility (goes to the relaxed / zero-accel fallback) and COUNTS it.
"""
from __future__ import annotations

import glob
import hashlib
import math
import os

import numpy as np
from numba import njit


def _numba_cache_guard():
    """Numba's on-disk cache (cache=True) is keyed on the DEFINING file only: a cached
    rt_kernels function keeps the machine code of the rt_qp functions it was compiled
    against even after rt_qp.py changes (observed: a stale, 4x slower GI inside
    rt_kernels.solve_with_fallback). Purge the rt_qp/rt_kernels cache entries whenever
    either source file changes (content hash stamp)."""
    here = os.path.dirname(os.path.abspath(__file__))
    srcs = [os.path.join(here, f) for f in ("rt_qp.py", "rt_kernels.py")]
    try:
        h = hashlib.sha1(b"".join(open(p, "rb").read() for p in srcs if os.path.exists(p))).hexdigest()
        cdir = os.path.join(here, "__pycache__")
        stamp = os.path.join(cdir, "rt_numba_cache.stamp")
        old = open(stamp).read().strip() if os.path.exists(stamp) else None
        if old != h:
            for f in glob.glob(os.path.join(cdir, "rt_qp.*.nb[ic]")) + glob.glob(os.path.join(cdir, "rt_kernels.*.nb[ic]")):
                try:
                    os.remove(f)
                except OSError:
                    pass
            os.makedirs(cdir, exist_ok=True)
            with open(stamp, "w") as fh:
                fh.write(h)
    except OSError:
        pass


_numba_cache_guard()

ST_OPTIMAL = 0
ST_INFEASIBLE = 1
ST_ITERCAP = 2
ST_NOTPD = 3

TOL_VIOL = 1e-9        # raw row violation tolerance (scaled by max(1,|e_i|)); old solver used 1e-8
EPS_DEP = 1e-20        # ||d2||^2 <= EPS_DEP*||d||^2  -> new row numerically dependent on the active set


# ---------------------------------------------------------------------------
# factorisation (cache-miss path only: once per distinct cost-weight set)
# ---------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def chol_jinv(G, J):
    """J <- L^{-T} with G = L L^T (J upper triangular). Returns False if G is not PD."""
    n = G.shape[0]
    L = np.zeros((n, n))
    for j in range(n):
        s = G[j, j]
        for k in range(j):
            s -= L[j, k] * L[j, k]
        if s <= 0.0:
            return False
        L[j, j] = math.sqrt(s)
        for i in range(j + 1, n):
            s = G[i, j]
            for k in range(j):
                s -= L[i, k] * L[j, k]
            L[i, j] = s / L[j, j]
    # Linv (lower): L Linv = I, forward substitution per column
    for c in range(n):
        for i in range(n):
            J[i, c] = 0.0
    for c in range(n):          # column c of Linv -> row c of J (J = Linv^T)
        for i in range(c, n):
            s = 1.0 if i == c else 0.0
            for k in range(c, i):
                s -= L[i, k] * J[c, k]
            J[c, i] = s / L[i, i]
    return True


# ---------------------------------------------------------------------------
# constraint primitives. ids: [0,m) general, [m, m+nt) lower bounds, [m+nt, m+2nt) upper
# convention (quadprog): c_i . y >= e_i ; general: c = (-A_i, +e_slack), e = -b_i
#
# J is stored TRANSPOSED ("JT", JT[j, k] = J[k, j]) so that every hot loop -- d = J^T c
# (row dot products), z = J[:, q:] d[q:] (row axpys) and the Givens rotations of two J
# COLUMNS (two JT rows) -- runs over contiguous memory. Dot products use
# fastmath(reassoc, contract, nsz) so LLVM can vectorise the reductions (no nnan/ninf:
# the solver uses inf step lengths).
# ---------------------------------------------------------------------------
_FM = {"reassoc", "contract", "nsz"}


@njit(cache=True, nogil=True, fastmath=_FM)
def _dot_row(A, i, y, n):
    # NOT numba-inlined: numba IR inlining would drop the fastmath flags (-> a latency-bound
    # scalar reduction, measured 10x slower); LLVM still inlines it with the flags intact.
    s = 0.0
    for k in range(n):
        s += A[i, k] * y[k]
    return s


@njit(cache=True, nogil=True, fastmath=_FM)
def _matvec(A, y, m, n, out):
    for i in range(m):
        s = 0.0
        for k in range(n):
            s += A[i, k] * y[k]
        out[i] = s


@njit(cache=True, nogil=True, inline="always")
def _cdot(i, y, A, sidx, m, n, nt):
    if i < m:
        s = -_dot_row(A, i, y, n)
        if sidx[i] >= 0:
            s += y[n + sidx[i]]
        return s
    if i < m + nt:
        return y[i - m]
    return -y[i - m - nt]


@njit(cache=True, nogil=True, inline="always")
def _erhs(i, b, lo, hi, m, nt):
    if i < m:
        return -b[i]
    if i < m + nt:
        return lo[i - m]
    return -hi[i - m - nt]


@njit(cache=True, nogil=True, fastmath=_FM)
def _jtc(i, J, A, sidx, m, n, nt, d, nsb, act):
    """d <- J^T c_i  (J stored transposed: d[j] = JT[j, :] . c_i). Rows j < nsb are the
    PURE slack-bound block (relaxed start): JT row j = e_{n+k_j}/sigma with k_j = act[j]-m-n,
    zero x-part -> no dot product needed."""
    if i < m:
        sl = sidx[i]
        for j in range(nsb):
            d[j] = J[j, n + sl] if (sl >= 0 and act[j] - m - n == sl) else 0.0
        for j in range(nsb, nt):
            s = 0.0
            for k in range(n):
                s -= J[j, k] * A[i, k]
            if sl >= 0:
                s += J[j, n + sl]
            d[j] = s
    elif i < m + nt:
        k = i - m
        for j in range(nt):
            d[j] = J[j, k]
    else:
        k = i - m - nt
        for j in range(nt):
            d[j] = -J[j, k]


@njit(cache=True, nogil=True, fastmath=_FM)
def _rot_rows(J, r0, r1, cs, sn, nt):
    for k in range(nt):
        ja = J[r0, k]
        jb = J[r1, k]
        J[r0, k] = cs * ja + sn * jb
        J[r1, k] = -sn * ja + cs * jb


@njit(cache=True, nogil=True)
def _drop(l, q, nt, J, R, act, u, iact):
    """Remove active constraint at position l (0<=l<q); u[q] (the pending row's
    multiplier) shifts down with the rest. Returns new q. R is stored TRANSPOSED
    (R[c, r] = R_math[r, c]) so column shifts are contiguous row copies."""
    iact[act[l]] = 0
    for j in range(l, q - 1):
        act[j] = act[j + 1]
        u[j] = u[j + 1]
        for k in range(j + 2):
            R[j, k] = R[j + 1, k]
    u[q - 1] = u[q]
    for j in range(l, q - 1):
        a = R[j, j]
        bb = R[j, j + 1]
        h = math.hypot(a, bb)
        if h == 0.0:
            continue
        cs = a / h
        sn = bb / h
        R[j, j] = h
        R[j, j + 1] = 0.0
        for k in range(j + 1, q - 1):
            ra = R[k, j]
            rb = R[k, j + 1]
            R[k, j] = cs * ra + sn * rb
            R[k, j + 1] = -sn * ra + cs * rb
        _rot_rows(J, j, j + 1, cs, sn, nt)
    for k in range(q):
        R[q - 1, k] = 0.0
    return q - 1


@njit(cache=True, nogil=True)
def _add(p, q, nt, J, R, act, iact, d):
    """Append constraint p (d = J^T c_p, computed with the CURRENT J). Returns new q."""
    for j in range(nt - 1, q, -1):
        bb = d[j]
        if bb == 0.0:
            continue
        a = d[j - 1]
        h = math.hypot(a, bb)
        cs = a / h
        sn = bb / h
        d[j - 1] = h
        d[j] = 0.0
        _rot_rows(J, j - 1, j, cs, sn, nt)
    for k in range(q + 1):
        R[q, k] = d[k]
    act[q] = p
    iact[p] = 1
    return q + 1


@njit(cache=True, nogil=True)
def _drop_any(l, q, nsb, nt, J, R, act, u, iact):
    """Drop active position l. If l is inside the pure slack-bound block [0, nsb) it is first
    SWAPPED with the block's last position (exact: the block is diagonal in R and its JT rows
    are pure), so the Givens re-triangularisation only touches the (q - nsb) non-pure columns
    instead of O(q). Returns (q, nsb)."""
    if l < nsb:
        e = nsb - 1
        if l != e:
            ta = act[l]
            act[l] = act[e]
            act[e] = ta
            tu = u[l]
            u[l] = u[e]
            u[e] = tu
            for k in range(nt):
                tj = J[l, k]
                J[l, k] = J[e, k]
                J[e, k] = tj
            for k in range(q):          # math rows l <-> e  (RT columns)
                tr = R[k, l]
                R[k, l] = R[k, e]
                R[k, e] = tr
            for k in range(nsb):        # math cols l <-> e  (RT rows; pure block only)
                tr = R[l, k]
                R[l, k] = R[e, k]
                R[e, k] = tr
        l = e
        nsb = e
    q = _drop(l, q, nt, J, R, act, u, iact)
    return q, nsb


@njit(cache=True, nogil=True, fastmath=_FM)
def _backsub(R, d, r, q, nsb):
    for j in range(q):
        r[j] = d[j]
    for j in range(q - 1, -1, -1):
        rj = r[j] / R[j, j]
        r[j] = rj
        if j >= nsb and rj != 0.0:
            for k in range(j):
                r[k] -= rj * R[j, k]


@njit(cache=True, nogil=True, fastmath=_FM)
def _row_norms(A, sidx, m, n, rn):
    for i in range(m):
        s = 0.0
        for k in range(n):
            s += A[i, k] * A[i, k]
        if sidx[i] >= 0:
            s += 1.0
        rn[i] = math.sqrt(s) if s > 0.0 else 1.0


@njit(cache=True, nogil=True, fastmath=_FM)
def _zstep(J, d, q, nt, t, z, x):
    """z = J[:, q:] d[q:] (JT rows axpy); x += t z."""
    for k in range(nt):
        z[k] = 0.0
    for j in range(q, nt):
        dj = d[j]
        if dj != 0.0:
            for k in range(nt):
                z[k] += dj * J[j, k]
    for k in range(nt):
        x[k] += t * z[k]


@njit(cache=True, nogil=True)
def gi_core(n, ns, m, A, b, sidx, g, lo, hi, J, R, act, u, x, iact, d, z, r, q0, max_iter, stats, rn):
    """Goldfarb-Idnani dual active set, in place. On entry (J [stored transposed], R[:, :q0],
    act[:q0], u[:q0], x, iact) must describe the minimiser of the problem restricted to the
    q0 initially-active constraints (q0 = 0: x = -J J^T g). stats[0] <- #iterations
    (adds + drops), stats[1] <- final q, stats[2] <- #drops. rn: (>= 2m) scratch (row norms, A x).
    Returns status."""
    nt = n + ns
    q = q0
    nsb = q0 if ns > 0 else 0      # pure slack-bound block (gi_init_slack start) -- see _jtc / _drop_any
    it = 0
    ndrop = 0
    ntot = m + 2 * nt
    status = ST_OPTIMAL
    _row_norms(A, sidx, m, n, rn)
    ax = rn[m:2 * m]
    while True:
        # ---- step 1: most violated inactive constraint (normalised by ||c||) ----
        _matvec(A, x, m, n, ax)                     # one vectorised A x per outer iteration
        p = -1
        best = 0.0
        sp = 0.0
        for i in range(ntot):
            if iact[i] != 0:
                continue
            if i < m:
                e = -b[i]
                s = -ax[i]
                if sidx[i] >= 0:
                    s += x[n + sidx[i]]
                s -= e
            elif i < m + nt:
                e = lo[i - m]
                s = x[i - m] - e
            else:
                e = -hi[i - m - nt]
                s = -x[i - m - nt] - e
            tol = TOL_VIOL * max(1.0, abs(e))
            if s >= -tol:
                continue
            v = s / rn[i] if i < m else s
            if p < 0 or v < best:
                best = v
                p = i
                sp = s
        if p < 0:
            status = ST_OPTIMAL
            break
        u[q] = 0.0
        # ---- step 2: add p (possibly after dropping blocking constraints) ----
        done_p = False
        while not done_p:
            if it >= max_iter:
                status = ST_ITERCAP
                break
            it += 1
            _jtc(p, J, A, sidx, m, n, nt, d, nsb, act)
            dn2 = 0.0
            zn2 = 0.0
            for j in range(nt):
                dn2 += d[j] * d[j]
                if j >= q:
                    zn2 += d[j] * d[j]
            # r = R^{-1} d[:q]  (column-oriented back substitution on RT rows; the pure
            # slack block of R is diagonal, so its columns carry nothing above the diagonal)
            _backsub(R, d, r, q, nsb)
            t1 = np.inf
            l = -1
            for j in range(q):
                if r[j] > 0.0:
                    tt = u[j] / r[j]
                    if tt < t1:
                        t1 = tt
                        l = j
            if zn2 > EPS_DEP * dn2 and zn2 > 0.0:
                t2 = -sp / zn2
            else:
                t2 = np.inf
            if t1 == np.inf and t2 == np.inf:
                status = ST_INFEASIBLE
                break
            if t2 == np.inf:
                # dual-only step, then drop the blocking constraint
                for j in range(q):
                    u[j] -= t1 * r[j]
                u[q] += t1
                q, nsb = _drop_any(l, q, nsb, nt, J, R, act, u, iact)
                ndrop += 1
                continue
            t = t1 if t1 < t2 else t2
            _zstep(J, d, q, nt, t, z, x)
            for j in range(q):
                u[j] -= t * r[j]
            u[q] += t
            if t2 <= t1:
                q = _add(p, q, nt, J, R, act, iact, d)
                done_p = True
            else:
                q, nsb = _drop_any(l, q, nsb, nt, J, R, act, u, iact)
                ndrop += 1
                sp = _cdot(p, x, A, sidx, m, n, nt) - _erhs(p, b, lo, hi, m, nt)
        if status != ST_OPTIMAL:
            break
    stats[0] = it
    stats[1] = q
    stats[2] = ndrop
    return status


@njit(cache=True, nogil=True, fastmath=_FM)
def gi_init_cold(n, J0T, g, J, x, iact, ntot):
    """q0 = 0 start: J <- J0 (both stored transposed), x <- -J0 J0^T g, no active constraints."""
    for i in range(n):
        for j in range(n):
            J[i, j] = J0T[i, j]
    for i in range(ntot):
        iact[i] = 0
    for i in range(n):
        x[i] = 0.0
    for j in range(n):
        s = 0.0
        for k in range(n):
            s += J0T[j, k] * g[k]
        for i in range(n):
            x[i] -= J0T[j, i] * s
    return 0.0


@njit(cache=True, nogil=True, fastmath=_FM)
def gi_init_slack(n, ns, m, J0xT, sigma, g, J, R, act, u, x, iact):
    """Relaxed-fallback start: y = (x_unc, s = 0) with ALL slack lower bounds
    s_k >= 0 active (valid dual-feasible start: their multipliers u_k = g_s[k] =
    RHO_SLACK > 0). J = [slack cols | x cols] of blockdiag(J0x, I/sigma) (stored
    transposed: JT rows 0..ns-1 = slack columns); R = I/sigma."""
    nt = n + ns
    ntot = m + 2 * nt
    for i in range(ntot):
        iact[i] = 0
    for i in range(nt):
        for j in range(nt):
            J[i, j] = 0.0
    for k in range(ns):
        J[k, n + k] = 1.0 / sigma
    for j in range(n):
        for i in range(n):
            J[ns + j, i] = J0xT[j, i]
    for i in range(ns):
        for j in range(ns):
            R[i, j] = 0.0
        R[i, i] = 1.0 / sigma
        act[i] = m + n + i          # lower bound on variable n+i
        iact[m + n + i] = 1
        u[i] = g[n + i]             # = RHO (+ sigma^2 * 0)
    for i in range(nt):
        x[i] = 0.0
    for j in range(n):
        s = 0.0
        for k in range(n):
            s += J0xT[j, k] * g[k]
        for i in range(n):
            x[i] -= J0xT[j, i] * s
    return ns


# ---------------------------------------------------------------------------
# convenience (tests / offline): full solve with fresh allocations
# ---------------------------------------------------------------------------
class GIWorkspace:
    """Preallocated arrays for problems up to nt_max variables / m_max rows."""

    def __init__(self, nt_max: int, m_max: int):
        self.nt_max, self.m_max = nt_max, m_max
        self.J = np.zeros((nt_max, nt_max))
        self.R = np.zeros((nt_max, nt_max))
        self.act = np.zeros(nt_max + 1, np.int64)
        self.u = np.zeros(nt_max + 1)
        self.x = np.zeros(nt_max)
        self.iact = np.zeros(m_max + 2 * nt_max, np.int64)
        self.d = np.zeros(nt_max)
        self.z = np.zeros(nt_max)
        self.r = np.zeros(nt_max)
        self.stats = np.zeros(4, np.int64)
        self.rn = np.zeros(2 * m_max)


def factor(G):
    """J0T = (L^{-T})^T = L^{-1}, the TRANSPOSED storage gi_core works on."""
    G = np.ascontiguousarray(G, dtype=np.float64)
    J = np.zeros_like(G)
    if not chol_jinv(G, J):
        raise np.linalg.LinAlgError("G not positive definite")
    return np.ascontiguousarray(J.T)


def solve_dense(H, f, lo, hi, A, b, max_iter=1000, J0=None, ws=None):
    """Hard QP exactly as solve_constrained_box_qp poses it (0.5(H+H^T)+1e-8 I).
    Returns (x, status, stats, active_ids, u)."""
    n = H.shape[0]
    G = 0.5 * (H + H.T) + 1e-8 * np.eye(n)
    if J0 is None:
        J0 = factor(G)
    A = np.ascontiguousarray(A, dtype=np.float64)
    m = A.shape[0]
    if ws is None:
        ws = GIWorkspace(n, m)
    sidx = -np.ones(m, np.int64)
    gi_init_cold(n, J0, f, ws.J, ws.x, ws.iact, m + 2 * n)
    st = gi_core(n, 0, m, A, np.asarray(b, float), sidx, np.asarray(f, float), np.asarray(lo, float),
                 np.asarray(hi, float), ws.J, ws.R, ws.act, ws.u, ws.x, ws.iact, ws.d, ws.z, ws.r, 0,
                 max_iter, ws.stats, ws.rn)
    q = int(ws.stats[1])
    return ws.x[:n].copy(), st, ws.stats.copy(), ws.act[:q].copy(), ws.u[:q].copy()


def kkt_residuals(H, f, lo, hi, A, b, x, act, u, n=None, sidx=None):
    """(max primal violation, min multiplier, stationarity inf-norm) of a GI result
    (quadprog convention: G x + g = sum_active u_i c_i)."""
    nt = H.shape[0]
    n = nt if n is None else n
    m = A.shape[0]
    G = 0.5 * (H + H.T) + 1e-8 * np.eye(nt)
    sidx = -np.ones(m, np.int64) if sidx is None else sidx
    Ax = A @ x[:n]
    for i in range(m):
        if sidx[i] >= 0:
            Ax[i] -= x[n + sidx[i]]
    viol = max(float(np.max(Ax - b, initial=-np.inf)), float(np.max(lo - x)), float(np.max(x - hi)))
    grad = G @ x + f
    Cu = np.zeros(nt)
    for k, i in enumerate(act):
        i = int(i)
        if i < m:
            c = np.zeros(nt)
            c[:n] = -A[i]
            if sidx[i] >= 0:
                c[n + sidx[i]] = 1.0
        elif i < m + nt:
            c = np.zeros(nt); c[i - m] = 1.0
        else:
            c = np.zeros(nt); c[i - m - nt] = -1.0
        Cu += u[k] * c
    return viol, (float(np.min(u)) if len(u) else 0.0), float(np.max(np.abs(grad - Cu)))
