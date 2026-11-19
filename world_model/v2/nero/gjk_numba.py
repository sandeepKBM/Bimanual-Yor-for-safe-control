"""gjk_numba.py -- Numba port of clearance._gjk (Gilbert distance between the convex hulls of two point sets).

Same loop, tolerances (tol=1e-10, maxit=60, nv<1e-12 -> 0, keep lam>1e-12) and support rule as the scipy version; the nnls step (min |P^T lam|,
sum lam = 1, lam >= 0, solved there with an M=1e3 penalty row) is replaced by the EXACT closest point to the origin on the simplex spanned by
the current points (<= 4): every subset's affine-hull minimiser is solved in closed form and the smallest valid (lam >= 0) one is taken.
fastmath OFF.  Flag BIMANUAL_LEGACY_GJK_SCIPY=1 restores the scipy path (see clearance.py).
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _solve_subset(P, idx, k, lam):
    """min-norm point on the affine hull of P[idx[:k]]; fills lam[:k] (barycentric); returns False if degenerate."""
    if k == 1:
        lam[0] = 1.0
        return True
    n = k - 1
    E = np.empty((3, n))
    p0 = P[idx[0]]
    for j in range(n):
        for c in range(3):
            E[c, j] = P[idx[j + 1], c] - p0[c]
    A = np.empty((n, n))
    b = np.empty(n)
    for i in range(n):
        s = 0.0
        for c in range(3):
            s += E[c, i] * p0[c]
        b[i] = -s
        for j in range(n):
            t = 0.0
            for c in range(3):
                t += E[c, i] * E[c, j]
            A[i, j] = t
    # Gaussian elimination with partial pivoting
    scale = 0.0
    for i in range(n):
        if A[i, i] > scale:
            scale = A[i, i]
    if scale <= 0.0:
        return False
    for col in range(n):
        piv = col
        mx = abs(A[col, col])
        for r in range(col + 1, n):
            if abs(A[r, col]) > mx:
                mx = abs(A[r, col])
                piv = r
        if mx <= 1e-14 * scale:
            return False
        if piv != col:
            for c in range(n):
                tmp = A[col, c]
                A[col, c] = A[piv, c]
                A[piv, c] = tmp
            tmp = b[col]
            b[col] = b[piv]
            b[piv] = tmp
        for r in range(col + 1, n):
            f = A[r, col] / A[col, col]
            for c in range(col, n):
                A[r, c] -= f * A[col, c]
            b[r] -= f * b[col]
    mu = np.empty(n)
    for i in range(n - 1, -1, -1):
        s = b[i]
        for c in range(i + 1, n):
            s -= A[i, c] * mu[c]
        mu[i] = s / A[i, i]
    tot = 0.0
    for j in range(n):
        lam[j + 1] = mu[j]
        tot += mu[j]
    lam[0] = 1.0 - tot
    return True


@njit(cache=True)
def _closest(P, m, lam_out):
    """closest point to the origin on conv(P[:m]) (m <= 4); lam_out[:m] = its barycentric weights."""
    best = 1e300
    idx = np.empty(4, np.int64)
    sub = np.empty(4)
    v = np.zeros(3)
    for mask in range(1, 1 << m):
        k = 0
        for i in range(m):
            if (mask >> i) & 1:
                idx[k] = i
                k += 1
        if not _solve_subset(P, idx, k, sub):
            continue
        ok = True
        for j in range(k):
            if sub[j] < -1e-13:
                ok = False
                break
        if not ok:
            continue
        x = 0.0
        y = np.zeros(3)
        for j in range(k):
            for c in range(3):
                y[c] += sub[j] * P[idx[j], c]
        for c in range(3):
            x += y[c] * y[c]
        if x < best:
            best = x
            for i in range(m):
                lam_out[i] = 0.0
            for j in range(k):
                lam_out[idx[j]] = max(sub[j], 0.0)
            for c in range(3):
                v[c] = y[c]
    return v


@njit(cache=True)
def gjk(Wa, Wb, tol, maxit):
    na = Wa.shape[0]
    nb = Wb.shape[0]
    v = np.empty(3)
    for c in range(3):
        v[c] = Wa[0, c] - Wb[0, c]
    S = np.empty((5, 3))
    for c in range(3):
        S[0, c] = v[c]
    m = 1
    lam = np.zeros(5)
    w = np.empty(3)
    Pt = np.empty((5, 3))
    for _ in range(maxit):
        nv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
        if nv < 1e-12:
            return 0.0
        # support of A - B in direction -v: argmax Wa.(-v) , argmin Wb.(-v)
        ia = 0
        best = Wa[0, 0] * -v[0] + Wa[0, 1] * -v[1] + Wa[0, 2] * -v[2]
        for i in range(1, na):
            s = Wa[i, 0] * -v[0] + Wa[i, 1] * -v[1] + Wa[i, 2] * -v[2]
            if s > best:
                best = s
                ia = i
        ib = 0
        best = Wb[0, 0] * -v[0] + Wb[0, 1] * -v[1] + Wb[0, 2] * -v[2]
        for i in range(1, nb):
            s = Wb[i, 0] * -v[0] + Wb[i, 1] * -v[1] + Wb[i, 2] * -v[2]
            if s < best:
                best = s
                ib = i
        for c in range(3):
            w[c] = Wa[ia, c] - Wb[ib, c]
        if nv - (v[0] * w[0] + v[1] * w[1] + v[2] * w[2]) < tol + 1e-9 * nv:
            break
        for c in range(3):
            S[m, c] = w[c]
        m += 1
        v = _closest(S, m, lam)
        k = 0
        for i in range(m):
            if lam[i] > 1e-12:
                for c in range(3):
                    Pt[k, c] = S[i, c]
                k += 1
        # recompute v from kept (as the scipy version: lam[keep] @ P[keep])
        for c in range(3):
            v[c] = 0.0
        kk = 0
        for i in range(m):
            if lam[i] > 1e-12:
                for c in range(3):
                    v[c] += lam[i] * S[i, c]
                kk += 1
        for i in range(k):
            for c in range(3):
                S[i, c] = Pt[i, c]
        m = k
    return np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
