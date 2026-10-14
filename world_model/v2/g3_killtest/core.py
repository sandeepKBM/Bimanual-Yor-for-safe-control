"""core.py -- G3 kill test: reduced-order payload dynamics, exact grid HJ, particle kernels.

Planar, horizontal tray. Feedback linearization (tau = M a + c) makes the carry arm an
exact double integrator, so the tray acceleration a(t) is COMMANDED EXACTLY (up to the
torque box, assumed not binding for |a| <= 8 m/s^2). The payload is the only uncertain
subsystem. Three payload classes (theta = uncertain parameters):

  T  TIPPING block/stack on the tray (Housner rocking block, uniform, width 2b known,
     COM height h unknown): at rest while |a| <= g b/h; rocks about a corner otherwise;
     impacts at psi=0 with Housner restitution r = 1 - 1.5 sin^2(alpha). FAIL |psi| >= alpha.
  S  SLIDING loose item (Coulomb mu unknown, mu_s = mu_k): sticks while |a| <= mu g;
     slides otherwise. FAIL |x_rel| >= edge.
  P  PENDULUM-like visible payload (hanging item; omega, zeta unknown):
     phi'' = -w^2 sin(phi) - 2 zeta w phi' - (a w^2/g) cos(phi). FAIL |phi| >= phi_max.

All dynamics: dt = 0.01 s control hold, SUB semi-implicit sub-steps. Everything numba.
Normalized margin l in [M_LO, 1] (l <= 0 <=> failed). SIMULATION ONLY: no hardware.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

G = 9.81
DT = 0.01
SUB = 4
CAM_EVERY = 4            # 25 Hz torso camera
A_FB = 4.0               # fallback |a| bound [m/s^2]
N_U = 17
U_FB = np.linspace(-A_FB, A_FB, N_U)
TIP, SLIDE, PEND = 0, 1, 2
CID = {"T": TIP, "S": SLIDE, "P": PEND}
TIP_B = 0.025            # half-width [m] (known)
SLIDE_EDGE = 0.10        # item starts centred; fails at 10 cm displacement
PEND_PHI_MAX = 0.5       # rad
REST_EPS = 0.05          # rad/s: post-impact angular speed below which the block settles
M_LO = -0.25


# ============================================================================ dynamics
@njit(cache=True)
def step(cid, s0, s1, a, th0, th1, dt, sub):
    h = dt / sub
    if cid == 0:
        alpha = np.arctan(TIP_B / th0)
        R = np.sqrt(TIP_B * TIP_B + th0 * th0)
        k = 0.75 / R                      # m R / I_o with I_o = 4/3 m R^2 (uniform block)
        sa = np.sin(alpha)
        r = 1.0 - 1.5 * sa * sa
        ac = G * TIP_B / th0
        psi = s0
        psid = s1
        for _ in range(sub):
            if psi == 0.0:
                if psid == 0.0:
                    if a > ac:
                        sg = 1.0
                    elif a < -ac:
                        sg = -1.0
                    else:
                        continue                      # stick: exactly at rest
                else:
                    sg = 1.0 if psid > 0.0 else -1.0
            else:
                sg = 1.0 if psi > 0.0 else -1.0
            e = alpha - sg * psi
            psidd = k * (a * np.cos(e) - sg * G * np.sin(e))
            psid_n = psid + h * psidd
            psi_n = psi + h * psid_n
            if sg * psi_n <= 0.0:                     # crossed the flat pose -> impact
                psi = 0.0
                psid = r * psid_n
                if abs(psid) < REST_EPS:
                    psid = 0.0
            else:
                psi = psi_n
                psid = psid_n
        return psi, psid
    elif cid == 1:
        fr = th0 * G
        x = s0
        v = s1
        for _ in range(sub):
            if v == 0.0:
                if abs(a) <= fr:
                    continue                          # stick
                acc = -a + (fr if a > 0.0 else -fr)
                v = h * acc
            else:
                acc = -a - (fr if v > 0.0 else -fr)
                vn = v + h * acc
                if vn * v <= 0.0:                     # friction brings it to rest
                    v = 0.0
                    continue
                v = vn
            x += h * v
        return x, v
    else:
        w2 = th0 * th0
        c = 2.0 * th1 * th0
        ag = a * w2 / G
        phi = s0
        phid = s1
        for _ in range(sub):
            phidd = -w2 * np.sin(phi) - c * phid - ag * np.cos(phi)
            phid += h * phidd
            phi += h * phid
        return phi, phid


@njit(cache=True)
def margin(cid, s0, s1, th0, th1):
    if cid == 0:
        alpha = np.arctan(TIP_B / th0)
        m = (alpha - abs(s0)) / alpha
    elif cid == 1:
        m = (SLIDE_EDGE - abs(s0)) / SLIDE_EDGE
    else:
        m = (PEND_PHI_MAX - abs(s0)) / PEND_PHI_MAX
    if m < M_LO:
        m = M_LO
    return m


def grid_halfwidths(cid, th0, th1):
    """Per-theta state-grid half widths (hi0, hi1); grids are symmetric with an exact 0."""
    if cid == TIP:
        alpha = np.arctan(TIP_B / th0)
        R = np.sqrt(TIP_B ** 2 + th0 ** 2)
        p = np.sqrt(0.75 * G / R)
        return 1.25 * alpha, 2.0 * p * alpha
    if cid == SLIDE:
        return 1.2 * SLIDE_EDGE, 2.0
    return 1.4 * PEND_PHI_MAX, 2.5 * th0 * PEND_PHI_MAX


def make_axes(cid, TH, n0, n1):
    AX = np.empty((len(TH), 2))
    for i, (a, b) in enumerate(TH):
        h0, h1 = grid_halfwidths(cid, a, b)
        AX[i] = (2 * h0 / (n0 - 1), 2 * h1 / (n1 - 1))
    return AX


# ============================================================================ grid HJ
@njit(cache=True)
def interp2(V, d0, d1, x0, x1):
    n0, n1 = V.shape
    f0 = x0 / d0 + 0.5 * (n0 - 1)
    f1 = x1 / d1 + 0.5 * (n1 - 1)
    if f0 < 0.0:
        f0 = 0.0
    if f0 > n0 - 1.000001:
        f0 = n0 - 1.000001
    if f1 < 0.0:
        f1 = 0.0
    if f1 > n1 - 1.000001:
        f1 = n1 - 1.000001
    i0 = int(f0)
    i1 = int(f1)
    t0 = f0 - i0
    t1 = f1 - i1
    a = V[i0, i1] * (1 - t0) + V[i0 + 1, i1] * t0
    b = V[i0, i1 + 1] * (1 - t0) + V[i0 + 1, i1 + 1] * t0
    return a * (1 - t1) + b * t1


@njit(cache=True)
def solve_one(cid, th0, th1, d0, d1, n0, n1, U, iters, tol):
    """Undiscounted finite-horizon viability value V_n = min(l, max_u V_{n-1}(f(x,u))),
    semi-Lagrangian, bilinear interpolation. Returns V (float32), iterations, residual."""
    c0 = 0.5 * (n0 - 1)
    c1 = 0.5 * (n1 - 1)
    nu = U.shape[0]
    L = np.empty((n0, n1))
    NX0 = np.empty((n0, n1, nu))
    NX1 = np.empty((n0, n1, nu))
    for i in range(n0):
        x0 = (i - c0) * d0
        for j in range(n1):
            x1 = (j - c1) * d1
            L[i, j] = margin(cid, x0, x1, th0, th1)
            for u in range(nu):
                y0, y1 = step(cid, x0, x1, U[u], th0, th1, DT, SUB)
                NX0[i, j, u] = y0
                NX1[i, j, u] = y1
    V = L.copy()
    Vn = np.empty_like(V)
    resid = 0.0
    it = 0
    for it in range(iters):
        resid = 0.0
        for i in range(n0):
            for j in range(n1):
                best = -1e9
                for u in range(nu):
                    v = interp2(V, d0, d1, NX0[i, j, u], NX1[i, j, u])
                    if v > best:
                        best = v
                nv = L[i, j] if L[i, j] < best else best
                dv = abs(nv - V[i, j])
                if dv > resid:
                    resid = dv
                Vn[i, j] = nv
        tmp = V
        V = Vn
        Vn = tmp
        if resid < tol:
            break
    return V.astype(np.float32), it + 1, resid


@njit(parallel=True, cache=True)
def solve_all(cid, TH, AX, n0, n1, U, iters, tol):
    N = TH.shape[0]
    out = np.empty((N, n0, n1), np.float32)
    its = np.empty(N, np.int64)
    res = np.empty(N)
    for k in prange(N):
        V, it, r = solve_one(cid, TH[k, 0], TH[k, 1], AX[k, 0], AX[k, 1], n0, n1, U, iters, tol)
        out[k] = V
        its[k] = it
        res[k] = r
    return out, its, res


# ============================================================================ particle kernels
# Particles: IDX (N,) index into the theta grid (V tables per grid point), states S (N,2).
@njit(cache=True)
def obs_of(cid, s0, s1):
    return s0          # tilt psi [rad] / relative position x [m] / swing phi [rad]


@njit(cache=True)
def n_frames(cam_off, T):
    n = 0
    for t in range(T):
        if (cam_off + t + 1) % CAM_EVERY == 0:
            n += 1
    return n


@njit(cache=True)
def rollout(cid, S, IDX, THg, AXg, Vg, a_seq, t_split, cam_off, rec_idx):
    """Deterministic propagation of every particle through a_seq. Steps [0, t_split) = the
    ACTION (predicted observations at camera frames, min value / min margin recorded);
    steps [t_split, T) = the planned route afterwards (running-min value W).
    rec_idx: absolute step indices at which to record the state (start of that step)."""
    N = S.shape[0]
    T = a_seq.shape[0]
    TR = T - t_split
    nf = n_frames(cam_off, t_split)
    W = np.empty((N, TR), np.float32)
    OBS = np.empty((N, nf))
    Ssp = np.empty((N, 2))
    vmin = np.empty(N)
    lmin = np.empty(N)
    nr = rec_idx.shape[0]
    Srec = np.empty((N, nr, 2))
    for i in range(N):
        g = IDX[i]
        s0 = S[i, 0]
        s1 = S[i, 1]
        th0 = THg[g, 0]
        th1 = THg[g, 1]
        d0 = AXg[g, 0]
        d1 = AXg[g, 1]
        Vi = Vg[g]
        vm = 1e9
        lm = 1e9
        rm = 1e9
        f = 0
        r = 0
        if t_split == 0:
            Ssp[i, 0] = s0
            Ssp[i, 1] = s1
        for t in range(T):
            while r < nr and rec_idx[r] == t:
                Srec[i, r, 0] = s0
                Srec[i, r, 1] = s1
                r += 1
            s0, s1 = step(cid, s0, s1, a_seq[t], th0, th1, DT, SUB)
            v = interp2(Vi, d0, d1, s0, s1)
            if t < t_split:
                if v < vm:
                    vm = v
                lv = margin(cid, s0, s1, th0, th1)
                if lv < lm:
                    lm = lv
                if (cam_off + t + 1) % CAM_EVERY == 0:
                    OBS[i, f] = obs_of(cid, s0, s1)
                    f += 1
                if t == t_split - 1:
                    Ssp[i, 0] = s0
                    Ssp[i, 1] = s1
            else:
                if v < rm:
                    rm = v
                W[i, t - t_split] = rm
        while r < nr:
            Srec[i, r, 0] = s0
            Srec[i, r, 1] = s1
            r += 1
        vmin[i] = vm
        lmin[i] = lm
    return Ssp, W, OBS, vmin, lmin, Srec


@njit(cache=True)
def _systematic(w, u0):
    N = w.shape[0]
    out = np.empty(N, np.int64)
    c = w[0]
    j = 0
    for i in range(N):
        u = (u0 + i) / N
        while u > c and j < N - 1:
            j += 1
            c += w[j]
        out[i] = j
    return out


@njit(cache=True)
def pf_chunk(cid, S, IDX, THg, AXg, Vg, logw, a_seq, cam_off, Y, sigma, update,
             jit0, jit1, p_rough, n_th1, n_grid, seed):
    """Sequential importance resampling over (theta-grid index, state) for an EXECUTED chunk.
    Propagate each step; at camera frames weight by N(y; s0, sigma); if ESS < N/2 do
    systematic resampling + roughening (state jitter for MOVING particles only -- exact-rest
    T/S particles are left at rest; theta-index moves to a grid neighbour w.p. p_rough).
    Mutates S, IDX, logw in place. Returns per-particle min margin over the chunk and the
    number of resampling events."""
    np.random.seed(seed)
    N = S.shape[0]
    T = a_seq.shape[0]
    lmin = np.full(N, 1e9)
    f = 0
    n_rs = 0
    for t in range(T):
        for i in range(N):
            g = IDX[i]
            y0, y1 = step(cid, S[i, 0], S[i, 1], a_seq[t], THg[g, 0], THg[g, 1], DT, SUB)
            S[i, 0] = y0
            S[i, 1] = y1
            lv = margin(cid, y0, y1, THg[g, 0], THg[g, 1])
            if lv < lmin[i]:
                lmin[i] = lv
        if (cam_off + t + 1) % CAM_EVERY == 0:
            if update:
                y = Y[f]
                for i in range(N):
                    d = y - S[i, 0]
                    logw[i] -= d * d / (2.0 * sigma * sigma)
                m = logw.max()
                tot = 0.0
                for i in range(N):
                    tot += np.exp(logw[i] - m)
                lz = m + np.log(tot)
                w = np.empty(N)
                e2 = 0.0
                for i in range(N):
                    logw[i] -= lz
                    w[i] = np.exp(logw[i])
                    e2 += w[i] * w[i]
                if 1.0 / e2 < 0.5 * N:
                    n_rs += 1
                    idx = _systematic(w / w.sum(), np.random.random())
                    S2 = S.copy()
                    I2 = IDX.copy()
                    L2 = lmin.copy()
                    for i in range(N):
                        S[i, 0] = S2[idx[i], 0]
                        S[i, 1] = S2[idx[i], 1]
                        IDX[i] = I2[idx[i]]
                        lmin[i] = L2[idx[i]]
                        logw[i] = -np.log(N)
                    for i in range(N):
                        moving = S[i, 1] != 0.0 or (cid == 2) or (cid == 0 and S[i, 0] != 0.0)
                        if moving:
                            S[i, 0] += jit0 * np.random.standard_normal()
                            if not (cid == 1 and S[i, 1] == 0.0):
                                S[i, 1] += jit1 * np.random.standard_normal()
                        if p_rough > 0.0 and np.random.random() < p_rough:
                            g = IDX[i]
                            if n_th1 > 1 and np.random.random() < 0.5:
                                j1 = g % n_th1 + (1 if np.random.random() < 0.5 else -1)
                                if 0 <= j1 < n_th1:
                                    IDX[i] = (g // n_th1) * n_th1 + j1
                            else:
                                j0 = g // n_th1 + (1 if np.random.random() < 0.5 else -1)
                                if 0 <= j0 < n_grid // n_th1:
                                    IDX[i] = j0 * n_th1 + g % n_th1
            f += 1
    return lmin, n_rs


@njit(cache=True)
def true_rollout(cid, s0, s1, th0, th1, a_seq, cam_off):
    """Ground-truth payload through a_seq. Returns final state, OBS (noise-free) at frames,
    min margin, first failure step (-1 if none), max response amplitude, left-rest flag."""
    T = a_seq.shape[0]
    nf = n_frames(cam_off, T)
    OBS = np.empty(nf)
    f = 0
    lm = 1e9
    fail = -1
    amp = 0.0
    x_start = s0
    moved = False
    for t in range(T):
        s0, s1 = step(cid, s0, s1, a_seq[t], th0, th1, DT, SUB)
        if s1 != 0.0:
            moved = True
        lv = margin(cid, s0, s1, th0, th1)
        if lv < lm:
            lm = lv
        if lv <= 0.0 and fail < 0:
            fail = t
        dev = abs(s0 - x_start) if cid == 1 else abs(s0)
        if dev > amp:
            amp = dev
        if (cam_off + t + 1) % CAM_EVERY == 0:
            OBS[f] = obs_of(cid, s0, s1)
            f += 1
    return s0, s1, OBS, lm, fail, amp, moved


@njit(cache=True)
def next_values(cid, S, IDX, THg, AXg, Vg, a):
    N = S.shape[0]
    out = np.empty(N)
    for i in range(N):
        g = IDX[i]
        y0, y1 = step(cid, S[i, 0], S[i, 1], a, THg[g, 0], THg[g, 1], DT, SUB)
        out[i] = interp2(Vg[g], AXg[g, 0], AXg[g, 1], y0, y1)
    return out


@njit(cache=True)
def values_at(S, IDX, AXg, Vg):
    N = S.shape[0]
    out = np.empty(N)
    for i in range(N):
        g = IDX[i]
        out[i] = interp2(Vg[g], AXg[g, 0], AXg[g, 1], S[i, 0], S[i, 1])
    return out


@njit(cache=True)
def wquantile(vals, w, q):
    idx = np.argsort(vals)
    c = 0.0
    for k in range(idx.shape[0]):
        c += w[idx[k]]
        if c >= q:
            return vals[idx[k]]
    return vals[idx[-1]]


@njit(cache=True)
def robust_backup_action(cid, S, IDX, THg, AXg, Vg, w, U, q):
    """argmax_u of the weighted q-quantile over particles of V(f(x_i,u)) (robust greedy)."""
    best = -1e9
    bu = 0.0
    for k in range(U.shape[0]):
        vals = next_values(cid, S, IDX, THg, AXg, Vg, U[k])
        qv = wquantile(vals, w, q)
        if qv > best:
            best = qv
            bu = U[k]
    return bu, best
