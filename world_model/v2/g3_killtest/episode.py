"""episode.py -- belief (point-mass particle filter over the theta grid), undecided band U,
robust-tube certification, preposterior predictors (G3 value-level dU, theta-variance,
EIG, theta-width, SafeOpt-style expander), policies, counterfactual calibration.

Definitions (fixed by PREREG.md):
  * route = K_SEG min-jerk segments (+ dwell); nominal plan from segment k = route_k.
  * For particle i, W_it = running min over the plan prefix of V(s_t(theta_i); theta_i).
    p_t = P_post(W_t <= 0). Step t UNDECIDED iff QL < p_t < 1-QL  (QL = DELTA/2; the
    two-sided (1-DELTA) credible interval of the prefix value straddles 0).
    U_k = undecided fraction of route_k steps.
  * Robust tube: a chunk is CERTIFIED iff P_post(min V over chunk <= 0) <= QL. Segments are
    executed at the smallest time-scaling s in SCALES that certifies (a -> a/s^2).
  * Actions at decision point k: (a) execute segment k (as certified); (b) a certified probe;
    (c) neither (robust; same execution as (a), no probe). dU is measured "in time":
    for (a) the undecided steps of segment k itself stay counted (info arrives too late).
SIMULATION ONLY.
"""
from __future__ import annotations

import zlib

import numpy as np

from world_model.v2.g3_killtest import core as C
from world_model.v2.g3_killtest import config as K

DELTA = 0.05
QL = DELTA / 2
M_PRE = 64
SCALES = np.array([1.0, 1.1, 1.2, 1.35, 1.5, 1.7, 2.0, 2.4, 3.0, 4.0])
K_SEG = 5
DWELL = 25
SETTLE = 60
PROBE_A = np.round(np.arange(1.5, 6.51, 0.5), 2)
PROBE_TAU = (0.15, 0.3, 0.6)
PROBES = [(float(A), float(tau)) for tau in PROBE_TAU for A in PROBE_A]
MAX_PROBES_PER_DP = 2
MAX_PROBES_EP = 6
TAU_RULE = 0.5
U_GATE = 0.05          # gated policies only deliberate when U_k >= U_GATE
VOI_TOP = 4            # VOI evaluates (a) + the top-4 probes by predicted dU
K1_TOL = 0.05          # K1 primary: segment undecided at arrival iff >= 5% of its steps undecided
SO_L = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
SO_L_POLICY = 1.0
BACKUP_STEPS = 50
MAX_BACKUPS = 4
DG_WIDTH_GOAL = 0.10
DG_PROBE_MAX_S = 1.0
DG_BUDGET_S = 3.0
DG_TAU = 0.25
MJ_PEAK = 10 * np.sqrt(3) / 3
EMPTY = np.zeros(0, np.int64)

# mismatch arm (secondary): executed accel = gain * lag(a_cmd) + OU tracking noise
MM_LAG_S = 0.04
MM_OU_STD = 0.25
MM_OU_TC = 0.10
MM_GAIN = (0.92, 1.08)


# ============================================================================ routes / probes
def make_route(seed):
    rng = np.random.default_rng([int(seed), 1])
    typ = str(rng.choice(["ramp", "late", "random", "early"], p=[0.35, 0.25, 0.25, 0.15]))
    A0 = rng.uniform(0.8, 1.4)
    if typ == "ramp":
        A = np.sort(rng.uniform(1.5, 6.5, 4))
    elif typ == "late":
        A = np.r_[rng.uniform(1.0, 2.5, 3), rng.uniform(3.0, 6.5)]
    elif typ == "random":
        A = rng.uniform(1.5, 6.5, 4)
    else:
        A = np.sort(rng.uniform(1.5, 6.5, 4))[::-1]
    A = np.r_[A0, A]
    T = rng.uniform(0.5, 1.1, K_SEG)
    dirs = rng.choice([-1.0, 1.0], K_SEG)
    return dict(type=typ, segs=[dict(A=float(A[i]), T=float(T[i]), dir=float(dirs[i])) for i in range(K_SEG)])


def seg_chunk(seg, s, frac0=0.0):
    """Segment profile on [frac0, 1] (nominal time fraction), time-scaled by s, + dwell."""
    Tn, A = seg["T"], seg["A"]
    d = A * Tn ** 2 / MJ_PEAK
    n = max(1, int(round(s * Tn * (1 - frac0) / C.DT)))
    tau = frac0 + (1 - frac0) * (np.arange(n) + 0.5) / n
    a = seg["dir"] * d / Tn ** 2 * (60 * tau - 180 * tau ** 2 + 120 * tau ** 3) / s ** 2
    return np.r_[a, np.zeros(DWELL)]


def probe_chunk(b):
    A, tau = PROBES[b]
    n = int(round(tau / C.DT))
    return np.r_[A * np.sin(2 * np.pi * (np.arange(n) + 0.5) / n), np.zeros(SETTLE)]


def nominal_seq(segs, k):
    parts = [seg_chunk(segs[j], 1.0) for j in range(k, len(segs))]
    bounds = np.cumsum([0] + [len(p) for p in parts])[:-1].astype(np.int64)
    if not parts:
        return np.zeros(0), bounds, []
    return np.concatenate(parts), bounds, [len(p) for p in parts]


def key_rng(seed, key, salt=0):
    return np.random.default_rng([int(seed), zlib.crc32(repr(key).encode()), int(salt)])


# ============================================================================ belief
PF_REP = {"T": 2, "S": 2, "P": 1}        # particles per theta-grid point at t=0
PF_JIT0 = 0.25                            # roughening: state jitter = PF_JIT0 * sigma (observed coord)
PF_JIT1_RATE = {"T": 10.0, "S": 10.0, "P": 6.0}   # velocity jitter = PF_JIT0 * sigma * rate
PF_ROUGH = 0.05                           # prob. of moving to a neighbouring theta-grid point at resampling


class Belief:
    """Sequential PF over (theta-grid index, payload state). Grid arrays are shared (not copied)."""

    def __init__(self, cid, THg, Vg, AXg, cls, frozen=False, rep=None, n_th1=1):
        self.cid, self.THg, self.Vg, self.AXg, self.cls, self.frozen = cid, THg, Vg, AXg, cls, frozen
        G = len(THg)
        rep = PF_REP[cls] if rep is None else rep
        self.IDX = np.repeat(np.arange(G), rep).astype(np.int64)
        N = len(self.IDX)
        self.S = np.zeros((N, 2))
        self.logw = np.full(N, -np.log(N))
        self.n_th1 = n_th1
        self._E = None

    @property
    def G(self):
        return len(self.THg)

    def w(self):
        lw = self.logw - self.logw.max()
        w = np.exp(lw)
        return w / w.sum()

    def E(self):
        """sparse one-hot (N x G): particle -> theta-grid point."""
        if self._E is None:
            from scipy.sparse import csr_matrix
            N = len(self.IDX)
            self._E = csr_matrix((np.ones(N), (np.arange(N), self.IDX)), shape=(N, self.G))
        return self._E

    def wgrid(self, W=None):
        W = self.w()[None, :] if W is None else W
        return np.asarray((self.E().T @ W.T).T)

    def copy(self):
        b = object.__new__(Belief)
        b.__dict__ = dict(self.__dict__)
        b.S = self.S.copy()
        b.logw = self.logw.copy()
        b.IDX = self.IDX.copy()
        return b

    def sub(self, j):
        """single-particle belief (particle j)"""
        b = self.copy()
        b.IDX = self.IDX[j:j + 1].copy()
        b.S = self.S[j:j + 1].copy()
        b.logw = np.zeros(1)
        b._E = None
        return b


def theta_stats(Wg, cls):
    """Wg (M,G) GRID posterior weights -> normalized variance trace (M,), normalized 95% width (M,)."""
    c = K.CLASSES[cls]
    TH = K.particle_grid(cls)
    var = np.zeros(Wg.shape[0])
    width = np.zeros(Wg.shape[0])
    D = len(c["prior"])
    if D == 1:
        margs = [(Wg, TH[:, 0].copy())]
    else:
        n0, n1 = c["grid"]
        W3 = Wg.reshape(-1, n0, n1)
        margs = [(W3.sum(2), TH.reshape(n0, n1, 2)[:, 0, 0].copy()),
                 (W3.sum(1), TH.reshape(n0, n1, 2)[0, :, 1].copy())]
    for d, (Wd, ax) in enumerate(margs):
        lo, hi = c["prior"][d]
        m = Wd @ ax
        v = Wd @ (ax ** 2) - m ** 2
        var += np.maximum(v, 0) / ((hi - lo) ** 2 / 12.0)
        cdf = np.cumsum(Wd, 1)
        il = np.argmax(cdf >= QL, 1)
        ih = np.argmax(cdf >= 1 - QL, 1)
        width += (ax[ih] - ax[il]) / (hi - lo)
    return var / D, width / D


def entropy(w):
    w = np.asarray(w).ravel()
    w = w[w > 0]
    return float(-(w * np.log(w)).sum())


def logsumexp(X, axis):
    m = X.max(axis, keepdims=True)
    return (m + np.log(np.exp(X - m).sum(axis, keepdims=True))).squeeze(axis)


# ============================================================================ episode
class Episode:
    """One (class, sigma, seed, arm) episode context shared by all policy runs."""

    def __init__(self, cls, sigma_idx, seed, tables, arm="main"):
        self.cls, self.seed, self.arm = cls, int(seed), arm
        c = K.CLASSES[cls]
        self.cid = c["cid"]
        self.sigma = float(c["sigmas"][sigma_idx])
        self.sigma_label = c["sigma_labels"][sigma_idx]
        self.THg, self.Vg, self.AXg = tables
        self.n_th1 = c["grid"][1] if len(c["grid"]) > 1 else 1
        rng = np.random.default_rng([self.seed, 2])
        self.theta = K.sample_theta(cls, rng)
        self.route = make_route(self.seed)
        self.segs = self.route["segs"]
        self.gain = float(np.random.default_rng([self.seed, 3]).uniform(*MM_GAIN)) if arm == "mismatch" else 1.0
        self.nom_time = sum(len(seg_chunk(s, 1.0)) for s in self.segs) * C.DT
        self.jit0 = PF_JIT0 * self.sigma
        self.jit1 = PF_JIT0 * self.sigma * PF_JIT1_RATE[cls]
        self.info_cache = {}
        self.real_cache = {}
        self._oracle = None

    def belief(self, frozen=False):
        return Belief(self.cid, self.THg, self.Vg, self.AXg, self.cls, frozen=frozen, n_th1=self.n_th1)

    def oracle_belief(self):
        if self._oracle is None:
            TH = self.theta[None, :].copy()
            AX = C.make_axes(self.cid, TH, K.N_GRID, K.N_GRID)
            V, _, _ = C.solve_one(self.cid, TH[0, 0], TH[0, 1], AX[0, 0], AX[0, 1], K.N_GRID, K.N_GRID, C.U_FB,
                                  K.HJ_ITERS, K.HJ_TOL)
            self._oracle = (TH, V[None], AX)
        TH, V, AX = self._oracle
        return Belief(self.cid, TH, V, AX, self.cls, frozen=True, rep=1)

    def exec_filter(self, a_cmd, rng, st):
        if self.arm != "mismatch":
            return a_cmd
        out = np.empty_like(a_cmd)
        al, ou = st.lag, st.ou
        k = C.DT / MM_LAG_S
        rho = np.exp(-C.DT / MM_OU_TC)
        z = rng.normal(0, 1, len(a_cmd))
        for t in range(len(a_cmd)):
            al += k * (a_cmd[t] - al)
            ou = rho * ou + np.sqrt(1 - rho ** 2) * MM_OU_STD * z[t]
            out[t] = self.gain * al + ou
        st.lag, st.ou = al, ou
        return out


class State:
    def __init__(self, ep: Episode, belief: Belief):
        self.s = np.zeros(2)
        self.B = belief
        self.t = 0
        self.failed = False
        self.fail_t = None
        self.lmin = 1.0
        self.lag = 0.0
        self.ou = 0.0
        self.executed = []          # (excitation level, grid-level min margin) for the SafeOpt frontier
        self.probe_time = 0.0
        self.backup_time = 0.0
        self.n_resample = 0

    def copy(self):
        s = object.__new__(State)
        s.__dict__ = dict(self.__dict__)
        s.s = self.s.copy()
        s.B = self.B.copy()
        s.executed = list(self.executed)
        return s


def _grid_min(B, lmin):
    out = np.full(B.G, np.nan)
    np.fmin.at(out, B.IDX, lmin)
    return out


def _pf(ep, st, a_seq, cam_off, Y, key):
    B = st.B
    seed = int(key_rng(ep.seed, key, salt=31).integers(2 ** 31 - 1))
    lmin, nrs = C.pf_chunk(ep.cid, B.S, B.IDX, B.THg, B.AXg, B.Vg, B.logw, np.ascontiguousarray(a_seq, dtype=np.float64),
                           cam_off, np.ascontiguousarray(Y, dtype=np.float64), ep.sigma, not B.frozen,
                           ep.jit0, ep.jit1, PF_ROUGH, B.n_th1, B.G, seed)
    B._E = None
    st.n_resample += nrs
    return lmin


def execute(ep: Episode, st: State, chunk, key):
    """Execute a commanded accel chunk on the truth; PF-update the belief from camera frames."""
    rng = key_rng(ep.seed, key)
    cam_off = st.t % C.CAM_EVERY
    a_exec = ep.exec_filter(chunk, rng, st)
    s_start = st.s.copy()
    s0, s1, OBS, lm, fail, amp, moved = C.true_rollout(ep.cid, st.s[0], st.s[1], ep.theta[0], ep.theta[1],
                                                       a_exec, cam_off)
    y = OBS + rng.normal(0, ep.sigma, OBS.shape)
    lmin = _pf(ep, st, chunk, cam_off, y, key)
    ref = s_start[0] if ep.cid == C.SLIDE else 0.0
    snr2 = float(((OBS - ref) ** 2).sum() / ep.sigma ** 2) if OBS.size else 0.0
    if fail >= 0 and not st.failed:
        st.failed = True
        st.fail_t = (st.t + fail) * C.DT
    st.t += len(chunk)
    st.s = np.array([s0, s1])
    st.lmin = min(st.lmin, lm)
    st.executed.append((float(np.abs(chunk).max()), _grid_min(st.B, lmin)))
    return dict(moved=bool(moved), amp=float(amp), snr2=snr2, lmin=float(lm), failed=bool(fail >= 0))


def execute_stepwise(ep: Episode, st: State, n_max, ctrl, key):
    """Closed-loop execution; ctrl(st, i) -> accel or None (stop). Returns steps executed."""
    rng = key_rng(ep.seed, key)
    n = 0
    for i in range(n_max):
        a = ctrl(st, i)
        if a is None:
            return i
        a_exec = ep.exec_filter(np.array([a]), rng, st)[0]
        s0, s1 = C.step(ep.cid, st.s[0], st.s[1], a_exec, ep.theta[0], ep.theta[1], C.DT, C.SUB)
        st.s = np.array([s0, s1])
        lv = C.margin(ep.cid, s0, s1, ep.theta[0], ep.theta[1])
        st.lmin = min(st.lmin, lv)
        if lv <= 0 and not st.failed:
            st.failed = True
            st.fail_t = (st.t + 1) * C.DT
        cam_off = st.t % C.CAM_EVERY
        frame = (cam_off + 1) % C.CAM_EVERY == 0
        Y = np.array([s0 + rng.normal(0, ep.sigma)]) if frame else np.zeros(0)
        _pf(ep, st, np.array([float(a)]), cam_off, Y, key + (i,))
        st.t += 1
        n = i + 1
        if st.failed:
            return n
    return n


def _rollout(ep, B, a_seq, t_split, cam_off, rec=EMPTY, S=None):
    S = B.S if S is None else S
    return C.rollout(ep.cid, S, B.IDX, B.THg, B.AXg, B.Vg, np.ascontiguousarray(a_seq, dtype=np.float64),
                     t_split, cam_off, rec)


def certify(ep: Episode, st: State, chunk_fn, B=None):
    """Smallest s in SCALES whose chunk is certified under belief B (default st.B)."""
    B = st.B if B is None else B
    w = B.w()
    cam_off = st.t % C.CAM_EVERY
    for s in SCALES:
        ch = chunk_fn(s)
        _, _, _, vmin, _, _ = _rollout(ep, B, ch, len(ch), cam_off)
        if w @ (vmin <= 0) <= QL:
            return float(s), ch
    return None, None


def exec_backup(ep, st, key):
    def ctrl(stt, i):
        B = stt.B
        a, _ = C.robust_backup_action(ep.cid, B.S, B.IDX, B.THg, B.AXg, B.Vg, B.w(), C.U_FB, QL)
        return float(a)
    n = execute_stepwise(ep, st, BACKUP_STEPS, ctrl, key)
    st.backup_time += n * C.DT
    return n


def exec_segment(ep, st, k, hist, frac0=0.0, tag="s"):
    seg = ep.segs[k]
    s, ch = certify(ep, st, lambda s_: seg_chunk(seg, s_, frac0))
    n_bk = 0
    uncert = False
    while s is None and n_bk < MAX_BACKUPS and not st.failed:
        hist = hist + (("bk", k, n_bk),)
        exec_backup(ep, st, hist)
        n_bk += 1
        s, ch = certify(ep, st, lambda s_: seg_chunk(seg, s_, frac0))
    if s is None:
        s, uncert = float(SCALES[-1]), True
        ch = seg_chunk(seg, s, frac0)
    hist = hist + ((tag, k, s, round(frac0, 3)),)
    nm = execute(ep, st, ch, hist) if not st.failed else {}
    return hist, s, nm, n_bk, uncert


# ============================================================================ predictions
def so_frontier(st, wg, L):
    """SafeOpt-style Lipschitz frontier in peak-excitation space from executed chunks:
    F = max_c (e_c + LCB_c / L), LCB_c = grid-posterior QL-quantile of the chunk's min margin."""
    F = 0.0
    for e, lg in st.executed:
        ok = ~np.isnan(lg)
        if not ok.any() or wg[ok].sum() <= 0:
            continue
        ww = wg[ok] / wg[ok].sum()
        ell = C.wquantile(lg[ok], ww, QL)
        if ell > 0:
            F = max(F, e + ell / L)
    return F


def prepost(ep, B, w, OBS, Bmat, extra, nT, rng, nsafe0):
    """Preposterior Monte Carlo: sample truth particle j ~ posterior, simulate its (deterministic)
    observations + camera noise, batch-reweight all particles (no resampling), recompute U."""
    N = len(w)
    M = M_PRE
    js = rng.choice(N, M, p=w)
    if OBS.shape[1]:
        Y = OBS[js] + rng.normal(0, ep.sigma, (M, OBS.shape[1]))
        D = (Y * Y).sum(1)[:, None] + (OBS * OBS).sum(1)[None, :] - 2 * Y @ OBS.T
        LL = -np.maximum(D, 0) / (2 * ep.sigma ** 2)
    else:
        LL = np.zeros((M, N))
    lw = np.log(np.maximum(w, 1e-300))
    LW = lw[None, :] + LL
    mx = LW.max(1, keepdims=True)
    Wp = np.exp(LW - mx)
    Wp /= Wp.sum(1, keepdims=True)
    P = Wp @ Bmat
    und = (P > QL) & (P < 1 - QL)
    Up = (und.sum(1) + extra) / nT
    nsafe = (P <= QL).sum(1)
    # EIG on theta (states marginalized within each theta-grid point)
    Eg = B.E()
    A = np.asarray((Eg.T @ np.exp(LW - mx).T).T)                 # (M,G): sum_{i in g} w_i p(y|i) (scaled)
    wg0 = np.asarray(Eg.T @ w).ravel()
    gj = B.IDX[js]
    log_py_theta = np.log(np.maximum(A[np.arange(M), gj], 1e-300)) + mx[:, 0] - np.log(np.maximum(wg0[gj], 1e-300))
    log_py = np.log(np.maximum(A.sum(1), 1e-300)) + mx[:, 0]
    eig = float(np.mean(log_py_theta - log_py))
    Wg = np.asarray((Eg.T @ Wp.T).T)
    var, width = theta_stats(Wg, ep.cls)
    return dict(U=Up, var=var, width=width, eig=max(eig, 0.0), dsafe=float(np.mean(nsafe - nsafe0)) / nT, Wp=Wp)


def route_state(ep, st, k):
    """Current U over route_k + per-step undecided mask + per-segment flags."""
    B = st.B
    w = B.w()
    rk, bounds, lens = nominal_seq(ep.segs, k)
    cam_off = st.t % C.CAM_EVERY
    _, W, _, _, _, Srec = _rollout(ep, B, rk, 0, cam_off, bounds)
    Bm = (W <= 0).astype(np.float64)
    p = w @ Bm
    und = (p > QL) & (p < 1 - QL)
    seg_und = [bool(und[b:b + n].any()) for b, n in zip(bounds, lens)]
    return dict(rk=rk, bounds=bounds, lens=lens, und=und, p=p, U=float(und.mean()) if len(rk) else 0.0,
                seg_und=seg_und, nsafe=int((p <= QL).sum()), Srec=Srec, nT=len(rk), Bm=Bm)


def compute_info(ep: Episode, st: State, k, hist):
    if hist in ep.info_cache:
        return ep.info_cache[hist]
    B = st.B
    w = B.w()
    wg = B.wgrid()[0]
    rs = route_state(ep, st, k)
    nT = rs["nT"]
    len_k = rs["lens"][0]
    cam_off = st.t % C.CAM_EVERY
    var0, width0 = theta_stats(wg[None, :], ep.cls)
    info = dict(k=k, U=rs["U"], seg_und=rs["seg_und"], var0=float(var0[0]), width0=float(width0[0]),
                H0=entropy(wg), actions={})
    Fs = {L: so_frontier(st, wg, L) for L in SO_L}
    e_k = np.maximum.accumulate(np.abs(rs["rk"]))
    rk1, b1, _ = nominal_seq(ep.segs, k + 1)
    e_k1 = np.maximum.accumulate(np.abs(rk1)) if len(rk1) else np.zeros(0)
    und = rs["und"]
    und_after_k = und[len_k:]
    rng = key_rng(ep.seed, hist, salt=11)
    D = len(K.CLASSES[ep.cls]["prior"])
    aux = {}

    def finish(name, chunk, a_seq, t_split, extra, und_mask, e_prof, certified, s=None, seg_ids=(), rec=EMPTY):
        Ssp, W, OBS, vmin, lmin, Srec = _rollout(ep, B, a_seq, t_split, cam_off, rec)
        Bm = (W <= 0).astype(np.float64)
        nsafe0 = int(((w @ Bm) <= QL).sum()) if Bm.shape[1] else 0
        pp = prepost(ep, B, w, OBS, Bm, extra, nT, rng, nsafe0)
        dU = rs["U"] - float(np.mean(pp["U"]))
        ex = float(np.abs(chunk).max())
        u_marg = C.wquantile(lmin, w, 1 - QL)
        so = {}
        for L in SO_L:
            Fx = max(Fs[L], ex + max(u_marg, 0.0) / L)
            so[L] = float((und_mask & (e_prof > Fs[L]) & (e_prof <= Fx)).sum()) / nT
        info["actions"][name] = dict(
            dU=dU, dU_sd=float(np.std(pp["U"])), dU_safe=pp["dsafe"], nsafe0=nsafe0,
            r_g3=dU / rs["U"] if rs["U"] > 0 else 0.0,
            r_tvar=1 - float(np.mean(pp["var"])) / max(info["var0"], 1e-12),
            r_width=1 - float(np.mean(pp["width"])) / max(info["width0"], 1e-12),
            eig=pp["eig"], r_eig=1 - float(np.exp(-2 * pp["eig"] / D)),
            so=so, certified=bool(certified), dur=len(chunk) * C.DT, s=s, excitation=ex,
            p_unsafe_action=float(w @ (vmin <= 0)))
        aux[name] = (pp["Wp"].astype(np.float32), Srec, tuple(seg_ids))

    # (a) next nominal segment, executed as currently certified
    seg = ep.segs[k]
    s_k, ch_k = certify(ep, st, lambda s_: seg_chunk(seg, s_))
    cert_a = s_k is not None
    if not cert_a:
        s_k, ch_k = float(SCALES[-1]), seg_chunk(seg, float(SCALES[-1]))
    # (a): W from the rollout covers route_{k+1} only; segment k's own undecided steps stay counted
    finish("a", ch_k, np.r_[ch_k, rk1], len(ch_k), int(und[:len_k].sum()), und_after_k, e_k1, cert_a, s_k,
           seg_ids=tuple(range(k + 1, len(ep.segs))), rec=(len(ch_k) + b1).astype(np.int64))
    for b in range(len(PROBES)):
        ch = probe_chunk(b)
        _, _, _, vmin, _, _ = _rollout(ep, B, ch, len(ch), cam_off)
        if (w @ (vmin <= 0)) > QL:           # probe not certified-safe under the credible set
            continue
        finish(("p", b), ch, np.r_[ch, rs["rk"]], len(ch), 0, und, e_k, True,
               seg_ids=tuple(range(k, len(ep.segs))), rec=(len(ch) + rs["bounds"]).astype(np.int64))
    info["s_k"] = s_k
    info["cert_a"] = cert_a
    info["useg"] = float(und[:len_k].mean()) if len_k else 0.0
    probes = sorted([x for x in info["actions"] if x != "a"], key=lambda x: -info["actions"][x]["dU"])[:VOI_TOP]
    info["_aux"] = {x: aux[x] for x in ["a"] + probes}
    ep.info_cache[hist] = info
    return info


def voi(ep, st, k, info):
    """One-step lookahead expected remaining time per candidate action (value-level): for each
    hypothetical post-action posterior W'_m, every remaining segment is executed at its smallest
    certified scaling (states along the nominal plan after the action)."""
    if "voi" in info:
        return info["voi"]
    B = st.B
    cam_off = st.t % C.CAM_EVERY
    out = {}
    for x, (Wp, Srec, seg_ids) in info["_aux"].items():
        t0 = len(seg_chunk(ep.segs[k], info["s_k"])) * C.DT if x == "a" else info["actions"][x]["dur"]
        Wp64 = Wp.astype(np.float64)
        tot = np.zeros(Wp.shape[0])
        for jj, j in enumerate(seg_ids):
            seg = ep.segs[j]
            S0 = np.ascontiguousarray(Srec[:, jj, :])
            done = np.zeros(Wp.shape[0], bool)
            for s in SCALES:
                ch = seg_chunk(seg, s)
                _, _, _, vmin, _, _ = _rollout(ep, B, ch, len(ch), cam_off, S=S0)
                pu = Wp64 @ (vmin <= 0).astype(np.float64)
                new = (~done) & (pu <= QL)
                tot[new] += len(ch) * C.DT
                done |= new
                if done.all():
                    break
            tot[~done] += len(seg_chunk(seg, SCALES[-1])) * C.DT
        out[x] = t0 + float(tot.mean())
    info["voi"] = out
    return out


def stake_time(ep, st, k, info):
    """Extra time of the remaining undecided segments under the current robust tube."""
    rs = route_state(ep, st, k)
    tot = 0.0
    for j, flag in enumerate(rs["seg_und"]):
        if not flag:
            continue
        Bj = st.B.copy()
        Bj.S = np.ascontiguousarray(rs["Srec"][:, j, :])
        seg = ep.segs[k + j]
        s, _ = certify(ep, st, lambda s_: seg_chunk(seg, s_), B=Bj)
        s = SCALES[-1] if s is None else s
        tot += (s - 1.0) * seg["T"]
    return tot


# ============================================================================ realized (counterfactual)
def realize(ep: Episode, st: State, k, hist, action, info):
    """Execute `action` counterfactually on a copy (same noise key as a policy taking it) and
    measure realized shrinkage with the SAME in-time accounting as the prediction."""
    ck = (hist, action)
    if ck in ep.real_cache:
        return ep.real_cache[ck]
    st2 = st.copy()
    rs0 = route_state(ep, st, k)
    wg0 = st.B.wgrid()[0]
    len_k = rs0["lens"][0]
    if action == "a":
        seg = ep.segs[k]
        s = info["s_k"]
        nm = execute(ep, st2, seg_chunk(seg, s), hist + (("s", k, s, 0.0),))
        extra = int(rs0["und"][:len_k].sum())
        rs1 = route_state(ep, st2, k + 1) if k + 1 < len(ep.segs) else None
        mask0 = rs0["und"][len_k:]
        rk1, _, _ = nominal_seq(ep.segs, k + 1)
        e_prof = np.maximum.accumulate(np.abs(rk1)) if len(rk1) else np.zeros(0)
    else:
        b = action[1]
        nm = execute(ep, st2, probe_chunk(b), hist + (("p", b),))
        extra = 0
        rs1 = route_state(ep, st2, k)
        mask0 = rs0["und"]
        e_prof = np.maximum.accumulate(np.abs(rs0["rk"]))
    nsafe0 = info["actions"][action]["nsafe0"]
    if rs1 is not None:
        n_und = extra + int(rs1["und"].sum())
        nsafe1 = rs1["nsafe"]
    else:
        n_und, nsafe1 = extra, 0
    wg1 = st2.B.wgrid()[0]
    v0, wd0 = theta_stats(wg0[None, :], ep.cls)
    v1, wd1 = theta_stats(wg1[None, :], ep.cls)
    U1 = n_und / rs0["nT"]
    so_real = {}
    for L in SO_L:
        F0 = so_frontier(st, wg0, L)
        F1 = so_frontier(st2, wg1, L)
        so_real[L] = float((mask0 & (e_prof > F0) & (e_prof <= F1)).sum()) / rs0["nT"]
    out = dict(dU=rs0["U"] - U1, dU_safe=(nsafe1 - nsafe0) / rs0["nT"],
               r_tvar=1 - float(v1[0]) / max(float(v0[0]), 1e-12),
               r_width=1 - float(wd1[0]) / max(float(wd0[0]), 1e-12),
               dH=entropy(wg0) - entropy(wg1), so=so_real, failed=nm.get("failed", False),
               moved=nm.get("moved", False), amp=nm.get("amp", 0.0), snr2=nm.get("snr2", 0.0))
    ep.real_cache[ck] = out
    return out
