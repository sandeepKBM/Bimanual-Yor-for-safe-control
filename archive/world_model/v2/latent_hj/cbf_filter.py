"""cbf_filter.py -- faithful LatentCBF baseline for the physics-latent HJ scenario.

Reference: Nakamura, Bishop, Man, Johnson, Manchester, Bajcsy, "Generalizing Safety
Beyond Collision-Avoidance via Latent-Space Reachability Analysis" / LatentCBF,
arXiv 2511.18606.  Their filter (one-step, NO planning horizon):

    a* = argmin_a ||a - pi_nom(o)||   s.t.   Q(z,a) - eps >= alpha * (Q(z, pi_dag(z)) - eps)

solved by SAMPLING (zeroth order): draw actions from a MIXTURE of pi_nom and pi_dag (+
noise), keep the ones satisfying the constraint, pick the nearest to nominal; if none is
valid apply pi_dag.  alpha in [0.85, 0.925], eps in 0.05-0.2 (their value scale).

What is reused / what "Q" means here
  * V is OUR reachability value (reach_value.ValueNet, Fisac discounted backup on the analytic
    liquid-proxy latent, gamma = 0.99 = meta['gamma']).  Q(z,a) is its one-step backup:
        Q(z,a) = (1-gamma) l(z) + gamma * min{ l(z), V(F(z,a)) }
    F = ONE Bellman step of the physics latent under the pivot accel / cup ang-vel that the
    candidate JOINT acceleration `a` (ego arm, first FL-MPC block) produces:
        a_piv(j) = Jp a + Jpd (qd0 + j*DT*a),   w(j) = Jr (qd0 + j*DT*a),   j = 0..steps-1
    (exactly HorizonMaps' first-block maps, so d_value / g_cbf_row / f_cbf_sample share the
    same kinematics + the same PendContents._score).  steps=2 x DT=5 ms = 10 ms = the
    Bellman step V was trained with (Pv.dt=0.01, sub=4).
  * pi_nom  = the FL-MPC step's ego first-block joint acceleration with NO cup/value rows
              (the same nominal that c_switch's dry-run filters).
  * pi_dag  = the SAME fallback c_switch uses: argmax_u V(f(z,u)) over the discrete control
              set, mapped to a joint acceleration by damped least squares, clipped +-40.
  * Hard rows: CBFFLMPC captures the (a_ineq, b_ineq) FL-MPC hands the solver (collision,
    torque box, cup-tilt/table/leash if enabled).  A sample is admissible iff it satisfies
    every row that depends ONLY on the first blocks ("step-1 rows", rows_mode="block1")
    or every row ('all'); the rest of the nominal plan is held fixed, and each RHS is relaxed
    to the nominal's own value when the nominal is already infeasible (samples are never
    worse than the nominal on any captured row -> collision safety is not traded away).

Two modes (scenario.py wiring):
  f_cbf_sample : the filter above (sampling); overrides the ego torque of the FL-MPC step.
  g_cbf_row    : the same one-step condition, linearised (autograd, about the previous plan's
                 first block) as ONE slackable row inside FL-MPC via the SAME solver proxy
                 that carries d_value's rows (exact penalty rho_v, trust region, a-cap).
                 Isolates "one-step CBF" vs "horizon value rows".

SIMULATION ONLY. No hardware, no network.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np

from world_model.v2.latent_hj import flmpc_value as fv

mpc_fl = fv.mpc_fl
sc = fv.sc
CBF_MODES = ("f_cbf_sample", "g_cbf_row")
ACT_TOL = 0.05             # [rad/s^2] an override below this counts as "not intervening"
FB_QDD_MAX = 40.0          # == scenario.FB_QDD_MAX (c_switch fallback clip)


# =============================================================================== FL-MPC row capture
class _CaptureProxy(fv._SolverProxy):
    """_SolverProxy that also records the FIRST hard-QP call of a step (the rows FL-MPC builds
    before any relaxation): the exact collision + torque-box (+tilt/table/leash) rows."""

    def solve_constrained_box_qp(self, H, f, lower, upper, a_ineq=None, b_ineq=None, **kw):
        o = self._owner
        if o._cap is None and a_ineq is not None:
            o._cap = dict(A=np.array(a_ineq), b=np.array(b_ineq))
        return super().solve_constrained_box_qp(H, f, lower, upper, a_ineq, b_ineq, **kw)


class CBFFLMPC(fv.ValueFLMPC):
    """ValueFLMPC + capture of the hard rows of the last step (`last_rows`). Used ONLY by the
    CBF modes; identical solver behaviour otherwise."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._cap = None

    @property
    def last_rows(self):
        return self._cap

    def step(self, *args, value_rows=None, **kw):
        self._rows = (None, np.zeros(0)) if value_rows is None else value_rows
        self._cap = None
        real = mpc_fl.sc
        mpc_fl.sc = _CaptureProxy(real, self)
        try:
            out = mpc_fl.FLMPC.step(self, *args, **kw)
        finally:
            mpc_fl.sc = real
        return out


def dry_nominal(fl, args, kw, dummy_row=False):
    """Nominal (no cup/value rows) FL-MPC ego first-block accel WITHOUT touching fl's state
    (shallow copy; FLMPC rebinds, never mutates, its state arrays -- same trick as c_switch).
    dummy_row=True: solve with an INACTIVE value row (0 x <= 1) so the ValueFLMPC trust region /
    a-cap are applied exactly as in g_cbf_row -> the override then isolates the CBF row's effect."""
    fl_dry = copy.copy(fl)
    kw = dict(kw); kw["value_rows"] = None
    if dummy_row:
        kw["value_rows"] = (np.zeros((1, fl.n_blk)), np.ones(1), fl.last_plan_ego.copy())
    fl_dry.step(*args, **kw)
    return fl_dry.a_ego_prev.copy(), fl_dry


# =============================================================================== latent Q
class LatentQ:
    """Q(z,a) = (1-g) l(z) + g min{l(z), V(F(z,a))} for joint accelerations a, batched (torch)."""

    def __init__(self, contents, gamma, steps=2, dt=0.005):
        import torch
        self.t = torch
        self.c, self.gamma, self.steps, self.dt = contents, float(gamma), int(steps), float(dt)
        torch.set_num_threads(1)

    def _z0(self, z0):
        t = self.t
        return [t.tensor(np.asarray(v), dtype=t.float64) for v in z0]

    def margin(self, z0):
        n, nd, c = self._z0(z0)
        return self.c.fn["margin"](n[None].float(), c[None].float(), self.c.P)[0]

    def q(self, z0, jac, qd0, A, grad=False):
        """A (N,6) joint accels -> Q (N,) [, dQ/dA (N,6) when grad, summed-over-batch grads
        = per-row grads because rows are independent]."""
        t = self.t
        Jp, Jr, Jpd, _ = (t.tensor(np.asarray(m), dtype=t.float64) for m in jac)
        qd0t = t.tensor(np.asarray(qd0), dtype=t.float64)
        At = t.tensor(np.asarray(A, float), dtype=t.float64, requires_grad=grad)
        N = At.shape[0]
        n, nd, c = (v[None].expand(N, 3) for v in self._z0(z0))
        l0 = self.margin(z0).double()
        with t.set_grad_enabled(grad):
            for j in range(self.steps):
                qd_j = qd0t[None] + (j * self.dt) * At
                a_piv = At @ Jp.T + qd_j @ Jpd.T          # Jp a + Jpd qd(j)
                w = qd_j @ Jr.T
                n, nd, c = self.c.fn["step"](n, nd, c, a_piv, w, self.c.P, dt=self.dt, sub=self.c.sub,
                                             project=False)
            Vn = self.c._score(n.float(), nd.float(), c.float()).double()
            Q = (1.0 - self.gamma) * l0 + self.gamma * t.minimum(l0, Vn)
            if not grad:
                return Q.detach().numpy(), None
            g = t.autograd.grad(Q.sum(), At)[0]
        return Q.detach().numpy(), g.numpy()


def pi_dagger(net, dyn_fb, z0, jac, qd_l):
    """c_switch's fallback policy (scenario.run_episode, `fb_mode` branch), verbatim: greedy
    argmax_u V(f(z,u)) -> pivot accel / cup ang-vel target -> joint accel by damped LS, +-40."""
    import torch
    from world_model.v2.latent_hj import reach_value as rv
    zt = torch.tensor(np.concatenate(z0), dtype=torch.float32)[None]
    with torch.no_grad():
        _, _, ib = rv.greedy_next(net, dyn_fb, zt)
    u = dyn_fb.U[int(ib[0])].numpy().astype(float)
    Jp, Jr, Jpd, Jrd = jac
    w_now = Jr @ qd_l
    rhs = np.concatenate([u[:3] - Jpd @ qd_l, (u[3:] - w_now) / 0.1 - Jrd @ qd_l])
    J6 = np.vstack([Jp, Jr])
    qdd_fb = J6.T @ np.linalg.solve(J6 @ J6.T + 1e-2 * np.eye(6), rhs)
    return np.clip(qdd_fb, -FB_QDD_MAX, FB_QDD_MAX)


# =============================================================================== controller
class LatentCBF:
    def __init__(self, contents, net, dyn_fb, gamma=0.99, alpha=0.9, eps=0.15, n_samples=7600,
                 steps=2, dt=0.005, sig_lo=0.5, sig_hi=30.0, a_cap=60.0, rows_mode="all", seed=0):
        assert rows_mode in ("block1", "all")
        self.q = LatentQ(contents, gamma, steps, dt)
        self.net, self.dyn_fb = net, dyn_fb
        self.alpha, self.eps, self.N = float(alpha), float(eps), int(n_samples)
        self.sig_lo, self.sig_hi = float(sig_lo), float(sig_hi)
        self.a_cap = None if not a_cap or a_cap <= 0 else float(a_cap)
        self.rows_mode = rows_mode
        self.rng = np.random.default_rng(seed)
        self.last = {}

    # ---- the CBF condition
    def threshold(self, q_pi):
        return self.eps + self.alpha * (q_pi - self.eps)

    # ---- hard rows a candidate must satisfy: R a <= rhs  (block-0 ego accel, rest of plan fixed)
    def hard_rows(self, fl, a_nom):
        cap = fl.last_rows
        NV, nb = fl.NV, fl.n_blk
        A, b = cap["A"], cap["b"]
        x_nom = np.array(fl._last_plan)          # full [x_e; x_i] plan of the nominal solve
        x_nom[:NV] = a_nom                       # post-fallback/backstop accel actually applied
        if self.rows_mode == "block1":
            supp = np.ones(A.shape[1], bool); supp[:NV] = False; supp[nb:nb + NV] = False   # later blocks
            keep = ~np.any(np.abs(A[:, supp]) > 1e-12, axis=1)
            A, b = A[keep], b[keep]
        Ax = A @ x_nom
        R = A[:, :NV]
        rest = Ax - R @ a_nom
        rhs = np.maximum(b, Ax) - rest           # never worse than the nominal on any row
        return R, rhs

    def pi_dag(self, z0, jac, qd_l):
        return pi_dagger(self.net, self.dyn_fb, z0, jac, qd_l)

    # ---- f_cbf_sample ---------------------------------------------------------------------
    def filter(self, fl, z0, qd_l, jac, a_nom, a_pi=None):
        t0 = time.perf_counter()
        NV = fl.NV
        a_pi = self.pi_dag(z0, jac, qd_l) if a_pi is None else a_pi
        N, Nh = self.N, self.N // 2
        cen = np.vstack([np.tile(a_nom, (N - Nh, 1)), np.tile(a_pi, (Nh, 1))])          # mixture
        sig = np.exp(self.rng.uniform(np.log(self.sig_lo), np.log(self.sig_hi), (N, 1)))
        S = cen + sig * self.rng.standard_normal((N, NV))
        if self.a_cap is not None:
            S = np.clip(S, -self.a_cap, self.a_cap)
        S = np.vstack([a_nom[None], a_pi[None], S])              # exact endpoints: idx 0 = nominal, 1 = pi_dag
        Q, _ = self.q.q(z0, jac, qd_l, S)
        q_nom, q_pi = float(Q[0]), float(Q[1])
        thr = self.threshold(q_pi)
        ok_cbf = Q >= thr
        R, rhs = self.hard_rows(fl, a_nom)
        ok_hard = np.all(S @ R.T <= rhs[None] + 1e-9, axis=1)
        ok = ok_cbf & ok_hard
        fb = 0
        if ok.any():
            d = np.linalg.norm(S - a_nom[None], axis=1)
            d[~ok] = np.inf
            i = int(np.argmin(d))
        else:
            if ok_hard[1]:
                i, fb = 1, 1                                       # pi_dag itself (paper)
            else:                                                  # pi_dag breaks a hard row:
                qq = np.where(ok_hard, Q, -np.inf)                 # best-Q hard-admissible sample
                i, fb = int(np.argmax(qq)), 2
        a_star = S[i].copy()
        self.last = dict(q_nom=q_nom, q_pi=q_pi, q_star=float(Q[i]), thr=thr, fb=fb,
                         n_cbf=int(ok_cbf.sum()), n_hard=int(ok_hard.sum()), n_ok=int(ok.sum()),
                         ovr=float(np.linalg.norm(a_star - a_nom)), ms=1e3 * (time.perf_counter() - t0))
        return a_star

    # ---- g_cbf_row ------------------------------------------------------------------------
    def row(self, fl, z0, qd_l, jac):
        """Linearised one-step CBF condition as a slackable ego-block row (A_v x_e <= b_v):
        Q_ref + g^T (a - a_ref) >= eps + alpha (Q(z,pi_dag) - eps);  a_ref = previous plan's
        first block (refreshed every step, like d_value's x_ref)."""
        t0 = time.perf_counter()
        NV, nb = fl.NV, fl.n_blk
        x_ref = fl.last_plan_ego
        a_pi = self.pi_dag(z0, jac, qd_l)
        Qp, _ = self.q.q(z0, jac, qd_l, a_pi[None])
        thr = self.threshold(float(Qp[0]))
        Qr, g = self.q.q(z0, jac, qd_l, x_ref[None, :NV], grad=True)
        Av = np.zeros((1, nb)); Av[0, :NV] = -g[0]
        bv = np.array([float(Qr[0]) - float(g[0] @ x_ref[:NV]) - thr])
        self.last = dict(q_ref=float(Qr[0]), q_pi=float(Qp[0]), thr=thr, gnorm=float(np.linalg.norm(g[0])),
                         ms=1e3 * (time.perf_counter() - t0))
        return (Av, bv, x_ref.copy())

    def q_of(self, z0, jac, qd_l, a):
        Q, _ = self.q.q(z0, jac, qd_l, np.asarray(a, float)[None])
        return float(Q[0])


# =============================================================================== summary
def summarize_cbf(C, mode, kw):
    """extra per-episode metrics (only merged into the row of a CBF mode)."""
    C = {k: np.asarray(v, float) for k, v in C.items()}
    ovr, anom = C["ovr"], C["anom"]
    act = ovr > ACT_TOL
    eng = act.astype(int)
    out = dict(
        act_tol=ACT_TOL, cbf_alpha=kw["alpha"], cbf_eps=kw["eps"], cbf_N=kw["n"], cbf_steps=kw["steps"], cbf_rows=kw["rows"],
        ovr_mean=float(ovr.mean()), ovr_rel=float(ovr.mean() / max(anom.mean(), 1e-9)),
        ovr_p99=float(np.percentile(ovr, 99)), ovr_max=float(ovr.max()),
        anom_mean=float(anom.mean()),
        active_frac=float(act.mean()),
        cbf_unsat_frac=float(np.mean(C["q_star"] < C["thr"] - 1e-9)),      # applied action violates the CBF condition
        nom_unsafe_frac=float(np.mean(C["q_nom"] < C["thr"] - 1e-9)),      # nominal would have violated it
        filter_ms_mean=float(C["fms"].mean()), filter_ms_p99=float(np.percentile(C["fms"], 99)),
        filter_ms_max=float(C["fms"].max()),
        q_pi_mean=float(C["q_pi"].mean()),
    )
    if mode == "f_cbf_sample":
        fb = C["fb"]
        out.update(fb_frac=float(np.mean(fb > 0)), fb_pi_frac=float(np.mean(fb == 1)), fb_hard_frac=float(np.mean(fb == 2)),
                   n_ok_mean=float(C["n_ok"].mean()), n_hard_mean=float(C["n_hard"].mean()),
                   n_cbf_mean=float(C["n_cbf"].mean()))
    else:
        out.update(fb_frac=float(np.mean(C["q_star"] < C["thr"] - 1e-9)))   # no pi_dag switch in the row form
    out.update(engaged_frac=float(act.mean()), switches=int(np.sum(np.abs(np.diff(eng)))))
    return out


# =============================================================================== self-test
def _selftest():
    """Offline sanity checks on the real value net with random states / kinematics:
    (1) alpha=0 valid set == {Q >= eps};  (2) valid sets are NESTED and shrink with alpha
    (when Q(z,pi_dag) > eps);  (3) Q(z,a) is monotone in V(F(z,a));  (4) g_cbf_row's linearised
    row matches Q(a) to first order."""
    import os, sys
    import torch
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from world_model.v2.latent_hj import reach_value as rv, slosh_pend as sp
    scr = Path(os.environ.get("LATENT_HJ_SCR", "/common/users/ss5772/latent_hj_scratch"))
    net, meta = rv.load(scr / "v1" / "value.pt")
    for p in net.parameters():
        p.requires_grad_(False)
    P = sp.PendParams.from_meta(meta["P"]); P.dt, P.sub = 0.01, 4
    contents = fv.PendContents(P, net, "value"); contents._dt = 0.005
    LQ = LatentQ(contents, meta["gamma"], 2, 0.005)
    dyn = rv.PendDyn(P, device="cpu")
    torch.manual_seed(0); rng = np.random.default_rng(0)
    Z = dyn.sample_states(40).numpy()
    n_viol = 0; n_tot = 0; nest_ok = True; frac = {}
    alphas = [0.0, 0.3, 0.6, 0.85, 0.925, 0.99]
    for zi in Z:
        z0 = (zi[0:3], zi[3:6], zi[6:9])
        jac = tuple(rng.standard_normal((3, 6)) * s for s in (1.0, 1.0, 0.3, 0.3))
        qd0 = rng.standard_normal(6) * 0.5
        A = rng.standard_normal((2000, 6)) * 20.0
        Q, _ = LQ.q(z0, jac, qd0, A)
        a_pi = A[int(np.argmax(Q))]
        q_pi = float(Q.max())
        eps = 0.15
        prev = None
        for al in alphas:
            thr = eps + al * (q_pi - eps)
            ok = Q >= thr
            if al == 0.0:
                assert np.array_equal(ok, Q >= eps), "alpha=0 must equal {Q>=eps}"
            if q_pi > eps and prev is not None:
                nest_ok &= bool(np.all(ok <= prev))
            prev = ok
            frac.setdefault(al, []).append(ok.mean())
        # grad check
        a0 = A[0]; Q0, g = LQ.q(z0, jac, qd0, a0[None], grad=True)
        d = rng.standard_normal(6); d /= np.linalg.norm(d); h = 1e-3
        Q1, _ = LQ.q(z0, jac, qd0, (a0 + h * d)[None])
        n_tot += 1
        n_viol += abs((Q1[0] - Q0[0]) / h - g[0] @ d) > 5e-2 * (abs(g[0] @ d) + 1e-3)
    print("valid-set fraction vs alpha (mean over 40 random latents, eps=0.15):",
          {k: round(float(np.mean(v)), 4) for k, v in frac.items()})
    print("nested_ok:", nest_ok, " grad-check violations:", n_viol, "/", n_tot)
    assert nest_ok
    means = [np.mean(frac[a]) for a in alphas]
    assert all(means[i] >= means[i + 1] - 1e-12 for i in range(len(means) - 1)), "valid fraction must not grow with alpha"
    print("SELFTEST OK")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
