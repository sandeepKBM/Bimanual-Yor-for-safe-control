"""flmpc_value.py -- FL-MPC with a LINEARIZED LATENT-VALUE CONSTRAINT (ours).

LatentSafe (Nakamura et al., RSS 2025) filters with a least-restrictive SWITCH:
apply the task action unless Q(z, a_task) < eps, then apply the learned fallback
(bang-bang). Here the learned reachability value instead becomes a slackable
CONSTRAINT inside the FL-MPC QP, next to the hard rows FL-MPC already has
(arm-arm clearance, torque box, table floor, optional cup tilt), so the plan is
minimally and smoothly deformed instead of switched.

---------------------------------------------------------------------------- math
FL-MPC decision vector x = [x_e ; x_i], x_e = ego (cup arm) block accelerations
(NC blocks x NV joints). FL (tau = M a + c) makes the joint prediction EXACT:
    q(t)  = q0 + qd0 t + Phi_q(t)  x_e,     qd(t) = qd0 + Phi_qd(t) x_e,
    qdd(t) = S(t) x_e,   S(t) = (Phi_qd(t+1) - Phi_qd(t)) / dt   (block selector)
Robot part -> contents inputs (FL-MPC's exact double integrator + frozen-at-t0
pivot Jacobians, the same first-order kinematics the tilt/leash rows use):
    a_piv(t) = Jp S(t) x_e + Jpd (qd0 + Phi_qd(t) x_e)          (pivot accel, 3)
    w(t)     = Jr (qd0 + Phi_qd(t) x_e)                          (cup ang. vel, 3)
i.e. a_piv(t) = A0_t + A1_t x_e,  w(t) = W0_t + W1_t x_e  (affine, precomputed).
Contents part (the ONLY place a model of the cup contents enters): latent
z_{k+1} = F(z_k, a_piv(k), w(k)) rolled from the current latent z_0 (analytic
pendulum observer in Stage 0, WM posterior in Stage 1), then the learned
reachability value V at checkpoints k in K (block ends):
    V_k(x_e) = V( F^k(z_0; x_e) ) >= eps,  k in K                     (value rows)
Linearize about the previous plan x_ref (autograd through F and V):
    V_k(x_ref) + g_k^T (x_e - x_ref) >= eps,   g_k = dV_k/dx_e |_{x_ref}
    <=>  -g_k^T x_e - s_k <= V_k(x_ref) - g_k^T x_ref - eps,   s_k >= 0
with an exact-penalty slack s_k (cost rho*s + eps_s*s^2). For rho above the
row's optimal multiplier the solution equals the hard-constrained one whenever
that is feasible; otherwise it is the least-value-violating plan. The torque box
(and every other FL-MPC row) stays HARD -- only value rows get slack. FL-MPC's
own collision-relaxed fallback still runs unchanged when the hard rows clash.
Because the rows are first-order, the ego block is also kept in a TRUST REGION
|x_e - x_ref|_inf <= TRUST (40 rad/s^2) whenever value rows are present (box bounds,
never tighter than the accel box). Measured first-order error of V_k (validate_rows):
~0.003 / 0.03 / 0.12 / 0.34 (V units) for plan perturbations of 2 / 5 / 10 / 20 rad/s^2.
Without the trust region + with rho = 1e5 the rows produced 1000+ rad/s^2 commands
once the state was doomed (smoke dbg2/dbg3); default rho is 1e4.
Refresh (z_0, V_k, g_k, x_ref) at the WM rate (every `refresh_every` sim steps,
13 x 5 ms = 15.4 Hz) and hold the rows between refreshes (zero-order hold,
ContactGuard pattern of world_model/v2/model/shield.py).

---------------------------------------------------------------------------- plumbing
mpc_fl.FLMPC has no extra-inequality hook (only table_rows on the FIRST block,
torque convention, always HARD). ValueFLMPC therefore wraps the ONE solver call
site: during ValueFLMPC.step the module global `mpc_fl.sc` is swapped for a proxy
whose solve_constrained_box_qp appends the value rows + their slack columns and
delegates to the real solver (also inside FLMPC._relaxed_fallback). mpc_fl.py is
NOT edited. A clean optional hook is proposed in flmpc_extra_rows.diff.
SIMULATION ONLY.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO), str(REPO / "experiments" / "ab_smooth_vs_bangbang")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mpc_fl  # noqa: E402
import sim_common as sc  # noqa: E402


# =============================================================================== solver proxy
class _SolverProxy:
    """Forwards every attribute to sim_common except solve_constrained_box_qp."""

    def __init__(self, real, owner):
        self._real, self._owner = real, owner

    def __getattr__(self, k):
        return getattr(self._real, k)

    def solve_constrained_box_qp(self, H, f, lower, upper, a_ineq=None, b_ineq=None, **kw):
        o = self._owner
        Av, bv = o._rows[0], o._rows[1]
        x_ref = o._rows[2] if len(o._rows) > 2 else None
        if x_ref is not None and o.TRUST is not None:
            # TRUST REGION on the ego block around the linearization point: the
            # value rows are first-order in x_e, so keep x_e within +-TRUST of x_ref
            # (never tighter than the original box; x_ref itself is always inside).
            lower = lower.copy(); upper = upper.copy()
            nb = len(x_ref)
            lo_t, hi_t = x_ref - o.TRUST, x_ref + o.TRUST
            if o.A_CAP is not None:
                # absolute ego-accel cap while value rows are active: the trust region is
                # re-anchored every refresh, so alone it lets the plan ratchet +-TRUST per
                # step once the state is doomed (20-seed run: 3/20 blow-ups, |a| > 1000)
                lo_t = np.clip(lo_t, -o.A_CAP, o.A_CAP); hi_t = np.clip(hi_t, -o.A_CAP, o.A_CAP)
            lower[:nb] = np.maximum(lower[:nb], np.minimum(lo_t, upper[:nb] - 1e-9))
            upper[:nb] = np.minimum(upper[:nb], np.maximum(hi_t, lower[:nb] + 1e-9))
        if Av is None or len(bv) == 0:
            x, lam, feas = self._real.solve_constrained_box_qp(H, f, lower, upper, a_ineq, b_ineq, **kw)
            o._last_plan = np.array(x[:o._ntot]); o._last_slack = np.zeros(0); o._last_lam_v = np.zeros(0)
            return x, lam, feas
        n = H.shape[0]
        m = len(bv)
        na = n + m
        H2 = np.zeros((na, na)); H2[:n, :n] = H
        H2[n:, n:] = 2.0 * o.EPS_V_SLACK * np.eye(m)
        f2 = np.concatenate([f, o.RHO_V * np.ones(m)])
        lo2 = np.concatenate([lower, np.zeros(m)])
        hi2 = np.concatenate([upper, o.S_MAX_V * np.ones(m)])
        A0 = np.zeros((0, na)) if a_ineq is None else np.hstack([a_ineq, np.zeros((a_ineq.shape[0], m))])
        rows = np.zeros((m, na))
        rows[:, :Av.shape[1]] = Av                    # ego block columns (0 : n_blk)
        rows[:, n:] = -np.eye(m)
        A2 = np.ascontiguousarray(np.vstack([A0, rows]))
        b2 = np.concatenate([np.zeros(0) if b_ineq is None else b_ineq, bv])
        y, lam, feas = self._real.solve_constrained_box_qp(H2, f2, lo2, hi2, A2, b2, **kw)
        o._last_plan = np.array(y[:o._ntot]); o._last_slack = np.array(y[n:])
        o._last_lam_v = np.array(lam[-m:]) if lam is not None and len(lam) >= m else np.zeros(m)
        return y[:n], (lam[:-m] if lam is not None and len(lam) >= m else lam), feas


class ValueFLMPC(mpc_fl.FLMPC):
    """FLMPC + externally supplied ego-block inequality rows (A_v x_e <= b_v, slackable)."""

    def __init__(self, model, rho_v=1.0e4, eps_v_slack=1.0e-3, s_max_v=1.0e3, trust=40.0, a_cap=60.0, **kw):
        super().__init__(model, **kw)
        # s_max_v large: a slack that can hit its cap turns the row HARD again (with
        # s_max 10 a doomed state (V=-1) forced extreme plans -- 20-seed run)
        self.RHO_V, self.EPS_V_SLACK, self.S_MAX_V = float(rho_v), float(eps_v_slack), float(s_max_v)
        self.TRUST = None if trust is None else float(trust)
        self.A_CAP = None if a_cap is None else float(a_cap)
        self._rows = (None, np.zeros(0))
        self._ntot = 2 * self.n_blk
        self._last_plan = np.zeros(self._ntot)
        self._last_slack = np.zeros(0)
        self._last_lam_v = np.zeros(0)

    def step(self, *args, value_rows=None, **kw):
        self._rows = (None, np.zeros(0)) if value_rows is None else value_rows
        real = mpc_fl.sc
        mpc_fl.sc = _SolverProxy(real, self)
        try:
            out = super().step(*args, **kw)
        finally:
            mpc_fl.sc = real
        return out

    @property
    def last_plan_ego(self):
        return self._last_plan[:self.n_blk].copy()


# =============================================================================== horizon maps
class HorizonMaps:
    """Affine maps x_e -> (a_piv(t), w(t)), t = 0..NP-1, from FLMPC's exact DI
    tensors and the pivot Jacobians at t0 (see module docstring)."""

    def __init__(self, fl: mpc_fl.FLMPC):
        self.NP, self.NV, self.n_blk, self.DT, self.L = fl.NP, fl.NV, fl.n_blk, fl.DT, fl.L
        Pq, Pqd = fl.Phi_q_all, fl.Phi_qd_all
        self.S = (Pqd[1:] - Pqd[:-1]) / fl.DT          # (NP, NV, n_blk)
        self.Pqd = Pqd[:-1]                              # velocity at the start of step t
        self.checkpoints = np.arange(fl.L, fl.NP + 1, fl.L) - 1   # indices into the NP rollout (block ends)

    def build(self, qd0, Jp, Jr, Jpd, Jrd=None):
        A1 = np.einsum("ij,tjk->tik", Jp, self.S) + np.einsum("ij,tjk->tik", Jpd, self.Pqd)
        A0 = np.tile(Jpd @ qd0, (self.NP, 1))
        W1 = np.einsum("ij,tjk->tik", Jr, self.Pqd)
        W0 = np.tile(Jr @ qd0, (self.NP, 1))
        return A0, A1, W0, W1


def pivot_jacobians(model, data, ee_body, pivot_world, dofadr):
    import mujoco
    jp = np.zeros((3, model.nv)); jr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jp, jr, pivot_world, ee_body)
    jpd = np.zeros((3, model.nv)); jrd = np.zeros((3, model.nv))
    mujoco.mj_jacDot(model, data, jpd, jrd, pivot_world, ee_body)
    return jp[:, dofadr], jr[:, dofadr], jpd[:, dofadr], jrd[:, dofadr]


# =============================================================================== contents models
class PendContents:
    """Contents model = analytic liquid proxy (Stage 0 latent), torch-differentiable.
    score='value' -> learned reachability V; score='margin' -> instantaneous l
    (horizon-margin physics oracle, mode e)."""

    def __init__(self, P, value_net=None, score="value", sub=2):
        import torch
        from world_model.v2.latent_hj import slosh_pend as sp
        self.torch, self.P, self.net, self.score, self.sub = torch, P, value_net, score, sub
        self.fn = sp.torch_fns()
        torch.set_num_threads(1)

    def _score(self, n, nd, c):
        if self.score == "margin" or self.net is None:
            return self.fn["margin"](n, c, self.P)
        x = self.fn["feats"](n, nd, c, nd_scale=1.0 / self.P.w1)
        return self.net(x, self.fn["margin"](n, c, self.P))

    def value_now(self, z0):
        t = self.torch
        n, nd, c = (t.tensor(v, dtype=t.float64)[None] for v in z0)
        with t.no_grad():
            return float(self._score(n.float(), nd.float(), c.float())[0])

    def rollout_scores(self, z0, A0, A1, W0, W1, x, ck, need_grad=True):
        """-> (scores at checkpoints (m,), jacobian d scores/dx (m, n_blk) or None)."""
        t = self.torch
        xt = t.tensor(x, dtype=t.float64, requires_grad=need_grad)
        A0t, A1t, W0t, W1t = (t.tensor(v, dtype=t.float64) for v in (A0, A1, W0, W1))
        a = A0t + t.einsum("tik,k->ti", A1t, xt)
        w = W0t + t.einsum("tik,k->ti", W1t, xt)
        n, nd, c = (t.tensor(v, dtype=t.float64) for v in z0)
        dt = self._dt
        out = []
        cks = set(int(k) for k in ck)
        for k in range(a.shape[0]):
            n, nd, c = self.fn["step"](n, nd, c, a[k], w[k], self.P, dt=dt, sub=self.sub, project=False)
            if k in cks:
                out.append(self._score(n[None].float(), nd[None].float(), c[None].float())[0].double())
        s = t.stack(out)
        if not need_grad:
            return s.detach().numpy(), None
        J = t.stack([t.autograd.grad(s[i], xt, retain_graph=i < len(out) - 1)[0] for i in range(len(out))])
        return s.detach().numpy(), J.numpy()

    _dt = 0.005


# =============================================================================== value-row manager
class ValueConstraint:
    """Builds/holds the linearized value rows (A_v x_e <= b_v) at the WM rate."""

    def __init__(self, fl: ValueFLMPC, contents, eps=0.05, refresh_every=13):
        self.fl, self.contents, self.eps, self.refresh_every = fl, contents, float(eps), int(refresh_every)
        self.maps = HorizonMaps(fl)
        self.contents._dt = fl.DT
        self.k = 0
        self.rows = (None, np.zeros(0))
        self.last = dict(V=np.nan, V_ck=None, t_ms=0.0, refreshed=False)

    def update(self, z0, qd0, jac):
        """Call every control step BEFORE fl.step; returns value_rows for fl.step."""
        refreshed = False
        t0 = time.perf_counter()
        if self.k % self.refresh_every == 0 or self.rows[0] is None:
            A0, A1, W0, W1 = self.maps.build(qd0, *jac)
            x_ref = self.fl.last_plan_ego
            s, J = self.contents.rollout_scores(z0, A0, A1, W0, W1, x_ref, self.maps.checkpoints)
            A_v = -J
            b_v = s - J @ x_ref - self.eps
            self.rows = (A_v, b_v, x_ref.copy())
            self.last.update(V_ck=s, J_norm=float(np.linalg.norm(J)))
            refreshed = True
        self.k += 1
        self.last.update(t_ms=1e3 * (time.perf_counter() - t0), refreshed=refreshed)
        return self.rows

    def plan_q(self, z0, qd0, jac, x_plan):
        """Q(z, plan): min over checkpoints of the score along the given plan (switching filter)."""
        A0, A1, W0, W1 = self.maps.build(qd0, *jac)
        s, _ = self.contents.rollout_scores(z0, A0, A1, W0, W1, x_plan, self.maps.checkpoints, need_grad=False)
        return float(np.min(s)), s
