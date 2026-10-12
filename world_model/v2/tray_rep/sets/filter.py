"""sets/filter.py -- F5: backup-reachability safety filter on analytic ball sets. SIMULATION / SAVED DATA ONLY. CPU. No hardware.

Bridge interface (like F0-F4):   f(obs_hist, act_hist, u_nom, t) -> u (6,) = [tray accel (3), tray tilt rate (3)];  begin(ctx);  stats().

Each tick
  1. estimate the ball (estimate.BallKF: Kalman filter on the obs stream, covariance grows while samples are missing);
  2. predict the reachable sets of the ball for the plan "apply u_nom now, then the BACKUP law" (reach.ReachModel) -> a Prediction;
  3. evaluate the tray spec (tray_spec.TRAY_SPEC_FULL, compiled by the IR) on the predicted sets:
       robust margin >= 0     ->  u = u_nom unchanged  (status 'true')
       otherwise              ->  status 'unknown' (nominal margin >= 0) or 'false' (< 0), and
  4. solve the QP   min |D (u - u_nom)|^2   s.t. the IR's linear rows (support functions of the linearised reachable sets, every tick of the
     horizon) + the Piper box and slew limits, by qp.project (Hildreth). The solution is checked on the NONLINEAR prediction; if it fails
     the rows are re-linearised at the solution (up to `n_relin` times). Infeasible / still unsafe  ->  the BACKUP command is applied.
`status(obs_hist, u)` exposes the three-valued status of a given command for later BOED / probing: the filter never needs hidden mu_roll.

Spec string (bridge):  "F5"   or   "F5:H=60,k_sigma=2.5,mu_lo=0.0005,mu_hi=0.006,eps_scale=1,r_shrink=0.002,kd=3"
  calib_file=v2 | <path>: another mu_roll -> mu_eff map (default boed/calib.json); eps_a=<m/s^2>: model-error box (default EPS_A_DEFAULT=0.08), times eps_scale.
  calib=0: ablation, raw mu_roll as the resistance (no boed/calib.json map; a built-in sim-to-sim model gap).
"""
from __future__ import annotations

import time

import numpy as np

from world_model.v2.tray_rep import common as K          # noqa: F401
from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep.bridge.filters import Filter, project_box
from world_model.v2.tray_rep.sets import ir, qp
from world_model.v2.tray_rep.sets import reach as RC
from world_model.v2.tray_rep.sets import tray_spec as TS
from world_model.v2.tray_rep.sets.estimate import BallKF

DT = RC.DT
U_COLS = 6
EPS_A_DEFAULT = float(__import__("os").environ.get("BIMANUAL_F5_EPS_A", 0.08))   # 0.08 = the VAL calibration of the old physics (README); env var = rebaseline_v2 override


class F5SetFilter(Filter):
    name = "F5"

    def __init__(self, renv, H=60, k_sigma=2.5, mu_lo=5e-4, mu_hi=6e-3, eps_scale=1.0, r_shrink=0.002, theta_max=0.3, kd=3.0,
                 kp=2.0, fb_delay=5, sigma_a=None, n_relin=2, spec=None, tau_b=0.5, a_brake_frac=0.5, calib=1.0, calib_file="", eps_a=None, **kw):
        super().__init__(renv, **kw)
        self.rp = RC.ReachParams(H=int(H), mu_lo=float(mu_lo), mu_hi=float(mu_hi), k_sigma=float(k_sigma),
                                 eps_a=(EPS_A_DEFAULT if eps_a is None else float(eps_a)) * float(eps_scale), calib_file=str(calib_file or ""), kd=float(kd), kp=float(kp), fb_delay=int(fb_delay),
                                 tau_b=float(tau_b), a_brake_frac=float(a_brake_frac), calib=bool(float(calib)),
                                 a_max=float(renv.a_max), w_max=float(renv.w_max), alpha=float(renv.alpha or 1e3))
        self.model = RC.ReachModel(self.rp)
        self.r_shrink, self.theta_max, self.n_relin = float(r_shrink), float(theta_max), int(n_relin)
        self.spec_dict = spec or TS.TRAY_SPEC_FULL
        trk = renv.tracker
        self.nf = trk is None
        self.kf_args = dict(sigma_meas=0.003 if trk is None else trk.noise, lat=0 if trk is None else trk.lat,
                            sigma_a=(0.05 if self.nf else 0.10) if sigma_a is None else float(sigma_a), noise_free=self.nf)
        self.kf = BallKF(**self.kf_args)
        # fixed QP geometry: weights D (normalised deviation), box rows
        self.D = np.array([1.0 / renv.a_max] * 3 + [1.0 / renv.w_max] * 3)
        self.alpha = renv.alpha
        self._ents = TS.pred_entities("centre")
        self._poly_n = 16
        ang = (np.arange(self._poly_n) + 0.5) * 2 * np.pi / self._poly_n
        self._a_poly = np.stack([np.cos(ang), np.sin(ang)], 1)
        self.reset_stats()

    # ------------------------------------------------------------------
    def reset_stats(self):
        self._x = dict(x_f5_true=0, x_f5_unknown=0, x_f5_false=0, x_f5_qp=0, x_f5_backup=0, x_f5_relin=0, x_f5_infeas=0,
                       x_f5_ms=0.0, x_f5_ms_max=0.0, x_f5_n=0, x_f5_margin_min=1e9, x_f5_hw_end_mm=0.0)

    def begin(self, ctx):
        super().begin(ctx)
        self.reset_stats()
        self.kf = BallKF(**self.kf_args)

    # ------------------------------------------------------------------ pieces
    def _tray_state(self, o):
        return o[0:3].astype(np.float64), o[3:6].astype(np.float64), o[6:9].astype(np.float64)

    def _eff(self, u_nom, w0):
        """the nominal command as the arm realises it: Piper box, then the tilt-rate slew limit (the decision variable is the realised rate)."""
        u = project_box(u_nom, self.a_max, self.w_max)
        lim = (self.alpha or 1e3) * DT
        u[3:] = w0 + np.clip(u[3:] - w0, -lim, lim)
        return u

    def _spec(self, state):
        return TS.build_spec(self.spec_dict, self._ents, self.rp.H, self.r_shrink, self.theta_max, lo=1)

    def _margins(self, pred):
        st = TS.PredState(pred)
        sp = self._spec(st)
        return ir.margin(sp, st, True), ir.margin(sp, st, False), sp, st

    def backup_u(self, x, o):
        c0, vt0, _ = self._tray_state(o)
        a, w = RC.backup_cmd(c0, vt0, np.asarray(x, np.float64), self.model.par)
        return project_box(np.concatenate([a, w]), self.a_max, self.w_max)

    def _box_rows(self, w0):
        """Piper box + slew on the decision vector u = [a (3), w_realised (3)]: a_xy in a 16-gon inscribed in the disc, |a_z| <= a_max,
        |w_i| <= w_max, |w_i - w0_i| <= alpha dt."""
        A, b = [], []
        cs = np.cos(np.pi / self._poly_n)
        for n in self._a_poly:
            r = np.zeros(U_COLS)
            r[0], r[1] = n
            A.append(r); b.append(self.a_max * cs)
        for sgn in (1.0, -1.0):
            r = np.zeros(U_COLS); r[2] = sgn; A.append(r); b.append(self.a_max)
        lim = (self.alpha or 1e3) * DT
        for i in range(3):
            hi = min(self.w_max, w0[i] + lim)
            lo = max(-self.w_max, w0[i] - lim)
            r = np.zeros(U_COLS); r[3 + i] = 1.0; A.append(r); b.append(hi)
            r = np.zeros(U_COLS); r[3 + i] = -1.0; A.append(r); b.append(-lo)
        return np.array(A), np.array(b)

    def _bounds(self, w0):
        lim = (self.alpha or 1e3) * DT
        lo = np.array([-self.a_max] * 3 + list(np.maximum(-self.w_max, w0 - lim)))
        hi = np.array([self.a_max] * 3 + list(np.minimum(self.w_max, w0 + lim)))
        return lo, hi

    @staticmethod
    def _prune(A, b, u, lo, hi):
        """drop rows that cannot bind anywhere in the box [lo, hi] (sound): slack at u exceeds the largest possible change."""
        reach = np.maximum(A * (hi - u)[None], A * (lo - u)[None]).sum(1)
        keep = (b - A @ u) < reach + 1e-9
        return A[keep], b[keep]

    def _jac_steps(self, u, w0):
        d = np.array([0.05, 0.05, 0.05, 0.03, 0.03, 0.03])
        lim = (self.alpha or 1e3) * DT
        for i in range(3):
            if u[3 + i] + d[3 + i] > w0[i] + lim + 1e-12:
                d[3 + i] = -d[3 + i]
        return d

    # ------------------------------------------------------------------ main
    def __call__(self, obs_hist, act_hist, u_nom, t):
        t0 = time.perf_counter()
        x, P = self.kf.step(obs_hist)
        o = obs_hist[-1]
        c0, vt0, w0 = self._tray_state(o)
        u_eff = self._eff(np.asarray(u_nom, np.float64), w0)
        pred = self.model.predict(x, P, c0, vt0, w0, u_eff, jac=False)
        m_rob, m_nom, _, _ = self._margins(pred)
        X = self._x
        X["x_f5_n"] += 1
        X["x_f5_margin_min"] = min(X["x_f5_margin_min"], m_rob)
        X["x_f5_hw_end_mm"] += 1e3 * float(pred.hw[-1].max())
        if m_rob >= 0.0:
            X["x_f5_true"] += 1
            return self._done(u_nom, t0)
        X["x_f5_unknown" if m_nom >= 0.0 else "x_f5_false"] += 1
        u = self._solve(x, P, c0, vt0, w0, u_eff, o)
        return self._done(u, t0)

    def _done(self, u, t0):
        ms = 1e3 * (time.perf_counter() - t0)
        self._x["x_f5_ms"] += ms
        self._x["x_f5_ms_max"] = max(self._x["x_f5_ms_max"], ms)
        return u

    def _solve(self, x, P, c0, vt0, w0, u_eff, o):
        X = self._x
        u_lin = u_eff.copy()
        Ab, bb = self._box_rows(w0)
        lo, hi = self._bounds(w0)
        for it in range(self.n_relin + 1):
            pred = self.model.predict(x, P, c0, vt0, w0, u_lin, jac=True, d_u=self._jac_steps(u_lin, w0))
            st = TS.PredState(pred)
            sp = self._spec(st)
            A, b, _ = ir.rows(sp, st, st.jac(), robust=True)
            A = np.concatenate([A, Ab]); b = np.concatenate([b, bb])
            A, b = self._prune(A, b, u_eff, lo, hi)
            u, ok, sw, viol = qp.project(A, b, u_eff, self.D)
            if not ok:
                X["x_f5_infeas"] += 1
                break
            predu = self.model.predict(x, P, c0, vt0, w0, u, jac=False)
            m2, _, _, _ = self._margins(predu)
            if m2 >= -5e-4:                        # 0.5 mm of slack, inside the 2 mm r_shrink buffer: absorbs the QP residual
                X["x_f5_qp"] += 1
                X["x_f5_relin"] += it
                return u
            u_lin = u                                   # re-linearise at the (unsafe) solution
        X["x_f5_backup"] += 1
        return self.backup_u(x, o)

    # ------------------------------------------------------------------ three-valued status (for BOED / probes)
    def status(self, obs_hist, u):
        """status of command u: 'true' | 'false' | 'unknown'. Does not advance the estimator (call after __call__ of the same tick)."""
        x, P = self.kf.xs[self.kf.t].copy(), self.kf.Ps[self.kf.t].copy()
        o = obs_hist[-1]
        c0, vt0, w0 = self._tray_state(o)
        pred = self.model.predict(x, P, c0, vt0, w0, self._eff(np.asarray(u, np.float64), w0), jac=False)
        m_rob, m_nom, _, _ = self._margins(pred)
        return "false" if m_nom < 0 else ("true" if m_rob >= 0 else "unknown")

    def set_mu(self, mu_lo, mu_hi):
        """after a probe: narrow the rolling-resistance set (the constraint loosens)."""
        self.model.set_mu(mu_lo, mu_hi)

    def stats(self):
        s = dict(self._x)
        n = max(s["x_f5_n"], 1)
        s["x_f5_ms_mean"] = s["x_f5_ms"] / n
        s["x_f5_hw_end_mm"] = s["x_f5_hw_end_mm"] / n
        return s
