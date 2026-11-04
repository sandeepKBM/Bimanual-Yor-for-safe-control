"""qp_safety/monitor.py -- READ-ONLY constraint-violation monitor for FL-MPC (SIMULATION ONLY). Never changes a command.

StepMonitor.attach(fl) wraps fl.step on the INSTANCE (works for FLMPC and RTFLMPC) and records per call:
  ok (the controller's feasible flag), fallback level (RTFLMPC.last_fallback; py: 0 if ok else 'n/a'),
  and, when the caller passes table_rows = (a_e, b_e, a_i, b_i) with a @ tau <= b per arm, the violation of the
  RETURNED torques: v = max_k(a_k @ tau - b_k) (row units) and v / ||a_k|| (torque units).
QPRecorder.install() wraps sim_common.solve_constrained_box_qp (the py path's QP, also used inside
  FLMPC._relaxed_fallback) and records max(A x - b), max box violation and the returned feasible flag per solve.
  The rt path solves inside numba; its OPTIMAL status is a KKT certificate, and StepMonitor still checks its torques.
A violation counts when v > TOL_ABS + TOL_REL * |b_k|.
"""
from __future__ import annotations

import numpy as np

TOL_ABS, TOL_REL = 1e-6, 1e-9


def _viol(a, b, tau):
    a = np.asarray(a, float)
    if a.size == 0:
        return -np.inf, -np.inf, False
    b = np.asarray(b, float)
    r = a @ np.asarray(tau, float) - b
    k = int(np.argmax(r))
    nrm = float(np.linalg.norm(a[k])) or 1.0
    bad = bool(np.any(r > TOL_ABS + TOL_REL * np.abs(b)))
    return float(r[k]), float(r[k] / nrm), bad


class StepMonitor:
    def __init__(self, fl):
        self.fl = fl
        self.impl = getattr(fl, "impl", type(fl).__name__)
        self.reset()

    def reset(self):
        self.n_calls = 0
        self.n_not_ok = 0
        self.fallbacks = {}
        self.n_with_rows = 0
        self.n_rows_violated = 0
        self.n_rows_violated_while_ok = 0
        self.max_v = -np.inf
        self.max_vn = -np.inf
        self.v_hist = []

    @classmethod
    def attach(cls, fl):
        mon = cls(fl)
        inner = fl.step

        def step(*args, **kw):
            out = inner(*args, **kw)
            try:
                mon._record(out, kw.get("table_rows"))
            except Exception:          # bookkeeping must never break control
                pass
            return out

        fl.step = step
        fl.qp_monitor = mon
        return mon

    def _record(self, out, table_rows):
        te, ti, ok = out
        self.n_calls += 1
        ok = bool(ok)
        if not ok:
            self.n_not_ok += 1
        fb = getattr(self.fl, "last_fallback", None)
        key = str(fb) if fb is not None else ("ok" if ok else "not_ok")
        self.fallbacks[key] = self.fallbacks.get(key, 0) + 1
        if table_rows is None:
            return
        a_e, b_e, a_i, b_i = table_rows
        ve, vne, be = _viol(a_e, b_e, te)
        vi, vni, bi = _viol(a_i, b_i, ti)
        v, vn = max(ve, vi), max(vne, vni)
        if not np.isfinite(v):
            return
        self.n_with_rows += 1
        self.v_hist.append(v)
        self.max_v, self.max_vn = max(self.max_v, v), max(self.max_vn, vn)
        if be or bi:
            self.n_rows_violated += 1
            if ok:
                self.n_rows_violated_while_ok += 1

    def summary(self):
        vh = np.asarray(self.v_hist) if self.v_hist else np.zeros(0)
        return dict(impl=self.impl, n_calls=self.n_calls, n_not_ok=self.n_not_ok, fallbacks=dict(self.fallbacks),
                    n_calls_with_table_rows=self.n_with_rows, n_rows_violated=self.n_rows_violated,
                    n_rows_violated_while_ok=self.n_rows_violated_while_ok,
                    max_row_violation=(float(self.max_v) if np.isfinite(self.max_v) else None),
                    max_row_violation_torque_units=(float(self.max_vn) if np.isfinite(self.max_vn) else None),
                    p99_row_violation=(float(np.quantile(vh, 0.99)) if vh.size else None))


class QPRecorder:
    """Process-wide wrapper of sim_common.solve_constrained_box_qp (idempotent). Later installs of other wrappers
    (e.g. experiments/realtime/fast_qp.install) replace it; call install() after them."""
    _orig = None
    stats = None

    @classmethod
    def install(cls, sc=None):
        if sc is None:
            from .impl import _paths
            _paths()
            from world_model.v2.flmpc import mpc_fl
            sc = mpc_fl.sc                 # 2026-11-03: FL-MPC's QP now lives in flmpc._sc (was sim_common)
        if getattr(sc.solve_constrained_box_qp, "_qp_recorder", False):
            return cls
        cls._orig = sc.solve_constrained_box_qp
        cls.reset()
        orig = cls._orig

        def wrapped(H, f, lower, upper, a_ineq, b_ineq, *a, **k):
            out = orig(H, f, lower, upper, a_ineq, b_ineq, *a, **k)
            try:
                cls._record(out, lower, upper, a_ineq, b_ineq)
            except Exception:
                pass
            return out

        wrapped._qp_recorder = True
        sc.solve_constrained_box_qp = wrapped
        cls._sc = sc
        return cls

    @classmethod
    def uninstall(cls):
        if cls._orig is not None:
            cls._sc.solve_constrained_box_qp = cls._orig
            cls._orig = None

    @classmethod
    def reset(cls):
        cls.stats = dict(n_solves=0, n_feasible_flag=0, n_violated=0, n_violated_while_feasible=0,
                         max_row_violation=0.0, max_row_violation_normalised=0.0, max_box_violation=0.0, viol=[])

    @classmethod
    def _record(cls, out, lo, hi, A, b):
        x, _lam, feas = out[0], out[1], bool(out[2])
        s = cls.stats
        s["n_solves"] += 1
        s["n_feasible_flag"] += int(feas)
        x = np.asarray(x, float)
        bx = float(max(np.max(np.asarray(lo, float) - x, initial=0.0), np.max(x - np.asarray(hi, float), initial=0.0)))
        s["max_box_violation"] = max(s["max_box_violation"], bx)
        A = np.asarray(A, float)
        if A.size == 0:
            return
        b = np.asarray(b, float)
        r = A @ x - b
        k = int(np.argmax(r))
        v = float(r[k])
        nrm = float(np.linalg.norm(A[k])) or 1.0
        s["viol"].append(v)
        s["max_row_violation"] = max(s["max_row_violation"], v)
        s["max_row_violation_normalised"] = max(s["max_row_violation_normalised"], v / nrm)
        if np.any(r > TOL_ABS + TOL_REL * np.abs(b)):
            s["n_violated"] += 1
            if feas:
                s["n_violated_while_feasible"] += 1

    @classmethod
    def summary(cls):
        if cls.stats is None:
            return None
        s = {k: v for k, v in cls.stats.items() if k != "viol"}
        vh = np.asarray(cls.stats["viol"]) if cls.stats["viol"] else np.zeros(0)
        s["p99_row_violation"] = float(np.quantile(vh, 0.99)) if vh.size else None
        return s
