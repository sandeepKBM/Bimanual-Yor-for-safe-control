"""qp_safety tests (SIMULATION / synthetic only).  python -m pytest -q world_model/v2/qp_safety/tests"""
import numpy as np

from world_model.v2.qp_safety import QPRecorder, StepMonitor, make_flmpc
from world_model.v2.qp_safety.impl import _paths


def _near_parallel_qp(seed=0, n=12, m=30, spread=1e-2):
    rng = np.random.default_rng(seed)
    H = np.eye(n)
    f = -5.0 * rng.standard_normal(n)              # unconstrained optimum x* = -f
    a0 = rng.standard_normal(n)
    a0 /= np.linalg.norm(a0)
    A = a0 + spread * rng.standard_normal((m, n))
    b = A @ (-f) - 1.0 - 0.1 * rng.random(m)       # every row cuts off x*
    return H, f, -10 * np.ones(n), 10 * np.ones(n), A, b


def test_py_solver_can_report_feasible_with_violated_rows_and_gi_is_exact():
    _paths()
    import sim_common as sc
    import rt_qp
    worst_py, worst_gi, n_flag_lies = 0.0, 0.0, 0
    for s in range(20):
        H, f, lo, hi, A, b = _near_parallel_qp(seed=s)
        x, _lam, feas = sc.solve_constrained_box_qp(H, f, lo, hi, A, b)
        vpy = float(np.max(A @ x - b))
        xg, st, *_ = rt_qp.solve_dense(H, f, lo, hi, A, b)
        assert st == rt_qp.ST_OPTIMAL
        vgi = float(np.max(A @ xg - b))
        worst_py, worst_gi = max(worst_py, vpy), max(worst_gi, vgi)
        n_flag_lies += int(feas and vpy > 1e-6)
    print(f"py max viol {worst_py:.3e} (feasible-flag lies {n_flag_lies}/20), GI max viol {worst_gi:.3e}")
    assert worst_gi < 1e-9
    assert n_flag_lies > 0            # documents the hazard; if this ever fails the py solver was fixed upstream


def test_qprecorder_counts_violation_while_feasible():
    _paths()
    import sim_common as sc
    QPRecorder.install(sc)
    try:
        QPRecorder.reset()
        H, f, lo, hi, A, b = _near_parallel_qp(seed=1)
        x, _lam, feas = sc.solve_constrained_box_qp(H, f, lo, hi, A, b)
        s = QPRecorder.summary()
        assert s["n_solves"] == 1
        assert abs(s["max_row_violation"] - float(np.max(A @ np.asarray(x) - b))) < 1e-12
    finally:
        QPRecorder.uninstall()
    assert not getattr(sc.solve_constrained_box_qp, "_qp_recorder", False)


class _FakeFL:
    impl = "fake"
    last_fallback = 0

    def __init__(self, tau):
        self.tau = tau

    def step(self, *a, **k):
        return self.tau.copy(), self.tau.copy(), True


def test_stepmonitor_row_violation_and_passthrough():
    tau = np.array([1.0, 2.0, 0.0])
    fl = _FakeFL(tau)
    mon = StepMonitor.attach(fl)
    a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = fl.step(table_rows=(a, np.array([2.0, 1.5]), a, np.array([5.0, 5.0])))
    assert np.array_equal(out[0], tau) and out[2] is True          # outputs untouched
    s = mon.summary()
    assert s["n_rows_violated_while_ok"] == 1 and abs(s["max_row_violation"] - 0.5) < 1e-12
    fl.step()                                                       # no rows -> counted, no violation entry
    assert mon.summary()["n_calls"] == 2


def test_make_flmpc_impls():
    _paths()
    import mujoco
    import sim_common as sc
    m = sc.load_model() if hasattr(sc, "load_model") else None
    if m is None:
        return
    kw = dict(NC=4, L=8, W_POS_EGO=3000.0, W_VEL_EGO=15.0)
    py, rt = make_flmpc(m, impl="py", **kw), make_flmpc(m, impl="rt", **kw)
    assert py.impl == "py" and rt.impl == "rt"
    assert type(rt).__name__ == "RTFLMPC" and isinstance(rt, type(py))
