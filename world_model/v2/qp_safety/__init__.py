"""qp_safety -- FL-MPC solver choice + read-only constraint-violation monitor (SIMULATION ONLY).

Why: mpc_fl.FLMPC's default QP (sim_common/fast_qp dual coordinate ascent, 4 sweeps) has no final A x <= b check, so with
many near-parallel rows it can return feasible=True while rows stay violated. rt_flmpc.RTFLMPC is a drop-in subclass whose
Goldfarb-Idnani QP is exact (OPTIMAL = KKT certificate; iteration cap / infeasible -> the same graceful fallback).

    make_flmpc(model, impl=None, **FLMPC kwargs)   impl 'py' | 'rt'; default env BIMANUAL_FLMPC_IMPL (default 'py')
    StepMonitor.attach(fl)                         wraps fl.step: feasible / fallback / violation of caller table_rows
    QPRecorder.install()                           wraps sim_common.solve_constrained_box_qp: max(Ax-b) per py-path solve

Neither file in experiments/ (mpc_fl.py, rt_flmpc.py, rt_qp.py, fast_qp.py, sim_common.py) is edited.
"""
from .impl import IMPLS, default_impl, make_flmpc
from .monitor import QPRecorder, StepMonitor

__all__ = ["IMPLS", "default_impl", "make_flmpc", "StepMonitor", "QPRecorder"]
