"""qp_safety/impl.py -- pick the FL-MPC implementation (SIMULATION ONLY).

    py  mpc_fl.FLMPC      dual-coordinate-ascent QP (4 sweeps, no final residual check: 'feasible' is NOT a certificate)
    rt  rt_flmpc.RTFLMPC  same ctor / step(...) signature incl. table_rows; Goldfarb-Idnani exact QP, capped
                          (OPTIMAL = KKT certificate; cap or infeasible -> relaxed QP -> a = 0 -> torque backstop)
Control rate and action scaling are the caller's (unchanged): one QP per FLMPC.step, torques returned unscaled.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_AB = REPO / "experiments" / "ab_smooth_vs_bangbang"
_RT = REPO / "experiments" / "realtime"
IMPLS = ("py", "rt")
ENV = "BIMANUAL_FLMPC_IMPL"


def _paths():
    if str(_AB) not in sys.path:
        sys.path.insert(0, str(_AB))
    if str(_RT) not in sys.path:
        sys.path.append(str(_RT))


def default_impl() -> str:
    v = os.environ.get(ENV, "py").strip().lower()
    if v not in IMPLS:
        raise ValueError(f"{ENV}={v!r}; expected one of {IMPLS}")
    return v


def make_flmpc(model, impl: str | None = None, **kw):
    """FLMPC ('py') or RTFLMPC ('rt') with the same kwargs. The instance gets `.impl`."""
    _paths()
    impl = default_impl() if impl is None else str(impl).lower()
    if impl not in IMPLS:
        raise ValueError(f"impl={impl!r}; expected one of {IMPLS}")
    if impl == "rt":
        import rt_flmpc
        fl = rt_flmpc.RTFLMPC(model, **kw)
    else:
        import mpc_fl
        fl = mpc_fl.FLMPC(model, **kw)
    fl.impl = impl
    return fl
