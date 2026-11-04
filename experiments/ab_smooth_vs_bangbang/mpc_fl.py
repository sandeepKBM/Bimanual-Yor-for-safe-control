"""MOVED 2026-11-03 -> world_model/v2/flmpc/mpc_fl.py (code unchanged; equivalence-checked bit-identical on carry episodes).

This file is an alias kept so `import mpc_fl` (sys.path to this folder) still works: it IS the package module (same object, so
monkeypatches and isinstance checks agree).  Note: FL-MPC now resolves its QP / Jacobian helpers from world_model.v2.flmpc._sc,
not sim_common, so scripts that swap sim_common.solve_constrained_box_qp / sim_common.TAU_MAX (replay/, imitation/il_common,
realtime/fast_qp, realtime/bench_qp, actuation/run_actuation_eval) no longer reach FL-MPC; patch `mpc_fl.sc.<name>` instead.
"""
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from world_model.v2.flmpc import mpc_fl as _m  # noqa: E402

_sys.modules[__name__] = _m
