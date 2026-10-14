"""selftest.py -- prove the harness CATCHES regressions (and passes on unmodified code).

All injections are run-time monkeypatches inside THIS process (no file is edited, nothing on disk changes):
  baseline                : unmodified code                                   -> must PASS
  row_1e-12               : _assemble_fl returns b_ineq[0] + 1e-12            -> must FAIL
  tau_1e-12               : step() adds 1e-12 to tau_ego                       -> must FAIL (bitwise); a 1e-9 tolerance
                            check would have PASSED it (shown as 'tolerance-only')
  bypass_solver           : mpc_fl.sc replaced by a copy of sim_common holding a private reference to the real
                            solver (a refactor that binds the solver at import time instead of calling
                            `sc.solve_constrained_box_qp` as a module global) -> must FAIL (solver-call trace)
  valuefl_no_proxy        : ValueFLMPC.step stops swapping in the solver proxy (value rows silently vanish) -> must FAIL
  state_not_updated       : step() forgets to store a_ego_prev                -> must FAIL (side effect)
  fallback_scale_1e-12    : _torque_safe_scale returns a0*(1-1e-12)           -> must FAIL (fallback records only)
  rt_tau_1e-12            : RTFLMPC.step adds 1e-12 to tau                    -> must FAIL (RTFLMPC records)

  python selftest.py --records <dir> [--limit-per-class 25]
"""
from __future__ import annotations

import argparse
import contextlib
import sys
import types

import numpy as np

import check as ck
import replay_core as rc


@contextlib.contextmanager
def patch(obj, name, new):
    # for classes keep the RAW class-dict entry (staticmethod stays a staticmethod on restore)
    old = obj.__dict__[name] if isinstance(obj, type) else getattr(obj, name)
    setattr(obj, name, new)
    try:
        yield
    finally:
        setattr(obj, name, old)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=rc.DEFAULT_DATA_DIR)
    ap.add_argument("--limit-per-class", type=int, default=25)
    ap.add_argument("--atol", type=float, default=1e-9)
    args = ap.parse_args(argv)

    import mpc_fl
    import sim_common as sc
    import flmpc_value as fv
    import rt_flmpc

    all_paths = ck.find_records(args.records)
    if not all_paths:
        print("[selftest] no records"); return 2
    # A small, cheap, representative subset (infeasible-QP replays are slow): per class a few feasible records
    # plus relaxed-fallback (1) and zero-fallback (2) records so the fallback injections can be noticed.
    want = {("FLMPC", 0): 10, ("FLMPC", 1): 4, ("FLMPC", 2): 3, ("ValueFLMPC", 0): 4, ("ValueFLMPC", 1): 3,
            ("ValueFLMPC", 2): 2, ("RTFLMPC", 0): 3, ("RTFLMPC", 1): 3, ("RTFLMPC", 2): 2}
    got = {k: 0 for k in want}
    paths = []
    for p in all_paths:
        r = rc.load_record(p)
        k = (r["cls"], r["tags"]["fallback"])
        if k in want and got[k] < want[k] and (k[1] != 0 or r["tags"]["table_rows"] or got[k] < 3):
            got[k] += 1
            paths.append(p)
    print(f"[selftest] subset: {len(paths)} records {got}", flush=True)

    def run(label, expect_fail, only=None):
        summ, res = ck.run_check(paths, args.atol, None, only, verbose=False)
        n = len(res)
        nfail = sum(not r["exact"] for r in res)
        ntol = sum((not r["exact"]) and r["tol"] for r in res)
        by = {c: f"{s['records'] - s['exact']}/{s['records']}" for c, s in summ.items()}
        ok = (nfail > 0) if expect_fail else (nfail == 0)
        print(f"  {label:22s} expect {'FAIL' if expect_fail else 'PASS'} -> got {'FAIL' if nfail else 'PASS'} "
              f"({nfail}/{n} records differ; tolerance-only {ntol}; by class {by})  "
              f"{'OK' if ok else '*** HARNESS PROBLEM ***'}", flush=True)
        return ok

    allok = True
    print("[selftest] baseline (unmodified code) ...", flush=True)
    allok &= run("baseline", False)

    # 1. a row perturbed by 1e-12
    orig_asm = mpc_fl._assemble_fl

    def asm_pert(*a, **k):
        H, f, A, b = orig_asm(*a, **k)
        b = b.copy()
        b[0] += 1e-12
        return H, f, A, b
    with patch(mpc_fl, "_assemble_fl", asm_pert):
        allok &= run("row_1e-12", True)

    # 2. tau perturbed by 1e-12 (below any sane tolerance)
    orig_step = mpc_fl.FLMPC.step

    def step_tau(self, *a, **k):
        te, ti, fe = orig_step(self, *a, **k)
        return te + 1e-12, ti, fe
    with patch(mpc_fl.FLMPC, "step", step_tau):
        allok &= run("tau_1e-12", True, only={"FLMPC"})

    # 3. solver bound privately (bypasses the module-global patch point)
    ns = types.SimpleNamespace(**{k: getattr(sc, k) for k in dir(sc) if not k.startswith("__")})
    real_solver = sc.solve_constrained_box_qp
    ns.solve_constrained_box_qp = real_solver
    real_sc = mpc_fl.sc
    with patch(mpc_fl, "sc", ns):
        allok &= run("bypass_solver", True, only={"FLMPC", "ValueFLMPC"})

    # 4. ValueFLMPC without the solver proxy: value rows vanish
    def vstep(self, *a, value_rows=None, **k):
        self._rows = (None, np.zeros(0)) if value_rows is None else value_rows
        return mpc_fl.FLMPC.step(self, *a, **k)
    with patch(fv.ValueFLMPC, "step", vstep):
        allok &= run("valuefl_no_proxy", True, only={"ValueFLMPC"})

    # 5. side effect dropped
    def step_noprev(self, *a, **k):
        out = orig_step(self, *a, **k)
        self.a_ego_prev = np.zeros_like(self.a_ego_prev)
        return out
    with patch(mpc_fl.FLMPC, "step", step_noprev):
        allok &= run("state_not_updated", True, only={"FLMPC"})

    # 6. backstop scaled by (1 - 1e-12): only fallback steps can notice
    orig_tss = mpc_fl.FLMPC._torque_safe_scale

    def tss(M_arm, c_arm, a0, tau_max):
        return orig_tss(M_arm, c_arm, a0, tau_max) * (1.0 - 1e-12)
    with patch(mpc_fl.FLMPC, "_torque_safe_scale", staticmethod(tss)):
        allok &= run("fallback_scale_1e-12", True, only={"FLMPC", "ValueFLMPC"})

    # 7. realtime drop-in
    orig_rt = rt_flmpc.RTFLMPC.step

    def rt_step(self, *a, **k):
        te, ti, fe = orig_rt(self, *a, **k)
        return te + 1e-12, ti, fe
    with patch(rt_flmpc.RTFLMPC, "step", rt_step):
        allok &= run("rt_tau_1e-12", True, only={"RTFLMPC"})

    print("[selftest] after all patches are removed: unmodified code again ...", flush=True)
    allok &= run("baseline_again", False)
    print(f"[selftest] RESULT: {'ALL AS EXPECTED' if allok else 'PROBLEM: see lines marked ***'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
