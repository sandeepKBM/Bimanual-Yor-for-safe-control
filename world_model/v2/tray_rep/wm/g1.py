"""wm/g1.py -- the G1-tray gate. SIMULATION / SAVED DATA ONLY.

G1-tray PASSES iff, on the certified-savable TEST states of the oracle (label 1):
  (a) at least 300 such states exist
  (b) the N0 closed-loop failure on them is >= 30%     (N0 = WM trained on nominal data only; pooled over its seeds)
  (c) the oracle policy fails <= 2%                    (the certificate must be executable)
Hold-still failure is printed as a reference only (it is not a criterion).

N0 numbers come from <runs>/<setup>/N0_<mode>_b0_s*/result.json (the same states, T and cl-max as the oracle reference
below). The references are recomputed here over ALL certified states unless a cache exists.
  python -m world_model.v2.tray_rep.wm.g1 --data D --runs R/runs --setup piper --obs-mode full
Exit code 0 = PASS, 1 = FAIL, 2 = N0 results missing. Writes <runs>/<setup>/G1_<mode>.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import cl_eval_mj as CL
from world_model.v2.tray_rep import oracle as OR
from world_model.v2.tray_rep.wm import common as C
from world_model.v2.tray_rep.wm import run_arm as RA

N_MIN, FAIL_MIN, ORACLE_MAX = 300, 0.30, 0.02


def gate(n_states, n0_fail, n0_n, oracle_fail):
    """pure decision rule (unit-tested). Rates are fractions."""
    a = n_states >= N_MIN
    b = (n0_fail is not None) and n0_fail >= FAIL_MIN
    c = oracle_fail <= ORACLE_MAX
    return dict(a_n_savable_ge_300=bool(a), b_n0_fail_ge_30pct=bool(b), c_oracle_fail_le_2pct=bool(c),
                PASS=bool(a and b and c))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--oracle", default="")
    ap.add_argument("--runs", default=str(RA.RUNS))
    ap.add_argument("--setup", default=RA.DEFAULT_SETUP)
    ap.add_argument("--obs-mode", default="full")
    ap.add_argument("--cl-T", type=int, default=RA.CL_T)
    ap.add_argument("--cl-max", type=int, default=None)
    ap.add_argument("--n0-glob", default="", help="override the N0 result.json glob")
    a = ap.parse_args()
    data = Path(a.data)
    oracle_path = Path(a.oracle) if a.oracle else data / "oracle" / "test_v0.npz"
    o = OR.load_oracle(oracle_path)
    lab = o["label"]
    n_sav = int((lab == OR.SAVABLE).sum())
    counts = {OR.LABELS[k]: int((lab == k).sum()) for k in OR.LABELS}
    ref = RA.cl_references(data, oracle_path, a.cl_T, a.cl_max)
    pat = a.n0_glob or str(Path(a.runs) / a.setup / f"N0_{a.obs_mode}_b0_s*" / "result.json")
    files = sorted(glob.glob(pat))
    per_seed, k_fail, n_tot = {}, 0, 0
    for f in files:
        r = json.loads(Path(f).read_text())
        per_seed[r["seed"]] = dict(fail_rate=r["cl"]["fail_rate"], n=r["cl"]["n"], n_fail=r["cl"]["n_fail"])
        k_fail += r["cl"]["n_fail"]; n_tot += r["cl"]["n"]
    n0_fail = (k_fail / n_tot) if n_tot else None
    out = dict(setup=a.setup, obs_mode=a.obs_mode, oracle=str(oracle_path), oracle_label_counts=counts,
               n_certified_savable=n_sav, n0_per_seed=per_seed, n0_pooled_fail=n0_fail, n0_pooled_n=n_tot,
               n0_pooled_ci95=CL.wilson(k_fail, n_tot) if n_tot else None,
               oracle_policy_fail=ref["oracle"]["fail_rate"], oracle_policy_n=ref["oracle"]["n"],
               oracle_policy_ci95=ref["oracle"]["fail_ci95"],
               hold_still_fail_REFERENCE_ONLY=ref["hold_still"]["fail_rate"], hold_still_ci95=ref["hold_still"]["fail_ci95"])
    out.update(gate(n_sav, n0_fail, n_tot, ref["oracle"]["fail_rate"]))
    if not n_tot:
        out["PASS"] = False
        out["note"] = f"no N0 results found with {pat}"
    C.atomic_json(out, Path(a.runs) / a.setup / f"G1_{a.obs_mode}.json")
    print(json.dumps(out, indent=1, default=float), flush=True)
    print(f"[G1-tray {a.obs_mode}] {'PASS' if out['PASS'] else 'FAIL'}  states={n_sav}  N0 fail={n0_fail}  "
          f"oracle fail={ref['oracle']['fail_rate']:.4f}  hold-still (ref) {ref['hold_still']['fail_rate']:.3f}", flush=True)
    sys.exit(0 if out["PASS"] else (2 if not n_tot else 1))


if __name__ == "__main__":
    main()
