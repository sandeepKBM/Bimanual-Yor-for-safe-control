"""world_model/report_real_wm.py

Aggregates world_model/eval_results_real_wm/all_episodes.csv (the REAL-WM
eval-harness comparison, see run_wm_eval_harness.py) into a summary
table + comparison plot -- reuses eval/report.py's aggregate()/make_plot()
functions via import (eval/ stays read-only) with custom in/out paths under
world_model/ + artifacts/ instead of eval/report.py's own eval/results/
default (that script's module-level RESULTS_DIR/ARTIFACTS_DIR constants are
only used in its own main(), not inside aggregate()/make_plot(), so importing
those two functions directly is safe and edit-free).

Usage:
    python -m world_model.report_real_wm \\
        --csv world_model/eval_results_real_wm/all_episodes.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

WORLD_MODEL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = WORLD_MODEL_ROOT.parent
_EVAL_DIR = REPO_ROOT / "eval"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import report as eval_report  # noqa: E402  (eval/, read-only import; do not edit)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=str(WORLD_MODEL_ROOT / "eval_results_real_wm" / "all_episodes.csv"))
    p.add_argument("--out-csv", default=None)
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-png", default=str(REPO_ROOT / "artifacts" / "wm_eval_compare_real.png"))
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"no episodes found at {csv_path}; run world_model/run_wm_eval_harness.py first")
    df = pd.read_csv(csv_path)
    summary = eval_report.aggregate(df)

    out_csv = Path(args.out_csv) if args.out_csv else csv_path.parent / "summary.csv"
    out_json = Path(args.out_json) if args.out_json else csv_path.parent / "summary.json"
    out_png = Path(args.out_png)

    summary.to_csv(out_csv)
    with open(out_json, "w") as f:
        json.dump(json.loads(summary.reset_index().to_json(orient="records")), f, indent=2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    eval_report.make_plot(summary, out_png)

    print(summary.to_string())
    print(f"\nWrote {out_csv}\nWrote {out_json}\nWrote {out_png}")


if __name__ == "__main__":
    main()
