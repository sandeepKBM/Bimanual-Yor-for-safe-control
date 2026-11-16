"""world_model/run_wm_eval_harness.py

Runs eval/run_eval.py's 4-config comparison (none/geometric_only/
semantic_only/both) on the cup task with the REAL trained WM monitor
(world_model/wm_semantic_monitor.py's RealWMSpillMonitor) swapped in for
eval/safety_module.py's OracleSpillMonitor placeholder -- monkeypatches
`safety_module.SEMANTIC_REGISTRY["oracle"]` at runtime (no edits to eval/,
which is import-only per this task's constraints) so run_eval.py's existing
CONFIGS dict (which references the "oracle" key for semantic_only/both)
transparently uses the real model instead.

Writes results to `--out-dir` (default world_model/eval_results_real_wm/),
NOT eval/results/ -- eval/ stays read-only (run_eval.py already supports
`--out-dir`, so this needs no changes there).

Usage:
    CUDA_VISIBLE_DEVICES=<gpu> python -m world_model.run_wm_eval_harness \\
        --wm-checkpoint world_model/logdir/wm_cup_full/best.pt \\
        --head-checkpoint world_model/logdir/wm_cup_full/monitor_head_cup_full.pt \\
        --episodes 8 --duration-s 4.0 --risk-threshold 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORLD_MODEL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = WORLD_MODEL_ROOT.parent
_EVAL_DIR = REPO_ROOT / "eval"
_SIM_COMMON_DIR = REPO_ROOT / "experiments" / "ab_smooth_vs_bangbang"
for _p in (str(REPO_ROOT), str(_SIM_COMMON_DIR), str(_EVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_eval  # noqa: E402  (eval/, read-only import; do not edit)
import safety_module as sm  # noqa: E402  (eval/, read-only import; do not edit)

from world_model.wm_semantic_monitor import RealWMSpillMonitor  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wm-checkpoint", required=True)
    p.add_argument("--head-checkpoint", required=True)
    p.add_argument("--risk-threshold", type=float, default=0.5)
    p.add_argument("--configs", default="all")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--duration-s", type=float, default=4.0)
    p.add_argument("--out-dir", default=str(WORLD_MODEL_ROOT / "eval_results_real_wm"))
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    def _factory():
        return RealWMSpillMonitor(
            wm_checkpoint=args.wm_checkpoint,
            head_checkpoint=args.head_checkpoint,
            risk_threshold=args.risk_threshold,
        )

    sm.SEMANTIC_REGISTRY["oracle"] = _factory
    print(
        f"[run_wm_eval_harness] safety_module.SEMANTIC_REGISTRY['oracle'] -> RealWMSpillMonitor "
        f"(wm_checkpoint={args.wm_checkpoint}, head_checkpoint={args.head_checkpoint}, "
        f"risk_threshold={args.risk_threshold}) -- semantic_only/both configs now run the REAL model, "
        f"not the ground-truth oracle."
    )

    run_eval_argv = [
        "--configs", args.configs,
        "--episodes", str(args.episodes),
        "--seed-start", str(args.seed_start),
        "--duration-s", str(args.duration_s),
        "--out-dir", args.out_dir,
    ]
    sys_argv_backup = sys.argv
    sys.argv = ["run_eval.py"] + run_eval_argv
    try:
        run_eval.main()
    finally:
        sys.argv = sys_argv_backup


if __name__ == "__main__":
    main()
