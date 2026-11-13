"""Shared import shim for experiments/realtime/* scripts (NEW file, read-only
use of the repo's existing modules). Puts the repo root, scripts/,
experiments/ab_smooth_vs_bangbang and eval/ on sys.path and imports the
flagship demo module the same way `python scripts/make_cup_collect_safety_demo.py`
would, so monkeypatches applied to `demo.sc` / `demo.mpc_fl` hit exactly the
module objects run_episode() resolves at call time.

SIMULATION ONLY -- nothing here (or in anything that imports it) opens any
hardware / network connection.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "experiments" / "ab_smooth_vs_bangbang",
           REPO_ROOT / "eval", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import make_cup_collect_safety_demo as demo  # noqa: E402

sc = demo.sc
mpc_fl = demo.mpc_fl

SCRATCH = Path(os.environ.get("CLAUDE_JOB_DIR", "data/jobs")) / "tmp"
SCRATCH.mkdir(parents=True, exist_ok=True)

BASELINE_SEED0_FLMPC = dict(collision=False, placed_final=4, h_min=0.01929845984125282)
