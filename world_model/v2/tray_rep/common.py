"""common.py -- paths, constants and small helpers shared by the tray_rep modules.
SIMULATION / SAVED DATA ONLY. CPU only. No hardware, no network.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ROOT = Path("/common/users/ss5772/bimanual_data/tray_rep")
SMOKE = ROOT / "smoke"
TMP = ROOT / "tmp"
SPLITS = ("train", "val", "test")

# shard -> split rule (same as the pendulum wmdata and bead_rep): shard index % 5 == 4 -> test, == 3 -> val, else train
def shard_split(g: int) -> str:
    return "test" if g % 5 == 4 else ("val" if g % 5 == 3 else "train")


# the pipeline's window law (train_kinwm.make_branches / vdirect.branches)
PRE, LB, W_WIN = 60, 100, 160
