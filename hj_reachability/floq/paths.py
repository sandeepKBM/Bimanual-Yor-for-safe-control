"""sys.path setup shared by this package's scripts.

Adds (1) hj_reachability/ (this file's parent) so ``import dynamics`` /
``import mujoco_fk`` resolve to our numpy reference spec + real-Piper FK
oracle, and (2) VLAism's ur5e_benchmark/src so ``import ur5e_benchmark.floq``
resolves to the FOUND FLOQ implementation -- imported read-only, never
copied or edited (see hj_reachability/FLOQ_SETUP.md).

No name-collision risk here (unlike deepreach_dynamics.py's workaround for
this repo's ``dynamics.py`` vs. the DeepReach fork's ``dynamics/`` package):
this package never imports the DeepReach fork, so a plain ``sys.path.insert``
+ ``import dynamics`` / ``import ur5e_benchmark.floq`` is safe.
"""
from __future__ import annotations

import os
import sys

HJ_REACHABILITY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UR5E_BENCHMARK_SRC = "/common/users/ss5772/VLAism/ur5e_benchmark/src"

for _p in (HJ_REACHABILITY_ROOT, UR5E_BENCHMARK_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)
