"""Wrapper: run the (hash-identical) g3_killtest package with outputs redirected to $G3_OUT
(used because /common/users hit EDQUOT). SIMULATION ONLY."""
import os
import sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path  # noqa: E402
from world_model.v2.g3_killtest import config as K  # noqa: E402
OUT = Path(os.environ["G3_OUT"])
K.OUT = OUT
K.VAL_DIR = OUT / "values"
mode = sys.argv[1]
sys.argv = [sys.argv[0]] + sys.argv[2:]
if mode == "run":
    from world_model.v2.g3_killtest import runner
    runner.main()
else:
    from world_model.v2.g3_killtest import analyze
    analyze.main()
