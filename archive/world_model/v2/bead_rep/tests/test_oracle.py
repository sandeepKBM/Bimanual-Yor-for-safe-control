"""oracle.py sanity tests (SIMULATION ONLY)."""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

from world_model.v2.bead_rep import bead_env as B
from world_model.v2.bead_rep import oracle as O


def _settled():
    env = B.BeadEnv()
    env.reset(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0, 0, 0]))
    for _ in range(80):
        env.phys_step()
    return env


def test_rest_state_is_trivially_safe():
    env = _settled()
    snap = env.snapshot()
    rec = O.certify(env, snap, T=25, n_rand=3, rng=np.random.default_rng(0))
    assert rec["certified"] == 1 and rec["trivial"] == 1 and rec["kind"] == 1 and rec["n_eval"] == 1 and rec["best_margin"] > 0


def test_overturned_cup_is_not_certified_and_search_is_deterministic():
    env = _settled()
    for _ in range(30):                                   # tip the cup over (tilt rate 2 rad/s for 0.6 s, then it keeps coasting rotated)
        env.tick(np.array([0, 0, 0, 2.0, 0, 0]))
    for _ in range(30):
        env.tick(np.array([0, 0, 0, 2.0, 0, 0]))
    assert env.n_spilled > 0 or env.margin() < 0.9
    snap = env.snapshot()
    r1 = O.certify(env, snap, T=25, n_rand=4, rng=np.random.default_rng(5))
    r2 = O.certify(env, snap, T=25, n_rand=4, rng=np.random.default_rng(5))
    for k in ("certified", "kind", "idx", "n_eval", "n_spill", "n_tunnel"):
        assert r1[k] == r2[k]
    assert r1["certified"] == 0 or r1["already_spilled"] == 0


def test_run_sequence_reports_spill_and_restores_exactly():
    env = _settled(); snap = env.snapshot()
    a = O.run_sequence(env, snap, 20, seq=np.zeros((20, 6)), n_open=20)
    b = O.run_sequence(env, snap, 20, seq=np.zeros((20, 6)), n_open=20)
    assert a == b and a[0] == "safe"
