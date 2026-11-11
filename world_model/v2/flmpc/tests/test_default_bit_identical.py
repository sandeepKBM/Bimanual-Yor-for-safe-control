"""The full FL-MPC (mpc_fl.FLMPC: teleop, Sept cup demo, latent_hj, house) must be bit-identical to the committed (HEAD) source.
The HEAD file is loaded under another module name and both controllers are stepped on the same fixed states.  SIMULATION ONLY.
(Reference-by-HEAD: when this test is committed, HEAD == the file; re-point `REF` to a commit hash to compare against a fixed revision.)"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from world_model.v2.nero import adapters as A
from world_model.v2.nero import config as C
from world_model.v2.flmpc import mpc_fl as NOW

REF = "HEAD"
ROOT = Path(__file__).resolve().parents[4]
SRC = "world_model/v2/flmpc/mpc_fl.py"


def _load_ref(tmp_path):
    try:
        code = subprocess.run(["git", "show", f"{REF}:{SRC}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except Exception as e:                                 # pragma: no cover
        pytest.skip(f"git show failed: {e}")
    name = "world_model.v2.flmpc._mpc_fl_ref"
    f = tmp_path / "mpc_fl_ref.py"                       # numba cache=True needs a real file
    f.write_text(code)
    spec = importlib.util.spec_from_file_location(name, f)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)                         # `from . import _sc` resolves via the name's package (world_model.v2.flmpc)
    return mod


def _drive(mod, m, d0, hold_q, tau, extras):
    A_ = A
    lq, ld = A_.arm_addrs(m, "left")
    rq, rd = A_.arm_addrs(m, "right")
    fl = mod.FLMPC(m, NC=4, L=8, W_POS_EGO=3000.0, W_VEL_EGO=15.0, NV=C.N_ARM, tau_max_vec=tau)
    NP = fl.NP
    qe = np.tile(np.array(hold_q["left"]), (NP + 1, 1))
    qi = np.tile(np.array(hold_q["right"]), (NP + 1, 1))
    z = np.zeros((NP + 1, C.N_ARM))
    pairs = A_.PairBarriers(m)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(3)
    out = []
    for k in range(6):
        d.qpos[:] = d0.qpos
        d.qvel[:] = d0.qvel
        d.qpos[lq] += 0.02 * rng.standard_normal(7)
        d.qvel[ld] = 0.3 * rng.standard_normal(7)
        d.qvel[rd] = 0.3 * rng.standard_normal(7)
        mujoco.mj_forward(m, d)
        te, ti, ok = fl.step(m, d, lq, ld, rq, rd, qe, z, qi, z, pairs(d), **(extras if k % 2 else {}))
        out.append((te.copy(), ti.copy(), ok, fl.last_fallback))
    return out


def test_full_flmpc_matches_head(hold, tmp_path):
    m, d0, info = hold
    ref = _load_ref(tmp_path)
    tau = C.tau_limit()
    ex = dict(M_ego_extra=0.05 * np.eye(7), bias_ego_extra=0.3 * np.ones(7))
    a = _drive(ref, m, d0, info["hold_q"], tau, ex)
    b = _drive(NOW, m, d0, info["hold_q"], tau, ex)
    for (te0, ti0, ok0, fb0), (te1, ti1, ok1, fb1) in zip(a, b):
        assert np.array_equal(te0, te1) and np.array_equal(ti0, ti1) and ok0 == ok1 and fb0 == fb1
