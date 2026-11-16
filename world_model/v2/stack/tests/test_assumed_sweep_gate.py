"""ASSUMED-value sweep gate (owner-written, 11-16): adapters/assumed_sweep.py on box 1.0-1.5, static level, closed_chain default plan.  SIMULATION ONLY.

Each knob's patch must take effect in the fresh process (read-back == requested, status "ok") and must MOVE the checker:
  PATH_MARGIN_MIN 0.05 -> 0.0     the joint_pos margin rises by exactly 0.05 rad (the row is distance-to-limit minus the margin)
  EPS_CC_POS      2 mm -> 5 mm    a kw-default bound at def time: some closed_chain row margin changes (a plain attribute patch would not)
  CLEARANCE_MIN   5 mm -> 0       read-back only: it sets the ARM classes; the box collision row binds on obj_world, so its margin must NOT move
  U_OBJ_CLEAR_M   5 mm -> 0       (static level) the collision row margin rises by exactly 5 mm (obj_world d_min)
and an unchanged row keeps its margin (no other state leaks between the two runs).  Slow (6 oracle calls, ~100 s each).
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[4]
PY = sys.executable
CASES = {"PATH_MARGIN_MIN": "0.05,0.0", "EPS_CC_POS": "0.002,0.005", "CLEARANCE_MIN": "0.005,0.0", "U_OBJ_CLEAR_M": "0.005,0.0"}
NO_MOVE = {"CLEARANCE_MIN"}   # arm classes only; the box collision row binds on obj_world (shelf|box_wall_e at place, 7.1 mm planned), so the row margin cannot move


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    env = dict(os.environ, BIMANUAL_PLANNER="closed_chain", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
               PYTHONPATH=f"{ROOT}:{ROOT}/experiments/ab_smooth_vs_bangbang:{ROOT}/experiments/realtime")
    procs, outs = {}, {}
    for knob, vals in CASES.items():
        out = tmp_path_factory.mktemp(knob)
        outs[knob] = out
        procs[knob] = subprocess.Popen([PY, "-m", "world_model.v2.adapters.assumed_sweep", "--task", "box", "--cells", "1.0-1.5", "--knobs", knob,
                                        "--values", vals, "--level", "static", "--workers", "1", "--out", str(out)], cwd=ROOT, env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    res = {}
    for knob, p in procs.items():
        log, _ = p.communicate(timeout=3600)
        assert p.returncode == 0, log[-3000:]
        recs = [json.loads(l) for l in (outs[knob] / "runs.jsonl").read_text().splitlines() if l.strip()]
        res[knob] = {float(r["settings"][knob]): r for r in recs if r.get("knob") == knob}
    return res


def _margins(rec):
    return {r["name"]: r["margin"] for r in rec["rows"] if r.get("margin") is not None}


@pytest.mark.parametrize("knob", list(CASES))
def test_patch_took_effect_and_moved_a_row(runs, knob):
    recs = runs[knob]
    assert len(recs) == 2, sorted(recs)
    for v, r in recs.items():
        assert r["status"] == "ok", (knob, v, r.get("patch_errors"))
        for e in r["effective"][knob]:
            assert abs(float(e["value"]) - v) < 1e-12, (knob, e)
    (va, ra), (vb, rb) = sorted(recs.items())
    ma, mb = _margins(ra), _margins(rb)
    changed = [n for n in ma.keys() & mb.keys() if abs(ma[n] - mb[n]) > 1e-9]
    if knob in NO_MOVE:
        assert ma.get("collision") == mb.get("collision"), (ma.get("collision"), mb.get("collision"))
        return
    assert changed, (knob, ma, mb)
    if knob == "U_OBJ_CLEAR_M":
        assert abs((ma["collision"] - mb["collision"]) - 0.005) < 1e-6, (ma["collision"], mb["collision"])
    if knob == "PATH_MARGIN_MIN":
        assert abs((ma["joint_pos"] - mb["joint_pos"]) - 0.05) < 1e-6, (ma["joint_pos"], mb["joint_pos"])


def test_tilt_row_is_unmoved_by_static_knobs(runs):
    """obj_tilt comes from the stored tilt table: no static knob may move it (else state leaked or the row is recomputed)."""
    vals = {round(_margins(r).get("obj_tilt", float("nan")), 9) for recs in runs.values() for r in recs.values() if "obj_tilt" in _margins(r)}
    assert len(vals) <= 1, vals
