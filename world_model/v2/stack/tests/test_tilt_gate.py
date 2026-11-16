"""K4 gate (owner-written, 11-16): object-tilt row (stack/checks/tilt.py) on synthetic MEASURED tables.  SIMULATION ONLY (pure).

  level choice   the row uses the next measured level >= a_plan (never a lower one); above every level -> missing
  failures       a stopped or errored episode, or a peak >= the U limit, refutes; eps_upper counts every failure
  sampled        ok -> guarantee 'sampled' with n and the Clopper-Pearson bound for 0 failures in n
  old tables     no n_stopped / n_errors -> missing
"""
from __future__ import annotations

import copy

import pytest

from world_model.v2.stack.checks import tilt as T
from world_model.v2.stack.checks.dynamic import cp_upper


def _lvl(mx, n=20, stop=0, err=0, over=0):
    return dict(n=n - err, n_total=n, n_errors=err, n_stopped=stop, n_over=over, n_fail=max(stop + err + over, 0),
                p50=mx * 0.6, p95=mx * 0.9, max=mx, seeds=list(range(n)))


TABLE = {"file": "synthetic", "noise_model": "sim only", "planner": "closed_chain",
         "cells": {"box|1.0-1.5": {"u_limit_deg": 10.0, "levels": {"0.1": _lvl(2.0), "0.25": _lvl(9.5), "0.5": _lvl(4.0)}}}}


def test_level_choice_never_rounds_down():
    r = T.tilt_row(TABLE, "box|1.0-1.5", a_plan=0.2, u_limit_deg=10.0)        # must use 0.25 (9.5 deg), not 0.1 (2 deg)
    assert r.ok and r.margin == pytest.approx(0.5)
    r = T.tilt_row(TABLE, "box|1.0-1.5", a_plan=0.1, u_limit_deg=10.0)
    assert r.ok and r.margin == pytest.approx(8.0)
    assert T.tilt_row(TABLE, "box|1.0-1.5", a_plan=0.51, u_limit_deg=10.0).guarantee == "missing"
    assert T.tilt_row(TABLE, "box|9-9", a_plan=0.1, u_limit_deg=10.0).guarantee == "missing"
    assert T.tilt_row(None).guarantee == "missing" and not T.tilt_row(None).ok


def test_sampled_guarantee_and_bound():
    r = T.tilt_row(TABLE, "box|1.0-1.5", a_plan=0.5, u_limit_deg=10.0)
    assert r.ok and r.guarantee == "sampled" and r.n == 20
    assert r.eps_upper == pytest.approx(cp_upper(0, 20, 0.95), abs=1e-4)


@pytest.mark.parametrize("kw,word", [(dict(stop=1), "stopped"), (dict(err=2), "errored"), (dict(over=1), "over")])
def test_failures_refute(kw, word):
    t = copy.deepcopy(TABLE)
    mx = 10.5 if "over" in kw else 4.0
    t["cells"]["box|1.0-1.5"]["levels"]["0.5"] = _lvl(mx, **kw)
    r = T.tilt_row(t, "box|1.0-1.5", a_plan=0.5, u_limit_deg=10.0)
    assert not r.ok and r.guarantee == "refuted" and word in r.why
    nf = sum(kw.values())
    assert r.eps_upper == pytest.approx(cp_upper(nf, 20, 0.95), abs=1e-4)


def test_peak_equal_to_limit_refutes_and_old_table_missing():
    t = copy.deepcopy(TABLE)
    t["cells"]["box|1.0-1.5"]["levels"]["0.5"] = _lvl(10.0, over=1)
    assert not T.tilt_row(t, "box|1.0-1.5", a_plan=0.5, u_limit_deg=10.0).ok
    old = copy.deepcopy(TABLE)
    del old["cells"]["box|1.0-1.5"]["levels"]["0.5"]["n_stopped"]
    assert T.tilt_row(old, "box|1.0-1.5", a_plan=0.5, u_limit_deg=10.0).guarantee == "missing"
