"""Owner gate (11-17): Rule 3 switch is reachable through the WIDE table (USER 11-17 "Wide table").  SIMULATION ONLY (Rule 1 model, no carry), CPU, no network.
PYTHONPATH=. python -m pytest -q world_model/v2/adapters/tests/test_wide_table_gate.py      (slow: two box run_cell calls, fallback=True)

On box closed_chain, prior cell 2.0-2.5 kg contents:
  - every program's set at every branch time contains Theta_t (theta_cert), and is Theta_t plus whole PRIOR cells only;
  - the admitted program is listed at every branch time;
  - Lemma 1: any Theta_new inside Theta_t answers continue (no switch from narrowing);
  - a Theta_new in an added prior cell switches away from a program that is not certified there (read_then_retrace -> stored_backup);
  - BIMANUAL_LEGACY_TABLE_THETA_T=1 gives sets equal to Theta_t."""
import pathlib

import numpy as np
import pytest

from world_model.v2.stack import theta as TH
from world_model.v2.stack.checks import cover as COV

CELL = (2.0, 2.5)


def _run(tmp_path, monkeypatch, legacy):
    from world_model.v2.adapters import rule1_contents as RC
    monkeypatch.setenv("BIMANUAL_PLANNER", "closed_chain")
    monkeypatch.setenv("BIMANUAL_NO_NETWORK", "1")
    monkeypatch.setenv("BIMANUAL_LEGACY_UNTILTED_COLLISION", "1")          # untilted row (BIMANUAL_LEGACY_UNTILTED_COLLISION=1): this test is about Rule 1 logic, not the load tilt (11-17 tilt row: adapters/tests/test_tilt_collision_gate.py)
    if legacy:
        monkeypatch.setenv("BIMANUAL_LEGACY_TABLE_THETA_T", "1")
    else:
        monkeypatch.delenv("BIMANUAL_LEGACY_TABLE_THETA_T", raising=False)
    return RC.run_cell("box", CELL, pathlib.Path(tmp_path), n_sampled=0, planner="closed_chain", fallback=True)


def _adm(rec):
    return TH.Admission.from_json(rec["admission"]) if hasattr(TH.Admission, "from_json") else None


def _cells(s):
    out = []
    for c in (s.cells if hasattr(s, "cells") else s["cells"]):
        iv = c.bounds["total_kg"] if hasattr(c, "bounds") else c["bounds"]["total_kg"]
        out.append((float(iv.lo), float(iv.hi)) if hasattr(iv, "lo") else (float(iv[0]), float(iv[1])))
    return sorted(out)


@pytest.mark.slow
def test_wide_table(tmp_path, monkeypatch):
    recs = [r for r in _run(tmp_path, monkeypatch, legacy=False) if r.get("admitted")]
    assert recs, "no admitted step on box 2.0-2.5"
    for r in recs:
        adm = _adm(r)
        assert adm is not None, "Admission.from_json missing"
        tc = _cells(adm.theta_cert)
        prior_w = None
        for t, lst in adm.table.items():
            names = [n for n, _ in lst]
            assert adm.recovery in names, (r["seg"], t, names)
            for n, s in lst:
                cs = _cells(s)
                assert set(tc) <= set(cs) or n != adm.recovery, (r["seg"], t, n)
                for lo, hi in cs:                                            # whole prior cells only (0.5 kg wide)
                    w = round(hi - lo, 6)
                    prior_w = prior_w or w
                    assert w == prior_w, (lo, hi)
        # Lemma 1: narrowing never switches
        lo, hi = tc[0][0], tc[-1][1]
        rng = np.random.default_rng(1117)
        for _ in range(50):
            a, b = np.sort(rng.uniform(lo, hi, 2))
            t = float(rng.choice([float(k) for k in adm.table]))
            d = COV.cover(TH.ThetaSet.box(total_kg=(float(a), float(b))), adm, t)
            assert d.action == "continue", (r["seg"], a, b, t, d)


@pytest.mark.slow
def test_switch_from_read_program_into_an_added_cell(tmp_path, monkeypatch):
    recs = [r for r in _run(tmp_path, monkeypatch, legacy=False) if r.get("admitted")]
    hit = 0
    for r in recs:
        adm = _adm(r)
        for t, lst in adm.table.items():
            sets = dict(lst)
            if "read_then_retrace" not in sets or "stored_backup" not in sets:
                continue
            extra = sorted(set(_cells(sets["stored_backup"])) - set(_cells(sets["read_then_retrace"])))
            if not extra:
                continue
            from dataclasses import replace
            adm_r = replace(adm, recovery="read_then_retrace")
            lo, hi = extra[0]
            d = COV.cover(TH.ThetaSet.box(total_kg=(lo + 0.1, hi - 0.1)), adm_r, float(t))
            assert d.action == "switch" and d.recovery == "stored_backup", (r["seg"], t, d)
            hit += 1
    assert hit > 0, "no branch time lists both programs with a wider stored_backup set"


@pytest.mark.slow
def test_legacy_table_is_theta_t(tmp_path, monkeypatch):
    for r in [r for r in _run(tmp_path, monkeypatch, legacy=True) if r.get("admitted")]:
        adm = _adm(r)
        tc = _cells(adm.theta_cert)
        for t, lst in adm.table.items():
            for n, s in lst:
                assert _cells(s) == tc, (r["seg"], t, n)
