"""coffee-tray port (nero/contents/coffee.py): world, scene, capsule phase success.  CPU, ~1 min."""
import os

import numpy as np
import pytest


def _setup(variant):
    os.environ["BIMANUAL_COFFEE_VARIANT"] = variant
    from world_model.v2.nero.contents import coffee as CF
    CF.install()
    return CF


def test_prior_and_worlds():
    CF = _setup("unknown")
    ws = [CF.world_for(i) for i in range(20)]
    assert all(0.0 <= w.load_kg <= 3.0 for w in ws)
    assert all(abs(w.load_xy[0]) <= CF.INNER_X - CF.LOAD_MARGIN and abs(w.load_xy[1]) <= CF.INNER_Y - CF.LOAD_MARGIN for w in ws)
    from world_model.v2.stack import tasks as TK
    assert CF.prior() == (0.0, 3.0) == TK.prior_box("tray0")          # the ONE copy of the prior is stack/tasks.py
    os.environ["BIMANUAL_COFFEE_VARIANT"] = "known"
    assert CF.prior() == (0.0, 0.0) and CF.world_for(0).load_kg == 0.0


def test_capsule_phase_and_carry_known():
    CF = _setup("known")
    from world_model.v2.nero.contents import carry as CR
    ep = CR.CarryEpisode(CF.world_for(0), m_hat=0.0)
    start = CF.tray_ref(ep).copy()
    ep.ctl.mass = 0.0
    ep.run_phases([("hold", 0.3)])
    log = CF.run_capsule_phase(ep)
    assert not ep.track["viol"], ep.track["viol"]
    from world_model.v2.adapters import outcome_carry as OCC
    m = OCC.measure(ep, "tray0", False, False)                     # the one goal path: capsule terms inside, finished False -> not a success
    assert abs(m["measured"]["cap_rel_x"]) <= CF.INNER_X and abs(m["measured"]["cap_rel_y"]) <= CF.INNER_Y and m["result"]["success"] is False
    ep.ctl.mass = ep.m_nom
    ep.run_phases([("hold", 0.4)])
    rec = CR.run_segments(ep, dict(segments=[]), branch_backups=False)
    assert rec["end_in_R"][0]
    m = OCC.measure(ep, "tray0", True, False)
    assert m["result"]["success"] and m["goal"]["ok"], m["goal"]
    assert CF.general_result(ep, True)["result"]["success"]
