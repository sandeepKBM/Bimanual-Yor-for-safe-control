"""F3/BA6 (11-13): do the BaseMonitors wheel counters leak from a nominal run into a counterfactual roll-out (snapshot -> branch -> restore, stack/executor.py)?
Finding: NO.  snapshot() stores off_floor / max_off / flagged / flagged_ever and restore() puts them back (copies).  reset() does not clear off_floor / max_off by design
(a new window keeps the debounce running); that is a window semantic, not a leak across restore."""
import numpy as np

from world_model.v2.nero.contents import carry as CR, scenes as SC
from world_model.v2.nero.base import monitors as MON


def test_counters_restored_by_snapshot_restore():
    ep = CR.CarryEpisode(SC.TrayWorld(n_cups=2), base="swerve", goal_support="counter2")
    bm = ep.bmon
    bm.off_floor[:] = [3, 0, 7, 1]; bm.max_off = 1
    sn = ep.snapshot()
    bm.off_floor[:] = [20, 20, 20, 20]; bm.max_off = 4; bm.flagged.add("wheel_liftoff"); bm.flagged_ever.add("wheel_liftoff")   # a counterfactual branch
    ep.restore(sn)
    assert np.array_equal(bm.off_floor, [3, 0, 7, 1]) and bm.max_off == 1 and not bm.flagged and not bm.flagged_ever


def test_reset_keeps_debounce_counters_by_design():
    ep = CR.CarryEpisode(SC.TrayWorld(n_cups=2), base="swerve", goal_support="counter2")
    bm = ep.bmon
    bm.off_floor[:] = 5; bm.max_off = 2; bm.flagged.add("wheel_liftoff")
    bm.reset()
    assert not bm.flagged and np.all(bm.off_floor == 5) and bm.max_off == 2
    assert MON.LIFTOFF_STEPS == 10


def test_mark_contents_box_world():
    ep = CR.CarryEpisode(SC.BoxWorld(shelf=False), base="swerve", goal_support="counter2")
    r = ep.mark_contents("start")
    assert r is not None and len(r["xyz_world"]) == 3 and len(r["xyz_in_obj"]) == 3 and ep.contents_poses["start"] is r
