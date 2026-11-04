"""tests for contents/carry.py: weld wrench statics, stored backups, a nominal tray carry."""
import numpy as np
import pytest

from world_model.v2.nero.contents import carry as CR, scenes as SC


@pytest.fixture(scope="module")
def tray_ep():
    w = SC.sample_tray_world(2, np.random.default_rng(0))
    ep = CR.CarryEpisode(w)
    ep.run_phases([("hold", 0.4), ("move", (0, 0, 0.05), 0.8), ("hold", 1.0)])
    return ep


def test_claw_wrench_carries_the_weight(tray_ep):
    ep = tray_ep
    Fz = sum(ep.claw_wrench(s)[0][2] for s in CR.SIDES)
    W = SC.object_truth(ep.m, ep.d)["total_mass"] * 9.81
    assert abs(Fz - W) / W < 0.05                       # in the air, at rest: the two welds carry the whole object


def test_backups_are_stored_phases_only(tray_ep):
    ep = tray_ep
    for seg in ("lift", "carry", "place"):
        ph = ep.backup_phases(seg)
        assert ph[0][0] == "stop" and ("weld", False) in ph
        for p in ph:
            assert p[0] in ("stop", "to", "hold", "weld", "retreat")
            if p[0] == "to":
                assert p[1].startswith("start_")         # targets are poses stored at admission time, nothing chosen at trigger
    assert ep.backup_phases("release") is None


def test_nominal_tray_carry_ends_in_R():
    w = SC.sample_tray_world(2, np.random.default_rng(3), fill_max=0.1)
    r = CR.world_trial(w, "t", branch_backups=False)
    assert [s["seg"] for s in r["segments"]] == list(CR.SEGS)
    assert all(s["nominal"]["viol"] is None for s in r["segments"])
    assert r["end_in_R"][0]


def test_weld_solref_option_and_coop_split_zero_error():
    w = SC.sample_tray_world(2, np.random.default_rng(0))
    ep = CR.CarryEpisode(w, weld_solref=(0.004, 1.0))
    for s in CR.SIDES:
        assert np.allclose(ep.m.eq_solref[ep.weld_id[s]], (0.004, 1.0))
    from world_model.v2.nero import config as C            # default = config.carry_physics() (11-03 item 6: stiff weld; MuJoCo 0.02 s only under LEGACY_CARRY_PHYS)
    default = C.carry_physics()["weld_solref"] or (0.02, 1.0)
    assert np.allclose(CR.CarryEpisode(w).m.eq_solref[ep.weld_id["left"]], default)
    u, e_int, dq = ep._coop_split(0.0)                      # arms exactly at the commanded start pose
    assert np.linalg.norm(u) < 1e-9 and all(np.linalg.norm(dq[s]) < 1e-9 for s in CR.SIDES)
