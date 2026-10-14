"""PROPRIO-ONLY ZonoCtl: no tracked (d.xpos) tray / ball points; tray rim = hand-FK fixed offsets.  SIMULATION only, CPU.
python -m pytest -q world_model/v2/house/zono_rows/tests/test_proprio.py"""
import functools

import mujoco
import numpy as np

from world_model.v2.house.zono_rows import rows as R
from world_model.v2.house.zono_rows.ctl import ZonoCtl


@functools.lru_cache(maxsize=1)
def _C():
    from world_model.v2.house.zed_demo import rig as Z, execute as EX
    C, _gt = Z.build_composed(pitch_deg=30.0, dt=EX.DT, textures=False)
    mujoco.mj_forward(C.m, C.d)
    return C


def test_proprio_has_no_tracked_points_and_keeps_fk_tray():
    C = _C()
    kw = dict(zonos=None, tray_level=True, track_bodies=["tray", "ball"], tilt=True)
    off = ZonoCtl(C, proprio=False, **kw)
    assert off.tracked is not None and any("tracked:ball" in n for n in off.tracked.names)        # the privileged path is unchanged
    on = ZonoCtl(C, proprio=True, **kw)
    assert on.tracked is None                                                                      # nothing reads d.xpos of tray / ball
    names = list(on.kps["left"].names)
    assert any("tray" in n for n in names) and any("ball" in n for n in names)                    # rim + prior ball spheres are fixed offsets in the gripper frame
    drop = ZonoCtl(C, proprio=True, proprio_ball="drop", **kw)
    assert drop.tracked is None and not any("ball" in n for n in drop.kps["left"].names) and any("tray" in n for n in drop.kps["left"].names)


def test_proprio_env_default(monkeypatch):
    C = _C()
    monkeypatch.setenv("BIMANUAL_PROPRIO_ONLY", "1")
    assert ZonoCtl(C, zonos=None, track_bodies=["tray"]).proprio
    monkeypatch.setenv("BIMANUAL_PROPRIO_ONLY", "0")
    assert not ZonoCtl(C, zonos=None, track_bodies=["tray"]).proprio


def test_proprio_rows_run_and_match_tracked_clearance():
    """with a zonotope next to the tray the FK-rim rows are finite and the clearance seen by the FK rim is within a few mm of the tracked rim."""
    C = _C()
    m, d = C.m, C.d
    zono = (d.xpos[m.body("tray").id] + np.array([0.0, 0.25, 0.05]), np.diag([0.04, 0.04, 0.15]))
    hs = {}
    for tag, proprio in (("fk", True), ("trk", False)):
        ctl = ZonoCtl(C, zonos=[zono], tray_level=True, track_bodies=["tray"], tilt=True, proprio=proprio, d_act=0.4)
        dd = mujoco.MjData(m)
        mujoco.mj_copyData(dd, m, d)
        mujoco.mj_forward(m, dd)
        rl, rr, infos = ctl.zono_rows(dd)
        assert np.isfinite(rl[0]).all() and np.isfinite(rl[1]).all()
        hs[tag] = min(i.get("h_min", np.inf) for i in infos)
    assert np.isfinite(hs["fk"]) and abs(hs["fk"] - hs["trk"]) < 5e-3, hs
