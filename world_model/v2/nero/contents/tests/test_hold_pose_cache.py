"""hold_pose cache key = world geometry + solver fingerprint (11-19): a patched / edited solver input must not read or write another input's entry."""
import numpy as np

from world_model.v2.nero import clearance as CL
from world_model.v2.nero.contents import nyu_hold as NH, scenes as SC
from world_model.v2.nero.pan import scene as PS


def test_same_world_same_inputs_same_key():
    assert NH._hold_key(SC.BoxWorld()) == NH._hold_key(SC.BoxWorld())
    assert NH._hold_key(SC.BoxWorld()) != NH._hold_key(SC.TrayWorld())
    assert NH._hold_key(SC.BoxWorld()) != NH._hold_key(SC.BoxWorld(obj_yaw=0.05))


def test_patched_solver_inputs_change_the_key(monkeypatch):
    k0 = NH._hold_key(SC.BoxWorld())
    with monkeypatch.context() as mp:
        mp.setattr(NH, "SHIFT_M", NH.SHIFT_M + 0.001)
        assert NH._hold_key(SC.BoxWorld()) != k0
    with monkeypatch.context() as mp:
        mp.setattr(CL, "CLEARANCE_MIN", CL.CLEARANCE_MIN + 0.001)
        assert NH._hold_key(SC.BoxWorld()) != k0
    with monkeypatch.context() as mp:
        mp.setenv("BIMANUAL_HOLD_CACHE_TEST_FLAG", "1")
        assert NH._hold_key(SC.BoxWorld()) != k0
    with monkeypatch.context() as mp:                    # a change of the compiled robot model only (not of any constant): the model arrays are in the key
        orig = PS.apply_sim_defaults

        def patched(m):
            m = orig(m)
            m.jnt_range[:] = m.jnt_range * 0.999
            return m
        mp.setattr(PS, "apply_sim_defaults", patched)
        assert NH._hold_key(SC.BoxWorld()) != k0
    assert NH._hold_key(SC.BoxWorld()) == k0             # restored


def test_entry_round_trip_and_atomic(monkeypatch, tmp_path):
    q = {"left": np.arange(7) * 0.1, "right": -np.arange(7) * 0.1}
    calls = []

    def fake_solve(world=None, **kw):
        calls.append(1)
        return dict(q=q)
    monkeypatch.setenv("BIMANUAL_PLAN_CACHE", str(tmp_path))
    monkeypatch.setattr(NH, "_HOLD_MEMO", {})
    monkeypatch.setattr(NH, "solve", fake_solve)
    w = SC.BoxWorld()
    a = NH.hold_pose(w)
    assert len(calls) == 1
    files = list((tmp_path / "hold_pose").iterdir())
    assert len(files) == 1 and files[0].suffix == ".json" and not any(f.name.endswith(".tmp") for f in files)
    monkeypatch.setattr(NH, "_HOLD_MEMO", {})            # a new process: read the entry from disk
    b = NH.hold_pose(w)
    assert len(calls) == 1
    for s in q:
        assert np.array_equal(a[s], q[s]) and np.array_equal(b[s], q[s])
