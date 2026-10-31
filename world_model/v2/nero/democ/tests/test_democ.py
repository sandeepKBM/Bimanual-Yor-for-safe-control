"""Demo C scene v0 tests.  SIMULATION ONLY, CPU.   PYTHONPATH=. python -m pytest -q world_model/v2/nero/democ/tests   (~30 s)"""
import numpy as np
import pytest
import mujoco

from world_model.v2.nero import adapters as A
from world_model.v2.nero.democ import scene as S
from world_model.v2.nero.democ import kin as K


@pytest.fixture(scope="module")
def setup():
    p = S.DemoCParams()
    m = S.compose(p)
    pre = K.find_pregrasp(m, p)
    return p, m, pre


def test_scene_compiles_and_params_apply():
    p = S.DemoCParams(box_mass=0.9, box_com_offset=(0.01, 0.0, 0.0), box_mu=1.3)
    m = S.compose(p)
    b = m.body("democ_box").id
    assert abs(m.body_mass[b] - 0.9) < 1e-9
    assert np.allclose(m.body_ipos[b], [0.01, 0, 0])
    assert abs(m.geom_friction[m.geom("democ_box_geom").id][0] - 1.3) < 1e-9
    # pad geoms can collide with the box (bit check, same rule MuJoCo uses)
    gb = m.geom("democ_box_geom").id
    for side in ("left", "right"):
        for n in ("finger1_col", "finger2_col"):
            g = m.geom(f"{side}_arm_{n}").id
            assert (m.geom_contype[g] & m.geom_conaffinity[gb]) or (m.geom_contype[gb] & m.geom_conaffinity[g])


def test_written_mjcf_loads():
    path = S.write_mjcf(S.DemoCParams())
    m = mujoco.MjModel.from_xml_path(str(path))
    assert m.nbody > 30 and m.body("democ_box").id > 0


def test_initial_state_only_box_table_contacts(setup):
    p, m, pre = setup
    d = mujoco.MjData(m)
    K.set_state(m, d, p, pre["ql"], pre["qr"], K.FINGER_OPEN)
    assert K.bad_contacts(m, d) == []                     # nothing but box/pedestal/table/floor contacts
    names = {frozenset((m.geom(c.geom1).name, m.geom(c.geom2).name)) for c in d.contact[:d.ncon]}
    assert all(n <= set(K.NAMES_OK) for n in names)


def test_pregrasp_reachable_ik_residual(setup):
    p, m, pre = setup
    assert pre["el"] < 1e-6 and pre["er"] < 1e-6
    tg = K.box_grasp_targets(p, 0.0, pre["jaw_sign"])
    for side, q in (("left", pre["ql"]), ("right", pre["qr"])):
        pik = K.PoseIK(m, side)
        assert np.linalg.norm(pik.err(q, *tg[side])) < 1e-6
        assert np.all(q >= pik.lo) and np.all(q <= pik.hi)


# ------------------------------------------------------------------------------------------------ hold fix (opt-in)
def test_holdfix_default_is_off():
    from world_model.v2.nero.democ import hold_profile as H
    f = H.HoldFix()
    assert f.joint_k is None and f.grip_force_N is None and f.label() == "baseline"
    assert H.FIX_DEFAULT.label() == "fix" and H.FIX_DEFAULT.grip_force_N <= H.GRIP_F_MAX
    with pytest.raises(AssertionError):
        H.run(S.DemoCParams(), fix=H.HoldFix(grip_force_N=11.0))      # above the URDF gripper effort 10 N


def test_holdfix_easy_case_improves_hold(monkeypatch):
    """Short profile, 20 deg, light box, mu 0.8: fix must not drop, stay within limits, and reduce slip vs the baseline."""
    from world_model.v2.nero.democ import hold_profile as H
    monkeypatch.setattr(H, "SCHED", dict(settle=0.3, close=0.5, settle2=0.2, lift=0.8, hold1=0.6, rot=0.8, hold2=0.1, back=0.8,
                                         hold3=0.1, lower=0.8, hold4=0.1, release=0.3, tail=0.1))
    p = S.DemoCParams(box_mass=0.3, box_mu=0.8)
    rb = H.run(p, fix=H.HoldFix(), rot_deg=20.0, pad_mu=0.8)
    rf = H.run(p, fix=H.FIX_DEFAULT, rot_deg=20.0, pad_mu=0.8)
    assert not rf["box_dropped"]
    assert rf["tau_peak_Nm"]["max"] <= rf["tau_limit_Nm"] and max(v["max"] for v in rf["gripper_actuator_force_N"].values()) <= H.GRIP_F_MAX + 1e-6
    assert max(rf["slip_max_mm"].values()) < max(rb["slip_max_mm"].values())
