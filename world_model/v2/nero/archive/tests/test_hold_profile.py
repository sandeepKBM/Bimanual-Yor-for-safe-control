"""hold_profile tests, split out of democ/tests/test_democ.py on 2026-11-03 (hold_profile.py archived).  Not collected."""
import numpy as np
import pytest

from world_model.v2.nero.democ import scene as S


def test_holdfix_default_is_off():
    from world_model.v2.nero.archive.democ import hold_profile as H
    f = H.HoldFix()
    assert f.joint_k is None and f.grip_force_N is None and f.label() == "baseline"
    assert H.FIX_DEFAULT.label() == "fix" and H.FIX_DEFAULT.grip_force_N <= H.GRIP_F_MAX
    with pytest.raises(AssertionError):
        H.run(S.DemoCParams(), fix=H.HoldFix(grip_force_N=11.0))      # above the URDF gripper effort 10 N


def test_holdfix_easy_case_improves_hold(monkeypatch):
    """Short profile, 20 deg, light box, mu 0.8: fix must not drop, stay within limits, and reduce slip vs the baseline."""
    from world_model.v2.nero.archive.democ import hold_profile as H
    monkeypatch.setattr(H, "SCHED", dict(settle=0.3, close=0.5, settle2=0.2, lift=0.8, hold1=0.6, rot=0.8, hold2=0.1, back=0.8,
                                         hold3=0.1, lower=0.8, hold4=0.1, release=0.3, tail=0.1))
    p = S.DemoCParams(box_mass=0.3, box_mu=0.8)
    rb = H.run(p, fix=H.HoldFix(), rot_deg=20.0, pad_mu=0.8)
    rf = H.run(p, fix=H.FIX_DEFAULT, rot_deg=20.0, pad_mu=0.8)
    assert not rf["box_dropped"]
    assert rf["tau_peak_Nm"]["max"] <= rf["tau_limit_Nm"] and max(v["max"] for v in rf["gripper_actuator_force_N"].values()) <= H.GRIP_F_MAX + 1e-6
    assert max(rf["slip_max_mm"].values()) < max(rb["slip_max_mm"].values())
