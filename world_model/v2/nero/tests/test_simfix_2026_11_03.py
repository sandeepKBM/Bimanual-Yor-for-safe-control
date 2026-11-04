"""tests for the 11-03 sim-fix list: arm-arm contacts (item 5), carry physics defaults (6), coop wrench sensor / friction (8, 9),
commanded acceleration (10), logic bugs (12).  SIMULATION ONLY."""
import json
import os

import mujoco
import numpy as np
import pytest

from world_model.v2.nero import adapters as A, config as C
from world_model.v2.nero.contents import carry as CR, scenes as SC
from world_model.v2.nero.democ import kin as K

LEFT = lambda m, g: (m.body(m.geom_bodyid[g]).name or "").startswith("left_arm_")
RIGHT = lambda m, g: (m.body(m.geom_bodyid[g]).name or "").startswith("right_arm_")


def _arm_arm_contacts(m, d):
    out = []
    for c in d.contact[:d.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        if (LEFT(m, g1) and RIGHT(m, g2)) or (RIGHT(m, g1) and LEFT(m, g2)):
            out.append(float(c.dist))
    return out


def _drive_arms_together(steps=1500):
    """both TCPs are commanded to the OTHER arm's TCP (PD + bias, servos off); returns (min arm-arm contact dist [m], n steps with arm-arm contact, final y gap, initial y gap)."""
    m = SC.compose(SC.BoxWorld())
    for g in range(m.ngeom):                           # 11-03: only the arms collide here; since the NYU hold pose + SCENE_DY the box walls and table block the drive
        if not (LEFT(m, g) or RIGHT(m, g)):
            m.geom_contype[g] = m.geom_conaffinity[g] = 0
    d = mujoco.MjData(m)
    cap = json.loads(CR.CAP_FILE.read_text())["pot"]
    addr = {s: A.arm_addrs(m, s) for s in CR.SIDES}
    for s in CR.SIDES:
        d.qpos[addr[s][0]] = cap["q"]["W0_liftoff"][s]
    A.disable_arm_servos(m)
    mujoco.mj_forward(m, d)
    site = {s: m.site(f"{s}_arm_ee_grasp").id for s in CR.SIDES}
    p = {s: d.site_xpos[site[s]].copy() for s in CR.SIDES}
    tgt = {}
    for s, o in (("left", "right"), ("right", "left")):
        pik = K.PoseIK(m, s)
        _, R0 = pik.ik.pose(d.qpos[addr[s][0]])
        tgt[s], _ = pik.solve(p[o], R0, d.qpos[addr[s][0]], iters=200)
    dmin, n_con = 0.0, 0
    for _ in range(steps):
        d.qfrc_applied[:] = 0
        for s in CR.SIDES:
            q, dq = d.qpos[addr[s][0]], d.qvel[addr[s][1]]
            d.qfrc_applied[addr[s][1]] = np.clip(60.0 * (tgt[s] - q) - 12.0 * dq + d.qfrc_bias[addr[s][1]], -16, 16)
        mujoco.mj_step(m, d)
        ds = _arm_arm_contacts(m, d)
        if ds:
            n_con += 1
            dmin = min(dmin, min(ds))
    mujoco.mj_forward(m, d)
    return dmin, n_con, float(d.site_xpos[site["left"]][1] - d.site_xpos[site["right"]][1]), float(p["left"][1] - p["right"][1])


@pytest.mark.skipif(C.LEGACY_ARMCOL, reason="BIMANUAL_LEGACY_ARMCOL=1: no arm-arm contacts by design")
def test_arm_arm_contacts_exist_and_arms_do_not_pass_through():
    dmin, n_con, gap_end, gap0 = _drive_arms_together()
    assert n_con > 0                                   # left and right arm geoms DO touch when driven into each other
    assert dmin > -0.01                                # worst penetration over the run (measured -8.5 mm, NYU, arms-only contacts, 16 Nm PD drive; was -3.6 mm agx): soft-contact margin


@pytest.mark.skipif(not C.LEGACY_ARMCOL, reason="legacy only")
def test_legacy_armcol_has_no_arm_arm_contacts():
    dmin, n_con, _, _ = _drive_arms_together()
    assert n_con == 0


def test_arm_arm_does_not_add_other_pairs():
    """only left-right pairs are new: same-arm and arm-world/object pairs have the same collidable bits as before."""
    m = SC.compose(SC.BoxWorld())
    ct, ca = m.geom_contype, m.geom_conaffinity
    arm = [g for g in range(m.ngeom) if (LEFT(m, g) or RIGHT(m, g)) and m.geom_group[g] == 3]
    assert arm
    for g in arm:
        if LEFT(m, g):
            assert ct[g] == (0 if C.LEGACY_ARMCOL else C.ARM_ARM_BIT) and ca[g] == 1
        else:
            assert ct[g] == 0 and ca[g] == (1 if C.LEGACY_ARMCOL else 1 | C.ARM_ARM_BIT)
    for g in range(m.ngeom):                           # no other geom carries the bit
        if g not in arm:
            assert not (ct[g] & C.ARM_ARM_BIT or ca[g] & C.ARM_ARM_BIT)


# ---------------------------------------------------------------- item 6: one physics default
def test_carry_physics_default_is_shared_by_carry_episode():
    w = SC.BoxWorld()
    cp = C.carry_physics()
    ep = CR.CarryEpisode(w)
    assert (ep.fc is not None) == bool(cp["fric_comp"])
    assert ep.weld_solref == (None if cp["weld_solref"] is None else tuple(cp["weld_solref"]))
    exp = (0.02, 1.0) if cp["weld_solref"] is None else cp["weld_solref"]
    assert np.allclose(ep.m.eq_solref[ep.weld_id["left"]], exp)
    ep0 = CR.CarryEpisode(w, fric_comp=None, weld_solref=None)           # explicit None is honoured (old behaviour)
    assert ep0.fc is None and ep0.weld_solref is None
    assert ep.physics_flags()["legacy_carry_phys"] == C.LEGACY_CARRY_PHYS


def test_carry_physics_legacy_pair():
    assert C.WELD_STIFF == (0.004, 1.0)
    d = C.carry_physics()
    if C.LEGACY_CARRY_PHYS:
        assert d == dict(fric_comp=None, weld_solref=None)
    else:
        assert d["fric_comp"] is True and d["weld_solref"] == C.WELD_STIFF


# ---------------------------------------------------------------- item 10: commanded acceleration
def test_commanded_accel_t_min_clamp():
    r = CR.commanded_accel(0.05, 1.0)              # sqrt(6 * 0.05 / 1) = 0.55 s < T_MIN = 0.6 s -> clamped
    assert r["clamped"] and r["T"] == CR.T_MIN and abs(r["a_cmd_peak"] - 6 * 0.05 / 0.36) < 1e-4 and r["a_cmd_peak"] < 1.0
    r = CR.commanded_accel(0.18, 0.25)             # sqrt(6 * 0.18 / 0.25) = 2.08 s: not clamped, a_cmd == a_plan
    assert not r["clamped"] and abs(r["a_cmd_peak"] - 0.25) < 1e-4
    r = CR.commanded_accel(0.03, 0.5)              # exactly T_MIN (box place at a_nom): not clamped
    assert not r["clamped"] and abs(r["a_cmd_peak"] - 0.5) < 1e-3


# ---------------------------------------------------------------- item 12
def test_certify_sampled_no_world_is_not_a_pass():
    from world_model.v2.nero.pan.cert import admit as AD
    r = AD._certify_sampled([], "lift", family=(0.0,))
    assert r["ok"] is False and r["reason"] == "no world qualified" and r["n_worlds"] == 0 and r["no_world"]
    r = AD._certify_sampled([dict(tag="a", segments=[dict(seg="lift", nominal=dict(viol=None), family={0.0: dict(n_bad=0)})])], "carry", family=(0.0,))
    assert r["ok"] is False and r["no_world"]                           # the world never reached this segment
    r = AD._certify_sampled([dict(tag="a", segments=[dict(seg="lift", nominal=dict(viol=None), family={0.0: dict(n_bad=0)})])], "lift", family=(0.0,))
    assert r["ok"] is True and r["n_worlds"] == 1


def test_carry_cert_counts_pre_hold_viol():
    from world_model.v2.nero.contents import carry_cert as CC
    ce = dict(ok=True, n_worlds=4, eps_conf=0.95, counterexamples=[])
    recs = [dict(tag="a", pre_hold_viol=None), dict(tag="b", pre_hold_viol=dict(kind="tau_sat", t=0.1))]
    CC._count_pre_hold(ce, recs)
    assert ce["ok"] is False and ce["n_pre_hold_viol"] == 1 and ce["counterexamples"][0]["where"] == "pre_hold" and ce["eps_upper"] > 0
    ce2 = CC._count_pre_hold(dict(ok=True, n_worlds=4), [dict(tag="a", pre_hold_viol=None)])
    assert ce2["ok"] is True and ce2["n_pre_hold_viol"] == 0


def test_agentchord_success_ours_checks_goal_and_pre_hold():
    from world_model.v2.agentchord_probe.nero import execute as X
    good = dict(xy_err=0.01, z_err=0.005)
    assert X.success_ours(True, True, None, None, good) == (True, True, False)
    assert X.success_ours(True, True, None, None, dict(xy_err=X.GOAL_TOL_M + 0.01, z_err=0.0))[1] is False      # wrong place, otherwise clean
    assert X.success_ours(True, True, None, None, dict(xy_err=0.0, z_err=X.GOAL_TOL_M + 0.01))[1] is False
    assert X.success_ours(True, True, None, dict(kind="tau_sat"), good) == (True, False, True)                 # U entry in the pre-hold phase
    assert X.success_ours(False, True, None, None, good)[1] is False


def test_extract_json_takes_the_last_object():
    from world_model.v2.vlm.spec_eval import extract_json
    assert extract_json('draft {"a": 1} final {"a": 2}') == {"a": 2}
    assert extract_json('<think>{"a": 0}</think>```json\n{"a": 3, "n": {"b": [1]}}\n```') == {"a": 3, "n": {"b": [1]}}
    assert extract_json('{"a": 1} then a broken {"a": ') == {"a": 1}
    assert extract_json("nothing") is None and extract_json("") is None


def test_arm_geoms_nonempty_for_both_arms_in_contents_scene():
    m = SC.compose(SC.BoxWorld())
    assert len(A.arm_geoms(m, "left")) > 0 and len(A.arm_geoms(m, "right")) > 0
