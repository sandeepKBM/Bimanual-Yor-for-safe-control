"""drift tests: the pure-data stack/tasks.py table equals the nero values it was copied from; admit / payload / readers reproduce the old code."""
import math

import numpy as np
import pytest

from world_model.v2.adapters import contents_specs as CS
from world_model.v2.adapters import payload_envelope as PE
from world_model.v2.stack import tasks as TK
from world_model.v2.stack.checks import admit as AD
from world_model.v2.stack.checks import payload as PAY
from world_model.v2.stack.checks import probe_rung as PR


def _u(t, q):
    return TK.task(t).U_dict()[q]


def test_carry_thresholds():
    from world_model.v2.nero.contents import carry as CR
    for n in ("tray2", "tray4", "tray6", "box", "pot"):
        t = TK.task(n)
        assert t.U_dict()["obj_tilt_deg"].hi == CR.U_TILT_OBJ and t.R_dict()["obj_tilt_deg"].hi == CR.R_TILT_OBJ
        assert t.R_dict()["obj_speed"].hi == CR.R_SPEED
        assert t.U_dict()["drop_m"].hi == 0.01
    assert _u("tray2", "cup_tilt_deg").hi == CR.U_TILT_CUP and TK.task("tray2").R_dict()["cup_tilt_deg"].hi == CR.R_TILT_CUP
    p = TK.task("pot")
    assert p.U_dict()["lid_slide_m"].hi == CR.LID_SLIDE_M and p.U_dict()["lid_dz_m"].hi == CR.LID_OFF_Z and p.U_dict()["lid_tilt_deg"].hi == CR.LID_TILT_DEG
    assert p.R_dict()["lid_slide_m"].hi == CR.LID_SLIDE_M


def test_plans_and_goal_dxy():
    from world_model.v2.nero.contents import carry as CR
    from world_model.v2.stack.planners import closed_chain as CCP
    for n in ("tray2", "tray6", "box", "pot"):
        t = TK.task(n)
        assert [(s, tuple(d)) for s, d in CR.plan_moves(t.kind if t.kind != "tray" else "tray")] == [(s, tuple(d)) for s, d in t.moves]
        assert not hasattr(CCP, "GOAL_DXY")                                           # deleted 11-09: stack/tasks.py is the one copy
    assert TK.GOAL_DXY == {t.kind: t.goal_dxy for t in TK.TASKS.values() if t.backend == "carry" and not t.base}


def test_behind_goal_matches_scene():
    from world_model.v2.nero.base import scene as BS
    from world_model.v2.nero.contents import scenes as SC
    for n, w in (("tray6_behind", SC.TrayWorld(n_cups=6)), ("box_behind", SC.BoxWorld(shelf=False))):
        t = TK.task(n)
        d = BS.place_pose_xy(w) - np.asarray(w.obj_xy)
        assert t.base == "swerve" and t.goal_support == "counter2" and t.moves == ()
        assert np.allclose(t.goal_dxy, d, atol=1e-4), (n, t.goal_dxy, d)
        assert CS.build_world(n) == w                               # the builder makes the same world (box: no shelf)


def test_prior_has_one_copy_and_matches_contents():
    from world_model.v2.agentchord_probe.nero import tasks as AT
    from world_model.v2.nero.contents import probe, probe_trace
    from world_model.v2.vlm import contents_eval as CE
    assert not any(hasattr(m, "PRIOR") for m in (probe, probe_trace, AT))
    assert CE.PRIOR_FILL == TK.prior_box("tray2") and CE.PRIOR_BOX == TK.prior_box("box")
    for n in ("tray2", "tray4", "tray6", "box", "pot"):
        t = TK.task(n)
        assert (t.prior.lo, t.prior.hi) == TK.prior_box(n)
        if n == "pot":
            from world_model.v2.nero.contents import scenes as SC
            assert TK.prior_box(n) == tuple(SC.POT_CONTENTS_KG)          # scenes.POT_CONTENTS_KG = pan/theta.py THETA mass (the sim's own range)
        # total-mass prior = probe.prior_contents
        assert CS.prior_total(n) == pytest.approx(probe.prior_contents(n))


def test_object_dims_and_support_shape():
    from world_model.v2.nero.contents import objects as OB, scenes as SC
    assert TK.task("box").shape == TK.Rect(OB.BOX["lx"], OB.BOX["ly"]) and TK.task("tray2").shape == TK.Rect(OB.TRAY["lx"], OB.TRAY["ly"])
    assert TK.task("pot").shape.r == pytest.approx(SC.PS.ASSUMED["pan_radius"] + SC.PS.ASSUMED["pan_wall_t"])
    assert TK.task("box").empty_mass == OB.BOX["mass"] and TK.task("box").empty_com == (0.0, 0.0, OB.BOX["lz"] / 2)
    assert TK.task("tray2").empty_mass == OB.TRAY["mass"]
    assert CS.empty_props("pot")[0] == pytest.approx(SC.pot_empty_props()[0])
    assert TK.task("tray4").n_cups == 4


def test_pan_and_goal_constants():
    from world_model.v2.nero.pan import segments as S
    from world_model.v2.nero.pan.handle_grasp import LIFT
    t = TK.task("pan_lift")
    u, r = t.U_dict(), t.R_dict()
    assert (u["obj_tilt_deg"].hi, u["lid_mm"].hi, u["drop_mm"].hi, u["joint_margin_rad"].lo, u["tau_ratio"].hi) == (
        S.U["tilt_deg"], S.U["lid_mm"], S.U["drop_mm"], S.U["joint_tol"], S.U["effort"] + 1e-6)   # _monitor allows effort + 1e-6
    assert (r["obj_tilt_deg"].hi, r["abs(dz_mm)"].hi, r["lid_mm"].hi, r["support_frac"].lo, r["pan_speed"].hi, r["arm_speed"].hi, r["abs(yaw_deg)"].hi) == (
        S.R["tilt_deg"], S.R["dz_mm"], S.R["lid_mm"], S.R["support_frac"], S.R["pan_speed"], S.R["arm_speed"], S.R["yaw_deg"])
    g = dict(t.goal_terms)
    assert g["rise_mm"].lo == pytest.approx(0.8 * LIFT * 1e3)
    for n in ("box", "tray6", "pot"):                  # AgentChord ac_success (USER 11-09): xy <= 0.05, z <= 0.02, tilts < 45 deg strict
        gt = dict(TK.task(n).goal_terms)
        assert (gt["xy_err"].hi, gt["z_err"].hi, gt["obj_tilt_deg"].hi, gt["obj_tilt_deg"].strict) == (0.05, 0.02, 45.0, True)
    assert dict(TK.task("tray6").goal_terms)["cup_tilt_deg"].hi == 45.0


def test_coffee_tray_numbers():
    import json
    import pathlib
    import world_model.v2.agentchord_probe as AP
    geom = json.load(open(pathlib.Path(AP.__file__).parent / "coffee_tray" / "geometry.json"))
    t = TK.task("coffee_tray")
    assert (TK._CT_LX, TK._CT_LY, TK._CT_RIM_T) == (geom["tray"]["lx"], geom["tray"]["ly"], geom["tray"]["rim_t"])
    assert t.goal_dxy == tuple(geom["forward"])
    q = dict(t.goal_terms)
    assert q["abs(cap_rel_x)"].hi == pytest.approx(geom["tray"]["lx"] / 2 - geom["tray"]["rim_t"])
    assert q["abs(cap_rel_y)"].hi == pytest.approx(geom["tray"]["ly"] / 2 - geom["tray"]["rim_t"])


@pytest.mark.parametrize("task,cell", [("box", (0.5, 1.0)), ("box", (4.5, 5.0)), ("tray6", (0.125, 0.25)), ("tray2", (0.0, 0.05)), ("pot", (1.25, 1.5))])
def test_admit_equals_static_cell(task, cell):
    from world_model.v2.nero.contents import carry_cert as CC
    t = TK.task(task)
    w = CS.build_world(task)
    old = CC.static_cell(task, cell, w, planner="scripted")
    be = PE.EnvelopeBackend()
    new = AD.admit(t, CS.spec_from_cell(task, *cell), PE.scripted_path(task), be, be.support_extent(w, t.goal_support))
    assert new.action == old["action"]
    assert (new.a_admit is None) == (old["a_admit"] is None)
    if old["a_admit"] is not None:
        assert new.a_admit == pytest.approx(old["a_admit"])
    assert new.u_nom == pytest.approx(old["u_nom"], abs=1e-4) and new.u_one_nom == pytest.approx(old["u_one_nom"], abs=1e-4)
    assert new.rows[-1].margin == pytest.approx(old["rows"]["release"]["support_dest"]["margin_m"], abs=1e-4)
    assert new.rows[-1].ok == old["rows"]["release"]["support_dest"]["ok"]
    for seg in ("lift", "carry", "place"):
        r = next(x for x in new.rows if x.name == f"torque_payload[{seg}]")
        assert r.worst["u_max"] == pytest.approx(old["rows"][seg]["torque_payload"]["u_max"], abs=1e-4)
    assert new.verdict.admit == (not old["action"].startswith("refuse"))
    _same_dict_shape(new, old)


def _same_dict_shape(new, old):
    """item 15: Admission carries the fields static_cell's callers read (worst_nom / worst_one / contents_m / per-segment rows dict)."""
    assert list(new.contents_m) == pytest.approx(old["contents_m"]) and new.worst_nom.keys() == old["worst_nom"].keys() and new.worst_one.keys() == old["worst_one"].keys()
    assert list(new.seg_rows) == list(old["rows"]), (list(new.seg_rows), list(old["rows"]))
    for seg, rows in old["rows"].items():
        assert set(new.seg_rows[seg]) == set(rows), seg
        if "torque_payload" in rows:
            assert new.seg_rows[seg]["torque_payload"]["ok"] == rows["torque_payload"]["ok"]
            assert new.seg_rows[seg]["torque_payload"]["u_max"] == pytest.approx(rows["torque_payload"]["u_max"], abs=1e-4)
        if "support_dest" in rows:
            assert new.seg_rows[seg]["support_dest"]["ok"] == rows["support_dest"]["ok"]
            assert new.seg_rows[seg]["support_dest"]["margin_m"] == pytest.approx(rows["support_dest"]["margin_m"], abs=1e-4)


def _closed_chain_path(task):
    """the closed_chain planner's PLANNED path (samples, planned joint configurations, goal offset) as admit's PathSamples.  Lives in the test: an adapter
    importing stack.planners.closed_chain would reach the forbidden layers (dependency rule); the step-2 caller passes the path in."""
    from world_model.v2.stack.planners import closed_chain as CCP
    offs, per_seg, cap = CCP.static_path(task, None)
    return AD.PathSamples(offs, per_seg, cap, np.asarray(CCP.goal_offset(task, None), float))


# closed_chain planner (the planned path, not the scripted one): u_nom is finite here, so a payload refusal is actually exercised
#   box (4.5, 5.0): refused (payload u 1.7 and support); tray6 (0.10, 0.15): total ~3.0 kg, u_nom 0.91: slowed at U_ADMIT 0.9, full speed at U_ADMIT 1.0 (USER 11-09); pot (1.25, 1.5): admitted,
#   its COM 1.9 cm inside the bottom disc (the disc binds)
@pytest.mark.parametrize("task,cell,action", [("box", (4.5, 5.0), "refuse"), ("tray6", (0.10, 0.15), "two arms"), ("pot", (1.25, 1.5), "two arms")])
def test_admit_equals_static_cell_closed_chain(task, cell, action):
    from world_model.v2.nero.contents import carry_cert as CC
    t = TK.task(task)
    w = CS.build_world(task)
    old = CC.static_cell(task, cell, w, planner="closed_chain")
    assert old["action"].startswith(action) and old["u_nom"] != float("inf")
    be = PE.EnvelopeBackend()
    new = AD.admit(t, CS.spec_from_cell(task, *cell), _closed_chain_path(task), be, be.support_extent(w, t.goal_support))
    assert new.action == old["action"]
    assert (new.a_admit is None) == (old["a_admit"] is None)
    if old["a_admit"] is not None:
        assert new.a_admit == pytest.approx(old["a_admit"])
    assert new.u_nom == pytest.approx(old["u_nom"], abs=1e-4) and new.u_one_nom == pytest.approx(old["u_one_nom"], abs=1e-4)
    assert new.verdict.admit == (not old["action"].startswith("refuse"))
    sd = old["rows"]["release"]["support_dest"]
    sup = next(r for r in new.rows if r.name == "support_dest")
    assert sup.ok == sd["ok"] and sup.margin == pytest.approx(sd["margin_m"], abs=1e-4)
    if task == "pot":
        assert "disc" in sd and sup.worst["shape"]["kind"] == "Disc"
    _same_dict_shape(new, old)
    if task == "box":                                            # refused by the payload row: a failed torque row, guarantee refuted
        assert any(r.name.startswith("torque_payload") and not r.ok for r in new.rows) and new.verdict.guarantee == "refuted"


def test_admit_refused_counterpart():
    from world_model.v2.nero.contents import carry_cert as CC
    spec = CS.spec_from_cell("box", 0.5, 1.0)
    old = CC.refused_static("box", (0.5, 1.0), spec, "goal unreachable")
    new = AD.refused(TK.task("box"), spec, "goal unreachable")
    assert new.action == old["action"] and new.a_admit is None and new.u_nom == old["u_nom"] == float("inf") and new.a_open == old["a_open"]
    assert list(new.contents_m) == old["contents_m"] and new.why == old["why"] and new.planner == old["planner"]
    assert not new.verdict.admit and new.verdict.guarantee == "missing" and "goal unreachable" in new.verdict.why


def test_probe_rung_equals_old():
    from world_model.v2.nero.contents import probe as P
    for task, m in (("box", 2.0), ("box", 5.0), ("tray6", 4.0)):
        rest = PE.rest_backend(task)
        e = CS.empty_props(task)[0]
        r = PR.rung_row(rest, e, m, M_lo=m - 0.25)
        assert r.worst["u"] == pytest.approx(P.rung_torque_u(task, m), abs=1e-9)
        ok, d = P.Probe.admissible.__get__(type("X", (), dict(task=task, M_lo=m - 0.25))())(m)
        assert r.ok == ok
        assert PAY.U_ADMIT == __import__("world_model.v2.nero.contents.envelope", fromlist=["E"]).U_ADMIT
        assert AD.A_NOM == __import__("world_model.v2.nero.contents.envelope", fromlist=["E"]).A_NOM


def test_open_content_limit_matches_objects():
    from world_model.v2.nero.contents import objects as OB
    from world_model.v2.stack.checks import open_content as OCN
    assert OCN.G == OB.G
    assert OCN.spill_limit(0.08, 0.095, 0.06) == OB.cup_spill_limit(0.08, 0.095, 0.06)
