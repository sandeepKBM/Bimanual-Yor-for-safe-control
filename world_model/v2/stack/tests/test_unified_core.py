"""unit tests of the unified stack core (pure: tasks, goal, state, outcome, checks/*); no sim, no nero import."""
import math

import numpy as np
import pytest

from world_model.v2.stack import goal as GL
from world_model.v2.stack import outcome as OC
from world_model.v2.stack import registry as REG
from world_model.v2.stack import tasks as TK
from world_model.v2.stack.checks import dynamic as DY
from world_model.v2.stack.checks import monitors as MON
from world_model.v2.stack.checks import open_content as OCN
from world_model.v2.stack.checks import plan_shape as PSH
from world_model.v2.stack.checks import support as SUP
from world_model.v2.stack.checks import tilt as TL
from world_model.v2.stack.checks.row import Row, bound_margin, combine, inside, missing
from world_model.v2.stack.state import EndState, StepState
from world_model.v2.stack.types import Bound


class FakeContents:
    def __init__(self, m_lo, m_hi, P):
        self.m_lo, self.m_hi, self.P = m_lo, m_hi, np.asarray(P, float)

    def pi_vertices(self):
        return np.array([np.concatenate([[m], m * v]) for m in (self.m_lo, self.m_hi) for v in self.P])


class World:
    obj_xy = (0.38, -0.30)
    counter_top, shelf_top = 0.82, 0.97


def test_every_bound_is_tagged():
    for t in TK.TASKS.values():
        for tab in (t.U, t.R, t.goal_terms):
            for q, b in tab:
                assert b.tag in ("MEASURED", "MODEL", "ASSUMED") and b.src, (t.name, q)
        if t.prior:
            assert t.prior.tag == "ASSUMED"
        for _, b in t.reader:
            assert b.tag in ("MEASURED", "MODEL", "ASSUMED") and b.src
        if t.backend == "carry":                                     # shape / empty mass / empty COM are tagged facts (AGENTS.md bounds rule)
            tags = {f: (tag, src) for f, tag, src in t.geom_tags}
            assert set(tags) == {"shape", "empty_mass", "empty_com"} and all(tg in ("MEASURED", "MODEL", "ASSUMED") and sr for tg, sr in tags.values()), t.name
    assert {f: tag for f, tag, _ in TK.task("tray2").geom_tags}["shape"] == "MODEL" and {f: tag for f, tag, _ in TK.task("box").geom_tags}["shape"] == "ASSUMED"
    assert TK.task("pot").U_dict()["tau_ratio"].tag == "MODEL" and TK.task("tray2").U_dict()["tau_ratio"].tag == "MODEL"      # item 12


def test_scripted_moves_end_at_goal():
    for n in ("tray2", "tray4", "tray6", "box", "pot"):
        t = TK.task(n)
        assert t.goal_end_dxy() == pytest.approx(t.goal_dxy)
    assert TK.GOAL_DXY == {"tray": (-0.10, 0.0), "box": (-0.20, 0.0), "pot": (-0.10, 0.0)}


def test_shapes():
    assert isinstance(TK.task("box").shape, TK.Rect) and isinstance(TK.task("tray4").shape, TK.Rect)
    assert isinstance(TK.task("pot").shape, TK.Disc) and TK.task("pot").shape.r == pytest.approx(0.0925)
    assert TK.task("pan_lift").shape is None and TK.task("coffee_tray").shape is None          # nothing reads a pan / coffee-tray shape


def test_row_sign_and_combine():
    b = TK.ub(10.0, "ASSUMED", "x")
    assert bound_margin(8.0, b) == 2.0 and bound_margin(11.0, b) == -1.0
    assert bound_margin(3.0, TK.lb(5.0, "ASSUMED", "x")) == -2.0
    r1 = Row("a", True, 0.1, guarantee="proof")
    r2 = Row("b", True, 0.2, guarantee="sampled", n=5)
    v = combine([r1, r2], "t")
    assert v.admit and v.guarantee == "sampled"
    v = combine([r1, Row("c", False, -0.1, guarantee="proof", why="bad")])
    assert not v.admit and v.guarantee == "refuted" and "bad" in v.why
    v = combine([r1, TL.tilt_row()])
    assert v.admit and v.guarantee == "missing" and v.assumptions["missing_rows"] == ["tilt"]
    v = combine([r1, Row("m", False, -1.0, guarantee="runtime")])
    assert v.admit and v.guarantee == "proof" and v.assumptions["runtime_rows"] == ["m"]
    with pytest.raises(ValueError):
        Row("x", True, guarantee="bogus")


def test_support_rect_and_disc():
    P = [[-0.05, -0.05, 0.05], [0.05, 0.05, 0.05]]
    c = FakeContents(1.0, 1.0, P)
    sx, sy = (0.0, 0.5), (-0.5, 0.0)
    # Rect box 0.20 x 0.30, COM offset 5 cm from the centre: well inside -> margin = 0.10 - ... > 0
    r = SUP.support_row(TK.task("box"), (0.25, -0.25), sx, sy, 0.25, (0, 0, 0.075), c)
    assert r.ok and r.margin > 0 and r.guarantee == "proof"
    # support edge right at the centre: COM leaves the clipped rectangle
    r = SUP.support_row(TK.task("box"), (0.0, -0.25), sx, sy, 0.25, (0, 0, 0.075), c)
    assert not r.ok and r.margin < 0
    # disc binds where the bounding square does not: COM offset along the diagonal of the square
    far = FakeContents(5.0, 5.0, [[0.07, 0.07, 0.05]])
    rd = SUP.support_row(TK.task("pot"), (0.25, -0.25), sx, sy, 0.38, (0, 0, 0.03), far)
    assert not rd.ok or rd.margin < 0.0925 - 0.0           # margin is capped by the disc test
    r_sq = 0.0925 - float(np.hypot(*(5.0 * np.array([0.07, 0.07]) / 5.38)))
    assert rd.margin == pytest.approx(min(r_sq, 0.0925 - abs(5.0 * 0.07 / 5.38)))


def test_open_content_one_spill_formula():
    D, H, h = 0.08, 0.095, 0.085
    a = OCN.spill_limit(D, H, h)
    assert OCN.spill_ratio(D, H - h, a) == pytest.approx(1.0)
    assert OCN.spill_ratio(D, H - h, 0.5 * a) == pytest.approx(0.5)
    assert OCN.open_content_row(0.4, a).ok and not OCN.open_content_row(a + 0.1, a).ok
    assert OCN.open_content_row(0.4, a).margin == pytest.approx(a - 0.4)


def test_dynamic_cp_upper_and_family():
    assert DY.cp_upper(0, 50) == pytest.approx(1 - 0.05 ** (1 / 50))
    assert DY.cp_upper(5, 5) == 1.0
    ok = DY.counts_row("d", 30, 0)
    assert ok.ok and ok.guarantee == "sampled" and ok.n == 30 and ok.eps_upper == pytest.approx(round(DY.cp_upper(0, 30), 4))
    assert not DY.counts_row("d", 0, 0).ok
    recs = [dict(tag=str(i), segments=[dict(seg="lift", nominal=dict(viol=None), family={0.06: dict(n=3, n_bad=int(i == 1), first_bad=None),
                                                                                         0.03: dict(n=3, n_bad=0, first_bad=None)})]) for i in range(4)]
    r = DY.family_row(recs, "lift", (0.06, 0.03))
    assert r.ok and r.worst["backup_retreat"] == 0.03 and r.n == 4
    recs[2]["segments"][0]["family"][0.03]["n_bad"] = 1
    recs[2]["segments"][0]["family"][0.03]["first_bad"] = dict(t=1)
    assert not DY.family_row(recs, "lift", (0.06, 0.03)).ok


def _carry_state(**kw):
    now = dict(obj_x=0.28, obj_y=-0.30, obj_z=0.85, obj_z0=0.85 + 0.15 * 0, obj_tilt_deg=1.0, obj_speed=0.0, support_contact=1.0, cup_tilt_deg=0.0,
               tau_ratio=0.2, drop_m=0.0)
    now.update(kw)
    return EndState(t=1.0, now=now, worst=dict(now, spill_ratio=0.1, collision_flag=0.0, cup_off_tray_flag=0.0))


def test_goal_box_synthetic():
    w = World()
    g = GL.goal_for("box", w)
    assert g.ref["gx"] == pytest.approx(0.18) and g.ref["dz"] == pytest.approx(0.15)
    s = _carry_state(obj_x=0.19, obj_y=-0.30, obj_z=0.85 + 0.15 + 0.01)
    r = GL.check_goal(g, s)
    assert r.ok and r.worst["xy_err"] == pytest.approx(0.05 - math.hypot(0.01, 0.0)) and r.worst["z_err"] == pytest.approx(0.01)
    # AgentChord ac_success terms (USER 11-09): xy <= 0.05, z <= 0.02, object tilt < 45 (strict), mugs < 45 (tray only)
    assert GL.check_goal(g, _carry_state(obj_x=0.229, obj_y=-0.30, obj_z=1.0)).ok and not GL.check_goal(g, _carry_state(obj_x=0.231, obj_y=-0.30, obj_z=1.0)).ok
    assert not GL.check_goal(g, _carry_state(obj_x=0.18, obj_y=-0.30, obj_z=1.0 + 0.025)).ok
    assert GL.check_goal(g, _carry_state(obj_x=0.18, obj_y=-0.30, obj_z=1.0, obj_tilt_deg=44.9)).ok
    assert not GL.check_goal(g, _carry_state(obj_x=0.18, obj_y=-0.30, obj_z=1.0, obj_tilt_deg=45.0)).ok
    assert "cup_tilt_deg" not in dict(TK.task("box").goal_terms) and "cup_tilt_deg" in dict(TK.task("tray4").goal_terms)
    assert not GL.check_goal(g, _carry_state(obj_x=0.25, obj_y=-0.30, obj_z=0.85 + 0.15)).ok       # 7 cm off in x
    assert not GL.check_goal(g, _carry_state(obj_x=0.18, obj_y=-0.30, obj_z=0.85)).ok               # still on the counter, 15 cm low
    assert "z_err" in GL.check_goal(g, _carry_state(obj_x=0.18, obj_y=-0.30, obj_z=0.85)).why


def test_goal_pan_lift_and_coffee():
    g = GL.goal_for("pan_lift")
    base = dict(rise_mm=41.0, support_force_N=0.0, obj_tilt_deg=2.0, spill_flag=0.0, lid_mm=1.0)
    assert GL.check_goal(g, StepState(0, base, base)).ok
    for k, v in (("rise_mm", 39.0), ("support_force_N", 0.2), ("obj_tilt_deg", 7.0), ("spill_flag", 1.0), ("lid_mm", 6.0)):
        assert not GL.check_goal(g, StepState(0, dict(base, **{k: v}), {})).ok, k
    g = GL.goal_for("coffee_tray")
    now = dict(obj_x=0.10, obj_y=0.01, cap_rel_x=0.05, cap_rel_y=-0.1, cap_rel_z=0.01, obj_tilt_deg=2.0)
    assert GL.check_goal(g, StepState(0, now, {})).ok
    assert not GL.check_goal(g, StepState(0, dict(now, obj_x=0.2), {})).ok
    assert not GL.check_goal(g, StepState(0, dict(now, cap_rel_y=0.2), {})).ok
    gm = GL.goal_for("coffee_tray_min")                      # loose: >= 5 cm forward, <= 5 cm sideways
    assert GL.check_goal(gm, StepState(0, dict(now, obj_x=0.06), {})).ok
    assert not GL.check_goal(gm, StepState(0, dict(now, obj_x=0.04), {})).ok
    assert not GL.check_goal(gm, StepState(0, dict(now, obj_x=0.2, obj_y=0.08), {})).ok


def test_monitors_U_and_R():
    t = TK.task("tray4")
    s = _carry_state()
    assert MON.u_row(s, t).ok and MON.recoverable(s, t).ok
    bad = _carry_state()
    bad.worst["obj_tilt_deg"] = 12.0
    first, m = MON.first_violation(bad, t)
    assert first == "obj_tilt_deg" and m["obj_tilt_deg"] == pytest.approx(-2.0) and not MON.u_row(bad, t).ok
    bad = _carry_state()
    bad.worst["spill_ratio"] = 1.2
    bad.worst["obj_tilt_deg"] = 11.0
    assert MON.first_violation(bad, t)[0] == "obj_tilt_deg"          # table order = carry._check order
    fv = _carry_state()
    fv.first_viol = dict(kind="collision")
    assert not MON.u_row(fv, t).ok
    assert not MON.recoverable(_carry_state(obj_speed=0.01), t).ok
    assert not MON.recoverable(_carry_state(support_contact=0.0), t).ok
    assert not MON.recoverable(_carry_state(cup_tilt_deg=6.0), t).ok
    assert MON.recoverable(_carry_state(cup_tilt_deg=4.0), t).ok
    assert "U:" in MON.recoverable(fv, t, include_U=True).why or "U entered" in MON.recoverable(fv, t, include_U=True).why
    # a missing quantity is listed, not passed silently
    r = MON.recoverable(StepState(0, {"obj_speed": 0.0}, {}), t)
    assert "obj_tilt_deg" in r.assumptions["not_evaluated"]
    # box has no cup / lid rows; pot has lid rows
    assert "cup_tilt_deg" not in TK.task("box").R_dict() and "lid_slide_m" in TK.task("pot").R_dict()


def test_old_comparators_at_equality():
    """item 9: the Bounds reproduce the old code at the boundary (carry tau >= 1-1e-6 violates; pan margin <= 0 violates; pan effort allows 1+1e-6)."""
    t = TK.task("box")
    lim = t.U_dict()["tau_ratio"]
    assert not inside(bound_margin(1.0 - 1e-6, lim), lim) and inside(bound_margin(1.0 - 2e-6, lim), lim)
    pan = TK.task("pan_lift")
    ps = lambda **kw: StepState(0, {}, dict(dict(obj_tilt_deg=1.0, lid_mm=1.0, drop_mm=0.0, joint_margin_rad=0.5, tau_ratio=0.5, spill_flag=0.0, com_in_m=0.05,
                                                 collision_flag=0.0), **kw))
    assert MON.u_row(ps(), pan).ok
    assert not MON.u_row(ps(obj_tilt_deg=15.0), pan).ok and MON.u_row(ps(obj_tilt_deg=14.999), pan).ok
    assert not MON.u_row(ps(joint_margin_rad=0.005), pan).ok and not MON.u_row(ps(com_in_m=0.0), pan).ok
    assert MON.u_row(ps(tau_ratio=1.0 + 1e-6), pan).ok and not MON.u_row(ps(tau_ratio=1.0 + 2e-6), pan).ok
    rnow = dict(obj_tilt_deg=1.0, dz_mm=0.0, lid_mm=1.0, support_frac=1.0, pan_speed=0.0, arm_speed=0.0, spill_flag=0.0, released=1.0, yaw_deg=0.0, rim_in_m=0.02,
                reach=1.0, stable_margin_m=0.05)
    assert MON.recoverable(StepState(0, rnow, {}), pan).ok and MON.recoverable(StepState(0, rnow, {}), pan, version="v2").ok
    assert not MON.recoverable(StepState(0, dict(rnow, obj_tilt_deg=3.0), {}), pan).ok            # R needs margin > 0
    assert not MON.recoverable(StepState(0, dict(rnow, rim_in_m=0.0), {}), pan).ok and MON.recoverable(StepState(0, dict(rnow, rim_in_m=0.0), {}), pan, version="v2").ok
    assert not MON.recoverable(StepState(0, dict(rnow, yaw_deg=9.0), {}), pan).ok and MON.recoverable(StepState(0, dict(rnow, yaw_deg=9.0), {}), pan, version="v2").ok
    assert not MON.recoverable(StepState(0, dict(rnow, reach=0.0), {}), pan).ok and not MON.recoverable(StepState(0, dict(rnow, reach=0.0), {}), pan, version="v2").ok
    assert not MON.recoverable(StepState(0, dict(rnow, stable_margin_m=0.005), {}), pan, version="v2").ok          # (margin - 0.005) / 0.01 > 0
    assert MON.recoverable(StepState(0, dict(rnow, stable_margin_m=0.006), {}), pan, version="v2").ok
    assert "stable_margin_m" not in dict(pan.R) and "rim_in_m" not in dict(pan.R_v2) and "abs(yaw_deg)" not in dict(pan.R_v2)


def test_absent_quantity_is_missing_not_passed():
    """item 3 / 4: an unmeasured U or R quantity gives a missing row (success unknown), never a pass; empty tables are missing too."""
    t = TK.task("tray4")
    s = _carry_state()
    del s.worst["spill_ratio"]
    r = MON.u_row(s, t)
    assert not r.ok and r.guarantee == "missing" and "spill_ratio" in r.why and MON.unsafe(s, t) is None
    del s.now["obj_speed"]
    r = MON.recoverable(s, t)
    assert not r.ok and r.guarantee == "missing" and "obj_speed" in r.why
    o = OC.classify(True, _row(True), MON.unsafe(s, t), False, r)
    assert o.success is True and o.unsafe is None and o.safe_success is None and o.in_R.guarantee == "missing"     # R1: success = finished AND goal; unknown U -> safe_success unknown
    both = _carry_state(obj_tilt_deg=30.0)                                   # a violation elsewhere is still a plain failure
    del both.worst["spill_ratio"]
    assert MON.u_row(both, t).guarantee == "runtime" and MON.unsafe(both, t) is True
    for name in ("coffee_tray", "coffee_tray_min"):
        c = TK.task(name)
        assert MON.u_row(StepState(0, {}, {}), c).guarantee == "missing" and MON.recoverable(StepState(0, {}, {}), c).guarantee == "missing"
        assert MON.unsafe(StepState(0, {}, {}), c) is None
    for t_ in TK.TASKS.values():
        if t_.backend != "embodichain":
            assert t_.U and t_.R, t_.name


def test_recoverable_include_U_default_and_time_order():
    t = TK.task("tray4")
    fv = _carry_state()
    fv.first_viol = dict(kind="spill")
    assert "U entered: spill" in MON.recoverable(fv, t).why              # carry.in_R appends "U:<kind>": default include_U for carry tasks
    assert MON.recoverable(fv, t, include_U=False).ok
    pan = TK.task("pan_lift")
    assert pan.r_includes_U is False                                      # pan Episode.in_R does not
    bad = _carry_state(obj_tilt_deg=30.0)
    bad.worst["spill_ratio"] = 2.0
    bad.first_viol = dict(kind="spill")                                   # time order: spill came first although obj_tilt is earlier in the table
    r = MON.u_row(bad, t)
    assert "spill" in r.why and r.worst["_table_first"] == "obj_tilt_deg"
    nofv = _carry_state(obj_tilt_deg=30.0)
    assert "obj_tilt_deg" in MON.u_row(nofv, t).why                       # fallback: table order


def test_plan_shape_failed_row_has_no_positive_margin():
    sup = dict(counter=dict(top=0.82, x=(0.13, 0.53), y=(-0.45, -0.15)), shelf=dict(top=0.97, x=(0.06, 0.27), y=(-0.44, -0.16)))
    steps = [("lift", dict(dz=0.18)), ("move", dict(dx=-0.2, dy=0.0, dz=0.0)), ("place", dict(support="counter"))]       # ends on the wrong support, goal reached
    r, _ = PSH.check(steps, sup, (0.38, -0.30), (0.18, -0.30), "shelf", 0.03)
    assert not r.ok and r.margin <= 0.0


def test_verdict_guarantee_literal_has_missing():
    from typing import get_args
    from world_model.v2.stack.types import Guarantee
    assert "missing" in get_args(Guarantee)


def test_plan_shape():
    sup = dict(counter=dict(top=0.82, x=(0.13, 0.53), y=(-0.45, -0.15)), shelf=dict(top=0.97, x=(0.06, 0.27), y=(-0.44, -0.16)))
    steps = [("lift", dict(dz=0.18)), ("move", dict(dx=-0.2, dy=0.0, dz=0.0)), ("place", dict(support="shelf"))]
    r, deltas = PSH.check(steps, sup, (0.38, -0.30), (0.18, -0.30), "shelf", 0.03)
    assert r.ok, r.why
    assert deltas[2][2] == pytest.approx(-0.03)
    r, _ = PSH.check([("move", dict(dx=-0.2, dy=0, dz=0))] + steps[2:], sup, (0.38, -0.30), (0.18, -0.30), "shelf", 0.03)
    assert not r.ok and "dragging" in r.why
    r, _ = PSH.check(steps, sup, (0.38, -0.30), (0.5, -0.30), "shelf", 0.03)
    assert not r.ok and "tolerance" in r.why


def _row(ok):
    return Row("x", ok, 0.1 if ok else -0.1, guarantee="runtime")


@pytest.mark.parametrize("finished,goal,unsafe,pre,R,refused,exp", [
    (True, True, False, False, True, False, True),
    (False, True, False, False, True, False, False),     # not finished
    (True, False, False, False, True, False, False),     # goal missed
    (True, True, True, False, True, False, True),        # U entered: still a success (R1), but not a safe_success
    (True, True, False, True, True, False, True),        # U before the hold: same
    (True, True, False, False, False, False, True),      # not at rest in R: success is the goal rule only (R1); R is reported apart
    (True, True, False, False, True, True, False),       # refused
])
def test_outcome_truth_table(finished, goal, unsafe, pre, R, refused, exp):
    o = OC.classify(finished, _row(goal), unsafe, pre, _row(R), refused=refused)
    assert o.success is exp
    assert o.safe_success is bool(exp and not unsafe and not pre)       # safe_success = success AND not unsafe (incl. pre-hold)
    assert o.unsafe is bool(unsafe or pre)


def test_outcome_unknown_ingredient():
    assert OC.classify(True, missing("goal", "x"), False, False, _row(True)).success is None
    assert OC.classify(True, missing("goal", "x"), True, False, _row(True)).success is None and OC.classify(True, missing("goal", "x"), True, False, _row(True)).safe_success is False
    assert OC.classify(True, missing("goal", "x"), False, False, _row(True)).safe_success is None


def test_unknown_is_never_true_or_false():
    """item 11: finished None / unsafe None / pre-hold None give success None; any known failure still gives False."""
    ok = _row(True)
    assert OC.classify(None, ok, False, False, ok).success is None
    assert OC.classify(True, ok, None, False, ok).success is True and OC.classify(True, ok, None, False, ok).safe_success is None and OC.classify(True, ok, None, False, ok).unsafe is None
    assert OC.classify(True, ok, False, None, ok).safe_success is None and OC.classify(True, ok, True, None, ok).safe_success is False
    assert OC.classify(None, ok, True, False, ok).success is None and OC.classify(False, ok, None, None, ok).success is False
    assert OC.classify(None, ok, False, False, ok).finished is None
    theirs = dict(task="box", outcome="completed", goal=dict(xy_err=0.0, z_err=0.0), end_in_R=True, R_fail=[], pre_hold_viol=None)       # no `track`
    o = OC.load_legacy(theirs)
    assert o.unsafe is None and o.success is True and o.safe_success is None
    pan = dict(segments=[dict(seg="lift", nominal=dict(viol=None, success=True))])                                          # no status
    o = OC.load_legacy(pan)
    assert o.finished is None and o.success is None


def test_pre_hold_from_ours_record():
    """item 2: probe records carry carry.pre_hold_viol; a U entry before the hold fails the world."""
    ours = dict(outcome="carried", carry=dict(end_in_R=[True, []], segments=[dict(nominal=dict(viol=None))], pre_hold_viol=dict(kind="tau_sat")))
    o = OC.load_legacy(ours)
    assert o.success is None and o.safe_success is False and o.pre_hold_unsafe is True
    ours["carry"]["pre_hold_viol"] = None
    o = OC.load_legacy(ours)
    assert o.pre_hold_unsafe is False and o.success is None            # goal still missing in an ours record
    del ours["carry"]["pre_hold_viol"]
    assert OC.load_legacy(ours).pre_hold_unsafe is None                # not recorded = unknown, not False


def test_load_legacy():
    theirs = dict(task="box", outcome="completed", goal=dict(xy_err=0.02, z_err=0.01), end_in_R=True, R_fail=[], pre_hold_viol=None, track=dict(viol=None),
                  ac_success=False)
    o = OC.load_legacy(theirs)
    assert o.success is True and o.safe_success is True      # the record's own ac_success is NOT used (re-evaluated on the task's goal terms)
    theirs["goal"]["xy_err"] = 0.04
    assert OC.load_legacy(theirs).success is True            # 4 cm: inside the AgentChord 5 cm bound (the old 3 cm rule failed it)
    theirs["goal"]["xy_err"] = 0.06
    assert OC.load_legacy(theirs).success is False
    theirs["goal"].update(xy_err=0.0, tilt_deg=50.0)
    assert OC.load_legacy(theirs).success is False
    theirs.update(goal=dict(xy_err=0.0, z_err=0.0), pre_hold_viol=dict(kind="x"))
    assert OC.load_legacy(theirs).pre_hold_unsafe
    ours = dict(outcome="carried", carry=dict(end_in_R=[True, []], segments=[dict(nominal=dict(viol=None))]))
    o = OC.load_legacy(ours)
    assert o.success is None and o.goal.guarantee == "missing" and not o.unsafe
    ours["carry"]["segments"][0]["nominal"]["viol"] = dict(kind="spill")
    assert OC.load_legacy(ours).unsafe is True and OC.load_legacy(ours).safe_success is False
    assert OC.tally([dict(result=dict(success=True, unsafe=False, safe_success=True, in_R=True, refused=False)), ours])["unsafe"] == 1
    assert OC.load_legacy(dict(outcome="refused (placement)", carry={})).refused
    pan = dict(segments=[dict(seg="lift", nominal=dict(viol=None, status="done", success=True))])
    assert OC.load_legacy(pan).success is True and OC.load_legacy(pan).safe_success is None      # pre-hold not recorded in pan records: safe_success unknown


def test_registry_checkers_resolve():
    for n in REG.names("checkers"):
        assert callable(REG.get("checkers", n))
    assert TL.tilt_row().guarantee == "missing"
