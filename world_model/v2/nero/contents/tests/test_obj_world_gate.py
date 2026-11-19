"""K5a gate (owner-written, 11-16): object-world clearance in U (USER 11-16 D9, 5 mm ASSUMED).  SIMULATION ONLY.

  _box_sat         a lower bound on the true box-box distance (never above it), exact for axis-aligned face-face, negative when overlapping
  default record   the obj_world track key and the physics flag are always present (the row is always in U since 11-19)
  nominal carries  box (light cell) and tray4 never raise obj_world on the planned path.  11-18: the heavy box cell (2.5-3.0 kg) was dropped from this
                   list: its nominal carry pitches the box and comes 4.8 mm from the shelf (current_state finding 8, a real near-miss, not a test bug)
  tilt row         11-18: on that heavy cell the tilt-aware row (MEASURED tilt) refutes the closed_chain default plan
  geometry         load 2 mm beside the shelf side -> U; 20 mm beside -> not U; 3 mm above the shelf top (support approach) -> not U
  U table          tasks._u_carry lists obj_world_clear_m as a lower bound 0.005 m
"""
from __future__ import annotations

import numpy as np
import pytest
import mujoco

from world_model.v2.nero.contents import carry as CR
from world_model.v2.nero.contents import carry_cert as CC
from world_model.v2.nero.contents import envelope as E

TWO_BOXES = """<mujoco><worldbody>
<body name="a"><freejoint/><geom name="ga" type="box" size="0.10 0.05 0.02"/></body>
<body name="b" pos="0.5 0 0"><freejoint/><geom name="gb" type="box" size="0.03 0.07 0.04"/></body>
</worldbody></mujoco>"""


def _quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def test_box_sat_is_a_lower_bound_and_exact_face_face():
    m = mujoco.MjModel.from_xml_string(TWO_BOXES)
    d = mujoco.MjData(m)
    ga, gb = m.geom("ga").id, m.geom("gb").id
    rng = np.random.default_rng(5)
    ft = np.zeros(6)
    n_checked = 0
    for _ in range(400):
        d.qpos[:3] = rng.uniform(-0.05, 0.05, 3)
        d.qpos[3:7] = _quat(rng)
        d.qpos[7:10] = rng.uniform(-0.4, 0.4, 3)
        d.qpos[10:14] = _quat(rng)
        mujoco.mj_forward(m, d)
        dist = mujoco.mj_geomDistance(m, d, ga, gb, 1.0, ft)
        sat, n = CR._box_sat(d, m, ga, gb)
        assert abs(np.linalg.norm(n) - 1.0) < 1e-9
        if dist > 0.01:                                   # GJK reliable here
            n_checked += 1
            assert sat <= dist + 1e-9, (sat, dist)
    assert n_checked > 100
    # axis-aligned, separated along x: exact
    d.qpos[:] = 0
    d.qpos[3] = d.qpos[10] = 1.0
    for gap in (0.001, 0.004, 0.02):
        d.qpos[7] = 0.10 + 0.03 + gap
        mujoco.mj_forward(m, d)
        sat, n = CR._box_sat(d, m, ga, gb)
        assert abs(sat - gap) < 1e-12 and abs(n[0] - 1.0) < 1e-12
    d.qpos[7] = 0.10                                      # overlapping by 3 cm along x
    mujoco.mj_forward(m, d)
    assert CR._box_sat(d, m, ga, gb)[0] < 0


def _episode(task, cell, seed=1):
    w = CC.sample_world(task, tuple(cell), np.random.default_rng(seed))
    return CR.CarryEpisode(w, a_plan=E.A_NOM, m_hat=0.0, fric_comp=True, weld_solref=CC.WELD_STIFF)


def test_default_record_has_obj_world_key():
    ep2 = _episode("box", CC.cells("box")[0][0])
    ep2.reset_track()
    assert ep2.u_obj is True and ep2.track["obj_world_clear_min"] == float("inf")
    assert ep2.physics_flags().get("u_obj_world") == pytest.approx(0.005)


@pytest.mark.parametrize("task,ci", [("box", 0), ("tray4", 0)])        # ("box", 4) removed 11-18: finding 8, see test_heavy_box_near_miss_refuted_by_tilt_row
def test_nominal_carry_never_obj_world(task, ci):
    ep = _episode(task, CC.cells(task)[ci][0])
    ep.run_phases([("hold", 0.4)])
    mins = []
    for seg in CR.SEGS:
        ep.reset_track()
        ep.run_phases(ep.segment_phases(seg))
        assert "obj_world" not in ep.track["viol_all"], (task, seg, ep.track["viol_all"]["obj_world"])
        mins.append(ep.track["obj_world_clear_min"])
    assert all(x >= CR.U_OBJ_CLEAR_M for x in mins), mins


def _aabb(m, d, gids):
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in gids:
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
            r = m.geom_rbound[g]
            c = d.geom_xpos[g]
            lo, hi = np.minimum(lo, c - r), np.maximum(hi, c + r)
            continue
        R = d.geom_xmat[g].reshape(3, 3)
        ext = np.abs(R) @ m.geom_size[g]
        lo, hi = np.minimum(lo, d.geom_xpos[g] - ext), np.maximum(hi, d.geom_xpos[g] + ext)
    return lo, hi


def _shift_load(ep, delta):
    m, d = ep.m, ep.d
    for b in ep.carried:
        j = m.body_jntadr[b]
        if j >= 0 and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            a = m.jnt_qposadr[j]
            d.qpos[a:a + 3] += delta
    mujoco.mj_forward(m, d)


def test_geometry_beside_and_above_shelf():
    ep = _episode("box", CC.cells("box")[0][0])
    m, d = ep.m, ep.d
    mujoco.mj_forward(m, d)
    load = sorted({l for _, l in ep._ow_pairs})
    shelf = [m.geom("shelf").id]
    q0 = d.qpos.copy()

    def check_at(target_lo_hi_fn):
        d.qpos[:] = q0
        mujoco.mj_forward(m, d)
        llo, lhi = _aabb(m, d, load)
        slo, shi = _aabb(m, d, shelf)
        _shift_load(ep, target_lo_hi_fn(llo, lhi, slo, shi))
        ep.reset_track()
        ep._obj_world_check(0.0)
        return ep.track["obj_world_clear_min"], ep.track["viol_all"].get("obj_world")

    lc = None
    for gap, expect_u in ((0.002, True), (0.020, False)):
        def beside(llo, lhi, slo, shi, gap=gap):
            lc, sc = 0.5 * (llo + lhi), 0.5 * (slo + shi)
            delta = np.zeros(3)
            delta[2] = sc[2] - lc[2]                                    # vertically level with the shelf slab
            delta[1] = sc[1] - lc[1]
            delta[0] = (slo[0] - gap - lhi[0]) if lc[0] < sc[0] else (shi[0] + gap - llo[0])
            return delta
        clear, viol = check_at(beside)
        assert (viol is not None) == expect_u, (gap, clear, viol)
        assert clear == pytest.approx(gap, abs=1.5e-3), (gap, clear)

    def above(llo, lhi, slo, shi):
        lc, sc = 0.5 * (llo + lhi), 0.5 * (slo + shi)
        return np.array([sc[0] - lc[0], sc[1] - lc[1], shi[2] + 0.003 - llo[2]])
    clear, viol = check_at(above)
    assert viol is None, (clear, viol)
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)


def test_u_table_has_obj_world_row():
    from world_model.v2.stack import tasks as TK
    names = [n for n, _ in TK.task("box").U]
    assert "obj_world_clear_m" in names
    b = dict(TK.task("box").U)["obj_world_clear_m"]
    assert b.lo == pytest.approx(0.005) and b.tag == "ASSUMED"


@pytest.mark.slow
def test_heavy_box_near_miss_refuted_by_tilt_row(tmp_path, monkeypatch):
    """11-18 (finding 8, USER 11-17 "Checker uses measured tilt"): box 2.5-3.0 kg, a = 0.5, closed_chain default plan.  (An untilted row admitted it, owner run 11-18, margin +2.1 mm; that row is removed.)  With the per-plan tilt MEASURED (n = 20 ASSUMED, sim only) the row refutes it on obj_world (owner run 11-18: lift, shelf|box_lid,
    -5.4 mm at t = 0.365 s).  Sampled guarantee (the tilt cover is a max over n)."""
    from world_model.v2.adapters import collision_rows as RCOL
    from world_model.v2.stack.planners import closed_chain as CCP
    from world_model.v2.nero.contents import tilt_measure as TM
    monkeypatch.setenv("BIMANUAL_PLANNER", "closed_chain")
    cell, a = (2.5, 3.0), 0.5
    plan = CCP.plan_for(CCP._nominal_inputs("box"), a)
    tab, info = TM.measure_plan("box", None, cell, a, n=20, procs=8, out_dir=tmp_path / "tilt", plan_id="box_heavy_gate")
    spec = RCOL.tilt_from_table(tab, TM.cell_key("box", cell), a, source=info["table_path"])
    r = RCOL.collision_rows(plan, tilt=spec)
    assert not r.ok and r.guarantee == "refuted" and "obj_world" in (r.why or ""), (r.ok, r.guarantee, r.why)
