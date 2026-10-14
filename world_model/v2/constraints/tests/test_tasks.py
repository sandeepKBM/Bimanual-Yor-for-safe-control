"""pytest: the six tasks compile and agree with hand-coded geometry (small N; the full study is run_all.py)."""
import numpy as np
import pytest

from world_model.v2.tray_rep.sets import ir
from world_model.v2.constraints import common as C, ir_ext, geom_ref
from world_model.v2.constraints.tasks import t1_tray, t2_cup, t3_stack, t4_bibox, t5_door, t6_pour

N = 60


@pytest.mark.parametrize("name,task,tol_pos", [
    ("tray", t1_tray.make(1), 1e-12), ("tray_K6", t1_tray.make(6), 1e-12), ("cup", t2_cup.make(True), 1e-12), ("stack", t3_stack.make(False), 1e-12),
    ("stack_sub", t3_stack.make(True), 1e-12), ("box", t4_bibox.make(), 1e-8), ("door_exact", t5_door.make("AX"), 1e-12),
    ("aperture", t5_door.make("B"), 1e-12), ("pour", t6_pour.make(t6_pour.SPEC), 1e-12)])
def test_margin_matches_hand_coded(name, task, tol_pos):
    rng = np.random.default_rng(0)
    S = task.sample(N, rng)
    m = C.check_margin(task, S)
    assert m["sign_agree"] == 1.0
    assert (m["max_abs_err_pos"] < tol_pos) or np.isnan(m["max_abs_err_pos"])


def test_v0_clear_is_a_bound_not_exact():
    """v0 Clear (sd_origin grid) disagrees with the exact box-box distance; ClearX matches."""
    rng = np.random.default_rng(1)
    S = t5_door.make("A").sample(80, rng)
    v0 = C.check_margin(t5_door.make("A"), S)
    ex = C.check_margin(t5_door.make("AX"), S)
    assert ex["max_abs_err"] < 1e-9 and v0["max_abs_err"] > 1e-3


def test_rows_exact_when_geometry_is_a_polytope():
    rng = np.random.default_rng(2)
    t = t5_door.make("B")
    S = t.sample(40, rng)
    r = C.check_rows(t, S, rng, toy_fn=t5_door.toy_norot)
    assert r["slack_max_err"] < 1e-12 and r["sound_violations"] == 0
    t = t6_pour.make(t6_pour.SPEC_OVER_ONLY)
    r = C.check_rows(t, t.sample(40, rng), rng)
    assert r["grad_med_rel_err"] < 1e-8 and r["sound_violations"] == 0


def test_contact_kept_rows_translation_only():
    rng = np.random.default_rng(3)
    t = t4_bibox.make()
    S = t.sample(50, rng)
    ra = C.check_rows_per_atom(t, S, rng, toy_fn=t4_bibox.toy_norot)
    for k, r in ra.items():
        if "Contact" in k:
            assert r["sound_violations"] == 0 and r["slack_max_err"] < 1e-4


def test_not_convex_markers():
    ents = t4_bibox.build()[1]
    S = t4_bibox.make().sample(2, np.random.default_rng(0))
    with pytest.raises(ir.NotConvex):
        ir.rows(ir.Not(ir.Clear(ents["pad_L"], ents["box"], 0.003)), S[0], t4_bibox.toy(S[0], np.random.default_rng(0)))


def test_geom_ref_box_box():
    I = np.eye(3)
    h = np.array([0.1, 0.1, 0.1])
    assert abs(geom_ref.box_box_sd(I, np.zeros(3), h, I, np.array([0.5, 0, 0]), h) - 0.3) < 1e-12
    assert abs(geom_ref.box_box_sd(I, np.zeros(3), h, I, np.array([0.15, 0, 0]), h) + 0.05) < 1e-12
    assert abs(geom_ref.box_box_sd(I, np.zeros(3), h, I, np.array([0.3, 0.3, 0]), h) - 0.1 * np.sqrt(2)) < 1e-12


def test_dict_form_backward_compatible():
    """the v0 dict forms still build through ir_ext.from_dict (tray spec)."""
    t = t1_tray.make(1)
    spec, _ = t.build()
    assert isinstance(spec, ir.G)
