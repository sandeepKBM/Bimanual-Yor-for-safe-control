"""adapters/base_envelope vs the compiled swerve scene (SIMULATION ONLY, CPU)."""
import mujoco
import numpy as np
import pytest

from world_model.v2.adapters import base_envelope as BE
from world_model.v2.adapters import base_tip_model as BT
from world_model.v2.nero.contents import objects as OB
from world_model.v2.nero.contents import scenes as SC
from world_model.v2.stack.checks import base_motion as BM

TASK = "tray6_behind"


def _pm(env, a_b, al, w2):
    """independent Newton-Euler of the arm bodies: sum m J_com^T a (point masses), rigid motion about the rotation centre."""
    m, d = env.m, env.d
    out = {s: np.zeros(7) for s in env.arms}
    for b in range(1, m.nbody):
        if m.body_mass[b] <= 0:
            continue
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacBodyCom(m, d, jp, jr, b)
        a = env._accel_at(d.xipos[b], a_b, al, w2)
        for s in env.arms:
            out[s] += m.body_mass[b] * jp[:, env.da[s]].T @ a
    return out


def test_arm_id_matches_point_mass_newton_euler_and_is_affine():
    env = BE.envelope(TASK)
    z = env.arm_id((0, 0), 0, 0)
    for case in (((1.0, 0.0), 0, 0), ((0.5, -0.7), 0.3, 0.4), ((0, 0), 1.57, 2.46)):     # incl. centrifugal (needs mj_comVel)
        r, p = env.arm_id(*case), _pm(env, *case)
        for s in env.arms:
            assert np.allclose(r[s] - z[s], p[s], atol=0.02 + 0.03 * np.abs(p[s]))
    ex = [env.arm_id(c[0], c[1], c[2]) for c in (((1, 0), 0, 0), ((0, 1), 0, 0), ((0, 0), 1, 0), ((0, 0), 0, 1))]
    r = env.arm_id((0.3, -0.2), 0.5, 0.7)
    for s in env.arms:
        lin = z[s] + 0.3 * (ex[0][s] - z[s]) - 0.2 * (ex[1][s] - z[s]) + 0.5 * (ex[2][s] - z[s]) + 0.7 * (ex[3][s] - z[s])
        assert np.allclose(r[s], lin, atol=1e-9)


def test_static_load_split_balances_object_weight():
    env = BE.envelope(TASK)
    v = (1.8, np.array([0.0, 0.0, 0.03]))
    wr = env.claw_wrenches(v, (0, 0), 0.0, 0.0)
    assert sum(F[2] for F, _ in wr.values()) == pytest.approx((env.m_e + 1.8) * OB.G, rel=1e-9)
    assert sum(F[:2].sum() for F, _ in wr.values()) == pytest.approx(0.0, abs=1e-9)


def test_constants_match_sources():
    assert BE.MU_CUP == SC.TrayWorld().cup_mu
    from world_model.v2.nero import config as C
    env = BE.envelope(TASK)
    assert np.allclose(env.tau_lim, C.tau_limit()) and env.F_cap == pytest.approx(25.0155, abs=1e-3)
    g = BE.mug_geom(TASK)
    assert BM.ratio_limits(g)["spill"] * BM.G == pytest.approx(OB.cup_spill_limit(OB.CUP["D"], OB.CUP["H"], OB.CUP["H"] - OB.CUP["freeboard_min"]))
    assert g.zc_max == pytest.approx(OB.CUP["H"] / 2) and BM.ratio_limits(g)["slide"] < BM.ratio_limits(g)["tip"]


def test_rows_on_g0b_light_tray_and_heavy_box():
    """L1 measured: tray light admitted (model u_tau 0.098 vs G0b sim tau_ratio 0.101, static Fz 0.42 vs sim 0.55); heavy box refused on the static claw load (1.21; sim 1.18)."""
    from world_model.v2.nero.base import smoke as SM
    w = SM.task_world("tray6", None)
    c = BT.true_contents(TASK, w)
    r = BE.torque_row(TASK, c, "L1", "measured", world=w)
    assert r.ok and r.guarantee == "sampled" and r.worst["u_tau"] == pytest.approx(0.101, abs=0.02) and r.worst["u_Fz"] == pytest.approx(0.42, abs=0.05)
    o = BE.open_row(TASK, "L1", "measured")
    assert o.ok and o.guarantee == "sampled" and o.worst["term"] == "spill"
    assert not BE.open_row(TASK, "L3", "measured").ok               # card alpha at L3 (achieved peak, unfinished route) exceeds the spill limit
    wb = SM.task_world("box", SM.heaviest_seed(task="box")[1])
    rb = BE.torque_row("box_behind", BT.true_contents("box_behind", wb), "L1", "measured", world=wb)
    assert not rb.ok and rb.worst["binding"] == "claw_Fz" and 1.0 - rb.margin == pytest.approx(1.18, abs=0.06)
