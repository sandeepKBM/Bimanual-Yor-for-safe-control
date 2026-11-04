"""11-03: opt-in FL-MPC joint position/speed rows (q_lim/qd_lim), rules.py table rows, ctl='coopimp_safe'.  Targeted, short."""
import numpy as np
import mujoco

from world_model.v2.nero import adapters as A, config as C, ik as IK
from world_model.v2.nero import hold_test as HT
from world_model.v2.nero.contents import carry as CR, carry_cert as CC, rules as RU, tilt_sweep as TS


def _setup():
    m = A.compose_scene(lift=False, gravcomp=False, tau=C.tau_limits(None))
    d = mujoco.MjData(m)
    info = IK.set_hold_state(m, d)
    A.disable_arm_servos(m)
    ctl = HT.Ctl(m, info["hold_q"], C.tau_limit(None))
    return m, d, ctl


def _step(ctl, m, d, **kw):
    mujoco.mj_forward(m, d)
    te, ti, ok = ctl.fl.step(m, d, ctl.lq, ctl.ld, ctl.rq, ctl.rd, ctl.qe_ref, ctl.z, ctl.qi_ref, ctl.z, ctl.pairs(d), **kw)
    return te, ti, ok


def test_rules_none_identical():
    m, d, c1 = _setup()
    d.qvel[c1.ld[0]] = 0.3
    a = _step(c1, m, d)
    _, _, c2 = _setup()
    b = _step(c2, m, d, q_lim=None, qd_lim=None, lim_steps=None)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]
    assert c2.fl.n_limit_steps == 0


def test_velocity_limit_holds_or_slack_counted():
    m, d, c0 = _setup()
    vmax = RU.speed_limits()[0, 0]
    j = c0.ld[0]
    d.qvel[j] = 0.97 * vmax
    c0.qe_ref[:, 0] += np.linspace(0.0, 3.0, c0.qe_ref.shape[0])       # reference runs away in +j1: tracking wants to speed up
    _step(c0, m, d)
    a_free = c0.fl.a_ego_prev.copy()
    nv = c0.fl.NV
    v_free = d.qvel[j] + c0.fl.Phi_qd_all[1][0, :nv] @ a_free
    _, _, c1 = _setup()
    c1.qe_ref[:, 0] += np.linspace(0.0, 3.0, c1.qe_ref.shape[0])
    _step(c1, m, d, qd_lim=RU.speed_limits())
    a = c1.fl.a_ego_prev
    v = d.qvel[j] + c1.fl.Phi_qd_all[1][0, :nv] @ a
    assert v_free > vmax                                                 # the unconstrained plan would break the limit (test is meaningful)
    assert v <= vmax + 1e-6 or c1.fl.n_limit_slack > 0
    assert c1.fl.n_limit_steps == 1


def test_coopimp_safe_torque_box_and_no_postclip():
    name, w = TS.corners("box", (3.5, 4.0))[0]
    sp = CC.cell_spec("box", (3.5, 4.0))
    m_hat = 0.5 * (sp.contents.m_lo + sp.contents.m_hi)
    ep = CR.CarryEpisode(w, a_plan=0.5, m_hat=m_hat, fric_comp=True, weld_solref=CC.WELD_STIFF, ctl="coopimp_safe")
    assert ep.rules_mode == "full" and ep.ctl.rules is not None
    ep.run_phases([("hold", 0.1)])
    tr = ep.track
    assert tr["tau_ratio"] <= 1.0 + 1e-9
    assert tr.get("ci_postclip_steps", 0) == 0
    assert tr["fl_steps"] > 0


# ---- 11-03 follow-up: effective limit bounds, footprints, sound pruning, release phase ----
def test_past_limit_state_stays_feasible_with_effective_bounds():
    m, d, c = _setup()
    rl = RU.Rules(m)
    lo, hi = RU.joint_limits(m)
    j = 0
    d.qpos[c.lq[j]] = hi[0, j] + 0.008                                    # left j1 8 mrad past the MJCF range (MuJoCo soft limit allows it)
    c.qe_ref[:, j] = hi[0, j]                                            # the reference sits AT the limit
    d.qvel[c.ld[j]] = 0.0
    lo_e, hi_e, vm_e = rl.limit_bounds(d)
    assert hi_e[0, j] == d.qpos[c.lq[j]] and lo_e[0, j] == lo[0, j]
    assert rl.stats["n_q_past"] == 1 and rl.stats["q_past_joint"][0, j] == 1
    _step(c, m, d, q_lim=(lo_e, hi_e), qd_lim=vm_e)
    assert c.fl.last_feasible and c.fl.n_limit_fallback == 0
    _, _, c2 = _setup()
    c2.qe_ref[:, j] = hi[0, j]
    _step(c2, m, d, q_lim=(lo, hi), qd_lim=RU.speed_limits())            # raw MJCF bound: row unmet at step 1 within the torque box
    assert (not c2.fl.last_feasible) or c2.fl.n_limit_slack > 0


def test_speed_bound_effective_when_over_limit():
    m, d, c = _setup()
    rl = RU.Rules(m)
    v = RU.speed_limits()[0, 3]
    d.qvel[c.ld[3]] = 1.1 * v
    _, _, vm = rl.limit_bounds(d)
    assert vm[0, 3] == 1.1 * v


def _box_ep(ctl="lever"):
    name, w = TS.corners("box", (3.5, 4.0))[0]
    sp = CC.cell_spec("box", (3.5, 4.0))
    m_hat = 0.5 * (sp.contents.m_lo + sp.contents.m_hi)
    return CR.CarryEpisode(w, a_plan=0.5, m_hat=m_hat, fric_comp=True, weld_solref=CC.WELD_STIFF, ctl=ctl, rules="full")


def test_footprint_scope_and_pruned_rows_match_unpruned():
    import copy
    ep = _box_ep()
    ep.run_phases([("hold", 0.05)])
    m, d, ctl = ep.m, ep.d, ep.ctl
    names = [s["name"] for s in ctl.rules.surf]
    assert "pan_table_top" in names and "counter" in names
    mujoco.mj_forward(m, d)
    t = ctl.terms(d)
    Me, ce = RU.arm_dyn(m, d, ctl.ld, *t["left"])
    Mi, ci = RU.arm_dyn(m, d, ctl.rd, *t["right"])
    sound, none = RU.Rules(m, prune="sound"), RU.Rules(m, prune="none")
    for k in range(3):                                                    # a few states along a short motion
        mujoco.mj_forward(m, d)
        t = ctl.terms(d)
        Me, ce = RU.arm_dyn(m, d, ctl.ld, *t["left"])
        Mi, ci = RU.arm_dyn(m, d, ctl.rd, *t["right"])
        sols = []
        for rl in (sound, none):
            c2 = copy.deepcopy(ctl)
            c2.fl.a_box = 0.5                                             # small acceleration box (hard in the QP) so that the sound bound prunes rows
            kw = rl.fl_kwargs(d, Me, ce, Mi, ci, fl=c2.fl)
            te, ti, ok = c2.fl.step(m, d, c2.lq, c2.ld, c2.rq, c2.rd, c2.qe_ref, c2.z, c2.qi_ref, c2.z, c2.pairs(d),
                                    M_ego_extra=t["left"][0], bias_ego_extra=t["left"][1], M_int_extra=t["right"][0], bias_int_extra=t["right"][1], **kw)
            sols.append((te, ti, ok, kw))
        assert sols[0][2] == sols[1][2]
        # a pruned row would not have been violated: the unpruned solution satisfies every limit row that the pruned QP dropped
        lo_s, hi_s, vm_s = sound.limit_bounds(d)
        assert np.allclose(sols[0][0], sols[1][0], atol=1e-9) and np.allclose(sols[0][1], sols[1][1], atol=1e-9)
        d.qvel[ctl.ld[1]] += 0.05
    assert sound.stats["tab_cand"] + sound.stats["tab_out_scope"] > 0
    assert none.stats["tab_pruned"] == 0 and none.stats["q_pruned"] == 0 and none.stats["qd_pruned"] == 0
    assert sound.stats["tab_pruned"] > 0 and sound.stats["q_pruned"] > 0 and sound.stats["qd_pruned"] > 0     # pruning actually happened


def test_release_phase_goes_through_rules():
    ep = _box_ep()
    ep.run_phases([("hold", 0.02)])
    n0 = ep.ctl.rules.stats["calls"]
    ep.run_phases([("weld", False), ("hold", 0.02)])
    assert not ep.ctl.on and ep.ctl.rules.stats["calls"] - n0 == 4       # released steps also call fl_kwargs (same rule set)
