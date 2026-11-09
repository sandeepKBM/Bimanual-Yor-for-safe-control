"""flmpc_basic: no safety rows, torque saturates at the actuator limit on an infeasible QP, no fallback.  SIMULATION ONLY."""
import mujoco
import numpy as np
import pytest

from world_model.v2.nero import adapters as A
from world_model.v2.nero import config as C
from world_model.v2.flmpc.basic import FLMPCBasic
from world_model.v2.flmpc import mpc_fl


def _setup(hold, **kw):
    m, d, info = hold
    lq, ld = A.arm_addrs(m, "left")
    rq, rd = A.arm_addrs(m, "right")
    fl = FLMPCBasic(m, NC=4, L=8, W_POS_EGO=3000.0, W_VEL_EGO=15.0, NV=C.N_ARM, **kw)
    NP = fl.NP
    z = np.zeros((NP + 1, C.N_ARM))
    qe = np.tile(np.array(info["hold_q"]["left"]), (NP + 1, 1))
    qi = np.tile(np.array(info["hold_q"]["right"]), (NP + 1, 1))
    return m, d, fl, (lq, ld, rq, rd), qe, qi, z


def test_no_removed_rows_and_no_fallback(hold, monkeypatch):
    m, d, fl, (lq, ld, rq, rd), qe, qi, z = _setup(hold, tau_max_vec=C.tau_limit())
    seen = {}
    orig = mpc_fl._assemble_fl

    import world_model.v2.flmpc.basic as B

    def spy(*a, **k):
        out = orig(*a, **k)
        seen["shape"] = out[2].shape
        seen["npairs"] = a[14].shape[0]
        return out
    monkeypatch.setattr(B, "_assemble_fl", spy)
    monkeypatch.setattr(FLMPCBasic, "_relaxed_fallback", lambda *a, **k: (_ for _ in ()).throw(AssertionError("fallback called")))
    monkeypatch.setattr(FLMPCBasic, "_torque_safe_scale", staticmethod(lambda *a, **k: (_ for _ in ()).throw(AssertionError("scale called"))))
    mujoco.mj_forward(m, d)
    pairs = A.PairBarriers(m)(d)
    te, ti, ok = fl.step(m, d, lq, ld, rq, rd, qe, z, qi, z, pairs)       # pairs are accepted and ignored
    assert seen["npairs"] == 0 and seen["shape"][0] == 4 * C.N_ARM       # only the torque box rows (a_box is the variable box)
    assert ok and fl.last_fallback == 0 and fl.n_sat_steps == 0
    with pytest.raises(ValueError):                                      # a removed feature is refused, not silently dropped
        fl.step(m, d, lq, ld, rq, rd, qe, z, qi, z, pairs, table_rows=(1, 2, 3, 4))
    with pytest.raises(ValueError):
        fl.step(m, d, lq, ld, rq, rd, qe, z, qi, z, pairs, tilt_max_rad=0.1)


def test_torque_saturates_when_infeasible(hold):
    # limit below gravity bias on some joint and a = -M^-1 c unreachable -> hard QP infeasible -> unconstrained tracking solve, torque clipped at the limit
    m, d, fl, (lq, ld, rq, rd), qe, qi, z = _setup(hold, tau_max_vec=np.full(7, 0.5))
    fl.a_box = 0.01                                                      # test only: a tiny solver bound makes a = -M^-1 c unreachable (otherwise the torque box is always satisfiable)
    qe = qe + 0.05                                                       # a tracking error so the unconstrained solve commands something
    mujoco.mj_forward(m, d)
    te, ti, ok = fl.step(m, d, lq, ld, rq, rd, qe, z, qi, z)
    assert not ok and fl.n_sat_steps == 1 and fl.n_tau_clipped == 1 and fl.last_fallback == 0
    assert np.max(np.abs(te)) <= 0.5 + 1e-12 and np.max(np.abs(ti)) <= 0.5 + 1e-12
    assert np.isclose(np.max(np.abs(te)), 0.5)                           # it actually saturated


def _run_jump(hold, qd_max, steps=100, jump=0.6):
    """closed loop: the reference of left joint 1 jumps by `jump` rad; returns (peak |qd| of left joint 1, fl)."""
    m, d, info = hold
    mujoco.mj_resetData(m, d)
    from world_model.v2.nero import ik as IK
    IK.set_hold_state(m, d)
    m_, d_, fl, (lq, ld, rq, rd), qe, qi, z = _setup(hold, tau_max_vec=C.tau_limit(), qd_max=qd_max)
    qe = qe.copy(); qe[:, 0] += jump
    peak = 0.0
    for _ in range(steps):
        mujoco.mj_forward(m, d)
        te, ti, _ok = fl.step(m, d, lq, ld, rq, rd, qe, z, qi, z)
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[ld] = te
        d.qfrc_applied[rd] = ti
        mujoco.mj_step(m, d)
        peak = max(peak, abs(float(d.qvel[ld[0]])))
    return peak, fl


def test_speed_rows_limit_planned_speed(hold):
    v = 0.3                                                              # rad/s, test value (the research callers pass config.MANUAL speeds)
    p_free, fl0 = _run_jump(hold, None)
    p_lim, fl1 = _run_jump(hold, np.full(7, v))
    assert p_free > 2 * v                                                # the reference does demand more than the limit
    assert p_lim <= v + 0.02                                             # limited (small slack: tracking of the planned a, one-step lag)
    assert fl1.n_speed_binding_steps > 0 and fl0.n_speed_binding_steps == 0
    assert fl1.n_a_bound_steps == 0 and fl0.n_a_bound_steps == 0         # the solver bound never binds
    assert fl1.n_tau_clipped == 0 and fl1.last_fallback == 0


def test_speed_row_infeasible_is_counted_and_torque_saturated(hold):
    # initial speed above the limit cannot be met within a 1 Nm torque box over the horizon -> infeasible QP -> unconstrained solve + saturation, counted
    m, d, info = hold
    mujoco.mj_resetData(m, d)
    from world_model.v2.nero import ik as IK
    IK.set_hold_state(m, d)
    m_, d_, fl, (lq, ld, rq, rd), qe, qi, z = _setup(hold, tau_max_vec=np.full(7, 1.0), qd_max=np.full(7, 0.01))
    d.qvel[ld[0]] = 5.0
    mujoco.mj_forward(m, d)
    te, ti, ok = fl.step(m, d, lq, ld, rq, rd, qe, z, qi, z)
    assert (not ok) and fl.n_sat_steps == 1
    assert np.max(np.abs(te)) <= 1.0 + 1e-9 and fl.last_fallback == 0
