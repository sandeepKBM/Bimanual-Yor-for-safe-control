"""teleop/fastpath.py -- opt-in speed paths for the teleop sim loop (SIMULATION ONLY; pure MuJoCo, no robot code).

  BIMANUAL_TELEOP_FAST=1    ArmCtl.tau: (1) numba fast_barriers.all_pair_barriers instead of sim_common's pure-Python version (bit-identical),
                            (2) the minimal dynamics refresh below instead of a full mj_forward (bit-identical on every field FL-MPC reads).
  BIMANUAL_TELEOP_SPLIT=1   MuJoCo-recommended control split  mj_step1 -> controller reads the fresh state -> mj_step2  (no refresh at all).
                            Needs the 2-line hook in TeleopSim.tick (see begin()/finish() below).  Invalid with RK4 (asserted).
Default (both unset/0) = the old behaviour, bit-identical.

What FL-MPC (mpc_fl.FLMPC / rt_flmpc.RTFLMPC) + all_pair_barriers + the tilt cone READ from the MjData (verified by grep):
  qpos, qvel, qM (via mj_fullM), qfrc_bias, qfrc_passive, xpos, xmat, cdof, cdof_dot, cvel, subtree_com (the last four through the numba
  Jacobians of fast_barriers; sim_common.point_jac_at uses mj_jac/mj_jacDot = the same fields).  No contact, constraint or actuator field.
"""
from __future__ import annotations

import os

import numpy as np
import mujoco

ENV_FAST = "BIMANUAL_TELEOP_FAST"
ENV_SPLIT = "BIMANUAL_TELEOP_SPLIT"
_DERIVED = ("qM", "qfrc_bias", "qfrc_passive", "xpos", "xmat", "cdof", "cdof_dot", "cvel", "subtree_com")


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() not in ("", "0", "false", "no", "off")


def fast_enabled() -> bool:
    return _flag(ENV_FAST)


def split_enabled(m=None) -> bool:
    if not _flag(ENV_SPLIT):
        return False
    if m is not None and int(m.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_RK4):
        raise RuntimeError("BIMANUAL_TELEOP_SPLIT=1 is invalid with the RK4 integrator (mj_step1/mj_step2)")
    return True


def refresh_dynamics(m, d):
    """== the part of mj_forward whose results FL-MPC reads, without collision / constraint / actuation (0.11 ms vs ~5 ms here).
    Bit-identical to mj_forward on qM, qfrc_bias, qfrc_passive, xpos, xmat, cdof, cdof_dot, cvel, subtree_com (tested).
    Uses mj_makeM (NOT mj_crb: makeM adds the implicit-integrator armature terms; crb alone differs by ~1e-5)."""
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    mujoco.mj_makeM(m, d)
    mujoco.mj_comVel(m, d)
    mujoco.mj_passive(m, d)
    mujoco.mj_rne(m, d, 0, d.qfrc_bias)


def fast_all_pair_barriers():
    """fast_barriers.all_pair_barriers (numba, 0.2 ms, bit-identical, returns a lazy PairList; RTFLMPC reads its arrays
    directly, FLMPC-py materialises the dicts on first iteration)."""
    from ...qp_safety.impl import _paths
    _paths()
    import fast_barriers
    return fast_barriers.all_pair_barriers


# ------------------------------------------------------------------------------------------------------------------ control split
def begin(ctl, m, d, dn):
    """Replaces `mujoco.mj_copyData(self.dn, m, d)` in TeleopSim.tick's substep loop.
    split OFF, FAST OFF: exactly mj_copyData (old behaviour).
    split OFF, FAST ON : copy only what the refresh/kinematics reads (qpos, qvel, mocap pose, time) instead of the whole 1449-geom MjData
               (mj_copyData is ~0.7 ms/step); bit-identical torques (tested).
    split ON : mj_step1(m, d) -> d now holds the fresh kinematics / M / bias / passive for the current state; dn gets qpos, qvel
               and those fields (the caller then adds encoder noise to dn.qpos / dn.qvel as before; ctl.tau skips its refresh).
               Deviation vs the old path: M, bias, Jacobians, xmat are evaluated at the CLEAN state while q0/qd0 carry the encoder noise
               (old: everything at the noisy state).  Noise is 1e-4 rad / 2e-3 rad/s, so this is a 1e-4 relative change in the model terms."""
    if split_enabled(m) and getattr(ctl, "supports_split", False):
        mujoco.mj_step1(m, d)
        dn.qpos[:] = d.qpos
        dn.qvel[:] = d.qvel
        for f in _DERIVED:
            getattr(dn, f)[:] = getattr(d, f)
        ctl.skip_refresh = True
        ctl.split_pending = True
    elif getattr(ctl, "fast", False) and getattr(ctl, "supports_split", False):
        dn.qpos[:] = d.qpos
        dn.qvel[:] = d.qvel
        dn.mocap_pos[:] = d.mocap_pos
        dn.mocap_quat[:] = d.mocap_quat
        dn.time = d.time
        ctl.skip_refresh = False
        ctl.split_pending = False
    else:
        mujoco.mj_copyData(dn, m, d)
        ctl.skip_refresh = False
        ctl.split_pending = False


def finish(ctl, m, d):
    """Replaces `mujoco.mj_step(m, d)` after qfrc_applied is set."""
    if getattr(ctl, "split_pending", False):
        ctl.split_pending = False
        mujoco.mj_step2(m, d)
    else:
        mujoco.mj_step(m, d)


# ------------------------------------------------------------------------------------------------------------------ hook without editing sim.py
_HOOKS = (("mujoco.mj_copyData(self.dn, m, d)", "FP.begin(ctl, m, d, self.dn)"),
          ("mujoco.mj_step(m, d)", "FP.finish(ctl, m, d)"))


def install_split_hook(cls=None) -> bool:
    """Standalone way to get the 2-line sim.py hook (for harnesses / the web server) BEFORE sim.py carries it: rewrites the source of
    TeleopSim.tick (exactly the two substitutions in _HOOKS) and rebinds it.  No-op (returns False) if tick already uses FP.begin.
    Raises if the expected source lines are not found (sim.py changed -> apply the hook by hand)."""
    import inspect
    import textwrap
    import sys
    if cls is None:
        from .sim import TeleopSim as cls
    if getattr(cls.tick, "_fp_hooked", False):
        return False
    src = inspect.getsource(cls.tick)
    if "FP.begin(" in src:                 # sim.py already carries the hook
        return False
    src = textwrap.dedent(src)
    for old, new in _HOOKS:
        if src.count(old) != 1:
            raise RuntimeError(f"fastpath.install_split_hook: expected exactly one {old!r} in TeleopSim.tick; apply the hook by hand")
        src = src.replace(old, new)
    mod = sys.modules[cls.__module__]
    mod.FP = sys.modules[__name__]
    ns = mod.__dict__
    exec(compile(src, f"<fastpath-patched {cls.__module__}.tick>", "exec"), ns)
    fn = ns.pop("tick")                    # exec defined it in the module namespace; keep that clean
    fn._fp_hooked = True
    cls.tick = fn
    return True
