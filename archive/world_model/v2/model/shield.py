"""world_model/v2/model/shield.py -- 15 Hz WM spill shield API for FL-MPC (ContactGuard pattern).

Two rates, never mixed:
  * 200 Hz FL-MPC QP (experiments/.../mpc_fl.py): unchanged hard constraints; it
    only ever reads a SCALAR limit scale s in (0,1] from the shield (for its cup
    accel / tilt / EE-speed boxes). NO WM gradients, NO WM calls inside the QP.
  * 15 Hz shield tick: (1) observe the new camera frames + proprio, (2) take the
    FL-MPC's current planned joint accelerations (its horizon), convert them to
    the WM action space, (3) roll the plan (plus optional perturbed variants)
    through every ensemble member in IMAGINATION, (4) risk = pessimistic
    (ensemble mean - kappa*std) min-over-horizon Fisac value and/or max hazard,
    (5) map risk -> s for the next 66 ms of QP solves (hysteresis: tighten fast,
    relax slowly). Thresholds come from eval/conformal.py (calm-step conformal or
    the <= 1 FA/min event budget chosen on validation).

The WM must be trained with --action-key matching `plan_to_wm_actions(kind=...)`:
  'action_cup' (v2 data: commanded cup accel over the next WM interval,
               [lin(3) m/s^2, ang(3) rad/s^2], world) -- RECOMMENDED: it is the
               quantity FL-MPC's double-integrator plan determines directly;
  'action'     (absolute next-tick joint refs [left q(6), right q(6), fingers(2)]).
"""
from __future__ import annotations

import numpy as np

from world_model.v2.model.data import quat_mul, quat_conj


# --------------------------------------------------------------------------
# plan conversion (FL-MPC: tau = M a + c  =>  exact double integrator in q)
# --------------------------------------------------------------------------
def integrate_plan(q0, qd0, qdd, dt):
    """q0,qd0 (n,), qdd (K,n) planned accelerations -> q,qd (K+1,n) (exact for piecewise-constant a)."""
    K = len(qdd); q = np.empty((K + 1, len(q0))); qd = np.empty_like(q)
    q[0], qd[0] = q0, qd0
    for k in range(K):
        qd[k + 1] = qd[k] + dt * qdd[k]
        q[k + 1] = q[k] + dt * qd[k] + 0.5 * dt * dt * qdd[k]
    return q, qd


def _ang_vel(quat, dt):
    dq = quat_mul(quat[1:], quat_conj(quat[:-1]))
    dq *= np.sign(dq[:, :1] + 1e-12)
    w = 2.0 * dq[:, 1:] / dt
    return np.concatenate([w[:1], w], 0)


def plan_to_wm_actions(q_plan, dt, wm_hz, kind="action_cup", cup_pose_fn=None, fingers=(0.0, 0.0),
                       arm_slices=(slice(0, 6), slice(6, 12))):
    """q_plan (K+1, 12) integrated joint plan [left 6, right 6] at dt -> WM actions (K_wm, A)
    at wm_hz. cup_pose_fn(q_left (N,6), q_right (N,6)) -> (pos (N,3), quat (N,4)) is the FK
    cup pose (e.g. data.fk_grasp_sites + the fixed grasp offset), needed for 'action_cup'."""
    stride = max(1, int(round(1.0 / (wm_hz * dt))))
    n_wm = (len(q_plan) - 1) // stride
    if kind == "action":
        idx = np.arange(1, n_wm + 1) * stride
        ql, qr = q_plan[idx][:, arm_slices[0]], q_plan[idx][:, arm_slices[1]]
        return np.concatenate([ql, qr, np.tile(np.asarray(fingers, float), (n_wm, 1))], 1).astype(np.float32)
    if kind != "action_cup":
        raise ValueError(kind)
    pos, quat = cup_pose_fn(q_plan[:, arm_slices[0]], q_plan[:, arm_slices[1]])
    v = np.gradient(pos, dt, axis=0); a = np.gradient(v, dt, axis=0)
    w = _ang_vel(np.asarray(quat, float), dt); wd = np.gradient(w, dt, axis=0)
    acc = np.concatenate([a, wd], 1)
    return np.stack([acc[k * stride:(k + 1) * stride].mean(0) for k in range(n_wm)]).astype(np.float32)


# --------------------------------------------------------------------------
# shield
# --------------------------------------------------------------------------
class SpillShield:
    """Wraps a WM predictor exposing reset/observe_step/imagine (DinoWMPredictor,
    RSSMPredictor). risk in [0, 2]: 1 - (value lower confidence bound), so risk > 1
    <=> the pessimistic imagined Fisac value goes negative (spill predicted)."""

    def __init__(self, wm, risk_thr=1.0, kappa=1.0, s_min=0.3, relax=0.1, hazard_col=-1, hazard_thr=None):
        self.wm, self.thr, self.kappa, self.s_min, self.relax = wm, risk_thr, kappa, s_min, relax
        self.hcol, self.hthr = hazard_col, hazard_thr
        self.s = 1.0

    def reset(self):
        self.wm.reset(); self.s = 1.0

    def tick(self, obs_tokens_or_images, low_t, a_prev):
        return self.wm.observe_step(obs_tokens_or_images, low_t, a_prev)

    def assess(self, action_seqs: np.ndarray) -> dict:
        """action_seqs (N,K,A) in the WM action space (row 0 = the nominal FL-MPC plan)."""
        im = self.wm.imagine(action_seqs)
        vstd = im["vmin_std"] if im["vmin_std"] is not None else 0.0
        v_lcb = im["vmin_mean"] - self.kappa * vstd
        risk = 1.0 - v_lcb
        hz = im["hazard"][..., self.hcol]                      # (M,N,K)
        h_ucb = hz.mean(0).max(-1) + self.kappa * (hz.std(0).max(-1) if hz.shape[0] > 1 else 0.0)
        unsafe = risk > self.thr
        if self.hthr is not None:
            unsafe = unsafe | (h_ucb > self.hthr)
        return dict(risk=risk, v_lcb=v_lcb, hazard_ucb=h_ucb, unsafe=unsafe, raw=im)

    def limit_scale(self, risk_nominal: float) -> float:
        """Scalar for the QP's cup-accel / tilt / EE-speed boxes over the next tick:
        tighten immediately in proportion to the excess risk, relax by `relax`/tick."""
        excess = max(0.0, float(risk_nominal) - self.thr)
        target = float(np.clip(1.0 - excess, self.s_min, 1.0))
        self.s = target if target < self.s else min(target, self.s + self.relax)
        return self.s
