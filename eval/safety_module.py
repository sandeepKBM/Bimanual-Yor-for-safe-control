"""Pluggable safety-stack interfaces for the eval harness (eval/run_eval.py).

Two roles, matching docs/design/state_spaces.md's layer split:

  - GeometricSafety  (S1/S2): state (q, qdot of both arms) -> safe torque
    correction. REAL system later = the learned HJ value V(q) (DeepReach,
    trained on x_HJ=[q_L,q_R]) composed with the v4 cooperative HOCBF-QP
    (experiments/ab_smooth_vs_bangbang/RESULTS_v4.md condition C) -- V(q)
    (+ its gradient) replaces the analytic h(q)=dist-radii used below as the
    psi1 terminal/CBF value; everything downstream (psi1/psi2 HOCBF rows,
    the box-QP) is unchanged. PLACEHOLDER here = literally run_ab.py's C2
    law, reusing sim_common's own building blocks (all_pair_barriers /
    build_weighted_least_squares_qp / solve_constrained_box_qp / ALPHA1 /
    ALPHA2 / TAU_MAX / TorqueCommandFilter), not reimplemented.

  - SemanticSafety (S3): observation -> failure risk / veto. REAL system
    later = a trained DreamerV3 (RSSM) failure head reading z_t = RSSM(o<=t)
    where o_t = {image, proprio(26D), action} PER docs/design/
    state_spaces.md -- NEVER object pose or ground-truth flags. PLACEHOLDER
    here = an ORACLE that uses scripts/cup_task_utils.count_spilled plus a
    ground-truth ball-velocity extrapolation to flag imminent spill; this
    intentionally breaks the sim2real proprio rule (it IS the ground truth)
    -- the real monitor must switch its `assess()` input to the restricted
    SemanticObservation only.

Both roles are plain Python Protocols (structural typing) so a real model
can be dropped in by implementing `reset()`/`correct()` or `reset()`/
`assess()` with no changes to run_eval.py.

This module only imports (never edits) sim_common.py / cup_task_utils.py.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import mujoco

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SIM_COMMON_DIR = _REPO_ROOT / "experiments" / "ab_smooth_vs_bangbang"
for _p in (str(_REPO_ROOT), str(_SIM_COMMON_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sim_common as sc  # noqa: E402  (read-only import; do not edit)
from scripts import cup_task_utils as ct  # noqa: E402  (read-only import; do not edit)


# ---------------------------------------------------------------------------
# Geometric role
# ---------------------------------------------------------------------------
@dataclass
class GeometricInputs:
    """S1/S2-only inputs (docs/design/state_spaces.md): joint state of both
    arms + the ego's already-payload-coupled dynamics + its nominal task
    torque. Deliberately excludes anything semantic/visual."""
    model: mujoco.MjModel
    data: mujoco.MjData
    left_dofadr: np.ndarray
    right_dofadr: np.ndarray
    tau_task: np.ndarray
    M_ll: np.ndarray
    bias_l: np.ndarray


@dataclass
class GeometricResult:
    tau: np.ndarray
    active: bool
    feasible: bool
    min_h: float
    min_dist: float
    solve_time_s: float


@runtime_checkable
class GeometricSafety(Protocol):
    def reset(self) -> None: ...
    def correct(self, inputs: GeometricInputs) -> GeometricResult: ...


class NoGeometricSafety:
    """`none` role: pass the nominal task torque through unmodified. Still
    reports the (cheap, position-only) min arm-arm distance for logging."""

    def reset(self) -> None:
        pass

    def correct(self, inputs: GeometricInputs) -> GeometricResult:
        h, dist, *_ = sc.min_pair_distance(inputs.model, inputs.data)
        return GeometricResult(tau=inputs.tau_task, active=False, feasible=True,
                                min_h=h, min_dist=dist, solve_time_s=0.0)


class AnalyticHOCBFQPGeometricSafety:
    """PLACEHOLDER for the real geometric certificate.

    Byte-for-byte the run_ab.py "C2" law: analytic arm-arm distance barrier
    h(q) (sc.all_pair_barriers over the ~25x25 keypoint proxy), HOCBF
    psi1=hdot+alpha1*h / psi2=psi1dot+alpha2*psi1 rows, solved as a box-QP
    (sc.build_weighted_least_squares_qp + sc.solve_constrained_box_qp), then
    the standard low-pass/rate-limit TorqueCommandFilter.

    Swap-in point for the real HJ value: replace `h_p`/`hdot_p` (and the
    `sc.ALPHA1 * h_p` term inside psi1_p) with `V(q)` and its time-derivative
    from a trained DeepReach network -- the constraint-row algebra and QP
    call below are otherwise unchanged. See RESULTS_v4.md Sec.6 for the
    further step of composing this with a cooperative (both-arms) filter.
    """

    def __init__(self):
        self._filter: Optional[sc.TorqueCommandFilter] = None

    def reset(self) -> None:
        self._filter = sc.TorqueCommandFilter(
            6, sc.FILTER_LOWPASS_ALPHA, np.full(6, sc.FILTER_RATE_LIMIT_NM_S))

    def correct(self, inputs: GeometricInputs) -> GeometricResult:
        model, data = inputs.model, inputs.data
        t0 = time.perf_counter()
        M_ll_inv = np.linalg.inv(inputs.M_ll)
        pairs = sc.all_pair_barriers(model, data, inputs.left_dofadr, inputs.right_dofadr)
        a_rows, b_rows = [], []
        for p in pairs:
            n_p, Je_p, Ji_p = p["n"], p["Je"], p["Ji"]
            JeDot_p, JiDot_p = p["JeDot"], p["JiDot"]
            v_ego_p, v_int_p = p["v_ego"], p["v_int"]
            h_p, hdot_p, dist_p = p["h"], p["hdot"], p["dist"]
            qacc_int = data.qacc[inputs.right_dofadr]
            a_int_p = Ji_p @ qacc_int + JiDot_p @ data.qvel[inputs.right_dofadr]
            ndot_p = (np.eye(3) - np.outer(n_p, n_p)) @ (v_ego_p - v_int_p) / max(dist_p, 1e-6)
            hddot_rest_p = float(ndot_p @ (v_ego_p - v_int_p)
                                  + n_p @ (JeDot_p @ data.qvel[inputs.left_dofadr])
                                  - n_p @ a_int_p)
            psi1_p = hdot_p + sc.ALPHA1 * h_p
            A_h_p = n_p @ Je_p @ M_ll_inv
            const_term_p = (-float(n_p @ Je_p @ M_ll_inv @ inputs.bias_l) + hddot_rest_p
                            + sc.ALPHA1 * hdot_p + sc.ALPHA2 * psi1_p)
            a_rows.append(-A_h_p)
            b_rows.append(const_term_p)
        a_ineq = np.stack(a_rows, axis=0)
        b_ineq = np.array(b_rows)

        H_qp, f_qp = sc.build_weighted_least_squares_qp([(np.eye(6), inputs.tau_task, 1.0)])
        lower = -sc.TAU_MAX * np.ones(6)
        upper = sc.TAU_MAX * np.ones(6)
        tau_qp, _lam, feasible = sc.solve_constrained_box_qp(H_qp, f_qp, lower, upper, a_ineq, b_ineq)
        tau_cmd = self._filter.apply(tau_qp, sc.DT)
        solve_time = time.perf_counter() - t0

        h_min = float(min(p["h"] for p in pairs))
        dist_min = float(min(p["dist"] for p in pairs))
        return GeometricResult(
            tau=np.clip(tau_cmd, -sc.TAU_MAX, sc.TAU_MAX), active=True,
            feasible=bool(feasible), min_h=h_min, min_dist=dist_min,
            solve_time_s=solve_time)


# ---------------------------------------------------------------------------
# Semantic role
# ---------------------------------------------------------------------------
@dataclass
class SemanticObservation:
    """S3-restricted observation the REAL monitor may use (image + proprio +
    action only -- see docs/design/state_spaces.md). `model`/`data` are
    passed through only so the oracle placeholder can cheat; a real
    implementation must read ONLY image/proprio/action fields, never these."""
    model: mujoco.MjModel
    data: mujoco.MjData
    t: float


@dataclass
class SemanticVerdict:
    risk: float           # in [0, 1]
    veto: bool
    spilled_now: bool = False


@runtime_checkable
class SemanticSafety(Protocol):
    def reset(self) -> None: ...
    def assess(self, obs: SemanticObservation) -> SemanticVerdict: ...


class NoSemanticSafety:
    def reset(self) -> None:
        pass

    def assess(self, obs: SemanticObservation) -> SemanticVerdict:
        return SemanticVerdict(risk=0.0, veto=False, spilled_now=False)


def _ball_lin_vel(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> np.ndarray:
    jadr = model.body_jntadr[body_id]
    dofadr = model.jnt_dofadr[jadr]
    return data.qvel[dofadr:dofadr + 3].copy()


class OracleSpillMonitor:
    """PLACEHOLDER for the real WM failure head.

    An ORACLE: uses ct.count_spilled (already-spilled) plus a linear
    ground-truth extrapolation of each ball's position (current pos + its
    current linear velocity * horizon_s) to flag IMMINENT spill before it
    actually crosses the cup boundary. `risk` = fraction of balls
    already-or-predicted spilled; `veto=True` when risk >= risk_threshold.

    Real system later: same `assess(obs)->SemanticVerdict` interface, but
    `risk` comes from a trained DreamerV3 P(spill) head on
    z_t=RSSM(o_{<=t}) with o_t restricted to {image, proprio(26D), action}
    (docs/design/state_spaces.md S3) -- no object pose, no ground truth.
    """

    def __init__(self, horizon_s: float = 0.35, risk_threshold: float = 0.12):
        self.horizon_s = horizon_s
        self.risk_threshold = risk_threshold

    def reset(self) -> None:
        pass

    def assess(self, obs: SemanticObservation) -> SemanticVerdict:
        model, data = obs.model, obs.data
        cup_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ct.CUP_BODY_NAME)
        cup_pos = data.xpos[cup_bid].copy()
        cup_mat = data.xmat[cup_bid].reshape(3, 3).copy()

        spilled_now = ct.count_spilled(model, data) > 0
        ball_ids = ct.ball_body_ids(model)
        n_flag = 0
        for bid in ball_ids:
            p = data.xpos[bid].copy()
            v = _ball_lin_vel(model, data, bid)
            p_future = p + v * self.horizon_s
            local = cup_mat.T @ (p_future - cup_pos)
            radial = float(np.hypot(local[0], local[1]))
            height = float(local[2])
            if (radial > ct.CUP_RADIUS or height < -0.002 or height > ct.CUP_HEIGHT):
                n_flag += 1
        risk = float(n_flag) / max(len(ball_ids), 1)
        veto = spilled_now or (risk >= self.risk_threshold)
        return SemanticVerdict(risk=risk, veto=veto, spilled_now=spilled_now)


GEOMETRIC_REGISTRY = {
    "none": NoGeometricSafety,
    "hocbf_qp": AnalyticHOCBFQPGeometricSafety,
}
SEMANTIC_REGISTRY = {
    "none": NoSemanticSafety,
    "oracle": OracleSpillMonitor,
}
