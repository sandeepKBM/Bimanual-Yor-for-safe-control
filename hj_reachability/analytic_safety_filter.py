"""Analytic Hamilton-Jacobi fail-safe SAFETY FILTER for the bimanual arm-arm
crossing -- NO neural net, NO training.

This deploys the *analytic* pieces already living in ``dynamics.py``
(``BimanualPairAvoidance``) as a minimally-invasive velocity-level safety
filter on the real Piper crossing task, reproducing the fail-safe control
mechanism of NeHMO / HaMMaR (both use the saturated boundary-avoidance law
``u* = -ubar * P(x) gradV / ||P(x) gradV||``) WITHOUT their unreleased code.

WHAT VALUE / WHAT GRADIENT (honest, see the F2 note at the bottom):
  The "value" here is the *analytic signed-distance* clearance
    V(x) := -l(x),   l(x) = h = min-pair (arm-arm) keypoint clearance,
  i.e. exactly ``sim_common.min_pair_distance`` / ``dynamics.boundary_fn``.
  This is the terminal cost l(x) of the single-integrator differential game
  in ``dynamics.py`` -- NOT the converged time-to-collision BRT value a
  DeepReach net would learn. Using -l as V puts us in the standard avoid
  convention (V<=0 safe boundary crossed), and steering to *increase clearance*
  is exactly ``dynamics.BimanualPairAvoidance.optimal_control`` applied to the
  ego arm.

  The state-gradient is the EXACT analytic gradient of the clearance w.r.t.
  the ego arm's joint angles, obtained from the REAL Piper geometric Jacobian
  at the closest keypoint (mj_jac, the same FK-Jacobian oracle path
  ``mujoco_fk.py`` / ``sim_common.point_jac_at`` provide, verified vs
  finite-diff ~1e-10):
    d(dist)/dq_int = -(n . J_int),   n = (p_ego - p_int)/dist,
  so the VALUE gradient (V=-l) w.r.t. the ego (=right/int) joints is
    dvds = d(-l)/dq_int = +(n . J_int).
  Feeding that dvds into ``optimal_control`` yields the saturated fail-safe
  joint-velocity command
    u* = -ubar * dvds/||dvds|| = +ubar * grad(clearance)/||grad(clearance)||,
  i.e. drive the ego joints at the ball-bounded speed ``ubar`` in the
  direction that most increases arm-arm clearance -- the bang-bang boundary
  control.

EGO / DISTURBANCE CHOICE:
  We filter the RIGHT arm as EGO (it is the arm performing the crossing and
  the arm whose EE smoothness we measure), treating the LEFT (cup) arm as the
  neighbor/disturbance (it simply holds the lifted cup). This matches the
  HaMMaR / NeHMO ego-vs-worst-case-neighbor framing. NOTE the FK-oracle naming
  convention is the opposite (ego=left, int=right); here the *right* arm is the
  controlled ego, so its clearance gradient is the ``int`` gradient above.

ENGAGEMENT MODES:
  - "hard"  : bang-bang switch. When h < margin, REPLACE the nominal
              velocity command with the full saturated u* (the pure fail-safe
              law). Reproduces the discretely-switching HJ-optimal controller.
  - "blend" : deadzone/blend. Linearly blend nominal->u* as h sweeps the band
              [0, margin] (blend=0 at h>=margin, blend=1 at h<=0). A softer
              variant to test whether a deadzone reduces chatter.

An inner velocity servo (computed-torque: tau = M (Kv (v_cmd - qd)) + bias)
turns the (possibly overridden) velocity command into Piper joint torque,
mirroring how ``make_cup_collect_safety_demo.py`` converts commands to torque.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

_HJ_DIR = Path(__file__).resolve().parent
_REPO = _HJ_DIR.parent
_AB4 = _REPO / "experiments" / "ab_smooth_vs_bangbang"
for _p in (str(_HJ_DIR), str(_AB4)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sim_common as sc          # noqa: E402  real-geometry keypoint proxy + Jacobians
import dynamics as hj_dynamics   # noqa: E402  analytic BimanualPairAvoidance (optimal_control)


@dataclass
class FilterInfo:
    """Per-step diagnostics from the filter (for logging/verification)."""
    h: float                 # current arm-arm clearance l(x)
    engaged: bool            # did the safety law override (blend>0) this step?
    blend: float             # blend weight actually applied in [0,1]
    grad_norm: float         # ||dvds|| (value-gradient magnitude); 0 => degenerate/no pair
    u_safe: np.ndarray       # the saturated fail-safe joint-velocity (6,)
    v_cmd: np.ndarray        # the velocity command actually sent to the inner servo (6,)


class AnalyticHJSafetyFilter:
    """Analytic HJ fail-safe filter on the RIGHT (ego) arm's joint-velocity
    command. Stateless across steps except for the fixed config; call
    ``filter_velocity`` each control step with the live model/data.

    Parameters
    ----------
    right_dofadr : int array (6,)   -- the ego (right) arm's dof addresses
    u_bar        : float            -- saturated joint-velocity ball bound (rad/s)
    margin       : float            -- clearance [m] at/below which the filter engages
    mode         : "hard" | "blend"
    """

    def __init__(self, right_dofadr, *, u_bar: float = hj_dynamics.DEFAULT_U_BAR_EGO,
                 margin: float = 0.05, mode: str = "hard"):
        assert mode in ("hard", "blend"), mode
        self.right_dofadr = np.asarray(right_dofadr, dtype=int)
        self.u_bar = float(u_bar)
        self.margin = float(margin)
        self.mode = mode
        # analytic value model -- used ONLY for its optimal_control() (no oracle
        # load; that lazy attribute is never touched here). Configured to the
        # same ubar so the saturated magnitude matches the deployed servo.
        self._hj = hj_dynamics.BimanualPairAvoidance(
            hj_dynamics.BimanualPairAvoidanceConfig(u_bar_ego=self.u_bar, u_bar_int=self.u_bar))

    # -- analytic value + gradient at the current state -----------------------
    def clearance_and_grad(self, model, data):
        """Returns (h, dvds) where h=l(x) is the worst-pair arm-arm clearance
        and dvds = d(-l)/dq_right is the analytic VALUE gradient w.r.t. the ego
        (right) joints, via the real-geometry closest-keypoint Jacobian."""
        h, dist, eb, ib, n, eoff, ioff = sc.min_pair_distance(model, data)
        Ji = sc.point_jac_at(model, data, ib, ioff)[:, self.right_dofadr]  # 3x6
        # d(dist)/dq_int = -(n . J_int); value V=-l => dvds = +(n . J_int)
        dvds = n @ Ji
        return float(h), dvds

    # -- the fail-safe control law -------------------------------------------
    def safe_velocity(self, dvds):
        """Saturated fail-safe joint-velocity u* = optimal_control(dvds)
        = -ubar * dvds/||dvds|| (the analytic HJ boundary control)."""
        return self._hj.optimal_control(dvds)

    def _blend_weight(self, h: float) -> float:
        if self.mode == "hard":
            return 1.0 if h < self.margin else 0.0
        # blend: 0 at h>=margin, ramp to 1 at h<=0
        if h >= self.margin:
            return 0.0
        if h <= 0.0:
            return 1.0
        return float((self.margin - h) / self.margin)

    def filter_velocity(self, model, data, v_nom) -> FilterInfo:
        """Given the nominal ego (right-arm) joint-velocity command v_nom (6,),
        return the safety-filtered command + diagnostics."""
        v_nom = np.asarray(v_nom, dtype=float)
        h, dvds = self.clearance_and_grad(model, data)
        gnorm = float(np.linalg.norm(dvds))
        u_safe = self.safe_velocity(dvds)
        w = self._blend_weight(h)
        if gnorm < 1e-9:
            w = 0.0  # degenerate pair (parallel/zero gradient) -> no safe direction; keep nominal
        v_cmd = (1.0 - w) * v_nom + w * u_safe
        return FilterInfo(h=h, engaged=(w > 0.0), blend=w, grad_norm=gnorm,
                          u_safe=u_safe, v_cmd=v_cmd)


def velocity_servo_torque(M, bias, qd, v_cmd, kv: float = 50.0, tau_max: float = sc.TAU_MAX):
    """Inner computed-torque VELOCITY servo: tau = M (kv (v_cmd - qd)) + bias,
    clipped to +-tau_max. Mirrors the demo's computed-torque command->torque
    conversion (M @ qacc_des + bias), specialised to velocity tracking."""
    qacc_des = kv * (np.asarray(v_cmd) - np.asarray(qd))
    return np.clip(M @ qacc_des + bias, -tau_max, tau_max)
