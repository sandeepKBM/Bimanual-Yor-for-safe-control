"""v4 experiment: does a COOPERATIVE intruder dissolve the collision-vs-grasp
trade-off that every controller (C0/C1/C2/C3) failed in RESULTS_v3.md, on the
tight real-YOR geometry (scenes/scene_yor_canonical.xml, arms 0.185m apart)?

Ego (LEFT arm, grasping test_box) is FIXED across all three conditions below
to exactly the v3 "C2" controller (smooth HOCBF-QP, ALPHA1=ALPHA2=30,
sim_common.EgoController + the same per-pair HOCBF-QP construction as
run_ab.py's C2 branch) -- only the INTRUDER (RIGHT arm) model changes:

  A - Adversarial baseline (= v3's C2 row exactly): intruder is the existing
      open-loop position-servo sweep (sim_common.intruder_ctrl), unresponsive
      to the ego. Included as a same-code-path sanity check that this script
      reproduces RESULTS_v3.md's C2 numbers before trusting B/C.
  B - Cooperative symmetric: intruder is ALSO torque-controlled -- joint-space
      computed-torque PD toward its nominal intruder_ctrl(t) target, filtered
      by its OWN HOCBF-QP against the same faithful ~25-sphere/arm proxy
      (sim_common.all_pair_barriers), same class-K gains as the ego
      (ALPHA1=ALPHA2=30) -- i.e. both arms yield with equal authority.
  C - Cooperative, ego priority: identical to B except the intruder's own
      class-K gains are raised (ALPHA1=ALPHA2=90, 3x) so IT reacts earlier/
      harder while the ego's controller and gains are byte-identical to A/B
      -- the free arm (protecting nothing) does most of the yielding; the
      grasping ego is left as close as possible to its undisturbed C2
      behavior.

Decentralized-reactive HOCBF-QP construction (both ego and intruder solve
their OWN box-constrained QP each step, each treating the OTHER arm's
current measured acceleration, data.qacc[other_dofadr] from the previous
mj_forward/mj_step, as a fixed disturbance -- NOT a jointly-optimized/
centralized plan; see RESULTS_v4.md Sec. "Caveats"):

  ego QP (unchanged from run_ab.py's C2): for each proxy pair p,
    hddot = ndot.(v_ego-v_int) + n.(Je@qacc_ego + JeDot@qvel_ego)
                                - n.(Ji@qacc_int_KNOWN + JiDot@qvel_int)
    psi2 = hddot + ALPHA1*hdot + ALPHA2*psi1 >= 0, solved for tau_ego via
    qacc_ego = Minv_ego@(tau_ego - bias_ego)
    => a_row = -(n@Je@Minv_ego), b_row = const term (see run_ab.py C2).

  intruder QP (new, mirror-derived -- v_int enters hdot with a MINUS sign,
  so the intruder's own controllable term flips sign relative to the ego's):
    hddot = ndot.(v_ego-v_int) + n.(Je@qacc_ego_KNOWN + JeDot@qvel_ego)
                                - n.(Ji@qacc_int + JiDot@qvel_int)
    psi2 = hddot + ALPHA1_int*hdot + ALPHA2_int*psi1_int >= 0, solved for
    tau_int via qacc_int = Minv_int@(tau_int - bias_int)
    => a_row = +(n@Ji@Minv_int), b_row = const term (sign flip vs ego's).

Run:
    MUJOCO_GL=egl /common/users/ss5772/miniforge3/bin/conda run -n safe_env \\
        python /common/users/ss5772/bimanual/experiments/ab_smooth_vs_bangbang/run_ab4.py

Writes experiments/ab_smooth_vs_bangbang/logs/v4_{A,B,C}.npz +
logs/v4_summary.json (+ printed table).
"""

from __future__ import annotations

import os
import json
import numpy as np
import mujoco

import sim_common as sc


LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

CONDITIONS = ["A", "B", "C"]
# Intruder HOCBF class-K gains per condition. A: n/a (open-loop, no QP).
# B (symmetric): same ALPHA as the ego (30/30) -- equal reaction authority.
# C (ego-priority) asymmetry: NOT via a higher class-K gain (tried first --
# ALPHA_int=90 with the SAME +-40Nm torque budget instead destabilized the
# intruder's own QP, 34%/11% ego/int infeasible fraction, worse than A on
# every metric -- an aggressive class-K gain without matching actuation
# headroom is brittle under saturation, a real and reported negative result,
# see RESULTS_v4.md). Implemented instead via an EXTRA static margin added
# only to the intruder's own barrier value (INTRUDER_EXTRA_MARGIN below) --
# same ALPHA=30 as B (numerically well-behaved), but the intruder starts
# backing off at a larger true physical gap than the ego does, i.e. it does
# most of the yielding by treating the shared boundary as tighter than the
# ego does, not by reacting more violently once already close.
INTRUDER_ALPHA = {"A": None, "B": 30.0, "C": 30.0, "C-adapt": 30.0}
# v4-C* validated default (2026-09 distributional sweep, N=50 scenes): C's
# static extra margin is 0.03, NOT the original 0.06 above/below this line's
# history -- 0.03 dominates 0.06 on every axis (collision 4% vs 8%, grasp
# 82% vs 80%, task-success 74% vs 30%, chatter 3.29, qp_infeasible 0.009),
# so it replaces 0.06 as condition C's default here. 0.06 remains fully
# reproducible on demand via
# `run_sweep.py --coop-margins 0.06` (adds a separate 'v4-C-m0.06' column
# with ab4.INTRUDER_EXTRA_MARGIN["C"] overridden to 0.06 for that run only;
# see run_sweep.py's MARGIN_OVERRIDE/--coop-margins plumbing). A and B are
# unchanged.
INTRUDER_EXTRA_MARGIN = {"A": 0.0, "B": 0.0, "C": 0.03}
INTRUDER_KP_TASK = 600.0
INTRUDER_KD_TASK = 2.0 * (INTRUDER_KP_TASK ** 0.5)

# ---------------------------------------------------------------------------
# ADDITIVE (2026-09): conditions beyond A/B/C, added to push the v4-C
# collision rate (4/50 scenes, all coinciding with ego QP infeasibility --
# see the sweep diagnosis) further toward 0 without touching A/B/C's
# existing byte-identical behavior.
#
# "C-adapt" -- ego priority, but the intruder's extra barrier margin is no
# longer a fixed scalar (INTRUDER_EXTRA_MARGIN["C"] = 0.06 for ALL pairs at
# ALL times): it GROWS as a given ego/intruder keypoint pair's own distance
# shrinks, via _adaptive_margin_schedule below, so conservatism (the
# intruder backing off) is added ONLY on the specific pairs that are close/
# deep right now, while staying exactly as permissive as C=0.0 on pairs that
# are still far away -- intended to make the intruder retreat earlier/
# farther on the specific close pairs that would otherwise blow up the
# number of simultaneously-binding ego HOCBF-QP constraints at the deepest
# crossings (the diagnosed proximate cause of the 4 collisions).
# ---------------------------------------------------------------------------
COOPERATIVE_CONDITIONS = ("B", "C", "C-adapt")

ADAPTIVE_MARGIN_H_FAR = 0.20     # [m] h (the CBF value itself -- zero-referenced clearance, ALREADY
                                  # nets out each pair's own combined radius + D_MIN, unlike raw dist)
                                  # at/above which extra margin = ADAPTIVE_MARGIN_MIN (fully permissive).
                                  # Deliberately wide (a smoke check on scenes 0/2 -- the exact 2 scenes
                                  # in a 50-scene N=50 baseline sweep where A collides deep, depth_scale
                                  # ~1.08-1.11, and static C/0.06 avoided collision on BOTH -- showed a
                                  # narrower/purely-reactive H_FAR=0.08 schedule collides on both, worse
                                  # than a constant margin present from t=0: the intruder needs "runway"
                                  # to start backing off before closing velocity builds up, not just a
                                  # late proximity trigger. Widening H_FAR spreads SOME conservatism
                                  # across a much larger clearance range, closer in spirit to why a
                                  # constant margin works, while still going higher than the constant
                                  # only once truly close.)
ADAPTIVE_MARGIN_H_NEAR = 0.0     # [m] h at/below which extra margin saturates at ADAPTIVE_MARGIN_MAX
                                  # (right at/through the nominal safety boundary)
ADAPTIVE_MARGIN_MIN = 0.0        # [m]
ADAPTIVE_MARGIN_MAX = 0.10       # [m] -- mildly above the static C margin (0.06), applied only right
                                  # at the closest/deepest pairs


def _adaptive_margin_schedule(h_p: float) -> float:
    """Piecewise-linear extra margin vs. this pair's CURRENT h (not raw
    keypoint distance -- h already subtracts each pair's own combined
    sphere radius + D_MIN, see sim_common.all_pair_barriers, so it is
    directly comparable across the ~625 differently-sized pairs and
    directly comparable to the static C margin, both being clearance
    values in the same h-space). ADAPTIVE_MARGIN_MIN for h>=H_FAR,
    ADAPTIVE_MARGIN_MAX for h<=H_NEAR, linear ramp in between. (An earlier
    version of this schedule keyed on raw dist_p instead of h and used
    MAX=0.12/D_FAR=0.15m in dist-space -- an N=3 smoke check showed it
    over-triggers on large-radius pairs at what is actually still a modest
    true clearance, causing intruder-QP saturation/chatter; switched to
    h-space with a milder MAX for exactly that reason.)"""
    if h_p >= ADAPTIVE_MARGIN_H_FAR:
        return ADAPTIVE_MARGIN_MIN
    if h_p <= ADAPTIVE_MARGIN_H_NEAR:
        return ADAPTIVE_MARGIN_MAX
    frac = (ADAPTIVE_MARGIN_H_FAR - h_p) / (ADAPTIVE_MARGIN_H_FAR - ADAPTIVE_MARGIN_H_NEAR)
    return ADAPTIVE_MARGIN_MIN + frac * (ADAPTIVE_MARGIN_MAX - ADAPTIVE_MARGIN_MIN)


# condition -> per-pair margin function; None (A/B/C, unchanged) falls back
# to the existing constant INTRUDER_EXTRA_MARGIN[condition] lookup.
INTRUDER_MARGIN_SCHEDULE = {"C-adapt": _adaptive_margin_schedule}


def _resolve_extra_margin(condition: str, h_p: float) -> float:
    fn = INTRUDER_MARGIN_SCHEDULE.get(condition)
    if fn is not None:
        return fn(h_p)
    return INTRUDER_EXTRA_MARGIN.get(condition, 0.0)


def _solve_ego_qp_with_relaxation(H_qp, f_qp, lower, upper, a_ineq, b_ineq,
                                   slack_step=0.05, max_relax_iters=6, top_k=8):
    """Wraps sc.solve_constrained_box_qp UNCHANGED (same solver, same 6-dof
    decision variable -- no new QP formulation, no new decision variables)
    with a bounded constraint-relaxation retry for when the hard problem is
    genuinely infeasible (per the diagnosis: TAU_MAX=40Nm actuator
    saturation under ~625 simultaneous pairwise HOCBF constraints at the
    deepest crossings -- ALL 4/50 v4-C collision scenes had
    qp_infeasible_frac_ego in [0.24, 0.31] vs EXACTLY 0.0 for all 46 clean
    scenes).

    On infeasibility, relaxes (raises) b_ineq by `slack_step` for only the
    `top_k` MOST-violated rows (by residual a_ineq@x - b_ineq of the
    previous solve) and re-solves the identical hard QP, repeating up to
    `max_relax_iters` times. This is functionally a high-penalty soft
    constraint (only the minimum number of worst-offending rows are relaxed,
    only by the smallest number of `slack_step` increments needed to reach
    feasibility, only on steps where the hard problem has NO feasible
    point at all) without adding per-constraint slack decision variables to
    the ~625-constraint QP (which would multiply its per-step cost).
    Returns (x, lam, feasible, n_relax_iters_used)."""
    b = b_ineq.copy()
    x, lam, feasible = sc.solve_constrained_box_qp(H_qp, f_qp, lower, upper, a_ineq, b)
    if feasible:
        return x, lam, feasible, 0
    for it in range(max_relax_iters):
        residual = a_ineq @ x - b
        worst = np.argsort(-residual)[:top_k]
        if np.all(residual[worst] <= 1e-6):
            break
        b[worst] += slack_step
        x, lam, feasible = sc.solve_constrained_box_qp(H_qp, f_qp, lower, upper, a_ineq, b)
        if feasible:
            return x, lam, feasible, it + 1
    return x, lam, feasible, max_relax_iters


def intruder_ctrl_nominal(t: float) -> np.ndarray:
    """The cooperative intruder's OWN 'nominal sweep goal' reference: same
    ramp-in / hold / ramp-out timeline and same RIGHT_HOME_CTRL <->
    RIGHT_INTRUDE_CTRL endpoints as sc.intruder_ctrl, but WITHOUT the 5Hz/
    OSC_AMP oscillation.

    Empirically required (see RESULTS_v4.md "Confound found + fixed"): the
    oscillation in sc.intruder_ctrl is an ADVERSARIAL-only stress element
    (sim_common.py's own comment: chosen "near/above the ego task-PD's
    natural frequency ... what makes bang-bang (C1) chatter/destabilize"),
    tuned assuming a stiff, near-instantaneous MuJoCo position SERVO can
    track it. A torque-controlled computed-torque PD (finite gains, +-40Nm,
    rate-limited command filter -- the intruder's realistic actuation
    budget) CANNOT track a 5Hz joint-space oscillation: an isolated
    open-loop check (no collision QP at all) showed tracking error growing
    unboundedly (>1.4 rad by t=1.5s) purely from chasing this reference,
    fully independent of any collision-avoidance interaction -- a confound
    that would contaminate any A vs B/C comparison. Dropping the oscillation
    (ramp to 1.0 and HOLD, instead of ramp-and-oscillate) keeps the same
    intrusion-depth GOAL (reach + hold at RIGHT_INTRUDE_CTRL, then retreat)
    while being trackable: an isolated re-check with this reference gives
    <=0.16 rad transient error and ~0.04 rad steady-state offset, both
    stable, no divergence. Condition A is UNCHANGED (still uses
    sc.intruder_ctrl with the full oscillation, exactly reproducing v3)."""
    if t < sc.T_IN0:
        s = 0.0
    elif t < sc.T_IN1:
        s = sc._smoothstep(t, sc.T_IN0, sc.T_IN1)
    elif t < sc.T_HOLD_END:
        s = 1.0
    elif t < sc.T_OUT_END:
        s = 1.0 - sc._smoothstep(t, sc.T_HOLD_END, sc.T_OUT_END)
    else:
        s = 0.0
    return sc.RIGHT_HOME_CTRL + s * (sc.RIGHT_INTRUDE_CTRL - sc.RIGHT_HOME_CTRL)


def intruder_ctrl_nominal_dot(t: float, h: float = 1e-4) -> np.ndarray:
    return (intruder_ctrl_nominal(t + h) - intruder_ctrl_nominal(t - h)) / (2.0 * h)


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_angle(q):
    w = np.clip(abs(q[0]), -1.0, 1.0)
    return 2.0 * np.arccos(w)


def left_right_geom_sets(model):
    left_geoms, right_geoms = set(), set()
    for gi in range(model.ngeom):
        bid = model.geom_bodyid[gi]
        bname = model.body(bid).name
        if bname.startswith("left_arm_"):
            left_geoms.add(gi)
        elif bname.startswith("right_arm_"):
            right_geoms.add(gi)
    return left_geoms, right_geoms


def run_condition(condition: str, capture_steps=None, capture_cb=None,
                   ego_relax_on_infeasible: bool = False,
                   ego_relax_step: float = 0.05, ego_relax_max_iters: int = 6,
                   ego_relax_top_k: int = 8):
    """ego_relax_on_infeasible (default False, byte-identical to the
    original A/B/C behavior when off): if True, use
    _solve_ego_qp_with_relaxation instead of a single hard
    sc.solve_constrained_box_qp call for the EGO QP only (see that
    function's docstring) -- a bounded, reversible fallback for genuine
    actuator-saturation infeasibility, selectable per run without touching
    A/B/C's default path."""
    cooperative = condition in COOPERATIVE_CONDITIONS
    model = sc.load_model()
    data = sc.init_grasped_state(model)
    q0_left = data.qpos[sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)[1]].copy()
    sc.disable_left_arm_position_servos(model)
    if cooperative:
        sc.disable_right_arm_position_servos(model)

    ctrl = sc.EgoController(model, q0_left=q0_left)

    left_geoms, right_geoms = left_right_geom_sets(model)

    left_j7 = model.actuator("left_arm_joint7_pos").id
    left_j8 = model.actuator("left_arm_joint8_pos").id
    right_act_ids = [model.actuator(f"right_arm_joint{i}_pos").id for i in range(1, 7)]

    ee_body = model.body("left_arm_gripper_base").id
    right_ee_body = model.body("right_arm_gripper_base").id
    box_body = model.body("test_box").id
    box_geom = model.geom("test_box_geom").id
    pad7 = model.geom("left_arm_link7_pad").id
    pad8 = model.geom("left_arm_link8_pad").id

    tau_filter_ego = sc.TorqueCommandFilter(6, sc.FILTER_LOWPASS_ALPHA,
                                             np.full(6, sc.FILTER_RATE_LIMIT_NM_S))
    tau_filter_int = sc.TorqueCommandFilter(6, sc.FILTER_LOWPASS_ALPHA,
                                             np.full(6, sc.FILTER_RATE_LIMIT_NM_S))

    alpha1_int = alpha2_int = INTRUDER_ALPHA[condition]

    n_steps = sc.N_STEPS
    log = {k: np.zeros(n_steps) for k in
           ["t", "h", "dist", "hdot", "psi1", "psi2", "qp_feasible_ego",
            "qp_feasible_int", "contact_force", "collision", "box_z",
            "box_drift_trans", "box_drift_rot", "int_track_err", "int_qdes_norm",
            "ego_relax_iters"]}  # ego_relax_iters: 0 unless ego_relax_on_infeasible=True
                                  # AND the hard ego QP was infeasible this step (additive; 0 for A/B/C's default path)
    log["tau_ego"] = np.zeros((n_steps, 6))
    log["tau_int"] = np.zeros((n_steps, 6))
    log["q_ego"] = np.zeros((n_steps, 6))
    log["q_int"] = np.zeros((n_steps, 6))
    log["q_int_des"] = np.zeros((n_steps, 6))
    log["qvel_ego"] = np.zeros((n_steps, 6))
    log["ee_pos"] = np.zeros((n_steps, 3))
    log["box_pos"] = np.zeros((n_steps, 3))
    log["right_ee_pos"] = np.zeros((n_steps, 3))

    box_pos_rel0 = None
    box_quat_rel0 = None

    for k in range(n_steps):
        t = k * sc.DT
        if cooperative:
            # cooperative intruder pursues its own trackable nominal goal
            # (see intruder_ctrl_nominal docstring: same ramp/hold/retreat
            # timeline and endpoints, oscillation dropped -- an adversarial-
            # only stress element a torque-limited PD cannot track).
            q_int_des = intruder_ctrl_nominal(t)
            qd_int_des = intruder_ctrl_nominal_dot(t)
        else:
            q_int_des = sc.intruder_ctrl(t)
            data.ctrl[right_act_ids] = q_int_des
        data.ctrl[left_j7] = 0.0
        data.ctrl[left_j8] = 0.0

        mujoco.mj_forward(model, data)

        tau_task, M_ll, bias_l = ctrl.task_torque(data)
        bar = ctrl.barrier(data)
        h, hdot, n_vec = bar["h"], bar["hdot"], bar["n"]
        psi1 = hdot + sc.ALPHA1 * h

        M_ll_inv = np.linalg.inv(M_ll)
        pairs = sc.all_pair_barriers(model, data, ctrl.left_dofadr, ctrl.right_dofadr)

        # known/estimated OTHER-arm acceleration this step (decentralized:
        # each side treats the other's currently-realized qacc as a fixed
        # disturbance, exactly as run_ab.py's C2 already does for the
        # open-loop intruder).
        qacc_int_known = data.qacc[ctrl.right_dofadr]
        qacc_ego_known = data.qacc[ctrl.left_dofadr]

        # ---- ego HOCBF-QP (byte-identical construction to run_ab.py C2) ----
        a_rows, b_rows = [], []
        for p in pairs:
            n_p, Je_p, Ji_p = p["n"], p["Je"], p["Ji"]
            JeDot_p, JiDot_p = p["JeDot"], p["JiDot"]
            v_ego_p, v_int_p = p["v_ego"], p["v_int"]
            h_p, hdot_p, dist_p = p["h"], p["hdot"], p["dist"]
            a_int_p = Ji_p @ qacc_int_known + JiDot_p @ data.qvel[ctrl.right_dofadr]
            ndot_p = (np.eye(3) - np.outer(n_p, n_p)) @ (v_ego_p - v_int_p) / max(dist_p, 1e-6)
            hddot_rest_p = float(ndot_p @ (v_ego_p - v_int_p)
                                  + n_p @ (JeDot_p @ data.qvel[ctrl.left_dofadr])
                                  - n_p @ a_int_p)
            psi1_p = hdot_p + sc.ALPHA1 * h_p
            A_h_p = n_p @ Je_p @ M_ll_inv
            const_term_p = (-float(n_p @ Je_p @ M_ll_inv @ bias_l) + hddot_rest_p
                            + sc.ALPHA1 * hdot_p + sc.ALPHA2 * psi1_p)
            a_rows.append(-A_h_p)
            b_rows.append(const_term_p)
        a_ineq = np.stack(a_rows, axis=0)
        b_ineq = np.array(b_rows)
        H_qp, f_qp = sc.build_weighted_least_squares_qp([(np.eye(6), tau_task, 1.0)])
        lower = -sc.TAU_MAX * np.ones(6)
        upper = sc.TAU_MAX * np.ones(6)
        if ego_relax_on_infeasible:
            tau_ego_qp, _, feasible_ego, relax_iters = _solve_ego_qp_with_relaxation(
                H_qp, f_qp, lower, upper, a_ineq, b_ineq,
                slack_step=ego_relax_step, max_relax_iters=ego_relax_max_iters,
                top_k=ego_relax_top_k)
        else:
            tau_ego_qp, _, feasible_ego = sc.solve_constrained_box_qp(
                H_qp, f_qp, lower, upper, a_ineq, b_ineq)
            relax_iters = 0
        tau_ego_cmd = tau_filter_ego.apply(tau_ego_qp, sc.DT)
        tau_ego_cmd = np.clip(tau_ego_cmd, -sc.TAU_MAX, sc.TAU_MAX)
        data.qfrc_applied[ctrl.left_dofadr] = tau_ego_cmd

        feasible_int = True
        tau_int_cmd = np.zeros(6)
        if cooperative:
            # ---- intruder task torque: computed-torque PD to its own
            # nominal (scripted) sweep target, no payload ----
            M_full = sc.full_mass_matrix(model, data)
            M_rr = M_full[np.ix_(ctrl.right_dofadr, ctrl.right_dofadr)]
            bias_r = data.qfrc_bias[ctrl.right_dofadr]
            q_int = data.qpos[ctrl.right_qposadr]
            qvel_int = data.qvel[ctrl.right_dofadr]
            qacc_int_des = (INTRUDER_KP_TASK * (q_int_des - q_int)
                             + INTRUDER_KD_TASK * (qd_int_des - qvel_int))
            tau_task_int = M_rr @ qacc_int_des + bias_r
            M_rr_inv = np.linalg.inv(M_rr)

            # ---- intruder HOCBF-QP (mirror-derived sign flip vs ego's) ----
            ai_rows, bi_rows = [], []
            for p in pairs:
                n_p, Je_p, Ji_p = p["n"], p["Je"], p["Ji"]
                JeDot_p, JiDot_p = p["JeDot"], p["JiDot"]
                v_ego_p, v_int_p = p["v_ego"], p["v_int"]
                h_p, hdot_p, dist_p = p["h"], p["hdot"], p["dist"]
                h_p_int = h_p - _resolve_extra_margin(condition, h_p)  # C/C-adapt only:
                                                   # intruder treats the shared gap as tighter
                                                   # than the ego does (constant for C, grows
                                                   # as dist_p shrinks for C-adapt)
                                                   # (see INTRUDER_EXTRA_MARGIN above)
                a_ego_p = Je_p @ qacc_ego_known + JeDot_p @ data.qvel[ctrl.left_dofadr]
                ndot_p = (np.eye(3) - np.outer(n_p, n_p)) @ (v_ego_p - v_int_p) / max(dist_p, 1e-6)
                hddot_rest_int_p = float(ndot_p @ (v_ego_p - v_int_p)
                                          + n_p @ a_ego_p
                                          - n_p @ (JiDot_p @ qvel_int))
                psi1_p_int = hdot_p + alpha1_int * h_p_int
                A_h_int_p = n_p @ Ji_p @ M_rr_inv
                const_term_int_p = (float(n_p @ Ji_p @ M_rr_inv @ bias_r) + hddot_rest_int_p
                                     + alpha1_int * hdot_p + alpha2_int * psi1_p_int)
                ai_rows.append(A_h_int_p)
                bi_rows.append(const_term_int_p)
            ai_ineq = np.stack(ai_rows, axis=0)
            bi_ineq = np.array(bi_rows)
            Hi_qp, fi_qp = sc.build_weighted_least_squares_qp([(np.eye(6), tau_task_int, 1.0)])
            tau_int_qp, _, feasible_int = sc.solve_constrained_box_qp(
                Hi_qp, fi_qp, lower, upper, ai_ineq, bi_ineq)
            tau_int_cmd = tau_filter_int.apply(tau_int_qp, sc.DT)
            tau_int_cmd = np.clip(tau_int_cmd, -sc.TAU_MAX, sc.TAU_MAX)
            data.qfrc_applied[ctrl.right_dofadr] = tau_int_cmd

        # ---- logging ----
        log["t"][k] = t
        log["h"][k] = h
        log["dist"][k] = bar["dist"]
        log["hdot"][k] = hdot
        log["psi1"][k] = psi1
        log["qp_feasible_ego"][k] = float(feasible_ego)
        log["ego_relax_iters"][k] = float(relax_iters)
        log["qp_feasible_int"][k] = float(feasible_int)
        log["tau_ego"][k] = tau_ego_cmd
        log["tau_int"][k] = tau_int_cmd
        log["q_ego"][k] = data.qpos[ctrl.left_qposadr]
        log["q_int"][k] = data.qpos[ctrl.right_qposadr]
        log["q_int_des"][k] = q_int_des
        log["int_track_err"][k] = float(np.linalg.norm(data.qpos[ctrl.right_qposadr] - q_int_des))
        log["int_qdes_norm"][k] = float(np.linalg.norm(q_int_des - sc.RIGHT_HOME_CTRL))
        log["qvel_ego"][k] = data.qvel[ctrl.left_dofadr]
        log["ee_pos"][k] = data.xpos[ee_body]
        log["box_pos"][k] = data.xpos[box_body]
        log["right_ee_pos"][k] = data.xpos[right_ee_body]
        log["box_z"][k] = data.xpos[box_body][2]

        total_force = 0.0
        collision = False
        for i in range(data.ncon):
            con = data.contact[i]
            geoms = {con.geom1, con.geom2}
            if box_geom in geoms and (pad7 in geoms or pad8 in geoms):
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, force)
                total_force += float(np.linalg.norm(force[:3]))
            if (con.geom1 in left_geoms and con.geom2 in right_geoms) or \
               (con.geom2 in left_geoms and con.geom1 in right_geoms):
                collision = True
        log["contact_force"][k] = total_force
        log["collision"][k] = float(collision)

        R_gb = data.xmat[ee_body].reshape(3, 3)
        rel_pos = R_gb.T @ (data.xpos[box_body] - data.xpos[ee_body])
        gb_quat = data.xquat[ee_body].copy()
        box_quat = data.xquat[box_body].copy()
        rel_quat = quat_mul(quat_conj(gb_quat), box_quat)
        if box_pos_rel0 is None:
            box_pos_rel0 = rel_pos.copy()
            box_quat_rel0 = rel_quat.copy()
        log["box_drift_trans"][k] = float(np.linalg.norm(rel_pos - box_pos_rel0))
        drift_quat = quat_mul(quat_conj(box_quat_rel0), rel_quat)
        log["box_drift_rot"][k] = float(quat_angle(drift_quat))

        if capture_steps is not None and k in capture_steps and capture_cb is not None:
            capture_cb(condition, k, model, data)

        mujoco.mj_step(model, data)

    np.savez(os.path.join(LOG_DIR, f"v4_{condition}.npz"), **log)
    return log


def compute_metrics(condition: str, log: dict) -> dict:
    dt = sc.DT
    ee_pos = log["ee_pos"]
    right_ee_pos = log["right_ee_pos"]

    rms_ee_jerk, max_ee_jerk = sc.jerk_metrics(ee_pos, dt)
    rms_int_jerk, max_int_jerk = sc.jerk_metrics(right_ee_pos, dt)

    table_top_z = 0.55
    box_dropped = bool(np.min(log["box_z"]) < table_top_z - 0.01) or \
        bool(np.max(log["box_drift_trans"]) > 0.05)

    min_h = float(np.min(log["h"]))
    min_dist = float(np.min(log["dist"]))
    collision = bool(np.any(log["collision"] > 0.5))

    # intrusion-depth proxy: EE-EE distance (rest -> deepest reach).
    ee_gap = np.linalg.norm(ee_pos - right_ee_pos, axis=1)
    home_gap = float(ee_gap[0])
    min_gap = float(np.min(ee_gap))

    # intruder goal-tracking during the ramp-in+hold window (t in [T_IN1, T_HOLD_END]):
    # how close it gets to its OWN nominal (scripted) target despite the filter.
    hold_mask = (log["t"] >= sc.T_IN1) & (log["t"] <= sc.T_HOLD_END)
    if np.any(hold_mask):
        track_err_mean = float(np.mean(log["int_track_err"][hold_mask]))
        track_err_max = float(np.max(log["int_track_err"][hold_mask]))
    else:
        track_err_mean = track_err_max = float("nan")

    # crude oscillation/chatter proxy: sign changes of hdot per second during hold.
    hdot_hold = log["hdot"][hold_mask]
    n_sign_changes = int(np.sum(np.diff(np.sign(hdot_hold)) != 0)) if len(hdot_hold) > 1 else 0
    hold_dur = float(np.sum(hold_mask)) * dt
    hdot_sign_changes_per_s = n_sign_changes / hold_dur if hold_dur > 0 else 0.0

    qp_infeasible_frac_ego = float(np.mean(1.0 - log["qp_feasible_ego"]))
    qp_infeasible_frac_int = float(np.mean(1.0 - log["qp_feasible_int"]))
    # 0.0 for every step unless ego_relax_on_infeasible=True was passed to
    # run_condition (A/B/C's default path never touches this array).
    ego_relax_used_frac = float(np.mean(log["ego_relax_iters"] > 0))
    ego_relax_iters_mean = float(np.mean(log["ego_relax_iters"]))

    return dict(
        condition=condition,
        min_h=min_h, min_dist=min_dist, collision=collision,
        box_dropped=box_dropped,
        box_drift_trans_max=float(np.max(log["box_drift_trans"])),
        box_drift_rot_max_deg=float(np.max(log["box_drift_rot"]) * 180.0 / np.pi),
        rms_ee_jerk_ego=rms_ee_jerk, max_ee_jerk_ego=max_ee_jerk,
        rms_ee_jerk_int=rms_int_jerk, max_ee_jerk_int=max_int_jerk,
        home_ee_gap=home_gap, min_ee_gap=min_gap,
        int_track_err_mean=track_err_mean, int_track_err_max=track_err_max,
        hdot_sign_changes_per_s=hdot_sign_changes_per_s,
        qp_infeasible_frac_ego=qp_infeasible_frac_ego,
        qp_infeasible_frac_int=qp_infeasible_frac_int,
        ego_relax_used_frac=ego_relax_used_frac,
        ego_relax_iters_mean=ego_relax_iters_mean,
    )


def main():
    summary = {}
    for condition in CONDITIONS:
        print(f"=== running v4-{condition} ===")
        log = run_condition(condition)
        m = compute_metrics(condition, log)
        summary[condition] = m
        print(json.dumps(m, indent=2))

    # progress relative to the adversarial baseline's achieved intrusion depth
    home = summary["A"]["home_ee_gap"]
    adv_min = summary["A"]["min_ee_gap"]
    denom = max(home - adv_min, 1e-6)
    for c in CONDITIONS:
        prog = (home - summary[c]["min_ee_gap"]) / denom
        summary[c]["intruder_progress_frac_vs_A"] = float(prog)

    with open(os.path.join(LOG_DIR, "v4_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY TABLE ===")
    keys = list(summary["A"].keys())
    for k in keys:
        if k == "condition":
            continue
        row = "  ".join(f"{c}={summary[c][k]!s:>10}" for c in CONDITIONS)
        print(f"{k:28s}  {row}")


if __name__ == "__main__":
    main()
