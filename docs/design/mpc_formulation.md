# MPC formulation — linearization & cost (v7 / JointMPCv7)

> Status (2026-11-05): SUPERSEDED: mpc7 / JointMPCv7 is DEFUNCT (AGENTS.md); the controller is FL-MPC in world_model/v2/flmpc/. HISTORY only.

Review doc for the cooperative MPC-HOCBF (`experiments/ab_smooth_vs_bangbang/mpc7.py`,
`JointMPCv7`). Documents exactly how the (now true-dynamics) prediction enters
the QP cost + constraints. Companion to `MASTER.md` §4/§4a. Last updated 2026-09-08.

Symbols: `N_V = 6` joints/arm; `N_C` control blocks; `L` steps/block;
`N_P = N_C·L` prediction steps; `Δt = 0.005 s`. Two arms: ego = cup-holding
(left), int = collecting (right). Decision variable stacks both arms.

---

## 0. What changed (and why this doc exists)

The original v7 predicted the horizon with a **single frozen** `M⁻¹`/`bias` at
t=0 and a free-fall Euler rollout. Review (2026-09-08) showed that model was
wrong two ways: (a) the hand-rolled `M⁻¹(τ−bias)` Euler **omits MuJoCo's passive
(joint damping/spring) and constraint (cup-weld) forces**, so even a free-fall
nominal was ~716 mrad off the true trajectory; (b) linearizing about **free-fall**
is the wrong operating point for a controlled arm. Fixes:

- **Nominal via `mj_step`** (the TRUE evolution, incl. passive+constraint forces),
  rolled under the warm-start torque `τ_nom` (previous solution). Measured nominal
  error **716 → 0.00 mrad**.
- **Time-varying sensitivity**: `Φ` built from per-step `M(q_t)⁻¹` along that real
  trajectory (not one frozen `M⁻¹`).
- **Warm-start linearization**: linearize about `x_nom = τ_nom` (see §1).

Remaining validated limit: the linear prediction is accurate only on the **near
horizon** (36 mrad @20 ms → 811 @80 ms → 3409 @160 ms; open-loop response doubles
~every 40 ms), which is why the horizon is being shortened (see MASTER §4a / the
horizon sweep).

---

## 1. Decision variable & linearized state prediction

Decision variable (per-block joint torques, both arms):

    x = [ τ_e^(0) … τ_e^(N_C-1) , τ_i^(0) … τ_i^(N_C-1) ] ∈ ℝ^(2·N_C·N_V)

Predicted trajectory of each arm (the linearization):

    q(t)  = q_eff(t)  + Φ_q(t)·x
    q̇(t)  = q̇_eff(t)  + Φ_q̇(t)·x

- q_nom(t), q̇_nom(t): nominal rolled with TRUE MuJoCo dynamics (`mj_step`) under
  the warm-start torque τ_nom — EXACT (not linear).
- Φ_q(t), Φ_q̇(t) ∈ ℝ^(N_V × N_C·N_V): control→state sensitivity ∂q(t)/∂x.
- Warm-start shift so x is absolute torque but linearized about x_nom:

      q_eff(t)  = q_nom(t)  − Φ_q(t)·x_nom ,   x_nom = [τ_nom,…,τ_nom]
      q̇_eff(t)  = q̇_nom(t)  − Φ_q̇(t)·x_nom

  ⇒  q(t) = q_nom(t) + Φ_q(t)·(x − x_nom)   (standard linearization about x_nom).

Code: `JointMPCv7._rollout_arm` (mj_step nominal + per-step M_t), the shift +
Φ-build in `_sensitivity_core`, x_nom tiling in `_assemble`.

---

## 2. Sensitivity Φ — the time-varying linearization

Per-step acceleration sensitivity to torque (M now VARIES along the trajectory):

    ∂q̈_t/∂τ = M(q_t)⁻¹ =: M⁻¹_t

For control block b active on steps [b·L, (b+1)·L), double-integrate (the loop in
`_sensitivity_core`; qacc = M⁻¹_t inside the block, 0 outside):

    q̇_loc_{t+1} = q̇_loc_t + M⁻¹_t·Δt
    q_loc_{t+1} = q_loc_t + q̇_loc_t·Δt + ½·M⁻¹_t·Δt²

    Φ_q(t+1)[:, b·N_V:(b+1)·N_V]  = q_loc_{t+1}
    Φ_q̇(t+1)[:, b·N_V:(b+1)·N_V]  = q̇_loc_{t+1}

M⁻¹_t is the arm's own dofadr submatrix of the full mass matrix, plus a
**t=0-frozen payload reflection M_extra** for the ego arm (the cup's full coupling
lives only in the mj_step nominal, not in Φ — an approximation in the sensitivity).

---

## 3. Cost → QP (where the linearization enters)

Cost (per arm p=pos, v=vel; ego EE-hold; jerk across blocks; effort):

    J = Σ_t [ W_p·‖q(t) − q_ref(t)‖²  +  W_v·‖q̇(t) − q̇_ref(t)‖² ]
      + W_ee · Σ_t ‖ J_ee·(q_e(t) − q0_left) ‖²
      + W_j  · ‖Δ_b τ‖²      (block-to-block jerk)
      + R    · ‖x‖²          (effort)

ego ref = fixed hold q0_left; int ref = moving track q_int_des_traj(t).

Substitute q(t) = q_eff(t) + Φ_q(t)·x. For each term define

    A_t = Φ_q(t)                    (sensitivity)
    b_t = q_ref(t) − q_eff(t)       (error to close)

then  W_p·‖A_t·x − b_t‖² = W_p·( xᵀA_tᵀA_t x − 2 b_tᵀA_t x + b_tᵀb_t ).

Summing over t gives the box-QP   min_x  ½ xᵀ H x + fᵀ x   with

    H = 2 Σ_t ( W_p·A_tᵀA_t + W_v·A^v_tᵀA^v_t + W_ee·A^ee_tᵀA^ee_t ) + H_jerk + 2R·I
    f = −2 Σ_t ( W_p·A_tᵀb_t + W_v·A^v_tᵀb^v_t + W_ee·A^ee_tᵀb^ee_t ) + f_jerk

Code mapping (`_assemble`):

    Ae  = Phq_e[t] ,  be  = q0_left − qf_e[t]           # ego pos
    Aev = Phqd_e[t],  bev = −qdf_e[t]                    # ego vel (toward 0)
    Aee = Jp_ee@Phq_e[t], bee = −(Jp_ee@(qf_e[t]−q0_left))  # ego EE hold
    Ai  = Phq_i[t] ,  bi  = q_int_des_traj[t] − qf_i[t]  # int pos (moving ref)
    Aiv = Phqd_i[t],  biv = qd_int_des_traj[t] − qdf_i[t]# int vel
    H[..]+=2*W*A.T@A ;  f[..]+=-2*W*A.T@b
(qf_e/qf_i here are the q_eff returned by _sensitivity_core.)

Priority weights W_p, W_j, W_ee for the ego are grasp-fragility-scaled (FIX 2);
the int weights are fixed. Effort R and box |τ|≤TAU_MAX unchanged.

---

## 4. Collision & EE-speed constraints (also linearized)

**Arm–arm HOCBF**, first-order in the predicted states, with c_e = nᵀJ_e,
c_i = nᵀJ_i (barrier gradient), evaluated at t=0 (constant n, J over the horizon):

    h(t) ≈ h0 + c_e·(q_e(t) − q0_e) − c_i·(q_i(t) − q0_i) ≥ 0

Substitute q(t) = q_eff + Φ x  ⇒  linear row  aᵀx ≤ b :

    −(c_e·Φ_q^e(t))·x_e + (c_i·Φ_q^i(t))·x_i
        ≤  h0 + c_e·(q_eff_e(t) − q0_e) − c_i·(q_eff_i(t) − q0_i)

(the intruder carries an extra static margin in v4-C; in v7 the same pairs feed
the row). Applied at every one of the N_P predicted steps (`_coll_steps`).

**EE-speed cap** (`ee_speed_limit.ee_velocity_cap_rows`, FIX 3): single-step, on
the applied (first-block) torque, |v_next_axis| ≤ v_max with
v_next = J_ee·(q̇0 + Δt·M⁻¹(τ − bias)), affine in τ → 6 rows.

**Table floor** (`sim_common.table_plane_barrier_rows`): static half-space HOCBF
z_kp ≥ z_table − margin, per arm keypoint (incl. fingertip keypoints), embedded on
the first-block torque.

---

## 5. Honest caveats (for review)

1. State prediction is **first-order in x** (Φ from M⁻¹_t); valid only near-horizon
   → horizon must be short (open-loop divergence). Being tuned via a horizon sweep.
2. **Barrier linearized to first order** (constant n, J from t=0; no ṅ over the
   horizon) — another reason long horizons hurt.
3. M⁻¹_t sensitivity uses a **t=0-frozen payload reflection** for the ego arm
   (cup coupling exact only in the mj_step nominal).
4. Nominal rollout costs N_P `mj_step` calls per arm per control step (perf, not
   correctness — mitigated by the shorter horizon).
5. This gives a **short-horizon, feedback-relinearized** MPC (re-solves every Δt,
   applies first block). It is NOT a long-horizon global planner; the foresight is
   bounded by where the linearization holds.
