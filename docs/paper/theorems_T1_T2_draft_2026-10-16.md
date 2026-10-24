# T1 (soundness) and T2 (monotonicity) — draft, 2026-10-16

Scope: the base-carry gatekeeper as implemented in `world_model/v2/house/gatekeeper/` (Checkers 1+2), shared autonomy. Arm motion,
grasp loads (Checker 3) and probes (Checker 4) come later; the statements are written so they extend.
Status: proof sketches. Each one names the code row it relies on. Items marked **GAP** are places where code and proof disagree today.

## 0. Setting

- **Gatekeeper steps** are k = 0, 1, … at times t_k = kΔ (Δ = 0.1 s, K = 5 ticks). Trajectories are sampled at h = 0.05 s.
- **Robot state** is x = (p, v): planar base pose p = (x, y, θ) and body twist v = (v_x, v_y, ω). The robot footprint at pose p is
  F(p) = p ⊕ R(θ)·Rect.
- **Belief** b_k = (P_k, M_k, {H_i,k}), all SETS:
  - pose box P_k = {p : |p_xy − p̂_xy| ≤ e_xy, |θ − θ̂| ≤ e_θ};
  - friction interval M_k = [μ_lo, μ_hi];
  - for each tracked person i, the set H_i,k of initial states (position within e_p of c_i, velocity within e_v of v̂_i).
- **Reachable set.** Under motion model A2, H_i,k induces R_i,k(t) ⊆ ℝ², the positions person i can occupy at t_k + t:
  - speed model: disc(c_i, r0 + v_p·(t + ℓ));
  - accel model: the intersection of that disc and disc(c_i + v̂_i(t + ℓ), r0 + e_v(t + ℓ) + ½·a_p(t + ℓ)²);
  - here r0 = PERSON_CLEAR_R + e_p and ℓ = latency.
- **Candidate** τ = nominal (u_op held for T_S, cup-safe ramp) followed by backup (cup-safe stop). i_rest is the first sample with v ≡ 0.
- **Admission:** Adm(τ, b) ⇔ C1(τ, b) ∧ C2(τ, b), the implemented Checkers 1 and 2 (README §Checker 1/2), all margins ≥ 0.

**Safety requirements** for the true world w:
- **S1 (static):** min over the boundary of F(p(t)) of EDT ≥ d_wall − ε for all t.
- **S2 (passive people safety):** for all t with v(t) ≠ 0, dist(F(p(t)), person_i(t)) ≥ PERSON_CLEAR_R + d_person − ε. Contact with a person is
  possible only while the robot is at rest, which is the standard passive-safety notion (Bouraine et al. 2012). It is weaker than "never touched".
- **S3 (cup):** the cup does not slip on the tray: |a_cup(t)| ≤ μ_true·g (flat tray; tilt enters through the measured tray axis, as in the shield).

## 1. Assumptions (each one is named in GKConfig / ShieldCfg or listed as open)

- **A1 (belief soundness, at commit times):** at every t_k, the true pose is in P_k, μ_true ∈ M_k, every true person j within range R_max
  is tracked, and its state is in H_i(j),k.
  - The sweep audits this (n_belief_miss, n_untracked_near; truth is used for metrics only). OPEN: unseen and occluded people.
- **A2 (person motion):** |ṗ_j| ≤ v_p,max, and for the accel model also |p̈_j| ≤ a_p. Values: 1.5 m/s and 2 m/s², both ASSUMPTIONS.
- **A3 (execution):** the base realizes the committed twist profile with footprint deviation ≤ ε, where ε ≤ min(d_wall, d_person).
  Discretization is covered separately: fp_step/2 for the boundary sampling and the sweep term for inter-sample motion.
- **A4 (contact model):** Coulomb static friction between cup and tray; no slip if |a_cup| ≤ μ·g. The cup offset r is known.
- **A5 (map):** the EDT is exact up to the loc_margin folded into e_xy.
- **A6 (initialization):** at t_0 the robot is at rest and τ_rest (at rest from t = 0) satisfies Adm(τ_rest, b_0).
- **A7 (hardware):** committed profiles respect |a| ≤ A_HW and |v| ≤ V_STICK. Checker 1's hw rows enforce this.

## 2. Lemma 1 (one-shot soundness of the checks)

**Statement.** If Adm(τ, b_k) holds and A1–A5 and A7 hold at t_k, then executing τ from t_k satisfies S1–S3 on [t_k, ∞) for every world consistent
with b_k. This includes the rest phase, where S2 is vacuous because v = 0.

**Proof sketch, row by row.**
- **Wall.** EDT is 1-Lipschitz. The worst pose in P_k moves a boundary point q by at most e_xy + |q|·e_θ. Between samples the boundary
  moves at most ½h·max(|v_xy| + |ω|·max|q|), the sweep term. Boundary sampling error is at most fp_step/2. Hence true clearance ≥ the computed
  margin + d_wall − ε ≥ d_wall − ε.
- **Person (moving samples).** By A1 and A2 the true person lies in R_i,k(t). The robot is clear of the intersection whenever it is clear of
  EITHER disc, so the margin is max(m_speed, m_accel). The sweep term bounds the inter-sample motion of both the footprint and the disc boundary.
  S2 follows.
- **Cup.** Within one planner segment the body acceleration is constant, so ω(t) and v(t) are affine in t and a_cup(t) = a_b + α×r +
  ω×(v + ω×r) is a vector QUADRATIC in t.
  - The chord bound gives max over the segment of |a_cup| ≤ max(|a_cup(0)|, |a_cup(h)|) + (h²/8)·sup|a_cup''|, with
    |a_cup''| ≤ 2|α||a_b| + 2α²|r| in the planar case.
  - Code-limit values (|α| ≤ 2 rad/s², |a_b| ≤ 1.4 m/s², |r| ≤ 0.4 m, h = 0.05) give a correction of ≤ 2.8e-3 m/s².
  - That is far inside the 10 % slack of cup_margin = 0.9 (≈ 0.14 m/s²). With μ_lo ≤ μ_true (A1), |a_cup| ≤ 0.9·κ·μ_lo·g < μ_true·g, so S3 holds.
  - **GAP-1:** the code checks only segment ends. Either add the chord term explicitly, or keep cup_margin ≤ 1 − (h²/8)·sup|a''|/a_lim and
    state it in GKConfig.
- **Checker 2 static and rest rows** cover t ≥ t_rest. The passive row is not needed for S2, which only constrains motion. It is needed for
  progress (T3/T4), because a person reaching the robot before it stops would put the robot in motion inside a person disc. Here t_touch is the
  earliest time any disc can touch the stopped footprint, and the condition is t_rest ≤ t_touch. ∎

## 3. Theorem T1 (closed-loop soundness and an always-available backup)

**Statement.** Under A1 (at every commit time), A2–A7, the gatekeeper loop with the KEEP rule below satisfies S1–S3 for all t ≥ t_0. At every
time it is executing a trajectory that ends at rest in the safe-rest set T(b), so a certified backup always exists. Safety never requires
regime 2 unless A1 or A2 is falsified.

> **USER DECISION 2026-10-24: assume NO Shield exists (for now).** Regime 2 (hand-off to the reactive Shield) is not part of the system.
> - After the monitor fires (A1 or A2 falsified, e.g. a new person appears within reach mid-segment), T1 claims nothing, and nothing
>   uncertified may take over.
> - The response after falsification must therefore be defined inside the certified framework. Proposal, not built: re-check the committed
>   backup under the current belief; else search the backup set under the current belief; else a counted, reported "no certified action"
>   event (A1 coverage failure), executing the cup-safe brake.
> - The Shield stays only as an experimental baseline, never as a safety fallback.
(2026-10-22 wording fix: T(b) was called "task-ready", but Checker 2's rows (rest, static, passive person) only certify SAFE rest. Whether
the robot can continue from the rest pose is the separate ranking term C2-ready, `theorems_T3_T4_draft` §1b. T1 needs only safe rest.)

**KEEP rule required for T1.** If no candidate passes at step k, continue the committed τ_j. Do NOT re-certify τ_j under b_k. Instead run the
falsification monitor: each person's new observation must be consistent with the predicted set R_i,j(t_k − t_j), and the pose with P_j
propagated. Regime 2 fires only when that monitor fails.

**Proof (induction on k).**
- **Invariant I_k:** the trajectory executed on [t_k, t_k+1) is a suffix of some τ_j (j ≤ k) with Adm(τ_j, b_j), and the truth satisfied A1 at t_j.
- **Base:** A6 gives I_0.
- **Step:** at t_k+1 either
  - (a) a new τ passes Adm(·, b_k+1), and A1 at t_k+1 gives I_k+1; or
  - (b) we keep τ_j. Lemma 1 applied at t_j already covers [t_j, ∞) for every world consistent with b_j, and A2 propagates b_j's people sets
    over time, so the suffix stays safe. I_k+1 holds with the same j.
- Every τ_j ends at rest in T(b_j) (C2), so a backup is always being executed or available.
- Regime 2 fires only if the monitor detects an observation outside the predicted set, i.e. A1 or A2 is false. ∎

**GAP-2 (code ≠ T1).** gate.py's keep branch currently RE-CERTIFIES τ_j under the CURRENT belief b_k with falsify_tol.
- b_k is re-anchored on the new observation and is not intersected with the prediction from b_j, so it can be wider in places.
- Re-certification can therefore fail while the truth is still safe. The code then hands control to the Shield (regime 2), which has no
  certificate. That is conservative for progress and, worse, it weakens the guarantee.
- Fix (small): implement the KEEP rule above, i.e. a prediction-consistency monitor instead of re-certification. Count monitor failures as n_falsified.

**Remarks.**
- T1 is a set-membership / passive-safety version of the gatekeeper theorem (Agrawal & Panagou, T-RO 2024), with Bertsekas–Rhodes (1971)
  set-membership beliefs.
- What is new is the belief-wide cup row (S3 via μ_lo and the chord bound) and the explicit falsification monitor. What is not new is the
  induction pattern.
- Observations only enter through A1: T1 is conditional on the observation-error bounds (e_p, e_v, the conformal radius). The sweep's audit
  counters measure how often that condition fails in sim.

## 4. Theorem T2 (monotonicity), restated

Belief order: b ⊑ b' if P ⊆ P', M ⊆ M' (μ_lo' ≤ μ_lo), and for every person i, H_i ⊆ H_i' (e_p ≤ e_p', e_v ≤ e_v', a_p ≤ a_p', v_p ≤ v_p').

- **T2(a) (antitone admission).** For a FIXED trajectory τ, b ⊑ b' and Adm(τ, b') imply Adm(τ, b).
  - **Proof:** each margin is a closed-form lower bound that is non-increasing in every set-size parameter.
    - Wall: −e_xy − |q|·e_θ.
    - Person: radii increasing in e_p, e_v, a_p, v_p. The max of two discs' margins is monotone when both are.
    - Cup: a_lim ∝ μ_lo.
    - Passive: t_touch decreasing in all radii.
  - Property-tested (tests/test_gatekeeper.py, randomized). ∎
  - **Caveat:** latency ℓ is NOT a set-inclusion parameter in the accel model when v̂ ≠ 0, because it moves the disc centre. An older
    observation is a different belief, not a larger one. The speed model is monotone in ℓ.
- **Corollary (VLM rule).** A proposer, e.g. a VLM, that may only ENLARGE sets (widen μ, mass or e_p) can never make an inadmissible plan
  admissible. It affects completeness only. This is the formal basis of "VLM proposes, evidence narrows, certificate admits".
- **T2(b) (persistence under parameter learning).** If parameter sets are updated only by intersection with evidence, Θ_k+1 = Θ_k ∩ Θ_obs ≠ ∅
  (friction from a probe, mass from a lift), then every (segment, backup) admitted at step k stays admitted with respect to its parameter
  components at all later steps.
  - Viability-library entries therefore never need re-certification because of LEARNING. They are invalidated only by FALSIFICATION
    (Θ_k ∩ Θ_obs = ∅), which switches to regime 2.
  - **Proof:** Θ_k+1 ⊆ Θ_k, then T2(a). ∎
- **T2(c) (what is NOT monotone).** STATE sets grow with time and with occlusion: people leave view, pose drift accumulates.
  - A library entry certified under state precondition X is reusable at step k iff the current state belief ⊆ X. Lookup is therefore a
    containment test.
  - Occlusion-driven growth is exactly what T3 must count, with re-observation as a recovery skill ("look" back into the library).

## 4b. Status update (2026-10-16, later)
- **GAP-1 closed (opt-in `cup_chord=True`).**
  - The chord term is subtracted in checks.py.
  - Tests: the margin is reduced by exactly the bound, and the bound is sound over 300 random segments.
- **GAP-2 closed (opt-in `keep_rule="monitor"`, gatekeeper/monitor.py).**
  - The commit-time certificate is kept.
  - Each person's observation disc must meet the predicted reach set, and the pose must stay in the propagated box.
  - Unmatched persons count only if they could reach the remaining motion.
  - Tests: the monitor keeps a still-safe plan where re-certification falls to regime 2, and it flags teleports and pose jumps.
- **Set-membership filter (opt-in `belief_filter="intersect"`).**
  - Lens enclosing disc; disjoint observation and prediction counts n_obs_inconsistent.
  - The owner verified the closed form: chord-diameter disc when both caps are minor, otherwise the containing disc; sound.
  - Empirically it shrinks a standing person's conformal disc by only about 1%, because the observations are near-concentric.
- **Tracker runs:** one of the two people is untracked near the robot (n_untracked_near ≫ 0). A1 fails because of sensing coverage, so the
  remedy is LOOK, not filtering.

## 5. Checks still to do

1. GAP-1 cup chord term: make it explicit in the code (one line) or assert cup_margin against it in GKConfig.
2. GAP-2 KEEP rule: replace re-certification with the prediction-consistency monitor, then re-run the sweep; n_falsified should drop to the
   monitor's true positives.
3. A1 audit from the tracker sweep: n_belief_miss and n_untracked_near per episode. T1 is only as good as these numbers.
4. Unseen people: either a sensor-coverage assumption (every person within R_max is tracked) or occlusion regions as phantom-person sets.
   The second is the honest POMDP version and connects to T3.
5. Then T3 (finite completion bound with counted recoveries) and T4 (existence of recovery), built on the always-backup invariant above.
