# Smooth, Grasp-Aware Counter-Planning for Bimanual Manipulation

> Status (2026-11-05): SUPERSEDED (Sept 2026 Piper-era method; superseded by MASTER.md, itself superseded by docs/paper/claims_2026-11-02.md). HISTORY only.

> **EMPIRICAL ARC (the paper spine, as of 2026-09-06, on the faithful proper-YOR 0.185 m geometry):**
> 1. **Naive single-arm avoidance under an adversarial intruder → ALL fail.** On the real tight geometry, bang-bang, myopic HOCBF-QP, and geometric fabric *all* either collide or shed the grasp (v3, verified with a 0%-false-safe collision proxy). The collision-vs-grasp-retention trade-off is real and unsolved by reactive single-arm control.
> 2. **Exploiting the cooperative structure dissolves it.** Because we control *both* arms, the free (non-grasping) arm can also yield. With **ego priority**, condition C reaches the collision-free + grasp-retained + smooth quadrant nothing hit before (min_h +0.016, box drift 1.6 cm, ego jerk 25, intruder still 70% goal progress) (v4).
> 3. **The contribution is a *principled explicit priority* in decentralized cooperation.** Decentralized-reactive cooperation with a hand-tuned priority margin works (v4-C) but the margin is magic. A **joint/centralized grasp-aware MPC was tried (v5) and did NOT help** — 17–49 ms/step (not real-time) and priority did *not* emerge from the grasp constraint (the optimizer moved the ego, not the intruder; feasibility needed cost-tuning anyway). **Lesson: cooperative priority does not emerge for free from joint optimization — it must be stated explicitly.** So the contribution is a **principled explicit authority law** (grasp-state-driven right-of-way: the arm holding a fragile object yields least, scaled by grasp fragility) inside the real-time decentralized scheme — NOT a joint MPC. This still reframes the field's decentralized-adversarial assumption (NeHMO/HaMMAR) as the *wrong model* for a robot whose arms you jointly control.



**Status:** design draft · **Last updated:** 2026-09-05 · **Owner:** ss5772
**Platform:** YOR dual AgileX Piper (2× 6-DoF + gripper), MuJoCo sim → hardware
**One line:** Replace the jittery, last-moment, bang-bang collision-avoidance fallback of neural-HJ multi-arm planners with a *smooth, task-aligned, grasp-preserving* controller built from **MPC + High-Order CBFs + an HJ-reachability terminal safe set**, with grasp/closed-chain dynamics actually modeled.

---

## 1. Problem & gap

Two arms share a workspace. Each may be **holding an object** (single grasp, or a bimanual closed chain). We want each arm to reach its goal, avoid arm–arm (and arm–environment) collision, and **retain task success** — i.e. keep the grasp/object constraint satisfied — with **smooth** motion.

The recent neural-HJ-reachability line solves the *geometry* of this but leaves three gaps **its own authors state explicitly**:

- **NeHMO** (arXiv:2507.13940) — decentralized neural-HJR multi-arm avoidance. Adversarial/worst-case → conservative; deadlocks ("stuck when goals are close"). No grasp/manifold concept.
- **HaMMAR** (arXiv:2511.03591v2) — HJR on a task **manifold** (cup-upright / object-carrying). Runtime = learned-HJ-value → receding-horizon IPOPT traj-opt with a `V > ε` constraint, and a **discontinuous fallback to bang-bang HJ extremal control** `u* = −ū·P(x)∇V/‖P(x)∇V‖` when the solve fails. Admissions:
  - *"applying the fail-safe control … can also lead to **jittery motions**."*
  - Manifold enforced **only at planning-horizon endpoints** → *"may **drift off the constraint manifold** … segments between planning updates are not explicitly constrained."*
  - Proposed-but-**unbuilt** fix: *"integrating a **high-frequency low-level controller** with the planner."*
  - **Grasp dynamics not modeled**: kinematic velocity model `ẋ = u`; held object faked via IK matching; no mass/inertia/closed chain; **no grasp-stability / slip / spill metrics**.
- **Latent Safety Filters** (arXiv:2502.00935) — HJ reachability in a world-model latent space for un-hand-specifiable failures. Override is a **hard switch** `a = π_task if V>ε else π_safe`. Admission: *"a single control action … **maximally and at the last moment** … align them with the base policy's task performance should be explored."*

**The open problem we claim:** counter-planning that (1) avoids arm–arm collision, (2) holds the grasp/task manifold **continuously** (not just at horizon endpoints), and (3) is **smooth** (no bang-bang jitter), with grasp/closed-chain **dynamics modeled** in the safety certificate.

---

## 2. Contribution (honest positioning)

MPC-CBF (Zeng–Zhang–Sreenath 2021), HOCBF (Xiao–Belta 2019/2021), and reachability-terminal-set MPC (Herbert/Bansal/Tomlin) **already exist**. *Using MPC-HOCBF is not itself the novelty and must not be framed as such.* The contribution is the **combination on this problem**:

1. **HOCBF driven by modeled grasp/closed-chain dynamics.** Because we model the payload (second-order dynamics, torque/acceleration control), position-level barriers have **relative degree ≥ 2** — HOCBF is *required*, not decorative. This is exactly the modeling axis the HJ papers skip.
2. **A two-layer smooth architecture:** slow **reachability-MPC** (HJ value as terminal safe set + grasp manifold as per-stage constraint) feeding a **high-rate HOCBF-QP** safety filter. That fast inner loop *is* the "high-frequency low-level controller" HaMMAR named and never built.
3. **Smoothness & grasp stability as first-class, measured objectives** (jerk, SPARC, grasp force/slip, object-pose drift) — the HJ papers report none.
4. **Validation on cheap bimanual hardware** (dual-Piper) across single-arm-with-intrusion, bimanual handover, and crossing-path tasks.
5. *(Stretch)* **Fuse geometric + latent semantic safety** — HOCBF collision safety composed with a latent task-failure filter.

**Competitive risk (explicit):** the same lab already flagged "high-frequency low-level controller" as future work — they may be building the smoothing. Our insulation is the grasp-**dynamics** modeling, measured smoothness/grasp-stability, and hardware — the axes they show no interest in. Lead with those; do **not** lead with "we use MPC-HOCBF."

---

## 3. Modeling

### 3.1 Dynamics (the departure from the papers)
Per-arm rigid-body dynamics, stacked for the pair (`q ∈ ℝ¹²`):

```
M(q) q̈ + C(q, q̇) q̇ + g(q) = τ + Σ Jᵢ(q)ᵀ fᵢ
```

- Control input `u = τ` (torque) or `u = q̈` (acceleration / inverse-dynamics interface). Either way, **position/collision constraints are relative degree 2** in `u`.
- **Grasp:** a held object adds mass/inertia to the end-effector and, for a bimanual grasp, a **closed-kinematic-chain equality** coupling the two EE poses through the rigid object. This is modeled, not IK-faked.
- State `x = (q, q̇)` (add object free-body state when relevant).

### 3.2 Constraints
- **Collision barrier** `h_c(q) = sd(cap_L(q), cap_R(q)) − d_min ≥ 0` (signed distance between arm collision capsules; also arm–table, arm–self). Relative degree 2.
- **Grasp / task manifold** `c(x) = 0`:
  - single grasp: EE holds rigid transform to object ⇒ orientation/position constraint;
  - bimanual: closed-chain `T_L(q) = T_object = T_R(q)·T_offset`;
  - "cup-upright" style: `orientation_z(EE) = ẑ`.
  Enforced **continuously** (per-stage in MPC + in the inner QP), directly answering HaMMAR's drift.
- **Input & joint limits:** `u_min ≤ u ≤ u_max`, `q ∈ [q⁻, q⁺]`, `|q̇| ≤ q̇_max`.

### 3.3 HJ reachability value (learned)
Train a DeepReach network `V_θ(t, x_L, x_R)` for the **pairwise avoidance game** (reuse the lab's formulation and, if available, the existing `deepreach_export.py` / `hj_eval` tooling from `ur5e_benchmark`). The zero-superlevel set of `V_θ` is the safe backward-reachable set. We use `V_θ` as a **terminal certificate**, *not* as the myopic controller — avoiding the fragile double-differentiation of a neural barrier.

---

## 4. Method

### 4.1 HOCBF construction (relative degree 2)
With `ψ₀ = h_c`:

```
ψ₁(x) = ψ̇₀ + α₁(ψ₀)
ψ₂(x, u) = ψ̇₁ + α₂(ψ₁) = L_f² h_c + L_g L_f h_c · u + α₁'(·)ψ̇₀ + α₂(ψ₁) ≥ 0
```

`ψ₂` is **affine in `u`** ⇒ a linear inequality usable in both the MPC and the inner QP. `α₁, α₂` are class-𝒦 (tunable; start with linear `αᵢ(s)=γᵢ s`). The grasp manifold gets an analogous HOCBF form `ψ₂^grasp ≥ 0` (or is kept as a soft equality).

### 4.2 Outer loop — reachability-MPC (~10–20 Hz)
Discrete horizon `N`, decision vars `u₀..N₋₁`, states `x₁..N`, slacks `s ≥ 0`:

```
min   Σ_k [ ‖u_k − π_task(x_k)‖²_R          # least-restrictive, task-aligned
          + ‖Δu_k‖²_W + ‖Δ²u_k‖²_J ]        # smoothness / jerk (measured!)
      + ‖x_N − x_goal‖²_P                    # terminal task cost
      + ρ‖s‖²                                # slack penalty
s.t.  x_{k+1} = F(x_k, u_k)                  # 2nd-order dynamics (§3.1)
      ψ₂(x_k, u_k) ≥ −s_k                    # HOCBF collision (soft)
      c(x_k) = 0    (or ψ₂^grasp ≥ 0)        # grasp manifold, every stage
      V_θ(x_N) ≥ ε                           # HJ terminal safe set
      u_min ≤ u_k ≤ u_max, joint/vel limits
      x_0 = x(t)
```

The MPC emits a smooth reference `(x_ref, u_ref)`.

### 4.3 Inner loop — HOCBF-QP safety filter (~500 Hz–1 kHz)
Tracks the reference; guarantees inter-sample collision + grasp safety between MPC updates (kills the drift):

```
u*(t) = argmin_u ‖u − u_ref(t)‖²_R + ρ‖s‖²
        s.t. ψ₂^collision(x, u) ≥ −s,  ψ₂^grasp(x, u) ≥ 0,  u limits
```

Small, convex, fast. This layer is HaMMAR's un-built "high-frequency low-level controller."

### 4.4 Why this is smooth
Safety is a **constraint**, smoothness is in the **objective** (`Δu`, `Δ²u`), and the two layers are continuous — no discontinuous switch into extremal control. Contrast with the papers' bang-bang fallback.

---

## 5. Baselines & metrics

**Baselines**
- **B0** Task policy only (no safety) — task-success ceiling, collision floor.
- **B1** Bang-bang HJ extremal fallback (HaMMAR-style switch) — reproduces the jitter; our headline negative.
- **B2** Myopic HOCBF-QP only (no MPC, no HJ terminal) — reactive, may deadlock.
- **B3** First-order MPC-CBF on the velocity model — reproduces the papers' regime.
- **Ours** Two-layer reachability-MPC + HOCBF with grasp dynamics.

**Metrics**
- Task success rate; time-to-goal.
- Collision rate; min arm–arm distance.
- **Grasp stability:** object drop rate, max fingertip slip, contact-force variance, object-pose drift (spill proxy).
- **Smoothness:** RMS joint & EE jerk, control-rate, **SPARC** (spectral arc length).
- Conservativeness: path-length ratio vs unconstrained; **deadlock rate**.
- Compute: achieved solve time / control rate.

---

## 6. Experiments (the 3 scene tasks we're building)
Single scene, three scripted regimes (see `scenes/scene_bimanual_pickup.xml`):
- **E1 — single-arm pick + intrusion:** arm A grasps center box; arm B reaches into the shared zone. Tests smoothness/grasp-hold while avoiding. *Cleanest Figure-1.*
- **E2 — bimanual handover:** A picks, hands to B (closed chain). Tests continuous manifold enforcement.
- **E3 — both pick, crossing paths:** each arm to its box, paths cross. Tests decentralized-style avoidance without a shared grasp.

**Ablations:** horizon `N`; with/without HJ terminal set; HOCBF vs first-order CBF; hard vs soft manifold; smoothness-weight sweep; adversarial vs cooperative other-arm model.

---

## 7. Risks & mitigations
1. **Real-time** (top risk). NLP-MPC over 12-DoF + payload + HOCBF is heavy (HaMMAR already ~0.1 s/step). → SQP/convex MPC, short horizon (5–15), warm-start, and rely on the two-layer split (expensive layer slow, safety layer fast/convex).
2. **Recursive feasibility.** Hard collision + hard grasp + input limits → infeasible. → soft (slacked) CBF with high penalty; accept loss of *formal* guarantee (the papers already lost it via NN) in exchange for smoothness.
3. **Neural-value gradients** noisy from DeepReach. → use `V_θ` only as terminal constraint (flavor 3), not as differentiated barrier; smooth/distill if needed.
4. **HOCBF tuning** (`αᵢ`). → start linear, grid/tune; report sensitivity.
5. **Manifold over-constraint** vs collision. → soft/projected manifold when the two conflict.
6. **Smoothing is not free** (empirical, real_Cartpole). A jerk-continuous s-curve cleared a guard that bang-bang tripped — but tripped a *different* guard at longer durations. Smoothing shifts *which* constraint binds; it doesn't erase conservatism. → evaluate smoothness against *all* barriers/limits, and report the trade-off honestly; never claim a free win.

---

## 8. Implementation plan (milestones)
- **M0** Scene + actuated grasp in MuJoCo *(**DONE** 2026-09-05)* — `scenes/scene_bimanual_pickup.xml` (derived from YOR; table top z=0.20, 3 boxes, gripper actuation ported from `coney.xml`). Verified grasp: 8 contacts, ~12 N, box lifted 4 cm. `nq=37, nu=16` (6 arm + 2 finger DOF/arm). Position servos: arm ctrl=radians, finger ctrl=meters, `(0,0)`=closed, `(0.035,-0.035)`=open.
- **M1** 2nd-order dynamics interface + IK/teleop pick baseline (B0), no safety.
- **M2** Bang-bang HJ fallback baseline (B1) — reproduce jitter → **Figure 1 negative**. *(**DONE** 2026-09-05 — `experiments/ab_smooth_vs_bangbang/`.)*
- **M3** Inner HOCBF-QP (analytic collision + grasp) — smooth reactive filter (B2). *(**DONE** 2026-09-05.)* **Result:** C2 (smooth) RMS EE jerk **113** vs C1 (bang-bang) **10,682** m/s³ (~95×), torque-rate 17.7 vs 5,499; both collision-free; box-drift settles 0.62 cm (C2) vs 1.72 cm & climbing (C1). Caveats: barrier = analytic sphere-distance proxy (not trained HJ yet); ego mass = 6×6 arm block (payload not yet coupled); C2 rides boundary (min h −0.0035, soft-slack) — the smoothing↔margin trade (§7.6). Needed all-9 sphere-pair constraints (single-pair switch caused a collision — fixed). SPARC unreliable here (C0 barely moves); trust RMS/max jerk + torque-rate.
- **M4** Train DeepReach `V_θ` for the Piper pair (**offload to rlab via the slurm gateway**); add HJ terminal-set MPC (outer loop).
- **M5** Full two-layer; run E1–E3; metrics + ablations.
- **M6** *(stretch)* fuse latent semantic safety; sim→real on dual-Piper.

**Compute:** MuJoCo + control dev local on `westeros` (8× 2080 Ti); DeepReach training offloaded to `rlab` via `scripts/offload/submit_via_slurm_gateway.sh`; episode/data archival via the ported Box offload (`docs/offload.md`).

---

## 9. Related work (positioning)
- **Neural HJR multi-arm:** NeHMO (2507.13940), HaMMAR (2511.03591) — we reuse the learned value as a *terminal set*, replace bang-bang fallback with smooth MPC+HOCBF, model grasp dynamics they omit.
- **Latent safety:** Latent Safety Filters (2502.00935) — candidate semantic-safety layer for M6; DINO-WM world model, 15 Hz filter recipe.
- **CBF/MPC:** HOCBF (Xiao–Belta), MPC-CBF (Zeng et al.), reachability-terminal MPC (Herbert/Bansal/Tomlin) — building blocks, not contributions.

## 10. Open questions
- Cooperative (shared-state) vs adversarial other-arm model for bimanual — is the game-theoretic worst-case even the right prior when we control both arms? (Likely relax to shared-intent for E1/E2.)
- Does modeling grasp dynamics measurably change the reachable set vs the kinematic model, enough to matter?
- Can the HJ terminal set be trained once and reused across the 3 tasks, or is it task/manifold-specific?

---

## 11. Reusable assets & adopted conventions (harvested from `real_Cartpole` + `ur5e_benchmark`)

The two halves of this method already exist in-house and can be combined; **neither repo has arm–arm collision code** — that (and grasp-dynamics modeling) is the novel part we build.

**Half A — CBF / QP / smoothing (`real_Cartpole/controller_core/`):**
- `manipulability_cbf.py` — working **HOCBF** via `q̈ = M⁻¹(τ−bias)` (the exact relative-degree-2 pattern §4.1); analytic gradient + directional curvature. *Adapt:* 6→12 DoF, add an inter-arm-distance barrier row.
- `box_qp.py`, `constrained_box_qp.py` (+Numba) — generic DOF-agnostic box-QP, ~0.3–1 ms. **Import as-is** → inner HOCBF-QP core (§4.3).
- `x_task_yz_corridor_qp/controller.py` — pattern for stacking multiple HOCBF rows (task+corridor+manip+orientation) in one QP. *Adapt:* dual-arm torque var + cross-arm barrier.
- `filters.py::TorqueCommandFilter` — low-pass + slew-rate limiter. **Import as-is** → smoothness primitive.
- `damped_least_squares.py` + residual-gated warm-start QP (unmerged `hardware-corridor-qp` branch) — smooth without silently degrading tracking.

**Half B — HJ / DeepReach / eval / counter-planning (`ur5e_benchmark/src/ur5e_benchmark/`):**
- `grasp_boundary/deepreach_export.py` — DeepReach training-data format + deterministic SHA1 splits. *Adapt:* double state per-arm, add inter-arm proximity/contact features.
- `hj_eval/` (`hj_inference.py`, `dataset_reset.py`, `replay_analysis.py`) — value-function runtime query + exact-state replay eval + metric aggregation.
- `planning/receding_horizon_search.py`, `hj_planning/receding_horizon.py`, `steering/rollout_engine.py` — **sampling-based counter-planner** (snapshot→sample→rollout→score-with-HJ→restore→execute). Serves as baseline **B3′** and a fallback if NLP-MPC misses real-time.
- `safety_models/interface.py::SafetyValueModel` — clean seam to plug our bimanual value net.
- Adopt wholesale for hygiene: `benchmark/perturbations.py`, `episode_schema.py`, `artifact_keys.py` (SSOT), `artifacts/registry.py`, `cluster/slurm.py`.

**Adopted conventions / lessons:**
- **Rates:** task/policy ~20 Hz (dt=0.05), ~240-step episodes; inner HOCBF-QP ~500 Hz–1 kHz. Keep the two-rate split explicit.
- **Gripper sign convention:** ur5e hit a +1=close vs −1=close gotcha. Our scene: finger ctrl in **meters**, `(0,0)`=closed, `(0.035,−0.035)`=open. **Fix one convention for both grippers and document it** (done in M0).
- **Headless render:** `MUJOCO_GL=egl` works here (both smoke tests); ur5e uses `glfw`+`xvfb-run` and marks EGL legacy — keep EGL unless it breaks.
- **Units:** world-frame meters, RPY radians throughout logs.
- **Validation discipline:** define a *disqualifying* ground-truth failure case BEFORE comparing candidates; a design that wins on average but misses the one case that matters is rejected. Measure bang-bang vs jerk-continuous against the *same* guard.
- **Path hygiene:** adopt a `paths.py` root resolver with env overrides from day one (ur5e accrued 1500+ hardcoded paths without it).
- **Cluster gotchas:** `OPENBLAS_NUM_THREADS=1` per worker (BLAS thread explosion); watch per-user cgroup silent OOM; CPU-only Slurm jobs omit `--gres`/`--cpus-per-task`.
- **Skills to adapt:** real_Cartpole `.agents/skills/*` (sim-to-real-change-gate, hardware-preflight, control-loop-timing-audit, remote-compute-background-jobs).

**Caution (weigh before M6):** real_Cartpole logged an RL track record of 0/20, 0/20, 1/20 across three reward redesigns (~4.4 M steps) vs a fixed-gain classical baseline — evidence to confirm a *simpler* baseline fails before committing to a learned latent-space safety filter.

---

## 12. Architecture option: geometric fabric as the smooth layer

Geometric fabrics (Van Wyk/Ratliff, arXiv:2109.10443; safety follow-up arXiv:2405.02250) are a closed-form second-order acceleration policy `q̈_f = π(q_f,q̇_f)` that **composes** behaviors (goal attractor + arm–arm repulsion + joint-limit barrier + grasp impedance) into one command, is **provably stable + smooth by construction** (HD2 speed-invariant geometry + energization + damping), and runs at kHz. Grasp/contact is handled impedance-style (fabric target diverges from state → contact force), demonstrated on in-hand manipulation.

**Trade vs our HOCBF plan:** fabric safety is a **soft** high-impedance barrier (no forward-invariance; a strong driving force can overpower it) — the *opposite* end of the guarantee↔smoothness spectrum from HJ/CBF.

**Recommended synthesis (candidate primary architecture):** **fabric = smooth reactive layer, thin HJ/HOCBF filter = hard-safety gate.** The fabric keeps motion smooth, composed, grasp-preserving and far from the boundary, so the HJ/CBF gate rarely fires → rarely chatters, yet retains the formal guarantee the fabric lacks. This removes the NLP-MPC real-time risk (§7.1) *and* is literally HaMMAR's un-built "high-frequency low-level controller." Novelty sharpens: **HJ-reachability hard safety + geometric-fabric smooth avoidance + modeled grasp dynamics, bimanual.**
**Caveats:** fabric metric design/tuning needs expertise (hours); check the open-source `fabrics` package (Spahn et al.) before hand-rolling. Add **fabric avoidance** as a 3rd condition in the §5 A/B comparison (bang-bang vs HOCBF-QP vs fabric).

**Empirical update (v2, 2026-09-05):** ran fabric as C3 (self-rolled minimal, soft, no gate) under an aggressive oscillating intruder. The 2×2 — {hard,soft}×{smooth,jerky}:
- **C1 bang-bang (hard+jerky): DROPS the box** (35 cm / 101° grasp drift) — chatter shakes it loose, though arms stay apart.
- **C3 fabric (soft+smooth): COLLIDES** (sustained arm–arm contact) — smooth and holds the grasp far better than C1 (4.3 cm drift), but soft safety is overpowered.
- **C2 HOCBF-QP (hard+smooth): the ONLY condition that both avoids AND keeps the box** (jerk 308 — best on every axis).
Conclusion: *smoothness alone is unsafe; hard safety alone destroys the grasp; you need **hard+smooth**, which the HOCBF-QP already delivers.* **Architecture recommendation UPDATED: lead with the HOCBF-QP / MPC-HOCBF route (Route A). Demote fabric** to an optional smoothness/composability tool usable only *with* a hard gate — no longer the front-runner. Honest caveats: C3 fabric was minimal/untuned (a tuned fabric, or fabric+hard-gate, is still untested); the scenario was hold-pose-under-attack, not goal-reaching-past-the-other-arm, so MPC's horizon/deadlock advantage is still unstressed. **Payload coupling = null result** (box only 50 g → gravity torque negligible; EE offset 1.423→1.414 cm); to actually test the grasp-dynamics claim, use a heavier/eccentric payload or a closed-chain object.

---

## 13. Teleop constraint-legibility & sim-to-real confidence bounds

**Why here:** the geometric route (HOCBF-QP / fabric-gated, known dynamics) is orders faster than latent-world-model (Dreamer/DINO-WM) reachability, so it can preplan a horizon **and** expose a live safety signal at control rate — which the latent route structurally cannot. *Scope honesty:* this covers geometric failures (collision, grasp-manifold) only; semantic failures still need the slow latent route, which gives no cheap live measure.

### 13.1 Live "constraint-tightness" measure for teleop
At each instant the barriers + torque limits define a feasible safe-action set `𝒰_safe(x) = {u : ψ₂(x,u) ≥ 0 ∀ barriers, u ∈ [u⁻,u⁺]}`. Measures of how constrained the current region is:
- **Safe-action-set volume / inscribed-ball radius** (primary) — → 0 as a collision/grasp constraint binds.
- **HJ value margin** `V(x) − ε` — time-to-failure buffer.
- **Active-barrier / QP-slack** — which constraint binds, how hard.
Surface to operator: HUD gauge + workspace coloring; **haptic force-feedback, stiffness ∝ 1/tightness** — a reachability-grounded virtual fixture. Proactive legibility vs latent-safety's last-moment override.

### 13.2 Confidence bounds on the sim-to-real gap
Margin is computed in sim; bound its validity on hardware two ways:
- **Robust reachability:** bounded mismatch `‖d‖ ≤ d̄` → inflate required margin `ε(d̄)` for a hard worst-case bound.
- **Conformal calibration:** log (sim-predicted margin) vs (real closest-approach/violation) over rollouts → "at margin m, real stays safe w.p. ≥ 1−α." **Teleop is the calibration data source** (operator drives into tight regions on purpose).
Targets stated open problems: NeHMO future work = "conformal prediction for selecting the safety margin"; Latent-Safety = "characterize how errors propagate to assurances."
**Caveats:** conformal → marginal (population) coverage, not per-state; vanilla conformal assumes exchangeability → sequential teleop data needs time-series/episode-level conformal; bound valid only where teleop data covered the state space; robust bound needs a credible `d̄`.

**Deliverable/figure:** teleop clip with live tightness gauge + force feedback, beside a calibration plot (sim margin vs real closest-approach + fitted conformal coverage) — a *quantitative sim-to-real safety-gap* result.
