# MASTER — Bimanual YOR: cooperative safety + learned world model

Single source of truth for **what we are doing and why**. Supersedes the
earlier `method.md` / `state_spaces.md` on any conflict (they predate the
decisions below). Last updated: 2026-09-29 (see §1b status block — it supersedes conflicting lines in §2–§8).

---

## 1. Thesis (locked)

> **SUPERSEDED 2026-10-15 (user decision).** The learned world-model / latent-HJ thesis below is DROPPED. Current direction:
> classical planning + formal checkers + VLM proposals, evaluated as a POMDP with set-valued beliefs, autonomous AND shared-autonomy modes.
> Novelty = the CHECKERS (belief-wide admission, backup to a task-ready set, exact grasp-load corner check, probe value) + the THEOREMS
> (soundness, monotonicity, finite completion bound, recovery existence). Plan: docs/paper/YOR_Thesis_Full_Idea_and_Demonstration_Plan_2026-10-15.docx.
> Kept below for history.

Demonstrate **in sim** that a learned **world model (WM) enables learned HJ
reachability safety** for bimanual manipulation — the WM supplies a failure
predicate that geometric reachability *cannot* express, while a cooperative
controller keeps the two arms from colliding **and** retains the task,
smoothly. Grounded in NeHMO (2507.13940), HaMMAR (2511.03591v2), Latent
Safety Filters (2502.00935).

The sharp version, as a formal spec (STL): the task is
`⋀ᵢ F(cube_i in cup) ∧ G(no_arm_collision ∧ ¬spill ∧ grasp_held ∧ cup_in_view)`.
Every predicate is analytic **except `¬spill`**, which has no geometric
signal — that gap is exactly what the WM fills. STL is our **certification /
evaluation** layer (robustness score), **not** a control synthesizer (see §7).

**Extension (user-confirmed 2026-09-29):** a **BOED-style mechanism for shared
autonomy to update itself and learn on the fly**. The robot (or the operator,
on request) runs small, certified-safe probes chosen by how much they shrink
uncertainty in the safety tube/value, and the WM/value belief updates online.
- Scope: grasp/slip is out (classical control + wrist cameras).
- The primary testbed moves toward observable payloads; unobservable contents
  (opaque liquid) go to a separate, conservative system.

## 1b. Status 2026-09-29 (supersedes conflicting lines in §2–§8)

- **Controller:** FL-MPC (`experiments/ab_smooth_vs_bangbang/mpc_fl.py`) is the
  solution; v4-C\* (§4) is the **baseline**. The fair baseline is
  `--v4c-preset tuned` (acceleration-metric QP); the original torque-metric v4c
  collided 66% at N=50.
  - N=50 FL-MPC vs tuned v4c: collisions 0% / 0%, jerk 301 vs 616, cup tilt
    14.1/14.2° vs 18.3/30.1°, cubes placed 3.00 vs 2.54.
  - File: `bimanual_data/fakedsel/n50_faked_all_flmpc_v4ctuned.json`.
- **Do not cite the §5/§8 numbers:** the 4% collision floor, WM spill AUC
  0.999→0.746, and latent value 8/8→0/8. They came from **pre-fix** harness
  bugs: sideways-cup init, the lift-loss spill oracle, and the broken v4c metric.
  The old `latent_safety_value.pt` is policy evaluation (no max over actions),
  not reachability.
- **Spill:**
  - Plan-aware physics matches or beats every learned predictor (3941
    episodes); the cup contents are unobservable from the ZED.
  - A latent HJ value on a nominal-data WM collapses: its closed-loop fallback
    spills 79–84% of physics-savable states. Counterfactual evasive-action
    branches fix it (1.2–1.6%, 3 seeds). Details: `bimanual_data/wm_audit/REPORT.md`.
- **Value constraint:** FL-MPC with look-ahead value rows spills 8/50 vs 28–30/50
  for one-step LatentCBF and 30/50 for switching (`bimanual_data/latentcbf/final/`).
  It is not real-time yet: the value rows take 102 ms per step.
- **BOED core (tube-shrink forecasting), pre-registered kill test:**
  - It survives as a **predictor**.
  - As a one-step decision rule it loses to a reactive switch.
  - Next: execution-aware, multi-step value of information
    (`bimanual_data/g3_killtest/REPORT.md`).
- **Real time:** a numba FL-MPC rewrite plus a pipelined solver is in progress
  (`experiments/realtime/rt_flmpc.py`). Clean timing: the controller alone runs
  ~5 ms mean, with a ~17 ms p99 dominated by the QP.

---

## 2. Architecture — division of labor (locked this session)

Two safety channels enter the control at **different layers and rates**; they
do NOT both feed the QP.

| Concern | Handled by | State it uses | Enters control at |
|---|---|---|---|
| Arm–arm collision, task motion | **cooperative HOCBF-QP (v4-C\*)** + task sequencer | **exact joint state** of both arms | torque level, control rate |
| Object-of-contact failure (**spill**) | **WM (RSSM) → spill reachability value** | ZED image + proprio (WM latent) | task reference (freeze/slow), vision rate |

Key decision: **arms are handled analytically from exact joint state; the WM
is confined to the object of contact (cup + contents).** The WM never has to
encode 12-DoF arm geometry at cm precision (that was the latent-bottleneck
risk); contact is left **implicit** (gripper-closed = holding), no invented
sensors.

- **Learned HJ, two forms:** (a) arm–arm = HJ/HOCBF on `x=[q_L,q_R]∈ℝ¹²`
  (analytic barrier now; optionally a learned explicit-state value `V(x)` that
  drops into the same QP rows); (b) object/spill = a **Fisac
  discounted-safety-Bellman value `V_spill(z)`** over the frozen WM latent
  (the Latent Safety Filter). Same math as DeepReach in the limit; DeepReach
  itself is **demoted to an optional geometric baseline** (needs an analytic
  PDE, can't run in latent space).
- **FLOQ (explicit-state) = baseline stepping-stone, not the endpoint.** The
  latent value trainer is written fresh (FLOQ's flow-matching classifier
  doesn't transfer).

---

## 3. Observation / state design

- **Proprio (26-D):** `[q_L, q_R, qd_L, qd_R, grip_L, grip_R]` — both arms'
  joint pos+vel + gripper opening. Contact/grasp is **inferred** (gripper
  state), never sensed. No force/torque/contact sensor (hardware has none).
- **Vision:** ZED workspace camera. This is the WM's only image input and the
  sole signal for spill. (**Caveat:** the sim ZED is a *proposed torso mount*
  at z≈1.0 m; the real robot's ZED is a floor-facing nav cam — mount is
  confirm-on-hardware.)
- **WM = object only.** Current WM is trained on the full obs and works
  (spill AUC 0.999 in-dist); we simply don't use its latent for arms. Optional
  future: retrain a narrower object-only WM if spill prediction proves
  distracted under OOD.

---

## 4. The cooperative HOCBF-QP (v4-C\*)

Per arm, each step: turn "keep arms apart" into a torque correction.
- Barrier `h(q)` = min arm-arm keypoint distance (~25-keypoint faithful proxy,
  `all_pair_barriers`). Relative degree 2 wrt torque → **HOCBF cascade**
  `ψ₁=ḣ+α₁h`, enforce `ḧ+α₁ḣ+α₂ψ₁≥0` (α=30); `ḧ` is linear in τ via
  `q̈=M⁻¹(τ−c)` (`mj_fullM`/`qfrc_bias`/`mj_jac`).
- QP: `min‖τ−τ_task‖²` s.t. HOCBF rows + `|τ|≤40 Nm`
  (`solve_constrained_box_qp`, Numba). Least-restrictive safe torque.
- **Cooperative, decentralized:** both arms run their own QP; ego (cup) arm
  protected, intruder (collecting) arm carries an **extra 0.03 m margin**
  (`INTRUDER_EXTRA_MARGIN["C"]=0.03`) → yields earlier. Both arms still deflect
  via their own QP (the cup arm *does* move to stay safe).
- **Failure mode:** if the required τ exceeds ±40 Nm (deep/fast crossings,
  many active pairs) the QP is **infeasible** → guarantee lost → collision.

### 4a. Speed-holding + table floor (2026-09-07 — "fix them all" pass)

Three complaints — *arms still collide sometimes*, *arms sink into the table*,
*EE moves too fast* — trace to two structural gaps and one honest limit:

1. **No speed bound.** Neither v4-C nor v7 bounded EE velocity, only arm-arm
   *distance*. Measured on the flagship crossing, the right arm's scripted
   **return-to-home ("park") peaks at ~0.79 m/s** at the EE (release ~0.40,
   hover ~0.36) while the actual crossing ("carry") is only ~0.24 — so "moves
   too fast" is the fast return, the **same segment** where the right-arm
   keypoints sink **~3.4 cm below the tabletop** (measured, `--table-floor
   off`: right min-z 0.516 vs table 0.55). Fix = **one Cartesian speed cap
   `V_EE_MAX=0.28 m/s`**, applied two ways: (part 1) the scripted right-arm
   reference is **time-scaled** so no phase's EE speed exceeds the cap —
   computed from each phase's real Cartesian travel, carry already under the
   cap is untouched; (part 2) a **hard next-step EE-velocity constraint**
   (`ee_speed_limit.ee_velocity_cap_rows`, 6 box-QP rows, affine in the
   applied torque) on the right arm's QP under **both** controllers (v4-C
   reactive rows; v7 via `JointMPCv7(v_ee_max=…)`). Part 1 keeps nominal
   motion off the constraint; part 2 is the guarantee, even during a dodge.
2. **No table awareness.** Arm collision-proxy geoms don't collide with the
   table (contype), so nothing stopped a link sinking through it. Fix =
   `sim_common.table_plane_barrier_rows`, a static half-space HOCBF
   `h = z_kp − (z_table − margin) ≥ 0` (margin 6 mm so the gripper can still
   reach cubes resting on the surface), added to **both arms** in v4-C **and**
   now the MPC (`JointMPCv7.step(table_rows=…)`).
3. **Why MPC (v7) is the right tool, honestly.** A reactive QP can bound
   *instantaneous* speed but cannot *plan* the deceleration — it brakes late
   at the limit. MPC bounds speed **and** anticipates the crossing over the
   horizon, so it yields early and smoothly and hits the ±40 Nm infeasibility
   wall (the residual-4%-collision mechanism) less often. v7 was shelved
   earlier on an **unfair** comparison (untuned MPC drop-in vs heavily-tuned
   reactive, on a testbed that doesn't reward foresight); this pass gives v7
   the same speed + table constraints so the crossing comparison is fair.
   `--ee-speed-limit {on,off}` and `--table-floor {on,off}` gate the fixes for
   before/after A-B. **Residual 4% collision floor is honest and unchanged:
   under deep QP infeasibility the guarantee still breaks; MPC lowers the rate
   but does not eliminate it.**

---

## 5. Results to date

- **LEG A (proven, distributional, N=50 randomized geometries):** v4-C\*
  (margin **0.03**) dominates the adversarial baseline **and** the old 0.06:
  collision **68%→4%**, grasp **0%→82%**, task-success **0%→74%**, chatter
  median **10.7→3.3**. Non-monotonic optimum at 0.03 (too little → intrude &
  collide; too much → over-activate constraints → QP saturate → collide).
  Residual 4% = actuator-saturation floor, not a tuning gap. Verified in
  isolation (`artifacts/margin_grid/`, `artifacts/verify_m003/`).
- **WM (in-distribution):** spill AUC 0.999, topple 1.000, collision head
  1.000; reconstruction MSE 1.6e-3. **OOD-brittle:** 0.999→0.746 under DR;
  DR-retrain did **not** fix it (open problem).
- **Does-WM-help (earlier cup demo):** geometric-only spills → WM veto 0
  spilled. (Integrated A/B harness still needs a clean rebuild.)

---

## 6. Flagship task (current)

**Cube-collection into a held, lifted cup** (replaces stacking as flagship;
stacking kept as secondary/ablation). Left arm grasps + lifts the cup (balls
inside, kept below ZED level); right arm collects **scattered cubes** and
deposits them into the cup; cube placement forces arm–arm crossings.

- Scene: `scenes/scene_yor_cup_collect.xml` + `scripts/cup_collect_task_utils.py`
  — **built & verified** (loads nq=121; rests stable; 6 cubes right-arm
  reachable + cup left-arm reachable sub-mm; cube_4/5 induce real crossing
  contacts; cubes fit the cup; ZED sees cup+cubes+both arms+deposit zone).
- Controller (sequencer + v4-C\* port + grasp/lift + deposit) — **in build**
  (the L-effort milestone). Grasp/transit uses the repo-standard scripted-
  attach (`collect_stack`); **arm motion + safety filter are real.**
- Success predicate: `deposited≥K ∧ count_spilled==0 ∧ no arm collision ∧ no drop`.

---

## 7. Safety-module interface & recovery

- `eval/safety_module.py` Protocols: **GeometricSafety** (`correct(GeometricInputs)
  →GeometricResult`, torque level) and **SemanticSafety** (`assess(SemanticObservation)
  →SemanticVerdict{risk,veto}`, task level). `run_eval`: semantic gates the
  reference (freeze-clock), geometric filters the torque. Learned `V(x)` swaps
  into the HOCBF rows; `V_spill(z)` swaps into `assess()`.
- **Recovery:** *automatic* — geometric filter relaxes as arms separate; the
  freeze-clock resumes the deposit when spill risk clears (no snap). *Gaps to
  build* — (1) **infeasibility/boundary backup**: on QP-infeasible, switch from
  least-restrictive to a safety-descent (follow `−∇V` / max-`h`, or QP slack a
  la `C-adapt-relax`); (2) **deadlock timeout** so the collecting arm can
  always eventually deposit. Post-spill = freeze + fail (no re-level).
- **STL** = spec + robustness eval on top; not a synthesizer (STL-MPC over
  12-DoF contact is nonconvex/combinatorial — out of critical path).

---

## 8. Honest open items

- WM OOD brittleness (0.999→0.746) — a latent safety value inherits it;
  needs graceful-degradation + stress test.
- ZED mount is proposed/confirm-on-hardware.
- FLOQ-explicit is CPU-bound (flow-matching, GIL); baseline only.
- Latent HJ value + shield: **BUILT + validated + wired (2026-09-07).** Fisac
  discounted-safety-Bellman value `V_θ(z)` over the cup-collect WM latent
  (`world_model/train_latent_safety_value.py`, ckpt `latent_safety_value.pt`),
  `V<0 ⇔ unsafe`, imminent-failure separation **AUC 0.998** (V(safe)=+0.20,
  V(failure)=−1.0). Wired as the least-restrictive shield (`LatentValueShield`,
  `--monitor value`). Flagship spill result: **no shield 8/8 spilled → P(spill)
  classifier 3/8 → V(z) reachability shield 0/8** — it vetoes *anticipatorily*
  (V goes negative well before P(spill) rises). IMPORTANT: integrated as a
  shield (veto/slow the task), NOT a joint-space QP-barrier swap (no tractable
  ∂V/∂q through the WM). In-distribution only; inherits the WM's OOD brittleness.
- Flagship cup-collect controller: BUILT (crossing + spill scenarios,
  magnet-weld cube grasp, cup-in-barrier right-arm↔cup avoidance). Recovery:
  QP-infeasibility brake added; a deposit/deadlock timeout is still TODO.
- Real-time gap on the cooperative barrier assembly (~tens of ms).
- Gripper→wrist neck: cosmetic visual gap (mesh-swap), pending alignment fix;
  wrist iPhone-camera visual already fixed to a compact housing.
