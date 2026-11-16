# WM-requirement ladder — results

**Question (2026-09-10):** before building the spill world-model, how powerful does
it need to be / what degree of dynamics must it capture?

**Method:** measure the gap between a ladder of analytic spill predictors and the
ground-truth ball oracle (`count_spilled`). The rung where accuracy saturates is the
dynamics order the WM must capture.

- **rung 0** static tilt (memoryless, pose only)
- **rung 1** effective-gravity cone `up_eff=(a_cup−g)/‖·‖` (memoryless, +accel)
- **rung 2** + 1st slosh mode: damped 2-D oscillator driven by cup accel (2nd-order latent)

Ground truth: 8 loose balls, `scripts.cup_task_utils.count_spilled`. Cup dims: R≈0.028 m,
rim 0.080 m, fill 0.026 m ⇒ static overflow tilt θ_max≈63° (so spills are **dynamic**,
not static-tilt). Slosh ω≈24.6 rad/s (Abramson/NASA SP-106).

## Result (clean oracle, 24 seeds, 10 spilled / 14 clean negatives)

Run AFTER the spill-oracle fix (commit bb0f9be — see below); onsets are genuine
crossing-phase spills from a `count_spilled==0` baseline.

| rung | AUROC (pooled) | median lead |
|---|---|---|
| rung 0 static | 0.785 | ~2.3 s |
| rung 1 eff-gravity | 0.798 | ~2.3 s |
| rung 2 slosh (physics ω=24.6) | 0.812 | ~2.2 s |
| rung 2 slosh (best-fit ω=12.3, ζ=0.15) | **0.857** | ~2.2 s |

## Honest conclusion (moderate confidence)

- The analytic ceiling is **~0.80–0.86 AUROC** — far from perfect, so there is real
  residual for a learned WM to capture.
- At the **physics-derived ω, rung 2 ≈ rung 1** (0.81 vs 0.80); the 0.857 needs ω **fit**
  to the 10 events. So the evidence a low-order oscillator helps is **weak-to-moderate,
  fit-dependent** — not a clean saturation.
- **Bottom line:** the spill WM should carry **at least a compact low-order dynamic
  (slosh-like) latent**; the leftover gap to ~1.0 (unexplained by any analytic rung)
  warrants some extra learned richness, but a **full high-dimensional fluid model is not
  justified** by this evidence. A full DreamerV3 is likely more than the *spill* channel
  needs (it may still be wanted for richer semantics).

## Caveats

1. **Fit-dependence:** the rung-2 advantage rests on fitting ω on 10 events. Prior
   (broken-label) runs over-fit this to a spurious 0.92 — see the retraction in
   `flmpc`/git history. Treat the 0.857 as an upper-ish bound, 0.812 as the honest tie.
2. **Release-transient confound:** ~6/10 clean spills onset within ~80 steps of release
   (balls released with zero velocity may self-trigger). Firm up with a brief pre-release
   settle or releasing with the cup's velocity, and more seeds.
3. **Fidelity ceiling:** `reattach_cup_rigidly` pins the cup with ZEROED velocity, so the
   balls feel geometric wall-sweep, not true inertial forces — the whole spill mechanism
   is a crude analog of a dynamically-grasped real cup. Caps how faithful any spill study
   or WM trained on it can be.

## The spill-oracle fix (prerequisite for the above; commit bb0f9be)

`count_spilled` originally read 6–8/8 balls "out" from step 0 because
`init_cup_grasped_state`'s grasp never holds the cup (fingers don't contact the walls),
so cup+balls scatter during init. Fixed by carrying the balls with the kinematically
re-pinned cup using the canonical `home`-keyframe layout through the lift, then releasing
at the crossing. **All pre-fix spill metrics (incl. the project's "~92% spill") were
inflated by this artifact.**
