# Experiment plan: claims → evidence (2026-10-16, owner draft)

> **2026-10-26: partly superseded.** Current claims, scope (manipulation core; people safety is a borrowed component) and what is obsolete here: `docs/paper/claims_2026-10-26.md`.

The frame is fixed: novelty = the checkers (C1 admission, C2 backup→task-ready, C3 grasp-load corners, C4 probe value) plus the theorems
(T1 soundness, T2 monotonicity, T3 completion bound, T4 recovery existence). Everything else is borrowed.
Sim only: house h5 (MolmoSpaces/ProcTHOR), YOR base + 2 PiPER arms (NERO later), FL-MPC arms, 50 Hz base filter.
Small n = examples, not rates. Every rate gets n ≥ 20 seeds and a 95% CI.

## Claims and the experiment that can FALSIFY each

| # | Claim | Experiment | Metric | Falsified if |
|---|---|---|---|---|
| E1 | T1: no S1–S3 violation while A1/A2 hold | sweep (truth people; then tracker), all gatekeeper configs | wall / person-in-motion contacts, cup slip, spills; audit n_belief_miss, n_untracked_near | any violation in an episode with 0 audit misses |
| E2 | T1 needs the KEEP monitor (GAP-2) | recertify vs monitor, same seeds | n_falsified, regime-2 ticks, done-rate | monitor has more violations than recertify |
| E3 | T2(a) corollary: widen-only VLM cannot create admissions | pick planner with Nano / Qwen / LoRA / Codex answers, veto_mode=probe | admitted-but-infeasible count (vs the rule with exact sim mass) | any admitted plan infeasible for the true mass |
| E4 | Stop-safe filters livelock; recovery restores progress (motivates T3/T4) | corridor-with-standing-person + h5 carry: shield-brake, shield-escape, gatekeeper hold, + candidate set (B), + library recovery (MOVE/LOOK) | done-rate, time, number and kind of recoveries vs the T3 bound | done-rate does not rise with recovery, or the recovery count exceeds the bound |
| E5 | Belief quality drives progress (POMDP) | tracker: raw conformal e_p vs set-membership intersection vs LOOK action | filtered radius, done-rate, belief misses | intersection raises the miss rate above the conformal α without a progress gain |
| E6 | C3 exact corners decide 1 vs 2 arms correctly where the footprint rule fails | offline (80 held-out + dataset) + in-sim lifts of thin books / mugs / bowls | agreement with physics (sim lift success), not with the old rule | C3-admitted grasps slip in sim |
| E7 | C4 probe value beats rule-fallback and blind execution | pick objects with VLM-widened mass: rule fallback vs probe | unsafe lifts (true mass > limit), time to completion | probe mode is slower AND no safer |

## Baselines (same harness, same seeds)
- Shield (one-step CBF-QP polytope + emergency brake; also fallback=escape): what most shared-autonomy filters do.
- Gatekeeper with POINT beliefs (Agrawal & Panagou style: truth or mean people, no sets): shows what the set-valued belief costs and buys.
- Backup-CBF (Gurriet/Ames style), if time allows: the same backup with a CBF filter instead of commit/keep.
- Ablations: speed vs accel person model; hold vs candidate set; recertify vs monitor; no-intersect vs intersect; library off vs on.

## Scenarios (scripted, seeded)
1. h5 carry (lab → cafe), sloppy operator, 2 people (one standing near the corridor). Done.
2. Narrow corridor + standing person at a controlled offset (sweep the offset 0.5–1.2 m): the livelock curve.
3. Occlusion: a person appears from behind a door (tests A1 audit, LOOK recovery, T3 occlusion counting).
4. Low-friction tray (μ_true near μ_lo): cup-slip stress (S3 with the chord bound).
5. Pick set (mug, cube, bottle, crate, basket, thin book, bowl): C3 + C4 with VLM proposals.

## Run budget (CPU sweeps on westeros, nice 10; ~1–3 min/episode)
- E1/E2/E4/E5: 20 seeds × ~10 configs × 2 people sources ≈ 400 episodes, about 1 h at 20 workers.
- E6/E7: offline + about 100 sim lifts.

## Figures / tables for the paper
1. Safety–progress plane (done-rate vs min clearance / violations) for all configs.
2. Livelock curve vs person offset (scenario 2).
3. Recovery-count histogram vs the T3 bound.
4. Belief radius over time (raw vs intersect) with misses.
5. C3 decision table vs physics and VLMs.
6. Probe vs fallback table.

## Order
E2 + E5 (running now), E1 at scale, E4 (library integration), E6/E7 (C3/C4 agents running), then scenario 2/3 scripting.
