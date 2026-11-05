# Covered pan: our constraint layer vs a faithful VIZ-COAST (2026-10-28, updated 10-29)

> Status (2026-11-05): HISTORY: record of 2026-10-28/29 (see the 10-29 note inside: the 10-28 headline is superseded). The pan demo is no longer primary (docs/paper/demos_2026-11-01.md).

All simulation. Code:
- `world_model/v2/nero/pan/vizcoast/` (ours);
- `world_model/v2/nero/pan/vizcoast_faithful/` (baseline);
- `world_model/v2/nero/pan/observe_pose.py` (built 10-29);
- `world_model/v2/nero/pan/llm.py` (swappable proposer model).

Results:
- `results/pan_vizcoast_2026-10-28/`, `results/pan_vizcoast_faithful_2026-10-28/` (10-28);
- `results/pan_observe_pose_2026-10-29/` (E1 accuracy, E2 closed loop);
- `results/pan_fair_box_2026-10-29/` (same-world comparison, `SUMMARY.md`);
- `results/pan_llm_proposers_2026-10-29/` (gpt-5.6-sol as proposer for both layers).

Pre-registrations sit in each folder. Bound labels follow AGENTS.md "Bounds and assumptions".

> **10-29: the 10-28 headline is superseded.**
> - "Ours 32/32 corners + 30/30 interior" holds only on the ±5 mm / 2° box. It modelled the unbuilt observe_pose as "the pan is
>   within ±5 mm / 2° of nominal" (ASSUMED), which shrank the world set and not just the aim error.
> - VIZ-COAST's 42/100 ran on the full ±1.5 cm / 8° box, so the two numbers were not comparable.
> - The same-world comparison below replaces both.

## 10-29: same worlds for every plan

The worlds are 32 corners plus 30 random interior points, identical for every plan in a row.
- "full" = the ASSUMED prior box: ±1.5 cm, ±8°, μ×0.6–1.2, 0–0.4 kg.
- "±5 mm/2°" = the box from the 10-28 observe assumption.

VIZ-COAST plans are its five sampled push + clamp groundings: pinch 0.58–0.81, push 5.2–8.8 cm. With gpt-5.6-sol the VRM produced the
same four groundings as the scripted run (same stream seeds), so these rows also cover the sol plans.

| Worlds, aiming | VIZ-COAST, per plan (/62) | VIZ-COAST, all | Ours: pinch 0.65 / push 7 cm |
|---|---|---|---|
| full, nominal aim (open loop) | 20, 27, 36, 31, 23 | 137/310 (44%) | 28/62 (45%) |
| full, built observe_pose, zed noise | 29, 34, 48, 40, 30 | 181/310 (58%) | 46/62 (74%) (E2) |
| ±5 mm/2°, nominal aim | 32, 57, 54, 59, 54 | 256/310 (83%) | 62/62 (10-28 Stage B + interior) |

All rows are `sampled`, not proofs. Reading:
- With the same worlds and the same perception, our plan beats the average VIZ-COAST plan, but not its best one: pinch 0.755 / push
  8.8 cm holds 48/62 with observe. The gap comes from picking parameters, not from a different kind of plan.
- On the full box no single re-aimed plan holds everywhere. Every vertex failure of ours is at yaw −8°: either the left claw has no
  clean pre-grasp, or the pan is far left and heavy and the right claw loses the rim.
- VIZ-COAST reports every one of these plans as a success, because its streams check kinematics only.

## 10-29: our layer with the built observe_pose and gpt-5.6-sol

Stage B runs on the FULL box with observe_pose in the loop: depth fit at zed noise, claws re-aimed at the estimate
(`demo --observe-mode built`).
- Result: **no plan** (157 nodes, 8 certificate calls, 8 sim runs, 118 s).
- Pinch 0.6, 0.7 and 0.75 (with observe) were each falsified at a corner: no clean pre-grasp, or sag. 0.65 was already refuted by E2.
- The layer refuses where VIZ-COAST runs a plan that holds 29–48/62.

Sol as proposer (same model for both layers; `codex:gpt-5.6-sol:high`):
- **Ours.** Proposed: no lid grasp, no probe_mass ("tipping could spill"), refine the pose before pinching, lift only with handle +
  rim clamp + lid secured. It did not narrow the belief. Search shrank from 291 to 35 nodes in Stage A.
- **VIZ-COAST.** Four of five seeds gave push + clamp. Seed 4 gave a one-arm handle lift, which is physically impossible here
  (0.3 Nm capacity vs ≥0.43 Nm needed) and still reported as a success.
- A first sol run hit a bug in our API: it lower-cased the arm names while the PDDL showed `L`/`R`. Sol's correct program matched
  nothing, then relaxed into one-arm plans. Fixed; that run is kept as `vizcoast_sol/SUPERSEDED.txt`.

## observe_pose, built (E1, E2)

Method:
- One depth image from the virtual head camera, 5×5 median.
- Robust circle fit to the wall gives the centre; robust handle axis gives the yaw.
- Up to 3 images, then refuse. Claws re-aim at the estimate.

Residual over 200 random poses, max over n (not a bound). The noise model is ASSUMED and synthetic.

| noise | position p95 / max | yaw p95 / max | within 5 mm and 2° |
|---|---|---|---|
| clean | 0.04 / 0.06 mm | 0.21 / 0.40° | 200/200 |
| mid | 0.9 / 1.3 mm | 1.6 / 4.2° | 198/200 |
| zed (~1% of range, part spatially correlated) | 2.3 / 3.4 mm | 4.7 / 9.4° | 137/200 |
| zed + camera calibration error 3 mm / 0.3° | 6.2 / 8.1 mm | 5.1 / 8.8° | 105/200 |

- Position meets the old 5 mm assumption at zed noise. Yaw does not meet 2°: the handle is the only yaw cue.
- With a perfect estimate the plan still fails at yaw −8°, so better perception alone does not close the gap.

## 10-28 record (kept; superseded where marked)

- **Faithful VIZ-COAST.** Z3 sequential encoding, MuJoCo kinematic streams on the nominal scene, scripted VRM. Paper setup: Gemini 3
  Flash at medium reasoning; K=5 revise, relax on no plan.
  - VRM "none": one-arm lift by the handle, 0/5 lifted.
  - Push + clamp plans: 42/100 on the first 20 Set-A worlds (full box). The comparison with ours was SUPERSEDED (different box).
- **Ours, assumed observe.** Pinch 0.65 / push 7 cm passed 32/32 corners and 30/30 interior of the ±5 mm/2° box. The claim was
  CONDITIONAL on the ASSUMED residual and is SUPERSEDED for the full box (above).
- **Overload** (0–0.8 kg): no plan; the right claw loses the rim.
- **Proposer ablation (Stage A, scripted):**
  - good proposer: 291 → 148 nodes, 42 → 2 certificate calls;
  - wrong proposer: claims ignored, same plan;
  - over-pruning proposer: no plan, and the diagnosis names `forbid clamp_rim`.
- **Fixed 10-29 (VIZ-COAST baseline).** Clause order depended on Python's hash seed; it is now sorted.

## Caveats

- `certify.py` rejections are sound: a model fact or a real counterexample. Admissions are `sampled`: Stage B tests the 32 corners
  and stops after 2 failures. The claw capacity is a fit to 4 points, not a bound.
- probe_mass (exact mass) is still ASSUMED. The probe is not built, and neither is any backup to a task-ready set: the pipeline only
  lifts.
- Codex caveats apply to every sol result: Codex adds its own system prompt, temperature is not settable, and plan rate limits
  apply. Sol is a stand-in until all three methods (with AgentChord) move to one open-weight model.
- The camera is virtual and the depth noise is synthetic.

## Next

1. Choose the plan from the observation: certify per sub-box of the estimated pose (grasp and push as functions of the estimate)
   instead of one fixed grasp and push for the whole box.
2. Widen our pinch/push grid toward VIZ-COAST's best plan (0.755 / 8.8 cm).
3. One open-weight model for all three methods.
