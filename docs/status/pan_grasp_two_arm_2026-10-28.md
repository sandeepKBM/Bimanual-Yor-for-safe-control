# Covered pan: one claw vs two claws, grasp proposer, lift reliability (2026-10-28)

All simulation (MuJoCo, NERO 7-DoF arms, FL-MPC, real friction contact, no welds). Results:
`world_model/v2/nero/results/pan_handle_grasp_2026-10-28/`. Scene: `pan/covered.py` (PLA pan + loose lid + hidden puck; all
dimensions and densities are assumptions). Squeeze 10 N = MJCF gripper force limit, run in force mode (the stock MJCF gripper is a
position servo that squeezes only about 2.5 N on the 14 mm handle).

## 1. One claw cannot lift the covered pan

Code: `pan/handle_grasp.py`, `pan/handle_grasp_sweep.py`.

- Claw pitch-moment capacity grows with squeeze. Measured on the empty pan, pinch at 60% of the handle:

  | Squeeze | Capacity |
  |---|---|
  | 2.5 N | 0.05 Nm |
  | 5 N | 0.14 Nm |
  | 7.5 N | 0.22 Nm |
  | 10 N | 0.30 Nm |

- Needed at the grasp:

  | Case | Moment |
  |---|---|
  | empty | 0.43 Nm |
  | near handle | 0.66 Nm |
  | far side | 0.84–0.86 Nm |
  | overload | 1.72 Nm |

  At a 70–80% grasp the need rises to 0.47–1.95 Nm.
- Every case leaves the pan pivoting on its far rim: about 11° of droop in the claw, with the lid and puck staying put.
- Grasp position: the IK target is the fingertip point. From the handle end, only 60% or more along the handle is reachable; deeper, the palm hits the handle end. 60% and 80% never lift.
- Twist friction (condim 4) changes nothing.

## 2. Two claws on the same handle: no lift

The right claw was tried tilted at 30° and from the side, aimed so the fingers wrap the handle. In both grips the right claw acts as a hinge about the pitch axis, and the left claw cannot supply the couple.

One case is unexplained: with the left claw at 70% and the right side grip at 50%, the left fingers lost the handle.

## 3. Borrowed grasp proposer: graspnet-baseline

The user chose this one. AnyGrasp needs a per-machine license form from the user.

- **Setup:** env `/data/local/ss5772/graspnet` (torch 2.1.2 cu118, sm_75), commit 280c215, RealSense checkpoint (sha256 6068008…). Wrapper `pan/graspnet_proposer.py`; sim pipeline `pan/graspnet_sim.py`.
- **Camera and runtime:** virtual head camera; the cloud is accurate to under 1 mm. Peak GPU memory 662 MiB on westeros GPU 6 (released).
- **Proposals:** 86–94 per scene, about 60% of them on the edges of the stove block and counter. On the pan:

  | Part | Proposals |
  |---|---|
  | lid (a loose body) | 15–19 |
  | bottom | 10–11 |
  | handle | 8–11 |

- **Our check:** with our 80 mm fingers (graspnet assumes 60 mm), the left claw reaches 6 handle grasps per scene plus 1 lid grasp. All 10 lifted handle proposals fail (pivot, 9–27° tilt).
- **Lid grasp:** inconclusive, because the claw closed on air.
- **Reading:** a geometry-only proposer has no notion of mass, where it sits, or which parts are attached. Our physics check rejects every proposal it produced here.

## 4. Two-claw lift that works: push to overhang, clamp the far rim

Code: `pan/under_rim.py`. Videos are in `videos/`. Sequence:

1. Left claw pinches the handle at 60%.
2. It pushes the pan 6 cm along the handle axis, through the palm against the handle end. The heavy far side now hangs 4 cm over the stove's far edge.
3. Right claw comes in from the far side and clamps the lid + wall + bottom stack. This also secures the lid.
4. Both arms lift 5 cm and hold for 2 s.

It is a slide, not "set on the counter", because one claw cannot carry the pan and the counter is on the handle side.

Nominal runs, one per placement:

- **Held in 5 of 6:** lift 4.7–5.2 cm, tilt 2.6–4.9°, slip at most 6 mm, lid on, 0 QP fallbacks.
- **Overload (0.6 kg):** off the stove, but sags to 3.5 cm. The controller only knows the empty pan's mass.

## 5. Reliability of the two-claw lift (pre-registered: `robust/PREREG.txt`)

Planning uses the believed (nominal) pose; the true pan is perturbed: ±1.5 cm, ±8° yaw, friction ×0.6–1.2, hidden weight 0–0.4 kg at a random placement.

**30/60 held: 50%, Wilson 95% CI 38–62%** (predicted 60–85%).

| Failure | Count |
|---|---|
| no clean pre-grasp | 21 |
| right claw missed the rim | 4 |
| left claw lost the handle (yaw 7–8°) | 2 |
| slip/tilt | 3 |

- The left pinch at 60% leaves about 5 mm between the palm and the handle end. Any true shift of more than 5 mm toward the claw makes the open claw overlap the handle: with dy above +0.5 cm, 0/22 held.
- Given a clean start: 30/39 held (77%).

Mass curve, far-centre, no pose error:

| Hidden mass | Lift | Result |
|---|---|---|
| up to 0.4 kg | 4.7 → 4.3 cm | held |
| 0.5–0.8 kg | 3.9 → 2.7 cm | sags; slip grows to 8.5° |

No drop at any mass.

Mass known to the controller (`robust_mass_known/`): FL-MPC's payload model is given the true total mass, as if the probe had measured it exactly; far-centre, no pose error.

| Hidden mass | Lift | Tilt | Worst slip | Result |
|---|---|---|---|---|
| 0.2–0.6 kg | 5.0–5.2 cm | 3.0–4.0° | ≤ 9 mm / 3.2° | held |
| 0.7 kg | 5.0 cm | 3.0° | 11 mm / 3.8° | flagged: 0.19 N non-finger contact on the pan during the hold (probably a gripper body; not checked) |
| 0.8 kg | 5.0 cm | 3.0° | 12 mm / 4.1° | held |

So with the mass known, the sag is gone. The mass estimate the probe provides is what keeps heavy loads at the commanded height.

Post-hoc follow-up (not pre-registered; see `robust_L70_NOTE.txt`): the same 60 samples with the left pinch at 70% (`robust_L70/`).

**Result: 41/60 held, 68% (Wilson 95% CI 56–79%).** The pre-grasp failures are gone.

| Remaining failure | Count |
|---|---|
| right claw missed or lost the rim | 7 |
| slip/tilt | 8 |
| left claw lost the handle | 3 |
| still on support | 1 |

Main factor, yaw error:

| Yaw error | Held |
|---|---|
| within ±3° | 18/20 |
| beyond ±3° | 23/40 |

Friction trend: higher friction did worse (μ×1.0–1.2: 14/27; μ×0.6–0.8: 14/15). My unverified guess is that a higher pan–stove friction shortens the push, which reduces the overhang the right claw needs.

## What it means for the thesis

- Covered-pan transport needs two coordinated arms at these sizes. The one-arm versus two-arm choice is a physics question that no geometric proposer answers.
- The open-loop plan-from-belief lift is fragile mainly through grasp clearance against pose error, not through the lift dynamics. This is the hook for the certificate:
  - reject or re-aim grasps whose clearance is below the pose uncertainty;
  - feed the probe's mass estimate into FL-MPC, which fixes the sag.
- Not done:
  - closed-loop (perception-corrected) targets;
  - mass-informed controller;
  - graspnet on the other placements and with the right claw;
  - AnyGrasp itself;
  - carrying the lifted pan anywhere.
