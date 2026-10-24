# Observations, 2026-10-11 → 10-12 (bimanual YOR, sim only)

The lab notebook for the teleop / perception / language push. **[V]** = the owner verified it: re-ran the tests, re-ran a regression, or
read the result files and frames directly. **[A]** = agent-reported, not independently re-run yet. All numbers come from MuJoCo simulation.
Code: branch `perception-teleop-2026-10-12` (commit f04b891, plus the uncommitted work listed at the end).

---

## 1. Infrastructure

### Speed (teleop sim, h5 carry) [V]

| config | ms / tick (budget 20) | real-time factor |
|---|---|---|
| before | ~93-112 | 0.17-0.21 |
| `BIMANUAL_TELEOP_FAST=1` | ~29-41 | ~0.68 |
| + renderer in its own process (web) | n/a | 0.67, 15 fps |

- **Profile:**
  - Collision detection is CHEAP (`mj_collision` 0.17 ms with 1449 geoms), so voxelising MuJoCo collision would not help.
  - The cost was the Python pair barriers (8.7 ms vs 0.2 ms numba), a redundant `mj_forward` (3.9 ms), and the contact solver (noslip=4, ~58 condim-6 contacts, ~3.2 ms of a 4.9 ms step).
- **FAST mode is bit-identical** (100 ticks: qpos/qvel/ctrl identical). [V]
- **The mj_step1/step2 split adds ~nothing.** [A]
- **noslip=0 → 1.7x real time BUT tips the cup in raw seeds 1-2. Keep noslip=4.** [A]

### Browser teleop (keyboard + Xbox via Gamepad API) [V]

- 127.0.0.1-only, plus a token and an Origin check. Watchdog 250 ms; zero command on blur, hidden tab, pad disconnect or deadman release.
- The Xbox pad needs LB held to move. Start/Back = arm/disarm, RS click = camera.
- Input-to-ack latency: 39 ms median.
- Untested in a real browser and with a real pad.
- ROS conventions were taken from the public package sources. Only `keyboard_handler` is installed under /opt/ros/humble.

### Language model service [V]

- Qwen3.5-9B as a long-lived Slurm job on ilab3 (1x RTX 4500 Ada), reached through an NFS file queue (`vlm/daemon.py`, `vlm/service.py`).
  - Loaded in 15 s.
  - Image question 7 s wall; spec authoring 30 s wall.
  - User-approved scope: ilab3 1 GPU, offline weights, no hosted APIs.

---

## 2. Zonotope safety

- **Arm-only zonotope rows cannot protect the tray from the base.** [V] The kinematic base path carries the tray 106 mm into the bottle.
- **Base-twist zonotope CBF (`zono_rows/base_filter.py`)** [V], 24-run sweep:
  - Base filter on: ball kept 12/12, 55.6-55.8 mm clearance, 0 N, tilt 0.6°.
    - Identical across arm options: the base stops before the arms engage.
  - BUT the base stalls at 40% of the path. **The guard stops; it does not go around.**
  - Base filter off: ball lost 9/12.
- **Selective slack hazard** [V]: with track + selective slack and no base filter, the tray FLIPPED (147°, 1224 N, 123 unrecovered steps), vs 103 N with uniform slack.
  - The agent's claim "selective only changes the slack spread" was FALSE.
  - Root cause [A]: the hard tilt rows plus an accumulated tracking error. Selective slack released the arm early → 8-13 rad/s → infeasible → the a=0 fallback kept the velocity.
  - Fix `--ref-clip 0.25 --brake`: m64 now ≤17°, 64-75 N, 0 unrecovered. m8 not fixed (use ≥32 rows). [A]
- **Zonotope rows speed:** rows + base filter went from 6.3 to 2.3 ms median, still exact (2.5e-13). [V tests; timing A]
- **Flagship with zonotope guards** [V]:
  - Base + arm rows: done in 126 s, 0 beads, 0 contacts.
  - 1280 FL-MPC calls had rows, 0 violated, BUT the max row value was −19.7: **the arm rows never came close to binding** (the base guard stopped at 27% first).
  - The tray was placed 0.36 m from the planner's spot. The placement stand moves under any spot, so `placed_on_stand` is trivially true.

---

## 3. Grasp

- **Wrist-camera friction grasp** [V]: 30/30 succeeded on the first attempt (depth-only).
  - Grasp time 13.2 s median; slip ≤3.1 mm; grip ≥23.9 N; 0 beads.
  - Offsets tested only up to 2 cm / 3° (the 3.5 cm / 6° capture limit is untested).
- **Model assumptions (friction mode only):** finger kp 200→2000 (forcerange ±20 unchanged), finger armature 0.3, stiff pad solimp, 12 mm handle strips, counter shelf lowered 6 mm, dynamic base.
- **The left wrist camera is blind at the grasp pose**, so the final insertion is dead reckoning.
- **Wrist cameras:** they are in the UNMODIFIED YOR GitHub MJCF (`left/right_arm_egocentric`), but the YOR code has no driver. Confirm on hardware. [V]
- **Proprioception-only grasp** (`BIMANUAL_PROPRIO_ONLY=1`) [V: 51 tests pass; the 10-09 carry regression is identical; the bad-grasp JSONs show proprio = privileged outcome in all 4 pairs]:
  - Verify = finger gap + effort (noise + quantisation). Payload from joint torque: total error +2.4..+5.8%, but the L/R split is lopsided.
  - FK tray error 7-17 mm.
  - 11 runs, including deliberately bad grasps: 0 false-verify, 0 missed-verify.
  - Still privileged: the operator ALIGN cue (wrist cameras blind at the hold geometry) and the weld mode.

---

## 4. People

- **Person positions were PERFECT ground truth** (`sim.set_people` → `sim.people_state`). The forward camera never saw the walker, and walkers yield. [V]
- **BUG: the h5 walker never moved** [V]: its first track segment is stationary, and the clock only advances when |v|>0. So the old "raw 1/9 person contact" was the STANDING person.
- **Moving walker, truth positions, old shield:**
  - Root cause [V numbers, A timeline]: the robot is squeezed between the standing person, a wall and the walker. The needed lateral retreat is ≥0.27 m/s but the wall allows ≤0.14 → infeasible → brake.
  - A yielding walker → deadlock. An oblivious walker walks through the stopped robot.
  - NOT a tray/arm-extent problem: they are never more than 1.2 cm closer than the footprint.
- **PEOPLE_V2 shield** (predictive rows, relaxed tier, social yield / back-off), 9-seed sweep, 27 episodes per cell [V]:

| shield + people input | contact episodes | finished | spill |
|---|---|---|---|
| old + truth | 15 | 13 | 1 seed |
| **new + truth** | **0** | **24** | 0 |
| old + depth tracker | 12 | 16 | 0 |
| new + depth tracker | 3 (side crossings, outside the forward FOV) | 11 | 1 seed |

- **With the camera, perception is now the bottleneck:** FOV (no side/rear sensing), and the uncertainty margins make the robot too cautious.
- In progress: tracker in the live loop, look-before-move, conformal margins.

---

## 5. Language layer (house/lang) [V tests + JSONs + marks image]

- **Grounding (set-of-marks):** 62.5% of marks named correctly; 54/46% by distinct object.
  - The marks are over-segmented furniture. The lamp is missed.
  - **Self-reported confidence is a constant 0.9, so it carries no information.**
- **Spec authoring:** 3/3 compile and pass the semantic checks. The adversarial instruction needed one repair round.
  - It changes behaviour: a 15 cm bottle pad cuts the filtered placement progress from 26% to 2%.
- **KEY: the VLM friction prior is confidently wrong.** mu ∈ [0.30, 0.68] (twin image [0.42, 0.81]) vs sim truth 0.25.
  - Used raw, it would raise the shield's spill budget about 20% above what the cup can take.
  - **Owner fix:** `mu_policy="conservative"` (default). An unverified VLM prior may only WIDEN [0.18, 0.35], never raise mu_lo. Narrowing needs probe/BOED evidence.
  - Caveat: sim materials are flat colours, so mu is not visually grounded. A fair test needs textures that correlate with mu.
- **Latency:** model time 35-55 s per mission (grounding 21, authoring 9-30, mu 4.5); wall time is 2-3x higher through the queue.

---

## 6. Perception audit and object-centric state

- **Ground truth still in the control path** [V spot-checked]:
  - shield tray/cup (`shield.py:112/353/355`);
  - people_state;
  - the sim-built occupancy grid;
  - the mocap base pose;
  - base_filter tray/ball (`base_filter.py:246`, now FK under the proprio flag);
  - grasp contacts and masses (now gap/effort/torque under the flag);
  - flagship placement `xpos[tray]` (partly);
  - zed_demo self-mask from gt seg.
- **Already honest:** encoders with noise, stereo zonotopes, arm rows from FK, the people depth detector, the grasp depth estimator.
- **Object-centric state** (`house/ocs/state.py`, owner design):
  - one `WorldState` per tick;
  - per-object set uncertainty (ir.Unc) for safety, plus a Gaussian for estimators;
  - parent frames;
  - Belief intervals;
  - per-field provenance, with `privileged()` / `assert_honest()`.
  - Experiments E1-E4 are running (E4 headline: the false-safe rate of the VLM spec on perceived vs true state).

---

## 6b. Localization and perceived map (house/loc)

Verified: 16 tests pass; the 10-09 carry regression is identical; the carry JSONs match the report; 0 wall-contact ticks in all 24 smoke runs.

**Pose estimator.** The parameters are ASSUMPTIONS (`LocCfg`):
- wheel odometry with 1.5% scale error, heading bias and slip events;
- ZED VIO model: 30 Hz, 60 ms latency, drift 2 cm/√m, relocalization, dropouts;
- complementary fusion.

ATE is 4-8 cm rms over the carry. [A]

**Perceived wall map** (a mapping pass with head-ZED depth). [A]
- Mapping poses are ASSUMED to be 0.5x the online error, standing in for SLAM refinement. With that map: recall 100%, 0 false-free cells near the route, EDT error p99 5.7 cm.
- **At 1.0x pose error the map blurs badly:** recall 46%, EDT p95 16 cm, and one stress run reached 1.0 cm true wall clearance.
- So the honest result depends on the SLAM-refinement assumption.

**Smoke, seeds 0-1:**

| condition | carry min true wall clearance [V] | stress (operator drives into walls) [A] |
|---|---|---|
| truth pose + true grid | 25.7 / 23.0 cm | 9.9 / 10.3 cm |
| est pose + true grid | 24.7 / 28.0 cm | 6.4 / 6.3 cm |
| est pose + perceived grid | 24.0 / 20.0 cm | 9.0 / 5.0 cm |
| est + perceived + 1.5σ adaptive margin | 26.0 / 23.0 cm | 17.0 / 10.6 cm |

The stress JSONs have no clearance field, so the stress numbers are agent-reported.

**Zonotope base filter with the estimated pose:** 9 seeds, min true clearance 35.2 / 58.1 / 77.6 mm (min / mean / max) vs 55.8 mm with truth. 0 contacts and the ball kept in 9/9. Pose error moves the clearance by about ±21 mm. [A]

Not yet on the estimated pose: the people rows and cup rows (relative geometry). The mapper has no robot self-occlusion.

## 7. Process lessons (gotchas)

- **Contaminated regression reference:** a reference captured while another agent was mid-edit looked like a regression. Regress against ARCHIVED pre-change JSONs (`teleop/carry_9seeds_summary.json`).
- **`flagship --perception live` overwrites the shared perception cache:** the 10-10 pickles were replaced. Always pass `--out <new dir>`.
- **Counter perception is not repeatable across runs:** 16 / 10 / 26 / 25 zonotopes for the same seed. Cause unknown.
- **Agent claims that were wrong and caught on verification:**
  - "selective slack only spreads slack";
  - my own guesses: "lifted rows cause the flip"; "tray/arm extent causes the people contacts".
- **The QP 'feasible' flag is not proof the constraints hold:** always measure true distances.

## Pending
- P1: people tracker in the loop.
- P3: localization + perceived map.
- OCS experiments E1-E4.
- Native OCS consumption in shield/zono rows.
- The full-perception switch + an end-to-end coffee mission.
- Commit of the uncommitted work: lang, ocs, loc, people_track additions, grasp proprio, vlm daemon/service.

## 8. Where each tracked state comes from today (owner-verified from code, 2026-10-12)

| tier | inputs |
|---|---|
| REAL stereo (rendered L/R pair → SGBM) | flagship desk/counter object snapshot only (2 captures per mission) |
| Ray-cast depth + stereo-law noise model (depth_sim) → our perception | people tracker (head 160x90), wrist cameras (grasp), wall mapper |
| Noise models on true state | encoders, finger gap/effort, joint torque (legitimate proprioception sim); wheel odometry and ZED VIO (a statistical model, no images) |
| Raw simulator state in the control path (default) | shield people (tracker behind a flag), shield cup/tray spill rows, base pose and walls (perception behind flags), operator grasp-align cue |

**User requirement (10-12):** the shields must use states tracked from the stereo cameras. Plan:

1. house/stereo: a continuous head-ZED stereo sensor (rendered pair → SGBM, separate process) as a drop-in depth source for people, the mapper, and a new head-stereo cup/tray estimator, writing into OCS with Source.STEREO. Plus a study calibrating the noise model against SGBM, including the textureless-wall case. Agent launched.
2. After the people-in-loop and OCS agents finish, the shield reads ONLY the OCS WorldState, behind one full-perception switch with `assert_honest()`, followed by an end-to-end coffee mission.

## 9. GPU acceleration plan (user-approved 2026-10-12)

**Hardware.** RTX 2080 Ti (sm_75). torch 2.12+cu130 and numba.cuda work. OpenCV is CPU-only. No JAX/MJX/Warp/CuPy.
- GPUs 0-3 belong to the jenga project; bimanual uses 4-7. Right now 4 and 7 are saturated by another user, so **use 5/6**.
- Earlier agents ran on GPUs 1-3. That is corrected now.

**Startup vs per-call overhead (user question).**
- One-time costs (weights, CUDA context, JIT/engine build, first inference) are paid in a WARM-UP phase. The robot does not move until every component signals READY.
- Per-call kernel-launch, transfer and sync costs recur every step. They are negligible for big perception networks but dominate the 1-3 ms QPs, so the **FL-MPC / shield / zonotope QPs stay on the CPU**.
- MuJoCo physics stays on the CPU (single scene + noslip). GPU physics only pays off for batched sweeps.

**Workstreams launched:**
- **FoundationStereo + FoundationPose (model-free):** user-approved downloads (NVlabs repos + linked weights + PyPI/conda) into a separate env, /data/local/ss5772/conda_envs/foundation, with models in /data/local/ss5772/models. Out-of-process backends with warm-up/READY. Pose does 6D tracking of the tray and cup, onboarded from robot-captured reference views.
- **gpusense:** EGL depth buffer + the same noise model in torch, replacing CPU mj_ray; GPU log-odds map + EDT/gradient for the shield walls; warm-up/READY; benchmarks.
- **GPU MPC survey** (user saw "a GPU-based MPC"): candidates cuRobo, STORM, MPCGPU, ReLU-QP, DIAL-MPC, Hydrax, pytorch_mppi. Fit criteria: robust rollouts over the mu interval, non-convex specs, contact-rich prediction, raw speed.

**GPU MPC survey result** [A: Haiku agent, from READMEs/abstracts; not re-verified by the owner]
- Worth trying:
  - GATO (MIT, CUDA C++, batched constrained SQP): robust MPC with one batch item per friction value. Arm-only.
  - Hydrax (MIT, JAX/MJX sampling + risk measures): no hard constraints; we have no JAX.
  - pytorch_mppi: quick to prototype.
  - TurboMPC (MIT, JAX SQP with inequality constraints; no timing data).
- Poor fits: cuRobo (no verified 200 Hz MPC), STORM (stale; constraints as costs), MPCGPU (no constraints), ReLU-QP/cuOSQP/cuPDLP (no gain at our QP size), MJPC (CPU), DIAL-MPC (quadruped).
- **Conclusion:** none gives hard guarantees, so a GPU MPC belongs ABOVE the hard QP shield, as a planner.

**Decision:** an in-house torch MPPI planner (house/mppi): batched rollouts with mu SAMPLED over the interval (robust, CVaR), non-convex zonotope/wall costs. Goal: fix the "guard stops, never goes around" problem in S2/flagship. The existing hard filters certify every proposal. Agent launched.

## 10. People tracker in the live loop (P1)

Verified: 30 tests pass; the 10-09 carry regression is identical; calib/held-out JSONs present. Smoke numbers are [A], n = 2 seeds × 2 walkers.

- **Flags:** `TeleopSim(people_source="tracker")`, `run/flagship --people tracker [--people-look --people-conformal JSON]`. `sim.people_state` stays truth, for metrics only.
- **MECHANISM** (more important than "too cautious"): a newly born track gets 0.5 m inflation → the QP becomes infeasible → hardware brake → the cup slides → the IR veto latches for 800-1385 ticks → the robot is stuck.
  - Fix: `soft_floor` (`+soft`) caps how fast the inflation can make a row infeasible.
- **Smoke, NEW shield** (contact eps / finished / min TRUE clearance):

| condition | contact eps | finished | min TRUE clearance |
|---|---|---|---|
| truth | 0 | 4/4 | 0.24 |
| depth | 1 | 3/4 | -0.25 |
| depth+soft | 0 | 4/4 | 0.19 |
| depth+look | 0 | 4/4 | 0.26 |
| depth+conf (α .05) | 1 | 2/4 | -0.34 (worse than k-sigma) |
| depth+look+conf20 | 0 | 4/4 | 0.18 |

  - The look-glance costs 6-8 s and raises the walker-in-FOV share from 24% to 42% at best.
- **Conformal coverage on held-out seeds:**

| α | coverage |
|---|---|
| .05 | .927 capped / .941 uncapped |
| .10 | .892 |
| .20 | .779 |

  - The old k-sigma covered only .849 on the same data.
  - The large radius at 1-1.7 m comes from the detector's partial-view bias, so FIX THE DETECTOR BIAS to make conformal pay off.
- **9-seed sweep9b launched** (truth, depth, +soft, +look, +look+soft, +conf, +look+conf, +look+conf20) → /data/local/ss5772/molmospaces/people_shield/sweep9b.

## 11. Object-centric state, round 1 (house/ocs)

Verified: 13 tests pass. E4 tables read directly from e4_constraints.md.

**Producers:**
- **Honest:** base (odometry + encoders via loc), tray (arm FK + nominal grasp offset), cup (wrist RGB red blob → ray onto the FK tray plane; height/upright are priors), people (head depth + KF), obstacles (stereo zonotopes).
- **Privileged:** names (gt; VLM grounding not wired); the people re-cast and the stereo capture use the true camera/base pose (inherited sim shortcuts).

**Results:**
- **E1** [A, numbers in e1_equivalence]: shield inputs are bit-identical via OCS(truth); the carry through the OCS proxy reproduces direct runs.
- **E2:**
  - Calibrated sets: tray 0.94, cup 0.96, base 0.998 (k·σ, k≥2).
  - Engineering-prior sets UNDER-COVER: tray 0.69, cup 0.006.
  - Pooled conformal fails for the base (per-episode odometry bias: 0.685), so the base uses k·σ.
  - Worst: occluded persons, coverage 0.654 and a track on only 46% of occluded ticks (coasting drifts up to 0.8 m). The flagship tilt bound covers 0.80. Stereo sets contain only 20-44% of the desk/monitor front surface (partial views).
- **E4** (spec G(inside ∧ upright ∧ clear laptop ∧ clear person); laptop/person were substituted by existing entities):
  - Instantaneous: 0 false-safe in 6433 violated h5 ticks. But perception certifies `true` on only 2.3% of satisfied ticks: very conservative (3 cm cup slack vs a 5 cm cup set).
  - **Predictive G[0,1 s]: the `inside` atom is false-safe on 18/507 = 3.55%** [V]. The rollout's constant-twist model misses the cup sliding after braking. The conjunction is still 0 false-safe only because the other atoms came out `unknown`.
  - **Lesson:** the rollout must use the cup slip dynamics (slip_on_support / reach.py), not constant twist. Sets must shrink (better cup sensing) or the certifier stays useless.
  - Flagship E4 is uninformative (person→bottle substitution, violated on every tick).

## 12. Pose shortcuts in perception (owner-verified in code) + neural person detector (user-approved)

- **people_track/detectors.py:**
  - Background subtraction re-casts rays through the EXACT house model from the TRUE camera pose (l.130-134).
  - Detections go to the world frame with the TRUE camera pose (l.213).
  - The optional map filter uses `sim.occ` (the sim-built map).
- **flagship/perception.py:** the stereo twin is rendered at the true base pose (fine: that is the sensor's real position), BUT the cloud goes to the world frame with the TRUE camera pose (l.158).
- **Rule:** rendering from the true pose is physics. Interpreting the image with the true pose or map is a shortcut.
- **Fix (agent launched):**
  - people background from the PERCEIVED map + ESTIMATED pose;
  - stereo zonotopes transformed with the estimated pose and inflated by pose uncertainty;
  - a map-free neural person detector (permissive licence, separate env `/data/local/ss5772/conda_envs/detect`, out-of-process with warm-up/READY).
  - Caveat to measure: sim people are capsules/mannequins, so a COCO detector may not fire. No asset downloads without the user's OK.
- existing people_track files are frozen until sweep9b finishes (51/216 runs done at launch).
- **User approved (10-12)** a human-mesh download if needed. Scope: one permissively licensed (CC0/CC-BY/MIT/Apache) mesh from its official source, no login or non-commercial licences (no SMPL/Mixamo), stored in /data/local/ss5772/models/humans/, used as a VISUAL mesh only (capsule collision unchanged), flag-gated so defaults stay identical.
- **What MolmoSpaces offers for people** (owner-verified in /data/local/ss5772/molmospaces/src):
  - NO people tracker.
  - It does package **Microsoft Rocketbox humans (MIT)** as MuJoCo assets in `static` / `skinned` / `articulated` variants (version 20260817), lazily downloaded via its resource loader. Not yet on disk. These were sent to the detector agent as the preferred mesh source.
  - Per-house `_metadata.json` gives per-object category/room/static flags: ground truth for scoring VLM naming, and a hazard menu.
  - Per-house occupancy maps exist too, but they would be a PRIVILEGED map in the control path, so we keep the perceived map.
  - Our current people are NOT from MolmoSpaces: the h5 people are mocap capsules, and the office mannequins are our own primitive bodies (`office/scene.py _mannequin`, 1.70 m, coloured shirt).
- **User decision (10-12):** the WALKER uses the ARTICULATED Rocketbox human: visual only (contype/conaffinity 0, capsule collision unchanged), with a kinematic walking gait phase-locked to the walker's speed and a standing pose on pause/yield. Static mannequins may use the static variant. Flag-gated, default off. The cost (nv, mj_step, render, RTF) will be measured.

## 13. People 9-seed sweep (sweep9b) [V: owner per-episode recount]

NEW shield, 27 episodes per cell (9 seeds × 3 walker types).

| people input | contact episodes | finished | spill episodes | median time (s) |
|---|---|---|---|---|
| truth | 0 | 24 | 0 | 22.5 |
| depth tracker | 3 | 11 | 3 | 28.5 |
| + soft floor | 4 (report: 3) | 27 | 0 | 27.6 |
| + conformal (α .05) | 5 | 25 | 0 | 27.9 |
| + look | 0 | 15 | **9 (36 beads)** | 28.0 |
| **+ look + soft** | **0** | **27** | **0** | 30.3 |
| **+ look + conformal** | **0** | **27** | **0** | 28.7 |
| **+ look + conformal α .20** | **0** | **27** | **0** | 27.9 |

- **Conclusion:** the camera tracker reaches 0 contacts, 27/27 finished and 0 spills ONLY with look-before-move PLUS a non-explosive margin (soft floor or conformal). That beats the truth tracker's completion (24/27), at a cost of +5-8 s.
- **Look alone causes spills:** glances under a stuck/braking state shake the cup.
- **Soft or conformal alone still collides:** the forward FOV misses side crossings.
- **Recommended people default for full perception:** `depth+look+conf20` (fastest of the safe three).

## 14. GPU depth sensing and distance field (house/gpusense) [V: tests pass; mapping-equivalence JSON exact]

- **Depth equivalence:** GPU depth vs ray-cast is within 1 mm on 98.5% of head-camera interior pixels and 99.9% of wrist pixels [A].
- **Mapping:** the log-odds update and the EDT are exact vs the numba/scipy versions [V].
- **Steady-state ms per frame** [A]:

| item | CPU | GPU |
|---|---|---|
| people cam | 84 | 6.2 |
| wrist | 135 | 5.6 |
| head 640x360 | 1186 | 24 |
| map update | 238 | 22 |
| EDT | 41 | 6 |

- **Warm-up:** 10-14 s to READY (spawned processes, imports included).
- **RTF:** async GPU sensing costs the sim thread ~1.2 ms/tick. The absolute RTF was low (0.22) because the machine was overloaded (load 64-87 during the sweep).
- **Flag:** `BIMANUAL_GPUSENSE=1`; hook lines are in `gpusense.HOOKS`.
- **Not done:** the wrist RGB path.

## 15. Paper draft (2026-10-12)
- 4-page draft written and delivered: docs/paper/draft_2026-10-12/main.{tex,pdf}. Title: 'Confidently Wrong: Keeping Real-Time Bimanual Safety Sound When VLM Priors and Perception Are Miscalibrated'.
- Central claim: language may specify constraints and ENLARGE uncertainty sets; only calibrated evidence may SHRINK them; margins must be honest AND continuous.
- Coverage numbers are stated in the ROBOT frame (the frame the cup-on-tray constraint uses): engineering cup 0.006 / tray 0.69 vs calibrated 0.96 / 0.94. In the world frame they look fine (0.90 / 0.999) because base-pose uncertainty hides the relative error.
- † marks citations still to verify (Phys2Real, ActSafe, Lu-Chen L4DC, polynomial annihilation). The literature search is incomplete.

## 16. Pose-honest perception + neural person detector [V: 47 tests, regression identical, JSONs + a clip frame checked; detector numbers A]

- **People, perceived map + estimated pose** (capsules):
  - Ray-casting against the perceived map alone gives **1.35 false detections/frame, all at walls, and 34/47 false track IDs**.
  - Adding the perceived 2-D occupancy-grid filter gives 0 false detections, recall 0.99, tracker RMSE 4.5 cm including localization error.
- **Stereo zonotopes with the estimated pose:** fine in RELATIVE (capture-frame) planning. But absolute pose error after 12 m is mean 10-12 cm, p95 24 cm, and **the localizer's 2σ covers only ~70% (overconfident on long routes)**.
  - Inflating by the absolute uncertainty makes the desk plan infeasible (claimed clearance 2.5 cm).
  - Lesson: re-sense near the target and plan in the capture/robot frame. Calibrate the localization uncertainty as a function of distance travelled.
- **Neural detector:** Faster R-CNN R50-FPN v2, COCO, BSD-3, out-of-process, warm-up 13.7 s, ~43 ms/frame.
  - **0 detections on capsule people.**
  - On the articulated Rocketbox human (MIT): recall 0.93, 0.08 false/frame, 0 false tracks, centroid error ~19 cm (the person-size prior is suspected).
  - The depth-blob detector on the human mesh gives 0.8-3.2 false/frame.
  - The human mesh adds 0 DOF and no measurable mj_step cost.
- **Decisions (owner):**
  - The human mesh becomes a run flag (default off).
  - The full-perception preset uses neural (human mesh) / depth_perc+grid (capsules).
  - Next: a fusion test, a centroid-bias fix, and a 9-seed closed-loop sweep with the neural detector + look + conf20 (recalibrated).
- **Agent note:** it ran one read-only `git diff --stat` despite the no-git rule. Disclosed by the agent; no state changed.

## 17. GPU sampling planner above the hard filters (house/mppi) [V: 9 tests pass; S2 JSON read; h5/robustness numbers A]

- **Design:** torch MPPI / CEM / predictive sampling, captured as a CUDA graph.
  - Mu is sampled over [0.18, 0.35] with a CVaR spill cost.
  - Non-convex clearance from a GPU table of the perceived zonotopes, plus the wall EDT and person discs.
  - Its output passes through the UNCHANGED BaseFilter/shield.
- **S2 (bottle):** the base filter alone reached 44% progress (stalled), with 92% of steps intervened.

| | progress | ball | min true clearance | bottle force | steps intervened |
|---|---|---|---|---|---|
| planner + base filter (every K, CEM, PS) | **100%** | kept | 76 mm | 0 N | ~31% (4x smaller mean) |

  - It goes around the bottle (0.285 m lateral shift). **This fixes "the guard stops instead of going around".**
- **h5 carry, 10 seeds:** shield-changed ticks 39% vs 84%, min person distance 0.32 vs 0.21 m, 0 spills either way. NOT faster: dock 6.9 vs 6.6 s, task 15.2 vs 13.4 s (operator docking noise dominates).
- **Robustness, planner ALONE (shield bypassed), 3 seeds per cell, spills:**

| truth mu | robust CVaR | nominal 0.35 | nominal 0.25 | raw operator |
|---|---|---|---|---|
| 0.18 | 0/3 | 2/3 | 1/3 | 3/3 |

  - Evidence for the paper's parametric claim: planning with an overconfident nominal mu spills; robust-over-interval does not.
  - With the shield ON, every planner had 0 spills, so the shield decides safety.
- **Timing:** ms per plan, after warm-up (CUDA init 0.3 s, graph capture ~1 s, torch.compile 30-90 s once).

| K | graph | compiled |
|---|---|---|
| 256 | 9.5 | 4.5 |
| 1024 | 15.3 | 5.0 |
| 4096 | 41 | — |

- **Hooks:** smoke_s2 `nominal_fn`; teleop/run.py `cmd = hook(sim, cmd)` before `shield.filter`.

## 18. Continuous head-ZED stereo (house/stereo) [V: 5 tests; README tables read; closed-loop numbers A]

- **Pipeline:** a textured twin rendered at 640x360 in a separate process (StereoLink, READY gate) → SGBM or FoundationStereo (one backend interface) → people tracker / mapper / new payload estimator → OCS → shield via `ShieldView` (no shield edits).
- **The default depth noise model UNDERESTIMATES SGBM by ~1.4x.**
  - Refit: sigma_d 0.176 px, floor 4.8 mm, p_drop 0.106.
  - Not capturable by a simple noise model: depth-edge outliers (9.7%), slope-dependent error, a negative bias growing with range, and poor fill below 0.5 m.
- **Textureless surfaces:**

| twin | fill | outliers > 10 cm |
|---|---|---|
| speckle | 0.89 | 5.8% |
| flat walls | 0.49 | 21.5% |
| all flat | 0.29 | 39.5% |
| FoundationStereo, all flat | 0.99 | 26% |

  - Flat walls give WRONG depth, not just holes.
- **Tracker on SGBM:** recall and RMSE are the same as ray-cast, but **false tracks are ~40x higher**: 3.3 simultaneous, and 0.54 with min_pts=40.
- **Mapper:** spurious cells 0.06 → 0.28, and 0.45-0.50 with flat walls.
- **Payload estimator on SGBM:** tray tilt 0.24°, cup 1.8 mm, cup tilt 2.7°. Sets cover 99.9-100% at 640x360 but only 0.72-0.78 at 320x180.
- **Cost:** 125 ms/frame (640x360, loaded). In a separate process the RTF is unaffected (0.345 vs 0.348).
- **Closed-loop smoke** (shield fed from the stereo WorldState, 3 seeds × 30 s): 0 spills, 0 bumps, like truth.
- **Hook (owner applies at integration):** `stereo = SL.attach(sim)`; per tick `ws = stereo.step(sim)` → `shield.filter(ShieldView(sim, ws), cmd)`.
- **BUG FOUND:** `ocs/util.py` CUP_R/H = 0.045/0.115 (flagship cup), but the h5 cup is smaller (~0.032/0.08). The OCS E2/E4 h5 numbers (and the paper's 2.3% / 3.55% / coverage claims) may change. Fix + rerun launched → ocs/v2_cupfix.
- **Housekeeping:** stereo temp twin `.mjb` caches use 3.4 GB in /data/local/ss5772/molmospaces/percept/stereo. The agent's delete was blocked by permissions; left for the user to decide.

## 19. FoundationStereo + FoundationPose [A; README present, processes confirmed stopped]

- **Env:** `foundation` (torch 2.8+cu128, TRT 10.13); out-of-process (Unix socket + shared memory), warm-up 8-37 s.
- **FoundationStereo** (2080 Ti, loaded):

| config | Hz |
|---|---|
| ViT-L 640x360, TRT FP16 | 3.3 |
| ViT-L 320x180 | 10.6 |
| ViT-S 320x180 | 19 |

  - Depth error median 1 mm at 0.3-2 m.
  - **Flat walls: ~2 cm error, 0.5% invalid, vs SGBM 76% invalid.**
- **FoundationPose model-free:**
  - cup: 2 mm / 1-2°, even at 240°/s;
  - **tray: fails under fast motion** (20-65 mm, 36-93% frames lost);
  - onboarding 107-200 s (warm-up phase).
- The first-frame register mask in the benchmark was privileged (rendered segmentation). The onboarding masks were not.

## 20. Human-mesh detectors [V: table JSON spot-checked; agent regression identical]

| detector | false det/frame | centroid error | tracker RMSE |
|---|---|---|---|
| depth_perc+grid | 0.82 | 19.5 cm | 20 cm |
| neural | 0.08 | 19.3 cm | 20 cm |
| fusion (neural boxes gating depth) | 0.00 | 19.2 cm | 20 cm |

- Recall ~0.93 for all three.
- **The centroid bias is the r=0.22 person-size prior** (detections sit 13-15 cm behind the true axis). Visible-surface centroid + 3 cm gives fusion 9.3 cm centroid / 9.7 cm tracker RMSE. Neural alone reaches 17 cm (its boxes include background).
- **Flags:** `--human-mesh articulated`; `PerceptionCfg.preset("full")`. Conformal fitted on seeds 100-107 (in-sample only).
- **Untested closed loop:** ObservedMap.stamp and the shield's people logic now use the estimated pose.
- **9-seed sweep (neural / fusion + look + conf20, human mesh) launched** → percept/people2/sweep_hm.

## 21. Cup-size fix: corrected numbers and the episode-shift finding [V: v2_cupfix E2/E4 files read]

- **The h5 AND flagship sims use the SMALL cup** (r 0.0325, h 0.080). The 0.045/0.115 "paper cup" exists only as the flagship stereo-hull model.
- **E2, robot frame:** the engineering cup set was 0.006 → **0.852** visible / 0.000 occluded. The old 0.6% was an ARTIFACT of the wrong cup size. Calibrated: 0.959 → 0.976 visible. Tray unchanged (0.69 / 0.938).
- **E4 h5:**
  - `inside` atom false-safe: **1263/2507 instantaneous, 255/500 predictive**, concentrated in held-out seed 4 (1263/1269, 251/253).
  - Mechanism: the cup slid 3.1 cm while occluded; the held estimate was frozen with a 0.4 cm conformal radius learned on episodes with no occluded slip; the true margin was −2.3 cm vs perceived +0.8 cm.
  - The full conjunction is still 0 false-safe, only because the person/laptop atoms were 'unknown' (luck).
- **Lesson** (now in the paper): per-tick coverage ≠ episode safety; an episode-level shift defeats pooled calibration. While unobserved, sets must GROW with physics (the slip budget from the μ interval).
- **Paper v3:** the abstract, Table 1, the State paragraph and a new "Model class and episode-level shift" paragraph are corrected; the stale 0.6% / 2.3% / 3.55% claims are removed. v1/v2 are kept as main_v1.tex / main_v2.tex.
- **Open fix:** stereo/payload.py and live.py still call `U.cup_shape()` (flagship cup) for the h5 cup.
- **Recommended:** use the sim cup for flagship recordings (`OCS_FLAGSHIP_CUP=sim`).

## 22. Grasp v2 [V: 27 tests pass, weld regression identical; numbers A]

- **Time to verified grasp:** 13.5 → **5.6 s** median. Root cause of the slowness: unmodelled 0.3 Nm joint friction left the wrists 0.13 rad short; fixed with friction feed-forward.
- **Final pose error:** along-edge 11 → 2 mm, rotation 2.5° → 1.1°.
- **Capture map** (0-4 cm × 0-8°, 3 seeds per cell): **75/75 first-attempt successes**, look on or off. The lateral look only cuts along-edge error (at 4 cm: 4 vs 24 mm).
- **Place/release:** touchdown detected from joint torque 0.08-0.14 s after contact; 4/4 OK, 0 beads.
- **Proprio-only:** grasp 4/4, but the payload estimate is off by −15 to −23% (worse than the earlier 2-6%).
- **Flags** (BIMANUAL_GRASP_FAST/LOOK/FUSE/PLACE, FRIC_FF; default ON, friction mode only; ALIGN_CUE=cam off by default).
- **NOT done:** flagship `--grasp-mode friction` (RELEASE must wait for place, 5-8 s), the web assist-button smoke, video.
- **FoundationPose hook:** `grasp/fuse.py` (ExternalTrayPose).

## 23. SE(3)-anchored cup tracker; seed-4 mechanism corrected [V: v3_wrist / v3_anchored E4 files read, 24 OCS tests pass] (2026-10-14)
- **Correction to §21:** the cup did NOT drift while hidden on seed 4.
  - What happened: a hard stop (base 0.7 → 0 m/s in ~0.2 s; 4-5 m/s² at the tray, over the slip threshold μg ≈ 2.5) slid the cup from y = 0 to −7.06 cm.
  - The slide carried it out of the right wrist view. The last blob fix (t = 4.58 s, −3.96 cm) was taken MID-SLIDE, and the cup was never seen again.
  - The old producer froze that fix in a 0.4 cm box.
  - Odometry sees the stop (estimate 4.7-5.1 m/s² vs truth 4.0-4.9).
- **Shield on:** the recordings ran with the shield ON ("Shield(truth people)"). A person was 0.65 m away at the stop, so it was likely the shield's person/brake fallback, which ignores the ~1.4 m/s² spill budget. Not confirmed: the recordings do not log the shield mode.
- **New module `ocs/anchored.py`:** general tracker = object-frame zonotope + measurement sources + Coulomb-slide hypotheses (μ ∈ {0.18, 0.35} × 8 disturbance directions + nominal).
  - Velocities persist across fixes; the hypotheses are driven by odometry + FK only.
  - The blob source reproduces CupWristProducer exactly while visible (test).
- **Hidden-cup results:**
  - Seed 4: error 3.14 → 0.34 cm, coverage 0 → 100%, but the box is ±13 cm (wider than the tray). Conditioning `k_fals` 1-2 does not tighten it at the blob's ±1.7 cm noise.
  - Seed 2: error 0.40 → 0.13 cm, box ±3.6 cm.
- **E4 `inside`, test seeds 3-5:**
  - Instantaneous false-safe: wrist **1263/1269** → anchored **0/1269** (all now "false").
  - Predictive (1 s) false-safe: 251/253 → **8/253**.
  - Certify-when-true: 93.7% → 93.4%.
  - The 8 predictive misses are all in the 1 s BEFORE the stop (seed 4 t = 3.7-4.4, seed 2 t = 4.3-4.5). The rollout assumes the operator holds the twist (±0.6 m/s²) and cannot anticipate the shield/operator stop. Fix belongs in the rollout (it should include the shield's own brake profile), not in perception.
- **Head camera:** the recorded head frames contain 0 cup points (seeds 2 and 4, all 300 frames). Only the wrist cameras ever see the cup.
- Runs: `/data/local/ss5772/molmospaces/ocs/v3_wrist`, `v3_anchored` (`exp_e234 all --cup-tracker {wrist,anchored}`).
- FoundationPose source (`ocs/fp_source.py`, `--cup-tracker anchored_fp`) wired; its measurement pass is being built.

## 24. FoundationPose (model-free) cup source on the torso ZED [V: fp/head npz re-scored; v3_anchored_fp E2/E4 read; 31 OCS tests pass] (2026-10-14)
- **Wrist cameras are unusable for FoundationPose.**
  - Right wrist: 3-9 cm from the cup wall, inside the 0.10 m depth floor → 0/80 frames. A what-if 2 cm floor gives 25%, 8.5 mm, tilt 9.5° (p95 29°).
  - Left wrist: the cup is behind it in 100% of ticks.
  - The recorded LEFT blobs (≥3000 px on 85% of seeds 2/4 frames) are red-threshold false positives on walls; the off-tray gate rejected all of them (0 fixes used left, 6 seeds).
- **Torso ZED (proposed mount, pitch 36.3°)** sees the cup dead-centre at ~0.5 m. The recorded "head" frames had 0 cup points because that feed self-filters the payload.
- **FoundationPose on the head ZED** (640×360, stereo-law depth, non-privileged tray-plane mask, onboarding from 20 idealised ring views):
  - 100% of frames on all 6 seeds, 1-2 registers per episode, track 46-62 ms.
  - Error vs the FK tray frame ~1 mm (p95 < 2 mm); tilt ~2°.
  - It keeps the cup through the seed-4 stop (re-registers once).
- **The FK tray frame is the dominant error now.** The encoder-FK tray (nominal grasp relpose) is 28 mm (max 41 mm) off the true tray before the seed-4 stop, ~0 after. My first estimate of 0.4 ± 1.0 cm was dominated by the quiet tail; the subagent was right.
- **E4 `inside`, test seeds 3-5:**

  | tracker | instantaneous false-safe | predictive false-safe | certify-when-true |
  |---|---|---|---|
  | wrist (old) | 1263/1269 | 251/253 | 93.7% |
  | anchored blob | 0 | 8/253 | 93.4% |
  | anchored FoundationPose | 0 | **0/253** | **55.7%** |

  - Cup-in-tray error p50 0.19 cm (vs 0.60 for the blob).
  - The cost comes from `cup_fp_pos` = 2.85 cm (calibrated): the FK tray offset, not FoundationPose.
  - Next: measure the tray from the same image (cup relative to the visually observed tray, so the common-mode error cancels).
- **Pre-existing calibration bug:** `exp_e234.fit_calib` fits the blob affine/bias on ALL rows with a blob reading, including STALE held readings (seed 2: 1228 frozen rows). This inflates `cup_pos` to 1.73 cm; on fresh rows only it is 0.56 cm. It affects v2_cupfix and v3_wrist/anchored blob numbers. NOT fixed yet (it changes baseline numbers; needs the user's go).
- **Source selection:** AnchoredProducer now uses sources in priority order (FoundationPose first, blob fallback). The first version picked the min radius, which chose the blob on mis-calibrated radii. That run is kept as `v3_anchored_fp_minr_flawed`.
- **Runs:** `/data/local/ss5772/molmospaces/ocs/fp/head/fp_s{0..5}.npz`, `ocs/v3_anchored_fp`.

## 25. Four follow-ups and the combined run [V: all E4 counts re-scored from e234_raw.pkl; 60 OCS tests pass] (2026-10-15)
- **Seed-4 stop = the shield's own `brake_fallback`.** CONFIRMED by an exact re-simulation (base_v error 0.0).
  - At t=4.36 all 3 QP tiers were infeasible (person keep-out).
  - The brake = the hardware slew A_HW=[4,4,8] (teleop/shield.py:457, teleop/base.py:17). That is above mu_hi·g=3.43, so it always slides the cup. The cup-limited decel is 1.41 m/s².
  - The operator was still pushing 0.89 m/s.
- **Calibration stale-row fix** (now the default; `--calib-legacy` reproduces v3 exactly).
  - cup_pos: 1.73 → 0.59 cm. The old cup_pos_occ = 0.40 cm was an artifact (fitted and scored on the same frozen rows); honestly it is 4.6 cm.
  - The new radius under-covers the brake slide: on seed 4 at t=4.62–4.64 a fresh blob is 3.1 cm off. The calibration seeds contain no brake event (exchangeability broken by the episode).
- **Visual tray** (`ocs/fp_tray.py`; head-ZED depth-plane RANSAC + footprint fit; `fp/head_vtray`).
  - Tray position p95: FK 25–34 mm vs vision ~3.5 mm. The FK is right at p50 and wrong during transients.
  - cup_fp_pos: 2.85 → 0.34 cm.
  - Caveat: held-out errors were glanced at during development; redo the tuning blind before quoting.
- **`--pred-contingency hold+brake`:** the perceived pred status is 'true' only if true under {hold twist, shield brake now}. It removes every pre-brake pred false-safe, but costs certification at cruise speed: a 4 m/s² stop always slides the cup past the margin.
- **Live FoundationPose** (`ocs/fp_live.py`, `run_fp_live.py`; lock-step, seed 3).
  - Latency 0.26–0.49 s (render dominates). Coverage 12–30% of due frames vs 100% in replay; same error (~2/3.3 mm).
  - AnchoredProducer ignores measurement age, so the source inflates its radius (19–32 mm).
  - The sim runs 10–13× slower than real time without FP.
  - EGL gotcha: set MUJOCO_EGL_DEVICE_ID (CUDA_VISIBLE_DEVICES does not pick the EGL GPU).

| held-out seeds 3-5, 'inside' | false-safe inst | false-safe pred | certify inst | certify pred |
|---|---|---|---|---|
| wrist, v3 | 1263/1269 | 251/253 | 93.7% | 93.2% |
| anchored blob, v3 | 0/1269 | 8/253 | 93.4% | 93.0% |
| anchored FP (FK tray), v3 | 0/1269 | 0/253 | 55.7% | 56.9% |
| wrist, all fixes (v5) | 6/1269 | 3/253 | 99.6% | 74.4% |
| anchored blob, all fixes (v5) | 2/1269 | 0/253 | 99.6% | 74.4% |
| **anchored FP + visual tray, all fixes (v5)** | **0/1269** | **0/253** | **99.4%** | **70.5%** |

- Runs: `ocs/v4_{legacy,calib}_*`, `v4_brake_*`, `v4_vtray_*`, `v5_all_{wrist,anchored,anchored_fp}`, `fp_live/`.
- Next: gate the brake contingency on the shield's live infeasibility margin (recover pred certification). Optionally a cup-limited brake fallback (a controller change; needs the go).
