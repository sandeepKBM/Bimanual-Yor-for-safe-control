# Testbed candidate c_block_tower: 3 stacked blocks on a plate carried by both arms

Tags: VERIFIED = I ran or measured it. DERIVED = from math. ASSUMED = my choice or an unconfirmed spec.
Code: `/common/users/ss5772/bimanual/world_model/v2/testbeds/c_block_tower/`. Numbers: `/common/users/ss5772/bimanual_data/testbeds/c_block_tower/feasibility.json`.

## Verdict + 5 key numbers

| | |
|---|---|
| **Verdict** | **GO-with-changes, only as a BOED / friction-probing testbed. NO-GO as proof that a learned WM beats physics.** Plate motion plus observed block poses nearly decide the outcome. A 2-line analytic model is 96% right per branch. |
| 1. Nominal failure rate | **47.7%** (88 held-out spill_v2 trajectories, N=3, mu 0.5, plate tilt scaled by 0.35). 86% unscaled. VERIFIED. Tilt scale is a calibration knob. ASSUMED. |
| 2. Savable set | Of pre-failure states: **45.4% savable** (hold-still fails, some 85-set action holds for 2 s), 21.2% hold-still suffices, 33.3% not certified. 451 savable states from 71 failing episodes. VERIFIED. |
| 3. Lead time | Median **5 ticks (0.1 s)** from first-unsavable to failure (range 2-8). VERIFIED. The doomed region is thin. |
| 4. Analytic vs MuJoCo | Quasi-static analytic model: **96.3%** branch accuracy (72 states x 86 branches). It calls **all** doomed states safe (it has no block velocity). Its best-margin pick succeeds on 18 of 18 MuJoCo-savable states. VERIFIED. |
| 5. Speed | **5.4k ticks/s/core** (21.6k physics steps/s), 60 branches/s of 100 ticks, 0.2 s per fully certified state. Restore is bitwise exact. VERIFIED. |

Total compute: about 16 CPU-min on 8 cores. No GPU. No rendering. No hardware or network.

## 1. Failure and margin

Failure event (sticky, checked each 20 ms tick). Any one of these:
1. A block tilts more than 0.6 rad from the plate normal. The 0.6 rad threshold is ASSUMED.
2. A block centre leaves the support face below it. For block 0 the support is the plate.
3. A block drops below the plate.

Margin (VERIFIED in code, `tower_env.TowerEnv.margin`), computed from sim state in the plate frame:

    l = clip( min over blocks of min( (0.6 - tilt_i)/0.6 , support_i ) , -0.4 , 1 )
    support_i = (h - ||c_i - c_{i-1}||_inf) / h        (h = block half-width; block 0 uses the plate edge)

- l < 0 holds exactly when the event has fired.
- The margin is continuous.
- It has the same range and clip as the bead margin, [-0.4, 1].

Mapping to "spill": `not-spill` becomes `not-topple`. `l_t` is stored per tick. `onset` is the first tick with `l < 0`.
The privileged aux head `n, nd` can hold the block centres in the plate frame and their rates. ASSUMED, not built.

Physical thresholds (DERIVED, then checked in 2c) for N=3 cubes-ish blocks (5 x 5 x 4 cm):
- Slide at tilt atan(mu). That is 0.46 rad at mu 0.5.
- Tip at tilt atan(0.025 / (N * 0.02)). That is 0.395 rad for N=3 and 0.303 for N=4.
- Tip at horizontal accel g * 0.025 / 0.06 = 4.1 m/s^2. The pipeline's A_max is 5 m/s^2, so the action domain reaches the failure region.

## 2. Physics representation

### 2a. Reduced analytic model (`analytic.py`)

Plate state: pos 3, v 3, axis 2 (8 dimensions), driven by the same `kin_step` as bead_env.
Block part: observed block centres in the plate frame, frozen (2N numbers). Block velocities do not exist in this model.

Inputs: the 6-D u = [a, w]. For each tick and each sub-stack j (blocks j..N-1, equal mass, COM r_j in plate frame, DERIVED):
- Inertial accel of the COM: `a_c = a + w x (w x rho)`. Angular accel is dropped because u is piecewise constant.
- Support force per unit mass: `f = a_c + g e_z`. Split into normal `f_n` and tangential `f_t`.
- Slide margin: `(mu f_n - |f_t|) / (mu g)`.
- Tip margin: `(hs - ||zmp - c_{j-1}||_inf) / hs`, with `zmp = r_xy - (f_t/f_n) h_j`.

Predicted failure = margin < 0 for K ticks in a row (K = 1 to 8 gave the same result).
Parameters: mu (hidden), geometry (known), N, mass (drops out; see 2c).
Use as a labeller: it needs only the kinematic plate state plus observed block poses, so it runs any action sequence with no MuJoCo.

### 2b. MuJoCo representation (`tower_env.py`)

- Bodies: one plate (freejoint, box 30 x 20 x 1 cm, 5 kg, `gravcomp=1`) pinned each physics step with an inertial carry (qvel from the pose difference, as `pin_cup`). N free boxes, 5 x 5 x 4 cm, 0.06 kg. All values ASSUMED.
- Contacts: mu 0.5 on every geom, condim 3, `solref 0.01 1`. Option block copied from the pipeline: implicitfast, elliptic cone, impratio 10, timestep 0.005 s, 4 steps per tick.
- Speed (VERIFIED): 0.18 ms per tick with no margin check, 0.35 ms with it. A 100-tick branch with early exit runs at about 60 per second per core. A full certification search averages 73 branches and 0.2 s per state.
- Determinism (VERIFIED): restoring a snapshot twice gives bitwise equal margins. A fresh run and a restored run also match bitwise. Snapshot state is 131 numbers and costs 4 us.
- The numba `kin_step` matches `bead_env.kin_step` to 1.1e-16 (VERIFIED, `selftest`). bead_env itself is imported, not edited.
- Static stability (VERIFIED): 2 to 4 blocks sit still for 2 s. Drift is only the 3 mm initial offset.

MuJoCo traps I hit (all VERIFIED):
1. A light plate (0.3 kg) or one without `gravcomp` falls a little inside every step. On a tilted plate that is a fake tangential motion. Blocks crept 4 mm in 20 ticks at 0.12 rad. A 5 kg plate with `gravcomp=1` removes it.
2. `noslip_iterations`, cone type, impratio and solimp changed nothing in the creep test. The cause was trap 1, not the solver.
3. Setting `body_mass` after compile gives wrong dynamics. Set mass in the XML.
4. Equal coplanar boxes give a FatalError (9 contacts, max 8) in MuJoCo 3.9. I taper each layer by 0.3 mm. Branches also catch the error and count it as failed.

### 2c. Reduced model vs MuJoCo: where it is wrong

Ramp test (plate tilted about y at 0.1 rad/s; VERIFIED):

| case | MuJoCo failure tilt | analytic min(slide, tip) |
|---|---|---|
| N=1, mu 0.3 / 0.5 / 0.8 | 0.386 / 0.556 / 0.764 | 0.291 / 0.464 / 0.675 |
| N=3, mu 0.3 / 0.5 / 0.8 | 0.344 / 0.434 / 0.434 | 0.291 / 0.395 / 0.395 |

- MuJoCo fails 0.04 to 0.09 rad after the analytic threshold. That is the lag of the ramp and the 0.6 rad tilt rule. The ordering is right.
- For mu 0.5 and 0.8 the 3-stack fails at the same tilt. Tipping comes first, so friction is irrelevant there.
- Tripling block mass moves the failure tilt by at most 0.003 rad. Mass drops out. DERIVED and VERIFIED.
- Pre-slip creep exists. A 3-stack on a slow ramp (0.03 rad/s) moves 3 mm at 0.24 rad, well before failure. Residual creep over the first 100 ticks of the ramp (tilt reaches 0.2 rad) is 0.14 mm (N=1) to 1.7 mm (N=4).

Branch test: 72 states x 86 branches (hold, brake, 84 held constants; MuJoCo ground truth):

| metric | value |
|---|---|
| per-branch accuracy | 96.3% (precision 1.00, recall 0.96) |
| median error of failure tick | 8 ticks |
| MuJoCo states with some safe branch | 69.4% |
| analytic states with some safe branch | 100% |
| analytic best-margin pick succeeds on MuJoCo-savable states | 18 of 18 (small n) |

The gap is one specific thing. States that are already rocking or slipping are doomed. The analytic model sees only block positions, so it calls them safe.
This is the "truth is not proxy" gap, but it is small. It needs block velocity, not a neural network.

Sensitivity of the nominal failure outcome (88 episodes, N=3, s=0.35; VERIFIED):

| change | fail rate | episodes whose outcome flips vs mu 0.5 |
|---|---|---|
| mu 0.3 / 0.5 / 0.8 | 61% / 48% / 44% | 18% / 0 / 3% |
| mass x3 | 47% | 1% |
| initial offset 0 / 3 / 6 mm | 40% / 48% / 48% | 12.5% / 0 / 7% |
| noslip 10 | 41% | 7% |
| solref 0.02 | 49% | 3% |

- 22% of episodes fail at mu 0.3 xor mu 0.8, and failure comes 14 ticks earlier at mu 0.3.
- The solver settings change the result as much as a 3x mass change would. MuJoCo "truth" is only good to about 5% per episode.

## 3. Observability, sensor by sensor

| quantity | sensor | rate / noise / occlusion | status |
|---|---|---|---|
| plate pose, v, a, tilt (the whole u history) | joint encoders + FK of both arms | Piper CAN rate and encoder noise not checked here. ASSUMED fast and low-noise. | observed |
| block centres and tilts | ZED | At 1 m with f about 640 px (ASSUMED 90 deg HFOV), 5 cm is about 32 px, so 1 px is about 1.6 mm (DERIVED). Frame rate is ASSUMED 15-60 Hz against 50 Hz ticks. The arms, plate edge and upper blocks may occlude. | observed with lag and noise |
| block velocity, tilt rate | ZED, finite difference | Noise grows as 1/dt. Needed for the doomed boundary. | weak |
| mu (block-block, block-plate) | none directly | Slip onset is visible as mm-level motion (about 2 px per 3 mm). | hidden, probe-identifiable |
| initial block offsets and yaw | ZED | A 3 mm offset is about 2 px. | observed, coarse |
| mass, COM inside a block | none | Mass is irrelevant (2c). COM offset inside a block would matter and is hidden. | hidden |
| contact force | none (no sensor on the robot) | | not available |
| gripper slip | out of scope | | n/a |

Camera caveats (flag for hardware): the sim ZED is a proposed torso mount at z about 1.0 m. The real one is a floor-facing nav camera. It may not see the tray at all. Wrist cameras may. Block pose also needs a marker or a learned detector. ASSUMED.

Failure onset: the failure event is visible (blocks move). Inevitability is not. The state turns doomed about 5 ticks (0.1 s) before failure, at 2 to 6 camera frames. The deciding motion is sub-pixel (creep under 1.7 mm). So vision cannot see "doomed" in time. The plate kinematics, which the encoders give exactly, nearly decide it.

Safe probe (BOED angle, VERIFIED by the ramp test): tilt slowly and watch the bottom block.
- For N=1, first slip (3 mm) lands at 0.314 / 0.484 / 0.694 rad for mu 0.3 / 0.5 / 0.8, against atan(mu) 0.291 / 0.464 / 0.675. That identifies mu to about 0.02 rad.
- It works only when atan(mu) is below the tip angle. For N=3 that means mu below 0.42. Above that the tower tips first, but then mu does not matter for failure.
- So the probe is safe and useful only in the low-friction regime. It needs a tilt well under 0.395 rad.

**Class: partially observable, identifiable by probing** (mu, initial alignment). The main failure mechanism is observed, not hidden.

## 4. Feasibility probe (all VERIFIED unless tagged)

Setup: N=3, mu 0.5, plates and blocks as in 2b. Plate trajectories replayed from the spill_v2 held-out split (shard index % 5 == 4). The 200 Hz logged plate pose is pinned. Savable = search-certified over the same library as `bead_rep.oracle` (hold, brake, 84 held constants, 100 random 2-segment sequences plus brake tail). Horizon 100 ticks. Failure checked every 2 ticks in branches. Certified = an under-approximation.

**4a. Failure rate.** The spill_v2 cup trajectories tilt too far for a tower. Median max tilt is 0.88 rad. The tower tips near 0.4 rad. Unscaled replay fails 86% (21 episodes; 80-90% for all N, mu). I scale the plate tilt (polar angle) by s and keep accelerations. This is synthetic and ASSUMED.

| s | 0.25 | 0.30 | 0.35 | 0.40 |
|---|---|---|---|---|
| failure rate (88 eps) | 14.8% | 28.4% | 47.7% | 59.1% |

Main config s=0.35. Over 164 episodes the rate is 71/164 = 43%. Target band 20-60%: met.
Other stack sizes at s=0.3 (21 eps): N=2 at mu 0.5 fails 19%, N=4 fails 67%.

**4b. Savable set** (993 states from 71 failing episodes, 14 offsets before failure; 278 uniform states from non-failing episodes):

| ticks before failure | 1-3 | 4 | 5 | 6 | 8-20 | 30 | 40-50 |
|---|---|---|---|---|---|---|---|
| hold-still suffices | 0 | 0 | 0 | 1% | 3-8% | 80% | 93% |
| savable (hold fails, some action holds) | 0-1% | 10% | 45% | 80% | 92-94% | 20% | 7% |
| not certified | 100-99% | 90% | 55% | 18% | 0-3% | 0 | 0 |

- All pre-failure states pooled: 21.2% hold, 45.4% savable, 33.3% not certified (this mix depends on my choice of offsets).
- 451 savable states in total. 287 of them are 10 or more ticks before failure. 89 are 20 or more.
- Certificates: 374 by a held constant, 75 by brake, 2 by a random sequence. So the winning action is almost always "brake and level the plate".
- All 278 uniform states from non-failing episodes are safe under hold-still.
- Caution: the 451 are heavily correlated (71 episodes, about 6 per episode).

**4c. Lead time.** Median 5 ticks, range 2-8 (71 episodes). The doomed boundary is sharp. A state 8 or more ticks (0.16 s) before failure is 94% savable. A state 3 or fewer ticks before is lost.

**4d. Speed.** See the key-number box and 2b. A full 14-offset savable run over 164 episodes took 38 s wall on 8 cores.

## 5. Pipeline fit and WM value

Reused as is (by import, no edits): `bead_env.kin_step`, `brake_action`, `fallback_U`; `gen_nominal.episode_obs_act` formulas for `obs` and `act`; the `load_split` keys; the `bead_rep.oracle` library design.

Must change:
- Replace the cup and bead rig (`contents.rig_model`, `Oracle`) with `TowerEnv`. The snapshot is 131 numbers, not 286.
- Replace the margin and the `n, nd` aux fields.
- Rebuild the oracle to call the tower rig. It is cheaper than the bead oracle.
- The WM obs is `[axis, v, w]` with no block information. For the tower, add block pose from ZED to obs. Without it the WM has the same "hidden contents" problem as the cup.
- Cut the action domain. Tilt up to 1.5 rad is 4x past the tip angle, and a real Piper (feedforward cap 8 Nm) cannot do 5 m/s^2 with tilt rates of 2 rad/s. ASSUMED, not checked on hardware.
- spill_v2 trajectories need a tilt scale. Better: generate tower-specific carry trajectories.
- UNISafe code (IsaacLab) is not needed. It solves a different task (single-arm plucking).

**Where a learned WM adds value, honestly.**
- Physics plus observed state suffices for the decision. The analytic model gets 96% of branches right. Its best-margin pick saves all 18 savable states I tested. This is the same finding that killed the cup.
- A learned WM could help in three places only:
  1. The doomed boundary. It needs block velocity from noisy ZED poses. A rocking/sliding ODE with a filter should do this too.
  2. Hidden mu and alignment. A Bayesian filter on the analytic model can infer them from the past.
  3. Occlusion and sensor lag, where block poses drop out for several ticks.
- None of these needs the WM to supply a predicate that geometry cannot. The failure set is a function of plate kinematics plus mu, not of arm pose, but physics already knows that function.
- BOED value is real. mu moves 22% of nominal outcomes and 14 ticks of failure time, and a safe tilt probe identifies it (low-friction regime).

## 6. Effort, risks, verdict

Effort to a G1-like gate:
- Part 1 (at least 300 certified-savable held-out states): already met in-sample (451 from a quarter of the test split). The full 703-episode test split gives roughly 1.9k states, correlated. About half a day to wire the generator and certifier into the pipeline format. Replay plus certification cost about 1.5 core-hours for all 3941 episodes (DERIVED from 164 episodes in 38 s on 8 cores).
- Part 2 (nominal latent-HJ spill at least 30% with hold-still at most 10%): not measured here. The coverage failure from the earlier audit (93% of actions near u=0) probably repeats, so nominal latent-HJ spill at least 30% is likely. ASSUMED. Hold-still is below 10% on the near-failure savable band (3-8% at 8-20 ticks), but it is a function of how I sample states. Needs the existing latent-HJ training: about 1-2 days.
- Added work: a dynamic rocking/sliding baseline and a block-pose obs channel.

Top risks:
1. The thesis claim fails. Physics plus observed state decides this object, so a learned latent HJ can match physics but not beat it.
2. Thin savable window: 6 to 25 ticks, and the doomed boundary is 5 ticks. Real ZED lag may eat the window.
3. Tilt scaling, geometry, mu and the 0.6 rad threshold are all my assumptions. The 20-60% failure band depends on them.
4. MuJoCo contact numerics move about 5% of outcomes. Certified labels carry that noise.
5. The real torso ZED may not see the tray. The sim mount is proposed.
6. The action domain overshoots what the real arms can do.

**Verdict: GO-with-changes.** Use it for BOED / friction identification and as an honest control showing physics suffices. Do not use it as the headline WM-beats-physics testbed.
