# b_tray_ball at real-robot authority

Simulation only. Offline. CPU only. No robot, SDK, ROS or network was used for any robot.
Tags: VERIFIED = I ran or measured it. DERIVED = math from verified inputs. ASSUMED = my choice, not checked on hardware.

## Answer
1. **Yes, the tray + ball testbed still has a usable savable set at real-robot authority.** (VERIFIED, 20 episodes per row, MuJoCo oracle)
2. The 5 m/s^2 set is too strong. The set that fits ±8 Nm on every usable front-grip pose is **a_scale 0.3 (1.5 m/s^2), w_scale 0.5 (1 rad/s), tilt-rate slew 10 rad/s^2.** (DERIVED)
3. At that scale 48% of pre-failure states are certified savable, 22% need only a hold. The last certified lead is 3.5 ticks (70 ms). (VERIFIED)
4. The weak point is time, not size. A 70 ms lead equals the 50-100 ms perception latency. Early leads (4-20 ticks, 80-400 ms) are still 30-75% savable.

## 1. Robot model (YOR, unmodified)
- File: `YOR/robot/yor-description/robot-welded-base-and-lift.mjcf`. `git -C YOR status --short` is empty; HEAD 2cc6dd2 = origin/HEAD; `git fetch --dry-run` showed no new refs. (VERIFIED)
  I ran that one fetch against github.com (read-only, dry run). Nothing else left this machine.
- Changed in memory only: `body_gravcomp = 0` (the MJCF fakes gravity comp; the real arm pays it) and contact/limit/friction-loss off for `mj_inverse`.
- Arm: 3.6 kg per arm, joint armature 0.005, forcerange ±100, frictionloss 0.3. (VERIFIED from the file)
  MJCF has no joint-rate limits. The URDF in `YOR/piperlib/urdf` has 5 rad/s (j1-j5) and 3 rad/s (j6). I used those.
- Payload: tray 0.3 kg + ball 0.1 kg on top, one rigid body, held by both hands. Same tray as b_tray_ball (30 x 20 cm). (ASSUMED rigid 6-D grasp)
- Torque caps: (a) ±100 Nm (MJCF, `sim_common.TAU_MAX`). (b) real: ±8 Nm, wrist j4-j6 ±6.4 Nm, minus 0.3 Nm friction margin. The 8 / 6.4 figures come from memory notes; confirm on hardware. Which joints are "wrist" is ASSUMED.

## 2. Method (`world_model/v2/tray_rep/authority/`)
- `robot_authority.py`: IK puts both grasp points (fingertip midpoint) on the tray. `mj_inverse` gives each arm's torque for a tray accel. The two hand wrenches must sum to the tray's Newton-Euler wrench.
  Hand split: "ls" = plain load sharing (no tuned squeeze). "lp" = LP picks the best squeeze, bounded to 15 N and 0.5 Nm per hand (ASSUMED bounds). An unbounded split cheats: the hands load each other to cancel gravity.
- Poses: 12 tray positions x 2 grips = 24. Grips: "front" (hands on the near edge, fingers forward) and "side" (hands on the long edges, fingers inward). **17 of 24 reachable, 11 usable** (joint margin >= 0.1 rad). Poses and sizes are my choice (ASSUMED).
- **Independent check (VERIFIED):** a second model welds a tray to both grippers. Applying my torques in forward dynamics gives the target tray accel to within 0.04 m/s^2 and 0.10 rad/s^2 (4 poses, 8 motions, both splits).

## 3. What the arms can do (11 usable poses; per-pose rows in `summarize.py` output)
| quantity | ±100 Nm (MJCF) | real ±8 / 6.4 Nm |
|---|---|---|
| gravity, bare arm, worst joint (DERIVED) | same | 2.8 - 5.9 Nm (median 4.0) |
| static torque with tray, worst joint (DERIVED) | same, 3-7% of cap | 3.4 - 6.6 Nm; 44-86% of cap |
| max horizontal accel, one direction, front grip (DERIVED) | >= 70 m/s^2 (bisection bound 80) | 2.8 / 5.8 / 6.5 plain; 4.9 / 9.4 / 11 tuned (min / med / max) |
| same, side grip | >= 70 | 1.4 / 3.2 / 4.9 plain |
| max tilt accel alone, x/y axes (DERIVED) | >= 570 rad/s^2 | front 18 / 26 / 39 plain; side 9 / 29 / 62 |
| max tilt rate before a joint hits its limit (DERIVED) | same | 1.0 / 1.6 / 2.3 rad/s (min / med / max) |
| max tray speed before a joint hits its limit (DERIVED) | same | 0.40 / 1.07 / 1.34 m/s |
- Sideways (y) is the binding direction. Forward (x) allows 2x more.
- At ±100 Nm torque is not the limit. At ±8 Nm it is, and the arm's own weight eats most of it.
- Cross-check of the "7.4 of 8 Nm" claim: bare-arm gravity at the repo's LEFT_GRASP_Q pose is 7.2 Nm on j1 and j2 (VERIFIED). That is an extended reach-to-table pose. Tucked tray-carry poses use 2.8-5.9 Nm. So the claim holds for extended poses, not for all carry poses.
- Payload mass hardly matters: halving it lifts the median scale 0.72 to 0.80; removing it, to 0.88 (DERIVED). The arm's own weight dominates.

## 4. Recommended scale (DERIVED)
The test: every full-magnitude element of the 85-set must fit the cap, gravity included, in both phases (tilt rate ramping at alpha; tilt rate held). `recommend.py`, `grid.py`.
- Tilt rate is set by the joint-rate limit, not torque: w_scale 0.5 (1.0 rad/s = worst usable pose). Torque does not care about w (same result at 0.5 and 1.0).
- The tilt slew alpha is the real cost: at 40 rad/s^2 no scale fits; at 20 few do; at 10 rad/s^2 the table below.
Fraction of usable poses where all 85 elements fit (alpha = 10 rad/s^2, w_scale 1):
| a_scale (m/s^2) | front, plain | front, tuned | all 11, plain | all 11, tuned |
|---|---|---|---|---|
| 0.3 (1.5) | 100% | 100% | 82% | 100% |
| 0.5 (2.5) | 71% | 100% | 64% | 91% |
| 0.65 (3.3) | 57% | 100% | 55% | 82% |
| 1.0 (5) | 29% | 86% | 18% | 73% |
**Recommendation: front grip, a_scale 0.3, w_scale 0.5, slew 10 rad/s^2, acceleration clipped as a vector (not per axis).** One scalar. It fits every front pose with plain load sharing.
- If the controller knows its pose: 0.5-0.7 is fine in mid-workspace poses (median 0.72 plain, front).
- Per-axis (single-direction limits, not tested jointly): y allows about half of x. A split such as s_x 0.6 / s_y 0.3 is the next thing to try. Untested.
- Avoid the side grip and far-low poses: static load reaches 86% of the cap there.

## 5. Tray + ball savable set at that scale (MuJoCo oracle, 20 failing episodes x 13 leads = 260 states per row; VERIFIED)
Same 85-set scaled per axis, same brake rule, plus a tilt-rate slew of 10 rad/s^2 (ASSUMED model of the arm's rate loop). Carry: b_tray_ball's synthetic carry, durations stretched by 1/sqrt(a_scale), accel clipped at the scale. Intensity was raised until the nominal failure rate sat in band.
| row | a / w scale | carry intensity | nominal failure (n=100) | certified savable | hold ok | not certified | last certified lead (median) |
|---|---|---|---|---|---|---|---|
| **recommended** | 0.3 / 0.5 | 3.5 | 29% | **48%** (124 states) | 22% | 31% | **3.5 ticks (70 ms)** |
| mid | 0.5 / 0.8 | 2.0 | 24% | 42% (110) | 16% | 42% | 4 ticks (80 ms) |
| slew only | 1.0 / 1.0 | 1.7 | 43% | 41% (107) | 21% | 38% | 4 ticks (80 ms) |
- Reference, no slew, full authority (b_tray_ball report): 57% savable, 28% hold ok, 2 ticks. Slew 10 rad/s^2 alone costs about 2 ticks of lead (different episodes and lead grid, so rough).
- Differences between the three rows are inside the noise (20 episodes, states within an episode are correlated). Read them as "about the same".
- Savable by lead, recommended row: 1 tick 20%; 2: 40%; 3: 50%; 4: 75%; 5: 65%; 8: 60%; 15: 40%; 20: 30%. Hold-ok takes over at 15-20 ticks (50-65%).
- Nominal failure vs carry style, recommended scale (n=100):
  | carry | intensity 1.7 | 2.5 | 3.0 | 3.5 |
  |---|---|---|---|---|
  | stretched (slower) | 5% | 15% | 21% | 29% |
  | clipped (same paths) | 19% | - | 35% (at 3.0) | - |
  Peak tray speed p99: stretched 0.82 / 1.02 / 1.11 / 1.13 m/s; clipped 0.82 m/s at 1.7.
- **Speed flag.** The in-band stretched carries (intensity 3.5) peak at 1.1 m/s. That is at the median joint-rate limit (1.07) and over it in near-body poses (0.40). The clipped carry at intensity 1.7 (19% failure, 0.82 m/s) is the safer match. I did not run the oracle on it.

## 6. What this means
- Usable: 124 certified-savable states from 20 episodes (6 per episode). 300 needs about 50 episodes, about 14 CPU-min.
- The honest risk is latency. Savability is lost 70 ms before failure. ZED perception takes 50-100 ms (ASSUMED, not measured here). A shield sees most of its savable states 100 ms or more early, not the last 70 ms.
- Carries must be slow. The old 44% failure rate came from carries that need 5 m/s^2. At 1.5 m/s^2 the same carry style fails 5%. In-band failure now needs carry speed near the arm's rate limit, or clipped carries.
- If a larger set is needed, in order of effect (DERIVED unless noted):
  1. **Arm control mode.** Raise the usable torque above the ±8 Nm feedforward cap (for example, carry gravity through the PD position-offset term, if the motor's total limit allows). With caps x1.5 the median scale is 1.0 and the min 0.88 on the side grip (0.33 -> 1.0 on the front grip). Needs hardware confirmation.
  2. **Front grip, tucked pose.** Free. Median plain accel limit 5.8 m/s^2 (front) versus 3.2 (side). Median bare-arm gravity 4.0 versus 4.75 Nm.
  3. **Faster tilt loop.** Not measured in MuJoCo. It costs torque: at slew 20 rad/s^2 the 0.3 scale fits only 43% of front poses with plain sharing.
  4. **Lighter tray.** Small gain (median scale 0.72 -> 0.80 at 0.2 kg total payload). Not worth it.
  5. **Lip / dustpan, different ball, slower carries.** Not tested here. b_tray_ball already measured the lip (failure 85% to 10%). Slower carries cut failure to 5% at intensity 1.7, so they shrink the failing data set.

## 7. Caveats
- Poses, grips, squeeze bounds and the rigid grasp are my choices. A pinch grasp may not hold the moments I allow. (ASSUMED)
- Reflected motor inertia is the MJCF armature (0.005). A real geared joint has more, so the alpha limit is optimistic. (ASSUMED)
- Linear accel has no jerk limit in the testbed. (ASSUMED)
- The ball is lumped into the tray when sizing torques. Rolling-ball reaction force is ignored. (ASSUMED)
- Torque limits are static per joint. Motor speed-torque droop and PD gains are not modelled. (ASSUMED)
- 20 episodes per row. The scale-to-scale differences are not significant.

## Files
- Code: `/common/users/ss5772/bimanual/world_model/v2/tray_rep/authority/` (`robot_authority.py`, `recommend.py`, `grid.py`, `summarize.py`, `verify_fd.py`, `scaled_probe.py`)
- Data: `/common/users/ss5772/bimanual_data/tray_rep/authority/` (`robot_authority.json`, `recommend.json`, `grid.json`, `verify_fd.json`, `scaled_probe.json`)
- Step 4 cost: about 19 CPU-min (limit 30). Robot part: about 15 CPU-min.
