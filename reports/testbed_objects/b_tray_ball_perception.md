# Tray + ball testbed: point-cloud ball tracking (replaces privileged ball state)

Code: `world_model/v2/tray_rep/perception/` (README there). Data: `/common/users/ss5772/bimanual_data/tray_rep/perception/`.
Simulation only. CPU only. No camera driver, no SDK, no hardware. Tags: VERIFIED = measured in this work or read from code;
DERIVED = computed from verified numbers; ASSUMED = my choice, not documented.

## Answer

1. **No single camera is enough. Head camera aimed at the tray plus both wrist cameras (fused) is.**
   In the last 25 ticks before failure it sees the ball in 96% of frames, centre error 2.9 mm mean / 3.6 p95 with no delay,
   5.4 / 10.5 mm at 30 Hz and 3 ticks of delay (VERIFIED, 58 failing carries). Under the same condition the head camera alone
   sees 85% (p95 error 47 mm), the left wrist camera alone 47%. (Study is 200 carries; wrist depth is ASSUMED, see below.)
2. **The head camera as proposed (`head_cam`, `scene_yor_canonical.xml` line 59) cannot track the carried tray.**
   It looks at the table (36 degrees down). A tray carried 0.95 m high sits at the top edge of its image:
   the ball is visible in 25% of pre-failure frames (36% in the last 25 ticks; VERIFIED). Lowering the carry to 0.80 m
   (`head_low`) or turning the same camera at the tray (`head_aim`) fixes it: 92% / 90% (86% / 85% in the last 25).
3. **The two sensors fail in different places, so they fuse well.** The left wrist camera loses the ball when it rolls to the
   near edge (hands side; 4% visible) and to the right edge (40%). The head camera loses it at the far edge, close to the
   camera, inside its 0.3 m minimum range (31% visible). Fused: 91-100% on every edge. (VERIFIED, last 25 ticks.)
4. **The right wrist camera is useless here.** Its mount is 15 mm below the tray plane, so it sees the underside only (0 ball points).
   The real ZED as mounted today (low, floor-facing) sees no tray at all (0 points in 200 carries).
5. **What the testbed tolerates (DERIVED, not a WM measurement; no WM was trained here).** In the 70 ms savable lead the ball
   moves about 12 mm (median speed 0.17 m/s over the last 25 ticks, p95 0.32). So aim for edge-distance error well under that:
   mean <= 3 mm, p95 <= 8 mm, latency <= 3 ticks WITH the filter. The fused rig meets it (3.0 / 7.6 mm at 30 Hz, 3 ticks).
   A plain delay costs 3.4 mm per tick (VERIFIED); the filter cuts that to 0.9 mm per tick (VERIFIED, fused rig).
   Each tick of delay is not free: at 5 ticks the fused edge error is 4.7 mm mean (VERIFIED).

## What was built (all new files, nothing existing edited)

`depth_sim.py` (ray-cast depth camera and noise), `ball_estimator.py` (sphere fit and delay-aware Kalman filter), `pc_env.py` (`PCRepEnv`,
`pc_est` obs, K=128 cloud, snapshots), `gen_pc.py` (same episodes as `data_piper_full`), `camera_study.py`, `selftest_pc.py`.
`pc_est` obs (16) = tray 9 (exact) + ball x, y, vx, vy (Kalman on sphere fits) + n_points/100, fit rms/4 mm, ticks since last good/25.

## Camera facts

| item | value | tag |
|---|---|---|
| Real ZED: stereo RGB-D, SDK, HD720 or VGA, NEURAL depth | `YOR/robot/zed_pub_node.py` | VERIFIED |
| ZED model, baseline | not documented; baseline 0.12 m (ZED 2) used | ASSUMED |
| Real ZED pose today | low, forward (0.21 m, 0.06 m left of base), 23.8 deg tilt; height 0.25 m used | VERIFIED (extrinsic) / ASSUMED (height) |
| Sim head camera `head_cam`, fovy 55 | scene file line 59, a proposed mount | VERIFIED |
| Wrist cameras `egocentric`, fovy 53.1, on the Piper end link, iPhone housing mesh | `piper_{left,right}.xml` | VERIFIED |
| Wrist camera depth | not documented; the repo records RGB only. Depth treated as available | ASSUMED |
| Wrist poses in the tray frame: left (-0.123, +0.089, +0.015) m, right (-0.123, -0.089, -0.015) m, both 15 deg inward | forward kinematics of `front/mid_mid`; identical for 7 front poses | VERIFIED |
| Tray carry pose for the head mapping: 0.4 m ahead, 0.95 m high (`mid_low`: 0.80 m) | authority poses | ASSUMED |
| Depth noise: sigma = z^2/(f b) * 0.25 px, floor 0.5 mm: 1.6 mm at 0.7 m, 3.1 mm at 1 m (head); wrist floor 2 mm | disparity noise 0.25 px | ASSUMED |
| Grazing mask 80 deg, dropout 2%, outliers 0.2%, quantisation 1 mm, range head 0.3-5 m, wrist 0.1-2 m | | ASSUMED |
| 160 x 90 rays, 30 Hz, 3 ticks (60 ms) latency (same as `tracker_realistic`) | | ASSUMED |

## Camera study (200 Piper carries, seeds 100000-100199, 58 fail, 36,985 ticks; ball radius 3 cm; VERIFIED)

Rows are the Kalman output vs privileged truth, last 25 ticks before failure, at 30 Hz and 3 ticks of delay
(in brackets: no delay, 50 Hz). "vis" = share of captured frames with a valid fit.

| rig | vis all | vis last25 | median ball points | pos err mm mean / p95 | vel err m/s mean | edge err mm mean / p95 | loss events per failing carry |
|---|---|---|---|---|---|---|---|
| head (as proposed) | 0.25 | 0.36 | 0 | 45 / 126 (42 / 124) | 0.12 | 29 / 93 | 0.69 |
| head_low | 0.92 | 0.86 | 44 | 10 / 58 (5.6 / 44) | 0.065 | 5.4 / 33 | 0.16 |
| head_aim | 0.90 | 0.85 | 40 | 9.3 / 47 (6.9 / 46) | 0.067 | 5.2 / 26 | 0.16 |
| wrist_l | 0.85 | 0.47 | 691 | 27 / 95 (26 / 96) | 0.094 | 11 / 49 | 0.52 |
| wrist_r | 0.00 | 0.00 | 0 | 92 / 161 | 0.17 | 63 / 97 | 1.00 |
| wrist_both | 0.85 | 0.48 | 691 | 35 / 175 (31 / 179) | 0.096 | 11 / 49 | 0.53 |
| head + wrist | 0.89 | 0.69 | 709 | 13 / 72 (10 / 65) | 0.069 | 6.3 / 34 | 0.31 |
| **head_aim + wrist** | **0.97** | **0.96** | 745 | **5.4 / 10.5 (2.9 / 3.6)** | **0.052** | **3.0 / 7.6 (1.4 / 2.4)** | **0.03** |
| zed_nav (real ZED today) | 0.00 | 0.00 | 0 | 90 / 154 | 0.17 | 65 / 97 | 1.00 |

Full table, medians and the rate x latency sweeps: `camera_study.json`, `camera_study_table.md`.
Frame-level quality: false detections (valid fit with no ball ray) 0.0% on every rig; valid fits more than 20 mm off: 0.1% (fused).

**Caveat on the head rows (VERIFIED, DERIVED).** The nominal Piper carry (intensity 3.0) moves the tray far from home:
median 0.27 m, p95 0.68 m. A real carry would stay nearer. When the tray is within 0.30 m of home, `head_aim` and `head_low`
see the ball in 100% of frames with 0.3-0.4 mm mean error; the loss comes from ticks where the tray is more than 0.5 m
away (visibility 46-48%, error 17-21 mm). So the head rows are pessimistic for a robot that carries inside its workspace.
Wrist cameras do not depend on tray position (rigid), so the wrist rows are not affected.

## Latency and rate (fused rig `head_aim+wrist`, last 25 ticks, edge-distance error mean in mm; VERIFIED)

| delay (ticks) | 50 Hz | 30 Hz | 15 Hz |
|---|---|---|---|
| 0 | 1.4 | 1.5 | 1.8 |
| 1 | 1.8 | 1.9 | 2.3 |
| 3 | 2.8 | 3.0 | 3.7 |
| 5 | 4.4 | 4.7 | 5.7 |

For `head_aim` alone the same grid runs 3.5 to 8.0 mm. A plain stale measurement (no filter) would be 3.4 / 9.9 / 16.0 mm at 1 / 3 / 5 ticks.
Rate matters less than delay. Velocity error (fused, 30 Hz, 3 ticks) is 0.052 m/s mean against a 0.17 m/s ball.

## Cost of a 3,000-episode `pc_est` dataset (CPU, one core each; 8 workers for the wall time)

| rig | CPU s per episode | CPU hours | wall at 8 workers |
|---|---|---|---|
| `full` data today (reference, with snapshots) | 0.09 | 0.08 | 1 min |
| head_aim | 0.59 | 0.5 | 4 min |
| head_low | 0.55 | 0.5 | 3.5 min |
| wrist_l | 0.84 | 0.7 | 5 min |
| wrist_both | 1.38 | 1.2 | 9 min |
| head_aim + wrist | 1.56 (1.65 with the cloud) | 1.3 | 10 min |

VERIFIED (timed with snapshots on, 186 ticks per episode, machine loaded: 50 on 72 cores). Disk per rig: about 1.1 GB with snapshots
(each snapshot carries 449 sensor and filter numbers; DERIVED), about 0.1 GB with `--no-snaps`, plus about 0.5 GB for the cloud file (DERIVED from 0.9 KB per tick).
Ray casting is 1.5 to 10.8 ms and the fit 1.0 to 4.8 ms per frame (VERIFIED).

## Checks run (VERIFIED, `selftest_pc.py`, all pass, 95 s with the existing selftests)

- Ray distance equals analytic sphere/plane distance (max error 4 um / 0.06 um). Depth noise matches the stereo law (std ratio 1.01).
- Noise-free cloud: fit recovers the centre to 3e-12 m; the visible-point centroid is biased 21 mm, the fit is not. 3 gross outliers move the fit 0.1 mm.
- Same seed gives bitwise-equal obs. Snapshot and restore reproduce the rest of the episode bitwise (also with the packed cloud, and through the dataset round trip).
- Live 30 Hz / 3-tick env equals the offline replay of the 50 Hz log (diff 1e-8); the study relies on this.
- `PCRepEnv` physics, act, l, priv and the first 9 obs columns equal `TrayRepEnv` `full` bit for bit; two episodes equal `data_piper_full` shard 0.
- The phase-1a selftest (`tray_rep.selftest --fresh`) and `wm/selftest_1b` both pass.

## Not done / limits (honest)

- No world model was trained on `pc_est`. The tolerance in point 5 comes from ball motion and tracker error, not from WM behaviour.
- Only the tray and the ball block rays. The arms, fingers and body are not modelled, so every visibility number is an upper bound.
- Wrist depth, the ZED model and the noise values are ASSUMED. If the wrist cameras are RGB-only, only the head rows stand.
  The ZED node uses NEURAL depth (smoother than raw stereo), so the stereo sigmas are probably pessimistic.
- `head_aim` is a placement what-if (same position, new aim); it is not a scene camera and the mount is still a proposal.
- The Piper nominal carry wanders far from home (see caveat), which hurts the head rows.
- `env_from_meta` in existing code does not know `pc_est`; use `pc_env.env_from_meta_pc`. Oracle labels from `data_piper_full` apply because the episodes are identical.
