# tray_rep/perception: point-cloud ball tracking for the tray + ball testbed

The tray testbed used to feed the world model the ball's x, y, vx, vy straight from MuJoCo.
This folder replaces that with what a depth camera could give: a noisy point cloud, a sphere fit and a Kalman filter.
It adds a new observation mode, `pc_est`. Nothing outside this folder is edited.

Simulation only. CPU only. No GL, no GPU, no camera driver, no ZED SDK, no hardware, no network.
Python: `/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python`. Run from the repo root.
The modules pin threads for you (`OMP/OPENBLAS/MKL/NUMBA_NUM_THREADS=1`).

## Files

| file | what it does |
|---|---|
| `depth_sim.py` | Ray-cast depth camera (`mujoco.mj_multiRay`), noise model, camera specs and rigs, frame clock and delay queue. |
| `ball_estimator.py` | Cloud to tray frame, point selection, robust known-radius sphere fit, constant-velocity Kalman filter. |
| `pc_env.py` | `PCRepEnv`: subclass of `TrayRepEnv` with `obs_mode = pc_est`. Compact cloud. Snapshot and restore. `replay()` for offline sweeps. |
| `gen_pc.py` | Sharded, resumable generator. Same episodes as `data_piper_full`. Optional cloud file. |
| `camera_study.py` | The camera-placement study (about 200 carries, every rig). |
| `selftest_pc.py` | Smoke tests (about 15 s; `--existing` also runs the phase-1a selftest and `wm/selftest_1b`, about 80 s). |

## Run commands (owner)

```
cd /common/users/ss5772/bimanual ; PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
$PY -m world_model.v2.tray_rep.perception.selftest_pc --existing
$PY -m world_model.v2.tray_rep.perception.camera_study --n-eps 200 --workers 6        # about 2 min; writes camera_study.json + table
# a 3000-episode dataset for one rig (same episodes as data_piper_full; shard split by index % 5)
$PY -m world_model.v2.tray_rep.perception.gen_pc --config piper --camera head_aim --n-eps 3000 --workers 8
$PY -m world_model.v2.tray_rep.perception.gen_pc --config piper --camera head_aim --n-eps 3000 --merge
#   options: --save-cloud (K=128 cloud per tick, cloud_<g>.npz)  --rate-hz  --latency-ticks  --sigma-a  --no-snaps  --shard i/n
```
Outputs live under `/common/users/ss5772/bimanual_data/tray_rep/perception/` (default dataset
`data_piper_pc_est_<camera>`). The merged `train/val/test.npz` read with `tray_rep.store.load_split` (obs is 16 wide).
The oracle labels of `data_piper_full/oracle` apply as they are: same split, same episode order, same ticks.
`meta.json` has `obs_mode = pc_est` and `base_obs_mode = full`. Existing loaders that build an env from the meta
(`env.env_from_meta`) do not know `pc_est`; use `pc_env.env_from_meta_pc(meta)` for it.

In Python:
```python
from world_model.v2.tray_rep.perception.pc_env import PCRepEnv
env = PCRepEnv("head_aim", a_scale=0.3, w_scale=0.5, alpha=10.0)     # rate 30 Hz, latency 3 ticks from the camera spec
rec = env.reset(ball_xy=(0.004, -0.003), mu_roll=0.002, track_seed=7)  # rec["obs"] is 16-D
rec = env.tick(u)                                                      # rec["priv"], rec["l"] stay privileged truth
```

## Observation (`pc_est`, 16 numbers)

`[axis c (3), tray velocity v (3), realised axis rate w (3)]` exact (joint encoders and forward kinematics, as on the robot).
Then `[ball x, y, vx, vy]` from the Kalman filter (tray frame, clipped like `full`).
Then three quality numbers: `n_points / 100` (capped at 1), `fit rms / 4 mm` (1 when there is no fit), `ticks since the last good measurement / 25` (capped at 1).
`priv`, `l`, `n`, `nd`, `act` are the parent's: exact, privileged, for labels and aux heads only.

## Cameras (`camera=`)

| name | what it is | status |
|---|---|---|
| `head` | The scene camera `head_cam` (`scenes/scene_yor_canonical.xml` line 59), fovy 55, mapped to a tray carried 0.95 m high. | Mount is a PROPOSAL (confirm on hardware). It looks at the table, so the tray sits at the top edge of the image. |
| `head_low` | Same camera, tray carried 0.80 m high (`front/mid_low`). | Same caveat. |
| `head_aim` | Same camera position, turned to look at the tray. | WHAT-IF placement, not a scene camera. |
| `wrist_l`, `wrist_r` | YOR wrist cameras (`egocentric`, fovy 53.1), pose in the tray frame from forward kinematics of the front-grip carry pose. | Camera is real. Depth is ASSUMED (the repo records RGB only). |
| `wrist_both` | both wrist cameras | |
| `head+wrist`, `head_aim+wrist`, `head_low+wrist` | head camera plus both wrist cameras, clouds merged | |
| `zed_nav` | The real ZED as mounted today: low, forward, floor-facing (from `YOR/robot/zed_pub_node.py`). | Diagnostic. Height ASSUMED 0.25 m. It cannot see the tray. |

Wrist cameras sit rigidly on the Piper end link, so in a rigid grasp their pose in the tray frame is fixed:
left at (-0.123, +0.089, +0.015) m, right at (-0.123, -0.089, -0.015) m, both looking forward and 15 degrees inward.
The right camera is 15 mm below the tray plane, so it looks at the underside of the tray and never sees the ball.
(Same pose for all seven front-grip poses checked.)
Several cameras: clouds are merged in the tray frame and fitted together. The frame rate is the slowest camera's, the latency the largest.

## Noise model (`depth_sim.CamSpec`; ASSUMED unless marked)

1. Depth noise: `sigma_z = sqrt(floor^2 + (z^2 / (f * b) * sigma_disp)^2)`. `f` = focal length in pixels of the real
   sensor (HD720). Head: baseline 0.12 m (ZED 2; the ZED model is NOT documented), disparity noise 0.25 px, floor 0.5 mm.
   This gives 1.6 mm at 0.7 m and 3.1 mm at 1 m. Wrist: floor 2 mm plus a virtual 6 cm baseline.
   The real ZED node uses NEURAL depth, which is smoother than raw stereo; treat these sigmas as pessimistic or fair.
2. Grazing mask: a hit is invalid when the angle between ray and surface normal is above 80 degrees.
3. Dropout 2 % of hits. Gross outliers 0.2 % of hits (depth shifted 5 to 20 cm).
4. Depth quantisation 1 mm. Valid range: head 0.3 to 5 m, wrist 0.1 to 2 m (ZED 2 minimum is 0.3 m).
5. Frame rate 30 Hz against the 50 Hz tick (sample and hold; 3 frames in every 5 ticks). Latency 3 ticks (60 ms), the same numbers as `tracker_realistic` in `configs/piper.json`.
6. Grid: 160 x 90 rays per camera (about 1.5 to 4 ms per frame on one core). ROI option casts only around the tray.
7. Not modelled: the arms, fingers and body hiding the ball; motion blur; rolling shutter; depth-edge "flying pixels"; real depth biases.
   Only the tray and the ball block rays.

Frame noise is drawn from a generator seeded with (episode seed, capture tick). A restored snapshot sees the same noise.

## Estimator

1. Put points in the tray frame (the tray pose at the capture time is known). Keep points above the plane
   (`z > 2 mm + 3 sigma_z |ray z|`) and inside the footprint grown by the ball radius and 2 cm.
2. Fit a sphere of KNOWN radius (3 cm) by least squares: robust (Tukey weights, then a trim), started from the median point pushed away from the camera.
   With few points (< 30) the fit uses z = radius (ball on the tray). The centroid of the visible points is biased toward the camera
   (about 20 mm in the selftest); the fit is not.
3. Output: centre, number of inlier points, fit RMS, xy covariance of the centre (from the fit Jacobian, plus a term that grows fast below 15 points). Visible = at least 6 inliers and RMS under 6 mm.
4. Constant-velocity Kalman filter in the tray frame. Delayed frames are applied at their capture tick and re-predicted to now.
   Track loss: the filter keeps predicting, the covariance grows, after 4 ticks the velocity decays and the position is clipped to the footprint (+4 cm).
   `since` counts ticks since the last good arrival; a loss event is counted when it passes 10 (0.2 s); a frame after 60 ticks restarts the track.
   Process noise: ball acceleration std 1.5 m/s^2 (`--sigma-a`).

## Cost (CPU, one core)

One core, with snapshots, 186 ticks per episode: `full` data 0.09 s; head_aim 0.59 s; wrist_l 0.84 s; head_aim+wrist 1.56 s (3,000 episodes: 0.5 to 1.3 CPU hours).
Per frame: ray casting 1.5 to 10.8 ms, fit 1 to 4.8 ms. Details in `reports/testbed_objects/b_tray_ball_perception.md`.
Snapshots carry the sensor and filter state (449 numbers each), so snapshot files are about 2x larger than the `full` data. `--no-snaps` skips them.

## Caveats

- The perception layer sees no occluders other than tray and ball, so the numbers are optimistic where the arms or fingers would block the view.
- The head camera mapping assumes the front-grip carry pose `mid_mid` (tray 0.95 m high, 0.4 m ahead of the robot) or `mid_low` (0.80 m).
  The nominal carry law moves the tray up to a few tenths of a metre from home; the study reports how far.
- Wrist-camera depth is a what-if. If the wrist cameras are RGB-only, those rigs do not exist.
