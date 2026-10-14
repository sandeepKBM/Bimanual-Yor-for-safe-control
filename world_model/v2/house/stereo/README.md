# house/stereo: continuous head-ZED stereo sensor + consumers + calibration

SIMULATION ONLY (MuJoCo rendering + OpenCV; no robot code, no network). Author: stereo subagent, 2026-10-12.
Outputs / caches: `/data/local/ss5772/molmospaces/percept/stereo/` (`calib_640.json`, `track_*.json`, `payload_*.json`, `map_study.json`, `sweep*.json`, `bench_*.json`, `smoke_live_*.json`).

## Why

Everything except the flagship desk snapshot used RAY-CAST depth plus a stereo-law noise model (`depth_sim.CamSpec`): people tracker, wrist cameras, mapper.
That skips the real stereo failure modes (textureless surfaces, occlusion edges, wrong matches). This package renders the head ZED pair continuously, runs a
real matcher, and feeds the existing consumers (people tracker, mapper) and a NEW payload estimator (tray plane + cup) with that depth.

## Files

| file | content |
|---|---|
| `cfg.py` | `StereoCfg` (640x360 default, fovy 70, B 0.12, head mount of people_track.camera, SGBM params, texture mode, backend, flags) |
| `twin.py` | textured perception twin of a TeleopSim (same state layout, asserted): `build_twin_model(sim, cfg)` hooks `sim_common._patch_real_collision_geoms` during the unchanged `teleop.scene.build_rig`; `build_house_twin` (static, mapping). Texture modes `speckle` / `objects` (walls + floors flat) / `none` (flat colours = the textureless case). People get a random-dot CUBE texture (a 2-D texture is vertical stripes on a capsule: ambiguous matching). |
| `backends.py` | pluggable depth backend, one interface `depth(left, right, K, baseline) -> (depth_m, valid, ms)` + `warmup()`: `SGBMBackend` (cv2, CPU, left-pad fix for the dead strip) and `FoundationBackend` (imports `house/foundation/stereo_fs.py` when it exists; waits for its READY) |
| `sensor.py` | `StereoSensor`: state snapshot -> render pair (+ sensor noise) -> backend -> `Frame` (depth NaN=invalid, camera pose, FK self-mask, classes, optional people-free static depth, GT for evaluation) |
| `link.py` | `StereoLink`: the sensor in a SEPARATE PROCESS (shared-memory seqlock snapshot like `teleop/web/render_proc.py`). `start()` blocks until the child is READY (model load, EGL, backend warm-up, 2 dry captures) |
| `adapters.py` | stereo -> `head_frame` dict (people tracker / `ocs.PeopleProducer` input), `stereo_spec` (CamSpec of the stereo grid with the fitted noise), `make_people_producer`, `StereoHeadDepth` (mapper drop-in) |
| `payload.py` | head-stereo payload estimator + `StereoPayloadProducer` (ocs): tray plane + in-plane position + cup cylinder fit |
| `live.py` | `attach(sim, cfg, mode)` -> `StereoOCS.step(sim)` returns a `WorldState` the shield reads via `ocs.consumers.ShieldView` |
| `record.py`, `study_render.py`, `study_static.py`, `study_calib.py`, `study_track.py`, `study_payload.py`, `study_map.py`, `sweep.py`, `bench.py`, `smoke_live.py` | the experiments (commands in each docstring) |
| `tests/test_stereo.py` | CPU-only unit tests (geometry, backend interface + pad fix, head-frame conversion, plate window, synthetic tray + cup recovery) |

## Run

```
cd /common/users/ss5772/bimanual ; export PYTHONPATH=$PWD:$PWD/experiments/ab_smooth_vs_bangbang:$PWD/experiments/realtime OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 BIMANUAL_TELEOP_FAST=1 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=5   # 5 or 6 only
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python ; S=world_model.v2.house.stereo
$PY -m pytest -q world_model/v2/house/stereo/tests
$PY -m $S.record carry --seeds 0 1 2 3 --procs 4                               # CPU: 10 Hz state snapshots of the h5 carry (walking people)
$PY -m $S.study_render --seed 1 --tex speckle --every 1                        # GPU: stereo + GT + ray-cast frames  (also --tex objects|none --every 3, --res 320 180)
$PY -m $S.study_calib sample --tags speckle_s1_640x360 ... ; $PY -m $S.study_calib report --tags ... --tex-compare ...
$PY -m $S.study_track --tags speckle_s0_640x360 ... --noise <fitted.json> ; $PY -m $S.study_payload --tags ... --noise <fitted.json>
$PY -m $S.study_map --tex speckle objects none ; $PY -m $S.sweep ; $PY -m $S.bench parts ; $PY -m $S.bench rtf
$PY -m $S.smoke_live --seed 0 --source stereo --mode sync                      # closed loop: the shield reads the stereo WorldState
```

## What is simulated, and what is not (honest)

* Two pinhole cameras, parallel axes, B 0.12 m (ASSUMED ZED 2 nominal), fovy 70, head mount (0.05, -0.20955, 1.15 + lift), pitch 36.3 deg: a PROPOSED mount.
* The depth is whatever cv2.StereoSGBM does on the rendered, noise-corrupted 8-bit pair. A real ZED SDK (neural depth) will behave differently: textureless is
  the big difference (SGBM cannot do it). The backend interface exists so a learned stereo net can replace SGBM.
* The sim scene has flat colours; the twin adds a procedural speckle texture. The calibration below is for THAT texture. Real surfaces vary.
* Shortcuts inherited from the consumers: people_track's background subtraction uses a static model at the true camera pose (here: the sensor's own people-free
  render, same visual geometry); the people tracker is fed the TRUE base pose in the study (isolates the depth source; live mode uses the odometry estimate).
* The robot is masked by its known kinematics (a segmentation render of the known model, dilated 2 px): equals a perfect-FK self-mask.
* The FK tray prior for the payload estimator is `ArmFK.tray` with encoder noise; extrinsics (head mount) are exact in sim; a calibration error would add to every number below.

## Results (h5 carry with walking people, seeds 0-3, 10 Hz, 640x360, HH SGBM, speckle twin; GT = MuJoCo depth of the left camera)

### Depth calibration (scene pixels, interior = not edge, cos(incidence) > 0.5, textured)

| z [m] | fill | measured sigma (inlier std) [mm] | fitted law [mm] | default CamSpec [mm] | bias [mm] | p95 abs err [mm] | gross outlier (>10 cm) |
|---|---|---|---|---|---|---|---|
| 0.65 | 0.88 | 2.4 | 5.4 | 1.8 | 0.0 | 5 | 0.000 |
| 1.0 | 0.93 | 11.1 | 7.5 | 4.1 | -1.0 | 27 | 0.001 |
| 1.4 | 0.86 | 13.6 | 12.2 | 8.0 | -2.0 | 30 | 0.003 |
| 1.8 | 0.84 | 20.0 | 19.1 | 13.1 | -4.0 | 44 | 0.009 |
| 2.5 | 0.90 | 39.1 | 35.9 | 25.3 | -5.0 | 80 | 0.010 |
| 3.5 | 0.98 | 66.1 | 70.0 | 49.6 | -12 | 125 | 0.004 |
| 4.5 | 0.84 | 119 | 116 | 82 | -49 | 211 | 0.051 |

Fit: sigma_z^2 = floor^2 + (z^2/(f B) sigma_d)^2 with f = 257 px, floor 4.8 mm, **sigma_d = 0.176 px at 640x360** (= 0.35 px at HD720; the default CamSpec has 0.25 px at
HD720, i.e. it UNDERestimates SGBM by about 1.4x at range), **p_drop = 0.106** (default 0.02), p_out = 0.0036 (default 0.002) in the interior.
Where a simple independent-noise model cannot represent SGBM:
* depth-edge band (within ~4 px of a depth jump): fill 0.76, 9.7 % gross outliers, p95 19 cm (foreground fattening / flying pixels);
* error is not a function of range alone: slanted surfaces (floor at 0.8-1.4 m) are 2x worse than the law (window matching bias with slope);
* bias grows with range (-1.2 cm at 3.5 m, -4.9 cm at 4.5 m): the matcher favours the nearer surface; the noise model is zero-mean;
* errors are spatially correlated (staircase of the 1/16-px disparity, texture-dependent), not i.i.d. per pixel;
* low-texture patches, grazing surfaces (cos < 0.3: fill 0.73-0.83), and the closest range (< 0.5 m: fill 0.2-0.5) fail in clusters, not at random;
* OpenCV leaves the leftmost numDisparities columns dead (17 % of the image at 640): fixed by padding (fill 0.85 -> 0.90, person fill 0.78 -> 0.90).
At 320x180 the robust sigma is about 2x larger at every range (1.3 cm at 1 m, 3.9 cm at 1.8 m, 8.6 cm at 2.5 m).

### Textureless result (same frames, scene pixels 0.5-4 m)

| twin texture | scene fill | scene p95 abs err | gross outlier (>10 cm) of valid | person fill |
|---|---|---|---|---|
| speckle (everything textured) | 0.89 | 11 cm | 5.8 % | 0.90 |
| `objects` (walls + floor flat, objects/people/robot textured) | 0.49 | 111 cm | 21.5 % | 0.91 |
| `none` (flat colours as in the sim) | 0.29 | 137 cm | 39.5 % | 0.22 |

Flat walls do not only give holes: 20-40 % of the returned depths are wrong by > 10 cm (12-18 % by > 50 cm), because the matcher locks onto sensor noise / shading.
Parameter sweeps (`sweep.json`: uniqueness 8-25, speckle filter, block 3-9, texture-confidence gate) trade fill for outliers only partially (best: uniqueness 25 -> outliers 37 %, fill 0.23).
A local-texture gate kills the textureless garbage but also the valid person pixels. Conclusion: SGBM needs textured surfaces; real walls need a learned stereo backend.

### Tracker (people_track DepthDetector + MultiTracker, same recorded runs, true base pose, person visible = ZED FOV + line of sight + range < 4.5 m)

| depth source | detection (<0.5 m) | position RMSE | p95 | false tracks (ids) | false-track fraction* |
|---|---|---|---|---|---|
| ray-cast + noise model, same ZED geometry | 99.2 % | 2.9 cm | 3.9 cm | 1 | 0.013 |
| ray-cast, default tracker camera (fovy 55, 160x90) | 99.2 % | 3.2 cm | 2.8 cm | 1 | 0.018 |
| stereo SGBM, detector min_pts 6 (ray-cast default) | 99.4 % | 4.1 cm | 6.5 cm | 72 | 3.34 |
| **stereo SGBM, min_pts 40** | 99.4 % | 2.8 cm | 4.6 cm | 9 | 0.54 |

*average number of simultaneous confirmed tracks farther than 1 m from every real person. n = 1321 person-snapshots, 109 s. min_pts 40 was chosen on seeds 0-1
(1095 person-snapshots: false 0.22, RMSE 0.9 cm) and checked on held-out seeds 2-3 (226: false 0.92, RMSE 6.6 cm, detection 98.2 % vs ray-cast 97.3 %).
So stereo matches the ray-cast tracker on detection and accuracy but produces ~40x more false tracks: SGBM leaves scattered gross outliers (depth-edge flying
pixels, repeated texture) that form 6-40-point clusters. A cluster-size gate removes most; the rest needs the learned backend or a temporal consistency gate.

### Payload estimator (tray plane + red cup; FK tray prior; n = 1094 frames; same RGB and prior for both rows)

| | SGBM depth | ray-cast + noise model |
|---|---|---|
| tray + cup found | 100 % / 100 % | 100 % / 100 % |
| tray tilt error, median / p95 | 0.24 / 0.66 deg | 0.02 / 0.06 deg |
| tray plane offset error (normal), median | 0.7 mm | 0.3 mm |
| cup position, BASE frame (xy), median / p95 | 1.8 / 3.4 mm | 0.5 / 4.2 mm |
| cup position, TRAY frame error (x, y), median | 3.4 / 13.9 mm | 0.6 / 8.2 mm |
| tray in-plane position error (x, y), median | 4.4 / 15.0 mm | 0.7 / 7.5 mm |
| cup tilt error rel. tray, median / p95 | 2.7 / 7.9 deg | 1.8 / 1.9 deg |
| coverage of the ObjectState sets (k = 3 sigma + floors + tray in-plane half-width) | cup pos 100 %, cup tilt 99.9 %, tray tilt 100 %, tray in-plane 99.9 % | 100 / 100 / 100 / 100 % |
| ms per estimate | 79 | 73 |

The tray's in-plane position is only partly observable (arms occlude the plate edges): estimated by a plate-width window slid over the near-plane points;
the y bias (+15 mm stereo, +7.5 mm ray-cast) is structural, so the tray set carries the window half-width and the cup set (expressed in the estimated tray frame)
carries it too. The composed (base-frame) cup position is accurate to 2 mm because both come from the same measurement. FK-prior tray error in the recordings is up to 36 mm
(the grasp offset): the stereo plate window removes most of it. Cup tilt has a floor of ~2 deg (|angle| of a noisy estimate; polygonal translucent cup).
Caveats: coverage with the position set widened by the tray half-width is 100 % but the set is conservative (median 37 mm); the same sets at 320x180 cover only
0.78 (tray tilt) / 0.72 (tray in-plane), i.e. the floors need a per-resolution refit. The cup used is the SIM cup (r 0.032, h 0.08, data/contents.py);
`ocs.util.CUP_R/H` = 0.045/0.115 is the flagship paper cup (mismatch in ocs, not here).

### Mapper (loc/mapper.py with stereo as the sensor; scripted pass, 252 frames, 2 Hz, 320x180 grid from 640x360 pairs; stereo miss != free)

| sensor | recall @30 mm | false-free @30 mm | spurious (false-occupied) @30 mm | signed EDT error mean / p95 |
|---|---|---|---|---|
| ray-cast + noise | 0.977 | 0.023 | 0.057 | -0.8 / +0.8 cm |
| stereo, speckle | 0.974 | 0.026 | 0.282 | -2.7 / +2.0 cm |
| stereo, walls flat (`objects`) | 0.980 | 0.020 | 0.453 | -7.4 / +2.0 cm |
| stereo, `none` | 0.973 | 0.027 | 0.501 | -5.6 / +1.3 cm |

Recall of true obstacles is the same; stereo adds spurious occupied cells (the safe direction; deadlock risk), 5x more than ray-cast with textures and 8-9x with flat walls.

### Cost (this node, loaded: load average ~50; single-thread cv2; GPU 6; ms per frame; `bench_parts.json`)

| resolution | mode | render pair (+noise) | SGBM | FK self-mask | total | SGBM 4 threads |
|---|---|---|---|---|---|---|
| 320x180 | 3-way | 11 | 15 | 3 | 29 | 5 |
| 640x360 | HH | 33 | 200 | 10 | 243 | 199 |
| 640x360 | **3-way** | 32 | 83 | 10 | 125 | 38 |
| 640x360 | 5-path | 31 | 98 | 10 | 138 | 100 |
| 1280x720 | HH | 134 | 1410 | 37 | 1580 | - |
| 1280x720 | 3-way | 133 | 534 | 37 | 704 | 174 |

3-way matches HH on fill / outliers (sweep2: fill 0.888 vs 0.891, outlier 6.6 % vs 5.7 %) at 2.2x speed, so it is the live profile (`live.attach` default). Real-time factor of the
teleop loop (8 s sim at 10 Hz stereo, 640x360 3-way; the sim itself ran at 0.35x real time on this loaded node): no stereo 0.348; stereo IN-process (blocking) 0.231 (-34 %);
stereo in a SEPARATE process 0.345 (-1 %), frames delivered 8.0 Hz of sim time, publish-to-receive 147 ms (compute 140 ms), child READY (model load, EGL, warm-up) in 2.1 s.
At RTF 1 the child needs 10 captures/s: 3-way 640x360 single-thread (125 ms) gives 8 Hz, 4 threads (85 ms) gives 11 Hz; HH does not fit.

### Closed-loop smoke: the shield reads the stereo WorldState (`smoke_live.py`, h5 carry, 30 s, seeds 0-2, 640x360 3-way, `sync` mode, ShieldView)

Same scenario with the shield fed by the simulator state ("truth") and by the stereo WorldState; both: 0 spilled beads, 0 bump contacts, task not finished within 30 s (the shield yields to the walker).
Stereo-fed inputs over the run (estimate vs truth, localisation error excluded by comparing in the true base frame):

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| cup position error (base frame), median / p95 | 2.6 / 46 mm | 2.6 / 48 mm | 1.7 / 38 mm |
| tray tilt error p95 | 0.39 deg | 0.44 deg | 0.35 deg |
| cup visible (fresh stereo estimate) | 99.5 % | 99.5 % | 99.5 % |
| person recall, in front of the head 0.8-3 m (est. within 0.7 m) | 100 % | 100 % | 99.9 % |
| false person estimates (> 1 m from every real person) | 2 % | 6 % | 0 % |
| person position error, median | 4 mm | 4 mm | 4 mm |

(The cup p95 is the first second before the first sighting, where the engineering-prior cup with a wide set is used.)
`privileged()` of the WorldState is `[]`. This shows the plumbing and the estimate quality in the loop, not safety equivalence: it is one scenario family, 30 s.
Pitfalls found while building it (all fixed in `live.py`): (1) the frame is delivered late, so people frames must be placed with the odometry estimate AT CAPTURE and stamped with the
CAPTURE tick (the tracker re-applies them via its state history; `TrackerCfg.hist` raised to 96 ticks); (2) the payload estimator must use the encoders / lift of the EXPOSURE time;
(3) in `sync` mode the latency charged in sim time is the nominal one (150 ms), not the wall time of a loaded node (which gave 20-70 tick latencies and 25-60 % person recall).

### FoundationStereo through the same interface (`foundation_compat.json`; 10 frames per row, GPU 6, TRT engine, ~200 ms on the loaded node)

| twin texture | scene fill | outlier > 10 cm of valid | wrong > 50 cm | person fill |
|---|---|---|---|---|
| speckle | 0.992 | 0.9 % | 0.5 % | 1.00 |
| none (flat colours) | 0.993 | 25.9 % | 4.0 % | 1.00 |

(SGBM on the same textureless frames: fill 0.29, 39.5 % outliers, 12-18 % wrong > 50 cm.) Plugged in with `StereoCfg(backend="foundation")` + `backend_kw=dict(gpu=6)`; READY wait in `warmup()`.
Not yet run through the tracker / payload studies: that is the obvious next experiment.

## Hooks for the shield (owner applies; nothing in teleop/, shield.py, sim.py, people_track/, ocs/ was edited)

```python
# teleop/run.py run_episode (and the web server's loop), after `shield = ...`:
from world_model.v2.house.stereo import live as SL
from world_model.v2.house.ocs.consumers import ShieldView
stereo = SL.attach(sim, mode="async", seed=seed)        # builds the twin, spawns the sensor process, BLOCKS until READY (all warm-up paid)
# in the loop, replace  safe, info = shield.filter(sim, cmd)  by
ws = stereo.step(sim)                                    # publishes the snapshot, runs the producers, returns an ocs WorldState (privileged() == [])
safe, info = shield.filter(ShieldView(sim, ws), cmd)     # cup / tray / people (and base if base_from_ws=True) come from the stereo estimate
# after the loop:  stereo.close()
```
`ShieldView(sim, ws)` defaults to `base_from_ws=True` (the shield's wall / person geometry then uses the odometry estimate); the smoke above used `base_from_ws=False`.
`ws` holds: `tray` (stereo plane o FK prior), `cup` (stereo cylinder fit, tray frame; an engineering-prior cup with a wide set before the first sighting), PERSON objects (stereo depth ->
tracker), robot (odometry + encoders). Each object carries `source` (STEREO / TRACKER / ODOM / FK / PRIOR), `t_meas` = exposure time, `visible`, `unc`.
