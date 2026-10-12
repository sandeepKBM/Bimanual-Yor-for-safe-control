# MolmoSpaces house scenes in the bimanual YOR codebase: integration plan

Date: 2026-10-05. Status: PLAN. No repo file was edited. Smoke files: `/data/local/ss5772/molmospaces/`.
No credentials, hardware or GPU. Network: public GitHub clone and public Hugging Face reads only.
Tags: VERIFIED = read in repo/docs or measured. ASSUMED = not checked.

## 0. Bottom line
- Use houses first as a **ray-cast backdrop and route map**. Never under the payload-safety
  rig: a house costs 10x to 140x per physics step.
- Plain MuJoCo 3.9 loads a house with no install. Do not install `molmo_spaces` into the
  shared env (it pins `mujoco~=3.5.0`, torch 2.7, Python 3.11).
- YOR attaches into a house through `mjSpec` with no name clash. A mocap root body moves the
  robot with no new degrees of freedom, so FL-MPC sees the same model.
- Top risks: clutter speed, doorway width vs tray width, base-motion forces FL-MPC lacks.

## 1. Facts about MolmoSpaces
### 1.1 Project, code, license
- VERIFIED: paper arXiv 2602.11337 (AI2, RSS 2026). Repo `github.com/allenai/molmospaces`.
  Cloned at commit 713fd12 (2026-09-19) to `/data/local/ss5772/molmospaces/src` (169 MB).
- VERIFIED: README: "Upgrades in progress, for a stable release use v0.2.9" (PyPI `molmo-spaces`).
- VERIFIED: code license is Apache-2.0 (`LICENSE`).
- VERIFIED: data license on the HF dataset card is `odc-by` and `cc-by-4.0`. The README adds:
  some Objaverse assets are CC BY-NC / BY-NC-SA. Default policy uses them anyway. Set
  `MLSPACES_LICENSE_POLICY=commercial_safe` to exclude them. Use is "research and education".
- ASSUMED: the license of the THOR object pack itself (not read). Check before any release.
- ASSUMED: headline numbers (230k houses, 130k assets, 48k manipulable, 42M grasps) are from
  the blog/abstract. The README table confirms the families below.

### 1.2 Scene content and size (all VERIFIED from the HF size manifests)
| Family (code name) | Houses | Houses-only size | Notes |
|---|---|---|---|
| ithor (MSCrafted) | 120 | 0.5 GB | hand-made, many articulated objects |
| procthor-10k (MSProc) | 10k train + val/test | train 1.0 GB, val 0.09 GB | THOR objects |
| procthor-objaverse | ~100k train | 12 GB | Objaverse objects |
| holodeck-objaverse | ~100k train | 15.6 GB | LLM-made layouts |
| objects: thor | about 2k assets | 1.57 GB packed (about 2.4 GB on disk) | textures alone 0.64 GB |
| objects: objaverse | 129,725 files | 142 GB | do not fetch |

- VERIFIED: a house is a plain **MJCF XML** plus assets. Example `val_541.xml` (2.1 MB):
  meshes are `.obj` (walls, rooms, convex collision pieces), textures are `.png`.
  Objects are referenced as `../../objects/thor/<Category>/.../*.obj`. A `<name>_ceiling.xml`
  variant exists. Each house also ships `.json` (the ProcTHOR spec), `_metadata.json`
  (per object: category, asset_id, room_id, is_static, name map) and `_map.png`.
- VERIFIED: a house archive is tiny (0.01 to 0.4 MB packed); the object packs are the cost.
- VERIFIED: house XML defaults: `timestep 0.002`, `integrator implicitfast`, `cone elliptic`,
  `jacobian sparse`, `noslip_iterations 4`, `multiccd` on. Geom groups: 0 = visual (no
  collision), 3 = structural walls, 4 = dynamic/collision meshes.
- VERIFIED: sampling is file-level: pick an index (`val_N`); layout is fixed per file.
  Randomization (lights, poses) lives in task samplers.

### 1.3 Python API, tasks, robots, annotations
- VERIFIED: layers are Env (`MjModel` + `MjData`) / Task (gym-style) / TaskSampler (loads
  houses with **MjSpec**, places robot and objects). Batch size > 1 is "not well tested".
- VERIFIED: robots in the asset list: `rby1`, `rby1m` (wheeled, bimanual), `franka_fr3`,
  `franka_droid`, `floating_rum`, `floating_robotiq`, `i2rt_yam`, `g1`, Rocketbox humans.
  Code also has `bimanual_yam.py`. **No YOR, no Piper.** Robot base frame: +x forward, +z up.
- VERIFIED: assets download lazily (`molmospaces_resources`, R2 or HF); env vars
  `MLSPACES_CACHE_DIR`, `MLSPACES_ASSETS_DIR`.
- VERIFIED: annotations: per-object category/room/static flag (house `_metadata.json`),
  `ObjectMeta` text descriptions (Objaverse), DROID grasps, occupancy map, benchmarks
  (MS-Bench v1/v2). The task side is pick, pick-and-place, open/close, door opening, nav.
### 1.4 Dependencies and MuJoCo version
- VERIFIED: `pyproject.toml` pins `mujoco~=3.5.0` (the `[mujoco]` extra), a 3.7.1 wheel for
  the Filament extra, `torch~=2.7.0`, `numpy<3`, `mujoco-warp`, `open_clip_torch`, `opencv`.
- VERIFIED (measured): a MolmoSpaces house XML **compiles and runs on our MuJoCo 3.9.0** with
  the stock Python bindings. The deprecated `<size njmax nconmax>` line is accepted.
- ASSUMED: the `molmo_spaces` Python package itself works on 3.9. Not needed, so not tested.

## 2. Smoke test (done): load and measure

House `procthor-10k-val/val_541` (19.8 m x 19.8 m, 225 annotated objects) plus the THOR packs
it needs: 0.76 GB fetched by HF byte-range reads, no sign-up. Textures (0.64 GB) skipped; I
stripped `texture` refs in a scratch copy (physics and rays ignore them). Scripts in
`/data/local/ss5772/molmospaces/smoke/`. One thread, MuJoCo 3.9.0.

| Measurement | Result |
|---|---|
| nbody / ngeom / nmesh / njnt | 643 / 5,145 / 2,971 / 181 (139 free, 42 hinge/slide) |
| geoms by group | g0 visual 701, g3 walls 108, g4 collision 4,336 |
| load time (first, cold) | 7.4 s; compiled model 87 MB |
| `mj_step`, house alone, settled | **9.2 to 9.7 ms**, ncon about 690 (dt 0.002) |
| same, `noslip_iterations=0` | 6.1 ms |
| same, clutter made static (free joints removed) | **0.79 ms**, ncon 11 |
| same, contacts disabled | 1.5 ms |
| reference: `scene_yor_canonical.xml` | 0.07 ms/step (nbody 39, dt 0.005) |
| `mj_multiRay` 160x90 = 14,400 rays, all geoms, cutoff 20 m | **260 to 337 ms** (about 23 us/ray) |
| same, collision group 4 only | 107 to 118 ms (7 us/ray) |
| same, visual group 0 only | 196 to 218 ms |
| 1,600 rays (about 40x40 ROI), all geoms / group 4 | 42 ms / 14 ms |
| attach YOR (`yor_robot_derived.mjcf`) with `mjSpec.attach_body`, prefix "" | OK. nbody 679, ngeom 5,214, nu 16, 4 cams, **no name clash** |
| mocap root body wrapping YOR | OK. nv 892 (house 876 + robot 16). Robot moved exactly with the mocap body (0.499 m for 0.5 m) |

Reading the numbers: ray cost is linear in rays and geometry. A full 160x90 view per tick
is too slow; ROI casting (as `depth_sim.capture` does) keeps it near 15 to 45 ms. The 9.7 ms
step is the clutter; static clutter is 0.8 ms (11x the current rig). `mj_multiRay` threading
is unknown; re-time on a quiet node.

## 3. Integration design

### 3.1 Where code lives
New package `world_model/v2/house/` (sibling of `tray_rep/`; it imports `tray_rep`, never the
reverse): `assets.py` (paths, manifest, license policy, range download), `build.py`
(`load_house`, `compose` via `mjSpec`), `nav.py` (route planner, base command),
`perception_bridge.py` (ray backdrop for `depth_sim`), `selftest.py`, `README.md`.
Data root: env var `HOUSE_DATA=/data/local/ss5772/molmospaces/assets`. Never `/common/users`.

### 3.2 Two modes (the key decision)
- **Mode B, backdrop (default).** House compiled with free joints removed and collisions off.
  Used only for `mj_kinematics` and `mj_multiRay`. Tray, ball and arm geoms are posed in by
  hand each tick. Payload physics stays in the fast object rig. Serves goals B and C.
- **Mode D, dynamic house.** Full physics with YOR and payload. For goal A demos and a few
  validation runs. Furniture static; only the payload and chosen objects dynamic (about 1 ms).

### 3.3 Composing scene + YOR + payload
- Use `mjSpec`, not `<include>` (the house uses relative asset paths, the robot an absolute
  `meshdir`; `<include>` ASSUMED awkward, not tried). VERIFIED: `attach_body` handled both.
  Never edit `YOR/` or `scenes/`; build in memory, as `sim_common.load_model` does.
- Prefix `""` keeps existing name lookups (`left_arm_joint1`, ...). VERIFIED: no clash with
  `val_541`. The loader must raise on a clash in any other house.
- Robot root: wrap in a **mocap body** (`yor_root`). Moving the base = write `mocap_pos` and
  `mocap_quat`. No new degrees of freedom, so `mpc_fl.py` indexing is unchanged. VERIFIED
  by `mocap_probe.py`. Cost: base acceleration is not part of `tau = M a + c`. Treat it as a
  logged disturbance; cap base acceleration in Phase 2.
- Collision bits: house structure is `contype 8 / conaffinity 15`; clutter is `contype 1` or
  0. `sim_common._patch_real_collision_geoms` uses bits 4, 8, 16, 32 for arm-arm logic. Reuse
  that patch after attach, then test that arm geoms hit walls (bit 8 side) and furniture.
- Payload: build the tray + ball from `testbeds/b_tray_ball` (as `tray_rep` does), attach to
  the gripper frames with weld constraints (same as the flagship demo grasp) or, for Mode B,
  pose it kinematically from the 6-D command.
- House dt is 0.002; canonical YOR scene uses 0.005. Set the controller's dt and re-check
  stability (risk 4).

### 3.4 Base and navigation
- VERIFIED: in this repo's sim the YOR base is **welded**. `freejoint`, steer and drive joints
  are commented out in `scenes/yor_robot_derived.mjcf`. No sim base controller exists.
  `YOR/robot/base.py` (`BaseController`, `base_motor.py`) drives CAN motors. **Never import or
  run it** (hardware rule). `YOR/robot/yor_mujoco.py` is an arm-only sim wrapper.
- Plan: kinematic planar base via the mocap root. A* on the occupancy grid (`_with_occupancy`
  or built from geoms at 5 cm), inflated by footprint plus arm spread. Pure pursuit with jerk
  limits. Doors from `_metadata.json` (category `Doorway`); measure clear width.
- Not claimed: swerve-drive dynamics or wheel slip (no model).

### 3.5 Cameras and point clouds
- VERIFIED: `head_cam` in `scene_yor_canonical.xml` is a fixed **world** camera, not on the
  robot. YOR wrist cams `left/right_arm_egocentric` are on the arms (kept after attach).
  `front_camera`/`side_camera` track the base. For house runs, add a head camera to the
  robot (torso mount, proposed ZED mount: confirm on hardware, per AGENTS.md).
- `perception/depth_sim.py` casts against its own tiny `Scene` and passes `geomgroup=None`.
  Another agent owns that file. Do not edit it. Add an adapter in `house/perception_bridge.py`
  that exposes the same `m`, `d` interface over the Mode B model, plus the tray, ball and
  arm geoms posed in. Ask the owner for a `scene=` hook only if the adapter is not enough.
- Ray cost controls (measured above): ROI around the tray (already in `capture`); a
  `geomgroup` mask (group 4 hulls for cheap occlusion, group 0 for realistic; always keep tray,
  ball and arm groups); cutoff near 1.2 x `z_max`; sensor at its real 30 Hz rate.
- Realism comes from clutter between camera and tray, arm self-occlusion and varied head
  poses. Ray hits return geom ids, so points carry object labels for free.

### 3.6 Determinism and snapshot/restore
- Mode B is stateless apart from the posed geoms: repeatable by construction.
- Mode D: `mj_getState`/`mj_setState` with warmstart and `mocap_*`. Bitwise restore needs
  one thread and the same compile. Gate: restore then 200 ticks equals the straight run
  bitwise, else document the drift. Seed every house pick and pose randomization.

## 4. Phases

Effort = engineer-agent working days, with review. Data on `/data/local`.

### Phase 0: load and measure (about 80% done by this smoke), 0.5 day left
Deliverables: `house/assets.py`, per-house object-pack manifest, `selftest` loading 20 val houses.
Acceptance: all 20 compile on MuJoCo 3.9; table of nbody/ngeom/step/ray time and door widths
per house; disk < 4 GB. Rollback: delete `/data/local/ss5772/molmospaces`.

### Phase 1: YOR in one house, holding the tray, FL-MPC running, static carry, 4 days
Deliverables: `compose()` for Mode D; YOR placed in a clear room; arms hold tray + ball
(weld); FL-MPC runs a short in-place carry; furniture static.
Acceptance:
1. YOR spawns with zero contacts against house geoms.
2. FL-MPC controls on the composed model produce the same tau as on `scene_yor_canonical`
   for the same arm state, within 1e-6 (same `M`, `c`; only the world differs). Do this test
   with `mpc_fl` unmodified: the arm joints are found by name.
3. 5 s static carry: ball stays on the tray; zero arm-arm and arm-house collisions; step
   time under 1.5 ms (budget, see 5).
4. Snapshot/restore gate from 3.6.
Rollback: drop `house/build.py`; nothing else imports it.

### Phase 2: route-level carry through a doorway, with perception, 6 days
Deliverables: `nav.py`, a route through one doorway and one turn, perception via
`perception_bridge.py` (head + wrist cams), a demo video from a new script (flagship untouched).
Acceptance:
1. Base reaches the goal across the doorway, no collisions, clear margin >= 5 cm to the
   door frame (or the plan reports "no route", which is a valid result).
2. Ball stays on the tray in 8/10 seeds at base accel cap (record the cap). Base-motion
   disturbance on arm torque is logged and compared to the rig's disturbance range.
3. Point-cloud tracker runs on house rays: ROI cost under 20 ms per camera per frame;
   tracker RMS error reported with and without furniture occlusion; compare to the
   object-only rig number.
4. 10 seeds repeat bitwise (same seed, same single-thread build).
Rollback: remove `nav.py` and `perception_bridge.py`; Phase 1 stays valid.

### Phase 3: many houses for perception and benchmark data, 5 days
Deliverables: sampler over `procthor-10k-val` (1,000 houses, 0.09 GB), then train (1 GB);
cached stripped backdrop XML plus manifest per house; dataset writer; label table
(category, room) for goal C; small benchmark spec (route + payload + seed list) for goal D.
Acceptance: 200 houses, 3 views each, tracker run; failure log per house (load, spawn, route);
parallel, 8 workers, thread-pinned; throughput report; total disk < 15 GB.
Rollback: delete the data dir; no repo effect.

## 5. Performance budget
- The data rig stays object-only (0.07 ms class). `tray_rep` must never import `house`.
- Mode D target: <= 1.5 ms/step with static furniture (house alone measured 0.8 ms).
  Dynamic clutter costs 6 to 10 ms: not for sweeps.
- Tracker target: <= 20 ms per camera frame (ROI plus cutoff), sensor at 30 Hz.
- A 20 s route: 4,000 steps x 1.5 ms = 6 s, plus about 24 s of rays. About 30 s per run.

## 6. Risks and what NOT to do
1. **Licenses.** Data is odc-by / cc-by-4.0, some CC BY-NC Objaverse pieces; THOR pack license
   not read. Stay on THOR houses; do not redistribute assets or push them to GitHub.
2. **Quotas.** Do not fetch Objaverse (142 GB). THOR expands to 2.4 GB. Keep everything in
   `/data/local/ss5772/molmospaces` (now 3.3 GB; 907 GB free). Never `/common/users` or home.
3. **MuJoCo version.** Upstream pins 3.5; we run 3.9.0 and it loads. A 3.9.x change could shift
   physics. Pin a model hash in tests.
4. **Stability.** House contacts are soft (`solref 0.025`, `solimp 0.998`, elliptic, noslip 4).
   YOR with a 100 Nm torque box at our dt may differ. Phase 1 tests 2 and 3 catch it. Do not
   change house contact parameters globally; override only for payload geoms.
5. **Speed.** No Mode D with dynamic clutter in sweeps; no full 160x90 casts per tick.
6. **Robot fidelity.** Do not edit `YOR/` or limits. Do not claim a wheel model. Do not use
   `YOR/robot/base.py` (hardware CAN). The mocap base is a kinematic idealization; base
   accelerations are an unmodeled disturbance for FL-MPC. Report it that way.
7. **Doorway geometry.** Arms plus tray may be wider than a door (unmeasured). Phase 0
   measures it. Phase 2 may need a tucked-arm pose, untested for FL-MPC: a research item.
8. **Env and scope.** No `pip install` into `mujoco_ur5e` (use a venv under
   `/data/local/ss5772/molmospaces/venv` if ever needed). Pin data versions
   (`DATA_TYPE_TO_SOURCE_TO_VERSION`; upstream is mid-upgrade). Scope: houses stay out of WM training, HJ value work and the vdirect kill test.

## 7. Impact analysis
Intended change set: **new files only**, all under `world_model/v2/house/` (plus this doc).
No existing file changes.
- `repo_graph_agent impact` on `sim_common.py` and `mpc_fl.py` returned `[]` = UNKNOWN.
- Grep (VERIFIED): 73 files import `sim_common`; `mpc_fl` is imported by `rt_flmpc.py`,
  `_demo_env.py`, `gate_equivalence.py`, `flmpc_value.py`, `cbf_filter.py`, `shield.py`,
  `spill_predictors.py`, `keyboard_fl_teleop.py` and more. **Do not edit either file.** The
  house loader builds its own `MjModel` and passes it in, as `load_model()` does.
- `tray_rep/perception/*` is another agent's in-progress work: read-only for us.
  `depth_sim.capture` casts against `scene.m, scene.d`; the adapter must keep that interface.
- If a later phase needs a hook (for example `sim_common.load_model(scene=...)`), stop and ask:
  73 importers make it a wide change.

## 8. Open questions
1. Is a kinematic (mocap) base OK for route demos, with base acceleration logged?
2. Start with `procthor-10k`; textures (0.64 GB) only for videos?

## 9. Integrating the other workstreams
Rule: algorithms and data stay in the fast object-only rig. The house calibrates and validates.
Calibrate, do not import: measure house effects, then put them into the rig.

### 9.1 Rig or house
| Workstream | Fast rig | Needs the house |
|---|---|---|
| Monitor (WM, latent HJ, 85 actions) | training, value, ablations | route validation only |
| Bimanual tray rows in FL-MPC | build and test in `scene_yor_canonical` | base term, route runs |
| Authority per pose | table code (`authority/`) | evaluate along a route |
| Tracker / Bridge (A vs B) | noise data / main result | occlusion stats / "loaded plate" route |
| VLM hazard naming | none | labels, images, scoring |
| Set-based margins | method, calibration | occlusion recalibration, coverage |

### 9.2 Task space vs joint space (the key point)
VERIFIED: the monitor acts on `u = [tray accel (3), tilt rate (3)]`, from the 85-action set
times `a_scale`. `FLMPC.step` (`mpc_fl.py:540`) plans joint accelerations (`q_ddot = a`). Its
decision vector is `z = [ego blocks | second-arm blocks]` (`ntot = 2*n_blk`), with exact
prediction `q(t) = qnom(t) + Phi_q(t) z`. Task limits (cup tilt cone, leash, table; lines about
750-890) are rows linear in `z`, with the end-effector Jacobian frozen at t=0. The cup rows use
the left arm only. `step()` has no generic extra-rows argument (`extra_a` is local).

Plan: new file `world_model/v2/tray_fl/bimanual_tray.py` (subclass or wrapper). No edit to
`mpc_fl.py` until the owner approves one optional `extra_rows=None` kwarg (default
bit-for-bit, equivalence gate re-run; many importers, section 7).
1. **Closed chain.** Rigid grasp: `J_L a_L + Jd_L qd_L = J_R a_R + Jd_R qd_R = a_tray` (tray
   frame, 6-D). On the first block this is 6 equality rows `[J_L, -J_R] z0 = Jd_R qd_R -
   Jd_L qd_L`, plus a Baumgarte term on hand-to-hand pose error. If the QP lacks equality
   support, use tight inequality pairs or a heavy `W_chain` (ASSUMED easy; check). Later
   blocks use frozen-Jacobian rows through `Phi_q`, as the leash rows do.
2. **Monitor rows.** `a_tray = J_L a_L + Jd_L qd_L (+ base term)` is linear in `z`. Build A
   first: the monitor takes `u* = argmax_u Q(x,u)` over the 85 set; FL-MPC tracks `u*` with a
   box `|a_tray - u*.lin| <= delta`, and `u*.tilt` bounds the tray tilt rows. Then B: linearise
   the value, `Q(u0) + grad_u Q.(u - u0) >= eps`, and re-score the final `u` (non-convex).
   Priority: torque box HARD; collision rows with the slack fallback unchanged; chain rows
   hard; value rows slack with high penalty; tracking last. Spill stays the monitor's job.
3. **Mocap base term.** `a_tray_w = R_b(J a + Jd qd) + a_b + alpha_b x r + omega_b x (omega_b x r)
   + 2 omega_b x v_rel`. The planner knows the base terms ahead, so they are a constant on the
   row right-hand sides: the arms produce `u* - base term`. Base motion uses authority, so cap
   `|a_b|`. Base reaction forces are not modelled; do not claim them.
4. **Authority is pose-dependent.** `configs/piper.json`: `a_scale 0.3` (linear cap 1.5 m/s^2),
   `w_scale 0.5` (1 rad/s), slew 10 rad/s^2: constants (VERIFIED). `authority/robot_authority.py`
   shows the real limit depends on pose (inverse dynamics, squeeze LP, caps +-8 Nm, wrist 6.4
   ASSUMED, bisection per direction). House check: (a) offline, per waypoint (each 0.25 m and
   each lift or arm change) run it and store `a_max(dir, pose)` (time per pose not measured);
   (b) online, FL-MPC's torque box IS the per-pose authority if its cap is 8 Nm. VERIFIED risk:
   the sim cap is `TAU_MAX=100`. Plumb 8 Nm into the solver (check how `step` gets it; no
   monkeypatch). Until then house results are "optimistic authority"; (c) scale the 85 set:
   `a_scale(pose) = min(cfg, a_max(pose)/5)`; below half of cfg the planner slows or stops.
5. **Tests, no house:** static bimanual hold in `scene_yor_canonical` (chain residual < 1e-6);
   tray follows a scripted rig `u` (achieved vs commanded); 8 Nm cap agrees with `a_max(pose)`
   within 5%; infeasible case uses the slack fallback and no driver overrides its torque.

### 9.3 Point-cloud perception
Runs on Mode B: furniture static; tray, ball and both arms posed in as ray targets, so arms
occlude naturally. ROI first (your re-run: an empty group still costs about 65 ms, so ROI is
the main lever); budget 20 ms per camera frame. Output: ball estimate, Kalman covariance,
point count, fit RMS, dropout flag. From house runs fit the rig's `tracker_realistic` block
(today ASSUMED: 3 mm, 60 ms, 30 Hz, 2% dropout) plus an occlusion-dropout process (burst
length vs arm pose and furniture), then regenerate rig data. Needs the tracker stable first.

### 9.4 Bridge experiment, house version
Rig (`tray_rep/bridge/`; not present yet, VERIFIED by listing): myopic keypoint-cost filter
(A) vs look-ahead latent-HJ filter (B). The main result lives there. House "carry loaded
plate": same filters, same FL-MPC executor, same seeds; route 3 m, 90-degree turn, doorway,
turn (about 12 m); perception on. Route accel enters through `a_tray` (9.2 item 3), so no
retraining at first order; check coverage (turn accel `v^2/r` = 0.25 m/s^2 at 0.5 m/s, r 1 m;
base jerk) is inside the rig's training set. Metrics: ball lost, min door margin,
interventions, route time, achieved vs commanded `a_tray`; 10 seeds x 20 houses. Run only
after the rig result exists; if A and B tie there, skip it.

### 9.5 VLM hazard naming
Menu = MolmoSpaces `_metadata.json` categories (Chair, Doorway, DiningTable, Bowl, ...).
We write one table: category -> hazard family (obstacle, fragile, liquid container, threshold,
opening, person) and part names. The VLM only picks menu entries. Output is a PRIOR
(inflation, speed cap, rolling-resistance prior); it never certifies. Certification stays
with ray geometry, the monitor and the FL-MPC rows. Free scoring: ray hit -> geom -> body ->
category. House only, GPU later, outside the control loop.

### 9.6 Set-based robust margins
Input: Kalman `P`. Cheapest first: (1) shrink by `L_V sqrt(chi2 lambda_max(P))`, `L_V` fit on
rig data; (2) min of the value at ellipsoid support points (85 x 9 = 765 evals, one batch);
(3) conformal radius from held-out rig runs plus the outcome-conformal audit (vdirect
addendum F). Rig: method, calibration, coverage under the parametric occlusion model. House:
recalibrate with real occlusion; accept if empirical coverage >= 1 - alpha; report the
intervention cost.

### 9.7 Dependency-ordered timeline (engineer-agent days; each step needs those above)
1. Rig only, now, parallel: bridge (rig), tracker, set margins (rig), authority table.
   House Phase 0 (0.5 d) alongside.
2. Bimanual tray rows in `scene_yor_canonical`, tests 9.2 item 5: 5 d. Needs the 8 Nm
   plumbing and the hook decision. Blocks the rest.
3. House Phase 1 (4 d): rows from step 2 in a house, static carry.
4. Tracker in house, occlusion statistics, `tracker_realistic` refit: 3 d.
5. Set margins with house recalibration: 2 d.
6. House Phase 2 route (6 d), then loaded-plate bridge route (3 d). Needs 2, 3, 4 and the rig
   bridge result.
7. VLM menu and scoring (3 d), last; Phase 3 (5 d) in parallel.
Critical path: 5 + 4 + 9 = about 18 days after the rig tracker and bridge finish. Cut line:
stop after step 3; the house is then a perception and route demo.
