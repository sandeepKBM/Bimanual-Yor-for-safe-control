# The checker today, what the measured geometry set needs, and what to retire (2026-11-19)

Status: plan, nothing in parts 2 and 3 is built. Task: `box_bookcase` (lift the crate, the swerve base drives and turns, the arms insert
the crate into the bookcase compartment, lower, release). Paths are relative to `world_model/v2/` unless they start with `docs/`.

Source of part 1: a read-only code trace (11-19) of the current tree, legacy switches ignored (they are being removed). I have not run
every row; where a row's internals were not read, it says so. Notation: Θ = the hidden-contents set (total mass interval and the
centre-of-mass polytope). "Backend" = `adapters/base_sweep.py` (the one geometry source since 11-19). "Compiled scene" = the MuJoCo
model built from the world (`BS.compose(world)`).

---

## Part 1. What the checker does now

### 1.1 The pipeline, in order (`nero/contents/probe_select.py` `run_world`)

1. **Compile the proposer's plan** (`stack/proposer/compile.py` `_compile_behind`). It checks the plan's shape:
   - each step moves at least 5 mm;
   - the plan never drags the crate lower than 2 cm above a support;
   - the place point is over the goal support (oriented test);
   - the plan ends on the goal support, with a miss of at most 3 cm;
   - there is exactly one base move, and every value is in range.

   Then the planner builds the arm and base motion (`closed_chain.geometry_for`).
   On failure: the plan is rejected and Codex is asked again with the reasons.
2. **Probes on the contents** (weigh, claw readings). Each probe motion is first checked by the probe-limit rows. A contradiction
   between a reading and the belief ends the run.
3. **Coverage.** Every phenomenon the task lists must be measured or have an approved fallback. Otherwise: refuse.
4. **Static admission** (`checks/admit.py` via `adapters/static_admission.py`): torque/payload rows and the support row at the plan's
   end pose. On failure: slow the plan down if that is enough, else refuse and ask Codex again.
5. **Base speed level** (`adapters/base_envelope.pick_level`): the highest of three levels at which the base does not tip and the arm
   torques stay in range. No level passes: refuse.
6. **Planned-path rows** (`probe_select.planned_path_rows`): joint limits and speeds, reach, the two arms agree on the object pose,
   measured load tilt, collision. Any failure or missing input: refuse (planned path), ask Codex again.
7. **Rule 1, per step** (`adapters/rule1_contents.run_cell`): if the robot must stop at any time during this step, does a safe recovery
   exist (stop, hold, retrace, set down) for every θ in Θ? For the base move, the base-stop rows. A fixable failure goes back to Codex
   (up to 3 times); a non-fixable one refuses.
8. **Certificates** are bound to the plan (hash of the plan inputs). A mismatch refuses.
9. **Execution with runtime monitors** (`stack/executor.py`, `nero/contents/carry.py`, `nero/base/monitors.py`): every control step is
   checked in the simulation; a contradicting reading triggers Rule 3/Rule 4.

### 1.2 Every row

| Row | Where | When | What it checks | Over what | Guarantee | Geometry it reads | On fail |
|---|---|---|---|---|---|---|---|
| plan shape | `stack/proposer/compile.py` | compile | step size, no drag, place over support, end on goal ±3 cm | the plan | proof (arithmetic) | backend supports (top, extents, yaw), goal, start pose; crate size; crate start | reject, re-ask |
| roof clearance | `closed_chain.default_waypoint` | default plan only | crate top ≥ 5 mm under the compartment roof at lift height | one plan | proof | backend roof height, crate height, counter top | refuse |
| route form | `base_route.default_route_form` | default plan only | back-off and turn sweep clear | poses every 1 cm / 2° | MODEL, sampled poses | backend start pose, goal, required end pose; compiled scene | refuse |
| reach / IK | `closed_chain` `plan_geometry` | planner | both arms reach every path point inside joint limits | samples every 1 cm | MODEL | hold pose (`capture_nyu.json`), path from the plan's deltas | refuse |
| base height | `base_route.check` | planner | crate underside ≥ `CLEAR_M` (3 cm, ASSUMED) above the supports during the base move | one hold pose | proof | backend counter top, compartment top | refuse |
| base sweep | `base_route.check` → backend `sweep` | planner | robot and load stay ≥ 5 mm from every scene geom along the route | poses every 1 cm / 2°, arms held | MODEL (kinematic), sampled poses | **compiled scene** (convex-hull distances) | refuse, names the segment and pair |
| timing | `closed_chain` | planner | time-optimal timing inside speed limits | the path | MODEL | joint and speed limits | refuse |
| torque / payload | `checks/payload`, `admit.py` | admission | joint torques ≤ limit with the load split on both arms; claw force cap | every Θ vertex × acceleration box, 6 path samples per move | proof under MODEL (on the samples) | planner arm poses, crate empty mass/COM | slow down, else refuse |
| support (goal) | `checks/support.support_row` | admission | the total COM lies over the crate–support contact area at the plan's end pose | every Θ vertex (exact) | proof under MODEL; placement error not covered | backend support (oriented), plan end pose, crate footprint | refuse (placement) |
| coverage | `checks/coverage` | before admission | every phenomenon measured or on an approved fallback | the list | — | none | refuse (uncovered) |
| base tip | `checks/base_tip` | level pick, base stop | ZMP inside the wheel polygon on lines and turns | Θ vertices × MEASURED acceleration card | sampled (card n) | compiled scene masses at the hold pose; crate empty mass from the drawn crate | no level → refuse |
| base torque | `checks/base_motion` | level pick | arm torques with base acceleration | Θ vertices × acceleration corners | sampled | compiled scene (inverse dynamics at the hold pose) | same |
| joint / speed / reach / closed chain | `adapters/motion_rows`, `checks/motion` | planned path | margins on joints, speeds, reach, both arms agree on the object pose | every control sample | proof under MODEL | planned arrays, arm model | refuse, re-ask |
| load tilt | `tilt_measure.measure_plan`, `checks/tilt` | planned path | measured load tilt of THIS plan, base move included | n = 20 simulated carries from the belief cell (n ASSUMED) | MEASURED, sampled | **runs the compiled scene** | refuse; also feeds the collision rows |
| collision (arm–arm, arm–world, load–world) | `adapters/collision_rows`, `checks/collision` | planned path | minimum distance per class ≥ 5 mm, with a speed bound between samples and the measured load tilt | every 1 cm sample + swept bound | sampled (tilt makes it sampled) | **compiled scene** (`mj_geomDistance`, GJK) | missing tilt → MISSING; refuse |
| Rule 1 recovery torque | `rule1_contents` `motion_row` | per step | torques on every recovery motion (stop, hold, retrace, advance) | Θ vertices, branch times every 0.25 s | proof, capped to sampled by the 0.25 s grid | planner path | fixable → re-ask |
| Rule 1 recovery collision | `rule1_contents` `collision_row` | per step | collision rows on the recovery paths | 1 cm samples + refinement | sampled | **compiled scene** | fixable |
| landing at start | `rule1_contents` `landing_row` | per step | setting the crate back down on the counter is safe for all Θ | Θ | proof | **counter via `SC.support_tops`** (not the backend) | fixable |
| landing at goal | same | per step | setting it down on the compartment floor is safe | Θ | proof | backend goal support, goal xy | fixable |
| landing "hold" | same | per step | holding in the arms is not a safe landing (USER 11-17) | — | refuted | — | step refused |
| reading coverage | `checks/rule1` | per step | the claw-reading bins cover all possible readings | Θ | proof | claw read spec | not fixable |
| base stop (9 rows) | `rule1_contents.base_stop_record`, `checks/base_stop` | the base-move step | stop at any time: stop distance + MEASURED card extra + lateral/yaw drift (4 corners); remaining route clear; tip, torque, landing | branch times × 4 drift corners | sampled | backend sweep, start pose, levels; MEASURED stop card | mostly not fixable |
| certificate binding | `executor._check_certs` | before running | every step has an admitted certificate tied to this plan's inputs | — | arithmetic | plan-inputs hash (includes supports and goal) | refuse |
| runtime monitors | `carry._check`, `_obj_world_check`, `nero/base/monitors.py`, `in_R` | every control step | torque saturation, claw payload, tilt, drop, any non-grasp contact, load–world distance ≥ 5 mm, wheel lift-off, base contact, route finished, final rest on the goal support | the real simulated run | violation count (a monitor, not a guarantee) | **compiled scene contacts** | the segment stops |
| Rule 3 / Rule 4 | `executor._interrupt`, `adapters/rule4_contents` | on a contradicting reading | Rule 1 rows on a transfer from the current state | the new Θ | proof/sampled | as Rule 1 | uncertified cascade, counted |

Not read in detail: the inside of `rule4_contents.transfer_row` and of `tilt_measure.measure_plan`.

### 1.3 Where the scene geometry enters (all from the sim scene today)

| Source | What | Used by |
|---|---|---|
| backend `supports` | support top heights, extents, yaw, centre, roof height | plan shape, support row, base height, default plan, plan hash |
| backend `goal`, `object_box`, `required_end_pose`, `support_geo` | goal xy, pre-insert xy, heading, insert length; crate size, start xy, yaw | plan end pose, support row, route |
| `SC.support_tops` | counter top | path building, **Rule 1 start landing** (a second reader of the counter, outside the backend) |
| compiled scene | every board, counter, base, arm, crate geom | base sweep, collision rows, Rule 1 collision, tilt measurement, runtime monitors, base tip/torque masses |
| constants | `CLEAR_M` 3 cm, clearances 5 mm, drag 2 cm, goal miss 3 cm (ASSUMED) | the margins that a measured set will use up |

---

## Part 2. What the checker needs to use the measured geometry set

### 2.1 The rule

The world lies somewhere in the measured set G: one interval per field per entity (`stack/geometry.py`). The intervals come from probe
readings with errors from the noise study, so they are sampled, "max over n".
- **The checker must admit only if every row holds for every geometry in G.** Each row also states how it covers G: proof (all of G,
  under a stated model) or sampled (corners/draws, with n).
- **A field with no reading is MISSING**, so every row that needs it refuses (fail closed).
- The runtime monitors stay in the true simulated world. They watch what happens; they are not certificates.

### 2.2 How each kind of row covers G

| Kind | Rows | How it covers G | Guarantee | Difficulty |
|---|---|---|---|---|
| A. Closed form | plan shape, support (goal and start), base height, roof, goal check | interval arithmetic: shrink each support by its position/size half-widths; take the crate at its upper size; yaw by its interval (corners at the interval ends + a bounded remainder); heights by their worst end | proof under MODEL (over all of G) | low |
| B. Distance to the scene | base sweep, collision rows, Rule 1 recovery collision, base-stop continuation | build a **grown scene**: each board/counter becomes one box that contains every member of its set (centre ± half-width, upper size, the yaw interval swept); the carried crate at its upper size. If the robot clears the grown box by d, it clears every member by at least d. The existing rows then run once, unchanged. | same as the row today (MODEL / sampled poses), now over all of G | medium: one builder, then reuse |
| C. Mass model | base tip, base torque, torque/payload | the crate's size changes its empty centre of mass. Add the crate-size interval corners to the Θ vertex set. | as today (sampled card × vertices) | medium |
| D. Simulated | load-tilt measurement (n = 20 carries) | draw the carries' worlds from G (corners + random), not from the true world; state n | MEASURED, sampled | medium (costly in compute) |
| E. Planning | reach / IK, timing, default route | plan on one geometry in G (the midpoint), then rows A–D check the plan over all of G | the plan is a proposal; the rows certify | low |

### 2.3 The parts to build

| # | Part | File(s) | Owner gate (what it proves) |
|---|---|---|---|
| M1 | **Set backend**: same calls as `base_sweep` (`supports`, `goal`, `object_box`, `required_end_pose`, `support_geo`, `start_pose`, `compose`, `sweep`), answering from a `GeometrySet`. Point values = midpoints, for planning. Bounds = lo/hi, for the rows. `compose()` returns the grown scene. MISSING field → raises `GeometryMissing`. | new `adapters/geom_world.py` | on the 250 study fits: every grown box contains the true board; the crate at the upper size contains the true crate; a missing field refuses |
| M2 | Rows of kind A take bounds | `checks/support.py`, `compile.py`, `base_route.check`, `closed_chain.default_waypoint`, `rule1_contents.landing_row` (and the start landing moved from `SC.support_tops` into the backend) | brute force: when a row passes on G, it passes on 1000 random members of G |
| M3 | Rows of kind B run on the grown scene | `collision_rows`, `base_sweep.sweep`, `rule1_contents.collision_row` (via `compose()`) | when a row passes on the grown scene, no sampled member of G violates it |
| M4 | Kind C: crate-size corners in the vertex set | `base_tip_model`, `base_envelope`, `contents_specs` | worst vertex over the corners ≥ the value at 200 random members |
| M5 | Kind D: tilt worlds drawn from G | `tilt_measure.measure_plan` | n stated in the record; same seeds across methods |
| M6 | Plan hash over the set (interval bounds, not true values) | `closed_chain.PlanInputs.key` | two sets → two keys; the same set → the same key |
| M7 | Probes fill G: `run_probe` readings from the noise-study error table (after the box-fit fix and the study rerun) | `adapters/geom_probe_run.py` | the true world lies in G for the 50 study worlds (count, n stated) |
| M8 | Same-world comparison: truth backend vs set backend, worlds 0–49, same seeds, same Θ | `nero/results/...` | admission rate and refusal reasons side by side |

The truth backend (today's) stays as the test oracle and for the comparison. It is not a legacy path.

### 2.4 Values to rule on before results rest on them

1. **Set half-width** = max |error| over n (sampled) from the noise study. Add a safety factor or not (ASSUMED if added)?
2. **Planning point** = the midpoint of each interval (recommended), or the worst case?
3. **Tilt draws over G**: n (today 20, ASSUMED) and corners + random.
4. **Margins that the set uses up**: `CLEAR_M` 3 cm, 5 mm clearances, goal miss 3 cm. These stay as they are; the set's width is added on top.
5. **Robot's own start pose**: from the robot (odometry/model, MODEL), not measured by the cameras. (The base starts at a known pose in sim.)

---

## Part 3. Retire and delete

### 3.1 Legacy switches (running now)

About 60 `BIMANUAL_LEGACY_*` switches, `TAU_MIT16`, `U_OBJ_CONTACT=0`, the old checker with `T_CAP`, the v1 proposer, `load_legacy`,
`exact_torque.legacy_*` and the `mpc_fl` alias are being removed by two agents. They go in commit 2, on a green suite.

### 3.2 Not imported by any current code: delete

Facts from a read-only inventory (11-19; importer check by name, so dynamic loads could be missed; the suite after deletion is the
real check).

| Path | Files | Size | Last commit | What it is |
|---|---|---|---|---|
| `archive/` (world_model, experiments, hj_reachability, scripts) | 328 | 4.3 MB | 11-16 | v1 world model, HJ value nets, old demos |
| `experiments/` (ab_smooth_vs_bangbang, realtime, actuation, baselines, grasp_audit, imitation, slosh, offload_probe) | 113 | 1.6 MB | 11-03 | cup-era controllers, v4-C\* baseline, mpc7, realtime study |
| `eval/` | 98 | 0.1 MB | 09-21 | cup-era eval |
| `control/` | 11 | 0.1 MB | 09-28 | `box_qp`, `realgrasp_flmpc` |
| `collection/`, `teleop/`, `randomization/`, `offload/` | 23 | 0.3 MB | 09-28 | cup data collection, teleop, cup DR |
| `scripts/` | 26 | 0.4 MB | 11-16 | cup and YOR smoke scripts, old flagship cup demo |
| `research_notes/`, `reports/` | 26 | 0.9 MB | 10-13 | notes |
| `docs/archive/`, `docs/design/`, `docs/runbooks/` | 25 | 0.7 MB | 11-05 | superseded per `docs/README.md` |
| `world_model/v2/{latent_hj, qp_safety, bead_rep, constraints, eval, wm_audit}` | 32 | 0.2 MB | 11-16 | dropped thesis lines |
| `world_model/v2/nero/archive/` | 25 | 0.2 MB | 11-09 | old nero code |
| `scenes/` files no current code uses (`scene_bimanual_*`, `scene_yor_cup*`, `*_pan*`) | ~11 | ~0.2 MB | 10-29 | old scenes (keep the 5 used ones) |

Notes:
- `experiments/` holds the v4-C\* baseline (`run_ab4.py`) that AGENTS.md calls "kept for comparison", and the September cup demo.
  Deleting them retires those baselines; git keeps them.
- Old `docs/status/` notes marked history in `docs/README.md` (about 25): delete, or move to `docs/archive/` before it goes? Proposed: delete; git keeps them.

### 3.3 Still imported by current code: migrate, then delete

| Path | Size | Who imports it | Step |
|---|---|---|---|
| `world_model/v2/house/` | 5.5 MB | `stack/producers.py`, `stack/proposer/{mock,propose}.py`, `nero/contents/proposer_compare.py`, `stack/tests/test_coverage.py` (`house.lang.backend`, `house.ocs.state`) | move the two live modules into `stack/`, then delete |
| `world_model/v2/tray_rep/` | 1.4 MB | `stack/producers.py`, `stack/skills/pregrasp.py`, `stack/universes/{democ,pot}.py`, `vlm/...` (`sets.ir`) | move `sets.ir` into `stack/`; retire the democ/pot universes |
| `world_model/v2/nero/pan/` | 0.3 MB | `adapters/state_pan.py`, `nero/contents/proposer_compare.py` (`pan.llm` = the proposer model spec) | move `pan/llm.py` into `stack/proposer/`; retire the pan tasks |
| `world_model/v2/agentchord_probe/` | 0.3 MB | `nero/coffee_run.py`, `same_worlds.py`, `test_tasks_match_nero.py` | retire with the coffee tasks |
| `world_model/v2/nero/democ/`, `testbeds/`, `perception_zono/`, `data/` | 0.4 MB | `probe_liftoff.py`, `nero/adapters.py`, `nero/ik.py`, `vlm/...` | move the few used functions; delete |
| fixed-base tasks not used by the behind tasks (`tray0*`, `tray2/4`, `pot`, `pan_*`, `coffee_tray*`) | — | `stack/tasks.py` | remove from the task table; keep `box` and `tray6` (the behind tasks build on them) |

### 3.4 Results and data (not code)

- `world_model/v2/nero/results/`: 3108 tracked files. The pan-era runs (`pan_*`, `contact_model_2026-10-30`; 1158 files in
  `pan_fair_box_2026-10-29` alone) are generated experiment results. Proposed: copy them to the cold bank
  (`westeros:/data/local/ss5772/coldbank`, with a manifest), then remove them from git. Keep
  `base_card_G0b_2026-11-09/CARD.json` and `base_stop_card_2026-11-16/` (read by current code) and the 11-19 runs.
- Untracked data folders in the main checkout (`collection/data`, `world_model/data`, `hj_reachability/datasets`, `world_model/logdir`):
  datasets and logs. They are not touched without your explicit word.

### 3.6 Retired gates (USER 11-19: "Retire the gates")

Fixed-base task gates that fail once the collision rows always use the measured load tilt (the fixed-base box nominal plan is refused at
lift, 4.4 mm < 5 mm, n=4 tilt). Skip-marked now; deleted with the 3.2 list.

| Gate | Scope |
|---|---|
| `adapters/tests/test_wide_table_gate.py` | whole file |
| `stack/tests/test_assumed_sweep_gate.py` | whole file |
| `stack/tests/test_collision_gate.py::test_collision_rows_admit_nominal_plans` | one test |
| `stack/tests/test_rule1_gate.py::test_rule1_admits_only_inside_the_static_admission` | one test |
| `stack/tests/test_rule4_gate.py::test_box_trap_offgrid_set_down_is_a_safe_park` | one test |

### 3.5 How deletions happen

1. Legacy switches out → suite green → commit 2.
2. I give you one `!` line that runs `git rm -r` on the 3.2 list in the worktree. You run it.
3. Full suite. If green → commit 3. If anything breaks, I restore that path from git and report it.
4. 3.3 migrations one folder at a time (Sonnet agent moves the live pieces; I gate; then your `git rm` line; suite; commit).
5. 3.4: cold-bank copy with checksums → your `git rm` line → commit.

---

## Order of work from here

1. Finish the current series: legacy removal, box-fit fix, gates, error-table rerun, commits 1–2.
2. Retire 3.2 (commit 3), then 3.3 and 3.4.
3. Build M1–M8 (each with its gate), then the loop wiring and the first end-to-end run on a varied world.
