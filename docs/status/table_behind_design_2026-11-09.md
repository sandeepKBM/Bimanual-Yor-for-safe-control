# Table-behind carry: build design (2026-11-09)

> Status: DESIGN only. Nothing here is built. Simulation only. Paths are under `world_model/v2/` unless stated.
> Rulings this builds on (`docs/status/decisions.md`, 11-09): table-behind task on the real swerve wheels; success = AgentChord's rule
> (completed, object and mug tilt < 45 deg, xy <= 5 cm, z <= 2 cm), safety reported apart (any U incl. pre-hold; safe success);
> u <= 1.0; re-ask on every rejection, no cap; tray6 only in this task; FL-MPC basic is a tracker skill; the checker owns safety;
> 11-08 the LLM names phenomena, skills supply numbers. Related: `tracker_skills_design_2026-11-09.md` (T4 `base_ff`, cards),
> `probe_selection_build_spec_2026-11-09.md` (Φ, schema v2), `checker_design_2026-11-05.md` (rows 3.1-3.11).
> Code read at HEAD 6b2a6c7: `nero/base/{swerve,scene,smoke}.py` (commit 7d0edfd), `stack/{tasks,goal,outcome}.py`, `stack/checks/*`,
> `stack/proposer/{schema,compile}.py`, `stack/planners/closed_chain.py`, `nero/contents/{carry,envelope,same_worlds}.py`,
> `agentchord_probe/nero/{atoms,tasks,plan,coffee}.py`, their `atom_actions.py` and prompts (read only).

## 0. Facts that shape the design (MODEL numbers read from the compiled scene today)

- Base rotation centre c = mean of the 4 steer pivots = (0.000, -0.210) (MODEL, `scene.base_geometry`). Tray start xy = (0.38, -0.22).
  Goal xy on counter 2 = (-1.38, -0.199) (MODEL, `scene.place_pose_xy`), so goal_dxy = (-1.76, +0.021) m in the world frame.
- Tray centre is 0.38 m from c; a mug at a far tray corner is at most about 0.52 m from c (MODEL, tray 0.22 x 0.33 m).
- Route that works (11-09 smoke): back 0.5 m, turn 180 deg in place, forward 0.5 m. Turning at the pick pose hits the counter.
  End error 8 mm (one smoke, not a bound). Arms held by LeverCtl, which now runs `flmpc_basic` (`carry.py:145`).
- AgentChord's own atoms have **no base atom** (`/data/local/ss5772/agentchord/src/.../atom_actions.py`: grasp, place_on_table, three
  move atoms, back_to_initial_pose, rotate/orient_eef, open/close_gripper). Their `basic_background.txt` says the robot base frame is the
  only frame. Any base motion on their side is an addition we make (section 5).
- U today (`carry._check`) counts robot geoms touching anything except the grasp. It does **not** count the tray or a mug touching the
  world, nor the base body touching a counter (the base is not in `arm_g`; base-world contact is not checked). R reads geom `counter` only.

**Hand estimate (MODEL, quasi-static; G0 below must confirm it in sim).** Peak horizontal specific force at a far mug:
line segments a; turn r·sqrt(α² + ω⁴) with r = 0.52 m.

| Level | v, a, ω, α | line (m/s²) | turn (m/s²) | ratio to spill limit 2.45 | to slide μg 4.9 | to tip 8.3 |
|---|---|---|---|---|---|---|
| L1 (today's swerve `Limits`, ASSUMED) | 0.25, 0.25, 0.5, 0.5 | 0.25 | 0.29 | 0.12 | 0.06 | 0.04 |
| L2 (ASSUMED midpoint) | 0.5, 0.5, 1.0, 1.0 | 0.5 | 0.74 | 0.30 | 0.15 | 0.09 |
| L3 (YOR caps, MEASURED source `swerve.YOR_MAX_*`) | 1.0, 1.0, 1.57, 1.57 | 1.0 | 1.52 | 0.62 | 0.31 | 0.18 |

Limits: spill = g·2·freeboard/D with the coded worst freeboard 1 cm (`objects.CUP`, ASSUMED); slide μ_cup 0.5 (ASSUMED, `scenes.TrayWorld`);
tip = g·D/(2 z_c), z_c = H/2 for the empty mug (worst over fill; ASSUMED uniform mug). Per-claw vertical load does not change with
horizontal acceleration (the lever split's vertical part depends on g only, `envelope.lever_split`). Joint torque rises by the horizontal
inertia of links and load (load ≤ 3.6 kg × 1.52 m/s² ≈ 5.5 N) against 100 Nm. **So the quasi-static rows predict no tray6 failure even at
YOR's caps.** Motion failures, if they occur, must come from what the rows do not model: jerk at the trapezoid corners, base pitch and
bounce on soft wheel contacts, wheel slip, tracking error, contact chatter of free mugs (MuJoCo creep), and tray or base contact with the
counters. This is why the first build step is a sim smoke (G0), not the pipeline.

## 1. Task definition

New `TaskDef` row `tray6_behind` in `stack/tasks.py` (pure data, as the coffee `tray0` row):
- `kind="tray"`, `backend="carry"`, new field `base="swerve"` (default `""` = fixed base; the episode builder uses `base/scene.build_spec`).
- `world_ref` = `TrayWorld` with 6 mugs; `prior` = `_TRAY_PRIOR` (0-0.25 kg per mug, unchanged); `shape` Rect(0.22, 0.33).
- `start_support="counter"`, `goal_support="counter2"` (new support name; the world gets `counter2_top`, equal to `counter_top`).
- `goal_dxy=(-1.76, 0.021)` literal, `goal_frame="world"`. Test (`adapters/tests/test_tasks_match_nero.py`) asserts it equals
  `scene.place_pose_xy(world) - world.obj_xy` (MODEL), so a scene change is caught.
- `goal_terms` = AgentChord's rule: `_XY` 5 cm, `_Z` 2 cm, `_TILT_G` 45, `_CUP_TILT_G` 45 (all exist).
- New field `routes`: named base routes, as data: `("R1", (("line", -0.5), ("turn", +180), ("line", +0.5)))` (ASSUMED, from the smoke).
- `U` = `_u_carry(cups=True)` plus two new flags: `obj_world_contact_flag` (tray or a mug touches any world geom other than the support it
  rests on or is being placed on) and `base_collision_flag` (any base or torso geom touches a world geom other than the floor). Both MODEL
  readers in `adapters/state_carry.py` and monitors in `carry._check` (switched on by the task, so old tasks are bit-identical).
- `R` = `_r_carry(cups=True)` with `support_contact` read on `goal_support` (counter2), not the hard-coded `counter`.
- `rows` = existing three + `base_motion` (section 3) + `sweep` (section 2). If the catalog lands, `phenomena` = PH1, PH2, PH3, PH4, PH6, PH7.

**Worlds.** tray6, `probe._world("tray6", probe._seed("e2e", "tray6", i))`, i < 20: uniform on the full prior, the same list as
`same_worlds` (bounds rules 4 and 8). Smokes use the light world (`TrayWorld(n_cups=6)`) and the heaviest of 40 seeds (`smoke.heaviest_seed`).

**Success with a moved base.** Measured from sim truth in the **world frame**: object xy against the fixed goal on counter 2, z against
`obj_z0 + (counter2_top - counter_top)`, tilts from the body rotation. The base pose never enters the score, so a base end error shows up
as xy error unless the place step corrects for it (section 2). Safety: any U entry, including before the hold, by phase (section 6).

## 2. Plan vocabulary

**New step skill `base_move(route)`** in `stack/proposer/schema.py`. Args: `route` = a list of 1-4 segments, each
`{"line": d}` (metres along the robot's own +x, |d| ≤ 1.0, ASSUMED) or `{"turn": deg}` (|deg| ≤ 180, sign = CCW). No speed argument in v1:
as for arm moves today ("the admission chooses a_admit"), the admission picks the base level (section 3). Q4 asks whether the LLM may ask
for a slower level. Execution: each segment is one `swerve.trap_profile` at the level's (v, a) or (ω, α), a 0.5 s rest after each
(ASSUMED, smoke `gap`), P hold on the segment goal in the rest (`PathFollower.hold_twist`, exists).

**Composition.** A plan is `lift(dz)` → optional `move` → `base_move(route)` → optional `move` → `place("counter2")` → `release`.
Rules (compile time, `stack/proposer/compile.py`): during `base_move` the arms hold their joint references (no arm motion at the same
time, ASSUMED A4); `base_move` needs the object at least `MIN_CLEAR_M` above every support under it (no dragging off counter 1);
`place` must name the task's goal support. Arm steps before `base_move` are in the start base frame; arm steps after it are in the base
frame at the route's nominal end. Recovery skills during `base_move`: `hold` = decelerate the base at the level's a/α and keep holding;
`retrace` = drive the route back to the step's start; `set_down` = retrace all steps and set down on counter 1; `let_go` is accepted by
the schema and refused by the checker (it lands the tray off any support: U).

**Place after a base move.** The place segment is re-planned at the rest after `base_move` from the measured base pose (sim truth: no
odometry error, ASSUMED A5, as `swerve.pose` already states). Before execution the checker checks that place over the corners of the
base end-error box (MEASURED on the base card), so the re-plan cannot reach a pose the admission did not check.

**What `closed_chain` must check for a base plan** (new module `stack/planners/base_route.py`, called by `closed_chain`; keeps the big file
mostly untouched):
1. Reach and joint margins at both tables: today's SRS IK + DP path at table 1; at table 2 the same planner in the end base frame, run at
   the nominal end pose and at the 4 corners (x, y) × 2 (yaw) of the base end-error box. Any miss → `PlanRefused` → re-ask.
2. Swept clearance of the route: a kinematic copy of the scene, the floating base set along the route (1 cm per line step, 2 deg per turn
   step, ASSUMED A7), arms at the held q, tray and mugs at their held pose. `mj_geomDistance` of base, arms, tray and mugs against the
   world ≥ `CLEARANCE_MIN` 5 mm (existing ASSUMED) + the base path-tracking error (MEASURED, lateral and yaw × lever). A hit → `PlanRefused`.
   This is what refuses "turn at the pick pose".
3. Height over counter 2 during the advance: tray underside ≥ counter2 top + `CLEAR_M["tray"]` 0.05 m (existing ASSUMED) + base pitch ×
   lever (MEASURED). Today's smoke lifts only 2 cm; with base pitch that may be the first object-world contact.
4. Segment output: `Segment.base_ref` (route, level, profile), as in the tracker design A2, so the tracker and the checker read one copy.

**Proposer prompt** (`stack/proposer/prompt.py`, task text for `tray6_behind`): the two counters and their world positions; the goal on
counter 2 in world coordinates; that the base can move and the robot's +x turns with it; the `base_move` skill, its segment ranges and
that arms hold during it; route R1 as a known-good option with the scene fact "turning in place at the pick pose hits the counter";
that the planner refuses a route whose sweep collides. No contents prior and no sanity ranges (11-08). Same text facts go to AgentChord.

## 3. Checker rows with base motion

Notation: base frame B at the rotation centre c; base acceleration a_b (line segments), yaw rate ω and yaw acceleration α (turns).
A point at r (in B, from c) held still in B accelerates at `a_p = a_b + α ẑ×r − ω² r_xy` (the Coriolis term is 0 because arms hold).
Gravity stays (0, 0, −g): flat floor, base level (ASSUMED; base pitch is a MEASURED residual). Let μ = (a_bx, a_by, α, ω²).
Every row below is affine in μ for fixed θ, and affine in the contents first moments π = (m, m c) for fixed μ.

**3a. Joint torque (extends `checks/payload`, MODEL).** Per arm: `τ_s = ID_s(q; base ω, base a_b, α) + J_sᵀ h_s`. ID_s = the arm rows of
MuJoCo inverse dynamics of the full floating model with the arm q̇ = q̈ = 0 and the base twist and acceleration set (adapter
`adapters/base_envelope.py`). It contains the links' own inertia under base motion. h_s = lever split (`envelope.lever_split`) of the
object wrench with `acc = a_o − g` in place of `−g`, F = M_tot (a_b + α ẑ×r_c − ω² r_c,xy − g), r_c = total COM. The angular terms
`I α + ω×Iω` need second moments; use the outer box of the checker design 3.1 (MODEL, conservative). Worst case = the vertices of π ×
the corners of the μ box of the level (line: a_b ∈ ±a, ω = α = 0; turn: a_b = 0, α ∈ ±α, ω² ∈ [0, ω²]). Exact for the affine part:
**proof under MODEL**. Admit at u ≤ 1.0 (11-09).

**3b. Per-claw vertical load (MODEL).** Unchanged by horizontal μ (lever split, vertical part = λ g). Base pitch and bounce change it;
that part is the MEASURED excess r5 on the base card. Row = today's payload row + card excess: **sampled** once the excess is added.

**3c. Mug spill, slide, tip (`checks/open_content`, new rows, MODEL, quasi-static).** Specific force at mug i: f_i = a_i − g, with a_i from
the formula above at the mug's r_i plus the tray's own acceleration (zero while arms hold). Worst r_i = the tray footprint corners
(ASSUMED A10: mug positions are visible, but the corners are simpler and conservative). Spill: |f_xy|/f_z ≤ 2(H − h)/D with h the
fullest fill in Θ_t (monotone in fill, today's row). Slide: |f_xy|/f_z ≤ μ_cup. Tip: |f_xy|/f_z ≤ D/(2 z_c), z_c worst over the fill
interval (1-D, exact). **Proof under MODEL, quasi-static.** Jerk, chatter and creep are not covered (dynamic rows).

**3d. Tray tilt (`checks/tilt`, still MISSING today).** Planned tilt is 0 while arms hold. Bound = planned + r3 tilt residual per level,
MEASURED on the `flmpc_basic` base card. **Sampled.** U threshold 10 deg (ASSUMED, existing).

**3e. Base tip-over (new, MODEL).** Robot + load COM projected with the ZMP shift h_com·|a_h|/g must stay inside the wheel contact polygon
by a margin ≥ 0. Affine in π. **Proof under MODEL** (rigid, flat floor). Expected far from binding; it is cheap and closes the question.

**3f. Sweep clearance (section 2 item 2).** Proof for the reference; with the MEASURED tracking margin added it is **sampled**.

**3g. Dynamic trials (`checks/dynamic`).** Worlds from Θ_t, run the plan and each segment's backup from branch points (also inside
`base_move`). Pass = no U and end in S ∩ R. **Sampled**, Clopper–Pearson, n stated. This is the only row that sees jerk, slip, chatter.

**How admission picks the level.** Levels L1 < L2 < L3 (≤ the run's cap). All model rows are affine in μ, and the μ box grows with the level,
so pass/fail is monotone in the level. The admission takes the highest level ≤ cap at which every row passes on Θ_t; none → refuse →
Rule 2 T2 re-ask. "Slow down" is then an action: a recovery program "base level from the weigh reading" is checked for every θ in Θ_t at
the level it would choose. Sampled rows use the card measured at that level only (no extrapolation between levels unless AS3 is ruled).

**New MEASURED quantities.** Base card per level and segment kind (line, turn): path-tracking error (lateral, yaw) during the segment;
end error after the rest; actual peak base acceleration and jerk vs the command; base pitch and roll peak; wheel slip peak. Arm card
(`flmpc_basic`, then `+base_ff`) under base motion per level: object acceleration excess r4, tilt r3, TCP error r2, torque excess r5,
internal wrench r6, mug slide on the tray (mm) and mug tilt. Corners of Θ_0 + random interior worlds, n = 20 per condition (AS2).
Report p50/p95/max over n with the noise model (sim truth state, no odometry error, MJCF floor friction).

## 4. Tracker

`flmpc_basic` already reads `qfrc_bias` of the full model (`flmpc/basic.py:106`), so base-velocity terms are in its bias. What it lacks is
the base-acceleration coupling `M_arm,base a_b` and the payload wrench at `a_o − g`; base acceleration acts as a disturbance its feedback
removes. `base_ff` (tracker design T4) adds both from the **planned** base profile (AS8), through LeverCtl's existing `(M_extra,
bias_extra)` hook, so the FL-MPC core is not touched.

Build first: `flmpc_basic` alone, because it exists and both methods already run it. Measure its base card at L1-L3 (G1). Build `base_ff`
next, as a wrapper in a new file `nero/base/base_ff.py`, and measure it on the same worlds. Keep it if r3/r4/r5 drop at L2-L3; the checker
uses whichever card the plan's segment names. Both methods run the same tracker configuration in every comparison cell.

## 5. AgentChord side

Faithful to their framework: their TaskAgent → RecoveryAgent → CompileAgent, their prompts and their monitors; no checker, no admission,
no level choice. The only changes are the ones a base forces, each listed in `SETUP_DEVIATIONS`:
- **New atom `move_base(robot_name, route, **kwargs)`** added to a copy of their `atom_actions.txt` in their style ("Use the following
  functions exactly as defined"). It must appear in both arms' slots of one edge with the same route (as the coupled tray move,
  `atoms.ARM_TOL` logic); otherwise `rejected_arm_mismatch`. It drives the same swerve controller (`PathFollower` + `SwerveBase`) at the
  **cell's cap** (they have no speed reasoning), arms hold, same tracker and same feedforward mass (prior mid-contents, as today for theirs).
- **Frames.** Their prompt says the robot base frame is the only frame. Both methods get world coordinates (= the start base frame);
  atoms resolve world targets with the sim-truth base pose. This extends the existing deviation "coordinates are world-frame metres".
- **Monitors** (`monitor_object_held`, `monitor_object_moved`) run every `MON_PERIOD_S` during the base edge; an abort decelerates the
  base at the cap, then their recovery branch runs (their robot would freeze; a moving base cannot stop instantly).
- **No route help beyond the shared facts.** Their LLM gets the same task text and scene facts as ours (incl. route R1 and the
  turn-at-pick fact). A colliding route runs and the sim shows the collision.
- New module `agentchord_probe/nero/behind.py` (the `coffee.py` pattern: `install()`, task text, atom, `run_world`); `atoms.py`,
  `execute.py`, `tasks.py` unedited. Drop injection off (p = 0) unless the user asks.

## 6. Run protocol

- **Worlds:** tray6_behind, i < 20, the same list for both methods and every cell (bounds rule 8). Same proposer model for both:
  Codex `gpt-5.6-sol:high`, every request cached, then an `--offline` replay that must match (bounds rule 9).
- **Grid of caps (stage 1):** L1, L2, L3 as one knob (v, a, ω, α together). 3 cells × 2 methods × 20 worlds. One plan per method per cell
  (the caps are in the prompt facts only as robot facts).
- **Stage 2 (only if stage 1 shows failures):** split linear and yaw caps, 3 × 3, to attribute failures to lines vs the turn.
- **Report per cell and method:** success, unsafe (kind × phase: pre-hold, lift, arm carry, base line, base turn, place, release), safe
  success, refusals by row and guarantee type, plan_rejected and re-asks (count, reasons), base level chosen (ours), base end error,
  xy/z error at the goal, peak mug slide, tray tilt, spill ratio. Counts with Clopper–Pearson 95% bounds; n = 20 gives rates with wide
  intervals. Each row lists the box (tray6 prior, 0-0.25 kg per mug), the tracker card and the ASSUMED inputs.

## 7. Build order (what blocks what; smallest smoke first)

| # | Piece | Files | Blocks / waits for |
|---|---|---|---|
| G0 | Sim smoke: route R1 at L1/L2/L3, light + heaviest tray6, U incl. the two new flags (smoke-local readers), peaks of base accel, pitch, mug slide | `nero/base/smoke.py` (level arg) | Nothing. **Gates the rest: if L3 shows no U and no near-miss, ask the user (Q2) before building more.** |
| G1 | Base card + `flmpc_basic` card under base motion | new `nero/base/card.py`, `nero/results/base_card_*` | G0. Blocks 3b/3d/3f margins, place-corner checks |
| B1 | Task row, `counter2` support, U flags, R on goal support | `stack/tasks.py`, `adapters/state_carry.py`, `nero/contents/carry.py` (`_check`, `in_R`, task-gated) | Other agents edit `tasks.py`/`carry.py`: land after them |
| B2 | Base driver promoted from the smoke (`BaseDrive` → executor hook; segment `base_ref`) | new `nero/base/drive.py`; `stack/executor.py` hook | G0 |
| C1 | Rows 3a, 3c, 3e (pure) + MuJoCo ID adapter | new `stack/checks/base_motion.py`, new `adapters/base_envelope.py`, tests | Independent of B1/B2 |
| D1 | `base_move` skill, compile rules, sweep + table-2 reach | `stack/proposer/{schema,compile,prompt}.py`, new `stack/planners/base_route.py`, small hook in `closed_chain.py` | B1; proposer files are also edited by the probe-selection build (schema v2): coordinate |
| T1 | `base_ff` wrapper + its card | new `nero/base/base_ff.py` | G1, B2; tracker interface step 1 if it lands first |
| A1 | AgentChord `move_base` atom + task install | new `agentchord_probe/nero/behind.py`, prompt copy | B1, B2 |
| R1 | Same-world runner for the base task and the level grid | `nero/contents/same_worlds.py` (`--level`) | all above; `same_worlds.py` is shared with the probe fix |

Parallel without shared files: G0/G1 (base package) ‖ C1 (new checks + adapter) ‖ A1 drafting (new module). B1, D1 and R1 touch files
other agents are editing (tasks, carry, proposer, same_worlds) and should be sequenced with them, one owner per file.

## 8. New ASSUMED values (for the user to rule) and open questions

ASSUMED (new or newly load-bearing):
- A1 level grid: L1 = today's swerve `Limits` (0.25 m/s, 0.25 m/s², 0.5 rad/s, 0.5 rad/s²; not yet ruled); L2 = 0.5, 0.5, 1.0, 1.0; L3 = YOR caps
  1.0, 1.0, 1.57, 1.57 (values MEASURED from YOR source; their use as a cap is our choice).
- A2 route R1 and `RETREAT_M` = `ADVANCE_M` = 0.5 m; counter 2 = point mirror (both in `base/scene.py`, not yet ruled).
- A3 `base_move` ranges: line ≤ 1.0 m, turn ≤ 180 deg, ≤ 4 segments; rest 0.5 s after each segment.
- A4 arms hold still during base segments (no concurrent arm and base motion).
- A5 base pose for the place = sim truth (no odometry error).
- A7 sweep sampling 1 cm / 2 deg; clearance 5 mm reused; tray clearance over counter 2 = `CLEAR_M` 0.05 m reused.
- A8 backup during `base_move` decelerates at the level's a/α.
- A10 mug positions taken as the worst tray corner in rows 3c.
- μ_cup 0.5 and the empty-mug COM at H/2 now enter checker rows (existing values, newly load-bearing).
- AgentChord: base atom at the cell cap; world coordinates in their prompt.

Open questions:
- Q1 Rule the level grid. Any cell above YOR's caps? (Default: no; not robot-faithful.)
- Q2 The hand estimate says tray6 stays inside every quasi-static limit even at L3. If G0 agrees, what should create motion risk:
  concurrent arm + base motion, a floor threshold or ramp, taller or fuller mugs (less freeboard), a heavier object on the route (box,
  pot), or accept "no motion failures at robot caps" as the result?
- Q3 11-09 put visible loose items (PH6) out of scope. This task needs the mug slide and tip rows (MODEL, μ_cup ASSUMED). Add them here?
- Q4 Admission picks the base level, the LLM names only the route (default)? Or may the LLM ask for a slower level?
- Q5 Place from sim-truth base pose (default), or model an odometry error?
- Q6 Accept `move_base` as an added atom in AgentChord's vocabulary (a stated deviation)? Alternative: the base move is scripted between
  two halves of their plan (not chosen by their LLM).
- Q7 New U flags (object-world contact, base collision): this task only (default), or all carry tasks after a replay check?
- Q8 tray6 only, or tray4 as well?

Uncertainties: MuJoCo mug creep under base vibration may produce sim-only slides (Drake spike 10-30 saw 40× more creep in MuJoCo);
the YOR drive servo has no force limit, so the sim base may accelerate harder than the real one under P feedback; base pitch on soft
wheel contacts is unmeasured; the 8 mm end error is one smoke.
