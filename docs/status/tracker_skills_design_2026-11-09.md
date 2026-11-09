# Tracker skills, and what could help the proposer (design, 2026-11-09)

> Status: DESIGN only. Nothing here is built. Simulation only. Paths are under `world_model/v2/`.
> Rulings this builds on (`docs/status/decisions.md`, 11-09): "FL MPC is a skill to be used"; the FL-MPC skill is the basic one
> (`flmpc/basic.py`, being built: FL + acceleration MPC, torque box + acceleration box, no safety rows, no fallback); the checker alone
> owns safety; 11-08: the LLM names phenomena and choices, skills supply numbers. Related: `probe_selection_build_spec_2026-11-09.md`
> (skill catalog, cards, schema v2), `checker_design_2026-11-05.md` (`TrackerModel`, A2-A5, rows 3.1-3.7).

## A1. Candidate tracker skills

A tracker turns a planned reference into joint torques every 5 ms. It never decides safety. Each tracker below is a skill with its own
measured card (A3). The plain MuJoCo position servo is weighed and dropped (end of this section).

**T1 `flmpc_basic` (first; being built).** Feedback linearization `τ = M a + c` plus a joint-acceleration MPC with a 32-step preview
(NC 4 × L 8 × 5 ms = 160 ms, `nero/hold_test.FL_KW`). Constraints: actuator torque box (`config.tau_limit()`, MODEL) and acceleration box
(800 rad/s², ASSUMED, inherited). An infeasible step saturates the torque and is counted (`n_sat_steps`).
- Inputs: joint references q, qd (qdd) for both arms over the horizon; a payload feedforward (M_extra, bias_extra) from a load model.
- Needs from the plan: smooth joint references (closed_chain planner, SRS IK + TOPP-RA); the nominal load (see A2, "load model").
- Helps: stiff, previewed tracking for lift, carry, place; the weigh probe (today's calibration used the same core, `LeverCtl`).
- Cost: core exists (`flmpc/basic.py`). The driver to port: lever-split payload feedforward (`carry.LeverCtl.terms`) and the
  friction wrapper (`carry.FRIC_COMP`, default on via `config.carry_physics()`).
- Cannot: see base motion (its model is the arm block of M and `qfrc_bias`; base acceleration acts as an unmodelled disturbance);
  avoid internal force under the weld (two stiff trackers with different errors push against each other: 11-02 diagnosis, about 19 N
  along the grasp line and 1.2-1.7 Nm twist); treat the arms alike (weights left 3000 / right 600: `W_POS_EGO` vs `W_POS_INT`).

**T2 `ct_pd` (computed-torque joint PD + feedforward).** `τ = M (qdd_ref + Kp e + Kd ė) + c + Jᵀ h_ff`, clipped to the torque box. No QP,
no preview. Two gain presets: `stiff` and `soft` (A3/C, ASSUMED values).
- Inputs and plan needs: as T1, plus qdd_ref (TOPP-RA gives piecewise-constant path acceleration, so qdd jumps at knots).
- Helps: a plain baseline whose error dynamics are linear with known gains. The checker can then carry a MODEL bound on the tracking
  error (from a disturbance bound) next to the MEASURED one, and compare them. `soft` gives a compliant joint hold for base motion.
  It has the same command form as the NERO MIT mode (kp, kd, t_ff; `config.TORQUE_PRESETS`).
- Cost: small. Pieces exist: payload terms (`LeverCtl.terms`), arm dynamics (`contents/rules.arm_dyn`), friction wrapper.
- Cannot: preview (it lags sharp references); handle saturation gracefully (clip only); fix the weld internal force (same as T1).

**T3 `coop_imp` (cooperative object-level impedance, Caccavale-style).** Existing `contents/coop_imp.CoopImpCtl`: object pose impedance
from the object reference, a PI loop that drives the internal wrench to the squeeze target (0 N, USER 11-03), joint null-space PD.
Wrench estimate from joint torques (no wrist F/T sensor).
- Inputs: object reference x_obj(t) (Segment.x_obj), grasp matrix from the weld, load model, joint references for the null space.
- Helps: the closed chain (internal wrench is regulated, not a by-product); base acceleration and turns (the object is held
  compliantly, so a base disturbance moves the object a little instead of loading the weld); the "compliant" side of a stiff-vs-compliant
  comparison on the table-behind task.
- Cost: medium. The law exists but lives in `nero/contents` and subclasses `LeverCtl`; it must be ported to the tracker package (A2)
  without the carry episode. `coopimp_safe` (coopimp inside full FL-MPC) is not ported (its rows belong to the checker now).
- Cannot: track the object as tightly (object pose error is its main residual, by design); gains are ASSUMED and uncalibrated (USER 11-03:
  "gains need calibration"); the law was written from the paper's structure, with deviations listed in its docstring.

**T4 `base_ff` (an option on T1-T3, not a separate tracker).** Adds the known base motion to the tracker's model:
(a) arm-base coupling `τ += M_arm,base(q) a_base` (MODEL, off-diagonal block of the full MuJoCo mass matrix); (b) the payload wrench at the
object's acceleration `a_obj - g` instead of `-g` (lever split at the base's planned linear, centripetal and yaw acceleration).
- Inputs: the planned base profile (pose, twist, acceleration: `nero/base/swerve.trap_profile`, `PathFollower`). Planned, not sensed.
- Helps: the table-behind task (route back 0.5 m, turn 180°, forward 0.5 m; commit 7d0edfd). Smoke today: arms held by LeverCtl without
  this term, tilt < 0.5°, no U, at the ASSUMED swerve limits 0.25 m/s², 0.5 rad/s².
- Cost: small for (a) and (b). Needs `Segment.base_ref` (A2).
- Cannot: correct the base's own tracking error (wheel slip; drive droop about 18% in an in-place turn, `swerve.Limits.ki_wheel`).
  That error enters as a residual measured on the base card. The payload part uses the nominal load, so its error grows with the
  width of Θ_t times the acceleration (MODEL).

Stiff vs compliant hold during base motion is then a comparison of configurations, not a new tracker:
`flmpc_basic+base_ff` (stiff) vs `ct_pd:soft+base_ff` and `coop_imp+base_ff` (compliant), on the same worlds.

**Dropped: plain position servo (MuJoCo actuators).** Its gains are ASSUMED (`config.ASSUMED` kp 200/100, kv 20/10; the URDF has none). It
has no gravity or payload term, so its sag grows with the hidden load: the tracking error itself depends on θ. It would need its own card
per load and would mostly fail tilt and goal on the heavy box. It can still run once as a falsifier to show what the payload term buys.
**Not ported: full FL-MPC** (teleop and the Sept cup demo keep it; its rows would confound the checker, 11-07/11-09).

## A2. One tracker interface, and where it lives

Layering today (`stack/tests/depcheck.py`): `stack_pure` (types, tasks, goal, state, outcome, registry, checks) imports only itself;
`stack_engine` may not import flmpc, nero/contents, adapters; adapters and nero/contents must not reach flmpc.

- **Types (stack_pure, `stack/types.py`).** `Tracker` protocol: `reset(m, d, sides, load, base_ref)`, `set_segment(seg)`, `tau(d, t) -> {side:
  τ}`, `summary() -> TrackLog`. `TrackerCard` dataclass (A3), every number a `Bound`. The checker rows receive the card as an argument.
- **Registry (stack_pure).** New kind `TRACKERS` in `stack/registry.py`: `"flmpc_basic": "world_model.v2.trackers.flmpc_basic:FLMPCBasicTracker"`,
  others `None` until built. Strings resolve lazily, so no static import edge.
- **Implementations: new package `world_model/v2/trackers/`** (new depcheck bucket). It may import `flmpc`, the robot model (`nero/config`,
  `nero/adapters`) and mujoco. It may not import nero/contents, stack_engine, adapters or agentchord. Episodes (carry, base smoke, AgentChord
  port) get a tracker only through `registry.get("trackers", name)`. Depcheck rule to reword: "adapters and nero/contents must not reach
  flmpc" becomes "... reach flmpc only through the tracker registry" (memory note 11-09 says the same; user to confirm, C/Q6).
- **One split law.** The lever split (`contents/envelope.lever_split`) moves to a pure module (e.g. `stack/checks/split.py`), used by the tracker
  feedforward and by the checker torque row. One copy, so the checker models what the tracker commands. Test: equal to the old function.
- **Load model.** The feedforward needs a nominal load. Under 11-08 the LLM gives no numbers, so the nominal load = the centre of Θ_t
  (after a weigh) or of the prior box Θ_0 (before). It is a skill-supplied number; the card measures the error it causes (A3 "m_hat error").
- **Plan names a tracker per segment.** New fields `Segment.tracker: str | None` (None = the task default, `TaskDef.default_tracker`)
  and `Segment.base_ref` (None = base still). Recoveries name a tracker the same way (default: the tracker of the segment they leave).
  Proposer schema v2: optional `"tracker": "<id>"` on a step or a recovery item. An unknown id, or an id whose card does not cover the
  segment's conditions, is `plan_rejected[schema]` or a refusal `tracker_uncalibrated` (never extrapolated). A switch is allowed only at a
  segment boundary where the plan is at rest (ASSUMED, C).
- **What a tracker returns.** Per segment a `TrackLog` with the same keys as its card residuals (peaks of joint, TCP, object pose error,
  object acceleration excess, torque excess, internal wrench, settle time, saturation counts, step compute time). Plus per step the
  applied torque and the model torque, which the weigh reads. A runtime monitor row `tracker_out_of_card` (guarantee `runtime`) fires when a
  logged residual exceeds the card's max over n: that is evidence the card does not cover this run (counted; Rule 3 switches to the fallback).

## A3. The skill card each tracker needs for the checker

One card per configuration (tracker + gains + wrappers: payload ff, friction wrapper, coop reference layer, base_ff). Keep few
configurations. Stored as `nero/results/tracker_cards/<config>/CARD.json` with the commit, the planner and the scene.

| Residual | Unit | Feeds checker row | Tag |
|---|---|---|---|
| r1 joint tracking error, per joint, peak per segment | rad | 3.2 joint limits; 3.6 collisions (through FK) | MEASURED (T2 also MODEL) |
| r2 TCP pose error per claw | mm, deg | 3.6 clearance | MEASURED |
| r3 object pose error vs x_obj (position, tilt) | mm, deg | 3.7 tilt (row MISSING today), goal tolerance | MEASURED |
| r4 object acceleration excess over the plan (linear, angular) | m/s², rad/s² | 3.5 open content (spill, lid, mug slide), 3.1 | MEASURED |
| r5 torque excess τ_applied − τ_model(plan, θ) per joint | Nm | 3.1 torque (would replace the ASSUMED 0.9 margin, C/Q4) | MEASURED |
| r6 internal wrench under the weld | N, Nm | 3.1 split (checker A2 `H_I`) | MEASURED |
| r7 settle time to 5 mm/s; placement error at landing | s, mm | 3.9 landing S, 3.10 R (checker A4, A5) | MEASURED |
| r8 saturation steps (torque box; acceleration box for T1) | count | flags a sample outside the linear regime | MEASURED |
| r9 compute time per control step | ms | real-time check (5 ms step) | MEASURED |

**Conditions (the grid each card is measured on).** Task and object (box, pot, tray on the base route; coffee tray once it runs in the NERO
sim). Load: the corners of the prior box Θ_0 plus random interior worlds. Commanded peak TCP acceleration: 3 levels up to A_NOM 0.5 m/s².
Base: linear acceleration 0 / 0.125 / 0.25 m/s², yaw acceleration 0 / 0.25 / 0.5 rad/s² (tops = the ASSUMED swerve limits). Nominal-load
error: the true load anywhere in Θ_0 while the feedforward uses the box centre (worst case before a weigh).

**How measured.** Same worlds and seeds for every tracker (bounds rule 8). Worlds drawn from Θ_0, never from a narrowed set (rule 4).
References from the closed_chain planner. n = 20 per condition (ASSUMED, C). Report p50 / p95 / max over n for each residual (rule 6:
"max over n = 20", never "≤"). Noise model, stated on the card: state from the sim (no encoder noise, no delay, `delay_steps` 0); joint
torque read ±`probe.TAU_NOISE` 0.05 Nm uniform (ASSUMED, existing); friction = MJCF frictionloss 0.3 Nm (ASSUMED, existing); weld =
`config.WELD_STIFF`; physics substeps per `config.substeps_default()`. What it leaves out: hardware friction, torque-read bias, latency,
slosh (the sim has none).

**How the checker uses a card.** A row that adds a MEASURED residual to a MODEL row becomes `sampled` (n stated), not `proof`. Today's
tilt row is already sampled. A card applies only inside its measured conditions. Between grid levels the checker takes the max over the
enclosing levels; this assumes the residual is monotone in load and acceleration between levels (ASSUMED, C). For T2 the card also holds
the MODEL bound from the gains; both are reported side by side.

**Probe recalibration per tracker.** The weigh reads load from applied minus model joint torque, so EPS_LO, EPS_FREE and DL_STEP_MAX
(0.9332 kg, n = 349 rungs) depend on the tracker. Today's CALIB (`nero/results/probe_calib_poslift_2026-11-09/CALIB.json`) was measured
under `LeverCtl` (FL-MPC with its rows). Proposal: the weigh owns one fixed tracker, `flmpc_basic` + lever ff + friction wrapper, chosen
by the stack, not the LLM (ASSUMED, C). Recalibrate once under it (`probe.py calibrate`, n = 20 per task, same seeds), add a `tracker` field
to CALIB.json, and assert it at load the way `method` is asserted today. Report the LeverCtl and flmpc_basic numbers side by side and mark
the old one superseded (rule 5). Any other tracker that would run the weigh needs its own CALIB first.

## A4. Build and measure order (what blocks what)

1. Interface: `Tracker`, `TrackerCard`, `TrackLog` in `stack/types.py`; registry kind `TRACKERS`; depcheck bucket `trackers` + rule rewording;
   split law moved to a pure module (equality test). Blocks everything below. Needs the user's answer on Q6.
2. `trackers/flmpc_basic.py`: wraps `flmpc/basic.py` with the lever ff and friction wrapper. Gate: on the golden fixture
   (`stack/tests/golden/`) compare with today's LeverCtl runs; report where torques differ (old rows active) rather than expect equality.
3. Measurement harness `trackers/measure.py` (conditions grid → CARD.json). Blocks every card.
4. Card for `flmpc_basic` on box and pot (base still). Weigh recalibration under it. Blocks any probe or admission result under the new tracker.
5. Checker reads cards (rows 3.1, 3.2, 3.6, 3.7, H_I). Blocks: checker steps C5/C6. Needs Q4, Q5.
6. `Segment.base_ref` + `base_ff`; base card (swerve tracking error, slip) + `flmpc_basic+base_ff` card on the tray route.
   Blocks: the base-and-arm planner (it writes `base_ref`).
7. `ct_pd` (stiff, soft) and its card. Independent of 6 except for its base_ff card.
8. `coop_imp` ported as a tracker; card incl. r6. No weigh under it unless calibrated.
9. Proposer: `tracker` field in schema v2 + tracker cards in the prompt; mock fixtures (unknown id rejected; uncovered condition refused).
   Needs probe-selection step 2 (schema v2) landed.
10. Same-world comparison on the table-behind task: stiff vs compliant configurations.

## B. What could help the proposer make better proposals (ranked)

The proposer writes phenomena, probes, steps, recoveries, fallback, and (with A2) the tracker per segment. Ranked by value for cost.

1. **Skill cards in the prompt, tracker cards included.** Gives: what each skill does, its preconditions, its limits, and how each tracker
   behaves (e.g. "compliant: object error up to X mm at base accel Y"). These are robot facts, allowed by U10. Cost: low (catalog step 1).
   Risk: the LLM copies card numbers into step args; schema v2 already rejects numbers outside args. Never put a contents range on a card.
2. **Structured checker feedback on refusal (Rule 2 T2 re-ask).** Gives: which row refused, its margin and unit, the binding segment, joint
   and Θ_t vertex, the guarantee type, and which of the LLM's choices touch it (segment, tracker, route, recovery). Cost: low to medium
   (`propose.py:117-127` exists with no caller). Risk: the vertex is a point of the measured set, not truth, so no leak; never include the
   executing world. The LLM may "tune by feedback": cap re-asks (count is a result).
3. **A feasibility query tool for proof rows only.** Gives: "would this step pass reach / joint / static torque / support?" before the final
   answer. Cost: medium (tool loop with Codex, cache keyed on the query sequence for offline replay). Risk: the checker becomes a search
   oracle, so plan quality comes from the checker; must not run sampled rows, because resubmitting until a sampled row passes weakens
   the sampled guarantee (multiple testing). Sampled rows run once, on the final plan, with fresh seeds. Cap queries per plan.
4. **Base-route and stance options.** Gives: a short list of precomputed base routes and stances with reach and clearance facts (e.g. "turn in
   place at the pick pose hits the counter"), so the LLM picks a route by name instead of writing base numbers. Cost: medium (needs the base
   planner). Risk: low; options limit creativity (an LLM may propose a new route as "NEW", logged).
5. **Renders from the robot's own cameras** (ZED torso mount, wrist RealSense; both ASSUMED mounts). Gives geometry the JSON lacks (lid,
   handles, clutter, the second table). Cost: low to medium (GPU, `MUJOCO_GL=egl`). Risk: leaking hidden contents, e.g. mug fill seen from
   above; render with the contents hidden or from views the real sensors have (U5: no vision into the mugs).
6. **Worked examples (admitted and refused plans) from other tasks.** Gives format and the kind of recovery that passes. Cost: low. Risk: copying;
   leaking test-world facts if taken from the same world set; use other tasks or the prior box only.
7. **The current set Θ_t and the readings** (already in the re-ask). Gives what was measured. Risk: the LLM turns it into numeric beliefs; 11-08
   allows reading numbers, not writing contents numbers.
8. **Run history on other worlds** (which phenomena caused failures). Gives priors on what matters. Risk: train/test leak; keep a split.

## C. New ASSUMED values and open questions

New ASSUMED values this design introduces (raise before any claim rests on them):
- AS1 conditions grid: TCP accel 3 levels up to 0.5 m/s²; base linear 0 / 0.125 / 0.25 m/s²; yaw 0 / 0.25 / 0.5 rad/s².
- AS2 n = 20 worlds per condition for a card.
- AS3 residual monotone in load and acceleration between grid levels (lets the checker use a grid).
- AS4 tracker switches only at segment boundaries where the plan is at rest.
- AS5 `ct_pd` gains: `stiff` ω_n 40 rad/s, `soft` ω_n 10 rad/s, ζ = 1 (per joint, on the linearized double integrator).
- AS6 the weigh runs under one fixed tracker config (`flmpc_basic` + lever ff + friction wrapper).
- AS7 feedforward nominal load = centre of Θ_t (or Θ_0 before a weigh).
- AS8 `base_ff` uses the planned base acceleration, not a measured one.
- Reused, not new: a_box 800 rad/s² (raised by the basic-FL-MPC agent), TAU_NOISE 0.05 Nm, frictionloss 0.3, WELD_STIFF, swerve Limits,
  coop_imp gains, squeeze 0 N.

Open questions for the user:
- Q1 Which trackers after `flmpc_basic`: `ct_pd`, `coop_imp`, `base_ff` (all, or a subset)?
- Q2 Does the LLM pick the tracker per segment, or does the stack pick by rule and the LLM only sees the cards? Default when omitted?
- Q3 Does the weigh own one fixed tracker (one calibration), or recalibrate for each tracker the LLM may pick?
- Q4 Replace the ASSUMED carry margin U_ADMIT 0.9 by the measured torque excess r5 per tracker (ties to the 11-09 open item)?
- Q5 Rows that use measured residuals become `sampled`, not `proof`. Accept?
- Q6 Depcheck: new bucket `trackers` that may import flmpc; adapters and nero/contents reach it only through the registry. Accept?
- Q7 Are the friction wrapper and the 11-02 coop reference layer part of a tracker's configuration (own card each), or dropped?
- Q8 Keep FL-MPC's asymmetric arm weights (left 3000 / right 600) in `flmpc_basic`, or make them equal?

Uncertainties: the internal wrench under the weld may dominate the torque excess for the stiff trackers (unknown until r6 is measured);
TOPP-RA's qdd jumps may inflate T2 errors at knots; the swerve base card does not exist yet, so the base_ff residuals cannot be split from
base tracking error until it does.
