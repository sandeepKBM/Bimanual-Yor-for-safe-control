# Task-agnostic checker: design (2026-11-05)

> Status note (2026-11-17): **partly built.** The design below is not rewritten; lines that the code now contradicts carry
> "[11-17: ...]". Built / wired / tested: `docs/status/current_state_2026-11-17.md`. Decisions D1, D2, D4, D9 were ruled 11-16
> (`docs/status/decisions.md`). Map of the build steps (§7) to code, checked 11-17:
>
> | Step | State | Code |
> |---|---|---|
> | C0 types | built | `stack/theta.py` (Interval, Cell, ThetaSet, ReadingSpec, RecoveryProgram, Admission) |
> | C4 reach, closed chain, joint q / qd | built (planned-path rows) | `nero/contents/probe_select.planned_path_rows`, `stack/checks/motion.py` |
> | C5 exact-ID torque row | code and gate exist; wiring into the run path not checked | `adapters/exact_torque.py`, `stack/tests/test_exact_torque_gate.py` |
> | C6 support, open content, tilt, collision | built; tilt = MEASURED per plan; collision = swept bound (`SWEPT_MAX_DEPTH` 4); the load is modelled untilted (tilt-aware row parked) | `stack/checks/{support,open_content,tilt}.py`, `adapters/collision_rows.py`, `nero/contents/tilt_measure.py` |
> | C7 `admit` with fixed recoveries and programs (Rule 1) | built | `stack/checks/rule1.py`, `adapters/rule1_contents.py` |
> | C8 run-time cover table + fallback (Rule 3) | built; wide table | `stack/checks/cover.py`, `stack/checks/rule1.py:138-144`, `stack/executor.py` |
> | C9 R_K and the Rule 4 loop with a mock proposer | Rule 4 built (offline mock at run time); R_K depth 1 only (K, NODE_CAP recorded, unused) | `stack/checks/rule4.py`, `adapters/rule4_contents.py`, `nero/contents/runtime_rules.py` |
> | base-stop rows (not in the 11-05 design) | built | `stack/checks/base_stop.py` |
> | C1-C3, C10, gates G1-G4 | not checked for this note | — |
>
> Status (2026-11-05, superseded by the note above): **design only, nothing is built.** Statements about existing code carry file:line (worktree `contents-armside2`, paths under
`world_model/v2/` unless given in full). Everything else is a proposal. Simulation only. Problem statement: `docs/paper/claims_2026-11-02.md`
§1 (rows 11-04, 11-05) and §3 "Problem". The checker is built after planner step 2 (closed-chain path + TOPP-RA,
`docs/status/2026-11-03_handoff_notes.md:65-67`).

## 0. Symbols

- θ: world parameters of a universe (contents mass, COM, frictions, ...). θ*: the true value.
- Θ_0: prior box (`Universe.theta_box`, `stack/types.py:37-43`). B: the VLM's belief range. Θ_t: the set the probes show true at time t
  (ASSUMED, user 11-05: θ* ∈ Θ_t). F: the set the fallback is verified on (decision D1).
- π = (m, m·c) ∈ R⁴: contents mass and first moment (`nero/contents/objects.py:6-8`). π10 = (m, m·c, I_o): π plus the 6 second moments.
- u: utilisation = max(|τ_j| / τ_lim,j, arm vertical load / F_cap) (`nero/contents/envelope.py:17-29`).
- U: unsafe set. S: rest set (holding still is safe for every θ in the set). R: task-ready set; R_K: its depth-K version
  (`docs/paper/proposal_2026-10-27.md:98-112`).
- Recovery b is **viable for set Θ from state x**: every row of b passes for every θ ∈ Θ starting at x, and b ends in S ∩ R.

## 1. Scope

### 1.1 What the checker decides

Inputs: a `TaskPlan` (task steps, recoveries, fallback) already turned into motions by the planner; a `ThetaSet`; robot, scene and object
models; optionally a `Universe` to draw sampled worlds. Output: a `Verdict` with one `RowResult` per row, and for an admitted step an
`Admission` (chosen recovery + a runtime cover table). The checker never runs a probe, never chooses a plan, and never acts at runtime
beyond set-inclusion lookups.

| Call | Decides | Rule |
|---|---|---|
| `check_motion(plan, Θ, models)` | every row of one motion for every θ ∈ Θ | building block |
| `static_check(state, Θ, models)` | rest-pose torque/payload + support on Θ, no motion | §3 "static corner check" (before the probe on B, after it on Θ_t) |
| `admit(step, Θ_t, recoveries, models)` | ∃ one b viable for all of Θ_t from every branch point of the step | Rule 1 |
| `split_by_B(Θ_t, B)` | Θ_t ⊆ B, else the part D = Θ_t \ B | Rule 2 |
| `cover(Θ_new, admission)` | runtime: continue / switch to another precomputed b / fallback | Rule 3 |
| `check_target(x_goal, transfer, Θ_t)` | transfer admitted, a recovery viable at x_goal, x_goal ∈ R_K | Rule 4 (called by the orchestrator loop) |

### 1.2 What it replaces or wraps

| Existing | Today | In the checker |
|---|---|---|
| `envelope.Envelope` (`envelope.py:162-352`) | LP / lever / share utilisation at waypoints, exact corner check, level object, translation only, quasi-static | backend of the torque row in **legacy mode**; generalised in §3.1 |
| `carry_cert.static_cell` (`carry_cert.py:103-151`) | contents static rows per hard-coded cell (tray fill, box mass) at `carry.PLANS` path samples | becomes a call of `check_motion` on a cell; equivalence gate G1 |
| `carry_cert.support_row` (`carry_cert.py:80-100`) | COM over destination rectangle, geom/plan specific | support row (§3.4) with scene data |
| `static_cert.static_rows` (`static_cert.py:93-134`) | **pan/pot only**: torque (two_share split), support at rest on the stove, lid slide | same rows for the pot universe; gate G3 |
| `pan/cert/admit.certify` (`admit.py:77-126`) | sampled ∃b∀θ: first family member passing all worlds + Clopper–Pearson | dynamic sampled row (§3.11) |
| `carry.CarryEpisode._check` / `in_R` (`carry.py:489-567`) | U and R predicates, geom names `counter`/`shelf` (`carry.py:557`) | U list and R0 as data (§3.8-3.10) |
| `path_reach.py` | records IK miss and joint margin; no caller in the repo; unreachable is not in U (`carry.py:351-356`) | reach row that **refuses** (§3.3) |
| `rules.py` | runtime FL-MPC rows (table, q, qd) | **superseded 11-09: `rules.py` deleted (c8cf380); the tracker skill is basic FL-MPC, the checker alone owns safety** |
| `house/tamp/spec_cert.py` + `tray_rep/sets/ir.py` | spec-driven rows (Inside, Upright, Clear, Support, STL), continuous-time sweep term | language for geometric rows (§3.6) |

### 1.3 Code facts that differ from the task brief

1. `static_cert.py` is wired to the **pan Episode** (segments pinch/push/clamp, stove, lid; `static_cert.py:93-129`). The contents carry
   static rows live in `carry_cert.static_cell` (`carry_cert.py:103-151`), not in static_cert.
2. The contents torque row checks the **lever** split (`carry_cert.py:117,126,134`); `static_cert` checks the **share** split
   (`static_cert.py:109`). The current controller work `ctl="coopimp_safe"` commands the minimum-norm split W⁺h_o
   (`nero/contents/coop_imp.py:13-17`), which neither row checks. `CarryEpisode` still defaults to `ctl="lever"` (`carry.py:180`).
3. Contents dynamic rows branch every 0.25 s (`stack/executor.py:57`) with one stored backup, `family=(0.0,)` (`carry_cert.py:241`). The
   0.2 s spacing and "longest backup" choice are the pan path (`admit.py:17`, `pan/cert/backups.py:24`).
4. Object–world contact is **not in U** for the carry: `_check` skips any contact without an arm geom (`carry.py:532-542`), and the welded
   object is in no FL-MPC table row (`rules.py:21`; superseded 11-09: `rules.py` deleted, c8cf380). A box hitting the shelf edge counts only if it tilts > 10° or drops. [11-17: object-world contact is in U (D9, USER 11-16; `nero/contents/carry.py:140`), clearance 5 mm; the landing support is exempt at release (USER 11-17).]
5. `capture_nyu.json` field `o0` is the old pot origin (y = −0.30) while its TCP midpoint is at y = −0.22 (`scenes.py:38,48`,
   SCENE_DY). `carry_cert` builds o_ref from the TCPs (`carry_cert.py:111-112`), so its rows are not affected; a new adapter must not read `o0`.
6. Two torque functions: `config.tau_limits()` defaults to 100 Nm (`config.py:72-77`), `config.tau_limit()` to 16 Nm (`config.py:86-92`).
   The checker must take the limit explicitly (AGENTS.md: the arm torque limit is OPEN). [11-17: ruled 11-05: `config.tau_limit()` = 100 Nm
   (MJCF forcerange); 16 Nm only under `BIMANUAL_TAU_MIT16=1`.]
7. `types.Checker.check` takes a `sim` (`stack/types.py:152-156`); model rows need none. `registry.CHECKERS` is empty (`stack/registry.py:25`).
   [11-17: `registry.CHECKERS` now lists the unified rows (`stack/registry.py:32`).]
8. `config.py` has **no joint acceleration limit** (only position `config.py:25-26` and speed `config.py:42`). [11-17: still true for
   `config.py`; the planner has QDD_MAX 6 rad/s² (ASSUMED, USER 11-17, `stack/planners/closed_chain.py:68`) in TOPP-RA.]

## 2. Interfaces

All types are additive to `stack/types.py`; `Bound`, `ThetaBox`, `Segment`, `BimanualPlan`, `Verdict` stay.

```python
# ---- belief sets -------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Cell:                                   # one axis-aligned box in the universe's θ coordinates
    bounds: dict[str, Bound]                  # field -> [lo, hi] + tag (MEASURED/MODEL/ASSUMED) + src
    source: Literal["prior", "vlm_B", "probe", "vlm_outside_B"]
    provenance: dict = field(default_factory=dict)   # probe id + reading + noise bound; VLM model + reasoning level

@dataclass(frozen=True)
class ThetaSet:                               # a finite union of cells (exact for boxes; probes report intervals)
    cells: tuple[Cell, ...]
    def contains(self, other: "ThetaSet") -> bool: ...     # exact: other.minus(self) is empty
    def minus(self, other: "ThetaSet") -> "ThetaSet": ...   # box difference -> union of boxes
    def intersect(self, other) -> "ThetaSet": ...
    def corners(self) -> np.ndarray: ...                     # all cell vertices (for MONOTONE / AFFINE rows)

# ---- models (data, built once per universe; no geom names inside the rows) ------------------------------------------
@dataclass(frozen=True)
class RobotModel:
    mj: Any                                   # compiled MJCF (arms + NYU grippers), gravcomp off for dynamics
    sides: tuple = ("left", "right")
    q_lim: Bound_per_joint                    # config.URDF_LIMITS (MODEL), margin config/ carry.PATH_MARGIN_MIN (ASSUMED)
    qd_lim: Bound_per_joint                   # config.MANUAL joint_speed_deg_s (MEASURED-spec)
    qdd_lim: Bound_per_joint | None           # None today (config.py has none): decision D6
    tau_lim: np.ndarray; tau_src: str         # passed explicitly ([11-17: 100 Nm, USER 11-05])
    payload_N: float                          # (3 kg - gripper mass) g (envelope.py:20)
    collision: Any                            # convex collision geoms (clearance.py) + FL-MPC proxy spheres (proxy.py)

@dataclass(frozen=True)
class TrackerModel:                           # what the executing controller commands and how well it tracks
    name: str                                 # "lever" | "share" | "coopimp_safe" | ...
    split: Callable[[np.ndarray, np.ndarray], np.ndarray]   # (W grasp matrix, h_o object wrench) -> per-arm wrenches; linear in h_o
    internal_bound: Bound                     # |internal wrench| under the weld (A2)
    residual: dict[str, Bound]                # joint error, TCP error, object tilt; MEASURED per tracker, else ASSUMED (A3)

@dataclass(frozen=True)
class Support:  name: str; top_z: float; polygon_xy: np.ndarray; mu: Bound   # MODEL from the MJCF box geoms
@dataclass(frozen=True)
class SceneModel:
    supports: tuple[Support, ...]
    obstacles: tuple["ir.Entity", ...]        # tray_rep/sets/ir.py entities (shape + pose), world frame

class OpenContentRow(Protocol):               # pluggable, per object
    name: str
    dependence: Literal["monotone", "affine_pi", "blackbox"]
    def margin(self, f_obj: np.ndarray, R: np.ndarray, cell: Cell) -> tuple[float, dict]: ...  # f = a - g (specific force)

@dataclass(frozen=True)
class ObjectModel:
    name: str
    empty_mass: float; empty_com: np.ndarray; empty_inertia: np.ndarray  # MODEL (MJCF)
    def pi_vertices(self, cell: Cell) -> np.ndarray: ...     # (n, 4) [m, m cx, m cy, m cz]; objects.Contents.pi_vertices
    def pi10_box(self, cell: Cell) -> np.ndarray: ...        # outer box of second moments (§3.1)
    footprint_xy: np.ndarray                                  # contact footprint, object frame
    grasp: dict[str, np.ndarray]                              # object -> TCP transform per side (weld, ASSUMED)
    open_rows: tuple[OpenContentRow, ...]
    unsafe: dict[str, Bound]                                  # U thresholds (tilt, cup tilt, spill ratio, drop)

# ---- plans ------------------------------------------------------------------------------------------------------------
@dataclass
class MotionSpec:                             # what the checker needs from the planner, per Segment (planner step 2 output)
    knots: np.ndarray                         # time grid (TOPP-RA grid)
    q: dict[str, PPoly]                       # per side, piecewise polynomial in t: exact q, qd, qdd and per-piece extrema
    x_obj: ObjPath | None                     # object pose, twist, acceleration at t (None when not carrying)
    tracker: str                              # TrackerModel name
    ik_residual: dict[str, np.ndarray]        # FK-validated pose error per knot (SRS IK, kin/srs_ik.py)
    branch_dt: float = 0.25                   # where a recovery may start (design; stack/executor.py:57)

@dataclass
class FixedRecovery:
    plan: BimanualPlan                        # motion from a branch state to a landing; segments carry MotionSpec
    claimed_cover: ThetaSet                   # the part of the range it was written for (VLM claim; the checker verifies)
    landing: str                              # name of the S region (support name, or "hold")

@dataclass
class ReadingSpec:
    producer: str                             # stack/producers name; Source must not be TRUTH (WorldState.assert_honest)
    statistic: str                            # e.g. "vertical load from joint torques" (probe.py:7-10)
    model: Callable[[np.ndarray], Interval]   # y as a function of π (affine where possible)
    noise: Bound                              # MEASURED or ASSUMED (probe.TAU_NOISE 0.05 Nm is ASSUMED, probe.py:57)

@dataclass
class RecoveryProgram:                        # parameters set from readings during execution, never from θ*
    prefix: BimanualPlan                      # runs before the reading; must pass on the whole set
    reading: ReadingSpec
    bins: list[tuple[Interval, BimanualPlan]] # piecewise-constant selection: reading interval -> plan

Recovery = FixedRecovery | RecoveryProgram

@dataclass
class Step:
    call: SkillCall; plan: BimanualPlan; recoveries: list[Recovery]
@dataclass
class TaskPlan:
    steps: list[Step]
    fallback: Recovery
    fallback_set: ThetaSet                    # F (decision D1)
    proposer: dict                            # model, reasoning level, caveats (AGENTS.md bounds rule 9)

# ---- verdicts ---------------------------------------------------------------------------------------------------------
@dataclass
class RowResult:
    row: str; ok: bool; guarantee: Guarantee  # "proof" | "sampled" | "refuted"
    margin: float                             # > 0 inside
    worst: dict                               # θ or π vertex, time t, side, joint
    counterexample: dict | None               # refuted: vertex (model) or world id + seed + t + why (sim)
    n: int | None = None; eps_upper: float | None = None   # sampled rows: n worlds, Clopper-Pearson bound
    assumptions: list[Bound] = field(default_factory=list)
@dataclass
class CheckVerdict(Verdict):                  # Verdict.guarantee = weakest row: refuted < sampled < proof
    rows: list[RowResult] = field(default_factory=list)
    theta: ThetaSet | None = None
@dataclass
class Admission:
    step: int; recovery: Recovery; theta_cert: ThetaSet
    table: dict[float, list[tuple[Recovery, ThetaSet]]]   # branch time -> viable recoveries and their verified sets
    fallback_ok_on: ThetaSet                  # where the fallback passed (⊆ F)
    verdicts: list[CheckVerdict]

class Checker(Protocol):                      # replaces stack/types.py:152-156 (no sim argument for model rows)
    def check_motion(self, plan: BimanualPlan, theta: ThetaSet, models: "Models",
                     sampler: Universe | None = None) -> CheckVerdict: ...
```

**What the checker needs from the planner (step 2).** (1) per segment and side a piecewise-polynomial q(t) (TOPP-RA output) so q, qd, qdd
and their per-piece extrema are exact; (2) the object path x_obj(t) with twist and acceleration; (3) the grasp transforms; (4) FK-validated
IK residuals per knot; (5) the tracker name; (6) recovery motions in the same format, planned from each branch state (or one plan per branch
point); (7) for R_K, a call `plan_to(subgoal, from_state, depth=K, node_cap)` that returns a plan or "not found".

## 3. Row catalogue

Each row is `row(plan, Θ, models) -> RowResult`. Each declares how it depends on θ; the declaration fixes the best guarantee:
**affine in π** (or separately convex in π and in the motion scale) → the max over the set is at vertices → `proof` under MODEL;
**monotone** in one field → endpoints → `proof` under MODEL; **none** → `proof` under MODEL; **black box** → `sampled`.

| # | Row | θ-dependence | Best guarantee | Stays sampled |
|---|---|---|---|---|
| 3.1 | torque / payload | affine in π (π10 with rotation) | proof under MODEL at knots | between knots (D7), tracking feedback torque, friction |
| 3.2 | joint q / qd / (qdd) | none (+ tracking residual) | proof for the reference, also between knots | tracking residual |
| 3.3 | reach + closed chain | none | proof under MODEL (FK) | — |
| 3.4 | support at rest / destination | affine in π | proof under MODEL | placement residual, settling |
| 3.5 | open content (spill, lid, cup slide/tip) | monotone | proof under MODEL, quasi-static | slosh, impacts |
| 3.6 | collisions | none (+ residual) | proof for the reference with a sweep margin | tracking residual |
| 3.7 | object tilt | black box today | sampled (MEASURED residual) | all of it |
| 3.8 | unsafe set U | list of the rows above + monitors | as the rows | sim monitors |
| 3.9 | landing set S | support + static torque | proof (static part) | coming to rest |
| 3.10 | task-ready R / R_K | via planner | constructive witness | landing-state spread |
| 3.11 | dynamic trials | black box | sampled, n + ε | weld compliance, stiction, contact chaos |

### 3.1 Torque and payload

Per arm s at plan time t, with the plan's (q, qd, qdd):
`τ_s(t, θ) = ID_s(q, qd, qdd) + J_s(q)ᵀ h_s(t, θ)`, ID = MJCF inverse dynamics of the arm alone (MODEL).
Object wrench (Newton–Euler about o): `h_o(t, θ) = Y(R, ω, α, a) · π10 + h_empty(t)`, linear in π10 for a fixed motion.
Per-arm wrenches: `h_s = split_s(W, h_o) + internal`, split from the `TrackerModel` (lever `envelope.py:143-155`, share
`envelope.py:216-227`, coopimp W⁺ `coop_imp.py:13-17`), |internal| ≤ H_I (A2). For a split linear in h_o, τ is affine in π10, so
max |τ_j| over a polytope is at its vertices (exact). The best-split LP (`envelope.py:239-281`) stays as information only ("capability").

What changes from `envelope.py`:
- **Quasi-static dropped.** Today `G + M J⁺[a; 0]`, Coriolis dropped, angular acceleration 0 (`envelope.py:5-15,199`). New: exact ID from
  the plan's (q, qd, qdd). Legacy mode keeps the old formula for gate G1.
- **Rotation.** Today R0 is fixed (`envelope.py:177-184`; carry passes `np.eye(3)`, `carry_cert.py:113`). With R(t): gravity and moment arms
  rotate, still linear in π for fixed R. Angular terms `I_o α + ω × I_o ω` need the second moments, which π does not hold. The contents
  set gives no vertex set in π10; use an **outer box**: for contents inside polytope P, each entry of m·E[c cᵀ] lies in
  [m·min_P c_i c_j, m·max_P c_i c_j] (MODEL, conservative). The vertex max over this box is a proof on a superset. Contents that slide
  inside P while the object tilts stay covered quasi-statically (the row is ∀c ∈ P); impacts are not (sampled).
- **Motion scale.** Along a fixed path, TOPP-RA's torque constraint is affine in (s̈, ṡ²). Torque is affine in π for fixed (s̈, ṡ²) and
  affine in (s̈, ṡ²) for fixed π: separately convex, so the vertex check over (π vertices × scale endpoints) is exact. This is what makes
  a "slow down by a reading" recovery program checkable (§4.3). Today's `a_max` bisection relies on the same convexity (`envelope.py:11-13`).
- **Admission margin.** Admit only at u ≤ U_ADMIT = 0.9 (ASSUMED, `envelope.py:53`) for the feedback torque above the model. [11-17:
  `U_ADMIT = 1.0` (USER 11-09: the rated limits, no margin; `envelope.py:54`).]
- **Between knots:** D7.

### 3.2 Joint position, speed, acceleration

q within URDF_LIMITS (`config.py:25-26`) minus PATH_MARGIN_MIN 0.05 rad (ASSUMED, `carry.py:73`); |qd| ≤ datasheet speed (`config.py:42`).
On a piecewise polynomial the max of |q|, |qd| over each piece is exact (roots of the derivative), so this is a proof for the reference
including between knots. Tracking error adds the residual e_q (A3). **No joint acceleration limit exists** in `config.py`; qdd enters only
through the torque row unless D6 adds one.

### 3.3 Reach and closed chain (must be able to refuse)

At each knot: FK(q_s(t)) = x_obj(t)·G_s within PATH_IK_TOL (1e-4, ASSUMED, `carry.py:72`), and both arms agree on the object pose:
`‖FK_L G_L⁻¹ − FK_R G_R⁻¹‖ ≤ ε_cc` (A1). Under the weld a mismatch is an internal force (39.6 N on the scripted box path,
`2026-11-03_handoff_notes.md:25-26`). Between knots: sample densely and add a Lipschitz term from max |qd| (MODEL). A fail **refuses**
(today it is only recorded). SRS `psi_intervals` (`stack/kin/srs_ik.py:238`) can report the feasible arm-angle set as a margin.

### 3.4 Support at rest and at the destination

Generalises `carry_cert.support_row` (`carry_cert.py:80-100`) and `static_cert` support_rest (`static_cert.py:117-124`): for every π vertex,
signed distance of the total COM (xy) to the contact polygon = object footprint ∩ support polygon (both data), shrunk by the placement
residual δ_place (A4). Linear in π → proof under MODEL. Level supports only. Static tilt at rest = 0 follows when it passes (flat support,
rigid object, `static_cert.py:10-12`).

### 3.5 Open-content rows (pluggable)

Input: specific force f = a − g in the object frame and R(t). Each row is monotone in one θ field, so its worst case is an endpoint.
- cup spill: tan φ ≤ 2(H − h)/D, φ = angle of f in the mug frame, h = fullest fill of the cell (`objects.py:52-64`, `carry_cert.py:56-58`).
  Today the row compares the planned peak acceleration with a_max for a level object (`carry_cert.py:140-142`); with R(t) and the tilt
  residual the angle adds.
- lid slide: |f_xy|/f_z ≤ μ_lid,min (`static_cert.py:125-129`).
- cup slide / tip on the tray (new rows, sampled today as `cup_off_tray` / `cup_tipped`, `carry.py:519-531`): |f_xy|/f_z ≤ μ_cup (existing
  ASSUMED value in `scenes.TrayWorld`) and ≤ D/(2 z_c) (MODEL).
Guarantee: proof under MODEL, quasi-static. Slosh and impacts: sampled.

### 3.6 Collisions

Pairs: arm–arm, arm–world, **object–world**, arm–object (non-welded parts). At knots: convex-geom distances (`nero/clearance.py:1-17`);
between knots add the sweep term (h/2)·v_max of the farthest point, as `spec_cert.py` does for the base (`house/tamp/spec_cert.py:14-20`).
Clearance ≥ CLEARANCE_MIN 5 mm (ASSUMED, `clearance.py:14`) + TCP residual (A3). Geometric rows can be written in the IR
(`Clear`, `Support`, `Upright` with a time axis, `tray_rep/sets/ir.py:1-30`), so a VLM-written constraint uses the same evaluator.

### 3.7 Object tilt

No model row exists (claims §3, claim 1 row). Bound = planned tilt of R(t) + residual r_tilt(cell, acceleration) MEASURED from sim (claims
"Next": corner max × measured interior margin). Guarantee: sampled. U threshold 10° (ASSUMED, `carry.py:75`). [11-17: built as a per-plan
MEASURED row (`nero/contents/tilt_measure.measure_plan`, n = 20 ASSUMED, USER 11-17); numbers predate vref. The collision rows still
model the load untilted; the tilt-aware collision row is parked.]

### 3.8 Unsafe set U

One list of named predicates with thresholds, supplied by the object model and universe (today hard-coded in `carry._check`,
`carry.py:489-546`: tau_sat, obj_tilt, obj_drop, cup_tipped, spill, cup_off_tray, collision). Two evaluators read the same list: the model
rows above (before execution) and the sim monitors (sampled trials). Same names, so a sampled counterexample points at a row.
Object–world contact is added as a predicate (decision D9; today it is missing, §1.3 item 4).

### 3.9 Landing set S

A landing is in S when: the support row passes for every vertex (object set down), or the static torque row passes at the hold pose for
every vertex (object held still, proposal `:98-99`); plus "comes to rest" (speed ≤ R_SPEED 5 mm/s, `carry.py:76`), which only trials show.

### 3.10 Task-ready R

Three levels (decision D2):
- **R0** = today's `in_R`: still, level ≤ 3°, on counter/shelf, cups ≤ 5°, no U (`carry.py:548-567`). Says nothing about progress.
- **R1** = R0 + the next skill's `pre()` holds and its first motion passes rows 3.2-3.3 (as pan R_V2 adds a reach test,
  `pan/segments.py:961-983`).
- **R_K** = R1 + the planner finds, within depth K and a node cap, a sequence to the next subgoal whose every step is admitted under Θ
  (proposal `:108-112`). [11-17: R = R_K ruled 11-16 (D2); built at depth 1 only: one recovery program, then in R or the continuation;
  K = 3 and NODE_CAP = 50 are recorded, not used (`stack/checks/rule1.py:26-27`). A hold is not a safe landing (USER 11-17).] A returned admitted sequence is a constructive witness. "Not found" means not shown, not refuted.
The landing depends on θ: use the nominal landing plus a landing residual box (A5), corners sampled. By Lemma 1 (claims §3) a witness for Θ
stays valid for any sound narrowing Θ' ⊆ Θ, so witnesses are cached by (landing region, Θ). R_K runs before motion only.

### 3.11 Dynamic rows (sampled)

Worlds drawn from the checked set (cell or Θ_t; D11) through `Universe.worlds` + the executor; branch every `branch_dt`; from every branch
run the recovery; pass = no U and end in S ∩ R. Count and bound as `admit.certify` (`admit.py:77-126`, Clopper–Pearson). Covers what no model
row covers: weld compliance, joint friction / stiction, contact chaos, slosh, settling. A world refused or failed earlier is not a sample of
a later segment (`admit.py:103-109`).

**Composition.** A verdict is `proof` only if every row is proof; else `sampled` (n, ε); `refuted` if any vertex or world fails.
Refusals from proof rows are sound (the vertex is a counterexample under MODEL); admissions with any sampled row are sampled.

## 4. Rules 1-4 as checker calls

### 4.1 Before the probe

`static_check(rest state, B)` and the same on Θ_t after the probe (claims §3 "Probes"): rest-pose torque/payload and support, no motion.
It also tells the probe which verdicts are already the same for the whole set (`probe.py:23-24`: no probe needed then).

### 4.2 Rule 2: Θ_t ⊆ B?

`D = Θ_t.minus(B)`, exact for unions of boxes. D empty → the recoveries written for B apply (still checked on Θ_t by Rule 1). D not empty →
D's numbers and provenance go to the VLM, which writes plans for that part; the planner turns them into motions; `admit` checks them.
Open (D3): Rule 1 needs **one** recovery for all of Θ_t. A plan written only for D helps only if it (or a program) also covers Θ_t ∩ B.

### 4.3 Rule 1: admit a step

```
admit(step, Θ_t, recoveries):
  for b in order(recoveries):                       # VLM order, then fewer set-downs
      if not b.claimed_cover.contains(Θ_t): continue   # cheap filter; claims are re-checked below
      ok = all(viable(b, x_step(t_k), Θ_t) for t_k in branch_times(step))    # x_step(t_k): planned state + residual
      if ok: return Admission(step, b, Θ_t, table=precompute_table(step, F), fallback_ok_on=check_fallback(F))
  return None                                       # -> Rule 4
```
**Per-cell library.** A fixed plan is verified on the cells it claims; `admit` needs Θ_t inside one plan's verified cells (∃b∀θ: the union of
several plans' cells does not count). Narrow ranges therefore need a probe first.

**Program verified for every θ ∈ Θ_t** (parameters from readings, never θ*):
1. `check_motion(prefix, Θ_t)`: the part before the reading passes on the whole set.
2. Reading y = h(π) + n, |n| ≤ ν (`ReadingSpec`). For bin j with reading interval Y_j:
   Θ_j = {θ ∈ Θ_t : (h(θ) + [−ν, ν]) ∩ Y_j ≠ ∅}. Exact by interval arithmetic when h is affine in π (the probe's vertical load is).
3. `check_motion(plan_j from the prefix end, Θ_j)` for every bin.
4. Coverage: ∪_j Y_j ⊇ h(Θ_t) + [−ν, ν] (every possible reading picks a bin).
Then for every θ ∈ Θ_t and every reading it can produce, the executed recovery passes: Rule 1 holds for the program. Continuous programs are
checked as bins (finer bins, less conservative), or by the separate-convexity argument of §3.1 when the parameter is a motion scale.
The reading's producer must not be TRUTH (`WorldState.assert_honest`); the verdict records the producer and ν with its tag.

### 4.4 Rule 3: Θ_t changes during a motion

Precomputed before the motion (all LP / IK / sim work): the admission (b*, Θ_cert); for each branch time t_k the list of recoveries viable
from x(t_k) with the sets they were verified on (checked on the cells of F, not only Θ_t); the fallback checked on F from every branch point.
Online, on each change of Θ:
```
if Θ_cert.contains(Θ_new): continue                        # sound narrowing keeps every certificate (Lemma 1)
elif b := first(b for b, Θ_b in table[t_k] if Θ_b.contains(Θ_new)): switch to b
else: run fallback; certified only if F.contains(Θ_new), else log an uncertified event (counted, no claim)
```
Online work = box-inclusion tests and a table lookup. Design target: one control step (5 ms, `carry.py:61`); to be measured (p50/p95/max).
A change between branch points uses the latest branch point; that gap is covered only by the sampled rows, as today.

### 4.5 Rule 4: no viable recovery

The loop belongs to an orchestrator above the planner, not to the checker:
1. The checker returns, per candidate recovery, the failing rows and the worst vertex or world.
2. The VLM proposes target states (set down on support s, another hold pose, base move) with that feedback.
3. For each target the planner plans the transfer; `check_target` checks (a) Rule 1 for the transfer itself (it needs its own recovery),
   (b) at the target some recovery is viable for all of Θ_t, (c) the target is in R_K (§3.10).
4. First target passing (a)-(c) → move there, then retry the step. None after N_prop proposals or K attempts → certified refusal with the
   per-row reasons (claim 3). No retry from the same (state, Θ) (proposal `:128`).

## 5. Migration

### 5.1 Universes supply data only

| Universe | ThetaSet / cells | ObjectModel | SceneModel | U / R thresholds |
|---|---|---|---|---|
| contents (tray, box) | `ContentsUniverse.theta_box` (`stack/universes/contents.py:35-60`), cells `carry_cert.py:40-41`, seeds `carry_cert.py:185-187` | `objects.coffee_tray` / `objects.boxed` + mug spill row | `scenes.support_tops` (counter, shelf) + MJCF obstacles | `carry.py:72-77` |
| pot | `stack/universes/pot.py` (THETA, nominal ASSUMED, `pot.py:1-8`) | `objects.pan_like` + lid row | stove / table tops from MJCF | pan segments thresholds |
| Demo C | `stack/universes/democ.py` (box all ASSUMED, needs user OK, `democ.py:1-6`) | slab box, no contents | table, pedestal | democ |
| house | not built (`stack/registry.py:14`); `spec_cert` rows in IR | tray + cup | from `house/build` | gatekeeper |

### 5.2 Existing modules → row implementations

`envelope.Envelope` → torque row, legacy backend; `carry_cert.support_row` + `static_cert` support → support row; `objects.OpenLimit` →
`OpenContentRow`; `admit.certify` → dynamic row; `carry._check` / `in_R` → U list and R0; `path_reach.eval_moves` → reach row (refusing);
`clearance.report` + IR `Clear` → collision row; `carry_cert.static_cell` / `probe.static_interval` (`probe.py:93-95`) → thin callers of
`check_motion` on a cell. (Superseded 11-09: `rules.py` was deleted in c8cf380; there is no FL-MPC runtime row layer.)

### 5.3 Equivalence gates (before anything is retired)

A `LegacyPlanAdapter` turns the ScriptedPlanner plan (`stack/planners/scripted.py`) into knots at the same path samples (N_PER_MOVE = 6,
`carry_cert.py:42`) with the a_nom acceleration box, lever split, R = I, quasi-static ID.
- **G1** contents static: for tray2/4/6 × 3 fill cells and box × 9 mass cells, the new checker in legacy mode gives the same action,
  a_admit (within the bisection tol 1e-3), u_nom, per-segment u, u_best_split and support margin as `static_cell`. Same code path → equal floats.
- **G2** `probe.c_admit_max` per task equal.
- **G3** pot: `static_cert.static_rows` on the pot segments equal (when the pot universe runs).
- **G4** dynamic: `carry_cert` sampled records for the same `cell_seed` worlds byte-identical (executor unchanged).
Compare live code against live code: cached RESULT.json files were made on the old object position/cap (`2026-11-03_handoff_notes.md:70`).
Then report the new mode (exact ID, ClosedChainPlanner plan) beside legacy, marked as a different model.

## 6. Decisions for the user, and new ASSUMED bounds

### 6.1 Decisions

1. **D1 Fallback set F** (OPEN 11-05; [11-17: ruled 11-16: F = Θ_0]). (a) prior box Θ_0: strongest, may leave no feasible fallback (box 4.5-5 kg); (b) B: covers only
   changes inside the VLM range; (c) Θ_t widened by a margin (new ASSUMED bound); (d) union of B and Θ_0 ∩ reachable. Any choice is
   stored in the plan and the verdict; outside F the fallback runs uncertified and is counted.
2. **D2 Meaning of R.** [11-17: ruled 11-16: R_K.] R0 (today, no progress test) / R1 (+ next skill precondition and reach) / R_K (planner witness, choose K, node cap).
3. **D3 Rule 2 scope.** Does the VLM write plans for D = Θ_t \ B only, or for all of Θ_t whenever D is not empty (Rule 1 needs one recovery)?
4. **D4 Recovery form.** [11-17: ruled 11-16: programs first.] Per-cell fixed library (needs narrow Θ_t, so a probe first) vs reading-binned programs; cell and bin widths.
5. **D5 Torque row.** Model: legacy quasi-static vs exact ID from the plan. Split checked: what the tracker commands (lever / W⁺ for
   coopimp_safe), with the LP capability as information. Limit: 16 vs 100 Nm (AGENTS.md OPEN). [11-17: 100 Nm, USER 11-05.]
6. **D6 Joint acceleration limit.** [11-17: the planner uses QDD_MAX 6 rad/s² ASSUMED (USER 11-17); no robot limit in `config.py`.] None (torque row only) / an ASSUMED number / measured on hardware later.
7. **D7 Between knots.** Lipschitz / sweep margin (proof under MODEL, more conservative) vs dense sampling (sampled in time).
8. **D8 Object rotation in plans.** Allow (needs the π10 outer box) or keep the object level as a planner constraint for now.
9. **D9 Object–world contact in U** and the clearance margin (5 mm today). [11-17: ruled 11-16: in U, 5 mm ASSUMED.]
10. **D10 Static check sets.** Run the pre-probe static check on B only, or on Θ_0 too.
11. **D11 Sampled rows.** Worlds per set (n), target ε, and the set they are drawn from (cell vs Θ_t).

### 6.2 New ASSUMED bounds this design introduces (rule 2: raise before any claim rests on them)

| # | Bound | On what | Until |
|---|---|---|---|
| A1 | ε_cc closed-chain mismatch [m, rad] | planned reference (internal force under the weld) | derived from weld stiffness (MODEL) |
| A2 | H_I internal wrench under the weld [N, Nm] | tracker + weld | MEASURED from sim (39.6 N seen on the scripted box path) |
| A3 | tracking residuals e_q, TCP, r_tilt per tracker | executed vs planned motion | MEASURED (p50/p95/max over n, per cell and acceleration) |
| A4 | placement residual δ_place [m] | landing pose vs plan | MEASURED |
| A5 | landing residual box for R | landing state spread over θ | MEASURED |
| A6 | reading noise ν for each new program reading | sensing, not the world | MEASURED (probe's TAU_NOISE 0.05 Nm is an existing ASSUMED value) |
| A7 | K, node cap, N_prop | R_K search and Rule 4 loop | design parameters |
| A8 | joint acceleration limit | robot | only if D6 picks a number |
| A9 | Θ_t margin for F | fallback set | only if D1 picks (c) |
| A10 | online budget 5 ms | runtime decision time | design target, measured |

Reused, not new: U_ADMIT 0.9, A_NOM 0.5 m/s², A_MIN, U/R tilt thresholds, PATH_IK_TOL, PATH_MARGIN_MIN, CLEARANCE_MIN, branch_dt 0.25 s,
the weld, the probe-is-truth assumption. Unbuilt skills used inside recoveries (e.g. a set-down with a measured placement) enter only as
ASSUMED and make the verdict conditional (bounds rule 3).

## 7. Build order (after planner step 2)

| Step | Build | Test |
|---|---|---|
| C0 | types: Cell, ThetaSet, RowResult, CheckVerdict, Recovery types, TaskPlan, Admission; Checker protocol; registry entry | set ops vs brute-force grid on random boxes (contains / minus / intersect); JSON round trip |
| C1 | RobotModel, SceneModel (supports + obstacles from MJCF), ObjectModel for tray and box | supports equal `scenes.support_tops`; `pi_vertices` equal `objects.Contents.pi_vertices`; limits read from `config.py` only |
| C2 | legacy rows + LegacyPlanAdapter | gates G1, G2 |
| C3 | dynamic sampled row through Universe + executor | gate G4 |
| C4 | reach + closed chain + joint q/qd (+qdd per D6) on ClosedChainPlanner output | synthetic plans with a known fault (limit crossed between knots, IK miss) are refused with that counterexample; the scripted box path is refused (IK miss 0.15, handoff `:25`) |
| C5 | exact-ID torque row, split laws, π10 outer box | equals legacy at qd = 0, α = 0, R = I (1e-9); 10⁴ random π inside the set never exceed the vertex max; ID equals `mj_inverse` at random states |
| C6 | support, open-content, tilt, collision rows | support equals `support_row`; swept distance never above dense-sampled distance |
| C7 | `admit` with fixed recoveries and programs | a program with one failing bin is refused; a TRUTH-sourced reading raises; ∃b∀θ: a union of two per-cell plans is not accepted |
| C8 | runtime cover table + fallback | narrowing keeps b; a moved set picks another b or the fallback; decision time p50/p95/max over n |
| C9 | R0 / R1 / R_K and the Rule 4 loop with a mock proposer (no VLM service) | Demo C joint-limit trap state is not recoverable unless a set-down target is proposed |
| C10 | `carry_cert`, `probe`, `thresholds` call the checker; old code archived | only after G1-G4 pass and the user agrees |

Open uncertainties of this design: the π10 outer box may be loose enough to refuse cells that pass today; exact ID needs the planner's qdd to
be smooth (TOPP-RA gives piecewise-constant s̈, so qdd jumps at knots: rows must take both one-sided limits); the coopimp internal wrench has no
bound yet (A2), so its torque row is incomplete until it is measured.
