# Comparison plan — our cooperative bimanual safety-control vs NeHMO & HaMMAR

Written plan only (no experiments run). Purpose: define how to compare our
FL-MPC / v4-C\* cooperative bimanual safety controller against the two learned
Hamilton-Jacobi-reachability planners this project is grounded in
(MASTER.md §1). Scope: which comparisons are a fair head-to-head reproducible
in our MuJoCo YOR sim, which are conceptual-only, and the honest gap ledger.

**Source-of-truth for our system:** `docs/design/MASTER.md`,
`docs/design/mpc_formulation.md`, `AGENTS.md`,
`experiments/ab_smooth_vs_bangbang/mpc_fl.py` (FL-MPC),
`experiments/ab_smooth_vs_bangbang/run_ab4.py` (v4-C\*).

> **Metrics-provenance caveat (read first).** The NeHMO/HaMMAR numbers below
> were extracted from the arXiv HTML by a summarizer, and a first pass over the
> NeHMO PDF returned *fabricated* figures (hedged with "likely"/"inferred").
> The HTML-sourced numbers used here are mutually consistent (both papers UR5,
> zero-sum/adversarial, DeepReach-family) and are the ones tabulated, **but any
> number quoted from those two papers MUST be re-verified against the published
> PDF before it appears in a paper/talk.** Rows drawn from the papers are marked
> `[verify]`. Our own numbers are from MASTER.md/AGENTS.md and are internal.

---

## 0. One-paragraph orientation

Both papers solve **decentralized, adversarial (worst-case / zero-sum
differential-game) safe motion planning** using a **learned HJ-reachability
value function** as the safety certificate, at the **trajectory-optimization**
layer, on **UR5 fleets (2–5 arms)**, evaluated mostly on **reaching / carrying**
tasks. NeHMO learns an *unconstrained* inter-arm safety value; HaMMAR extends it
to a **task-manifold-constrained** value (e.g. cup-upright) — which is the
single closest external work to our flagship. Our system instead is a
**cooperative** (joint-QP, both arms yield) **feedback-linearized MPC** whose
safety is **anticipatory HOCBF hard constraints** (not a learned/formal HJ
certificate), tuned for **smoothness and real-time** on a **2-arm 12-DoF YOR**
bimanual cube-into-cup task, with a separate learned-WM latent HJ value handling
the non-geometric spill predicate (out of scope for both papers).

---

## 1. Three-system mapping table

| Axis | **NeHMO** (2507.13940) `[verify]` | **HaMMAR** (2511.03591v2) `[verify]` | **Ours** (FL-MPC + v4-C\*) |
|---|---|---|---|
| Problem framing | Decentralized safe multi-arm motion planning, high-D coupled C-space; others act unpredictably | Same, **plus task-induced manifold constraints** C(x)=0 (e.g. carry cup upright) | Cooperative bimanual arm–arm avoidance **while retaining a grasped-cup task**, smoothly, in real time |
| Safety mechanism | Learned **neural HJR value** capturing worst-case inter-arm safety; used as pairwise constraints in receding-horizon opt | Learned **manifold-constrained HJR value** (Hamiltonian projected onto manifold tangent space via P(x)); DeepReach NN; in receding-horizon opt | **Hard QP constraints**: anticipatory arm–arm HOCBF over the horizon, torque box, cup-upright (box / eff-gravity cone), cup-arm leash, table-floor; **graceful slack-QP fallback** + closed-form torque backstop on infeasibility |
| Cooperative vs unilateral | **Unilateral / defensive** — zero-sum game, treats other arms as adversarial disturbance, no access to their policies | **Unilateral / defensive** — same adversarial formulation, no knowledge of others' policies | **Cooperative** — FL-MPC is a **joint QP over both arms' accelerations**, so both yield; v4-C\* is decentralized-cooperative (each arm own QP, intruder carries +0.03 m margin ⇒ yields earlier) |
| Guarantee type | **Formal-ish**: forward-invariance under a *learned* HJ value (guarantee only as good as the NN approx of the PDE) | Same, on the constrained manifold | **Anticipatory heuristic**: first-order HOCBF rows over a linear horizon; **no formal reachability certificate**. Hard guarantee held is the **torque box** (never relaxed); collision rows are slackened when infeasible |
| Online compute / real-time | Planning **36±35 ms** dual-UR5 (IPOPT, RTX 4070); 43±52 ms @5 arms `[verify]` | Planning **0.10±0.087 s** 2-arm cup; 0.14±0.14 s doorway; 0.17±0.18 s @5 arms `[verify]` | FL-MPC ~**40 ms/step** (Numba box-QP, CPU); v4-C\* reactive, real-time. Exact analytic horizon (no MuJoCo rollout in the controller) |
| Task | Reaching, random start/goal, 2/3/5 UR5 + UR5-Kinova `[verify]` | 2D-circle particle (ablation), object-carrying, **cup-holding upright**, doorway-crossing; +real cup-carry validation `[verify]` | **Cube-collection into a held, lifted cup** (2-arm YOR, 6+6 DoF): left holds/lifts cup upright, right deposits scattered cubes; deposit forces arm–arm crossings |
| Reported metrics | dual-UR5: **92% succ / 2% coll / 36±35 ms**; 5-UR5: 66% / 8% / 43±52 ms; UR5-Kinova 79% / 6% / 46±46 ms; baselines CHOMP 86%/14%/570 ms, Ha et al. 79%/13% `[verify]` | cup-hold (2): **82% succ / 2% coll / 0.10 s**; doorway 71% / 1% / 0.14 s; 5-arm cup 57% / 0.17 s `[verify]` | v4-C\* N=50: coll **68%→4%**, grasp **0%→82%**, task **0%→74%**, chatter 10.7→3.3. FL-MPC N=50: coll **15%→0/50**, fling 7.45→1.80 m/s. Spill 46/50 (geometry-blind; WM's job) |

---

## 2. Shared comparison axes (where a same-case comparison is meaningful)

The three systems are not the same robot or task, so a fair comparison is on
**normalized per-episode outcome axes**, holding the *scenario class* (two arms
forced to cross, one carrying a tilt-constrained object) fixed:

1. **Collision rate** (episodes with any arm–arm contact / total). All three
   report this directly — the primary safety axis. Ours: v4-C\* 4%, FL-MPC 0/50;
   NeHMO 2% (dual), HaMMAR 2% (cup) `[verify]`.
2. **Task-success rate** (task completed *and* no collision *and* constraint
   held). Definitions differ — NeHMO/HaMMAR = reach/carry goal reached;
   ours = `deposited≥K ∧ count_spilled==0 ∧ no collision ∧ no drop`. Compare
   trends, not absolute values; report our success both with and without the
   spill term so the *geometric* success is on the papers' footing.
3. **Manifold / constraint-violation rate** — **only HaMMAR shares this.**
   Cup-upright tilt violation (fraction of steps the cup axis exceeds the tilt
   bound). Directly comparable: their manifold-constrained HJ value vs our
   cup-upright hard QP row (box or effective-gravity cone, `mpc_fl.py` block 3).
4. **Smoothness / jerk** — our differentiator; the papers report path length,
   not jerk. Compare our jerk-regularized profile (chatter 10.7→3.3;
   FL-MPC jerk term `W_JERK`) qualitatively against their path-length
   (NeHMO 8.5±4.6 rad, HaMMAR 7.7±3.1 rad `[verify]`) as a proxy; add a
   path-length column to our sweep so at least one axis is literally shared.
5. **Clearance margin** — min arm–arm distance maintained. Ours = HOCBF
   `H_MARGIN` (0.02 m base, up to 0.06 m urgent) + v4-C\* intruder +0.03 m.
   Report realized min-clearance distribution to compare against their safety
   value's realized margin.
6. **Online compute / solve time** — FL-MPC ~40 ms vs NeHMO ~36 ms (very close);
   HaMMAR ~100 ms. Same-axis, but note hardware differs (we are CPU box-QP; they
   use IPOPT + a GPU for the NN) — report our number CPU-only, theirs as quoted.
7. **Distributional robustness** — our N=50 *randomized-geometry* sweep vs their
   random start/goal sampling. Shared axis = success/collision **as a
   distribution over randomized layouts**, not a single demo.
8. **Cooperative yielding** — *not* shared as a metric (they are unilateral); it
   is a **qualitative axis** where we differ in kind, measured by "both arms
   deflect" (our joint QP) vs "ego arm defends against worst-case other."

Axes NOT shared (do not force a number): formal reachability guarantee (they
have a learned HJ certificate, we do not); multi-arm scaling to 5/30-DoF (we are
fixed 2-arm/12-DoF); non-geometric spill (ours only — handled by the WM latent
value, MASTER §7/§8, orthogonal to both papers).

---

## 3. Concrete protocol (per paper)

### 3A. HaMMAR (2511.03591v2) — **closest match; partial head-to-head reproducible**

**Why it's the right head-to-head:** it is the only external work with a
**cup-upright manifold constraint + 2-arm manipulation** — the same physical
scenario as our flagship. The comparison of guarantee *kind* (learned
manifold-constrained HJ value vs our analytic cup-upright hard QP row) is the
scientifically interesting one.

**What IS reproducible in our MuJoCo YOR sim (same-case, our-metrics):**
- Reuse **`scenes/scene_yor_cup_collect.xml`** + the **N=50 randomized-geometry
  crossing sweep** already used for the FL-MPC 15%→0/50 result
  (`experiments/ab_smooth_vs_bangbang/run_ab4.py` harness pattern; flagship
  driver `scripts/make_cup_collect_safety_demo.py --controller flmpc`).
- Enable the cup-upright rows two ways and sweep both:
  (a) per-axis tilt **box** (`cup_cone_facets=0`),
  (b) **effective-gravity cone** (`cup_cone_facets>0`) — the dynamic-spill-aware
  manifold, the fairest analog to HaMMAR's manifold constraint.
- Report on the **shared axes** of §2: collision, geometric success,
  cup-tilt-violation rate, min-clearance, path length, solve time.
- Add a **cup-carry-while-crossing** variant (left arm *transports* the upright
  cup across the workspace while the right arm crosses) to mirror HaMMAR's
  cup-carrying task rather than only the deposit task.

**What is NOT a byte-identical reimplementation (and why):**
- Different robot (YOR 6-DoF arms vs UR5), different task semantics (deposit
  cubes vs navigate cup to goal), and — crucially — a **different safety
  primitive** (their planner *is* a learned manifold-constrained HJ value inside
  trajectory optimization; ours is an analytic HOCBF hard constraint inside a
  QP). Re-implementing their DeepReach manifold-HJ trainer on YOR is a research
  project, not a comparison run.
- **Fairest qualitative comparison:** frame it as *two ways to keep a cup on its
  task manifold while avoiding an arm* — learned-HJ-certificate (theirs,
  formal-ish, ~100 ms, unilateral) vs analytic-HOCBF-in-cooperative-QP (ours,
  heuristic, ~40 ms, both arms yield). Tabulate the §2 axes side by side with the
  explicit note that robot/task differ, so the numbers are *positioned*, not
  claimed head-to-head-identical.
- **Optional stronger comparison (research-effort, flag as future):** train a
  manifold-constrained latent/explicit HJ value for the cup axis and drop it into
  our safety interface (`eval/safety_module.py` — a learned `V(x)` already swaps
  into the HOCBF rows per MASTER §7). That would make it a true
  same-controller / different-certificate ablation.

### 3B. NeHMO (2507.13940) — **conceptual analog; only partial reproduction**

**Natural analog on our side:** **v4-C\*** (decentralized reactive HOCBF-QP,
`run_ab4.py`) — both are decentralized per-arm safety; the difference is
*learned adversarial HJ value* (theirs) vs *analytic HOCBF barrier* (ours), and
*unilateral defensive* (theirs) vs *cooperative* (ours, intruder margin makes it
asymmetric-cooperative).

**What IS reproducible:**
- A **2-arm reaching sweep with randomized start/goal** in our YOR sim (their
  core task), scored on collision / success / planning-time / path-length — the
  four axes NeHMO reports. Our N=50 randomized-geometry harness already produces
  exactly these; we add a pure reaching scenario (no cup) so the *task* matches
  NeHMO rather than our cube-into-cup task.
- Report FL-MPC solve ~40 ms directly against NeHMO's 36±35 ms (nearly equal,
  and the honest headline: comparable compute, we drop collisions to 0/50 on
  *our* distribution but with **no formal guarantee**).

**What prevents a full head-to-head:**
- NeHMO's headline contribution is **scaling** (2→3→5 arms, up to 30-DoF, plus
  heterogeneous UR5-Kinova). Our sim is **fixed 2-arm bimanual (12-DoF)** — we
  cannot reproduce the 5-arm / heterogeneous results at all. State that plainly;
  our comparison is confined to the **dual-arm (12-DoF)** row of NeHMO's tables,
  which is exactly the regime we share.
- Robot morphology differs (UR5 vs YOR); their evaluator/IPOPT stack differs
  from our Numba box-QP. So even the dual-arm reaching comparison is
  *axes-aligned, not drop-in identical.*
- **Fairest qualitative comparison:** decentralized safety with a *learned
  worst-case HJ certificate that scales to many arms* (theirs) vs a
  *cooperative, real-time, smooth analytic controller specialized to a
  two-arm contact task* (ours). We concede scaling + formal certificate;
  we claim cooperation, smoothness, comparable dual-arm compute, and a
  task-aware (spill) channel they do not address.

### 3C. Sweeps to reuse (both papers)

- **N=50 randomized-geometry collision sweep** (the FL-MPC 15%→0/50 and v4-C\*
  68%→4% result) — primary distributional-robustness evidence; reuse verbatim,
  add path-length + realized-min-clearance + cup-tilt-violation columns so the
  §2 shared axes are all populated in one run.
- **Cone / inertial cup sweeps** (`cup_cone_facets`, inertial cup-carry from
  commit b55c9bd) — reuse for the HaMMAR cup-upright manifold comparison (3A).
- **v4-C\* margin grid** (`artifacts/margin_grid/`) — reuse to characterize the
  cooperative-yielding axis (the non-monotonic 0.03 optimum) that neither paper
  has an analog for.

---

## 4. Honest gap ledger

### Where we are WEAKER
- **No formal reachability guarantee.** Both papers carry a (learned) HJ
  certificate with forward-invariance semantics; our safety is an **anticipatory
  HOCBF heuristic** over a first-order-linearized barrier. On QP-infeasibility we
  *slacken the collision rows* (torque box stays hard) — safety is traded for
  feasibility exactly where it is hardest. HaMMAR's constrained-HJ value is a
  genuinely stronger correctness story on the manifold.
- **~40 ms solve, CPU box-QP, sim-only.** Comparable to NeHMO (~36 ms) but we
  have **no hardware validation**; HaMMAR reports real-world cup-carry runs. Our
  ZED mount is *proposed/confirm-on-hardware* (MASTER §3).
- **No multi-arm scaling.** NeHMO's central result (up to 5 arms / 30-DoF,
  heterogeneous) is outside our 2-arm sim entirely.
- **WM/latent-value channel is OOD-brittle** (spill AUC 0.999→0.746 under DR,
  unfixed — MASTER §8). A learned safety value inherits this; the papers' HJ
  values are analytic-dynamics-grounded and do not have this failure mode.
- **Our metrics are internally defined** and not yet cross-checked against an
  external benchmark; the papers report against named baselines (CHOMP, Ha et
  al.).

### Where we are STRONGER
- **Genuinely cooperative avoidance.** FL-MPC's joint QP makes **both** arms
  yield; both papers are unilateral/worst-case-defensive. For a *cooperating*
  bimanual system (one embodiment, shared task) this is the right model and
  avoids the conservatism of treating your own other arm as an adversary.
- **Smoothness as a first-class objective.** Jerk-regularized, exact-prediction
  FL-MPC (chatter 10.7→3.3; fling 7.45→1.80 m/s). The papers optimize
  feasibility/path-length, not jerk/actuator-fling — relevant for real hardware
  and for holding a fragile grasped object.
- **Exact horizon prediction with no in-loop rollout.** Feedback linearization
  makes the double-integrator prediction exact and the sensitivity Φ a constant
  precomputed matrix — no linearization divergence (the failure that killed
  mpc7), any horizon length, ~40 ms.
- **Task-aware non-geometric safety (spill).** The WM latent HJ value vetoes an
  *anticipatory* spill predicate with **no geometric signal** (V(z)<0 before
  P(spill) rises: 8/8→0/8 spilled, MASTER §8) — a failure mode **neither paper
  addresses**. HaMMAR's manifold constraint is the nearest analog but is a
  *geometric* orientation constraint, not a learned latent slosh predicate.
- **Hard-real-time on CPU** with no GPU NN inference in the safety loop (the HJ
  value NN is only in the orthogonal spill channel).

### Net positioning
Against **HaMMAR**: same scenario, weaker guarantee, faster + cooperative +
adds a learned latent spill channel HaMMAR lacks — the honest claim is
"cooperative & smooth & spill-aware, at the cost of formal manifold-HJ
correctness." Against **NeHMO**: comparable dual-arm compute and lower collision
on our distribution, but we concede scaling and the formal certificate; we win
on cooperation and smoothness. In both cases the strongest future move is to
**drop a learned (manifold-constrained) HJ value into our existing
`safety_module` HOCBF-row interface**, turning a positioning comparison into a
same-controller / different-certificate ablation.
