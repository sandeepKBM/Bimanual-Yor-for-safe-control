# Overnight 2026-10-08: analytic sets replace HJ; a simulator artifact explains the slow-carry drops

All of this ran on the tray-and-ball testbed (MuJoCo, simulation only, no hardware). Every experiment was pre-registered before its
TEST carries were touched, and every agent result was re-checked by me. The headline setup is the realistic tracker (3 mm noise,
60 ms latency, 2% dropout), with 600 TEST carries × 3 seeds.

## Five things to know

1. **The analytic set filter is good enough: HJ is not needed here** (Experiment A, PASS).
   - F5 is the set filter: ball reachable sets, then a QP, with a backup law.
   - It dropped **0 of 1800** carries. The latent-HJ filter drops 13.6%, and F5 makes the same path progress (0.339 vs 0.345).
   - The cost is intrusiveness: 15.5% of ticks overridden vs 3.5%.
   - It runs at 2.6 ms per tick.
   - It also drops 0 with a deliberately wrong friction map. It holds with the simulator artifact removed (point 5).
2. **A VLM prior buys efficiency, not safety, on this testbed** (Experiment B, the sets-vs-points test KILLED by a floor effect).
   - F5 is safe across the whole friction range, so a wrong friction belief rarely causes a drop.
   - The predicted danger is real but small: every drop of the point arm was a confidently wrong prior that overestimated mu 4-6×
     on a slippery ball (0.39 pp at a 40% confident-error rate).
   - The conformal set kept its 90% coverage at every error rate. The ±2σ band fell to 54%.
   - Knowing mu exactly cuts F5's overrides from 13.3% to 8.1%. A calibrated prior gets part of that. The more often the VLM is
     confidently wrong, the wider the calibrated set must be, and the less is left.
3. **At an executable speed, F5 changes the picture** (C-F5, exploratory).
   - With F5 guarding, the carry runs **2.3× faster with 0.78% drops**. The best single safe speed under the learned filter drops
     26.7%.
   - At that speed, knowing mu is worth -11 pp overrides, +13 pp goal reached and 5% time.
   - The sets-vs-points test on drops was KILLED again: +0.5 pp, CI includes 0.
4. **Probing and online shrinking did not pay under the frozen protocol** (Experiment C, P7 KILL on drops; P8 KILL on time).
   - P7, the worst-case speed rule after a probe, was 10% faster but dropped +5.9 pp more.
   - My decomposition: the extra drops came from the probe, not the speed choice. Probed carries run at the same slow speed as the
     baseline dropped +8.6 pp more.
   - P8, which shrinks the set from the carry's own motion, was safe, but its belief barely moved: 1.3% time saved.
5. **I found a simulator artifact that affects every v2 testbed.**
   - `world_model/v2/data/contents.pin_cup` re-pins a light (0.3 kg) tray or cup before each physics step, and the body stays
     dynamic during the step.
   - An off-centre load tilts it within the step, so balls drift outward on a level, still tray: +9.6 mm from x = 12 cm in 1.2 s.
   - A static tray or a 1000× heavier tray shows zero drift.
   - It barely touches the short bridge carries behind Experiments A and B (F3 19.0% vs 19.0%). It **causes all of F5's drops on
     slow carries** (8% → 0%), and about three quarters of P7's probe penalty (+4.0 → +1.0 pp on VAL).
   - It is also used by the loose-items, block-tower, opaque-fill and bead testbeds and the spill-data generator.
   - **I did not fix it**: fixing it changes all data.

## Verdicts

| experiment | prereg sha | primary verdict | key number |
|---|---|---|---|
| A: F5 vs latent HJ | 6cfa9627 | **PASS** (both setups) | 0.0% vs 13.6% drops; progress equal |
| B: calibrated set vs point prior | 049a7b5d | **KILL** (KB1); KB2 dead weight, no headroom | point - conf = +0.11 pp [0.00, +0.44] |
| C: P7 probe + worst-case speed | e091efd1 | **KILL** on drops | +5.94 pp [+4.50, +7.44]; time 0.902 |
| C addendum: P8 online shrink | 4bbb793e | **KILL** on time | time 0.987; drops -0.39 pp |
| C-F5 (exploratory): F5 at speed | 97f0b081 | KS1 sets-beat-points **KILL** | F5 @ 0.99: 0.78% drops, 2.3× faster than P6 |
| Robust chain R1b (learned value) | dc88d87c | **KILL** | -0.019 (-0.111, +0.078) |
| Robust chain R2 | dc88d87c | **fails** (the ACI residual set dominates) | ours +5.9 pp vs ACI |

Result files:
- `bimanual_data/tray_rep/sets/expA/RESULT_A.md`
- `vlmprior/expB/RESULT_B.md`
- `boed_p7/RESULT_C.md`
- `boed_p8/RESULT_CP8.md`
- `boed_f5/RESULT_CF5.md`
- `robust/RESULT_ROBUST.md`

## What this means for the paper (VLM priors and the sim-to-real gap, in real time)

- **Safety is no longer the bottleneck.** Analytic ball sets with a backup law are safe and real-time without a learned value. For
  the problem statement, the sharp question becomes what a prior is worth: on safety it buys little here, because the filter already
  covers the whole range.
- **The live mechanism is calibration against informativeness.**
  - Confident VLM errors are the only source of prior-caused drops.
  - Calibrating them away (conformal) keeps coverage but widens the set as the error rate rises.
  - Online shrinking recovers part: -5.4 pp of overrides at speed, against the oracle's -10.8 pp.
  - This is measurable, and it lives in efficiency.
- **For a safety claim about sets vs points,** you would need a regime where a wrong mu flips the safe action. Examples: faster carries,
  heavier or rolling objects near the authority limit, or a filter whose model-error box is not sized for the worst case. On this
  testbed the claim cannot be shown.

## Decisions for you

1. **Fix `pin_cup`?** Options: a mocap or weld kinematic body, or a very large carried-body inertia. Which testbeds and datasets
   should be regenerated? The spill data and the bead replication use it too.
2. **A closed-loop nominal planner for speed tests.** The BOED plan is open-loop in acceleration, so a filter that brakes leaves the
   tray off-goal. That caps F5's goal rate at 49-62% in C-F5, so time and goal-reached comparisons there are planner-limited.
3. **Re-run C on fixed physics?** On VAL, the probe penalty almost vanishes (+1.0 pp), so P7 might pass on time.
4. **Where should the prior's value be tested next?** Efficiency at speed (C-F5 style, with fixed physics and planner), or a harder
   regime for safety.

## Other notes

- **A revision of an earlier conclusion.** The earlier BOED result blamed P3's drops on "the decision boundary of the speed rule".
  The P7 decomposition and the artifact check show it was mostly the probe's after-state, amplified by the artifact.
- **Cost.** F5 runs in about 2.6 ms per tick on average. Its worst real tick is about 37 ms at the 99th
  percentile; the logged 12 s maxima are numba compiling on first use.
- **The online friction belief undercovers on the long carries:** 0.77 vs 0.90 target. It was calibrated on the short bridge carries.

## Files changed tonight

Code (all in the untracked tree `world_model/v2/tray_rep/`):
- **Edited by me:**
  - `sets/reach.py` and `sets/filter.py`: a `calib` switch, default unchanged.
- **New, mine:**
  - `sets/expA_analyze.py`, `sets/expB_analyze.py`
  - `sets/slurm/expA_*`, `sets/slurm/expB_*`
  - `boed/p7_analyze.py`, `boed/run_fxi.py`, `boed/fxi_analyze.py`, `boed/p8_analyze.py`
  - `boed/slurm/{p7,fxi,p8}_*`
- **Agent-built, verified by me:**
  - `sets/vlmprior/` (new)
  - `boed/p7.py`, `boed/p7_checks.py`, `boed/p8.py`, `boed/p8_checks.py` (new)
  - additive P7 hooks in `boed/episode.py`, `boed/run.py`, `boed/launch_local.py`
  - one F5P registration line in `bridge/filters.py`

Data and docs:
- New pre-registrations and results under `bimanual_data/tray_rep/{sets/expA, vlmprior, boed_p7, boed_f5, boed_p8}`.
- `robust/RESULT_ROBUST.md`.
- Appends to `tray_rep/JOBS.txt`.
- This report.
- Memory notes: `vlm-prior-set-experiments-2026-10-08`, `pin-cup-drift-artifact`.

Not touched: frozen BOED, bridge and robust result folders; checkpoints; datasets; git.

## Tests

**Run:**
- `pytest sets/tests` (44 pass) and `sets/vlmprior/tests` (19 pass).
- Bridge smoke checks: ALL PASS on a fresh directory.
- F5 true-state check on the 16 hardest carries: the ball stays ≥ 28 mm from the edge.
- Reproduction checks:
  - the bridge baselines match the stored records (drops, ticks, interventions);
  - F3 @ 0.4 equals the frozen P6 on 1800 records;
  - a fresh P8 null-belief run equals P6.
- The hidden-friction audits of F5, F5P, P7 and P8.
- The artifact diagnostics on VAL: static vs pinned tray, heavy tray, and impact on A, C and F5.

**Not run:**
- The noise-free C-F5.
- Any experiment on fixed physics beyond the VAL diagnostics.
- A real VLM.
- Anything on hardware.

## Rollback

The edits are in untracked files.
- **Restore F5's original behaviour:** remove the `calib` lines from `sets/reach.py` (the ReachParams field and `self._map`) and from
  `sets/filter.py` (`calib=1.0` and its pass-through).
- **Remove tonight's new code:**
  `find world_model/v2/tray_rep/sets/vlmprior world_model/v2/tray_rep/boed/{p7,p7_checks,p8,p8_checks,run_fxi,fxi_analyze,p7_analyze,p8_analyze}.py world_model/v2/tray_rep/sets/expA_analyze.py world_model/v2/tray_rep/sets/expB_analyze.py -delete`
- **Then remove the agent's hooks:** the P7 hooks in `boed/episode.py`, `run.py` and `launch_local.py`, and the F5P line in
  `bridge/filters.py`.
- Result folders can be deleted the same way. The frozen ones were never written.
