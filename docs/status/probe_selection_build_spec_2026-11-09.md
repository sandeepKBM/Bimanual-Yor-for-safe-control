# Build spec: LLM-driven probe selection (2026-11-09)

> Status: DESIGN, nothing built. Builds the 11-08 ruling ("the LLM proposes phenomena, skills supply the numbers") on top of the
> design page https://claude.ai/artifact/HfAnaFjXm6jYusvhFzCZXM (phenomena PH1-PH8, coverage rule, Rule 2 as T1/T2/T3, U1-U12).
> Simulation only. Paths are under `world_model/v2/`. Line numbers are from the worktree on 11-09. Other agents are editing
> `stack/{tasks,goal,outcome}.py`, `stack/checks/*`, `stack/proposer/*`, `nero/contents/{probe,same_worlds,carry_cert,envelope}.py`;
> each step below says which of them it depends on.

## 1. Current state

| What | Where | Fact |
|---|---|---|
| Proposer reply | `stack/proposer/schema.py:3-8, 119-125` | Fields `belief` {param: [lo, hi]}, `steps`, `recoveries`, `fallback`. No phenomena field, no probe field. |
| Numeric belief | `schema.py:58-79` `params()` | Per task "sanity ranges" (box contents 0-20 kg, pot puck 0-10 kg: ASSUMED). The prompt shows them (`prompt.py:75-76`). On 11-05 the model copied them into B. |
| Numeric `covers` | `schema.py:109, 277-300`; `prompt.py:89, 95` | Each recovery may claim a numeric sub-range of B. Not verified by anyone. |
| Step vocabulary | `schema.py:40-47`, `:25` | lift / move / place / release; recoveries hold / retrace / set_down / let_go. No probe skill, no base motion. |
| Rule 2 today | `proposer/compile.py:172-191` `probe_in_belief`; `nero/contents/probe.py:532-537` | Mass interval vs B on one dimension. Outside B -> "refuse (probe set outside B)". |
| Re-ask | `proposer/propose.py:117-127`; `prompt.py:128-141` | Exists; no caller in any run path (grep 11-09). |
| Probe = one fixed skill | `probe.py:403-412` `make_probe`; `:59-60` | Methods "poslift" (default, 11-09) and "ladder". Runs whenever the static verdict on Θ_0 refuses (`probe.py:517-521`); the LLM does not choose it. |
| What the probe reads | `probe.py:190-206` `Probe.read` | Vertical claw force only -> total mass. **It does not read moments: horizontal CoM is not measured today** (MODEL-possible per `probe_limits_2026-11-05.md`, not built). |
| Probe calibration | `nero/results/probe_calib_poslift_2026-11-09/CALIB.json` | MEASURED, n = 20 per task, noise-free reads: EPS_LO 0.0766 kg, EPS_FREE 0.0316 kg, DL_STEP_MAX 0.9332 kg (n = 349 rungs). Free interval: box 9/20, tray6 19/20, pot 20/20. Tracker: FL-MPC LeverCtl (pre-11-07). |
| Probe limit | `nero/contents/envelope.py:53`, `stack/checks/payload.py:15` | U_ADMIT 0.9 (ASSUMED) in code. Ruling 11-09 says the probe limit is u <= 1.0; not in code yet. |
| Rung admission row | `stack/checks/probe_rung.py:16-22` | Unified row exists (payload at rest + overshoot O_MAX 1.0 kg ASSUMED). Not yet called by the probe (plan step 2). |
| Skill registry | `stack/registry.py:16-18` | Only `pregrasp`. No measuring skills, no phenomenon table. |
| Skill protocol | `stack/types.py:137-143` | pre / plan / post. No "measures", no output set. |
| Task table | `stack/tasks.py:56-82` `TaskDef` | No phenomenon list. Prior per task at `tasks.py:216-218` (one copy). |
| Tilt row | `stack/checks/tilt.py:7-8` | Registered MISSING. |
| Codex run path | `nero/contents/same_worlds.py:16-21, 135-148, 179` | `--plan-source codex`: one plan per task, compiled, run with `--outside-b refuse|carry`. |

## 2. Target

1. The prompt shows the task, the scene facts, and one card per phenomenon and per built skill. It shows no contents prior and no
   sanity ranges of hidden parameters.
2. The LLM reply lists phenomena. For each it gives either `measure: <skill id>` or `unmeasurable: <reason>`. It may name a phenomenon
   outside the catalog (`id: "NEW"`). It lists the probes in order, then steps, recoveries and fallback. No number about the contents.
3. The stack maps each phenomenon to its catalog entry. A skill that does not measure that phenomenon rejects the plan
   (`plan_rejected[schema]`). Probe order must respect the skills' preconditions; a bad order rejects (never reordered or repaired).
4. Before each probe the checker evaluates the skill's own limit rows on the current Θ_t (for the weigh: `checks/probe_rung`).
   Fail -> the probe is skipped and recorded; its phenomena go to their fallback.
5. Each reading intersects Θ_t on its own θ fields. The prior box is never shrunk to model sensing (bounds rule 4).
   A reading disjoint from the previous sound set is T3: stop, counted event.
6. Coverage: the checker's own list Φ(task) decides. Each φ in Φ is measured (a skill ran) or on its approved fallback; else uncovered.
   The LLM's list changes only which probes run. A phenomenon the LLM omits sits on its fallback.
7. Rule 1 (`checks/admit`) runs on Θ_t. If no recovery is viable: Rule 2 T2 re-ask with the readings and the failing rows.
   T1 fires when a reading shows a phenomenon in Φ that the plan did not name. Then validation, coverage, Rule 1 again.
8. If the static verdict is already the same on all of Θ_0, a selected probe is not run ("no_probe_needed", as `probe.py:517-518`).

## 3. Data model

**Phenomenon** (`stack/skills/catalog.py`, pure data, no `nero` import):
`Phenomenon(id, name, sentence, theta_fields, rows, objects)`. Ids PH1-PH8 as on the design page: PH1 total mass, PH2 horizontal CoM,
PH3 CoM height, PH4 liquid slosh, PH5 hidden loose item, PH6 visible loose item (mug, capsule), PH7 per-mug fill, PH8 loose lid.

**SkillEntry** (same file): `id, status (built | design), measures [PH ids], theta_fields, output (interval | lower_bound | presence |
monitor_event), range (Bound), noise {read: Bound, residual: Bound, propagation: str}, guarantee (str), preconditions [ids or state
facts], own_limits [(row name, Bound)], needs_checker (bool), one_sided (bool), tracker (str), cost (str), calib (path), code
("module:attr", resolved through the registry), card (dict shown to the LLM)`. Every number is a `stack.types.Bound` with its tag.
Only `status == "built"` entries get a card and can be selected. The weigh is ONE entry, `weigh`; poslift vs ladder stays a stack
setting (`make_probe(method=)`), not an LLM choice.

**Fallback table** (same file): `FALLBACK[(task kind, PH)] = Fallback(kind, set or row, tag, status)` with status `approved | pending`.
Already in code, no new assumption: PH2 and PH3 = container interior (MODEL, the contents prism P); PH7 = min(fill_max, total fill hi)
(MODEL, `probe.interval_spec`); PH8 = lid-slide row, μ_lid 0.5 (ASSUMED, existing). Pending (blocking question Q1): PH4, PH5, PH6.

**Φ per task**: new field `TaskDef.phenomena: tuple[str, ...]` in `stack/tasks.py` (the one per-task table). Values in §4.

**Registry**: new kind `MEASURE` in `stack/registry.py`: `"weigh": "world_model.v2.adapters.skills_contents:Weigh"` (adapter layer,
because it drives the carry sim). Design entries are registered as `None` (placeholder: `get` raises NotImplementedError).

**Proposer schema v2** (replaces `belief` and `covers`; exact JSON):
```
{"phenomena": [
   {"id": "PH1", "why": "<one sentence>", "handle": {"measure": "weigh"}},
   {"id": "PH3", "why": "<one sentence>", "handle": {"unmeasurable": "<one sentence>"}},
   {"id": "NEW", "name": "<short name>", "why": "<one sentence>", "handle": {"unmeasurable": "<one sentence>"}}],
 "probes": [{"skill": "weigh"}],
 "steps": [{"skill": "lift", "args": {"dz": 0.05}, "why": "..."}, ...],
 "recoveries": {"0": [{"skill": "set_down", "args": {}}], ...},
 "fallback": [{"skill": "let_go", "args": {}}]}
```
Validation (strict, never repairs, as `schema.py:8`): required keys exactly these five; `id` in the catalog or "NEW"; no duplicate id;
`measure` names a built skill whose `measures` contains the id; every skill in `probes` is named by some `measure` handle and vice versa;
the probe order meets each skill's preconditions; numbers appear only in step `args`; `why`/`name`/reason strings ≤ MAX_WHY (ASSUMED 400).
`unmeasurable` for a phenomenon that has a built skill is accepted and recorded (completion cost, not a safety issue).
The old format stays replayable behind `BIMANUAL_LEGACY_PROPOSER_V1=1` (11-05..11-09 caches are keyed on the old prompt).

**Missing skill** (open ruling 11-08, not decided here). When no built skill measures φ, the options are:
(a) refuse while φ is in Φ; (b) a designer range (a new ASSUMED bound per φ: A-S5 μ_min 0.3, A-S6 μ_i,min 0.2, A-S7 k_dyn 2);
(c) worst case to the robot limits (e.g. μ = 0: no lateral acceleration, refuses most carries); (d) build a skill (only PH5 has a
plausible one, `shake_listen`, one-sided); (e) keep φ out of Φ for now and state it as a limit (today's behaviour).
For a "NEW" name outside Φ (U1): F = log only; E = refuse; H = log, with a per-run switch "refuse on new names".

## 4. Skills per task

Built today: `weigh` only (PH1). Design only: `hold_moment` (PH2), `shake_listen` (PH5, one-sided), `watch_item` (PH6, PH8, runtime).

| Task | Φ (checker list) | Measured by a built skill | No built skill -> fallback |
|---|---|---|---|
| box | PH1, PH2, PH3, PH5 | PH1 `weigh`: interval 9/20, else lower bound or rung_not_admissible 11/20 (MEASURED, poslift CALIB, n = 20) | PH2, PH3 interior (MODEL); PH5 pending Q1 |
| tray2/4/6 | PH1, PH2, PH3, PH4, PH6, PH7 | PH1 total (tray6 interval 19/20, MEASURED n = 20; tray2/4 not calibrated under poslift) | PH2, PH3, PH7 MODEL; PH4 pending Q1 (sim has no slosh); PH6 mugs (free bodies, `cup_mu`) pending Q1 |
| pot | PH1, PH2, PH3, PH5, PH8 | PH1 interval 20/20 (MEASURED n = 20) | PH2, PH3 MODEL; PH5 puck pending Q1; PH8 lid row (ASSUMED μ_lid 0.5, existing) |
| coffee tray | PH6 (capsule) | none: runs only in AgentChord's sim (embodichain); our skills do not run there | out of this build until the task exists in the NERO sim |
| table behind (tray on swerve base) | PH1, PH2, PH3, PH4, PH6, PH7 | PH1 as tray | base accelerations and turns add lateral load: PH3, PH4, PH6 bind harder; the tilt row is MISSING (`checks/tilt.py`); base tip-over is a checker row on PH1/PH3, not a new phenomenon; the step vocabulary has no base motion (`schema.py:25`) |

## 5. Build steps (one commit each, with its test)

| # | Commit | Files | Test | Waits for |
|---|---|---|---|---|
| 1 | Catalog: phenomena, skill entries, fallback table, cards | new `stack/skills/catalog.py`; `stack/registry.py` (kind MEASURE); `stack/tasks.py` (`phenomena`) | `stack/tests/test_catalog.py`: every number a tagged Bound; each Φ id exists; design entries not selectable; cards deterministic and free of prior numbers; dependency test passes | tasks.py edits by the other agent landed |
| 2 | Proposer schema v2 + prompt + mock replies | `stack/proposer/{schema,prompt,mock}.py`; `stack/tests/test_proposer.py` | each bad-reply class rejected with its reason (numeric field, wrong skill, unknown id, probe without handle, bad order, `covers`); v1 replay under the legacy switch; prompt holds no hidden truth and no sanity ranges | other agent's proposer edits landed |
| 3 | Weigh as a skill (adapter) | new `adapters/skills_contents.py` (wraps `probe.make_probe`); reads the probe limit from one constant | Θ and `why` equal today's `_e2e_job` on the golden seeds, bit-identical; own-limit row = `checks.probe_rung` | **plan step 2 "probe" switch** (`rung_torque_u`, `Probe.admissible` -> `checks.payload`/`probe_rung`); the u <= 1.0 constant change |
| 4 | Probe runner | new `stack/probing.py` (core: no `nero` import) | fake skills: own limit fails -> skipped + recorded; Θ_t only shrinks; T3 contradiction -> stop event; order checked | step 1 |
| 5 | Coverage row (report mode first) | new `stack/checks/coverage.py`; `stack/checks/admit.py` takes the coverage Θ_t | property test: drop or shuffle the LLM's phenomena, same readings -> same admission verdict; enforce mode refuses an uncovered φ; NEW names follow the U1 switch | steps 1, 4; **plan step 2 "contents admission" switch**; enforce mode waits for Q1 |
| 6 | Rule 2 T1/T2/T3 + re-ask in the run path | `stack/proposer/{prompt,propose}.py` (`reask_prompt` v2: readings + failing rows + worst vertex); `nero/contents/probe.py` (`_e2e_job`); retire `compile.probe_in_belief` | mock: T2 fires, second plan admitted; N_reask (ASSUMED A-S8 = 2) cap -> certified refusal; T3 counted | step 5; **plan step 2 "goal/success" switch**; Q2 |
| 7 | `same_worlds --plan-source codex` on v2 | `nero/contents/same_worlds.py` | records phenomena, handles, probes run and skipped, coverage row, re-asks, proposer record (bounds rule 9) | step 6; plan step 2 (method registry) |

Steps 1, 2 and 4 do not depend on the step-2 switch. Steps 3, 5, 6 and 7 do.

## 6. Questions

**Blocking** (the build stops at the named step without them):
- **Q1 (U2 + U6 + U8).** What does the checker assume for PH4 slosh, PH5 hidden loose item and PH6 visible loose item, which no built
  skill measures and no row covers today? (a) refuse while they are in Φ; (b) designer ranges, new ASSUMED bounds (μ_min 0.3, μ_i,min
  0.2, k_dyn 2); (c) out of Φ for now, stated as a limit (today's behaviour). Blocks step 5 enforce mode. Until then coverage runs in report mode.
- **Q2 (U7 + Rule 2 text).** Approve Rule 2 as T1 (a reading shows a phenomenon in Φ the plan did not name), T2 (no plan recovery is
  viable on Θ_t), T3 (a reading contradicts the earlier sound set: stop, counted)? And when the weigh returns only a lower bound:
  (a) T2 re-ask; (b) plain Rule 1 refusal, no re-ask (today); (c) T1. Blocks step 6.

**Defaults the build proceeds with** (each reversible by a switch or a data change):
- U1: build H. The switch default is "log" (= F). New names never add a row.
- U3: the weigh's own limits are checked by the checker before it runs. Moving skills are not built, so U3 waits.
- U4: worst-case read noise per sample (today, `probe.TAU_NOISE` 0.05 Nm ASSUMED). U5: no vision into the mugs.
- U7 (b)-(e): the re-ask shows the readings; it asks for a full new plan; "unmeasurable" then measured -> T2 only; re-ask only at rest.
- U10: cards show each skill's range and resolution (robot facts), never a contents range.
- New N1: numeric `covers` is removed (11-08: no contents numbers from the LLM); Rule 1 tries every recovery on all of Θ_t.
  Regime labels come later, with programs.
- New N2: the stack runs only the probes the LLM selects. An omitted built skill is recorded; its φ goes to its fallback.
- New N3: probe numbers stay conditional on the D8 tracker (calibrated under LeverCtl) and on the NERO torque read (hardware fact, open).

## 7. Verification

1. Unit tests per step (§5), run with
   `PYTHONPATH=. OMP_NUM_THREADS=1 ... timeout 1800 python -m pytest -q world_model/v2/stack/tests world_model/v2/nero/contents/tests`.
   Golden fixture (`stack/tests/golden/`) unchanged for admission, U and R.
2. Mock-LLM fixture run, offline, `BIMANUAL_NO_NETWORK=1`: `same_worlds --plan-source codex --plan-spec mock:fixture_v2`, n = 2 per task
   (box, tray6, pot). Mock variants: (i) names PH1 -> weigh; Θ_t equals today's probe; (ii) omits PH1 -> no probe, more refusals,
   no admission outside today's; (iii) wrong skill -> plan_rejected; (iv) NEW phenomenon -> watch list; (v) numeric field -> rejected.
3. Live Codex `gpt-5.6-sol:high` (approved 11-05), n = 2 worlds per task (box, tray6, pot), every request cached. Then rerun
   `--offline` from the cache and diff the records (must be identical). Report per task: Φ recall of the LLM's list, wrong-skill claims,
   plan_rejected count, probes run and skipped, coverage row, re-asks, outcome and safety. n = 2 gives examples, not rates.
