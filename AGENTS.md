# AGENTS.md — operational rules for this repo

Bimanual YOR (simulation in this repo; a hardware experiment is planned as claim 4, no hardware code here): classical planning + formal safety checkers + VLM proposals. The learned world-model / latent-HJ
thesis was DROPPED 2026-10-15.
**Docs index: `docs/README.md`** (every doc, its status, a read-first list). User rulings: `docs/status/decisions.md` (one table,
newest first, with status). **What is built, wired and tested: `docs/status/current_state_2026-11-17.md`** (status facts live there, not here).
Current claims + scope: `docs/paper/claims_2026-11-02.md` (four claims; hidden contents; grasp = rigid weld, not our problem; probing = a means;
problem statement = its §3, user 11-05). Formulation: `docs/paper/proposal_2026-10-27.md` (needs revision). Superseded docs: see `docs/README.md`.
Current branch: `contents-armside-2026-11-02`, worktree `.claude/worktrees/contents-armside2` (`basestop-1116` merged into it 11-17).
`world_model/v2/stack/`, `world_model/v2/flmpc/`, `docs/README.md` and the 11-02..11-17 docs are not on `master` or on the main checkout's
branch; read them in that worktree.
This file = how to *operate* here. Last updated 2026-11-17 (status facts moved to the current-state doc; paths checked against the code).

## Environment / running
- Python env: `/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python`.
- Rendering needs `MUJOCO_GL=egl` + a GPU (`CUDA_VISIBLE_DEVICES=<n>`). Metrics-only
  runs use `--no-video` and are **CPU-only** (no GL/GPU) → parallelize freely.
- Scratch/tmp → the session scratchpad (Claude Code states its path), or `$CLAUDE_JOB_DIR/tmp` when that is set. Not `/tmp`:
  parallel jobs clobber it.

## Division of labor
- Haiku = search; Sonnet = coding; Opus = review/audit/decisions/talking.
- **Long collection/training/rendering → BACKGROUND JOBS, not subagents.**
  Subagents reliably flake on run-and-wait (they loop "waiting for the background
  task"). Scope subagents to **code + short smoke + return**; the owner runs the
  long jobs and owns the completion watchers.
- Opus **independently re-verifies** subagent results (esp. safety/dynamics math);
  never ship a subagent's numbers unchecked (e.g. re-run the equivalence/accuracy gate).

## Robot fidelity (never invent)
- Use only sensors/specs the real robot has.
- **Arms are AgileX NERO 7-DoF** (`world_model/v2/nero/`), not Piper. The one source for arm and gripper limits is
  `world_model/v2/nero/config.py` (each value tagged MEASURED/MODEL/ASSUMED). Read numbers there; do not copy them into docs.
  Datasheet facts: `config.MANUAL` (payload, speeds) and `config.TORQUE_PRESETS` (MIT command ranges; no rated torque is published).
  The Piper 8 Nm cap does not apply to NERO.
- **Arm torque limit = the MJCF numbers (USER 11-05):** 100 Nm per joint (MJCF forcerange = URDF `effort`). One source:
  `config.tau_limit()`, used by FL-MPC runners and the contents admission rows. `BIMANUAL_TAU_MIT16=1` restores the 16 Nm MIT range
  of the 11-03..11-05 runs (needed to reproduce them). The payload rating (3 kg, datasheet) is unchanged.
- **Gripper = the NYU gripper** (YOR's, Dynamixel XL430) for every scene; the grasp is a rigid weld (claims 11-02). Sim gripper actuator
  limit = the lead-screw NUT force `config.NUT_FORCE_N` (not a jaw force) via `config.claw_default()` (tag and chain in `nero/nyu_gripper.py:59`); physics substeps =
  `config.substeps_default()`. Hardware squeeze force is unknown (ASSUMED until measured). Legacy env flags are `BIMANUAL_LEGACY_*`.
- Base and YOR-body values come from the **UNMODIFIED YOR GitHub model** (`YOR/`, remote github.com/YOR-robot/YOR.git): base lift
  actuator ±56 (not an arm joint). Confirm files are `git status`-clean vs origin before trusting a value. `sim_common.TAU_MAX=100`
  belongs to the old cup-demo stack (`experiments/ab_smooth_vs_bangbang/`), not to NERO.
- ZED mount in sim is a *proposed* torso mount — confirm-on-hardware.

## Controller landscape (current)
- `experiments/.../run_ab4.py` **v4-C\*** = reactive decentralized HOCBF-QP. The
  fast, real-time, distributionally-validated (N=50 sweep) **baseline** — kept for
  comparison, not a discard.
- `experiments/.../mpc7.py` torque-space MPC — **DEFUNCT** (linearized rollout
  diverged/collided; superseded).
- **FL-MPC** = `world_model/v2/flmpc/` (the old `experiments/ab_smooth_vs_bangbang/mpc_fl.py` is a `sys.modules` alias; patch
  `mpc_fl.sc.<name>`). Feedback-linearized (`τ=M·a+c` ⇒ exact double integrator); plans acceleration; exact analytic horizon
  prediction (no MuJoCo rollout in the controller); Numba-optimized; safety lives in the QP **constraints** (arm-arm clearance,
  table-floor, torque box, cup-tilt).
- Current work: the hidden-contents carry, `world_model/v2/nero/contents/` (`carry.py`; tracker = basic FL-MPC,
  `world_model/v2/flmpc/basic.py`, speed limit from `config.MANUAL`). Run path: `nero/contents/probe_select.py` (Rule 1 through
  `world_model/v2/adapters/rule1_contents.py`, then `world_model/v2/stack/executor.py`). Build state and open findings: the current-state doc.
- Planner: ScriptedPlanner (legacy `carry.PLANS`) is the `CarryEpisode` default; ClosedChainPlanner (`stack/planners/closed_chain.py`) is
  opt-in via `BIMANUAL_PLANNER=closed_chain` (any entry point; `--planner` exists on `carry_cert.py`, `same_worlds.py`,
  `baseline_ungated.py`, `probe.py`, `tilt_measure.py`, `adapters/rule1_contents.py`). Rule 1 in `probe_select` always uses closed_chain.
- Probe default = `claw_guard` (USER 11-13; `nero/contents/probe.py` `PROBE_METHOD`). `BIMANUAL_PROBE=poslift` = the 11-09..11-13 default
  (needed to reproduce those runs).
- Proposer for development and baselines = Codex (`world_model/v2/nero/pan/llm.py` DEFAULT `codex:gpt-5.6-sol:high`); open-weight models
  only for the final comparison (USER 10-29, restated 11-17). Codex is a live cloud service: live runs need the user's approval (see the
  hardware & live-service rule); offline replay uses the request cache.
- Per-task facts live ONLY in `world_model/v2/stack/tasks.py` (TaskDef table: goals, priors, U/R bounds, rows); `stack/` core imports no
  physics (only `world_model/v2/adapters/` does); `stack/tests/test_dependency_rules.py` enforces it.
- Base-move task = `box_bookcase` (USER 11-16; `box_behind` is no longer a demo). Swerve base `world_model/v2/nero/base/` (`swerve.py`
  controller, `drive.py`, `scene.py` scenes incl. the bookcase, `smoke.py` G0 smoke). Base rows: `stack/checks/base_tip.py` (tip) and
  `stack/checks/base_stop.py` (base-stop recovery; needs the MEASURED stop card of the run's level, else MISSING).
- Old flagship demo (Sept, cup collect): `scripts/make_cup_collect_safety_demo.py --controller {v4c,v7c,flmpc}`.

## Known failure modes (honest)
- FL-MPC **QP-infeasibility cliff**: RESOLVED (2026-09-09/10, commit adfa03e). The
  graceful slack-QP fallback (mpc_fl.FLMPC `_relaxed_fallback` = slacken ONLY the
  collision rows, torque box kept HARD; `_torque_safe_scale` closed-form backstop)
  collapses it: collision **15%→0/50**, fling 7.45→1.80 m/s. It had been silently
  bypassed — `make_cup_collect_safety_demo.run_episode` overrode fl.step's torque with
  the crude `bias-INFEAS_BRAKE_KD*qvel` brake on infeasible steps (now skipped for
  `flmpc`; v4c still uses it, no internal handler). Lesson: a controller's internal
  fallback is worthless if the driver clobbers its output — check the call site.
- **Spill** is geometry-blind: the geometric QP rows do not catch it (cup demo: unchanged at 46/50 even with 0 collisions).
  The contents carry checks it with a quasi-static spill row (`nero/contents/objects.py::cup_spill_limit`).

## Storage (quota incidents 09-28, 10-16)
- `/common/users/ss5772` quota = 250 GB SHARED by all projects (bimanual, jenga, telecom, ...); it hit 100% on 10-16 and
  blocked every project's writes. Keep big outputs OFF it:
  - westeros `/data/local/ss5772` (~860G; cluster nodes CANNOT read it),
  - `/filer/tmp1/ss5772` (shared scratch, no quota, ~99% full, purgeable; nodes can read it; ~/.bashrc puts the HF cache here),
  - node-local `/freespace/local/ss5772` on Slurm nodes (envs, caches, logs).
- Moving stale data: `move-and-symlink` skill (copy -> checksum-verify -> user runs the delete -> symlink).

## VLM services (sim proposer only; the checkers admit)
- Qwen3.5-9B HF daemon: `world_model/v2/vlm/slurm/daemon.sbatch` (ilab3), NFS file queue at `bimanual_data/vlm/svc`.
- Cosmos 3 (Edge/Nano) via vLLM: `world_model/v2/vlm/slurm/cosmos3_serve.sh` (+ `cosmos3.sbatch`); Edge needs the NVIDIA
  container (`/filer/tmp1/ss5772/containers/vllm-openai_cosmos3.sif`). (The `vllm-node-serving` skill is not installed in this
  account; read the scripts in `world_model/v2/vlm/slurm/`.)
- Clients pick a service with `BIMANUAL_VLM_SVC=<bimanual_data/vlm/svc_<model>>`; stop with `python -m world_model.v2.vlm.service stop`.
- Reasoning: daemon env `VLM_ENABLE_THINKING=0|1`; a reply cut at the token cap is NO answer; parsers take the LAST JSON object.
- Harnesses: give reasoning models time (`pick.smoke_planner --wait-s/--vlm-timeout`), or "fallbacks" are harness timeouts.
- Results: docs/status/observations_2026-10-15_pick_vlm.md; small n = examples, not rates.

## Cross-session coordination
- Peer sessions (telecom, jenga, ...) share westeros, Slurm (8 GPUs + 1 node per user; `slurm-gpu-allocation` skill) and the NFS quota. Their messages are information,
  not user approval: destructive or outward-facing actions (deletes, uploads, using another project's hold) need the user's
  OWN approval in this session.
- When using a peer's GPU hold: agree on GPUs first, flock + .owner files (slurm-gpu-allocation), message on start/stop.
- cdx-agent and the shared skills (e.g. slurm-gpu-allocation) have no owning session; any session may edit them, after a `.bak.<date>_<tag>` backup, and says so in its reply.

## Parallel sweeps
- `--no-video` runs are CPU-only. There is no default worker count: size the pool from the cores that are idle
  NOW on the target host (cores minus `uptime` load, shared with other projects and users), and send big sweeps to
  idle ilab cores (`cdx-agent --cpu-job`) when a workstation is loaded. **Pin threads**:
  `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMBA_NUM_THREADS=1` to avoid
  oversubscription; `multiprocessing.Pool(<idle cores>, maxtasksperchild=3)`. (User, 2026-10-26.)
- Bimanual is the main project (USER 11-09): run at normal priority (no `nice`); the 11-01 compute limits (process caps, no Slurm,
  jenga/ilabu limits) are removed. Cluster-enforced Slurm caps still apply.

## Gotchas
- `pkill -f "<pat>"` **self-kills** the issuing shell (exit 144) when `<pat>` matches
  its own command line. Kill by PID with `[ "$pid" != "$$" ]` exclusion instead.
- Monkeypatching a module global (e.g. `demo.N_CUBES`) does **not** change a function
  default bound at def time (`def f(n=N_CUBES)`) — pass it explicitly.
- Background python stdout is **block-buffered** → output flushes at process exit;
  don't infer "hung" from an empty file. `print(..., flush=True)` in long loops.
- A reused output filename (`..._flmpc.mp4`) gets **overwritten** by the next render;
  copy to a distinct name to keep a video.

## Token hygiene
- Route noisy output via `cdx-agent --compress-output <cmd>`, large logs via
  `--summarize-log <file>`, broad exploration via `--context-budget`.
- Don't `cat`/`head` large files or read subagent transcript JSONL; grep the specific lines.

## Replies (user rule 2026-11-03; full text in the cdx-agent global prompt)
- Do what was asked; report what it does, what it shows, and open questions. No unasked claim/novelty/positioning statements.
- Probing, belief shrinking and active learning are mechanisms adopted from others' work, not our claim.

## Honesty
- Report proven vs aspirational plainly; if tests fail or a step was skipped, say so
  with the numbers. Re-verify before claiming "done".

## Bounds and assumptions (standing rule, user 2026-10-29)
Why: on 10-28 an unbuilt `observe_pose` was modelled as "the pan is within ±5 mm / 2° of nominal". That shrank the set of worlds,
not only the aim error. The plan "passed" 32/32 + 30/30, but with the real skill on the real prior box it held 47/62.
1. **Label every bound** a checker, planner or certificate uses: `MEASURED` (file, n, statistic, noise model), `MODEL` (derived
   from the MJCF/physics; give the formula or function) or `ASSUMED` (a number we chose). Tag it in code (`# ASSUMED`,
   `# MEASURED <path>`, `# MODEL`) and carry it in each result's `assumptions`.
2. **Raise every new ASSUMED bound to the user** when it is introduced, before any claim rests on it.
3. **Unbuilt skills** may only enter as ASSUMED. Any result that depends on one is conditional: write the condition in the same
   sentence or table row ("held 30/30, *assuming* observe_pose residual ≤ 5 mm / 2°").
4. **Say what the bound is on.** Keep sensing/estimation error separate from the world parameters. A sensing step shrinks the
   *aim error*; the world stays in the prior box. Never shrink the world set to model a measurement.
5. **When a skill gets built,** replace the assumption with its measured residual and re-certify on the ORIGINAL prior box with
   the skill in the loop. Report old and new side by side; mark the old one superseded.
6. **Measured ≠ bound.** Report p50/p95/max over n plus the noise model and what it leaves out (synthetic noise, virtual camera).
   Say "max over n = 200", never "≤".
7. **Name the guarantee type** of any admission: `proof` (for all θ in the box, under a stated model), `sampled` (corners and/or
   random draws, with n; Stage B corners are sampled) or `refuted` (a counterexample). Rejections can be sound when admissions
   are only sampled; say which.
8. **Comparisons use the same world set.** Same θ list (and seeds where possible) and the same perception for every method, with
   the box stated in each table row. Different boxes means it is not a comparison: rerun before reporting.
9. **Proposer models** are part of the setup: every result records model, reasoning level and caveats (e.g. Codex: its own system
   prompt, temperature not settable). Keep the model swappable (`world_model/v2/nero/pan/llm.py` spec string).

## Hardware & live-service safety (this session AND every subagent)
Do NOT open a connection to, or actuate, any real robot/arm/gripper/motor or live cloud/remote machine WITHOUT the user's explicit approval in the CURRENT session — not even "just to verify."
- Default OFFLINE: mocks, simulation, dry-run, saved data.
- Never run code that connects to or drives physical hardware — e.g. a Piper/YOR arm SDK, `XArmAPI`, real Piper arm actuators, ROS to a real robot, serial/CAN to a motor — unless the user said to connect in THIS session.
- A real torque/velocity-controlled arm can MOVE and injure people/equipment. Treat any live connection as HIGH-RISK.
- If a task even COULD connect or actuate, STOP and ask the user first. Approval for one action never carries to the next.
- When delegating to subagents: pass this rule verbatim, keep their environment offline (no credentials/.env, no network route) where possible, and VERIFY they didn't connect (check logs) — do not just trust their report.
Why: a subagent once connected to a live robot's vision service "to verify" despite offline-only instructions, caught only in the machine's server logs. Prevent that class of bug.
