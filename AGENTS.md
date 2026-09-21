# AGENTS.md — operational rules for this repo

Bimanual YOR: cooperative arm-arm safety + learned world model. Design source of
truth: `docs/design/MASTER.md` (+ `docs/design/mpc_formulation.md`). This file =
how to *operate* here. Last updated 2026-09-08.

## Environment / running
- Python env: `/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python`.
- Rendering needs `MUJOCO_GL=egl` + a GPU (`CUDA_VISIBLE_DEVICES=<n>`). Metrics-only
  runs use `--no-video` and are **CPU-only** (no GL/GPU) → parallelize freely.
- Scratch/tmp → `$CLAUDE_JOB_DIR/tmp` (not `/tmp`; parallel jobs clobber `/tmp`).

## Division of labor
- Haiku = search; Sonnet = coding; Opus = review/audit/decisions/talking.
- **Long collection/training/rendering → BACKGROUND JOBS, not subagents.**
  Subagents reliably flake on run-and-wait (they loop "waiting for the background
  task"). Scope subagents to **code + short smoke + return**; the owner runs the
  long jobs and owns the completion watchers.
- Opus **independently re-verifies** subagent results (esp. safety/dynamics math);
  never ship a subagent's numbers unchecked (e.g. re-run the equivalence/accuracy gate).

## Robot fidelity (never invent)
- Use only sensors/specs the real robot has. Robot limits come from the
  **UNMODIFIED YOR GitHub model** (`YOR/`, remote github.com/YOR-robot/YOR.git):
  arm joints `effort=100` / MJCF piper-class `forcerange ±100` → `sim_common.TAU_MAX=100 Nm`;
  fingers ±20; base lift actuator ±56 (not an arm joint). Confirm files are
  `git status`-clean vs origin before trusting a value.
- ZED mount in sim is a *proposed* torso mount — confirm-on-hardware.

## Controller landscape (current)
- `experiments/.../run_ab4.py` **v4-C\*** = reactive decentralized HOCBF-QP. The
  fast, real-time, distributionally-validated (N=50 sweep) **baseline** — kept for
  comparison, not a discard.
- `experiments/.../mpc7.py` torque-space MPC — **DEFUNCT** (linearized rollout
  diverged/collided; superseded).
- `experiments/.../mpc_fl.py` **FL-MPC** = the solution. Feedback-linearized
  (`τ=M·a+c` ⇒ exact double integrator); plans acceleration; exact analytic horizon
  prediction (no MuJoCo rollout in the controller); Numba-optimized; safety lives in
  the QP **constraints** (arm-arm clearance, table-floor, torque box, cup-tilt).
- Flagship demo: `scripts/make_cup_collect_safety_demo.py --controller {v4c,v7c,flmpc}`.

## Known failure modes (honest)
- FL-MPC **QP-infeasibility cliff**: RESOLVED (2026-09-09/10, commit adfa03e). The
  graceful slack-QP fallback (mpc_fl.FLMPC `_relaxed_fallback` = slacken ONLY the
  collision rows, torque box kept HARD; `_torque_safe_scale` closed-form backstop)
  collapses it: collision **15%→0/50**, fling 7.45→1.80 m/s. It had been silently
  bypassed — `make_cup_collect_safety_demo.run_episode` overrode fl.step's torque with
  the crude `bias-INFEAS_BRAKE_KD*qvel` brake on infeasible steps (now skipped for
  `flmpc`; v4c still uses it, no internal handler). Lesson: a controller's internal
  fallback is worthless if the driver clobbers its output — check the call site.
- **Spill** is geometry-blind → the WM/latent-value shield's job, never the geometric QP
  (unchanged at 46/50 even with 0 collisions).

## Parallel sweeps
- `--no-video` runs are CPU-only. Use ~78% of cores (72→56 workers). **Pin threads**:
  `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMBA_NUM_THREADS=1` to avoid
  oversubscription. `multiprocessing.Pool(56, maxtasksperchild=3)`.

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

## Honesty
- Report proven vs aspirational plainly; if tests fail or a step was skipped, say so
  with the numbers. Re-verify before claiming "done".

## Hardware & live-service safety (this session AND every subagent)
Do NOT open a connection to, or actuate, any real robot/arm/gripper/motor or live cloud/remote machine WITHOUT the user's explicit approval in the CURRENT session — not even "just to verify."
- Default OFFLINE: mocks, simulation, dry-run, saved data.
- Never run code that connects to or drives physical hardware — e.g. a Piper/YOR arm SDK, `XArmAPI`, real Piper arm actuators, ROS to a real robot, serial/CAN to a motor — unless the user said to connect in THIS session.
- A real torque/velocity-controlled arm can MOVE and injure people/equipment. Treat any live connection as HIGH-RISK.
- If a task even COULD connect or actuate, STOP and ask the user first. Approval for one action never carries to the next.
- When delegating to subagents: pass this rule verbatim, keep their environment offline (no credentials/.env, no network route) where possible, and VERIFY they didn't connect (check logs) — do not just trust their report.
Why: a subagent once connected to a live robot's vision service "to verify" despite offline-only instructions, caught only in the machine's server logs. Prevent that class of bug.
