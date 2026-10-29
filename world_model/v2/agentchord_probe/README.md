# agentchord_probe: stress-testing AgentChord's plans

Plan: `/common/home/ss5772/.claude/plans/frolicking-forging-treehouse.md` (approved 2026-10-27).
Target: AgentChord (Xu et al., arXiv 2605.11951, RSS 2026), code `github.com/EDEM-AI/AgentChord`, Apache-2.0.

## Layout

| What | Where |
|---|---|
| their code (unedited) | `/data/local/ss5772/agentchord/src`, commit `74fe7dd756964b737952d26e9e65d4d5e2e29ec3` (2026-05-15) |
| container | `/data/local/ss5772/agentchord/containers/embodichain.sif`, from `docker://dexforce/embodichain:ubuntu22.04-cuda12.8` |
| assets | `/data/local/ss5772/agentchord/data`; fetched from `huggingface.co/datasets/DexForceAI/embodichain_data`, not their `hf-mirror.com` default |
| runs | `/data/local/ss5772/agentchord/runs/<run>/` (`meta.json`, `episodes.jsonl`, `traj_*.npy`, `run.log`, `gateway.log`, `hw_audit.txt`) |
| plans and LLM log | `/data/local/ss5772/agentchord/runs/plans/<plan>/` (their 3 JSONs + `llm_log.jsonl`) |

## Language-model backends

All backends go through `gateway.py`, a local OpenAI-compatible endpoint. Their `LLM_URL` points at it.

| Backend | Model | Caveats (stored on every log row) |
|---|---|---|
| `codex` | `gpt-5.6-terra` (user choice 10-27). `gpt-5.5` and `gpt-5` are rejected for Codex on a ChatGPT account. In Codex's model list, sol is the "workhorse" (closest to the paper's GPT-5), terra is "balanced" and luna is "fast". | Codex adds its own agent system prompt; temperature is not settable; ChatGPT plan rate limits |
| `qwen-jenga` (**DROPPED by the user 10-27**: text only, so the planner would get no image; kept in code, unused) | `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` on jenga's live vLLM (ilab3:8781, via `ssh -N -L 18781:127.0.0.1:8781 ilab3`) | **text only, so the camera image is replaced by a note**; 16k context; output ≤ 2048 tokens; 1 request in flight; pause during TOWER evals; available until about 10-31 |
| `replay` | answers from `llm_log.jsonl` | none |

## Hardware safety

The AGENTS.md hardware rule applies. Their repo includes a real-robot path (`embodichain/deploy/run_agent_realworld.py`: ROS 2,
Piper, CAN), which we never run:
- `ac_run.py` blocks imports of `embodichain.deploy`, ROS, CAN and robot SDKs;
- the container runs `--containall`, with no home directory, credentials or devices beyond the GPU;
- `launch.sh` greps every run log for hardware traces and writes the result to `hw_audit.txt`.

## Setup fixes (2026-10-28)

| Problem | Fix |
|---|---|
| The image has no simulator | DexForce wheel `dexsim_engine` 0.3.11, installed into a venv on top of the image's py310 (sha256 checked; licence proprietary, so not redistributed) |
| 0-byte NVIDIA library placeholders in the image | host driver libraries put first on `LD_LIBRARY_PATH` |
| `pytorch_kinematics` crashes on `--device cuda` (their `SerialChain` call passes no device, and pk 0.10.0's `Chain.__init__` is not device-clean on CUDA) | `ac_run.patch_pk_device()` deep-copies the chain to CPU, builds the sub-chain there, then `.to(device)` |
| GPU lock still held after a run (the watchdog's `sleep` inherited the lock fd) | `launch.sh` is called with `9>&-` in `smoke_noop.sh` and `e0_grid.sh` |
| tensordict pulled in torch 2.14 | tensordict pinned to 0.8.3, so the image's torch 2.7.0+cu128 stays |
| cvxpy 1.4 rejects ortools 9.14 (GLOP and PDLP disabled) | ortools pinned to 9.7.2996 |
| texture sets `DexsimMaterials/{WoodTable,Plate,Spoon,Bottle,Cup}` never published | **substitutes** chosen by category from DexForce's published `materials/BackgroundTexture.zip`; see `data/DexsimMaterials/PROVENANCE.json`. These are not the authors' textures, so the planner's camera images differ from theirs. Checked 10-28 (`texchk2_*/agent_obs_*.png`): textures render; in single-pour, cup and bottle are both blue but differ clearly in shape. |
| LeRobot recorder | `--filter_dataset_saving` (their own flag) |
| `./outputs` inside the read-only clone | runs use a writable working folder that links to the clone |

## GPU isolation (incident 2026-10-28)

DexSim renders through Vulkan, which ignores `CUDA_VISIBLE_DEVICES`. With `--nv` and `CUDA_VISIBLE_DEVICES=6`, every simulator
process ran on physical GPU 0 (jenga's fallback server GPU; jenga confirmed no harm).

`launch.sh` now:
- runs with `--nvccli` and `NVIDIA_VISIBLE_DEVICES=$GPU`, so only `/dev/nvidia$GPU` exists inside the container;
- runs a watchdog that SIGKILLs any of the run's processes that shows up on another GPU UUID (logged in `gpu_watchdog.txt`).

Before scaling up, check placement with `nvidia-smi --query-compute-apps=gpu_uuid,pid`.

## What their code actually does (read 10-27; differs from the paper)

1. **Monitors** fire on a single step: `monitor_object_moved` compares the pose with the previous step's pose (> 1–2 cm jump), and
   `monitor_object_held` checks whether the gripper is closed below a threshold. Only these two monitor functions exist. There is no
   K-step window (paper Eq. 12). A slow drift of under 1–2 cm per step is invisible by construction.
2. **Pour success** (`pour_water.py::is_task_success`) only checks that neither the bottle nor the cup is tipped ≥ 45°. Not checked:
   the bottle reaching the cup, pouring, the return pose. There is no water in the sim.
3. **Task prompts are scripted**: they give the step order and the numeric offsets and targets, so the model mostly turns given steps
   into atomic-action calls.
4. **Retry loop**: `run_env.generate_function` re-runs an episode that failed, until one succeeds. `ac_run.py` runs each episode once.
5. **No recovery branch for a fired monitor** gives a `RuntimeError`. The graph allows at most 1000 transitions.
6. **Error injection** is interactive only (the `f` key) and teleports the object (`set_local_pose`).
7. **The shipped example plans are stale** (verified 10-28). They were committed on 05-12 (`bfde624`). On 05-14, commit `95a0884`
   changed `hold_lost` from "arm–object distance > thr" (default 0.05) to "gripper opening < thr" (default 0.01). The examples kept 0.05;
   the compiled Rearrangement graph also has the old 0.05 default. In the sim the gripper opening is 0.05 when open and 0.0297 when
   holding the bottle, so the monitor fires right after every successful grasp. The result is a regrasp loop, then an IK error:
   SinglePourWater fails 3/3 at p = 0 (`gripchk_single_paper_cpu`). E0 therefore uses `paperfix` (`runs/plans/e0_<task>_paperfix`,
   `PATCH.json`), which changes only those thresholds, to 0.01 (what the current compiler fills in).
8. **CPU only**: `atom_actions.grasp` calls `np.arctan2` on torch tensors, so `--device cuda` crashes. CPU is also their CLI default.

## Pre-registered for E0 (written before any sim run)

- **Protocol** (our reading of "with probability p, an object held by the end-effector is randomly dropped at each atomic action"):
  - "held" means lifted more than 3 cm above its initial height;
  - the drop is at a uniformly random step of the action;
  - the default mode is `upright` (teleported down to its initial height); the paper does not say which mode it used, so `fallen`
    (their `fallen_object`) is tested in E2.
- **Strict pour success:**
  - at some logged step the bottle is within 3 cm of cup + [0.05, −0.10, 0.125], tilted at least 60°;
  - the bottle ends within 3 cm (xy) of [0.7, −0.1];
  - bottle and cup end tipped less than 45°;
  - DualPourWater adds: the cup ends within 3 cm of [0.6, 0.1].
- **Do-nothing control** (`--noop`): reset, then score. Expected: their success = true, strict success = false.
- **Backend:** GPT only (Codex `gpt-5.6-terra`, with the camera image passed by `-i`; checked 10-27). Qwen dropped.
- **E0 pass:** with the GPT backend, their success is within ±3 trials out of 20 of paper Table I (p = 0.05 and 0.10, 3 tasks).
- **Kill:** if their success is below 70% at p = 0, the setup is broken; debug it before E1–E3.

## Coffee tray: the paper's weakest bimanual task (added 10-28, user choice; written before any coffee-tray run)

On their 3 sim tasks AgentChord is at 95–100%, so that is not where it can be beaten. The weak tasks are real-robot only (Table II):
- fold towel 55% (deformable; out of our scope);
- **setup coffee tray 60%**;
- dual-arm pour 80%;
- handover 85%.

Coffee tray is a two-arm carry of a tray holding a payload, which is our area. Its failures (slow slide, one-hand slip, tilt, payload)
are below or outside what their monitors measure: a single-step 1–2 cm pose jump, and gripper opening. Their release has no assets,
env or prompt for it, so we built it in **their** sim (`coffee_tray/`) and leave their code unedited:
- **Steps** (paper appendix): grasp capsule (L), place it into the tray (L), grasp tray (LR), place tray forward (LR).
- **Prompt** (`task_prompt.txt`): written by us, in the style of their sim prompts (scripted steps and numeric offsets).
- **Geometry and masses** (`build.py`): our choice, because the paper gives none.
  - tray 0.22 × 0.34 m with 3 cm rims, 0.30 kg; capsule r = 1.9 cm, h = 2.8 cm, 12 g;
  - their table, robot, cameras and randomization style (from their rearrangement config).
- **Two-arm tray grasp:** their `grasp()` reads one grasp pose per object. `ac_run.patch_tray_grasp` sets the calling arm's rim
  pose (top-down, their rotation) before calling it. Nothing else of theirs is changed.
- **Success** (paper criterion made numeric; thresholds are ours):
  - capsule inside the tray's inner footprint and within 4 cm of it in z;
  - tray within 3 cm (xy) of its start position + [0.10, 0];
  - tray tilt < 10°.
- **Drops:** `--held-rule grip`, meaning a gripper is closed (< 0.045) and the object is raised ≥ 0.5 cm. This is a change from the
  pre-registered "lifted > 3 cm" rule, which never fires in rearrange (fork and spoon rise only ~1 cm; 0/6 drops applied in
  `e0d_pf_Rearrangement_*`). E0 pour runs keep the pre-registered rule; rearrange p > 0 is rerun with `grip`.

## Files

- `gateway.py`: the OpenAI-compatible relay (backends, JSONL log, replay).
- `ac_run.py`: runs inside the container (import block, scripted injector, per-step pose log, strict success).
- `launch.sh`: starts the gateway, then the container, then the hardware audit.
- `tests/test_gateway.py`: offline tests (fake backend, replay, token cap, image strip).
