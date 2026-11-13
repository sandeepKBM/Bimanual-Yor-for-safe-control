# SafeTeleop HUD — browser teleoperation of a simulated YOR mobile bimanual robot

This branch (`feature/safeteleop-beta`) holds only the SafeTeleop application and the code it imports. The base is the
Assignment 4 (Alpha) snapshot `7439778`. Simulation only: nothing here connects to a real robot.

```
browser (index.html + app.js)          aiohttp server (teleop/web/server.py, 127.0.0.1, token + Origin check)
  keyboard / Xbox pad (Gamepad API) --->  InputHub (web/inputs.py): deadman, arm/disarm, Esc latch, 250 ms watchdog
  JPEG frames + JSON telemetry  <-----    SimWorker thread, 50 Hz tick (teleop/sim.py TeleopSim)
                                            base: Shield QP (teleop/shield.py) over operator twist -> safe twist
                                            arms: ObjectFrame jog -> IK refs -> FL-MPC (experiments/ab_smooth_vs_bangbang/mpc_fl.py,
                                                  rt kernels in experiments/realtime)
                                            ghost: teleop/ghost.py (0.8 s operator-intent preview, own MjData)
                                          render process (web/render_proc.py): shared-memory state snapshot -> MuJoCo
                                            renderer (ego / chase / top / wrist_l / wrist_r) -> JPEG
```

## Install

1. Python 3.12 and the pinned packages: `pip install -r requirements.txt` (torch is only needed by optional modules that
   this application does not load; the teleop stack needs mujoco, numpy, scipy, numba, aiohttp, opencv-python, Pillow).
2. Robot model: `git submodule update --init YOR` (YOR-robot/YOR, pinned commit). Only the MJCF meshes under
   `YOR/robot/yor-description` and `YOR/robot/arm/mujoco/assets` are read. `scenes/yor_robot_derived.mjcf` refers to them
   by paths relative to `scenes/`. Do not init YOR's own nested submodules (hardware drivers); they are not needed.
3. House assets (not redistributed here): MolmoSpaces / ProcTHOR-10k validation house 5 and the THOR object pack.
   Put them under `data/molmospaces/assets` (or set `HOUSE_DATA=<dir>`):
   ```
   $HOUSE_DATA/dl/val_shard0.tar               (val houses; val_5 is extracted on first use)
   $HOUSE_DATA/objects/thor/                   (THOR object XMLs + meshes)
   $HOUSE_DATA/../nav_cache/occ_5_0.010.npz    (built on first launch if missing, about 1 min)
   ```
   If the data already lives elsewhere: `mkdir -p data && ln -s <existing molmospaces dir> data/molmospaces`.
   `data/` is ignored by git; outputs of the headless tools also default to `data/...` (each has an env override).
4. Rendering: `MUJOCO_GL=egl` with a GPU (the default the server sets), or `MUJOCO_GL=osmesa` on CPU only
   (works, but measured 0.3 fps streaming on a loaded 72-core host; physics is unaffected).

## Launch

Run from the repository root:

```bash
export PYTHONPATH=$PWD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
MUJOCO_GL=egl python -m world_model.v2.house.teleop.web.server \
  --scene h5 --task carry --shield --fast --split --render-mode process
# prints:  open http://localhost:8765/?token=<token>
# remote host: ssh -L 8765:localhost:8765 <host>, then open that URL locally
```

The server listens on 127.0.0.1 only. The sim build took 98 s on the first launch and 29 s on a reset in our CPU-only
run (numba compile and house load); telemetry reports `ready: false` until it is done.

Useful flags: `--no-shield`, `--no-ghost`, `--view ego|chase`, `--people on|off|static`, `--footprint fixed|live`,
`--fallback brake|escape`, `--d-wall-door M`, `--task carry|coffee`, `--grasp-mode friction` (assisted friction grasp,
with `--spawn-at cafe_counter_pick`), `--lang off|auto|live|mock` (language panel; `live` needs a VLM service),
`--fps`, `--width/--height`, `--port`, `--token`, `--render-mode thread` (fallback without a render process).

## Controls

| Keyboard | Action | Keyboard | Action |
|---|---|---|---|
| W/S, A/D, Q/E | base forward/back, left/right, yaw | R / F | lift up / down |
| Arrows, PageUp/PageDown, T/Y | tray jog x/y, z, yaw | G / H | grasp / release |
| Enter | arm (clears the Esc latch) | Esc | latched stop (disarm) |
| Space | zero all motion | Backspace | reset episode (rebuilds the sim) |
| V | shield on/off | O | operator-intent ghost on/off |
| C | cycle camera | B | wrist inset on/off |
| N / M | tuck tray / normal carry | Z / X, 1–3 | speed x0.9 / x1.1, presets |

Xbox pad ("standard" mapping, press a button once after focusing the tab): hold LB (deadman); right stick = base
vx/vy, left stick X = yaw, D-pad up/down = lift; LB+RB = tray mode; Start = arm, Back = disarm, X = shield, Y = preset,
A/B = grasp/release, RS click = camera, LS click = inset, D-pad left/right = tuck/carry.

Every new connection starts disarmed. Keyboard keys are hold-to-move; key release, tab blur, pad unplug, socket close
and 250 ms without input all zero the command.

## HUD

Shield state and intervention fraction; clearances to walls and people; raw (`op`) vs filtered (`safe`) command; ghost
gap (requested vs permitted); cup state; real-time factor, tick time, stream fps, render time, round-trip and
input-to-ack latency. In the 3D view the ghost shows where the robot would be after 0.8 s under the raw command
(orange = the shield is changing it, cyan = no change), with requested (orange) and permitted (green) arrows.

## Tests

```bash
export PYTHONPATH=$PWD MUJOCO_GL=osmesa OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
python -m pytest -q world_model/v2/house/teleop/tests world_model/v2/house/teleop/web/tests
python -m pytest -q world_model/v2/house/{lang,grasp,ocs,pick,zono_rows,people_track}/tests
# end-to-end (needs a running server started with --token smoke123):
python -m world_model.v2.house.teleop.web.smoke_client --token smoke123 --out data/smoke
```

The smoke client connects over the WebSocket. It checks that motion is zero while disarmed, then drives, checks the
watchdog stop, the Esc latch and re-arming, and saves the last frame.

## What is in this branch

- `world_model/v2/house/teleop/` (+ `web/`, `tests/`): the application.
- Imported by it, kept unchanged: `house/{build,assets,phase1,lift,nav,...}.py`, `house/zono_rows`, `house/people_track`,
  `house/grasp`, `house/lang`, `house/ocs`, `house/pick`, `house/flagship` (perception helpers), `house/loc`,
  `house/office`, `house/zed_demo` (parts), FL-MPC (`experiments/ab_smooth_vs_bangbang`), the rt kernels
  (`experiments/realtime`), `control/`, `eval/{safety_module,scene_setup}.py`, `scripts/*task_utils.py`,
  `scripts/make_cup_collect_safety_demo.py`, `world_model/v2/{tray_rep,bead_rep,testbeds,qp_safety,latent_hj,
  perception_zono,vlm,wm_audit,data}` (only the imported files).
- `third_party/controller_core/`: four box-QP / filter modules from the author's controller repository (provenance in
  its `__init__.py`).

Some kept modules belong to older research code (the cup-collect demo, the tray study, an old world-model monitor).
They stay because the FL-MPC import chain loads them at import time (`experiments/realtime/_demo_env.py` imports
`scripts/make_cup_collect_safety_demo.py`, which imports `sim_common`, `run_ab4`, `mpc7`, the cup task utilities and
`control/realgrasp_flmpc`). Removing them would mean editing that chain. The world-model monitor (`world_model/*.py`)
needs `dreamerv3` + torch only when it is constructed; teleop never constructs it.

## Known limitations

- House assets are an external download (license: see MolmoSpaces); the scene list is `h5` only (`office` is a hook).
- No explicit Start → Active → Outcome episode state machine yet; reset is Backspace, outcome is the cup state.
- `--grasp-mode friction` assists only the tray grasp; `house/pick` (object proposals, selection API, pick skill) is not
  connected to the browser.
- One browser client at a time (a new connection takes over).
