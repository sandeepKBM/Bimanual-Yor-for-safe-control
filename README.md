# Bimanual YOR — safe cooperative bimanual control (simulation)

A MuJoCo simulation of the **YOR bimanual mobile manipulator** carrying a tray with a cup of beads through
procedurally generated houses (MolmoSpaces / ProcTHOR) under sloppy human teleoperation, with a real-time
**safety shield** in the loop:

- **Arms:** feedback-linearized MPC (FL-MPC, `τ = M·a + c`) at 200 Hz. Arm–arm clearance, torque box, tray-tilt cone and
  zonotope obstacle rows (higher-order CBFs) are hard QP constraints.
- **Base:** a 50 Hz 3-variable QP (`vx, vy, ω`). It enforces a friction/jerk no-spill budget, wall clearance (distance field),
  predictive person rows and zonotope control-barrier rows, and falls back to lift → peel → brake.
- **Perception:** stereo depth (SGBM or FoundationStereo) is turned into point clouds, then zonotopes, then an
  **object-centric world state** with calibrated (conformal) uncertainty. People are tracked with a depth or neural detector
  plus a Kalman tracker, and the base pose comes from an odometry/VIO estimator.
- **Language:** an on-prem VLM (Qwen3.5-9B) grounds object names, authors constraints in a small set language and gives a
  friction prior. That prior can only **widen** the engineered interval, never narrow it.
- **Planning:** an optional GPU MPPI planner proposes base motions above the hard filters.

**Simulation only.** Nothing here connects to or drives real hardware. `YOR/robot/base.py` and the YOR hardware
extras are never imported.

---

## System requirements

| | Tested | Notes |
|---|---|---|
| OS | Ubuntu 22.04.5, Linux 5.15, x86_64 | Any recent Linux should work; macOS/Windows untested |
| Python | 3.12.13 (main env) | 3.11 for the optional FoundationStereo/Pose and detector envs |
| GPU | NVIDIA RTX 2080 Ti (11 GB), driver 580.173.02 (CUDA 13.0) | Needed for rendering (EGL), video and the GPU perception/planner. Metrics-only runs (`--no-video`) are CPU-only. |
| VLM (optional) | ≥ ~20 GB VRAM GPU (ran on a Slurm node) | Only for the language features |
| Disk | ~3.2 GB house assets + run outputs (a full flagship run with video is ~350 MB) | Keep outputs off small home quotas |
| Tools | `git`, `conda`/`mamba` | `ffmpeg` comes with `imageio-ffmpeg` |

A real-time factor below 1 is normal on this hardware. The browser teleop measured RTF ≈ 0.67 at 15 fps.

---

## Installation

### 1. Clone (with the YOR robot model submodule)

```bash
git clone --recursive https://github.com/sandeepKBM/Bimanual-Yor-for-safe-control.git bimanual
cd bimanual
# already cloned without --recursive:
git submodule update --init
```

`YOR/` is the **unmodified** upstream robot description (github.com/YOR-robot/YOR). Every robot limit (joint
torques ±100 N·m, fingers ±20, lift ±56) is read from it. Do not edit it.

### 2. Main Python environment

```bash
conda create -n mujoco_ur5e python=3.12 -y
conda activate mujoco_ur5e
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130   # CUDA build; pick the index matching your driver
pip install -r requirements.txt
```

### 3. Make paths match your machine

The code was developed in `/common/users/ss5772/bimanual`, and some absolute paths are still baked in:

```bash
# robot / scene MJCFs use an absolute meshdir:
sed -i "s|/common/users/ss5772/bimanual|$PWD|g" scenes/yor_robot_derived.mjcf scenes/bimanual_piper_pickup.xml
```

- Output and data defaults point to `/data/local/ss5772/...`. Pass `--out <dir>` (most entry points accept it) or set
  the env vars the modules read (`HOUSE_DATA`, `OCS_OUT`, `ZONO_ROWS_OUT`, `FOUNDATION_OUT`, `PEOPLE_SHIELD_OUT`, …).
- Some modules put scratch files under `$CLAUDE_JOB_DIR/tmp`. Set `CLAUDE_JOB_DIR` to any writable scratch directory.
- Not every path is configurable yet. If a module fails on a missing `/data/local/ss5772/...` path, that path needs
  overriding.

### 4. Per-shell environment

```bash
export PYTHONPATH=$PWD:$PWD/experiments/ab_smooth_vs_bangbang:$PWD/experiments/realtime
export MUJOCO_GL=egl                      # headless GPU rendering
export CUDA_VISIBLE_DEVICES=0             # a free GPU
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1   # avoid oversubscription
```

`experiments/ab_smooth_vs_bangbang` (FL-MPC, `sim_common.py`) and `experiments/realtime` (the exact real-time QP used
by `--flmpc-impl rt`) must be on `PYTHONPATH`.

### 5. House assets (MolmoSpaces — not redistributed here)

The houses come from **MolmoSpaces** (AI2, `github.com/allenai/molmospaces`): procthor-10k-val scenes plus the THOR
object packs. Its code is Apache-2.0 and its data ODC-BY / CC-BY-4.0; the THOR object pack carries its own licence.
The assets are **not** in this repo. Download them following the MolmoSpaces instructions and lay them out under `$HOUSE_DATA`:

```
$HOUSE_DATA/                              (default /data/local/ss5772/molmospaces/assets, ~3.2 GB)
  dl/val_shard0.tar                       procthor-10k-val shard (zstd members; houses are extracted on demand)
  dl/thor_*.tar.zst                       THOR object packs
  objects/thor/                           extracted THOR objects
  scenes/procthor-10k-val/                extracted houses (created by the code)
```

Optional: an articulated **Rocketbox** human (MIT; MolmoSpaces `humans_rocketbox_articulated`) for the visual-only
walker meshes.

### 6. Check the install

```bash
python -m world_model.v2.house.selftest phase0 --ids 0 1 2     # compile + step + ray-cast 3 houses
python -m pytest -q world_model/v2/house                       # unit tests for the house stack
```

---

## Quick start

### Browser teleop (keyboard / Xbox pad)

```bash
python -m world_model.v2.house.teleop.web.server --scene h5 --task carry --shield --fast --split --render-mode process
# prints:  open http://localhost:8765/?token=<token>
```

The server listens on `127.0.0.1` only. From a laptop, either:
- forward the port with an SSH tunnel: `ssh -L 8765:localhost:8765 <server>`. Any free local port works, e.g. `-L 18765:localhost:8765`.
- or use VS Code Remote-SSH: Ports → Forward a Port → `8765`.

Then open the printed URL (with `?token=`) in a normal browser and press **Enter** to arm.

| Keys | Action | Keys | Action |
|---|---|---|---|
| W / S | base forward / back | A / D | base left / right |
| Q / E | turn left / right | R / F | lift up / down |
| Arrows | tray x / y | PgUp / PgDn | tray z |
| T / Y | tray yaw | G / H | grasp / release |
| 1 / 2 / 3 | speed preset | V | shield on / off |
| C | cycle camera | B | wrist inset |
| Space | zero all motion | Esc | latched stop (Enter clears) |
| Backspace | reset episode | | |

Xbox pad: hold **LB** as the deadman. The right stick drives the base, the left stick X turns, and LB+RB switches to tray mode.

Useful flags:
- `--task coffee` and `--scene office` change the task and scene.
- `--no-shield` starts with the shield off.
- `--lang live|mock` turns on the language features.
- `--port`, `--fps`, `--width/--height` set the port and the stream rate and size.

### Headless teleop episode (raw vs shielded, same simulated operator)

```bash
python -m world_model.v2.house.teleop.run --mode both --task carry --seed 0 --max-s 30 --out runs/teleop
python -m world_model.v2.house.teleop.run --mode shield --task coffee --max-s 110 --video --out runs/teleop
```

### Flagship coffee run (record both sides, then render the side-by-side video)

```bash
python -m world_model.v2.house.flagship.simulate --side raw    --seed 0 --perception off  --out runs/flagship
python -m world_model.v2.house.flagship.simulate --side shield --seed 0 --perception live --out runs/flagship
python -m world_model.v2.house.flagship.render   --seed 0 --dir runs/flagship --out runs/flagship/flagship_coffee_run.mp4   # 1920x1080, 25 fps
```

`--perception cached` re-uses the saved perception pickles and runs CPU-only. Always give a **new** `--out` directory
for live perception so earlier caches are not overwritten.

---

## Optional components (separate environments)

These run as separate server processes. The main env talks to them over Unix sockets, shared memory or a file queue,
and each one warms up before the run starts.

**FoundationStereo + FoundationPose (model-free)** — `world_model/v2/house/foundation/`
- Env: Python 3.11 (conda-forge `cmake ninja eigen boost pybind11`), `torch==2.8.0` and `torchvision==0.23.0` from the
  PyTorch cu128 index.
- Python packages: `timm==1.0.30 omegaconf==2.4.0 kornia==0.8.3 onnx onnxruntime onnxscript tensorrt-cu12==10.13.3.9`
  `opencv-python trimesh xatlas rtree diffrp-nvdiffrast==0.3.3.1`.
- Code and weights: clone `NVlabs/FoundationStereo` and `NVlabs/FoundationPose` and download the weights from their READMEs.
  The server scripts expect them in `/data/local/ss5772/models/{FoundationStereo,FoundationPose}` (edit `FS`/`FP` in
  `stereo_fs_server.py` / `pose_fp_server.py`).
- Build notes:
  - CUDA toolkit 12.x `nvcc` with system gcc 11, and `TORCH_CUDA_ARCH_LIST` set for your GPU.
  - FoundationPose's `bundlesdf/mycuda` needs `-std=c++17`, and `.type()` changed to `.scalar_type()`.
  - `pytorch3d` and `kaolin` are not installed: a `pytorch3d.transforms` shim plus `dense_octree.py` replace them.
- On an 11 GB card, FoundationPose needs ≤ 640×360 inputs.

**Neural person detector** — `world_model/v2/house/people_track/detect_server.py`
- Env: Python 3.11 with `torch==2.8.0` and `torchvision==0.23.0` (cu128). Uses torchvision Faster R-CNN (BSD) weights
  cached in `$TORCH_HOME`.
- Start: `TORCH_HOME=<dir> <detect-env>/bin/python world_model/v2/house/people_track/detect_server.py --sock <path> --warm 20`

**VLM service** — `world_model/v2/vlm/`
- Qwen3.5-9B through `transformers` (`AutoModelForImageTextToText`). Weights live on node-local disk and run offline
  (`HF_HUB_OFFLINE=1`). Needs ~20 GB VRAM.
- `vlm/slurm/daemon.sbatch` runs `vlm/daemon.py` as a long-lived Slurm job that polls a shared-filesystem request queue.
  Edit the node, env and queue paths in the script.
- Client: `python -m world_model.v2.vlm.service status | ask "<question>" | stop`

---

## Repository map

```
world_model/v2/house/        current system: houses + teleop + shield + perception
  teleop/                    TeleopSim (50 Hz tick = 4 physics steps), base Shield QP, ArmCtl → FL-MPC, browser UI (web/)
  zono_rows/                 zonotope faces → CBF rows for the base filter and the FL-MPC arm rows (numba)
  flagship/                  scripted coffee-run mission, perception service, renderer of the comparison video
  stereo/  foundation/       stereo twin cameras, SGBM / FoundationStereo backends, FoundationPose
  gpusense/  loc/            GPU depth / distance field; odometry/VIO pose estimator + log-odds mapper
  ocs/                       object-centric WorldState (objects, robot, free space, provenance, uncertainty sets)
  people_track/              person detectors, Kalman tracker, conformal margins, people shield
  grasp/                     wrist-camera friction grasp, place, proprioceptive payload estimate
  lang/                      VLM grounding, constraint-spec authoring, friction prior (widen-only)
  mppi/                      GPU MPPI planner (CUDA graph, CVaR over friction)
  office/  zed_demo/         office scene + route; earlier ZED perceive→plan→execute demo
world_model/v2/tray_rep/     tray + beads env; sets/ = zonotope fit, constraint IR language, QP filter, reachability
world_model/v2/vlm/          VLM runner, prompts, grounding, daemon + client
world_model/v2/*             other research packages (latent HJ, world-model audits, testbeds, …)
experiments/ab_smooth_vs_bangbang/   FL-MPC (mpc_fl.py), v4c HOCBF baseline, sim_common.py (on PYTHONPATH)
experiments/realtime/        exact real-time QP + numba barriers for FL-MPC (on PYTHONPATH)
control/ collection/ eval/ scripts/ hj_reachability/ teleop/ world_model/*.py
                             earlier tabletop bimanual work (cup-collect task, data collection, HJ reachability)
scenes/                      MJCF / XML scenes, YOR derived model (camera mount = proposed torso ZED mount)
YOR/                         upstream robot model (git submodule, unmodified)
```

## Notes

- **Robot fidelity:** limits come only from the unmodified YOR model. The sim's torso ZED mount is a proposal, not
  measured hardware.
- **Parallel sweeps:** `--no-video` runs are CPU-only. Pin threads (see step 4) and use a process pool.
- **Licences:** third-party components (YOR, MolmoSpaces assets, FoundationStereo/FoundationPose, Rocketbox,
  Qwen weights) keep their own licences. Check each upstream before reuse or redistribution.
