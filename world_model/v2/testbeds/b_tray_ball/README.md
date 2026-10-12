# b_tray_ball -- feasibility probe (ball on a rim-less tray / dustpan)

Simulation only. CPU only. No rendering, no hardware, no network. No existing repo file is edited.
Python: `/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python`. Run from the repo root. Pin threads
(`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMBA_NUM_THREADS=1`).

| file | what |
|---|---|
| `tray_env.py` | `TrayEnv`: MuJoCo rig, kinematic tray, `tick(u)` at 50 Hz, `snapshot()/restore()`, signed-distance margin. Imports `bead_rep.bead_env` (unedited) for the action meaning; `kin_step_fast` is a numba twin of `bead_env.kin_step`. |
| `analytic.py` | Two reduced models: `ne_rollout` (full Newton-Euler rolling ball, 9 numbers) and `qs_rollout` (quasi-static 5/7 model, 4 numbers). |
| `carry.py` | Documented synthetic nominal carry (min-jerk transport + noisy anti-roll tilt loop). |
| `probe.py` | `nominal` (failure rate) and `savable` (certified-savable set, lead time, model-vs-MuJoCo gap). |
| `authority.py` | Fine-lead and actuator-authority sweep with the QS model. |
| `hidden.py` | Hidden-parameter sensitivity and a safe-probe identifiability check. |
| `selftest.py` | Bitwise determinism, snapshot/restore, fast-vs-reference kin_step, speed. |

```
cd /common/users/ss5772/bimanual
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
$PY -m world_model.v2.testbeds.b_tray_ball.selftest                       # ~10 s, 1 core
$PY -m world_model.v2.testbeds.b_tray_ball.probe nominal --n 200 --workers 8   # ~10 s wall
$PY -m world_model.v2.testbeds.b_tray_ball.probe savable --eps 48 --scan 200 --workers 8   # ~3 min wall, ~24 CPU-min
$PY -m world_model.v2.testbeds.b_tray_ball.authority                      # ~4 min, 1 core
$PY -m world_model.v2.testbeds.b_tray_ball.hidden --workers 4             # ~12 s wall
```
Results go to `/common/users/ss5772/bimanual_data/testbeds/b_tray_ball/feasibility.json` (each stage merges its key).
Scratch: `.../b_tray_ball/tmp`. Report: `/common/users/ss5772/bimanual/reports/testbed_objects/b_tray_ball.md`.
