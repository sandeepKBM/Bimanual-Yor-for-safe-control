# flagship -- teleoperated coffee run, shield OFF vs ON, stereo zonotopes (SIMULATION ONLY)

Offline MuJoCo only.  No robot driver, no `YOR/robot/base.py`, no `YOR/extra/*`, no ZED SDK, no ROS, no network.  New files only; every existing module
(`office/`, `teleop/`, `zed_demo/`, `lift.py`) is imported read-only (two runtime attribute patches, both in this package: `teleop.scene.TRAY_HOLD_Z/COUNTER_TOP_Z`
while the rig is built, and `zed_demo.stereo.error_at_range` made robust to empty range bins).

| file | what |
|---|---|
| `scene.py` | flagship world: office + desk clutter (zed_demo) + counter extras (milk jug, sugar jar, napkin box, cup stack) + serving-ledge plate + placement stand; `new_sim`, `build_twin`, `set_stand` |
| `mission.py` | `CoffeeRunOperator` (state machine on top of `SloppyOperator`), `PathFollower`, `NullServices` |
| `perception.py` | `Twin` (speckle-textured twin + ZED stereo + SGBM + zonotopes + planners), `PerceptionService` (live / cached), place-planner switch |
| `simulate.py` | one side headless (CPU physics) -> `run_<side>_s<seed>.npz/.json` (per-tick qpos, mocap, ctrl, stand pose, scalars) |
| `analyze.py` | post-hoc clearance of each recorded run to the stereo zonotopes and to the true object geoms |
| `render.py` | replay -> 1920x1080 25 fps side-by-side video (+ `--preview`), no re-simulation; operator-intent ghost on the shield side (`--no-ghost` = the 10-10 video) |
| `tests/` | state machine, operator determinism, replay determinism, schedule |

## Pipeline / commands
```
cd /common/users/ss5772/bimanual ; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=$PWD
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python ; F=world_model.v2.house.flagship
$PY -m $F.simulate --side raw    --seed 0 --perception off                       # CPU only
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<free,!=0> $PY -m $F.simulate --side shield --seed 0 --perception live   # CPU physics; GPU only for the 2 stereo captures
#   (re-run shield with --perception cached  -> fully CPU, re-uses perception_*.pkl)
$PY -m $F.analyze --tag raw_s0 ; $PY -m $F.analyze --tag shield_s0
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<free,!=0> $PY -m $F.render --seed 0 [--preview]
$PY -m pytest -q world_model/v2/house/flagship/tests
```
Outputs in `/data/local/ss5772/molmospaces/flagship/` (override `FLAGSHIP_OUT`).

## Design decisions (see also the task report)
* **Operator**: closed-loop `SloppyOperator` model on each side with the same seed (same noise realisation / delay / bump times), each reacting to its own robot.
  Open-loop replay is not used: a shield that slows the robot would leave an open-loop stream far off the route.
* **Payload**: the robot drives OUT with empty hands; the tray + cup (8 beads) wait on a serving ledge at the counter; the return carries them.  The base shield uses the
  cup/IR rows only while the tray is held (keep-out rows always).
* **Lift**: tray ledge top = 0.90 m (counter height); the lift is raised by `H_PICK` = 0.056 m at the pre-dock to put the grippers at the handles, +0.06 m after the
  grasp, back to 0 for the drive home.  The ledge is a hand-off slab (TeleopSim `tp_counter`) plus a decorative plate.  No fall-back to the hand-off shelf at hold height was needed.
* **Desk**: the desk (0.734 m) is below the lowest hold height (0.85 m; the lift cannot descend) -> after the (planned or naive) move the tray is lowered onto a small
  PLACEMENT STAND (`tp_stand`, moved under the spot).  Documented simplification.
* **Perception**: stereo is rendered on a twin of the scene (same spec + speckle textures, ZED cameras); at a capture the twin's robot is posed at the main sim's base pose.
  The cup is not in the twin: the payload zonotope group gets cup points instead (self-filtered payload).  The dynamic walker is not in the twin.
* **Autonomous phases** (`PLACE`, `ABORT`, `RELEASE`) run without the base shield: the zonotope plan is the safety mechanism (shield side); the raw side drives the operator's
  naive straight line to the desk centre, halts on first contact, retraces, and the mission fails.
* **zono_rows switch**: `--place-backend zono_rows` calls `world_model.v2.house.zono_rows.flagship_backend(result, obs)` (expected to return `dict(ok, spot_t, path_t)`);
  absent module / function -> falls back to the zed_demo planner and logs it.

## Operator-intent ghost (2026-10-30, `teleop/ghost.py`)
The shield side shows a translucent robot where the operator's RAW command would put it (cyan = shield passing, orange = shield
intervening), plus arrows above the robot: orange = requested base motion, green = permitted (the shield QP's output this tick, not a
guarantee over the horizon).  Each tick both commands are rolled through the sim's own base model for 0.8 s (`--ghost-h`, a display choice)
from the measured state; the ghost is the robot composed with the raw-minus-permitted difference, so it sits on the robot when the shield
passes the command.  The 10-10 runs record the raw base twist (`s_u_op`) but not the raw tray jog / lift rate: in the video the ghost is
base-only and its arms ride along at the measured joints (the live web teleop also predicts the arms).  Output defaults to
`*_ghost.mp4`, so the 10-10 video is never overwritten.  CPU render: `MUJOCO_GL=osmesa LP_NUM_THREADS=2 ... --jobs 2`.

## Known issues / findings
* **People that pause freeze** (read-only `teleop/sim.py::set_people`): a person's track clock only advances while the current track velocity is non-zero, and
  `PersonTrack.at()` returns zero velocity at `t <= first time`.  Consequences: a track that starts at t=0 never starts, and any pause (zero-velocity segment, or the
  hold after the last point) freezes the walker there for the rest of the run.  `scene.launch_fix` prepends a point at t=-1 so tracks start; the corridor walker then
  walks west at 0.42 m/s from t=0 and stands at (9.5, 3.55) from t=18 s (that is the track's end hold -- by design here).  The optional cafeteria pacing walker
  (`--walker`, default OFF) pauses at its ends -> froze on the route and stalled the shield side for 80+ s, so the cafeteria persons are the three static mannequins.
* The shield's footprint-vs-occupancy clearance (`wall`) is meaningless while the tray is held over the desk (autonomous phases `PLACE/ABORT/RELEASE/DONE`): masked in
  the metrics and the video.
* mocap index 0 must stay `yor_root` (`teleop/sim.py`, `compose.set_base_pose` assume it): the placement stand is a static body moved through `model.body_pos`.
