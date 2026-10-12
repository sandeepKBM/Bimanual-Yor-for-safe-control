# world_model/v2/house/teleop -- teleoperated coffee run with a shared-control safety layer

SIMULATION ONLY (offline MuJoCo). No robot driver, no `YOR/robot/base.py`, no `YOR/extra/*`, no network, no device-to-robot link.
"Teleop" = a simulated operator command stream, plus an optional keyboard mode in the MuJoCo passive viewer (key_callback only).

| file | what |
|---|---|
| `operator.py` | `SloppyOperator` (closed-loop human model, deterministic per seed), `rollout_stream` / `ReplayOperator` (open-loop recorded stream), `KeyboardOperator` (viewer `key_callback`, `inject()` for headless tests) |
| `shield.py` | `Shield` (base governor QP + keep-out + IR monitor + jog/lift governor), `Passthrough` (shield off), `ShieldCfg` |
| `sim.py` | `TeleopSim`: one 50 Hz teleop tick = 4 physics steps; FL-MPC every physics step; `Cmd`; grasp/release (weld activation) |
| `arm.py` | `ArmCtl` (phase1.Ctl + tilt cone), `ObjectFrame` (tray-frame 6-DoF jog -> DLS IK -> joint refs of the unmodified FL-MPC) |
| `base.py` | mocap base kinematics, slew limit, cup-point acceleration (`cup_accel_body`) |
| `geom.py` | footprint / EDT clearance / rectangle-to-point distance |
| `scene.py` | scene registry (`h5`, `office` hook), rig composition (house/office + YOR + lift + tray + cup with beads + shelves + people), hold pose |
| `run.py` | one episode raw vs shielded -> metrics JSON (+ optional MP4) |
| `live.py` | optional live keyboard session (needs a display; not exercised headless) |

```
cd /common/users/ss5772/bimanual ; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=$PWD
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
$PY -m world_model.v2.house.teleop.run --mode both --task carry --scene h5 --seed 0 --max-s 30      # CPU smoke, outputs -> /data/local/ss5772/molmospaces/teleop
$PY -m world_model.v2.house.teleop.run --mode both --task coffee --scene h5 --max-s 110              # full pick/carry/place (several minutes per run)
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<free gpu, not 0> $PY -m world_model.v2.house.teleop.run --mode shield --task carry --video
$PY -m pytest -q world_model/v2/house/teleop/tests                                                    # headless unit smoke
```

## What is simulated

* Base = kinematic mocap (as phase2): commanded body twist (vx, vy, wz), hardware slew `base.A_HW` = (4, 4, 8) (m/s^2, m/s^2, rad/s^2) -- an ASSUMPTION
  (real YOR base limits unknown), full-stick `base.V_STICK` = (0.9, 0.45, 1.2). Base acceleration reaches the tray only through the two tray welds (phase2
  limitation: the mocap base gives the arms no fictitious force). Measured: tray acceleration equals the commanded base-point acceleration to <1 %.
* Bimanual abstraction: the operator commands the tray (object frame). Base twist moves the whole robot; the tray-frame jog (6-DoF) moves the hand targets
  rigidly with the object frame; both arms follow through the unmodified FL-MPC (`mpc_fl.py`, 200 Hz, all of its constraints incl. arm-arm clearance, torque box,
  tray tilt cone 0.10 rad).
* Cup with 8 beads (bead_rep / contents.py cup + bead geometry, nominal contents) standing on the tray at tray-frame (-0.045, 0) (the free region reported by the
  lift/perception agent; cup-to-arm clearance 24.7 mm at the hold pose, checked in `set_hold`). Cup-tray friction mu = 0.25 (all cup geoms set; the cup walls ship
  mu = 0.6 and would otherwise win MuJoCo's max-combine). Spill = a bead leaving the cup (cup-frame `Oracle.out_mask`, sticky); `cup_lost` = cup centre leaves the tray.
* Grasp (SIMPLIFICATION, grasping is out of scope): when both grippers are within 3.5 cm / 6 deg of the tray handles (nominal weld relpose) the two tray welds are
  activated at the current relative pose (no snap). Release = weld deactivation. The tray rests on a hand-off shelf (slab at the hold height) at the counter / desk.
* People: scripted capsules (mocap) that stop when the robot footprint is within 0.15 m of their torso; office mannequins are static spec bodies.

## The mu / theta sets (shield)

mu in [0.18, 0.35] (sim truth 0.25), kappa = 0.8, theta_spill = 0.30 rad (F5 default tilt set), tray tilt bound theta_tray = 0.05 rad (commanded jog) with the FL-MPC cone 0.10,
IR upright bound 0.10. In-plane effective-gravity ratio r = min(kappa mu_lo, tan theta_spill) = 0.144 -> admissible cup acceleration |a| <= 1.41 m/s^2 on a level tray
(less when the measured tray axis is tilted). Jerk bound 15 m/s^3 on the cup-point acceleration.

## Control rates

physics 200 Hz; FL-MPC 200 Hz; operator + shield + base command 50 Hz; shield QP is a 3-variable projection with ~60-100 rows (<1 ms numba Hildreth).
In this Python build FL-MPC costs ~14 ms/call, so a run is ~4-5x slower than real time.
