# A/B/C: smooth HOCBF-QP vs bang-bang HJ-style fail-safe (E1: pick + intrusion)

Single-arm pick + intrusion experiment on `scenes/scene_bimanual_pickup.xml`.
LEFT arm (ego) grasps `box_center` and is torque-controlled (computed-torque
joint-space PD holding its initial pose) on its 6 arm joints; RIGHT arm
(intruder) sweeps into the ego's workspace via a scripted, open-loop
position-servo trajectory, identical across all three conditions. Only the
ego's *safety* law differs:

- **C0** -- no safety filter (task PD only). Establishes the scenario is
  non-trivial (arms actually collide without an avoidance layer).
- **C1** -- bang-bang HJ-style fail-safe: hard, discontinuous switch to
  extremal avoidance torque `tau = tau_max * normalize(J^T n)` whenever
  `h < H_TRIGGER`, else task PD. No smoothing.
- **C2** -- smooth relative-degree-2 HOCBF-QP safety filter: every step,
  solve `min ||tau - tau_task||^2` subject to `psi2 >= 0` (one row per
  ego/intruder sphere pair -- see "Barrier" below) and torque limits, reusing
  `real_Cartpole/controller_core/constrained_box_qp.py::solve_constrained_box_qp`
  as the QP core, then smooth the result with
  `real_Cartpole/controller_core/filters.py::TorqueCommandFilter`
  (low-pass + slew-rate limit).

## Run

```bash
cd /common/users/ss5772/bimanual/experiments/ab_smooth_vs_bangbang
MUJOCO_GL=egl /common/users/ss5772/miniforge3/bin/conda run -n safe_env python run_ab.py
MUJOCO_GL=egl /common/users/ss5772/miniforge3/bin/conda run -n safe_env python make_plots.py
```

`run_ab.py` writes per-timestep logs to `logs/{C0,C1,C2}.npz` and a metric
summary to `logs/summary.json`. `make_plots.py` reads those and writes the
comparison plots to `/common/users/ss5772/bimanual/artifacts/ab_*.png`.
Deterministic (no randomness anywhere in the loop; the only "search" --
picking the intruder's target joint config -- was done once offline and is
hard-coded as `RIGHT_INTRUDE_CTRL` in `sim_common.py`).

## Files

- `sim_common.py` -- scene/model setup, grasp initialization (reuses the
  teleport-to-pad-midpoint-then-close-fingers technique already validated in
  `scripts/test_pickup_scene.py`), the intruder trajectory, the barrier /
  Jacobian / mass-matrix helpers, SPARC and jerk metric utilities, and the
  `EgoController` base class (task PD + single-pair barrier used by C0/C1).
- `run_ab.py` -- the three conditions' control loops, per-step logging, and
  metric computation.
- `make_plots.py` -- the comparison figures.
- `logs/` -- raw npz logs + `summary.json` (all three conditions).

## Key parameters (identical across conditions unless noted)

| param | value |
|---|---|
| dt (= model timestep) | 0.002 s |
| episode duration | 4.0 s (2000 steps) |
| ego arm | LEFT (torque control, joint1-6) |
| intruder arm | RIGHT (position servo, scripted) |
| barrier spheres | {link3, link5, gripper_base} per arm (3x3 = 9 pairs for C2's QP; single argmin pair for C0/C1 logging and C1's control) |
| r_ego, r_int, d_min | 0.05, 0.05, 0.08 m |
| combined radius (h=0 boundary) | 0.18 m |
| H_TRIGGER (C1 switch) | 0.05 m |
| alpha1, alpha2 (HOCBF gains) | 15.0, 15.0 |
| tau_max | 40 Nm / joint |
| task PD gains | Kp=600, Kd=2*sqrt(600) (critically damped) |
| TorqueCommandFilter | lowpass alpha=0.25, rate limit 400 Nm/s |
| intruder timeline | ramp in [0.5,1.5]s, hold to 2.7s, ramp out to 3.7s |

## Honest caveats / approximations (see the top-level report for full discussion)

- The barrier is an **analytic sphere-distance proxy**, explicitly a stand-in
  for a trained HJ value (no DeepReach value is used here).
- C2's HOCBF-QP constrains **all 9 ego/intruder sphere pairs simultaneously**
  (not just the instantaneous closest pair) -- this was a deliberate fix after
  an early version (single argmin-pair constraint) let a discrete
  closest-pair *switch* produce an unconstrained jump in h/hdot right as the
  intruder passed by, causing a real collision. Constraining every pair
  removes that gap; it is exactly what "all-pairs" is doing in the code and
  is called out in `sim_common.py::all_pair_barriers`'s docstring.
- The `hddot` term for the QP's psi2 uses `mj_jacDot` (fully analytic
  Jacobian-rate, MuJoCo 3.6 API) and the intruder's *actual* instantaneous
  `data.qacc` (from `mj_forward` with its current position-servo command) --
  no finite-differencing was needed for the drift terms in this
  implementation.
- The computed-torque task PD uses the **6x6 left-arm-only block** of the
  full mass matrix (`M_ll`) and `qfrc_bias` restricted to those 6 DOF; it
  ignores dynamic coupling with the (also serially attached) finger DOF and
  any contact-constraint forces from holding the box. This is why a
  finite (~1-1.5 cm at the chosen gains) steady EE-position offset appears
  during the initial settle in all three conditions -- a real, reported
  limitation of proportional-only computed torque without integral action,
  not a bug (see the Kp comment in `sim_common.py`).
- Grasp initialization reuses `scripts/test_pickup_scene.py`'s "teleport the
  box to the open-finger pad midpoint, close with gravity off, then restore
  gravity" trick verbatim (that script and the scene file itself were left
  untouched, per the task's read-only constraints).
