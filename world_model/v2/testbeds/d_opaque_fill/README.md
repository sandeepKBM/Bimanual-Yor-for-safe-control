# d_opaque_fill -- opaque container with hidden fill (feasibility probe)

Simulation only. CPU only. No hardware, no network. Imports `bead_rep.bead_env` (kin_step, quat_axis, fallback_U),
`data.contents` (pin_cup, ROBOT path), `latent_hj.slosh_pend` and `latent_hj.spill_data` read-only. Edits no existing file.

| file | what |
|---|---|
| `opaque_env.py` | MuJoCo rig: open cylinder (inner R 30 mm, rim 120 mm, shell 0.15 kg) with n grains (r 6 mm, 1400 kg/m3). Kinematic pose pin, snapshot/restore, continuous margin, wrench log. Also `PendModel` (analytic slosh pendulum for this container). |
| `lib.py` | replay of logged `spill_v2` carries, held-action branch search (certificate), helpers. |
| `run_probe.py` | the feasibility probe. Phases: `p1` (replay), `p2` (savable set, MuJoCo and pendulum), `probe` (identification), `torque` (joint torque via the YOR left-arm Jacobian). |

## Run
```
cd /common/users/ss5772/bimanual
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
$PY -m world_model.v2.testbeds.d_opaque_fill.run_probe --phase p1 --procs 8            # ~7 CPU-min
$PY -m world_model.v2.testbeds.d_opaque_fill.run_probe --phase p2 --procs 8 --states 32
$PY -m world_model.v2.testbeds.d_opaque_fill.run_probe --phase probe
$PY -m world_model.v2.testbeds.d_opaque_fill.run_probe --phase torque
$PY -m world_model.v2.testbeds.d_opaque_fill.selftest                                   # equivalence checks
```
Output: `/common/users/ss5772/bimanual_data/testbeds/d_opaque_fill/feasibility.json` (phases merge into one file); scratch in `.../tmp`.

## Model notes
* Solver: CG + elliptic cone, 50 iterations, dt 5 ms. Newton costs O(nv^3) (3 ms/step at 47 grains). Pyramidal CG explodes at mu < 0.2.
* The loose hex layout is settled inside temporarily tall walls (`H_TALL`), then the walls shrink to the real rim height. Exits flagged during the settle are cleared.
* Fill level is set by grain count; the settled height `env.top` is measured (packing fraction ~0.57).
* `mu=0.03, drag=3` is a crude liquid proxy. It is NOT a fluid (see report section 2c).
