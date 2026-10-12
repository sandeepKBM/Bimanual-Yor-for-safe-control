# c_block_tower: feasibility probe (simulation only, CPU only)

A stack of N blocks on a plate. Two arms carry the plate. In sim the plate moves kinematically under the pipeline's
6-D command u = [accel(3), axis angular velocity(3)] (the 85-element fallback set, 50 Hz ticks, tilt limit 1.5 rad).
Report: `/common/users/ss5772/bimanual/reports/testbed_objects/c_block_tower.md`.
Numbers: `/common/users/ss5772/bimanual_data/testbeds/c_block_tower/feasibility.json`.

| file | what |
|---|---|
| `tower_env.py` | `TowerEnv`: MuJoCo rig, `tick(u)`, `snapshot()/restore()`, continuous margin `l`, numba twin of `bead_env.kin_step` (`selftest()`, error ~1e-16). Imports `bead_rep/bead_env.py`; edits nothing. |
| `analytic.py` | Reduced model: rigid column, quasi-static slide and tip (ZMP-like) margins per sub-stack. |
| `probe.py` | Stages `grid`, `savable`, `gap`, `sens`, `ramp`, `selftest`. |

## Run (about 16 CPU-min in total; use 8 cores)
```
cd /common/users/ss5772/bimanual; PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
M=world_model.v2.testbeds.c_block_tower.probe
$PY -m $M selftest
$PY -m $M ramp                                                             # 5 s   slip/tip onset vs atan(mu), mass, noslip
$PY -m $M grid --nshard 4 --nproc 8 --Ns 3 --mus 0.5 --ss 0.25 0.3 0.35 0.4 --tag _t2   # 15 s  nominal failure rate vs tilt scale
$PY -m $M savable --nshard 8 --nproc 8 --n 3 --mu 0.5 --s 0.35 --extra 3 \
     --back 1 2 3 4 5 6 8 10 12 15 20 30 40 50 --tag _main                # 40 s  savable set, hold-still, lead time
$PY -m $M gap --nshard 4 --nproc 8 --n 3 --mu 0.5 --s 0.35 --max-fail-eps 24 --back 4 12 30 --tag _main   # 30 s  analytic vs MuJoCo
$PY -m $M sens --nshard 4 --nproc 8 --s 0.35 --tag _main                   # 20 s  hidden-parameter and solver sensitivity
```
Intermediate JSON goes to `<bimanual_data>/testbeds/c_block_tower/tmp/` (create it first).

## Design notes
- Plate: 5 kg, `gravcomp=1`. A light plate, or one that falls under gravity inside a step, makes the blocks creep by a fake 4 mm per 20 ticks.
- Blocks taper by 0.3 mm per layer. Identical coplanar boxes trigger a MuJoCo 3.9 FatalError ("9 contacts, max 8").
- `tilt scale s` multiplies the polar angle of the logged spill_v2 plate axis (azimuth and accelerations unchanged). It is a calibration knob, ASSUMED.
- The spill_v2 trajectories are replayed from `/common/users/ss5772/datasets/bimanual_wm_v2/spill_v2` (test split rule: shard index % 5 == 4).
- All geometry, mass and friction values are assumptions. No real object spec exists.
