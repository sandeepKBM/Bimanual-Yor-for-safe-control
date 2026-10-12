# a_loose_items -- feasibility prototype (dish / tray with N loose candy-like spheres)

SIMULATION ONLY, CPU only, no network, no hardware. Edits no existing file; imports `world_model.v2.bead_rep.bead_env`
(kin_step, Snapshot, brake_action, fallback_U), `world_model.v2.data.contents` (pin_cup) and `wm_audit.common.load_split`.
Report: `reports/testbed_objects/a_loose_items.md`. Numbers: `/common/users/ss5772/bimanual_data/testbeds/a_loose_items/feasibility.json`.

| file | what |
|---|---|
| `tray_env.py` | `TrayEnv(TrayCfg)`: kinematic container + N spheres, `tick(u)` at 50 Hz, `snapshot()/restore()` (bitwise), margin `l` in [-0.8, 1], `softmin_margin`. `beta_deg=0` = straight tray, `45` = dish. |
| `carry.py` | nominal carries: `replay` (spill_v2 test `act`, first 300 ticks) and `synthetic` (random held actions, too violent). |
| `probe.py` | feasibility probe: failure rate, savable set (hold / brake / constants / random 2-seg), lead time, hold-still spill rate. |
| `reduced.py` | reduced analytic model (independent rolling points, 6N+10 state, no item-item contact). |
| `gap.py` | reduced model vs MuJoCo: open-loop episode label, plan-aware label from mid-episode state (privileged vs camera-like). |
| `sens.py` | hidden-parameter sensitivity (friction, bounce, radius). |

Run (from the repo root; pin threads; ~1-3 CPU-min each):
```
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python; M=world_model.v2.testbeds.a_loose_items
$PY -m $M.probe --beta 45 --h 0.025 --n 10 --episodes 100 --workers 6 --out <json>   # dish
$PY -m $M.probe --beta 0  --h 0.015 --n 10 --episodes 100 --workers 6 --out <json>   # straight tray
$PY -m $M.gap --episodes 40 --out <json>
$PY -m $M.sens <json>
```
