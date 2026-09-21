# OOD generalization test: cup-WM spill monitor under domain randomization

**No retraining** -- existing checkpoints only: WM `world_model/logdir/wm_cup_full/best.pt`,
monitor `world_model/logdir/wm_cup_full/monitor_head_cup_full.pt`.

## DR test set

`randomization/collect_cup_dr.py` (new, wraps `collection/collect_cup.py` by
monkeypatching `cameras.build_yor_cup_model` to call
`randomization.domain_randomization.randomize()` on the compiled MjModel).
40 episodes, seeds 5000-5039, `collection/data/cup_dr_test` -> converted via
`dataset_adapter` -> `world_model/data/dreamer_fmt/cup_dr_test`.

**Families/ranges**: default `RandomizationConfig` -- visual (light pos
+/-0.4m, intensity 0.5-1.6x, color +/-0.15; material color +/-0.20,
reflectance 0-0.4; camera pos +/-0.01m, angle +/-2deg) and dynamics
(friction 0.7-1.3x, mass 0.8-1.2x, damping 0.8-1.2x, actuator gain
0.85-1.15x) both ON. All nonzero relative to training data
(`cup_yor_full`), which used **zero** randomization -- so this is a genuine
OOD test. **Caveat**: `scene` family's cup/ball qpos0 jitter does not
propagate through `collect_cup.py`'s keyframe-based reset (documented in
the script's docstring) -- only its table `body_pos` jitter (direct,
reset-independent) applied. Visual/dynamics were unaffected by this gap.

**Outcome mix**: spilled=20/40, collision=10/40, clean=10/40 (same 4-behavior
cycling as training, seeds only shift the DR draw).

## Monitor metrics: OOD vs in-distribution (per_step spill signal, same metric code)

| | in-dist (`monitor_cup_full_perstep`) | OOD (`monitor_cup_dr_ood`) | delta |
|---|---|---|---|
| spill precision | 0.978 | 0.607 | -0.371 |
| spill recall | 0.986 | 1.000 | +0.014 |
| **spill ROC-AUC** | **0.9995** | **0.746** | **-0.253** |
| collision precision | 0.885 | 0.867 | -0.018 |
| collision recall | 0.936 | 0.428 | -0.508 |
| **collision ROC-AUC** | **0.995** | **0.932** | **-0.063** |

## Verdict: does NOT generalize well

Spill AUC drops sharply (0.999 -> 0.75, still >> chance but far from
production-usable); collision AUC degrades much more mildly (0.995 -> 0.93).
Spill precision collapses (0.98 -> 0.61) while recall stays ~1.0 -- the head
over-triggers spill under DR rather than losing all signal, i.e. probabilities
are systematically shifted/miscalibrated, not random.

**Likely cause: visual shift**, not motion/proprio-only keying. Evidence: the
spill signal is the one label most dependent on the ZED image (scattered
balls / cup tipping are the only spill evidence; cup motion itself is
identical, open-loop-scripted teleportation, in-dist and OOD alike -- so
proprio/action sequences for spill behaviors are essentially unchanged).
Collision, by contrast, comes from a fixed scripted right-arm sweep whose
joint trajectory doesn't depend on visuals at all, and its AUC held up far
better -- consistent with the WM latent's collision-relevant component
leaning on kinematics/proprio (robust to material/light/camera jitter) while
its spill-relevant component leans on rendered appearance (not robust to it).
Dynamics-family jitter (friction/mass/damping/gain) was co-varied with
visual in this run, so its individual contribution isn't isolated here.

## Blockers

None. GPU: `CUDA_VISIBLE_DEVICES=3`, `mujoco_ur5e` env for WM/eval,
`safe_env` for DR collection (per env split in `world_model/README.md`).

## Rollback

`rm -rf collection/data/cup_dr_test world_model/data/dreamer_fmt/cup_dr_test
artifacts/monitor_cup_dr_ood*`; `rm -f randomization/collect_cup_dr.py
world_model/eval_monitor_ood.py world_model/RESULTS_WM_DR.md`. No existing
files were modified.

---

# DR-retrain pipeline: submitted (job 273122)

**Goal**: fix OOD spill-monitor brittleness above (0.999 in-dist -> 0.746
under DR) by retraining the WM+monitor head on DR data instead of clean
data, then re-evaluating OOD + a clean-set regression check.

## What was written

`world_model/dr_retrain.sbatch` (new) -- 4 finite stages on one allocated
node/GPU, `MUJOCO_GL=egl`:
- S1 `randomization.collect_cup_dr` (200 eps, seed 6000, `safe_env`) ->
  `collection/data/cup_dr_train`
- S2 `world_model.train_wm` (2000 steps, bs16/bl64, `mujoco_ur5e`) ->
  `world_model/logdir/wm_cup_dr`
- S3 `world_model.monitor_head` (300 epochs, spill+collision,
  `per_step` signal) -> `world_model/logdir/wm_cup_dr/monitor_head_cup_dr.pt`
- S4a OOD eval on the held-out DR test (`cup_dr_test` ->
  `dreamer_fmt/cup_dr_test_v2`, tag `dr_on_drtest`)
- S4b regression check on the original clean eval set (`cup_yor_full`,
  `--skip-convert`, tag `dr_on_clean`)

Mirrors `hj_reachability/floq/train_floq.sbatch`'s pattern: generous
`--time=12:00:00`, all-finite stages so the script self-exits, `_DONE`
sentinel as the last line
(`world_model/logdir/wm_cup_dr/_DONE`), stage logs redirected to
`world_model/logs/dr_retrain_273122.log`.

## Node selection

`sinfo`/`scontrol` via the gateway: `rlab1` was the only fully **IDLE**
node in `unlimited` (`AllocTRES=` empty, 3x free `RTX 4500 Ada`,
`Gres=gpu:4500_ada:3`) -- every other rlab/ilab node was `mix`/`mix-`
(partially or fully allocated). sbatch pins `--nodelist=rlab1
--gres=gpu:4500_ada:1`.

## EGL check on rlab1 (pre-submit)

Direct ssh smoke test, `safe_env`, `MUJOCO_GL=egl`:
`mujoco.Renderer(...).render()` returned a `(64, 64, 3)` frame ->
**EGL_OK**. (One warning, `failed to open /dev/dri/card1: Permission
denied` -- expected outside a Slurm GPU cgroup; it fell through to an
accessible card and rendered fine. Under the sbatch's actual
`--gres=gpu:4500_ada:1` allocation this resolves cleanly to the assigned
GPU.) Not a blocker.

## Submission

`bash scripts/offload/submit_via_slurm_gateway.sh world_model/dr_retrain.sbatch`
-> **job 273122**, `unlimited` partition, node `rlab1`. Confirmed via
`squeue -j 273122` (PD, reason `Priority`, not `Resources` -- should start
promptly given the target node was idle).

- Status: `source scripts/offload/slurm_gateway.sh && slurm_gateway_squeue -j 273122`
- Cancel: `slurm_gateway_scancel 273122`
- Sentinel: `world_model/logdir/wm_cup_dr/_DONE`
- Stage log: `world_model/logs/dr_retrain_273122.log`
- Slurm stdout/err: `bimanual_wm_dr_retrain_273122.out` / `.err` (gateway
  submit cwd)
- Autorelease armed in background (PID 3251391, this session):
  `slurm_autorelease.sh 273122 world_model/logdir/wm_cup_dr/_DONE`

## Blockers

None.

## Rollback

`scancel 273122` (or `slurm_gateway_scancel 273122`); `rm -f
world_model/dr_retrain.sbatch`; `rm -rf collection/data/cup_dr_train
world_model/data/dreamer_fmt/cup_dr_train
world_model/data/dreamer_fmt/cup_dr_test_v2 world_model/logdir/wm_cup_dr
world_model/logs/dr_retrain_273122.log
world_model/logs/autorelease_273122.log
bimanual_wm_dr_retrain_273122.{out,err}`. No existing files modified.
