# tray_rep: tray + free ball, MuJoCo as ground truth (phase 1a)

This is the data and evaluation layer for the "coverage vs caution" experiment on a real-physics testbed.
The pendulum proxy is replaced by MuJoCo. A ball rolls on a rim-less tray. The tray is moved by the
same 6-number action as the cup rig. The ball leaves the tray = failure.

Simulation and saved data only. CPU only. No GPU, no rendering, no hardware, no network.
No existing file is edited. The rig comes from `testbeds/b_tray_ball` (imported, not copied).
Python: `/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python`. Run from the repo root.
Pin threads (the modules do it for you with `setdefault`):
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMBA_NUM_THREADS=1`.

This is phase 1a: infrastructure only. No world model was trained here.
`DESIGN_WM_PORT.md` lists what the existing training and value code needs to run on this data.

## Files

| file | what it does |
|---|---|
| `env.py` | `TrayRepEnv`: wraps `TrayEnv`. Hidden `mu_roll`, `a_scale`, `obs_mode`, `tick(u)`, `snapshot()/restore()`. |
| `gen_nominal.py` | Sharded, resumable generator of nominal carries. Writes a `load_split`-compatible dataset plus snapshots. |
| `store.py` | Reads a dataset: logged rows, snapshots, `load_split` (superset of the wm_audit loader). |
| `branches.py` | MuJoCo counterfactual branches from snapshots. Root laws `R` and `Rc`. |
| `oracle.py` | Certifies "savable" states with MuJoCo (b's library search). Labels: hold_ok, savable, uncertified. |
| `cl_eval_mj.py` | Closed-loop evaluator on certified states, for any policy. Two reference policies. |
| `selftest.py` | Runs every smoke test (about 50 s, 4 processes). |
| `DESIGN_WM_PORT.md` | What to change in kin_wm, value code, trainer, cl_eval. Impact analysis. |
| `configs/piper.json` | Phase 1b: Piper-feasible authority and carry settings, plus the `tracker_realistic` entry. |
| `wm/` | Phase 1b: the world model, value, policy, arm runner, G1 gate, launchers. See "Phase 1b" below. |
| `authority/` | The authority study (robot torque limits, scaled savable set). Read-only reference. |

## What the data means

- Episode = one nominal carry of 4 to 5 s (about 225 ticks at 50 Hz). The tray moves in 3 min-jerk transports.
  A noisy tilt loop tries to keep the ball still. The law is `b_tray_ball/carry.py`, intensity 1.7 (parameters are ASSUMED).
- Hidden per episode: `mu_roll` (rolling resistance), log-uniform in [0.0005, 0.006] m. It is stored as `hid`.
  It is never in `obs`. It decides 35 to 50% of the failure outcomes, so the world model must infer it from the ball's motion.
- Row t of every array = the state at tick t. `act[t]` = the action applied from t to t+1, as `[a (3), w_eff (3)]`.
- `obs_mode = full`: `obs` (13) = tray axis (3), tray velocity (3), realised axis rate (3), ball x, y, vx, vy in the tray frame (4).
- `obs_mode = tray_only`: `obs` (9) = the first nine. The ball is invisible. The model must dead-reckon it from tray motion
  (the ball starts near the tray centre, so this is possible in principle; branch windows that start mid-episode lose that anchor).
- `l` = margin, in [-0.5, 1]. `l < 0` means the ball centre is off the tray. `onset` = first tick with `l < 0`, or -1.
  The episode keeps running after the ball falls (`l` stays -0.5). Ball values are clipped (position 0.5 m, velocity 3 m/s).
- `n`, `nd` = ball position and velocity in the tray frame (3 each; privileged, aux head only).
- `priv` (11) = ball position (3), velocity (3), angular velocity (3), signed distance d, lift. Privileged.
- `feat` (10, in branches) = `priv` scaled to O(1): the analogue of `train_kinwm.feats_t`.
- `a_scale` (default 1.0) multiplies the 85-element fallback set, the uniform branch box and the brake limits.
  Another job is computing the real Piper scale. Set it with `--a-scale` (it is also stored in `meta.json`).
  The nominal carry is not scaled (it is the operator).

## Run commands (owner)

```
cd /common/users/ss5772/bimanual ; PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
R=/common/users/ss5772/bimanual_data/tray_rep
$PY -m world_model.v2.tray_rep.selftest --fresh                      # about 50 s; summary in $R/smoke/selftest_summary.json

# 1. nominal data (3000 episodes = 120 shards of 25; split 60/20/20 by shard index % 5)
$PY -m world_model.v2.tray_rep.gen_nominal --obs-mode full      --n-eps 3000 --workers 16
$PY -m world_model.v2.tray_rep.gen_nominal --obs-mode tray_only --n-eps 3000 --workers 16   # same episodes, 9-D obs
$PY -m world_model.v2.tray_rep.gen_nominal --obs-mode full --n-eps 3000 --merge             # writes $R/data_full/{train,val,test}.npz
#    resumable: rerun the same command, finished shards are skipped. Array style: --shard i/n.
#    options: --a-scale 0.4  --lip (dustpan)  --intensity 1.7  --seed0 100000

# 2. branches (train split). R = make_branches law; Rc = uniform over not-yet-failed roots
$PY -m world_model.v2.tray_rep.branches --data $R/data_full --split train --law R  --n 10000 --workers 16 --out $R/branches/R_10k.npz
$PY -m world_model.v2.tray_rep.branches --data $R/data_full --split train --law Rc --n 10000 --workers 16 --out $R/branches/Rc_10k.npz
#    torch: world_model.v2.tray_rep.branches.load_branches(path, dev) -> dict(obs, act, l, feat, priv)

# 3. oracle on TEST states (writes $R/data_full/oracle/test_v0.npz)
$PY -m world_model.v2.tray_rep.oracle run --data $R/data_full --split test --n 2000 --workers 16 --tag test_v0
#    array style: --shard i/n  then  oracle merge --data ... --tag test_v0 --n-parts n
#    more certificates: --d-holds 5,10,25 (about 3x the CPU)

# 4. closed-loop reference policies
$PY -m world_model.v2.tray_rep.cl_eval_mj --data $R/data_full --oracle $R/data_full/oracle/test_v0.npz --T 50 --out $R/cl_ref_v0.json
```

In Python, for your own policy `pi(obs_hist) -> u`:
```python
from world_model.v2.tray_rep import cl_eval_mj as CL
res, per_state = CL.evaluate(pi, data_dir, oracle_npz, split="test", T=50)
```
`pi` may carry `needs_acts = True` (then it gets `act_hist` too) and a `begin(ctx)` hook. See the module docstring.

## Certified savable: what it means

A state is certified savable when hold-still fails within 2 s and some sequence in a fixed library survives 2 s.
The library is b's: each of the 85 fallback actions held 0.2 s, then a brake (stop and level the tray); plus brake-only.
It is an existence proof. It under-counts the true savable set. "Uncertified" means ambiguous, not "doomed".

## Smoke results (VERIFIED, 2026-10-05, `selftest --fresh`: all checks pass, 48 s, 4 processes)

| check | result |
|---|---|
| determinism, snapshot/restore (same env and fresh env) | bitwise equal over 80 ticks |
| `mu_roll` effect | ball x after 30 tilted ticks: 0.106 m (0.0005) vs 0.014 m (0.006) |
| gen, 40 episodes, both modes | failure 20/40 = 50% (flat tray, intensity 1.7); 0 tunnels; `wm_audit.common.load_split` reads all three splits; full[:, :9] == tray_only bitwise |
| larger nominal run (200 episodes, scratch) | failure 43%; by `mu_roll` bin: 77%, 57%, 19%, 6% (low to high) |
| resume, regeneration | rerun skips all 10 shards; a regenerated episode equals the stored one bitwise; a restored snapshot equals the logged state |
| branches, 50 each (40-episode set) | shapes (50,160,13/6/-/10); logged prefix and root state equal the dataset; worker count does not change results; R roots already failed: 32%, Rc: 0% |
| branch outcomes | 96-98% of branches fail within 2 s; median first failure tick 19 (R), 28 (Rc); 42% (R) / 68% (Rc) survive the first 25 ticks |
| oracle, 20 TEST states | hold_ok 15%, savable 60%, uncertified 25%; 0.54 s per state |
| oracle, 70 states (100-episode set) | hold_ok 17%, savable 54%, uncertified 29% |
| cl_eval_mj, 20 certified states, T = 50 | hold-still fails 45% (9/20); oracle policy fails 0% (0/20) |
| `a_scale = 0.4` (oracle and branches) | runs; savable falls from 53% to 45% on the same 40 states |

Untested: a world model on this data (none trained); the `--lip` dustpan path through the generator;
Slurm array runs; the 3000-episode run itself; noisy or delayed ball observations (obs is noise-free);
`--d-holds` with more than one hold length.

## Cost (measured on a loaded host, load average 9 to 13, so pessimistic)

| job | unit cost | full scale |
|---|---|---|
| env tick | 0.37 ms per tick (2.7k ticks/s/core) | |
| nominal episode | about 0.14 CPU-s (225 ticks + snapshots) | 3,000 episodes: about 7 CPU-min (0.12 CPU-h) per obs mode; disk about 0.4 GB |
| oracle | 0.54 to 0.57 CPU-s per state | 2,000 states: about 19 CPU-min (0.32 CPU-h); with `--d-holds 5,10,25` about 1 CPU-h |
| branches | 0.037 CPU-s per branch | 3,000: about 2 CPU-min; 10,000: about 6 CPU-min (0.1 CPU-h) |

Everything here fits on one 16-core node in minutes. No Slurm array is needed.

## Findings to know before using the data

1. The R branch law is very aggressive on a rim-less tray. 96-98% of branches end in failure. Only about half survive 0.5 s.
   Counterfactual data is therefore failure-heavy. The pendulum recipe was tuned on a proxy with a rim; it may need milder
   branches here (a lower gain, more hold-still). `a_scale` lowers the whole law.
2. About 1/3 of R roots are states where the ball has already left the tray (the nominal episode runs on after failure).
   Rc removes them. This is the R-versus-Rc contrast.
3. The nominal failure rate depends on `mu_roll`, so it is partly a property of the sampled range [0.0005, 0.006].
   Mean 43-50% is inside the 20-60% target. A narrower range moves it.
4. Savability is lost 2 to 5 ticks before the edge (b's finding). R's branch states are not near that boundary on purpose.


# Phase 1b: world model, value and arm runner on this data

Simulation and saved data only. The plan is `DESIGN_WM_PORT.md`. The new code is in `wm/`. No old file outside `tray_rep` was edited.
GPU is needed only for the arms. Everything else is CPU. Set `CUDA_VISIBLE_DEVICES=<n>` for a GPU run.
(`tray_rep.common` sets it to empty if you leave it unset, so a forgotten variable gives a CPU run, not a wrong GPU.)

## New knobs in the phase-1a code (defaults reproduce phase 1a; the phase-1a selftest still passes)

| knob | where | meaning |
|---|---|---|
| `w_scale` (default = `a_scale`) | env, branches, oracle, brake | scales the tilt-rate columns of the 85-set and the brake rate limit |
| `alpha` (default off) | env, oracle, cl_eval | tilt-rate slew limit [rad/s^2]. The env slews the commanded rate. `w_slew` is stored in every snapshot (`snap_wslew`). The oracle uses realised library sequences (authority/scaled_probe model) |
| `--carry-mode clip` | gen_nominal | the authority agent's carry: same paths, accel vector clipped at `5*a_scale`, tilt rate at `2*w_scale` |
| `--config piper` | gen_nominal | a 0.3, w 0.5, alpha 10, clip, intensity 3.0. Data goes to `data_piper_<mode>` |
| tracker: `--tracker realistic` or `--obs-noise-m --obs-latency-ticks --obs-rate-hz --obs-dropout-p` | env, gen_nominal | ball tracker model, `full` mode only. Default (0, 0, 50, 0) = today's obs, bitwise. The first nine obs numbers stay exact. Ball velocity = finite difference of the noisy positions between accepted samples. State is in the snapshot. Stored in meta.json |

Carry intensity for the Piper config: **3.0**. In clip mode the failure rate is flat in the intensity, because the clip saturates.
Measured with the random `mu_roll` prior (300 episodes each, then 600): intensity 2.0: 31% / 28%; 2.5: 33% / 31%; 3.0: 32% / 31%; 3.5: 32%; 4.0: 32%.
So the nominal failure rate is about **31%** (900 episodes at 3.0). That is at the low end of the 30-40% target; no setting of the carry moves it
much. Only the `mu_roll` range does. About 6% of episodes at 3.0 are shorter than the 161 ticks the trainer needs for a window, so the WM never trains
on them (2.5 would lose 1%). Tray speed p99 is 0.79 m/s at 3.0.
Oracle on the Piper config (60 TEST states of the smoke set): savable 48%, hold ok 28%, uncertified 23%. The authority study measured 48% too.

## wm/ files

| file | what it does |
|---|---|
| `common.py` | data loading (`load_data`), width-generic `pad_episodes` / `filtered_latents`, the fixed normalisation (`get_norm`) |
| `model.py` | `build_wm(d_obs)`, `load_wm(path)`. The checkpoint stores `d_obs`; `kin_wm.load_wm` cannot reload a 13-wide model |
| `trainer.py` | `TrayTrainer`: the vdirect Trainer with `d_obs`, aux head over `feats_tray(priv)`, tray branches from npz, resume |
| `dyn.py` | `TrayDyn`: the `Dyn` for `reach_value.train`. Scaled `U`, `P.m_lo = -0.5`, slew-aware, tilt projection kept |
| `value.py` | state bank and `train_value` (same hyper-parameters as vdirect) |
| `policy.py` | `WMGreedy`: filters the real MuJoCo obs stream; each tick `argmax_u V(f_WM(z, u))` over the 85-set |
| `run_arm.py` | one cell: WM -> value -> gate -> closed loop -> references -> `result.json` -> `_DONE`. Resumable |
| `prep.py` | CPU prep: `norm`, `branches`, `refs` |
| `g1.py` | the G1-tray gate |
| `selftest_1b.py` | fast checks of the pieces above (about 20 s) |
| `launch_local.py`, `slurm/*.sbatch`, `slurm/tr_tasks.txt` | full-scale launchers (not launched) |

Recipe (vdirect's): WM = 15000 steps, batch 64, window 160, lr 1e-3 cosine, branch-frac 0.5, aux weight 1; value = 10000 iterations, batch 2048,
gamma 0.8 to 0.98, K 4, lam 0.5, bank = filtered TRAIN latents + 200k imagined states. All are command-line parameters
(`--wm-steps`, `--value-iters`, `--value-batch`, `--n-imag`, `--branch-frac`).
Normalisation: nominal TRAIN ticks plus an equal number of ticks from 2000 law-R branches of the TRAIN split (seed 12345), cached in `<data>/norm/`.
Obs modes: `full` (13) and `tray_only` (9). Both read their own dataset (`data_piper_full`, `data_piper_tray_only`). The two have the same episodes,
so one oracle file serves both (the launchers copy it).

## G1-tray gate

`python -m world_model.v2.tray_rep.wm.g1 --data <data dir> --setup piper --obs-mode full`. PASS iff on the certified-savable TEST states:
(a) at least 300 exist, (b) the N0 closed-loop failure is at least 30% (pooled over N0 seeds), (c) the oracle policy fails at most 2%.
Hold-still failure is printed as a reference only. Exit code 0 = pass, 1 = fail, 2 = no N0 results.

## Full-scale order (owner runs it; nothing was launched)

All commands: `python -m world_model.v2.tray_rep.wm.launch_local --stage <s>` (add `--dry-run` to see them). Slurm twins are in `wm/slurm/`.
1. `data`: 3000 episodes per obs mode with `--config piper`, then merge. Both modes (same seeds, same episodes).
2. `oracle`: 2000 TEST states on `data_piper_full`; copied to `data_piper_tray_only`. Check that at least 300 are certified savable.
3. `prep`: the fixed norm; branch files for R and Rc at 300 / 1000 / 3000, seeds 0-2, both modes (36 files, 51.6k branches); CL references.
4. `arms --only N0`: 6 cells (3 seeds x 2 modes), GPUs 4-7, one job per GPU. Then `g1`. Stop here if G1 fails.
5. `arms`: all 42 cells (N0 6, R 18, Rc 18). Done cells are skipped.
6. `g1` again, then compare R and Rc against N0, hold-still and the oracle policy in each `result.json`.

Cost estimate. Measured on CPU: data 0.14 CPU-s per episode, oracle 0.96 CPU-s per state (slew library, loaded host), branches 0.04 CPU-s each,
a WM step 1.5 CPU-s and a value iteration 2.2 CPU-s at batch 256 (so use GPUs). GPU numbers are NOT measured here (GPU 4 was busy);
they come from the vdirect logs on the same cards: WM 15000 steps about 10 min, value 10000 iterations at batch 2048 about 25 min.

| step | CPU | GPU |
|---|---|---|
| data, both modes | 14 CPU-min | none |
| oracle, 2000 states | 32 CPU-min | none |
| norm + 36 branch files (51.6k branches) | about 40 CPU-min | none |
| one arm cell | closed loop + refs: about 3 CPU-min | about 35-40 GPU-min |
| 6 N0 cells | 0.3 CPU-h | about 4 GPU-h |
| all 42 cells | about 2 CPU-h | about 26 GPU-h (about 6.5 h wall on 4 GPUs) |
| total | about 3.5 CPU-h | about 26 GPU-h |

## Point-cloud mode (`pc_est`, 16-D obs): train N0 and Rc 3000 on point-cloud-tracked data

The phase-1b pipeline also runs on the datasets of `perception/gen_pc.py` (ball x, y, vx, vy come from a depth-camera point
cloud, a sphere fit and a Kalman filter; see `perception/README.md`). Everything reads the obs width from the data (16 here).
Code that restores a logged state now builds its env with `env.make_env(meta)`: for `pc_est` this is `PCRepEnv` (so a branch or a
closed-loop state restarts with the sensor and estimator state saved in the snapshot, and the tracker keeps running online);
for `full` and `tray_only` it is `env_from_meta` as before. Used by `branches.py`, `cl_eval_mj.py`, `oracle.py`, `wm/run_arm.py`.
`run_arm --obs-mode` accepts `pc_est`. The labels (oracle) are privileged truth and do not depend on the observation, so the
oracle of `data_piper_full` is reused (`prep link_oracle` copies it after checking that the test episodes are identical).

Owner steps after the shards of `data_piper_pc_head_aim_wrist` are all written (CPU only; `P=` python of mujoco_ur5e, run from the repo root,
`D=/common/users/ss5772/bimanual_data/tray_rep/data_piper_pc_head_aim_wrist`):
```
1. merge:        $P -m world_model.v2.tray_rep.perception.gen_pc --config piper --camera head_aim+wrist --n-eps 3000 --out $D --merge
2. oracle:       $P -m world_model.v2.tray_rep.wm.prep link_oracle --data $D          # copies data_piper_full/oracle/test_v0.{npz,json}; stops if episodes differ
3. norm:         $P -m world_model.v2.tray_rep.wm.prep norm --data $D --workers 16
4. branches:     $P -m world_model.v2.tray_rep.wm.prep branches --data $D --laws Rc --budgets 3000 --seeds 0,1,2 --workers 16
5. refs:         $P -m world_model.v2.tray_rep.wm.prep refs --data $D                 # hold-still + oracle policy on all certified states
6. arms (GPU):   see below
```
(step 6, exact line: `sbatch --array=0-5%6 --export=ALL,TASKS=$PWD/world_model/v2/tray_rep/wm/slurm/tr_tasks_pc.txt world_model/v2/tray_rep/wm/slurm/tr_arm_generic.sbatch`).
Results go to `runs/piper_pc/{N0,Rc}_pc_est_b*_s*/result.json`; the gate for N0 is
`wm.g1 --data $D --setup piper_pc --obs-mode pc_est`.

Things to know. (1) Branches and the closed loop run the camera ray caster and the filter every tick, so they are slower than `full`:
about 3 to 8 ms per tick per core (head_aim+wrist is the dearest rig), i.e. roughly 0.5 s per 100-tick branch. 3000 Rc branches x 3 seeds
is about 1.3 CPU-h; the 2000-branch norm about 20 CPU-min; the refs about 0.5 CPU-h (972 states x 2 policies x 50 ticks). Pre-make them with steps 3 to 5
before the GPU jobs, otherwise each GPU job makes them on 2 workers. (2) Pass `--n-ref` the same everywhere (default 2000 = the cached
norm of step 3). (3) The oracle policy and hold-still failure rates do not depend on the observation, so refs on `pc_est` should
match `data_piper_full`'s (a check on the restore path).
Smoke (2026-10-05, CPU, 125 episodes, 4 workers): `smoke_pc/data_piper_pc_smoke` (3 train / 1 val / 1 test shard; test = the first 25 test episodes of
data_piper_full, `link_oracle --allow-subset`), N0 and Rc b50 with wm-steps 300, value-iters 200 (batch 128), closed loop on 10 certified states.
Smoke numbers only show the plumbing runs; an untrained 300-step WM says nothing about performance.
