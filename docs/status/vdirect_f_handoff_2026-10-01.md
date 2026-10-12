# Handoff: vdirect confirming-falsifier test (addendum F), 2026-10-01

The owner APPROVED the design on 2026-10-01 ("Approved design go hit it").

**Why this handoff exists:** the previous session's auto-mode classifier blocked every shell command and every write
outside the repo, for the rest of that session. Code is written but has NOT been syntax-checked, smoke-tested or run.
A fresh session (or this one, out of auto mode) should run the steps below in order.

**Safety:** simulation and saved data only. No hardware, robot, Arrakis or Jetson. Box is approved for backups only.

## What is written (untracked; rollback = git checkout of the 2 edited files + rm of the new ones)

**`world_model/v2/vdirect/branches.py`:**
- `RandomScorer`: each segment holds one uniformly random fallback-set element.
- `falsify_directed(...)`: the F / F0 arms.

**`world_model/v2/vdirect/run_arm.py`:**
- Arms F, F0, Rr.
- `VD_SRC_ROOT` sets where shared/ + NOM pre-trains are read from (default VD_ROOT, so old behaviour is unchanged).
- The V / Vn / D code path is unchanged except for one net-assignment branch (Rr only).

**New files:**
- `world_model/v2/vdirect/oc_eval.py`: the outcome-conformal audit.
- `world_model/v2/vdirect/analyze_f.py`: KF1 / KF2, diagnostics, guard, OC table, plot. It imports analyze.py and
  analyze_cb.py UNCHANGED.
- `world_model/v2/vdirect/slurm/`:
  - `vf_tasks.txt`: 33 cells.
  - `vf_grid_array.sbatch`: GPU, %12.
  - `vf_cons_array.sbatch`: CPU.
  - `vf_oc_array.sbatch`: GPU, with `OC_LIST`.
  - `vf_oc_grid_runs.txt`: 51 lines.
  - `vf_oc_new_runs.txt`: 33 lines.

The design (arms, criteria KF1 / KF2, diagnostics, guard, OC audit, prior, limitations) is in the PREREG text at the
end of this file.

## Steps

0. Create the output tree:
   `mkdir -p /common/users/ss5772/bimanual_data/vdirect_f/{logs,oc,cb/cons,shared_oc}`.
   Test-write a file there first, because of the hidden /common/users quota.
1. Write the PREREG text below to `/common/users/ss5772/bimanual_data/vdirect_f/PREREG_ADDENDUM_F.md`, verbatim.
2. Static checks:
   `$PY -m py_compile world_model/v2/vdirect/{branches,run_arm,oc_eval,analyze_f}.py`
   (PY = /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python), and `bash -n` on each `vf_*.sbatch`.
3. CPU smoke (CUDA_VISIBLE_DEVICES="", pinned threads, scratch under $CLAUDE_JOB_DIR/tmp). Results are not used.
   - **Source root:** SRC=$scratch/vf_smoke_src, with `shared -> /common/users/ss5772/bimanual_data/vdirect_ilab/shared`
     as a symlink. Run `VD_ROOT=$SRC $PY -m world_model.v2.vdirect.run_arm --arm NOM --seed 0 --member 0 --smoke`.
   - **New arms:** `VD_ROOT=$scratch/vf_smoke VD_SRC_ROOT=$SRC $PY -m world_model.v2.vdirect.run_arm --arm {F,F0,Rr} --budget 30 --seed 0 --smoke`.
     Check rounds.json: hit_rate, n_probe and n_branch add up to 30, and root_kind is set.
   - **Equivalence:** run `--arm D --budget 30 --seed 0 --smoke` with the NEW code and with `git stash` (old code). Same
     CPU and seed, so result.json must be identical; this proves the existing arms are unchanged.
   - **OC:** `VD_OC_CACHE=$scratch/vf_smoke/shared_oc $PY -m world_model.v2.vdirect.oc_eval --run $scratch/vf_smoke/runs/smoke_F_b30_s0 --out $scratch/vf_smoke/oc/smoke_F_b30_s0.json --smoke`.
4. **Freeze:**
   - sha256sum of branches.py, run_arm.py, oc_eval.py, analyze_f.py, slurm/vf_* goes into the freeze block, with the
     UTC time.
   - The addendum's own sha256 goes into PREREG_ADDENDUM_F.sha256.
   - Any code fix after the smoke happens BEFORE the freeze.
5. **Submit** (rlab1 gateway; check node AllocMem first, per docs/runbooks slurm rules):
   - `ssh rlab1 "cd /common/users/ss5772/bimanual && sbatch world_model/v2/vdirect/slurm/vf_grid_array.sbatch"` gives
     job G.
   - `ssh rlab1 "cd ... && sbatch --nice=100 --array=0-50%4 --export=ALL,OC_LIST=world_model/v2/vdirect/slurm/vf_oc_grid_runs.txt world_model/v2/vdirect/slurm/vf_oc_array.sbatch"`
   - `ssh rlab1 "cd ... && sbatch --dependency=afterany:G --array=0-32%8 --export=ALL,OC_LIST=world_model/v2/vdirect/slurm/vf_oc_new_runs.txt world_model/v2/vdirect/slurm/vf_oc_array.sbatch"`
   - `ssh rlab1 "cd ... && sbatch --dependency=afterany:G world_model/v2/vdirect/slurm/vf_cons_array.sbatch"`
6. **Watch:** sentinel files only (runs/*/_DONE, oc/*._DONE, cb/cons/*._DONE). Never `pgrep -f` loops. Retry each
   failed cell once.
7. **Analyse:** `$PY -m world_model.v2.vdirect.analyze_f` writes vdirect_f/analysis/{tables_f.md, summary_f.json,
   falsifier_sample_efficiency.png}. Re-verify the verdicts against the per-seed numbers before reporting.
8. **Box:** launch `scripts/box_backup_watcher.sh` (written, never launched; syntax unchecked) with `setsid nohup`.

## PREREG text (write verbatim to bimanual_data/vdirect_f/PREREG_ADDENDUM_F.md)

> # vdirect addendum F: the confirming falsifier ("predator-prey") — pre-registration
>
> **Written:** 2026-10-01, BEFORE any F / F0 / Rr / R-300 run (smoke runs at toy settings excepted; not used). DRAFT
> until the freeze block is filled (code SHA-256, this file's SHA-256, UTC time); any later change is a listed
> deviation. Simulation only, as PREREG.md. Approved by the owner 2026-10-01.
>
> **1. Claim.** A confirming falsifier fixes the latent-HJ collapse with fewer true-system branches than random
> branching, and better than model-directed (ensemble-uncertainty) selection. It runs the current value's own greedy
> fallback on the TRUE system, keeps only CONFIRMED failures (V(z) > 0 but true margin < 0) as counterexamples, and
> branches broad alternative actions from them. Motivation:
> - PREREG.md killed value-directed selection (K1–K3), which never confirmed anything.
> - The literature check (reports/Coverage versus caution latent safety.md) found that directed data beats random only
>   with truth-confirmed selection (Dallas 2604.13325, RECON 2609.33336, CEGIS / ISAACS-style confirmation).
>
> **2. Fixed components.** Identical to PREREG.md §2: data, kin_wm trainer and 15,000 steps with fixed
> normalization, branch format, final value recipe, closed-loop metric on the same 2000 TEST states (1989 savable),
> and the hierarchical paired bootstrap (10,000). Reused read-only from bimanual_data/vdirect_ilab: shared/, the 9 NOM
> pre-trains and the round-0 probes. Comparators reused unchanged: R {1k, 3k, 10k, 30k}, V / U / Vn / D {1k, 3k, 10k},
> B0 (N0 / M / P), CB.
>
> **3. New arms.** All use V's schedule (NOM 3k, then 3 rounds × (b_r = split_budget(b, 3), add, 4k steps); probe =
> the round-0 probe, then a fresh 4,000-iteration probe; same 60k candidate set, seed 100000 + 1000·s + r). Every
> true-system rollout is one budget unit, including probes.
> - **F, probes** (n_p = round(0.5·b_r)): uniform draws from the top 4·n_p ACCEPTED candidates (V > 0, smallest
>   positive V). The current greedy fallback runs closed loop on the true pendulum (p_greedy = 1 = cl_eval's
>   wm_greedy) for 100 ticks. Confirmed counterexample = true margin < 0 within 50 ticks. Probes are kept as data.
> - **F, branches** (b_r − n_p): roots are the confirmed states, round-robin over a shuffled order. 3–15-tick
>   segments each hold ONE uniformly random fallback-set action (broad, not perturbations). With no failures, the
>   roots are the 10% lowest-min-margin probes ("closest calls", recorded).
> - **F0:** probes only, using the whole b_r (literal predator-prey).
> - **Rr:** make_branches (R's law) per round under V's warm start and schedule, with no probe value. It controls
>   the schedule: at 1k, the round-based arms were ~18–21% vs R 6.8%.
> - **R at 300:** original R recipe, seed s + 1.
>
> **4. Grid.** F {300, 1k, 3k, 10k}; F0 {300, 1k, 3k}; Rr {300, 1k, 3k}; R {300}; seeds {0, 1, 2}; 33 cells. One
> retry; missing → INCOMPLETE, never imputed.
>
> **5. Criteria** (as coded in analyze_f.py).
> - **KF1:** at b ∈ {300, 1k}: mean_R / mean_F ≥ 2 (floor 0.5/1989) AND the 95% CI of R − F lies above 0. KILL if
>   either fails at either budget. These budgets were chosen AFTER seeing the PREREG grid: the literature predicts
>   low-budget gains, and R at 3k is bimodal (one seed at 22%), which makes a 3k CI criterion uninformative. 3k and 10k
>   are diagnostics only.
> - **KF2:** at b ∈ {1k, 3k}, CI(U − F). PASS iff F is significantly better at ≥ 1 budget and U significantly better
>   at neither.
> - **Headline:** survives only if KF1 AND KF2 pass.
> - **Diagnostics** (CIs of X − F): Rr {300, 1k, 3k}; F0 {300, 1k, 3k}; V, D {1k, 3k, 10k}; R {3k, 10k}; U {10k};
>   per-round hit rate and root kind.
> - **Conservatism guard:** false interventions on clearly safe TEST states (cb_eval cons unchanged). FLAG if cons_F −
>   cons_R > 0.5 pp at any budget where both exist.
> - **Secondary OC audit** (oc_eval.py, every run, plus N0 / M / P):
>   - certified set {V ≥ τ};
>   - calibration on 2000 VAL states (cl_eval protocol, seed 1) with the value's own greedy fallback for 50 ticks;
>   - τ candidates from half A (m-th largest V, m = 100…1000 step 50), fixed-sequence on half B, one-sided
>     Clopper-Pearson, ε = 5%, β = 5%;
>   - reported on TEST: certified savable fraction, fail rate among certified, certified clearly-safe fraction, and
>     the same at τ = 0;
>   - expected in advance: N0 certifies little of the savable set;
>   - P's heads were not saved, so they are refit with the run seed (flagged).
>
> **6. Prior.** About 35–40% that the headline passes. Most likely: F ≈ U ≈ R at ≥ 3k; F < Rr at 300–1k if the 1k
> deficit is the schedule; F0 ≈ D. KF1 fail + F < Rr = confirmation helps, but not enough to beat front-loaded random.
>
> **7. Limitations.**
> 1. Confirmation is vacuous on this proxy (truth = labeller).
> 2. Selection compute is reported, not charged; probes are charged.
> 3. Probe states are TRAIN, test states TEST, OC calibration VAL (disjoint).
> 4. 3 seeds, one proxy.
> 5. Same ilab cluster as the comparators; cross-machine float noise is reported as within-noise if it appears.
>
> **8. Compute.** About 18 GPU-h (33 cells × 25–40 min), ~2–3 h wall on the 12-GPU cap; then cons (CPU), OC
> (GPU), analyze_f. Smoke first (CPU, --smoke).
>
> **9. Freeze block.** freeze_utc: <pending>; sha256 of code: <pending>; this file's sha256 →
> PREREG_ADDENDUM_F.sha256.
