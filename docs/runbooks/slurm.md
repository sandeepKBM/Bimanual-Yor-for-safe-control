# Slurm rule: generous `--time`, but self-release the moment work finishes

> Status (2026-11-05): CURRENT as a general rule for writing sbatch scripts. The examples (FLOQ, wm_train, hj_reachability) are from the dropped learned-WM work. Who may submit where: AGENTS.md and the slurm-gpu-allocation skill; check the user's current compute rules before any submit.

Memory: `slurm-generous-time-autorelease`. Root-caused 2026-09-06 from FLOQ
job 272949, which was CANCELLED at its `--time=02:00:00` limit while still
mid-training (dataset build alone took ~13 min; training on the real
4000-episode dataset needs much longer than the 2h budgeted).

## The rule

1. **Set a generous `--time` ceiling** on every sbatch -- this is a safety
   net, not a target. On this cluster the `unlimited` partition's own
   `TIMELIMIT` is `infinite`, so there is no cluster-side reason to under-ask;
   default to `12:00:00` (or longer for known-slow jobs) unless you have a
   tight, measured upper bound.
2. **Make the job command finite.** A Slurm batch job only runs until
   `--time` if its script hangs or trains open-ended. Every training
   entrypoint must have a real stopping condition (max epochs/steps) so the
   process exits, the sbatch script reaches its last line, and Slurm frees
   the node immediately -- not after sitting idle until the timer fires.
   - `archive/hj_reachability/floq/configs/bimanual_floq_v1.yaml`: `epochs: 15`,
     `max_train_batches: 0` (0 = all batches, not unbounded) -- already
     finite, confirmed by the local smoke completing on its own.
   - `archive/world_model/wm_train.sbatch`, `archive/hj_reachability/train.sbatch`: same
     rule applies -- verify their training configs have a defined
     epoch/step cap before raising `--time` on them.
3. **Auto-release belt-and-suspenders.** At the end of every sbatch script,
   after the real work, write a `_DONE` sentinel file and let the script
   exit:
   ```bash
   mkdir -p "${OUTPUT_DIR}"
   date -u +"%Y-%m-%dT%H:%M:%SZ" > "${OUTPUT_DIR}/_DONE"
   ```
   For any job whose command *might* hang despite (2), run
   `scripts/offload/slurm_autorelease.sh <job_id> <sentinel_path>` in the
   background from a place with squeue/scancel (or gateway) access -- it
   polls for the sentinel and `scancel`s the job the instant it appears,
   so the node is freed without waiting for `--time` even in the hang case.

## Applying it to a new sbatch script

- Pick a `--time` that is generous relative to your real estimate (not the
  cluster max) -- e.g. 2-3x your expected wall time, or `12:00:00` as a
  reasonable repo default.
- Confirm the training/eval command has a hard stop (epoch/step cap, not
  "run forever").
- Add the `_DONE` sentinel write as the script's last statement.
- Optionally launch `slurm_autorelease.sh` alongside submission for extra
  safety on long/unattended runs.

## Submitting from a host with no local `sbatch` (westeros/arrakis)

Use `scripts/offload/submit_via_slurm_gateway.sh <script.sbatch>` -- it now
calls `sbatch` on the gateway host (`rlab1` by default) directly (fixed
2026-09-06; it previously `bash`-ran the script on the login node, so
nothing was ever enqueued -- see archive/hj_reachability/FLOQ_SETUP.md for the incident). It also
strips an unsupported `#SBATCH --cpus-per-task` line automatically (this
cluster's `unlimited` partition rejects that flag).

Status/cancel from a workstation without local Slurm binaries:
```bash
source scripts/offload/slurm_gateway.sh
slurm_gateway_squeue -u "$USER"
slurm_gateway_scancel <job_id>
```
