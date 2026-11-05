# Slurm rules (copied from vendor: ur5e_benchmark `.agents/skills/slurm-rules`)

> Status (2026-11-05): SUPERSEDED for node choice: the 2026-08 node allow-list below ("ALL ilab and rlab") is stale. Current cluster rules: AGENTS.md and the slurm-gpu-allocation skill (8 GPUs + 1 node per user). HISTORY: verbatim vendor copy of 2026-09-06.

Copied verbatim 2026-09-06 from
`/common/users/ss5772/VLAism/ur5e_benchmark/.agents/skills/slurm-rules/SKILL.md`
for the FLOQ multi-GPU DDP relaunch (job replacing 273272). We follow these
conventions when submitting from `bimanual` via
`scripts/offload/submit_via_slurm_gateway.sh` / `scripts/offload/slurm_gateway.sh`
(both already ported from the same vendor repo -- see their file headers).

We don't have our own `.agents/skills/` or `.claude/skills/` directory in
`bimanual` yet, so this lives under `docs/runbooks/` per the mission
instruction ("our skills dir if we have one, else docs/runbooks/").

**Known internal inconsistency spotted while applying this (2026-09-06,
flagging for whoever reads this next):** the vendor file's own
"NODE ALLOW/OFF-LIMITS" section is explicitly marked the *single source of
truth* and says `ALLOWED (user, 2026-08-07): ALL ilab (1-4) and ALL rlab
(1-7) nodes`, with only the non-ilab/rlab reserved nodes (`first, traceai,
ra6kb1, ra6kb2, aries, cm01-cm04, trustai`) off-limits. But a few lines later,
inside the single-GPU node *preference-order* bullet, there's a leftover
note: "a4000 (rlab4) still sits on an OFF-LIMITS node -- do not use." That
note contradicts the source-of-truth section right above it in the same
file/date, and reads like a stale carry-over from when OpenVLA-7B's >16GB
VRAM need ruled out a4000/rlab4 (context: "a4000 previously left OpenVLA-7B
with a KV-cache OOM"). FLOQ's model here is a small MLP (a few 256-wide,
3-layer heads) with batch=256 tensors -- it fits trivially in a4000's 16GB,
and our own currently-running single-GPU FLOQ job (273272) is already on
rlab4. We treat rlab4 as ALLOWED for this workload per the source-of-truth
list, not the stale VRAM-fit note, and record that judgment call here rather
than silently overriding it.

---

## Core Rules

- Read `AGENTS.md` and `docs/runbooks/workstation_and_slurm.md` first (vendor
  repo paths; see vendor repo root).
- Classify the host before you act:
  - workstation (Arrakis/Westeros): do not run local `sbatch`
  - Slurm login (`rlab1`): submit to `rlab*`
  - compute node: run only inside the allocation
- Prefer canonical gateway wrappers over ad hoc submission commands.
- **NODE ALLOW/OFF-LIMITS** (as of 2026-08-07, vendor source of truth):
  - The real gate is VRAM, not the node list. VLA work needs >16GB VRAM
    (OpenVLA-7B does NOT fit a4000/16GB or 2080Ti/11GB; pi0 fits in <=16GB
    so any allowed GPU works for it).
  - **ALLOWED: ALL `ilab` (1-4) and ALL `rlab` (1-7) nodes** -- pick by GPU
    fit + freeness, not node identity. (rlab1 is also the Slurm login/gateway.)
  - **OFF-LIMITS without explicit approval:** only `first, traceai, ra6kb1,
    ra6kb2, aries, cm01-cm04, trustai`.
  - ilab4 (8x RTX PRO 5000 Blackwell, 48GB) is approved; smoke-test with
    `attn_implementation: eager` first (known SDPA segfault on that card for
    unrelated OpenVLA work -- irrelevant to FLOQ's plain MLP/PyTorch ops).
  - Query LIVE GPU type + free capacity before choosing a target -- don't
    trust any hardcoded node table. We used:
    `scontrol show node` (via the gateway) and read `Gres=`/`AllocTRES` per
    node, since this cluster's `sinfo -o '%G'` custom format printed empty
    columns for us.
- **Never leave a job PENDING on a node that's full.** Check free-GPU counts
  before choosing `--nodelist`, pick an idle node with the required GPU type.
- **Cluster-wide GPU ceiling: max 12 GPUs total per user/campaign at once,
  max 6 GPUs on any single server/rack.**
- **Do not pass `--cpus-per-task`/`--ntasks`/`--mincpus`** on this cluster's
  `unlimited` partition -- rejected outright. (`submit_via_slurm_gateway.sh`
  already strips a `--cpus-per-task` line automatically if present.)
- **If the job needs more than 1 GPU, do not auto-pick a multi-GPU node --
  confirm the node/partition with the user first** (multi-GPU is a bigger
  shared-resource commitment). For this FLOQ relaunch, the user's own mission
  instructions explicitly directed "maximize GPU use across rlab/ilab nodes,
  multi-GPU single node" -- treated as that confirmation; node/GPU-count
  choice (rlab4, 3x a4000) is documented in the mission report.
- Run `--help`, a dry-run, or a smoke path before any long queue launch.
  (We ran a 2-GPU torchrun smoke test locally on westeros -- 8x local
  2080Ti -- with `--epochs 1 --max-train-batches 5` before submitting the
  real multi-hour job.)
- Retarget a `PENDING` job only if it has not started and a compatible idle
  node exists.
- Verify artifacts on disk, not just Slurm job state.

## Common Checks

- Use `squeue`/`sinfo`/`scontrol` (via `scripts/offload/slurm_gateway.sh`'s
  `slurm_gateway_squeue`/`slurm_gateway_exec` helpers from a workstation with
  no local Slurm binaries) to inspect queue state and node availability.
- Multi-stage submit scripts using `set -euo pipefail` can abort the entire
  batch if one stage's submission call isn't guarded -- check the actual
  submitted-job list after a multi-stage submit.

## Escalate Early

- Stop if a command would delete protected data or change shared launch
  behavior.
- Resolve ambiguous host or partition selection before submitting jobs.
- A large, exclusive, long-running job (multi-GPU, full-node, many hours) is
  a bigger commitment than a quick validation job -- confirm the target node
  with the user before submitting.

## Slurm-generous-time-autorelease (bimanual-local rule, see `slurm.md` in
this same directory)

Generous `--time` ceiling + a finite training loop (fixed epoch count) +
`_DONE` sentinel written as the script's last statement, so the node
deallocates immediately once training finishes rather than waiting for
`--time`. Kept unchanged for the DDP relaunch.
