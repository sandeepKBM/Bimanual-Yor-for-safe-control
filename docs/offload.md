# Bimanual Compute/Storage Offload (Box Episode Uploader + Slurm Gateway)

This is a thin **adapter** layer for the bimanual project. It does not copy
the upload/archive engine or the Slurm gateway logic — it reuses them, as-is,
from the existing `ur5e_benchmark` project. See "Port decision" below.

## What This System Does

Two independent pieces, both ported as adapters:

1. **Box episode uploader/watcher** — archives completed local episode
   folders into `.tar.zst` bundles, writes SHA256 + manifest sidecars,
   uploads to Rutgers Box using `rclone`, verifies remote presence/size, then
   marks source episodes as uploaded with `.box_uploaded.json`. This never
   touches the network in `--dry-run` mode.
2. **Compute offload / Slurm gateway** — lets you submit bimanual Slurm jobs
   (e.g. the uploader sbatch, or any other job) from a workstation without a
   local `sbatch` (arrakis, westeros) by transparently going through the
   Slurm login node `rlab1` over SSH.

## Why Box Is Archive-Only (cold storage rule)

Box is cold storage, not live rollout output storage. Collectors always
write locally first (`SOURCE_ROOT`). The uploader only processes finalized
episodes (marked `_SUCCESS`) after collection completes. Source deletion is
disabled by default and requires two explicit flags
(`--delete-after-upload` **and** `--i-understand-delete-source`). The same
rule applies to checkpoints (see `box_checkpoint_archive.py` in the source
project) — keep the best checkpoint local, archive the rest only after
verified upload, never make Box the live resume path.

## Port decision: wrapper-reuse (not a copy)

**Coupling check result: LOW.** `ur5e_benchmark.cli.box_episode_uploader`
and `ur5e_benchmark.archive.box_episode_archive` depend only on the Python
stdlib plus one small local helper module
(`ur5e_benchmark.utils.episode_markers`, itself stdlib-only). Importing the
top-level `ur5e_benchmark` package additionally pulls in PyYAML (via
`ur5e_benchmark.config`), which is present in the `safe_env` and `fiper`
conda envs — both were verified to import the uploader module successfully:

```bash
PYTHONPATH=/common/users/ss5772/VLAism/ur5e_benchmark/src \
  conda run -n safe_env python -c "import ur5e_benchmark.cli.box_episode_uploader"
```

Because coupling is low, we did **not** extract/copy any engine code into
`bimanual/`. Instead:

- `bimanual/offload/config.sh` sets bimanual-specific defaults
  (`SOURCE_ROOT`, `SPOOL_ROOT`, `RCLONE_REMOTE`, `RUN_NAME`,
  `COLLECTOR_NAME`) and points `PYTHONPATH` at the read-only source tree
  `/common/users/ss5772/VLAism/ur5e_benchmark/src`.
- `bimanual/scripts/offload/run_box_episode_uploader.sh` sources that
  config and calls `python -m ur5e_benchmark.cli.box_episode_uploader ...`
  with bimanual arguments. This mirrors the original
  `scripts/run_box_episode_uploader.sh` in `ur5e_benchmark` almost line for
  line, just re-rooted.
- `bimanual/scripts/offload/watch_box_episode_uploader.sh`,
  `slurm_box_episode_uploader.sbatch`, `slurm_gateway.sh`, and
  `submit_via_slurm_gateway.sh` are likewise adapted wrappers, not copies of
  engine logic (the gateway script *is* the reusable logic, ported with only
  the `REPO`/`REPO_ROOT` default changed to the bimanual project path;
  `SLURM_GATEWAY_HOST` still defaults to `rlab1`).

No file under `/common/users/ss5772/VLAism/ur5e_benchmark` was modified.

### Known constraint: `--collector-name` is a closed enum

`box_episode_uploader.py`'s `--collector-name` argparse flag only accepts
`openvla | openpi | ball | t5 | auto | unknown`. `"yor_bimanual"` is **not**
a legal value there, and we did not edit the source CLI to add it. The
adapter instead:

- defaults `COLLECTOR_NAME` to `unknown`,
- defaults `RUN_NAME` to `yor_bimanual` (a free-form string, unrestricted),
  which is what actually appears in every archive/remote path
  (`{run_name}__{collector_name}__...`).

If you want a different (legal) collector name, override `COLLECTOR_NAME`
with one of the enum values above.

## Bimanual Defaults

Set in `bimanual/offload/config.sh` (override any of these via env var
before invoking the scripts):

| Variable | Default |
|---|---|
| `SOURCE_ROOT` | `/common/users/ss5772/bimanual/episodes` |
| `SPOOL_ROOT` | `/common/users/ss5772/bimanual/box_spool` |
| `RCLONE_REMOTE` | `rutgers_box:UR5E_ARCHIVE/bimanual` (new subpath, separate from the `ur5e_benchmark` archive tree) |
| `RUN_NAME` | `yor_bimanual` |
| `COLLECTOR_NAME` | `unknown` |
| `UR5E_BENCHMARK_SRC` | `/common/users/ss5772/VLAism/ur5e_benchmark/src` |

## Auth: not in this repo

Auth material is **never** stored in this repo. The adapter only references
existing secrets by path, matching the existing `ur5e_benchmark` convention:

- `RCLONE_CONFIG` → `$HOME/.config/rclone/rclone.conf` (if present)
- `RCLONE_PASSWORD_COMMAND` → `$HOME/.secrets/rclone-pass.sh` (if present)

If those don't exist yet on this machine, set them up **once, manually,
outside of any repo** (see
`/common/users/ss5772/VLAism/ur5e_benchmark/scripts/setup_rclone_box_auth_template.sh`
and `/common/users/ss5772/VLAism/ur5e_benchmark/docs/box_episode_archival.md`
for the exact steps: `rclone config`, create a remote named `rutgers_box`,
`rclone config encryption set`, then test with
`rclone lsd rutgers_box: --ask-password=false`). This adapter does not
create, print, or commit any secret value.

## How To Run

### Dry run (safe, offline — recommended first)

```bash
SOURCE_ROOT=/common/users/ss5772/bimanual/episodes \
SPOOL_ROOT=/common/users/ss5772/bimanual/box_spool \
RUN_NAME=yor_bimanual \
COLLECTOR_NAME=unknown \
RCLONE_REMOTE=rutgers_box:UR5E_ARCHIVE/bimanual \
DRY_RUN=1 \
bash /common/users/ss5772/bimanual/scripts/offload/run_box_episode_uploader.sh
```

`--dry-run` returns a JSON report of what *would* be archived/uploaded. It
never calls `tar`, `rclone`, or touches the network — verified by both
reading the engine source (`_dry_run_report` runs before any rclone/tar
invocation) and an actual smoke run (see below).

### Real upload (touches the network — only run this deliberately)

```bash
export RCLONE_PASSWORD_COMMAND="$HOME/.secrets/rclone-pass.sh"

SOURCE_ROOT=/common/users/ss5772/bimanual/episodes \
SPOOL_ROOT=/common/users/ss5772/bimanual/box_spool \
RUN_NAME=yor_bimanual \
COLLECTOR_NAME=unknown \
RCLONE_REMOTE=rutgers_box:UR5E_ARCHIVE/bimanual \
BW_LIMIT=30M \
bash /common/users/ss5772/bimanual/scripts/offload/run_box_episode_uploader.sh
```

### Start the watcher (infinite poll loop — never start this from an assistant session)

```bash
SLEEP_MINUTES=30 \
SOURCE_ROOT=/common/users/ss5772/bimanual/episodes \
SPOOL_ROOT=/common/users/ss5772/bimanual/box_spool \
RUN_NAME=yor_bimanual \
RCLONE_REMOTE=rutgers_box:UR5E_ARCHIVE/bimanual \
nohup bash /common/users/ss5772/bimanual/scripts/offload/watch_box_episode_uploader.sh \
  > /common/users/ss5772/bimanual/box_spool_watch.log 2>&1 &
```

Launch this only when you deliberately want continuous draining of
`SOURCE_ROOT`; it runs `while true` with a `sleep` between passes and will
not exit on its own. Prefer `tmux`/`nohup`/the Slurm sbatch wrapper over
running it in a foreground shell.

### Submit an offload Slurm job via the gateway (from arrakis/westeros)

```bash
RUN_NAME=yor_bimanual \
SOURCE_ROOT=/common/users/ss5772/bimanual/episodes \
SPOOL_ROOT=/common/users/ss5772/bimanual/box_spool \
RCLONE_REMOTE=rutgers_box:UR5E_ARCHIVE/bimanual \
bash /common/users/ss5772/bimanual/scripts/offload/submit_via_slurm_gateway.sh \
  scripts/offload/slurm_box_episode_uploader.sbatch
```

This detects whether the current host has a local `sbatch`; if not
(arrakis/westeros), it SSHes to `rlab1` (override with
`SLURM_GATEWAY_HOST`/`SLURM_GATEWAY_USER`) and runs `sbatch` there against
the bimanual repo. You can also `source
scripts/offload/slurm_gateway.sh` directly to get `slurm_gateway_squeue`,
`slurm_gateway_scancel`, `slurm_gateway_sacct`, `slurm_gateway_sinfo` helpers
for any other bimanual Slurm job, not just the uploader.

## Restore

`box_episode_restore.py` in the source project is a standalone CLI (only
stdlib deps) that downloads an archive + checksum (+ optional manifest),
verifies the checksum, and extracts. No bimanual-specific wrapper was
created for it since it takes fully explicit paths as CLI args; call it the
same way, with the same `PYTHONPATH`:

```bash
PYTHONPATH=/common/users/ss5772/VLAism/ur5e_benchmark/src python \
  -m ur5e_benchmark.cli.box_episode_restore \
  --archive-path /tmp/yor_bimanual__unknown__episodes_000001_000001.tar.zst \
  --checksum-path /tmp/yor_bimanual__unknown__episodes_000001_000001.tar.zst.sha256 \
  --manifest-path /tmp/yor_bimanual__unknown__episodes_000001_000001.tar.zst.manifest.json \
  --output-root /tmp/restored_episodes
```

## Smoke Test (already run, offline only)

A tiny fake episode directory was created at
`bimanual/artifacts/offload_smoke/episodes/ep_000001/` (`_SUCCESS`,
`summary.json`, `metadata.json` — no real data) and a `--dry-run` was
executed against it via the wrapper script in the `safe_env` conda env with
`MIN_AGE_SECONDS=0`. It correctly discovered 1 eligible episode, planned 1
batch, and printed the remote paths it *would* use — with no archive file
created, no `rclone`/`tar` invocation, and no network call. Rerun with:

```bash
SOURCE_ROOT=/common/users/ss5772/bimanual/artifacts/offload_smoke/episodes \
SPOOL_ROOT=/common/users/ss5772/bimanual/artifacts/offload_smoke/spool \
RUN_NAME=yor_bimanual_smoke \
COLLECTOR_NAME=unknown \
RCLONE_REMOTE=rutgers_box:UR5E_ARCHIVE/bimanual \
MIN_AGE_SECONDS=0 \
DRY_RUN=1 \
PYTHON_BIN=python \
conda run -n safe_env bash /common/users/ss5772/bimanual/scripts/offload/run_box_episode_uploader.sh
```

## Self-Healing Behavior (inherited from the reused engine, unchanged)

- Network disconnects: uploader retries with backoff.
- Interrupted uploads: rerun uploader; pending spool archives are retried first.
- Failed batches: source data remains local and is never deleted by default.
- Slurm timeout/session death: rerun uploader.
- Invalid token / MFA churn: uploader writes `.box_reauth_required.json` in
  `SOURCE_ROOT` and exits with reconnect instructions
  (`rclone config reconnect rutgers_box:`).

## Security and Operational Warnings

- **Auth is in `~/.secrets`, not in this repo.** No rclone config, tokens,
  or passwords live under `bimanual/`. Reference them only by path/env var.
- Do not run multiple uploader processes against the same spool tree with
  the same rclone config.
- Do not copy a writable rclone token/config across machines and use
  concurrently.
- Do not commit secrets/tokens/passwords to git.
- Do not use Box as a live filesystem output target; it is archive-only.
- Source deletion is disabled by default and requires explicit two-flag
  acknowledgement (`--delete-after-upload` + `--i-understand-delete-source`).
- Never start `watch_box_episode_uploader.sh` unattended without first
  validating a `--dry-run` and, if doing a real upload, confirming
  `RCLONE_REMOTE` points at the intended `UR5E_ARCHIVE/bimanual` subpath
  (not the `ur5e_benchmark` archive tree).

## Rollback

Everything created by this port lives under these paths only; nothing in
`ur5e_benchmark` or `bimanual/YOR` was touched:

```bash
rm -rf /common/users/ss5772/bimanual/offload \
       /common/users/ss5772/bimanual/scripts/offload \
       /common/users/ss5772/bimanual/docs/offload.md \
       /common/users/ss5772/bimanual/artifacts/offload_smoke
```
