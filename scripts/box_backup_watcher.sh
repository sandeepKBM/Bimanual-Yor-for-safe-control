#!/bin/bash
# Rutgers Box cold-archive watcher for bimanual results that exist ONLY on westeros local NVMe.
# /data/local is temporary scratch; Box is the cold copy. COPY only: never deletes local data or Box data.
# Follows VLAism/ur5e_benchmark docs/box_episode_archival.md and pi0_box_drain.sh:
#   - live run tree (SYNC_DIRS): incremental `rclone copy --checksum` every $INTERVAL s,
#     then `rclone check --one-way` (Box SHA1 vs local)
#   - static caches (BUNDLE_DIRS): one-time tar|zstd stream split into 4 GB parts (Box recommends 4-5 GB,
#     max 50 GB), plus a .sha256 per part and a file manifest; parts are uploaded and then checked; local
#     spool parts are removed only after a verified upload
# Only one rclone runs at a time per config (flock on $LOCK): the doc forbids concurrent uploaders.
# Stop: touch $STATE/STOP.   Status: $STATE/status.txt   Log: $STATE/watcher.log
# Restore a bundle: rclone copy $BOX_ROOT/bundles/<name> . && sha256sum -c *.sha256 \
#                   && cat <name>.tar.zst.part-* | zstd -d | tar -xf -
set -u
export RCLONE_PASSWORD_COMMAND="${RCLONE_PASSWORD_COMMAND:-$HOME/.secrets/rclone-pass.sh}"
BOX_ROOT="${BOX_ROOT:-rutgers_box:BIMANUAL_ARCHIVE/westeros_data_local}"
LOCAL="${LOCAL:-/data/local/ss5772}"
STATE="${STATE:-/common/users/ss5772/bimanual_data/box}"
SPOOL="${SPOOL:-$LOCAL/box_spool}"
LOCK="${LOCK:-$HOME/.config/rclone/rutgers_box.lock}"
INTERVAL="${INTERVAL:-1800}"
ZSTD="${ZSTD:-/common/users/ss5772/miniforge3/bin/zstd}"
SYNC_DIRS=(bimanual_runs)
BUNDLE_DIRS=(bimanual_cache/dinowm_cache_zed_px112)
EXCL=(--exclude 'tmp/**' --exclude 'jitcache/**' --exclude '__pycache__/**' --exclude '*.lock')
RC=(rclone --ask-password=false --transfers 2 --checkers 4 --low-level-retries 5 --retries 3 --timeout 120s)

mkdir -p "$STATE" "$SPOOL"
LOG="$STATE/watcher.log"
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
rc() { flock -w 3600 "$LOCK" timeout 3h "${RC[@]}" "$@" >> "$LOG" 2>&1; }
need_reauth() {
  if tail -n 40 "$LOG" | grep -qiE "invalid_grant|unauthori[sz]ed|token expired|couldn't decrypt config"; then
    echo "Box auth failed at $(date -Is); re-run 'rclone config reconnect rutgers_box:' then relaunch" \
      > "$STATE/BOX_REAUTH_REQUIRED"
    log "auth failure -> wrote BOX_REAUTH_REQUIRED, exiting"
    exit 3
  fi
}

bundle() {
  local rel=$1 name remote
  name="$(echo "$rel" | tr / _).tar.zst"
  remote="$BOX_ROOT/bundles/$name"
  [ -e "$STATE/$name.box_verified" ] && return 0
  if [ ! -e "$SPOOL/$name.built" ]; then
    log "bundle $rel -> $SPOOL/$name.part-*"
    rm -f "$SPOOL/$name".part-* "$SPOOL/$name".*.sha256
    tar -C "$LOCAL" -cf - "$rel" | "$ZSTD" -T4 -3 -q | split -b 4G -d -a 3 - "$SPOOL/$name.part-" || {
      log "bundle $rel FAILED (tar/zstd/split)"; return 1; }
    (cd "$SPOOL" && sha256sum "$name".part-* > "$name.sha256")
    (cd "$LOCAL" && find "$rel" -type f -printf '%s\t%TY-%Tm-%Td\t%p\n' | sort -k3) > "$SPOOL/$name.manifest.tsv"
    touch "$SPOOL/$name.built"
  fi
  rc copy "$SPOOL" "$remote" --checksum --include "$name.part-*" --include "$name.sha256" \
    --include "$name.manifest.tsv"
  need_reauth
  if rc check "$SPOOL" "$remote" --one-way --include "$name.part-*" --include "$name.sha256"; then
    cp "$SPOOL/$name.sha256" "$SPOOL/$name.manifest.tsv" "$STATE/"
    date -Is > "$STATE/$name.box_verified"
    rm -f "$SPOOL/$name".part-* "$SPOOL/$name.built"
    log "bundle $rel VERIFIED on Box ($remote); spool parts removed"
  else
    log "bundle $rel check FAILED; spool kept, retry next pass"
  fi
}

log "watcher start pid=$$ box=$BOX_ROOT sync=${SYNC_DIRS[*]} bundles=${BUNDLE_DIRS[*]} interval=${INTERVAL}s"
while [ ! -e "$STATE/STOP" ]; do
  summary=""
  for d in "${SYNC_DIRS[@]}"; do
    rc copy "$LOCAL/$d" "$BOX_ROOT/$d" --checksum "${EXCL[@]}"
    need_reauth
    if rc check "$LOCAL/$d" "$BOX_ROOT/$d" --one-way "${EXCL[@]}"; then st=OK; else st=MISMATCH; fi
    log "sync $d: $st"
    summary+="$d=$st "
  done
  for b in "${BUNDLE_DIRS[@]}"; do
    bundle "$b"
    if [ -e "$STATE/$(echo "$b" | tr / _).tar.zst.box_verified" ]; then summary+="$b=VERIFIED "; else summary+="$b=PENDING "; fi
  done
  echo "last_pass=$(date -Is) pid=$$ $summary" > "$STATE/status.txt"
  for _ in $(seq $((INTERVAL / 60))); do [ -e "$STATE/STOP" ] && break; sleep 60; done
done
log "STOP file seen, exiting"
