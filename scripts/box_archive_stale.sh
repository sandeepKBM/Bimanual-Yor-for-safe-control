#!/bin/bash
# One-shot Rutgers Box archive of the data made on the OLD physics (before the pin_cup fix of 2026-10-08;
# docs/status/pin_cup_fix_2026-10-08.md, docs/status/stale_inventory_2026-10-08.md). User request 2026-10-08:
# "delete that old cache and data stuff or send it all to BOX storage" -> archive to Box, verify, THEN delete local bulk data.
# Same bundle format and lock as scripts/box_backup_watcher.sh: tar | zstd | split 4 GB parts + .sha256 + file manifest,
# rclone copy --checksum, rclone check --one-way; one rclone at a time (flock on the watcher's lock).
# A local path is deleted ONLY if its bundle is verified on Box (marker file) AND it is listed in DELETE_NOW.
# DEFER items are archived but kept: running experiments still read them; delete them later with DELETE_DEFERRED=1.
# Restore: rclone copy $BOX_ROOT/<name> . && sha256sum -c <name>.sha256 && cat <name>.part-* | zstd -d | tar -xf -
set -u
export RCLONE_PASSWORD_COMMAND="${RCLONE_PASSWORD_COMMAND:-$HOME/.secrets/rclone-pass.sh}"
BOX_ROOT="${BOX_ROOT:-rutgers_box:BIMANUAL_ARCHIVE/stale_pre_pinfix_2026-10-08/bundles}"
STATE="${STATE:-/common/users/ss5772/bimanual_data/box_stale}"
SPOOL="${SPOOL:-/data/local/ss5772/box_spool_stale}"
LOCK="${LOCK:-$HOME/.config/rclone/rutgers_box.lock}"
ZSTD="${ZSTD:-/common/users/ss5772/miniforge3/bin/zstd}"
DELETE_DEFERRED="${DELETE_DEFERRED:-0}"
RC=(rclone --ask-password=false --transfers 2 --checkers 4 --low-level-retries 5 --retries 3 --timeout 120s)
BD=/common/users/ss5772/bimanual_data
TR=$BD/tray_rep

# name | parent dir | relative dir | delete policy: ALL | sub:<a,b,...> | DEFER | KEEP
ITEMS=(
  "spill_v2|/common/users/ss5772/datasets/bimanual_wm_v2|spill_v2|ALL"
  "jobtmp_wm2|$BD/jobtmp|wm2|sub:gen_v2_64"
  "bead_replication|$BD|bead_replication|sub:data,smoke_out,tmp"
  "wm_audit|$BD|wm_audit|KEEP"
  "tray_rep_data_piper_full|$TR|data_piper_full|DEFER"
  "tray_rep_data_piper_trk_full|$TR|data_piper_trk_full|DEFER"
  "tray_rep_data_piper_tray_only|$TR|data_piper_tray_only|DEFER"
  "tray_rep_data_piper_pc_head_aim_wrist|$TR|data_piper_pc_head_aim_wrist|DEFER"
  "tray_rep_runs|$TR|runs|DEFER"
  "tray_rep_tmp|$TR|tmp|DEFER"
  "tray_rep_smoke|$TR|smoke|DEFER"
  "tray_rep_smoke1b|$TR|smoke1b|DEFER"
  "tray_rep_smoke_pc|$TR|smoke_pc|DEFER"
  "tray_rep_perception|$TR|perception|DEFER"
  "tray_rep_results_bridge|$TR|bridge|KEEP"
  "tray_rep_results_boed|$TR|boed|KEEP"
  "tray_rep_results_boed_p7|$TR|boed_p7|KEEP"
  "tray_rep_results_boed_p8|$TR|boed_p8|KEEP"
  "tray_rep_results_boed_f5|$TR|boed_f5|KEEP"
  "tray_rep_results_vlmprior|$TR|vlmprior|KEEP"
  "tray_rep_results_robust|$TR|robust|KEEP"
  "tray_rep_results_sets|$TR|sets|KEEP"
)

mkdir -p "$STATE" "$SPOOL"
LOG="$STATE/archive.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
rc() { flock -w 7200 "$LOCK" timeout 4h "${RC[@]}" "$@" >> "$LOG" 2>&1; }

del_tree() {   # find -delete (rm -rf is blocked by policy); refuses anything outside the expected roots
  local p=$1
  case "$p" in /common/users/ss5772/bimanual_data/*|/common/users/ss5772/datasets/bimanual_wm_v2/*|/data/local/ss5772/bimanual_cache/*) ;;
    *) log "REFUSE delete outside roots: $p"; return 1;; esac
  [ -e "$p" ] || { log "already gone: $p"; return 0; }
  log "delete $p ($(du -sh "$p" 2>/dev/null | cut -f1))"
  find "$p" -depth -delete && log "deleted $p"
}

bundle() {   # name parent rel -> 0 if verified on Box
  local name=$1 parent=$2 rel=$3 f="$1.tar.zst" remote="$BOX_ROOT/$1"
  [ -e "$STATE/$f.box_verified" ] && return 0
  [ -d "$parent/$rel" ] || { log "missing $parent/$rel, skip"; return 1; }
  if [ ! -e "$SPOOL/$f.built" ]; then
    log "bundle $parent/$rel -> $SPOOL/$f.part-*"
    find "$SPOOL" -maxdepth 1 -name "$f.part-*" -delete
    tar -C "$parent" -cf - "$rel" | "$ZSTD" -T4 -3 -q | split -b 4G -d -a 3 - "$SPOOL/$f.part-" || { log "bundle $name FAILED"; return 1; }
    (cd "$SPOOL" && sha256sum "$f".part-* > "$f.sha256")
    (cd "$parent" && find "$rel" -type f -printf '%s\t%TY-%Tm-%Td\t%p\n' | sort -k3) > "$SPOOL/$f.manifest.tsv"
    touch "$SPOOL/$f.built"
  fi
  rc copy "$SPOOL" "$remote" --checksum --include "$f.part-*" --include "$f.sha256" --include "$f.manifest.tsv"
  if rc check "$SPOOL" "$remote" --one-way --include "$f.part-*" --include "$f.sha256"; then
    cp "$SPOOL/$f.sha256" "$SPOOL/$f.manifest.tsv" "$STATE/"
    date -Is > "$STATE/$f.box_verified"
    find "$SPOOL" -maxdepth 1 \( -name "$f.part-*" -o -name "$f.built" \) -delete
    log "VERIFIED on Box: $name ($remote)"
    return 0
  fi
  log "check FAILED: $name (spool kept)"; return 1
}

# 0. the DINO-WM cache: already a verified bundle from the watcher (2026-10-01). Delete the local copy only if its file list
#    and sizes still equal that bundle's manifest.
DW=/data/local/ss5772/bimanual_cache/dinowm_cache_zed_px112
WM=/common/users/ss5772/bimanual_data/box/bimanual_cache_dinowm_cache_zed_px112.tar.zst
if [ -e "$WM.box_verified" ] && [ -d "$DW" ]; then
  if diff <(cut -f1,3 "$WM.manifest.tsv") <(cd /data/local/ss5772 && find bimanual_cache/dinowm_cache_zed_px112 -type f -printf '%s\t%p\n' | sort -k2) > /dev/null; then
    log "dinowm cache unchanged since its verified Box bundle"; del_tree "$DW"
  else
    log "dinowm cache CHANGED since the bundle; not deleting"
  fi
fi

for it in "${ITEMS[@]}"; do
  IFS='|' read -r name parent rel pol <<< "$it"
  if bundle "$name" "$parent" "$rel"; then
    case "$pol" in
      ALL) del_tree "$parent/$rel" ;;
      sub:*) IFS=',' read -ra subs <<< "${pol#sub:}"; for s in "${subs[@]}"; do del_tree "$parent/$rel/$s"; done ;;
      DEFER) if [ "$DELETE_DEFERRED" = 1 ]; then del_tree "$parent/$rel"; else log "archived, local kept (deferred): $parent/$rel"; fi ;;
      KEEP) log "archived, local kept: $parent/$rel" ;;
    esac
  fi
done
log "DONE"
date -Is > "$STATE/DONE"
