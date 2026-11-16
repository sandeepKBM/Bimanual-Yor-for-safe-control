#!/bin/bash
# CB arm + conservatism evaluation: CPU ONLY (no GPU), 8 workers at nice 10, pinned threads. SIMULATION / SAVED DATA
# ONLY. Reads the ilab grid's model files read-only; writes only under $VD_ROOT/cb. Resumable: any output with a
# ._DONE sentinel is skipped, so re-running this script later picks up grid runs that finished after the first pass.
#   setsid nohup bash world_model/v2/vdirect/cb_jobs.sh > $VD_ROOT/cb/logs/jobs.log 2>&1 < /dev/null &
source /common/users/ss5772/bimanual/world_model/v2/vdirect/env.sh
export CUDA_VISIBLE_DEVICES=""
CB=$VD_ROOT/cb; G=/common/users/ss5772/bimanual_data/vdirect_ilab/runs
mkdir -p $CB/cl $CB/cons $CB/logs
echo "[cb_jobs] start $(date -u +%FT%TZ) pid=$$ host=$(hostname)"
run() {   # run <out.npz> <cb_eval args...>
  local out=$1; shift
  [ -f "$out._DONE" ] && return 0
  nice -n 10 $PY -m world_model.v2.vdirect.cb_eval "$@" --out "$out" > "$CB/logs/$(basename "$out" .npz).log" 2>&1
  echo "EXIT $? $(basename "$out") $(date -u +%FT%TZ)" >> "$CB/logs/exit_codes.txt"
}
cons_lane() {   # cons_lane <run names...>  (skips grid runs that are not finished yet)
  for r in "$@"; do
    [ -f "$G/$r/_DONE" ] || { echo "SKIP-NOT-DONE $r" >> "$CB/logs/exit_codes.txt"; continue; }
    run "$CB/cons/$r.npz" cons --run "$G/$r"
  done
}
# 6 workers: robust-fallback closed loop (secondary CBr), 3 seeds x 2 state halves
for s in 0 1 2; do for k in 0 1; do
  run "$CB/cl/robust_s${s}_part${k}.npz" cl --run "$G/B0_b0_s$s" --policy robust --part $k --nparts 2 &
done; done
# worker 7: CB (paper rule) closed loop, then B0 conservatism (CB/CBo/N0/M/P), then half of the other runs
RUNS=$(cd $G && ls -d R_b* V_b* U_b* Vn_b* D_b* 2>/dev/null | sort)
A=$(echo "$RUNS" | awk 'NR%2==1'); B=$(echo "$RUNS" | awk 'NR%2==0')
( for s in 0 1 2; do run "$CB/cl/nominal_s${s}_part0.npz" cl --run "$G/B0_b0_s$s" --policy nominal --part 0 --nparts 1; done
  cons_lane B0_b0_s0 B0_b0_s1 B0_b0_s2 $A ) &
# worker 8: the other half
( cons_lane $B ) &
wait
date -u +%FT%TZ > $CB/_DONE_JOBS
echo "[cb_jobs] end $(date -u +%FT%TZ)"
