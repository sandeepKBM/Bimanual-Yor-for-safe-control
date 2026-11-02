#!/bin/bash
# jenga request 11-01 (stricter than the user's 72): if westeros load1 > 40 on 3 checks in a row, pause (SIGSTOP) half of OUR running
# pool workers (keep >= 3 running); when load1 < 32 on 3 checks in a row, resume (SIGCONT) them all. Our PIDs only: workers are the
# spawn children of the main PID in main.pid. Exits (after resuming everything) when the run is done.
O=/data/local/ss5772/pan_cert/run158b
HI=40; LO=32; hi=0; lo=0
log() { echo "$(date -Is) $*" >> $O/guard40.log; }
workers() { for k in $(pgrep -P "$(cat $O/main.pid 2>/dev/null)" 2>/dev/null); do grep -qa spawn_main /proc/$k/cmdline 2>/dev/null && echo $k; done; }
state() { awk '{print $3}' /proc/$1/stat 2>/dev/null; }
log "guard40 start"
while [ ! -f $O/_ALL_DONE ]; do
  sleep 60
  L=$(cut -d' ' -f1 /proc/loadavg)
  if awk -v l=$L -v m=$HI 'BEGIN{exit !(l>m)}'; then hi=$((hi+1)); else hi=0; fi
  if awk -v l=$L -v m=$LO 'BEGIN{exit !(l<m)}'; then lo=$((lo+1)); else lo=0; fi
  run=(); stp=()
  for k in $(workers); do [ "$(state $k)" = "T" ] && stp+=($k) || run+=($k); done
  if [ $hi -ge 3 ] && [ ${#run[@]} -gt 3 ]; then
    n=$(( ${#run[@]} / 2 )); [ $(( ${#run[@]} - n )) -lt 3 ] && n=$(( ${#run[@]} - 3 ))
    kill -STOP ${run[@]:0:$n}; log "load $L > $HI x3: paused $n of ${#run[@]} running (${run[*]:0:$n})"; hi=0
  elif [ $lo -ge 3 ] && [ ${#stp[@]} -gt 0 ]; then
    kill -CONT ${stp[@]}; log "load $L < $LO x3: resumed ${#stp[@]} (${stp[*]})"; lo=0
  fi
done
for k in $(workers); do kill -CONT $k 2>/dev/null; done
log "guard40 exit (run done)"
