#!/bin/bash
# run158b launcher (USER 11-01 rules): westeros CPU only, <= 12 procs, nice 19, 1 thread; if load1 > 72 on 3 checks in a row,
# stop OUR run (by PID) and restart with half the procs (min 3). Finished worlds are kept (atomic per-world files, resume skips them).
cd /common/users/ss5772/bimanual
export PYTHONPATH=$PWD MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMBA_NUM_THREADS=1 LP_NUM_THREADS=1
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
OUT=/data/local/ss5772/pan_cert/run158b
REPO=world_model/v2/nero/results/contact_model_2026-11-01/run158b
mkdir -p $OUT
P=${PROCS:-12}
LOAD_MAX=72
for S in full iid overload; do
  tries=0
  while [ ! -f $OUT/_DONE_$S ] && [ $tries -lt 5 ]; do
    tries=$((tries+1))
    echo "$(date -Is) start set=$S procs=$P load=$(cut -d' ' -f1 /proc/loadavg)" >> $OUT/guard.log
    nice -n 19 $PY -m world_model.v2.nero.pan.cert.contact_falsify --out $OUT --set $S --procs $P >> $OUT/log_$S.txt 2>&1 &
    MAIN=$!
    echo $MAIN > $OUT/main.pid
    hi=0
    while kill -0 $MAIN 2>/dev/null; do
      sleep 60
      L=$(cut -d' ' -f1 /proc/loadavg)
      if awk -v l=$L -v m=$LOAD_MAX 'BEGIN{exit !(l>m)}'; then hi=$((hi+1)); else hi=0; fi
      if [ $hi -ge 3 ] && [ $P -gt 3 ]; then
        KIDS=$(pgrep -P $MAIN)
        echo "$(date -Is) load $L > $LOAD_MAX x3: stopping main $MAIN kids $KIDS, procs $P -> $((P/2 < 3 ? 3 : P/2))" >> $OUT/guard.log
        kill $MAIN $KIDS 2>/dev/null
        sleep 5
        for k in $KIDS; do kill -0 $k 2>/dev/null && kill -9 $k; done
        P=$((P/2 < 3 ? 3 : P/2))
        break
      fi
    done
    wait $MAIN 2>/dev/null
  done
done
$PY -m world_model.v2.nero.pan.cert.contact_falsify --out $OUT --report > $OUT/report_stdout.txt 2>&1
cp $OUT/SUMMARY.txt $OUT/guard.log $REPO/ 2>/dev/null
echo "done $(date -Is)" > $OUT/_ALL_DONE
