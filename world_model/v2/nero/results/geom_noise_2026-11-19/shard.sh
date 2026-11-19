#!/bin/bash
# shard.sh <n_gpus> <procs_per_gpu> <worlds A:B> <out> [extra study args, e.g. --seeds 0,1,2,3 --noise clean,zed2i_lab]
# Worlds are split round-robin over n_gpus*procs_per_gpu processes.  CODE (env) = the code root (default: the worktree).
NG=$1; PP=$2; WR=$3; OUT=$4; shift 4
CODE=${CODE:-/common/users/ss5772/bimanual/.claude/worktrees/basestop-1116}
PY=${PY:-/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python}
A=${WR%%:*}; B=${WR##*:}; NP=$((NG*PP))
mkdir -p "$OUT/logs"
pids=(); k=0
for ((p=0; p<NP; p++)); do
  ws=(); for ((w=A+p; w<B; w+=NP)); do ws+=($w); done
  [ ${#ws[@]} -eq 0 ] && continue
  g=$((p % NG))
  (cd "$CODE" && for w in "${ws[@]}"; do
     OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$g CUDA_VISIBLE_DEVICES=$g PYTHONPATH="$CODE" \
       "$PY" -u -m world_model.v2.nero.results.geom_noise_2026-11-19.study --worlds $w:$((w+1)) --out "$OUT" "$@"
   done) > "$OUT/logs/proc$p.log" 2>&1 &
  pids+=($!); k=$((k+1))
done
fail=0; for p in "${pids[@]}"; do wait $p || fail=$((fail+1)); done
echo "shard done: $k procs, $fail failed, $(ls "$OUT/runs" 2>/dev/null | grep -c '\.json$') run files in $OUT/runs"
[ $fail -eq 0 ]
