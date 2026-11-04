#!/bin/bash
#SBATCH -J bim_qwen36_serve
#SBATCH -w ilab4
#SBATCH --gres=gpu:5000_Blackwell:2
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /common/users/ss5772/bimanual_data/vlm_serve/slurm_%j.out
# Serve Qwen3.6-27B-FP8 (vLLM, TP=2) on 2 ilab4 RTX PRO 5000 Blackwell (user request 11-03).  No robot, no cloud API.
# Weights staged to node-local /freespace (never mmap weights from NFS).  One text smoke request, then hold until
# $SV/STOP_<jobid> exists or the time limit.  Endpoint is node-local 127.0.0.1:$PORT (clients: ssh -L via ilab4, or run on ilab4).
set -u
SV=/common/users/ss5772/bimanual_data/vlm_serve; mkdir -p $SV
NL=/freespace/local/ss5772/vlm_serve; HUB=$NL/hf/hub; SRC=/common/users/ss5772/bimanual_data/hf/hub
M=models--Qwen--Qwen3.6-27B-FP8
hostname; df -h /freespace/local | tail -n 1
mkdir -p $HUB
# the hub keeps weights in a shared hub/blobs/ (symlinked from models--*/blobs), so copy the whole hub and checksum the
# dereferenced snapshot files (11-03: copying only models--* got the links, and an empty-vs-empty checksum "passed")
if [ ! -f $HUB/.staged_ok_v2 ]; then
  rsync -a $SRC/ $HUB/ || { echo "stage failed"; exit 6; }
  (cd $SRC/$M/snapshots && find -L . -type f -exec md5sum {} + | sort -k2) > $NL/src.md5
  (cd $HUB/$M/snapshots && find -L . -type f -exec md5sum {} + | sort -k2) > $NL/dst.md5
  n=$(grep -c safetensors $NL/src.md5); echo "safetensors files: $n"
  [ "$n" -gt 0 ] && cmp -s $NL/src.md5 $NL/dst.md5 && touch $HUB/.staged_ok_v2 || { echo "checksum mismatch or no weights"; exit 7; }
fi
du -shL $HUB/$M/snapshots
cat > $NL/hold_$SLURM_JOB_ID.sh <<'EOH'
echo "$(hostname) $ENDPOINT $MID $(date -Is)" > /common/users/ss5772/bimanual_data/vlm_serve/ENDPOINT_$SLURM_JOB_ID
curl -s -m 600 $ENDPOINT/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MID\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word OK.\"}],\"max_tokens\":2048}" \
  > /common/users/ss5772/bimanual_data/vlm_serve/smoke_$SLURM_JOB_ID.json
echo "smoke exit $? $(date -Is)"
until [ -e /common/users/ss5772/bimanual_data/vlm_serve/STOP_$SLURM_JOB_ID ]; do sleep 30; done
echo "STOP seen $(date -Is)"
EOH
MID=Qwen/Qwen3.6-27B-FP8 NAME=qwen3.6-27b-fp8 HUB=$HUB OUT=$SV CMDS=$NL/hold_$SLURM_JOB_ID.sh BIND_EXTRA=/freespace/local \
  GPU_UTIL=0.85 MAXLEN=20480 MAXSEQ=8 PORT=8795 \
  bash /common/users/ss5772/bimanual/.claude/worktrees/contents-armside2/world_model/v2/vlm/slurm/vlm_serve_run.sh
