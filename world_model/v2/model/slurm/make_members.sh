#!/usr/bin/env bash
# Write one sbatch per ensemble member (env vars do NOT cross the ssh gateway) and submit.
#   bash world_model/v2/model/slurm/make_members.sh train_dino.sbatch 5 [--submit]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); TPL=$1; N=${2:-5}; SUBMIT=${3:-}
for m in $(seq 0 $((N - 1))); do
  f="$HERE/.${TPL%.sbatch}_m$m.sbatch"
  sed -e "s/^MEMBER=0 /MEMBER=$m /" -e "s/--job-name=\(.*\)/--job-name=\1_m$m/" "$HERE/$TPL" > "$f"
  echo "wrote $f"
  if [ "$SUBMIT" = "--submit" ]; then
    bash /common/users/ss5772/bimanual/scripts/offload/submit_via_slurm_gateway.sh "$f"
  fi
done
