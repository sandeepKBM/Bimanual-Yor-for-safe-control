#!/usr/bin/env bash
# CPU stage (westeros or an ilab CPU node; NOT a GPU job): shards -> canonical prepared dir.
#   bash world_model/v2/model/slurm/prep_westeros.sh <shard dir from gen_spill_data.py> [ood rule]
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
SHARDS=${1:?shard dir}; OOD=${2:-"contents.n_balls>10"}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
$PY -m world_model.v2.model.data prep --source "v2:$SHARDS" --out "$PREP" --workers 16 --ood-rule "$OOD"
