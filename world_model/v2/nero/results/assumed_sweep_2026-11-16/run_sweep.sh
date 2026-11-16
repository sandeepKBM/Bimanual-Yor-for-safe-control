#!/bin/bash
# ASSUMED-value break-even sweep, box, closed_chain default plan (owner, 2026-11-16).  SIMULATION ONLY.  Gate: stack/tests/test_assumed_sweep_gate.py (5 passed).
set -u
R=/common/users/ss5772/bimanual/.claude/worktrees/contents-armside2
D=$R/world_model/v2/nero/results/assumed_sweep_2026-11-16
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
export BIMANUAL_PLANNER=closed_chain OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONPATH=$R:$R/experiments/ab_smooth_vs_bangbang:$R/experiments/realtime
cd $R
S="$PY -m world_model.v2.adapters.assumed_sweep --task box --workers 12"
$S --cells 1.0-1.5,2.5-3.0 --level static --out $D/static \
   --knobs PATH_MARGIN_MIN,MARGIN_WANT,PATH_MARGIN_BOTH,PSI_STEP_DEG,DY_RANGE,CLEARANCE_MIN,CLEAR_M_BOX,U_OBJ_CLEAR_M,TAU_NOISE,BIN_WIDTH_KG,A_BACKUP,A_MIN,EPS_CC_POS,EPS_CC_ROT
$S --cells 1.0-1.5 --level episode --seeds 3 --out $D/episode --knobs U_OBJ_CLEAR_M,SUPPORT_APPROACH_DEG,CLEAR_M_BOX
$S --cells 2.5-3.0 --level rule4 --out $D/rule4 --knobs STOP_K_MAX
for c in "PATH_MARGIN_MIN=0.015,U_OBJ_CLEAR_M=0.002" "PATH_MARGIN_MIN=0.015,U_OBJ_CLEAR_M=0.002,CLEAR_M_BOX=0.04"; do
  $S --cells 1.0-1.5 --level static --combo "$c" --out $D/combo
done
echo SWEEP_DONE
