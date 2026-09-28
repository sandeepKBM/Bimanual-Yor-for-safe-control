#!/bin/bash
# Full Stage-2 comparison (>=20 seeds) + refresh-rate ablation + learned-latent modes.
# SIMULATION ONLY. CPU-only (no GL/GPU needed). ~16 workers, threads pinned.
# Outputs: world_model/v2/latent_hj/results/scenario_<tag>.json ; tabulate with make_table.
set -e
source /common/users/ss5772/latent_hj_scratch/env.sh      # LATENT_HJ_SCR, TMPDIR, pinned threads, $PY, cd repo
S=$LATENT_HJ_SCR
M=world_model.v2.latent_hj.scenario
SEEDS=${SEEDS:-0-29}
W=${W:-16}
# Stage-0 value v1: validate_stage0 train --out $S/v1 --iters 20000 --A-max 10 --W-max 3 --m-lo -2 --grid $S/grid_v1_g099.npz
# (1) main table: every-step value refresh (analytic latent, Stage-0 value)
$PY -u -m $M --modes a_none,b_box,b_cone,c_switch,d_value,e_margin --seeds $SEEDS --workers $W \
    --ckpt $S/v1/value.pt --eps-v 0.3 --eps-l 0.3 --refresh 1 --a-cap 60 --trust 10 --tag full_r1 > $S/full_r1.log 2>&1
# (2) refresh-rate ablation: 50 Hz (WM tick) and 15.4 Hz (DINO-WM rate)
for R in 4 13; do
  $PY -u -m $M --modes c_switch,d_value,e_margin --seeds $SEEDS --workers $W \
      --ckpt $S/v1/value.pt --eps-v 0.3 --eps-l 0.3 --refresh $R --a-cap 60 --trust 10 --tag full_r$R > $S/full_r$R.log 2>&1
done
# (3) learned latent (Stage 1, kin_wm): value/switch on the WM posterior, 50 Hz WM tick
$PY -u -m $M --modes c_switch_wm,d_value_wm --seeds $SEEDS --workers $W --wm $S/wm/wm.pt \
    --wm-value $S/wm_v/value.pt --eps-v 0.3 --refresh 4 --tag full_wm > $S/full_wm.log 2>&1
for t in full_r1 full_r4 full_r13 full_wm; do
  $PY -m world_model.v2.latent_hj.make_table world_model/v2/latent_hj/results/scenario_$t.json > $S/table_$t.md
done
