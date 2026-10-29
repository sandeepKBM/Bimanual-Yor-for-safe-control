set -x
P="/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python -m world_model.v2.nero.pan.vizcoast_faithful.run_baseline"
V=world_model/v2/nero/results/pan_vizcoast_faithful_2026-10-28/planning.json
$P --out world_model/v2/nero/results/pan_fair_box_2026-10-29/viz_full_obs_zed_v2 --plans-from $V --execute -1 --worlds full --observe zed --procs 24
$P --out world_model/v2/nero/results/pan_fair_box_2026-10-29/viz_obsbox_open --plans-from $V --execute -1 --worlds obsbox --procs 24
echo ALLDONE
