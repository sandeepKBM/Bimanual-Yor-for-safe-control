# Stale-data inventory (pin_cup fix, 2026-10-08)

Rule: a folder is STALE if it was produced by a pin_cup code path before 2026-10-08 09:00 local.
pin_cup paths: `world_model/v2/testbeds/{a_loose_items,b_tray_ball,c_block_tower,d_opaque_fill}`,
`world_model/v2/tray_rep`, `world_model/v2/bead_rep`, `world_model/v2/data/gen_spill_data.py`, `world_model/v2/data/contents.py` (run_rig).

Snapshot taken 2026-10-08 ~12:52. Mtimes are from `find -printf` / `ls`. Producers marked "guess" were inferred from folder names, READMEs, meta.json, or import graphs, not from a run log.

Timing check: `contents.py` mtime is 2026-10-08 12:08:55, not ~09:00 as stated in the task. Nothing in the inventory falls between 09:00 and 12:08 except the `box` archive (12:32, which holds a pre-fix cache). The post-fix `*_v2` datasets (12:45) are after both times, so the classification does not depend on which time is right. `rebaseline_v2/env.sh` runs `unset BIMANUAL_LEGACY_PIN`, so those datasets use the fixed pin. Any data made with `BIMANUAL_LEGACY_PIN=1` reproduces the old physics and should be treated as stale.

## Affected folders, sorted by size

| Path | Size | Newest mtime | Producer | STALE |
|---|---|---|---|---|
| `/data/local/ss5772/bimanual_cache/dinowm_cache_zed_px112` (bimanual_data/dinowm is a symlink to it) | 17G (bimanual_cache total) | 2026-09-24 23:33 (dir) | guess: `dinowm/shards.py`, built from spill_v2 (`DATA = .../spill_v2`) | YES |
| `/common/users/ss5772/datasets/bimanual_wm_v2/spill_v2` (outside requested roots; source of the cache above) | 8.7G | 2026-09-24 18:59 | `gen_spill_data.py` (rig mode, confirmed by `run_gen.sh`, which calls the pin_cup path) | YES |
| `/common/users/ss5772/bimanual_data/tray_rep` (mixed; see sub-table) | 5.3G | 2026-10-08 12:51 (live `wrongpoint` run) | `tray_rep` / `b_tray_ball` (pin_cup) | MIXED: about 5.1G stale, about 0.2G post-fix |
| `/common/users/ss5772/bimanual_data/jobtmp/wm2` | 3.5G | 2026-09-24 18:19 | guess: `gen_spill_data.py` (`gen_v2_64.log` format matches its shard output) | YES |
| `/data/local/ss5772/pose_est_render` | 2.7G | 2026-09-29 23:29 (dir) | UNKNOWN: no producer in the repo; names (`tower_transfer`, `aligned_stack`) suggest block-tower renders (c_block_tower?) | UNKNOWN (check owner) |
| `/common/users/ss5772/bimanual_data/bead_replication` | 844M | 2026-10-01 19:50 | `bead_rep/bead_env.py` (pin_cup); `bead_rep/README.md` | YES |
| `/common/users/ss5772/bimanual_data/wm_audit` (+ repo copy `world_model/v2/wm_audit/results`, 168K) | 159M | 2026-09-29 00:44 | guess: `wm_audit/obs_probe.py` (`DATA = spill_v2`) and `kin_wm`, trained on spill_v2 | YES |
| `/common/users/ss5772/bimanual_data/model_uq` (new today) | 138M | 2026-10-08 12:50 | `world_model/v2/model_uq/collect.py`, which imports `tray_rep` and `tray_env` (pin_cup path) | NO if its inputs are post-fix (inputs not traced) |
| `/common/users/ss5772/bimanual_data/latentcbf` | 13M | 2026-09-29 00:44 | guess: `latent_hj/scenario.py` (own carry physics, no pin_cup import); `cmds/` mention spill 18 times | UNCERTAIN (lean NO) |
| `/common/users/ss5772/bimanual_data/testbeds` | 5.9M | 2026-10-05 05:03 | testbeds a/b/c/d (pin_cup) | YES |
| `/data/local/ss5772/bimanual_runs/wm_audit` (mirror) | 2.3M | 2026-09-28 23:08 | same as wm_audit above | YES |
| `/common/users/ss5772/bimanual/world_model/v2/latent_hj/results` (repo) | 2.7M | 2026-09-25 01:49 | guess: `latent_hj/scenario.py` / `validate_rows.py` (no pin_cup or spill_v2 import) | UNCERTAIN (lean NO) |
| `/common/users/ss5772/bimanual_data/box` (archive of the dinowm cache, plus `/data/local/ss5772/box_spool` 20K) | 82K | 2026-10-08 12:32 | archive of the stale dinowm cache | YES (archive made after cutoff, but its contents are the pre-fix cache) |
| `/common/users/ss5772/bimanual_data/dinowm` (`gate_full.json`, `gate_full.log`) | 20K | 2026-09-25 00:29 | `dinowm/gains_gate.py` on the cache | YES |

### tray_rep sub-table (`/common/users/ss5772/bimanual_data/tray_rep`), sorted by size

| Sub-path | Size | Newest mtime | STALE |
|---|---|---|---|
| `data_piper_full` | 1.6G | 2026-10-05 11:49 (meta) | YES |
| `data_piper_trk_full` | 1.5G | 2026-10-05 11:50 (meta) | YES |
| `data_piper_tray_only` | 791M | 2026-10-05 11:50 (meta) | YES |
| `data_piper_pc_head_aim_wrist` | 668M | 2026-10-05 17:48 | YES |
| `runs` | 153M | 2026-10-05 19:00 | YES |
| `bridge` | 125M | 2026-10-08 04:38 | YES |
| `boed` | 56M | 2026-10-05 18:58 | YES |
| `vlmprior` | 79M | 2026-10-08 05:08 | YES |
| `robust` | 67M | 2026-10-08 05:39 | YES |
| `boed_p7` | 50M | 2026-10-08 05:33 | YES |
| `boed_p8` | 46M | 2026-10-08 05:35 | YES |
| `tmp` | 47M | 2026-10-05 11:55 | YES |
| `perception` | 34M | 2026-10-05 17:35 | YES |
| `sets` | 30M | 2026-10-08 05:39 | YES |
| `smoke` + `smoke1b` + `smoke_pc` | 23M + 41M + 27M | 2026-10-05 17:35 | YES |
| `boed_f5` | 4.2M | 2026-10-08 05:28 | YES |
| `logs`, `authority`, `rebaseline_v2` (see below) | 463K, 131K, - | - | see below |
| `data_piper_full_v2` | 79M | 2026-10-08 12:46 | NO (post-fix) |
| `data_piper_trk_full_v2` | 85M | 2026-10-08 12:46 | NO (post-fix) |
| `rebaseline_v2` | 13M | 2026-10-08 12:46 | NO (post-fix; env unsets BIMANUAL_LEGACY_PIN) |
| `wrongpoint` | 27K | 2026-10-08 12:51 (still writing) | NO (post-fix) |

## Not affected (no pin_cup or spill path found)

- `/common/users/ss5772/bimanual_data/`: `vdirect`, `vdirect_ext`, `vdirect_f`, `vdirect_g`, `vdirect_ilab`, `vdirect_rb`, `vdirect_depth_diag` (pendulum, per the brief); `flmpc_replay` 140M (`experiments/.../replay`, no world_model import); `ctrlfix` 21M, `fakedsel` 228K, `n50_metricfix` 1.2M, `rt_flmpc` 237K, `v4c_retune` 49M, `videos` 21M, `tmp_replay` 37M (FL-MPC and v4c cup-collect demos, which do not import world_model); `diag` 21K (`drop_diag.py` uses the cup-collect demo); `g3_killtest` 126M (its own MuJoCo episodes, no pin_cup, contents or spill import; guess); `lit` 904M (papers); `scratch_pycache` 2.5M (bytecode only); `slurm_tmp` 73K (cleanup logs); `README_home_quota.txt`.
- `/data/local/ss5772/`: `bimanual_runs` except the wm_audit mirror (`rt_flmpc` 1.1G, `vdirect` 187M, `vf_smoke` 85M, `fakedsel` 392K, `detcheck` 16K, `tmp` 3.6M; `latentcbf` 2.6M is uncertain, see above); `tfb_out` 4.3G (trafficflowbench); `data` 56M and `outputs` (pems, traffic); `tmp` 22M; `molmospaces` 3.3G; `jenga_data` 229G, `jenga_datagen` 34G, `jenga_monitor` 18G, `jenga_expert_data` 880M, `jenga_planner_dumps` 500M (separate TOWER/jenga session; folder names only, not code-traced, and my import grep timed out).
- Repo: `experiments/actuation/results` 468K; `experiments/ab_smooth_vs_bangbang/replay` 120K; `experiments/slosh/data` 93M (pendulum residual models; guess).

## Totals

- Confirmed or likely STALE: about 35.3G. This is bimanual_cache 17G + spill_v2 8.7G + tray_rep stale about 5.1G + jobtmp/wm2 3.5G + bead_replication 844M + wm_audit 159M + testbeds 5.9M + wm_audit mirror 2.3M + box 82K + dinowm 20K.
- Adding pose_est_render (2.7G) if it turns out stale gives about 38G.
- Sizes are `du -h` rounded values; none of the `du` calls timed out.

## Caveats

- `tray_rep` is still being written (`wrongpoint`, 12:51) and `model_uq` is new today. The snapshot will go stale.
- Producers marked "guess" are not verified against run logs.
- `latent_hj` and `latentcbf` are marked uncertain because `kin_wm` and `validate_stage0` import `spill_data`, so they may consume spill_v2 directly.
