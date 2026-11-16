# archive/ — learned world model and HJ-reachability work (archived 2026-11-16)

History only; nothing current imports this tree. The learned world-model / latent-HJ thesis was dropped on 2026-10-15
(`docs/paper/claims_2026-11-02.md` §1). Files keep their original repo paths under `archive/` (`git log --follow` works).
The code is not runnable from here: its imports still name the original paths (`world_model.v2.latent_hj...`).

| Archived path | What it was |
|---|---|
| `world_model/*.py`, `*.sbatch`, `*.yaml`, `*.md`, `eval_results_*` | v1 learned WM (DreamerV3-style) on the cup scenes: training, spill monitor head, `wm_semantic_monitor` (RealWMSpillMonitor, LatentValueShield), latent safety value |
| `world_model/v2/dinowm/` | latent world-model interface for the spill shield / latent HJ agent |
| `world_model/v2/latent_hj/` | latent-space HJ reachability (LatentSafe-style), grid HJ, kin WM, scenarios |
| `world_model/v2/model/`, `model_uq/` | v2 spill models; state-dependent model-error bounds (F5UQ) |
| `world_model/v2/vdirect/` | learned-WM falsifier arms (pre-registered) |
| `world_model/v2/g3_killtest/` | pre-registered K1–K4 kill test |
| `world_model/v2/wm_audit/` | closed-loop audit of latent reachability values |
| `world_model/v2/bead_rep/` | cup-with-beads rig for the latent-HJ replication |
| `world_model/v2/eval/` (harness, metrics, conformal, predictors, run_eval, slosh_bridge, test_metrics) | prediction-framed spill evaluation harness |
| `world_model/v2/tray_rep/wm/`, `tray_rep/robust/` | WM + latent-HJ value port onto tray_rep; perception-error tests of that filter |
| `hj_reachability/` | DeepReach / FLOQ HJ value training and datasets export |
| `experiments/wm_requirement/` | "how strong must the WM be" spill-predictor ladder (09-10) |
| `experiments/residual_dynamics/` | learned MLP/SIREN residual of the FL arm dynamics |
| `scripts/make_safety_demo.py` | 4-segment demo of the WM spill monitor (cup scene) |

**Kept in place (still imported by current code via `tray_rep/sets`, `tray_rep/selftest`, `testbeds/`):**
`world_model/v2/latent_hj/{__init__,reach_value,slosh_pend,spill_data}.py`, `world_model/v2/bead_rep/{__init__,bead_env,util}.py`,
`world_model/v2/wm_audit/common.py`, `world_model/v2/eval/{__init__,demo_runner,test_demo_runner}.py`.
`tray_rep/bridge/filters.py` still has a lazy import of `tray_rep.wm` inside its WM filter; that one filter no longer runs.

**Data:** compressed and checksum-verified in `westeros:/data/local/ss5772/coldbank/bimanual-wm-hj-2026-11-16/`
(one tar.gz per source directory, `SOURCES.tsv` = original paths, `manifests/` = per-file sha256, `MANIFEST.sha256`, `README.md`).

`scripts/make_cup_collect_safety_demo.py` no longer depends on the WM: `--config geometric_spill` and `--monitor` were removed;
`--config geometric_only` is the only config.
