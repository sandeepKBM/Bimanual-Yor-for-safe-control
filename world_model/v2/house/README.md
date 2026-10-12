# world_model/v2/house

MolmoSpaces (procthor-10k-val) houses for the bimanual YOR in plain MuJoCo 3.9. SIMULATION only. Plan:
`docs/status/molmospaces_integration_plan.md`. `tray_rep` must never import this package.

- `assets.py`   data root (`HOUSE_DATA`, default `/data/local/ss5772/molmospaces/assets`), house extraction from the local
                val shard, texture-stripped XML copies, door table from the ProcTHOR json.
- `build.py`    `load_house`, `find_clear_spots`, `compose` (YOR under a mocap root + tray/ball + two weld constraints),
                `set_hold_state` (IK hold pose, weld relpose capture), `disable_arm_servos`.
- `selftest.py` `python -m world_model.v2.house.selftest phase0`  (20 houses: compile, step, ray, door table).
- `phase1.py`   `python -m world_model.v2.house.phase1 --house 17 --x 3.55 --y 10.05` (acceptance 1-4);
                `--sweep1` = acceptance 1 in all 20 houses.
- `results/`    JSON outputs of the runs reported in the status note.

Conventions: tray is held by two `mjEQ_WELD` (gripper_base <-> tray), never pinned; furniture static (all house joints
deleted); arms see every house collision geom (house conaffinity OR 0b1100); tray geoms stay at contype/conaffinity 1/1,
so arm<->tray contacts are filtered. Hold pose: tray long axis across the robot, hands on the short rims (the long-rim
grasp puts the finger keypoints inside FL-MPC's arm-arm margin and the QP is infeasible at hold).
