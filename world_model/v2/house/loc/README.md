# house/loc -- simulated localization + mapping (SIMULATION ONLY)

Replaces two privileged reads in the teleop control path. Truth stays for metrics / eval / the operator model. All sensor numbers are ASSUMPTIONS
(`LocCfg`, `MapCfg`); the real YOR odometry / ZED error statistics are unknown to us.

| file | what |
|---|---|
| `estimator.py` | `PoseEstimator`: wheel odometry (scale/bias/noise/slip) + ZED VIO (30 Hz, 60 ms latency, drift per m, relocalization jumps, dropouts) -> complementary filter. `.pose`, `.v_odo`, `.sigma_xy`, `.errors()` (ATE/RPE) |
| `mapper.py` | scripted mapping pass with the head-ZED depth (depth_sim stereo noise law) -> 3-D log-odds (5 cm height bins) in the ESTIMATED frame -> 2-D occupancy (`PerceivedMap`, `OccGrid`); `map_error` (vs the true grids); `true_grid_zband` |
| `zono_est.py` | wraps `zono_rows.base_filter.BaseFilter.points` so the base CBF sees the robot in the ESTIMATED frame |
| `smoke.py` | h5 carry / wall-push stress smokes for the conditions truth / est_true / est_perc / est_perc_m |

## Flags (all default OFF = bit-identical)
* `TeleopSim(pose_source="est", loc_cfg=LocCfg(), loc_seed=, occ_source="perceived", occ_map=<npz|PerceivedMap|OccGrid|None>)`;
  `sim.ctrl_pose()` = what the control path reads; `sim.occ_ctrl` = the shield's grid; `sim.occ`/`sim.pose()` = truth (metrics).
  `observe()` additionally carries `pose_est`, `v_odo` when `pose_source="est"` (for capture-pose / arm-row feed-forward consumers).
* `ShieldCfg(loc_margin=, loc_margin_k=)`: wall buffer += loc_margin + k * sim.loc.sigma_xy.
* `teleop.run --pose est --occ perceived [--occ-map f.npz] [--loc-scale k] [--loc-margin m] [--loc-seed s]`.
* `zono_rows.smoke_s2 run ... --base-filter --pose-source est [--loc-scale k] [--loc-seed s]`.
* Maps: `/data/local/ss5772/molmospaces/percept/loc/maps/perceived_h5.npz` (built on first use if missing: ~75 s CPU).
