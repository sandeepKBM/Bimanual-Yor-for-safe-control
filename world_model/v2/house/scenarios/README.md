# scenarios/ -- scripted, seeded shared-autonomy base-carry scenarios (h5, sim only)

Two scenarios built WITHOUT editing teleop/gatekeeper/checkers/pick; used through their public APIs only.

## How people are placed / controlled
* A scenario is a `SceneSpec` factory registered in `teleop.scene.SCENES` under a key that encodes every parameter
  (`h5@corridor:{...}`); `TeleopSim(scene_name=key)`, `run_episode(scene=key)`, `run_gk.run_one(scene=key)` build it unchanged.
  `SceneSpec.people` = the single scripted person (mocap capsule r=0.22); p_walker / p_stand are removed.
* `ScenSim(TeleopSim)`, installed as `teleop.run.TeleopSim` by `with patched():`, defaults `people_clock="time"` and the scenario's `PersonPolicy`
  (the legacy clock never starts a track whose first segment is stationary, which the occlusion person needs), and logs a per-tick trace
  (stall time) and, for occlusion, the head-camera visibility (`people_track.detectors.Visibility`: frustum + line of sight).
* Teleop hook that would remove the monkeypatch (not needed, not made): `run_episode(..., sim_cls=TeleopSim, sim_kw=None)` forwarded to the constructor.

## Geometry (h5, 1 cm occupancy grid)
Route: lab_start (3.1, 6.09) -> (2.134, 5.075) -> south along x = 2.134 -> counter (2.135, 2.0); length 4.476 m.  The y = 4.2..3.3 stretch is a 1.86 m
wall-bounded passage (walls x = 0.94 / 2.80; route 0.67 m from the east wall, 1.15 m from the west wall), doorway plane y = 3.3.  The robot footprint
(teleop/geom.FP) reaches 0.64 m to the RIGHT of travel (west) and 0.13 m left (east) of the base origin the route describes.

### corridor (`corridor.py`, `--param d`)
Standing person (zero velocity, fixed all episode) at lateral offset `d` from the route centreline at arc length `arc_s` (default 3.6 -> y = 2.88),
`side` = `right` (west, default, footprint-heavy side; contact with the footprint swath starts at d ~ 0.86) or `left`.  The torso circle must be on free floor
(occupancy EDT >= 0.22 + 0.02) or `ScenarioError` is raised (never silently moved); max valid d at the default arc: 2.4 m right.  In the passage itself
(arc 2.5..3.2) only d <= 0.91 (right) / 0.40 (left) is free, so use `--scen-kw '{"arc_s": 2.75}'` for narrow-passage runs with d <= 0.9.

### occlusion (`occlusion.py`, `--param t_appear`)
Person stands at hide (0.45, 2.9) behind the wall y = 3.3 (x <= 0.94), walks east past the corner (0.95, 2.9) at `t_appear + jitter_s * U(-1, 1)`
(seeded by the episode seed; default jitter 0.5 s), crosses the route at (2.134, 2.9) and leaves at (3.0, 2.9), speed <= 1.2 m/s (default 1.0).
Hidden from the head camera on every pose of the north approach (tests); visible once past the corner; blind zone (< ~0.6 m in front of the base) is real.
Nominal operator (no people) is at the doorway plane at ~5 s, so t_appear 4..5 s is a crossing conflict.  `oblivious=true` = never yields.

## CLI
```
python -m world_model.v2.house.scenarios.run_scen --scenario corridor --param 0.4:1.4:0.2 --configs shield_brake,shield_escape,gk_accel_hold,gk_accel_hold_trk \
    --seeds 0-4 --workers 4 --max-s 40 --out /data/local/ss5772/molmospaces/scenarios/corridor
python -m world_model.v2.house.scenarios.run_scen --scenario occlusion --param 3.5,4.5,5.5 --configs gk_accel_hold,gk_accel_hold_trk \
    --gk-extra '{"gk_a3": {"person_model": "accel", "a_p": 3.0}}' --seeds 0-4
```
`--configs` names: shield_brake, shield_escape, gk_speed_hold, gk_accel_hold, gk_accel_hold_trk, plus any key of `--gk-extra` (JSON GKConfig overrides).
Workers capped at 4 (CPU only, `nice 10`, threads pinned).  Outputs `rows.jsonl` (resumable per scenario/param/scen_kw/config/seed) and `summary.json`.
Row fields: task_done, task_time_s, min_person_distance_m, min_wall_clearance_m, person_contact_ticks, physical_house_contacts, spilled, final_dist_to_goal_m,
stall_s (base speed < 0.03 m/s before the task is done = the livelock measure), first_visible_t (occlusion), `gk` counters (gatekeeper configs).

## Tests
`python -m pytest -q world_model/v2/house/scenarios/tests` (CPU, builds 4 TeleopSim rigs).
