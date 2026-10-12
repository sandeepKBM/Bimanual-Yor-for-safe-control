# world_model/v2/house/office  -- lab -> corridor -> cafeteria (coffee-run scene)

Custom single-floor MuJoCo scene for the YOR bimanual coffee run. SIMULATION only; no robot/driver code is imported
(`YOR/robot/base.py`, `YOR/extra/*` never touched), no network. Plain MuJoCo 3.9 + the local THOR object pack
(`/data/local/ss5772/molmospaces/assets/objects/thor`). Existing house files (`build/lift/nav/phase*`) are reused by import only.

Layout (MuJoCo xy, m): LAB x[0,8] y[0,6] | door1 (x 8.0-8.2, y-centre 3.0) | CORRIDOR x[8.2,17.8] y[2,4] | door2 (x 17.8-18.0) | CAFETERIA x[18,28] y[-1,7].
Doorways 1.10 m clear (hold footprint 0.774 m wide + 2x0.10 = 0.974 m). Walls 2.8 m boxes, no ceiling.

## API
- `scene.load_office_spec() -> mujoco.MjSpec` static scene, 0 joints (like `build.load_house_spec(static=True)`). Collision geoms: walls group 3, rest group 4,
  contype 8 / conaffinity 15 (house structural class); visuals group 0/1. Cameras `overview_top` (orthographic), `overview_lab`, `overview_cafe`, `overview_oblique`.
- `scene.office_meta() -> dict` `rooms{lab,corridor,cafeteria: polygon}`, `doors[{id,centre,wall_dir,normal,clear_w,design_w,...}]`,
  `waypoints{lab_start, door1, door2, cafe_counter_pick, lab_desk_place: {xy, heading[rad], target_xyz?}}`, `persons[{name,xy,radius,height}]` (bodies `person_0..2`),
  `counter_height` (0.90), `desk_top_height`, `tray_pick_pad`, `desks`, `tables`, `footprint`. Waypoint `xy`/`heading` are the **robot base origin** pose
  (stand-offs sit `standoff_clear`=0.07 m from the counter/desk edge for the hold footprint); `target_xyz` is the counter pad / desk spot.
- `compose.compose_office(spawn_xy=None, yaw=None, with_tray=True, lift=True, mode="locked", dt=None, gravity=CANON, ball=True, head_cam=True, ...) -> build.Composed`
  (= `build.compose` / `lift.compose_lift` against the office; `lift=False` = no lift joints). Adds camera `head_cam` on `yor_root` (ASSUMED mast mount,
  pos (0.05,-0.21,1.15), 36.3 deg down, fovy 55; rides with the base, not the lift). `compose.new_hold_office(spawn_xy, yaw, dt, lift_h, servos_off=False)` = compose +
  `build.set_hold_state` + finger/arm ctrl + `lift.set_lift`. `compose.set_base_pose(C, x, y, yaw)` moves the mocap base and carries the tray/ball rigidly.
- `nav_grid.build_office_grid(res=0.01) -> nav.Grid` (parallel upward rays, same rule as `nav.build_occupancy`; ~5 s, cached in `$OFFICE_OUT/nav_cache`),
  `nav_grid.office_doors()` (`nav.door_list`-compatible dicts), `nav_grid.in_room(xy)`.
- `route.plan_leg(F, C, start(x,y,yaw|None), goal, doors, ...)` = `nav.plan_route` generalised to a door sequence + end headings;
  `route.plan_mission(F, C, meta)` = lab_start -> door1 -> door2 -> cafe_counter_pick | slide back 0.7 m, rotate in place | -> door2 -> door1 -> lab_desk_place,
  returns the concatenated `nav.Traj` (`marks` give the segment sample ranges) and a fine-grid validation (`validation`). Uses nav's `astar`, `door_passage`, `fillet`,
  `time_parameterize` (v 0.3 m/s, a 0.25), `validate_traj`, `poses_clearance`.
- `thor.place(spec, category, model, name, xy, yaw, z, tint=...)` static THOR object loader (textures stripped, joints deleted, receptacle sites removed,
  collision AABB centred on `xy`, underside at `z`). `thor.facing(dir)` -> yaw for a desired front direction.

## Run the smoke (CPU except renders; ~45 s)
    cd /common/users/ss5772/bimanual
    OMP_NUM_THREADS=1 PYTHONPATH=. /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python -m world_model.v2.house.office.smoke [--no-render] [--no-route] [--out DIR]
Picks the first GPU != 0 with <500 MiB used (nvidia-smi) for EGL. Outputs to `/data/local/ss5772/molmospaces/office/` (override `OFFICE_OUT`):
`smoke_results.json`, `office_overview.png`, `office_overview_oblique.png`, `office_head_cam.png`, `office_head_cam_3poses.png`, `office_standoff_*.png`,
`office_route.png`, `route_traj.npz` (t, x, y, yaw, segment marks).

## Notes / limits
- Hand-off at the counter is holonomic (slide back, rotate in place): the hold footprint swept by a rotation at the stand-off would hit the counter.
- Stand-offs are footprint-clearance poses; reaching the counter pad (~0.2-0.4 m beyond the tray) / desk spot needs the manipulation phase. The tray starts welded in
  the hands at the hold pose (not on the counter); the green pad is only a marker.
- Nav treats everything with underside < 1.15 m as an obstacle (nav.ROBOT_TOP), so wall shelves above that are ignored by planning.
- People are static capsule+sphere mannequins (torso radius 0.22 m); `person_*` bodies have no joints.
