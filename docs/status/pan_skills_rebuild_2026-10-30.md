# Pan action skills rebuild (2026-10-30)

> Status (2026-11-05): HISTORY: record of 2026-10-30. The pan demo is no longer primary (docs/paper/demos_2026-11-01.md); the pan push/clamp tests were archived 11-03.

USER decisions (10-30): stop hand-writing skills; retrofit with mink (MuJoCo-native IK) AND rebuild RoboTwin's skill layer in
MuJoCo; add a torso-lift / base skill. Building blocks first (each skill checked alone against its contract,
`pan/skills_check.py`), then the cert v2 algorithm on top (`pan/cert/`).

## Why
`segments.py` skills are open-loop scripts: a world-frame delta, IK once per phase (`PoseIK`), FL-MPC tracking, a fixed grasp.
Measured (lib3 full+iid+overload, 158 worlds; skills_check smoke I03/V10):
- approach refuses 69/158 (left: all yaw < 0; right: 37/39 dx > 0 and yaw > 0; mostly no IK solution at all);
- clamp shoves the pan ~2.1 mm (+1.7 mm lid on I03);
- lift is uneven: I03 tilt 5 deg; V10 the pan slips (claws +47/+55 mm, pan +15 mm, tilt 10 deg);
- the lift backup inherits it (smokeB2: lift 0/39 on I03, 6/54 on OI03; pinch/push/clamp backups 100%).

## Borrowed pieces
| piece | source (license) | what we take | what we do not take |
|---|---|---|---|
| IK, limits | mink 1.2.0 (Apache-2.0), `bimanual_data/pylibs/mink120` (pip --target; mink 1.3 needs MuJoCo 3.10, we stay on 3.9.0) | FrameTask, RelativeFrameTask, PostureTask, DofFreezingTask, ConfigurationLimit, VelocityLimit, CollisionAvoidanceLimit | 1.2.0's collision-limit per-step bound is 1/dt loose (fixed upstream after 1.3.0): use a small gain and keep our own clearance check as the judge |
| skill layer | RoboTwin 2.0 (MIT), read from a shallow clone | Action list DSL (move / open / close per arm), lockstep two-arm scheduler, annotated contact / functional points (JSON, object frame), grasp-candidate sweep (rotations about the contact point) + direction priors, place-pose algebra | SAPIEN, mplib/curobo planners, its two-arm lift (two independent z moves: the same flaw as ours), its "first reachable" ranking |
| stage pattern | MoveIt Task Constructor (idea only) | generator / propagator stages with explicit pre / post interfaces | ROS |
| tracking | FL-MPC (ours, unchanged) | joint-reference tracking with payload terms | no edits |

## Architecture: `world_model/v2/nero/pan/skills/` (new package; segments.py stays the simulator, v1 skills kept for lib3)
- `annot.py`: pan annotation in the pan frame (MODEL, from scene params): left contact points along the handle
  (frac grid), right clamp points on the rim stack, functional points (pan centre, lid seat), approach-direction priors.
- `kin.py`: mink wrapper. Per-arm and two-arm solves; arm-vs-env and arm-vs-arm collision limits; joint limits; torso
  slider DoF; object-frame task for two-arm moves (both claws keep their grasp offset to one pan-frame target). Output =
  joint waypoints; clearance re-checked with the existing `_arm_env_clearance` (independent of mink).
- `actions.py`: RoboTwin Action DSL + lockstep scheduler, executed through the existing Episode path interface (joint
  waypoints -> FL-MPC). New action kinds: guarded_close (stop on claw wrench / gap rate), torso_to(z).
- `grasp.py`: candidate generation (contact points x rotations x approach tilt x torso height), filtered by IK + clearance
  along the whole approach / push / lift path, ranked by min clearance and joint margin.
- `pan_skills.py`: approach, pinch, push, clamp (guarded), lift (object-frame), torso, and the backup skills
  (release, right-back, lower), each with its contract from `skills_check.POST`.
- Episode gets additive hooks only (`skills="v2"` flag, a joint-path entry point); default stays v1.

## Torso / base
- Torso: `build_variant(lift=True)` enables the YOR telescope sliders (0..0.208 m each, coupled, +-56 N, from the YOR model).
  Moved only before the approach (arms free), then held. FL-MPC sees a fixed arm base during arm motion.
- Base: the YOR swerve base / floating base are commented out in the derived MJCF. A planar base (x, y, yaw) would be a NEW
  model assumption: added only if grasp variants + torso leave approach refusals, and raised first.

## Gates
- K0: mink smoke on the pan scene (done 10-30: 3 iters, 0.7 ms/iter, frozen DoFs unchanged).
- K1: approach v2 on the full/iid/overload worlds: coverage vs v1 (69/158 refused), same worlds and seeds.
- K2: each skill's contract over the same worlds (`skills_check`, v1 vs v2 side by side).
- K3: cert v2 on top: gate B rerun, library, Phase C (plan `frolicking-forging-treehouse.md`).

## New bounds (ASSUMED unless marked; raised to the user 10-30)
skills_check POST thresholds; grasp candidate ranges (handle frac, rotation, tilt); mink min distance (= existing 5 mm
clearance); velocity limits; guarded-close threshold; torso range used; two-arm payload share.
