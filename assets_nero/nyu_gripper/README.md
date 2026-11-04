# NYU gripper (YOR gripper_body mesh) for the NERO wrist-camera variant

Opt-in; the default AgileX-gripper arm (`assets_nero/nero_arm.xml`) is unchanged. Since 2026-11-03 (USER) `config.GRIPPER_VARIANT` defaults to `nyu`; `BIMANUAL_GRIPPER=agx` = the AgileX gripper of every earlier result.
Scenes are built through `build_variant.variant_xml()`; call sites use `nero/gripper.py` (adapter). Grasps are the rigid weld (USER 11-03); `NUT_FORCE_N` = 231 N (MODEL) is not used by any result.

Purpose (USER 11-03): the model exists for the WRIST CAMERAS only (what they see, where they sit). Grasps stay the rigid weld.
Squeeze not modelled; grasp = rigid weld; owner's estimate 25-44 N tip, see memory.

## Source
`YOR/robot/arm/mujoco/assets/gripper_body.stl` (github.com/YOR-robot/YOR, commit `2cc6dd2`, `git status --short robot/arm/mujoco` clean at build time;
the user confirmed 11-03 it is the NYU RUM / Stick-V2-style gripper). Rebuild: `python -m world_model.v2.nero.nyu_gripper split|write`.
Split rule (trimesh, vertices merged 1e-6 m, mm -> m): the 2 connected components reaching x < -100 mm = `finger_a` (low y) / `finger_b`
(high y); everything else (palm, housing, phone pole, screws) = `palm.stl`. Collision pieces `palm_c0..3`, `finger_{a,b}_c0..4`: faces binned by
position along mesh z (palm) or along hinge->tip (jaws, only the arm beyond the hinge; the root plates/tails behind it interleave in z and carry no collision geom);
each piece is collided as its convex hull. The phone mesh is not placed (no `iphone.stl` copy). Mesh frame = YOR gripper_base frame, jaws point along -x.
Paper arXiv 2602.11150 Sec 3.4: "custom grippers with an angular jaw". It gives no pivot/range/force.

## Mechanism (owner measurement from the mesh, 11-03)
XL430 -> lead screw along mesh x -> nut+pin -> straight cam slot in each jaw root plate -> each jaw hinges about mesh z. Model: slide joint `gripper_nut`
(0..21.9 mm) + two hinge jaws `gripper_joint1/2` tied to the nut by `THETA_POLY` (joint equality). One position actuator `gripper_pos` on the nut,
ctrl = nut position (0 = open mesh pose); a POSE parameter, no force claim. The helix may be a spring (owner); the user saw a spring of unknown role: not modelled.
Needs the 1 ms physics step (file sets timestep 0.001; 5 ms oscillated).

## Numbers
| name | value | label |
|---|---|---|
| hinge A / B (mesh mm) | (-49.29,-5.59) / (-49.33,42.44) | MEASURED (section hole centre at z=29, n=1 mesh) |
| jaw mirror plane y, jaw z mid | 18.42, 29.0 mm | MEASURED |
| mounting standoff max x | 73.2 mm | MEASURED |
| nut axis (y,z), pin x open, travel | (18.2,29.5), -45.6 mm, 21.9 mm | MEASURED / MODEL (owner, pin-in-slot solve) |
| THETA_POLY, THETA_MAX | a0..a4 in config of nyu_gripper.py, 0.793 rad | MODEL (owner fit, max err 0.011 deg) |
| mass / COM / inertia | 0.45 kg, YOR piper_left.xml values | MODEL (YOR MJCF) |
| jaw masses | surface-area share of 0.45 kg; nut 5 g out of the palm share | ASSUMED |
| mesh->flange rotation | euler (pi,-pi/2,0) (YOR uses 3.14,-1.57,0) | ASSUMED idealisation |
| adapter | coaxial on the jaw mirror axis, standoffs flush with the NERO flange face (z=0) | ASSUMED |
| ee_grasp site | 0.165 m along approach (mid-pad) | ASSUMED |
| jaw hinge range 0.82 rad, damping 0.2, armature 0.005; nut armature 0.01; pose servo kp 1e4, kv 100, force 500 N | | ASSUMED |
| wrist cameras | `<side>_wrist_rgb`, `<side>_wrist_depth` on gripper_base (no body, no mass, no geoms) | USER 11-03: the real wrist camera is an Intel RealSense D435i on the wrist protrusion facing the fingers |
| RGB camera position | at the YOR phone position (28.5,15.0,117.0) mm in the gripper_base mesh frame | MODEL (YOR MJCF phone position); that the RealSense sits at the phone spot = ASSUMED |
| depth camera position | RGB + (-65 mm) along the image x axis (left imager) | MEASURED offset from the Menagerie D435i mesh vertices (numbers only; RGB pupil x=+32.5, left IR x=-32.5 mm) |
| fovy RGB / depth | 42.5 / 57 deg | datasheet (Intel: RGB 69.4x42.5, depth 86x57 deg, 16:9), +/-3 deg per unit |
| image size | 640x360 (`CAM_RES`, both cameras) | ASSUMED default render size (16:9) |
| camera direction | along the jaws (mesh -x) | USER 11-03 (faces the fingers) |
| camera tilt | 16.0 deg down, 0.0 deg yaw | MEASURED 11-03: YOR piper_left.xml iphone mesh as placed in gripper_base, area-weighted back-face normal (9663 mm2) = (-0.961, 0.001, -0.276); n = 1 assembly; RUM mount CAD parts not assembled (copied to bimanual_data/cad/rum_mount_2026-11-03). ~~20 deg ASSUMED~~ superseded |

D435i 72 g not added: no D435i body, mesh or mass is in the model (the cameras are massless; `assets_nero/realsense_d435i/` is not referenced). The iPhone
mesh is not placed either. YOR's own MJCF camera orientation (look dir (-0.259,0,0.966): up the pole, ~113 deg off the fingers, no fingers in its render) is not
used: it contradicts the user statement that the camera faces the fingers. Only its position is used.

## Collision
Same treatment as the AgileX fingers: geoms use class `nero_collision` (contype 0 / conaffinity 1, the repo bit-patching scheme applies identically),
body-pair excludes are the AgileX set (palm-jaw, flange-jaw), jaw-jaw is not excluded. The geometry differs from the AgileX gripper (jaw tips 159 mm apart open
vs 100 mm; phone pole 157 mm), so contacts with carried objects are not guaranteed identical; the weld/carry scenes keep the jaws at one fixed pose.

## Not wired
`pan/handle_grasp._set_fingers` and `contents/carry.py` set `*_arm_gripper_joint*` qpos and `*_arm_gripper_pos` ctrl in AgileX units (0.03 m = open);
with this file ctrl 0.03 m would exceed the nut travel and 0.03 rad on the jaw joints is ~open. A variant hook is needed before using the nyu arm in a scene.
