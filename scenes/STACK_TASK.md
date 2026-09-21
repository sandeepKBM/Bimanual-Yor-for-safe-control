# Bimanual STACKING task -- PROPER YOR robot

Genuinely bimanual pick-and-stack: scattered reach-partitioned blocks + a
shared central tower whose toppling is the WM-visible failure. Built on
`scenes/scene_yor_canonical.xml`'s robot (`scenes/yor_robot_derived.mjcf`,
read-only, unmodified) and the same load-time patterns as the cup task
(`scripts/cup_task_utils.py` / `collection/CUP_DATA_V2.md`).

## 1. Scene: `scenes/scene_yor_stack.xml`

`<include file="yor_robot_derived.mjcf"/>` (unmodified) + the same
floor/table/lighting conventions as `scene_yor_canonical.xml` (table top
z=0.55) + 2 new flanking platforms + 8 graspable 4cm blocks + a visual tower
zone marker. No fixed camera declared in the file itself (ZED attached at
load time, see section 3).

### Reach-partition (per `scenes/YOR_CANONICAL.md`'s FK-sweep envelope: left y
in [-0.58,0.60], right y in [-1.01,0.17], overlap y in [-0.58,0.17])

| Zone | Body | y-range | Reachable by |
|---|---|---|---|
| Center table (unchanged from canonical) | `table`, pos (0.42,-0.21,0) | [-0.51,0.09] | BOTH (shared) |
| Left platform | `left_platform`, pos (0.42,0.35,0) | [0.23,0.47] | LEFT ONLY (past right's y=0.17 max) |
| Right platform | `right_platform`, pos (0.42,-0.75,0) | [-0.87,-0.63] | RIGHT ONLY (past left's y=-0.58 min) |

Both platforms are solid plinths (floor to z=0.55) at the SAME x=0.42 as the
canonical table (verified clear of the base/wheel footprint there); widening
only in y keeps that clearance (verified: 0 platform/table-vs-base contacts
at rest).

### 8 blocks (4cm cube, 50g, same friction/condim as canonical `test_box`)

| Block | Position | Color | Side |
|---|---|---|---|
| `block_l1` | (0.34, 0.29, 0.57) | red | left-only |
| `block_l2` | (0.42, 0.36, 0.57) | orange | left-only |
| `block_l3` | (0.50, 0.42, 0.57) | yellow | left-only |
| `block_r1` | (0.34, -0.69, 0.57) | blue | right-only |
| `block_r2` | (0.42, -0.76, 0.57) | cyan | right-only |
| `block_r3` | (0.50, -0.82, 0.57) | purple | right-only |
| `block_s1` | (0.20, -0.45, 0.57) | green | shared (table corner) |
| `block_s2` | (0.64, -0.02, 0.57) | white | shared (table corner) |

### Tower zone

Flat, non-colliding (`contype=0 conaffinity=0`) yellow disc marker,
`tower_zone_marker`, at the table center (0.42,-0.21,0.551) -- the exact
point `YOR_CANONICAL.md` verified both arms' grasp sites can reach to within
1-2cm. Neither arm alone can supply all the blocks needed for a >=3 tower
(l-blocks unreachable by right, r-blocks unreachable by left) -- genuine
bimanual necessity.

### ZED + collision fix (load-time, in-memory)

`scripts.stack_task_utils.load_yor_stack_model()`: `mujoco.MjSpec.from_file`
+ `scripts.zed_camera.add_zed_camera` (mounted on `Lift_Top`, reused
unmodified) + `sim_common._patch_real_collision_geoms` (reused unmodified,
arm-arm collision fix) + a NEW compensating fix,
`_restore_finger_object_collision` (see section 4). Verified:
`nq=72, nv=64, nu=16, ncam=5` (4 static + `zed_cam`).

## 2. Label functions: `scripts/stack_task_utils.py`

- `tower_height(model, data, ...)` -- counts blocks forming a CONTIGUOUS,
  upright stack centered at the tower zone: horizontal distance < 6cm AND
  upright (local z-axis within ~32deg of world z, i.e. not lying on a side --
  important since a cube resting on ANY face has the same center height) AND
  z matches the expected resting height for level i
  (`table_top + block_half + i*2*block_half`), stopping at the first gap.
- `count_toppled(model, data, ...)` -- blocks within a WIDER 12cm radius of
  the tower zone that are tilted over (>35deg from upright) or knocked off
  the table (z < table_top-3cm). **This is the WM-visible tower failure
  label** -- fires 0->N the instant a placement misses badly enough to fall
  over, or an arm bump/collision disturbs the stack.
- `block_dropped(model, data, drop_z=0.30)` -- True if any block fell off a
  table/platform toward the floor.
- `collision(model, data)` -- real left_arm_*/right_arm_* body-geom contact,
  same body-name-prefix proxy as `collect_cup.py`'s `_left_right_contact`.
- `is_stacked_success(model, data, k=3)` -- `tower_height >= k and
  count_toppled == 0`.

## 3. Demonstrator: `collection/collect_stack.py`

**How both arms coordinate**: each arm independently fetches its OWN
reach-exclusive block first (LEFT gets `block_l1`, RIGHT gets `block_r1` --
neither could reach the other's), then LEFT additionally reaches ACROSS the
body midline to fetch a SHARED block (`block_s1`, on the table's right-ish
corner) for the 3rd tower level -- demonstrating cross-reach into the other
arm's territory, not just parallel independent work. Each subsequent block's
target (x,y) is chosen from the PREVIOUS block's actual landed position (not
the fixed nominal center), so the stack is self-consistent even though no
single placement is pixel-perfect.

**Transport-fidelity finding (important, see the module's own docstring for
the full writeup)**: this shared robot's arm actuators ship with low
position gains (kp 5-10, `scenes/YOR_CANONICAL.md`) that produce a
non-shrinking (does not improve with more settle time) several-cm
steady-state tracking error -- consistent with `teleop/README.md`'s own
finding that its cup-pickup smoke test never achieved a real grasp at all. A
real BUG was found and fixed along the way:
`sim_common._patch_real_collision_geoms` (reused for the arm-arm `collision`
label) also silently remaps the finger pad geoms' contype/conaffinity
(same "group=3" class as the arm-link collision proxies), making the
fingers physically incapable of colliding with anything but the other arm --
`_restore_finger_object_collision` in `scripts/stack_task_utils.py` compensates
this in-memory, in our own loader, without touching the shared
`sim_common.py`. Given the remaining PD-lag-vs-mm-precision gap (a
tuning property of the shared, read-only robot MJCF), the demonstrator uses a
hybrid: the arm's reach/carry is a real (if imprecise) `bt.drive()` IK
animation, while the HELD BLOCK follows its own exactly-scripted waypoint
path (position AND orientation pinned each step -- orientation pinning was
necessary too: the tracking arm's fingers can still incidentally overlap a
forcibly-positioned block, and a single step's contact-resolution torque on
an unconstrained quat silently tumbles it), decoupled from the arm's
tracking error -- but the block is DETACHED (real gravity/contact physics
resumes) 1.5cm above its target, so whether a placement is clean, misplaced,
or topples the tower is a genuine physics outcome of the final drop, never
scripted. This mirrors this repo's own precedent (`collect_cup.py` scripts
the cup's motion via direct qpos writes for its ENTIRE episode, no contact
grasp attempted at all) but goes further: only the transit leg is scripted,
never the task-relevant failure physics.

**Behavior families** (`STACK_BEHAVIOR_LEVELS`, seed % 3):
- `clean_stack` -- centered placements, 1.5cm release gap -> clean 3-tower.
- `topple_misplace` -- identical for the first 2 blocks; the 3rd (cross-reach)
  block is released 4.5cm off-center from a 5cm gap -> real physics topples
  the tower.
- `collision_bump` -- each arm picks its own block, then BOTH arms (+ carried
  blocks) are commanded to the SAME point near the tower center
  simultaneously, reproducing `YOR_CANONICAL.md`'s documented "both grasp
  sites to one point -> crossed/tangled arms" finding -- a real left/right
  arm-arm contact.

## 4. Verify (headless)

- **Scene load**: PASS -- `nq=72, nv=64, nu=16, ncam=5`.
- **Table/platform vs base/wheel contacts at rest**: 0 (verified).
- **Grasp + place on tower, tower build >=3 + topple (0->toppled)**: PASS --
  `run_clean_stack` reaches `tower_height=3, toppled=0`; a standalone 2-block
  build + misplaced 3rd block shows `toppled: 0 -> 1` at the exact moment of
  the bad drop (`artifacts/stack_pretopple.png` vs `artifacts/stack_topple.png`).
- **ZED visibility**: PASS -- `artifacts/stack_scene.png` (start: platforms +
  scattered colored blocks + yellow tower marker, both arms in frame),
  `artifacts/stack_built.png` (3-block tower standing), `artifacts/stack_topple.png`
  (tower reduced after the misplace).
- **Smoke collection** (`collection/data/stack_yor_smoke`, 10 episodes,
  seed 0-9, seed%3 cycling the 3 behaviors):

  ```
  episode_000000 seed=0 clean_stack      success=True  tower_height=3 toppled=0 collision=False steps=4740
  episode_000001 seed=1 topple_misplace  success=False tower_height=2 toppled=1 collision=False steps=4790
  episode_000002 seed=2 collision_bump   success=False tower_height=0 toppled=1 collision=True  steps=1510
  episode_000003 seed=3 clean_stack      success=True  tower_height=3 toppled=0 collision=False steps=4740
  episode_000004 seed=4 topple_misplace  success=False tower_height=2 toppled=1 collision=False steps=4790
  episode_000005 seed=5 collision_bump   success=False tower_height=0 toppled=1 collision=True  steps=1510
  episode_000006 seed=6 clean_stack      success=True  tower_height=3 toppled=0 collision=False steps=4740
  episode_000007 seed=7 topple_misplace  success=False tower_height=2 toppled=1 collision=False steps=4790
  episode_000008 seed=8 collision_bump   success=False tower_height=0 toppled=1 collision=True  steps=1510
  episode_000009 seed=9 clean_stack      success=True  tower_height=3 toppled=0 collision=False steps=4740
  wrote 10 episodes in 194.0s (success=4 toppled=6 collision=3)
  ```
  Outcome mix: **4/10 clean success**, **3/10 topple (no collision)**,
  **3/10 collision** -- deterministic per-behavior (this demonstrator is
  scripted, not stochastic, so the SAME behavior always produces the SAME
  outcome; the mix comes entirely from the seed%3 family cycling).
  Episode schema verified: all `episode.py` `STEP_ARRAY_KEYS` present
  (`rgb_head/rgb_wrist_right` = ZED, `left_q/left_qd/right_q/right_qd` +
  `left_gripper_state/right_gripper_state` = the 6 components of the 26-D
  `proprio`, `box_pose` = focus-block pose label, `collision`/`dropped`)
  plus extras `rgb_zed`/`rgb_wrist_left_ego`/`rgb_wrist_right_ego` (aux
  cams) and `tower_height`/`count_toppled` (dense progress/failure labels).

## Genuine bimanual necessity + cross-reach

Confirmed by construction and by the collection run: `block_l1`/`l2`/`l3`
sit at y in [0.23,0.47], past the right arm's max reach (y=0.17); `block_r1`/
`r2`/`r3` sit at y in [-0.87,-0.63], past the left arm's min reach (y=-0.58).
Every `clean_stack`/`topple_misplace` episode uses BOTH arms' exclusive
blocks plus a cross-reach shared block -- no single-arm demonstrator could
assemble a 3-tower here. `collision_bump` additionally drives both arms to
literally the same point, producing a real geometric arm/arm intersection
(not just a bounding-box coincidence), per `YOR_CANONICAL.md`'s own
documented finding.

## Figures

- `artifacts/stack_scene.png` -- ZED view, scene at rest (blocks scattered,
  tower zone marked).
- `artifacts/stack_built.png` -- ZED view, clean 3-block tower standing.
- `artifacts/stack_pretopple.png` / `artifacts/stack_topple.png` -- ZED view
  before/after the misplaced 3rd block, `count_toppled` 0->1.

## Blockers

None outstanding. The demonstrator is fully deterministic per behavior
(scripted waypoints + fixed seeds), so a bigger collection would currently
repeat the same 3 outcomes per family; adding per-episode randomization
(release-gap jitter, xy-offset jitter, which shared block/side is used) is a
natural follow-up if more OUTCOME diversity within a family is wanted, not
done here (out of scope: task asked for a MIX across families, which this
delivers).

## Rollback

Delete `scenes/scene_yor_stack.xml`, `scripts/stack_task_utils.py`,
`collection/collect_stack.py`, this file, `collection/data/stack_yor_smoke/`,
`collection/data/stack_yor_dbg*/`, and the `artifacts/stack_*.png` files.
Nothing under `YOR/`, `scenes/scene_yor_canonical.xml`,
`scenes/yor_robot_derived.mjcf`, `scripts/zed_camera.py`,
`scripts/cup_task_utils.py`, `experiments/ab_smooth_vs_bangbang/sim_common.py`,
`teleop/`, or `collection/episode.py`/`collection/perturbations.py` was
edited -- only imported/read.
