# Collision fidelity — what is (and isn't) physically represented

Audit of the cup-collect model (`load_yor_cup_collect_model`), by semantic geom
category. Collision predicate: two geoms touch iff
`(contype_a & conaffinity_b) | (contype_b & conaffinity_a) != 0`.

Bit scheme (`sim_common._patch_real_collision_geoms`): left-arm geoms → contype
`0b0100`(4)/conaffinity `0b1000`(8); right-arm → `8`/`4`; manipulable objects
(cup/ball/cube) and world (table/floor) ship at `1`/`1`.

## ✅ Properly represented
| pair | status |
|---|---|
| **arm ↔ arm** (left links vs right links) | 169/169 collide — the safety-critical one; full gripper housing now included |
| **arm ↔ table + legs** | **65/65** (added 2026-09-14 — see below) |
| **arm ↔ floor** | **13/13** (added) |
| object ↔ table / floor | full (cup/cube/ball rest correctly) |
| object ↔ object | full (balls-in-cup, cube-cube, ball-cube) |

## 🔒 Intentionally excluded (must stay off)
| pair | why |
|---|---|
| left gripper ↔ its **grasped cup** / balls | cup is kinematically PINNED to the left EE; a hard contact would fight the pin |
| right arm ↔ the **cube it is holding** | picked cubes are WELDED to the gripper; contact would fight the weld |

## ✅ Now represented (added 2026-09-14, full-fidelity pass)
| pair | status | how |
|---|---|---|
| **arm ↔ self** (non-adjacent links, each arm) | **156/156** | per-arm SELF bits (16 left / 32 right, disjoint so they never cross arms or reach objects); 16 `<exclude>`s (parent-child + the two gripper fingers link7↔link8) → 0 spurious contacts at rest |
| **right arm ↔ cup** (non-holding arm) | **169/169** | cup conaffinity += bit 8 (right) only; left/holder stays 0/169 |
| **arm ↔ cubes not held** | **78/78** | cubes get both arm bits; a runtime toggle (`set_cube_arm_collision`) clears them on the weld rising edge and restores on release, so only the *held* cube is exempt |

Bit scheme: left arm contype/conaff = (4|16)/(8|16), right = (8|32)/(4|32); table/floor/leg
conaffinity += 12; cup += 8; cube += 12 (runtime-cleared while held). Verified: full matrix
correct, all isolations intact, 0 self-contacts at rest, full pick→place episode stable
(no NaN, no spurious contacts, cubes placed).

Remaining (low priority): arm ↔ balls (0/104) — balls live inside the cup, negligible.

## The arm↔table fix (2026-09-14)
Arm↔table was previously *deliberately* masked off (`sim_common` comment ~L234/641),
with the analytic `table_plane_barrier` QP row as the only protection — so an arm could
geometrically pass through the table legs/sides. Fix: `_patch_real_collision_geoms` now
ORs the arm bits (`4|8`) into the **conaffinity of every table/floor/leg geom** (matched
by geom name), so both arms' contype overlaps them. Applied ONLY to fixed world
structure — never to cup/ball/cube — so the pinned-cup and welded-cube isolations are
untouched. Verified: arm↔table 65/65, arm↔floor 13/13, arm↔arm still 169/169, gripper↔cup
still 0/169; a flmpc episode runs stable (no NaN, no spurious contacts, arm stays above
the table). The analytic barrier is kept as the proactive anticipatory layer; the geom
contact is now the hard ground-truth backstop.
