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

## ❌ Still missing (candidate next fidelity steps)
| pair | now | note / difficulty |
|---|---|---|
| **arm ↔ self** (non-adjacent links, each arm) | 0/156 | real arm can't self-intersect. MED — needs adjacent-link `<exclude>`s to avoid spurious parent-child contacts |
| **right arm ↔ cup** (the non-holding arm) | 0/169 | should avoid knocking the cup. EASY-ish — give the cup conaffinity bit 8 (right) but NOT bit 4 (left, which holds it) |
| **arm ↔ cubes not being held** | 0/78 | HARD — the *held* cube changes per pick, so it needs a runtime `<exclude>` of the current welded cube, not a static bit |
| arm ↔ balls | 0/104 | LOW — balls live inside the cup |

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
