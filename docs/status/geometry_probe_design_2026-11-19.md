# Geometry by probing: design note (2026-11-19, draft, not built)

Status: design only. Nothing here is built. User rulings of 11-19 are in `docs/status/decisions.md`.

## 1. What this changes

Today Codex gets the scene geometry as numbers in its first prompt: every support as a box (top height, x and y extents), the crate's
size and tab height, and the goal xy (`stack/proposer/prompt.py` `task_spec`, `scene_facts`). Under the 11-19 rulings:

- **No scene geometry in the first prompt** ("no clues"). Codex gives priors and establishes the geometry by probes.
- **Robot facts stay**: what the robot knows about itself (joint ranges, link lengths, speeds, torque limits, claw, sensors).
- **Codex sees photos only.** The room is a rectangle with one lab camera on each wall (four cameras). Codex gets their RGB photos.
  It does not get depth: it cannot use depth well, and it does not need to.
- **The robot software uses depth.** The measuring skills (run by the robot, not by Codex) use the lab cameras' depth to turn a
  request ("measure the crate") into numbers with an error.
- **Model queries are allowed** (for example: joint room of the arms at an object pose).
- **Lab-fixed sensors are allowed**, pose known by calibration. Sim depth is fine for now.

Why: finding 9 (the box lift runs out of J7 room at 0.18 m) shows that a plan can fail on how the arms, the object and the scene
interact. The 11-17 ruling forbids giving Codex the answer. Instead Codex finds such facts itself, with probes, and the checker admits
only what the probes established.

## 2. Roles

| Part | Sees | Does | Never does |
|---|---|---|---|
| Codex (proposer) | task text, robot facts, four photos, the probe catalog, the geometry structure (empty) | fills priors, designs probes, then plans on the measured structure | sees depth, numbers it did not ask for, hidden truth |
| Robot software (measuring skills, model queries) | lab camera RGB + depth, joint encoders, claw readings, the robot model | runs each admitted probe, writes MEASURED / MODEL entries with their error | invents a value; widens a reading's error to make a plan pass |
| Checker | the structure (MEASURED and MODEL entries only), Θ_t for the contents | admits probes before they run; certifies the plan over the whole measured geometry set × Θ_t | uses a Codex prior as a fact |

## 3. Sensors (sim)

- **Room**: a rectangle around the task area. Size ASSUMED, large enough for every task scene (the bookcase scene spans about 2.5 m).
- **Four lab cameras**, one per wall, centred on the wall, 2.2 m high (ASSUMED), each aimed at the work area. Same intrinsics for all:
  70° vertical field of view, 1280 × 720 (ZED-2i-like, ASSUMED). Poses known exactly in sim (the calibration error is a later item).
- **Photos for Codex**: RGB from the four cameras at the initial state (arms at the reset pose, objects settled), before any probe. No
  hidden truth is visible (the crate lid is closed). Built from `nero/contents/plan_images.py` (new mode `room4`).
- **Depth for the robot software**: the same four cameras. Clean sim depth plus a noise model labelled ASSUMED (the existing
  `zed` level of `nero/pan/observe_pose.py` is a candidate: about 1 % of range, correlated, 10 % dropout).
- Rendered checks of two of these poses: `/common/users/ss5772/downloads/bimanual/2026-11-19_ceiling_view/`. The robot's own ZED at
  the YOR code mount sits under the counter in our low-lift scenes (`.../2026-11-19_zed_view/`), so it is not used now.

## 4. The geometry structure

One record per entity. Every field holds the same five things:

```
field: {
  prior:    [lo, hi]   # Codex's range, from the photos (ASSUMED; never used by the checker)
  set:      [lo, hi]   # what the checker may use: from readings only (empty until a probe fills it)
  source:   "MEASURED" | "MODEL" | "ASSUMED"
  readings: [ {probe, value, error, noise_model, n, camera_ids, time} ... ]
  why:      Codex's one-line reason for the prior
}
```

Entities and fields (first version, box and tray tasks):

| Entity | Fields |
|---|---|
| each support (counter, shelf, bookcase level) | top height; x and y extents (or centre + size + yaw); thickness |
| the carried object | footprint (x, y), height, grasp-tab height and positions, pose on its support (x, y, yaw) |
| other obstacles Codex names | box extents |
| goal | the support named in the task text, and the target pose on it |

The goal is named in words ("put the crate on the shelf"), not given as numbers. Codex must find where that support is.

## 5. Probe catalog (first version)

| Probe | Run by | Output | Guarantee type | Notes |
|---|---|---|---|---|
| `measure_box(entity)`: fit a box to the entity's points in the lab cameras' depth | robot software | its fields as a range: fit ± error | sampled (error from a measured noise study, n stated) | needs a segmentation of the entity; in sim the segmentation pass (as `observe_pose` uses) |
| `measure_support(name)`: plane fit of a support's top + its edges | robot software | top height, extents | sampled | same depth pipeline |
| `arm_room(object_pose)`: joint room of both arms holding the object at a pose (best over elbow angle) | robot model | per arm: min joint room and which joint | MODEL (exact under the robot model) | reuses `closed_chain._nodes` / `_best_margin`; this is how Codex would meet finding 9 |
| `path_clear(waypoints)`: the path's worst clearance against the CURRENT geometry set | robot model + structure | min clearance, the pair, where | MODEL on the measured set | uses only the structure's `set` fields |
| `touch(support, point)` (later) | robot, guarded claw move | contact height at a point | sampled | a physical probe: the checker admits it first |

A probe that moves the robot (touch) is admitted by the checker before it runs, as the contents probes are today. Camera probes and
model queries move nothing.

## 6. The loop

1. **First prompt**: task text, robot facts, the four photos, the probe catalog, the empty structure.
2. **Codex fills priors** for every geometry field (and the contents belief, as today), each with a reason. It asks for probes.
3. **Checker admits the probes** (only the ones that move the robot need rows).
4. **Robot software runs them** and writes readings. The field's set = the reading's range. A reading outside Codex's prior is not
   discarded: the set follows the reading, and the contradiction is recorded and shown to Codex (as D1 does for contents).
5. **Codex plans** on the structure, and may ask for more probes (bounded number of rounds, ASSUMED).
6. **Checker certifies** the plan over every geometry in the measured set and every θ in Θ_t. A field with no reading is MISSING, so
   any row that needs it refuses (fail closed).
7. **T4 re-ask** as today: failing rows go back to Codex as evidence, with the measured numbers only.

## 7. Soundness rules (from the AGENTS.md bounds rule)

- The checker reads only `set` fields with source MEASURED or MODEL. A prior never narrows anything.
- A depth error used in a `set` is measured: p50 / p95 / max over n with the noise model stated, labelled as sim depth. Report it as
  "max over n = …", never "≤".
- Sensing error is kept separate from the world: a reading gives the set of geometries consistent with it; it does not move the world.
- Comparisons (for example given geometry vs probed geometry) use the same worlds, the same cameras and the same noise.

## 8. Checker changes this needs

- **Collision rows over a geometry set**: each support and the object vary inside their measured ranges. First approach: inflate each
  support by its range and inflate the object by its size range (sound: the inflated volume contains every member of the set). Then
  the existing rows run once. `adapters/collision_rows.py`, the planner's shelf test (`closed_chain._collides`), Rule 1 recovery rows.
- **The planner's own clearance rule** (`CLEAR_M`, 30 mm ASSUMED over the highest support) is applied to Codex's plans today. With
  measured geometry, the checker's collision rows decide; the planner rule should not override Codex's path. Needs a ruling.
- **Goal check**: the plan ends on the support named in the task, inside its measured extents.
- **Static admission and Rule 1** take support heights from the structure instead of the scene file.

## 9. What exists to reuse

| Need | Existing code |
|---|---|
| send photos to Codex | `stack/proposer/propose.py` (`images=`, `image_note`), `nero/contents/plan_images.py` |
| probe catalog format | `stack/skills/catalog.py` (`Phenomenon`, `SkillEntry`, `cards`) |
| a reading → a set; contradiction handling | `stack/theta.py` (`ReadingSpec`, `theta_from_reading`, `reading_range`) |
| depth render + noise + shape fit | `nero/pan/observe_pose.py` |
| joint room at a pose | `stack/planners/closed_chain.py` (`_nodes`, `_best_margin`, `_psi_grid`) |
| re-ask with failing rows | `nero/contents/probe_select.py` (T4), `stack/proposer/prompt.py` (`reask_evidence_block`) |

## 10. Values to rule on before any result rests on them (all ASSUMED today)

1. Room size and the four camera poses (height 2.2 m, centred on each wall).
2. Camera intrinsics (70° vertical, 1280 × 720).
3. The sim depth noise model for `measure_box` / `measure_support`.
4. The number of probe rounds Codex may use.
5. Whether `CLEAR_M` stays as a planner rule once geometry is measured (section 8).
6. The first task to run it on (box is the smallest: one crate, a counter, a shelf).

## 11. Build order (each step with its own gate test)

1. `room4` cameras in the scene + photos for Codex; render check.
2. The geometry structure (schema, read / write, source tags); prompt text without scene numbers.
3. Model queries `arm_room`, `path_clear`.
4. `measure_box` / `measure_support` on sim depth; a noise study (n stated) that gives the error.
5. Checker over the geometry set (inflation), the goal check, `CLEAR_M` per the ruling.
6. One box run with the mock proposer end to end; then Codex (live requests within the cap).
