# One world-facts interface for every decision component (proposal, 2026-10-22)

> Status (2026-11-05): SUPERSEDED: proposal of 2026-10-22 for the gatekeeper/shield/tamp people-avoidance stack (dropped 11-04); never approved as written. The WorldState + producers idea is used by world_model/v2/stack/ (types.py, producers.py). HISTORY.

Status: PROPOSAL, awaiting user approval. No code moved yet.

## Why
The 10-22 cup-row bug: the gatekeeper and the shield read `sim.d.xpos[sim.cup]` directly and never asked whether the cup was on the robot. The planner
had its own idea (`World.belief_at(holding=...)`), so plan and execution disagreed. Every component that reads the simulator for a FACT makes its own
assumptions about that fact, and a VLM interface later would need a separate code path into each one.

Rule after the overhaul: decision code reads only
1. a `WorldState` (facts + set-valued uncertainty + provenance),
2. the robot's own proprioception and limits (pose estimate, twist, encoders, `a_hw`, footprint),
3. static config.

Who writes the facts is a PRODUCER: sim truth (ablation), perception, the executor (it knows what it grasped), and later the VLM. The checkers never
change when a producer is swapped. VLM proposes facts/specs with `Source.VLM`; the checkers still decide (unchanged theory: T1 is over the belief set).

## What exists
`house/ocs` already has most of it (README, round 1 2026-10-12): `WorldState{robot, objects, relations, free}`, `Source` provenance,
`privileged()` audit, producers (Truth, Base, TrayFK, CupWrist, People, ObstacleZono, Names), consumers (`to_people_state`, `to_occ`,
`to_ir_entities`, `ShieldView`) and an equivalence test (truth -> OCS -> shield is bit-identical). It was never wired into the gatekeeper, tamp
or checkers.

## Coupling to remove (from the 10-22 map; facts only, proprioception and metrics excluded)
| component | direct sim reads of facts |
|---|---|
| gatekeeper/belief.py, gate.py | cup pose, attachment (`cup_attached`), people xy + velocity, occupancy grid |
| teleop/shield.py | tray/cup pose + tilt, people + ids/radii |
| tamp/world.py | tray/cup pose, masses (`body_mass`), cup_world/cup_rel, hard-coded names ("tray", "counter", "desk") |
| tamp/executor.py | post-conditions read tray pose/qvel, surfaces, `sim.grasped`, `bead_out` (these are CHECKS: may stay on truth as an audit, labelled) |
| checkers/library.py | scene routes/waypoints (config, fine), "cup" hard list |
| pick/ | prop poses/masses (scene building), name keywords |

## Schema additions (additive, nothing renamed)
- **Attachment as a fact**: `Relation("held_by", oid, "robot", status)` with status in {closing, held, releasing}; `ws.attached(oid)` replaces
  `geom.cup_attached(sim)`. Producers: executor (weld/assist state, `Source.EXEC`), later VLM/perception.
- **Object registry**: `ObjectState.kind` + `affordances` (graspable, support surface, payload) + skill list, so tamp stops hard-coding "tray"/"desk".
- **Physical parameter sets with provenance**: mass and mu as intervals (`params`), `Source.PRIOR | TRUTH | VLM | PROBE`. C3 reads only these.
- **People as objects**: `kind="person"`, position set, velocity estimate + error (`e_v`), staleness. Gatekeeper Belief people come from here.
- **Surfaces/docks** as objects with pose sets (the planner's `SURFACES`).
- **Task spec as data**: goal (PDDL literal), constraints (IR/STL specs, `house/lang` already authors these), optional preference weights.
  Hand-written today; the VLM later writes the same object.

## Adapters (thin; the checkers keep their math)
- `gatekeeper.belief.belief_from_ws(ws, robot, gk, cfg)`; `build_belief(sim, ...)` becomes `belief_from_ws(TruthProducer(sim)...)`.
- Shield takes a view (`ocs.ShieldView`) everywhere, not `sim`.
- `tamp.world.world_from_ws(ws, ...)`; `world_from_sim` = truth producer + it.
- `checkers/library` Ctx from `ws`.

## Migration order (each step bit-identical on the truth producer, proven by an equivalence test before the next)
1. Commit the current tree as a baseline (gatekeeper/, tamp/, checkers/ are UNTRACKED today -> no rollback for a refactor).
2. Schema additions + `ws.attached`, people objects, registry, parameter sets (+ tests).
3. Gatekeeper: `belief_from_ws`; equivalence: same Belief fields as `build_belief` on recorded states, gatekeeper suite unchanged.
4. Shield: all call sites through `ShieldView`; equivalence on the teleop carry regression.
5. tamp: `world_from_ws`, names from the registry; equivalence: same plans + certs on h5 (23 tests + a recorded plan JSON).
6. checkers/library Ctx from `ws`.
7. Guard test: grep-based test that fails if a decision module reads `sim.d`, `sim.cup`, `sim.tray`, `sim.grasped`, `sim.people_state`, `body_mass`
   outside the producer/adapter allowlist (metrics/audit lines allowlisted by marker).
8. Mock VLM producer (no network): writes names, `held_by`, mass/mu ranges and a constraint spec with `Source.VLM`; run the h5 TAMP episode on
   it with checkers unchanged. This proves the VLM interface needs no new code in the checkers.

## Not changed
Theory (T1-T4, C1-C4 math), units, control rate (50 Hz), accel limits, horizons, the `cup_row` flags (they become `ws.attached` internally).
Executor post-conditions and episode metrics may keep reading truth, labelled as audit (they judge outcomes; they do not decide).

## Open decisions for the user
- Approve the order above (and step 1: commit the baseline).
- Scope: include pick/ and flagship/ now, or only gatekeeper + shield + tamp + checkers first.
