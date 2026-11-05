# TAMP layer status, 2026-10-17 (owner notes, full)

> Status (2026-11-05): SUPERSEDED. The TAMP/PDDLStream people-avoidance work (10-17..10-26) is no longer the thesis topic (shared autonomy dropped 11-04). Current claims: docs/paper/claims_2026-11-02.md; current planner decisions: docs/status/2026-11-03_handoff_notes.md. HISTORY: record of 2026-10-17.


**User direction change (10-17):**
- The viability library felt "not right". Diagnosis: it was wired into the 50 Hz gatekeeper as a stall rescue (E4: 0/2789 recoveries).
  - Likely cause: r_cup must match to 1e-6 m (library.py:228) but is read live each step (belief.py:131). Not instrumented yet.
  - Structural cause: entries only replay the gatekeeper's own candidate family; a resting robot matches only rest-start entries.
- The user chose to build the PLANNER layer: PDDL spec -> PDDLStream -> our checkers as streams -> sim execution. VLM ignored for now.
- The library moves up: it becomes the planner's certified-action cache; recovery = replanning.

**Built tonight (all uncommitted, new files only, world_model/v2/house/tamp/):**
- PDDLStream at /data/local/ss5772/external/pddlstream (GPL-3.0, dormant upstream since 2023-10). FD release build OK; the examples solve on py3.12.
- Survey (Sonnet): no maintained alternative with user streams + belief. Tampura is the belief-space option; cuTAMP is not conservative (spheres) and has no mobile base.
- types.py (plan contract), pddl/fetch_domain.pddl + fetch_stream.pddl, world.py (world from TeleopSim truth), streams.py (checkers as streams), planner.py (goal string -> plan).
- Streams map to checkers:
  - s-dock = C2 static at rest.
  - s-grasp = C3 corners + executor-skill filter.
  - s-place = support footprint.
  - t-reach = hold-pose kinematics.
  - s-route-free / s-route-hold = C1+C2 static route certificate (T3 H0, no people).
- RESULT (owner-run, inline): "(On tray desk)" solves in 0.2 s: move_free, pick, move_holding (reverse), place.
  - C3 keeps the bimanual squeeze (margin 0.61). It rejects the edge pinch (corner m=0.41 kg) and the top pinch (jaw width).
- Executor (Sonnet agent, STOPPED mid-run at shutdown): executor.py, exec_smoke.py, tests/test_executor.py exist, NOT reviewed.
  - Its debug run dbg_none.json (hand-written plan, filter none, no people) shows 4/4 actions ok, 21.8 s sim. UNVERIFIED by the owner.
  - The shield/gatekeeper with-people runs were not done.

**Open / next (in order):**
1. tests/test_planner.py: 3 pass, 5 FAIL under pytest only. FD translate's `import constraints` gets the wrong module ("module 'constraints' has no attribute 'Assignment'").
   - The inline run works. Suspect a `constraints` module on pytest's sys.path (eval/, experiments/* are on it) shadowing FD's translate/constraints.py.
   - Fix in planner._import_pddlstream (purge the foreign sys.modules entries / order the paths). Do not rename repo modules.
2. Review executor.py (do not trust the agent's numbers). Run exec_smoke: none / shield / gatekeeper(accel, set, monitor), seed 0, people on.
3. Wire planner -> executor end to end (run_tamp.py). Add a replan-on-failure loop: on a gatekeeper stall, replan from the sim state (WAIT / LOOK / other route as actions).
4. Pick-props domain with C3 + C4: a probe action whose outcome is unknown at plan time -> determinize + replan.
5. Cheap test from the gatekeeper side, still valid: hop = add T_S 0.05 to t_switch (T4(c) E3), plus the C2 "do not park inside the frozen band" margin.

**Rules unchanged:** sim only; nice 10 CPU; ≤ 3-4 agents; review subagent output before trusting it.
