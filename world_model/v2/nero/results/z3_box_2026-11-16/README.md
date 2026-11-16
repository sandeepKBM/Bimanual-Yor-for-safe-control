# Z3 planner on box, first run (2026-11-16, ilab3 job 1295671)
- Driver `z3_box_run.py`: closed_chain, Z3 (stack/z3plan) + checker oracle (adapters/z3_oracle_contents), cost [("a","max")], max_iters=150.
  Variants: A = checker as-is; B = carry.PATH_MARGIN_MIN 0.03 (ASSUMED override).  Cells 1.0-1.5, 2.5-3.0, 4.5-5.0 kg.
- Result: NO answer in any run: all 6 stopped at the iteration cap (log[-1].source = "max_iters"); this is NOT "no plan exists".
  ~150 oracle calls per run, 10-16 min; ~60 % failed the box compile rules (pure arithmetic, discovered one oracle call at a time),
  ~17 % plan_geometry, ~13 % joint_pos (1.0-1.5); every tried plan had a = 0.5.  B = A on every row: 0.03 rad is still above the
  J7 room (0.021 rad; sweep break-even ~0.019 rad), so the joint_pos refusals are the same.
- Follow-up: v2 (compile-rule prefilter, learned constraints that drop knobs a failure does not depend on, explicit unsat vs cap).

# v2 (2026-11-16, ilab3 job 1296082, `z3_box_run_v2.py`, files `v2_*.json`)
Grid (ASSUMED, `z3_oracle_contents.domains("box")`): zc 0.02..0.20 step 0.02 (10), place_dx/dy +-0.01 step 0.005 (5 x 5), a {0.05, 0.1, 0.25, 0.5},
recovery {stored_backup, read_then_retrace} = 2000 plans.  Prefilter (compile rule "lowering", pure arithmetic): 1400 rejected -> only zc 0.16 / 0.18 / 0.20 remain.
Variant C = carry.PATH_MARGIN_MIN 0.015 rad (ASSUMED, sweep break-even); every C result is conditional on it.

| variant | cell kg | status | oracle calls | wall s | failing rows (count over the calls) |
|---|---|---|---|---|---|
| A as-is | 1.0-1.5 | unsat | 75 | 1166 | plan_geometry 50, joint_pos 20, static place/release support_dest 5 |
| A as-is | 2.5-3.0 | unsat | 102 | 1867 | plan_geometry 50, a_le_a_admit 27, static support_dest 16, joint_pos 9 |
| A as-is | 4.5-5.0 | unsat | 75 | 869 | plan_geometry 50, torque_payload (lift/move/place) 25, support_dest 23 |
| C 0.015 | 1.0-1.5 | unsat | 95 | 3680 | plan_geometry 50, rule1[2:s2_place] (both recoveries) 20+20, support_dest 5 |
| C 0.015 | 2.5-3.0 | unsat | 111 | 5899 | plan_geometry 50, a_le_a_admit 27, support_dest 16, rule1 lift/move/place 9 each |
| C 0.015 | 4.5-5.0 | unsat | 75 | 870 | as A 4.5-5.0 |

- "unsat" = no plan on THIS grid passes; it says nothing about plans off the grid (finer zc, wider place window).
- plan_geometry, one oracle call per place offset (25 each): zc 0.16 "every lateral lift offset collides with the shelf"; zc 0.20 "right arm 0.072 rad outside a
  joint limit at the lift top".  Only zc 0.18 is buildable on the 0.02 grid.
- zc 0.18, 1.0-1.5: A fails on joint_pos (J7, as the default plan); C fails on Rule 1 at the place step for both recoveries: (i) the stop overshoot
  (0.5 v T = 18 mm at t = 0.25 s) runs past the last planned sample (end pose unchecked -> refused), (ii) obj_world shelf|box_wall_e 7.1 mm planned,
  swept bound 2.1 mm < U_OBJ_CLEAR_M 5 mm.  A +-1 cm place offset does not change either (both are about lowering onto the shelf).
- The ilab3 /dev/shm semaphores of the run's outer Pool vanished at ~03:25 (another ss5772 job ended on the node at 03:25:40; cause not confirmed);
  the per-run workers had already started and wrote their files, the log holds ~1M lines of respawn tracebacks.  No oracle call reports an error row.
- Not re-checked: the fresh-oracle recheck only runs on a found plan (none found).
