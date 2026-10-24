# Owner notes 2026-10-21 (interpretation only; numbers are in memory files + JSONs)

## State
- Track 1 (base recovery around people): planner tests 11/11 (FD `constraints` clash fixed in tamp/planner._bind_fd_translate);
  executor reviewed OK; run_tamp.py = plan -> execute -> replan (held-grasp planning G1, mid-route suffix routes G2, mu check added).
- Livelock curve (run_tamp --stand-offset, seed irrelevant: executor is noise-free): gatekeeper done only d>=1.0 m, shield d>=0.9;
  stuck runs are safe (0 contacts) and replanning re-picks the same route (3/3). Out: /data/local/ss5772/molmospaces/tamp/run_tamp/stand_sweep_1021/.
- Track 2 (Demo C, NERO box turnover): world_model/v2/nero/democ/ built; joint-6 trap +45/-54 deg verified (null-space can't fix);
  FL-MPC got right-arm payload hooks (M_int_extra/bias_int_extra, default None, bit-identical); --payload-ff internal halves slip (~6 mm).
- Prior art read in full: Borquez hybrid HJ (low risk), ARMTD, ARMOUR, WAITR (memory files *-verdict-*). Receding-horizon
  "keep last certified plan" and robust mass-interval torque/contact constraints are THEIRS; C3 must be pitched as grasp-hold +
  mode switch + mu lower-bound set.

## Root cause found for the re-stall (drives the next build)
s-route certifies routes with NO people (T3 H0 by design); people are only seen by the 50 Hz gatekeeper at execution. So the planner
cannot see a blocked doorway and returns the same plan.

## Recovery-actions build (phase 1, started 10-21)
1. People-aware certification of the NEAR part of a route (current set-valued people belief incl. standing people) so a blocked
   segment is rejected at plan time; far part stays H0 (gatekeeper re-checks at execution; T1 unchanged).
2. Route streams become generative: the scene route + lateral-shift variants, each certified (walls + people) -> a "go around" plan.
3. RETREAT (MOVE): back to the last certified rest pose on the executed path (library-entry style), certified; then replan.
4. WAIT: when no plan certifies, hold at rest (T1 backup) for dt, replan; counted, budgeted.
Phase 2: LOOK (tracker people; reuse gatekeeper/gate_lib Looker) + checkers/library.Library containment as the MOVE generator.
Measure: rerun the stand-offset sweep; done-rate vs d for shield / gatekeeper / ours; recovery counts vs the T3 bound.

## Open user decisions
- prior_art_2026-10-15.md: add verified Borquez/ARMTD/ARMOUR/WAITR rows? (asked, no answer yet)
- Nothing committed this session.

## 10-22: option 2 build (USER approved "option 2 and planner-chosen speed")
- Sonnet implement agent launched: route stream emits gatekeeper-format timed segments (Traj, short, each with cup-safe backup),
  certified from the PREDICTED moving state, speed chosen from SPEEDS=(0.3,0.2,0.12) fastest-first; Route gains v_max+segments;
  Streams(cert_mode="rest" default | "segments"); tamp/seg_gate.py SegmentGatekeeper re-checks + commits each segment at its
  boundary (LibGatekeeper._start_traj precedent), keep monitor unchanged; refusal -> "segment_refused" -> ladder.
  run_tamp --cert segments; logs n_segments planned/admitted/refused (mismatch rate). Edits only in tamp/.
- Smokes: d=0.7/0.9 seed 0, both cert modes -> /data/local/ss5772/molmospaces/tamp/run_tamp/seg_1022/.
- OWNER TODO on return: review diff (tamp/ only; gatekeeper/ must be untouched: git diff --stat), rerun tamp tests (was 17/17),
  check legacy --cert rest reproduces stand_sweep_1021 JSONs, read smoke JSONs, then the full stand-offset sweep on ilab
  (cdx-agent --cpu-job, threads=1).
