# Handoff bimanual

> Status (2026-11-05): HISTORY: machine-generated handoff snapshot of 2026-10-22; not current guidance.
Generated: 2026-10-22T00:29:44-04:00
Repo root: /common/users/ss5772/bimanual
Current: perception-teleop-2026-10-12 @ 66cd2d8 docs: root README (system requirements, install, quick start, optional envs, repo map); regenerate requirements.txt
## Worktrees
- perception-teleop-2026-10-12: 66cd2d8 docs: root README (system requirements, install, quick start, optio... (ahead ?, behind ?, dirty 23)
## Slurm
queued/running:
  685224 jenga_hold PENDING 0:00  Priority
  685179 r7D6 PENDING 0:00  Priority
  685178 r7T54 PENDING 0:00  Priority
  532911 jenga_cap_coder RUNNING 5-04:57:55 rlab2 None
finished:
  655849                       s0A_base    COMPLETED   00:08:57      rlab3
  655853                       s0E_cont    COMPLETED   00:08:55      rlab4
  655854                    s0F_recover    COMPLETED   00:17:51      ilab3
  655977                    s0B_noalign CANCELLED b+   00:00:00 None assi+
  656002                      s0C_ffv15    COMPLETED   00:08:34      rlab4
  656003                      s0D_ffv25 CANCELLED b+   00:00:00 None assi+
  657277                        s0G_z11 CANCELLED b+   00:00:00 None assi+
  657278                       s0H_z11v CANCELLED b+   00:00:00 None assi+
  657966                          p4_ds CANCELLED b+   00:00:00 None assi+
  658264                        p4_rb48 CANCELLED b+   00:00:00 None assi+
  683393                        wcalS17 CANCELLED b+   00:00:00 None assi+
  683394                         wcalH2 CANCELLED b+   00:00:00 None assi+
  683516                        wcalS17 CANCELLED b+   00:00:00 None assi+
  683517                         wcalH2 CANCELLED b+   00:00:00 None assi+
  683527                         wcalT0 CANCELLED b+   00:00:00 None assi+
  685176                          r7S17 CANCELLED b+   00:00:00 None assi+
  685177                           r7H2 CANCELLED b+   00:00:00 None assi+
## Local GPUs
- GPU 0: 1479 MiB, 0% util, owners: ss5772
- GPU 1: 6457 MiB, 11% util, owners: jhd79
- GPU 2: 8269 MiB, 12% util, owners: jhd79, ss5772
- GPU 3: 5351 MiB, 0% util, owners: jhd79
- GPU 4: 9461 MiB, 74% util, owners: jhd79
- GPU 5: 5373 MiB, 0% util, owners: jhd79
- GPU 6: 5486 MiB, 4% util, owners: jhd79
- GPU 7: 6858 MiB, 0% util, owners: jhd79, ss5772
## tmux sessions
- boxoff
- ilabujobs
- ilabukeep
- ilabuwatch
- tower
- towerops
## Reports
- perception-teleop-2026-10-12:
  (none)
## Open items
- docs/status/molmospaces_integration_plan.md
  ### Phase 0: load and measure (about 80% done by this smoke), 0.5 day left
  Deliverables: `house/assets.py`, per-house object-pack manifest, `selftest` loading 20 val houses.
  Acceptance: all 20 compile on MuJoCo 3.9; table of nbody/ngeom/step/ray time and door widths
  per house; disk < 4 GB. Rollback: delete `/data/local/ss5772/molmospaces`.
  ## 8. Open questions
  1. Is a kinematic (mocap) base OK for route demos, with base acceleration logged?
  2. Start with `procthor-10k`; textures (0.64 GB) only for videos?
## Notes (agent-written interpretation)
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
1. P
## Resume
Base rules: /common/users/ss5772/codex_tools/base/AGENTS.md
Start a fresh session in /common/users/ss5772/bimanual and say: "Read /common/users/ss5772/bimanual/docs/archive/status/2026-10-22_handoff_auto.md and continue from its open items."
- Full owner notes: docs/status/2026-10-22_owner_notes.md
