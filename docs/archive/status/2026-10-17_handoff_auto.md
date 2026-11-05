# Handoff bimanual

> Status (2026-11-05): HISTORY: machine-generated handoff snapshot of 2026-10-17 (branch perception-teleop-2026-10-12); not current guidance.
Generated: 2026-10-17T01:39:27-04:00
Repo root: /common/users/ss5772/bimanual
Current: perception-teleop-2026-10-12 @ 66cd2d8 docs: root README (system requirements, install, quick start, optional envs, repo map); regenerate requirements.txt
## Worktrees
- perception-teleop-2026-10-12: 66cd2d8 docs: root README (system requirements, install, quick start, optio... (ahead ?, behind ?, dirty 22)
## Slurm
queued/running:
  534060 tower_dynF2b RUNNING 32:25 rlab4 None
  533599 tower_v3n RUNNING 32:53 rlab4 None
  533597 tower_dynF1 RUNNING 3:08:26 rlab3 None
  532911 jenga_cap_coder RUNNING 6:07:39 rlab2 None
finished:
  532916             tower_uncert_wmrep CANCELLED b+   00:05:01      ilab1
  532918             tower_uncert_wmrep CANCELLED b+   00:04:58      ilab1
  532928                  tower_dyn_cpu    COMPLETED   00:06:25      rlab4
  532942                  tower_dyn_cpu    COMPLETED   00:03:37      rlab4
  532944                     tower_eval      TIMEOUT   04:00:03      rlab4
  532946                       push_est    COMPLETED   00:03:03      rlab4
  532947                       ws5_push CANCELLED b+   00:00:00 None assi+
  532950                     ws6_insert       FAILED   00:00:04      rlab6
  532951                       ws5_push CANCELLED b+   00:00:00 None assi+
  532952                     insert_est    COMPLETED   00:55:50      rlab3
  532953                    tower_dynE1    COMPLETED   01:33:06      rlab1
  532954                    tower_dynE2    COMPLETED   00:58:59      ilab3
  532955                       ws5_push    COMPLETED   00:04:25      rlab1
  532982                    tower_v4seg    COMPLETED   02:28:58      rlab3
  532983                     ws6_insert CANCELLED b+   00:00:00 None assi+
  532984                 tower_v4fix_bt       FAILED   00:01:09      rlab3
  532989                     ws6_insert    COMPLETED   00:05:19      rlab4
  532990                       ws5_push    COMPLETED   00:03:16      rlab1
  533001                  uncert_pytest    COMPLETED   00:01:20      ilab1
  533010                  uncert_pytest    COMPLETED   00:04:13      ilab1
  533011                       tower_s2    COMPLETED   01:40:12      ilab3
  533014            jenga_cap_coder_llm CANCELLED b+   00:10:47      rlab2
  533022                  uncert_pytest    COMPLETED   00:04:09      ilab1
  533284                    tower_v4seg    COMPLETED   02:27:01      rlab1
  533598                    tower_dynF2 CANCELLED b+   02:33:01      rlab3
## Local GPUs
- GPU 0: 785 MiB, 0% util, owners: ss5772
- GPU 1: 2 MiB, 0% util, owners: (none)
- GPU 2: 1 MiB, 0% util, owners: (none)
- GPU 3: 1 MiB, 0% util, owners: (none)
- GPU 4: 1 MiB, 0% util, owners: (none)
- GPU 5: 1 MiB, 0% util, owners: (none)
- GPU 6: 3 MiB, 0% util, owners: (none)
- GPU 7: 1 MiB, 0% util, owners: (none)
## tmux sessions
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
## Owner notes (bimanual, 2026-10-17 night) -- READ docs/status/2026-10-17_tamp_status.md FIRST (full notes + ordered open items)
- Direction (USER 10-17): build the planner layer. PDDL spec -> PDDLStream -> our checkers as streams -> sim. VLM parked. The library moves to the planner level.
- Done: PDDLStream built (/data/local/ss5772/external/pddlstream). New package world_model/v2/house/tamp/. "(On tray desk)" plans in 0.2 s with every value checker-certified.
- Next 1: fix test_planner.py (5 fail under pytest only: a foreign `constraints` module shadows FD's translate).
- Next 2: review the agent-written executor.py (unreviewed). Run exec_smoke none / shield / gatekeeper.
- Next 3: run_tamp.py end to end + replan loop.
- The "MolmoSpaces Phase 0" open item below is STALE (phase1/phase2 and the h5 sweeps already exist).
## Resume
Base rules: /common/users/ss5772/codex_tools/base/AGENTS.md
Start a fresh session in /common/users/ss5772/bimanual and say: "Read /common/users/ss5772/bimanual/docs/archive/status/2026-10-17_handoff_auto.md and continue from its open items."
