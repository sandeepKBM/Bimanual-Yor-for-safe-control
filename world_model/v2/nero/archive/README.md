# nero/archive -- friction-pinch / initial-understanding scripts (archived 2026-11-03, USER: every grasp is a rigid weld now)

Not imported by live code.  Moved with `git mv` (history kept; nothing deleted).  Relative imports inside the moved files were rewritten to
absolute `world_model.v2.nero...` paths so they still import from here (they still need live modules such as pan.segments, pan.under_rim).
`conftest.py` here sets `collect_ignore_glob = ["*"]`: pytest does not collect archive/tests.  Run one by path with
`pytest --noconftest -p no:cacheprovider <file>` (not run since the move).

| old path (under world_model/v2/nero/) | new path (under archive/) | last commit touching old path | why archived |
|---|---|---|---|
| pan/hardworlds.py | pan/hardworlds.py | 55f1ea5 | friction-pinch hard-world pipeline; THETA, C_R, world_from_theta copied verbatim to pan/theta.py (live: contents/envelope.py) |
| pan/skills/grasp.py | pan/skills/grasp.py | c9c9822 | friction grasp-candidate coverage; _rot copied to pan/rim_geometry.py (live: pan/segments.py) |
| pan/cert/contact_model.py | pan/cert/contact_model.py | 0a31baf | friction two-claw hold LP (claw twist/internal force); grasp is a weld now |
| pan/cert/contact_falsify.py | pan/cert/contact_falsify.py | 0a31baf | MuJoCo falsifier of contact_model |
| pan/handle_grasp_sweep.py | pan/handle_grasp_sweep.py | 85be2c0 | friction handle-grasp sweep script |
| democ/grasp_smoke.py | democ/grasp_smoke.py | a177648 | real-friction box grasp smoke |
| democ/grasp_cert.py | democ/grasp_cert.py | 0a31baf | friction grasp certificate |
| democ/grasp_cert_calib.py, grasp_cert_score.py, grasp_cert_xcheck.py, grasp_cert_compliant_xcheck.py | democ/ (same names) | 0a31baf | grasp_cert calibration / scoring / cross-checks |
| democ/hold_profile.py, democ/hold_grid.py | democ/ (same names) | 8f453c1 | real-friction hold profile (no weld) and its grid merge |
| pan/tests/test_contact_model.py | tests/test_contact_model.py | 0a31baf | tests only archived contact_model |
| democ/tests/test_grasp_cert.py | tests/test_grasp_cert.py | 0a31baf | tests only archived grasp_cert |
| democ/tests/test_democ.py (2 tests: test_holdfix_*) | tests/test_hold_profile.py | 8f453c1 | tests only archived hold_profile; cut out of test_democ.py unchanged apart from the import path |
| pan/tests/test_skills_actions.py | tests/test_skills_actions.py | git mv (11-03) | AgileX removal: all 5 tests were skipif(is_nyu) (pan push/clamp flow dropped 11-01); skipif markers dropped in the archived copy |
| pan/tests/test_cert_runtime.py (test_stored_backups_are_fixed, test_v1_backup_flagged + the `ep` fixture) | tests/test_cert_runtime_pushclamp.py | git mv n/a (new file; the two tests were cut out) | AgileX removal: both were skipif(is_nyu); the 2 gate tests stay in pan/tests/test_cert_runtime.py |

New live modules (verbatim copies of values and code, single source now): pan/rim_geometry.py (RIGHT_APPROACH, RIGHT_JAW, APPROACH_IN,
EDGE_CLEAR, _perturb_pose from pan/under_rim.py; _rot from skills/grasp.py; under_rim.py now imports the first five from it) and pan/theta.py.

Kept live on purpose (still reachable): pan/under_rim.py, pan/observe_eval.py, pan/vizcoast/, pan/vizcoast_faithful/, pan/llm.py, and
pan/handle_grasp.run()/main() -- pan/segcert.worlds (used by pan/cert/runtime.py and run_trials.py) imports vizcoast_faithful.run_baseline.worlds
and observe_eval._random_thetas, and run_baseline / graspnet_sim call handle_grasp.run and under_rim.run.

Not edited (still name the old paths): results/contact_model_2026-11-01/** and results/contact_model_2026-10-30/** scripts
(import pan.cert.contact_falsify, pan.under_rim, ...), and the dated docs docs/paper/claims_2026-10-2*.md, claims_2026-11-02.md,
docs/status/pan_grasp_two_arm_2026-10-28.md.  Comments in contents/objects.py and pan/handle_grasp.py mention hardworlds / grasp_smoke by name.
