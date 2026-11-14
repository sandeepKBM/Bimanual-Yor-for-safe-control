CALIB.json for probe method "claw_guard" (per-claw load guard, Stage 0b 11-12); the probe default since 2026-11-13 (user ruling).
- Source: /common/users/ss5772/bimanual_data/contents/probe_fix_2026-11-12/calib_claw_guard/CALIB.json (copied unchanged), n = 20 calibration worlds per task (box, tray6, pot), code rev in
  /common/users/ss5772/bimanual_data/contents/probe_fix_2026-11-12/CODE_REV; per-rung records: /common/users/ss5772/bimanual_data/contents/probe_fix_2026-11-12/calib_claw_guard/calibrate.jsonl (not copied, 1.1 MB).
- EPS_LO 0.0542 kg, EPS_FREE 0.0061 kg (total load), EPS_CLAW 0.0332 kg and K_STEP_CLAW 2.1149 kg/mm (per claw): MEASURED, max over n
  (in-sample; not bounds). The joint-torque read noise (TAU_NOISE +-0.05 Nm per joint) is ASSUMED and added per read.
- Decisions use per-arm loads from joint torques only; the simulator's per-claw constraint force is logged as a diagnostic.
sha256: 26f3fc58a309d4a68f3dafdbc61232ba740adbb88130c0600bf1a4bb38768e4b
