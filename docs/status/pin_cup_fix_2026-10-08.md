# pin_cup carrier fix (2026-10-08)

> Status (2026-11-05): HISTORY: record of the 2026-10-08 pin_cup fix (BIMANUAL_LEGACY_PIN=1 = old physics).

## Bug

`world_model/v2/data/contents.pin_cup` sets the carried body's (tray or cup) pose and velocity before each `mj_step`. The body stayed
DYNAMIC during the step, and it was light (tray 0.3 kg, cup 0.03 kg). So it reacted to its own contents within every step:
- An off-centre load tilted it and was pushed outward. The next pin snapped the body back.
- Gravity also made it fall 0.25 mm inside every step.

Net effect: on a level, still carrier, the contents drifted outward. A real tray or cup holds them still.

## Fix

`contents.make_carrier_kinematic` sets the carrier's mass to 1e4 kg and inertia to 1e2 kg m², and turns on gravity compensation.
`mj_setConst` then runs on a scratch MjData. `pin_cup` calls it on first use, so every caller gets it.
- The pose and velocity still come only from the pin. The carry kinematics (inertial carry, tilt) are unchanged.
- **`BIMANUAL_LEGACY_PIN=1` restores the old behaviour.** Use it to reproduce any result made before this date.

## Verification

Still-carry drift: 2 s, level, carrier held still after settling; carrier-frame xy displacement of the contents (mean / max):

| env | legacy | fixed |
|---|---|---|
| testbeds/b_tray_ball (and all of tray_rep) | 12.7 / 12.7 mm | 0.000 / 0.000 mm |
| testbeds/a_loose_items | 0.04 / 0.04 mm | 0.000 / 0.000 mm |
| testbeds/d_opaque_fill (60 grains) | 7.4 / 23.1 mm, steady churn | 1.8 / 7.0 mm, then 0.5-0.8 mm per 2 s of compaction |
| bead_rep (8 beads in a cup) | 1.2 / 2.3 mm | 0.014 / 0.022 mm |

Further checks:
- The carrier no longer falls inside a step: 0.000 mm, against 0.245-0.327 mm.
- The steady contact normal force is exactly m g.
- The carry still acts on the ball. Legacy vs fixed: inertial lag -16.1 vs -17.2 mm (0.5 m/s² for 0.5 s); tilt roll 54.6 vs 55.9 mm.
- Tests: sets + vlmprior 63/63, bead_rep 11/11.
- Bridge smoke checks:
  - With the legacy flag: ALL PASS.
  - Fixed: `f0_matches_stored` fails on episode 2, which dropped under the old physics. This is expected: the stored data predates
    the fix.
  - Determinism and resume pass in both.

## What it invalidates (needs regeneration on fixed physics before results are trusted)

- **tray_rep:**
  - the datasets data_piper_full, data_piper_trk_full, data_piper_tray_only, data_piper_pc_head_aim_wrist, and their oracle labels,
    branches and norms;
  - the trained F3 values (`runs/`);
  - the bridge, BOED (tables and arms), robust chain, and Experiments A / B / C / C-F5 / P8;
  - F5's eps_a calibration (`sets/validate_reach.py`), `boed/calib.json` (the mu_eff map was measured on the old physics), and the
    online-belief calibration in `vlmprior/online_calib.json`.
- **The cup spill data** (`data/gen_spill_data.py`) and anything trained on it.
- **bead_rep:** phases 0-1.
- **The testbed-objects study (2026-10-05):** a_loose_items and d_opaque_fill.

Not affected:
- c_block_tower: it already used a heavy, gravity-compensated plate and does not call `pin_cup`.
- Code that never calls `pin_cup`.

Size of the effect on the old results (VAL diagnostics; details in `reports/overnight_2026-10-08_vlm_prior_series.md`):
- Short bridge carries barely change: F3 19.0% vs 19.0%.
- Long 20-s BOED carries do change: all of F5's slow-carry drops, and about three quarters of P7's probe penalty, came from the bug.

Rollback: `git checkout world_model/v2/data/contents.py`, or keep the code and set `BIMANUAL_LEGACY_PIN=1`.
