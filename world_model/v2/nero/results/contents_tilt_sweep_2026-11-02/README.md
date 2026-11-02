# Tilt / spill use of the carry plan vs acceleration (2026-11-02, simulation only)

Code: `contents/tilt_sweep.py`. Raw: `/data/local/ss5772/contents/tilt_sweep_2026-11-02/runs.jsonl` (720 runs, 0 crashes).
Physics: rigid weld (ASSUMED), stiff weld solref (0.004, 1) (MODEL choice), wrist friction compensation `carry.FRIC_COMP` (gains
ASSUMED), FL-MPC with the lever payload split, m_hat = cell centre. Plan = `carry.PLANS` (ASSUMED, user-confirmed). Nominal runs only.
Worlds: the corners of each belief cell (box: mass {lo, hi} x contents at the 4 floor corners = 8; tray: all mugs at {lo, hi} packed
into each corner + a lean layout = 12), at peak plan accelerations 0.1, 0.25, 0.5, 1.0 m/s^2 (0.5 = a_nom).
Budgets: object tilt 10 deg, mug tilt 20 deg (ASSUMED); spill ratio 1 (MODEL quasi-static surface; slosh NOT modelled).
Guarantee type: SAMPLED (corners). Not a proof.

## Findings (max over corners; see SUMMARY.txt)
- Tilt is set mostly by the LOAD, not by the motion. Box at a = 0.1: 2.4 deg (0.5-1 kg) -> 7.1 deg (3.5-4 kg); going from a = 0.1 to
  1.0 adds only 1.0-2.6 deg. Roughly +1 deg per +0.5 kg at a = 0.5.
- Object tilt ~= the object-level tracking error (e_obj within ~0.1-0.5 deg of tilt): tilt is imperfect tracking under the load.
- Worst corner is always the heaviest load with the contents at (+x, -y).
- Box: tilt stays <= 0.8 of the 10 deg budget up to 3.0-3.5 kg at a <= 0.25 and up to 2.5-3.0 kg at a = 1.0; unsafe runs start at
  3.5-4.0 kg (a >= 0.5) and 4.0-4.5 kg (all a), first by joint torque saturation (tau = 1.00), then tilt > 10 deg.
- Trays (2/4/6 mugs, all fills): tilt <= 3.4 deg (0.34 of budget), mug tilt <= 3.4 deg, spill ratio <= 0.31 even at a = 1.0.
  The tray refusals (6 mugs >= 50 g) come from the 3 kg arm payload spec, not from tilt or spill.

## Do the corners bound the interior? (INTERIOR_VS_CORNER.txt, a = 0.5)
Interior = the sampled worlds of the certificate run (admitted cells) and of the ungated baseline (refused cells), n = 20 per cell.
17/18 cells: interior max <= corner max. Exception: tray4 (0.125, 0.25 kg/mug): interior tilt 2.50 deg vs corner 2.36 deg (+6%);
the tray corners use packed layouts only, a spread layout can tilt slightly more. Spill: interior <= corners in all 18 cells.
So the corner max is NOT a bound by itself; a measured margin (here max interior/corner = 1.06, n = 20 per cell) must be added,
or the tray corner set widened (spread layouts).
