# ASSUMED-value sweep, task box

Every swept value stays **ASSUMED (break-even, not measured)**. A break-even is the value that would be needed, not an accepted one. Baseline = today's values.

## cell 1.0-1.5
baseline: ok=False, failing rows: episode_U (margin -0.0008000000000000004)

| knob | today | target rows (margin) | break-even | change | next failing row(s) at break-even | justify (measurement that would make it MEASURED) |
|---|---|---|---|---|---|---|
| CLEAR_M_BOX | 0.03 | - | None | | knob's rows pass today; the baseline fails on other rows | the measured object-pose error at the lift / carry (perception + tracking) over the shelf edge |
| SUPPORT_APPROACH_DEG | 30 | episode_U (-0.0008) | 45 deg (linear estimate of the zero crossing 39.2308, not a run) | +15 deg | none: admits end to end | the geometry that defines 'resting on / approaching a support top' (a definition; justify by the measured approach direction of the real set-down) |
| U_OBJ_CLEAR_M | 5 mm | episode_U (-0.0008) | 4 mm (linear estimate of the zero crossing 0.0042, not a run) | -0.001 m | none: admits end to end | the object pose + shape error vs the world geometry (perception / calibration), measured on the robot |

## not swept

- rule1.K: not swept: no consumer. rule1.K and rule1.NODE_CAP are only RECORDED (rule1.py:153-154 assumptions, rule1_contents.py:621/643 `worst`, rule4.py:81); no search reads them (the R_K witness is the depth-1 planner admission). Raising them cannot change a verdict.
- rule1.NODE_CAP: not swept: no consumer (see rule1.K)
- rule4.MAX_ROUNDS / RETRACE_HALVINGS: not requested here; rule4_contents module globals, add to KNOBS if a Rule 4 near-miss needs them
