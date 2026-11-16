# ASSUMED-value sweep, task box

Every swept value stays **ASSUMED (break-even, not measured)**. A break-even is the value that would be needed, not an accepted one. Baseline = today's values.

## cell 2.5-3.0
baseline: ok=False, failing rows: rule4[hold_here@round0] (margin None), rule4[hold_lower@round0] (margin None)

| knob | today | target rows (margin) | break-even | change | next failing row(s) at break-even | justify (measurement that would make it MEASURED) |
|---|---|---|---|---|---|---|
| STOP_K_MAX | 4 | rule4[hold_here@round0], rule4[hold_lower@round0] | none on grid | | target rows fail | the longest stop time the task tolerates (0.3 s * 2^k); a task-timing ruling, plus the measured stop deceleration the arms track |

## not swept

- rule1.K: not swept: no consumer. rule1.K and rule1.NODE_CAP are only RECORDED (rule1.py:153-154 assumptions, rule1_contents.py:621/643 `worst`, rule4.py:81); no search reads them (the R_K witness is the depth-1 planner admission). Raising them cannot change a verdict.
- rule1.NODE_CAP: not swept: no consumer (see rule1.K)
- rule4.MAX_ROUNDS / RETRACE_HALVINGS: not requested here; rule4_contents module globals, add to KNOBS if a Rule 4 near-miss needs them
