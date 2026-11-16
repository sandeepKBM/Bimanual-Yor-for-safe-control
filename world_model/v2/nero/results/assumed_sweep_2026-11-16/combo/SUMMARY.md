# ASSUMED-value sweep, task box

Every swept value stays **ASSUMED (break-even, not measured)**. A break-even is the value that would be needed, not an accepted one. Baseline = today's values.

## combos

- {'PATH_MARGIN_MIN': 0.015, 'U_OBJ_CLEAR_M': 0.002, 'CLEAR_M_BOX': 0.04} cell [1.0, 1.5]: ok=False, failing: plan_geometry (None)  [ASSUMED (break-even, not measured)]

## not swept

- rule1.K: not swept: no consumer. rule1.K and rule1.NODE_CAP are only RECORDED (rule1.py:153-154 assumptions, rule1_contents.py:621/643 `worst`, rule4.py:81); no search reads them (the R_K witness is the depth-1 planner admission). Raising them cannot change a verdict.
- rule1.NODE_CAP: not swept: no consumer (see rule1.K)
- rule4.MAX_ROUNDS / RETRACE_HALVINGS: not requested here; rule4_contents module globals, add to KNOBS if a Rule 4 near-miss needs them
