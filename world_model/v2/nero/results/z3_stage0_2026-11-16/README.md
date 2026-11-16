# Z3 Stage 0 (2026-11-16): is torque/payload utilisation non-decreasing in the acceleration a?
- What: `mono.py` evaluates `Envelope.worst(contents, keys, a, payload.MODE)` on the closed_chain DEFAULT box path (`closed_chain.static_path("box")`),
  every box cell (`carry_cert.cells("box")`), every segment, a = 0 .. 1.0 m/s^2 step 0.025.
- Result (`mono.json`): no decrease anywhere (violations = []); smallest step per cell 0.0009 .. 0.0041 (same for lift/carry/place in a cell).
- Use: the Z3 oracle may report `block_a_ge` for torque/payload failures (fails at a => fails at every a' >= a).  ASSUMED for other paths
  (measured on the default path only) and between grid points (sampled on the 0.025 grid).
