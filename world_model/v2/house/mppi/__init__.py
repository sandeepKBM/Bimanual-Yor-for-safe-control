"""house/mppi -- GPU sampling-MPC (MPPI / CEM / predictive sampling) PLANNER that sits ABOVE the hard safety layer.  SIMULATION ONLY.

The planner only PROPOSES a base twist (and optionally a tray tilt jog).  The proposal goes through the existing, unchanged hard filters
(teleop shield base QP / zono_rows BaseFilter), which stay the safety authority.  See README-free docstrings in each module:
  dynamics.py   batched torch rollout: slew-limited planar base, cup inertial acceleration, tray tilt
  fields.py     clearance fields: 3-D perceived-zonotope sd_lb lookup grid, 2-D wall EDT, person discs
  cost.py       CarryProblem: progress / intent + clearance + mu-robust (CVaR) spill risk + smoothness
  planner.py    SamplingPlanner (MPPI | CEM | PS), warm-up + CUDA-graph capture, READY flag
  s2.py         S2 desk scene: planner + BaseFilter closed loop (experiment driver)
  carry.py      h5 teleop carry: planner as a command proposer in front of the shield (experiment driver)
"""
