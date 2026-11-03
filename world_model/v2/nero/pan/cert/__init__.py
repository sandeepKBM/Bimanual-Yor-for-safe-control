"""pan/cert -- certificate v2 for the covered pan (plan approved 2026-10-29).  SIMULATION ONLY.

Every segment gets a STORED backup that does not depend on theta, certified for every theta in the robot's belief Theta_t
(exists b, for all theta) from every state the segment can reach, ending in S and R.  The robot gates on that certificate before
the segment and keeps the backup armed while it runs.  The certificate simulates the robot's BELIEF, never the truth; truth only
scores the result.

    belief.py       Theta_t: pose prior (world) x aim residual x friction x contents polytope in (m, s_u, s_w); torque-row update
    model_check.py  exact corner checks over the belief (MODEL), e.g. COM over the support at rest, planned path accelerations
"""
