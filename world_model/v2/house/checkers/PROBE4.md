# Checker 4 -- probe decision value (probe4.py)  [sim only; all probe models are ASSUMPTIONS]

Belief b = mass in [m_lo, m_hi] (+ optional mu interval). A(b) = candidate plans the checker accepts for EVERY member (corners; monotone checkers).
`decide(b, cands, probes)` -> `("execute", cand)` | `("probe", probe, value)` | `("infeasible", reason)`.

Value of a probe: true states on a uniform grid over b; posterior = b ∩ [m* ± 2σ]; frac = share of outcomes with A(posterior) non-empty;
net = frac − cost_s / 120 s. Pick argmax net among SAFE probes with frac > 0 (criterion="worst": lexicographic (all-outcomes-resolve, net)).
Note: net may be <= 0 for a barely-useful probe; it is still returned (progress > 0) and the value is reported.

Probe models (ASSUMED, not measured): LIFT_TEST_1 (1 arm, σ_m 0.20 kg, 6 s), LIFT_TEST_2 (2 arms, 0.15, 8 s), WEIGH (bimanual hold, 0.05, 15 s),
ASK (σ_m 0.10 kg, σ_mu 0.10, 30 s), PUSH_TEST (σ_mu 0.05, 5 s). Safe lift: m_hi <= n_arms * 1.5 kg * 0.9 (quasi-static, a = 0.2 m/s^2) AND grasp geometry exists.

Plug-in: pass `checker(cand, m, mu) -> bool | (bool, why)` (e.g. a wrapper over checkers/grasp3.check_grasp); default = pick/assign.py.

Planner (opt-in): `GraspPlanner(veto_mode="probe")` or env `BIMANUAL_PICK_VETO_MODE=probe`. Belief = [min(prior, VLM lo), max(prior, VLM hi)].
VLM plan vetoed AND rule plan inadmissible on that belief -> `ProbePlan` (mode "probe") or infeasible Plan instead of the rule plan.
Decision keys added only in probe mode: `veto_mode`, `probe`; source "vlm+veto+probe".
KNOWN INTEGRATION GAP: pick/session.py (not edited) treats any mode != "infeasible" as feasible and plans bimanual for non-"single" -> it must
learn mode "probe" (surface to UI / run the probe) before veto_mode="probe" is used in live execution.
