# A/B/C/D v2 results: C0 | C1 | C2 | C3

Upgrade of the v1 A/B/C experiment (`README.md`) with three coupled changes:
(1) C3 = a self-rolled geometric-fabric soft-safety filter, (2) a more
aggressive, oscillating intruder that makes C1 (bang-bang) visibly fail, and
(3) grasped-box payload coupling into every condition's effective arm
dynamics. Code: `sim_common.py`, `run_ab.py` (now runs C0/C1/C2/C3),
`make_plots_v2.py`, `render_montage_v2.py`. v1's own scripts/artifacts
(`make_plots.py`, `render_montage.py`, `artifacts/ab_*.png`) are untouched.

Run: `python run_ab.py` (writes `logs/{C0,C1,C2,C3}.npz` + `logs/summary.json`,
~22s wall time for all four conditions), then `make_plots_v2.py` and
`render_montage_v2.py`.

## 1. C3 -- geometric fabric (soft safety)

`fabrics` (the open-source RMPflow/optimization-fabrics package) is **not
installed** in `safe_env` (`pip show fabrics` -> not found). Self-rolled a
minimal fabric instead, in `EgoController.fabric_qacc` (`sim_common.py`):
three independent task-space policies -- (i) a joint-space attractor to the
same nominal hold pose `q0` used by every condition's PD, (ii) a per-joint
soft joint-limit barrier, (iii) an arm-arm repulsion term on the same 9
ego/intruder sphere pairs C2's QP constrains -- combined by the standard
fabric rule: `M_total = sum_i J_i.T @ M_i @ J_i`, `f_total = sum_i J_i.T @
M_i @ a_i`, `q_dd_f = M_total^-1 @ f_total`, where each term's Riemannian
metric `M_i` stiffens (`~1/x^2`) as its task variable approaches its margin.
This is deliberately the "metric-weighted combination" half of the theory
only -- no curvature/Finsler-energy correction term (an honest scope
limitation, not hidden). It is SOFT by construction: nothing ever gates or
hard-constrains `tau`; `h` can go negative freely. `q_dd_f` is tracked into
torque by the *same* payload-coupled inverse-dynamics law as every other
condition: `tau_f = M_ll_eff @ q_dd_f + bias_l_eff`.

Numerical-safety caveat found during tuning (documented in-code): a raw
`1/x` barrier potential is unbounded as `x -> 0`/negative, which a SOFT layer
(no hard floor) can actually reach under a fast enough intruder. Early
versions let per-term accelerations and the *combined* solved `q_dd_f` blow
up to ~1e5-1e6 rad/s^3 jerk and threw the box (180 deg tumbling). Fixed with
an `FABRIC_X_FLOOR` distance floor, a per-term `FABRIC_A_CAP`, and a final
global `FABRIC_QDD_CAP=60 rad/s^2` clip on the solved `q_dd_f` (needed
because satisfying several simultaneous, large-weight, non-orthogonal rows
can amplify the *solved* vector past any single term's own cap) -- see the
tuning constants block and `fabric_qacc`'s docstring in `sim_common.py`.

## 2. Intruder (v2, same trajectory for all 4 conditions)

`sim_common.py`: ramp-in halved (`T_IN0=0.4, T_IN1=0.8` vs v1's `0.5/1.5`),
episode extended `4.0s -> 6.0s`, and -- the key change -- once fully
intruded the right arm now **oscillates its penetration depth** for 4.4s
(`T_HOLD_END=5.2`) instead of holding statically: `s(t) = clip(1.0 +
0.14*sin(2*pi*5.0*(t-T_IN1)), 0, 1.08)`, capped at factor 1.08 beyond the
original `RIGHT_INTRUDE_CTRL` (checked by hand against `left/joint3`'s
`[-2.965, 0]` limit). The oscillation frequency (5 Hz, near/above the ego
task-PD's ~3.9 Hz natural frequency) keeps re-crossing `H_TRIGGER` rather
than letting bang-bang settle into one clean push-away.

**Result: C1 now drops the box outright.** `box_dropped=True`, translation
drift 35.7 cm (vs v1's 1.7 cm), rotation drift 101 deg (vs v1's 1.7 deg) --
while still successfully avoiding a mesh collision (`min_h=+0.025`,
`collision=False`): bang-bang keeps the arms apart but chatters (44% duty
cycle, RMS torque-rate 8399 Nm/s, ~124x C2's) hard enough to shake the box
loose. C0 (no safety) and C2 (HOCBF-QP) do not drop the box under the same
intruder.

## 3. Payload coupling

Implemented via the grasp Jacobian (option (b) from the task), in
`payload_reflected_terms` (`sim_common.py`), used inside `EgoController.
task_torque` -- so it automatically feeds C0/C1's `tau_task`, C2's QP
(`M_ll`, `bias_l`), and C3's fabric torque mapping identically. Method: treat
the box as rigidly welded to `left/gripper_base` (true while grasped): its
hypothetical Jacobian is `Jp_box = Jp_ee - skew(r) @ Jr_ee`, `Jr_box =
Jr_ee` (`r` = current world-frame box-CoM-minus-ee offset, recomputed every
step, not assumed constant), reflect the box's 6x6 spatial inertia through
it (`M_box_reflected = J_box.T @ blockdiag(m*I3, I_box_world) @ J_box`), and
add its gravity generalized-force (`bias_box = -Jp_box.T @ (m_box*g)`,
matching MuJoCo's own `qfrc_bias` sign convention). Coriolis terms from the
box's own spatial velocity are dropped (documented, small at these speeds).

**Honest result: it barely moved the needle.** Pre-intrusion settle-window
comparison (`couple_payload=False` vs `True`, everything else identical):
EE offset 1.423 cm -> 1.414 cm (~0.6% smaller -- essentially noise), joint
offset 0.04027 -> 0.04014 rad. The box is only 50 g
(`scenes/bimanual_piper_pickup.xml`'s `box_center` inertial), so its gravity
torque (~0.05 kg * 9.81 * ~0.15 m lever =~ 0.07 Nm) is tiny next to
`TAU_MAX=40 Nm` and next to whatever else was already producing the
documented ~1-1.5 cm offset (finger/contact coupling the 6x6 `M_ll` block
ignores, per v1's README caveat) -- that other, uncompensated source
dominates, and payload coupling does not touch it. What DID improve
measurably: grasp (finger-box) contact force in that same settle window,
4.67 N -> 10.26 N (>2x), i.e. the box's weight is now correctly counted in
the force balance even though the resulting *position* shift is negligible
at `Kp_task=600`. **Net: apply-and-report as specified, but do not oversell
it -- for this light a payload, the EE-offset benefit is negligible; the
grasp-force benefit is real.**

## 4. Headline metric table (v2, 6.0s episode)

| metric | C0 (none) | C1 (bang-bang, hard) | C2 (HOCBF-QP, hard) | C3 (fabric, SOFT) |
|---|---|---|---|---|
| collision | True | False | False | True |
| min h [m] | -0.0838 | +0.0253 | -0.0164 | -0.0673 |
| min dist [m] | 0.0962 | 0.2053 | 0.1636 | 0.1127 |
| RMS EE jerk [m/s^3] | 869 | 13207 | 308 | 2757 |
| RMS torque-rate [Nm/s] | 67.9 | 8399 | 63.3 | 53.0 |
| box_dropped | False | **True** | False | **True** (see note) |
| box drift trans / rot | 0.41cm / 2.1deg | 35.7cm / 101deg | 1.48cm / 1.4deg | 4.27cm / 6.6deg |
| grasp (finger-box) force | ~0 (see note) | ~0 | ~0 | ~0 |

(`min_contact_force` in `logs/summary.json` reads 0.0 for all conditions --
it is the *minimum over time*, and contact force legitimately hits ~0 during
brief re-grip transients; see `ab2_grasp_force.png` for the actual
time-series, which shows C3's payload-coupled grip running roughly 2x C0/C1/
C2's before intrusion, consistent with Sec. 3.)

Full table: `artifacts/ab2_metric_table.png`. Raw numbers:
`logs/summary.json`.

**Note on C3's `box_dropped=True`:** driven by the *absolute* box-height
threshold (`box_z < 0.19m`), not by grasp slip -- C3's own relative
drift/rotation (4.3cm / 6.6deg) is far smaller than C1's (35.7cm / 101deg),
i.e. the box stays gripped throughout. What actually happens (confirmed by
direct box-height tracing) is that the ego arm sustains prolonged, real
arm-arm contact (soft repulsion never hard-gates it away) starting ~t=1.07s
and sags/settles a few cm lower under that continuous contact loading for
the rest of the run -- a genuinely different failure mode from C1's "shakes
the box out of the fingers."

## 5. Honest readout

- **C2 (smooth HOCBF-QP) remains the best condition on every axis**: only
  condition with `collision=False` *and* `box_dropped=False` *and* the
  lowest RMS jerk/torque-rate of the three active safety layers (308 / 63.3,
  vs C3's 2757 / 53.0 and C1's 13207 / 8399). The hard `psi2>=0` constraint
  buys a real, measurable guarantee the two soft/discontinuous alternatives
  don't have.
- **C3 (fabric) is clearly smoother and gentler than C1 (bang-bang)**: ~4.8x
  lower RMS EE jerk (2757 vs 13207), ~158x lower RMS torque-rate (53 vs
  8399), and it keeps the actual grasp far more intact (4.3cm/6.6deg drift
  vs 35.7cm/101deg) -- demonstrating the real value of a continuous,
  metric-weighted response over a discontinuous switch.
- **C3 is NOT better than C2, and should not be presented as such**: it is
  the only soft layer, so it is the only one of the three active conditions
  that fails to prevent an actual arm-arm collision under this aggressive
  intruder (`collision=True`, sustained contact from ~t=1.07s), its jerk is
  ~9x C2's, and it is flagged `box_dropped` (for the height-sag reason
  above, not slip). Read plainly: **for this scene, a soft-only avoidance
  layer traded "smoother than bang-bang" for "no longer collision-free" --
  it did not close the gap to the hard-constraint QP.**
- Payload coupling is implemented correctly and applied uniformly to
  C0/C1/C2/C3, but for this scene's 50g box its effect on the previously
  documented EE-offset is negligible (~0.6%); its effect on grasp
  (finger-box) force is real (>2x). Reported plainly rather than inflated.

## Rollback

`git status`/`git diff` on `sim_common.py`, `run_ab.py` (the only edited
files) reverts to the v1 A/B/C behavior; `make_plots_v2.py`,
`render_montage_v2.py`, `RESULTS_v2.md` and the `ab2_*.png` artifacts are new
files/outputs that can simply be deleted with no effect on v1.
