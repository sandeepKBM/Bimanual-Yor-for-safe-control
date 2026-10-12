# Testbed (b) `b_tray_ball`: ball on a rim-less tray or dustpan, carried bimanually

Tags: VERIFIED = I ran or measured it. DERIVED = math. ASSUMED = not checked.
Numbers file: `/common/users/ss5772/bimanual_data/testbeds/b_tray_ball/feasibility.json`.
Code: `/common/users/ss5772/bimanual/world_model/v2/testbeds/b_tray_ball/` (README there).

```
+------------------------------------------------------------------------------------------+
| VERDICT: GO-with-changes.                                                                 |
| The testbed works: failure is inside the action domain, the rate is in band, and the      |
| savable set is large. But physics plus the observed ball state predicts failure about as  |
| well as MuJoCo does. A learned WM adds little. The value is the hidden rolling            |
| resistance, which a safe probe can identify. Reframe the WM claim around that.            |
|                                                                                          |
| 5 key numbers (MuJoCo, flat tray, ball r=3 cm, 100 g)                                     |
| 1. Nominal failure rate: 44% (n=200); dustpan 37.5%. Max tilt p99 = 0.39 rad (limit 1.5). |
| 2. Savable set (426 pre-failure states): 57% savable, 28% hold-still is enough,           |
|    15% not certified. Within 10 ticks of failure: 69% / 8% / 24%.                         |
| 3. Lead: savability is lost about 2 ticks (40 ms) before failure at full authority;       |
|    3 ticks at 0.4x and 5 ticks (100 ms) at 0.2x authority (QS model).                     |
| 4. Reduced models vs MuJoCo: 97.4% label agreement (QS), 96.7% (NE); median ball          |
|    position error 0.4-0.6 cm after 0.6 s. Physics + observed state suffices.              |
| 5. Hidden rolling resistance (0.0005 to 0.006 m) flips 35-40% of failure labels           |
|    (fail rate 85% to 10%). Mass, radius, sliding friction flip 1-3 of 40.                 |
| Speed: 2.9k ticks/s on one core = 29 two-second rollouts/s.                               |
+------------------------------------------------------------------------------------------+
```

## 1. Failure and margin

**Failure event (DERIVED, VERIFIED in sim).** The ball centre leaves the tray footprint.
After that the ball falls. The event is sticky.
Flat tray: 30 x 20 cm (ASSUMED size), four open sides.
Dustpan: the same plate with a 2 cm back lip at x = -0.15 m. Three sides are open.

**Margin.** `d` = signed distance from the ball centre to the open edge, measured in the tray frame.
`d > 0` inside the footprint. Outside, `d = -dist(centre, footprint)`. The lip side is ignored.
`l = clip(d, -0.04, 0.08) / 0.08`, so `l` is in [-0.5, 1].
The bead margin uses 0.05 m; a tray is bigger, so I widened the range to 0.08 m (a choice, not a rule).

**Mapping to "spill" in the pipeline.**
- `l < 0` is spill. `onset` = first tick with `d < 0`.
- `n` and `nd` (privileged, aux head) = ball position and velocity in the tray frame.
  For the cup they are hidden. Here the ZED sees them (see section 3).
- `obs`/`act` keep the 9-D and 6-D layout. The ball state must be added to `obs` (section 5).

**Geometric `l` or velocity-aware `l`?** I keep `l` purely geometric.
- HJ needs `l` to be a function of state only. `V(z)` then supplies the braking distance.
- A margin like `d - v_out^2 / (2 a_stop)` is already an approximation of `V`. Using it as `l` would count dynamics twice.
- It is still useful as a baseline: a "myopic shield" that reacts to `d` only.
- Speed matters here: the ball moves 1 cm per 20 ms at 0.5 m/s, and the savable boundary sits 2-5 ticks from failure
  (section 4c). A purely geometric `l` is nearly the final answer, which is why `V` adds little (section 5).

**Contact loss.** Tilt up to 0.39 rad (p99) never lifts the ball (VERIFIED: 0 tunnels, `N > 0`).
A fall-through guard exists (`z < r - 2 cm`). It never fired.

## 2. Physics representation

### 2a. Reduced analytic models (DERIVED; checked against MuJoCo in 2c)

**Inputs.** `u = [a (3), w (3)]` at 50 Hz, the same as bead_env.
`a` = tray-origin linear acceleration. `w` = commanded axis angular velocity.
The tray follows `bead_env.kin_step` (tilt limit 1.5 rad). Tray rotation `Omega = w - (w.c)c`.

**Parameters.** `r` radius, `m` mass, `k = I/(m r^2)` (0.4 solid, 0.667 thin shell),
`mu` sliding friction, `mu_roll` rolling-resistance length [m] (torque = `mu_roll * N`).
Mass cancels in rolling. `r` matters through the margin and `mu_roll / r`.
Hidden or uncertain: `mu_roll` (most), `k` (soft or hollow payload), `mu` (little), CoM offset (if non-uniform).

**QS model: quasi-static 5/7 model. State (x, y, vx, vy) = 4.** In the tray frame:

    rho_ddot = [ P_t(g - a) - (mu_roll/r)(N/m) * dir(rho_dot) ] / (1 + k)        N/m = -n.(g - a)

- Level tray: ball acceleration = -(5/7) a. Tilted tray: `a = (5/7) g sin(theta)` without resistance.
- With resistance: `a = (g sin(th) - (mu_roll/r) g cos(th)) / (1 + k)`. VERIFIED against MuJoCo (th = 0.1: 0.25 measured, 0.23 predicted).
- Static hold angle: `tan(th_c) = mu_roll / r`. For `mu_roll` = 0.0005 / 0.002 / 0.006 m and r = 3 cm:
  1.0 / 3.8 / 11.3 degrees. A tray tilted past `th_c` rolls the ball. This threshold depends on a hidden parameter. For the cup it is fixed.
- Ball at rest on an accelerating tray needs tilt `phi = atan(a/g)` toward the acceleration. Hence "anti-roll" carrying.
- Slip starts when `tan(phi_eff) > 3.5 mu` (solid sphere, `(2/7)|tangential demand| > mu N/m`).
  With `mu` = 0.4 that is 54 degrees, never reached. Only `mu < ~0.2` matters (VERIFIED: mu 0.15 flips 2/40 labels).

**NE model: full Newton-Euler rolling ball.** World-frame `(p, v, omega)` = 9 numbers.
Reduced: ball (x, y, vx, vy, spin) = 5, plus tray axis `c` = 2. Per 5 ms step it solves a 9x9 linear system
(`m A = F + m g`; `k m r^2 alpha = -r n x F + tau_rr`; `A - r alpha x n = A_contact + r omega x n_dot`).
It includes centrifugal, Euler and Coriolis terms of the tray rotation. It switches to sliding if `|F_t| > mu N`, and to airborne if `N < 0`.

**Use as labeller.** Both are ball-state-to-failure maps. Given `(ball, tray, mu_roll, k)` and a planned `u` sequence,
failure is a closed-form-fast rollout (QS: about 1 ms per 100 ticks in numba, DERIVED from wall time).
This is a plan-aware physics baseline and a cheap HJ label source.

### 2b. MuJoCo representation (VERIFIED)

- Bodies: `tray` (free joint, pinned each step with the bead_env inertial carry, box 0.15 x 0.10 x 0.003 half-sizes, optional lip box),
  `ball` (free joint, sphere r = 0.03 m, m = 0.1 kg, explicit inertia `k m r^2`).
- Contact: `condim=6`, friction `(0.4, 0.002, 0.002)` (sliding, torsional, rolling). Same on both geoms.
  `solref="0.01 1"`. The default 0.02 let the ball sink 5 mm and distorted rolling; 0.01 gives 1.3 mm.
- Options copied from the cup rig: `implicitfast`, elliptic cone, `impratio=10`, `timestep=0.005`. 4 steps per WM tick.
- Speed: 11.5k physics steps/s = 2.9k ticks/s = 29 two-second rollouts/s on one core (python driver; kin_step in numba).
- Determinism: bitwise identical on re-run (VERIFIED). Snapshot/restore: bitwise (`mj_getState` + kinematic state). 0 tunnels in all carry runs (about 1,000 episodes).
- `kin_step_fast` (numba) equals `bead_env.kin_step` to 5e-16 over 4,000 steps. `bead_env` is imported unedited.

### 2c. Reduced model vs MuJoCo gap (VERIFIED, flat tray)

Test: 426 pre-failure states x 87 action sequences, 2 s each. Model started from the MuJoCo state.

| model | label agreement | false-safe (MuJoCo fails, model safe) | false-fail | median ball error, 0.6 s |
|---|---|---|---|---|
| QS (4 numbers) | 97.4% | 2.2% | 5.6% | 0.6 cm (p90 1.5 cm) |
| NE (9 numbers) | 96.7% | 1.1% | 22.5% | 0.4 cm (p90 0.7 cm) |

- State-class agreement (hold-ok / savable / not-certified): QS 91.5%, NE 87.3%.
- The exact model is not better at labels. Its false-fails come from the 1 cm boundary: median `d` at the sampled states is 5 cm.
  I did not find the cause of the NE false-fails (rolling-resistance smoothing is one suspect; ASSUMED, not tested).
- Where the models are wrong or untested:
  1. **Same convention.** Both models use the MuJoCo rolling-friction convention. Agreement does not validate real tomato physics.
     A real tomato is soft, rate-dependent, not a perfect sphere, with non-uniform mass. None of that is simulated (ASSUMED gap).
  2. **Rotation impulses.** `w` is piecewise constant per tick, so tray angular acceleration is an impulse (up to 400 rad/s^2).
     A real arm tilts smoothly. The ball 10 cm from the origin sees a 0.2 m/s kick (DERIVED).
  3. **Lip.** Lip bounce is not in the analytic models. Gap measured on the flat tray only.
  4. **Tray is rigid and kinematic.** Real tray flex and arm vibration are absent (ASSUMED).
- Net: unlike the cup (truth differed from the pendulum proxy), here the proxy is nearly the truth, inside MuJoCo.

## 3. Observability (real sensors only)

| quantity | source | rate / noise | note |
|---|---|---|---|
| tray pose, axis `c`, tray velocity | encoders + FK of both Piper arms | ASSUMED 100-200 Hz, mrad-level | needs a rigid grasp (grasp/slip is out of scope); hand-eye and mount calibration add mm |
| ball position in tray frame | ZED: sphere centre in image, ray-plane intersect with FK tray plane | ASSUMED 30 Hz, 2 mm per frame (sim figure), latency 50-100 ms | no depth needed because the tray plane is known from FK |
| ball velocity | differentiate / filter the above | raw `sigma_v = sqrt(2) * 2 mm * 30 Hz = 0.085 m/s` (DERIVED) | failure speeds are 0.2-0.5 m/s, so filtering is needed |
| ball spin | not observable | none | `k` and spin enter only through the model |
| `mu_roll`, `k`, `mu`, CoM, mass | none directly | none | hidden |
| force/torque/contact | none | none | not used |

- **Visibility.** The ball is on top of the tray, 6 cm wide. Tilt stays below 0.4 rad, so the tray does not hide it. The forearm or gripper can hide it near the edges (ASSUMED).
  **Blocker:** the real ZED is a floor-facing nav cam, not the proposed torso mount. If the ball is not in view, the object is unobservable. Wrist cameras are a fallback (ASSUMED).
- **Latency vs lead.** Perception latency (50-100 ms) is as long as the lead near the savable boundary (40-100 ms).
  The controller must act on a forecast, not on the last frame. This is a filtering job.
- **Doomed-to-failure onset is not hidden.** The ball state and the parameters fix the outcome. The ball is seen drifting before the edge.
  Early on, "doomed" is ambiguous: hold-still still works in 17% of states at leads of 15 ticks or less (table below).
- **Mass is not needed.** Rolling accel does not depend on mass. mass x2 flips 1/40 labels (VERIFIED).
- **Safe probe (VERIFIED in MuJoCo, noise simulated).** Start with the ball at the centre. Tilt the tray about y by `theta` at 1 rad/s and hold 0.3 s.
  Fit `x(t)` for the ball acceleration. The 0.12 rad probe moves the ball at most 4.8 cm (edge is 15 cm along x), so it stays safe for all tested parameters.

  | `mu_roll` [m] | a(0.12 rad) [m/s^2] | a(0.30 rad) [m/s^2] | note |
  |---|---|---|---|
  | 0.0005 | 0.71 | 2.01 | 0.30 rad probe leaves the tray (20 cm) |
  | 0.002 | 0.38 | 1.66 | 0.30 rad probe leaves the tray (16 cm) |
  | 0.006 | -0.02 | 0.97 | ball does not move at 0.12 rad (inside static hold) |

  With 2 mm noise at 30 Hz the acceleration error is `sigma_a` ~ 0.15 m/s^2. The 0.12 rad probe separates the extremes by 0.7 m/s^2 (about 5 sigma);
  neighbours (0.002 vs 0.006) differ by 0.4 (about 2.7 sigma). `k` is only weakly identifiable (shell vs solid differ by 0.25 m/s^2 at 0.30 rad, under 2 sigma),
  and it is confounded with `mu_roll`. Only the gentle probe is safe in every case, so probe design must be certified per state (the BOED angle).
- **Class: partially observable.** The state is fully observable (if the ball is in view). The hidden parameter `mu_roll` is identifiable by probing. `k` is weakly identifiable.

## 4. Feasibility probe (VERIFIED)

Method. Object-only MuJoCo rig, same kinematic integrator and 85-element set as bead_env.
The spill_v2 cup trajectories cannot be replayed: in the 703 test episodes, 11.8% of ticks have tilt above 1.5 rad
and the 90th percentile tilt is 98 degrees (they pour). I used a documented synthetic carry (`carry.py`): 3 min-jerk transports of 15-45 cm
with a noisy tilt loop that tries to align the tray normal with effective gravity. Parameters are ASSUMED. I raised the intensity until the rate was in band,
so the 44% is partly tuned. The table shows the sweep.

**4a. Nominal failure rate (MuJoCo).**

| tray | carry intensity | n | failure rate | median failure time |
|---|---|---|---|---|
| flat | 1.0 | 100 | 2% | 3.5 s |
| flat | **1.7 (nominal)** | 200 | **44%** | 2.3 s |
| flat | 2.0 | 100 | 59% | 1.8 s |
| dustpan | 1.7 | 200 | 37.5% | 2.3 s |

The lip removes one of four sides and lowers the rate by about 15% relative. The failure needs tilt of only 0.1-0.4 rad, far inside the 1.5 rad domain.
(Cup: spills only at 1.7-1.9 rad.) The rate is steep in the intensity, so the band is easy to hit and easy to miss.

**4b. Savable set.** 48 failing episodes, 9 states each at 2, 5, 10, 15, 20, 30, 40, 60, 80 ticks before onset = 426 states.
Oracle = 85 fallback actions, each held 10 ticks (0.2 s), then "brake" (stop the tray, level it), 2 s horizon. Plus hold-still (u = 0) and brake-only.
"Savable" = hold-still fails and some library sequence survives. This under-approximates the true savable set. "Not certified" is ambiguous, not "doomed".

| class | all 426 | lead <= 10 ticks | lead <= 15 ticks |
|---|---|---|---|
| hold-still suffices | 28% | 8% | 17% |
| savable (hold fails, some action works) | 57% | 69% | 64% |
| not certified | 15% | 24% | 19% |

By lead (savable / hold-ok): 2 ticks 65% / 0%; 5: 73% / 4%; 10: 69% / 19%; 15: 50% / 44%; 30: 48% / 25%; 60: 59% / 39%; 80: 43% / 50%.
Near failure, hold-still almost never works. Far from failure it often does, because the cause was a later acceleration.
Certified-savable states: 243 from 48 episodes (about 5 per episode). 300 needs about 60 episodes, roughly 25 CPU-min of oracle.

**4c. Lead time.** Last certified-savable lead per episode: median 2 ticks (40 ms), the smallest sampled lead.
At lead 1 tick, QS finds 42% savable. So savability lasts almost until the edge at full authority (a up to 5 m/s^2, w up to 2 rad/s).

Authority sweep (QS model, 478 states, scaled action set and brake limits; DERIVED from QS, not MuJoCo-verified):

| authority (a, w) | savable overall | not certified at lead 1 / 2 / 3 | last-certified lead (median) |
|---|---|---|---|
| 1.0x (5 m/s^2, 2 rad/s) | 56% | 56% / 29% / 25% | 2 ticks |
| 0.4x, 0.5x | 47% | 81% / 54% / 33% | 3 ticks |
| 0.2x, 0.25x | 39% | 94% / 83% / 69% | 5 ticks (100 ms) |
| 0.1x, 0.15x | 40% | 96% / 92% / 77% | 5 ticks |

The savable fraction at leads of 8 ticks or more stays near 50% at every authority. Only the last 2-4 ticks are lost.
**Risk:** 5 m/s^2 is not feasible on the real Piper: the +-8 Nm ff cap is almost used by gravity on j1/j2 (7.4 Nm). Real authority is likely the low rows.
Real lead time is then 60-100 ms, about the perception latency (ASSUMED, not measured).

**4d. Speed (VERIFIED).** MuJoCo 29 two-second rollouts/s/core. Oracle: 0.8 s per state (87 rollouts, early stop on failure).
Compute used: about 32 CPU-min (limit 30; the oracle run alone was 24 CPU-min), at most 8 cores, no GPU.

**Hidden-parameter sensitivity (VERIFIED, same 40 open-loop carries replayed).**

| change | failure rate (nominal 45%) | labels flipped of 40 |
|---|---|---|
| mu_roll 0.0005 | 85% | 16 |
| mu_roll 0.006 | 10% | 14 |
| shell ball (k 0.667) | 38% | 3 |
| mu 0.15 / 0.8 | 50% / 43% | 2 / 1 |
| radius 2.5 / 3.5 cm | 40% / 53% | 2 / 3 |
| mass 0.2 kg | 48% | 1 |

`mu_roll` is the one parameter that matters. It is a rolling-resistance length of a soft contact, and its real value is not known (ASSUMED range).

## 5. Pipeline fit and WM value

**Reused directly (ASSUMED from the bead_rep README and code; I did not run these).**
- `wm_audit.common.fallback_U` and the action meaning, via `bead_env.kin_step` (imported).
- The `load_split` dataset format (`obs, act, l, n, nd, lens, onset, seed`), the snapshot/branch machinery pattern of `bead_rep/gen_nominal.py` and `oracle.py`.
- `latent_hj` / `vdirect` training and the counterfactual-branch recipe (action coverage). I did not read their code.

**What must change.**
1. `obs` must include the ball: `[c(3), v(3), w(3), ball x, y, vx, vy (4)] = 13`. Without it the WM cannot see the ball,
   and the object turns into the unobservable cup case. With the real ZED the ball is observed, so adding it respects the robot-fidelity rule.
2. `l` range [-0.5, 1]; `n, nd` become ball position and velocity (now observable, no longer aux-only).
3. A real nominal carry source (this probe's synthetic carry is a stand-in), randomised `mu_roll` per episode, and a payload-detection failure model (latency, dropout).
4. Counterfactual branch generator on `TrayEnv.snapshot()`. At 29 rollouts/s/core, 30k two-second branches cost about 17 core-minutes. That is cheaper than the cup.

**Where a learned WM adds value. Honest answer: little for the predictor.**
- Physics plus the observed ball state predicts failure with 97% label agreement and 0.4-0.6 cm error (section 2c).
  That is the situation that killed the cup. The state is observed, the dynamics are simple, and there is no unobservable content.
- The savable boundary is within 2-5 ticks of the edge. `V` is close to `l` plus a braking distance. A learned `V(z)` has little to add over that formula.
- Where physics + state falls short:
  1. **Hidden `mu_roll`** (35-40% label flips). The WM must infer it from ball motion history. The fair baseline is physics plus an online `mu_roll` estimator, which I expect to tie or win (ASSUMED, not tested).
  2. **Non-modelled payloads:** soft or egg-shaped tomato, non-uniform mass, lip bounce. These are the only places a learned model can beat the closed form. They are not in this rig yet.
  3. **Latency and noise:** forecasting a ball seen 50-100 ms late with 2 mm noise. A filter does this, learned or not.
- So the thesis claim for this object is not "a WM supplies the predicate". It is "a certified safe probe identifies `mu_roll`, and the safety value uses it". That is the BOED part.

## 6. Effort to a G1-like gate, risks, verdict

**Gate pieces.**
- >= 300 certified-savable held-out states: already 243 from 48 episodes. About 60 held-out episodes and 25 CPU-min give 300+. Easy.
- Hold-still <= 10%: hold-still suffices in 8% of states within 10 ticks of failure, and 28% overall.
  The overall figure fails a 10% bar unless the gate counts only near-failure states. I do not know how G1 defines it. Needs the owner's definition.
- Nominal latent-HJ spill >= 30%: NOT MEASURED. No WM was trained. Since near-edge savability is almost geometric, a value trained on nominal data may rescue
  most savable states, which would fail this bar. I expect the bar to be hard here (ASSUMED).

**Effort (ASSUMED).** About 4-6 working days: carry source and parameter randomisation (1 d), observation change plus dataset generator on `TrayEnv` (1-2 d),
WM and latent-HJ training with the existing code (1-2 d), oracle certification of held-out states (0.5 d), a physics + `mu_roll` estimator baseline (1 d).
Compute is small (tens of CPU-min for data, existing GPU recipe for training).

**Top risks.**
1. **WM value.** Physics + state may tie the learned WM on every metric. The thesis needs a reframing toward parameter inference.
2. **Real authority and latency.** Lead time at realistic authority is 60-100 ms, about the latency of the ZED. The 5 m/s^2 action set overstates what Piper can do.
3. **Camera.** The real ZED is floor-facing. If the ball is not visible, there is no state, and the object fails the fidelity rule.
4. **Sim-to-real.** `mu_roll` and its range come from MuJoCo's convention. A real soft tomato may roll differently. The carry distribution and its 44% rate are tuned, not measured.
5. **Steep rate.** The failure rate moves from 2% to 59% over a 2x change in carry intensity, so the band is fragile.
6. **Perception of the probe.** The gentle probe separates `mu_roll` extremes at about 5 sigma, but neighbours at about 2.7 sigma with an assumed 2 mm noise.

**Verdict: GO-with-changes.** The object is the cleanest testbed measured so far: failure sits well inside the action domain,
the failure rate is in band, the savable set is large and certifiable cheaply, and a hidden parameter with a 12x range is identifiable by a safe probe.
The change needed: do not claim the learned WM beats physics. Make the claim about identifying `mu_roll` with certified probes, add the ball to `obs`,
and settle the G1 definition of "hold-still" before building.

## Files
- Report: `/common/users/ss5772/bimanual/reports/testbed_objects/b_tray_ball.md`
- Code: `/common/users/ss5772/bimanual/world_model/v2/testbeds/b_tray_ball/` (`tray_env.py`, `analytic.py`, `carry.py`, `probe.py`, `authority.py`, `hidden.py`, `selftest.py`, `README.md`)
- Numbers: `/common/users/ss5772/bimanual_data/testbeds/b_tray_ball/feasibility.json` (and `savable_rows.npz`)
