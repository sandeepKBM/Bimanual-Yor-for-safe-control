# Testbed candidate: a_loose_items (low-rim dish or tray holding N loose candy-like items)

Competitors: LatentSafe and LatentCBF (Skittles bag, real robot only). Failure = any item leaves the container.
Tags: VERIFIED = I ran or measured it. DERIVED = math. ASSUMED = not checked.
All runs: MuJoCo, CPU only, no GPU, no rendering, no hardware, no network. About 25 CPU-min used (estimate; the machine was shared).

## Verdict + 5 key numbers

| | |
|---|---|
| Verdict | **GO-with-changes (weak).** The dish is solved by a 70-state physics model. Only a straight low-rim tray leaves room for a learned WM, and there hold-still spills too often (19%). |
| 1. Failure rate, nominal carry (dish, N=10, 100 replayed spill_v2 action sequences) | **50%** (tray 45%, N=20 dish 50%). In band. VERIFIED |
| 2. Savable set (200 pre-failure states, dish N=10) | hold-still suffices **85%**; savable but not by hold-still **10.5%**; no certified save **4.5%**. All the non-trivial cases lie within 20 ticks of the failure. VERIFIED |
| 3. Lead time, first-unsavable to failure | median **3.5 ticks (70 ms)**, p90 5.5 ticks (tray: median 5.5). VERIFIED |
| 4. Physics vs MuJoCo (dish) | The reduced model fed the true item state predicts the 2 s outcome with **AUC 0.93, 91.5% accuracy**. With 3 mm position noise and no velocity: AUC 0.93 (same). Physics plus observed state suffices. VERIFIED |
| 5. Sim speed | **445 ticks/s** at N=10 (628 at N=5, 231 at N=20, 120 at N=30), one core. A 2 s branch costs about 0.2 s. VERIFIED |

Other facts that decide the verdict:
- Spills happen at tilt about 0.8 rad (dish) and 1.0-1.5 rad (tray). This is inside the 1.5 rad action domain. The cup spilled only at 1.7-1.9 rad. VERIFIED
- Hidden parameters (friction, bounce, item radius) flip at most 3% of spill labels. There is almost nothing for a BOED probe to learn. VERIFIED
- The real ZED is a floor-facing nav cam, so it probably cannot see the container contents at all. ASSUMED, confirm on hardware.

## 1. Failure and margin

Container. A flat floor disc (radius 6 cm) with 16 inclined wall boxes (8 mm thick). `beta_deg` is the wall angle from vertical.
- Dish: beta = 45 deg, rim lip 2.5 cm above the floor.
- Straight tray: beta = 0, rim 1.5 cm (also tried 2.5 cm).
- Items: spheres, radius 7 mm, density 900, friction 0.4, N = 10 (5 to 30 tested). A Skittle is about 14 x 7 mm, so a sphere is an ASSUMED analogue.
- Contact settings copied from the cup rig: condim 6, rolling 3e-4, solref 0.004/1, dt 5 ms.

Failure event. An item centre crosses the rim lip, going outward. Formally, in the container frame, with `s_i = (RF + z_i tan(beta) - rad_i) cos(beta)` (positive inside the wall plane):
- `l_i = s_i` if `s_i < 0`, else `max(H - z_i, s_i)`.
- `l_i < 0` exactly when `s_i < 0`. VERIFIED (it is the definition used by the sticky exit check).
- The rig margin is `l = clip(min_i l_i, -0.02, 0.025) / 0.025`, in [-0.8, 1].

Why this form: my first version used `max(H - z, s)` for all items. A fallen item then read l = +1, because H - z is large. I caught this in a static tilt test and fixed it. VERIFIED

Many items.
- Hard min over items: it follows the single worst item. Median warning (margin below 0.3 and staying below) comes **9 ticks** before the first exit (dish). VERIFIED
- Soft-min (log-sum-exp, tau = 4 mm): warns **14 ticks** before the exit on the dish. On the tray it stays low for 60 ticks because items often rest near the low rim, so it is a weak signal there. VERIFIED
- Energy margin (not implemented, DERIVED): `l_E = min_i [ U_rim(g_eff) - (0.7 |v_i|^2 + g_eff . p_i) ]`. The 0.7 is (1/2)(7/5) for a rolling solid sphere. `U_rim` is the lowest lip point under the effective gravity `g_eff = R^T(g - a)`. It is a necessary-not-sufficient signal, because `g_eff` changes in time.

Maps onto "spill" in the pipeline.
- `onset` = first tick that holds a rim exit. `l` is the current margin.
- `n`, `nd` = centroid of the items still inside, in the container frame, and its rate. These are privileged aux targets, as in bead_rep.
- Range [-0.8, 1] differs from the pendulum's [-1, 1] and the bead rig's [-0.4, 1]. Feed it through the bead_rep path, not through the pendulum value code. DERIVED
- Tunnel guard: an item whose first exit lies more than 12 mm below the rim height, or below the floor, is flagged TUNNEL and excluded. Tunnel episodes seen: **0** in all runs. VERIFIED

## 2. Physics representation

### 2a. Reduced analytic model (`reduced.py`)
- Container pose: driven by the pipeline's kinematic integrator (`bead_env.kin_step`). Input `u = [a(3), w(3)]`, the same 85-element set, held for 4 sub-steps of 5 ms.
- State per item: centre `p_i` (3) and velocity `v_i` (3) in the container frame. Total dimension **6N + 10** (70 for N = 10). Items do not touch each other.
- Dynamics: `a_eff = R^T (g - a_o) - 2 w_t x v - w_t x (w_t x p)`, with `w_t = R^T w`.
  - Free flight: `p'' = a_eff`.
  - Floor and wall contact: inelastic (the normal velocity is removed).
  - Rolling without slip while in contact: the tangential acceleration is multiplied by kappa = 5/7.
- Static escape tilt: `theta_crit = pi/2 - beta`. DERIVED. Measured for the dish: 0/10 items out at 0.7 rad, 3/10 at 0.8, 10/10 at 0.9 (prediction 0.785). VERIFIED
- Parameters. Known from CAD: ring geometry, rim height, item radius. Hidden or assumed: friction mu, bounce, rolling resistance, item shape, N, kappa. The model has no mu, no bounce and no item-item contact.
- Use (i), labeller for HJ: only valid for the dish (see 2c). Use (ii), plan-aware physics baseline: strong on the dish. VERIFIED

### 2b. MuJoCo representation (`tray_env.py`)
- Bodies: one free-joint container body, pinned each 5 ms step with `contents.pin_cup` (inertial carry). N free-joint spheres.
- Solver: option block copied from `yor_robot_derived.mjcf` (implicitfast, elliptic cone, impratio 10). No extra substeps.
- Speed, one core, shared machine, dynamic replayed carry (VERIFIED):

| N | 5 | 10 | 20 | 30 |
|---|---|---|---|---|
| ticks/s (1 tick = 4 steps) | 628 | 445 | 231 | 120 |
| 100-tick hold branch (s) | 0.11 | 0.14 | 0.16 | 0.22 |

- Determinism: snapshot and restore are bitwise (`mj_getState` INTEGRATION + WARMSTART plus the inertial-carry state). Replaying 60 ticks after a restore gave identical qpos and margins. A fresh reset is identical. VERIFIED
- Item count versus speed: N = 5 to 10 is cheap. N = 30 is 3.7x slower than N = 10. N = 10 to 20 keeps a sensible margin signal. ASSUMED for the choice of N.
- Item-item contact chaos: at N = 10 the outcome is not chaotic in the sense that matters. Changing friction, bounce or radius flips at most 1 of 30 episode labels, and the median onset shift is 1-5 ticks. VERIFIED (30 episodes, one layout seed each)

### 2c. Reduced model versus MuJoCo (`gap.py`, replayed spill_v2 test actions)

| container | open-loop episode label | plan-aware label from a mid-episode state (privileged state) | same, camera-like state |
|---|---|---|---|
| dish (40 episodes, 531 states) | agree 90%, TPR 100%, FPR 27%, AUC 0.87 | acc 91.5%, TPR 99.6%, FPR 14.3%, AUC 0.934 | acc 91.5%, AUC 0.932 |
| straight tray, rim 1.5 cm (20 episodes, 298 states) | agree 50%, TPR **0%**, AUC 0.78 | acc 70%, TPR **5%**, AUC 0.81 | AUC 0.72 |

(VERIFIED. "Camera-like" = 3 mm Gaussian position noise and zero relative velocity.)
- Dish: the escape is a slow roll up a slope. The reduced model captures it. Its errors are false alarms (FPR 14-27%), from missing friction loss and item-item blocking. The truth does not diverge from the proxy here.
- Tray: the escape is a hop. An item rolls fast into a 1.5 cm lip and bounces over it. The reduced model has no bounce and misses 95% of spills. Here truth does differ from the proxy, and a better reduced model with restitution could still close it. ASSUMED
- Zero velocity costs nothing on the dish, because items are mostly at rest relative to the container in the sampled states. VERIFIED

## 3. Observability (only the sensors on the real robot)

| quantity | source | status |
|---|---|---|
| Container pose, velocity | joint encoders of both Piper arms plus forward kinematics. The container is held rigidly (grasp slip is out of scope). | Observable. Acceleration comes from differentiating encoders, or from the commanded `u`. ASSUMED noise |
| Item positions | ZED, if the container is in view | Observable in principle. See below. |
| Item velocities | finite difference of detections | Noisy. The dish result says it does not matter. VERIFIED in sim |
| Item count, shape, mass, friction, bounce | none | Hidden. Weak effect (at most 3% label flips, VERIFIED). |

Camera.
- Size: a 14 mm item at 0.6 m spans about 16 px, if the focal length is about 700 px (HD720). DERIVED from an ASSUMED focal length.
- Rate: 30 fps is typical, up to 60 fps. ASSUMED. Latency is not known, so I assume 50-100 ms. ASSUMED
- Rim occlusion: the near rim hides a strip `h / tan(e)` of the floor for camera elevation `e`. For `h` = 2.5 cm: 4.3 cm at 30 deg (36% of a 12 cm floor), 1.4 cm at 60 deg. A 45 deg dish wall shows more than a vertical wall. DERIVED
- Arm occlusion: the arms hold the container edges. Not measured.
- Mount: the sim ZED is a proposed torso mount at z about 1.0 m. The real ZED is a floor-facing nav cam. **Most likely it cannot see the container. ASSUMED, confirm on hardware.**
- No force, torque or contact sensor exists. Torque estimation from motor current is not used here.

Failure onset. The onset is visible when it happens (an item crosses the rim). It is not hidden. The unsavable point comes **3.5 ticks (70 ms)** before the first exit on the dish and 5.5 ticks (110 ms) on the tray. VERIFIED. This is about the same as one camera frame plus latency (ASSUMED). A vision-based reaction cannot use that window. The controller must act on the carry plan about 20 ticks earlier. Every state that is 21 or more ticks before the exit is saved by hold-still (dish, VERIFIED).

Probing (BOED angle). A small tilt (for example 0.3 rad) shows whether items roll (acceleration kappa g sin(theta), kappa = 5/7) or slide. But the hidden parameters change fewer than 4% of spill labels. A probe has almost nothing to learn.

Class: **observable if the camera sees the container; physics suffices. Partially observable only because of the camera question.** For the dish, nothing hidden matters, so it is not "identifiable by probing" in any useful sense.

## 4. Feasibility probe (VERIFIED, `probe.py`, `gap.py`, `sens.py`)

Carry. First 300 ticks of the spill_v2 TEST action sequences (`act` = [a, w_eff]), replayed through the bead_env integrator with the 1.5 rad tilt projection. This is not the logged cup pose. The logged cup tilt reaches 2.37 rad, outside the action domain.
A synthetic carry (random held actions from the 85 set) was also tried. It is too violent (100% failure), so I do not use it for rates.

Savable test. A state is "certified savable" if one sequence avoids any rim spill for 100 ticks (2 s). Library, with early exit: hold `u=0`, then brake (decelerate and level), then the 84 held constants, then 40 random two-segment sequences followed by the brake tail. This is an under-approximation. "Not certified" means ambiguous, not doomed. Pre-failure state = a snapshot within 100 ticks before the first rim exit.

| config | episodes | fail rate | states | hold-still suffices | brake suffices | savable, not by hold | not certified | hold-still spill, all states | lead median (IQR) |
|---|---|---|---|---|---|---|---|---|---|
| dish, N=10 | 100 | **50%** | 200 | 85% | 8% | 10.5% | 4.5% | 7.5% | 3.5 (2.5-5.3) |
| dish, N=20 | 40 | 50% | 80 | 84% | 11% | 12.5% | 3.8% | 5% | 3.5 (2.5-4.5) |
| tray, rim 1.5 cm, N=10 | 100 | 45% | 180 | 58% | 32% | 36% | 5.6% | 19% | 5.5 (3.5-7.5) |
| dish, synthetic carry | 40 | 100% | 152 | 69% | 8% | 22% | 8.6% | 60% | 3.5 |

("Brake suffices" overlaps the "savable, not by hold" column.) The rate of tunnel artifacts was zero in every row.

By distance to the failure (dish, N=10):

| ticks before first exit | states | hold-still suffices | savable, not by hold | not certified |
|---|---|---|---|---|
| 1-20 | 50 | 40% | 42% | 18% |
| 21-60 | 90 | 100% | 0% | 0% |
| 61-100 | 60 | 100% | 0% | 0% |

Straight tray, same bins: 1-20: hold 0%, savable-not-hold 77%, not certified 23%. 21-60: hold 59%. 61-100: hold 92%.

Tuning caveat. The failure rate is a design choice of the container. A straight tray with a 2.5 cm rim failed in 1 of 30 replayed episodes (probe `q1`, 30 episodes; the run printed 1/30). I picked the dish (50%) and the 1.5 cm tray (45%) because they land in the 20-60% band. Expect a reviewer to ask why. VERIFIED (rates), the concern is a judgement.

Sampling caveat. 50 failing episodes, 200 states, one layout seed per episode. Failure rate 50% over 100 episodes has a binomial standard error of 5 points. The savable fractions come from states that are strongly correlated inside an episode.

Sim speed: see 2b.

## 5. Pipeline fit and WM value

Reuse directly.
- `bead_env.kin_step`, `Snapshot`, `brake_action`, `fallback_U` (imported, not edited).
- `contents.pin_cup`.
- The bead_rep oracle recipe (hold, brake, constants, random two-segment). I wrote a smaller version in `probe.py`.
- The dataset format (`obs, act, l, n, nd, lens, onset, seed`). Change `l` range handling to [-0.8, 1].

Must change.
- Observation. `kin_wm` sees only a 9-D kinematic vector `[axis, v, w]`. It sees no items. If the camera sees the items, add item features (they must be permutation invariant: centroid, spread, nearest-to-rim distance, soft-min margin). If the camera does not, the WM must infer item state from kinematic history. I did not test that case.
- Carry generator. Replace the act replay with the 200 Hz cup poses from spill_v2 (`latent_hj/spill_data.load_dense`), or accept the tilt projection.
- Item shape. Move from spheres to ellipsoids or lenses with rolling resistance (the real candy is a lens).

Where a learned WM adds value (honest).
- Dish: **it does not.** A 6N+10 physics model with the true item state gets AUC 0.93 and TPR 99.6%, and loses nothing when item velocity is unknown. Hidden parameters barely matter. A simple rule (is the effective tilt above `pi/2 - beta`?) captures most of the structure. This is the same problem that killed the cup, in the other direction: the contents are observable, physics suffices.
- Straight low-rim tray: possible. The reduced model finds 5% of spills (AUC 0.81). The cause is bounce over the lip, which is a contact event. A WM trained on MuJoCo could learn it. But a reduced model with restitution may also fix it, and I did not test that.
- Unobserved items (camera cannot see): a WM that infers slosh state from the kinematic history has a real job. Physics cannot do this without knowing the layout. This is the strongest argument, and it hinges on the unverified ZED mount. If the ZED cannot see the container, this object matches the thesis story. If it can, it does not.

## 6. Effort to a G1-like gate, risks, verdict

Gate: at least 300 certified-savable held-out states, nominal latent-HJ spill at least 30%, hold-still at most 10%.

Reading of the gate. I read "hold-still at most 10%" as the hold-still spill rate over all sampled states. Dish N=10: 7.5% (passes), N=20: 5% (passes), tray: 19% (fails), synthetic: 60% (fails). The meaning of "certified-savable" is ambiguous. If hold-still saves count, 300 states is trivial. If only non-trivial saves count, see below. I did not measure latent-HJ spill (no WM trained).

Non-trivial savable states, DERIVED from the bins. The non-trivial region is the last 20 ticks. Dish: 0.42 x 20 = about 8 states per failing episode at stride 1. About 50 failing test episodes give about 400 raw states. About 100 test episodes suffice, about 10 CPU-min for certification. The states within an episode are correlated, so the count of independent states is lower. Also, `test` has 703 episodes, so there is room.

Effort (ASSUMED):
- Integrate with `load_dense` replay, ellipsoid items, item-feature obs: 2-3 days.
- Port the full bead_rep oracle (300 random sequences) and generate train, val and test: 1-2 days, plus a compute day on CPU.
- Train kin_wm with item features, run the value gate: 2-3 days.
- About 1-1.5 weeks to a G1-like gate.

Top risks.
1. Physics suffices on the dish (AUC 0.93). The WM-value claim could fail the same way the cup did. VERIFIED
2. The savable set is thin and late. It sits in the last 20 ticks, with a median lead of 70 ms. The hold-still policy saves 85% of pre-failure states. A value function sees few states where the action matters. VERIFIED
3. Failure rate depends on container design (beta, rim height). It is tunable, so it looks chosen. VERIFIED
4. The ZED may not see the container on the real robot. It decides which story the object tells. ASSUMED
5. Spheres overstate rolling. The real candy is a lens. Hidden-parameter insensitivity may be a sphere artifact. ASSUMED
6. The tray variant breaks the hold-still gate (19%). The dish passes it by a small margin (7.5% against 10%; n = 200, so the interval is wide). VERIFIED

**Verdict: GO-with-changes (weak).** The dish has a healthy failure rate (50%) and an in-domain, non-empty savable set, but a 70-state physics model solves it. Continue only with the straight low-rim tray (hop-driven spills, 5% physics TPR), after retuning the rim height so hold-still spills at most 10%, or if the real ZED cannot see the container.

## Files
- Code: `/common/users/ss5772/bimanual/world_model/v2/testbeds/a_loose_items/` (`tray_env.py`, `carry.py`, `probe.py`, `reduced.py`, `gap.py`, `sens.py`, `README.md`).
- Numbers: `/common/users/ss5772/bimanual_data/testbeds/a_loose_items/feasibility.json` (raw per-episode json and logs in `tmp/`).
