# Testbed d_opaque_fill: opaque container with hidden fill

Date 2026-10-05. Simulation and saved data only. CPU only. About 18 CPU-min used (limit 30).
Code: `world_model/v2/testbeds/d_opaque_fill/`. Numbers: `/common/users/ss5772/bimanual_data/testbeds/d_opaque_fill/feasibility.json`.
Tags: VERIFIED = I ran or measured it. DERIVED = math. ASSUMED = not checked.

## Verdict and 5 key numbers

**Verdict: NO-GO as the main latent-HJ testbed. GO-with-changes as the conservative-system / identification case.**
Reason: the savable set is large, but the doomed set is empty. A worst-case-fill pendulum already bounds the failure, and no safe probe reveals the fill.

| # | Number | Tag |
|---|---|---|
| 1 | Nominal failure rate **41%** (29/70 episodes; 14 `spill_v2` carries x 5 hidden fill levels). It rises with fill: 21, 36, 36, 50, 64%. This is inside the 20-60% band. | VERIFIED |
| 2 | Savable set: **84 of 84** hold-spills states (2 to 10 ticks before the first spill) are certified savable. **0** are unsavable. Hold-still suffices in 2 of 86. | VERIFIED |
| 3 | Lead time: hold-still dooms the state only **4 ticks** (80 ms, IQR 2-6) before the spill. First-unsavable lead is about 0 ticks. | VERIFIED |
| 4 | Analytic pendulum, with the true fill: **0** missed spills, **22/70** false spills. It calls 73% of episodes spills (MuJoCo: 41%). It labels **0/84** savable states as savable. | VERIFIED |
| 5 | Fill 0.05-0.21 kg changes joint-1/2 torque by **0.15-0.61 Nm** (static). Resolution is 0.25 Nm, headroom 0.65 Nm. Dynamic part: at most 0.10 Nm. A gentle probe sees the contents move rigidly (coupling 1.00-1.03 at every fill and type). | VERIFIED + DERIVED |

Sim speed: a 2 s branch costs 0.6-2.3 s of CPU (40-165 grains). Restore is bitwise exact. VERIFIED.

## 1. Failure and margin

**Failure.** Open container: any grain leaves through the rim. Same predicate as the cup (grain centre above the rim plane, or outside the outer radius). Sticky once it happens. VERIFIED (`opaque_env.py`, selftest).

**Closed bottle.** I considered three failure events and rejected all of them as a dynamic testbed.
- Tip or drop from shifting mass: this is grasp/slip, which is OUT of scope. ASSUMED.
- Spill: impossible when closed.
- Load limit: the arm must hold the extra mass inside the ±8 Nm feed-forward cap. I computed this with the YOR left-arm Jacobian at `LEFT_GRASP_Q`. Gravity at that pose is [-7.35, -7.36, -2.2, -0.93, 0.45, -0.05] Nm (VERIFIED: matches the owner's number). Shell 0.15 kg plus fill 0.21 kg adds about 1.0 Nm to joints 1 and 2, which is above the 0.65 Nm headroom. The dynamic slosh part is at most 0.10 Nm. So this failure is set by mass, which is static and predictable. It is not a good HJ target. DERIVED from VERIFIED wrenches, for one fixed arm pose. The arm moves in a real carry, so the Jacobian changes.

I use the **open container** as the object. The closed bottle is the same rig plus the load-limit note.

**Margin.** Per grain, in the container frame: `l_i = H - z_i` if radius <= R_out, else `min(H - z_i, 0) - (radius - R_out)`. Then `l = clip(min_i l_i, -0.02, 0.05) / 0.05`, which lies in [-0.4, 1]. It is the bead-rig margin with this container's R and H. `l < 0` if and only if a (non-tunnelled) grain is out. VERIFIED (selftest).

**Map to the pipeline.** `l`, `onset`, `n`, `nd` and `obs/act` follow the `load_split` keys. It can be a drop-in for `bead_rep` data. ASSUMED (not built).

**Limit of the margin.** At low fill, `l` stays at 1 until the grains are near the rim. At the sampled pre-spill states the median `l` is 0.09 (2 ticks before), 0.15 (4), 0.25 (6), 0.27 (about 10), 0.87 (about 30). VERIFIED. The margin gives about 0.2 s of warning.

## 2. Physics representation

### a. Reduced analytic model (labeller and plan-aware baseline)
- State: bob direction n (3), rate nd (3), container axis c (3). That is 9 numbers, 6 degrees of freedom. Same as `slosh_pend.py`.
- Input: u = [a (3), w (3)] from the 85-element set, held for one 20 ms tick.
- Equations (DERIVED, first slosh mode, Ibrahim): `w1^2 = (g xi / R) tanh(xi h / R)`, `L = g / w1^2`, `xi = 1.841`.
  `n'' = f/L - |n'|^2 n - 2 zeta w1 (n' - w x n)`, where `f` is `(g - a_pivot)` projected orthogonal to n, and the pivot sits h above the origin.
  Spill when `phi = angle(-n, c) >= phi* = atan((H_free - h) / R)`.
- Parameters: R = 0.030 m, H_free = 0.100 m (known). **Hidden:** fill height h, damping zeta, fill type.
- This container: `phi*` runs from 1.16 rad (h = 0.031 m) down to 0.44 rad (h = 0.086 m). That is inside the 1.5 rad action domain. This is the reason to use a tall, high-fill container.
- Code: `opaque_env.PendModel`. Matches `wm_audit.common.pend_rollout_torch` to 2e-15 on 60 ticks. VERIFIED.

### b. MuJoCo representation
- Bodies (VERIFIED, `opaque_env.py`): one free-joint container (12 box walls, thick cylinder floor, shell 0.15 kg), pinned kinematically each step with inertial carry (`contents.pin_cup`). N free-joint spheres: r = 6 mm, density 1400 kg/m3, friction (0.5, 0.001, 0.0001), condim 3, solref "0.01 1".
- Fill: 40-165 grains, settled height 0.031-0.086 m, mass 0.05-0.21 kg. The packing fraction is about 0.57. The loose layout settles inside temporarily tall walls, then the walls shrink.
- Timestep 5 ms. Integrator `implicitfast` (robot options). Solver CG, 50 iterations, **elliptic cone**. Newton costs O(nv^3): 3.1 ms/step at 47 grains, against 1.3 ms with CG. Pyramidal CG explodes at mu < 0.2. VERIFIED.
- Speed (loaded machine, load about 20; pessimistic): 1.5, 2.8, 3.8, 4.8, 5.8 ms/step at 40, 80, 110, 140, 165 grains. A 2 s branch costs 0.6, 1.1, 1.5, 1.9, 2.3 s. VERIFIED.
- Determinism: snapshot and restore replays with max diff 0.0 (`selftest.py`). VERIFIED.
- Tunnel: 12 of 98 sampled states had a wall or floor tunnel flag. I excluded them. The first rig run had a 4 mm floor and tunnelled. I thickened it to 20 mm. VERIFIED.
- Liquid: MuJoCo has no fluid. A low-friction, damped grain proxy (mu 0.03, drag 3/s) behaves like grains: critical tilt 1.39 and 1.14 rad, against 1.38 and 1.15 for rice at 103 and 155 grains (VERIFIED, ad hoc console run, saved in JSON). Its failure rate (43%) matches rice (50%) at 140 grains. So **liquid is analytic only** (the pendulum). A true liquid needs SPH or a different engine. ASSUMED.

### c. Reduced model versus MuJoCo (the gap)
Same carries, same true fill height. 70 episodes. VERIFIED.
- Failure rate: pendulum 73% against MuJoCo 41%. By fill (40 to 165 grains): 50, 64, 71, 86, 93% against 21, 36, 36, 50, 64%.
- The pendulum never misses a MuJoCo spill (0 false negatives). It raises 22 false spills. It is a **conservative bound** for grains.
- When both spill, the pendulum spills a median 22 ticks (0.44 s) earlier.
- Critical tilt (ramp 0.4 rad/s, ad hoc run): MuJoCo 1.38 and 1.15 rad, pendulum 0.93 and 0.65 rad, at 103 and 155 grains.
- Cause (DERIVED): a liquid surface stays level, so the margin falls when the surface reaches the rim. A grain pile holds its slope by friction and jams in a narrow bore. So a tilt that spills liquid does not spill grains. Spill also needs the axis to point past horizontal for low fill (median tilt at spill 1.66 rad at 40 grains).
- Savability: at the 84 savable states, the pendulum calls the state already spilled (`phi >= phi*`), so it finds 0 savers. MuJoCo finds a saver in all 84. The pendulum is wrong about *savable*, not only about *spill*.
- Savers are hard horizontal accelerations (about 2.5 m/s2, held for 2 s, no tilt change). They push grains into the wall. A real arm cannot follow that for 2 s. This is a limit of the pipeline's held-constant certificate. ASSUMED relevance.

## 3. Observability (only the listed sensors)

| Quantity | Observable? | How, rate, noise | Tag |
|---|---|---|---|
| Container pose, velocity (grasp is rigid) | yes | Joint encoders and forward kinematics. Rate and noise not measured here. | ASSUMED |
| Grain positions, fill height, fill type | **no** | Opaque walls. ZED (real mount is floor-facing) and wrist cams see the outside only. An open top is visible only from above, and no sensor has that view. | ASSUMED |
| Fill mass | partly | Joint torque from motor current. The repo notes `tau_` is uncalibrated current (TODO). **Confirm on hardware.** Static torque per kg: [2.87, 2.89, 1.02, 1.14, 0.58, 0.25] Nm/kg for joints 1-6. | DERIVED |
| Commanded torque | yes | Known to the controller. It is not a measured load. | |
| Spill onset | **no** | The first grain exit is hidden. A camera sees flying grains only afterwards. | ASSUMED |

**Mass resolution.** One 0.25 Nm step at joint 1 or 2 equals 0.087 kg. The fill range is 0.05-0.21 kg, so that is 2 or 3 bins. At joints 4-6 (0.05 Nm steps) one step equals 0.2 kg or more. Averaging may help. Mass maps to fill height only if bulk density is known, and it is hidden (rice, beans, water differ by about 20%). DERIVED.

**Lead time.** In the rig, hold-still becomes doomed a median 4 ticks (80 ms) before the first exit. The pendulum would warn about 22 ticks earlier but is too eager. VERIFIED. The controller cannot use the event itself. It must predict from the observed container motion.

**Probe.**
- A gentle probe (hold, then +1.5 and -1.5 m/s2 pulses of 0.2 s each) shows contents moving as one rigid body. Coupling is 1.00-1.03 for all five fills and for rice, friction 0.2 and liquid proxy. Ring-down is at most 0.008 N. The probe gives mass only (static Fz). VERIFIED.
- A tilt probe would reveal fill through the moment `M = m g h_com sin(theta)`. At 0.9 rad, M about the container origin is 0.014 Nm (40 grains) to 0.10 Nm (165 grains). A 20% error in bulk density moves M by about 0.003-0.02 Nm. That is below one 0.05 Nm step on joints 4-6. DERIVED.
- The probe that does reveal the fill is a near-critical tilt (1.1-1.4 rad). It is unsafe by definition: the critical tilt is the unknown. The only readout is a spill, and that is hidden too.

**Class: unobservable for the failure-relevant quantity** (critical tilt). Mass is partly identifiable if motor-current torque is trustworthy. Not identifiable by a safe probe with these sensors.

## 4. Feasibility probe (measured)

Rig: `opaque_env.OpaqueEnv`. It imports `bead_env.kin_step`, `fallback_U` and `contents.pin_cup` unchanged. Carries: the first 14 logged `spill_v2` episodes of shard 4 (a real nominal distribution, not synthetic). The container follows the logged 200 Hz pose, with free grains. 14 carries x fills (40, 80, 110, 140, 165 grains, mu 0.5), plus 14 at 140 grains with the liquid proxy.

**a. Failure rate.** 29/70 = **41%** (target 20-60%). VERIFIED. By fill (settled height, mass): 21% (0.031 m, 0.05 kg), 36% (0.049, 0.10), 36% (0.062, 0.14), 50% (0.072, 0.18), 64% (0.086, 0.21).
- 6 of the 14 carries peak below 0.8 rad of tilt and never spill.
- 72% of spills start below 1.5 rad, so they sit inside the action domain. Median tilt at spill falls from 1.66 (40 grains) to 1.24 rad (165 grains). The cup had 1.7-1.9 rad for all spills. A near-full container fixes that.
- The hidden fill prior is my choice. The 41% is a statement about that prior and these carries.

**b. Savable set.** 86 non-tunnel states, sampled 2, 4, 6 and about 10 ticks before the first rim exit (and a few at 30 and 60). Certificate: some single held action of the 85-set gives zero spill for 100 ticks. Search is gentle-first. This under-approximates savability.
- Hold-still spills in 84 of 86. All 84 are certified savable. 0 are unsavable.
- Hold-still suffices in 2 of 86. At 30 ticks before the spill, hold-still suffices in 13 of 13 states: the carry is still driving the tilt up at that time.
- Fraction "savable and hold-still spills": **84/86 = 98%** of the states in the last 0.2 s. The doomed fraction is about 0.
- To get 300 savable states you need about 18 CPU-min (3.5 s per state). VERIFIED (the 84 cost 390 CPU-s).

**c. Lead time.** First-unsavable to failure: about 0 ticks (0 of 24 unsavable at 2 ticks, tunnels excluded). Hold-doom to failure: median 4 ticks, IQR 2-6. The sampling grid has a 2-tick floor. VERIFIED.

**d. Speed.** See section 2b. Total probe compute about 960 CPU-s plus about 150 CPU-s of development. VERIFIED.

**Effective load (what the arms feel).** Wrench of the grains on the container (momentum balance, VERIFIED: static Fz matches mass to 1%), mapped through the YOR left-arm Jacobian at `LEFT_GRASP_Q`, grasp point 6 cm up the axis.

| Fill | Static joint-1/2 torque | Joints 3 / 4 / 5 / 6 | Dynamic part, joints 1/2 (peak) |
|---|---|---|---|
| 0.051 kg | 0.15 Nm | 0.05 / 0.06 / 0.03 / 0.01 | 0.03 |
| 0.127 kg | 0.36 Nm | 0.13 / 0.14 / 0.07 / 0.03 | 0.07-0.08 |
| 0.209 kg | 0.60 Nm | 0.21 / 0.24 / 0.12 / 0.05 | 0.08-0.10 |

Compare: resolution 0.25 Nm (joints 1-3), 0.05 Nm (joints 4-6); cap ±8 Nm (wrist about ±6.4). The mass is visible at joints 1-3 as 1 to 2 steps. The dynamic part is below one step everywhere. Gravity on joints 1/2 is already 7.35 of 8 Nm. A fill above 0.17 kg, with the shell, leaves no headroom. DERIVED (two carries, one arm pose).

## 5. Pipeline fit and where a WM adds value

**Reused directly.**
- `bead_rep.bead_env`: kinematic integrator, 85-element set, bitwise snapshot idea.
- `latent_hj.spill_data`: `spill_v2` carries.
- `wm_audit`: `fallback_U` and `pend_rollout_torch` (checked equal to my numpy pendulum).
- `kin_wm` and `vdirect`: unchanged on the data side if I write the `load_split` keys.

**Must change.**
- A new fill generator with the hidden fill drawn per episode, and its reset from a settled state. Done in the prototype.
- A margin-to-`l` writer in the `load_split` format.
- A per-state fill label kept out of the WM input.
- A Piper cap check: the 85-element set has savers at 2.5 m/s2 that the arm cannot hold.

**Where a learned WM helps.**
- Kinematics alone cannot reveal the fill, because the rig moves the container by command. So a WM cannot infer the fill. Its value is the **fill-marginalised** spill probability. It would learn the granular correction to the pendulum (73% against 41%).
- Plan-aware physics plus the observed state suffices for a **safe** bound: a pendulum at the worst-case fill has 0 false negatives in 70 episodes. VERIFIED. This is the same result that ended the cup.
- The pendulum is wrong for savability (0 of 84). That gap is real, but a WM trained on MuJoCo labels would learn it only from kinematic features. It cannot use the fill.
- Honest summary: for this object, physics plus observed state does the safety job. The WM adds calibration, not new information.

## 6. Effort, risks, verdict

**What the conservative system should certify.** "No spill for any fill up to the bound." The margin is monotone in fill (failure rate and critical tilt both move one way, VERIFIED on 5 levels), so the certificate reduces to the fullest allowed fill. A mass reading from torque tightens the bound. A fill-independent tilt cap near 1.0 rad certifies all five fills. It would curtail 8 of the 14 carries (57%, they exceed 1.0 rad). That is the price of not knowing the fill. DERIVED from VERIFIED ramp and carry tilts.

**Effort to a G1-like gate.** (At least 300 certified-savable held-out states, nominal latent-HJ spill at least 30%, hold-still at most 10%.)
- Savable states: 84 already exist. 300 take about 18 CPU-min. The count is not a risk.
- Data: 3941 `spill_v2` carries x one random fill at about 5 s each is about 5.5 CPU-h. ASSUMED scaling from 418 CPU-s for 84 episodes.
- Not measured: the latent-HJ spill rate of a nominal-trained WM, and the hold-still rate. I did not train a WM. The gate is not tested.
- Estimate: 1 to 1.5 weeks of agent-assisted work. Mostly dataset conversion and WM training. ASSUMED.

**Top risks.**
1. The doomed set is empty. Every pre-spill state is savable until about 2 ticks out. A nominal-trained value then has almost nothing to miss, or it fails by predicting the average over fills. The G1 spill-on-savable test may pass trivially or be uninformative.
2. Grain size: 6 mm grains in a 60 mm bore jam. Real rice (1-3 mm) flows more freely. No real-rice or real-water check exists. ASSUMED.
3. Liquid is not simulated. Only the pendulum covers it, and it is 1.8 times too eager on grains.
4. Held-constant savers (2.5 m/s2 for 2 s) are not arm-feasible. The certificate is pipeline-consistent but optimistic about arms.
5. Mass identification needs motor-current torque. ASSUMED, unverified.
6. Sample size: 14 carries, 70 episodes, 86 states, one shard, one arm pose, mu 0.5 mainly.

**Use in the thesis.** Use this object as the **"unobservable contents -> separate conservative system" case**. It matches Kim et al.'s prediction gap. Show: worst-case fill gives a safe bound; the cost is conservatism (57% of carries curtailed); mass from torque is the only partial identifier, at 2 to 3 bins. Do not use it as the main latent-HJ testbed. The main testbed needs observable contents and a non-empty doomed set.

**Verdict: NO-GO as main latent-HJ testbed. GO-with-changes as the conservative-system / BOED-identification case** (changes: add real-torque readback check on hardware, a true liquid or a pendulum-only liquid arm, more carries).
