# A/B Safety-Filter Experiment v3: Re-pointed to the Canonical YOR Scene

Re-points `sim_common.py` / `run_ab.py` from the old simplified rig
(`scenes/scene_bimanual_pickup.xml`, arms 0.5m apart, floor-mounted) to the
canonical proper-YOR scene (`scenes/scene_yor_canonical.xml`, arms 0.185m
apart, lift-mounted at z~0.67m) and re-runs the C0/C1/C2/C3 comparison on the
real geometry.

## 1. Re-point changes

- **Scene / timestep**: `SCENE = scenes/scene_yor_canonical.xml`. `DT`
  changed from 0.002 to **0.005s** (the canonical scene's own
  `<option timestep>`; the old assert on `model.opt.timestep` caught this
  immediately). `N_STEPS` recomputed (1200 steps @ 6.0s).
- **Names**: `left/joint{1..6}` -> `left_arm_joint{1..6}` (mirrored for
  right/fingers/gripper_base/actuators), `box_center` -> `test_box`
  (geom `test_box_geom`, freejoint `test_box_free`), `left/link7_pad` ->
  `left_arm_link7_pad`, etc. `EGO_BARRIER_BODIES` = `left_arm_link3,
  left_arm_link5, left_arm_gripper_base` (mirrored `right_arm_*`) --
  identical to the old `link3/link5/gripper_base` convention, just renamed.
- **DOF indexing**: unchanged in structure. `joint_addrs()` already resolved
  DOF/qpos addresses by joint *name* via `model.jnt_dofadr`/`jnt_qposadr`,
  which is robust to `nv` growing (23 nq / 22 nv now: 6+2 left arm+fingers,
  6+2 right, 6 box freejoint, vs. the old rig's smaller model) -- no
  index-arithmetic changes were needed, only the joint-name lists. `mj_fullM`
  sub-block selection (`M[np.ix_(left_dofadr, left_dofadr)]`) is unaffected
  by nv size since it operates on the resolved DOF index arrays.
- **Grasp init (NEW step)**: the old rig's "home" keyframe already parked
  the gripper at the box; the canonical scene's "home" keyframe is a generic
  ready pose, not positioned at the box. Added an IK step (damped-least-
  squares, same technique as `scripts/verify_yor_canonical.py`) to compute
  `LEFT_GRASP_Q` (hardcoded, found via `probe_v3.py`) and teleport the left
  arm to it before the existing box-teleport-to-pad-midpoint + close-fingers-
  with-gravity-off trick (unchanged from v1/v2).
- **Intruder pose**: `RIGHT_HOME_CTRL`/`RIGHT_INTRUDE_CTRL` re-derived (also
  via `probe_v3.py`, IK aiming the right grasp site at the left grasp site
  with the left arm frozen at `LEFT_GRASP_Q`) for the new 0.185m geometry.
  Same ramp/oscillate/ramp-out timeline as v2.

## 2. Scene collision-filter bug (found + worked around)

The canonical scene's `<default class="collision">` sets `contype="0"
conaffinity="1"` for every robot link. Two `contype=0` geoms never generate a
contact against each other (MuJoCo's test is
`(contype1&conaffinity2)||(contype2&conaffinity1) != 0`) -- so **the two
arms of the shipped scene cannot physically collide with each other at all**
(they still collide normally with table/floor/box). Patching
contype/conaffinity directly on the existing fitted-capsule collision geoms
fixed the filter, but the full-model collision pipeline then *still* silently
produced zero contacts for a verified-penetrating pair
(`mj_geomDistance` reported -3.7cm overlap for the two `gripper_base`
capsules; an isolated 2-capsule model at the exact same world pose *did*
generate the contact) -- neither `mjDSBL_MIDPHASE` nor `mjDSBL_NATIVECCD`
fixed it; root cause not fully identified.

**Workaround** (in-memory only, via `mujoco.MjSpec`, scene files never
touched): add an explicit SPHERE geom at the origin of each of the 3
canonical collision-proxy bodies per arm, radius `R_EGO`/`R_INT` (literally
realizing the "sphere radius stand-in" the module's own constants already
described). Bit scheme uses two otherwise-unused high bits (`0b0100`/`0b1000`)
exclusively for the left<->right proxy pairing, so it collides with the
opposite arm's proxy only -- not the table, not its own arm's other links (an
earlier bit0-based attempt accidentally caused the sphere to also collide
with the arm's *own* other geoms via the shared conaffinity=1 default,
which visibly knocked the box out of the grasp during the finger-close
settle). Confirmed empirically (`ncon` includes the sphere pair; self- and
world-collision unaffected).

## 3. C0 validity + retuned params

C0 **still collides** (`collision=True`, `min_h=-0.032`, `min_dist=0.068m`)
-- scenario is valid. `box_dropped=False` for C0 (drift 2.4cm).

Retuned (documented rationale in `sim_common.py`), identical across C1/C2/C3:

| param | v2 (old rig) | v3 (real rig) |
|---|---|---|
| R_EGO, R_INT | 0.05, 0.05 | 0.035, 0.035 |
| D_MIN | 0.08 | 0.03 |
| COMBINED_RADIUS | 0.18 | 0.10 |
| H_TRIGGER | 0.05 | 0.04 |
| ALPHA1, ALPHA2 | 15, 15 | **30, 30** (doubled -- see below) |
| FABRIC_REPULSION_MARGIN | 0.10 | 0.08 (kept at 2x H_TRIGGER) |
| RIGHT_HOME/INTRUDE_CTRL | old-rig FK | re-derived via `probe_v3.py` IK |

R_EGO/R_INT/D_MIN/H_TRIGGER scaled down ~2x to match the probed geometry (min
proxy separation ~0.23m at rest vs. ~0.04m at full IK intrusion, roughly half
the old rig's corresponding scale). An alpha sweep {15,25,30,40,50,70,90} at
fixed geometry (see below) showed `min_h` monotonically improving with alpha
but **`box_dropped=True` at every single value tested** -- ALPHA=30 (2x v2,
matching the 2x geometry scale-down) was kept as the most principled,
non-cherry-picked choice; it is *not* the value that best "solves" C2, since
no value in the sweep did.

## 4. Headline result: the v1/v2 verdict does NOT fully hold on real geometry

| metric | C0 | C1 | C2 | C3 |
|---|---|---|---|---|
| safety | none | bang-bang (hard) | HOCBF-QP (hard) | fabric (soft) |
| collision | **True** | False | **True** | False |
| min h [m] | -0.032 | +0.034 | -0.030 | +0.091 |
| box_dropped | **False** | True | True | True |
| box drift max [m] | 0.024 | 0.111 | 0.059 | 0.288 |
| rms EE jerk | 134 | 8280 | 139 | 19 |
| rms torque-rate | 20.5 | 3471 | 18.4 | 5.1 |

- **C1** (bang-bang): avoids the hard mesh collision (as before) but destroys
  the grasp even more violently than v2 (box drift 11cm, 102 deg rotation,
  chattering right at `H_TRIGGER`) -- unchanged qualitative story.
- **C2** (smooth HOCBF-QP, "the hero" in v1/v2): on the real geometry it
  **still collides** (`min_h=-0.030`, essentially matching C0's `-0.032`) AND
  **still drops the box** (drift 5.9cm, just over the 5cm threshold). The QP
  is feasible every step (`qp_infeasible_frac=0.0`) and uses only ~1-8Nm of a
  40Nm budget -- it is not torque-starved. The alpha sweep above shows a
  genuine trade-off, not a bug: pushing alpha high enough to clear the
  mesh-collision threshold (e.g. alpha=40-90) reliably avoids contact but
  drift grows to 10-15cm, *worse*. **The clean 2x2 separation (only C2
  avoids AND keeps the grasp) does not reproduce on the 0.185m rig.**
- **C3** (fabric, soft): on v2 this was expected to still collide (soft, no
  hard guarantee); here it instead **avoids** the hard-collision threshold
  (`min_h=+0.091`, largest margin of all four) but drops the box worst of
  all (drift 28.8cm, catastrophic) -- the opposite failure mode from v2.

## 5. Why: geometric deadlock, not a tuning bug

The intruder is fully open-loop (fixed joint-space target, unresponsive to
the ego's position) -- on the old 0.5m rig the ego had a roomy reachable
volume to retreat into while holding the grasp. On the real 0.185m rig the
ego's retreat volume is much smaller (arms mounted 2.7x closer, at z~0.67 on
a common lift/base), so escaping the intruder's fixed approach point requires
motion large/fast enough to disturb the grasp beyond the 5cm drop threshold
almost regardless of which safety law is used -- the "avoid AND hold" corner
of the 2x2 that v1/v2 demonstrated is much smaller (possibly empty, within
this controller's authority) on the real geometry. This is the honest,
qualitative change from the tighter rig: **collision-avoidance vs.
grasp-retention is now a much sharper trade-off**, not a solved problem for
any of C1/C2/C3 as tuned.

## 6. Figures (`artifacts/ab3_*.png`)

`ab3_distance_h.png`, `ab3_ee_jerk.png`, `ab3_grasp_force.png`,
`ab3_box_drift.png`, `ab3_summary_bars.png`, `ab3_metric_table.png`,
`ab3_render_montage.png`. Old `ab_*`/`ab2_*.png` untouched.

## 7. Blockers / caveats

- The root cause of the full-model capsule-capsule contact silently dropping
  a verified-penetrating pair (section 2) was not fully diagnosed (only
  worked around) -- worth a real MuJoCo issue report if this recurs
  elsewhere in this repo.
- `probe_v3.py` (kept in this directory) is the one-off script used to derive
  `LEFT_GRASP_Q`/`RIGHT_HOME_CTRL`/`RIGHT_INTRUDE_CTRL`; not imported at
  runtime.

## 8. Rollback

Revert `sim_common.py` and `run_ab.py` to their pre-v3 state (old rig,
`left/`/`right/` names, `DT=0.002`, v2 params); delete `probe_v3.py`,
`make_plots_v3.py`, `render_montage_v3.py`, `logs/{C0,C1,C2,C3}.npz` +
`logs/summary.json` (regenerated in place, not versioned), and
`artifacts/ab3_*.png`. Nothing under `scenes/` or `YOR/` was touched.
