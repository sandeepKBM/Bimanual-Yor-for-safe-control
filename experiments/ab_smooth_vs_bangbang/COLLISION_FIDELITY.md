# Collision-model fidelity: diagnosis + fix (scene_yor_canonical.xml)

Investigates and fixes the arm-arm collision-model fidelity gap flagged in
`RESULTS_v3.md` §2/§7: a verified-penetrating cross-arm contact that MuJoCo
"silently dropped", worked around there with an ad-hoc 3-sphere-per-arm proxy
(`link3/link5/gripper_base`) that is ALSO the HJ/HOCBF safety barrier. This
doc supersedes that workaround.

## 1. Root cause of the contact drop (concrete, verified)

**Single, confirmed cause**: `yor_robot_derived.mjcf`'s
`<default class="collision"><geom ... contype="0" conaffinity="1" .../>`
applies to every robot link's REAL collision geom (fitted capsules -- MuJoCo
auto-fits a capsule primitive to each link mesh via `type="capsule"
mesh="linkN"` -- plus box finger pads). Two `contype=0` geoms never generate
a contact regardless of overlap (`(contype1&conaffinity2)||(contype2&
conaffinity1) == 0` when both contypes are 0). So **as shipped, the two arms
are structurally incapable of colliding with each other** (table/floor/box
contacts are unaffected -- those geoms use the ordinary
`contype=conaffinity=1` default). No `<contact><exclude>` targets arm-arm
pairs, and `nconmax`/`njmax` are both `-1` (dynamic, MuJoCo 3.6 auto-resizes)
-- buffer limits are not a factor.

RESULTS_v3.md additionally reported a *second*, unexplained drop even after
patching contype/conaffinity on the 3 barrier bodies (gripper_base-vs-
gripper_base allegedly still not registering). Re-investigated here
(`diag_contact_drop.py`): patching contype/conaffinity on **all** ~13
per-arm "collision"-class (group==3) geoms (not just the 3 barrier bodies)
and re-running `mj_forward` at the experiment's own
`LEFT_GRASP_Q`/`RIGHT_INTRUDE_CTRL` pose reproduces **zero** missing
contacts: all 10 true-penetrating cross-arm geom pairs (via
`mj_geomDistance`), including gripper_base-vs-gripper_base, appear correctly
in `data.contact`. No separate broadphase bug is reproducible once the
filter fix is applied to the real geoms -- the likely explanation is that
only 3 of ~13 collision geoms per arm were patched in the original
debugging pass, so most cross-arm pairs were still `contype=0`.

## 2. Proxy fidelity: BEFORE (old 3-sphere proxy)

Ground-truth oracle: `mj_geomDistance` over all cross-arm "collision"-class
(group==3) geom pairs -- confirmed above to match MuJoCo's own contact
generation exactly (0/10 missed).

Sampled 2000 configs (1000 uniform over full joint ranges + 1000
"task-region": left arm at `LEFT_GRASP_Q` + noise, right arm interpolated
along the `RIGHT_HOME_CTRL`->`RIGHT_INTRUDE_CTRL` schedule + noise -- the IK
solution for both grippers reaching the shared workspace point (0.42,-0.21)).

- True collisions (real geoms overlap): 358/2000 (17.9%).
- **FALSE-SAFE** (old proxy h>=0 but true penetration): **71/358 = 19.8%**
  of true collisions (3.55% of all configs). Worst case: 5.5cm undetected
  penetration.
- Missed link pairs (top): `iphone_body` (wrist-cam mount, not covered by
  ANY of the 3 old bodies) x many, `link0`, `link1`, `link2`, `link4`,
  `link6`, `link7`/`link8` (fingers) vs the opposite arm's links -- i.e.
  exactly the links the 3-sphere set never modeled, plus the wrist-cam
  mount, which turned out to be one of the most frequent real collision
  points.

(`collision_fidelity_assess.py`)

## 3. The fix

Per arm, replaced 3 fixed-radius body-origin spheres with a **multi-keypoint
proxy read directly from the real per-link geometry**:

- Bodies: `link0, link1, ..., link6, gripper_base, link7, link8,
  iphone_body` (11 bodies/arm, i.e. every link + the wrist-cam mount).
- Per body: 2 endpoint spheres at the fitted capsule's own ends (radius =
  that capsule's real radius, computed from `geom_pos/geom_quat/geom_size`
  of the model's OWN "collision"-class geom -- not a guessed constant), +1
  extra midpoint sphere for capsules with half-length > 5cm (link2, link3,
  gripper_base, iphone_body -- avoids a shaft-center gap for long links). 1
  bounding sphere per box geom (finger pads).
- Result: ~25 spheres/arm (625 cross-arm pairs). `h` per pair = `dist -
  (r_ego + r_int + D_MIN)`, `D_MIN=0.03` unchanged; per-pair radius now
  varies by link instead of the old fixed `R_EGO=R_INT=0.035`.
- Real-geometry contype/conaffinity fix: patch ALL left/right "collision"
  geoms (not just barrier bodies) with the same two-high-bit scheme as v3
  (`0b0100`/`0b1000`, sharing no bit with the scene's own bit0-based
  defaults) -- no synthetic sphere geoms needed any more; MuJoCo's own
  contact pipeline now reliably detects true cross-arm contacts directly.

Implemented in `sim_common.py`: `LINK_NAMES`, `_body_keypoints`,
`barrier_keypoints`, `_patch_real_collision_geoms`, `point_jac_at`/
`point_jacdot_at`, and rewritten `min_pair_distance`/`all_pair_barriers`/
`EgoController.barrier`. `all_pair_barriers` computes each keypoint's
Jacobian ONCE (O(n_ego+n_int) `mj_jac` calls) and reuses it across all 625
pairs, not once per pair. `EGO_BARRIER_BODIES`/`INT_BARRIER_BODIES`/
`R_EGO`/`R_INT`/`COMBINED_RADIUS` kept as legacy constants (documented,
unused by the corrected path) so nothing importing them breaks.

**AFTER**: re-ran the same 2000-config assessment (`collision_fidelity_fix_test.py`)
with 2 independent seeds: **0/699 false-safes across both** (0.00%), down
from 19.8%/3.55%. (Two intermediate variants were tried and rejected:
link1-6+gripper alone = 11.2% still missed mostly on `link0`/`iphone_body`;
adding link7/8/iphone but not link0 = 5.7%, missed mostly on `link0`.)

## 4. Re-verification

`diag_contact_drop.py` + `render_collision_fidelity_check.py`: driving the
arms to `LEFT_GRASP_Q`/`RIGHT_INTRUDE_CTRL` and to the worst-h step of a live
C0 rollout both confirm the fixed model's proxy `h` and MuJoCo's own contact
list agree with ground truth (`mj_geomDistance`); rendered frame at the
worst-h C0 step (t=1.09s, proxy h=-0.0458, `mj_geomDistance`=-0.40cm,
genuinely penetrating) saved to `artifacts/collision_fidelity_check.png`.

Quick C0+C2 re-check (`quick_recheck_c0_c2.py`, corrected proxy, full DT/duration,
not the full C0-C3 sweep):

| metric | C0 (before: -0.032) | C2 (before: -0.030) |
|---|---|---|
| collision | **True** (min_h=-0.0458, min_dist=0.030) | **True** (min_h=-0.0309, min_dist=0.041) |
| box_dropped | False (drift 2.5cm) | **True** (drift 23.1cm, worse than v3's 5.9cm) |
| qp_infeasible_frac | -- | 0.0 (QP still feasible every step) |

**The "naive methods fail" conclusion SURVIVES the fidelity fix**: C0 still
collides at essentially the same margin as before; C2 (the smooth HOCBF-QP
"hero" of v1/v2) still collides (min_h close to C0's) AND drops the box even
more badly than under the coarse proxy -- consistent with RESULTS_v3.md §5's
"geometric deadlock" explanation, now confirmed against faithful (not
false-safe) collision geometry. No naive/soft controller in this experiment
achieves both collision-avoidance AND grasp-retention on the real geometry.

## 5. What `hj_reachability/mujoco_fk.py` must change to match (REPORT ONLY -- not edited)

`mujoco_fk.py` is a hand-copied, standalone duplicate of the OLD proxy and is
now doubly stale:

1. It already used **different** constants than `sim_common.py` even before
   this fix (`R_EGO=R_INT=0.05, D_MIN=0.08, COMBINED_RADIUS=0.18` vs.
   `sim_common.py`'s retuned `0.035/0.035/0.03` / `COMBINED_RADIUS=0.10`) --
   a pre-existing inconsistency independent of this investigation.
2. It must be updated to the corrected multi-keypoint proxy: replace
   `EGO_BARRIER_BODIES`/`INT_BARRIER_BODIES` (3 bodies) with `LINK_NAMES`
   (11 bodies), replace `keypoints()`'s 3 raw `xpos` points per arm with the
   ~25 per-arm keypoints (`body_id, local_offset, radius`) from
   `sim_common.barrier_keypoints`/`_body_keypoints`, and replace
   `min_pair_distance`/`boundary_fn`'s single global `COMBINED_RADIUS` with
   the per-pair `r_ego + r_int + D_MIN`.
3. `closest_pair_info`'s analytic gradient (`_point_jac`, currently querying
   `mj_jac` at `data.xpos[body_id]`, i.e. body origin) must query at the
   keypoint's WORLD offset point instead (`sim_common.point_jac_at`), since
   most corrected keypoints are not at body origins.
4. **Consequence**: any dataset already exported from this oracle (the "200k
   dataset" for DeepReach/HJ training) was generated against the old,
   false-safe-prone 3-point boundary function and is inconsistent with the
   corrected HOCBF barrier now in `sim_common.py` -- it should be treated as
   stale and regenerated after `mujoco_fk.py` is updated to match.

## 6. Blockers / caveats

- `all_pair_barriers` now costs ~50 `mj_jac`/`mj_jacDot` calls/step (up from
  6) -- ~17ms/step measured, fine for the quick C0/C2 recheck here but would
  meaningfully slow a full C0-C3 sweep at scale; not otherwise a blocker.
- `min_contact_force` stayed 0.0 in both C0/C2 quick reruns despite confirmed
  real penetration -- MuJoCo's contact SOLVER still only weakly resists these
  capsule-capsule overlaps (soft `solref`/`priority=1` per the `collision`
  default class); not investigated further here, out of scope (this doc is
  about barrier fidelity, not contact-response tuning).

## 7. Rollback

Delete `diag_contact_drop.py`, `collision_fidelity_assess.py`,
`collision_fidelity_fix_test.py`, `quick_recheck_c0_c2.py`,
`render_collision_fidelity_check.py`, `COLLISION_FIDELITY.md`, and
`artifacts/collision_fidelity_check.png`; revert `sim_common.py`'s
`LINK_NAMES`/`_body_keypoints`/`barrier_keypoints`/
`_patch_real_collision_geoms`/`point_jac_at`/`point_jacdot_at`/
`min_pair_distance`/`all_pair_barriers`/`EgoController.barrier` to the v3
(`_add_collision_proxy_spheres`, 3-body `EGO_BARRIER_BODIES`) versions.
Nothing under `scenes/`, `YOR/`, or `hj_reachability/` was touched.
