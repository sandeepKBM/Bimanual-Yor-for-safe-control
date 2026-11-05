# State & Observation Spaces

> Status (2026-11-05): SUPERSEDED: Sept 2026 draft for the HJ / learned-world-model layers (dropped 2026-10-15). HISTORY only.

**Status:** design draft · 2026-09-06 · defines the state each layer operates on, why, and its use.

The system has **three safety/planning layers, each on its own state representation** — plus a set of **privileged sim labels** that are used for supervision/eval but are NOT runtime inputs. Keeping these separate is what prevents a world model that "cheats" in sim and fails on hardware.

---

## S1 — Reachability configuration state (geometric collision safety)

- **State:** `x_HJ = [q_L, q_R] ∈ ℝ¹²` — the two arms' 6-DoF joint angles (radians). No velocities.
- **Model:** kinematic single-integrator differential game — ego `ẋ = u_L`, intruder `ẋ = u_R` (worst-case disturbance), `‖u‖ ≤ ū`.
- **Failure set:** `l(x) = min_pair ‖p_i(FK(q)) − p_j(FK(q))‖ − d_min` (arm–arm sphere distance, real Piper FK via MuJoCo).
- **Learned object:** `V(t, x)` = time-to-unavoidable arm–arm collision (DeepReach).
- **Use:** the safety **certificate** — terminal safe set / CBF value for the planner and the smooth filter.
- **Why joints + kinematic:** articulated collision constraints live in configuration space; low-dim and trainable; matches NeHMO/HaMMAR. Position-level (first-order).
- **Excludes:** velocities, the held object, any semantic/visual state.

## S2 — Control dynamic state (smooth counter-planning)

- **State:** `x_ctrl = [q_L, q̇_L, q_R, q̇_R] ∈ ℝ²⁴`, **plus grasped-payload params** (mass/inertia) folded into the ego arm's effective `M(q)` and bias.
- **Why second-order:** the collision barrier `h(q)` is **relative degree 2** in torque (control appears only in `ḧ`) → needs `(q, q̇)` and HOCBF.
- **Use:** the **HOCBF-QP filter** computes the smooth safe torque `τ* = argmin ‖τ − τ_task‖² s.t. ψ₂(x,τ) ≥ 0`, at kHz.
- **Compose with S1:** `V(q)` (S1) is the anticipatory terminal/CBF value on position; S2 executes it smoothly at second order. The fast inner QP uses the analytic distance barrier; `V` provides the long-horizon guarantee.

## S3 — Semantic latent state (task-failure monitor)

- **Observation:** `o_t = { image: RGB 64×64×6 (two cams concatenated), proprio: ℝ²⁶, action: ℝ¹⁶ }`.
- **State:** `z_t = RSSM(o_{≤t}, a_{<t})` — the DreamerV3 latent.
- **Failure head:** `P(spill) / P(task-failure)` from `z_t` (+ imagined rollout).
- **Use:** the **semantic safety layer** (~10–15 Hz) — catches failures that can't be written as a constraint on `q`.
- **Why latent/visual:** spill depends on ball dynamics / tilt-rate, observable only from vision (see the accel-induced-spill-while-upright result).

---

## Observations vs. privileged labels — the sim2real line (CRITICAL)

| | Available on the **real robot** at test? | Used for |
|---|---|---|
| RGB (cameras) | ✅ | WM observation |
| Arm `q`, `q̇` | ✅ (encoders) | S1/S2 state; WM proprio |
| Gripper state | ✅ | WM proprio |
| Action | ✅ | WM input |
| **Object 6-DoF pose** | ❌ (no external tracker) | **LABEL / eval only** |
| **`spilled_count`, `collision`, `dropped`** | ❌ (sim ground truth) | **supervision + eval only** |

**Rule:** the world model's `proprio` observation **must exclude object pose and the ground-truth flags.** If the WM sees object pose, it will predict spill from the privileged pose instead of from vision and will not transfer. Object pose / spill / collision are **training targets and eval signals**, never WM inputs.

- **Revised `proprio` (26-D):** `[q_L(6), q_R(6), q̇_L(6), q̇_R(6), grip_L(1), grip_R(1)]`. *(Down from the collector's current 33-D, which included the 7-D object pose — that must move from `proprio` to a separate label field. → action item for `collection/dataset_adapter.py`.)*

### Real YOR sensor suite (verified from the model + `zed_pub_node.py`) — the WM may ONLY use these
- **Cameras (physical):** `left_arm_egocentric` / `right_arm_egocentric` (wrist, fixed, 53° fovy); two wrist **iPhones**; a **ZED** stereo (RGB + **depth** + point cloud + pose, via `zed_pub_node`). → **WM image channels = the egocentric wrist cams** (+ possibly ZED RGB-D once its mount is verified to face the workspace).
- **NOT real / do not feed the WM:** our invented `head_cam` (drop it); the MJCF `front_camera`/`side_camera` (they are `mode="track"` *visualization* cams, not sensors).
- **Proprioception (physical):** joint encoders `q, q̇`, gripper state; base odometry (`base_site` framepos/quat/lin/ang-vel — the only `<sensor>` block, for the mobile base).
- **No force/torque, no tactile.** → **grasp force/slip is NOT directly sensed on hardware**; the grasp/spill monitor must run on vision + joint effort/current only. Ground-truth contact force exists in sim as a *label/eval signal*, never a runtime input. This is a documented sim2real gap for grasp sensing.

---

## Division of labor — which layer owns which failure

- **Arm–arm collision** → S1/S2 geometric (exact, hard).
- **Grasp retention** (object slip/drop) → S2 grasp dynamics + measured metrics.
- **Spill / topple / semantic** → S3 world model.
- **Held object vs. the *other* arm** → **OPEN** (see decisions): the barrier currently spheres the *arms only*; a held cup can hit the other arm.

## Frames & units

Joints in radians; positions in world-frame meters; gripper in meters (`0` = closed); action convention per the gripper-sign decision (`gripper_state = qpos(joint7)/0.035 ∈ [0,1]`).

---

## Open decisions to lock

1. **Object pose OUT of WM proprio** (26-D obs), kept as a label. — *recommended; needs adapter edit.*
2. **Add the held object to the collision barrier?** (extend S1/S2 sphere set with held-object spheres so the cup-vs-other-arm collision is covered). — *decision.*
3. **Keep HJ kinematic (S1) vs. train a second-order value?** — *recommend kinematic + S2 executes; second-order value is heavier and less necessary given the two-layer split.*
4. **HJ dataset sampling:** uniform joint sweep gives only ~4.9% unsafe and under-covers the task's shared region — **oversample near the shared workspace / boundary** so `V` is well-resolved where the task operates. — *decision.*
