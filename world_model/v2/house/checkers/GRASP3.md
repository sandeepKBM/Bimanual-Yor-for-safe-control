# Checker 3 -- exact grasp-load corner check (`grasp3.py`)  [SIM ONLY]

## Model
Object box (a >= b footprint, h), frame x||a, y||b, z up. CoM = centre + delta, delta in a box. Quasi-static load the grasp must supply at contact centre p_c:

    F = m (a_obj + 9.81 ez),     tau = (CoM - p_c) x F          (a_obj in an accel box incl. lift accel)

Grasp wrench set (soft-finger parallel jaw, squeeze N = F_grip_max per jaw, normalised gauge, feasible iff G <= 1):

    G = |f_n|/(2N) + || ( |f_t|/(2 mu N),  tau_n/(2 mu N (2/3) R_eq),  tau_t1/(N e2),  tau_t2/(N e1) ) ||_2 ,    and  |f_n|/N <= 1
    R_eq = sqrt(e1 e2 / pi);  (e1,e2) = pad/object contact patch along (t1,t2);  n = closing axis.

The f_n term is the exact friction-budget loss when one jaw carries a normal imbalance (N1+N2 = 2N-|f_n|).

## Why corners are exact (and when)
w = (F, tau) is multi-affine in (m, delta_x,y,z, a_x,y,z) (each variable appears with degree 1). G is a convex gauge, so G(w(.)) is convex in each variable separately;
a function convex in each coordinate attains its max over a box at a vertex. => the 2 x 8 x 8 = 128-corner check is EXACT for the JOINT box (also with a bounded acceleration
set, where w is bilinear in (m,a) and (m delta, a) -- fixed-a linearity is not needed). Capacity is increasing in mu => mu = mu_lo is worst. Hence enlarging mass_range / com_box /
accel_box or lowering mu_lo can only remove feasible candidates; margins are non-increasing (property test `test_monotonicity_property`; vertex-max test `test_corners_are_exact_max_over_box`).
Not exact: the gauge itself is a MODEL (ellipsoidal limit surface), and bimanual load split is assumed equal.

## Candidates (all produce contact geometry; `access` = precondition NOT checked)
| name | closing axis n | contact centre p_c (object frame) | width needed | arms |
|---|---|---|---|---|
| topdown_pinch | y (across b) | (0,0,h/2-e2/2), patch (min(.03,a), min(.03,h)) | b <= 0.060 | 1 |
| edge_pinch (h<=b) | z (across h) | (0, b/2-ins/2, 0), ins=2.5 cm | h <= 0.060 | 1 (lower jaw must reach under the edge) |
| rim_pinch (rim categories) | y (radial) | (0, b/2-t/2, h/2-bite/2), bite 2 cm, patch 1 x 2 cm | wall 1 cm | 1 |
| handle_loop_pinch (has_handle) | y (radial) | (0, b/2+3 cm, 0), patch 1.5 x 3 cm | bar 1.5 cm | 1 |
| handle_shank_pinch (Pan/Pot/Ladle/utensils) | z | (a/2-4 cm,0,0) | 1.5 cm | 1 |
| bimanual_squeeze_a / _b | footprint axis, span in [0.25,0.45] | (0,0,0) | span | 2 (equal split; squeeze 2N per hand) |

Payload (arm, not grip): 1 arm m_hi <= SINGLE_MAX_KG (0.75, assign's ASSUMED); bimanual m_hi (0.5 + com_s/span) <= 0.75 (equal split + CoM shift along the squeeze axis). margin = min(1-G, payload margin, width margin).

## Reused constants (pick/assign.py)
GRIPPER_OPEN_M 0.07 (YOR arm.py:12), JAW_MARGIN_M 0.005, SINGLE_MAX_KG 0.75 (ASSUMED), MIN_HAND_SEP_M 0.25, MAX_SPAN_M 0.45, FINGER_FORCE_N 20 (finger forcerange, yor_robot_derived.mjcf:86) = F_grip_max per jaw.
Pad face 3x3 cm from the sim finger geom (yor_robot_derived.mjcf:346).

## NEW ASSUMPTIONS (named in code; none is a verified robot spec)
MU_LO 0.4 / MU_HI 0.8 (sim pads are tuned to mu=2, not real); COM_FRAC 0.05 (CoM half-box = 5% of extent; 10% sensitivity below); ACCEL box x,y +-1, z [-1,+1.5] m/s^2;
wall 0.75 cm (jaw check 1 cm); rim bite 2 cm, rim contact length 1 cm; edge inset 2.5 cm; handle: bar 1.5 cm, offset 3 cm, length 6 cm, inner gap 2 cm, shank grip 4 cm from the end (bbox-centre CoM);
bimanual paddle = 2 pads (6x3 cm), squeeze = 2 x F_grip_max per hand (arm assumed able); torsion = (2/3) mu N R_eq; category rim list (Cup Mug Bowl Pot Pan GarbageCan Vase Plate Kettle).
Categories/handles come from `vlm/ftdata/category_priors.json` (has_handle). Mass ranges are the real-world priors (not sim masses).

## API
`check_grasp(extent, mass_range, com_box=None, mu_range=(.4,.8), category=None, has_handle=None, accel_box=..., has_rim=None) -> GraspReport(candidates ranked, feasible, best, worst_margin)`;
`decide(...) -> {mode: single|bimanual|infeasible, arms, arm, grasp, margin, reason, fallback_arms, report}`. infeasible = no candidate passes at ALL corners -> Checker 4's probe.
`fallback_arms` = arms of the least-bad candidate (only for comparing to VLM answers).

## Offline evaluation (noisy extents + real-world mass-range priors; no simulation)
Held-out `teacher_eval/eval_set.jsonl`: 80 records = 64 'arms' (rule feasible) + 16 'name' (rule-INFEASIBLE instances, dropped from arms). Default assumptions:
- rule 2-arm (bimanual) -> Checker 3: 23 stay bimanual, **6 become single** (4 Newspaper + 2 Spatula, edge_pinch), 0 infeasible. Rule single -> single 39/39.
- rule-infeasible (16): 7 still infeasible, **5 become feasible** (3 RemoteControl edge_pinch single; 1 Box, 1 Pillow bimanual squeeze).
- Agreement with VLM arms on the 64 (old rule: Codex 39/64 = 61%, Nano 40/64 = 62.5%): **Checker 3 Codex 41/64 = 64%, Nano 46/64 = 72%** (n=64, small: examples, not rates).
  Books stay bimanual (mass 0.65-1.3 kg: edge-pinch pitch torque >> pad couple), so the VLMs' "pinch the edge" for books is NOT confirmed by the load model.
- Sensitivity: CoM 10% -> 16 of the 29 2-arm rule cases become infeasible (the 1.3 kg book is payload/friction marginal: margin 0.05 at 5%); Codex/Nano agreement on decided-only cases 84%/86% (49 decided).
  mu_lo 0.6 identical to default; mu_lo 0.3 -> 1 more infeasible.
Full dataset (`sim_pick_v1/full/{train,test_cat}.jsonl`, 6336 unique samples; per-sample, not per-asset; ~5 assets/category):
- rule-infeasible 2543 -> **781 (31%) become feasible: 507 single, 274 bimanual**; 1762 stay infeasible. Rule-single 2681 all stay single. Rule-bimanual 1112 -> 205 single, 14 infeasible.
- Single-arm grasps used: edge_pinch 373, handle_loop 83, rim 51; bimanual squeeze 274.
- Mugs: 59/59 feasible via handle_loop (m_hi 0.5 kg; max feasible m_hi 0.54 kg); at CoM 10% only 10/59. Cups: 51/96 via rim (max m_hi 0.40 kg; 1/96 at CoM 10%). SprayBottle 24/24 handle.
- **Bowls: 0/128 feasible** (median 19.8 cm dia, m_hi 0.8 kg): rim pitch torque with 20 N jaws; max feasible m_hi = 0.28 kg. Bowl is too wide (<25 cm) to squeeze -> infeasible -> probe. Plates: 46/119 bimanual, 73 infeasible. Books 59/218 edge_pinch single (light ones), Pot/Pan/Kettle/Laptop/Vase/Statue stay infeasible.
Caveats: cliffs (Cup/Mug) depend on COM_FRAC, pad size, rim bite, F_grip_max (all ASSUMPTIONS); edge_pinch assumes the lower jaw can get under the edge; "edge" is applied to any object with h<=b (e.g. potato = pinch across the thinnest side); bimanual internal squeeze is not checked against arm force limits; no sim grasp verification.
