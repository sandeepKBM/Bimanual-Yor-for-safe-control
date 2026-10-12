# FL-MPC: options for a cleaner constraint abstraction

Status: design proposal, 2026-10-05. Read-only analysis. No code was changed.
Tags: **[V]** = verified (read in code or run, with file:line). **[I]** = inferred.
Short paths: `mpc_fl` = experiments/ab_smooth_vs_bangbang/mpc_fl.py; `sc` = experiments/ab_smooth_vs_bangbang/sim_common.py;
`rt_flmpc`, `rt_kernels`, `rt_qp`, `rt_pipe` = experiments/realtime/*.py; `demo` = scripts/make_cup_collect_safety_demo.py;
`fv` = world_model/v2/latent_hj/flmpc_value.py.

## 0. Summary

- `FLMPC.step()` is one 470-line method (mpc_fl:540-1009). It builds every constraint family by hand and has no generic row input. [V]
- The same rows exist three times: Python (`mpc_fl.step`), numba (`rt_kernels.assemble_rows`), and a partial copy (`mpc_fl_decentralized.step`). [V]
- Because there is no hook, 15 files patch FL-MPC internals at run time. These patch points are now a hidden API. A refactor that moves them breaks results silently. [V]
- Recommendation: build the replay harness first, then add a minimal hook (option A) that keeps every patch point. Put task-space rows in a new file on top of it (E). Port the hook to the numba path when the first new constraint must run in real time. Do the registry (C) later, and only if needed. Do not do the shared kernel layer (D) now.

## 1. Current architecture

Terms. **FL** (feedback linearization): per arm, apply τ = M(q)·a + c(q, q̇), so q̈ = a exactly. The MPC then plans the
acceleration a. **QP**: quadratic program. **Decision vector** x: the stacked planned accelerations. **Block**: one piece of a piecewise-constant
acceleration, held for L steps. **Slack**: an extra variable that lets a row be violated at a cost.

```
 caller (demo / scenario.py / teleop)
   | pairs_all (sc.all_pair_barriers), table_rows (sc.table_plane_barrier_rows, torque convention),
   | refs q/qd (NP+1 x NV) per arm, payload M_ego_extra/bias_ego_extra, per-call options
   v
 FLMPC.step (mpc_fl:540)                                               Python unless marked
   1 q0, qd0 per arm; M, c = qfrc_bias - qfrc_passive from MuJoCo (565-583)
   2 [F5] forward-predict over the command delay (585-591)
   3 drift q_nom = q0 + qd0*t (593-598); top-K closest pairs (600-612)
   4 [adaptive_yield, default off] margin/weight tweaks (614-689)
   5 _assemble_fl -> H, f, rows [collision | torque box]               numba (79-198)
   6 [F6] per-joint torque RHS rewrite (705-718)
   7 rows += tilt/cone (720-834), leash (836-877), table (879-897), speed cap (899-927)
   8 sc.solve_constrained_box_qp(H, f, box, A, b)   <- patch point S1    numba (external)
   9 infeasible -> _relaxed_fallback (440-513) -> else a = 0; then _torque_safe_scale (976-995)
  10 tau = M a0 + c per arm (1007-1009) -> returns (tau_e, tau_i, feasible)
   v
 caller: friction feed-forward, filter, clip to +-sc.TAU_MAX, mj_step (demo:2969-2974)

 rt lane:  RTFLMPC.step (rt_flmpc:283) = Python copy of 1-4 -> rt_core (rt_kernels:391): same rows in numba,
           Goldfarb-Idnani QP, relaxed QP, a = 0, backstop, tau.
           RTFLMPC.step_lean (490) -> _prepare (522, Python) -> rt_full (704, numba, 72 args): robot-only M/c,
           pairs, table rows, payload, then rt_core.  PipelinedRTFLMPC (rt_pipe:81) runs rt_full on a thread.
```

**Decision vector [V]** (mpc_fl:104-157, 556-561). x = [ego arm | other arm]. Each half is [block 0 | ... | block NC-1], NV joints per block.
The "ego" arm is the left (cup) arm; the "int" arm is the right arm. The demo uses NC=4, L=8, NV=6 (demo:2136): 48 variables,
horizon NP=32 steps = 160 ms at DT = 5 ms (checked by import). Simple bounds |x| <= a_box = 800 rad/s² (mpc_fl:934-935).

**Prediction [V]** (mpc_fl:58-76, 377-380). q(t) = q0 + q̇0·t + Φq(t)·x_arm and q̇(t) = q̇0 + Φqd(t)·x_arm, with Φq(t) = [cq[t,b]·I] per block
(semi-implicit Euler, like MuJoCo). The tensors are constant and built once. The prediction is exact when M and c are exact.
Every geometric row freezes its Jacobians and normals at t = 0, so the rows are first-order in geometry.

**Cost [V]** (`_assemble_fl`, 108-157): position and velocity tracking for both arms over t = 1..NP, effort weight R_ACC, and
jerk weight W_JERK against the previous command. H does not depend on the state for a given ego weight. The numba path caches its factor (rt_flmpc:192-213).

**Row families, in stacking order [V].** Order matters. The Python solver sweeps the rows one by one, so a new row order changes the bits.

| # | Family | Python builder (mpc_fl) | Columns | Class | Enabled by |
|---|---|---|---|---|---|
| 1 | Arm-arm clearance, K_PAIRS x horizon steps (64 rows in the demo) | `_assemble_fl` 168-181 | both arms, all blocks | relaxable | always |
| 2 | Torque box, 4·NV rows | `_assemble_fl` 183-196; F6 RHS 705-718 | own arm, block 0 | hard | always |
| 3 | Cup tilt box / effective-gravity cone / true cone | step 720-834 | ego only | hard | left_ee_body + tilt_max_rad; cup_cone_facets; cone_mode |
| 4 | Leash (EE displacement box) | step 836-877 | ego only | hard | leash_ee_body + leash_max_m |
| 5 | Table floor (caller rows on torque, mapped to accel) | step 879-897 | own arm, block 0 | hard | table_rows |
| 6 | EE speed cap | step 899-927 | int only | relaxable | ee_speed_cap or v_ee_max + body |

"Hard" rows are never relaxed. "Relaxable" rows are hard in the first solve and get a slack only in the fallback. There is no
always-soft class and no equality class. [V]

**Solver [V].** `sc.solve_constrained_box_qp` comes from real_Cartpole/controller_core/constrained_box_qp.py:62-180. It is dual
coordinate ascent: for each row in turn it bisects a multiplier, and each probe solves a box QP. It allows inequality rows only.
Its `feasible` flag is a per-row heuristic, not a certificate. `fast_qp.install` swaps in a bit-identical faster copy (fast_qp:262-267).

**Fallback [V]** (mpc_fl:939-995). If the hard QP fails, `_relaxed_fallback` adds one slack per relaxable row (cost RHO·s + EPS·s²,
0 <= s <= S_MAX) and keeps all other rows hard. If that also fails, a = 0. In both cases `_torque_safe_scale` then scales a0 toward 0
until the torque box holds exactly (1041-1063). The relaxable rows are found **by index**: the first m_coll rows plus the speed rows (927, 951-956).

**Python vs compiled [V].** Only `_assemble_fl` (cost plus families 1-2) is numba. Families 3-6, adaptive yield, delay prediction,
the fallback and the backstop are Python. The MuJoCo calls (`full_mass_matrix`, `body_jac_full`) are Python (sc:598-627).

**Torque limit [V].** Scalar `sc.TAU_MAX = 100` (sc:130), read at call time (mpc_fl:703, 705). Optional per-joint `tau_max_vec` (F6, 405-411)
widens a joint's box to |c_j| when gravity alone exceeds the cap (709-712).

### Siblings and ports [V]

- **RTFLMPC** (rt_flmpc:119) subclasses FLMPC but overrides `step`. Its Python gather is "a line-for-line copy" (rt_flmpc:290).
  `rt_kernels.assemble_rows` (40-311) rebuilds every family in numba in the same order. The QP is Goldfarb-Idnani (**GI**, an exact
  dual active-set method, rt_qp.py), with an iteration cap that counts as infeasible. It reuses from mpc_fl only the config, the Φ
  tensors, `_predict_over_delay`, and `_assemble_fl` to build H once.
  - G1 gate (bimanual_data/rt_flmpc/g1/g1_flagship_s0.json): rows match the Python path to 2.2e-16 (A) and 2.8e-14 (b), but are not
    bit-identical (A bit-identical on 0 of 5736 steps). Solutions differ by up to 9.4 rad/s² (p99 6.1) because the solvers differ.
  - The rt path is therefore validated statistically. G3, N=50: 0 collisions and 0 spills on py, rt and rt_pipe (rt_flmpc/PROGRESS.md, 16:58).
  - G4 timing: rt in-loop mean 0.26 ms, max 0.47 ms; py mean 8.1 ms, p99 29.8 ms.
- **FLMPCDecentralized** (mpc_fl_decentralized:99) overrides `step` with its own copy of the tilt-box, leash and table code. It has
  already drifted: no F1/F5 kwargs, and it raises on the cone (121-125).
- **ValueFLMPC / CBFFLMPC** (fv:121; cbf_filter.py:72) add learned-value rows. With no hook, `ValueFLMPC.step` swaps the module global
  `mpc_fl.sc` for a proxy (fv:137-145). The proxy's solver call appends soft rows (ego block only, slack cost rho_v, s <= s_max_v)
  and narrows the ego bounds to a trust region (fv:70-118). It also acts inside the fallback.
  - An unapplied hook exists: world_model/v2/latent_hj/flmpc_extra_rows.diff. It is stale (it predates F1/F5/F6) and does not
    reproduce the proxy: it has no trust region, it uses S_MAX=5 instead of s_max_v=1e3, and it orders the slack columns differently in the fallback.
- **FastFLMPC** (fast_barriers:410) only pre-sorts pairs. It is installed by replacing `mpc_fl.FLMPC` (446).
- **MJPCController** (experiments/baselines/mjpc/mjpc_planner.py:108) is a duck-typed stand-in. Its `step` takes `**kw`.

## 2. Call sites

grep (outside YOR/ and _snap/) finds 43 files that mention mpc_fl, FLMPC or rt_flmpc [V]: 23 import them directly, 4 reach FL-MPC
through `fv` or the demo, and the rest only mention it in comments. Most experiment scripts reach FL-MPC through `demo.run_episode`.
run_ab4.py (v4c) has no reference. v4c shares only the demo module and the solver.

| File(s) | How it gets FL-MPC | Constructor kwargs | Calls | Reads / writes / patches | Critical |
|---|---|---|---|---|---|
| demo | FLMPC, FLMPCDecentralized, RTFLMPC, PipelinedRTFLMPC (2159-2163, 585-594) | NC=4, L=8, W_POS_EGO=800; optional adaptive_yield set, W_JERK, W_POS_INT, ee_speed_cap, ee_speed_body_int, delay_steps, tau_max_vec, cone_mode (2136-2157) | `step(pairs, M_ego_extra, bias_ego_extra, table_rows, left_ee_body, tilt_max_rad, leash_ee_body, leash_max_m, cup_cone_facets, ego_hold)` (2737, 2803); `step_lean` (2724, 2793); `configure_lean` (2205) | NP, a_ego_prev, a_int_prev; **writes** `_cup_axis_local` (2715-2791); replace_last_applied / push_applied (2995-2997); n_relaxed, n_zero_fallback, n_taumax_relaxed, rt_log_fields (3092-3097) | **YES** (flagship) |
| experiments/ab_smooth_vs_bangbang/comprehensive_eval.py | demo.run_episode | via demo globals; --ctrlfix-tau-max, --controller-impl | — | episode metrics | **YES** (N=50) |
| experiments/realtime: rt_gate_g1, rt_gate_g2, rt_gate_g4, rt_unit_branches, gate_equivalence, rt_capture_qp, rt_g3_driver | FLMPC, FastFLMPC, RTFLMPC | demo kwargs | step, step_lean | last_M/c_e/i, a_*_prev, _a_queue, _cup_axis_local, last_fallback, records; patch `sc.solve_constrained_box_qp` and `FLMPC.step` to record QPs | **YES** (rt gates) |
| rt_flmpc, rt_pipe, rt_jit_time | subclass / build | all FLMPC kwargs + it_hard, it_relax, budget_ms | super().__init__, `_assemble_fl` (for H), `_predict_over_delay` | Phi, coll_steps, tilt_steps, W, tau_max_vec | **YES** |
| fv, cbf_filter.py, scenario.py, validate_rows.py | ValueFLMPC, CBFFLMPC | NC=4, L=8, W_POS_EGO=3000, W_VEL_EGO=15, rho_v, trust, a_cap (scenario.py:70, 229-231) | `step(..., value_rows=(A, b, x_ref))` (scenario.py:367-369) | Phi, NP, n_blk, last_M_e, last_c_e, last_plan_ego, last_rows; swap `mpc_fl.sc` | **YES** (latentcbf/final results) |
| mpc_fl_decentralized.py | subclass | same as demo | own step | `_assemble_fl`, `_relaxed_fallback`, `_torque_safe_scale`, Phi, tilt_steps | medium |
| fast_barriers, fast_qp, profile_flmpc_step, bench_qp, bench_components | FastFLMPC / timing | — | — | replace `mpc_fl.FLMPC`, `sc.solve_constrained_box_qp`; wrap `FLMPC.step` | medium |
| experiments/actuation/run_actuation_eval.py | demo | — | — | patches `sc.TAU_MAX`, `mpc_fl._assemble_fl` (reads positional args 21, 23, 25), `FLMPC._torque_safe_scale` (117-151) | medium (Piper limits) |
| experiments/imitation/il_common.py (+ eval_policy) | demo | — | — | wraps `FLMPC.step` and the solver; assumes the first solver call is the main QP (348-370) | low-medium |
| experiments/grasp_audit/a_flmpc_ee_audit.py | demo | — | — | patches `FLMPC.step`, `_relaxed_fallback` (39-58) | low |
| experiments/baselines/mjpc/* | MJPCController stand-in | `**kw` | `step(**kw)` | swaps `demo.mpc_fl` (run_mjpc_compare.py:80-94) | low |
| teleop/keyboard_fl_teleop.py, scripts/smoke_realgrasp_flmpc_lift.py | FLMPC | NC=4, L=8, W_POS_EGO=800 | step (+ table_rows, payload) | a_*_prev, `_cup_axis_local`, last_feasible, `_torque_safe_scale` | low |
| comment-only: control/*, residual_dynamics/*, wm_requirement/spill_predictors.py, model/shield.py, ... | — | — | — | some cite mpc_fl line numbers (e.g. control/state_adapter.py:11) that any refactor makes stale | — |

### The hidden API: patch points that any change must keep [V]

- **S1.** `FLMPC.step` calls the solver as the module global `sc.solve_constrained_box_qp(H, f, lower, upper, a_ineq, b_ineq)`. The first
  call per step is the hard QP. A second call happens only in the fallback. Nine files depend on this.
- **S2.** `mpc_fl.sc` is read at call time (`fv` and cbf_filter swap it).
- **S3.** `mpc_fl._assemble_fl` is called through the module global with today's positional arguments. The torque rows are its last 4·NV rows (run_actuation_eval).
- **S4.** `FLMPC.step`, `_relaxed_fallback` and `_torque_safe_scale` are looked up on the class at call time.
- **S5.** The demo looks up `mpc_fl.FLMPC` at call time (fast_barriers.install; the MJPC swap).
- **S6.** Callers write `fl._cup_axis_local`.

If S1 breaks, nothing fails loudly: the value rows disappear and the gates record nothing. [I]

## 3. Pain points

1. **No generic row input.** Each constraint is a kwarg pair plus a hand-written block in step() (720-927). [V]
2. **One-sided arms.** Tilt, cone, leash and the payload terms (`M_ego_extra`, `bias_ego_extra`) exist for the ego arm only. The speed cap exists for the other arm only. There is no `bias_int_extra`. [V]
3. **Only two row classes**, hard and relaxable. There is no always-soft class and no equality class. ValueFLMPC builds its soft class outside the controller. [V]
4. **Fragile fallback.** It finds relaxable rows by index (927, 951-956). A row inserted before row m_coll would change which rows get a slack, and no test would fail. [V code; I consequence]
5. **Three copies of the rows** (mpc_fl.step, rt_kernels.assemble_rows, part of mpc_fl_decentralized.step) and three of the Python gather (mpc_fl.step, rt_flmpc.step, rt_flmpc._prepare). Each new family means three edits and a new G1 / unit-branch run. [V]
6. **The numba path is closed.** Its workspace size is a fixed formula over the known families (rt_flmpc:175-181), and rt_full takes 72 positional arguments. A new family changes the signatures of rt_core, rt_full, `_prepare` and rt_pipe. [V]
7. **The torque limit lives in four places:** `sc.TAU_MAX` (the Python path reads it per call; rt_flmpc:152 copies it once at construction; rt_pipe:140, 337 read it per call), `tau_max_vec`, and the demo's outer clip. run_actuation_eval patches the global and the kernel instead of using `tau_max_vec`, so its Piper-limit fix never reaches the rt path. [V]
8. **The F6 rule is wrong for hardware.** F6 widens a joint's box to |c_j| (709-712). The real Piper drops an out-of-range command (memory note: ±8 Nm, wrist ±6.4). [V code; I hardware]
9. **Equality rows are inaccurate on the Python solver.** I tested 20 random 48-variable QPs, each with 6 equality rows written as ± pairs plus 20 inequalities. The Python solver reported "feasible" 20/20 but left an equality error of median 2e-2 and max 9.5e-2. GI (`rt_qp.solve_dense`) solved the same QPs to an error below 6.1e-15. [V, synthetic only]
10. **No way to inspect the QP.** Users capture it by patching the solver (cbf_filter `_CaptureProxy`, il_common, rt_capture_qp). [V]

## 4. What the four needs require

- **N1, the rigid tray held by both grippers (closed chain) [I].**
  - Rows: both grippers must produce the same tray twist at chosen horizon steps. That is six rows per step on both arms' columns.
    These are equalities, so they are accurate only on GI (pain point 9). The other route is to eliminate the left arm's variables:
    with frozen Jacobians and Φ = scalar·I per block, the left arm's block accelerations follow from the right arm's. This halves
    the decision vector and keeps the solver inequality-only.
  - Dynamics: FL stays exact only if the tray's inertia is reflected onto the arms. With a fixed load split, each arm gets its own
    payload term, so both arms need `M_extra` and `bias_extra` (today only the ego arm has them). Optimizing the internal squeeze
    force would need extra variables.
  - rt: the numba path computes payload terms for the cup and the cube inside the kernel (rt_kernels:551). The tray needs a third case.
- **N2, rows from a learned task-space monitor.** This is the value-row pattern already proven in `fv`:
  - affine maps from x to the tray acceleration and tilt rate (`fv.HorizonMaps`, 153) [V];
  - the gradient of V turned into soft rows, plus trust-region bounds, refreshed at 15 Hz and held between refreshes [V];
  - the needs: an always-soft class with its own rho, eps and s_max, a bound-narrowing input, and rows on any columns.
    The value step costs 102 ms in Python (MASTER §1b), so for the rt path the rows must be computed off the control loop.
- **N3, base acceleration.** The canonical sim scene has no base joint: nv = 22, and the only non-arm joint is `test_box_free` (checked by loading the model). [V]
  - The FL law becomes τ = M_aa·a + M_ab·q̈_base + c, so both arms need a bias term. [I]
  - The arm-arm clearance rows do not change, because both arms ride the same base. [I]
  - The world-frame rows (table, leash, cone, speed cap) need an offset for the base motion over the horizon. Today those rows take no offset. [V]
- **N4, per-run torque caps.** `tau_max_vec` already exists on both paths [V]. What is missing is one limits object used by the box rows, the backstop, the outer clip and the rt path, plus an explicit rule for "gravity exceeds the cap". This is not a row problem.
- **The rt path.** No Python-only hook reaches the rt path, because RTFLMPC overrides `step`. The rt path needs a "precomputed extra rows" input to rt_core and rt_full, or new numba row builders. [V]

## 5. Options

**A. Minimal hook: `extra_rows=None` on step(), then the same input on the numba path.**
- **What changes.** `FLMPC.step` gets `extra_rows`: a list of blocks (A of shape (m, ntot), b, kind = hard / relax / soft, and rho, eps, s_max for soft rows), plus optional `bounds=(lo, hi)` to narrow the box. The new rows go **after** all existing rows, so the existing row order and slack indices do not change. Relax rows join the fallback's slack set. Soft rows get slack columns in both solves, placed after the fallback's own slacks (the same layout as ValueFLMPC). Equalities are ± pairs.
- **Add-ons.** Per-arm `M_extra` / `bias_extra` (keep `M_ego_extra` as an alias), a per-call torque limit, and `self.last_qp` (H, f, bounds, A, b, row tags) for capture users. The solver is still called through S1.
- **rt port (A-rt).** rt_core and rt_full gain one precomputed block (A, b, kind, m), and the workspace grows. Soft rows need slack columns in the hard GI solve; the slack set-up already exists for the relaxed solve (`gi_init_slack`). Until A-rt lands, RTFLMPC raises NotImplementedError when it gets `extra_rows`.
- **Call sites.** None must change. The demo passes `extra_rows` only when it is set, because FLMPCDecentralized.step has no `**kw`. `fv` and cbf_filter can move off the proxy later.
- **Risk.** Low. The new code runs only when `extra_rows` is not None. S1-S6 stay intact.
- **Needs.** N2 yes. N1 rows yes, but equalities are accurate only on rt and there is no elimination; the dynamics come through the per-arm extras. N3 yes (the caller computes the offsets). N4 only through the per-call limit. rt only after A-rt.
- **Effort.** A-py 1.5 days; A-rt 2.5 days. [I]

**B. Split step() into stages; new constraints become subclasses in new files.**
- **Today.** This is impossible without editing mpc_fl. The only override points are the whole step (FLMPCDecentralized copied it and drifted) or the solver patch point (the `fv` proxy).
- **What changes.** A pure move refactor: `_gather_state`, `_dynamics` (M, c plus extras), `_core_qp` (the `_assemble_fl` call plus F6), `_optional_rows` (families 3-6 in the same order), `_extra_rows(ctx)` returning [] (the hook), `_solve_with_fallback`, `_realize`. step() keeps its signature and calls the stages. New files then hold e.g. `TrayFLMPC`, which overrides `_dynamics` and `_extra_rows`.
- **Call sites.** None. Keep the names step, `_relaxed_fallback`, `_torque_safe_scale` and the `_assemble_fl` call (S3, S4).
- **Risk.** Low to moderate: about 470 lines move, but the bitwise replay (T1) catches any change. Combining subclasses (tray + monitor + base) needs mixins, which pushes the design toward C.
- **Needs.** Everything A gives, plus an overridable dynamics stage (N1 dynamics, N3) and variable elimination for the tray through `_core_qp`. It does not reach rt, because RTFLMPC overrides step.
- **Effort.** 2.5 days. [I]

**C. A constraint registry (plugin protocol).**
- **What changes.** A `RowFamily.build(ctx) -> RowBlock(A, b, kind, slack params, tag)` interface. `ctx` holds the state, M and c, the Φ tensors, a Jacobian cache, the references and the per-call options. FLMPC keeps an ordered list [KernelCore (collision + torque), Tilt/Cone, Leash, Table, SpeedCap] plus user families. Today's kwargs build the default list. The fallback picks slack rows by kind, not by index. Row tags make capture and logging easy.
- **Files.** mpc_fl (step rewritten) and a new rows module.
- **Call sites.** None if the kwargs stay.
- **Care point.** adaptive_yield shares state across families (`tilt0_now`, g, `leash_max_eff`; 645-689, 856-864). Port it with care, or retire it; rt already refuses it (rt_flmpc:125-126).
- **Risk.** Moderate. Bitwise output is possible (same numpy expressions in the same order), but every optional path moves.
- **Needs.** N1 rows, N2 and N3 fit naturally. N4 through a TorqueBox family. An equality class can later route to GI or to elimination. It does not reach rt unless D follows.
- **Effort.** 5 days. [I]

**D. One QP description shared by the Python and numba paths.**
- **What changes.** Each family is written once as an allocation-free numba row writer (pull out the loops already in `rt_kernels.assemble_rows`). Both `mpc_fl.step` and rt_core call the same writers. A writer for precomputed blocks covers learned rows.
- **Files.** rt_kernels, mpc_fl, rt_flmpc, rt_pipe, mpc_fl_decentralized.
- **Call sites.** None in API terms. But the Python path's numbers change: numba loops add terms in a different order from numpy (G1: up to 2.2e-16 in A and 2.8e-14 in b). The Python solver and the closed loop amplify such differences, so episodes no longer reproduce bit for bit.
- **Risk.** High to existing results. Every Python result becomes an "old baseline", and the N=50 sweeps must be re-run (T5).
- **Needs.** A single source, so each future family lands on the rt path at once. Numba has no clean polymorphism, though: each family is still an edit through the kernel signature chain (pain point 6).
- **Effort.** 8-10 days. [I]

**E. Task-space front end (frames to rows), on top of A.**
- **What changes.** A new file, e.g. experiments/ab_smooth_vs_bangbang/flmpc_task.py, generalizes `fv.HorizonMaps`. For a frame (a body plus offset on either arm, or a tray frame) it builds affine maps from x to displacement, velocity, acceleration, small-angle rotation and angular velocity at chosen horizon steps (frozen J, J̇·q̇ drift, optional base-motion offset). Users declare `TaskConstraint(frame, quantity, steps, C, d, kind)` or a relative constraint between two frames. The output is row blocks for A (or families for C). The existing families stay as they are, which keeps the bits.
- **Call sites.** None. **Risk.** None to existing results.
- **Needs.** N2 (V gradient times the maps), N1 rows (relative twist = 0) and N3 (the offset term) all fit naturally. N4 no. rt: the rows are built in Python and passed as a precomputed block, so A-rt is required. A 15 Hz refresh fits the budget; per-step tray rows add Python time on the rt path. [I]
- **Effort.** 3 days on top of A. [I]

### Comparison

| | A hook | B stages | C registry | D shared kernels | E task front end |
|---|---|---|---|---|---|
| Edit to mpc_fl.py | ~60 lines | move ~470 lines | rewrite step | rewrite rows | none |
| Other files | rt_kernels, rt_flmpc, rt_pipe (A-rt) | new subclass files | new rows module | rt_kernels, rt_flmpc, rt_pipe, decentralized | new file |
| Call sites forced to change | 0 | 0 | 0 | 0 | 0 |
| Bitwise on the default path | yes | yes | yes, with care | **no** | yes (additive) |
| Risk to existing results | low | low-moderate | moderate | high | none |
| N1 tray | rows + per-arm extras; exact equalities on rt only | + elimination by override | yes | yes | rows natural |
| N2 monitor rows | yes | yes | yes | yes | best fit |
| N3 base accel | yes (caller offsets) | yes | yes | yes | natural |
| N4 torque caps | per-call limit only | via stage | TorqueBox family | yes | no |
| Real-time numba path | with A-rt | no | no | yes | via A-rt |
| Effort (engineer-days) | 1.5 + 2.5 | 2.5 | 5 | 8-10 | 3 |

## 6. Recommendation

Order: **harness, then A, then the limits fix, then E, then A-rt; C later if needed; not D for now.**

1. **Replay harness first** (T0/T1, 0.5-1 day). Without it, no option can be proven safe.
2. **A-py**, implemented as a small split around the solve only. Pull out `_solve_with_fallback` and the extra-row append; leave the row blocks where they are. Keep S1-S6. Include the three kinds, bound narrowing, per-arm M/bias extras and `last_qp`. Prove T1 (bitwise) and T2 (ValueFLMPC parity), then move `fv` and cbf_filter off the module-global proxy.
3. **The limits object (N4)**, 1 day. Use one limit for the box rows, the backstop, the demo's outer clip and the rt path. State the rule for |c_j| > cap explicitly (question Q2). Fix the rt copy-at-construction. Retire run_actuation_eval's patch.
4. **E in a new file** for the monitor rows, the tray rows and the base offsets.
5. **A-rt** when the first new family must run in real time. The 15 Hz monitor rows are the likely first.
6. **Tray dynamics.** Decide Q1. If you choose elimination, build a separate `TrayFLMPC` on the step-B split. Keep it off the default path.
7. **C** only if three or more new families pile up inside FLMPC itself, or once adaptive_yield is retired. **D** only once the Python path no longer has to be a bitwise reference (Q3).

**Why not C now.** For the four needs it gives what A + E give, edits three times as much of the file the flagship numbers depend on, and still does not reach rt.

**Top risks.**
- **R1:** breaking S1 silently (the value rows vanish). Guard: T1 checks the sequence of solver calls; a test checks that ValueFLMPC still sees its rows.
- **R2:** a different slack-column order in the fallback, so the hook cannot reproduce the `fv` results bit for bit. Guard: put the soft slacks after the collision slacks.
- **R3:** tray equalities on the Python solver look feasible but are off by up to ~1e-1 (pain point 9). Guard: the Python path rejects `eq` rows, or the tray uses GI or elimination.
- **R4:** the hook exists on the Python path but not on rt, so the two drift. Guard: RTFLMPC raises until A-rt lands.

## 7. Test plan (proves "no behaviour change")

- **T0, capture.** Write a new script modeled on rt_capture_qp. Record for every step the inputs (qpos, qvel, act, pairs, refs, kwargs, `_cup_axis_local`, a_prev, delay queue) and the outputs (H, f, bounds, A, b, x, τ_e, τ_i, feasible, last_fallback, solver-call count).
  - Cases: the flagship flmpc run, seed 0, ctrlfixes off and on (delay, taumax, true cone, speed cap); a stress run with H_MARGIN raised to force the fallback; one value-row seed from scenario.py (d_value); one rt_unit_branches case per branch.
  - Store under /common/users/ss5772/bimanual_data/ (quota notes).
- **T1, bitwise replay on the Python path.** Run new code with default arguments. `np.array_equal` must hold for every recorded output on every step, and the solver-call sequence must match (this protects S1). Allowed difference: zero. Needed for A, B, C, E.
- **T2, hook parity.** ValueFLMPC through the proxy vs through the hook on the value-row capture. Target: bitwise. If the layout cannot match exactly, require |Δτ| <= 1e-9, identical fallback decisions, and identical last_plan_ego.
- **T3, rt gates.**
  - `rt_unit_branches.py`, and `rt_gate_g1.py --profile flagship --seed {0,1,2}`. Every count must equal the stored g1_*.json (rows_count_equal 5736, A_maxabs <= 2.2e-16, fallback_agree, stress_feas_agree 559/573).
  - Old rt vs new rt with an empty extra block: bitwise.
- **T4, episodes.**
  - `gate_equivalence.py --mode e2e`: 0 differing series; collision False, placed_final 4, h_min 0.0192984598...
  - The flagship demo with `--no-video --controller flmpc`, seed 0, with `--controller-impl py` and `rt`: logs identical to a pre-change run.
  - Also smoke-run the patch users once each (run_actuation_eval baseline seed 0, the il_common override meter, a_flmpc_ee_audit): same numbers as before.
- **T5, N=50.** Required only if T1 is not bitwise (D, or any change to Python-path floating point).
  - Run comprehensive_eval for seeds 0-49 on py and rt, then `rt_gate_g3.py` against the stored results.
  - Pass: collisions 0/50, spills 0, placed_strict 3; h_min_ref within 2 mm; tilt within 1°; fling within 0.05 m/s. Report jerk (N=50 reference: 301) and SPARC.
  - Add a 10-seed spot check of tuned v4c, because v4c shares the demo.
  - Run these as background jobs with 56 workers and pinned threads (AGENTS.md).
- **T6, timing.** G4 pinned to one core with GC off. With an empty extra block, the rt in-loop p99 must stay within 10% of 0.385 ms. Report the cost of the extra block separately.

## 8. Open questions for you

- **Q1. Tray closed chain.** Equality rows (exact only with GI) or eliminating the left arm's variables? Fixed load split or optimized squeeze force?
- **Q2. Torque cap below gravity.** On the ±8 Nm Piper, j1/j2 gravity is ~7.4 Nm. When |c_j| exceeds the cap, should the controller widen the box (sim F6), refuse and brake, or report infeasible? The hardware drops out-of-range commands.
- **Q3. Is the Python path still the bitwise reference**, or may rt become the single source? A "yes, rt is the source" answer reopens D.
- **Q4. Base acceleration.** Is it planned and known ahead (feed-forward) or measured only? Lift only, or a planar base too?
- **Q5. Monitor rows.** Always soft (exact penalty), or hard when the monitor is confident? Does the 15 Hz refresh stay?
- **Q6.** May the six research scripts that patch FL-MPC internals (`fv`, cbf_filter, il_common, a_flmpc_ee_audit, run_actuation_eval, run_mjpc_compare) be moved to the new hook in a one-time edit, or must they keep working untouched? (The other nine patchers are perf installers and gates.)
- **Q7.** Should FLMPCDecentralized get the hook, or stay frozen as an ablation?
