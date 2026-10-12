# DESIGN: what must change to run the world-model and value code on the tray dataset

Status: a plan. Nothing here was edited or run against the old code. I read the files named below.
Line numbers are from 2026-10-05 and will drift.

## 0. The short version

- The old code is wired to the pendulum in four ways: obs is 9 numbers, the privileged "feat" comes from the
  pendulum's bob state, the fallback set is a fixed 85-element array, and every ground truth (labels, "doomed",
  closed-loop outcome, counterfactual branches) is the pendulum integrator `pend_rollout_torch`.
- The tray dataset keeps the first three. It breaks the fourth on purpose: MuJoCo is the truth.
- Do not edit the old files. At least nine other modules import them (section 8), and the pendulum results must stay
  reproducible. Port by writing new files (suggested home: `tray_rep/wm/`) that copy the small parts that need a change
  and import the rest.
- Smallest port, in order: (1) a new trainer with `d_obs` and `FEAT_DIM` as parameters, (2) a new `Dyn` for the value,
  (3) a new gate and closed-loop driver that call `tray_rep.oracle` and `tray_rep.cl_eval_mj`. `reach_value.train` needs no change.

## 1. What the tray dataset provides (the contract)

| thing | pendulum | tray_rep |
|---|---|---|
| `obs` | 9 = axis(3), v(3), w(3) | `full`: 13 = the 9 + ball x, y, vx, vy (tray frame). `tray_only`: the same 9 |
| `act` | 6 = a(3), w_eff(3) | same 6, same meaning |
| `l` | in [-1, 1] | in [-0.5, 1]; < 0 = ball centre off the tray |
| `n`, `nd` | bob direction / rate (unit-ish vectors) | ball position / velocity in the tray frame (3 each). NOT unit vectors |
| `feat` (aux target, 10) | `feats_t(n, nd, c)` yaw-invariant pendulum features | `env.feats_tray(priv)`: 10 scaled numbers from `priv` |
| hidden parameter | none (h fixed) | `hid` = `mu_roll` per episode (not in obs) |
| fallback set | 85 = 17 accels x 5 tilt rates, `C.fallback_U()` | the same 85 times `a_scale` (`TrayRepEnv.U`) |
| `load_split` keys | obs act l n nd lens start onset seed | the same, plus `hid`, `priv`, `tunnel` (`tray_rep.store.load_split`). `wm_audit.common.load_split(name, data=dir)` also works and ignores the extras |
| branch tensors | `(N,160,9/6/-/10)` torch | `(N,160,13 or 9 / 6 / - / 10)` numpy in an npz; `branches.load_branches` gives torch |
| ground truth | pendulum integrator | MuJoCo snapshots (`tray_rep.store.Store.snapshot_at`) |

Index convention that matters: the first three obs columns stay the tray axis. Code that reads `obs[:, :3]` as the axis keeps working.

## 2. `latent_hj/kin_wm.py` (the WM)

| where | now | needed |
|---|---|---|
| `build_model(d_obs=9, ...)` L109 | already takes `d_obs`. Encoder, decoder, margin head and the normalisation buffers use it | call it with `d_obs = 13` (`full`) or `9`. No code change, but every caller must pass it (see below) |
| `load_wm(path)` L215 | calls `build_model()` with the default 9 | load fails for a 13-D checkpoint. Save `d_obs` in the checkpoint `args` and read it back (new loader in the port) |
| margin head `tanh` | output in [-1, 1] | fine for `l` in [-0.5, 1]. It cannot reach exactly 1; negligible |
| `WMDyn` L266 | `U = sp.control_set(P)`: unscaled | scaled set for `a_scale != 1`. Used by `kin_wm` value/gate commands only; the audit code uses `ProjWMDyn` (section 4) |
| `latent_bank` L298 | pads obs with a hard-coded `(.., 9)` at L308 | `d_obs` |
| `cmd_build`, `_episode_arrays` | build the pendulum dataset from spill_v2 | not used; `tray_rep.gen_nominal` replaces it |
| `cmd_value` L332 | `sp.PendParams()`, `P.dt` | the value's `P` is only used for `dt` and `m_lo`; see section 5 |
| `KinWMObserver` L224 | online filter from a 200 Hz cup pose | pendulum/cup specific. Not needed in simulation. For hardware it needs a ball tracker input; out of scope |
| `load_split(data, name)` L101 | note the argument order differs from `common.load_split(name, data)` | use `tray_rep.store.load_split(name, data)` |

Model question, not code: in `tray_only` mode the ball is not in `obs`. The GRU must dead-reckon it from the tray
motion history (the ball starts near the tray centre). Windows of 160 ticks that start mid-episode lose that anchor, so
`tray_only` training should use more from-start windows (the trainer takes 30% now). In closed loop, give the filter the
whole logged history from tick 0, as `cl_eval_mj` does.

In `full` mode the hidden `mu_roll` has to be inferred from how the ball accelerates under tray tilt. The GRU can do that
only if the history contains tilt. A state restarted from a quiet moment has a poorly determined latent.

## 3. `wm_audit/train_kinwm.py` and `vdirect/trainer.py` (training)

Both train the same model with the same hard-coded numbers. Both need the same changes.

| where | now | needed |
|---|---|---|
| `feats_t` (train_kinwm L31) | `sp.torch_fns()["feats"](n, nd, c)`: pendulum features | drop. Use `feat = env.feats_tray(priv)`; the dataset has `priv`. `add_feats` (train_kinwm L140-145, trainer `train_data` L33-38) builds `feat` from `n, nd, obs[:, :3]`: replace with `feats_tray(D["priv"])` |
| aux head | `nn.Linear(wm.d_h + 9, 256) ... nn.Linear(256, 10)` (train_kinwm L133, trainer L76) | `d_h + d_obs` in, `FEAT_DIM` out (10 here, so the output width is unchanged, but its meaning changes) |
| `kin_wm.build_model()` (train_kinwm L132; trainer L75, L167; run_rb L56) | default `d_obs = 9` | `build_model(d_obs=...)` at all five sites |
| data loading (train_kinwm L127, L247; trainer L33; value_gate L71, L94, L145) | `C.load_split("train")` uses the global `WMDATA` default | pass `data=<tray dir>`; thread it through `train_data`, `build_bank`, `gate`, `evaluate` |
| `make_branches` (train_kinwm L37) | pendulum integrator, torch | replace with `tray_rep.branches` output. The pipeline reads `BR["obs"|"act"|"l"|"feat"]`, which the npz has |
| `BR["obs"].reshape(-1, 9)` (train_kinwm L162; trainer `compute_norm` L52) | 9 | `d_obs` |
| `compute_norm` (trainer L45) | builds reference branches with the pendulum `make_branches` | use a fixed reference sample of tray branches (law R, fixed seed) |
| `subset_eps` (common L45) | keeps only obs, act, l, n, nd, lens, onset, seed | also carry `priv`/`hid` when `--frac < 1` is used |
| `evaluate` (train_kinwm L244) | `dg.counterfactual`: pendulum ground truth | drop, or replace by `tray_rep` branches/oracle states |
| margin loss | MSE to `l` | unchanged. `l` is continuous and saturates at -0.5 after failure |

Branch effect to watch: 96-98% of R branches fail (README, finding 1). With `branch_frac = 0.5` the margin head sees
mostly negative `l`. The pendulum recipe was not tuned for this. Check the branch-failure fraction before training.

## 4. `wm_audit/value_gate.py` (`ProjWMDyn`, gate)

`ProjWMDyn` is the `Dyn` object `reach_value.train` calls (`step`, `margin`, `features`, `U`, `sample_states`, `P`).

| where | now | needed |
|---|---|---|
| `self.U` L38 | `C.fallback_U()` | `TrayRepEnv.U` (scaled by `a_scale`). This is the fallback set of the whole value |
| `self.P = C.P0` L40 | `PendParams`: `tilt_lim = 1.5`, `m_lo = -1.0` | `reach_value.train` reads `dyn.P.m_lo` as the value floor (`v_lo`). Give the tray dyn a small object with `m_lo = -0.5` (the tray margin floor) and `tilt_lim = 1.5` |
| the tilt projection in `step` | zeroes `w` when the WM's own axis nears 1.5 rad | keep. It reads obs columns 0..2 (the axis). Tray tilt stays below 0.4 rad (p99), so it rarely fires |
| `step`, `margin`, `features` | generic in `dh` and `z = [h, o_n]` | no change. `features` is `z`, so the value net input is `d_h + d_obs` wide |
| `build_bank` L69 | `C.load_split("train")`; `C.filtered_latents` | needs the data path and a 13-wide `pad_episodes` (below) |
| `C.pad_episodes` (common L81-84) | hard-coded `(.., 9)` | `d_obs` from `D["obs"].shape[1]`. `filtered_latents` calls it, so every consumer breaks for 13 until it is fixed (new copy in the port) |
| `gate()` L86 | `doomed = (l > 0) & (V_phys <= 0)`; `V_phys` from `load_phys_value` + `phys_V(n, nd, c)` | no physics value exists for the tray. Replace the label with the MuJoCo oracle: use `oracle` labels on TEST states. Define positives as certified savable (label 1). Report (a) P(V_wm > 0 | savable) = the accepted fraction, (b) P(V_wm > 0 | hold_ok), and closed-loop failure (section 6). Do not call uncertified "doomed". The "doomed recall" metric loses its meaning |
| FL-MPC gradient-cosine test (L145-177) | compares `dV/du` to the physics value | drop for phase 1 |

## 5. `latent_hj/reach_value.py`

- `train(dyn, ...)`: no change. It only uses the `Dyn` interface. Pass `device`, and make sure `dyn.P.m_lo = -0.5`.
- `ValueNet(dx, ..., v_lo)`: `dx = d_h + d_obs`. `v_lo` comes from `dyn.P.m_lo` (above). `load(path)` takes `v_lo` from `meta["P"]["m_lo"]`;
  the port must write `meta = dict(dx=..., P=dict(m_lo=-0.5, ...))` or the reload uses -1.
- `PendDyn`: pendulum only; do not use.
- The discount in time: the pendulum run used one WM step per backup (`P.dt = DT_WM`). Keep `gamma` annealing as is; the tray horizon of interest
  (savability lost 2-5 ticks before failure) is shorter than the pendulum's, so check that `gamma_final = 0.98` per 20 ms tick is not too myopic or too long.
  This is a tuning question, not a code change.

## 6. `vdirect/cl_eval.py` (closed loop)

| where | now | needed |
|---|---|---|
| `cl_states` L27 | picks 2000 TEST states by `V_phys` (1500 near, 500 safe) | use `tray_rep.oracle` output: certified-savable states (label 1) |
| `_rollout` L82, `reference_outcomes` L109 | truth = `pend_rollout_torch` (batched on GPU) | truth = MuJoCo. `tray_rep.cl_eval_mj.evaluate` does it per state on CPU: 50 ticks of MuJoCo plus one policy call per tick |
| `Policy.q` L65 | `V(f_WM(z, u))` over `base.U` | same math; the policy needs `U` scaled and the latent kept between calls |
| WM filtering inside `_rollout` | feeds `o = [c, v + DT_WM u, w_obs]` from the pendulum | the observation now comes from MuJoCo (`rec["obs"]`, 13 or 9 numbers). Do not synthesise it |
| `phys_succ_values`, winner's-curse block (L135, L183-250) | `V_phys` of the true one-tick successor; pendulum one-step truth | no analogue. Drop for phase 1. A MuJoCo version needs one extra oracle call per (state, action): too costly |
| references `phys_greedy`, `hold` | pendulum | `hold` = `cl_eval_mj.hold_still`. `phys_greedy` has no analogue; the closest is `OraclePolicy` (privileged upper bound) |
| `Vp0 > 0` as "savable" | physics value | oracle label 1 |

How a WM policy plugs into `cl_eval_mj.evaluate`: write a class with `begin(ctx)` and `__call__(obs_hist)`.
`begin` filters the logged history `obs_hist[:t+1]`, `act_hist[:t]` into the GRU state (`C.filter_fast`), then each call
feeds the newest obs and the last action it returned, scores the 85 actions with `Policy.q`, and returns `U[argmax]`.
Cost: 2000 states x 50 ticks x one 85-way Q call is about 100k GPU calls; minutes. If that is too slow, run all states
in lock step (N `TrayRepEnv` objects, one batched WM call per tick). That batch runner is not built.

## 7. `vdirect/branches.py` and the other arms (not needed for R / Rc)

`rollout_branches` runs value-/uncertainty-directed branches closed loop on the PENDULUM, with the WM choosing actions
each tick. On the tray this needs MuJoCo in the loop: N envs stepped in lock step, one batched WM call per tick.
`tray_rep.branches.run_branch` is open loop (actions drawn in advance), so R and Rc are covered but V, U, F, and the
DAgger-style arms are not. Add a per-tick action callback to a lock-step runner when those arms are needed.

## 8. Impact: who imports what

Method: (1) `python3 -m repo_graph_agent impact --repo ... --files <f>`; (2) `.codex_graph/repo_graph.json`
`reverse_import_index` by module stem; (3) a plain `grep` of the current tree. Results:

- The graph is stale (written 2026-09-29; the vdirect `rb*`, `run_rb`, `depth_diag` files are newer). The reverse index has
  no entry for `kin_wm`, `reach_value`, `value_gate`, `trainer`, `cl_eval`, `train_kinwm`, `branches`. It does have `common`, but that
  is a different `common` (grasp_audit). An empty or missing entry is UNKNOWN, so I used grep as the ground truth.
- The `impact` command returns a superset (it follows imports transitively): for `kin_wm` it also lists `cbf_filter`, `flmpc_value`,
  `grid_hj`, `spill_data`, `validate_*`; grep shows `flmpc_value` does not import `kin_wm`. Treat its list as an upper bound.

Direct importers by grep (current tree, `tray_rep` excluded):

| module | imported by |
|---|---|
| `latent_hj/kin_wm.py` | `latent_hj/scenario`; `wm_audit/{closed_loop, diagnose, train_kinwm, value_gate}`; `vdirect/{trainer, cb_eval, oc_eval, depth_diag, run_rb}` (`load_wm` / `build_model` at 11 call sites outside `kin_wm.py`, none passing `d_obs`) |
| `latent_hj/reach_value.py` | `latent_hj/{kin_wm, cbf_filter, scenario, validate_stage0}`; `wm_audit/{common, value_gate, closed_loop}`; `vdirect/{run_arm, run_rb, cb_eval, oc_eval, depth_diag}` |
| `wm_audit/value_gate.py` (`ProjWMDyn`) | `wm_audit/closed_loop`; `vdirect/{branches, cl_eval, cb_eval, oc_eval, run_arm}`; `vdirect/pess.py` also uses it and has a similar projection of its own (L129) |
| `wm_audit/train_kinwm.py` | `vdirect/{trainer, branches, rb, rb_selftest, run_arm}` (`feats_t`, `make_branches`) |
| `wm_audit/common.py` | most of the above. Other testbeds (`a_loose_items`, `c_block_tower`, `d_opaque_fill`) use `fallback_U` or `load_split` by name; `bead_rep/bead_env.py` imports it, and `b_tray_ball` and this package depend on `bead_env` |
| `vdirect/trainer.py` | `vdirect/{run_arm, run_rb, depth_diag, rb_selftest}` |
| `vdirect/cl_eval.py` | `vdirect/{run_arm, oc_eval}` |

Consequence: do not change `common.fallback_U`, `pad_episodes`, `load_split` defaults, `kin_wm.build_model` defaults or `ProjWMDyn.U`
in place. They are shared by the pendulum arms, the bead replication and the other testbeds.
Slurm and shell launchers that call these modules by name (`kin_wm build/train/value`, `train_kinwm`, `vdirect.run_arm`) would also
break on a signature change; I did not enumerate them.

## 9. Suggested order of work (phase 1b)

1. `tray_rep/wm/common.py`: `load_split` with data path, `pad_episodes(d_obs)`, `filtered_latents`, scaled `fallback_U`, `feats` from `priv`.
2. `tray_rep/wm/train.py`: a copy of `train_kinwm.main` with the dims as arguments, branches from `tray_rep.branches`. Run `--branch-frac 0` (nominal) and `0.5` with R and Rc.
3. `tray_rep/wm/dyn.py`: `ProjWMDyn` for the tray (scaled `U`, `P.m_lo = -0.5`). Train the value with `reach_value.train`.
4. `tray_rep/wm/policy.py`: a WM policy for `cl_eval_mj.evaluate` (section 6), then the lock-step batch runner if speed needs it.
5. Gate on oracle labels; closed loop on certified states; compare hold-still, WM-greedy (nominal data), WM-greedy (R), WM-greedy (Rc), and the oracle policy.
   The claim to test: nominal data fails on many savable states; branch data fixes it; more caution does not.
6. Baseline the real job of the WM: a physics model with an online `mu_roll` estimate (b_tray_ball `analytic.py` has the reduced models). If it ties the WM,
   say so; the WM's value is then only in unmodelled payloads.

## 10. Open questions for the owner

- `a_scale`: the data is generated at 1.0 in the demo commands. The Piper-feasible value (another job) should be fixed before the 3000-episode run,
  because the branch box, the fallback set and the oracle all depend on it. The nominal carry does not.
- Whether R's branch law should be milder on this testbed (96-98% failing branches).
- `tray_only` vs `full` as the headline setting. The ZED may not see the ball (b_tray_ball report, section 3).
- Observation noise and delay for the ball (2 mm, 30 Hz, 50-100 ms in the report). The data is noise-free now.
