# bead_rep -- MuJoCo cup-with-beads replication, Phases 0-1 (code + smoke only)

Design: `/common/users/ss5772/bimanual_data/bead_replication/DESIGN.md`. SIMULATION / SAVED DATA ONLY: no hardware, no
network, CPU only. No existing repo file is edited; everything is imported from `world_model/v2/data/contents.py`
(rig_model, Oracle, pin_cup) and `wm_audit/common.py` (fallback_U, pendulum labeller).

Owner decisions applied: fixed nominal contents (8 beads, `Contents(**NOMINAL)`); "savable" = search-certified by the
oracle library (an UNDER-approximation; not-certified = ambiguous, never "doomed"); compute = ilab CPU arrays (you submit).

## Files
| file | what |
|---|---|
| `bead_env.py` | Phase 0. `BeadEnv` (cup-only rig, kinematic cup, `tick(u)` at 50 Hz, `snapshot()/restore()` of the full state, tunnel guard), `kin_step/kin_tick` (cup integrator with the pendulum labeller's action meaning), `bead_margin` (DESIGN Q2), `brake_action`, `Snapshot`. |
| `gen_nominal.py` | Phase 1 generator (replay of the `spill_v2` cup trajectories), `--merge`, `--verify`. |
| `oracle.py` | search-certified savable check; subcommands `candidates`, `run`, `merge`. |
| `util.py` | NFS-race-tolerant `safe_mkdir` (catches FileExistsError/OSError, re-checks `is_dir()` after a short sleep), atomic npz writes, sentinels, source-shard / split index. |
| `slurm/bead_gen.sbatch`, `slurm/bead_oracle.sbatch` | CPU array templates (NOT submitted). |
| `tests/` | `python -m pytest world_model/v2/bead_rep/tests -q` (11 tests, ~12 s, CPU). |

## Action meaning (matches `common.pend_rollout_torch` exactly; tested to 1e-10)
u = (a[3], w[3]) held for one WM tick = 4 physics steps (5 ms): `a` = cup-ORIGIN linear accel (semi-implicit Euler, so the
backward 2nd difference of the pose position == a), `w` = commanded axis angular velocity. Per step
`c' = normalize(c + dt w x c)`, then the tilt projection (`arccos(c_z) > 1.5` -> `c = [h_xy sin 1.5, cos 1.5]`); the
quaternion follows the axis by the minimal rotation c -> c' (no spin about the axis, like the labeller's
`w_real = c x c'/dt`). `tick(u)` returns `obs=[axis, v, w_eff]`, `act=[a, w_eff]` (closed_loop.py semantics: v is
`v + DT_WM a`), the bead margin `l`, the sticky rim-spill count and the tunnel flag. U = `fallback_U()` (85 elements).

State/label timing (== `contents.run_rig`): "state at sim step k" = cup pose K_k known (not yet pinned), beads after k
mj_steps; one step = pin K_k -> exit check -> mj_step -> K_{k+1}. The label at step k is computed in that state.

## Margin and tunnel guard
`l = clip(min_i l_i, -0.02, 0.05)/0.05` in [-0.4, 1], `l_i = H - z_i` if `rad <= R` else `min(H - z_i, 0) - (rad - R)`
(cup frame). Continuous across the rim (tested); `l_i < 0` <=> `Oracle.out_mask` for every non-tunnel point (tested).
Each ball's FIRST exit is classified from its cup-frame z: `z > H - 12 mm` -> rim spill (sticky, status 1); else side-wall /
floor **tunnel** artifact (status 2): the ball is excluded from the margin and the spill count and the episode/branch is
flagged `tunnel`. (A floor tunnel has `l_i = H - z > 0`, so only the guard sees it.)

## Dataset format (`<out>/<split>.npz` after `--merge`; read with `wm_audit.common.load_split(split, data=<out>)`)
Per-shard files `<out>/<split>/shard_<g>.npz` (g = global source-shard index 0..179), same keys, plus
`snap_<g>.npz`, sentinel `shard_<g>._DONE`. Default `<out>` = `/common/users/ss5772/bimanual_data/bead_replication/data`.

| key | shape | meaning |
|---|---|---|
| `obs` (N,9) f32, `act` (N,6) f32 | | **bit-identical** to the pendulum `wmdata` (same formulas as `kin_wm._episode_arrays`; verified by `--verify`) |
| `l` (N,) f32 | | bead margin (above) at each WM tick, in [-0.4, 1] (NOT the pendulum's [-1, 1]); it is the *current* margin, not sticky |
| `n` (N,3), `nd` (N,3) f32 | | **substitutes** for the pendulum bob direction/rate: centroid of the beads still inside the cup, in the cup frame [m] (zeros if none), and its backward-difference rate [m/s] (0 at t=0). Privileged, aux head only, never a WM input. NOT unit vectors: do not feed to `phys_V`/`feats_t`/`load_phys_value` (no bead analogue of V_phys) |
| `lens` (E,) | | ticks per episode (== pendulum wmdata), `start` is derived by `load_split` |
| `onset` (E,) | | first tick containing a rim exit (`onset200 // 4`), -1 if none. The first tick with `l < 0` is `onset` or `onset+1` |
| `seed` (E,) | | spill_v2 episode seed (join key to the pendulum wmdata) |
| `onset200`, `tunnel`, `n_rim` (E,) | | sim step of the first rim exit; tunnel-flagged episode; #rim-spilled balls at the end |
| `bpos` (N,8,3) f32, `n_in` (N,) | | optional aux: bead centres in the cup frame (clipped to +-0.3 m), #balls inside the cup |

Dropped vs pendulum wmdata: its `onset` (pendulum onset; join by `seed` if wanted).

**Episode split / mapping (recoverable and verified).** The generator reuses `spill_data.shard_split` on the sorted 180
`spill_v2` shards (idx%5==4 -> test, ==3 -> val, else train) and replays every episode in shard order, so
episode order, `lens` and `seed` equal the pendulum `wmdata` (train 2420 / val 818 / test 703 = 3941). `--verify` matches by
seed and compares lens/obs/act bitwise (smoke: 47/47 train + 1/1 val episodes match, same order).

**Snapshots** (`snap_<g>.npz`): every 5th WM tick from tick 20 (every 100 ms), `snap_state` (286 = `mj_getState(INTEGRATION|WARMSTART)`),
`snap_prev` (qpos after the last pin, for the inertial carry), `snap_has_prev`, `snap_kin` (pos3, quat4, v3 backward diff),
`snap_status`/`snap_exit_step` (per-ball first-exit bookkeeping), `snap_k` (sim step), `snap_ep` (episode within the shard file), `snap_tick`.
About 5-7 MB per shard (~1.2 GB for all 180). `oracle.StateStore` restores any (split, ep, t): exact on grid ticks, otherwise
nearest earlier snapshot + replay of the logged cup poses.

## Oracle (DESIGN Q5)
Certified savable <=> some library sequence keeps beads in for T=100 ticks (no rim spill, no tunnel). Library, early exit:
(1) hold u=0 -- safe => **trivially safe** (`trivial=1`, `kind=1`; the DESIGN pre-filter drops these), (2) brake (decel to 0 + axis to
vertical, then hold), (3) the other 84 held constants, (4) 300 random 2-segment sequences (3-15 ticks, uniform in the 85-set)
**followed by the brake controller** (my reading: DESIGN left the tail open and a coasting tail tunnels). Output per state:
`certified, kind, idx, trivial, best_margin` (max over valid sequences of the min tick margin; a spilled sequence counts <= -1e-3, so
`best_margin>0 <=> certified`), `hold_margin, hold_spill_tick, n_eval, n_spill, n_tunnel, out_of_domain, l0, wall_s`.
**`out_of_domain=1`**: the state's tilt already exceeds the action set's 1.5 rad clamp; the first step would snap the kinematic cup ~20 deg
in 5 ms and tunnel the beads (smoke: all 386 branches tunnelled), so it is not searched (never reported as doomed).

## Run commands (owner)
```
cd /common/users/ss5772/bimanual ; PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
mkdir -p /common/users/ss5772/bimanual_data/bead_replication/logs          # sbatch --output dir must pre-exist (already created)
# tests / local smoke (<=1-2 cores, nice):
nice -n 10 $PY -m pytest world_model/v2/bead_rep/tests -q
nice -n 10 $PY -m world_model.v2.bead_rep.gen_nominal --out /common/users/ss5772/bimanual_data/bead_replication/smoke_out --only-gidx 0 --splits train --limit-eps 3
# Phase 1 (ilab CPU array, via the rlab1 gateway):  scripts/offload/submit_via_slurm_gateway.sh world_model/v2/bead_rep/slurm/bead_gen.sbatch
$PY -m world_model.v2.bead_rep.gen_nominal --merge            # after all _DONE (60 sub-shards); then:
$PY -m world_model.v2.bead_rep.gen_nominal --verify           # mapping + obs/act parity vs wmdata (reads wmdata only)
# Phase 2 oracle:
$PY -m world_model.v2.bead_rep.oracle candidates --split val --out states_val.csv --n-near 1500 --n-safe 500
sbatch --export=ALL,STATES=$PWD/states_val.csv,TAG=val_v0 world_model/v2/bead_rep/slurm/bead_oracle.sbatch   # via the gateway
$PY -m world_model.v2.bead_rep.oracle merge --tag val_v0      # -> oracle/val_v0.npz + val_v0_summary.json
```
`candidates` is a saved-label proxy (margin > 0, tilt <= 1.5 now, the LOGGED continuation spills within 2 s), not the savable set.
Add `--shard i/n` (CLI) or the sbatch `NTASK/NPROC` env if you change the array size.

## Smoke results (westeros, nice 10, 1 core per process, load ~55-60/72, so pessimistic)
* Unit tests: 11 pass (bitwise restore incl. a fresh env, margin continuity/consistency, tunnel guard, parity with `pend_rollout_torch`
  incl. tilt projection and accel meaning, `run_rig` bitwise replay parity, brake limits, oracle determinism).
* Generator: 3.7 s/episode mean (2.1 ms/sim step; 0.8-3.3 ms/step range with host load), ~82 s per 22-23-episode source shard, maxrss 164 MB.
* Oracle: hold branch (100 ticks) 1.0-1.8 s (mean 1.3 s over 44 trivial states); a full-length 100-tick branch 0.8 s (random 2-seg + brake) to 1.4 s (held 5 m/s^2).
* See the sizing and the findings in the session summary (and the open risks below).

## Open risks / findings
1. **No hold-spill states inside the action domain.** All 6/6 logged bead spills in the 47 smoke episodes occur at tilt 1.7-1.9 rad (97-108 deg), 14-35 ticks
   after the tilt crossed the 1.5 rad clamp; 44/44 near-boundary candidates (tilt <= 1.5, logged spill within 2 s) and 0/118 random synthetic in-domain states are
   **trivially safe under hold**. So the "hold spills within T" pre-filter of DESIGN Q5 may leave almost no states (G1 needs >= 300). Random 2-seg branches from a near-boundary
   state do spill (2/25), so R branches have signal, but the savable/doomed contrast may not exist with 8 nominal beads. Levers for the owner: more/larger beads (contents.RANGES), a bead-specific tilt limit > 1.5, or a different pre-filter.
2. 11.6% of logged ticks have tilt > 1.5 (out of domain); candidates exclude them.
3. Margin saturates at 1.0 on 71% of ticks (freeboard > 5 cm), 13% in (0,1), 17% < 0; only 12.8% of episodes spill (pendulum: 56%), always as whole-cup tipping (8/8 balls).
4. Branch actions have no velocity feedback (closed_loop semantics): speeds > ~2.4 m/s tunnel; guard flags them, oracle treats them as invalid.
5. Tilt-projection + logged-state tilt > 1.5 -> impulsive snap (see Oracle).
6. Timings are from a heavily loaded host.
