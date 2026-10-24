# Certified viability library (`library.py`) -- SIMULATION ONLY

Recovery = get back INTO a growing set of states that earlier gatekeeper steps already certified (not "go home"). Theory: `docs/paper/theorems_T1_T2_draft_2026-10-16.md` section 4 (T2a antitone admission, T2b persistence under intersection, T2c state sets grow -> containment test).

## Definitions
- **Entry** = (X, V, end state, margins, progress, provenance). `X` (class `Ctx`) is the GROWN belief the segment was admitted under: pose box (`pose_wall`, `e_xy`, `e_th`; `pose_people`), velocity box (`v`, `v_tol`), mu interval (and optional mass), per-person context (centre, position error, age, v_hat), person-model constants (`v_p_max`, `a_p`, `e_v`), map key, buffers (`d_wall`, `d_person`), `a_cup_max`, `a_hw`, `v_cap`, `r_cup`, footprint. `V` = nominal + backup twist profile (a `gatekeeper.traj.Traj`). `progress` = task-route distance gained start -> rest pose (`RouteProgress`).
- **Growth** (`grow={"xy","p","mu"}`): bisect the largest pose-box / person-error / mu_lo under which the FIXED trajectory still passes check1+check2 (sound by T2a); the final joint region is re-checked. Without it a truth-source belief has e_xy = 0 and an entry would only match its exact pose.
- **Containment** `contained(b, X, path)` (all conditions monotone => `b ⊑ X` implies every check1/check2 margin at b >= the entry's):
  map/footprint/docks/r_cup/person model equal; b's `v_p_max,a_p,d_person,d_wall` <= X's; `a_hw,v_cap,a_cup_max` >= X's; mu and mass sets of b inside X's; `|v_b-v_X|_inf + e_v,b <= v_tol`;
  pose: `|c_b-c_X| + rp|dth| + e_xy,b <= e_xy,X` and `|dth| + e_th,b <= e_th,X` (rp = path radius: a rigid shift of the centre moves path point k by <= |dxy| + |rel_k||dth|);
  every person of b is FAR (b's own speed discs never reach the swept footprint) or COVERED by a person of X (`|dc| + shift + r0_b <= r0_X`, age and, for the accel model, velocity/accel bounds).
- Hard failures (map, parameters, centre outside box, uncovered person) cannot be fixed by looking; soft failures (state-set radii `e_xy, e_th, e_p, age, v`) can.

## API
`Library(gk, progress_fn, grow)`; `add(traj, bel, gk, ...)` (re-runs check1/2: never inserts an uncertified entry; dominance dedupe/merge; kd-tree index); `hook()` -> callback `f(traj, bel, gk, *, t, step, T_S, dev, v0)`; `lookup(belief, v)` -> containment matches ranked by progress (`verify=True` re-runs the checkers); `recover(belief, v)` -> `Recovery(mode, entry, legs, look, reason)`; `learn(mu=, mass=)` (intersection, T2b; empty => falsification); `falsified(kind, observed)` (mu, mass, v_p, a_p, e_p, pose, map); `monitor_failure(tags)`; `drop_where`.

## recover() order
1. **direct**: any applicable entry (best progress).
2. **move**: for the k nearest rest-start entries: stop leg (if moving) + straight translation + rotation (holonomic base), each fitted by bisection on the speed multiplier and CERTIFIED with check1+check2 under the belief re-anchored at the leg start and AGED by the time already elapsed (`latency_s += t_off`, speed model). The arrival belief (pose = actual end pose, aged by the connector duration) must be contained in the entry's X. Cheapest = connector time + remaining route / v_nom.
3. **look**: no connector, but only soft (state-set radius) deficits stand between b and some entry: returns `{entry_id, reduce: {name: (current, required_max)}}` for the perception side. Else **none** (fall back to the gatekeeper's plain backup / regime 2).

## Integration into gate.py (not done here)
```python
# Gatekeeper.__init__:      self.lib = None            # a checkers.library.Library
# Gatekeeper._commit(...):  if self.lib is not None: self.lib.add(tr, bel, self.gk, v0=tr.V[0], t=t, step=self.n_steps, T_S=T_S, dev=dev)
# where n_falsified += 1:   self.lib.monitor_failure(tags=<suspect assumptions>)   # or lib.falsified("v_p", observed_speed) etc.
# regime-2 / no candidate:  rc = self.lib.recover(bel, v0)  -> "direct": track rc.entry.traj(); "move": track rc.legs[i].traj in order (re-run gk_step each
#                           tick; abort when a leg stops being certified); "look": hand rc.look to the perception layer
```
`bel` must be the belief the candidate was actually certified under (the particle-filter-refined one in the current gate.py `_commit(..., bel, t)`). `demo run_demo` shows a subclass doing exactly this.

## Limits (honest)
- Velocity box is exact (`v_tol=0` default): moving entries only match their own twist; MOVE can only land in rest-start entries.
- Connector = straight line + pure rotation per leg; it fails near walls/people even when a curved path exists.
- Aging dominates MOVE near people: at v_p_max=1.5 the person disc grows ~1.5 m/s * connector time, so MOVE near people is refused (sound, conservative). Demo cases B/C vs B2 show it.
- Accel person model: aged centres are not modelled -> only the far test applies to aged beliefs.
- Static map assumed identical (map key); a map change drops the entries via `falsified("map", grid)`.
- Not yet measured: whether using the library in closed loop reduces regime-2 time / task time. Only offline structure + soundness are tested.

## Run
`nice -n 15 python -m pytest -q world_model/v2/house/checkers/tests/test_library.py`; demo: `python -m world_model.v2.house.checkers.library --episode-s 4` (CPU, ~70 s) -> `/data/local/ss5772/molmospaces/gatekeeper/library_demo/library_demo_seed0.json`.
