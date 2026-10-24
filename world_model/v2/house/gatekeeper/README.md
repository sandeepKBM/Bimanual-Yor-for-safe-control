# gatekeeper -- Checkers 1+2 for the shared-autonomy base carry (SIMULATION ONLY)
Gatekeeper pattern (Agrawal & Panagou, T-RO 2024) over set-valued beliefs. Drop-in for the Shield: `run.run_episode(..., filt=Gatekeeper())`.
CLI: `python -m world_model.v2.house.gatekeeper.run_gk --seed 0 --max-s 30 [--compare] [--v-p-max X] [--person-model speed|accel] [--candidates hold|set] [--ab]`. Tests: `pytest world_model/v2/house/gatekeeper/tests`.

## Loop (every K=5 ticks = 0.1 s; between steps the committed twist profile is tracked)
1. for T_S in (1.0, 0.6, 0.3, 0.1): plan nominal (u_op held T_S, cup-safe ramp) + backup (cup-safe stop); commit the first that passes Checker 1 AND 2.
2. none passes -> keep executing the committed trajectory (re-based at now) if it is still certified under the CURRENT belief (n_keep_backup).
3. not certified (model falsified) -> n_falsified, REGIME 2: internal teleop Shield (fallback="escape") issues the command until a candidate passes again.

## Checker 1 (worst case over the belief, exact inflation)
- wall:   EDT(p_k + R(th_k) q_j) - e_xy - |q_j| e_th - fp_step/2 - sl_k >= d_wall(p_k)   for all samples k, boundary points q_j
  (|R(th+d)q - R(th)q| <= |q||d|, EDT is 1-Lipschitz; sl_k = 0.5 h max(|v_xy| + |wz| max|q|) over neighbouring samples = inter-sample sweep)
- person: dist(c_i, Rect(p_k)) - (PERSON_CLEAR_R + e_p + v_p (t_k + latency)) - d_person - 0.5 h (v_pt + v_p) >= 0 while moving (k < i_rest)
- cup:    |a_cup(t)| <= a_cup_max = cup_margin * min(kappa mu_lo, tan theta_spill) * g, a_cup = a_b + alpha x r + w x (v_b + w x r) (segment start AND end)
- hw:     |a_i| <= A_HW_i, |v_i| <= V_STICK_i
## Checker 2 (backup ends in T(b))
- rest: v reaches 0 within t_back_max;  static: min_j EDT(p_rest + R q_j) - e_xy - |q_j| e_th - fp_step/2 >= d_wall(p_rest)
- passive person: t_touch - t_rest >= 0, t_touch = (dist(c_i, Rect(p_rest)) - r0)/v_p - latency (stopped before any reachable disc can touch);
  a trajectory at rest from t=0 is passive-safe by definition (a person walking up to a stopped robot does NOT falsify it).
Margins are returned in their own SI units (geometry m, passive s, cup/hw m/s^2); ok iff all >= 0.
Monotone for a FIXED trajectory (property-tested): enlarging e_xy, e_th, e_p, v_p, latency, d_wall, d_person or lowering mu_lo never turns reject into admit.

## Planner
Per 0.05 s step: a_des = (target - v)/h, scaled into the A_HW box (ray kept), then largest s with |a_cup(v, s a_des)| <= plan_frac*a_cup_max (quadratic in s),
re-verified at the segment end. Held twists are capped so the steady centripetal cup term <= steady_frac * limit (leaves braking headroom). Backup = same
rule with target 0 (spin decelerated too). If the start state is already outside the cup set it brakes at the hw box and Checker 1 reports `cup`.

## ASSUMPTIONS (named in GKConfig; the project constants A_HW, V_STICK, ShieldCfg mu_lo/kappa/theta_spill/d_wall/d_person/loc_margin, PERSON_CLEAR_R, FP are reused)
v_p_max=1.5 m/s (max person speed), e_p=0 (people source = truth), latency_s=0, cup_margin=0.9, k_pose=1 (pose box = k*sim.loc.sigma_xy + loc_margin; 0 for truth pose),
th_over_xy=0.15 rad/m (e_th from e_xy), plan_frac=0.97, steady_frac=0.6, falsify_tol=0.01 m, fp_step=0.03, h=0.05, T_S list, K=5.

## OPT-IN A -- velocity-aware person reachable set  (`GKConfig.person_model = "speed"` (default, bit-identical) | `"accel"`)
Per person: observed centre c (error e_p inside r0 = PERSON_CLEAR_R + e_p), velocity estimate v_hat (`sim.people_state` velocity = TRUE velocity of the truth
source; `vel_source="fd"` = finite difference over the previous gk step, e_v inflated by a_p dt, first step v_hat = 0 / e_v = v_p_max), and
ASSUMPTIONS e_v = 0.2 m/s (|v_true(0) - v_hat| <= e_v), a_p = 2.0 m/s^2 (|accel| <= a_p), v_p_max = 1.5 m/s (|v| <= v_p_max). With t_e = t + latency:
    D_speed(t) = disc( c ,            r0 + v_p t_e )                              (original)
    D_accel(t) = disc( c + v_hat t_e, r0 + e_v t_e + a_p t_e^2 / 2 )              (|v(s) - v_hat| <= e_v + a_p s, integrate)
The person is in the INTERSECTION D_speed ∩ D_accel, so the robot is safe if it is clear of EITHER disc => margin = max(margin_speed, margin_accel)
(never worse than the speed model). Checker 1 (moving samples): margin_accel = dist(c + v_hat t_e, Rect(p_k)) - (r0 + e_v t_e + a_p t_e^2/2) - d_person
- 0.5 h (vfar_k + |v_hat| + e_v + a_p (t_e + h)). Checker 2 passive: touching needs both discs => t_touch = max(t_speed, t_accel); t_accel = positive root of
a_p t^2/2 + (e_v + |v_hat|) t = dist(c, Rect(p_rest)) - r0, i.e. t = (-b + sqrt(b^2 + 2 a_p x))/a_p (b = e_v+|v_hat| from the 1-Lipschitz rect distance; sound lower bound).
A standing person (v_hat = 0): D_accel is the smaller disc for t < t* = 2 (v_p_max - e_v)/a_p (= 1.3 s; 1.5 s for e_v = 0).
Monotone (property-tested) in e_v, a_p, v_p_max, e_p, d_person (and in latency when v_hat = 0: a non-zero v_hat moves the centre, so an OLDER observation of a
person walking away can legitimately give a larger margin -- the speed disc stays monotone in latency).

## OPT-IN B -- certified candidate set  (`GKConfig.candidates = "hold"` (default, bit-identical) | `"set"`)
Each step: hold-u_op over T_S in t_switch first (deviation 0 => commit, stop; nothing else is spent). Only if all fail, candidate twists u_c (body frame, clipped to V_STICK):
scaled s u_op (s = .75/.5/.25); heading of (vx,vy) rotated +/-20, +/-40 deg; lateral sidestep u_op + (0, +/-0.5 V_STICK_y, 0); pure rotation (0,0,+/-0.5 V_STICK_wz);
back-off 0.4 V_STICK along the unit body-frame direction away from the binding hazard (reason `person`/`passive_person`: nearest person -> robot; `wall`: +grad EDT).
Each = u_c held T_S in cand_t_switch (0.6, 0.1) then the cup-safe backup (`plan_candidate`), certified by Checkers 1 AND 2 on the same belief. Candidates are ordered by
deviation  dev = || (u_c - u_op) / V_STICK ||_2 ; the first certified one wins, equal-deviation ties go to the larger T_S. At most `cand_max_checks` = 24 (candidate,T_S)
plan+check attempts per step (optional wall-clock cap `cand_time_budget_ms`, off by default for determinism). No certified candidate -> keep-backup / regime 2 as before.
Counters: n_deviate (<= n_commit), mean_deviation (over committed gk steps, 0 for hold), mean_deviation_when_deviating, n_cand_checks, cand_checks_max_per_step.
`mean_deviation` is on the requested twist before the cup-safe ramp; the episode-level comparable to the Shield is `intervention_mean_norm` of run.summarize.

## OPT-IN C -- tracker people source  (`GKConfig.people_source = "truth"` (default, bit-identical) | `"tracker"`; + `perception`, `conformal`, `k_sigma`=2, `seed`)
The gatekeeper owns a `people_track.PerceptionStack` (built like `people_track.source.make_shield`; stepped once per sim tick in `filter()` exactly as `PerceivedShield.filter` does) and
builds the belief from the TRACKS (`sim.people_state` = truth is then used ONLY by the audit). Per track i (`proxy.track_info`):
c_i = track mean xy; v_hat_i = track mean velocity (accel model); e_p_i = `ConformalMargin.radius(since_upd, age, range)` if `conformal` is given, else k_sigma sqrt(lam_max(P_pp));
e_v_i = k_sigma sqrt(lam_max(P_vv)); stale_i = time since the track's last measurement update. In the checkers e_p / e_v / latency are per person: e_p + e_p_i, e_v + e_v_i (global e_v = 0 in
tracker mode), latency_s + stale_i for the radius growth of THAT person (speed AND accel disc); the accel-disc centre c + v_hat t is not advanced by stale_i (c is already predicted to now).
Regime 2 uses a `PerceivedShield` (same stack) on the tracker proxy -> no truth in any decision. CPU-only conditions (ray casting by `mj_multiRay`, no GL/GPU): `depth` (default;
DepthDetector), `seg`, `param:SIG:LAT:DROP`; `depth_perc`/`neural`/`fusion`/`full` (est pose / GPU server) are rejected. Conformal json used by sweep.py: `/data/local/ss5772/molmospaces/percept/people/calib_out/conformal.json`
(people_track.calib fit, depth detector, h5 carry, alpha 0.05). NOTE its score is e_T = |dp + 0.5 s dv| (horizon-propagated), not the pure time-0 position error |dp| -> using it as e_p is an approximation (audit measures it).
ASSUMPTION AUDIT (metrics only, every gk step; counters in `counters()` / sweep summary): `n_belief_miss` = true person matched (nearest track within `audit_match_r`=1 m) but |true - c| > e_p (belief set at t=0 does
not contain it); `n_untracked_near` = true person within `audit_near_r`=3 m of the footprint with no track within 1 m (unseen/occluded/dropped); also `n_audit_person_steps`, `n_audit_matched`, `max_miss_excess_m`.
sweep.py configs `shield_brake_trk`, `shield_escape_trk` (PerceivedShield via run.py's `--people tracker` path), `gk_accel_hold_trk`; `--configs` default is unchanged (truth configs only).

## OPT-IN D -- T1 theory fixes (defaults bit-identical; code in monitor.py)
- `keep_rule="monitor"` (GAP-2): no candidate passes -> keep tau_j WITHOUT re-certifying it under the current belief; regime 2 (n_falsified, reason in `falsify_reasons`) only if the falsification monitor fires:
  pose: |ctrl_pose - pose_pred_j(t - t_j)|_xy > e_xy_j + e_xy_now + falsify_tol (or heading, e_th); people (only while the kept profile still moves): each current observation disc D(c, e_p + v_p latency) must
  meet ALL predicted discs (speed; accel if used) of some commit-time person after dt = t - t_j; an unmatched person (new track / left its set) is a falsification iff
  min_k dist(c, Rect(p_k)) < PERSON_CLEAR_R + R + d_person + v_p_max * T_move (T_move = remaining moving time). No commit yet / after regime 2: the old re-certify path.
- `belief_filter="intersect"`: per person, previous disc predicted (speed: r + v_p dt; accel: centre + v_hat dt, r + e_v dt + a_p dt^2/2, enclosing disc of the two) intersected with the observation disc via the smallest enclosing
  disc of the lens (closed form, `monitor.disc_enclose`); disjoint -> `n_obs_inconsistent`, reset to the observation. Filters only the track's own e_p (global gk.e_p stays additive). Truth source: no-op.
  CAVEAT: with a 95 % conformal e_p, intersecting many sets lowers coverage (union bound); `n_belief_miss` measures it.
- `cup_chord=True` (GAP-1): Checker 1 cup row adds (h^2/8)(2|alpha||a_b| + 2 alpha^2 |r|) to every segment's start/end |a_cup|.
sweep.py configs: gk_accel_hold_mon, gk_accel_hold_trk_mon, gk_accel_hold_trk_mon_int (summary: n_obs_inconsistent, mean_raw_radius, mean_filtered_radius, falsify_reasons).

## NOT covered yet
- Occluded / unseen people: with people_source="tracker" a person with NO confirmed track is NOT in the belief (open assumption; `n_untracked_near` measures it; an appearing person is only caught as falsification). Unseen-space speed cap of PerceivedShield is not used by the gatekeeper (only in regime 2 via filter_base it is not applied).
- Arm motion / live footprint changes during the carry (fixed GM.FP unless cfg.footprint="live" via the internal Shield; tray tilt not in a_cup_max).
- Recovery library: no escape from inside a violated buffer except regime 2 (the Shield); a stopped robot near people is frozen by design (see below).
- IR veto / cup-slip monitor not used as a falsifier; doorway buffers (d_wall_door) ignored; no operator-intent model (u_op held constant).
- v_p_max=1.5 makes the reachable disc outgrow the robot (0.9 m/s): any motion within ~2 m of a person is rejected -> conservative, task may stall.
- Pose-estimation error beyond the box and open-loop tracking drift are only caught by falsify_tol.

## Smoke A/B (seed 0, h5, carry, people on, 30 s; /data/local/ss5772/molmospaces/gatekeeper/smoke_AB_seed0.json)
speed+hold (= current): not done (livelock, goal 2.63 m); accel+hold: done 20.5 s; speed+set: not done (goal 2.21 m, 188 deviations); accel+set: done 22.6 s; shield: done 9.0 s.
No spill / wall / person contact in any config. Min person distance (to the centre): shield 0.19, accel+hold 0.30, accel+set 0.27 m. Step time: set mode mean 8-14 ms, max 30-34 ms.
