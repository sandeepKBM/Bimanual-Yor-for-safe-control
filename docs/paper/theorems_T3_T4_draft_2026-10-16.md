# T3 (finite completion with counted recoveries) and T4 (existence of a progress-preserving recovery) — draft, 2026-10-16

> **2026-10-26: partly superseded.** Current claims, scope (manipulation core; people safety is a borrowed component) and what is obsolete here: `docs/paper/claims_2026-10-26.md`.

Builds on `theorems_T1_T2_draft_2026-10-16.md`: same notation (x, b = (P, M, {H_i}), R_i,k, Adm = C1 ∧ C2, T(b), S1–S3, ⊑), A1–A7 and the
KEEP rule. New: A8, H0–H4, commit rule R_Φ. Scope: base carry (`house/gatekeeper`); T3 is skill-generic (carries to arm recoveries),
T4(b)–(e) are base-specific. Status table in §6. Code has no LOOK skill, no Φ logging, no progress-monotone commit rule (GAP-3…5). Nothing
was run except the arithmetic marked in §4.

## 0. Motivation (empirical, 10-16; one sweep each — motivation, not evidence for the theorems)

- Truth people, 20 seeds, 45 s: shield 20/20 (15.9 s); gatekeeper speed-only person model 0/20 (livelock); accel model 5/20 (31.7 s);
  0 contacts / 0 spills everywhere. Seed 0: a STANDING person ≈ 0.65 m from the footprint rejects every candidate.
- Tracker people (depth + conformal, seeds 0–1): gk_accel_trk 0/2, shield_trk 2/2. Conformal e_p = 0.42 m ⇒ keep-out 0.30 + 0.42 + 0.10 =
  0.82 m vs ≈ 0.65 m: uncertifiable. One person untracked ≈ 50 % of near steps (A1 violated).
- Min footprint–person-centre distance: shield 0.18 m, gk accel 0.24 m, both < PERSON_CLEAR_R + d_person = 0.40 m. For the gatekeeper this
  is S2-consistent only if it happened at rest; robot speed at that instant is not logged (§5).

Safety (T1) holds; progress fails. T3/T4 are about that gap.

## 1. Additions to the setting

- **Progress.** Route Γ (sim: operator out-route; autonomous: planner route), goal set G. V(p) = remaining arc length from the projection
  of p_xy on Γ to G; V = 0 exactly on G; L_V-Lipschitz in a tube around Γ (L_V = 1 on straight pieces). V̄(b) := max_{p∈P} V(p).
- **Prediction.** Pred(b, τ, t) = open-loop belief after executing τ for t without observations (pose box + drift; person sets by A2,
  velocity sets carried; parameters fixed). Monotone in ⊑; an exact semigroup for our box/disc models (r0 + v_p(s+t) and
  r0 + e_v(s+t) + ½a_p(s+t)² compose exactly when |v − v̂| ≤ e_v + a_p s is carried).
- **A8 (set-membership update).** b_k+1 ⊑ Pred(b_k, τ, Δ): observations only intersect. **GAP-5:** the code re-anchors on each observation
  (same root as T1's GAP-2); remedy 1 of the 10-16 memo implements A8.
- **Rest-point progress Φ (key construction).** τ_j = committed trajectory (KEEP rule), rest time t_rest:
  Φ_k := V̄(Pred(b_k, τ_j, t_rest − t_k)) — the progress kept even if every later candidate is rejected. Φ = 0 ⇔ certified completion.
- **Set size.** σ(b) ∈ ℝ^d_≥0: e_xy, e_θ, and per person near Γ the t = 0 radius PERSON_CLEAR_R + e_p,i (+ e_v,i, accel). b ⊑ b' ⇒ σ ≤ σ'.
- **Segments.** Motion is cut at rest instants; a segment is a maximal rest-to-rest motion or a rest interval, typed by checks at commit
  time: **PROG** ΔΦ ≤ −δ (or reaches Φ = 0); **MOVE** 0 < ΔΦ ≤ L_V r (certified back-off/detour; r bounds the rest-point displacement
  against Γ incl. pose growth); **LOOK** ΔΦ ≤ 0 with an observation whose post-condition on σ is certified (dwell, in-place rotation);
  **WAIT** rest interval; **NEUTRAL** other motion with ΔΦ ≤ 0. **Episode** = maximal run of non-PROG segments; m := max MOVEs per episode.
- **Commit rule R_Φ (required).** Inside a motion segment commit a new candidate only if Φ_new ≤ Φ_k, else KEEP (τ_j stays certified by T1);
  only the recovery protocol commits MOVEs, and they are counted. **GAP-3:** gate.py commits the first passing T_S in (1.0, 0.6, 0.3, 0.1);
  a T_S = 0.3 commit after a T_S = 1.0 commit pulls the rest point back — an UNCOUNTED Φ increase.

## 1b. C2-ready: the backup should end where progress can resume (added 2026-10-22)

Why: C2 certifies SAFE rest only. In the 10-22 sweep (24 scenarios), 9 failures of the segment executor ended at a safe rest pose from which
no PROG was admissible. The walker had stopped ~1.2 m away, facing the footprint's 0.64 m side, so the robot waited until it ran out of recoveries.
C2 cannot see this, because nothing in it looks past the rest instant.

- **Re-observed belief at rest.** For a candidate τ with rest time t_rest and rest pose p_rest, and a hypothesis η ∈ {walk, stop}:
  - b̃_η = b re-anchored at p_rest;
  - every person i at ĉ_i = c_i + v̂_i·t_rest, with the time-0 radius (age reset: the robot observes again when it stops) [STRUCK 10-24 (i)/(iv): ranking heuristic only, not usable in T3];
  - velocity v̂_i (walk) or 0 (stop);
  - everything else unchanged.

  This is a NOMINAL prediction (where we expect to see the person), not a set that contains the truth. It is used for progress only.
- **Definition.** Ready_η(τ, b) ⇔ ∃ m ∈ L(p_rest) with Adm(m, b̃_η) and V(p_rest) − V(end(m)) ≥ δ_min (or m reaches G). Here:
  - L(p) = the route-continuation segments from rest at p (speeds 0.3 / 0.2 / 0.12 m/s, T_S ∈ t_switch, current and zero lateral offset);
  - δ_min = gk.c2_ready_dmin = 0.05 m (ASSUMPTION).

  Score(τ) = #{η : Ready_η}. Depth 1: m itself needs only Adm = C1 ∧ C2, not Ready again.
- **Rule (ranking, never a hard row).** Among the candidates with Adm(τ, b), commit the first one with full score in try order. Try order is:
  current offset, then 0, then the configured offsets; T_S longest first. If none has full score, commit the best score. If nothing is
  admissible, behave exactly as before.

**Prop 1b.1 (safety neutrality).** Ranking only chooses among admitted candidates, so T1, Lemma 1 and T2 hold unchanged.

**Prop 1b.2 (conditional progress at the rest point).** Let τ be committed with Ready_η(τ, b), let the robot come to rest at p_rest (refusal or
end of the backup), and let the belief observed there satisfy b_obs ⊑ b̃_η. Then a PROG with δ ≥ δ_min is admissible at b_obs.
*Proof.* m is admitted at b̃_η; T2(a) gives Adm(m, b_obs). ∎
- **Caveat (exact containment).** b_obs ⊑ b̃_η needs the observed centre to be exactly the predicted one, because the radii are equal.
- **Robust version (not implemented).** Inflate b̃_η's position error by a tolerance ζ (and the velocity error likewise). Then any
  re-observation within ζ of the prediction is covered. ζ trades readiness for robustness: ζ = 0 today.

**Relation to H0 and T3/T4.**
- Ready is H0 checked ONLINE, at the rest pose the robot would actually reach, with the predicted people. H0 is the same question offline,
  with no people.
- When every committed candidate is ready and the hypotheses hold (Prop 1b.2), each rest is followed by an admissible PROG. The WAIT/MOVE
  part of an episode is then not needed, and T3's count holds with δ = δ_min and m = 0.
- "No admitted candidate is ready" is an EARLY warning of a T4 situation. It is seen while the robot is still moving, before the stuck rest.
- Prop 3.3 still applies: a person standing in a too-narrow corridor makes every candidate unready, and nothing can fix that.

**Interaction with R_Φ (GAP-3).** Ranking can prefer a shorter T_S, whose rest point lies behind the longer candidate's. Relative to the previously
committed rest point this can pull Φ back. Log it as a counted pull-back (n_ready_pullback, TODO) so Lemma 3.1's accounting stays honest.

**Code.** gatekeeper/ready.py (belief_at_rest, ready_score); tamp/segments.py next_certified_segment(ready=...) and lateral offsets
(gk.seg_offsets; holonomic vy tracking, heading along the path); flags `--c2-ready rank`, `--seg-offsets`. Default off (bit-identical).
**Status:** definition and Props 1b.1/1b.2 proved (one line each, from T1/T2(a)); the effect on the 10-22 failures is NOT yet measured.

## 2. Two lemmas

**Lemma 3.0 (recursive feasibility from admission).** Under T1's hypotheses, A8, and no pose-box growth at rest, an admissible action always
exists: KEEP while moving, τ_rest at rest.
*Proof.* T1's invariant: the robot runs a suffix of an admitted τ_j and is at rest within T̄_S + t_back_max (C2 rest row; 1.0 + 6.0 s). At
rest τ_rest has no moving samples, so C1 is the static row at one pose; C2 static passed for Pred(b_j) and b_k ⊑ Pred(b_j) (A8), so it
passes now (T2(a)); C2 passive holds at rest by definition. ∎
Enwerem 2610.07674 ASSUMES recursive feasibility (their Assumption 2); here it is derived. What is not free is PROGRESS feasibility —
isolated by H0–H3 and T4. Inherits T1's GAP-2.

**Lemma 3.1 (progress accounting).** Under A8 and R_Φ, Φ never increases except at MOVE commits (by ≤ L_V r). Hence, a posteriori, every
run has N_P ≤ 1 + (Φ_0 + L_V r N_M)/δ.
*Proof.* KEEP: Φ_k+1 = V̄(Pred(b_k+1, τ_j, ·)) ≤ V̄(Pred(Pred(b_k, τ_j, Δ), τ_j, ·)) = Φ_k (A8, Pred monotone + semigroup, V̄ monotone).
Non-MOVE commits: R_Φ. LOOK: xy box unchanged, V depends on xy only. Each non-final PROG lowers Φ by ≥ δ; Φ ≥ 0. ∎
A rejection never loses banked progress — the formal content of "progress-preserving". It also makes per-step replanning countable: one
commit from rest banks the whole planned segment.

## 3. Theorem T3 (finite completion with counted recoveries)

> **USER DECISIONS (these override the statement below; the restated T3 is waiting for approval).**
> - 2026-10-23: **H3 (fairness) STRUCK.** People need not ever clear the way. Prop 3.3 then shows no completion bound of the form
>   below exists, so T3 has to change form, not just lose a line.
> - 2026-10-24: **T3 may not assume the belief returns to its original expectation.** Struck wherever it is used:
>   (i) re-observing resets a person's set to its time-0 radius ρ_i0 / r0 (§1b age reset, T4(c) hop–observe repeat, No-escape remark);
>   (ii) a LOOK brings σ back to ≤ σ_L (H2, occlusion Remark, Lemma 3.2);
>   (iii) a tracked person's set does not grow while the robot waits (H1, Lemma 3.2);
>   (iv) people end up where the nominal prediction put them (§1b b̃_η; Prop 1b.2 needs b_obs ⊑ b̃_η).
>   The data already contradicts (i)/(ii): see the §5 H2 note (conformal radius not monotone in age or range; old tracks 0.45 m).
>   The set size after an observation is a MEASURED quantity in T3, not a hypothesis. Affected lines are tagged [STRUCK 10-24].
> - 2026-10-24: **No Shield is assumed** (see the T1 draft, §3). Falsification events have no Shield fallback; T3 counts them as measured
>   events, not as recoveries.

**Hypotheses** (besides A1 at commit times, A2–A8, KEEP, R_Φ):
- **H0 (static route certifiability).** ∃ σ*, D*, δ: at every closed-loop rest state with σ(b) ≤ σ* and every belief person centre ≥ D*
  from the route window W(p) (footprint swept by the next T̄_S of Γ plus braking), C(b) contains an admissible PROG. A property of
  map + library + checkers; checkable offline.
- **H1 (bounded growth, coverage).** [STRUCK 10-24 (iii): "tracked does not grow"] Unobserved components grow at most linearly: σ_c(Pred(b, τ, t)) ≤ σ_c(b) + γ_c t. Occluded person
  γ = v_p,max (from A2); pose box γ_x = drift while moving, 0 at rest; a continuously tracked person does not grow. Coverage: at rest,
  every person in W(p) ⊕ v_p,max(T̄_S + t_brake) is observed (fails today: unseen people are not in the belief, A1-open).
- **H2 (look efficacy / viewpoint reachability).** [STRUCK 10-24 (ii)] ∃ σ_L < σ*, m_L, T̄_L: from every closed-loop rest state, ≤ m_L consecutive admissible
  LOOKs give σ ≤ σ_L. Without a view-existence assumption a re-observe certificate is vacuous (prior_art Claim C). Deterministic only
  given detection; with dropout it becomes high-probability (GAP).
- **H3 (fairness / environment liveness).** [STRUCK 10-23] (a) Passive: every rest interval of length W̄ contains an instant (lasting ≥ Δ + latency) when no
  true person is within D* + σ* of W(p). (b) Reactive: (a) holds after ≤ m library MOVEs, each admissible when called (T4(d) gives a
  checkable sufficient condition). (a) ⇒ m = 0.
- **H4 (proposer persistence).** In a PROG segment, outside ≤ T_c := 2T_ramp + T̄_S + t_back_max of start-up and final coast, Φ falls at
  rate ≥ v_min. Shared autonomy: assumes the operator commands route progress (operator-idle time excluded).

**Protocol π_rec** (at rest, no PROG): LOOK while some binding σ_c > σ* (≤ m_L); else WAIT observing (≤ W̄); then MOVE (≤ m), repeat.

**Statement.** Under these hypotheses, with δ > m L_V r and Φ_0 = V̄(b_0), the loop satisfies S1–S3 for all t (T1) and reaches Φ = 0 with
- N_P ≤ N̄_P := ⌈Φ_0 / (δ − m L_V r)⌉ PROG segments (m = 1: the ⌈V0/(δ − L_V r)⌉ form);
- N_M ≤ m N̄_P, N_L ≤ (m+1) m_L N̄_P, N_W ≤ (m+1) N̄_P; i.e. N = N_P + R with R ≤ N̄_P (2m + 1 + (m+1) m_L) counted recoveries;
- T_done ≤ (Φ_0 + m L_V r N̄_P)/v_min + N̄_P (T_c + T_E), T_E := (m+1)(m_L T̄_L + W̄) + m T̄_M.

*Proof.* (i) Episodes e_0 … e_n−2 precede the (n−1)-th PROG, so ≤ m(n−1) MOVEs do. If an n-th PROG is needed, Φ > 0 after the (n−1)-th:
0 < Φ_0 − (n−1)δ + m(n−1)L_V r (Lemma 3.1) ⇒ n ≤ ⌈Φ_0/(δ − mL_V r)⌉. (ii) Every episode ends (Lemma 3.2), so while Φ > 0 a next PROG
occurs; with (i) the run ends at Φ = 0. (iii) PROG segments lower Φ by ≤ Φ_0 + L_V r N_M in total at rate ≥ v_min outside T_c (H4);
episodes last ≤ T_E. ∎

**Lemma 3.2 (episode ≤ T_E; sketch).** [STRUCK 10-23/10-24: uses H3 and resets (ii)/(iii)] Segments are rest-to-rest, so an episode starts at rest. H2: σ ≤ σ_L < σ* within m_L T̄_L. Waiting at
rest, tracked components and the pose box do not grow (H1). H3(a): within W̄ the true window is clear by D* + σ*; A1 puts every belief
centre ≥ D* from W(p); H0 gives a PROG. H3(b): repeat after each of ≤ m MOVEs. **GAP-T3-1:** a person who clears W(p) by leaving the field
of view becomes a phantom set growing at v_p,max that can re-block; H1's coverage clause excludes this, unimplemented.

**Remark (occlusion count).** [STRUCK 10-24 (ii)] If σ grows at ≤ γ while moving and a LOOK resets it to ≤ σ_L, growth-triggered LOOK episodes number
≤ 1 + γ T_mov/(σ* − σ_L). Occluded person: γ = 1.5 m/s, σ* − σ_L = 0.2 m ⇒ one LOOK per 0.13 s of motion near the occluder — finite but
useless unless the view geometry clears the phantom.

**Tightness.** 1-D, L_V = 1, PROG = exactly δ; an adversary allowed by H3(b) with m = 1 forces a MOVE of exactly r in every episode
(incl. e_0). Before the n-th PROG, Φ = V0 + nr − (n−1)δ; it completes iff V0 ≤ n(δ − r). So N_P = ⌈V0/(δ − r)⌉: attained exactly. With
r ≥ δ the same adversary keeps Φ ≥ V0 forever while all other hypotheses hold: δ > mL_V r cannot be dropped.

**Proposition 3.3 (no completion bound without fairness).** People fixed at c; 𝒮(c) := {p : S1 at p, dist(F(p), c_j) ≥ PERSON_CLEAR_R +
d_person − ε ∀j} (closed). A C¹ pose path satisfying S1–S2 that starts in 𝒮(c) stays in the path-component of 𝒮(c) containing p(0); one
starting outside never moves.
*Proof.* U := {t : p(t) ∉ 𝒮(c)} is open, and S2 forces ṗ = 0 on U. A component (a, b) of U with a > 0 has p ≡ p(a) ∈ 𝒮(c) (a ∉ U), a
contradiction; if 0 ∈ U then p ≡ p(0) on [0, b) and b < ∞ contradicts likewise. So U = ∅ (or no motion), and p([0,T]) is a path-connected
subset of 𝒮(c). ∎
**Corollary.** If that component misses G, no S1–S2-compliant controller ever completes, whatever its belief; so no bound of T3's form holds
under A1–A8 + H0–H2, and H3 is necessary. Instance: a person standing in a corridor every path to G must traverse, free width
w < 0.22 (torso) + 0.40 + 0.74 (min footprint extent) + 0.10 (d_wall) ≈ 1.46 m. At the tracker's e_p = 0.42 the certified passage needs
≈ 1.88 m: corridors of 1.46–1.88 m are passable for an omniscient S2 controller, blocked for any controller certified at that radius (the
price of uncertainty; T4(e)).

**New vs prior art.** Enwerem 2610.07674: E[T_r] ≤ (E V0 + B0)/(c α1(r)) — expectation, Bayesian finite-hypothesis belief, recursive
feasibility assumed, no state-set growth, no safety check over the belief. T3: (1) deterministic worst case over set-valued beliefs;
(2) feasibility derived (Lemma 3.0); (3) progress at the certified rest point, so rejections never lose progress (Lemma 3.1);
(4) occlusion growth and re-observation inside the count; (5) exact tightness and a spec-level impossibility making fairness necessary
(Prop 3.3). van Wijk 2610.17337: recurrence (gap ≤ T, full state), no count. Borquez/Peng/Bansal 2609.17430: hybrid reach-avoid, full
state, single reach, no switch count (10-15 read). OcclusionCBF 2609.06342 / SEAMLiS 2607.09959: sensing actions admitted for safety only.
The counting step is a standard ranking-function argument (no novelty claimed). Strongest attack — "H0 is Enwerem's Assumption 2 renamed":
H0 is static (map + library at σ ≤ σ*, no people) and checked offline; the dynamic part is isolated in H1–H3 and audited (§5).

## 4. Theorem T4 (existence of a progress-preserving recovery; forced retreat)

Setting: at rest at p (Lemma 3.0), belief b, no admissible PROG in C(b). A **progress-preserving recovery (PPR)** is an admissible segment
with ΔΦ ≤ 0 that strictly lowers the well-founded measure μ(b) = (⌈Φ/δ_q⌉, ⌈Σ_c σ_c/η⌉) ∈ ℕ² (lexicographic): a V-reducing hop/detour, or a
LOOK lowering σ by ≥ η. Notation: d_i := dist(c_i, F(p)); ρ_i0 := PERSON_CLEAR_R + e_p,i + d_person; sw_0 := first-sample sweep term of
the implemented person row (speed 0.5h·v_p = 0.0375 m; accel 0.5h(e_v + a_p h) = 0.0075 m; the row takes the max of the two margins).

**T4(a) LOOK certificate (sufficient, finite).** Let λ be a LOOK with Adm(λ, b), duration T_L, and sensor post-condition σ_c(b') ≤ σ_L,c(λ)
on the components it observes (calibrated radius at λ's view geometry). If (i) σ_c(b) − σ_L,c ≥ η for some observed c, and (ii) a finite
cover {Q_j} of the possible observed centres R_i(T_L) ⊕ σ_L exists such that each b_j (person set Q_j ⊕ σ_L, rest Pred(b, λ, T_L)) admits
some PROG in C(b_j) or is person-blocked (dist(Q_j, W(p)) < D*), then λ is a PPR, and afterwards either a PROG is admissible or the episode
is H3's.
*Proof.* Safe by T1; Φ unchanged (in place); σ drops by A8 + the post-condition; if the observed centre is in Q_j then b' ⊑ b_j, so every
PROG admitted at b_j is admitted at b' (T2(a)). ∎ Cost |cover| × |C| checker calls. The PROG may depend on the outcome (it is chosen after
observing); λ itself needs one non-anticipative backup (rest).
*H2 at today's sensor (data).* conformal.json cell3, updated tracks: the radius is not monotone in age or range — old tracks 0.45 / 0.42 /
0.20 m at r < 1.0 / 1.0–1.7 / 1.7–2.4 m; mid-age 0.18 / 0.14 / 0.37 m. A long-tracked standing person at close range keeps e_p ≈ 0.42 m;
neither waiting (age only grows) nor approaching certifies σ_L < 0.42. **Prediction:** the tracker freezes are not LOOK-resolvable with this
calibration. Caveat: the score is horizon-propagated e_T, not time-0 position error (README); E4 tests it.

**T4(b) Frozen radius (proved from the implemented rows).** If d_i ≤ ρ_i0 + sw_0 (+ v_p ℓ) for some tracked i, every candidate with any
motion fails C1-person at sample 0 (the current pose), for ANY library: only rest-type actions (WAIT, sensing without base motion) are
admissible — no hop, detour or retreat. Thresholds: 0.41 m with truth people, 0.83 m at e_p = 0.42. The spec itself (Prop 3.3) freezes every
controller only below 0.40 − ε; the difference e_p + sw_0 + ε is the checker's own freeze band.

**T4(c) Hop lemma (V-reducing PPR; sketch).** If all d_i > ρ_i0 + sw_0, let t̄ be the largest t with ρ_i0 + g_i(t) + sw(t) + a_r t²/4 ≤ d_i
∀i, where g = v_p t (speed) or min(v_p t, e_v t + ½a_p t²) (accel, v̂ = 0) and a_r ≤ plan_frac·a_cup_max (a_cup_max = 0.9·0.8·0.18·g =
1.27 m/s²). A straight rest-to-rest bang-bang hop of duration t̄ along Γ passes C1-person (worst case: straight at c_i), C2-passive (the
moving row at t̄ implies t_touch ≥ t̄ in both models), cup/hw (pure translation at a_r), wall/static if the static slack ≥ hop length, and
lowers Φ by δ_hop ≥ (a_r t̄²/4) cos ψ − L_V γ_x·(hop length). [STRUCK 10-24 (i):] Re-observing at rest resets tracked sets to ρ_i0, so hop–observe repeats and
T3 applies with δ = min δ_hop.
Back-of-envelope (NOT run; d ≈ 0.65 m, a_r = 1.27 m/s², h = 0.05): speed model t̄ ≈ 0.14 s, δ_hop ≈ 6 mm (≈ 2.5 cm/s with Δ = 0.1 s);
accel t̄ ≈ 0.34–0.38 s, δ_hop ≈ 4 cm (≈ 9 cm/s); tracker e_p = 0.42: frozen (T4(b)). The shortest implemented candidate (T_S = 0.1 s +
cup-safe backup) moves longer than 0.14 s. **Prediction:** the speed-model livelock is LIBRARY-blocked, not frozen; a hop primitive
unfreezes it at a few cm/s.
*No-escape remark.* For translation, d grows at ≤ |v_xy| ≤ 1.0 m/s < v_p,max = 1.5 m/s, and a_r ≈ 1.27 < a_p = 2, so ρ_i(t) − d_i(t)
increases along every motion: every admissible motion near a person is a bounded hop, and only re-observation (which resets the set [STRUCK 10-24 (i)])
makes repeated motion possible — near people MOVE and LOOK are inseparable. Rotation (corner speed can exceed 1.5 m/s): open.

**T4(d) Forced retreat (characterization).** Let 𝒮_in(b) be Prop 3.3's 𝒮 with each keep-out shrunk to
disc(c_i, PERSON_CLEAR_R + d_person − ε − e_p,i) — inside the true keep-out for every world in b. SPEC_in(b): the path-component of 𝒮_in(b)
containing P has inf V ≥ V̄(b) − δ_0. If SPEC_in(b) holds and the people stand still, then (1) for every world consistent with b no
S1–S2-compliant controller lowers V by ≥ δ_0 (Prop 3.3; every true component lies inside an inner one); (2) no LOOK removes the obstruction
(it holds for every world in b); (3) so every recovery that can lead to progress is WAIT (H3a) or a V-INCREASING MOVE (H3b): the retreat is
forced by the spec, not by checker conservatism. Precedent for this kind of impossibility: Restorative Control 2610.09442 (abstract level
per the 10-15 cross-check; not re-read).
*Retreat certificate.* Time-reverse the executed rest-to-rest profile since the last rest. The cup load is invariant: v_b → −v_b, ω → −ω
leave v̇_b, α, ω×v_b and ω×(ω×r) unchanged, so |a_cup| matches at mirrored times; hw rows are symmetric; wall/static rows pass if the stored
margins exceed the pose shift + box growth since then (EDT 1-Lipschitz, T2(a)). Only person/passive rows need a fresh check. Result: a MOVE
with r = path length + growth.

**T4(e) Gap (honest).** If e_p,i ≥ PERSON_CLEAR_R + d_person − ε (tracker: 0.42 > 0.40) the inner disc is empty and SPEC_in is never
certifiable: at that radius a fundamental block is indistinguishable from an uncertainty block. In general a band ≈ 2e_p,i wide per person
lies between SPEC_in and the outer tests (T4(a) at σ_L, T4(c)) where T4 certifies neither a PPR nor a forced retreat. Sufficient, not
complete.

**Arms (thesis).** T3 is skill-generic. Arm analogues of T4(a)–(d) (null-space/regrasp PPR) need fibre-connectivity and one-hand-hold
certificates — not done. On the current 6-DoF PiPER pair the closed-chain null space is 6 + 6 − 12 = 0: every arm-level recovery changes
the grasp or object pose.

## 5. Assumption → sim measurement → falsifier

| Item | Measured by (exists / ADD) | Falsified if |
|---|---|---|
| A1 | n_belief_miss, n_untracked_near, max_miss_excess_m (exist) | > 0 at commit steps (tracker seeds 0–1: ≈ 50 % untracked near ⇒ T1/T3 do not apply there) |
| A8 | P_k+1 ⊆ Pred(P_k), H_i,k+1 ⊆ Pred(H_i,k): n_A8_viol (ADD) | > 0 with the set-membership filter on |
| R_Φ / Lemma 3.1 | Φ_k trace + segment types (ADD); n_keep_backup (exists) | Φ rises at a non-MOVE step |
| H0 | offline route sweep: rest poses on Γ at σ = σ*, no people; fraction with an admissible PROG (ADD) | < 1 |
| H1 | per-component dσ/dt vs γ (stale_i, since_upd exist); pose-box growth vs drift | rate > γ; a person in W(p) unobserved at rest |
| H2 | n_look, attempts/episode, σ before/after (ADD); conformal cells (exist) | attempts > m_L or σ_after > σ_L |
| H3 | truth-only audit: blocked-interval lengths (window not clear by D* + σ*) (ADD) | > W̄ (a scripted standing person violates it by construction) |
| H4 | operator idle fraction; Φ rate in PROG segments (ADD) | rate < v_min outside T_c |
| T3 | N_P, N_M, N_L, N_W, episode lengths vs N̄_P, T_E (ADD); task_done, task_time_s (exist) | a run with all hypothesis counters clean exceeds the bound |
| S2 / T1 | robot speed at the min-person-distance instant (ADD) | v ≠ 0 at distance < 0.40 m |
| T4 | per frozen step: class F (T4b) / L (hop exists) / U (oracle PROG at σ_L or σ = 0) / S (SPEC_in) / gap (ADD) | a certified LOOK leaves σ > σ_L; a predicted-admissible hop is rejected; an S2-compliant run crosses a SPEC_in block |

Falsifying experiments:
- **E1 (tightness/livelock):** 1-D corridor, scripted person who yields only after a back-off of r; sweep r/δ, compare N_P with
  ⌈V0/(δ − r)⌉; r ≥ δ must never complete.
- **E2 (fairness necessary):** a person standing forever in a 1.2 m corridor; the gatekeeper must stall at rest; any controller that
  finishes must violate S2 while moving (tests the shield too).
- **E3 (hop):** add a rest-to-rest hop of duration ≤ t̄; seed 0 / speed model predicted to finish; rerun the 20-seed truth sweep.
- **E4 (LOOK floor):** refit the conformal score as time-0 position error per (age, range) cell; if the close-range radius stays ≥ 0.25 m,
  tracker freezes are U at the sensor floor (need A8 or another sensor model).
- **E5 (classification):** label every frozen step of sweep_truth and *_trk as F / L / U / S / gap.

## 6. Proof status

| Claim | Status | Depends on / gap |
|---|---|---|
| Lemma 3.0 feasibility from admission | proved | T1; GAP-2 in code |
| Lemma 3.1 Φ accounting | proved | A8 (code violates, GAP-5); R_Φ (missing, GAP-3); exact Pred semigroup (holds for our models) |
| T3 count N_P ≤ ⌈Φ_0/(δ − mL_V r)⌉ | proved (arithmetic) | given δ, m |
| Lemma 3.2 episode ≤ T_E; T3 time bound | sketch | GAP-T3-1 (phantom re-blocking / FOV coverage); H4 |
| Occlusion LOOK count | sketch | H1 rate; useless near occluders |
| Tightness; necessity of δ > mL_V r | proved | 1-D example |
| Prop 3.3 no bound without fairness | proved | standing people; C¹ pose path |
| T4(a) LOOK certificate | proved (given a finite cover) | H2 post-condition; current calibration fails it at close range |
| T4(b) frozen radius | proved | implemented C1 row at sample 0 |
| T4(c) hop lemma | sketch; numbers are estimates | not run; no hop primitive in the library (GAP-4) |
| T4(d) forced retreat via SPEC_in | proved for standing people | moving people: conjecture |
| T4(d) time-reversal retreat | cup/hw invariance proved | needs a rest-to-rest reversal; stored margins |
| T4(e) completeness | NOT claimed | ≈ 2e_p gap band exists |
| No-escape remark | proved for translation | rotation open |

Open before claiming: GAP-2/5 (set-membership update + KEEP monitor), GAP-3 (R_Φ), GAP-4 (LOOK/hop primitives, Φ and segment logging),
unseen people as phantom sets (A1, H1 coverage), a high-probability version of H2/A1 (conformal coverage and detection dropout are
probabilistic), and the arm instance of T4.
