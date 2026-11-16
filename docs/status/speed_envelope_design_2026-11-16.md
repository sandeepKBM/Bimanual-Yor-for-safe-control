# Speed envelope design (2026-11-16, DRAFT, not built)

User request (11-16): the VLM plans the high level. A fast modulator tunes the speed at control rate, inside a bound that the checker has
made legal. Order set by the user: place fix first (done separately: "a stop in the last T s of a move lets the move finish"),
then this design, starting with the monotone question. This note covers what the checker would certify, what must be measured first,
and the open values.

## 1. What changes

Today the checker admits **one** timing per step: TOPP-RA at one acceleration `a` (admission picks the highest level that passes).
In the new scheme the checker admits a **set of timings along the same path**. The path is fixed (same joint samples). Only the speed
along it may change. At each path sample `s_i` the certificate gives:

- a top path speed `sdot_max(s_i)`;
- an allowed range of path acceleration `[sddot_min(s_i), sddot_max(s_i)]`;
- a limit on how fast the modulator may change acceleration (a jerk bound; ASSUMED, to be chosen).

Rule 1 must hold for **every** timing in that set, from every branch point. The modulator may pick any timing inside the set.

## 2. Rows and how each one depends on speed and acceleration

| Row | Depends on | Shape for one θ | Check needed |
|---|---|---|---|
| torque / payload (`payload_envelope`) | `sddot` linearly, `sdot^2` linearly (fixed path: M q' sddot + (M q'' + C) sdot^2 + g) | convex region in (`sdot^2`, `sddot`) per sample | corners of that region; θ corners only if torque is affine in θ (true for mass at a fixed centre of mass; CHECK for the box contents model) |
| open_content (object acceleration) | object acceleration = p' sddot + p'' sdot^2 | linear in (`sdot^2`, `sddot`) | same as torque |
| joint speed (`motion_rows`) | `sdot` | `sdot * |q'(s)| <= qd_max` | the speed cap |
| tilt (`tilt_row`, measured table) | measured only at levels a = 0.1 / 0.25 / 0.5 on the DEFAULT plan | not a model: a table | envelope must stay inside a measured level; other plans: MISSING (as today) |
| collision swept term | speed | grows with speed (to confirm) | at `sdot_max` if monotone |
| Rule 1 stop rows | speed at the branch (`v`), stop time T | Hermite stop decel `v / T`, overshoot `0.5 v T`; finish = plan decel | at `sdot_max` if monotone |
| Rule 3 cover | the bin | a smaller bin can only widen the envelope | stored envelope per bin = a cover-table entry |

## 3. The monotone question (step 1, measure before building)

The envelope check is cheap only if each row gets **worse** (or no better) as speed and acceleration rise. Then checking the top edge
of the envelope covers everything under it. Status:

- torque / payload utilisation vs `a`: MEASURED non-decreasing on the default box path, every box cell, a = 0..1.0 step 0.025
  (`nero/results/z3_stage0_2026-11-16/`). ASSUMED for other paths and between grid points. This is for a single scalar `a`
  (TOPP-RA re-timed), not for a free speed profile.
- Rule 1 (stop and recovery rows) vs `a`: NOT measured. Possible now that `rule1_contents.run_cell(..., a=...)` uses the plan's `a`.
- collision swept term vs speed: NOT measured.
- tilt vs `a`: three measured levels on the default plan only.

Step-1 measurement (to run): box cells x a grid, per row family, pass/fail and margin; report any a where a row passes at a higher
`a` but fails at a lower one (a counterexample = no top-edge shortcut for that row).

## 4. The modulator

- Runs every control step (5 ms) inside the tracker (the FL-MPC path-speed variable is the natural place).
- Picks the next path acceleration and **clips** it to the certified range at the current sample (a lookup and a min/max).
- Its own logic need not be trusted: any choice inside the range is legal. Candidates: track nominal time; slow down on a
  watcher deviation; a STELA-style duration adjustment.
- The VLM does not set these numbers. It may state an intent ("place gently"), which maps to a narrower envelope.

## 5. Open values (ASSUMED until set or measured)

- Stop time T (`STOP_T_S`, 0.3 s): sweep added to `assumed_sweep` (user 11-16: yes).
- Jerk bound of the modulator.
- Envelope discretisation (per path sample or per segment).
- Whether torque is affine in all box contents parameters (mass only, or also centre of mass).

## 6. Not decided

- Whether the envelope replaces the single admitted `a` or sits on top of it (the admitted `a` as the envelope's top edge is the
  smallest change).
- How Rule 4 targets interact with an envelope (the transfer move would carry its own envelope).
