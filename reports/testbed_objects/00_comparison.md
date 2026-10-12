# Testbed objects: side-by-side comparison (2026-10-05)

Four Sonnet agents ran the same template on the four shortlisted objects. Each built a small object-only MuJoCo rig
on the bead_env design and measured feasibility inside the pipeline's action domain. Per-object reports are in this
folder. The owner checked every number below against its `feasibility.json` and re-ran the lead object (b) on fresh
seeds.

## Table

| | a: loose items (dish) | b: tray + ball | c: block tower | d: opaque fill |
|---|---|---|---|---|
| Verdict | GO-with-changes (weak) | **GO-with-changes** | GO-with-changes (probing only) | NO-GO as main; GO as conservative case |
| Nominal failure rate | 50% (replay) | 44% (synthetic); owner re-run 45.5% | 48% (replay, tilt x0.35) | 41% (replay, 5 fills) |
| Fails inside the action domain | yes (~0.8 rad) | yes (p99 tilt 0.39 rad) | yes (~0.4 rad) | 72% of spills below 1.5 rad |
| Savable, but not by hold-still | 10.5% (42% within 20 ticks) | 57% | 45% | 84/84 (no doomed states) |
| Lead time, first-unsavable to failure | 3.5 ticks | ~2 ticks at full authority | 5 ticks | ~0-4 ticks |
| Physics + observed state predicts failure | AUC 0.93 | 97% of labels | 96% of branches | pendulum is a safe bound but marks 0/84 savable |
| Hidden parameter that matters | none (<=3% flips) | **rolling resistance: 35-50% flips** | friction: 22% of outcomes | fill: only mass, ~2 bins |
| Safe probe identifies it | nothing to learn | **yes, ~5 sigma (0.12 rad tilt)** | only at low friction | no |
| Observability (if the camera sees it) | full | full + 1 hidden parameter | partial | unobservable |
| Sim cost | 0.2 s per 2 s branch (N=10) | 29 rollouts/s/core | 60 branches/s/core | 0.6-2.3 s per branch |

## What holds across all four

1. **Physics plus observed state predicts failure well for every visible object** (93-97%). This repeats the cup
   result. On these objects a learned world model will not beat physics as a plain failure predictor. The model needs
   a different job: to infer hidden parameters from motion history, or to work when the state is not directly seen.
2. **Only object b has a hidden parameter that both matters and can be found safely.** Rolling resistance flips
   35-50% of outcomes, and a gentle 0.12 rad tilt identifies it. This fits north star 2: certified-safe probes that
   shrink uncertainty online.
3. **The camera is an open hardware question.** Every result assumes the ZED sees the object. The sim mount is a
   proposed torso mount, but the real ZED is a floor-facing nav cam. If the robot cannot see the tray, the
   observability classes above change.
4. **The action set is too strong for the real arm.** The 85-action set (5 m/s^2) exceeds what the Piper can do
   under its +-8 Nm feedforward cap. At realistic authority, lead times and savable sets shrink. A Piper-feasible
   action set is needed before any gate.
5. **The cup carries do not transfer.** The spill_v2 trajectories tilt too far for these objects. The agents used
   scaled replays or synthetic carries tuned into the 20-60% failure band. The final carry distribution must be
   fixed and justified before data collection.
6. **G1 is untested for every object.** No agent trained a world model. The definition of "certified-savable"
   (do hold-still saves count?) also needs one fixed reading.

## Recommendation

- **Main testbed: b (tray + ball).** Give the world model a real job: carry a latent belief over rolling
  resistance, and test it against physics plus an explicit rolling-resistance estimator. Certified-safe tilt probes
  identify the parameter online.
- **Secondary: c (block tower)** for friction probing.
- **Conservative-system case: d (opaque fill),** certified against the worst-case fill.
- **Drop a,** or keep it as a weak backup.

**Order of work:**
1. Settle the camera question on hardware.
2. Build a Piper-feasible action set.
3. Run b phase 1: a dataset with randomized rolling resistance, a world model, the nominal latent HJ value and the G1
   gate.
