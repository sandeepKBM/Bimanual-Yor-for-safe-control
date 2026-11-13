# SafeTeleop Beta: episode lifecycle + pick guidance (protocol and rules)

Scope: Assignment 5 essentials. Simulation only. All additions are opt-in or additive. With `--episode off` and a task other
than `pick`, the server's physics and commands are the same as the Alpha branch head. The `--episode on` default adds
telemetry, plus one behaviour: a disarm when an episode ends.

## 1. Episode lifecycle (`teleop/web/episode.py`, pure logic, no MuJoCo)

States: `IDLE -> ACTIVE -> OUTCOME -> (reset) -> IDLE`.

| From | Event | To | Effect |
|---|---|---|---|
| IDLE | operator arms (hub `enabled` goes False -> True) | ACTIVE | start the clock (sim time) and the metric accumulators |
| ACTIVE | success predicate true | OUTCOME | result `success`; the server disarms the hub (not latched) |
| ACTIVE | failure predicate true | OUTCOME | result `fail`; disarm |
| ACTIVE | operator `end` event (key `K`) | OUTCOME | result `ended` (reason "operator"); disarm |
| ACTIVE | `--episode-timeout S` > 0 elapsed | OUTCOME | result `fail` (reason "timeout") |
| any | `reset` (Backspace; existing rebuild) | IDLE | metrics cleared; the server disarms the hub (not latched), so the new episode starts on the next Enter / Start |

The sim keeps ticking in every state; motion needs arming as before. In OUTCOME, arming (Enter / Start) is ignored by
the episode: the hub may arm again, but the server keeps every motion command zero until a reset.

Predicates (evaluated once per tick from the same values the telemetry uses):
- all tasks, failure: cup `LOST`, or `beads_out > 0` (spill).
- `carry`, success: distance from the base to waypoint `cafe_counter_pick` <= `GOAL_R`, base speed < `STOP_V`, cup ok.
- `coffee`, success: with `--grasp-mode friction`, the assist reports a successful place (`assist.place is True`).
  Otherwise there is no automatic success: the operator ends with `K`.
- `pick`, success: pick status `holding` or `done`. A failed pick is not terminal: the operator can select again.

Constants (ASSUMED, user-approved 2026-11-12): `GOAL_R = 0.30 m`, `STOP_V = 0.05 m/s`.

Outcome metrics (`episode.outcome.metrics`): `duration_s` (sim), `wall_s`, `interv_frac` (fraction of ACTIVE ticks with
intervention_norm > 0.02), `min_wall_clr`, `min_person_clr`, `beads_out`, `n_estop`, `n_shield_fallback` (ticks labelled
brake_fallback / escape_fallback), `max_cup_tilt`, `dist_to_goal` (carry / coffee).

Telemetry key `episode`: `{state, t_active, goal, outcome: null | {result, reason, metrics}}`.

## 2. Pick guidance (`--task pick`; `teleop/web/pick_link.py`)

Server flags: `--task pick` (new choice), `--pick-props default` (house/pick/props.py set name), `--pick-planner rule|vlm|vlm+veto`
(default `rule`), `--pick-backend mock|auto|live` (default `mock`; `live` needs the VLM service and is never used in tests).
With `--task pick` the server builds `TeleopSim(task="pick", pick_props=..)` and one `PickSession`. The mock backend gets
the same `mock_labels` dict as `house/pick/smoke.py` (a documented privileged shortcut for offline use).

Threading rule: WebSocket handlers only enqueue requests. The sim thread drains the queue between ticks and calls
`PickSession.propose(sim, blocking=False)`, `request_with_snapshot(sim, id)` and `cancel()`. It calls `ps.step(sim)` once
per tick, before `sim.tick()`.

Client -> server messages:
- `{"type":"pick_propose"}`: refresh proposals (worker thread renders the head camera; non-blocking).
- `{"type":"pick_select","id":N}`: start the pick of proposal N. Accepted only when ready, episode ACTIVE (or `--episode off`),
  hub enabled, pick status in `idle|failed|holding|done`, and proposal N exists and is feasible. Otherwise it is ignored,
  and the reason goes to `pick.last_reject`.
- `{"type":"pick_cancel"}`: `ps.cancel()` (release + retreat if a skill runs).

Interlock (sim thread, every tick): while pick status is `planning|ready|executing|cancelling`, the operator's base twist, lift
velocity and tray jog are set to zero before the shield filter. The skill expects a still base (`pick/skill.py`). Esc
(latched stop), disarm, the watchdog stop and a client disconnect call `ps.cancel()` if a skill is running.

Telemetry key `pick` (only with `--task pick`): `{status, selected, error, skill: null|{state, phase info..}, proposals:[{id, label,
box2d:[x0,y0,x1,y1], mode, arm, feasible, why}], frame:[640,360], snap_pose:[x,y,yaw,lift], stale: bool, ready_to_pick: bool,
last_reject, backend, planner}`.
- `box2d` is in pixels of the 640x360 head camera, the ZED **left eye** (`ocs/fp_render.head_cam_base(left=True)`: the mount
  centre + 0.06 m along base +y, half the 0.12 m baseline). With `--task pick` the browser `ego` view is rendered from that eye
  (`render_proc.EGO_LEFT_EYE_DY`), so the boxes line up; other tasks keep the centre view. Scale by `width/640, height/360` for
  another `--width/--height`.
- `stale` = the base moved > 0.02 m or > 1 deg, or the lift moved > 0.01 m, since the snapshot (ASSUMED, user-approved 2026-11-12). Stale boxes are drawn
  grey and are not clickable.
- `ready_to_pick` = episode ACTIVE, hub enabled, status idle, at least one feasible proposal, not stale.

`hello` gains `pick: {planner, backend} | null` and `episode: true|false`.

## 3. Browser (static/app.js, index.html)

- Episode banner: IDLE "Press Enter / Start to begin". ACTIVE: timer, goal. OUTCOME: result card (success / fail / ended,
  reason, metrics) and "Backspace = new episode".
- Pick panel (when `hello.pick`): Scan button (pick_propose), proposal list (label, mode, arm, feasible / why), Cancel button,
  status / skill phase / error, a READY TO PICK indicator, last_reject.
- Box overlay: a canvas over the video, drawn only when `tel.cam == "ego"`. Boxes scale from 640x360 to the displayed image
  size. Feasible boxes are drawn green, infeasible red with the reason, stale ones grey; the selected box is thicker. Clicking
  inside a non-stale feasible box sends `pick_select`.
- Key `K` = end the episode (an entry in `inputs.KBD_EVENTS`: `"KeyK": "episode_end"`, a server event).
