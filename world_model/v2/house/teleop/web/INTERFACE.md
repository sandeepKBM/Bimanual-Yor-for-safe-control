# teleop/web -- browser keyboard + Xbox teleop of the SIMULATED YOR

SIMULATION ONLY. The server drives the MuJoCo sim; it imports no robot driver, no ROS, no joystick library, and binds `127.0.0.1` only.
All mapping lives in one place: `inputs.py` (`CONFIG`, `KBD_AXES`, `KBD_EVENTS`, `PAD_*`). The browser fetches the keyboard maps from the server (`hello`), so the JS never duplicates them (the help tables in `index.html` are static text; update them with the maps).

## Launch
```
# server (westeros)
cd /common/users/ss5772/bimanual ; export PYTHONPATH=$PWD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python -m world_model.v2.house.teleop.web.server \
    --scene h5 --task carry --shield            # flags: --port 8765 --scene h5|office --task carry|coffee --shield/--no-shield --fps 15 --flmpc-impl rt|py --grasp-mode ... --token ...
    #   --view ego|chase (default ego)   --people on|off|static (default on)   --footprint fixed|live (default fixed)   --d-wall-door M (opt-in, default off)
    # doorway driving (h5, door|4|7): add  --footprint live --d-wall-door 0.05  and press N (tuck) before the door; see "Doorways" below
# laptop
ssh -L 8765:localhost:8765 westeros.cs.rutgers.edu
# browser: open the URL the server prints, http://localhost:8765/?token=<token>
```
The random token (printed at startup) is required for the page and the websocket, and the websocket also checks `Origin`: westeros is multi-user, so any other local user (or any web page in your laptop browser, via the tunnel) could otherwise reach `localhost:8765`.
One browser controls at a time (a new connection takes over). Every connection starts DISARMED. The sim pauses while no browser is connected.

## Safety model
| rule | behaviour |
|---|---|
| armed | motion is zero until armed: keyboard `Enter`, pad `Start`. Every new connection starts disarmed. |
| pad deadman | gamepad motion needs armed AND `LB` held (teleop_twist_joy `enable_button`). Releasing LB gives one explicit zero and a hard zero (no low-pass tail). |
| keyboard | hold-to-move: keydown sets, keyup clears; each key is its own deadman. Key release to idle = hard zero (no ramp tail). |
| `Space` | zero everything (still armed). Motion keys still held stay ignored until released (no surprise resume). |
| `Esc` | LATCHED stop (disarm). Only `Enter` clears it; pad `Start` does not. |
| pad `Back` | disarm (YOR convention; unlatched). |
| watchdog | a source with no message for 250 ms contributes zero command; every source silent 3 s -> disarmed. |
| blur / hidden | client clears all held keys and sends zero + `focus:false`; pad axes/buttons are zeroed. Pad unplugged -> `pad_gone` -> zero + disarm. Websocket closed -> zero + disarm. |
| send rate | 50 Hz while any input is active; one explicit zero message on active->idle; 10 Hz idle heartbeat (liveness for the watchdog and the RTT). |
| shield | `V` / pad `X` toggles the shield; the shield is ON by default (`--no-shield` to start with it off). Grasp/release only act when armed. |

Watchdog 250 ms sits between moveit_servo `incoming_command_timeout` (0.1 s) and twist_mux / teleop_twist_keyboard (0.5 s).

## Signs (ROS REP-103, in `sim.Cmd`)
+vx forward, +vy LEFT, +wz CCW, +z up. Browser stick axes are +right/+down, so the pad map inverts them (sign -1). Tray-jog channels: `jx jy jz` in the base frame, `jroll`, `jyaw`.
Full-scale: base `base.V_STICK` = (0.9, 0.45, 1.2); tray jog 0.08 m/s / 0.15 rad/s (= ShieldCfg limits); lift 0.12 m/s. All multiplied by the speed scale (presets 0.30/0.60/1.00, default medium).

## Keyboard (`KeyboardEvent.code`, layout independent)
| keys | action | keys | action |
|---|---|---|---|
| W / S | vx +/- | A / D | vy left / right |
| Q / E | wz CCW / CW | R / F | lift up / down |
| Arrow Up/Down | tray x +/- | Arrow Left/Right | tray y +/- |
| PageUp / PageDown | tray z +/- | T / Y | tray yaw +/- |
| G / H | grasp / release | 1 / 2 / 3 | speed preset slow/medium/fast |
| Z / X | speed x0.9 / x1.1 (capped at full stick) | Enter | arm / clear Esc latch |
| Space | zero everything | Esc | latched stop |
| V | shield on/off | C | cycle camera (ego, chase, top, wrist_l, wrist_r) |
| B | wrist inset on/off | Backspace | reset episode |
| **N** | TUCK: tray to the narrow doorway carry pose (offset +0.11, +0.11, 0 m, yaw +0.33 rad from the normal carry; ~4 s) | **M** | back to the normal carry pose |
N / M move the tray through the normal tray-jog path (so the shield's jog governor applies), only while armed; any tray-jog key, Space, Esc, disarm or a disconnect cancels the motion. HUD: `footprint ... tray tucked|carry|to_tuck|to_carry|custom`; the page has Tuck / Normal carry buttons (same keys).
Ctrl/Alt/Meta combos are ignored (browser shortcuts keep working).

## Xbox pad (browser Gamepad API, "standard" mapping; plugged into the LAPTOP, nothing installed on the server)
Press any pad button once while the tab is focused (browsers hide pads until then).
| input | action |
|---|---|
| **LB held** | deadman: no LB, no motion |
| right stick (LB held) | base vx (up = forward), vy (left = +) -- YOR `joystick.py` convention |
| left stick X (LB held) | base wz (left = CCW) |
| D-pad up / down (LB held) | lift up / down |
| LB + RB held | tray mode: left stick = tray x/y, right stick Y = tray z, right stick X = tray yaw, LT / RT (analog `buttons[6/7].value`) = tray roll -/+ |
| A / B | grasp / release |
| X | shield on/off |
| Y | cycle speed preset (YOR used LB for presets; LB is now the deadman) |
| Start / Back | arm / disarm |
| RS click / LS click | cycle camera / wrist inset |
| D-pad left / right | TUCK tray / normal carry (the motion runs only while LB is held; releasing LB pauses it) |
Deadzone 0.05, rescaled; low-pass alpha 0.7 per 20 ms tick (YOR uses 0.9 at its own loop rate; the sim base also slews at `A_HW`).
Conflict resolved: on the "standard" layout `buttons[8]` is both "View" and "Back". It is DISARM (safety first); camera cycle moved to RS click.
If `gamepad.mapping != "standard"` the HUD shows a warning and the triggers (roll) are disabled. Caveat: the sticks/buttons are still read by index in that case, so check the layout before driving.

## Messages (websocket `/ws?token=...`)
client -> server (JSON):
```
{"type":"input","src":"kbd","seq":N,"t":ms,"keys":["KeyW","ShiftLeft"],"focus":true}
{"type":"input","src":"pad","seq":N,"t":ms,"connected":true,"mapping":"standard","id":"...","axes":[LX,LY,RX,RY],"buttons":[17 x value 0..1]}
{"type":"pad_gone"}            {"type":"ping","id":K}
```
server -> client: binary `0x01 + JPEG` frames; JSON `{"type":"hello",...}` once (key maps, presets, cameras); `{"type":"tel",...}` at 20 Hz; `{"type":"pong","id":K}`; `{"type":"bye"}`.
Telemetry (HUD): `ready enabled latched src preset_name scale watchdog stopped tuck pad{connected,mode,deadman,warn} ack{kbd,pad} shield interv_frac interv_now rtf render_fps render_ms tick_ms beads_out n_beads cup_state cup_tilt wall_clr person_clr speed wz pose lift_h grasped can_grasp infeasible bumps cmd_twist safe_twist cam inset impl people fp_mode fp_w fp_l fp_ms tray_off`.
New: `tuck` carry|tucked|to_tuck|to_carry|custom|unknown; `people` on|off|static; `fp_mode` fixed|live and `fp_w`/`fp_l` the shield footprint width/length [m] used this tick (`wall_clr` in the HUD is still measured with the FIXED box); `fp_ms` mean cost of the live footprint [ms/tick]; `tray_off` = commanded tray offset [dx, dy, dz, yaw]; `render_ms` mean render+JPEG time of the stream process.
Latency: "net RTT" = ping/pong; "input->ack" = echoed `seq` seen in telemetry (up to 50 ms telemetry period). Real-time factor = sim seconds per wall second over the last ~4 s.

## Doorways (live footprint, tuck, doorway buffer)
The shield's wall and person rows use a footprint rectangle in the robot frame.  `--footprint fixed` (default) is `geom.FP`, the phase-1 hold-pose box of every robot geom + tray: x [-0.165, 0.577], y [-0.644, 0.130] = 0.774 m wide, and it ignores where the arms / tray are.  `--footprint live` (`ShieldCfg.footprint="live"`) recomputes the base-frame xy box of the CURRENT geoms each shield tick: 51 collision geoms (arms, grippers, base, tray, cup) as convex-hull point sets (mesh hull vertices; circumscribed samples for capsules), one gather+einsum over ~4500 points, + `live_pad` 0.02 m (box() 0.45 ms; whole shield filter 1.4 ms fixed -> 1.9 ms live per tick, measured idle; budget 20 ms; one-time hull build 28 ms).  Only the footprint rectangle changes; every other row (cup, jerk, hardware, IR veto, people CBF) is the same code.
Tuck (N): the tray offset (+0.11, +0.11, 0) m / +0.33 rad (inside the ObjectFrame jog box 0.12 m and yaw cap 0.35 rad; no limit is changed) pulls the right elbow in: live box 0.72 m -> ~0.64 m wide (+0.02 pad per side).  It also extends the box ~0.18 m FORWARD (0.75 m ahead of the base origin), which matters for people rows and turning.  Close-in offsets (dx < 0) do NOT narrow it (IK leaves the box).
h5 door|4|7 clear opening is 0.85-0.864 m (frame to frame; design 1.024).  With the default buffer d_wall = 0.10 m the shield needs footprint + 0.20 m = 0.88 m (tucked + pad) -> still blocked.  `--d-wall-door M` (`ShieldCfg.d_wall_door`, `doorways`, `door_radius` 0.5 m) lowers the wall buffer to M only while the footprint is within 0.5 m of a doorway centre; the pad already covers 2 cm, so M = 0.05 leaves >= 5 cm physical clearance per side when centred.  Headless results (scripted pad operator, h5 start -> door|4|7 -> room|7; grid door span 3.44..4.29 = 0.85 m; tucked live box 0.643 m + 2x0.02 pad = 0.683 m): default d_wall 0.10 -> stalls at the door (needs 0.883 m); `--d-wall-door 0.08` -> still stalls (7 mm theoretical slack); `--d-wall-door 0.05` + tuck + live -> PASSES (min hull-vertex clearance to the house EDT 0.10 m, no contacts, 0 beads out, cup tilt <= 1.9 deg); live WITHOUT tuck (0.77 m) or the fixed footprint (0.774 m) stall even at 0.05.  A person standing still at the walker's start (3.6, 3.8), 1.1 m beyond the door, blocks the advance at the door line (person keep-out 0.30 + 0.10 m against the 0.76 m-long tucked footprint).  Headless check: `python -m world_model.v2.house.teleop.door_test --footprint live --tuck --people off --d-wall-door 0.05 --out DIR` (scripted pad operator through the same InputHub + shield).

## People
`--people on` (default): the scripted people walk; the web sim runs `TeleopSim(people_clock="time")`.  Cause of the old "h5 walker never moves": with the legacy clock (`people_clock="legacy"`, still the TeleopSim default for the offline studies) a track's clock only advances while the CURRENT segment velocity is non-zero, and p_walker's first segment (t 0..2 s) is stationary, so it stayed at (3.6, 3.8) -- in the living room, 1.1 m south of door|4|7 -- forever.  With the time clock it waits 2 s, walks to (0.9, 3.8) by t = 10 s and stops (it yields while the footprint is within 0.15 m of it).  `--people static`: everybody stands at the track start (the old frozen behaviour, explicit: walker at (3.6, 3.8)); `--people off`: no people, no person keep-out.  The shield keeps a person disc of PERSON_CLEAR_R 0.30 + d_person 0.10 = 0.40 m around each person, measured to the footprint rectangle.

## ROS `joy` / `teleop_twist_joy` source (future)
Same interface: subclass `InputSource` (`feed(msg, now)` parses a `sensor_msgs/Joy`-like dict `{"axes":[..],"buttons":[..]}` into `Intent` channels + events; `intent()` is inherited), and hand it to `InputHub([...sources])`. For a standard Xbox `joy` node the axes order differs from the browser (ROS `joy`: 0 LX, 1 LY, 2 LT, 3 RX, 4 RY, 5 RT, 6/7 dpad; sticks left/up = +) so give it its own `PAD_*` dict, e.g. `PAD_BASE_ROS = {"vx": (4, +1), "vy": (3, +1), "wz": (0, +1)}`, and `enable_button` -> `PAD_DEADMAN_BUTTON`. Sign: ROS joy already uses +left/+up, so the browser inversion is dropped. The hub keeps the arm/disarm, speed, watchdog and arbitration unchanged. NOTE: only `keyboard_handler` is installed under `/opt/ros/humble`; do not assume `joy` / `teleop_twist_joy` exist there.

## Conventions and where ROS packages disagree
| topic | teleop_twist_keyboard (master) | teleop_twist_joy (`xbox.config.yaml`) / ros2 `joy.cpp` | moveit_servo / twist_mux | here |
|---|---|---|---|---|
| release detection | terminal raw mode has no keyup: a key keeps its last speed until `k`/another key (latching) | `enable_button` held; releasing publishes one zero twist | commands timeout instead of release events | browser gives real keyup: hold-to-move + explicit zero on keyup; watchdog as backstop |
| timeouts | none in the node; relies on `twist_mux` 0.5 s (humble topics yaml) | none in the node (sends zero on deadman release) | moveit_servo `incoming_command_timeout` 0.1 s; twist_mux 0.5 s | 250 ms (between the two) |
| repeat rate | key autorepeat of the terminal (OS dependent) | `joy` autorepeat_rate 20 Hz, `coalesce_interval` 1 ms | servo expects a steady stream | 50 Hz while active, 10 Hz idle heartbeat |
| axis sign | +x forward, +y left via `u/o/j/l`, `z` +/- are speed scales | joy axes: left/up = +; REP-103 output | REP-103 | REP-103 output; browser axes (+right/+down) inverted in the map |
| enable button | none | `enable_button` (LB) held; `enable_turbo_button` optional | n/a | pad: LB deadman + Start/Back arm session (YOR); keyboard: each key is its own deadman, Esc latched, Enter clears |
| deadzone | n/a | 0.05 (joy `deadzone`); YOR joystick.py 0.05 | n/a | 0.05 rescaled |
Speed keys: teleop_twist_keyboard uses `q/z` = x1.1/x0.9 for all speeds; ours is `X`/`Z` (x1.1/x0.9) because `Q/E` are yaw. W3C Gamepad spec: "standard" mapping guarantees the index layout above; triggers are `buttons[6]/[7]` with an analog `.value`; `axes` are +right/+down.
Sources consulted via the owner's research: teleop_twist_keyboard master, teleop_twist_joy `xbox.config.yaml`, ros2 `joy.cpp`, moveit_servo `servo_parameters.yaml`, twist_mux humble topics yaml, W3C Gamepad spec (not re-fetched by this agent).

## Ego camera
The default view is `ego`: the torso ZED head camera (PROPOSED mount, confirm on hardware): base-frame position (0.05, -0.20955, 1.15 + lift) m, pitched 36.3 deg down, 70 deg vertical FOV, rendered as a free camera from the base pose + lift (the teleop rig has no head-camera body).  `C` cycles ego, chase, top, wrist_l, wrist_r; with the ego view the inset shows the chase view.  `--view chase` starts the old default.  Stream: `--fps 15`, 640x360, JPEG q70, rendered in the render process (HUD shows the server-side render ms).

## Known limits
* Real-time factor is about 0.17 in the h5 scene at the time of writing: the controller is not the bottleneck (rt FL-MPC ~1 ms/step); `arm.ArmCtl.tau` (`all_pair_barriers` ~16 ms/call, `mj_forward` 8 ms) and `mj_step` (8 ms) per physics step dominate. The operator therefore sees ~6x slow motion; the watchdog runs in wall time.
* Tests: `python -m pytest -q world_model/v2/house/teleop/web/tests` (mapping, deadzone, deadman, watchdog, latch). End-to-end smoke: `python -m world_model.v2.house.teleop.web.smoke_client --out DIR` against a running server (`--token`).
