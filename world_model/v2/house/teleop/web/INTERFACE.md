# teleop/web -- browser keyboard + Xbox teleop of the SIMULATED YOR

SIMULATION ONLY. The server drives the MuJoCo sim; it imports no robot driver, no ROS, no joystick library, and binds `127.0.0.1` only.
All mapping lives in one place: `inputs.py` (`CONFIG`, `KBD_AXES`, `KBD_EVENTS`, `PAD_*`). The browser fetches the keyboard maps from the server (`hello`), so the JS never duplicates them (the help tables in `index.html` are static text; update them with the maps).

## Launch
```
# server (westeros)
cd /common/users/ss5772/bimanual ; export PYTHONPATH=$PWD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python -m world_model.v2.house.teleop.web.server \
    --scene h5 --task carry --shield            # flags: --port 8765 --scene h5|office --task carry|coffee --shield/--no-shield --fps 15 --flmpc-impl rt|py --grasp-mode ... --token ...
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
| V | shield on/off | C | cycle camera (chase, top, wrist_l, wrist_r) |
| B | wrist inset on/off | Backspace | reset episode |
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
Telemetry (HUD): `ready enabled latched src preset_name scale watchdog stopped pad{connected,mode,deadman,warn} ack{kbd,pad} shield interv_frac interv_now rtf render_fps tick_ms beads_out n_beads cup_state cup_tilt wall_clr person_clr speed wz pose lift_h grasped can_grasp infeasible bumps cmd_twist safe_twist cam inset impl`.
Latency: "net RTT" = ping/pong; "input->ack" = echoed `seq` seen in telemetry (up to 50 ms telemetry period). Real-time factor = sim seconds per wall second over the last ~4 s.

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

## Known limits
* Real-time factor is about 0.17 in the h5 scene at the time of writing: the controller is not the bottleneck (rt FL-MPC ~1 ms/step); `arm.ArmCtl.tau` (`all_pair_barriers` ~16 ms/call, `mj_forward` 8 ms) and `mj_step` (8 ms) per physics step dominate. The operator therefore sees ~6x slow motion; the watchdog runs in wall time.
* Tests: `python -m pytest -q world_model/v2/house/teleop/web/tests` (mapping, deadzone, deadman, watchdog, latch). End-to-end smoke: `python -m world_model.v2.house.teleop.web.smoke_client --out DIR` against a running server (`--token`).
