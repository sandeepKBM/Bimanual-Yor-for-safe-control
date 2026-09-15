# Teleop — keyboard/slider control of the bimanual YOR arms over SSH

The MuJoCo sim runs on this headless server; you control it from your laptop
browser via an SSH port-forward. (MuJoCo's native `mujoco.viewer` is GLFW/X11 and
fails headless — mjviser streams to the browser instead, so no display is needed.)

## Quick start (mjviser web panel)

```bash
# 1) on your LAPTOP, open an SSH tunnel to this box:
ssh -L 8080:localhost:8080  <you>@<this-box>

# 2) in that SSH session, start the panel:
/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python teleop/mjviser_panel.py

# 3) on your laptop, open:
http://localhost:8080
```

You get live 3D + **one slider per actuator** (16: `left/right_arm_joint1..6_pos`
plus the finger `joint7/8_pos`). Dragging a slider sets that joint's PD target and
the arm tracks it live. `--model scenes/scene_yor_cup_collect.xml` loads the
table/cup scene instead of the bare arms; `--port N` changes the port (forward the
same N). `--smoke` starts the server, steps the sim, and exits (CI check, no browser).

Verified: server binds, 16 sliders map to `data.ctrl`, 300 sim steps stable.

## Next: "drive one arm, the other dodges" (smooth-control demo)

The payoff combining this with the FL-MPC `ego_hold` change (mpc_fl.FLMPC): teleop
the RIGHT arm by slider, and run the FL-MPC on the LEFT arm with `ego_hold=False`
so it has no position spring-back — it just moves away smoothly from the approaching
teleoped arm and stays put. Sketch (to build next): a `step_fn` that (a) reads the
right-arm slider targets straight into `data.ctrl[right]`, (b) each step calls
`fl.step(...)` for the LEFT arm only (right arm treated as the moving intruder in the
clearance pairs), applying its returned torque to the left dofs. This shows the
collision-avoidance + smoothness live and under human control.
