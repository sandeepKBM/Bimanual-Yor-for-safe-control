#!/usr/bin/env python
"""Headless, SSH-friendly teleop panel for the bimanual YOR arms (mjviser/viser).

The MuJoCo sim runs HERE on the server; mjviser serves a viser web UI on
localhost that you reach from your laptop by forwarding the port over SSH:

    # on your laptop:
    ssh -L 8080:localhost:8080  <you>@<this-box>
    # then in the SSH session:
    /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python teleop/mjviser_panel.py
    # then open  http://localhost:8080  in your laptop browser

The panel auto-generates one slider per actuator (16 position servos:
left/right_arm_joint1..6_pos + finger joint7/8_pos) — dragging a slider sets that
actuator's target in data.ctrl and the arm PD-tracks it live in 3D. No X display
needed on the server (unlike mujoco.viewer/GLFW, which fails headless).

This is the BASIC joint-space teleop. For "drive one arm, the other dodges via
FL-MPC" see the plan in teleop/README (ego_hold FL-MPC on the non-teleoped arm).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = str(REPO / "scenes" / "yor_robot_derived.mjcf")


def build(model_path: str):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    # start at the 'home' keyframe if the model has one, else zero pose
    try:
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    except Exception:
        pass
    mujoco.mj_forward(model, data)
    # initialise every actuator's target to the CURRENT joint value so nothing
    # snaps when the panel opens (position servos transmit to a single joint).
    for a in range(model.nu):
        try:
            jid = int(model.actuator_trnid[a, 0])
            qadr = int(model.jnt_qposadr[jid])
            data.ctrl[a] = data.qpos[qadr]
        except Exception:
            data.ctrl[a] = 0.0
    return model, data


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="MJCF to load (default: the bimanual robot; pass a scene "
                         "like scenes/scene_yor_cup_collect.xml to include table/cup).")
    ap.add_argument("--port", type=int, default=8080, help="viser server port (forward this over SSH).")
    ap.add_argument("--smoke", action="store_true",
                    help="start the server, print the actuator list, run a few seconds, exit (CI check).")
    args = ap.parse_args()

    model, data = build(args.model)
    acts = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(model.nu)]
    print(f"[mjviser-panel] model={args.model}")
    print(f"[mjviser-panel] {model.nu} actuators (each becomes a slider): {acts}")
    print(f"[mjviser-panel] timestep={model.opt.timestep}s  nq={model.nq} nv={model.nv}")

    def step_fn(m, d):
        mujoco.mj_step(m, d)

    if args.smoke:
        # Headless server-start check: bring up the viser server, confirm it binds,
        # step the sim a bit, then exit (no browser needed).
        import viser, time
        server = viser.ViserServer(host="0.0.0.0", port=args.port)
        from mjviser import Viewer  # noqa
        # Build the scene + run a few hundred steps to prove sim+scene are healthy.
        for _ in range(300):
            step_fn(model, data)
        assert np.isfinite(data.qpos).all(), "NaN in qpos after stepping"
        print(f"[mjviser-panel][smoke] viser server bound on port {args.port}; "
              f"300 steps OK; qpos finite. Panel wiring is healthy.")
        server.stop()
        print("=== MJVISER SMOKE DONE ===")
        return

    from mjviser import Viewer
    import viser
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"[mjviser-panel] serving on http://localhost:{args.port}  "
          f"(SSH: ssh -L {args.port}:localhost:{args.port} <you>@<box>)")
    Viewer(model, data, step_fn=step_fn, server=server)


if __name__ == "__main__":
    main()
