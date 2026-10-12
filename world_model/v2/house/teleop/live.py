"""teleop/live.py -- OPTIONAL live keyboard teleop in the MuJoCo passive viewer (needs a display).  SIMULATION ONLY.

    python -m world_model.v2.house.teleop.live --task coffee --shield      # keys: see operator.KeyboardOperator

The only input device is the viewer window's key_callback (arrow keys etc.).  It drives the SIMULATED mocap base; nothing here can
reach a robot.  Not exercised in the headless smoke (no display): the key handling is unit-tested via KeyboardOperator.inject.
"""
from __future__ import annotations

import argparse
import time

from . import operator as OP, shield as SH
from .sim import TeleopSim, TICK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["carry", "coffee"], default="carry")
    ap.add_argument("--scene", default="h5")
    ap.add_argument("--shield", action="store_true")
    a = ap.parse_args()
    import mujoco.viewer
    carry = a.task == "carry"
    sim = TeleopSim(a.scene, grasped=carry, counter=not carry, desk=not carry)
    kb = OP.KeyboardOperator()
    sh = SH.Shield(occ=sim.occ, a_hw=sim.a_hw) if a.shield else SH.Passthrough()
    with mujoco.viewer.launch_passive(sim.m, sim.d, key_callback=kb.key_callback) as v:
        while v.is_running():
            t0 = time.perf_counter()
            cmd = kb.step(sim.observe())
            safe, info = sh.filter(sim, cmd)
            sim.tick(safe)
            v.sync()
            time.sleep(max(0.0, TICK - (time.perf_counter() - t0)))


if __name__ == "__main__":
    main()
