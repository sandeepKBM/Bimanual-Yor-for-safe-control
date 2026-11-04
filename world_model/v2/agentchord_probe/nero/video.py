"""agentchord_probe/nero/video.py -- re-run ONE world of a cached AgentChord plan and record an mp4 (CPU osmesa render).  SIMULATION ONLY.

Same run_world as `exec` (same seed rule, same physics, same monitors); frames are grabbed inside atoms._step, so recording does not change the
episode.  Two views side by side (contents_eval 'torso' + a wide side view), with the current edge, sim time, object tilt and the first
violation written on the frame.
  python -m world_model.v2.agentchord_probe.nero.video --plan P/plan.json --task tray4 --cell 0 --i 0 --out clip.mp4
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np

from . import atoms as A
from . import tasks as T

SIDE = dict(lookat=[0.30, -0.30, 0.95], distance=1.9, azimuth=-90.0, elevation=-18.0)      # wide view, shows the shelf in the box task


def _tilt_deg(xmat):
    z = np.asarray(xmat).reshape(3, 3)[:, 2]
    return float(np.degrees(np.arccos(np.clip(z[2], -1.0, 1.0))))


def record(plan, task, cell_idx, i, out, seeds="baseline", fps=25, w=640, h=480):
    import cv2
    import imageio.v2 as imageio
    import mujoco
    from world_model.v2.nero.contents import carry as CR
    from world_model.v2.vlm import contents_eval as CE
    from .execute import run_world

    cams = [CE._cam(CE.VIEWS["torso"]), CE._cam(SIDE)]
    stride = max(1, int(round(1.0 / (fps * CR.DT))))
    st = dict(r=None, frames=[], edge="", viol=None)
    orig_step, orig_plan = A._step, A._plan_edge

    def plan_edge(env, L, R):
        st["edge"] = f"{getattr(L, 'fn', None)} | {getattr(R, 'fn', None)}"
        return orig_plan(env, L, R)

    def step(env, monitors):
        hit = orig_step(env, monitors)
        ep = env.ep
        if ep.k % stride == 0:
            if st["r"] is None:
                st["r"] = mujoco.Renderer(ep.m, h, w)
            imgs = []
            for c in cams:
                st["r"].update_scene(ep.d, camera=c)
                imgs.append(st["r"].render().copy())
            fr = np.ascontiguousarray(np.concatenate(imgs, axis=1))
            tilt = _tilt_deg(ep.d.xmat[ep.obj])
            v = ep.track.get("viol")
            if v and st["viol"] is None:
                st["viol"] = f"{v.get('kind')} at t={v.get('t', 0):.2f}s"
            lines = [f"{task}  cell {T.cell_list(task)[cell_idx]}  world {i}   t={ep.k * CR.DT:5.2f}s",
                     f"edge: {st['edge']}", f"object tilt {tilt:5.2f} deg (limit 10, ASSUMED)"]
            if st["viol"]:
                lines.append("first violation: " + st["viol"])
            for k, s in enumerate(lines):
                cv2.putText(fr, s, (10, 24 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            st["frames"].append(fr)
        return hit

    A._step, A._plan_edge = step, plan_edge
    try:
        cell = T.cell_list(task)[cell_idx]
        rec = run_world(dict(plan=plan, task=task, cell=list(cell), i=i, seeds=seeds))
    finally:
        A._step, A._plan_edge = orig_step, orig_plan
        if st["r"] is not None:
            st["r"].close()
    imageio.mimsave(out, st["frames"], fps=fps, codec="libx264", quality=7, macro_block_size=8)
    summ = dict(out=out, task=task, cell=list(cell), i=i, seed=rec.get("seed"), outcome=rec.get("outcome"), reason=rec.get("reason"),
                ac_success=rec.get("ac_success"), success_ours=rec.get("success_ours"), any_U=rec.get("any_U"), R_fail=rec.get("R_fail"),
                theta=rec.get("theta"), n_frames=len(st["frames"]), plan=plan)
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump(summ, f, indent=1, default=float)
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--cell", type=int, default=0, help="cell index")
    ap.add_argument("--i", type=int, default=0)
    ap.add_argument("--seeds", default="baseline", choices=("cert", "baseline"))
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    print(json.dumps(record(a.plan, a.task, a.cell, a.i, a.out, a.seeds, a.fps), default=float))


if __name__ == "__main__":
    main()
