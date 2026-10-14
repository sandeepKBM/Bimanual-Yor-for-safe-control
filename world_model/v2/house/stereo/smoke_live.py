"""stereo/smoke_live.py -- closed-loop smoke: the teleop carry with the SHIELD reading the stereo-derived WorldState (ShieldView) vs the simulator state.  SIMULATION ONLY.

    python -m world_model.v2.house.stereo.smoke_live --seed 0 --source stereo --mode sync --max-s 30
Reports task time, spilled beads, bump contacts, shield-vs-truth input errors (cup / tray / person) over the run, stereo frame stats.  GPU: CUDA_VISIBLE_DEVICES 5 or 6.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("BIMANUAL_TELEOP_FAST", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from .cfg import StereoCfg
from .twin import OUT

TICK = 0.02


def run(seed, source, mode, max_s, res=(640, 360), tail_s=1.0):
    from world_model.v2.house.teleop.sim import TeleopSim
    from world_model.v2.house.teleop import operator as OP, shield as SH
    from world_model.v2.house.ocs.consumers import ShieldView
    from . import live as LV
    sim = TeleopSim("h5", grasped=True, seed=seed, with_people=True, people_clock="time")
    sc = sim.scene
    wp = sc.waypoints
    op = OP.SloppyOperator(seed, "carry", sc.out_route, pick_pose=wp["cafe_counter_pick"], place_pose=wp["lab_desk_place"], params=None, back_route=sc.back_route,
                           back_mode=sc.back_mode)
    shield = SH.Shield(None, occ=sim.occ_ctrl, a_hw=sim.a_hw)
    shield.docks = [wp["cafe_counter_pick"][:2]]
    st = None
    ready_ms = None
    if source == "stereo":
        t0 = time.time()
        st = LV.attach(sim, StereoCfg(W=res[0], H=res[1], rate_hz=10.0, mode="3way"), mode=mode, seed=seed)
        ready_ms = st.ready_ms
        print(f"[smoke] stereo READY in {ready_ms/1e3:.1f}s (mode {mode})", flush=True)
    err = dict(cup_xy=[], tray_tilt=[], cup_tilt=[], person=[], cup_vis=[], person_recall=[], person_recall_fov=[])
    done_t = None
    t_loop = time.time()
    for k in range(int(max_s / TICK)):
        obs = sim.observe()
        cmd = op.step(obs)
        if st is not None:
            ws = st.step(sim)
            view = ShieldView(sim, ws, base_from_ws=False)
            d = sim.d
            bx, by, byaw = sim.base.x, sim.base.y, sim.base.yaw
            cb, sb = math.cos(byaw), math.sin(byaw)
            Rb = np.array([[cb, -sb, 0], [sb, cb, 0], [0, 0, 1]])
            # tray / cup in the BASE frame from the estimate (robot-frame chain: no localisation error) vs truth in the true base frame
            Pt, Pc = ws.objects["tray"].pose, ws.objects["cup"].pose
            cup_b = np.asarray(Pt.p) + np.asarray(Pt.R) @ np.asarray(Pc.p)
            cup_true_b = Rb.T @ (d.xpos[sim.cup] - np.array([bx, by, 0.0]))
            err["cup_xy"].append(float(np.linalg.norm((cup_b - cup_true_b)[:2])))
            Rt_true = Rb.T @ d.xmat[sim.tray].reshape(3, 3)
            err["tray_tilt"].append(float(abs(math.acos(np.clip(np.asarray(Pt.R)[2, 2], -1, 1)) - math.acos(np.clip(Rt_true[2, 2], -1, 1)))))
            err["cup_vis"].append(bool(ws.objects["cup"].visible))
            tp = np.array([xy for xy, _ in sim.people_state]).reshape(-1, 2)
            est_w = []
            for o_ in ws.objects.values():
                if o_.kind.value == "person":
                    pr = np.asarray(o_.pose.p)
                    est_w.append((Rb @ pr)[:2] + np.array([bx, by]))      # estimate seen from the robot, placed with the TRUE base (isolates the perception error)
            est_w = np.array(est_w).reshape(-1, 2)
            for e in est_w:
                err["person"].append(float(np.min(np.linalg.norm(tp - e, axis=1))) if len(tp) else 9.0)       # false track if far from every real person
            for pxy in tp:
                rel = pxy - np.array([bx, by])
                rng_ = float(np.linalg.norm(rel))
                brg = abs((math.atan2(rel[1], rel[0]) - byaw + math.pi) % (2 * math.pi) - math.pi)
                hit = bool(len(est_w) and np.min(np.linalg.norm(est_w - pxy, axis=1)) < 0.7)
                if rng_ < 3.0:
                    err["person_recall"].append(hit)
                if 0.8 < rng_ < 3.0 and brg < math.radians(40):         # geometrically in front of the head camera (no line-of-sight test)
                    err["person_recall_fov"].append(hit)
            safe, info = shield.filter(view, cmd)
        else:
            safe, info = shield.filter(sim, cmd)
        o = sim.tick(safe)
        if op.done and done_t is None:
            done_t = o["t"]
        if done_t is not None and o["t"] - done_t >= tail_s:
            break
    wall = time.time() - t_loop
    res_ = dict(seed=seed, source=source, mode=mode, sim_t=sim.t, done_t=done_t, spilled=int(sim.bead_out.sum()), bumps=int(sim.bump_contacts), wall_s=wall, rtf=sim.t / wall, ready_ms=ready_ms)
    if st is not None:
        res_.update(frames=st.sensors.n_frames, stale_ms=float(np.mean(st.sensors.stale_ms)) if st.sensors.stale_ms else None,
                    cup_xy_err_mm_med=1e3 * float(np.median(err["cup_xy"])), cup_xy_err_mm_p95=1e3 * float(np.percentile(err["cup_xy"], 95)),
                    tray_tilt_err_deg_p95=math.degrees(float(np.percentile(err["tray_tilt"], 95))), cup_visible_frac=float(np.mean(err["cup_vis"])),
                    person_false_frac=float(np.mean(np.asarray(err["person"]) > 1.0)) if err["person"] else None, person_err_m_med=float(np.median(err["person"])) if err["person"] else None, person_recall_3m=float(np.mean(err["person_recall"])) if err["person_recall"] else None, n_person_ticks=len(err["person"]), person_recall_front_0p8_3m=float(np.mean(err["person_recall_fov"])) if err["person_recall_fov"] else None, n_front_ticks=len(err["person_recall_fov"]), privileged=st.pipe.step(st.view).privileged())
        st.close()
    print(json.dumps(res_), flush=True)
    return res_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", choices=["truth", "stereo"], default="stereo")
    ap.add_argument("--mode", default="sync")
    ap.add_argument("--max-s", type=float, default=30.0)
    ap.add_argument("--res", type=int, nargs=2, default=[640, 360])
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    r = run(a.seed, a.source, a.mode, a.max_s, tuple(a.res))
    Path(OUT / (a.name or f"smoke_live_{a.source}_{a.mode}_s{a.seed}.json")).write_text(json.dumps(r, indent=1, default=float))
    os._exit(0)


if __name__ == "__main__":
    main()
