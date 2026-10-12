"""smoke.py -- short smoke for the office scene: compile, compose, step 2 s, robot/scene contacts, route, 3 PNGs.  SIMULATION only.

    PYTHONPATH=/common/users/ss5772/bimanual python -m world_model.v2.house.office.smoke [--out DIR] [--no-render] [--no-route]

Rendering uses MUJOCO_GL=egl on the first GPU (never GPU 0) with < 500 MiB used.  Outputs (default /data/local/ss5772/molmospaces/office):
  office_overview.png  (ortho top view + robot at lab_start)  office_head_cam.png (robot head camera)  office_route.png (occupancy + route)
  smoke_results.json   all numbers printed here.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def pick_gpu() -> str | None:
    """First GPU index != 0 with < 500 MiB in use and < 20 % utilisation (nvidia-smi), else None."""
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout
        for line in o.strip().splitlines():
            i, mem, util = (int(v) for v in line.replace(" ", "").split(","))
            if i != 0 and mem < 500 and util < 20:
                return str(i)
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("OFFICE_OUT", "/data/local/ss5772/molmospaces/office"))
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--no-route", action="store_true")
    ap.add_argument("--steps-s", type=float, default=2.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    if not a.no_render:
        os.environ.setdefault("MUJOCO_GL", "egl")
        g = pick_gpu()
        if g is None:
            print("no free non-zero GPU: skipping rendering", flush=True)
            a.no_render = True
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = g
            res["gpu"] = g
    import mujoco
    from .. import build as B, lift as L, nav as N, phase1 as P1
    from . import scene as S, compose as CP, nav_grid as G, route as R

    # ---- 1. compile the static scene
    t = time.time()
    sp = S.load_office_spec()
    m0 = sp.compile()
    meta = S.office_meta()
    res["static"] = dict(ngeom=int(m0.ngeom), nbody=int(m0.nbody), njnt=int(m0.njnt), nmesh=int(m0.nmesh), compile_s=round(time.time() - t, 2),
                         free_joints=int(sum(m0.jnt_type == mujoco.mjtJoint.mjJNT_FREE)),
                         persons=[p["name"] for p in meta["persons"]], counter_height=meta["counter_height"],
                         desk_top_height=round(meta["desk_top_height"], 3), doors_clear_w=[d["clear_w"] for d in meta["doors"]])
    assert m0.njnt == 0, "static scene must have no joints"
    print("static:", res["static"], flush=True)

    # ---- 2. compose robot + tray (+ lift) and step 2 s
    t = time.time()
    C = CP.new_hold_office(dt=0.002, lift_h=0.0)
    m, d = C.m, C.d
    res["compose"] = dict(compose_s=round(time.time() - t, 1), nbody=int(m.nbody), ngeom=int(m.ngeom), nv=int(m.nv), nu=int(m.nu),
                          nmocap=int(m.nmocap), lift_mode=C.info["lift_mode"], ik_err=C.info["ik_err"], model_hash=C.info["model_hash"])
    cls = P1.geom_class(m)

    def robot_scene(cc):
        return {k: v for k, v in cc.items() if "H" in k and k != "HH"}

    mujoco.mj_forward(m, d)
    q_hold, mocap0 = d.qpos.copy(), d.mocap_pos[0].copy()
    cc0 = P1.contact_counts(m, d, cls)
    md0 = P1.robot_house_min_distance(m, d, cls)
    tray = m.body("tray").id
    tray0 = d.xpos[tray].copy()
    worst = {}
    n = int(round(a.steps_s / m.opt.timestep))
    for k in range(n):
        mujoco.mj_step(m, d)
        if k % 25 == 0:
            for key, v in robot_scene(P1.contact_counts(m, d, cls)).items():
                worst[key] = max(worst.get(key, 0), v)
    cc1 = P1.contact_counts(m, d, cls)
    res["step"] = dict(sim_s=a.steps_s, steps=n, contacts_t0_all=cc0, contacts_t0_robot_vs_scene=robot_scene(cc0),
                       contacts_end_all=cc1, contacts_end_robot_vs_scene=robot_scene(cc1), max_robot_vs_scene_during=worst,
                       min_dist_robot_scene_t0_m=md0[0], min_dist_pair_t0=[md0[1], md0[2]],
                       tray_drift_m=float(np.linalg.norm(d.xpos[tray] - tray0)), finite=bool(np.isfinite(d.qpos).all()))
    print("step:", {k: v for k, v in res["step"].items() if k != "contacts_end_all"}, flush=True)
    # lift variant still works in the office: raise to full height and check for contacts, then lower again
    L.set_lift(C, L.H_MAX)
    res["lift"] = dict(h_set=L.H_MAX, h_state=round(L.lift_state(C), 4), robot_vs_scene=robot_scene(P1.contact_counts(m, d, cls)),
                       tray_z=round(float(d.xpos[tray][2]), 3))
    L.set_lift(C, 0.0)
    print("lift:", res["lift"], flush=True)

    # ---- 3. occupancy + mission route
    mission = None
    F = None
    if not a.no_route:
        t = time.time()
        F = G.build_office_grid()
        C5 = N.coarsen(F, 5)
        res["grid"] = dict(res_fine=F.res, shape=list(F.occ.shape), occupied_frac=round(float(F.occ.mean()), 4), build_s=round(time.time() - t, 1))
        t = time.time()
        mission = R.plan_mission(F, C5, meta, verbose=False)
        res["mission"] = dict(ok=bool(mission["ok"]), reason=mission.get("reason"), plan_s=round(time.time() - t, 1),
                              validation=mission.get("validation"),
                              door_passages={k: [dict(door=dp["door"], u=dp["u"], clearance=dp["clearance"]) for dp in v.get("door_passages", [])]
                                             for k, v in mission["legs"].items()},
                              chosen={k: v.get("chosen") for k, v in mission["legs"].items()},
                              peak={k: dict(speed=v["route"]["peak_speed"], acc=v["route"]["peak_acc"], jerk=v["route"]["peak_jerk"],
                                            wz=v["route"]["peak_wz"], end_err_m=v["route"]["end_err_m"]) for k, v in mission["legs"].items() if v.get("route")})
        print("mission:", json.dumps(res["mission"], default=float)[:1500], flush=True)
        if mission["ok"]:
            tr = mission["traj"]
            np.savez_compressed(out / "route_traj.npz", t=tr.t, x=tr.x, y=tr.y, yaw=tr.yaw, marks=np.array([(0, i0, i1) for _, i0, i1 in mission["marks"]]))
            # replay in the full composed model (kinematic base, forward only): contacts robot-vs-scene along the route
            d.qpos[:] = q_hold                           # fresh hold state (undo the 2 s of settling: ball position etc.)
            d.qvel[:] = 0.0
            d.mocap_pos[0] = mocap0
            mujoco.mj_forward(m, d)
            idx = np.arange(0, len(tr.t), int(round(0.25 / (tr.t[1] - tr.t[0]))))
            nz, md_min, md_at = 0, 9.0, None
            for j, i in enumerate(idx):
                CP.set_base_pose(C, float(tr.x[i]), float(tr.y[i]), float(tr.yaw[i]))
                cc = robot_scene(P1.contact_counts(m, d, cls))
                nz += int(sum(cc.values()) > 0)
                if j % 12 == 0:
                    dist = P1.robot_house_min_distance(m, d, cls, radius=2.0, distmax=1.0)
                    if dist[0] < md_min:
                        md_min, md_at = dist[0], (float(tr.x[i]), float(tr.y[i]), dist[1], dist[2])
            res["replay"] = dict(n_poses=int(len(idx)), poses_with_robot_scene_contact=nz, min_geom_dist_sampled_m=float(md_min), at=md_at)
            print("replay:", res["replay"], flush=True)
            CP.set_base_pose(C, *meta["waypoints"]["lab_start"]["xy"], meta["waypoints"]["lab_start"]["heading"])
        d.qpos[:] = q_hold
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

    # ---- 4. renders
    if not a.no_render:
        from PIL import Image
        r = mujoco.Renderer(m, 700, 1920)
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera="overview_top")
        Image.fromarray(r.render()).save(out / "office_overview.png")
        r2 = mujoco.Renderer(m, 1000, 1400)
        r2.update_scene(d, camera="overview_oblique")
        Image.fromarray(r2.render()).save(out / "office_overview_oblique.png")
        # head camera: lab_start, then at door1 (inside corridor) and at the counter
        hc = mujoco.Renderer(m, 480, 640)
        tiles = []
        for nm, pose in (("lab_start", meta["waypoints"]["lab_start"]["xy"] + [0.0]),
                         ("door1", [8.1 - 1.2, 3.257, 0.0]), ("cafe_counter_pick", meta["waypoints"]["cafe_counter_pick"]["xy"] + [math.pi / 2])):
            CP.set_base_pose(C, pose[0], pose[1], pose[2])
            hc.update_scene(d, camera="head_cam")
            tiles.append(hc.render())
        Image.fromarray(tiles[0]).save(out / "office_head_cam.png")
        Image.fromarray(np.concatenate(tiles, axis=1)).save(out / "office_head_cam_3poses.png")
        # robot close-ups from the overview camera at the two stand-offs
        for nm in ("cafe_counter_pick", "lab_desk_place"):
            w = meta["waypoints"][nm]
            CP.set_base_pose(C, w["xy"][0], w["xy"][1], w["heading"])
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [w["xy"][0], w["xy"][1] + 0.4, 0.6]
            cam.distance, cam.azimuth, cam.elevation = 3.4, 55.0, -22.0    # camera south-west of the robot, looking north-east
            r2.update_scene(d, cam)
            Image.fromarray(r2.render()).save(out / f"office_standoff_{nm}.png")
        res["renders"] = sorted(str(p) for p in out.glob("office_*.png"))

    # ---- 5. occupancy + route PNG
    if F is not None:
        _plot_route(out / "office_route.png", F, meta, mission)
        res.setdefault("renders", []).append(str(out / "office_route.png"))

    (out / "smoke_results.json").write_text(json.dumps(res, indent=1, default=float))
    print("wrote", out / "smoke_results.json", flush=True)
    return 0


def _plot_route(path, F, meta, mission):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle
    from .. import nav as N
    fig, ax = plt.subplots(figsize=(20, 7.4), dpi=100)
    ny, nx = F.occ.shape
    ext = [F.x0, F.x0 + nx * F.res, F.y0, F.y0 + ny * F.res]
    img = np.where(F.occ, 0.15, 0.95)
    ax.imshow(img, extent=ext, origin="lower", cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    for nm, poly in meta["rooms"].items():
        P = np.array(poly)
        ax.add_patch(Polygon(P, closed=True, fill=False, ec="tab:blue", lw=0.8, ls=":"))
        ax.text(P[:, 0].mean(), P[:, 1].max() + 0.2, nm, color="tab:blue", ha="center", fontsize=9)
    for p in meta["persons"]:
        ax.add_patch(Circle(p["xy"], p["radius"], fc="tab:red", alpha=0.6))
        ax.text(p["xy"][0], p["xy"][1] + 0.3, p["name"], fontsize=7, color="tab:red", ha="center")
    fp = meta["footprint"]
    if mission is not None and mission.get("traj") is not None:
        tr = mission["traj"]
        cols = {"A_drive_to_counter": "tab:green", "slide_back": "tab:orange", "rotate": "tab:orange", "B_drive_to_desk": "tab:purple"}
        for nm, i0, i1 in mission["marks"]:
            ax.plot(tr.x[i0:i1], tr.y[i0:i1], color=cols[nm], lw=1.6, label=nm)
        step = int(round(9.0 / (tr.t[1] - tr.t[0])))
        pts = np.arange(0, len(tr.t), step)
        for i in list(pts) + [len(tr.t) - 1]:
            c, s = math.cos(tr.yaw[i]), math.sin(tr.yaw[i])
            corners = [(fp["x"][0], fp["y"][0]), (fp["x"][1], fp["y"][0]), (fp["x"][1], fp["y"][1]), (fp["x"][0], fp["y"][1])]
            W = [(tr.x[i] + c * a - s * b, tr.y[i] + s * a + c * b) for a, b in corners]
            ax.add_patch(Polygon(W, closed=True, fill=False, ec="tab:cyan", lw=0.7))
        for dr in meta["doors"]:
            u = [dp for leg in mission["legs"].values() for dp in leg.get("door_passages", []) if dp.get("door") == dr["id"]]
            ax.plot(*dr["centre"], "y*", ms=12)
            ax.text(dr["centre"][0], dr["centre"][1] + 0.9, f"{dr['id']} clear {dr['clear_w']:.2f} m", ha="center", fontsize=8)
    for nm, w in meta["waypoints"].items():
        ax.plot(*w["xy"], "o", color="k", ms=5)
        ax.annotate("", xy=(w["xy"][0] + 0.6 * math.cos(w["heading"]), w["xy"][1] + 0.6 * math.sin(w["heading"])), xytext=tuple(w["xy"]),
                    arrowprops=dict(arrowstyle="->", color="k"))
        ax.text(w["xy"][0] + 0.1, w["xy"][1] - 0.45, nm, fontsize=8)
    v = (mission or {}).get("validation") or {}
    ax.set_title(f"office occupancy (1 cm, obstacles dark) + coffee-run route  |  ok={ (mission or {}).get('ok')}  min footprint clearance "
                 f"{v.get('min_clearance_m', float('nan')):.3f} m  door-zone min {v.get('door_zone_min_clearance_m')}", fontsize=9)
    ax.set_aspect("equal")
    ax.legend(loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
