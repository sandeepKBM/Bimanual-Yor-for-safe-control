"""teleop/door_test.py -- headless scripted-operator doorway test of the browser teleop stack (SIMULATION ONLY).

A scripted "operator" drives the SAME path a browser pad does: Xbox-pad messages -> web.inputs.InputHub (arm, tuck, deadzone, low-pass, speed preset)
-> Shield (live or fixed footprint) -> TeleopSim.  It drives from the h5 start through door|4|7 (Bedroom -> LivingRoom, measured clear opening 0.864 m) and
reports: success (base centre in room|7), robot<->house physical contact count (arms / tray / cup; the mocap base geoms never collide), the smallest clearance of any robot hull vertex to the 1 cm house EDT near the door, beads / cup
tilt, the shield's labels.  The sim clock drives the hub (no wall-clock), so the run is deterministic and as fast as the sim.

    cd <repo> ; export PYTHONPATH=$PWD:$PWD/experiments/ab_smooth_vs_bangbang:$PWD/experiments/realtime
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=6 CUDA_VISIBLE_DEVICES=6 $PY -m world_model.v2.house.teleop.door_test --footprint live --tuck --people off --out DIR [--video f.mp4]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

os.environ.setdefault("BIMANUAL_FLMPC_IMPL", "rt")
os.environ.setdefault("BIMANUAL_TELEOP_FAST", "1")           # bit-identical numba barriers (see fastpath.py)

DOOR_Y = 4.94                      # wall line of door|4|7 (map frame)
ROOM7 = dict(x=(3.29, 4.94), y=(0.0, 4.94))
HEADING = -math.pi / 2             # drive south through the door


def door_span(occ, y=4.90, x_lo=3.30, x_hi=4.95):
    """(x_left, x_right) of the free run in the occupancy grid across the door line (the frame-to-frame clear opening)."""
    ix = lambda x: int(round((x - occ.x0) / occ.res))
    iy = int(round((y - occ.y0) / occ.res))
    xs = np.arange(x_lo, x_hi, occ.res)
    free = np.array([not occ.occ[iy, ix(x)] for x in xs])
    best, cur = (0, 0), None
    for i, f in enumerate(free):
        if f and cur is None:
            cur = i
        if (not f or i == len(free) - 1) and cur is not None:
            end = i if not f else i + 1
            if end - cur > best[1] - best[0]:
                best = (cur, end)
            cur = None
    return float(xs[best[0]]), float(xs[best[1] - 1] + occ.res)


class PadDriver:
    """emulates the browser pad: builds 'standard'-mapping messages (axes +right/+down, buttons 0..1) and feeds the hub with the SIM clock."""

    def __init__(self, hub):
        from .web import inputs as IN
        self.hub, self.IN, self.seq = hub, IN, 0
        self.btn = np.zeros(17)

    def send(self, now, vx=0.0, vy=0.0, wz=0.0, lb=True, press=()):
        IN = self.IN
        dz = self.hub.cfg["pad_deadzone"]
        inv = lambda a: 0.0 if abs(a) < 1e-9 else math.copysign(abs(a) * (1 - dz) + dz, a)   # inverse of the rescaled dead-zone
        ax = [0.0] * 4
        ax[3] = -inv(vx)                          # PAD_BASE vx: (axis 3, -1)
        ax[2] = -inv(vy)                          # vy: (axis 2, -1)
        ax[0] = -inv(wz)                          # wz: (axis 0, -1)
        b = np.zeros(17)
        b[IN.PAD_DEADMAN_BUTTON] = 1.0 if lb else 0.0
        for i in press:
            b[i] = 1.0
        self.seq += 1
        self.hub.feed(dict(type="input", src="pad", seq=self.seq, connected=True, mapping="standard", axes=ax, buttons=list(b)), now)


def run(a):
    from . import geom as GM, shield as SH, scene as SC
    from .sim import TeleopSim, TICK
    from .web import inputs as IN
    t_start = time.time()
    kw = dict(scene_name="h5", grasped=True, seed=a.seed, with_people=a.people != "off")
    if a.people == "on":
        kw["people_clock"] = "time"
    sim = TeleopSim(**kw)
    if a.people == "static":
        SC.freeze_people(sim.scene)
        sim.set_people()
    cfg = SH.ShieldCfg(footprint=a.footprint, live_pad=a.live_pad)
    occ = sim.occ
    xl, xr = door_span(occ)
    xc = 0.5 * (xl + xr)
    if a.d_wall is not None:
        cfg.d_wall = a.d_wall
    if a.d_wall_door is not None:
        cfg.d_wall_door, cfg.doorways = a.d_wall_door, ((xc, DOOR_Y),)
    if a.relax:
        cfg.relax_tier = True
    if a.d_person is not None:
        cfg.d_person = a.d_person
    shield = SH.Shield(cfg, occ=occ, a_hw=sim.a_hw)
    shield.docks = []
    hub = IN.InputHub()
    pad = PadDriver(hub)
    lf = GM.LiveFootprint(sim.m, pad=0.0)
    rec = dict(t=[], pose=[], v=[], label=[], qp=[], interv=[], beads=[], tilt=[], wall_live=[], door_dist=[], fp_y=[], person=[])
    video = None
    if a.video:
        video = VideoRig(sim, a.video, fps=a.vfps)

    dt = TICK
    state = dict(t=0.0)
    log = []

    def step(vx=0.0, vy=0.0, wz=0.0, lb=True, press=(), nlog=True):
        now = sim.t
        pad.send(now, vx, vy, wz, lb, press)
        obs = sim.observe()
        from .arm import _log_R
        obs["tray_offset"] = (sim.of.dp.copy(), float(_log_R(sim.of.dR)[2]))
        cmd = hub.step(obs, now=now, dt=dt)
        safe, info = shield.filter(sim, cmd)
        o = sim.tick(safe)
        k = sim.k
        if k % 5 == 0:
            pose = sim.pose()
            box = lf.box(sim.d, pose, pad=0.0)
            w = GM.boundary_pts(box, step=0.02)
            R = GM.rot(pose[2])
            wl = float(sim.occ.lookup(*(w @ R.T + pose[:2]).T).min())
            rec["t"].append(sim.t); rec["pose"].append(pose.tolist()); rec["v"].append(sim.base.v.tolist())
            rec["label"].append(info["label"]); rec["qp"].append(info["qp"]); rec["interv"].append(float(info["intervention_norm"]))
            rec["beads"].append(int(o["beads_out"])); rec["tilt"].append(float(o["cup_tilt"])); rec["wall_live"].append(wl)
            rec["fp_y"].append(box["y"]); rec["person"].append(float(o["pdist"]))
            if np.hypot(pose[0] - xc, pose[1] - DOOR_Y) < 2.0:
                w3 = lf.world_pts(sim.d)                       # every hull vertex of every robot collision geom vs the 1 cm EDT of the house (door frames, leaf, walls, furniture)
                rec["door_dist"].append(float(sim.occ.lookup(w3[:, 0], w3[:, 1]).min()))
            else:
                rec["door_dist"].append(float("nan"))
        if video is not None:
            video.maybe(sim, k)
        return o, info

    # ---- phase 0: heartbeat, arm (pad Start), tuck
    for _ in range(25):
        step(lb=False)
    step(lb=False, press=(9,))
    for _ in range(3):
        step(lb=False)
    assert hub.enabled, "not armed"
    tuck_s = None
    if a.tuck:
        step(press=(14,))                                   # D-pad left = tuck
        t0 = sim.t
        while hub.tuck_state() != "tucked" and sim.t - t0 < 25.0:
            step()
        t1 = sim.t
        for _ in range(int(1.5 / dt)):                      # let the arm references settle
            step()
        tuck_s = t1 - t0
    pose0 = sim.pose()
    box = lf.box(sim.d, pose0, pad=a.live_pad)
    fp_w = box["y"][1] - box["y"][0]
    yc = 0.5 * (box["y"][0] + box["y"][1])
    # footprint centre (base-frame y offset yc): at heading -90 deg  world_x = root_x + y_b
    root_x = xc - yc
    wps = [(root_x, 6.0, HEADING, "align", 0.025), (root_x, a.y_end, HEADING, "through", 0.04)]
    out = dict(args=vars(a), door_x=[xl, xc, xr], door_width=xr - xl, fp_box_after_tuck=box, fp_width=fp_w, fp_centre_y=yc, tuck_s=tuck_s,
               tray_off=[hub.tray_off[0].tolist(), hub.tray_off[1]], root_x_target=root_x, tuck_state=hub.tuck_state(),
               beads_after_tuck=int(sim.bead_check()), cup_tilt_after_tuck=float(sim.cup_rel()[1]))
    # ---- drive
    V = IN.CONFIG["presets"][IN.CONFIG["preset_default"]]
    from .base import V_STICK, wrap
    vmax = dict(vx=0.45, vy=0.25, wz=0.55)
    result = "ok"
    for wi, (wx, wy, wyaw, nm, tol) in enumerate(wps):
        t0 = sim.t
        last_prog = (sim.t, 1e9)
        while True:
            x, y, yaw = sim.pose()
            e = np.array([wx - x, wy - y])
            eb = GM.rot(yaw).T @ e
            dist = float(np.linalg.norm(e))
            eyaw = wrap(wyaw - yaw)
            if dist < tol and abs(eyaw) < 0.03 and np.linalg.norm(sim.base.v[:2]) < 0.05:
                break
            if nm == "through":                              # steady forward speed, lateral / yaw correction against the door axis
                vx = float(np.clip(1.2 * eb[0], 0.0, a.v_through))
                vy = float(np.clip(3.0 * eb[1], -vmax["vy"], vmax["vy"]))
            else:
                gain = 1.5
                vx = float(np.clip(gain * eb[0], -vmax["vx"], vmax["vx"]))
                vy = float(np.clip(gain * eb[1], -vmax["vy"], vmax["vy"]))
                if nm == "align" and abs(eyaw) > 0.3:         # rotate first, then translate (the tucked tray reaches 0.75 m ahead)
                    vx *= 0.3; vy *= 0.3
            wz = float(np.clip(1.5 * eyaw, -vmax["wz"], vmax["wz"]))
            scale = hub.scale
            o, info = step(vx / (V_STICK[0] * scale), vy / (V_STICK[1] * scale), wz / (V_STICK[2] * scale))
            if dist < last_prog[1] - 0.02:
                last_prog = (sim.t, dist)
            if sim.t - last_prog[0] > a.stall_s:
                result = f"stalled in '{nm}' at pose {np.round(sim.pose(), 3).tolist()} label={info['label']} qp={info['qp']}"
                break
            if sim.t - t0 > a.seg_timeout:
                result = f"timeout in '{nm}' at pose {np.round(sim.pose(), 3).tolist()}"
                break
        if result != "ok":
            break
    for _ in range(int(1.0 / dt)):
        step(lb=True)
    if video is not None:
        video.close()
    pose = sim.pose()
    r = np.array(rec["door_dist"], float)
    inside = bool(ROOM7["x"][0] < pose[0] < ROOM7["x"][1] and ROOM7["y"][0] < pose[1] < ROOM7["y"][1])
    inside_fp = bool(ROOM7["x"][0] < pose[0] + yc < ROOM7["x"][1] and pose[1] < DOOR_Y)
    labels = {}
    for l in rec["label"]:
        labels[l] = labels.get(l, 0) + 1
    out["trace"] = [[round(t, 2), *np.round(p, 3).tolist(), l, round(dd, 3)] for t, p, l, dd in zip(rec["t"], rec["pose"], rec["label"], rec["door_dist"])][::2]
    out.update(result=result, final_pose=pose.tolist(), base_centre_in_room7=inside, fp_centre_in_room7=inside_fp,
               bump_contacts=int(sim.bump_contacts), min_hull_clear=float(np.nanmin(r)) if np.isfinite(r).any() else None,
               min_wall_live=float(np.min(rec["wall_live"])), beads_out=int(sim.bead_check()), n_beads=int(sim.rig.n_beads),
               max_cup_tilt=float(np.max(rec["tilt"])), cup_lost=sim.cup_lost_t is not None, labels=labels,
               n_fallback=shield.n_fallback, fp_ms=(shield.fp_ms[0] / max(shield.fp_ms[1], 1)),
               min_person_dist=float(np.min(rec["person"])) if rec["person"] else None,
               sim_s=sim.t, wall_s=time.time() - t_start,
               success=bool(result == "ok" and inside and sim.bump_contacts == 0 and sim.bead_check() == 0 and not (sim.cup_lost_t is not None)
                            and (not np.isfinite(r).any() or np.nanmin(r) > 0.0)))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, f"{a.name}.json"), "w") as f:
            json.dump(out, f, indent=1, default=float)
    return out


class VideoRig:
    """headless MuJoCo renderer (EGL) -> mp4: left = fixed world camera over the door, right = ego (torso head camera)."""

    def __init__(self, sim, path, fps=15, size=(480, 360)):
        import cv2
        import mujoco
        from .web import render_proc as RP
        self.cv2, self.mj, self.RP = cv2, mujoco, RP
        self.m = sim.m
        W, H = size
        sim.m.vis.global_.offwidth, sim.m.vis.global_.offheight = max(W, 640), max(H, 480)
        self.r = mujoco.Renderer(sim.m, H, W)
        self.fovy0 = float(sim.m.vis.global_.fovy)
        self.cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(self.cam)
        self.cam2 = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(self.cam2)
        self.every = max(1, int(round(1.0 / (fps * 0.02))))
        self.vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * W, H))
        self.n = 0

    def frame(self, sim):
        m, d = self.m, sim.d
        m.vis.global_.fovy = self.fovy0
        c = self.cam
        c.lookat[:] = (4.0, 4.9, 0.6); c.distance, c.elevation, c.azimuth = 4.2, -50.0, 0.0     # bird's-eye over the door, looking along +x... (map frame)
        self.r.update_scene(d, camera=c)
        a = self.r.render().copy()
        m.vis.global_.fovy = self.RP.ego_cam(self.cam2, sim.base.x, sim.base.y, sim.base.yaw, sim.lift_actual())
        self.r.update_scene(d, camera=self.cam2)
        b = self.r.render().copy()
        m.vis.global_.fovy = self.fovy0
        img = np.concatenate([a, b], 1)
        self.cv2.putText(img, f"t={sim.t:5.1f}s", (8, 20), self.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return img

    def maybe(self, sim, k):
        if k % self.every == 0:
            self.vw.write(self.frame(sim)[:, :, ::-1])
            self.n += 1

    def close(self):
        self.vw.release()
        self.r.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--footprint", choices=["fixed", "live"], default="live")
    ap.add_argument("--tuck", action="store_true")
    ap.add_argument("--people", choices=["off", "on", "static"], default="off")
    ap.add_argument("--live-pad", dest="live_pad", type=float, default=0.02)
    ap.add_argument("--d-wall", dest="d_wall", type=float, default=None)
    ap.add_argument("--d-wall-door", dest="d_wall_door", type=float, default=None)
    ap.add_argument("--d-person", dest="d_person", type=float, default=None)
    ap.add_argument("--relax", action="store_true", help="enable ShieldCfg.relax_tier")
    ap.add_argument("--y-end", dest="y_end", type=float, default=4.2)
    ap.add_argument("--v-through", dest="v_through", type=float, default=0.25)
    ap.add_argument("--stall-s", dest="stall_s", type=float, default=12.0)
    ap.add_argument("--seg-timeout", dest="seg_timeout", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="door")
    ap.add_argument("--out", default="data/molmospaces/teleop_door")
    ap.add_argument("--video", default=None)
    ap.add_argument("--vfps", type=float, default=15.0)
    a = ap.parse_args(argv)
    o = run(a)
    print(json.dumps({k: v for k, v in o.items() if k not in ("args", "trace")}, indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
