"""flagship/render.py -- replay the two recorded runs (raw | shield) into the side-by-side flagship video.  SIMULATION ONLY.

    cd /common/users/ss5772/bimanual ; export PYTHONPATH=$PWD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<free gpu, not 0> $PY -m world_model.v2.house.flagship.render --seed 0 [--preview] [--out FILE]

No re-simulation: qpos / mocap / stand pose per tick come from run_raw_s<seed>.npz and run_shield_s<seed>.npz; the ZED panels from perception_*.pkl.
1920x1080, 25 fps.  Layout: header | two 960x580 chase views | per side: HUD (400 px) + minimap or ZED panel (560 px).
Sync: one shared sim clock t (both runs start at t=0); a side that finishes early holds its last frame ("DONE").  Time compression (x6) only where
neither side has anything going on (no person within 2 m, not near a door, drive phases only, no event within 3 s).  Freeze (1.6 s) on the first
spill / person contact / collision of either side (wall touches only in the HUD); the raw side then continues (its run does not end) so the end card covers the full mission.
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np

from . import scene as FS
from . import mission as MS
from world_model.v2.house.teleop import ghost as GH

TICK = 0.02
W, H, FPS = 1920, 1080, 25
HEAD_H = 70
VW, VH = 960, 580
BY = HEAD_H + VH                  # bottom row y
BH = H - BY                       # 430
HUD_W, MAP_W = 400, 560
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
RED, GREEN, AMBER, WHITE, GREY = (255, 90, 80), (90, 230, 130), (255, 190, 60), (240, 240, 245), (150, 155, 170)
BG = (12, 14, 22)
SPEED_C = 6.0
FREEZE_S = 1.6
PERC_S = 4.5
CARD_S = 3.0
END_S = 9.0
_fonts = {}


def font(sz, bold=True):
    k = (sz, bold)
    if k not in _fonts:
        from PIL import ImageFont
        _fonts[k] = ImageFont.truetype(FONT_B if bold else FONT_R, sz)
    return _fonts[k]


# ------------------------------------------------------------------------------------------------ run data
class Run:
    def __init__(self, tag, out=FS.OUT):
        z = np.load(Path(out) / f"run_{tag}.npz")
        self.a = {k: z[k] for k in z.files}
        j = json.loads((Path(out) / f"run_{tag}.json").read_text())
        self.meta, self.metrics = j["meta"], j["metrics"]
        self.tag, self.side = tag, self.meta["side"]
        self.n = len(self.a["s_t"])
        self.T = self.n * TICK
        self.ph = self.a["s_phase"]
        ana = Path(out) / f"run_{tag}_analysis.json"
        self.analysis = json.loads(ana.read_text()) if ana.exists() else None
        s = self.a
        held = s["s_grasped"] > 0
        acc = np.hypot(s["s_tray_acc"][:, 0], s["s_tray_acc"][:, 1])
        self.acc = np.where(held, acc, 0.0)
        self.acc_peak = np.maximum.accumulate(self.acc)
        auto = np.isin(self.ph, [MS.PHASE_ID[p] for p in MS.AUTONOMOUS + ("DONE",)])
        self.wall = np.where(auto, 9.0, s["s_wall"])            # footprint-vs-occupancy clearance is meaningless while the tray is placed over the desk
        self.wall_min = np.minimum.accumulate(self.wall)
        self.pd_min = np.minimum.accumulate(s["s_pdist"])
        self.pcont = np.cumsum((s["s_pdist"] <= 0) & (np.r_[False, s["s_pdist"][:-1] > 0]))
        self.wcont = np.cumsum((self.wall <= 0) & (np.r_[False, self.wall[:-1] > 0]))
        self.gh = None
        self._cam()

    def ghost_track(self, H=GH.H_LOOK):
        """operator-intent ghost per tick (teleop/ghost.py) from the recorded u_op / u_safe.  Base (+ lift) only: the 10-10 runs record the
        raw base twist but not the raw tray jog / lift rate, so the ghost arms ride with the ghost base at the measured joint angles."""
        s = self.a
        active = ~np.isin(self.ph, [MS.PHASE_ID[p] for p in MS.AUTONOMOUS + ("DONE",)])
        self.gh = GH.track(s["s_pose"], s["s_v"], s["s_u_op"], s["s_u_safe"], s["s_interv"], s["s_interv_aux"], active, H=H)
        vis = self.gh[:, 19] > 0
        print(f"[render] ghost {self.tag}: visible {vis.mean():.0%} of ticks, max gap {self.gh[:, 25].max() * 100:.0f} cm / "
              f"{math.degrees(self.gh[:, 26].max()):.0f} deg (H = {H} s)", flush=True)
        return self.gh

    def idx(self, t):
        return int(np.clip(round(t / TICK) - 1, 0, self.n - 1))

    def done_at(self, t):
        """t = this run's OWN sim time"""
        return t >= self.T

    # phase-aligned playback: reference clock (= the shield run's sim time) <-> this run's own sim time, piecewise linear through the milestones
    ref_k = None
    own_k = None

    def set_warp(self, ref_knots, own_knots):
        self.ref_k, self.own_k = np.asarray(ref_knots, float), np.asarray(own_knots, float)

    def own(self, t_ref):
        if self.ref_k is None:
            return t_ref
        k = self.ref_k
        if t_ref >= k[-1]:
            return float(self.own_k[-1] + (t_ref - k[-1]))
        return float(np.interp(t_ref, k, self.own_k))

    def ref(self, t_own):
        if self.ref_k is None:
            return t_own
        k = self.own_k
        if t_own >= k[-1]:
            return float(self.ref_k[-1] + (t_own - k[-1]))
        return float(np.interp(t_own, k, self.ref_k))

    def rate(self, t_ref, d=0.5):
        return (self.own(t_ref + d) - self.own(t_ref)) / d if self.ref_k is not None else 1.0

    def _cam(self):
        """smoothed camera parameters per tick: azimuth(from heading), distance, elevation."""
        yaw = np.unwrap(self.a["s_pose"][:, 2])
        al = 1 - math.exp(-TICK / 0.9)
        ys = np.empty_like(yaw)
        ys[0] = yaw[0]
        for i in range(1, len(yaw)):
            ys[i] = ys[i - 1] + al * (yaw[i] - ys[i - 1])
        self.cyaw = ys
        close = np.isin(self.ph, [MS.PHASE_ID[p] for p in ("SENSE_C", "RAISE", "CREEP", "ALIGN", "LIFT", "BACKOUT", "LOWER_C", "CREEP_D", "SENSE_D", "PLACE", "ABORT", "RELEASE", "DONE")])
        tgt = np.where(close, 1.0, 0.0)
        c = np.empty(len(tgt)); c[0] = tgt[0]
        al2 = 1 - math.exp(-TICK / 1.2)
        for i in range(1, len(tgt)):
            c[i] = c[i - 1] + al2 * (tgt[i] - c[i - 1])
        self.cdist = 3.5 + (2.7 - 3.5) * c
        self.cel = -56.0 + (-33.0 + 56.0) * c
        self.caz = -12.0 + (-35.0 + 12.0) * c

    # events ---------------------------------------------------------------------------------------------
    def events(self):
        s = self.a
        out = []
        t = s["s_t"]

        def first(mask):
            return int(np.argmax(mask)) if mask.any() else None
        i = first(s["s_beads_out"] > 0)
        if i is not None:
            out.append(dict(t=float(t[i]), kind="spill", i=i))
        i = first(s["s_pdist"] <= 0)
        if i is not None:
            out.append(dict(t=float(t[i]), kind="person", i=i))
        i = first(self.wall <= 0)
        if i is not None:
            out.append(dict(t=float(t[i]), kind="wall", i=i))
        i = first(s["s_bump_cum"] > 0)
        if i is not None:
            out.append(dict(t=float(t[i]), kind="collision", i=i))
        return out


# ------------------------------------------------------------------------------------------------ schedule
def interesting_mask(runs, door_xy=((8.1, 3.0), (17.9, 3.0)), dt=0.1, tmax=None):
    """bool per dt bin: something worth watching at 1x on any still-running side."""
    T = max(r.ref(r.T) for r in runs) if tmax is None else tmax
    nb = int(math.ceil(T / dt))
    m = np.zeros(nb, bool)
    ev_t = [r.ref(e["t"]) for r in runs for e in r.events()]
    for b in range(nb):
        t = (b + 0.5) * dt
        for r in runs:
            to = r.own(t)
            if r.done_at(to):
                continue
            i = r.idx(to)
            ph = MS.PHASES[r.ph[i]]
            pose = r.a["s_pose"][i]
            if ph in ("SENSE_C", "ALIGN", "LIFT", "SENSE_D", "PLACE", "ABORT", "RELEASE"):
                m[b] = True
            elif r.a["s_pdist"][i] < 1.3 or any(math.hypot(pose[0] - dx, pose[1] - dy) < 2.2 for dx, dy in door_xy):
                m[b] = True
            elif math.hypot(pose[0] - 1.8, pose[1] - 2.0) < 1.5 and to < 8.0:
                m[b] = True
        if any(abs(t - te) < 2.0 for te in ev_t):
            m[b] = True
    # grow by 1 s each side, then drop compress runs shorter than 3 s
    k = int(1.0 / dt)
    g = m.copy()
    for s in range(1, k + 1):
        g[:-s] |= m[s:]
        g[s:] |= m[:-s]
    comp = ~g
    idx = np.flatnonzero(np.diff(np.r_[0, comp.astype(int), 0]))
    for a, b in zip(idx[::2], idx[1::2]):
        if (b - a) * dt < 3.0:
            comp[a:b] = False
    return comp, dt


def build_schedule(runs, percs, cfg):
    """list of frame dicts.  kinds: card (name), sim (t, speed, banner, perc=(kind, p)), freeze (same as sim, t fixed)."""
    comp, dt = interesting_mask(runs)
    T = max(r.ref(r.T) for r in runs) + 1.0
    sched = []
    n_title, n_end = int(CARD_S * FPS), int(END_S * FPS)
    for i in range(n_title):
        sched.append(dict(kind="card", name="title", p=i / max(n_title - 1, 1), sec="title"))
    # one-shot windows keyed by sim time
    wins = []
    shield = next(r for r in runs if r.side == "shield")
    raw = next(r for r in runs if r.side == "raw")
    for kind, ph_name in (("counter", "SENSE_C"), ("desk", "SENSE_D")):
        if kind in percs:
            ts = [e for e, p in zip(*[[x[0] for x in shield.meta["events"]], [x[1] for x in shield.meta["events"]]]) if p == ph_name]
            if ts:
                wins.append(dict(t=float(ts[0]) + 0.1, kind="perc", which=kind))
    for r in runs:
        for e in r.events():
            if e["kind"] == "wall":
                continue                       # wall touches are counted in the HUD / end card, no freeze (spill / person / collision freeze)
            if r.side == "raw" or cfg.get("freeze_shield_events", True):
                wins.append(dict(t=r.ref(e["t"]), kind="event", side=r.side, ev=e))
    # keep the earliest freeze per (side, kind); merge freezes closer than 1 s
    wins.sort(key=lambda w: w["t"])
    t = 0.0
    wi = 0
    speed_prev = 1.0
    while t < T:
        i_b = min(int(t / dt), len(comp) - 1)
        tgt = SPEED_C if comp[i_b] else 1.0
        speed_prev += (tgt - speed_prev) * 0.08
        sp = speed_prev if abs(speed_prev - tgt) > 0.02 else tgt
        step = sp / FPS
        while wi < len(wins) and wins[wi]["t"] <= t + step:
            w = wins[wi]
            wi += 1
            if w["kind"] == "perc":
                sec = f"{w['which']}_perc"
                for k in range(int(PERC_S * FPS)):
                    sched.append(dict(kind="sim", t=w["t"], speed=0.0, perc=(w["which"], k / (PERC_S * FPS - 1)), sec=sec))
            else:
                ev = w["ev"]
                sec = f"event_{w['side']}_{ev['kind']}"
                for k in range(int(FREEZE_S * FPS)):
                    sched.append(dict(kind="sim", t=w["t"], speed=0.0, banner=(w["side"], ev["kind"], ev["i"]), sec=sec, fz=k / (FREEZE_S * FPS)))
        sched.append(dict(kind="sim", t=t, speed=sp, sec="drive_compressed" if sp > 1.5 else "sim"))
        t += step
    for i in range(n_end):
        sched.append(dict(kind="card", name="end", p=i / max(n_end - 1, 1), sec="end"))
    return sched


# ------------------------------------------------------------------------------------------------ minimap
class MiniMap:
    S = 40.0                                     # px per metre

    def __init__(self, sim, meta):
        import cv2
        occ = sim.occ
        self.cv2 = cv2
        x0, x1, y0, y1 = -0.5, 28.6, -1.5, 7.6
        self.x0, self.y1 = x0, y1
        Wp, Hp = int((x1 - x0) * self.S), int((y1 - y0) * self.S)
        img = np.full((Hp, Wp, 3), (26, 28, 36), np.uint8)
        for nm, col in (("lab", (50, 58, 70)), ("corridor", (60, 56, 46)), ("cafeteria", (66, 54, 44))):
            poly = np.array([[(x - x0) * self.S, (y1 - y) * self.S] for x, y in meta["rooms"][nm]], np.int32)
            cv2.fillPoly(img, [poly], col)
        # occupancy (1 cm grid -> map): max-pool via resize of the bool grid
        ny, nx = occ.occ.shape
        gx0, gy0 = occ.x0, occ.y0
        full = np.zeros((Hp, Wp), np.uint8)
        occu = occ.occ.astype(np.uint8)
        # grid pixel (ix,iy) -> world x=gx0+ix*res ; resample by integer stride
        stride = max(int(round(1.0 / (self.S * occ.res))), 1)
        sub = occu[::1, ::1]
        ys, xs = np.nonzero(sub)
        wx = gx0 + (xs + 0.5) * occ.res
        wy = gy0 + (ys + 0.5) * occ.res
        px = ((wx - x0) * self.S).astype(int); py = ((y1 - wy) * self.S).astype(int)
        ok = (px >= 0) & (px < Wp) & (py >= 0) & (py < Hp)
        full[py[ok], px[ok]] = 255
        full = cv2.dilate(full, np.ones((2, 2), np.uint8))
        img[full > 0] = (205, 205, 215)
        self.img = img
        self.meta = meta

    def w2p(self, x, y):
        return int((x - self.x0) * self.S), int((self.y1 - y) * self.S)

    def draw(self, pose, trail, people, wps, w=MAP_W, h=BH, zonos=None, extra=None):
        cv2 = self.cv2
        cx, cy = self.w2p(pose[0], pose[1])
        x0, y0 = cx - w // 2, cy - h // 2
        pad = np.full((h, w, 3), (26, 28, 36), np.uint8)
        sx0, sy0 = max(x0, 0), max(y0, 0)
        sx1, sy1 = min(x0 + w, self.img.shape[1]), min(y0 + h, self.img.shape[0])
        if sx1 > sx0 and sy1 > sy0:
            pad[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = self.img[sy0:sy1, sx0:sx1]
        P = lambda x, y: (self.w2p(x, y)[0] - x0, self.w2p(x, y)[1] - y0)
        for name, (wx, wy, col) in wps.items():
            p = P(wx, wy)
            cv2.circle(pad, p, 7, col, -1, cv2.LINE_AA)
            cv2.putText(pad, name, (p[0] + 9, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        if zonos:
            for (lo, hi, col) in zonos:
                a, b = P(lo[0], hi[1]), P(hi[0], lo[1])
                cv2.rectangle(pad, a, b, col, 1, cv2.LINE_AA)
        for (px, py, r) in people:
            cv2.circle(pad, P(px, py), max(int(r * self.S), 3), (80, 200, 120), -1, cv2.LINE_AA)
        if len(trail) > 1:
            pts = np.array([P(x, y) for x, y in trail], np.int32)
            cv2.polylines(pad, [pts], False, (255, 210, 90), 2, cv2.LINE_AA)
        # footprint
        from world_model.v2.house.teleop import geom as GM
        fp = GM.FP
        R = GM.rot(pose[2])
        corners = np.array([[fp["x"][0], fp["y"][0]], [fp["x"][1], fp["y"][0]], [fp["x"][1], fp["y"][1]], [fp["x"][0], fp["y"][1]]])
        wc = corners @ R.T + pose[:2]
        cv2.polylines(pad, [np.array([P(x, y) for x, y in wc], np.int32)], True, (90, 170, 255), 2, cv2.LINE_AA)
        hd = np.array([math.cos(pose[2]), math.sin(pose[2])]) * 0.5 + pose[:2]
        cv2.line(pad, P(pose[0], pose[1]), P(hd[0], hd[1]), (90, 170, 255), 2, cv2.LINE_AA)
        if extra:
            extra(pad, P)
        return pad


# ------------------------------------------------------------------------------------------------ 3-D views
class Views:
    def __init__(self, sim, percs, gpu_ok=True):
        import mujoco
        self.mj = mujoco
        self.sim, self.m = sim, sim.m
        self.d = mujoco.MjData(self.m)
        self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, VW)
        self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, VH)
        self.r = mujoco.Renderer(self.m, VH, VW, max_geom=40000)
        self.cam = mujoco.MjvCamera()
        self.stand_id = self.m.body("tp_stand").id
        self.percs = percs
        self.ghost = GH.Drawer(self.m)
        from world_model.v2.house.zed_demo import viz as VZ
        self.VZ = VZ
        self.shown = {}
        for kind, r in percs.items():
            from .perception import zono_unpack
            lst = []
            ci = 0
            for z in r["zonos"]:
                if z["kind"] != "obj" or z["n_pts"] < 250 or z["gt_key"] in ("wall", "floor", "counter", "desk_1"):
                    continue
                lst.append(dict(Zi=zono_unpack(z["Zi"]), color=VZ.color_of(ci), label=z["gt_key"], z=z))
                ci += 1
            self.shown[kind] = lst
            if kind == "desk":
                sup = [z for z in r["zonos"] if z["kind"] == "support"]
                self.support = zono_unpack(sup[0]["Zi"]) if sup else None

    def render(self, run: Run, i, which_zono=None, plan=None, naive=None, ghost=None):
        mj, m, d = self.mj, self.m, self.d
        a = run.a
        d.qpos[:] = a["qpos"][i]
        d.mocap_pos[:] = a["mocap_pos"][i]
        d.mocap_quat[:] = a["mocap_quat"][i]
        st = a["stand"][i]
        m.body_pos[self.stand_id] = st[:3] if st[3] > 0 else FS.STAND_PARK
        mj.mj_forward(m, d)
        pose = a["s_pose"][i]
        yaw = run.cyaw[i]
        cam = self.cam
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = (pose[0] + 0.35 * math.cos(yaw), pose[1] + 0.35 * math.sin(yaw), 0.80)
        cam.distance, cam.elevation, cam.azimuth = float(run.cdist[i]), float(run.cel[i]), math.degrees(yaw) + float(run.caz[i])
        self.r.update_scene(d, camera=cam)
        scn = self.r.scene
        if ghost is not None:
            self.ghost.add(scn, d, ghost, z_arrow=GH.Z_ARROW_CHASE)
        VZ = self.VZ
        if which_zono:
            for zd in self.shown.get(which_zono, []):
                VZ.add_zono_scene(scn, zd["Zi"], zd["color"])
            if which_zono == "desk" and getattr(self, "support", None) is not None:
                VZ.add_zono_scene(scn, self.support, (230, 210, 120), fill=0.08, width=0.003)
        if plan is not None:
            VZ.add_polyline(scn, plan["line"], [0.1, 1.0, 0.35, 0.95], 0.007)
            VZ.add_tray_ghost(scn, plan["spot"][:2], plan["spot"][2], plan["yaw"], [0.1, 1.0, 0.35, 0.35])
        if naive is not None:
            VZ.add_polyline(scn, naive["line"], [1.0, 0.15, 0.1, 0.95], 0.007)
            VZ.add_tray_ghost(scn, naive["spot"][:2], naive["spot"][2], naive["yaw"], [1.0, 0.15, 0.1, 0.35])
        return self.r.render().copy()


# ------------------------------------------------------------------------------------------------ text helpers
def draw_text(dr, xy, text, sz=22, fill=WHITE, bold=True, anchor="la"):
    dr.text(xy, text, font=font(sz, bold), fill=fill, anchor=anchor)


def bar(dr, x, y, w, h, frac, col, back=(40, 44, 58), mark=None):
    dr.rectangle([x, y, x + w, y + h], fill=back)
    f = float(np.clip(frac, 0, 1))
    dr.rectangle([x, y, x + int(w * f), y + h], fill=col)
    if mark is not None:
        mx = x + int(w * np.clip(mark, 0, 1))
        dr.line([mx, y - 3, mx, y + h + 3], fill=WHITE, width=2)


def sparkline(dr, x, y, w, h, vals, vmax, col, bound=None):
    dr.rectangle([x, y, x + w, y + h], fill=(22, 25, 36))
    if bound is not None:
        by = y + h - int(h * min(bound / vmax, 1))
        dr.line([x, by, x + w, by], fill=(120, 120, 140), width=1)
    if len(vals) < 2:
        return
    v = np.clip(np.asarray(vals) / vmax, 0, 1)
    pts = [(x + int(w * k / (len(v) - 1)), y + h - int(h * q)) for k, q in enumerate(v)]
    dr.line(pts, fill=col, width=2)


EV_TEXT = dict(spill="SPILL", person="HIT A PERSON", wall="HIT A WALL", collision="COLLISION")


# ------------------------------------------------------------------------------------------------ frame composition
class Composer:
    def __init__(self, runs, percs, views, minimap, cfg):
        self.raw = next(r for r in runs if r.side == "raw")
        self.shd = next(r for r in runs if r.side == "shield")
        self.runs = runs
        self.percs, self.views, self.mm, self.cfg = percs, views, minimap, cfg
        self.sim = views.sim
        meta = minimap.meta
        wp = meta["waypoints"]
        self.wps = {"door1": (*wp["door1"]["xy"], (240, 215, 60)), "door2": (*wp["door2"]["xy"], (240, 215, 60)),
                    "counter": (*wp["cafe_counter_pick"]["xy"], (80, 220, 120)), "desk": (*wp["lab_desk_place"]["xy"], (255, 140, 60)),
                    "lab": (*wp["lab_start"]["xy"], (90, 140, 255))}
        self.static_people = [(p["xy"][0], p["xy"][1], p["radius"]) for p in meta["persons"]]
        self.pnames = [self.sim.m.body(int(np.flatnonzero(self.sim.m.body_mocapid == i)[0])).name for i in range(self.sim.m.nmocap)]
        self.place_vis = {}
        self.gf = {}
        for r in runs:
            pi = r.meta.get("place_info", {})
            if pi.get("spot_t") is not None:
                tray = r.a["s_tray_pos"]
                b0 = np.array(pi["base0"])
                st = r.a["stand"]
                sidx = np.flatnonzero(st[:, 3] > 0)
                if len(sidx):
                    s = st[sidx[0]]
                    line = np.array([[*xy, 0.86] for xy in self._path_xy(r, pi)])
                    self.place_vis[r.side] = dict(spot=(s[0], s[1], s[2] + 0.003 + 0.006), yaw=float(r.a["s_pose"][sidx[0], 2] + math.pi / 2), line=line, t0=float(r.a["s_t"][sidx[0]]))

    def _path_xy(self, r, pi):
        # world path of the tray centre during PLACE: recorded tray positions in the PLACE phase
        i0 = np.flatnonzero(r.ph == MS.PHASE_ID["PLACE"])
        if len(i0) == 0:
            return []
        return r.a["s_tray_pos"][i0[::10], :2]

    # --------------------------------------------------------------------------------------- side panels
    def side_images(self, run: Run, t, perc=None):
        t = run.own(t)
        i = run.idx(t)
        a = run.a
        done = run.done_at(t)
        zono_kind = None
        ph = MS.PHASES[run.ph[i]]
        plan = naive = None
        if run.side == "shield":
            if "counter" in self.percs and run.meta["events"]:
                tc = next((e[0] for e in run.meta["events"] if e[1] == "SENSE_C"), None)
                tb = next((e[0] for e in run.meta["events"] if e[1] == "BACK"), None)
                if tc is not None and t >= tc and (tb is None or t < tb):
                    zono_kind = "counter"
            td = next((e[0] for e in run.meta["events"] if e[1] == "SENSE_D"), None)
            if "desk" in self.percs and td is not None and t >= td:
                zono_kind = "desk"
                pv = self.place_vis.get("shield")
                if pv is not None and t >= pv["t0"] - 0.1:
                    plan = pv
        else:
            tp_ = self.place_vis.get("raw")
            td = next((e[0] for e in run.meta["events"] if e[1] == "SENSE_D"), None)
            if tp_ is not None and td is not None and t >= td:
                naive = tp_
        gf = GH.GhostFrame.from_vec(run.gh[i]) if (run.gh is not None and not done) else None
        self.gf[run.side] = gf
        view = self.views.render(run, i, zono_kind, plan, naive, ghost=gf)
        return view, i, ph, t

    @staticmethod
    def fmt_cm(v):
        return ">5 m" if v > 5.0 else f"{v * 100:.0f} cm"

    def hud(self, run: Run, i, ph, t, dr, x0, y0, banner_on=False):
        a = run.a
        done = run.done_at(t)
        col_side = GREEN if run.side == "shield" else RED
        draw_text(dr, (x0 + 12, y0 + 8), f"phase  {ph if not done else 'DONE'}", 22, GREY)
        draw_text(dr, (x0 + HUD_W - 14, y0 + 8), f"t = {min(t, run.T):.0f} s", 22, WHITE, anchor="ra")
        nb = int(a["s_beads_out"][i])
        lost = a["s_cup_lost"][i] > 0
        draw_text(dr, (x0 + 12, y0 + 40), "BEADS SPILLED", 20, GREY)
        draw_text(dr, (x0 + 12, y0 + 62), f"{nb} / 8", 54, RED if nb > 0 else GREEN)
        draw_text(dr, (x0 + 200, y0 + 78), "CUP LOST" if lost else "cup on tray" if a["s_grasped"][i] > 0 else "cup on ledge", 24, RED if lost else GREY)
        acc = float(run.acc[i])
        draw_text(dr, (x0 + 12, y0 + 130), f"tray accel {acc:4.1f} m/s2   peak {run.acc_peak[i]:4.1f}", 21, WHITE)
        bar(dr, x0 + 12, y0 + 160, HUD_W - 30, 12, acc / 8.0, RED if acc > 1.41 else GREEN, mark=1.41 / 8.0)
        draw_text(dr, (x0 + 12, y0 + 176), "mark = 1.41 m/s2 no-spill bound", 14, GREY, False)
        wcl, pcl = float(run.wall[i]), float(a["s_pdist"][i])
        draw_text(dr, (x0 + 12, y0 + 202), f"wall/obstacle  {self.fmt_cm(wcl)}  (min {self.fmt_cm(run.wall_min[i])})", 19, RED if run.wall_min[i] <= 0 else WHITE)
        draw_text(dr, (x0 + 12, y0 + 228), f"person        {self.fmt_cm(pcl)}  (min {self.fmt_cm(run.pd_min[i])})", 19, RED if run.pd_min[i] <= 0 else WHITE)
        draw_text(dr, (x0 + 12, y0 + 254), f"contacts: person {int(run.pcont[i])}  wall {int(run.wcont[i])}  other {int(a['s_bump_cum'][i] > 0)}", 19, WHITE)
        if run.side == "shield":
            lab = run.meta["labels"][int(a["s_label"][i])]
            iv = float(a["s_interv"][i])
            on = iv > 0.02
            draw_text(dr, (x0 + 12, y0 + 288), "SHIELD ACTIVE" if on else "shield: monitoring", 24, AMBER if on else GREEN)
            draw_text(dr, (x0 + 12, y0 + 316), f"{lab if on else ''}", 18, AMBER)
            bar(dr, x0 + 12, y0 + 342, HUD_W - 30, 14, iv / 0.5, AMBER)
            draw_text(dr, (x0 + 12, y0 + 358), f"intervention |du|/Vmax = {iv:.2f}", 14, GREY, False)
        else:
            draw_text(dr, (x0 + 12, y0 + 288), "SHIELD OFF", 24, RED)
            draw_text(dr, (x0 + 12, y0 + 316), "operator sticks go straight to the base", 16, GREY, False)
        j0 = max(0, i - 400)
        sp = run.acc[j0:i + 1]
        sparkline(dr, x0 + 12, y0 + 384, HUD_W - 30, 36, sp[::4], 8.0, RED if run.side == "raw" else GREEN, bound=1.41)

    def side_panel(self, run: Run, t, perc, banner):
        return self.side_images(run, t, perc)

    # --------------------------------------------------------------------------------------- perception panel
    def perc_panel(self, which, p):
        import cv2
        from .perception import zono_unpack
        r = self.percs[which]
        left = r["left"]
        K, o, Rm = r["K"], r["origin"], r["R"]
        s = 500.0 / 1280.0
        img = cv2.resize(left, (500, int(720 * s)), interpolation=cv2.INTER_AREA).copy()
        shown = self.views.shown[which]
        nshow = int(round(np.clip((p - 0.40) / 0.30, 0, 1) * len(shown)))
        VZ = self.views.VZ
        for zd in shown[:nshow]:
            VZ.draw_zono_image(img, zd["Zi"], K, o, Rm, zd["color"], s, 1)
        for zd in shown[:nshow]:
            VZ.label_image(img, zd["label"].replace("_a", "").replace("_desk1", "").replace("_counter", ""), zd["Zi"], K, o, Rm, zd["color"], s)
        panel = np.full((BH, MAP_W, 3), (14, 16, 24), np.uint8)
        panel[:img.shape[0], 30:530] = img[:BH]
        # disparity thumbnail (wipe in at 0.2-0.4)
        from world_model.v2.house.zed_demo import stereo as ST
        dm = ST.colormap_disp(r["disp"].astype(np.float64))
        th_w = 205
        th = cv2.resize(dm, (th_w, int(th_w * 720 / 1280)), interpolation=cv2.INTER_NEAREST)
        wipe = float(np.clip((p - 0.15) / 0.2, 0, 1))
        y1 = img.shape[0] + 4
        tile = np.full_like(th, 20)
        k = int(th_w * wipe)
        tile[:, :k] = th[:, :k]
        panel[y1:y1 + th.shape[0], 4:4 + th_w] = tile[:BH - y1]
        return panel, y1, th_w, nshow, len(shown)

    def perc_text(self, which, p, dr, y1, th_w):
        r = self.percs[which]
        x = MAP_W + 0  # placeholder (text drawn relative to panel origin by caller)
        lines = []
        zs = [z for z in r["zonos"] if z["kind"] == "obj"]
        pe = r["ev_scene"].get("p95_abs", float("nan"))
        if which == "counter":
            ap = r["approach"]
            lines = [f"{len(zs)} stereo zonotopes (counter)", f"stereo p95 err {pe * 1000:.0f} mm (infl.)",
                     f"approach clearance >= {ap['approach']['min_clear_m'] * 100:.0f} cm", f"back-out clearance >= {ap['backout']['min_clear_m'] * 100:.0f} cm"]
            head = "ZED capture at the counter"
        else:
            pl = r["plan"]
            if pl.get("ok"):
                lines = [f"{len(zs)} zonotopes (desk clutter)", f"{pl['n_feasible']} feasible free spots",
                         f"planned path clearance {pl['path_min_exact_clear_m'] * 100:.1f} cm", f"naive centre spot: {pl['naive_exact_clear_m'] * 100:.0f} cm -> bottle"]
            head = "ZED capture at the desk"
        return head, lines

    # --------------------------------------------------------------------------------------- full frame
    def frame(self, f):
        import cv2
        from PIL import Image, ImageDraw
        if f["kind"] == "card":
            return self.card(f)
        t = f["t"]
        canvas = np.full((H, W, 3), BG, np.uint8)
        perc = f.get("perc")
        ban = f.get("banner")
        imgs = []
        for sidx, run in enumerate((self.raw, self.shd)):
            view, i, ph, to = self.side_panel(run, t, perc, ban)
            canvas[HEAD_H:HEAD_H + VH, sidx * VW:(sidx + 1) * VW] = view
            imgs.append((run, i, ph, to))
        # minimaps / perception
        for sidx, (run, i, ph, to) in enumerate(imgs):
            a = run.a
            x0 = sidx * VW
            trail = a["s_pose"][max(0, i - 500):i + 1:10, :2]
            people = list(self.static_people)
            for k, nm in enumerate(self.pnames):
                if nm.startswith("p_"):
                    people.append((float(a["mocap_pos"][i, k, 0]), float(a["mocap_pos"][i, k, 1]), 0.22))
            pan = self.mm.draw(a["s_pose"][i], trail, people, self.wps)
            if run.side == "shield" and perc is not None:
                pan, y1, th_w, nshow, nshown = self.perc_panel(perc[0], perc[1])
            canvas[BY:BY + BH, x0 + HUD_W:x0 + HUD_W + MAP_W] = pan[:BH, :MAP_W]
        pil = Image.fromarray(canvas)
        dr = ImageDraw.Draw(pil, "RGBA")
        # header
        dr.rectangle([0, 0, VW - 2, HEAD_H - 6], fill=(70, 24, 24))
        dr.rectangle([VW + 2, 0, W, HEAD_H - 6], fill=(18, 62, 36))
        draw_text(dr, (14, 6), "SHIELD OFF", 34, RED)
        draw_text(dr, (14, 44), "raw operator, same operator model + seed", 17, (230, 200, 200), False)
        draw_text(dr, (W - 16, 6), "SHIELD ON", 34, GREEN, anchor="ra")
        draw_text(dr, (W - 16, 44), "same operator + safety shield + stereo zonotopes", 17, (200, 230, 210), False, anchor="ra")
        sp = f.get("speed", 1.0)
        draw_text(dr, (W // 2, 8), "sync: same mission stage", 18, GREY, False, anchor="ma")
        draw_text(dr, (W // 2, 36), (f"x{sp:.0f} TIME-LAPSE" if sp > 1.5 else "paused" if sp == 0 else "real time (x1)"), 22, AMBER if sp > 1.5 or sp == 0 else GREY, anchor="ma")
        for sidx, (run, i, ph, to) in enumerate(imgs):
            x0 = sidx * VW
            self.hud(run, i, ph, to, dr, x0, BY)
            rate = run.rate(t)
            if abs(rate - 1.0) > 0.03 and sp > 0 and not run.done_at(to):
                draw_text(dr, (x0 + VW - 14, HEAD_H + 8), f"playback x{rate * sp:.2f}" if sp > 1.5 else f"playback x{rate:.2f}", 18, AMBER, anchor="ra")
            if run.done_at(to):
                dr.rectangle([x0 + 20, HEAD_H + 14, x0 + 330, HEAD_H + 56], fill=(0, 0, 0, 170))
                draw_text(dr, (x0 + 30, HEAD_H + 20), f"MISSION DONE  {run.metrics.get('mission_time_s') or run.T:.0f} s", 24, GREEN)
            draw_text(dr, (x0 + 12, HEAD_H + VH - 28), f"3rd-person view   {ph}", 17, (230, 230, 235), False)
            gf = self.gf.get(run.side)
            if gf is not None and gf.alpha > 0:
                self.ghost_legend(dr, gf, x0 + VW - 492, HEAD_H + VH - 132)
        # perception text
        if perc is not None:
            head, lines = self.perc_text(perc[0], perc[1], dr, 0, 0)
            x0 = VW + HUD_W
            r = self.percs[perc[0]]
            y1 = int(720 * 500.0 / 1280) + 4
            draw_text(dr, (x0 + 220, BY + y1 + 2), head, 18, (150, 200, 255))
            for k, ln in enumerate(lines):
                draw_text(dr, (x0 + 220, BY + y1 + 28 + 24 * k), ln, 16, WHITE, False)
            draw_text(dr, (x0 + 10, BY + 8), "ZED 2 stereo (sim): left eye + zonotopes", 17, (255, 255, 255))
            raw_x = HUD_W
            draw_text(dr, (raw_x + 20, BY + 12), "no stereo stack on this side", 20, GREY)
            if perc[0] == "desk":
                draw_text(dr, (raw_x + 20, BY + 40), "operator aims at the desk centre", 18, RED)
        # banners
        if ban is not None:
            side, kind, ei = ban
            run = self.raw if side == "raw" else self.shd
            x0 = 0 if side == "raw" else VW
            msg = EV_TEXT.get(kind, kind.upper())
            if kind == "spill":
                nbo = int(run.a['s_beads_out'][ei])
                msg = f"SPILL: {nbo} bead{'s' if nbo != 1 else ''} out"
            colb = (200, 30, 30, 220) if side == "raw" else (230, 150, 20, 220)
            dr.rectangle([x0 + 40, HEAD_H + 200, x0 + VW - 40, HEAD_H + 290], fill=colb)
            draw_text(dr, (x0 + VW // 2, HEAD_H + 215), f"{'RAW' if side == 'raw' else 'SHIELD'}: {msg}", 44, WHITE, anchor="ma")
        pil.save  # noqa
        return np.asarray(pil)

    @staticmethod
    def ghost_legend(dr, gf, x, y):
        tx = GH.describe(gf)
        col = (255, 150, 40) if gf.warn else (80, 205, 255)
        dr.rectangle([x, y, x + 482, y + 122], fill=(0, 0, 0, 175))
        draw_text(dr, (x + 10, y + 6), "GHOST = operator's command before the shield", 16, col)
        draw_text(dr, (x + 10, y + 28), f"({GH.H_LOOK:.1f} s look-ahead; the solid robot runs the shield's command)", 13, GREY, False)
        draw_text(dr, (x + 10, y + 50), f"requested  {tx['requested']}", 16, (255, 150, 40), False)
        draw_text(dr, (x + 10, y + 72), f"permitted  {tx['permitted']}", 16, (60, 235, 110), False)
        draw_text(dr, (x + 10, y + 96), tx["gap"] + "   (arrows: orange req., green perm.)", 14, WHITE, False)

    # --------------------------------------------------------------------------------------- cards
    def card(self, f):
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (W, H), BG)
        dr = ImageDraw.Draw(img)
        if f["name"] == "title":
            draw_text(dr, (W // 2, 300), "Coffee run, teleoperated", 76, WHITE, anchor="ma")
            draw_text(dr, (W // 2, 410), "language-free version: a sloppy operator,", 44, (200, 215, 240), False, anchor="ma")
            draw_text(dr, (W // 2, 470), "a safety shield, stereo zonotopes", 44, (200, 215, 240), False, anchor="ma")
            draw_text(dr, (W // 2, 640), "left: shield off      right: shield on      same operator, same seed", 30, GREY, False, anchor="ma")
            draw_text(dr, (W // 2, 1000), "Simulation", 26, GREY, False, anchor="ma")
        else:
            self.end_card(dr)
        return np.asarray(img)

    def end_card(self, dr):
        R, S = self.raw.metrics, self.shd.metrics
        Ra, Sa = self.raw.analysis or {}, self.shd.analysis or {}
        draw_text(dr, (W // 2, 50), "Result: same operator, with and without the shield", 50, WHITE, anchor="ma")
        draw_text(dr, (330, 150), "metric", 26, GREY)
        draw_text(dr, (1000, 150), "SHIELD OFF", 30, RED, anchor="ma")
        draw_text(dr, (1500, 150), "SHIELD ON", 30, GREEN, anchor="ma")
        f1 = lambda v, u="": "n/a" if v is None else f"{v:.1f}{u}"
        zc = lambda A: (None if A.get("min_zonotope_clearance_m") is None else A["min_zonotope_clearance_m"] * 100)
        rows = [
            ("beads spilled", f"{R['beads_spilled']} / {R['n_beads']}", f"{S['beads_spilled']} / {S['n_beads']}", R["beads_spilled"] > 0, S["beads_spilled"] > 0),
            ("cup lost", "yes" if R["cup_lost"] else "no", "yes" if S["cup_lost"] else "no", R["cup_lost"], S["cup_lost"]),
            ("person contacts (episodes)", f"{R['person_contact_episodes']}", f"{S['person_contact_episodes']}", R["person_contact_episodes"] > 0, S["person_contact_episodes"] > 0),
            ("wall contacts (episodes)", f"{R['wall_contact_episodes']}", f"{S['wall_contact_episodes']}", R["wall_contact_episodes"] > 0, S["wall_contact_episodes"] > 0),
            ("physical collisions (arm / tray / base)", f"{R.get('physical_collision_episodes', 0)}", f"{S.get('physical_collision_episodes', 0)}", R.get("physical_collision_episodes", 0) > 0, S.get("physical_collision_episodes", 0) > 0),
            ("min wall clearance", f"{R['min_wall_clearance_m'] * 100:.0f} cm", f"{S['min_wall_clearance_m'] * 100:.0f} cm", R["min_wall_clearance_m"] <= 0, S["min_wall_clearance_m"] <= 0),
            ("min person clearance", Composer.fmt_cm(R['min_person_clearance_m']), Composer.fmt_cm(S['min_person_clearance_m']), R["min_person_clearance_m"] <= 0, S["min_person_clearance_m"] <= 0),
            ("peak tray accel (carrying)", f"{f1(R['peak_tray_accel_ms2_held'], ' m/s2')}", f"{f1(S['peak_tray_accel_ms2_held'], ' m/s2')}", (R["peak_tray_accel_ms2_held"] or 0) > 1.41, (S["peak_tray_accel_ms2_held"] or 0) > 1.41),
            ("mission time", f"{f1(R['mission_time_s'], ' s')}" + ("" if R.get("placed_on_stand") else " (failed)"), f"{f1(S['mission_time_s'], ' s')}" + ("" if S.get("placed_on_stand") else " (failed)"), not R.get("placed_on_stand"), not S.get("placed_on_stand")),
            ("shield interventions", "-", f"{S['intervention_episodes']} ({100 * S['intervention_fraction_gt2pct']:.0f}% of ticks)", False, False),
            ("min clearance to stereo zonotopes", f"{f1(zc(Ra), ' cm')}", f"{f1(zc(Sa), ' cm')}", (zc(Ra) is not None and zc(Ra) <= 0.5), (zc(Sa) is not None and zc(Sa) <= 0.5)),
            ("tray placed on the free desk spot", "yes" if R.get("placed_on_stand") else "no", "yes" if S.get("placed_on_stand") else "no", not R.get("placed_on_stand"), not S.get("placed_on_stand")),
        ]
        y = 200
        for name, a, b, bad_a, bad_b in rows:
            draw_text(dr, (330, y), name, 28, WHITE, False)
            draw_text(dr, (1000, y), a, 30, RED if bad_a else (150, 235, 160), anchor="ma")
            draw_text(dr, (1500, y), b, 30, RED if bad_b else (150, 235, 160), anchor="ma")
            dr.line([300, y + 44, 1620, y + 44], fill=(40, 44, 58), width=1)
            y += 56
        draw_text(dr, (W // 2, 930), "Same closed-loop operator model and seed on both sides (each reacts to its own robot). Persons and walls are the same.", 20, GREY, False, anchor="ma")
        draw_text(dr, (W // 2, 960), "Zonotopes come from simulated stereo (SGBM) on a perception twin of the scene; raw side is measured against the same zonotopes.", 20, GREY, False, anchor="ma")
        draw_text(dr, (W // 2, 1010), "Simulation", 24, GREY, False, anchor="ma")


# ------------------------------------------------------------------------------------------------ main
def pick_preview(sched, shield=None, max_s=14.5):
    """<= max_s of video: slices of the key sections (title, time-lapse, first raw event, both perception windows, the planned placement, end card)."""
    secs = {}
    for k, f in enumerate(sched):
        secs.setdefault(f["sec"], []).append(k)
    plan = []

    def take(sec, frac, dur):
        ks = secs.get(sec)
        if ks:
            a = int(frac * len(ks))
            plan.extend(ks[a:a + int(dur * FPS)])
    take("title", 0.55, 0.6)
    take("drive_compressed", 0.0, 1.4)
    for sec in ("event_raw_spill", "event_raw_person", "event_raw_collision"):
        if sec in secs:
            take(sec, 0.0, 1.4)
            break
    take("counter_perc", 0.45, 2.6)
    take("desk_perc", 0.5, 2.4)
    if shield is not None:
        tp = next((e[0] for e in shield.meta["events"] if e[1] == "PLACE"), None)
        if tp is not None:
            ks = [k for k, f in enumerate(sched) if f["kind"] == "sim" and f["speed"] > 0 and tp + 1.0 <= f["t"] <= tp + 1.0 + 2.4]
            plan.extend(ks[:int(2.4 * FPS)])
    take("end", 0.4, 1.5)
    return [sched[k] for k in plan][: int(max_s * FPS)]


def set_milestone_warp(raw, shd, names=("SENSE_C", "BACK", "CREEP_D", "SENSE_D")):
    """phase-aligned playback: the reference clock is the shield run's sim time; the raw run is time-warped (piecewise linear) so that both reach the
    same mission stages at the same video time (the two operators are ~20 s apart otherwise).  Never more than the shield's own pace is imposed on it."""
    ev_s = {n: t for t, n in shd.meta["events"]}
    ev_r = {n: t for t, n in raw.meta["events"]}
    ks, ko = [0.0], [0.0]
    for n in names:
        if n in ev_s and n in ev_r and ev_s[n] > ks[-1] + 1.0 and ev_r[n] > ko[-1] + 1.0:
            ks.append(ev_s[n]); ko.append(ev_r[n])
    ks.append(shd.T); ko.append(raw.T)
    if ks[-1] > ks[-2] + 1.0 and ko[-1] > ko[-2] + 1.0:
        raw.set_warp(ks, ko)
        print("[render] raw playback warp (ref=shield t -> raw t):", [(round(a, 1), round(b, 1)) for a, b in zip(ks, ko)], flush=True)


def run_parallel(a):
    """spawn --jobs subprocesses (render.py --part i/N) round-robin over --gpus, then concat the parts with the ffmpeg binary bundled with imageio."""
    import subprocess
    import imageio_ffmpeg
    gpus = [g for g in a.gpus.split(",") if g] or ["1"]
    out = a.out or str(Path(a.dir) / default_name(a))
    procs = []
    for i in range(a.jobs):
        cmd = [sys.executable, "-m", "world_model.v2.house.flagship.render", "--seed", str(a.seed), "--dir", a.dir, "--out", out, "--part", f"{i}/{a.jobs}"]
        for k, v in (("--raw-tag", a.raw_tag), ("--shield-tag", a.shield_tag), ("--perception-dir", a.perception_dir)):
            if v:
                cmd += [k, v]
        if a.no_freeze_shield:
            cmd.append("--no-freeze-shield")
        if a.window:
            cmd += ["--window", a.window]
        if a.preview:
            cmd.append("--preview")
        if a.no_ghost:
            cmd.append("--no-ghost")
        cmd += ["--ghost-h", str(a.ghost_h)]
        gl = os.environ.get("MUJOCO_GL", "egl")
        env = dict(os.environ, MUJOCO_GL=gl, **({"CUDA_VISIBLE_DEVICES": gpus[i % len(gpus)]} if gl == "egl" else {}))
        procs.append(subprocess.Popen(cmd, env=env))
    rc = [p.wait() for p in procs]
    if any(rc):
        raise SystemExit(f"a render part failed: {rc}")
    parts = [out.replace(".mp4", f".part{i}.mp4") for i in range(a.jobs)]
    lst = out.replace(".mp4", ".parts.txt")
    Path(lst).write_text("".join(f"file '{p}'\n" for p in parts))
    subprocess.check_call([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out])
    print(f"[render] concatenated {a.jobs} parts -> {out}", flush=True)


def default_name(a):
    """the ghost video never overwrites the 10-10 one."""
    base = "flagship_preview" if a.preview else "flagship_coffee_run"
    return f"{base}.mp4" if a.no_ghost else f"{base}_ghost.mp4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dir", default=str(FS.OUT))
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--no-freeze-shield", action="store_true")
    ap.add_argument("--raw-tag", default=None)
    ap.add_argument("--shield-tag", default=None)
    ap.add_argument("--perception-dir", default=None)
    ap.add_argument("--window", default=None, help="t0,t1 sim-time window only (debug)")
    ap.add_argument("--dump-every", type=int, default=0, help="also write every Nth frame as PNG next to the output (debug)")
    ap.add_argument("--jobs", type=int, default=0, help="N>1: render N contiguous parts in parallel subprocesses (one per GPU in --gpus) and concat")
    ap.add_argument("--gpus", default="", help="comma list of GPU ids for --jobs (never 0)")
    ap.add_argument("--part", default=None, help="i/n: render only frames i::n chunk (contiguous), writes <out>.part<i>.mp4")
    ap.add_argument("--no-ghost", action="store_true", help="the 10-10 video without the operator-intent ghost")
    ap.add_argument("--ghost-h", type=float, default=GH.H_LOOK, help="ghost look-ahead [s] (display choice)")
    a = ap.parse_args()
    if a.jobs > 1:
        return run_parallel(a)
    d = Path(a.dir)
    raw = Run(a.raw_tag or f"raw_s{a.seed}", d)
    shd = Run(a.shield_tag or f"shield_s{a.seed}", d)
    pdir = Path(a.perception_dir or d / "perception")
    percs = {}
    for k in ("counter", "desk"):
        if (pdir / f"perception_{k}.pkl").exists():
            percs[k] = pickle.load(open(pdir / f"perception_{k}.pkl", "rb"))
    from world_model.v2.house.office import scene as OS
    meta = OS.office_meta()
    sim = FS.new_sim(a.seed, cafe_walker=raw.meta.get("walker", False))
    runs = [raw, shd]
    set_milestone_warp(raw, shd)
    for rn in runs:
        h = rn.meta.get("model_hash")
        if h is not None and h != sim.rig.info.get("model_hash"):
            print(f"[render] WARNING: model hash of {rn.tag} ({h}) != rebuilt model ({sim.rig.info.get('model_hash')}): replay may be mis-indexed", flush=True)
    cfg = dict(freeze_shield_events=not a.no_freeze_shield)
    sched = build_schedule(runs, percs, cfg)
    print(f"[render] schedule {len(sched)} frames = {len(sched) / FPS:.1f} s", flush=True)
    sections = {}
    for f in sched:
        sections[f["sec"]] = sections.get(f["sec"], 0) + 1
    print("[render] sections (frames):", sections, flush=True)
    if a.window:
        w0, w1 = (float(x) for x in a.window.split(","))
        sched = [f for f in sched if f["kind"] == "sim" and w0 <= f["t"] <= w1]
    if a.preview:
        sched = pick_preview(sched, shd)
        print(f"[render] preview {len(sched)} frames = {len(sched) / FPS:.1f} s", flush=True)
    if a.part:
        i, n = (int(x) for x in a.part.split("/"))
        L = int(math.ceil(len(sched) / n))
        sched = sched[i * L:(i + 1) * L]
    out = a.out or str(d / default_name(a))
    if a.part:
        out = out.replace(".mp4", f".part{a.part.split('/')[0]}.mp4")
    if not a.no_ghost:
        GH.H_LOOK = a.ghost_h
        shd.ghost_track(a.ghost_h)              # the raw side has no filter (u_safe = u_op): no ghost
    views = Views(sim, percs)
    mm = MiniMap(sim, meta)
    comp = Composer(runs, percs, views, mm, cfg)
    import imageio.v2 as imageio
    wr = imageio.get_writer(out, fps=FPS, codec="libx264", quality=8, macro_block_size=None, pixelformat="yuv420p")
    t0 = time.time()
    keyframes = {}
    for k, f in enumerate(sched):
        fr = comp.frame(f)
        wr.append_data(fr)
        if a.dump_every and k % a.dump_every == 0:
            import cv2
            Path(out).with_suffix("").mkdir(exist_ok=True)
            cv2.imwrite(str(Path(out).with_suffix("") / f"f{k:05d}_t{f.get('t', 0):.1f}.png"), fr[:, :, ::-1])
        if f["sec"] not in keyframes and f["kind"] != "card":
            keyframes[f["sec"]] = (k, fr)
        if k % 100 == 0:
            print(f"[render] {k}/{len(sched)}  {time.time() - t0:.0f}s", flush=True)
    wr.close()
    if a.preview:
        import cv2
        kd = Path(out).with_suffix("")
        kd.mkdir(exist_ok=True)
        for sec, (k, fr) in keyframes.items():
            cv2.imwrite(str(kd / f"key_{sec}.png"), fr[:, :, ::-1])
        for nm in ("title", "end"):
            fr = comp.card(dict(name=nm, p=1.0, kind="card", sec=nm))
            cv2.imwrite(str(kd / f"key_{nm}.png"), fr[:, :, ::-1])
    print(f"[render] wrote {out}  {len(sched)} frames in {time.time() - t0:.0f}s", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
