"""gen.py -- sharded, resumable generator of the pick-proposer LoRA dataset (SIM ONLY, offline; MuJoCo + EGL, no network, no robot, no VLM service).

  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=<g> CUDA_VISIBLE_DEVICES=<g> python -m world_model.v2.vlm.ftdata.gen --houses 2 3 5 --workers 2 --views-per-instance 2 --out DIR --limit 200

Per (house, free hand-sized instance, view): free MuJoCo camera (fovy 70) aimed near the object: elevation = pitch 36.3 +- 8 deg (uniform), azimuth uniform, distance chosen so the object
box is ~60-250 px tall (log-uniform target, distance clipped to [0.35, 2.4] m), object shifted off the image centre, light (model light + headlight) jittered.  The frame is 640x360,
the yellow box is drawn by pick.propose.draw_boxes exactly as session._crop_png does.  Robot frame = the head camera's horizontal frame anchored at HEAD_POS (x fwd, y left; centre line
y = -0.21).  Per-view RNG = crc32(seed, house, instance, view): the output is independent of --workers; resume skips sample ids already present in out/shards/*.jsonl.
Output: out/images/h<house>/h<house>_i<idx>_v<view>.png ; out/shards/h<house>_w<worker>.jsonl (Qwen chat records, one per line; arms task + ~30% name task).
Occlusion guard: the visible-mask bbox must cover >= 80 % of the projected-mesh bbox (both dims) and stay >= 6 px inside the frame.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import re
import time
import zlib
from pathlib import Path

import numpy as np

from . import labels as L
from . import priors as P

SCENE_DIR = Path(os.environ.get("HOUSE_DATA", "/data/local/ss5772/molmospaces/assets")) / "scenes" / "procthor-10k-val"
W, H = 640, 360
FOVY = 70.0
PITCH_DEG = 36.3
PITCH_JIT = 8.0
PX_RANGE = (60.0, 250.0)             # target box height [px]
PX_ACCEPT = (35.0, 320.0)
DIST_RANGE = (0.35, 2.4)
MARGIN_PX = 6
VIS_FRAC = 0.8
STRUCT = {"Doorway", "Window", "Painting", "Wall", "Floor", "Ceiling", "Door"}
MAX_DIM_M, MAX_SIM_MASS_KG, MIN_DIM_M = 0.5, 5.0, 0.02
P_NAME_TASK = 0.30
F_PX = (H / 2) / math.tan(math.radians(FOVY) / 2)


def seed_for(seed, house, inst, view):
    return zlib.crc32(f"{seed}|{house}|{inst}|{view}".encode()) & 0xFFFFFFFF


# ------------------------------------------------------------------------------------------------------------------ camera math (pure numpy)
def cam_axes(az_deg, el_deg):
    """MuJoCo free camera: forward (looks down for el < 0), right, up (world frame)."""
    az, el = math.radians(az_deg), math.radians(el_deg)
    fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    right = np.array([math.sin(az), -math.cos(az), 0.0])
    up = np.cross(right, fwd)
    return fwd, right, up


def project(Pw, cam_pos, fwd, right, up, f=F_PX, w=W, h=H):
    rel = Pw - cam_pos
    z = rel @ fwd
    return np.stack([w / 2 + f * (rel @ right) / z, h / 2 - f * (rel @ up) / z], 1), z


def make_cam(ctr, az, el, dist, ou, ov, f=F_PX):
    """lookat shifted so that the object centre lands (ou, ov) px right/down of the image centre.  -> dict(lookat, distance, azimuth, elevation, pos, fwd, right, up)."""
    fwd, right, up = cam_axes(az, el)
    look = ctr - (ou / f * dist) * right + (ov / f * dist) * up
    return dict(lookat=look, distance=dist, azimuth=az, elevation=el, pos=look - dist * fwd, fwd=fwd, right=right, up=up)


def robot_frame(ctr, cam):
    """object centre in the robot frame (head-camera horizontal frame anchored at HEAD_POS): x fwd, y left, z above the robot base plane."""
    rel = ctr - cam["pos"]
    xh = np.array([math.cos(math.radians(cam["azimuth"])), math.sin(math.radians(cam["azimuth"])), 0.0])
    yh = np.array([-xh[1], xh[0], 0.0])
    return np.array([L.HEAD_X + rel @ xh, L.HEAD_Y + rel @ yh, L.HEAD_Z + rel[2]])


def sample_cam(rng, ctr, Pw, maxdim):
    """-> cam dict or None.  Iterates the distance on the PROJECTED mesh bbox so that the box height hits a log-uniform target."""
    for _ in range(6):
        az = float(rng.uniform(0, 360))
        el = -(PITCH_DEG + float(rng.uniform(-PITCH_JIT, PITCH_JIT)))
        tgt = math.exp(rng.uniform(math.log(PX_RANGE[0]), math.log(PX_RANGE[1])))
        ou, ov = float(rng.uniform(-0.28 * W, 0.28 * W)), float(rng.uniform(-0.18 * H, 0.18 * H))
        dist = float(np.clip(F_PX * 0.9 * maxdim / tgt, *DIST_RANGE))
        for _it in range(3):
            cam = make_cam(ctr, az, el, dist, ou, ov)
            uv, z = project(Pw, cam["pos"], cam["fwd"], cam["right"], cam["up"])
            if z.min() < 0.05:
                break
            ph = float(uv[:, 1].max() - uv[:, 1].min())
            dist = float(np.clip(dist * ph / tgt, *DIST_RANGE))
        else:
            cam = make_cam(ctr, az, el, dist, ou, ov)
            uv, z = project(Pw, cam["pos"], cam["fwd"], cam["right"], cam["up"])
            if z.min() < 0.05:
                continue
            x0, y0, x1, y1 = uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()
            if x0 >= MARGIN_PX and y0 >= MARGIN_PX and x1 <= W - MARGIN_PX and y1 <= H - MARGIN_PX and PX_ACCEPT[0] <= y1 - y0 <= PX_ACCEPT[1]:
                cam["proj_bbox"] = [float(x0), float(y0), float(x1), float(y1)]
                cam["target_px"] = tgt
                return cam
    return None


# ------------------------------------------------------------------------------------------------------------------ box drawing (identical to session._crop_png)
def draw_selected_box(rgb, box2d):
    """RGB frame -> RGB frame with the planner's yellow selection box (pick.propose.draw_boxes, selected, no labels)."""
    from world_model.v2.house.pick import propose as PR
    return PR.draw_boxes(rgb, [dict(id=0, box2d=list(box2d), label="", mode="")], selected=0, labels=False)


def write_png(path, rgb):
    import cv2
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, rgb[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 4])      # same BGR flip as session._crop_png


# ------------------------------------------------------------------------------------------------------------------ house loading / instances
def load_house(h, scene_dir=SCENE_DIR):
    """val_<h>.xml with absolute mesh paths and the missing-on-disk textures dropped (-> plain colours).  -> (MjModel, n_textures_dropped)."""
    import mujoco
    s = (scene_dir / f"val_{h}.xml").read_text()
    names = set()
    for mt in re.finditer(r'<texture [^>]*name="([^"]+)"[^>]*file="([^"]+)"[^>]*/>', s):
        if not (scene_dir / mt.group(2)).resolve().exists():
            names.add(mt.group(1))
    s = re.sub(r'<texture [^>]*file="[^"]+"[^>]*/>\s*', lambda m: "" if any(f'name="{n}"' in m.group(0) for n in names) else m.group(0), s)
    for n in names:
        s = s.replace(f' texture="{n}"', "")
    s = re.sub(r'(<(?:mesh|texture|hfield)[^>]* )file="([^"]+)"', lambda mt: f'{mt.group(1)}file="{(scene_dir / mt.group(2)).resolve()}"', s)
    m = mujoco.MjModel.from_xml_string(s)
    m.vis.global_.fovy = FOVY
    m.vis.global_.offwidth = max(W, int(m.vis.global_.offwidth))
    m.vis.global_.offheight = max(H, int(m.vis.global_.offheight))
    m.vis.map.znear = 0.01
    return m, len(names)


def obb_from_points(Pw):
    """footprint min-area rectangle (a >= b) + height.  -> dict(a, b, h, ctr (3,), yaw_world [rad] of the a axis, aabb (3,))."""
    import cv2
    hull = cv2.convexHull(Pw[:, :2].astype(np.float32))
    (cx, cy), (w, hh), ang = cv2.minAreaRect(hull)
    if w < hh:
        w, hh, ang = hh, w, ang + 90.0
    zlo, zhi = float(Pw[:, 2].min()), float(Pw[:, 2].max())
    return dict(a=float(w), b=float(hh), h=zhi - zlo, ctr=np.array([cx, cy, 0.5 * (zlo + zhi)]), yaw_world=math.radians(ang), aabb=(Pw.max(0) - Pw.min(0)))


def enumerate_instances(m, d, md, table=None):
    """deterministic (metadata order) list of free hand-sized instances: dict(idx, body, bid, category, asset_id, mass_sim, obb, geoms (visual mesh geom ids)).
    -> (instances, unknown_categories)."""
    import mujoco
    table = table or P.load()
    kids = {}
    for b in range(1, m.nbody):
        kids.setdefault(int(m.body_parentid[b]), []).append(b)
    gbody = {}
    for g in range(m.ngeom):
        gbody.setdefault(int(m.geom_bodyid[g]), []).append(g)
    free_b = {int(m.jnt_bodyid[j]) for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}
    out, unknown = [], set()
    for k, v in md.items():
        cat = v["category"]
        if cat in STRUCT:                 # NB: metadata is_static is True for e.g. Books that DO have a free joint -> the freejoint test below decides
            continue
        try:
            bid = int(m.body(k).id)
        except KeyError:
            continue
        if bid not in free_b:
            continue
        stack, geoms = [bid], []
        while stack:
            b = stack.pop()
            geoms += [g for g in gbody.get(b, []) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 0]
            stack += kids.get(b, [])
        if not geoms:
            continue
        pts = []
        for g in geoms:
            a = int(m.geom_dataid[g])
            V = m.mesh_vert[m.mesh_vertadr[a]:m.mesh_vertadr[a] + m.mesh_vertnum[a]]
            pts.append(V @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        Pw = np.concatenate(pts)
        obb = obb_from_points(Pw)
        mass = float(m.body_subtreemass[bid])
        if obb["a"] > MAX_DIM_M or obb["h"] > MAX_DIM_M or mass >= MAX_SIM_MASS_KG or min(obb["b"], obb["h"]) < 0.004 or max(obb["a"], obb["b"], obb["h"]) < MIN_DIM_M:
            continue
        if cat not in table:
            unknown.add(cat)
            continue
        step = max(1, len(Pw) // 3000)
        out.append(dict(idx=len(out), body=k, bid=bid, category=cat, asset_id=v["asset_id"], mass_sim=mass, obb=obb, geoms=geoms, pts=Pw[::step].astype(np.float64)))
    return out, sorted(unknown)


# ------------------------------------------------------------------------------------------------------------------ lighting
class Lights:
    def __init__(self, m):
        self.m = m
        self.o = dict(ld=m.light_diffuse.copy(), la=m.light_ambient.copy(), lp=m.light_pos.copy(), hl_a=m.vis.headlight.ambient.copy(), hl_d=m.vis.headlight.diffuse.copy())

    def jitter(self, rng):
        m, o = self.m, self.o
        s = float(rng.uniform(0.55, 1.45))
        m.light_diffuse[:] = np.clip(o["ld"] * s * rng.uniform(0.9, 1.1, (m.nlight, 3)), 0, 1.5)
        m.light_ambient[:] = np.clip(o["la"] + rng.uniform(-0.05, 0.1, (m.nlight, 3)), 0, 1)
        m.light_pos[:] = o["lp"] + rng.normal(0, 0.6, (m.nlight, 3))
        ha, hd = float(rng.uniform(0.2, 0.5)), float(rng.uniform(0.2, 0.65))
        m.vis.headlight.ambient[:] = ha
        m.vis.headlight.diffuse[:] = hd
        return dict(light_scale=round(s, 3), headlight_ambient=round(ha, 3), headlight_diffuse=round(hd, 3))


# ------------------------------------------------------------------------------------------------------------------ resume
def done_ids(out):
    ids = set()
    for f in glob.glob(os.path.join(out, "shards", "*.jsonl")):
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:      # noqa: BLE001  (a line cut by a kill)
                    continue
                if r["meta"].get("task") == "arms" or r["meta"].get("sample_id", "").endswith("|arms"):
                    ids.add(r["meta"]["sample_id"].split("|")[0])
    return ids


# ------------------------------------------------------------------------------------------------------------------ worker
def make_renderers(m, mujoco):
    """(rgb renderer with the model's MSAA, segmentation renderer WITHOUT multisampling: MSAA blends the packed segment ids at silhouettes -> spurious ids)."""
    m.vis.quality.offsamples = 4
    Rc = mujoco.Renderer(m, H, W)
    m.vis.quality.offsamples = 0
    Rs = mujoco.Renderer(m, H, W)
    return Rc, Rs


def render_view(R, m, d, inst, cam, mujoco):
    Rc, Rs = R
    obj_lut = np.zeros(m.ngeom + 1, bool)
    obj_lut[np.array(inst["geoms"]) + 1] = True
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = cam["lookat"]
    c.distance, c.azimuth, c.elevation = cam["distance"], cam["azimuth"], cam["elevation"]
    Rc.update_scene(d, camera=c)
    rgb = Rc.render().copy()
    Rs.enable_segmentation_rendering()
    Rs.update_scene(d, camera=c)
    seg = Rs.render().copy()
    ids = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
    return rgb, obj_lut[ids + 1]


def worker(args):
    wid, nw, houses, out, vpi, limit_per_house, seed, policy, scene_dir = args
    import mujoco
    t_w = time.time()
    R = None
    n_new = 0
    table = P.load()
    names = P.display_names()
    done = done_ids(out)
    os.makedirs(os.path.join(out, "shards"), exist_ok=True)
    for house in houses:
        shard = os.path.join(out, "shards", f"h{house}_w{wid}.jsonl")
        m, ndrop = load_house(house, Path(scene_dir))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        md = json.loads((Path(scene_dir) / f"val_{house}_metadata.json").read_text())["objects"]
        insts, unknown = enumerate_instances(m, d, md, table)
        if unknown:
            print(f"[w{wid}] house {house}: categories with NO prior (skipped): {unknown}", flush=True)
        order = list(range(len(insts)))
        np.random.default_rng(zlib.crc32(f"{seed}|order|{house}".encode())).shuffle(order)
        # the per-house limit counts arms-samples over ALL workers: instance order is global, worker w takes positions w::nw; limit -> first ceil(limit/vpi) instances
        n_inst = len(order) if limit_per_house is None else min(len(order), math.ceil(limit_per_house / vpi))
        mine = order[:n_inst][wid::nw]
        print(f"[w{wid}] house {house}: {len(insts)} instances, {len(mine)} mine, {ndrop} textures dropped", flush=True)
        if R is not None:
            R[0].close()
            R[1].close()
        R = make_renderers(m, mujoco)
        lights = Lights(m)
        fh = open(shard, "a")
        for ii in mine:
            inst = insts[ii]
            for view in range(vpi):
                sid = f"h{house}_i{inst['idx']:04d}_v{view}"
                if sid in done:
                    continue
                rng = np.random.default_rng(seed_for(seed, house, inst["idx"], view))
                obb = inst["obb"]
                maxdim = max(obb["a"], obb["b"], obb["h"])
                rec = None
                for attempt in range(4):
                    cam = sample_cam(rng, obb["ctr"], inst["pts"], maxdim)
                    if cam is None:
                        continue
                    lt = lights.jitter(rng)
                    rgb, mask = render_view(R, m, d, inst, cam, mujoco)
                    npx = int(mask.sum())
                    if npx < 120:
                        continue
                    ys, xs = np.nonzero(mask)
                    bx = [int(xs.min()) - 2, int(ys.min()) - 2, int(xs.max()) + 2, int(ys.max()) + 2]      # same convention as propose.proposals_from_frame
                    pb = cam["proj_bbox"]
                    mw, mh = bx[2] - bx[0], bx[3] - bx[1]
                    if mw < VIS_FRAC * (pb[2] - pb[0]) or mh < VIS_FRAC * (pb[3] - pb[1]) or bx[0] < MARGIN_PX - 2 or bx[1] < MARGIN_PX - 2 or bx[2] > W - MARGIN_PX + 2 or bx[3] > H - MARGIN_PX + 2:
                        continue
                    rec = (cam, lt, rgb, bx, npx, attempt)
                    break
                if rec is None:
                    continue
                cam, lt, rgb, bx, npx, attempt = rec
                cat = inst["category"]
                prior = table[cat]
                vol = obb["a"] * obb["b"] * obb["h"]
                mrange = P.mass_range(cat, vol, table)
                a2, b2, h2 = L.noisy_extent(rng, obb["a"], obb["b"], obb["h"])
                cen = robot_frame(obb["ctr"], cam)
                cen = [round(float(cen[0]), 4), round(float(cen[1]), 4), round(float(cen[2]), 4)]
                yaw_r = round(float((obb["yaw_world"] - math.radians(cam["azimuth"]) + math.pi / 2) % math.pi - math.pi / 2), 4)
                label, lkind = L.pick_label(rng, cat, table)
                rl = L.rule_label(a2, b2, h2, cen, yaw_r, mrange[1], policy)
                img = os.path.join(os.path.abspath(out), "images", f"h{house}", f"{sid}.png")
                write_png(img, draw_selected_box(rgb, bx))
                meta = dict(sample_id=sid, task="arms", house=house, instance=inst["idx"], view=view, body=inst["body"], category=cat, asset_id=inst["asset_id"],
                            mass_sim_kg=round(inst["mass_sim"], 4), true_extent_obb=[round(obb["a"], 4), round(obb["b"], 4), round(obb["h"], 4)],
                            true_extent_aabb=[round(float(v), 4) for v in obb["aabb"]], noisy_extent=[a2, b2, h2], centroid=cen, yaw=yaw_r,
                            side="left" if cen[1] > L.A.CENTRE_Y else "right", mass_range=mrange, mass_hi_used=mrange[1],
                            rule_mode=rl["rule_mode"] if rl else "infeasible", infeasible_policy=policy, label_shown=label, label_kind=lkind,
                            box2d=bx, mask_px=npx, box_h_px=bx[3] - bx[1], cam=dict(azimuth=round(cam["azimuth"], 2), elevation=round(cam["elevation"], 2),
                                                                                  distance=round(cam["distance"], 3)), light=lt, attempt=attempt, seed=seed)
                lines = []
                if rl is not None:
                    ans = L.answer_obj(rl, mrange, prior)
                    lines.append(L.arms_record(img, label, (a2, b2, h2), cen, ans, meta))
                if rng.random() < P_NAME_TASK:
                    lines.append(L.name_record(img, prior.display, dict(meta, sample_id=sid + "|name", task="name")))
                for r in lines:
                    fh.write(json.dumps(r) + "\n")
                fh.flush()
                n_new += 1
        fh.close()
    if R is not None:
        R[0].close()
        R[1].close()
    return dict(worker=wid, new_views=n_new, seconds=round(time.time() - t_w, 1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--houses", type=int, nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--views-per-instance", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="total arms-samples (split evenly over the houses); default all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--infeasible", choices=L.INFEASIBLE_POLICIES, default="least_bad")
    ap.add_argument("--scene-dir", default=str(SCENE_DIR))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    lph = None if a.limit is None else math.ceil(a.limit / len(a.houses))
    t0 = time.time()
    jobs = [(w, a.workers, a.houses, a.out, a.views_per_instance, lph, a.seed, a.infeasible, a.scene_dir) for w in range(a.workers)]
    if a.workers == 1:
        res = [worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(a.workers) as pool:
            res = pool.map(worker, jobs, chunksize=1)
    n = sum(r["new_views"] for r in res)
    el = time.time() - t0
    print(json.dumps(dict(new_views=n, seconds=round(el, 1), views_per_hour=round(n / el * 3600) if el > 0 else None, workers=res)))


if __name__ == "__main__":
    main()
