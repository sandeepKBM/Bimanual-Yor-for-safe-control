"""grounding/make_house.py -- MolmoSpaces house views (YOR + held tray composed by house.build.compose): RGB + per-pixel object-id (geom -> body
-> house-object metadata category) + metric depth, from the proposed head camera (depth_sim _HEAD_POS_ROBOT/_HEAD_R in the robot frame) and one
oblique view (behind/above the robot, looking along the tray direction). SIMULATION ONLY. House XMLs are texture-stripped (textures not fetched):
surfaces are flat material colours, harder for a VLM than textured images. Run on westeros (house data is not on Slurm nodes) with MUJOCO_GL=egl and
CUDA_VISIBLE_DEVICES in 4-7:   python -m world_model.v2.vlm.grounding.make_house --out $DATA/house
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

from world_model.v2.house import assets as A, build as B  # noqa: E402
from world_model.v2.tray_rep.perception.depth_sim import _HEAD_POS_ROBOT, _HEAD_R  # noqa: E402
from world_model.v2.vlm.grounding import common as C  # noqa: E402
from world_model.v2.vlm.grounding.make_tabletop import free_cam, F_PX, W, H  # noqa: E402

STRUCT = {"wall", "floor", "ceiling", "doorway", "window", "door", "room", "doorframe"}
MIN_Q, MIN_ANY = 400, 150
SPOTS = [(17, (3.55, 10.05)), (1, (8.03, 3.8)), (541, None)]


def camel_words(cat):
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", cat).lower()


def lookat_R(pos, target):
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, -f], 1)


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def geom_object_map(m, idx):
    """-> (gl int32[ngeom] object index or -1, objs list of dict(oid, name, category, structural))."""
    meta = A.metadata(idx)
    body_name = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    owner = {}
    for hn, md in meta.items():
        for bn in md["name_map"]["bodies"]:
            owner[bn] = hn
    objs, oid_of = [], {}
    gl = np.full(m.ngeom, -1, np.int32)

    def get(name, cat):
        if name not in oid_of:
            oid_of[name] = len(objs)
            objs.append(dict(oid=len(objs), name=name, category=cat, structural=cat.lower() in STRUCT))
        return oid_of[name]

    for gi in range(m.ngeom):
        b = int(m.geom_bodyid[gi])
        found = None
        while b > 0:
            bn = body_name(b)
            if bn in owner:
                found = owner[bn]
                break
            if bn == "yor_root":
                found = "@robot"
                break
            if bn in ("tray",):
                found = "@tray"
                break
            if bn == "ball":
                found = "@ball"
                break
            if re.match(r"(wall|room)_", bn):
                found = "@" + bn.split("_")[0]
                break
            b = int(m.body_parentid[b])
        if found is None:
            continue
        if found.startswith("@"):
            gl[gi] = get(found, found[1:])
        else:
            gl[gi] = get(found, meta[found]["category"])
    return gl, objs


def build_queries(objs, seg):
    """per-category queries over objects with >= MIN_ANY visible pixels. unique = exactly one such instance."""
    ob = []
    for o in objs:
        msk = seg == o["oid"]
        area = int(msk.sum())
        ob.append(dict(o, area=area, bbox=C.mask_bbox(msk) if area else None))
    byc = {}
    for o in ob:
        if o["structural"] or o["category"] in ("robot", "ball", "wall", "room") or o["area"] < MIN_ANY:
            continue
        byc.setdefault(o["category"], []).append(o)
    objq = []
    for cat, L in byc.items():
        if max(x["area"] for x in L) < MIN_Q:
            continue
        objq.append(dict(phrase=camel_words(cat) if cat != "tray" else "tray", targets=[x["oid"] for x in L], unique=len(L) == 1,
                         area=int(sum(x["area"] for x in L)), category=cat))
    objq.sort(key=lambda q: -q["area"])
    return ob, objq[:12]


def render_view(m, d, cam, gl):
    r = mujoco.Renderer(m, H, W)
    r.update_scene(d, camera=cam)
    rgb = r.render().copy()
    r.enable_depth_rendering()
    r.update_scene(d, camera=cam)
    dep = r.render().copy()
    r.disable_depth_rendering()
    r.enable_segmentation_rendering()
    r.update_scene(d, camera=cam)
    sg = r.render().copy()
    r.close()
    gid = np.where(sg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), sg[..., 0], -1)
    seg = np.where(gid >= 0, gl[np.clip(gid, 0, None)], -1).astype(np.int16)
    return rgb, dep, seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.DATA / "house"))
    ap.add_argument("--yaws", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    for s in ("img", "seg", "depth"):
        (out / s).mkdir(parents=True, exist_ok=True)
    from PIL import Image
    meta = []
    for idx, spawn in SPOTS:
        if spawn is None:
            m0, d0 = B.load_house(idx, True)
            mujoco.mj_forward(m0, d0)
            spawn = tuple(B.find_clear_spots(idx, m0, d0, top=1)[0][k] for k in ("x", "y"))
        best = None
        for k in range(a.yaws):
            yaw = 0.3 + k * np.pi / 2
            Cc = B.compose(idx, spawn, yaw=yaw, with_tray=True, dt=0.005)
            B.set_hold_state(Cc)
            B.disable_arm_servos(Cc.m)
            mujoco.mj_forward(Cc.m, Cc.d)
            m, d = Cc.m, Cc.d
            m.vis.global_.offwidth, m.vis.global_.offheight, m.vis.global_.fovy = W, H, 55.0
            m.vis.map.znear, m.vis.map.zfar = 0.05, 30.0
            gl, objs = geom_object_map(m, idx)
            origin_r = _HEAD_POS_ROBOT
            Rg = Rz(yaw) @ _HEAD_R
            og = Rz(yaw) @ origin_r + np.array([spawn[0], spawn[1], 0.0])
            views = {"head": (og, Rg)}
            pr = np.array([-0.7, -0.21, 1.6])
            Ro = lookat_R(pr, pr + np.array([np.cos(np.deg2rad(-22)), 0, np.sin(np.deg2rad(-22))]))
            views["obl"] = (Rz(yaw) @ pr + np.array([spawn[0], spawn[1], 0.0]), Rz(yaw) @ Ro)
            res = {}
            for vn, (o_, R_) in views.items():
                cam = free_cam(o_, R_)
                rgb, dep, seg = render_view(m, d, cam, gl)
                ob, objq = build_queries(objs, seg)
                res[vn] = (o_, R_, rgb, dep, seg, ob, objq)
            score = len(res["head"][6]) + len(res["obl"][6])
            print(f"house {idx} yaw {yaw:.2f}: queries head={len(res['head'][6])} obl={len(res['obl'][6])}", flush=True)
            if best is None or score > best[0]:
                best = (score, yaw, res)
        _, yaw, res = best
        for vn, (o_, R_, rgb, dep, seg, ob, objq) in res.items():
            iid = f"house{idx}_{vn}"
            Image.fromarray(rgb).save(out / "img" / f"{iid}.png")
            np.savez_compressed(out / "seg" / f"{iid}.npz", seg=seg)
            np.savez_compressed(out / "depth" / f"{iid}.npz", depth=np.round(dep * 1000).clip(0, 65535).astype(np.uint16), origin=o_, R=R_, f_px=F_PX)
            meta.append(dict(id=iid, domain="house", scene=f"house{idx}", house=idx, view=vn, yaw=float(yaw), spawn=list(spawn), image=str(out / "img" / f"{iid}.png"),
                             seg=str(out / "seg" / f"{iid}.npz"), depth=str(out / "depth" / f"{iid}.npz"), W=W, H=H, objects=ob, objq=objq, refs=[]))
            print(iid, [(q["phrase"], q["unique"], q["area"]) for q in objq], flush=True)
    (out / "meta.json").write_text(json.dumps(meta, default=float))


if __name__ == "__main__":
    main()
