"""grounding/make_tabletop.py -- tabletop rig scenes (perception_zono geometry) rendered with MuJoCo (RGB + per-pixel object-id) from the
head camera (head_aim: the proposed ZED position re-aimed at the tray; the as-mounted 'head' looks at the table) and one oblique view
(head_hi, 45 deg), plus the SAME scene's point clouds from perception_zono.sensing (n3mm and nominal noise) for the downstream test.
SIMULATION ONLY. Run in mujoco_ur5e env with MUJOCO_GL=egl and CUDA_VISIBLE_DEVICES=<bimanual GPU 4-7>.

  python -m world_model.v2.vlm.grounding.make_tabletop --out $DATA/tab
Outputs: <out>/img/<id>.png, <out>/seg/<id>.npz (seg int16 HxW: -1 bg, 0 tray, k object), <out>/cloud/<id>.npz, <out>/meta.json
"""
from __future__ import annotations

import argparse
import json
import os
import zlib
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

from world_model.v2.perception_zono import geometry as G, sensing as S  # noqa: E402
from world_model.v2.perception_zono.run_study import mix_scene  # noqa: E402
from world_model.v2.vlm.grounding import common as C  # noqa: E402

W, H = C.IMG_W, C.IMG_H
VIEWS = {"head": "head_aim", "obl": "head_hi"}          # name used in ids -> perception_zono rig
PALETTE = {"red": (0.85, 0.12, 0.10), "green": (0.15, 0.65, 0.20), "blue": (0.15, 0.30, 0.85), "yellow": (0.95, 0.85, 0.15),
           "orange": (0.95, 0.50, 0.10), "purple": (0.55, 0.20, 0.70), "cyan": (0.10, 0.75, 0.80), "pink": (0.95, 0.45, 0.70),
           "white": (0.95, 0.95, 0.95), "gold": (0.90, 0.75, 0.20)}
NOUN = {"ball": "ball", "box": "box", "block": "block", "bottle": "bottle", "mug": "mug", "L": "L-shaped bracket", "cup": "cup",
        "dish": "dish", "bead": "bead", "item": "small ball", "tray": "tray"}
KIND_COLOR = {"ball": "red", "box": "blue", "block": "purple", "bottle": "green", "mug": "yellow", "L": "orange", "cup": "white",
              "dish": "cyan", "bead": "gold", "item": "gold"}
MIN_PX = 150          # an object is a VLM query target only if >= this many visible pixels at 1280x720


def _j(rng, a=0.004):
    return float(rng.uniform(-a, a))


def build_scene(name, seed):
    """-> (objs, colors{oid: colorname}, refs[list of dict(target, wrong, side, noun, anchor_noun)], group{oid: group})"""
    rng = np.random.default_rng(zlib.crc32(f"{name}|{seed}".encode()))
    refs = []
    if name in ("mix4", "tower3", "cupbeads", "dishitems", "Lblock"):
        objs = mix_scene(name, rng)
    elif name.startswith("single_"):
        k = name.split("_", 1)[1]
        x, y, yaw = float(rng.uniform(0.0, 0.07)), float(rng.uniform(-0.04, 0.04)), float(rng.uniform(0, 2 * np.pi))
        objs = [G.proto(k).moved(x, y, yaw)]
    elif name == "touch_boxblock":      # box and block flush side by side (faces touch, gap 0.1 mm)
        blk = G.Obj("blk", "block", [G.Prim("box", (0.025, 0.025, 0.02), np.array([0.0, 0.0, 0.02]))])
        objs = [G.proto("box").moved(0.04, -0.03, 0.0), blk.moved(0.04, 0.0201, 0.0)]
    elif name == "touch_bottlemug":     # bottle body touching the mug wall
        objs = [G.proto("bottle").moved(0.02, -0.06, 0.0), G.proto("mug").moved(0.02, 0.011, 0.0)]
    elif name == "touch_ballbox":
        objs = [G.proto("box").moved(0.03, 0.0, 0.0), G.proto("ball").moved(0.03, 0.0 + 0.025 + 0.03, 0.0)]
    elif name.startswith("ref_"):        # two identical objects + a bottle/box anchor; "the X left/right of the anchor"
        k = name.split("_", 1)[1]
        anchor = G.proto("bottle" if k != "ball" else "box").moved(0.08 + _j(rng, 0.01), _j(rng, 0.01), 0.0)
        a = G.proto(k).moved(-0.04 + _j(rng, 0.01), -0.055 + _j(rng, 0.005), 0.0)
        b = G.proto(k).moved(-0.04 + _j(rng, 0.01), 0.055 + _j(rng, 0.005), 0.0)
        objs = [a, b, anchor]
    else:
        raise KeyError(name)
    for i, o in enumerate(objs):
        o.oid = i + 1
    colors, group = {}, {}
    pal = list(PALETTE)
    same_noun = {}
    for o in objs:
        same_noun.setdefault(o.kind, []).append(o)
    for i, o in enumerate(objs):
        if name.startswith("ref_") and o.kind == name.split("_", 1)[1] and o.oid in (1, 2):
            colors[o.oid] = "yellow" if o.kind != "block" else "orange"
        elif len(same_noun[o.kind]) > 1 and o.kind not in ("bead", "item"):
            colors[o.oid] = pal[same_noun[o.kind].index(o) % 8]
        else:
            colors[o.oid] = KIND_COLOR[o.kind]
    if name.startswith("ref_"):
        k = name.split("_", 1)[1]
        anchor_noun = NOUN["bottle" if k != "ball" else "box"]
        refs = [dict(target=1, wrong=2, side="left", noun=NOUN[k], anchor_noun=anchor_noun),
                dict(target=2, wrong=1, side="right", noun=NOUN[k], anchor_noun=anchor_noun)]
    return objs, colors, refs


def render_xml(objs, colors):
    """MJCF for rendering: same primitives as G.Scene (names o<oid>_<j>), plus tray, table, floor, light, sky."""
    sc_all = [G.proto("tray").moved(0.0, 0.0, 0.0)] + objs
    sc_all[0].oid = 0
    geoms, glabel = [], []
    q = np.zeros(4)
    for o in sc_all:
        rgb = (0.80, 0.80, 0.85) if o.oid == 0 else PALETTE[colors[o.oid]]
        for j, p in enumerate(o.prims):
            mujoco.mju_mat2Quat(q, p.R.reshape(9))
            typ = {"box": "box", "cyl": "cylinder", "sphere": "sphere"}[p.kind]
            sz = " ".join("%.6f" % x for x in p.size)
            geoms.append(f'<geom name="o{o.oid}_{j}" type="{typ}" size="{sz}" pos="%.6f %.6f %.6f" quat="%.8f %.8f %.8f %.8f" '
                         f'rgba="{rgb[0]} {rgb[1]} {rgb[2]} 1" contype="0" conaffinity="0"/>' % (*p.pos, *q))
            glabel.append(o.oid)
    xml = f"""<mujoco><visual><global offwidth="{W}" offheight="{H}" fovy="55"/><quality shadowsize="4096"/>
<headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/><map znear="0.01" zfar="20"/></visual>
<asset><texture type="skybox" builtin="gradient" rgb1="0.75 0.82 0.92" rgb2="0.95 0.95 0.97" width="64" height="64"/></asset>
<worldbody><light pos="0.6 -0.4 2.2" dir="-0.2 0.15 -1" diffuse="0.7 0.7 0.7" castshadow="true"/>
<geom name="floor" type="plane" size="4 4 0.1" rgba="0.55 0.55 0.58 1" contype="0" conaffinity="0"/>
<geom name="table" type="box" size="0.7 0.7 0.5" pos="0 0 0.482" rgba="0.60 0.47 0.34 1" contype="0" conaffinity="0"/>
{''.join(geoms)}</worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    gl = np.full(m.ngeom, -1, np.int32)           # table / floor = -1
    for gi in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi)
        if nm and nm.startswith("o"):
            gl[gi] = int(nm[1:].split("_")[0])
    return m, gl


def free_cam(origin, R, dist=1.0):
    fwd = -np.asarray(R)[:, 2]
    assert abs(np.asarray(R)[2, 0]) < 1e-6, "camera must be roll-free (right vector horizontal)"
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
    cam.elevation = float(np.degrees(np.arcsin(fwd[2])))
    cam.distance = dist
    cam.lookat[:] = np.asarray(origin) + fwd * dist
    return cam


def render_rgb_seg(m, d, cam, geom_label):
    r = mujoco.Renderer(m, H, W)
    r.update_scene(d, camera=cam)
    rgb = r.render().copy()
    r.enable_segmentation_rendering()
    r.update_scene(d, camera=cam)
    sg = r.render().copy()
    r.close()
    gid = np.where(sg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), sg[..., 0], -1)
    seg = np.where(gid >= 0, geom_label[np.clip(gid, 0, None)], -1).astype(np.int16)
    return rgb, seg


def project(pts, origin, R, f_px, Wd=W, Hd=H):
    """world points -> pixel (u, v) in a Wd x Hd image of a camera at origin with camera-to-world R (looks along -z), + z-depth."""
    pc = (np.asarray(pts) - origin) @ R
    z = -pc[:, 2]
    zz = np.where(z > 1e-6, z, 1e-6)
    return np.stack([Wd / 2.0 + f_px * pc[:, 0] / zz, Hd / 2.0 - f_px * pc[:, 1] / zz], 1), z


F_PX = (H / 2.0) / np.tan(np.deg2rad(55.0) / 2.0)

SCENES = (["mix4"] * 3 + ["tower3"] * 3 + ["cupbeads"] * 2 + ["dishitems"] * 2 + ["Lblock"] * 2 +
          ["touch_boxblock", "touch_bottlemug", "touch_ballbox"] +
          [f"single_{k}" for k in ("ball", "box", "block", "bottle", "mug", "L", "cup", "dish")] +
          ["ref_mug", "ref_block", "ref_ball", "ref_box"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.DATA / "tab"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    for s in ("img", "seg", "cloud"):
        (out / s).mkdir(parents=True, exist_ok=True)
    meta, cnt = [], {}
    scenes = SCENES[:a.limit] if a.limit else SCENES
    for name in scenes:
        seed = cnt.get(name, 0)
        cnt[name] = seed + 1
        objs, colors, refs = build_scene(name, seed)
        sc = G.Scene(objs)                                        # raycast model for the point clouds (re-assigns oids 1..n)
        rm, gl = render_rgb_scene = render_xml(objs, colors)
        d = mujoco.MjData(rm)
        mujoco.mj_forward(rm, d)
        for vname, rig in VIEWS.items():
            iid = f"tab_{name}_s{seed}_{vname}"
            clouds = {}
            for cond in ("n3mm", "nominal"):
                cl = S.capture(sc, rig, cond, zlib.crc32(iid.encode()) & 0x7FFFFFFF)
                clouds[cond] = cl
            v0 = clouds["n3mm"].views[0]
            cam = free_cam(v0.origin, v0.R)
            rgb, seg = render_rgb_seg(rm, d, cam, gl)
            from PIL import Image
            Image.fromarray(rgb).save(out / "img" / f"{iid}.png")
            np.savez_compressed(out / "seg" / f"{iid}.npz", seg=seg)
            np.savez_compressed(out / "cloud" / f"{iid}.npz", **{f"{c}_{k}": getattr(clouds[c], k) for c in clouds for k in ("pts", "label", "cam")},
                                origin=v0.origin, R=v0.R)
            # objects table
            nouns = [NOUN[o.kind] for o in objs]
            ob = []
            for o in [sc.tray] + objs:
                m_ = seg == o.oid
                area = int(m_.sum())
                bb = C.mask_bbox(m_) if area else None
                if o.oid == 0:
                    noun, color = "tray", None
                else:
                    noun, color = NOUN[o.kind], colors[o.oid]
                multi = nouns.count(noun) > 1 and o.oid != 0
                is_ref_inst = name.startswith("ref_") and o.oid in (1, 2)
                phrase = noun if not multi else f"{color} {noun}"
                ob.append(dict(oid=o.oid, kind=o.kind, noun=noun, color=color, phrase=phrase, area=area, bbox=bb,
                               queryable=bool(area >= MIN_PX and not is_ref_inst), ref_instance=is_ref_inst))
            objq = [dict(phrase=x["phrase"], targets=[x["oid"]], unique=True) for x in ob if x["queryable"]]
            queries = []
            for rf in refs:
                t = [x for x in ob if x["oid"] == rf["target"]][0]
                w = [x for x in ob if x["oid"] == rf["wrong"]][0]
                an = [x for x in ob if x["oid"] == max(o.oid for o in objs)][0]
                ok = t["area"] >= MIN_PX and w["area"] >= MIN_PX and an["area"] >= MIN_PX
                # image-space truth of the relation (mask centroids)
                cu = lambda oid: float(np.nonzero(seg == oid)[1].mean()) if (seg == oid).any() else np.nan
                rel_ok = (cu(rf["target"]) < cu(an["oid"])) if rf["side"] == "left" else (cu(rf["target"]) > cu(an["oid"]))
                queries.append(dict(phrase=f"{rf['noun']} {rf['side']} of the {rf['anchor_noun']}", target=rf["target"], wrong=rf["wrong"],
                                    valid=bool(ok and rel_ok), side=rf["side"]))
            meta.append(dict(id=iid, domain="tab", scene=name, seed=seed, view=vname, rig=rig, image=str(out / "img" / f"{iid}.png"),
                             seg=str(out / "seg" / f"{iid}.npz"), cloud=str(out / "cloud" / f"{iid}.npz"), W=W, H=H, objects=ob, objq=objq, refs=queries))
            print(iid, [(x["phrase"], x["area"]) for x in ob], flush=True)
    (out / "meta.json").write_text(json.dumps(meta))
    print("wrote", len(meta), "images")


if __name__ == "__main__":
    main()
