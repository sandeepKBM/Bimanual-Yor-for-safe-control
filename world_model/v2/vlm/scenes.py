"""vlm/scenes.py -- render MuJoCo scenes (offscreen, EGL) with KNOWN ground-truth physical parameters, for the VLM smoke tests.
SIMULATION ONLY. Run with the mujoco_ur5e env (needs MUJOCO_GL=egl + a GPU node); the VLM env never imports this.

Scene kinds
  tray_ball   tray + ball on a table (tomato proxy of testbeds/b_tray_ball). truth: mu_roll [m], mu (sliding), ball radius [m]
  tray_door   same, plus a door frame (two posts + lintel) between the tray and a destination table.
  cup_beads   cup with beads (data/contents.py rig). truth: bead friction, number of beads, bead radius [m]
Sets (how truth relates to appearance)
  rand        truth drawn from the SIM POPULATION, independent of appearance (colour random) -> the image carries NO info about
              mu_roll; any skill is prior knowledge only.  mu_roll ~ logU[5e-4, 6e-3], mu ~ U[0.25, 0.7].
  mat         ball "material" (steel/wood/rubber/foam/tomato) sets truth through MATERIAL_TABLE (*lognormal noise). The
              appearance->physics link is OUR ASSUMPTION (the sim has none); it asks whether the VLM can exploit a link when one exists.

Outputs: <out>/<scene_id>_{a,b}.png and <out>/meta.json  (list of dicts: id, kind, set, truth, images, instruction, entities).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from world_model.v2.testbeds.b_tray_ball import tray_env as TE  # noqa: E402
from world_model.v2.data import contents as C  # noqa: E402

MATERIAL_TABLE = {   # name: (rgba, mu_roll [m], mu)
    "steel": ((0.72, 0.74, 0.78, 1), 6e-4, 0.35),
    "wood": ((0.62, 0.42, 0.22, 1), 1.5e-3, 0.45),
    "rubber": ((0.12, 0.12, 0.14, 1), 3.0e-3, 0.70),
    "foam": ((0.95, 0.85, 0.25, 1), 5.0e-3, 0.60),
    "tomato": ((0.85, 0.15, 0.10, 1), 2.0e-3, 0.50),
}
RAND_COLOURS = [(0.72, 0.74, 0.78, 1), (0.62, 0.42, 0.22, 1), (0.12, 0.12, 0.14, 1), (0.95, 0.85, 0.25, 1), (0.85, 0.15, 0.10, 1),
                (0.2, 0.55, 0.9, 1), (0.25, 0.7, 0.3, 1), (0.9, 0.9, 0.9, 1)]
MU_ROLL_RANGE = (5e-4, 6e-3)
MU_RANGE = (0.25, 0.7)

TABLE_RGBA = "0.55 0.45 0.35 1"
FLOOR = '<geom name="floor" type="plane" size="3 3 0.1" rgba="0.75 0.75 0.78 1" contype="0" conaffinity="0"/>'
LIGHT = '<light pos="0.5 -0.5 3" dir="-0.1 0.2 -1" diffuse="0.9 0.9 0.9"/>'


def _deco(extra: str) -> str:
    return f"{LIGHT}\n{FLOOR}\n{extra}"


def _table(name, x, y, top, hx=0.3, hy=0.3, rgba=TABLE_RGBA, collide=False):
    hz = top / 2.0
    ct = 'contype="1" conaffinity="1" friction="1 0.005 0.0001"' if collide else 'contype="0" conaffinity="0"'
    return (f'<geom name="{name}" type="box" size="{hx} {hy} {hz:.4f}" pos="{x} {y} {hz:.4f}" rgba="{rgba}" {ct}/>')


def _door(x=0.55, gap_half=0.22):
    posts = "".join(f'<geom name="door_post_{s}" type="box" size="0.04 0.03 0.55" pos="{x} {sy*(gap_half+0.03):.3f} 0.55" '
                    f'rgba="0.9 0.9 0.85 1" contype="0" conaffinity="0"/>' for s, sy in (("l", 1), ("r", -1)))
    lintel = (f'<geom name="door_lintel" type="box" size="0.04 {gap_half+0.06:.3f} 0.03" pos="{x} 0 1.12" '
              f'rgba="0.9 0.9 0.85 1" contype="0" conaffinity="0"/>')
    return posts + lintel


def _colour_geoms(xml: str, names, rgba):
    for n in names:
        xml = re.sub(rf'(<geom name="{n}")', rf'\1 rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"', xml, count=1)
    return xml


def tray_model(p: TE.BallParams, rgba, door=False):
    xml = TE.tray_xml(p, lip=False, home=(0.0, 0.0, 1.0))
    xml = _colour_geoms(xml, ["ball_g"], rgba)
    xml = _colour_geoms(xml, ["tray_top"], (0.82, 0.82, 0.86, 1))
    extra = _table("start_table", 0.0, 0.0, 0.9935, 0.22, 0.17, collide=True)
    extra += _table("target_table", 1.1 if door else 0.85, 0.0, 0.99, 0.2, 0.2, rgba="0.45 0.55 0.65 1")
    if door:
        extra += _door()
    xml = xml.replace("</worldbody>", _deco(extra) + "\n</worldbody>")
    return mujoco.MjModel.from_xml_string(xml)


def cup_model(c: C.Contents, rgba):
    extra = _deco(_table("start_table", 0.0, 0.0, 0.9995, 0.10, 0.10, collide=True) + _table("target_table", 0.7, 0.0, 0.998, 0.12, 0.12,
                                                                           rgba="0.45 0.55 0.65 1"))
    m = C.rig_model(c, extra_xml=extra)
    m.mat_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "cup_mat")] = (0.85, 0.2, 0.2, 0.35)   # see-through cup
    for g in range(m.ngeom):
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("ball_"):
            m.geom_rgba[g] = rgba
    return m


def render(m, d, cams, size=(448, 576)):
    r = mujoco.Renderer(m, *size)
    ims = []
    for cam in cams:
        r.update_scene(d, camera=cam)
        ims.append(r.render().copy())
    r.close()
    return ims


def cams_for(kind):
    def mk(az, el, dist, look):
        c = mujoco.MjvCamera()
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.azimuth, c.elevation, c.distance, c.lookat[:] = az, el, dist, look
        return c
    if kind == "cup_beads":
        return [mk(30, -25, 0.45, [0, 0, 1.03]), mk(110, -55, 0.40, [0, 0, 1.03])]
    if kind == "tray_door":
        return [mk(35, -28, 1.7, [0.5, 0, 1.0]), mk(90, -20, 1.9, [0.5, 0, 1.0])]
    return [mk(35, -28, 0.95, [0.1, 0, 1.0]), mk(100, -50, 0.85, [0.1, 0, 1.0])]


ENTITY_DOC_V1 = {
    "tray_ball": {"tray": "the tray as a rigid body (use for upright / clear / support)",
                  "tray_top": "the tray's top-face footprint region (use as the container B in inside / support)",
                  "ball": "the ball (its centre point)"},
    "tray_door": {"tray": "the tray as a rigid body (use for upright / clear)",
                  "tray_top": "the tray's top-face footprint region (container B in inside / support)",
                  "ball": "the ball (its centre point)",
                  "door_post_left": "the left door post, a solid obstacle (use in clear)",
                  "door_post_right": "the right door post, a solid obstacle (use in clear)",
                  "target_table": "the destination table-top region (container B in inside)"},
    "cup_beads": {"cup": "the cup as a rigid body (use for upright)",
                  "cup_interior": "the cup's inner footprint region (container B in inside)",
                  "bead": "a bead (its centre point)",
                  "target_table": "the destination table-top region"},
}
INSTRUCTION = {
    "tray_ball": "Carry the tray to the table without dropping the ball.",
    "tray_door": "Carry the tray through the doorway to the far table without dropping the ball and without hitting the door frame.",
    "cup_beads": "Carry the cup to the far table without spilling any beads.",
}


def make_scenes(out: Path, n_tray=12, n_door=4, n_cup=12, seed=0):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    meta = []

    def save(sid, ims):
        from PIL import Image
        paths = []
        for k, im in zip("ab", ims):
            pth = out / f"{sid}_{k}.png"
            Image.fromarray(im).save(pth)
            paths.append(str(pth))
        return paths

    # ---- tray scenes
    plan = [("tray_ball", "rand")] * n_tray + [("tray_ball", "mat")] * n_tray + [("tray_door", "rand")] * n_door + \
           [("tray_door", "mat")] * n_door
    for i, (kind, sset) in enumerate(plan):
        r_ball = float(rng.uniform(0.02, 0.04))
        if sset == "rand":
            mu_roll = float(np.exp(rng.uniform(*np.log(MU_ROLL_RANGE))))
            mu = float(rng.uniform(*MU_RANGE))
            rgba, mat = RAND_COLOURS[int(rng.integers(len(RAND_COLOURS)))], "n/a"
        else:
            mat = list(MATERIAL_TABLE)[i % len(MATERIAL_TABLE)]
            rgba, mr, mu0 = MATERIAL_TABLE[mat]
            mu_roll = float(mr * np.exp(rng.normal(0, 0.25)))
            mu = float(np.clip(mu0 * np.exp(rng.normal(0, 0.1)), *MU_RANGE if mu0 < 0.7 else (0.25, 0.8)))
        p = TE.BallParams(radius=r_ball, mu=mu, mu_roll=mu_roll)
        m = tray_model(p, rgba, door=(kind == "tray_door"))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        for _ in range(80):
            mujoco.mj_step(m, d)
        sid = f"{kind}_{sset}_{i:02d}"
        meta.append(dict(id=sid, kind=kind, set=sset, material=mat if sset == "mat" else "n/a",
                         truth=dict(mu_roll=mu_roll, mu=mu, ball_radius=r_ball),
                         images=save(sid, render(m, d, cams_for(kind))), instruction=INSTRUCTION[kind],
                         entities=ENTITY_DOC[kind]))
    # ---- cup scenes
    for i in range(n_cup):
        c = C.sample_contents(rng, nominal_prob=0.0)
        rgba = RAND_COLOURS[int(rng.integers(len(RAND_COLOURS)))]
        m = cup_model(c, rgba)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        for _ in range(120):
            mujoco.mj_step(m, d)
        sid = f"cup_beads_rand_{i:02d}"
        meta.append(dict(id=sid, kind="cup_beads", set="rand", material="n/a",
                         truth=dict(bead_friction=c.friction, n_beads=c.n_balls, bead_radius=c.radius,
                                    bead_rolling=c.rolling),
                         images=save(sid, render(m, d, cams_for("cup_beads"))), instruction=INSTRUCTION["cup_beads"],
                         entities=ENTITY_DOC["cup_beads"]))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/common/users/ss5772/bimanual_data/vlm/scenes")
    ap.add_argument("--n_tray", type=int, default=12)
    ap.add_argument("--n_door", type=int, default=4)
    ap.add_argument("--n_cup", type=int, default=12)
    a = ap.parse_args()
    mt = make_scenes(Path(a.out), a.n_tray, a.n_door, a.n_cup)
    print(len(mt), "scenes ->", a.out)


# v2 entity docs: v1 told the VLM to use tray_top / cup_interior as B of `support`, but ir.Support needs a rigid BoxShape (B = tray).
ENTITY_DOC = {k: dict(v) for k, v in ENTITY_DOC_V1.items()}
ENTITY_DOC_V2 = {k: dict(v) for k, v in ENTITY_DOC_V1.items()}
for _k in ("tray_ball", "tray_door"):
    ENTITY_DOC_V2[_k]["tray"] = "the tray as a rigid body (use for upright, clear, and as B of support)"
    ENTITY_DOC_V2[_k]["tray_top"] = "the tray's top-face footprint region (use ONLY as the container B of inside; never in support / clear / upright)"
ENTITY_DOC_V2["tray_door"]["target_table"] = "the destination table-top region (use ONLY as the container B of inside)"
ENTITY_DOC_V2["cup_beads"]["cup_interior"] = "the cup's inner footprint region (use ONLY as the container B of inside)"
ENTITY_DOC_V2["cup_beads"]["target_table"] = "the destination table-top region (use ONLY as the container B of inside)"
