"""selftest.py -- Phase 0 (20 val houses: compile + timing + doors) and Phase 1 acceptance (see phase1.py).

    python -m world_model.v2.house.selftest phase0 [--ids 0 1 2 ...] [--out results/phase0.json]

SIMULATION only, CPU only, single thread. Timings are wall-clock on a shared node (load average printed).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import time
from pathlib import Path

import numpy as np
import mujoco

from . import assets as A
from . import build as B

HERE = Path(__file__).resolve().parent


def _time_steps(m, d, n_settle=300, n_time=300, chunk=50):
    for _ in range(n_settle):
        mujoco.mj_step(m, d)
    ts = []
    for _ in range(n_time // chunk):
        t = time.perf_counter()
        for _ in range(chunk):
            mujoco.mj_step(m, d)
        ts.append((time.perf_counter() - t) / chunk * 1e3)
    return float(st.median(ts)), float(min(ts)), int(d.ncon)


def _rays(m, d, pnt, n_dir_grid, mask, cutoff=20.0, reps=3):
    W, H = n_dir_grid
    fov = np.deg2rad(55)
    f = (H / 2) / np.tan(fov / 2)
    u, v = np.meshgrid(np.arange(W) - W / 2 + .5, np.arange(H) - H / 2 + .5)
    dirs = np.stack([np.full_like(u, f), -u, -v], -1).reshape(-1, 3)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    n = len(dirs)
    gid = np.zeros(n, np.int32)
    dist = np.zeros(n)
    gm = None if mask is None else np.array(mask, np.uint8)
    mujoco.mj_multiRay(m, d, pnt, dirs.ravel(), gm, 1, -1, gid, dist, None, n, cutoff)
    t = time.perf_counter()
    for _ in range(reps):
        mujoco.mj_multiRay(m, d, pnt, dirs.ravel(), gm, 1, -1, gid, dist, None, n, cutoff)
    return (time.perf_counter() - t) / reps * 1e3, float((gid >= 0).mean())


def measure_doors(idx, m, d):
    """Clear opening width per door: horizontal rays along the wall from the door-frame body, +- direction, at three
    heights, door leaves + handles moved out of the ray mask. Returns list of dicts (design + measured)."""
    md = A.metadata(idx)
    sj = {dr["id"]: dr for dr in A.spec_json(idx)["doors"]}
    out = []
    saved = m.geom_group.copy()
    for k, v in md.items():
        if v["object_id"] not in sj or v["category"] not in ("Doorway", "Doorframe"):
            continue
        dr = sj[v["object_id"]]
        h = dr["holePolygon"]
        design = abs(h[1]["x"] - h[0]["x"])
        bodies = v["name_map"]["bodies"]
        root = m.body(next(iter(bodies))).id
        m.geom_group[:] = saved
        for bn, asset in bodies.items():
            a = asset.lower()
            if "_door_" in a or "handle" in a or a.endswith("_door"):
                bid = m.body(bn).id
                for gi in range(m.ngeom):
                    if m.geom_bodyid[gi] == bid:
                        m.geom_group[gi] = 5
        mm = re.match(r"wall\|\d+\|([-\d.]+)\|([-\d.]+)\|([-\d.]+)\|([-\d.]+)", dr["wall0"])
        x0, z0, x1, z1 = (float(g) for g in mm.groups())
        wd = np.array([x1 - x0, z1 - z0, 0.0])
        wd /= np.linalg.norm(wd)
        c = d.xpos[root]
        widths = []
        for zz in (0.4, 1.0, 1.7):
            p = np.array([c[0], c[1], zz])
            dirs = np.stack([wd, -wd])
            gid = np.zeros(2, np.int32)
            dist = np.zeros(2)
            mujoco.mj_multiRay(m, d, p, dirs.ravel(), B.GROUP_PHYS, 1, -1, gid, dist, None, 2, 3.0)
            if (gid >= 0).all():
                widths.append(float(dist.sum()))
        out.append(dict(id=v["object_id"], asset=dr["assetId"], design=design,
                        measured=(min(widths) if widths else float("nan")), n_valid=len(widths)))
    m.geom_group[:] = saved
    return out


def phase0_house(idx):
    r = dict(house=idx)
    A.extract_house(idx)
    r["missing_meshes"] = len(A.missing_mesh_files(idx))
    t = time.perf_counter()
    m = mujoco.MjModel.from_xml_path(str(A.make_stripped(idx)))   # dynamic, as shipped
    r["load_s"] = time.perf_counter() - t
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r.update(nbody=int(m.nbody), ngeom=int(m.ngeom), nmesh=int(m.nmesh), njnt=int(m.njnt),
             nfree=int((m.jnt_type == mujoco.mjtJoint.mjJNT_FREE).sum()), nv=int(m.nv),
             size_mb=m.nbuffer / 1e6, dt=float(m.opt.timestep), model_hash=B.model_hash(m))
    r["step_dyn_ms"], r["step_dyn_min_ms"], r["ncon_dyn"] = _time_steps(m, d)
    del m, d
    t = time.perf_counter()
    ms = B.load_house_spec(idx, True).compile()
    r["load_static_s"] = time.perf_counter() - t
    ds = mujoco.MjData(ms)
    mujoco.mj_forward(ms, ds)
    r["nv_static"] = int(ms.nv)
    r["step_static_ms"], r["step_static_min_ms"], r["ncon_static"] = _time_steps(ms, ds)
    mujoco.mj_forward(ms, ds)
    spots = B.find_clear_spots(idx, ms, ds, top=1)[0]
    r["clear_spot"] = dict(room=spots["type"], x=spots["x"], y=spots["y"], free_r=spots["free_r"])
    pnt = np.array([spots["x"], spots["y"], 1.2])
    ms2 = ms
    r["ray14400_all_ms"], r["ray_hit"] = _rays(ms2, ds, pnt, (160, 90), None)
    r["ray14400_g4_ms"], _ = _rays(ms2, ds, pnt, (160, 90), [0, 0, 0, 0, 1, 0])
    r["ray1600_all_ms"], _ = _rays(ms2, ds, pnt, (40, 40), None)
    r["ray1600_g4_ms"], _ = _rays(ms2, ds, pnt, (40, 40), [0, 0, 0, 0, 1, 0])
    doors = measure_doors(idx, ms, ds)
    r["doors"] = doors
    return r


def phase0(ids, out):
    rows = []
    for i in ids:
        t = time.time()
        r = phase0_house(i)
        rows.append(r)
        print(f"val_{i}: nbody {r['nbody']} ngeom {r['ngeom']} dyn {r['step_dyn_ms']:.2f} ms static {r['step_static_ms']:.2f} ms "
              f"ray14400 {r['ray14400_all_ms']:.0f} ms doors {len(r['doors'])} free_r {r['clear_spot']['free_r']:.2f} "
              f"({time.time() - t:.0f}s)", flush=True)
        Path(out).write_text(json.dumps(dict(load_avg=os.getloadavg(), mujoco=mujoco.__version__, rows=rows), indent=1))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["phase0"])
    ap.add_argument("--ids", type=int, nargs="*", default=A.VAL20)
    ap.add_argument("--out", default=str(HERE / "results" / "phase0.json"))
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.cmd == "phase0":
        phase0(a.ids, a.out)
