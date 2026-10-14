"""nav_grid.py -- occupancy adapter: office static scene -> nav.Grid (same semantics as nav.build_occupancy), doors, standoffs.  SIMULATION only.

nav.build_occupancy is keyed on a ProcTHOR house index (loads the house itself).  This module reproduces its ray rule on the
office model: UPWARD rays from z=-0.3 against collision groups 3+4, thin decals (top < nav.Z_BASE) skipped, anything whose
underside is < nav.ROBOT_TOP (1.15 m) is an obstacle.  Rows are cast in parallel (fork pool); result cached under
<office out dir>/nav_cache keyed by the model hash.  `nav.*` functions that take a Grid (coarsen, free_maps, astar, door_passage,
poses_clearance, validate_traj, time_parameterize, ...) then run unchanged.
"""
from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import mujoco

from .. import nav as N
from . import scene as S

OUT_DIR = Path(os.environ.get("OFFICE_OUT", "/data/local/ss5772/molmospaces/office"))
CACHE = OUT_DIR / "nav_cache"

_G = {}


def office_model():
    """(m, d) of the static office, forwarded.  Cached per process."""
    if "m" not in _G:
        m = S.load_office_spec().compile()
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        _G["m"], _G["d"] = m, d
    return _G["m"], _G["d"]


def model_key(m) -> str:
    h = hashlib.sha1()
    for a in (m.geom_pos, m.geom_size, m.geom_quat, m.body_pos):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(np.array([m.ngeom, m.nbody], np.int64).tobytes())
    return h.hexdigest()[:12]


def grid_bounds(margin=0.3):
    P = np.concatenate([np.array(p) for p in S.office_meta()["rooms"].values()])
    x0, y0 = P[:, 0].min() - margin, P[:, 1].min() - margin
    x1, y1 = P[:, 0].max() + margin, P[:, 1].max() + margin
    return math.floor(x0 * 20) / 20, math.floor(y0 * 20) / 20, x1, y1


def _rows(args):
    iy0, iy1, x0, y0, nx, res = args
    m, d = office_model()
    top = N._geom_top(m, d)
    occ = np.zeros((iy1 - iy0, nx), bool)
    mask = np.array([0, 0, 0, 1, 1, 0], np.uint8)
    up = np.array([0.0, 0.0, 1.0])
    gid = np.zeros(1, np.int32)
    ray = mujoco.mj_ray
    pnt = np.array([0.0, 0.0, 0.0])
    for iy in range(iy0, iy1):
        y = y0 + (iy + 0.5) * res
        for ix in range(nx):
            x = x0 + (ix + 0.5) * res
            z0 = -0.3
            for _ in range(6):
                pnt[0], pnt[1], pnt[2] = x, y, z0
                dist = ray(m, d, pnt, up, mask, 1, -1, gid)
                if dist < 0:
                    break
                zh = z0 + dist
                if zh >= N.ROBOT_TOP:
                    break
                if top[int(gid[0])] > N.Z_BASE:
                    occ[iy - iy0, ix] = True
                    break
                z0 = zh + 0.002
    return iy0, occ


def build_office_grid(res=0.01, use_cache=True, workers=None, verbose=False) -> N.Grid:
    """Fine (1 cm default) occupancy Grid of the static office (with EDT).  ~1-2 min uncached at 1 cm on 16 workers."""
    m, d = office_model()
    x0, y0, x1, y1 = grid_bounds()
    nx, ny = int(math.ceil((x1 - x0) / res)), int(math.ceil((y1 - y0) / res))
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"occ_office_{model_key(m)}_{res:.3f}.npz"
    if use_cache and f.exists():
        z = np.load(f)
        return N.Grid(float(z["res"]), float(z["x0"]), float(z["y0"]), z["occ"]).with_edt()
    workers = workers or min(16, max(1, (os.cpu_count() or 2) // 2))
    band = 40
    jobs = [(a, min(a + band, ny), x0, y0, nx, res) for a in range(0, ny, band)]
    occ = np.zeros((ny, nx), bool)
    with mp.get_context("fork").Pool(workers) as pool:
        for iy0, o in pool.imap_unordered(_rows, jobs):
            occ[iy0:iy0 + o.shape[0]] = o
    np.savez_compressed(f, res=res, x0=x0, y0=y0, occ=occ)
    if verbose:
        print(f"occupancy {nx}x{ny} saved {f}", flush=True)
    return N.Grid(res, x0, y0, occ).with_edt()


def office_doors() -> list:
    """nav.door_list-equivalent entries (numpy centre / wall_dir / normal) for the two doorways."""
    out = []
    for dr in S.office_meta()["doors"]:
        e = dict(dr)
        for k in ("centre", "wall_dir", "normal"):
            e[k] = np.array(dr[k], float)
        e["bodies"] = []
        e["asset"] = "office_wall_gap"
        out.append(e)
    return out


def in_room(xy, meta=None):
    """room name containing xy ('lab' / 'corridor' / 'cafeteria') or None (doorway slabs return 'door')."""
    from .. import build as B
    meta = meta or S.office_meta()
    for k, poly in meta["rooms"].items():
        if B._in_poly(tuple(xy), poly):
            return k
    for dr in meta["doors"]:
        c = np.array(dr["centre"])
        if abs(xy[0] - c[0]) <= dr["thickness"] / 2 and abs(xy[1] - c[1]) <= dr["clear_w"] / 2:
            return "door"
    return None
