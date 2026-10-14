"""proxy.py -- sphere ("keypoint") proxies for NERO links, for FL-MPC's arm-arm clearance rows.

Format = sim_common.barrier_keypoints: list of (body_id, local_offset(3,), radius).  Spheres are fitted to the vertices
of each link's OFFICIAL collision STL (in the body frame, incl. the URDF collision-origin transform) so that EVERY mesh
vertex lies inside its nearest sphere (conservative cover: proxy gap <= true gap; verified in tests/test_adapters.py).
Method: greedy farthest-point seeding + a few Lloyd steps until the largest sphere radius <= r_target or n == n_max.
The fitted set is cached as proxy_spheres.json next to this file (regenerate: python -m world_model.v2.nero.proxy).
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from . import config as C
from . import urdf_fk as U

CACHE = Path(__file__).with_name("proxy_spheres.json")
# link -> (r_target [m], n_max): small, near-round links get 1-2 spheres, long links 4-5
FIT = {"base_link": (0.05, 3), "link1": (0.05, 3), "link2": (0.05, 5), "link3": (0.05, 5), "link4": (0.05, 4), "link5": (0.05, 5),
       "link6": (0.045, 2), "flange_link": (0.04, 1), "gripper_base": (0.05, 4),
       "gripper_link1": (0.04, 3), "gripper_link2": (0.04, 3), "link7": (0.02, 1)}
BODY = {"base_link": "link0", "link1": "link1", "link2": "link2", "link3": "link3", "link4": "link4", "link5": "link5", "link6": "link6",
        "link7": "link7", "flange_link": "flange", "gripper_base": "gripper_base", "gripper_link1": "finger1",
        "gripper_link2": "finger2"}


def load_stl(fn):
    b = Path(fn).read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    a = np.frombuffer(b[84:84 + 50 * n], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return a["v"].reshape(-1, 3).astype(float)


def fit_spheres(V, r_target, n_max, iters=8):
    V = np.unique(np.round(V, 5), axis=0)
    cen = [V.mean(0)]
    for _ in range(n_max):
        Cc = np.array(cen)
        for _ in range(iters):
            lab = np.argmin(((V[:, None, :] - Cc[None]) ** 2).sum(-1), 1)
            Cc = np.array([V[lab == k].mean(0) if np.any(lab == k) else Cc[k] for k in range(len(Cc))])
        lab = np.argmin(((V[:, None, :] - Cc[None]) ** 2).sum(-1), 1)
        rad = np.array([np.linalg.norm(V[lab == k] - Cc[k], axis=1).max() if np.any(lab == k) else 0.0 for k in range(len(Cc))])
        cen = list(Cc)
        if rad.max() <= r_target or len(cen) >= n_max:
            break
        # split the worst sphere: seed at its farthest vertex
        k = int(np.argmax(rad))
        far = V[lab == k][np.argmax(np.linalg.norm(V[lab == k] - Cc[k], axis=1))]
        cen.append(far)
    Cc = np.array(cen)
    lab = np.argmin(((V[:, None, :] - Cc[None]) ** 2).sum(-1), 1)
    rad = np.array([np.linalg.norm(V[lab == k] - Cc[k], axis=1).max() if np.any(lab == k) else 0.0 for k in range(len(Cc))])
    keep = rad > 0
    return Cc[keep], rad[keep] + 1e-4


def build():
    links, _ = U.parse()
    out = {}
    for un, (rt, nm) in FIT.items():
        L = links[un]
        stem = Path(L.coll_mesh).stem
        which = "agx_gripper" if "agx_gripper" in L.coll_mesh else "nero"
        V = load_stl(C.MESH_DIR[which] / f"{stem}.stl")
        R = U.rpy_to_R(*L.coll_rpy)
        V = V @ R.T + L.coll_xyz                         # collision mesh -> link/body frame
        c, r = fit_spheres(V, rt, nm)
        out[BODY[un]] = dict(centers=c.round(6).tolist(), radii=r.round(5).tolist())
    return out


def load():
    if not CACHE.exists():
        CACHE.write_text(json.dumps(build(), indent=1))
    return json.loads(CACHE.read_text())


def keypoints(model, prefix: str, spec=None):
    """[(body_id, offset(3,), radius)] for the arm `prefix` ('left_arm_' / 'right_arm_')."""
    spec = spec or load()
    out = []
    for body, d in spec.items():
        bid = model.body(prefix + body).id
        for c, r in zip(d["centers"], d["radii"]):
            out.append((bid, np.array(c), float(r)))
    return out


if __name__ == "__main__":
    CACHE.write_text(json.dumps(build(), indent=1))
    s = load()
    print({k: (len(v["radii"]), max(v["radii"])) for k, v in s.items()}, sum(len(v["radii"]) for v in s.values()))
