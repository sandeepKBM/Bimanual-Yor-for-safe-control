"""proxy.py -- sphere ("keypoint") proxies for NERO links, for FL-MPC's arm-arm clearance rows.

Format = sim_common.barrier_keypoints: list of (body_id, local_offset(3,), radius).  Spheres are fitted to the vertices
of each link's OFFICIAL collision STL (in the body frame, incl. the URDF collision-origin transform) so that EVERY mesh
vertex lies inside its nearest sphere (conservative cover: proxy gap <= true gap; verified in tests/test_adapters.py).
Method: greedy farthest-point seeding + a few Lloyd steps until the largest sphere radius <= r_target or n == n_max.
The fitted set is cached as proxy_spheres_nyu.json next to this file (regenerate: python -m world_model.v2.nero.proxy).  proxy_spheres.json is the fit of the arm links
(base_link..flange; the same STLs) that build_nyu() reuses; its gripper entries are not used.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from . import config as C
from . import urdf_fk as U

CACHE = Path(__file__).with_name("proxy_spheres.json")                  # arm-link spheres reused by build_nyu() (its gripper entries are dropped there)
CACHE_NYU = Path(__file__).with_name("proxy_spheres_nyu.json")          # 11-03 F1: NYU gripper (palm incl. phone pole/clamp + both jaws)
# nyu gripper bodies, fitted to the compiled mesh of the visual geom (full palm.stl / finger_a.stl / finger_b.stl) in the BODY frame; (r_target, n_max)
# n_max for the palm is large: the NYU palm (housing + 157 mm phone pole) is much longer; ASSUMED fit budget, not tuned.
FIT_NYU = {"gripper_base": (0.05, 6), "finger1": (0.04, 4), "finger2": (0.04, 4)}


def cache_path():
    return CACHE_NYU
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


def build_nyu():
    """arm links (base_link..flange) = the proxy_spheres.json entries (same STLs, no change); gripper_base/finger1/finger2 = spheres on the NYU meshes.
    The mesh vertices come from the COMPILED model (MuJoCo mesh_vert + the visual geom's pose in its body frame), i.e. exactly the
    placement the scene uses (palm: adapter pose; jaws: pivot-centred body frame, so the spheres move with the hinge)."""
    import mujoco
    from . import build_variant as BV
    out = {k: v for k, v in json.loads(CACHE.read_text()).items() if k not in FIT_NYU}
    m = mujoco.MjSpec.from_string(BV.variant_xml(lift=False, gravcomp=False)).compile()
    for body, (rt, nm) in FIT_NYU.items():
        verts = {}
        for side in C.SIDES:
            b = m.body(f"{side}_arm_{body}").id
            g = [i for i in range(m.ngeom) if m.geom_bodyid[i] == b and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
                 and (m.geom(i).name or "").endswith("_vis")]
            assert len(g) == 1, (body, g)
            g = g[0]
            a, n = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
            R = np.zeros(9); mujoco.mju_quat2Mat(R, m.geom_quat[g])
            verts[side] = m.mesh_vert[a:a + n].astype(float) @ R.reshape(3, 3).T + m.geom_pos[g]
        assert np.allclose(verts["left"], verts["right"], atol=1e-9), body     # both sides are the same subtree
        c, r = fit_spheres(verts["left"], rt, nm)
        out[body] = dict(centers=c.round(6).tolist(), radii=r.round(5).tolist())
    return out


def load():
    cp = cache_path()
    if not cp.exists():
        cp.write_text(json.dumps(build_nyu(), indent=1))
    return json.loads(cp.read_text())


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
    cache_path().write_text(json.dumps(build_nyu(), indent=1))
    s = load()
    print({k: (len(v["radii"]), max(v["radii"])) for k, v in s.items()}, sum(len(v["radii"]) for v in s.values()))
