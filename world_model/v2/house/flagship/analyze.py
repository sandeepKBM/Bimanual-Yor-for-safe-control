"""flagship/analyze.py -- post-hoc clearance analysis of a recorded run (replay of qpos/mocap; no physics).  SIMULATION ONLY.

For each side: min distance of the robot's own collision geoms (arms, hands, chassis, tray) to
  (a) the INFLATED stereo zonotopes of the counter objects / the desk clutter (lower bound, ZUnion.outside_depth; 0 = inside), from the SHIELD side's captures
      (the raw side has no perception, but is measured against the same zonotopes);
  (b) the true MuJoCo collision geoms of those objects (ground truth, mj_geomDistance).
Windows: counter = phases RAISE..LOWER_C (+ the final 1 s of OUT); desk = CREEP_D..DONE.  Every STRIDE ticks.
Writes run_<tag>_analysis.json.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import mujoco

from . import scene as FS
from . import mission as MS

OUT = FS.OUT
COUNTER_KEYS = ("coffee_machine", "mug_counter", "milk_jug", "sugar_jar", "napkin_box", "cup_stack", "counter")
DESK_KEYS = ("bottle_a", "mug_desk1", "book_a", "pencup_a", "box_a", "monitor_1", "lamp_1", "desk_1")
PERSON_KEYS = ("person_0", "person_1", "person_2")


def zono_union(zonos, kinds=("obj",), drop_keys=()):
    from world_model.v2.house.flagship.perception import zono_unpack
    from world_model.v2.perception_zono.zonoutil import ZUnion
    zs = [zono_unpack(z["Zi"]) for z in zonos if z["kind"] in kinds and z["gt_key"] not in drop_keys]
    return ZUnion(zs) if zs else None


def replay_setup(seed=0, cafe_walker=False):
    sim = FS.new_sim(seed, cafe_walker=cafe_walker)
    return sim


def analyze(tag, seed=0, perception_dir=None, stride=5, out=OUT, sim=None):
    out = Path(out)
    from world_model.v2.house.zed_demo import perceive as PC
    from world_model.v2.house import phase1 as P1
    z = np.load(out / f"run_{tag}.npz")
    meta = json.loads((out / f"run_{tag}.json").read_text())["meta"]
    pdir = Path(perception_dir or out / "perception")
    percs = {}
    for k in ("counter", "desk"):
        f = pdir / f"perception_{k}.pkl"
        if f.exists():
            percs[k] = pickle.load(open(f, "rb"))
    sim = sim or replay_setup(seed, meta.get("walker", False))
    m, d = sim.m, sim.d
    keys = PC.gt_keys(m)
    cls = P1.geom_class(m)
    col = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
    own = [g for g in range(m.ngeom) if cls[g] in "LRBT" and col[g] and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]
    sets = {"counter": COUNTER_KEYS, "desk": DESK_KEYS}
    gt_geoms = {k: np.array([g for g in range(m.ngeom) if keys[g] in sets[k] and col[g] and m.geom_group[g] in (3, 4)]) for k in sets}
    persons = np.array([g for g in range(m.ngeom) if keys[g] in PERSON_KEYS and col[g]])
    U = {}
    if "counter" in percs:
        U["counter"] = zono_union(percs["counter"]["zonos"], ("obj",), drop_keys=("wall",))
    if "desk" in percs:
        U["desk"] = zono_union(percs["desk"]["zonos"], ("obj", "support"))
    ph = z["s_phase"]
    P = MS.PHASE_ID
    win = {"counter": (ph >= P["RAISE"]) & (ph <= P["LOWER_C"]), "desk": ph >= P["CREEP_D"]}
    ft = np.zeros(6)
    res = {"counter": dict(frames=0), "desk": dict(frames=0)}
    stand_id = m.body("tp_stand").id
    for wname, mask in win.items():
        idx = np.flatnonzero(mask)[::stride]
        zmin, gmin, gobj, zt = 9.0, 9.0, {}, None
        for i in idx:
            d.qpos[:] = z["qpos"][i]
            d.mocap_pos[:] = z["mocap_pos"][i]
            d.mocap_quat[:] = z["mocap_quat"][i]
            st = z["stand"][i]
            m.body_pos[stand_id] = st[:3] if st[3] > 0 else FS.STAND_PARK
            mujoco.mj_forward(m, d)
            if wname in U and U[wname] is not None:
                Pr = np.concatenate([PC.geom_pts(m, d, g) for g in own])
                dz = float(U[wname].outside_depth(Pr).min())
                if dz < zmin:
                    zmin, zt = dz, float(z["s_t"][i])
            for b in gt_geoms[wname]:
                for a in own:
                    if np.linalg.norm(d.geom_xpos[a] - d.geom_xpos[b]) - m.geom_rbound[a] - min(m.geom_rbound[b], 6.0) > 0.5:
                        continue
                    v = mujoco.mj_geomDistance(m, d, a, int(b), 0.5, ft)
                    if 0.0 <= v < 0.5 and abs(np.linalg.norm(ft[3:] - ft[:3]) - v) > 1e-4:      # spurious zero (same sanity check as zed_demo ClearMon)
                        continue
                    if v < 0.5:
                        k = str(keys[b])
                        gobj[k] = min(gobj.get(k, 9.0), float(v))
                        gmin = min(gmin, float(v))
        res[wname] = dict(frames=int(len(idx)), min_zonotope_clearance_m=zmin if zmin < 9 else None, at_t=zt,
                          min_true_clearance_m=gmin if gmin < 9 else None, per_object_true_min_m={k: round(v, 4) for k, v in sorted(gobj.items())})
    m.body_pos[stand_id] = FS.STAND_PARK
    vals = [v["min_zonotope_clearance_m"] for v in res.values() if v.get("min_zonotope_clearance_m") is not None]
    res["min_zonotope_clearance_m"] = float(min(vals)) if vals else None
    tv = [v["min_true_clearance_m"] for v in res.values() if isinstance(v, dict) and v.get("min_true_clearance_m") is not None]
    res["min_true_object_clearance_m"] = float(min(tv)) if tv else None
    (out / f"run_{tag}_analysis.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perception-dir", default=None)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    print(json.dumps(analyze(a.tag, a.seed, a.perception_dir, out=a.out), indent=1))
    sys.stdout.flush()
    os._exit(0)
