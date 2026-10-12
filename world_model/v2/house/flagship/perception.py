"""flagship/perception.py -- ZED stereo capture -> zonotopes -> plan, on a PERCEPTION TWIN of the flagship world.  SIMULATION ONLY.

The main sim (flat colours) and the twin (same spec + speckle textures + ZED stereo cameras, see scene.build_twin) share every static object; at a capture
the twin's robot is posed at the main sim's base pose (set_base_pose) and the stereo pair is rendered there.  Pipeline = zed_demo (SGBM, self-filter,
RANSAC planes, DBSCAN, zonotopes inflated by the measured stereo p95).  Differences to zed_demo that are flagged in the README:
  * counter ROI is wider/deeper; clusters beyond the 1.5 m calibration range are re-inflated with the z^2 law (reinflate_far);
  * the payload (tray + cup) zonotope group gets cup points (the twin carries the testbed ball instead of the cup);
  * ground truth (MuJoCo depth/segmentation) is used for evaluation numbers and the labels printed in the video only.
Two modes: 'live' (twin + EGL renderer, needs a GPU slot only for the 2 captures) and 'cached' (load pickles written by a live run -> CPU only).
Place-planner switch: backend 'zed' (default, zed_demo.plan) | 'zono_rows' (hook for the FL-MPC zonotope-row work; falls back to 'zed' if absent).
"""
from __future__ import annotations

import importlib
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np

from . import scene as FS

OUT = FS.OUT
COUNTER_ROI = ((0.05, 2.0), (-2.5, 1.0))
ZMAX = 1.55
CUP_R, CUP_H = 0.045, 0.115
CALIB_RANGE = 1.49


def _robust_error_at_range(ev, z, key="p95"):
    """stereo.error_at_range that skips range bins without valid matches (None) and extrapolates with the z^2 law from the nearest valid bin."""
    b = [r for r in ev["bins"] if r.get(key) is not None]
    for r in b:
        if r["lo"] <= z < r["hi"]:
            return float(r[key])
    r = min(b, key=lambda r: abs(0.5 * (r["lo"] + r["hi"]) - z))
    c = 0.5 * (r["lo"] + r["hi"])
    return float(r[key] * (z / c) ** 2) if z > c else float(r[key])


def _patch_stereo():
    from world_model.v2.house.zed_demo import stereo as ST
    ST.error_at_range = _robust_error_at_range            # runtime patch of the module attribute (zed_demo files are not edited)


def zono_pack(Z):
    return dict(c=np.asarray(Z.c, float), G=np.asarray(Z.G, float))


def zono_unpack(d):
    from world_model.v2.tray_rep.sets.zono import Zonotope
    return Zonotope(d["c"], d["G"])


def reinflate_far(res, calib=CALIB_RANGE):
    """clusters measured beyond the stereo calibration range: delta *= (range/calib)^2 (quadratic stereo error law), Zi recomputed."""
    from world_model.v2.house.zed_demo import perceive as PC
    n = 0
    for z in res["zonos"]:
        if z["range_m"] > calib:
            f = (z["range_m"] / calib) ** 2
            z["delta"] = float(z["delta"] * f)
            z["Zi"] = PC.inflate_world(z["Zx"], z["delta"])
            n += 1
    return n


class Twin:
    """perception twin + ZED rig; payload zonotopes (robot's own geometry incl. a cup) at the build pose."""

    def __init__(self, seed=0, pitch=30.0):
        import cv2
        cv2.setNumThreads(1)
        _patch_stereo()
        import mujoco
        from world_model.v2.house import phase1 as P1
        from world_model.v2.house.zed_demo import rig as ZR, perceive as PC, stereo as ST, plan as PL
        self.mujoco, self.ZR, self.PC, self.ST, self.PL = mujoco, ZR, PC, ST, PL
        self.C = FS.build_twin(pitch)
        self.m, self.d = self.C.m, self.C.d
        self.rig = ZR.ZedRig(self.m, seed=seed)
        self.K = self.rig.K
        self.cls = P1.geom_class(self.m)
        self.keys = PC.gt_keys(self.m)
        self.sgbm = ST.make_sgbm()
        self.base0 = np.array(self.C.info["spawn"], float)
        self.yaw0 = float(self.C.info["yaw"])
        self.tray_id = self.m.body("tray").id
        self.payload = self._payload()
        self.qpos0, self.mocap0 = self.d.qpos.copy(), self.d.mocap_pos.copy()
        self.seed = seed

    def _payload(self):
        """PL.payload_zonotopes + cup points merged into the 'tray' group."""
        from world_model.v2.perception_zono.fit import fit_hull_zono
        m, d, PC = self.m, self.d, self.PC
        self.mujoco.mj_forward(m, d)
        pay = dict(self.PL.payload_zonotopes(self.C, self.cls))
        tray_c = d.xpos[self.tray_id]
        Rt = d.xmat[self.tray_id].reshape(3, 3)
        ang = np.linspace(0, 2 * math.pi, 12, endpoint=False)
        pts = []
        for zz in (0.006, CUP_H):
            for a in ang:
                pts.append(tray_c + Rt @ np.array([-0.045 + CUP_R * math.cos(a), CUP_R * math.sin(a), zz]))
        # tray group: rebuild from its geoms + cup ring points
        gs = [g for g in range(m.ngeom) if self.cls[g] in "TP" and (m.geom_contype[g] or m.geom_conaffinity[g])
              and m.geom_type[g] != self.mujoco.mjtGeom.mjGEOM_PLANE]
        P = np.concatenate([PC.geom_pts(m, d, g) for g in gs] + [np.array(pts)])
        pay["tray"] = fit_hull_zono(P, order=2)
        return list(pay.items())

    def payload_at(self, base_xy, lift=0.0, extra=(0.0, 0.0)):
        """payload translated to base position base_xy (+extra), arms/tray raised by `lift` (same yaw as the build pose)."""
        from world_model.v2.tray_rep.sets.zono import Zonotope
        out = []
        t = np.array([base_xy[0] - self.base0[0] + extra[0], base_xy[1] - self.base0[1] + extra[1], 0.0])
        for n, Z in self.payload:
            up = lift if (n.startswith(("left_arm", "right_arm")) or n == "tray") else 0.0
            out.append((n, Zonotope(Z.c + t + np.array([0, 0, up]), Z.G)))
        return out

    def capture(self, pose, kind, tries=3):
        """pose=(x,y,yaw) of the main sim's base.  Returns a result dict (plan inside for 'desk', approach inside for 'counter')."""
        import mujoco
        from world_model.v2.house.office import compose as CP
        PC, PL = self.PC, self.PL
        m, d = self.m, self.d
        if abs(pose[2] - self.yaw0) > 0.2:
            raise ValueError("twin payload assumes the build yaw (pi/2)")
        d.qpos[:], d.mocap_pos[:] = self.qpos0, self.mocap0       # robot + virtual tray back at the build pose (undo hiding / previous capture)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        CP.set_base_pose(self.C, float(pose[0]), float(pose[1]), float(pose[2]))
        base = np.array(pose[:2], float)
        yaw = float(pose[2])
        if kind == "counter":                      # the real hands are empty at the counter: hide the twin's virtual tray/ball
            for jn in ("tray_free", "ball_free"):
                a = m.jnt_qposadr[m.joint(jn).id]
                d.qpos[a + 2] = -9.0
            mujoco.mj_forward(m, d)
        tray0 = d.xpos[self.tray_id].copy()
        roi = COUNTER_ROI if kind == "counter" else ((0.05, 1.47), (-1.30, 0.80))
        best = None
        for it in range(tries):
            t0 = time.time()
            o = self.rig.capture(d)
            res = PC.run(m, d, self.rig, o, self.cls, self.keys, base, yaw, seed=self.seed + it, sgbm=self.sgbm, roi=roi, zmax=ZMAX)
            n_far = reinflate_far(res)
            sup = [z for z in res["zonos"] if z["kind"] == "support"]
            ok = bool(sup) if kind == "desk" else True
            best = (o, res, sup, n_far, time.time() - t0)
            if ok:
                break
        o, res, sup, n_far, t_wall = best
        origin, R = self.rig.left_pose(d)
        out = dict(kind=kind, pose=[float(v) for v in pose], tray0=tray0.tolist(), left=o["left"], disp=res["disp"].astype(np.float32),
                   K=dict(self.K), origin=origin, R=R, n_far_reinflated=n_far, timing=res["timing"], capture_wall_s=t_wall,
                   z_desk=res["z_desk"], ev_scene={k: v for k, v in res["ev_scene"].items() if k != "bins"})
        zl = []
        for i, z in enumerate(res["zonos"]):
            zl.append(dict(id=i, kind=z["kind"], gt_key=z["gt_key"], n_pts=z["n_pts"], delta=z["delta"], range_m=z["range_m"],
                           Z=zono_pack(z["Z"]), Zi=zono_pack(z["Zi"])))
        out["zonos"] = zl
        if kind == "desk":                          # the robot's own payload zonotopes at the current base pose (used by the zono_rows place backend only)
            out["payload"] = [(n, np.asarray(Z.c, float), np.asarray(Z.G, float)) for n, Z in self.payload_at(np.array(self.d.mocap_pos[0][:2]))]
        try:
            rec = PC.gt_recovery(res)
            out["gt_recovered"] = f"{sum(v['recovered'] for v in rec.values())}/{len(rec)}"
        except Exception:
            out["gt_recovered"] = None
        if kind == "desk":
            out["plan"] = self._plan_desk(res, sup, tray0)
        else:
            out["approach"] = self._approach_counter(res, base)
        return out

    # ------------------------------------------------------------------------------------------------ desk plan
    def _plan_desk(self, res, sup, tray0):
        PL = self.PL
        obst = [(f"Z{i}:{z['gt_key']}", z["Zi"]) for i, z in enumerate(res["zonos"])]
        if not sup:
            return dict(ok=False)
        # payload zonotopes are built at the desk stand-off pose; the actual base pose is a pure translation of payload and tray0
        base_now = np.array(self.d.mocap_pos[0][:2])
        pay = self.payload_at(base_now)
        info = PL.plan_task(pay, obst, sup[0]["Z"], tray0[:2])
        ok = info["n_feasible"] > 0 and info.get("path_t") is not None
        plan = dict(ok=bool(ok), n_feasible=int(info["n_feasible"]), c_at_spot=info["c_at_spot"], t_map_s=info["t_map_s"])
        if ok:
            path_t = np.asarray(info["path_t"], float)
            ex = [PL.exact_clearance(pay, obst, t) for t in path_t[::max(len(path_t) // 25, 1)]]
            naive = np.stack([np.linspace(0, 0.0, 60), np.linspace(0, 0.405, 60)], 1)
            exn = [PL.exact_clearance(pay, obst, t) for t in naive[::6]]
            plan.update(spot_t=np.asarray(info["spot_t"], float), path_t=path_t, spot_xy=(tray0[:2] + info["spot_t"]).tolist(),
                        path_min_exact_clear_m=float(min(x[0] for x in ex)), path_limiting_pair=min(ex, key=lambda x: x[0])[1],
                        naive_exact_clear_m=float(min(x[0] for x in exn)), naive_limiting_pair=min(exn, key=lambda x: x[0])[1],
                        naive_path_t=naive, path_len_m=float(np.linalg.norm(np.diff(path_t, axis=0), axis=1).sum()))
            # clearance heat map for the video (tray-offset grid)
            Cm = info["Cm"]
            plan["map"] = dict(xs=info["xs"], ys=info["ys"], C=Cm.C.astype(np.float32), reach=info["reach"], on_desk=info["on_desk"])
        return plan

    # ------------------------------------------------------------------------------------------------ counter approach check
    def _approach_counter(self, res, base):
        """swept-volume check of the final approach (pre-dock -> dock, lift = h_pick) and the back-out (dock -> pre-dock, lift = h_pick + 0.06) against the
        inflated object zonotopes (exact zonotope-zonotope distance of the whole payload)."""
        PL = self.PL
        obst = [(f"Z{i}:{z['gt_key']}", z["Zi"]) for i, z in enumerate(res["zonos"]) if z["kind"] == "obj"]
        fwd = np.array([math.cos(FS_DOCK_YAW), math.sin(FS_DOCK_YAW)])
        out = {}
        for name, sgn, lift in (("approach", +1, FS.H_PICK), ("backout", -1, FS.H_PICK + FS.LIFT_CARRY_EXTRA)):
            vals = []
            s_list = np.linspace(0.0, 0.7, 15) if name == "approach" else np.linspace(0.0, 0.7, 8)
            b0 = base if name == "approach" else base + 0.7 * fwd
            for s in s_list:
                pay = self.payload_at(b0 + sgn * s * fwd, lift=lift)
                vals.append(PL.exact_clearance(pay, obst, (0.0, 0.0)))
            k = int(np.argmin([v[0] for v in vals]))
            out[name] = dict(min_clear_m=float(vals[k][0]), pair=vals[k][1], at_s=float(s_list[k]))
        return out

    def close(self):
        try:
            self.rig.close()
        except Exception:
            pass


FS_DOCK_YAW = math.pi / 2


def strip_heavy(r):
    """pickle-friendly copy (keeps images; drops the map for 'counter')."""
    return r


class PerceptionService:
    """services object for CoffeeRunOperator: sense_counter / sense_desk.  mode 'live' | 'cached'."""
    name = "zed"

    def __init__(self, mode="live", cache_dir=None, backend="zed", seed=0, log=print):
        self.mode, self.dir = mode, Path(cache_dir or OUT / "perception")
        self.backend, self.seed, self.log = backend, seed, log
        self.twin = None
        self.results = {}
        self.t_sense = {}

    def _get(self, kind, obs):
        if self.mode == "cached":
            with open(self.dir / f"perception_{kind}.pkl", "rb") as f:
                r = pickle.load(f)
            self.log(f"[perception] cached {kind}")
        else:
            if self.twin is None:
                self.log("[perception] building twin ...")
                self.twin = Twin(self.seed)
            t0 = time.time()
            r = self.twin.capture(obs["pose"], kind)
            self.log(f"[perception] {kind}: {len(r['zonos'])} zonotopes, capture+perceive {time.time() - t0:.1f}s, gt_recovered {r['gt_recovered']}")
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.dir / f"perception_{kind}.pkl", "wb") as f:
                pickle.dump(r, f)
        r["t_sim"] = float(obs["t"])
        self.results[kind] = r
        return r

    def sense_counter(self, obs):
        r = self._get("counter", obs)
        return r

    def sense_desk(self, obs):
        r = self._get("desk", obs)
        plan = r["plan"]
        if self.backend == "zono_rows":
            plan = _zono_rows_plan(r, obs, self.log) or plan
        if not plan.get("ok"):
            self.log("[perception] desk plan INFEASIBLE -> operator falls back to naive")
            return None
        return plan


def _zono_rows_plan(r, obs, log):
    """hook for the concurrent zono_rows work (zonotope HOCBF rows in FL-MPC).  Expected: world_model.v2.house.zono_rows.flagship_backend(result, obs) ->
    dict(spot_t, path_t, ok=True) or None.  Absent -> None (caller keeps the zed_demo planner)."""
    try:
        mod = importlib.import_module("world_model.v2.house.zono_rows")
        fn = getattr(mod, "flagship_backend", None)
        if fn is None:
            log("[perception] zono_rows present but has no flagship_backend(result, obs): using the zed_demo planner")
            return None
        return fn(r, obs)
    except ImportError:
        log("[perception] zono_rows not importable: using the zed_demo planner")
        return None
