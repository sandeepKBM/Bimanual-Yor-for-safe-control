"""ocs/record.py -- record a run ONCE (truth + the sensor streams the producers need) so E2-E4 replay offline without re-simulating.  SIMULATION ONLY.

    python -m world_model.v2.house.ocs.record carry --seeds 0 1 2 3 4 5 [--procs 6]      # h5 carry, shield ON (truth people), people_clock="time"
    python -m world_model.v2.house.ocs.record flagship                                  # replays the flagship desk-start run (qpos per tick) + cached stereo zonotopes

Per 50 Hz tick (state at the START of the tick, i.e. exactly what the shield reads): truth (base, tray, cup, people, joint q), encoders (q + the FL-MPC noise),
loc PoseEstimator output (wheel odometry + ZED VIO), person visibility (sim line of sight; EVALUATION only).  Per 10 Hz capture: the head-camera depth
frame (noisy world points z in (0.1, 2.1) m + the static-model re-cast depth for background subtraction; people_track.camera.HeadCamera) and the wrist-camera
red-blob centroids (RGB colour threshold; grasp/wrist.WristCam.seg_cup_proprio).  Output: <OUT>/rec/<name>.pkl.gz (views.Recording).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("BIMANUAL_TELEOP_FAST", "1")

import numpy as np

from world_model.v2.tray_rep.sets.zono import Zonotope

OUT = Path(os.environ.get("OCS_OUT", "/data/local/ss5772/molmospaces/ocs"))
TICK = 0.02
HEAD_HZ = 10.0


class LiveSensors:
    """head camera (depth frames at HEAD_HZ) + wrist cameras (cup blob) on a running TeleopSim.  Truth enters only as what the cameras image."""

    def __init__(self, sim, seed=0, head_hz=HEAD_HZ):
        from world_model.v2.house.people_track import camera as CAM, detectors as DET
        from world_model.v2.house.grasp.wrist import WristCam
        self.cam = CAM.HeadCamera(sim, CAM.head_spec(W=160, H=90, rate_hz=head_hz, latency_ticks=3))
        self.vis = DET.Visibility(self.cam)
        self.wc = {s: WristCam(sim.m, s) for s in ("left", "right")}
        self.rate = head_hz * TICK
        self.seed = seed

    def due(self, k):
        return math.floor((k + 1) * self.rate) > math.floor(k * self.rate)

    def head_frame(self, sim):
        k = sim.k
        if not self.due(k):
            return None
        fr = self.cam.render(k, self.seed + 7)
        z = fr["pts"][:, 2]
        keep = (z > 0.10) & (z < 2.1)                          # rows the depth detector would drop anyway (floor / ceiling)
        sub = dict(origin=fr["origin"], R=fr["R"], pts=fr["pts"][keep], z=fr["z"][keep], pix=fr["pix"][keep], person=fr["person"][keep], geom=None, tick=k)
        zs = self.cam.static_depth(sub, np.arange(len(sub["pts"])))
        return dict(tick=k, origin=fr["origin"].copy(), R=fr["R"].copy(), base_true=np.array([sim.base.x, sim.base.y, sim.base.yaw]),
                    pts=sub["pts"].astype(np.float32), z=sub["z"].astype(np.float32), pix=sub["pix"].astype(np.uint8), zs=zs.astype(np.float32),
                    person=sub["person"].astype(np.int8))

    def wrist(self, sim):
        k = sim.k
        if not self.due(k):
            return None
        out = {}
        for s, w in self.wc.items():
            r = w.seg_cup_proprio(sim.d, tick=k, seed=self.seed)
            out[s] = None if r is None else (int(r[0]), float(r[1]), float(r[2]))
        return out

    def visibility(self, sim):
        return [self.vis.of(sim, xy) for xy, _ in sim.people_state]


class Collector:
    def __init__(self):
        self.rows = {}

    def add(self, **kw):
        for k, v in kw.items():
            self.rows.setdefault(k, []).append(v)

    def arrays(self):
        out = {}
        for k, v in self.rows.items():
            out[k] = np.asarray(v)
        return out


def _snapshot(c: Collector, sim, sensors, loc, rng, with_vis=True):
    from world_model.v2.house.teleop.arm import ENC_Q, ENC_V
    d, ctl = sim.d, sim.ctl
    tL, tR = d.qpos[ctl.lq].copy(), d.qpos[ctl.rq].copy()
    tdL, tdR = d.qvel[ctl.ld].copy(), d.qvel[ctl.rd].copy()
    npeople = len(sim.people_state) if sim.with_people else 0
    wr = sensors.wrist(sim) if sensors is not None else None
    c.add(t=sim.t, base=np.array([sim.base.x, sim.base.y, sim.base.yaw]), base_v=sim.base.v.copy(), lift=sim.lift_actual(),
          tL=tL, tR=tR, tdL=tdL, tdR=tdR,
          qL=tL + rng.normal(0, ENC_Q, 6), qR=tR + rng.normal(0, ENC_Q, 6), qdL=tdL + rng.normal(0, ENC_V, 6), qdR=tdR + rng.normal(0, ENC_V, 6),
          lift_enc=sim.lift_actual() + rng.normal(0, 1e-4),
          tray_p=d.xpos[sim.tray].copy(), tray_R=d.xmat[sim.tray].copy(), tray_q=d.xquat[sim.tray].copy(),
          cup_p=d.xpos[sim.cup].copy(), cup_R=d.xmat[sim.cup].copy(), cup_q=d.xquat[sim.cup].copy(),
          ppl_xy=np.array([xy for xy, _ in sim.people_state], float).reshape(npeople, 2) if npeople else np.zeros((0, 2)),
          ppl_v=np.array([v for _, v in sim.people_state], float).reshape(npeople, 2) if npeople else np.zeros((0, 2)),
          ppl_vis=np.array(sensors.visibility(sim), float).reshape(npeople, 3) if (sensors is not None and npeople and with_vis) else np.zeros((npeople, 3)),
          loc_pose=loc.pose.copy(), loc_sigma=float(loc.sigma_xy), loc_v=loc.v_odo.copy(),
          wrist_due=wr is not None,
          wrist=np.array([[np.nan] * 3 if (wr is None or wr[s] is None) else list(wr[s]) for s in ("left", "right")], float))
    return wr


def record_carry(seed: int, out: Path = OUT, max_s: float = 30.0, tail_s: float = 1.0, people_clock: str = "time", verbose=True):
    from world_model.v2.house.teleop.sim import TeleopSim
    from world_model.v2.house.teleop import operator as OP, shield as SH
    from world_model.v2.house.loc.estimator import PoseEstimator
    from world_model.v2.house.ocs.views import Recording
    t_wall = time.time()
    sim = TeleopSim("h5", grasped=True, seed=seed, with_people=True, people_clock=people_clock)
    sc = sim.scene
    wp = sc.waypoints
    op = OP.SloppyOperator(seed, "carry", sc.out_route, pick_pose=wp["cafe_counter_pick"], place_pose=wp["lab_desk_place"], params=None,
                           back_route=sc.back_route, back_mode=sc.back_mode)
    shield = SH.Shield(None, occ=sim.occ_ctrl, a_hw=sim.a_hw)
    shield.docks = [wp["cafe_counter_pick"][:2]]
    sp = wp["lab_start"]
    loc = PoseEstimator(None, seed=seed, true_pose0=(sp[0], sp[1], sp[2]))
    sensors = LiveSensors(sim, seed)
    rng = np.random.default_rng(70_000 + seed)
    c = Collector()
    frames = {}
    done_t = None
    for k in range(int(max_s / TICK)):
        obs = sim.observe()
        cmd = op.step(obs)
        _snapshot(c, sim, sensors, loc, rng)
        fr = sensors.head_frame(sim)
        if fr is not None:
            frames[sim.k] = fr
        safe, info = shield.filter(sim, cmd)
        o = sim.tick(safe)
        loc.update(sim.t, sim.pose(), TICK)
        if op.done and done_t is None:
            done_t = o["t"]
        if done_t is not None and o["t"] - done_t >= tail_s:
            break
    A = c.arrays()
    meta = dict(scene="h5", seed=seed, task="carry", dt=TICK, people_names=[p.name for p in sc.people], people_clock=people_clock, mu_true=0.25,
                n=len(A["t"]), wall_s=time.time() - t_wall, head_hz=HEAD_HZ, shield="Shield(truth people)", fk_scene="h5")
    rec = Recording(A, meta, frames)
    path = Path(out) / "rec" / f"carry_s{seed}.pkl.gz"
    rec.save(path)
    if verbose:
        print(f"[record] carry seed {seed}: {meta['n']} ticks, {len(frames)} head frames, {meta['wall_s']:.0f}s -> {path}", flush=True)
    return str(path)


# ====================================================================================== flagship desk (replay of a recorded run)
def obb_zono(P: np.ndarray) -> Zonotope:
    """tight yaw-oriented box (min horizontal area) of world points -> Zonotope (the truth obstacle set)."""
    xy = P[:, :2]
    best = None
    for a in np.deg2rad(np.arange(0, 90, 2.0)):
        c, s = math.cos(a), math.sin(a)
        R = np.array([[c, s], [-s, c]])
        q = xy @ R.T
        lo, hi = q.min(0), q.max(0)
        area = float(np.prod(hi - lo))
        if best is None or area < best[0]:
            best = (area, a, lo, hi)
    _, a, lo, hi = best
    c, s = math.cos(a), math.sin(a)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    zlo, zhi = P[:, 2].min(), P[:, 2].max()
    cen_l = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (zlo + zhi) / 2])
    half = np.array([(hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, (zhi - zlo) / 2])
    return Zonotope(Rz @ cen_l, Rz @ np.diag(half))


def truth_obstacles(sim, keys):
    from world_model.v2.house.zed_demo import perceive as PC
    import mujoco
    m, d = sim.m, sim.d
    mujoco.mj_forward(m, d)
    gk = PC.gt_keys(m)
    out = []
    for key in sorted(keys):
        gs = [g for g in range(m.ngeom) if gk[g] == key and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]
        if not gs:
            continue
        P = np.concatenate([PC.geom_pts(m, d, g) for g in gs])
        out.append((key, obb_zono(P)))
    return out


def record_flagship(run_dir: Path, out: Path = OUT, seed: int = 0, name: str = "flagship_desk"):
    import mujoco
    from world_model.v2.house.flagship import scene as FS
    from world_model.v2.house.office import scene as OS
    from world_model.v2.house.loc.estimator import PoseEstimator
    from world_model.v2.house.ocs.views import Recording
    import pickle
    run_dir = Path(run_dir)
    z = np.load(run_dir / "run_shield_s0.npz", allow_pickle=True)
    with open(run_dir / "perception" / "perception_desk.pkl", "rb") as f:
        pres = pickle.load(f)
    wd = OS.office_meta()["waypoints"]["lab_desk_place"]
    sim = FS.new_sim(seed, cafe_walker=False, spawn=(wd["xy"][0], wd["xy"][1] - 0.15, wd["heading"]), grasped=True, counter=False)
    keys = {zz["gt_key"] for zz in pres["zonos"] if zz["kind"] == "obj" and zz["gt_key"]} | {"desk_1"}
    obstacles = truth_obstacles(sim, keys)
    sensors = LiveSensors(sim, seed)
    T = len(z["s_t"])
    loc = PoseEstimator(None, seed=seed, true_pose0=tuple(z["s_pose"][0]))
    rng = np.random.default_rng(70_000 + seed)
    c = Collector()
    d = sim.d
    dtq = TICK
    for k in range(T):
        d.qpos[:] = z["qpos"][k]
        d.mocap_pos[:] = z["mocap_pos"][k]
        d.mocap_quat[:] = z["mocap_quat"][k]
        d.qvel[:] = 0.0
        if k > 0:                                   # arm joint velocities from the recorded qpos differences (the run only stored qpos)
            d.qvel[sim.ctl.ld] = (z["qpos"][k][sim.ctl.lq] - z["qpos"][k - 1][sim.ctl.lq]) / dtq
            d.qvel[sim.ctl.rd] = (z["qpos"][k][sim.ctl.rq] - z["qpos"][k - 1][sim.ctl.rq]) / dtq
        mujoco.mj_forward(sim.m, d)
        sim.t = float(z["s_t"][k])
        sim.k = k
        sim.base.x, sim.base.y, sim.base.yaw = (float(v) for v in z["s_pose"][k])
        sim.base.v = np.asarray(z["s_v"][k], float)
        sim.people_state = []
        _snapshot(c, sim, sensors, loc, rng, with_vis=False)
        loc.update(sim.t, sim.pose(), TICK)
    A = c.arrays()
    A["ppl_xy"] = np.zeros((T, 0, 2)); A["ppl_v"] = np.zeros((T, 0, 2)); A["ppl_vis"] = np.zeros((T, 0, 3))
    meta = dict(scene="flagship", seed=seed, task="flagship_desk_shield", dt=TICK, people_names=[], mu_true=0.25, n=T, zono_t=float(pres.get("t_sim", 0.0)), fk_scene="flagship",
                head_hz=HEAD_HZ, source_run=str(run_dir))
    rec = Recording(A, meta, {}, pres, obstacles)
    path = Path(out) / "rec" / f"{name}.pkl.gz"
    rec.save(path)
    print(f"[record] flagship desk: {T} ticks, {len(obstacles)} truth obstacles -> {path}", flush=True)
    return str(path)


def _carry_job(a):
    return record_carry(*a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["carry", "flagship"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--run-dir", default=str(OUT / "flagship_desk"))
    a = ap.parse_args()
    if a.what == "carry":
        if a.procs > 1 and len(a.seeds) > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(min(a.procs, len(a.seeds)), maxtasksperchild=1) as pool:
                pool.map(_carry_job, [(s, Path(a.out)) for s in a.seeds])
        else:
            for s in a.seeds:
                record_carry(s, Path(a.out))
    else:
        record_flagship(Path(a.run_dir), Path(a.out))


if __name__ == "__main__":
    main()
