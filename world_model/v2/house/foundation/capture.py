"""capture.py -- render the head-ZED stereo pairs + ground truth from RECORDED teleop carry snapshots (stereo/record.py).  SIMULATION ONLY.

Runs in the mujoco env (EGL; use CUDA_VISIBLE_DEVICES=5 or 6).  Re-uses the other agent's textured perception twin (stereo/twin.py) and StereoSensor
read-only (no file of theirs is edited); the backend is a null backend, so this step only renders.

    python -m world_model.v2.house.foundation.capture stereo --tex speckle objects none --res 320x180 640x360 1280x720 --n 10
    python -m world_model.v2.house.foundation.capture pose --res 1280x720 --every 1

Output  <OUT>/stereo/<tex>_<W>x<H>.npz   : left, right (N,H,W,3 u8), gt_depth (N,H,W f32, m), gt_seg (N,H,W i32 geom id, -1 none), wall (N,H,W bool, EVAL ONLY),
                                         cls (N,H,W u8 FK class), origin, R, f, B, idx
        <OUT>/pose/seq_s<seed>_<W>x<H>.npz : same per consecutive 10 Hz snapshot + tray/cup truth world poses (EVAL ONLY)
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("BIMANUAL_TELEOP_FAST", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

OUT = Path(os.environ.get("FOUNDATION_OUT", "/data/local/ss5772/molmospaces/percept/foundation"))
SNAP = Path("/data/local/ss5772/molmospaces/percept/stereo/snap")


class NullBackend:
    name = "null"

    def warmup(self):
        pass

    def depth(self, left, right, K, baseline):
        z = np.full(left.shape[:2], np.nan, np.float32)
        return z, np.zeros(z.shape, bool), 0.0


def wall_geoms(m):
    """geom ids belonging to the room shell (walls / floor / ceiling bodies): textured or flat depending on the twin mode.  EVALUATION ONLY."""
    out = np.zeros(m.ngeom, bool)
    for g in range(m.ngeom):
        nm = (m.body(int(m.geom_bodyid[g])).name or "") + "|" + (m.geom(g).name or "")
        if nm.startswith(("wall_", "room_")) or "|wall_" in nm or "|room_" in nm:
            out[g] = True
    return out


def parse_res(s):
    w, h = s.lower().split("x")
    return int(w), int(h)


def run_stereo(texs, ress, n, seed, every_pick):
    from world_model.v2.house.teleop.sim import TeleopSim
    from world_model.v2.house.stereo.cfg import StereoCfg
    from world_model.v2.house.stereo import twin as TW
    from world_model.v2.house.stereo.sensor import StereoSensor
    snap = np.load(SNAP / f"carry_s{seed}.npz", allow_pickle=True)
    sim = TeleopSim("h5", grasped=True, seed=seed, with_people=True, people_clock="time")
    idx = np.linspace(20, len(snap["t"]) - 20, n).astype(int)
    od = OUT / "stereo"
    od.mkdir(parents=True, exist_ok=True)
    for tex in texs:
        for (W, H) in ress:
            t0 = time.time()
            cfg = StereoCfg(W=W, H=H, tex=tex, gt=True, keep_pair=True, self_mask=True, seed=seed)
            m = TW.build_twin_model(sim, cfg)
            s = StereoSensor(m, cfg, backend=NullBackend())
            wg = wall_geoms(m)
            rec = {k: [] for k in ("left", "right", "gt_depth", "gt_seg", "wall", "cls", "origin", "R")}
            for i in idx:
                s.load(snap["qpos"][i], snap["mpos"][i], snap["mquat"][i], snap["eq"][i])
                fr = s.capture(int(snap["k"][i]), float(snap["t"][i]))
                L, R = fr["pair"]
                rec["left"].append(L)
                rec["right"].append(R)
                rec["gt_depth"].append(fr["gt_depth"].astype(np.float32))
                gs = fr["gt_seg"].astype(np.int32)
                rec["gt_seg"].append(gs)
                rec["wall"].append((gs >= 0) & wg[np.clip(gs, 0, None)])
                rec["cls"].append(fr["cls"])
                rec["origin"].append(fr["origin"])
                rec["R"].append(fr["R"])
            A = {k: np.asarray(v) for k, v in rec.items()}
            np.savez(od / f"{tex}_{W}x{H}.npz", f=cfg.f, B=cfg.baseline, idx=idx, **A)
            print(f"[capture] {tex} {W}x{H}: {len(idx)} frames, {time.time() - t0:.0f}s", flush=True)
            s.close()


def run_pose(res, seed, every, max_frames):
    from world_model.v2.house.teleop.sim import TeleopSim
    from world_model.v2.house.stereo.cfg import StereoCfg
    from world_model.v2.house.stereo import twin as TW
    from world_model.v2.house.stereo.sensor import StereoSensor
    W, H = res
    snap = np.load(SNAP / f"carry_s{seed}.npz", allow_pickle=True)
    sim = TeleopSim("h5", grasped=True, seed=seed, with_people=True, people_clock="time")
    cfg = StereoCfg(W=W, H=H, tex="speckle", gt=True, keep_rgb=True, keep_pair=True, self_mask=True, seed=seed)
    m = TW.build_twin_model(sim, cfg)
    s = StereoSensor(m, cfg, backend=NullBackend())
    idx = list(range(0, len(snap["t"]), every))[:max_frames]
    rec = {k: [] for k in ("left", "right", "gt_depth", "gt_seg", "cls", "origin", "R")}
    t0 = time.time()
    for j, i in enumerate(idx):
        s.load(snap["qpos"][i], snap["mpos"][i], snap["mquat"][i], snap["eq"][i])
        fr = s.capture(int(snap["k"][i]), float(snap["t"][i]))
        L, R = fr["pair"]
        rec["left"].append(L)
        rec["right"].append(R)
        rec["gt_depth"].append(fr["gt_depth"].astype(np.float32))
        rec["gt_seg"].append(fr["gt_seg"].astype(np.int16))
        rec["cls"].append(fr["cls"])
        rec["origin"].append(fr["origin"])
        rec["R"].append(fr["R"])
        if j % 50 == 0:
            print(f"[capture pose] {j}/{len(idx)} {time.time() - t0:.0f}s", flush=True)
    A = {k: np.asarray(v) for k, v in rec.items()}
    # geom ids of the tray + cup (so the harness can build truth masks: EVALUATION / privileged first-frame prompt only)
    tray_g = np.flatnonzero(m.geom_bodyid == m.body("tray").id)
    cup_g = np.flatnonzero(m.geom_bodyid == m.body("cup").id)
    od = OUT / "pose"
    od.mkdir(parents=True, exist_ok=True)
    np.savez(od / f"seq_s{seed}_{W}x{H}.npz", f=cfg.f, B=cfg.baseline, idx=np.array(idx), t=snap["t"][idx], tray_p=snap["tray_p"][idx], tray_R=snap["tray_R"][idx],
             cup_p=snap["cup_p"][idx], cup_R=snap["cup_R"][idx], tray_geoms=tray_g, cup_geoms=cup_g, **A)
    print(f"[capture pose] {len(idx)} frames {W}x{H} {time.time() - t0:.0f}s", flush=True)
    s.close()


def _rot(yaw, pitch, roll):
    cz, sz, cy, sy, cx, sx = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return Rz @ Ry @ Rx


def _quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def run_scripted(res, seed, n, hz, snap_i):
    """hard tracking test: robot (camera) frozen at snapshot `snap_i`; tray + cup + beads follow a SCRIPTED rigid motion about the tray centre -- a chirp in yaw / pitch / roll
    (amplitudes 40 / 15 / 15 deg, frequency 0.15 -> ~1 Hz over the run, i.e. up to ~240 deg/s) + a +-4 cm translation.  Truth = the commanded poses."""
    from world_model.v2.house.teleop.sim import TeleopSim
    from world_model.v2.house.stereo.cfg import StereoCfg
    from world_model.v2.house.stereo import twin as TW
    from world_model.v2.house.stereo.sensor import StereoSensor
    W, H = res
    snap = np.load(SNAP / f"carry_s{seed}.npz", allow_pickle=True)
    sim = TeleopSim("h5", grasped=True, seed=seed, with_people=True, people_clock="time")
    cfg = StereoCfg(W=W, H=H, tex="speckle", gt=True, keep_rgb=True, keep_pair=True, self_mask=True, seed=seed)
    m = TW.build_twin_model(sim, cfg)
    s = StereoSensor(m, cfg, backend=NullBackend())
    balls = [m.body(i).name for i in range(m.nbody) if (m.body(i).name or "").startswith("ball_")]
    adr = lambda nm: int(m.jnt_qposadr[int(m.body_jntadr[m.body(nm).id])])
    q0 = snap["qpos"][snap_i].copy()
    a_t, a_c = adr("tray"), adr("cup")
    def pose_of(q, a):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q[a + 3:a + 7].copy())
        return q[a:a + 3].copy(), R.reshape(3, 3)
    tp0, tR0 = pose_of(q0, a_t)
    cp0, cR0 = pose_of(q0, a_c)
    bp0 = {b: q0[adr(b):adr(b) + 3].copy() for b in balls}
    rec = {k: [] for k in ("left", "right", "gt_depth", "gt_seg", "cls", "origin", "R")}
    TP, TRm, CP, CRm, tt = [], [], [], [], []
    t0 = time.time()
    for k in range(n):
        t = k / hz
        ph = 2 * np.pi * (0.15 * t + 0.04 * t * t)
        yaw, pitch, roll = np.radians(40) * np.sin(ph), np.radians(15) * np.sin(ph * 0.8 + 1.0), np.radians(15) * np.sin(ph * 1.1 + 2.0)
        dp = 0.04 * np.array([np.sin(ph * 0.7), np.cos(ph * 0.9), 0.6 * np.sin(ph * 0.5 + 0.5)])
        Rd = _rot(yaw, pitch, roll)
        q = q0.copy()
        def put(a, p0, R0):
            p = tp0 + dp + Rd @ (p0 - tp0)
            R = Rd @ R0
            q[a:a + 3] = p
            q[a + 3:a + 7] = _quat(R)
            return p, R
        p_t, R_t = put(a_t, tp0, tR0)
        p_c, R_c = put(a_c, cp0, cR0)
        for b in balls:
            q[adr(b):adr(b) + 3] = tp0 + dp + Rd @ (bp0[b] - tp0)
        s.load(q, snap["mpos"][snap_i], snap["mquat"][snap_i], snap["eq"][snap_i])
        fr = s.capture(k, t)
        L, Rr = fr["pair"]
        rec["left"].append(L); rec["right"].append(Rr); rec["gt_depth"].append(fr["gt_depth"].astype(np.float32))
        rec["gt_seg"].append(fr["gt_seg"].astype(np.int16)); rec["cls"].append(fr["cls"]); rec["origin"].append(fr["origin"]); rec["R"].append(fr["R"])
        TP.append(p_t); TRm.append(R_t.ravel()); CP.append(p_c); CRm.append(R_c.ravel()); tt.append(t)
        if k % 50 == 0:
            print(f"[capture scripted] {k}/{n} {time.time() - t0:.0f}s", flush=True)
    A = {k: np.asarray(v) for k, v in rec.items()}
    tray_g = np.flatnonzero(m.geom_bodyid == m.body("tray").id)
    cup_g = np.flatnonzero(m.geom_bodyid == m.body("cup").id)
    od = OUT / "pose"
    od.mkdir(parents=True, exist_ok=True)
    np.savez(od / f"scripted_s{seed}_{W}x{H}.npz", f=cfg.f, B=cfg.baseline, idx=np.arange(n), t=np.array(tt), tray_p=np.array(TP), tray_R=np.array(TRm), cup_p=np.array(CP),
             cup_R=np.array(CRm), tray_geoms=tray_g, cup_geoms=cup_g, hz=hz, **A)
    print(f"[capture scripted] {n} frames {time.time() - t0:.0f}s", flush=True)
    s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["stereo", "pose", "scripted"])
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--snap-i", type=int, default=100)
    ap.add_argument("--tex", nargs="*", default=["speckle", "objects", "none"])
    ap.add_argument("--res", nargs="*", default=["320x180", "640x360", "1280x720"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=10 ** 9)
    a = ap.parse_args()
    if a.what == "stereo":
        run_stereo(a.tex, [parse_res(r) for r in a.res], a.n, a.seed, None)
    elif a.what == "scripted":
        run_scripted(parse_res(a.res[0]), a.seed, a.n, a.hz, a.snap_i)
    else:
        run_pose(parse_res(a.res[0]), a.seed, a.every, a.max_frames)
    os._exit(0)


if __name__ == "__main__":
    main()
