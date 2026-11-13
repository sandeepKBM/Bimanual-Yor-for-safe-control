"""grasp/smoke.py -- tray pick at the counter with the wrist-camera-guided FRICTION grasp, lift, ~5 s of base motion.  SIMULATION ONLY.

  cd <repo> ; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=$PWD
  PY=python
  $PY -m world_model.v2.house.grasp.smoke --seeds 0 1 2 --seg depth            # metrics only, CPU
  $PY -m world_model.v2.house.grasp.smoke --seeds 0 --seg depth --video data/molmospaces/grasp/pick.mp4   # needs MUJOCO_GL=egl + GPU

Per seed: the tray (with cup + 8 beads) is placed on the counter at a seeded random offset (+-2 cm, +-3 deg) from where the hands expect it
(the operator's alignment error); Cmd.grasp starts the assisted grasp; after HOLD the lift rises 0.10 m and the base executes a scripted
~5 s manoeuvre (reverse, then a curve) with the twist target ramped at <= 1.0 m/s^2 (inside the cup-spill admissible acceleration; no
shield is used here).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from world_model.v2.house.teleop.sim import TeleopSim, Cmd, TICK
from world_model.v2.house.teleop import scene as SC
from . import util as U


def drive_profile(t, v_max=0.5, w_max=0.45, a=1.0):
    """scripted base twist target at time t [s] since the start of the manoeuvre: reverse straight, then a reversing curve, then stop.
    ramps are limited to `a` m/s^2 (and a/2 rad/s^2)."""
    ramp = lambda x, vmax, acc: float(np.clip(x * acc, 0.0, vmax))
    if t < 2.0:
        return np.array([-ramp(t, v_max, a), 0.0, 0.0])
    if t < 4.0:
        tt = t - 2.0
        return np.array([-v_max, 0.0, ramp(tt, w_max, a * 0.5)])
    tt = t - 4.0
    v = max(0.0, v_max - tt * a)
    w = max(0.0, w_max - tt * a * 0.5)
    return np.array([-v, 0.0, w])


def run_episode(seed=0, seg="depth", log_both=False, drive_s=5.0, offset_m=0.02, offset_deg=3.0, max_grasp_s=40.0, video=None,
                verbose=True, scene="h5", cfg_kw=None, shift_local=None, grasp_only=False, place=False, stand_dz=0.04, lift_to=0.10, ring=None):
    sc = SC.get_scene(scene)
    t_wall = time.time()
    sim = TeleopSim(scene, counter=True, seed=seed, spawn=sc.waypoints["cafe_counter_pick"], grasp_mode="friction", with_people=False)
    a = sim.assist
    a.cfg.seg, a.cfg.log_both, a.cfg.seed = seg, log_both, seed
    for k, v in (cfg_kw or {}).items():
        setattr(a.cfg, k, v)
    rng = np.random.default_rng(1000 + seed)
    if ring is not None:               # capture-range sweep: EXACT magnitudes -- displacement `offset_m` along tray direction ring[0] [deg], yaw ring[1] * offset_deg
        Rt0 = sim.d.xmat[sim.tray].reshape(3, 3)
        dloc = offset_m * np.array([math.cos(math.radians(ring[0])), math.sin(math.radians(ring[0]))])
        U.perturb_tray(sim, Rt0[:2, :2] @ dloc, math.radians(ring[1] * offset_deg))
    else:
        U.perturb_tray(sim, rng.uniform(-offset_m, offset_m, 2), math.radians(rng.uniform(-offset_deg, offset_deg)))
    if shift_local is not None:       # DEBUG bad grasp: move tray+cup+beads by (dx, dy) m along the TRAY x / y axes (x = the approach axis of both hands)
        Rt = sim.d.xmat[sim.tray].reshape(3, 3)
        U.perturb_tray(sim, Rt[:2, :2] @ np.asarray(shift_local[:2], float), 0.0)
    rec = _Recorder(sim, video) if video else None
    out = dict(seed=seed, seg=seg, proprio=bool(a.cfg.proprio), shift_local=shift_local, disable_est=bool(a.cfg.disable_est))

    def step(cmd):
        o = sim.tick(cmd)
        if rec:
            rec.frame(o)
        return o

    for _ in range(25):
        step(Cmd())
    t_trig = sim.t
    n = int(max_grasp_s / TICK)
    for i in range(n):
        step(Cmd(grasp=(i == 0)))
        if a.state in ("HOLD", "FAILED"):
            break
        if verbose and i % 100 == 0:
            print(f"  [seed {seed} {seg}] t={sim.t:5.1f} {a.state}", flush=True)
    out["grasp_state"] = a.state
    out["grasp_wall_s"] = time.time() - t_wall
    if a.state == "HOLD" or grasp_only:
        mt0 = a.metrics()
        out["grasp_verified"] = a.state == "HOLD"
        out["tug_rise_true_mm"] = mt0.get("tug_rise_true_mm")         # TRUE tray rise during the tug (eval): did the tray really come up with the hands?
        out["truly_held"] = bool(a.state == "HOLD" and (mt0.get("tug_rise_true_mm") or 0.0) > 0.5 * a.cfg.tug * 1e3)
        out["payload_est_kg"] = mt0.get("payload_est_kg")
        out["payload_err_frac"] = mt0.get("payload_err_frac")
        out["verify_by"] = mt0.get("verify_by", "pad_contacts")
        out["grip_effort_N"] = mt0.get("grip_effort_N")
        out["grip_gap_mm"] = {k: v * 1e3 for k, v in (mt0.get("grip_gap_m") or {}).items()} or None
        out["fail_log"] = mt0.get("fail_log")
    if grasp_only:
        out.update(success=a.state == "HOLD", metrics=_clean(a.metrics()), fail_reason=a.fail_reason, attempts=a.metrics()["attempts"])
        return out
    if a.state != "HOLD":
        out.update(success=False, metrics=_clean(a.metrics()), fail_reason=a.fail_reason)
        if rec:
            rec.close()
        return out
    for _ in range(25):
        step(Cmd())
    # lift 0.10 m
    nb0 = int(sim.bead_out.sum())
    for i in range(int(4.0 / TICK)):
        o = step(Cmd(lift_v=0.12 if sim.lift_actual() < lift_to else 0.0))
        if sim.lift_actual() >= lift_to - 0.001:
            break
    for _ in range(int(0.5 / TICK)):
        step(Cmd())
    out["lift_m"] = float(sim.lift_actual())
    if place:
        return _place_phase(sim, a, step, rec, out, stand_dz)
    rel0 = {s: a._hand_rel(s) for s in ("left", "right")}
    zlog, tilt_log, cup_log = [], [], []
    # base manoeuvre
    t0 = sim.t
    cup_min = []
    for i in range(int(drive_s / TICK)):
        o = step(Cmd(twist=drive_profile(sim.t - t0)))
        tilt_log.append(o["tray_tilt"])
        cup_log.append(o["cup_tilt"])
    for _ in range(int(1.0 / TICK)):
        o = step(Cmd())
    if rec:
        rec.close()
    mt = a.metrics()
    tray_rise = float(sim.d.xpos[sim.tray][2])
    rel1 = {s: a._hand_rel(s) for s in ("left", "right")}
    cont = {s: a.pad_contacts(s) for s in ("left", "right")}
    sl = np.array(a.slip_log)
    out.update(
        success=bool(mt["success"]) and sim.cup_lost_t is None,
        grasp_success=bool(mt["success"]), attempts=mt["attempts"],
        time_to_grasp_s=float(mt["t_done"] - mt["t_trigger"]) if mt["t_done"] is not None else None,
        pad_contacts_end={s: dict(upper=cont[s][0], lower=cont[s][1]) for s in cont},
        grip_force_end_N={s: cont[s][2] for s in cont},
        slip_carry_max_mm=[float(sl[:, 1].max()), float(sl[:, 2].max())] if len(sl) else None,
        contact_slip_carry_max_mm=[float(sl[:, 5].max()), float(sl[:, 6].max())] if len(sl) else None,
        slip_carry_end_mm={s: float(np.linalg.norm(rel1[s][0] - rel0[s][0]) * 1e3) for s in rel1},
        slip_cam_tug_mm=mt.get("slip_cam_mm"), tug_slip_true_mm=mt.get("tug_slip_true_mm"), cam_check=mt.get("cam_check"),
        beads_spilled=int(sim.bead_out.sum()), cup_lost=sim.cup_lost_t is not None,
        max_tray_tilt_deg=float(np.max(tilt_log)), max_cup_tilt_deg=float(np.max(cup_log)),
        tray_z_end=tray_rise, infeasible_fl_steps=int(sim.ctl.n_infeasible), wall_s=time.time() - t_wall,
        est=_est_summary(mt["est_log"], log_both), metrics=_clean(mt))
    out["tray_fk_err_mm"] = mt.get("tray_fk_err_mm")
    return out


def _place_phase(sim, a, step, rec, out, stand_dz):
    """PLACE: the placement surface (the pick counter slab) is raised by `stand_dz` (a stand 4 cm higher than the pick shelf, so the touchdown does not
    coincide with the lift axis' own lower limit), the held tray is lowered onto it with the lift axis until the joint-torque touchdown detector fires, the
    fingers open, the hands retreat, and the tray's rest is checked (tilt, speed, beads, still supported)."""
    import mujoco
    m = sim.m
    for nm in ("tp_counter", "tp_desk"):
        try:
            b = m.body(nm)
        except Exception:
            continue
        m.body_pos[b.id][2] += stand_dz
    mujoco.mj_forward(m, sim.d)
    nb0 = int(sim.bead_out.sum())
    for i in range(int(1.0 / TICK)):
        step(Cmd())
    a.release()                                             # Cmd.release == a.release(): HOLD -> PLACE_LOWER
    t0 = sim.t
    for i in range(int(25.0 / TICK)):
        step(Cmd())
        if a.state == "IDLE" and "place" in a.metrics_ and a.metrics_["place"].get("ok") is not None:
            break
    for _ in range(int(0.5 / TICK)):
        step(Cmd())
    if rec:
        rec.close()
    pl = dict(a.metrics_.get("place", {}))
    pl.pop("m_hist", None)
    out.update(place=pl, place_ok=bool(pl.get("ok")), beads_spilled=int(sim.bead_out.sum()) - nb0, success=bool(pl.get("ok")) and sim.cup_lost_t is None,
               grasp_success=True, time_to_grasp_s=float(a.metrics_["t_done"] - a.metrics_["t_trigger"]), metrics=_clean(a.metrics()))
    out["place_detect_delay_s"] = None if pl.get("true_touch_t") is None or pl.get("detect_t") is None else float(pl["detect_t"] - pl["true_touch_t"])
    return out


def _clean(mt):
    mt = dict(mt)
    mt.pop("est_log", None)
    return mt


def _est_summary(rows, both):
    """estimate error stats over PRE-state, confident (>= 0.55) estimates, per segmenter: full 3D position error, the grasp-relevant
    components (tray-frame x = edge distance, z = height), rotation error."""
    res = {}
    for key, pre in (("pri", "err_"), ):
        pass
    def collect(prefix, side=None):
        xs = []
        for r in rows:
            if r["state"] not in ("PRE", "APPROACH"):
                continue
            if side and r["side"] != side:
                continue
            if prefix == "":
                if not r.get("ok") or r["conf"] < 0.55 or "err_xyz" not in r:
                    continue
                xs.append((r["err_xyz"], r["err_deg"]))
            else:
                if f"{prefix}err_xyz" not in r or r.get(f"{prefix}conf", 0) < 0.55:
                    continue
                xs.append((r[f"{prefix}err_xyz"], r[f"{prefix}err_deg"]))
        return xs
    out = {}
    for name, prefix in (("driving", ""), ("priv", "priv_"), ("depth", "depth_")):
        for side in ("left", "right"):
            xs = collect(prefix, side)
            if not xs:
                continue
            e = np.array([x[0] for x in xs])
            out[f"{name}_{side}"] = dict(n=len(xs), pos3d_mm=float(np.linalg.norm(e, axis=1).mean()), x_mm=float(np.abs(e[:, 0]).mean()),
                                         y_mm=float(np.abs(e[:, 1]).mean()), z_mm=float(np.abs(e[:, 2]).mean()),
                                         rot_deg=float(np.mean([x[1] for x in xs])))
    return out


class _Recorder:
    """3-panel video: world view of the hands + tray, left and right WRIST CAMERA RGB insets; state text overlay.  Needs MUJOCO_GL=egl."""

    def __init__(self, sim, path, every=2, size=(480, 360), inset=(224, 168)):
        import mujoco
        self.mujoco = mujoco
        self.sim, self.path, self.every, self.k = sim, path, every, 0
        self.frames = []
        self.size, self.inset = size, inset
        self.r_main = mujoco.Renderer(sim.m, height=size[1], width=size[0])
        self.r_in = mujoco.Renderer(sim.m, height=inset[1], width=inset[0])
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 1.1, -28, 0
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def frame(self, o):
        self.k += 1
        if self.k % self.every:
            return
        mj, sim = self.mujoco, self.sim
        from PIL import Image, ImageDraw
        d = sim.d
        tray = d.xpos[sim.tray]
        self.cam.lookat[:] = tray + np.array([0.0, 0.0, 0.05])
        # azimuth: look at the tray from the robot side (robot yaw + 180 deg)
        yaw = sim.base.yaw
        self.cam.azimuth = math.degrees(yaw) + 90 + 180 - 40
        self.r_main.update_scene(d, camera=self.cam)
        main = self.r_main.render().copy()
        panels = []
        for s in ("left", "right"):
            self.r_in.update_scene(d, camera=f"{s}_arm_egocentric")
            panels.append(self.r_in.render().copy())
        img = Image.new("RGB", (self.size[0] + self.inset[0], self.size[1]))
        img.paste(Image.fromarray(main), (0, 0))
        img.paste(Image.fromarray(panels[0]), (self.size[0], 0))
        img.paste(Image.fromarray(panels[1]), (self.size[0], self.inset[1]))
        dr = ImageDraw.Draw(img)
        a = sim.assist
        dr.text((6, 4), f"t={sim.t:5.1f}s  grasp: {a.state}  attempt {a.attempt}  beads out {int(sim.bead_out.sum())}", fill=(255, 255, 0))
        dr.text((self.size[0] + 4, 2), "left wrist cam", fill=(255, 255, 255))
        dr.text((self.size[0] + 4, self.inset[1] + 2), "right wrist cam", fill=(255, 255, 255))
        if a.state in ("CLOSE", "VERIFY", "HOLD"):
            n = {s: a.pad_contacts(s) for s in ("left", "right")}
            dr.text((6, self.size[1] - 16), f"pad contacts L {n['left'][0]}+{n['left'][1]}  R {n['right'][0]}+{n['right'][1]}   grip N {n['left'][2]:.0f}/{n['right'][2]:.0f}",
                    fill=(0, 255, 255))
        self.frames.append(np.asarray(img))

    def close(self):
        import imageio.v2 as imageio
        w = imageio.get_writer(self.path, fps=25, codec="libx264", quality=7, macro_block_size=None)
        for f in self.frames:
            w.append_data(f)
        w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seg", default="depth", choices=["depth", "priv"])
    ap.add_argument("--log-both", action="store_true", help="also log the other segmenter's error every cycle (paired comparison)")
    ap.add_argument("--drive-s", type=float, default=5.0)
    ap.add_argument("--video", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--proprio", action="store_true", help="PROPRIO-ONLY assist (same as BIMANUAL_PROPRIO_ONLY=1): gripper gap/effort verify, estimated payload, hand-FK tray")
    ap.add_argument("--disable-est", action="store_true", help="DEBUG: ignore the wrist-camera estimate (grasp where the hands expect the tray)")
    ap.add_argument("--shift-local", type=float, nargs=2, default=None, help="DEBUG bad grasp: shift the tray by (dx dy) m along its own x / y axes")
    ap.add_argument("--offset-m", type=float, default=0.02, help="random tray placement error (+-m) from where the hands expect it")
    ap.add_argument("--offset-deg", type=float, default=3.0)
    ap.add_argument("--grasp-only", action="store_true", help="stop after the grasp state machine (no lift / drive)")
    ap.add_argument("--place", action="store_true", help="after the grasp: lift, raise the shelf 4 cm, lower onto it with torque-detected touchdown, open, retreat, check the rest")
    ap.add_argument("--stand-dz", type=float, default=0.04)
    ap.add_argument("--lift-to", type=float, default=0.10, help="carry lift height before the place [m]; < 0.03 -> the descent is done with the arms (tray-frame jog)")
    args = ap.parse_args()
    kw = {}
    if args.proprio:
        kw["proprio"] = True
    if args.disable_est:
        kw["disable_est"] = True
    res = []
    for sd in args.seeds:
        r = run_episode(sd, seg=args.seg, log_both=args.log_both, drive_s=args.drive_s, video=args.video if sd == args.seeds[0] else None,
                        cfg_kw=kw, shift_local=args.shift_local, grasp_only=args.grasp_only,
                        offset_m=args.offset_m, offset_deg=args.offset_deg, place=args.place, stand_dz=args.stand_dz, lift_to=args.lift_to)
        res.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "metrics"}, default=float), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(res, open(args.out, "w"), default=float, indent=1)


if __name__ == "__main__":
    main()
