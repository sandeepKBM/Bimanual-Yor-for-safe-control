"""teleop/run.py -- one coffee-run episode, RAW vs SHIELDED with the same sloppy operator (seed).  SIMULATION ONLY.

    python -m world_model.v2.house.teleop.run --mode both --task carry --seed 0 --max-s 30 --out /data/local/ss5772/molmospaces/teleop
    python -m world_model.v2.house.teleop.run --mode shield --task coffee --max-s 110 --video   # (needs MUJOCO_GL=egl + a free GPU != 0)

--stream closed (default): the operator is the closed-loop human model (same seed => same noise realisation, reaction delay and bump
times; the commands react to the robot state, as a real operator watching the camera would).  --stream open: the command stream is
pre-recorded on a ghost base (operator.rollout_stream) and replayed bit-identically in both runs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from . import base as BS
from . import operator as OP
from . import shield as SH
from .sim import TeleopSim, TICK

OUT = Path(os.environ.get("TELEOP_OUT", "/data/local/ss5772/molmospaces/teleop"))


class Recorder:
    """optional MP4 recorder (EGL).  Follow camera on the base; HUD with spill / clearance / intervention."""

    def __init__(self, sim, path, every=2, W=960, H=540):
        import mujoco
        self.mj = mujoco
        m = sim.m
        m.vis.global_.offwidth, m.vis.global_.offheight = W, H
        self.r = mujoco.Renderer(m, H, W)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.frames, self.every, self.path, self.W, self.H = [], every, path, W, H

    def frame(self, sim, k, info, label):
        if k % self.every:
            return
        from PIL import Image, ImageDraw, ImageFont
        d = sim.d
        yaw = sim.base.yaw
        self.cam.lookat[:] = (sim.base.x + 0.3 * math.cos(yaw), sim.base.y + 0.3 * math.sin(yaw), 0.8)
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 2.6, -30.0, math.degrees(yaw) - 150.0     # follow camera (behind-left of the robot)
        self.r.update_scene(d, camera=self.cam)
        img = Image.fromarray(self.r.render())
        dr = ImageDraw.Draw(img, "RGBA")
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
        dr.rectangle([0, 0, self.W, 30], fill=(0, 0, 0, 160))
        dr.text((8, 5), f"{label}   t={sim.t:5.1f}s  beads out {int(sim.bead_out.sum())}/{sim.rig.n_beads}  wall clr {info['wall']*100:5.1f} cm  "
                f"person {info['pdist']*100:5.0f} cm  shield: {info.get('label', '')}", font=f, fill=(255, 255, 255))
        self.frames.append(np.asarray(img.convert("RGB")))

    def close(self):
        import imageio.v2 as imageio
        w = imageio.get_writer(self.path, fps=1.0 / (TICK * self.every), codec="libx264", quality=8, macro_block_size=None)
        for f in self.frames:
            w.append_data(f)
        w.close()


def run_episode(shielded, seed=0, task="carry", scene="h5", max_s=30.0, stream="closed", video=None, cfg=None, op_params=None,
                tail_s=1.0, log_every=0, with_people=True, a_hw=None, verbose=True, people_source="truth", people_kw=None, loc_kw=None, filt=None):
    carry = task == "carry"
    sim = TeleopSim(scene, grasped=carry, seed=seed, with_people=with_people, counter=not carry, desk=not carry, a_hw=a_hw, people_source=people_source,
                     **(loc_kw or {}))        # loc_kw: pose_source/loc_cfg/loc_seed/occ_source/occ_map (house/loc); default = truth
    sc = sim.scene
    wp = sc.waypoints
    if stream == "open":
        st = OP.rollout_stream(seed, sc, max_s=max_s, params=op_params)
        op = OP.ReplayOperator(st)
    else:
        op = OP.SloppyOperator(seed, task, sc.out_route, pick_pose=wp["cafe_counter_pick"], place_pose=wp["lab_desk_place"],
                               params=op_params, back_route=sc.back_route, back_mode=sc.back_mode)
    shield = SH.Shield(cfg, occ=sim.occ_ctrl, a_hw=sim.a_hw) if shielded else SH.Passthrough()
    if shielded and people_source != "truth":            # head-camera tracker as the people source (people_track); sim.people_state stays the metric truth
        shield = sim.make_shield(cfg, **(people_kw or {}))
    if filt is not None:                                  # custom filter object (e.g. gatekeeper.Gatekeeper) replaces the Shield/Passthrough; default None = unchanged
        shield = filt
    if shielded:
        shield.docks = [wp["cafe_counter_pick"][:2], wp["lab_desk_place"][:2]] if not carry else [wp["cafe_counter_pick"][:2]]
    rec = Recorder(sim, str(video)) if video else None
    label = "SHIELD ON" if shielded else "SHIELD OFF (raw)"
    rows = []
    ctl_ms = []
    done_t = None
    t_wall = time.time()
    n = int(max_s / TICK)
    for k in range(n):
        obs = sim.observe()
        cmd = op.step(obs)
        safe, info = shield.filter(sim, cmd)
        t0 = time.perf_counter()
        o = sim.tick(safe)
        ctl_ms.append((time.perf_counter() - t0) * 1000 / 4)
        o.update(info)
        v_now = sim.base.v - 0  # (pre-tick velocity is in obs)
        u_hw = np.clip(cmd.twist, obs["v"] - sim.a_hw * TICK, obs["v"] + sim.a_hw * TICK)   # what the raw run would execute
        o["interv_vs_raw"] = float(np.linalg.norm((u_hw - safe.twist) / BS.V_STICK))
        o["u_op"] = cmd.twist.copy(); o["u_safe"] = safe.twist.copy(); o["phase"] = op.phase
        rows.append(o)
        if rec:
            rec.frame(sim, k, o, label)
        if verbose and log_every and k % log_every == 0:
            print(f"[{label}] t={o['t']:5.1f} ph={op.phase} pose=({o['pose'][0]:.2f},{o['pose'][1]:.2f},{math.degrees(o['pose'][2]):.0f}) "
                  f"v={o['v'].round(2)} beads_out={o['beads_out']} wall={o['wall']:.2f} pd={o['pdist']:.2f} tilt={o['tray_tilt']:.2f} "
                  f"lab={o['label']}", flush=True)
        if op.done and done_t is None:
            done_t = o["t"]
        if done_t is not None and o["t"] - done_t >= tail_s:
            break
    if rec:
        rec.close()
    summ = summarize(sim, op, shield, rows, shielded, seed, task, done_t, ctl_ms, time.time() - t_wall)
    if sim.loc is not None or sim.occ_source != "true":        # house/loc runs: localization error + the (true-grid) clearance the control path could not see
        summ.update(pose_source=sim.pose_source, occ_source=sim.occ_source, loc_margin_m=float(shield.cfg.loc_margin) if shielded else None)
        if sim.loc is not None:
            summ.update({"loc_" + k: v for k, v in sim.loc.errors().items()})
    return summ, rows


def summarize(sim, op, shield, rows, shielded, seed, task, done_t, ctl_ms, wall_s):
    t = np.array([r["t"] for r in rows])
    wall = np.array([r["wall"] for r in rows]); pd = np.array([r["pdist"] for r in rows])
    iv = np.array([r["intervention_norm"] for r in rows]); ivr = np.array([r["intervention"] for r in rows])
    ivr2 = np.array([r["interv_vs_raw"] for r in rows])
    tray_tilt = np.array([r["tray_tilt"] for r in rows]); cup_tilt = np.array([r["cup_tilt"] for r in rows])
    acc = np.array([np.hypot(*r["tray_acc"][:2]) for r in rows])
    rel = np.array([r["rel"] for r in rows])
    off0 = np.hypot(rel[0, 0] - rel[0, 0], 0)
    slip = np.hypot(rel[:, 0] - rel[0, 0], rel[:, 1] - rel[0, 1])
    labels = {}
    for r in rows:
        if r["intervention_norm"] > 0.02:
            labels[r["label"]] = labels.get(r["label"], 0) + 1
    pose = rows[-1]["pose"]
    goal = sim.scene.waypoints["cafe_counter_pick"] if task == "carry" else sim.scene.waypoints["lab_desk_place"]
    n_out = int(sim.bead_out.sum())
    s = dict(
        mode="shielded" if shielded else "raw", task=task, seed=seed, scene=sim.scene.name, sim_s=float(t[-1]), wall_s=wall_s,
        task_done=op.done, task_time_s=done_t, final_phase=op.phase,
        spills_beads_out=n_out, spilled=bool(n_out > 0 or sim.cup_lost_t is not None), cup_lost_at_s=sim.cup_lost_t,
        bead_out_times_s=[float(x) for x in sim.bead_out_t[sim.bead_out]],
        max_tray_tilt_deg=float(tray_tilt.max()), max_cup_tilt_deg=float(cup_tilt.max()),
        max_cup_slip_on_tray_m=float(slip.max()),
        peak_tray_acc_ms2=float(acc.max()), p99_tray_acc_ms2=float(np.percentile(acc, 99)),
        min_wall_clearance_m=float(wall.min()), wall_collision_ticks=int((wall <= 0).sum()),
        min_person_distance_m=float(pd.min()) if np.isfinite(pd.min()) else None, person_contact_ticks=int((pd <= 0).sum()),
        person_within_0p5m_ticks=int((pd < 0.5).sum()),
        physical_house_contacts=int(sim.bump_contacts),
        intervention_fraction_gt2pct=float((iv > 0.02).mean()), intervention_mean_norm=float(iv.mean()),
        intervention_mean_when_active_norm=float(iv[iv > 0.02].mean()) if (iv > 0.02).any() else 0.0,
        intervention_max_norm=float(iv.max()),
        intervention_vs_raw_exec_mean_norm=float(ivr2.mean()), intervention_vs_raw_exec_fraction_gt5pct=float((ivr2 > 0.05).mean()),
        intervention_vs_raw_exec_max_norm=float(ivr2.max()), intervention_mean_ms=float(ivr.mean()),
        intervention_labels_ticks=labels,
        shield_qp_fallbacks=int(shield.n_fallback), shield_ir_vetoes=int(shield.n_veto),
        flmpc_infeasible_steps=int(sim.ctl.n_infeasible),
        flmpc_qp=sim.ctl.qp_stats(),
        final_pose=[float(x) for x in pose], final_dist_to_goal_m=float(math.hypot(pose[0] - goal[0], pose[1] - goal[1])),
        ctl_ms_per_physics_step_median=float(np.median(ctl_ms)),
        peak_base_speed=float(max(np.hypot(*r["v"][:2]) for r in rows)),
        peak_yaw_rate=float(max(abs(r["v"][2]) for r in rows)),
        peak_base_accel_ms2=float(max(np.hypot(*r["acc_body"][:2]) for r in rows)),
        lift_max_m=float(max(r["lift_h"] for r in rows)),
    )
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["raw", "shield", "both"], default="both")
    ap.add_argument("--task", choices=["carry", "coffee"], default="carry")
    ap.add_argument("--scene", default="h5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-s", type=float, default=30.0)
    ap.add_argument("--stream", choices=["closed", "open"], default="closed")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--no-people", action="store_true")
    ap.add_argument("--people", choices=["truth", "tracker"], default="truth", help="people source of the SHIELD (tracker = head-camera KF tracker; truth stays the metric)")
    ap.add_argument("--people-look", action="store_true", help="tracker source: look-before-move glance")
    ap.add_argument("--people-conformal", default=None, help="tracker source: conformal margin json (people_track.calib fit)")
    ap.add_argument("--people-perception", default="depth", help="tracker source: detector condition (depth | seg | param:SIG:LAT:DROP)")
    ap.add_argument("--pose", choices=["truth", "est"], default="truth", help="base pose read by the control path (shield wall keep-out): truth | est (house/loc odometry+VIO)")
    ap.add_argument("--occ", choices=["true", "perceived"], default="true", help="static-obstacle grid read by the shield: true house grid | perceived (head-ZED mapping pass)")
    ap.add_argument("--occ-map", default=None, help="perceived grid npz (loc.mapper); default: build/cache for the scene")
    ap.add_argument("--loc-seed", type=int, default=None, help="estimator seed (default = --seed)")
    ap.add_argument("--loc-scale", type=float, default=1.0, help="multiply the odometry/VIO error parameters (pessimism sweep)")
    ap.add_argument("--loc-margin", type=float, default=0.0, help="shield wall buffer added for localization/map error [m]")
    ap.add_argument("--loc-margin-k", type=float, default=0.0, help="shield wall buffer: x estimator 1-sigma proxy")
    a = ap.parse_args()
    loc_kw = None
    cfg = None
    if a.pose != "truth" or a.occ != "true":
        from ..loc.estimator import LocCfg, scaled
        loc_kw = dict(pose_source=a.pose, occ_source=a.occ, occ_map=a.occ_map, loc_seed=a.loc_seed, loc_cfg=scaled(LocCfg(), a.loc_scale))
    if a.loc_margin or a.loc_margin_k:
        cfg = SH.ShieldCfg(loc_margin=a.loc_margin, loc_margin_k=a.loc_margin_k)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    modes = ["raw", "shield"] if a.mode == "both" else [a.mode]
    res = {}
    for mname in modes:
        vp = out / f"teleop_{a.task}_s{a.seed}_{mname}.mp4" if a.video else None
        summ, rows = run_episode(mname == "shield", a.seed, a.task, a.scene, a.max_s, a.stream, vp, log_every=a.log_every,
                                 with_people=not a.no_people, people_source=a.people, loc_kw=loc_kw, cfg=cfg,
                                 people_kw=dict(perception=a.people_perception, look=a.people_look, conformal=a.people_conformal, seed=a.seed))
        res[mname] = summ
        np.savez_compressed(out / f"teleop_{a.task}_s{a.seed}_{mname}_ts.npz",
                            t=[r["t"] for r in rows], pose=[r["pose"] for r in rows], u_op=[r["u_op"] for r in rows],
                            u_safe=[r["u_safe"] for r in rows], beads_out=[r["beads_out"] for r in rows], wall=[r["wall"] for r in rows],
                            pdist=[r["pdist"] for r in rows], tray_tilt=[r["tray_tilt"] for r in rows],
                            interv=[r["intervention_norm"] for r in rows], tray_acc=[r["tray_acc"] for r in rows],
                            rel=[r["rel"] for r in rows])
        (out / f"teleop_{a.task}_s{a.seed}_{mname}.json").write_text(json.dumps(summ, indent=1, default=str))
        print(json.dumps(summ, indent=1, default=str), flush=True)
    (out / f"metrics_{a.task}_s{a.seed}.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
