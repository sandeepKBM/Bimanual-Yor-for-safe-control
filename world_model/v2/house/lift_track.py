"""lift_track.py -- ball tracker and object segmentation sweeps over lift height x head mount x head pitch.  SIMULATION ONLY.
Used by lift_study.py (track / zono).  Head = the ASSUMED torso/mast head of perception_bridge._spec_head_new."""
from __future__ import annotations

import json
import time

import numpy as np
import mujoco

from world_model.v2.house import build as B
from world_model.v2.house import lift as L
from world_model.v2.house import perception_bridge as PB
from world_model.v2.house import perception_study as PS

OUT = PS.OUT
MOUNTS = ("lift", "mast")
PITCHES = (PB.HEAD_PITCH_CURRENT, 30.0, 45.0)
ARM_CATS = ("arm_L", "arm_R", "robot_base")
GRID_X = np.linspace(-0.07, -0.02, 6)      # feasible region of the hold pose (perception_study.FEAS_X)
GRID_Y = np.linspace(-0.06, 0.06, 5)


def occlusion_map(sc, base0, rig, mount, pitch, hit_min=8):
    """static ball on a 6 x 5 grid of the FEASIBLE part of the tray: fraction of positions the head sees (>= hit_min ball points), the fraction whose head
    line of sight is blocked by the robot's own geometry, and by the house."""
    cams, specs = PB.make_rig(sc, rig, mount, pitch)
    head = cams[0]
    vis_head = vis_any = blk_robot = blk_other = 0
    hits = []
    n = 0
    for gx in GRID_X:
        for gy in GRID_Y:
            sc.place_base(base0[0], base0[1], (gx, gy))
            tp, tR = sc.tray_pose()
            fr = [c.capture(sc, tp, tR, n, 7) for c in cams]
            hh = int((fr[0].geom == sc.gid_ball).sum())
            ha = sum(int((f.geom == sc.gid_ball).sum()) for f in fr)
            hits.append(hh)
            vis_head += hh >= hit_min
            vis_any += any(int((f.geom == sc.gid_ball).sum()) >= hit_min for f in fr)
            b, cat, _, _ = sc.los(head.pose()[0], sc.d.xpos[sc.body_ball])
            if b:
                blk_robot += cat in ARM_CATS + ("tray",)
                blk_other += cat not in ARM_CATS + ("tray",)
            n += 1
    sc.place_base(base0[0], base0[1])
    return dict(n=n, head_visible=vis_head / n, any_visible=vis_any / n, head_los_blocked_robot=blk_robot / n, head_los_blocked_house=blk_other / n,
                head_hits_median=float(np.median(hits)))


def cmd_track(a):
    h = a.lift
    C = L.new_hold_lift(PS.HOLD["house"], PS.HOLD["spawn"], PS.HOLD["yaw"], 0.005, lift_h=h)
    sc = PB.HouseScene(C)
    base0 = (np.array(C.info["spawn"]), C.info["yaw"])
    mujoco.mj_forward(sc.m, sc.d)
    sc.place_base(*base0)
    rows = []
    for mount in MOUNTS:
        for pitch in PITCHES:
            row = dict(lift=h, mount=mount, pitch=float(pitch))
            row["map"] = occlusion_map(sc, base0, "headN", mount, pitch)
            for rig in ("headN", "headN+wrist"):
                f = lambda t: sc.place_base(base0[0], base0[1], PS.ball_path(t))
                r, trk = PS.run_track(sc, rig, 200, f, 30.0, 3, seed=2, warm=25, eval_los=True, head_mount=mount, head_pitch=pitch)
                caps = max(r["n_caps"], 1)
                blk = r["los_blockers"].get("head", {})
                r["head_arm_blocked_frac"] = sum(v for k, v in blk.items() if k in ARM_CATS) / caps
                r["head_house_blocked_frac"] = r["los_house_blocked_frac"][0]
                r["head_cam_ms"] = r["cam_ms"][0]["mean"]
                for k in ("cam_ms", "los_blockers"):
                    r.pop(k, None)
                row[rig] = r
            rows.append(row)
            m_, r_, f_ = row["map"], row["headN"], row["headN+wrist"]
            print("lift %.3f %-4s pitch %.1f | map head vis %.2f any %.2f blocked robot %.2f house %.2f | head only valid %.2f lost %.2f err %.1f/%.1f(alive %.1f) armblk %.2f ms %.0f | fused valid %.2f lost %.2f err %.1f alive %.1f" % (
                h, mount, pitch, m_["head_visible"], m_["any_visible"], m_["head_los_blocked_robot"], m_["head_los_blocked_house"], r_["valid_frac"], r_["lost_frac"],
                r_["err_mm"]["mean"], r_["err_mm"]["p95"], r_["err_alive_mm"]["mean"], r_["head_arm_blocked_frac"], r_["head_cam_ms"], f_["valid_frac"], f_["lost_frac"],
                f_["err_mm"]["mean"], f_["err_alive_mm"]["mean"]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"lift_track_L{h:.3f}.json").write_text(json.dumps(rows, indent=1))


def cmd_zono(a):
    from world_model.v2.perception_zono.score import Truth
    house = a.house
    m0, d0 = B.load_house(house, True)
    mujoco.mj_forward(m0, d0)
    spot = B.find_clear_spots(house, m0, d0, top=1)[0]
    del m0, d0
    C = L.new_hold_lift(house, (spot["x"], spot["y"]), 0.0, 0.005, lift_h=0.0)
    sc = PB.HouseScene(C)
    clusters = PS.find_clusters(sc, a.want)
    print(f"house {house}: clusters {[[sc.md[k]['category'] for k in c] for c in clusters]}", flush=True)
    rows = []
    t0 = time.time()
    XC = 0.85
    for ci, keys in enumerate(clusters):
        objs = [PB.MeshObj(sc, k, oid=i + 1) for i, k in enumerate(keys)]
        rng = np.random.default_rng(100 + ci)
        Ts = [Truth(o, rng, n_surf=3000, n_q=0, n_mc=60000) for o in objs]
        lo = np.min([o.aabb()[0] for o in objs], 0)
        hi = np.max([o.aabb()[1] for o in objs], 0)
        ctr = (lo + hi) / 2
        for yaw_deg in (0, 90, 180, 270):
            yaw = np.deg2rad(yaw_deg)
            base = ctr[:2] - B._rotz(yaw)[:2, :2] @ np.array([XC, -0.209550])
            for h in (0.0, 0.2, 0.416):
                L.set_lift(C, h)
                sc.place_base(base, yaw)
                # reject poses where the robot BASE (wheels / column) is in contact with the house; arm / tray contacts are recorded only
                cls = sc.cls
                n_base = n_arm = 0
                for k in range(sc.d.ncon):
                    c = sc.d.contact[k]
                    g1, g2 = c.geom1, c.geom2
                    if sc.is_house[g1] != sc.is_house[g2]:
                        o = g2 if sc.is_house[g1] else g1
                        if cls[o] == "B":
                            n_base += 1
                        else:
                            n_arm += 1
                if n_base > 0:
                    continue
                for mount in MOUNTS:
                    for pitch in PITCHES:
                        cams, _ = PB.make_rig(sc, "headN", mount, pitch, W=480, H=270)       # the rig study's 480 x 270 grid
                        seed = (house * 1000 + ci * 100 + yaw_deg // 90 * 10 + int(h * 10)) & 0x7FFFFFFF
                        with sc.hide("tray"):
                            rec = PS.zono_trial((sc, keys, objs, Ts, yaw, "headN", "nominal", seed, cams))
                        rec.update(house=house, cluster=ci, cats=[sc.md[k]["category"] for k in keys], lift=h, mount=mount, pitch=float(pitch),
                                   yaw_deg=yaw_deg, n_arm_contacts=n_arm)
                        rows.append(rec)
        print(f"house {house} cluster {ci} done, {len(rows)} trials ({time.time() - t0:.0f}s)", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"lift_zono_h{house}.json").write_text(json.dumps(rows))
