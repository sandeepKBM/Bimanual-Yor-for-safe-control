"""pick/smoke.py -- headless smoke / evaluation of the pick backend (SIMULATION ONLY).

    python -m world_model.v2.house.pick.smoke --objects pp_bottle,pp_crate --seeds 0 --planner vlm+veto --backend mock --out data/molmospaces/pick
Success (user spec) = the object is lifted >= 5 cm above its start AND stays >= 5 cm for >= 2 s of the HOLD, with ZERO robot<->furniture contacts (arm geoms vs house / table
geoms; props, the stowed tray and the target itself are not furniture).  Truth (the prop's body pose) is read ONLY here, for evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import mujoco

from world_model.v2.house.teleop import sim as S
from . import props as PP
from . import propose as PR
from . import session as SE

LABELS = {"pp_bottle": "bottle", "pp_mug": "mug", "pp_cube": "box", "pp_crate": "crate", "pp_basket": "basket"}   # MOCK labeller table (offline stand-in for a VLM)


class Video:
    def __init__(self, path, fps=15, size=(640, 360)):
        import cv2
        self.cv2 = cv2
        self.w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (size[0] * 2, size[1]))
        self.size = size
        self.path = path

    def add(self, ego, chase):
        cv2 = self.cv2
        a = cv2.resize(ego, self.size)
        b = cv2.resize(chase, self.size)
        self.w.write(np.concatenate([a, b], 1)[:, :, ::-1])

    def close(self):
        self.w.release()


class Chase:
    def __init__(self, sim, W=640, H=360):
        self.m = sim.m
        self.m.vis.global_.offwidth = max(W, int(self.m.vis.global_.offwidth))
        self.m.vis.global_.offheight = max(H, int(self.m.vis.global_.offheight))
        self.r = mujoco.Renderer(self.m, H, W)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def render(self, sim, target=None):
        """chase view: looks at `target` (robot-frame point, default the table centre) from the front-left of the robot, 0.9 m away."""
        root = sim.m.body("yor_root").id
        Rr = sim.d.xmat[root].reshape(3, 3)
        yaw = np.arctan2(Rr[1, 0], Rr[0, 0])
        c = sim.d.xpos[root] + Rr @ (np.array([0.5, -0.21, 0.85]) if target is None else np.asarray(target, float) + np.array([0.0, 0.0, 0.08]))
        self.cam.lookat[:] = c
        self.cam.distance = 1.4 if target is None else 0.95
        self.cam.azimuth = float(np.degrees(yaw)) + 150.0
        self.cam.elevation = -28.0
        self.r.update_scene(sim.d, camera=self.cam)
        return self.r.render().copy()


def truth_z(sim, name):
    return float(sim.d.xpos[sim.m.body(name).id][2])


_ARM = {}


def furniture_contacts(sim):
    """arm geoms (yor_root subtree) touching house / table geoms.  The stowed tray and the props are not furniture and not robot."""
    from .skill import _arm_geom_mask
    if id(sim.m) not in _ARM:
        _ARM[id(sim.m)] = _arm_geom_mask(sim)
    arm = _ARM[id(sim.m)]
    d = sim.d
    n = 0
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = c.geom1, c.geom2
        if (sim.house_geom[g1] and arm[g2]) or (sim.house_geom[g2] and arm[g1]):
            n += 1
    return n


def run_one(gt, planner="rule", backend="mock", seed=0, props="default", video=None, max_s=70.0, out_dir=None, log=print, sim=None, ps=None, forced_plan=None):
    t_build = time.time()
    sim = sim or S.TeleopSim("h5", with_people=False, task="pick", pick_props=props, seed=seed)
    t_build = time.time() - t_build
    for _ in range(25):
        sim.tick(S.Cmd())
    own_ps = ps is None
    ps = ps or SE.PickSession(planner=planner, backend=backend, mock_labels=LABELS, out_dir=out_dir, seed=seed)
    t0 = time.time()
    props_ = ps.propose(sim)
    t_prop = (time.time() - t0) * 1000
    # wait (briefly) for the mock/live VLM answers, then re-decide
    for _ in range(200):
        if all(ps.planner.status(ps.planner.key(p["id"], dict(extent=p["extent"], centroid=np.array(p["centroid"])))) in ("done", "rule") for p in props_):
            break
        time.sleep(0.05)
    ps.refresh_decisions()
    props_ = ps.proposals()
    tp = PP.truth(sim, gt)
    c_true = PR.to_robot_frame(tp["pos"][None], sim.pose())[0]
    cand = [(np.linalg.norm(np.array(p["centroid"][:2]) - c_true[:2]), p) for p in props_]
    res = dict(object=gt, seed=seed, planner=planner, backend=ps.planner.backend_name(), n_proposals=len(props_), proposal_ms=round(t_prop, 1), build_s=round(t_build, 1),
               true=dict(pos_robot=np.round(c_true, 3).tolist(), size=tp["size"].tolist(), mass=tp["mass"]))
    if not cand or min(cand, key=lambda t: t[0])[0] > 0.08:
        res.update(success=False, failure="NOT_PROPOSED", note="no proposal near the object (occluded / outside the view)")
        return res
    prop = min(cand, key=lambda t: t[0])[1]
    res["proposal"] = dict(id=prop["id"], label=prop["label"], extent=prop["extent"], centroid=prop["centroid"], mode=prop["mode"], arm=prop["arm"], planner=prop["planner"],
                           est_err_mm=dict(centroid_xy=round(1000 * float(np.linalg.norm(np.array(prop["centroid"][:2]) - c_true[:2])), 1)))
    z0 = truth_z(sim, gt)
    vid = ego = chase = None
    if video:
        vid = Video(video)
        ego = PR.EgoCam(sim, 640, 360)
        chase = Chase(sim)
    ps.request_with_snapshot(sim, prop["id"])
    t_req = time.time()
    n = 0
    fur = 0
    zs = []
    t_sim0 = sim.t
    step_ms = []
    while sim.t - t_sim0 < max_s:
        _t = time.perf_counter()
        ps.step(sim)
        step_ms.append((time.perf_counter() - _t) * 1000)
        sim.tick(S.Cmd())
        fur += furniture_contacts(sim)
        n += 1
        st = ps._status
        if ps._skill is not None and ps._skill.state in ("HOLD", "LIFT", "SQUEEZE", "CLOSE"):
            zs.append((sim.t, truth_z(sim, gt) - z0))
        if vid is not None and n % 3 == 0:
            snap = ego.snapshot(sim)
            fr = ego.render(snap, noise=False)
            pr = ps.proposals() if False else prop
            e = PR.draw_boxes(fr["rgb"], [dict(id=prop["id"], box2d=prop["box2d"], label=prop["label"], mode=prop["mode"], arm=prop["arm"])], selected=prop["id"])
            if ps._skill is not None:
                import cv2
                cv2.putText(e, f"{ps._skill.state} t={sim.t - t_sim0:.1f}s", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            vid.add(e, chase.render(sim, prop["centroid"]))
        if st in ("done", "failed") or (st == "idle" and n > 50 and ps._skill is None and ps._plan_fut is not None and ps._plan_fut.done() and ps.history):
            if st == "failed" or st == "done":
                break
    t_wall = time.time() - t_req
    sk = ps._skill
    hold_zs = [z for t, z in zs if sk is not None and sk.hold_t0 is not None and t >= sk.hold_t0]
    lifted_max = max([z for _, z in zs], default=0.0)
    hold_min = min(hold_zs, default=0.0)
    hold_dur = (sim.t - sk.hold_t0) if (sk is not None and sk.hold_t0 is not None) else 0.0
    ok = (st == "done") and hold_min >= 0.05 and hold_dur >= 2.0 and fur == 0
    sm = np.array(step_ms) if step_ms else np.zeros(1)
    res.update(step_ms=dict(p50=round(float(np.percentile(sm, 50)), 3), p99=round(float(np.percentile(sm, 99)), 3), max=round(float(sm.max()), 3), n=len(sm)), success=bool(ok), status=st, error=ps._error, lifted_max_cm=round(lifted_max * 100, 1), hold_min_lift_cm=round(hold_min * 100, 1), hold_s=round(hold_dur, 2),
               furniture_contacts=int(fur), skill_s=round(sim.t - t_sim0, 2), wall_s=round(t_wall, 1), plan_ms=ps._t.get("plan_ms"), infeasible_steps=int(sim.ctl.n_infeasible),
               timeline=[(s_, round(t - t_sim0, 2)) for s_, t in (sk.timeline if sk else [])], info=(sk.info if sk else {}), events=list(ps._events))
    if vid is not None:
        vid.close()
        ego.close()
        res["video"] = vid.path
    if own_ps:
        res["vlm_log"] = ps.vlm_log()
        ps.close()
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default="pp_bottle,pp_mug,pp_cube,pp_crate")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--planner", default="rule")
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--props", default="default")
    ap.add_argument("--out", default=SE.OUT_DIR)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--video-seeds", default="0", help="seeds that get a video when --video is set")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        for gt in a.objects.split(","):
            vp = os.path.join(a.out, f"pick_{a.tag}{gt}_s{seed}.mp4") if (a.video and str(seed) in a.video_seeds.split(",")) else None
            r = run_one(gt, a.planner, a.backend, seed, a.props, vp, out_dir=a.out)
            print(json.dumps({k: v for k, v in r.items() if k not in ("vlm_log", "timeline")}, default=str), flush=True)
            rows.append(r)
    with open(os.path.join(a.out, f"summary_{a.tag}{a.objects.replace(',', '+')}_{a.planner.replace('+', '_')}.json"), "w") as f:
        json.dump(rows, f, indent=1, default=str)
    for r in rows:
        r.pop("vlm_log", None)
    print("SUCCESS", sum(1 for r in rows if r["success"]), "/", len(rows))


if __name__ == "__main__":
    main()
