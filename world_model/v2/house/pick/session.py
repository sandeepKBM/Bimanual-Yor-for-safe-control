"""pick/session.py -- PickSession: the API the teleop UI drives.  SIMULATION ONLY.

    ps = PickSession(planner="rule" | "vlm" | "vlm+veto", backend="mock" | "live" | "auto")
    props = ps.propose(sim)         # [dict(id, box2d, label, extent, centroid, mode, arm, ...)]  (blocking=True: waits; blocking=False: returns the cache, refreshes in the worker)
    ps.request(id)                  # start the pick of proposal `id` (plans in the worker, then the skill runs inside step())
    ps.cancel()                     # release + retreat to READY (or drop a pending plan)
    ps.step(sim)                    # SIM THREAD, once per 50 Hz tick, BEFORE sim.tick(): O(1) bookkeeping + reference writes; never renders / plans / calls the VLM
    ps.state()                      # JSON-safe dict for the HUD

Threads: ONE worker thread does every heavy thing (EGL render of the head camera, instance masks -> boxes + depth 3D, mock/live VLM labelling, IK + path collision checks, reach
probes); the planner has its own single VLM thread (planner.GraspPlanner) so a slow VLM never delays the geometry.  step() only (a) snapshots the physics state when a refresh was
asked for, (b) instantiates / advances the PickSkill once a plan is ready.
Frames: boxes in image pixels (640x360 head camera); centroid / extent in the ROBOT frame (+x forward, +y left, z up; origin = base root); extent = (a, b, h) footprint OBB sizes
(a >= b) and height; `yaw` = direction of the `a` axis.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import mujoco

from . import assign as A
from . import planner as PL
from . import propose as PR
from . import skill as SK

OUT_DIR = os.environ.get("PICK_OUT", "data/molmospaces/pick")


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [round(float(v), 4) for v in o.ravel()]
    if isinstance(o, (np.floating, float)):
        return round(float(o), 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    if hasattr(o, "to_json"):
        return _jsonable(o.to_json())
    return o


class PickSession:
    def __init__(self, planner="rule", backend="mock", mock_labels=None, W=PR.W_DEF, H=PR.H_DEF, log=None, out_dir=None, vlm_timeout_s=10.0, seed=0, n_fuse=5, mock_policy="default"):
        self.log = log or (lambda *a: None)
        self.W, self.H, self.seed, self.n_fuse = W, H, seed, n_fuse
        self.out_dir = out_dir or OUT_DIR
        self.mock_labels = dict(mock_labels or {})           # MOCK only: {gt body name: label}; used by the mock labelling rule (documented privileged shortcut for offline tests)
        self._marks = {}
        self.backend_kind = backend
        rules = PL.mock_pick_rules(mock_policy) + [(r"Marks to name", self._mock_label_rule)]
        self.planner = PL.GraspPlanner(planner=planner, backend=backend, timeout_s=vlm_timeout_s, log=self.log, mock_rules=rules)
        self._ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pickwork")
        self._lock = threading.Lock()
        self._cam = None
        self._props = []                  # latest proposals (dicts, with private '_' keys)
        self._ids = {}                    # id -> last centroid (stable ids across refreshes)
        self._next_id = 1
        self._fut_prop = None
        self._snap_req = False
        self._snap = None
        self._sel = None                  # requested id
        self._plan_fut = None
        self._wp = None
        self._plan_info = None
        self._skill = None
        self._status = "idle"
        self._error = None
        self._t = dict(propose_ms=None, plan_ms=None, render_ms=None)
        self._events = []
        self._last_frame = None
        self.history = []                 # finished picks (dict)

    # ------------------------------------------------------------------ mock labeller (MockVLM 'Marks to name' rule)
    def _mock_label_rule(self, text, system, images, n):
        import re
        ids = [int(x) for x in re.findall(r"Marks to name: \[([0-9, ]*)\]", text) for x in x.split(",") if x.strip()]
        return "\n".join(f"{i} | {self._marks.get(i, 'object')} | 0.8" for i in ids)

    # ------------------------------------------------------------------ propose (worker does the work)
    def propose(self, sim, blocking=True, timeout=120.0):
        """snapshot the physics state (cheap, caller's thread) and refresh the proposals in the worker.  blocking=False -> returns the cached list immediately."""
        if self._fut_prop is None or self._fut_prop.done():
            m = sim.m
            d = mujoco.MjData(m)
            mujoco.mj_copyData(d, m, sim.d)
            snap = dict(d=d, base=sim.pose().copy(), lift=sim.lift_actual(), t=sim.t, sim_m=m, robot_geom=np.array(sim.robot_geom, bool), ctl=sim.ctl)
            self._snap = snap
            self._fut_prop = self._ex.submit(self._propose_job, snap)
        if blocking:
            self._fut_prop.result(timeout=timeout)
        return self.proposals()

    def proposals(self):
        with self._lock:
            return [_jsonable(p) for p in self._props]

    def _ensure_cam(self, snap):
        if self._cam is None:
            class _S:                       # the bits EgoCam reads off a sim
                pass
            s = _S()
            s.m, s.robot_geom = snap["sim_m"], snap["robot_geom"]
            self._cam = PR.EgoCam(s, self.W, self.H)
        return self._cam

    def _propose_job(self, snap):
        t0 = time.time()
        try:
            cam = self._ensure_cam(snap)
            mujoco.mj_copyData(cam.d, cam.m, snap["d"])
            sn = self._snapshot_pose(snap)
            t1 = time.time()
            frame = cam.render(sn, seed=self.seed)
            frame["inst"] = PR.instance_masks(cam, frame)                     # PRIVILEGED instance partition
            t2 = time.time()
            raw = PR.proposals_from_frame(frame)
            if self.n_fuse > 1:                                              # depth noise is ~5 mm at 0.6 m: median-fuse the per-frame estimates of n_fuse noisy frames
                allr = [raw]
                for k in range(1, self.n_fuse):
                    fk_ = cam.render(sn, seed=self.seed + 1000 * k)
                    fk_["inst"] = PR.instance_masks(cam, fk_)
                    allr.append(PR.proposals_from_frame(fk_))
                raw = PR.fuse_estimates(allr)
            t3 = time.time()
            props = self._finish(raw, frame, snap)
            with self._lock:
                self._props = props
                self._last_frame = frame
            self._t.update(render_ms=round((t2 - t1) * 1000, 1), geometry_ms=round((t3 - t2) * 1000, 1), propose_ms=round((time.time() - t0) * 1000, 1))
        except Exception as e:      # noqa: BLE001
            import traceback
            self._error = f"propose: {type(e).__name__}: {str(e)[:200]}"
            self.log(traceback.format_exc())
            raise

    def _snapshot_pose(self, snap):
        from world_model.v2.house.ocs import fp_render as FR
        o_b, R_b = FR.head_cam_base(snap["lift"])
        b = snap["base"]
        cy, sy = math.cos(b[2]), math.sin(b[2])
        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        return Rz @ o_b + np.array([b[0], b[1], 0.0]), Rz @ R_b, b.copy(), float(snap["lift"])

    def _finish(self, raw, frame, snap):
        """stable ids, labels (lang backend), rule plan, reach probes, VLM requests, decisions."""
        # stable ids by nearest previous centroid (non-privileged association)
        used, props = set(), []
        for r in raw:
            best, bd = None, 0.06
            for pid, c in self._ids.items():
                dd = float(np.linalg.norm(np.asarray(c[:2]) - r["centroid"][:2]))
                if dd < bd and pid not in used:
                    best, bd = pid, dd
            if best is None:
                best = self._next_id
                self._next_id += 1
            used.add(best)
            self._ids[best] = r["centroid"].copy()
            r["id"] = best
            props.append(r)
        props.sort(key=lambda r: r["box2d"][0])
        # labels through the lang backend (MOCK unless backend='live'/'auto' was asked for explicitly)
        from world_model.v2.house.lang import grounding as GR
        for k, r in enumerate(props, 1):
            self._marks[k] = self.mock_labels.get(r["_gt"], "object")
        gprops = []
        for r in props:
            half = np.array(r["extent"][:2] + (r["extent"][2],)) / 2
            G = np.diag(half)
            gprops.append(GR.Proposal(r["id"], r["centroid"], G, r["centroid"], G, tuple(float(v) for v in r["box2d"]), r["n_pts"], 0.0, "obj"))
        try:
            g = GR.ground(self.planner.backend(), frame["rgb"], gprops, out_dir=None, tag="pick")
            for r in props:
                r["label"] = g.names[r["id"]][0]
                r["label_conf"] = round(float(g.names[r["id"]][1]), 2)
        except Exception as e:      # noqa: BLE001
            for r in props:
                r["label"], r["label_conf"] = "object", 0.0
            self._events.append(f"labelling failed ({type(e).__name__}): labels = 'object'")
        # reach probes + VLM requests
        scr = SK.Scratch(types_sim(snap))
        for r in props:
            est = dict(extent=r["extent"], centroid=r["centroid"], yaw=r["yaw"])
            rule = A.assign(r["extent"], r["centroid"], r["yaw"])
            reach = {}
            if rule.mode == "single":
                for arm in SK.SIDES:
                    reach[arm] = SK.check_reach(scr, dict(extent=r["extent"], centroid=r["centroid"], yaw=r["yaw"], top_z=r["top_z"], base_z=r["base_z"]), "single", arm)
            r["_reach"] = reach
            png = self._crop_png(frame, r)
            self.planner.request(r["id"], r["label"], est, png)
        out = []
        for r in props:
            self._decide_into(r)
            out.append(r)
        return out

    def _crop_png(self, frame, r):
        try:
            import cv2
            os.makedirs(self.out_dir, exist_ok=True)
            x0, y0, x1, y1 = r["box2d"]
            im = PR.draw_boxes(frame["rgb"], [dict(id=r["id"], box2d=r["box2d"], label="", mode="")], selected=r["id"], labels=False)
            path = os.path.join(self.out_dir, f"_crop_{r['id']}.png")
            cv2.imwrite(path, im[:, :, ::-1])
            return path
        except Exception:      # noqa: BLE001
            return None

    def _decide_into(self, r):
        est = dict(extent=r["extent"], centroid=r["centroid"], yaw=r["yaw"])
        d = self.planner.decide(r["id"], est, reach=r.get("_reach") or None)
        plan = d["plan"]
        r.update(mode=plan.mode, arm=plan.arm, grasp=plan.grasp, why=plan.why, width=plan.width, mass_est=plan.mass_est, axis=plan.axis,
                 planner=dict(source=d["source"], status=d["status"], rule=d["rule"].to_json(), vlm=d["vlm"], vetoed=d["vetoed"], reasons=d["reasons"],
                              fallback=d["fallback"], latency_s=d["latency_s"]))
        r["feasible"] = plan.mode != "infeasible"
        r["_plan"] = plan
        return r

    def refresh_decisions(self):
        """re-evaluate the planner decisions of the cached proposals (VLM answers that arrived since); cheap, any thread."""
        with self._lock:
            for r in self._props:
                self._decide_into(r)

    # ------------------------------------------------------------------ request / cancel
    def request(self, obj_id):
        """start the pick of proposal `obj_id`.  False if unknown / busy."""
        with self._lock:
            tgt = next((p for p in self._props if p["id"] == obj_id), None)
        if tgt is None or self._skill is not None and self._skill.state not in ("FAILED", "DONE_RELEASED") or (self._plan_fut is not None and not self._plan_fut.done()):
            return False
        self._sel = obj_id
        self._status = "planning"
        self._error = None
        self._wp = None
        self._skill = None
        self._plan_fut = self._ex.submit(self._plan_job, tgt, self._latest_snapshot())
        return True

    def _latest_snapshot(self):
        return self._snap

    def request_with_snapshot(self, sim, obj_id):
        """request + the physics snapshot the planner needs (collision scratch) -- the sim thread calls this (cheap copy)."""
        m = sim.m
        d = mujoco.MjData(m)
        mujoco.mj_copyData(d, m, sim.d)
        self._snap = dict(d=d, base=sim.pose().copy(), lift=sim.lift_actual(), t=sim.t, sim_m=m, robot_geom=np.array(sim.robot_geom, bool), ctl=sim.ctl)
        return self.request(obj_id)

    def _plan_job(self, tgt, snap):
        t0 = time.time()
        try:
            scr = SK.Scratch(types_sim(snap))
            self.refresh_decisions()
            plan = next(p for p in self.proposals_raw() if p["id"] == tgt["id"])["_plan"]
            obj = dict(extent=tgt["extent"], centroid=tgt["centroid"], yaw=tgt["yaw"], top_z=tgt["top_z"], base_z=tgt["base_z"])
            if plan.mode == "infeasible":
                raise SK.PlanError("INFEASIBLE", plan.why)
            if plan.mode == "single":
                other = "right" if plan.arm == "left" else "left"
                wp = SK.plan_single(scr, obj, [plan.arm, other])
                if wp.arms[0] != plan.arm:
                    self._events.append(f"arm {plan.arm} failed, {wp.arms[0]} used: {wp.note}")
            else:
                obj["width_axis"] = plan.width
                wp = SK.plan_bimanual(scr, obj, plan.axis)
            self._wp = wp
            self._plan_info = dict(plan=plan.to_json(), arms=list(wp.arms), plan_ms=round((time.time() - t0) * 1000, 1), ik=_jsonable(wp.ik))
            self._t["plan_ms"] = self._plan_info["plan_ms"]
            self._status = "ready"
        except SK.PlanError as e:
            self._status, self._error = "failed", f"{e.code}: {e.msg}"
            self._plan_info = dict(code=e.code, msg=e.msg, plan_ms=round((time.time() - t0) * 1000, 1))
        except Exception as e:      # noqa: BLE001
            import traceback
            self._status, self._error = "failed", f"PLAN_ERROR: {type(e).__name__}: {str(e)[:160]}"
            self.log(traceback.format_exc())

    def proposals_raw(self):
        with self._lock:
            return list(self._props)

    def cancel(self):
        self._events.append("cancel")
        if self._skill is not None and self._skill.state not in ("FAILED", "DONE_RELEASED"):
            self._skill.cancel()
            self._status = "cancelling"
        else:
            self._sel, self._wp, self._status = None, None, "idle"

    # ------------------------------------------------------------------ the sim-thread tick
    def step(self, sim):
        if self._skill is None and self._status == "ready" and self._wp is not None:
            self._skill = SK.PickSkill(sim, self._wp, log=self.log)
            self._status = "executing"
            self._wp = None
        if self._skill is not None:
            self._skill.step(sim)
            st = self._skill.state
            if st == "HOLD":
                self._status = "done" if self._skill.done else "holding"
            elif st == "FAILED":
                self._status, self._error = "failed", f"{self._skill.fail[0]}: {self._skill.fail[1]}"
                self._finish_pick(sim)
            elif st == "DONE_RELEASED":
                self._status = "idle"
                self._finish_pick(sim)
            elif self._status not in ("cancelling",):
                self._status = "executing"

    def _finish_pick(self, sim):
        if self._skill is None:
            return
        self.history.append(dict(id=self._sel, status=self._status, error=self._error, timeline=[(s, round(t, 2)) for s, t in self._skill.timeline],
                                 furniture_contacts=self._skill.furniture_contacts))
        self._skill = None

    # ------------------------------------------------------------------ HUD
    def state(self):
        sk = self._skill
        d = dict(status=self._status, selected=self._sel, error=self._error, backend=self.planner.backend_name(), planner=self.planner.planner,
                 timing=dict(self._t), events=self._events[-5:],
                 skill=None if sk is None else dict(sk.phase_info(), mode=sk.wp.mode, arms=list(sk.arms), furniture_contacts=sk.furniture_contacts))
        with self._lock:
            d["proposals"] = [_jsonable({k: p[k] for k in ("id", "box2d", "label", "extent", "centroid", "mode", "arm", "grasp", "why", "feasible", "planner")}) for p in self._props]
        d["vlm_log_n"] = len(self.planner.records)
        return d

    def vlm_log(self):
        return _jsonable(list(self.planner.records))

    def close(self):
        self.planner.close()
        if self._cam is not None:
            try:
                self._ex.submit(self._cam.close).result(timeout=20)      # the EGL context must be freed in the thread that made it
            except Exception:      # noqa: BLE001
                pass
            self._cam = None
        self._ex.shutdown(wait=True, cancel_futures=True)


class types_sim:
    """minimal sim-like view for skill.Scratch(snapshot): m, d (copy), ctl joint index arrays, robot masks."""

    def __init__(self, snap):
        self.m = snap["sim_m"]
        self.d = snap["d"]
        self.ctl = snap["ctl"]
