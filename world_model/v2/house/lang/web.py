"""lang/web.py -- language layer for the browser teleop (instruction box -> 'Ask VLM' -> HUD).  SIMULATION ONLY; loaded only with `--lang` (default off).

WebLang runs one grounding + authoring + mu-prior session in a background thread (the sim thread is never blocked: a VLM call takes seconds) and exposes
    state()           JSON-safe dict for the HUD (status idle|running|done|error, grounded names, pretty spec, compile status, mu interval, latencies, marks image)
    take_pending()    the finished LangResult (or CLEAR) for the SIM thread to apply to its Shield (server.SimWorker does that between ticks)
Image + proposals come from `source`:
    'sim'       the sim's own camera (head_cam, else front_camera) rendered offscreen from a COPY of the MjData; proposals = class-agnostic instance segments of the
                house geometry (the segmentation masks stand in for a perception front-end: the VLM names them, ground-truth names are never shown to it)
    'snapshot'  a saved flagship perception result (stereo zonotopes, speckled twin render) -- the real perception front-end, but of the flagship office world
"""
from __future__ import annotations

import base64
import re
import threading
import time

import numpy as np

from . import backend as BE, grounding as GR, session as SE, hooks as HK

CLEAR = "CLEAR"
MAX_INSTRUCTION = 400


def clean_instruction(s):
    s = re.sub(r"[\x00-\x1f\x7f]+", " ", str(s)).strip()
    return s[:MAX_INSTRUCTION]


def proposals_from_sim_seg(sim, cam=None, width=960, height=540, min_px=350, max_marks=GR.MAX_MARKS):
    """(rgb uint8, [Proposal], {id: gt key}) from the sim's camera.  Instance segmentation of the HOUSE geoms (robot, tray, cup excluded), one proposal per object key,
    zonotope = AABB of the object's geom points.  The gt keys (object names from the scene file) are returned for EVALUATION only."""
    import mujoco
    from world_model.v2.house import phase1 as P1
    from world_model.v2.house.zed_demo import perceive as PC
    from world_model.v2.tray_rep.sets.zono import Zonotope
    m = sim.m
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:] = sim.d.qpos, sim.d.qvel
    d.mocap_pos[:], d.mocap_quat[:] = sim.d.mocap_pos, sim.d.mocap_quat
    mujoco.mj_forward(m, d)
    names = [m.camera(i).name for i in range(m.ncam)]
    cam = cam or ("head_cam" if "head_cam" in names else "front_camera")
    m.vis.global_.offwidth, m.vis.global_.offheight = max(width, int(m.vis.global_.offwidth)), max(height, int(m.vis.global_.offheight))
    r = mujoco.Renderer(m, height, width)
    try:
        r.update_scene(d, camera=cam)
        rgb = r.render().copy()
        r.enable_segmentation_rendering()
        r.update_scene(d, camera=cam)
        seg = r.render()
    finally:
        r.close()
    cls, keys = P1.geom_class(m), PC.gt_keys(m)
    gid = seg[:, :, 0].astype(int)
    typ = seg[:, :, 1].astype(int)
    ok = (typ == int(mujoco.mjtObj.mjOBJ_GEOM)) & (gid >= 0) & (gid < m.ngeom)
    bykey = {}
    ys, xs = np.nonzero(ok)
    for g in np.unique(gid[ok]):
        if cls[g] != "H" or m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        sel = ok & (gid == g)
        bykey.setdefault(keys[g], []).append((g, sel))
    props, gt = [], {}
    for pid, (key, items) in enumerate(sorted(bykey.items())):
        mask = np.zeros(ok.shape, bool)
        for _, s in items:
            mask |= s
        if mask.sum() < min_px:
            continue
        yy, xx = np.nonzero(mask)
        P = np.concatenate([PC.geom_pts(m, d, g) for g, _ in items if len(PC.geom_pts(m, d, g))] or [np.zeros((0, 3))])
        if len(P) < 4:
            continue
        lo, hi = P.min(0), P.max(0)
        Z = Zonotope.from_box(lo, hi)
        props.append(GR.Proposal(pid, Z.c, Z.G, Z.c, Z.G, (float(xx.min()), float(yy.min()), float(xx.max()), float(yy.max())), int(mask.sum()), 0.0, "obj"))
        gt[pid] = str(key)
    props.sort(key=lambda p: -(p.box[2] - p.box[0]) * (p.box[3] - p.box[1]))
    return rgb, props[:max_marks], gt


def _b64_jpeg(path, width=640):
    import cv2
    im = cv2.imread(path)
    if im is None:
        return None
    s = width / im.shape[1]
    im = cv2.resize(im, (width, int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf.tobytes()).decode() if ok else None


class WebLang:
    def __init__(self, backend="auto", cache_dir=None, out_dir="data/molmospaces/lang/web", source="sim", snapshot=None, log=print, mu_queries=4):
        self.kind, self.cache_dir, self.out_dir, self.source, self.snapshot, self.log = backend, cache_dir, out_dir, source, snapshot, log
        self.mu_queries = mu_queries
        self._sess = None
        self._lock = threading.Lock()
        self._thread = None
        self._pending = None
        self.last = None                     # last finished LangResult
        self.version = 0
        self._state = dict(status="idle")
        self._img = None

    def backend_name(self):
        return getattr(self._sess.be, "name", self.kind) if self._sess else self.kind

    def _session(self):
        if self._sess is None:
            be = BE.make_backend(self.kind, cache_dir=self.cache_dir, log=self.log)
            self._sess = SE.LangSession(be, mu_queries=self.mu_queries, out_dir=self.out_dir, log=self.log)
        return self._sess

    def _set(self, **kw):
        with self._lock:
            self._state = dict(self._state, **kw)
            self.version += 1

    def busy(self):
        return self._thread is not None and self._thread.is_alive()

    def submit(self, instruction, sim=None):
        ins = clean_instruction(instruction)
        if not ins:
            return False
        if self.busy():
            return False
        self._set(status="running", instruction=ins, t_start=time.time(), error=None, backend=self.backend_name())
        self._thread = threading.Thread(target=self._run, args=(ins, sim), daemon=True, name="lang")
        self._thread.start()
        return True

    def clear(self):
        with self._lock:
            self._pending = CLEAR
        self.last = None
        self._set(status="idle", hud=None, marks_jpg=None, instruction="")

    def take_pending(self):
        with self._lock:
            p, self._pending = self._pending, None
        return p

    def _capture(self, sim):
        if self.source == "sim" and sim is not None:
            try:
                rgb, props, gt = proposals_from_sim_seg(sim)
                if props:
                    return rgb, props, gt, "sim camera + instance segments"
                self.log("[lang] sim capture produced no proposals; using the snapshot")
            except Exception as e:      # noqa: BLE001
                self.log(f"[lang] sim capture failed ({type(e).__name__}: {e}); using the snapshot")
        import pickle
        pkl = self.snapshot or "data/molmospaces/flagship/perception/perception_desk.pkl"
        with open(pkl, "rb") as f:
            res = pickle.load(f)
        return np.asarray(res["left"]), GR.proposals_from_result(res), {z["id"]: z["gt_key"] for z in res["zonos"]}, "saved flagship perception (stereo zonotopes)"

    def _run(self, ins, sim):
        t0 = time.time()
        try:
            sess = self._session()
            rgb, props, gt, src = self._capture(sim)
            lr = sess.run(ins, rgb, props, tag=f"web{self.version}")
            hud = lr.hud()
            hud["source"] = src
            hud["total_wall_s"] = round(time.time() - t0, 2)
            self.last = lr
            with self._lock:
                self._pending = lr
            self._set(status="done", hud=hud, marks_jpg=_b64_jpeg(lr.grounding.marks_png) if lr.grounding.marks_png else None, error=None)
        except Exception as e:      # noqa: BLE001
            self.log(f"[lang] FAILED {type(e).__name__}: {e}")
            self._set(status="error", error=f"{type(e).__name__}: {str(e)[:200]}")

    def state(self):
        with self._lock:
            s = dict(self._state)
        if s.get("status") == "running":
            s["elapsed_s"] = round(time.time() - s.get("t_start", time.time()), 1)
        s["version"] = self.version
        return s
