"""lang/hooks.py -- the (flag-gated, default-off) consumers of a LangResult.  SIMULATION ONLY.

    apply_to_shield(shield, lr)            teleop.shield.Shield.set_lang: mu set + IR-monitor spec from the language layer
    clearance_result(res, lr.clearance_by_zono)   copy of a flagship perception result whose Zi are inflated so the zonotope rows / base filter keep d from named objects
    LangPerceptionService                  flagship PerceptionService that runs grounding + authoring at the SENSE phases (--lang)
    render_head(sim, ...)                  one flat-colour head-camera frame of the MAIN sim (mu prior image)

Clearance semantics: a perceived zonotope Zi is inflated by the box [-(d - base_margin), +(d - base_margin)]^3 (exact Minkowski sum; the L_inf box contains the Euclidean
ball, so the real clearance is >= d, up to sqrt(3) d at the corners).  base_margin = the smallest margin the rows already add (arm rows 0.02 m, base filter 0.03 m);
a requested d below it changes nothing (the rows never relax below their default).
"""
from __future__ import annotations

import copy
import time

import numpy as np

BASE_MARGIN = 0.02


def apply_to_shield(shield, lr):
    shield.set_lang(lr)
    return shield


def inflate_pack(zp, r):
    """{'c','G'} + Minkowski sum with the box of half-width r: G <- [G | r I3]."""
    G = np.asarray(zp["G"], float)
    return dict(c=np.asarray(zp["c"], float), G=np.concatenate([G, r * np.eye(3)], axis=1))


def clearance_result(res, clearance_by_zono, base_margin=BASE_MARGIN):
    """-> (new result, {zono id: added inflation m}).  The input is not modified."""
    out = dict(res)
    out["zonos"] = []
    added = {}
    for z in res["zonos"]:
        d = clearance_by_zono.get(z["id"])
        if d is not None and d - base_margin > 1e-6:
            z2 = dict(z)
            z2["Zi"] = inflate_pack(z["Zi"], d - base_margin)
            z2["lang_clearance_m"] = float(d)
            out["zonos"].append(z2)
            added[z["id"]] = float(d - base_margin)
        else:
            out["zonos"].append(z)
    return out, added


def render_head(sim, width=640, height=360, cam="head_cam", cam_pose=None, f_px=None):
    """RGB uint8 frame of the MAIN sim (flat colours, real cup + tray).  cam_pose=(origin, R) (the perception camera's world pose, R columns = camera axes, looks along -R[:,2])
    renders through a free camera at exactly that pose (fovy from f_px); else the named camera (head_cam, else front_camera).  None if unavailable (no camera / no EGL)."""
    import math
    import mujoco
    m = sim.m
    names = [m.camera(i).name for i in range(m.ncam)]
    try:
        m.vis.global_.offwidth, m.vis.global_.offheight = max(width, int(m.vis.global_.offwidth)), max(height, int(m.vis.global_.offheight))
        r = mujoco.Renderer(m, height, width)
        fovy0 = float(m.vis.global_.fovy)
        try:
            if cam_pose is not None:
                origin, R = np.asarray(cam_pose[0], float), np.asarray(cam_pose[1], float)
                fwd = -R[:, 2]
                c = mujoco.MjvCamera()
                c.type = mujoco.mjtCamera.mjCAMERA_FREE
                c.distance = 1.0
                c.lookat[:] = origin + fwd
                c.azimuth = math.degrees(math.atan2(fwd[1], fwd[0]))
                c.elevation = math.degrees(math.asin(max(-1.0, min(1.0, fwd[2]))))
                if f_px:
                    m.vis.global_.fovy = math.degrees(2 * math.atan(0.5 * height / (f_px * height / 720.0)))
                r.update_scene(sim.d, camera=c)
            else:
                nm = cam if cam in names else ("front_camera" if "front_camera" in names else None)
                if nm is None:
                    return None
                r.update_scene(sim.d, camera=nm)
            return r.render().copy()
        finally:
            m.vis.global_.fovy = fovy0
            r.close()
    except Exception:       # noqa: BLE001
        return None


class LangPerceptionService:
    """flagship PerceptionService + language layer.  Built by simulate.run_side when --lang is given (see there).  Composition, not inheritance of
    behaviour: sense_* call the wrapped service, then ground / author on the capture and (desk) inflate the zonotopes for the named-object clearances."""
    name = "zed+lang"

    def __init__(self, inner, session, instruction, out_dir, on_result=None, log=print, sim=None, image="main"):
        self.inner, self.session, self.instruction, self.image = inner, session, instruction, image
        self.out_dir, self.on_result, self.log, self.sim = out_dir, on_result, log, sim
        self.lang = {}
        self.t_lang = {}

    def __getattr__(self, k):                 # results, mode, twin, backend ... of the wrapped service
        return getattr(self.inner, k)

    def _lang(self, kind, r):
        from .grounding import proposals_from_result
        import os
        os.makedirs(self.out_dir, exist_ok=True)
        t0 = time.time()
        props = proposals_from_result(r)
        im = None
        if self.image == "main" and self.sim is not None:            # flat-colour render of the MAIN sim through the perception camera's exact pose (same K / origin / R as the marks)
            im = render_head(self.sim, int(r["K"]["W"]), int(r["K"]["H"]), cam_pose=(r["origin"], r["R"]), f_px=float(r["K"]["f"]))
        src = "main-sim render at the perception camera pose" if im is not None else "perception twin left image"
        if im is None:
            im = np.asarray(r["left"])
        self.log(f"[lang] image: {src}")
        lr = self.session.run(self.instruction, im, props, tag=f"lang_{kind}")
        self.lang[kind] = lr
        self.t_lang[kind] = time.time() - t0
        if self.on_result is not None:
            self.on_result(lr, kind)
        return lr

    def sense_counter(self, obs):
        r = self.inner.sense_counter(obs)
        self._lang("counter", r)
        return r

    def sense_desk(self, obs):
        inn = self.inner
        r = inn._get("desk", obs)
        lr = self._lang("desk", r)
        if lr.clearance_by_zono:
            r2, added = clearance_result(r, lr.clearance_by_zono)
            r["zonos"] = r2["zonos"]                               # in place: services.results['desk'] (arm rows) sees the inflated zonotopes
            self.log(f"[lang] clearance inflation (m) by zono id: {added}")
            tw = getattr(inn, "twin", None)
            if tw is not None:
                self._replan(tw, r)
        plan = r["plan"]
        if inn.backend == "zono_rows":
            from world_model.v2.house.flagship.perception import _zono_rows_plan
            plan = _zono_rows_plan(r, obs, inn.log) or plan
        if not plan.get("ok"):
            inn.log("[perception] desk plan INFEASIBLE -> operator falls back to naive")
            return None
        return plan

    def _replan(self, tw, r):
        """re-run the zed_demo desk planner on the inflated zonotopes (live twin only)."""
        try:
            from world_model.v2.house.flagship.perception import zono_unpack
            res = dict(zonos=[dict(gt_key=z["gt_key"], Zi=zono_unpack(z["Zi"])) for z in r["zonos"]])
            sup = [dict(Z=zono_unpack(z["Z"])) for z in r["zonos"] if z["kind"] == "support"]
            r["plan"] = tw._plan_desk(res, sup, np.asarray(r["tray0"], float))
            self.log("[lang] desk plan recomputed on the inflated zonotopes")
        except Exception as e:      # noqa: BLE001
            self.log(f"[lang] re-plan failed ({type(e).__name__}: {e}); keeping the stereo plan")
