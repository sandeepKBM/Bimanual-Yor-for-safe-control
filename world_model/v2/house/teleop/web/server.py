"""teleop/web/server.py -- browser teleop for the SIMULATED bimanual YOR (aiohttp, 127.0.0.1 only).  SIMULATION ONLY.

    cd <repo> ; export PYTHONPATH=$PWD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 $PY -m world_model.v2.house.teleop.web.server --scene h5 --task carry --shield
    # laptop:  ssh -L 8765:localhost:8765 <server>   then open the URL (with ?token=...) the server prints.

Threads: the asyncio loop only does websocket I/O; the SIM THREAD owns MuJoCo: 50 Hz tick = hub.step -> shield -> sim.tick.  Rendering + JPEG
encode (--fps) run in a separate PROCESS (--render-mode process, default; render_proc.py: shared-memory state snapshot, own EGL context) so they
never contend with the sim for the GIL; --render-mode thread = the old in-thread renderer.  --fast / --split opt in to teleop/fastpath.py.  The sim paces itself to wall clock when it is fast enough; otherwise it runs flat out (real-time factor < 1,
shown in the HUD).  The sim is paused while no browser is connected.  Imports no robot driver, no ROS, no joystick library.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import asyncio
import collections
import inspect
import json
import math
import secrets
import sys
import threading
import time
import traceback
from urllib.parse import urlparse

import cv2
import numpy as np
from aiohttp import web, WSMsgType

from . import inputs as IN
from . import episode as EP
from . import pick_link as PLK

HERE = os.path.dirname(os.path.abspath(__file__))
DOORWAYS = {"h5": ((3.875, 4.94),)}                 # map-frame doorway centres for --d-wall-door (h5: door|4|7 frame-to-frame span 3.45..4.31 m at y = 4.94)
from . import render_proc as RP
from .. import ghost as GH

CAMERAS = RP.CAMERAS                                 # ("ego", "chase", "top", "wrist_l", "wrist_r"): ego = torso head camera (proposed ZED mount), the default view
WRIST = {"wrist_l": "left_arm_egocentric", "wrist_r": "right_arm_egocentric"}


class SimWorker(threading.Thread):
    def __init__(self, args, hub: IN.InputHub):
        super().__init__(daemon=True, name="sim")
        self.a, self.hub = args, hub
        self.stop_ev = threading.Event()
        self.clients = 0
        self.ready = False
        self.error = ""
        self.telemetry: dict = dict(ready=False)
        self.frame: bytes | None = None
        self.frame_id = 0
        self.cam_i = CAMERAS.index(getattr(args, "view", "ego"))
        self.inset = True
        self.shield_on = bool(args.shield)
        self.ghost_on = bool(getattr(args, "ghost", True))
        self.gf = None                                         # last ghost frame (teleop/ghost.py); visual only
        self.wall_hist = collections.deque(maxlen=200)       # (wall, sim_t)
        self.interv_hist = collections.deque(maxlen=100)
        self.tick_ms = collections.deque(maxlen=50)
        self.render_ms = collections.deque(maxlen=30)
        self.render_n = collections.deque(maxlen=60)
        self.frame_wall = 0.0
        self.episode = None                                   # episode.Episode (None with --episode off)
        self.pick = None                                      # pick_link.PickLink (only --task pick)
        self._ep_events: list = []
        self.lang = None                                      # house/lang WebLang (only with --lang; default off = nothing below runs)

    # ------------------------------------------------------------------ build / reset
    def build(self):
        import mujoco
        from .. import shield as SH
        from ..sim import TeleopSim
        a = self.a
        os.environ["BIMANUAL_FLMPC_IMPL"] = a.flmpc_impl
        if a.fast:
            os.environ["BIMANUAL_TELEOP_FAST"] = "1"            # else: whatever the environment says (default off)
        if a.split:
            os.environ["BIMANUAL_TELEOP_SPLIT"] = "1"
            from .. import fastpath as FP
            FP.install_split_hook()                              # no-op once sim.py carries the 2-line hook itself
        carry = a.task == "carry"
        people = getattr(a, "people", "on")
        pick = a.task == "pick"
        if pick:                                                  # free arms + pick table + props (house/pick); same kwargs as pick/smoke.py
            kw = dict(scene_name=a.scene, seed=a.seed, with_people=people != "off", task="pick", pick_props=a.pick_props)
        else:
            kw = dict(scene_name=a.scene, grasped=carry, counter=not carry, desk=not carry, seed=a.seed, with_people=people != "off")
        if people == "on":
            kw["people_clock"] = "time"      # the walker's track clock runs with sim time.  The legacy clock only advances while the CURRENT segment velocity is non-zero, so a track
                                             # that starts with a stationary segment (h5 p_walker: t 0..2 s) never left its start (3.6, 3.8) -- see INTERFACE.md "People"
        if a.grasp_mode is not None:
            kw["grasp_mode"] = a.grasp_mode
        if getattr(a, "spawn_at", None):                           # start the robot at a scene waypoint (e.g. cafe_counter_pick: hands at the tray, for the assisted grasp)
            from .. import scene as SCN
            kw["spawn"] = SCN.get_scene(a.scene).waypoints[a.spawn_at]
        t0 = time.time()
        self.sim = TeleopSim(**kw)
        sim = self.sim
        if people == "static":               # everybody stands still where the track starts (the old frozen-walker behaviour, now explicit)
            from .. import scene as SCN
            SCN.freeze_people(sim.scene)
            sim.set_people()
        wp = sim.scene.waypoints
        cfg = SH.ShieldCfg(footprint=getattr(a, "footprint", "fixed"), fallback=getattr(a, "fallback", "brake"))
        if getattr(a, "d_wall_door", None) is not None:        # opt-in doorway buffer (default off)
            cfg.d_wall_door = a.d_wall_door
            cfg.doorways = tuple(DOORWAYS.get(a.scene, ()))
        self.shield = SH.Shield(cfg, occ=sim.occ, a_hw=sim.a_hw)
        self.hub.tuck_goal = None
        self.shield.docks = [] if pick else ([wp["cafe_counter_pick"][:2]] if carry else [wp["cafe_counter_pick"][:2], wp["lab_desk_place"][:2]])
        goal = None if pick else wp["cafe_counter_pick"][:2] if carry else wp["lab_desk_place"][:2]
        self.episode = EP.Episode(a.task, goal, timeout_s=getattr(a, "episode_timeout", 0.0), grasp_mode=a.grasp_mode) if getattr(a, "episode", "on") == "on" else None
        self._ep_events = []
        self.passthru = SH.Passthrough()
        self.ghost = GH.Predictor()                             # operator-intent ghost: re-created (= cleared) on every episode reset
        self.arm_intent = GH.ArmIntent(sim.of, iters=4)
        self.ghost_draw = GH.Drawer(sim.m)
        self.gf = None
        if getattr(a, "lang", "off") != "off":                    # language layer (flag-gated): instruction box -> grounding + spec + mu prior -> shield
            if self.lang is None:
                from world_model.v2.house.lang import web as LW
                self.lang = LW.WebLang(a.lang, cache_dir=a.lang_cache, source=a.lang_source, snapshot=a.lang_snapshot, log=lambda m: print(m, flush=True))
            elif self.lang.last is not None:                      # episode reset: the new shield keeps the language-authored spec / mu set
                self.shield.set_lang(self.lang.last)
        W, H = a.width, a.height
        sim.m.vis.global_.offwidth, sim.m.vis.global_.offheight = max(W, 640), max(H, 480)
        self.mj = mujoco
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.fovy0 = float(sim.m.vis.global_.fovy)
        if a.render_mode == "thread":
            self._make_thread_renderers()
        for _ in range(25 if pick else 3):                            # warm-up (numba compile of the FL-MPC) with a zero command; pick: props settle (as pick/smoke.py)
            sim.tick(self.hub.to_cmd(IN.Intent(), 1.0))
        if pick:                                                     # one PickSession per built sim (its camera / snapshots belong to this model); MOCK labels unless --pick-backend says otherwise
            from ...pick import session as PSE
            from ...pick import smoke as PSM
            if self.pick is not None:
                self.pick.close()
            self.pick = PLK.PickLink(PSE.PickSession(planner=a.pick_planner, backend=a.pick_backend, mock_labels=PSM.LABELS, seed=a.seed,
                                                     log=lambda m: print(f"[pick] {m}", flush=True)))
        if a.render_mode == "process":
            if getattr(self, "link", None) is None:
                self.link = self._make_link()
            else:
                self.link.reload(sim.m)
        self.build_s = time.time() - t0
        print(f"[sim] built {a.scene}/{a.task} FL-MPC impl={getattr(sim.ctl.fl, 'impl', '?')} in {self.build_s:.1f}s", flush=True)

    def _make_thread_renderers(self):
        a, sim = self.a, self.sim
        self.r = self.mj.Renderer(sim.m, a.height, a.width)
        self.r_in = self.mj.Renderer(sim.m, a.inset_h, a.inset_w)

    def _make_link(self):
        import tempfile
        from . import render_proc as RP
        return RP.RenderLink(self.sim.m, self.a, self._on_frame, os.environ.get("TELEOP_RENDER_TMP") or tempfile.gettempdir())

    def _on_frame(self, ms, jpg):                                     # render-reader thread: just store the bytes
        self.frame = jpg
        self.frame_id += 1
        self.render_ms.append(ms)
        self.render_n.append(time.perf_counter())

    def close(self):
        if self.pick is not None:
            self.pick.close()
            self.pick = None
        link = getattr(self, "link", None)
        if link is not None:
            link.close()
            self.link = None

    def _close_renderers(self):
        if self.a.render_mode != "thread":
            return
        for r in (getattr(self, "r", None), getattr(self, "r_in", None)):
            try:
                r.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ rendering
    def _render_one(self, r, camname, size):
        sim = self.sim
        self.sim.m.vis.global_.fovy = self.fovy0
        if camname in WRIST:
            r.update_scene(sim.d, camera=WRIST[camname])
        elif camname == "ego":
            self.sim.m.vis.global_.fovy = RP.ego_cam(self.cam, sim.base.x, sim.base.y, sim.base.yaw, sim.lift_actual())
            r.update_scene(sim.d, camera=self.cam)
            self.ghost_draw.add(r.scene, sim.d, self.gf, which="arms", z_arrow=GH.Z_ARROW_EGO)
        else:
            yaw = sim.base.yaw
            cam = self.cam
            if camname == "top":
                cam.lookat[:] = (sim.base.x, sim.base.y, 0.5)
                cam.distance, cam.elevation, cam.azimuth = 5.0, -80.0, math.degrees(yaw)
            else:                                                     # chase: behind the robot, looking along its heading
                cam.lookat[:] = (sim.base.x + 0.5 * math.cos(yaw), sim.base.y + 0.5 * math.sin(yaw), 0.8)
                cam.distance, cam.elevation, cam.azimuth = 2.4, -25.0, math.degrees(yaw)
            r.update_scene(sim.d, camera=cam)
            self.ghost_draw.add(r.scene, sim.d, self.gf)                # no ghost in the wrist views
        return r.render()

    def render_frame(self):
        main = CAMERAS[self.cam_i]
        img = self._render_one(self.r, main, None)
        if self.inset:
            other = "chase" if (main in WRIST or main == "ego") else ("wrist_l" if main != "wrist_l" else "wrist_r")
            ins = self._render_one(self.r_in, other, None)
            h, w = ins.shape[:2]
            img = img.copy()
            img[-h - 6:-6, 6:6 + w] = ins
            cv2.rectangle(img, (6, img.shape[0] - h - 6), (6 + w, img.shape[0] - 6), (255, 255, 255), 1)
        ok, buf = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, self.a.quality])
        if ok:
            self.frame = buf.tobytes()
            self.frame_id += 1

    # ------------------------------------------------------------------ main loop
    def run(self):
        try:
            self.build()
            self.ready = True
            self.loop()
        except Exception:
            self.error = traceback.format_exc()
            print(self.error, flush=True)
            self.telemetry = dict(ready=False, error=self.error.splitlines()[-1])

    def handle_server_events(self):
        for ev in self.hub.pop_server_events():
            if ev == "shield_toggle":
                self.shield_on = not self.shield_on
            elif ev == "camera_next":
                self.cam_i = (self.cam_i + 1) % len(CAMERAS)
            elif ev == "inset_toggle":
                self.inset = not self.inset
            elif ev == "ghost_toggle":
                self.ghost_on = not self.ghost_on
                self.ghost.reset()
            elif ev == "episode_end":
                self._ep_events.append(ev)
            elif ev == "reset":
                self.ready = False
                self.telemetry = dict(ready=False, msg="resetting")
                self._close_renderers()
                self.build()
                if self.episode is not None:                  # a new episode starts disarmed in IDLE (Enter / Start begins it); --episode off keeps the Alpha behaviour
                    with self.hub.lock:
                        self.hub._stop("new episode - Enter / Start to begin")
                self.interv_hist.clear()
                self.wall_hist.clear()
                self.ready = True

    def apply_lang(self):
        """sim thread: install a finished language result on the shield between ticks (the VLM thread only queues it)."""
        p = self.lang.take_pending()
        if p is None:
            return
        try:
            self.shield.set_lang(None if isinstance(p, str) else p)
            if not isinstance(p, str):
                print(f"[lang] shield updated: mu [{p.mu.lo:.3f}, {p.mu.hi:.3f}], spec {p.spec.status}, clearances {p.spec.clearances}", flush=True)
        except Exception as e:      # noqa: BLE001
            print(f"[lang] could not apply the result to the shield: {type(e).__name__}: {e}", flush=True)

    def loop(self):
        a = self.a
        from ..sim import TICK
        from ..arm import _log_R
        next_t = time.perf_counter()
        last_render = 0.0
        info = dict(intervention_norm=0.0, label="", qp="")
        o = None
        while not self.stop_ev.is_set():
            if self.clients <= 0:                                     # nobody watching: pause, stay safe
                self.hub.disconnect(None)
                if self.pick is not None:
                    self.pick.on_disconnect()
                time.sleep(0.05)
                next_t = time.perf_counter()
                self.wall_hist.clear()
                continue
            self.handle_server_events()
            if self.lang is not None:
                self.apply_lang()
            t0 = time.perf_counter()
            sim = self.sim
            obs = sim.observe()
            obs["tray_offset"] = (sim.of.dp.copy(), float(_log_R(sim.of.dR)[2]))     # commanded tray offset from the normal carry pose: the tuck / carry presets drive it
            cmd = self.hub.step(obs)
            ep = self.episode
            pk = self.pick
            if pk is not None:
                st_ = self.hub.status()
                pk.pre_tick(sim, self.hub.enabled, latched=self.hub.latched, watchdog=bool(st_.get("watchdog")),
                            ep_state=None if ep is None else ep.state, ready=self.ready)
                cmd = pk.interlock(cmd)                               # no operator base / lift / jog motion while a skill runs
            if ep is not None and ep.blocks_motion:
                cmd = PLK.zero_cmd(cmd, motion_only=False)            # OUTCOME: every command stays zero until a reset
            sh = self.shield if self.shield_on else self.passthru
            safe, info = sh.filter(sim, cmd)
            self.gf = self.ghost_frame(sim, cmd, safe, info) if self.ghost_on else None     # read-only: before sim.tick, from the same state
            lab = str(info.get("label", ""))
            if sh is self.shield and lab != getattr(self, "_last_lab", "") and lab in ("brake_fallback", "escape_fallback"):
                try:                                                  # log WHY the shield fell back: the three tightest keep-out rows + every tier's QP status
                    _A, _b, _t = self.shield._keep_rows(sim, sim.base.v)
                    tight = [(_t[i], round(float(_b[i]), 3)) for i in np.argsort(_b)[:3]]
                except Exception as e:
                    tight = f"n/a ({type(e).__name__})"
                print(f"[shield] t={sim.t:.2f} {lab} pose=({sim.base.x:.2f},{sim.base.y:.2f},{sim.base.yaw:.2f}) v={np.round(sim.base.v, 2).tolist()} "
                      f"cmd={np.round(np.asarray(getattr(cmd, 'base', [0, 0, 0]), float), 2).tolist()} tiers={getattr(self.shield, 'dbg', {})} tightest={tight}", flush=True)
            self._last_lab = lab
            if pk is not None:
                pk.step(sim)
            o = sim.tick(safe)
            if ep is not None:
                self.episode_tick(ep, sim, o, info)
            t1 = time.perf_counter()
            self.tick_ms.append((t1 - t0) * 1e3)
            self.wall_hist.append((t1, sim.t))
            self.interv_hist.append(float(info.get("intervention_norm", 0.0)) > 0.02)
            now = t1
            if a.render_mode == "process":
                if self.link.alive():
                    self.link.publish(sim, self.cam_i, self.inset, None if self.gf is None else self.gf.to_vec())    # memcpy only; the child renders at --fps
                else:
                    print("[render] child process died; falling back to in-thread rendering", flush=True)
                    self.link.close(); self.link = None
                    a.render_mode = "thread"
                    self._make_thread_renderers()
            elif now - last_render >= 1.0 / a.fps:
                last_render = now
                self.render_frame()
                self.render_ms.append((time.perf_counter() - now) * 1e3)
                self.render_n.append(time.perf_counter())
            self.telemetry = self.make_telemetry(sim, obs, o, info, cmd, safe)
            next_t += TICK
            slack = next_t - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            elif slack < -0.5:
                next_t = time.perf_counter()

    def episode_tick(self, ep, sim, o, info):
        """sim thread, after sim.tick: advance the episode state machine from the values the telemetry uses; disarm the hub when an episode ends."""
        lost = sim.cup_lost_t is not None
        tel = dict(cup_lost=lost, beads_out=int(o["beads_out"]), cup_state="LOST" if lost else ("SPILLED" if o["beads_out"] > 0 else "ok"),
                   speed=float(np.hypot(sim.base.v[0], sim.base.v[1])), pose=[float(x) for x in sim.pose()],
                   interv_now=float(info.get("intervention_norm", 0.0)), shield_label=str(info.get("label", "")),
                   wall_clr=float(o["wall"]), person_clr=float(o["pdist"]), cup_tilt=float(o["cup_tilt"]), latched=self.hub.latched,
                   assist=({"place": sim.assist.metrics_.get("place", {}).get("ok")} if getattr(sim, "assist", None) is not None else None),
                   pick=None if self.pick is None else {"status": self.pick.status})
        evs, self._ep_events = self._ep_events, []
        ep.update(sim.t, time.time(), self.hub.enabled, tel, evs)
        if ep.consume_disarm():
            with self.hub.lock:
                self.hub._stop("episode over - Backspace = new episode")

    def ghost_frame(self, sim, cmd, safe, info):
        """operator-intent ghost (teleop/ghost.py): raw command vs the shield's, from the current measured state.  Visual only."""
        active = self.shield_on
        arm_dq = self.arm_intent.dq(cmd.jog, safe.jog) if active else None
        return self.ghost.step(sim.pose(), sim.base.v, cmd.twist, safe.twist, float(info.get("intervention_norm", 0.0)),
                               float(info.get("intervention_aux", 0.0)), lift=(sim.lift_h, sim.lift_v, cmd.lift_v, safe.lift_v),
                               arm_dq=arm_dq, active=active)

    def ghost_telemetry(self):
        gf = self.gf
        if gf is None:
            return dict(on=False)
        tx = GH.describe(gf)
        return dict(on=True, shown=gf.alpha > 0, warn=bool(gf.warn), requested=tx["requested"], permitted=tx["permitted"], gap=tx["gap"])

    def make_telemetry(self, sim, obs, o, info, cmd, safe):
        st = self.hub.status()
        rtf = float("nan")
        if len(self.wall_hist) >= 20:
            w0, s0 = self.wall_hist[0]
            w1, s1 = self.wall_hist[-1]
            rtf = (s1 - s0) / max(w1 - w0, 1e-6)
        fps = float("nan")
        if len(self.render_n) >= 3:
            fps = (len(self.render_n) - 1) / max(self.render_n[-1] - self.render_n[0], 1e-6)
        lost = sim.cup_lost_t is not None
        n_b = int(sim.rig.n_beads)
        fp = self.shield.fp_now
        rms = float(np.mean(self.render_ms)) if self.render_ms else float("nan")
        wall, pd = float(o["wall"]), float(o["pdist"])
        return dict(
            ready=True, t=round(sim.t, 2), **st, shield=self.shield_on, shield_label=str(info.get("label", "")),
            interv_frac=round(float(np.mean(self.interv_hist)) if self.interv_hist else 0.0, 3),
            interv_now=round(float(info.get("intervention_norm", 0.0)), 3),
            rtf=round(rtf, 2) if rtf == rtf else None, render_fps=round(fps, 1) if fps == fps else None,
            tick_ms=round(float(np.mean(self.tick_ms)), 1) if self.tick_ms else None,
            beads_out=int(o["beads_out"]), n_beads=n_b, cup_lost=lost, cup_tilt=round(float(o["cup_tilt"]), 1),
            cup_state="LOST" if lost else ("SPILLED" if o["beads_out"] > 0 else "ok"),
            wall_clr=round(wall, 3) if np.isfinite(wall) else None, person_clr=round(pd, 3) if np.isfinite(pd) else None,
            speed=round(float(np.hypot(sim.base.v[0], sim.base.v[1])), 3), wz=round(float(sim.base.v[2]), 3),
            pose=[round(float(x), 3) for x in sim.pose()], lift_h=round(float(o["lift_h"]), 3),
            grasped=bool(o["grasped"]), can_grasp=bool(obs["can_grasp"]), infeasible=bool(o["infeasible"]),
            bumps=int(o["bump_contacts"]), cmd_twist=[round(float(x), 3) for x in cmd.twist],
            safe_twist=[round(float(x), 3) for x in safe.twist], cam=CAMERAS[self.cam_i], inset=self.inset,
            impl=getattr(sim.ctl.fl, "impl", "?"), task=self.a.task, scene=self.a.scene, ghost=self.ghost_telemetry(),
            people=getattr(self.a, "people", "on"), render_ms=round(rms, 1) if rms == rms else None,
            fp_mode=self.shield.cfg.footprint, fp_w=round(fp["y"][1] - fp["y"][0], 3), fp_l=round(fp["x"][1] - fp["x"][0], 3),
            fp_ms=round(self.shield.fp_ms[0] / max(self.shield.fp_ms[1], 1), 3),
            tray_off=([round(float(x), 3) for x in self.hub.tray_off[0]] + [round(self.hub.tray_off[1], 3)]) if self.hub.tray_off is not None else None,
            **({"episode": self.episode.to_json()} if self.episode is not None else {}),
            **({"pick": self.pick.telemetry(sim.pose(), sim.lift_actual(), None if self.episode is None else self.episode.state, self.hub.enabled)}
               if self.pick is not None else {}),
            **({"assist": dict(state=str(sim.assist.state), attempt=int(sim.assist.attempt), t_state=round(float(sim.assist.t_state), 1),
                               fail=str(sim.assist.fail_reason or ""), place=sim.assist.metrics_.get("place", {}).get("ok"))}
               if getattr(sim, "assist", None) is not None else {}))


# ------------------------------------------------------------------------------------------------------------------ web app
def make_app(args, hub, worker, token):
    routes = web.RouteTableDef()
    state = dict(ws=None)

    def authed(req):
        return secrets.compare_digest(req.query.get("token", ""), token)

    @routes.get("/")
    async def index(req):
        if not authed(req):
            return web.Response(status=403, text="missing/invalid ?token=  (see the server's startup line)")
        return web.FileResponse(os.path.join(HERE, "static", "index.html"), headers={"Cache-Control": "no-store"})

    @routes.get("/app.js")
    async def appjs(req):
        return web.FileResponse(os.path.join(HERE, "static", "app.js"), headers={"Cache-Control": "no-store"})

    @routes.get("/ws")
    async def ws_handler(req):
        if not authed(req):
            return web.Response(status=403, text="bad token")
        origin = req.headers.get("Origin")
        # any loopback origin: the laptop end of an ssh / VS Code tunnel may sit on a different local port than --port
        if origin and urlparse(origin).hostname not in ("localhost", "127.0.0.1"):
            return web.Response(status=403, text="bad origin")
        ws = web.WebSocketResponse(max_msg_size=1 << 16, heartbeat=None)
        await ws.prepare(req)
        old = state["ws"]
        if old is not None and not old.closed:
            await old.send_str(json.dumps(dict(type="bye", why="another browser took control")))
            await old.close()
        state["ws"] = ws
        worker.clients = 1
        hub.disconnect(None)                                          # every new connection starts DISABLED
        print(f"[ws] client connected from {req.remote}", flush=True)
        await ws.send_str(json.dumps(dict(type="hello", kbd_axes={k: list(v) for k, v in IN.KBD_AXES.items()}, kbd_events=IN.KBD_EVENTS,
                                          presets=list(hub.cfg["presets"]), preset_names=list(hub.cfg["preset_names"]),
                                          watchdog_s=hub.cfg["watchdog_s"], cameras=list(CAMERAS), task=args.task, scene=args.scene,
                                          lang=(args.lang if getattr(args, "lang", "off") != "off" else None),
                                          pick=(dict(planner=args.pick_planner, backend=args.pick_backend) if args.task == "pick" else None),
                                          episode=getattr(args, "episode", "on") == "on")))

        async def writer():
            last_frame, last_tel, last_lang = -1, 0.0, -1
            while not ws.closed:
                if worker.lang is not None and worker.lang.version != last_lang:       # HUD panel of the language layer (sent on change, incl. the running timer every 1 s)
                    last_lang = worker.lang.version
                    try:
                        await ws.send_str(json.dumps(dict(type="lang", **worker.lang.state()), allow_nan=False, default=lambda x: None))
                    except ValueError:
                        pass
                if worker.frame is not None and worker.frame_id != last_frame:
                    last_frame = worker.frame_id
                    await ws.send_bytes(b"\x01" + worker.frame)
                now = time.monotonic()
                if now - last_tel >= 0.05:
                    last_tel = now
                    try:
                        await ws.send_str(json.dumps(dict(type="tel", **worker.telemetry), allow_nan=False, default=lambda x: None))
                    except ValueError as e:                           # NaN slipped into telemetry: log, keep streaming
                        print("[ws] telemetry not JSON-clean:", e, flush=True)
                await asyncio.sleep(0.006)

        wt = asyncio.ensure_future(writer())
        try:
            async for m in ws:
                if m.type == WSMsgType.TEXT:
                    try:
                        msg = json.loads(m.data)
                        if not isinstance(msg, dict):
                            continue
                        if msg.get("type") == "input":
                            hub.feed(msg)
                        elif msg.get("type") == "ping":
                            await ws.send_str(json.dumps(dict(type="pong", id=msg.get("id"))))
                        elif msg.get("type") == "pad_gone":
                            hub.disconnect("pad")
                        elif msg.get("type") in ("pick_propose", "pick_select", "pick_cancel") and worker.pick is not None and worker.ready:
                            t_ = msg["type"]
                            if t_ == "pick_propose":
                                worker.pick.enqueue_propose()
                            elif t_ == "pick_cancel":
                                worker.pick.enqueue_cancel()
                            elif isinstance(msg.get("id"), int) and not isinstance(msg.get("id"), bool):
                                worker.pick.enqueue_select(msg["id"])
                        elif msg.get("type") == "lang" and worker.lang is not None and worker.ready:
                            worker.lang.submit(msg.get("instruction", ""), sim=worker.sim)       # background thread; the sim keeps running
                        elif msg.get("type") == "lang_clear" and worker.lang is not None:
                            worker.lang.clear()
                    except Exception as e:                                # malformed message: ignore, never crash the loop
                        print("[ws] bad message:", repr(e), flush=True)
                elif m.type == WSMsgType.ERROR:
                    break
        finally:
            wt.cancel()
            if state["ws"] is ws:
                state["ws"] = None
                worker.clients = 0
                hub.disconnect(None)
            print("[ws] client disconnected", flush=True)
        return ws

    app = web.Application()
    app.add_routes(routes)
    return app


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="browser teleop server (SIMULATION ONLY, binds 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--scene", default="h5", choices=["h5", "office"])
    ap.add_argument("--task", choices=["carry", "coffee", "pick"], default="carry")
    ap.add_argument("--shield", dest="shield", action="store_true", default=True)
    ap.add_argument("--no-shield", dest="shield", action="store_false")
    ap.add_argument("--fps", type=float, default=15.0, help="render/stream rate")
    ap.add_argument("--no-ghost", dest="ghost", action="store_false", default=True, help="start with the operator-intent ghost off (key O toggles)")
    ap.add_argument("--view", choices=["ego", "chase"], default="ego", help="initial camera: ego = torso head camera (robot's-eye view, the default), chase = behind the robot.  C cycles all views")
    ap.add_argument("--people", choices=["on", "off", "static"], default="on",
                    help="on = scripted people walk (sim-time track clock); static = they stand still at their track start; off = no people (no person keep-out)")
    ap.add_argument("--footprint", choices=["fixed", "live"], default="fixed",
                    help="shield footprint: fixed = phase-1 hold-pose box (conservative, 0.77 m wide); live = box of the robot's CURRENT arm / tray / cup pose (tuck with N narrows it)")
    ap.add_argument("--fallback", choices=["brake", "escape"], default="brake",
                    help="shield fallback when every QP tier is infeasible: brake = emergency brake (holds the robot even inside a violated buffer); "
                         "escape = first try the keep-out rows with violated demands capped at the hardware slew (moves out of the buffer), then brake")
    ap.add_argument("--d-wall-door", dest="d_wall_door", type=float, default=None,
                    help="opt-in: wall buffer [m] inside a doorway (within 0.7 m of its centre) instead of ShieldCfg.d_wall = 0.10.  Default off")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--inset-w", dest="inset_w", type=int, default=224)
    ap.add_argument("--inset-h", dest="inset_h", type=int, default=168)
    ap.add_argument("--quality", type=int, default=70, help="JPEG quality")
    ap.add_argument("--flmpc-impl", choices=["py", "rt"], default="rt", help="FL-MPC implementation (rt = exact QP, fastest)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--token", default=None, help="access token (default: random, printed at startup)")
    ap.add_argument("--render-mode", dest="render_mode", choices=["process", "thread"], default="process",
                    help="process = renderer in a separate process from a shared-memory state snapshot (default); thread = old in-thread renderer")
    ap.add_argument("--fast", action="store_true", help="BIMANUAL_TELEOP_FAST=1: numba barriers + minimal dynamics refresh (bit-identical, ~3.5x faster tick)")
    ap.add_argument("--split", action="store_true", help="BIMANUAL_TELEOP_SPLIT=1: mj_step1/mj_step2 control split (installs the sim.py hook in memory)")
    ap.add_argument("--lang", choices=["off", "auto", "live", "mock"], default="off",
                    help="language layer (house/lang): instruction box in the page; a VLM (live daemon via the file-queue service, or the offline mock) grounds objects, authors the IR spec "
                         "+ a mu prior and installs them on the shield.  auto = live if the daemon is up else mock.  Default off = nothing changes")
    ap.add_argument("--lang-source", choices=["sim", "snapshot"], default="sim", help="image + object proposals: the sim camera + instance segments, or a saved flagship perception result")
    ap.add_argument("--lang-snapshot", default=None, help="--lang-source snapshot: perception_desk.pkl path (default: the flagship desk capture)")
    ap.add_argument("--lang-cache", default="data/molmospaces/lang/web_cache", help="VLM response cache (replays are offline)")
    ap.add_argument("--grasp-mode", dest="grasp_mode", default=None, help="passed to TeleopSim(grasp_mode=...) if that kwarg exists")
    ap.add_argument("--spawn-at", dest="spawn_at", default=None, help="scene waypoint to spawn at (e.g. cafe_counter_pick); default = the scene start")
    ap.add_argument("--episode", choices=["on", "off"], default="on", help="episode lifecycle (IDLE/ACTIVE/OUTCOME, disarm at the end; telemetry key `episode`).  off = Alpha behaviour")
    ap.add_argument("--episode-timeout", dest="episode_timeout", type=float, default=0.0, help="sim seconds after which an ACTIVE episode fails (0 = no timeout)")
    ap.add_argument("--pick-props", dest="pick_props", default="default", help="--task pick: prop set name (house/pick/props.py SETS)")
    ap.add_argument("--pick-planner", dest="pick_planner", choices=["rule", "vlm", "vlm+veto"], default="rule")
    ap.add_argument("--pick-backend", dest="pick_backend", choices=["mock", "auto", "live"], default="mock", help="mock = offline labels (privileged shortcut); live needs the VLM service and is not used in tests")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    from ..sim import TeleopSim
    if a.grasp_mode is not None and "grasp_mode" not in inspect.signature(TeleopSim.__init__).parameters:
        sys.exit("TeleopSim has no grasp_mode kwarg in this checkout; omit --grasp-mode")
    token = a.token or secrets.token_urlsafe(9)
    hub = IN.InputHub()
    worker = SimWorker(a, hub)
    worker.start()
    app = make_app(a, hub, worker, token)
    print(f"[web] SIMULATION ONLY.  listening on 127.0.0.1:{a.port} (loopback only)", flush=True)
    print(f"[web] open  http://localhost:{a.port}/?token={token}   (after: ssh -L {a.port}:localhost:{a.port} <server>)", flush=True)
    try:
        web.run_app(app, host="127.0.0.1", port=a.port, print=None, handle_signals=True)
    finally:
        worker.stop_ev.set()
        worker.close()


if __name__ == "__main__":
    main()
