"""teleop/web/render_proc.py -- out-of-process renderer for the browser teleop (SIMULATION ONLY; pure MuJoCo + OpenCV, no robot code).

The sim thread of server.py used to render + JPEG-encode in-thread (and, with the GIL, every other server thread contended with it).
Here the renderer is a separate spawned PROCESS with its own MuJoCo model copy and EGL context:

  parent (sim thread)  once per tick:  RenderLink.publish(sim, cam_i, inset)   -> memcpy of qpos / mocap / eq_active into a shared-memory
                                         float64 vector guarded by a seqlock (the parent never blocks, never waits for the renderer)
  child (this module)  every 1/fps s:   copy the latest consistent snapshot -> mj_kinematics + mj_camlight -> render (+ inset) -> JPEG
                                         -> pipe -> a parent reader thread that just stores the bytes.

What the renderer needs from MjData: qpos (robot, tray, cup, beads), mocap_pos / mocap_quat (base root, moving people), eq_active.  All other
visual state is derived by kinematics.  The model reaches the child as a binary .mjb written by mj_saveModel (so no scene code is imported).
The child exits when the parent dies (ppid watch) or on a stop message.  Binding / token / origin checks stay in server.py (unchanged).
"""
from __future__ import annotations

import math
import os
import time

from world_model.v2.house.teleop import ghost as GH      # light: numpy + mujoco + teleop.base

GH0 = 9              # header: [seq, base_x, base_y, base_yaw, cam_i, inset, sim_t, tick_id, lift_h] + ghost frame (GH.G_LEN floats)
HDR = GH0 + GH.G_LEN
CAMERAS = ("ego", "chase", "top", "wrist_l", "wrist_r")
WRIST = {"wrist_l": "left_arm_egocentric", "wrist_r": "right_arm_egocentric"}
# ego view = the torso ZED head camera (PROPOSED mount, confirm on hardware): same constants as ocs/fp_render (HEAD_POS / HEAD_PITCH_DEG / HEAD_FOVY), duplicated here so the
# renderer child imports nothing heavy.  The teleop rig has no head camera body, so the view is a free camera placed from the base pose + lift (rides the lift).
EGO_POS = (0.05, -0.20955, 1.15)       # base frame at lift 0 [m]
EGO_PITCH_DEG = 36.3                   # looks along base +x, pitched down
EGO_FOVY = 70.0                        # vertical field of view [deg]


def ego_cam(cam, x, y, yaw, lift):
    """fill a free MjvCamera `cam` with the head-camera pose for base pose (x, y, yaw) and lift height; returns the fovy [deg] the model must be rendered with."""
    c, s = math.cos(yaw), math.sin(yaw)
    p = math.radians(EGO_PITCH_DEG)
    org = (x + c * EGO_POS[0] - s * EGO_POS[1], y + s * EGO_POS[0] + c * EGO_POS[1], EGO_POS[2] + lift)
    fwd = (math.cos(p) * c, math.cos(p) * s, -math.sin(p))
    cam.lookat[:] = (org[0] + fwd[0], org[1] + fwd[1], org[2] + fwd[2])        # free camera: pos = lookat - distance * forward
    cam.distance, cam.elevation, cam.azimuth = 1.0, -EGO_PITCH_DEG, math.degrees(yaw)
    return EGO_FOVY


def layout(m):
    """float64 vector layout for model m: header | qpos | mocap_pos | mocap_quat | eq_active."""
    n_qpos, n_mp, n_mq, n_eq = m.nq, 3 * m.nmocap, 4 * m.nmocap, m.neq
    return dict(nq=n_qpos, nmocap=m.nmocap, neq=n_eq, size=HDR + n_qpos + n_mp + n_mq + n_eq)


class Snapshot:
    """Seqlock'd float64 vector over a numpy buffer (writer: parent; reader: child)."""

    def __init__(self, buf, lay):
        import numpy as np
        self.a = np.ndarray((lay["size"],), dtype=np.float64, buffer=buf)
        self.lay = lay
        o = HDR
        self.qpos = self.a[o:o + lay["nq"]]; o += lay["nq"]
        self.mpos = self.a[o:o + 3 * lay["nmocap"]]; o += 3 * lay["nmocap"]
        self.mquat = self.a[o:o + 4 * lay["nmocap"]]; o += 4 * lay["nmocap"]
        self.eq = self.a[o:o + lay["neq"]]

    # ---- writer
    def write(self, d, base_x, base_y, base_yaw, cam_i, inset, t, tick_id, lift_h=0.0, ghost=None):
        a = self.a
        s = a[0]
        a[0] = s + 1.0 if (int(s) % 2 == 0) else s + 2.0       # odd = write in progress
        a[1:9] = (base_x, base_y, base_yaw, cam_i, float(inset), t, tick_id, lift_h)
        a[GH0:HDR] = ghost if ghost is not None else 0.0        # alpha 0 = no ghost
        self.qpos[:] = d.qpos
        self.mpos[:] = d.mocap_pos.reshape(-1)
        self.mquat[:] = d.mocap_quat.reshape(-1)
        self.eq[:] = d.eq_active
        a[0] = a[0] + 1.0                                       # even = consistent

    # ---- reader
    def read_into(self, d, tries=50):
        """copy a consistent snapshot into MjData d; returns the header (copy) or None if it never became consistent."""
        a = self.a
        for _ in range(tries):
            s0 = a[0]
            if int(s0) % 2:
                time.sleep(0.0002)
                continue
            hdr = a[:HDR].copy()
            d.qpos[:] = self.qpos
            d.mocap_pos[:] = self.mpos.reshape(-1, 3)
            d.mocap_quat[:] = self.mquat.reshape(-1, 4)
            d.eq_active[:] = self.eq.astype(d.eq_active.dtype)
            if a[0] == s0:
                return hdr
        return None


def render_main(shm_name, lay, model_path, ctrl, out, cfg):
    """child entry point (spawned).  ctrl: Connection parent->child (('reload', path) | ('stop',)); out: Connection child->parent."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import cv2
    import numpy as np
    import mujoco
    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(name=shm_name)
    snap = Snapshot(shm.buf, lay)
    ppid = os.getppid()
    W, H, IW, IH, fps, quality = cfg["w"], cfg["h"], cfg["iw"], cfg["ih"], cfg["fps"], cfg["quality"]

    def load(path):
        m = mujoco.MjModel.from_binary_path(path)
        m.vis.global_.offwidth, m.vis.global_.offheight = max(W, 640), max(H, 480)
        d = mujoco.MjData(m)
        return m, d, mujoco.Renderer(m, H, W), mujoco.Renderer(m, IH, IW)

    m, d, r, r_in = load(model_path)
    ghost = GH.Drawer(m)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)

    fovy0 = float(m.vis.global_.fovy)

    def one(rr, name, hdr):
        m.vis.global_.fovy = fovy0
        if name in WRIST:
            rr.update_scene(d, camera=WRIST[name])
        elif name == "ego":
            m.vis.global_.fovy = ego_cam(cam, hdr[1], hdr[2], hdr[3], hdr[8])
            rr.update_scene(d, camera=cam)
            ghost.add(rr.scene, d, GH.GhostFrame.from_vec(hdr[GH0:HDR]), which="arms", z_arrow=GH.Z_ARROW_EGO)
        else:
            x, y, yaw = hdr[1], hdr[2], hdr[3]
            if name == "top":
                cam.lookat[:] = (x, y, 0.5)
                cam.distance, cam.elevation, cam.azimuth = 5.0, -80.0, math.degrees(yaw)
            else:
                cam.lookat[:] = (x + 0.5 * math.cos(yaw), y + 0.5 * math.sin(yaw), 0.8)
                cam.distance, cam.elevation, cam.azimuth = 2.4, -25.0, math.degrees(yaw)
            rr.update_scene(d, camera=cam)
            ghost.add(rr.scene, d, GH.GhostFrame.from_vec(hdr[GH0:HDR]))      # no ghost in the wrist views (it would cover them)
        return rr.render()

    last_tick = -1.0
    next_t = time.perf_counter()
    try:
        while True:
            if os.getppid() != ppid:                      # parent died
                break
            while ctrl.poll():
                msg = ctrl.recv()
                if msg[0] == "stop":
                    return
                if msg[0] == "reload":
                    r.close(); r_in.close()
                    m, d, r, r_in = load(msg[1])
                    ghost = GH.Drawer(m)
                    fovy0 = float(m.vis.global_.fovy)
                    last_tick = -1.0
            now = time.perf_counter()
            if now < next_t:
                time.sleep(min(next_t - now, 0.01))
                continue
            hdr = snap.read_into(d)
            if hdr is None or hdr[7] == last_tick:        # nothing new (sim paused) -> nothing to render
                time.sleep(0.01)
                continue
            t0 = time.perf_counter()
            last_tick = hdr[7]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_camlight(m, d)
            main = CAMERAS[int(hdr[4]) % len(CAMERAS)]
            img = one(r, main, hdr)
            if hdr[5] > 0.5:
                other = "chase" if (main in WRIST or main == "ego") else ("wrist_l" if main != "wrist_l" else "wrist_r")
                ins = one(r_in, other, hdr)
                h, w = ins.shape[:2]
                img = img.copy()
                img[-h - 6:-6, 6:6 + w] = ins
                cv2.rectangle(img, (6, img.shape[0] - h - 6), (6 + w, img.shape[0] - 6), (255, 255, 255), 1)
            ok, buf = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                out.send((1e3 * (time.perf_counter() - t0), buf.tobytes()))
            next_t = max(next_t + 1.0 / fps, time.perf_counter() - 0.5 / fps)
    finally:
        try:
            r.close(); r_in.close()
        except Exception:
            pass
        shm.close()


class RenderLink:
    """parent side: owns the shared memory, the child process and the frame-reader thread."""

    def __init__(self, sim_model, args, on_frame, tmpdir):
        import multiprocessing as mp
        from multiprocessing import shared_memory
        import threading
        self.ctx = mp.get_context("spawn")
        self.tmpdir = tmpdir
        self.on_frame = on_frame
        self.lay = layout(sim_model)
        self.shm = shared_memory.SharedMemory(create=True, size=8 * self.lay["size"])
        self.snap = Snapshot(self.shm.buf, self.lay)
        self.snap.a[:] = 0.0
        self.tick_id = 0
        self.cfg = dict(w=args.width, h=args.height, iw=args.inset_w, ih=args.inset_h, fps=args.fps, quality=args.quality)
        self.path = self._save(sim_model)
        self.ctrl_p, ctrl_c = self.ctx.Pipe()
        self.out_r, out_w = self.ctx.Pipe(duplex=False)
        self.proc = self.ctx.Process(target=render_main, args=(self.shm.name, self.lay, self.path, ctrl_c, out_w, self.cfg),
                                     daemon=True, name="render")
        self.proc.start()
        ctrl_c.close(); out_w.close()
        self._stop = False
        self.reader = threading.Thread(target=self._read, daemon=True, name="render-reader")
        self.reader.start()

    def _save(self, m):
        import mujoco
        m.vis.global_.offwidth, m.vis.global_.offheight = max(self.cfg["w"], 640), max(self.cfg["h"], 480)
        path = os.path.join(self.tmpdir, f"render_model_{os.getpid()}_{time.time_ns()}.mjb")
        mujoco.mj_saveModel(m, path, None)
        return path

    def _read(self):
        while not self._stop:
            try:
                ms, jpg = self.out_r.recv()
            except (EOFError, OSError):
                break
            self.on_frame(ms, jpg)

    def alive(self):
        return self.proc.is_alive()

    def reload(self, sim_model):
        """a new model (episode reset).  Same nq / nmocap / neq is required (same scene+task)."""
        assert layout(sim_model) == self.lay, "reset changed the state layout"
        old = self.path
        self.path = self._save(sim_model)
        self.ctrl_p.send(("reload", self.path))
        self._rm_later = old                     # removed at close(): the child may still be mid-load

    def publish(self, sim, cam_i, inset, ghost=None):
        self.tick_id += 1
        self.snap.write(sim.d, sim.base.x, sim.base.y, sim.base.yaw, cam_i, inset, sim.t, self.tick_id, sim.lift_actual(), ghost)

    def close(self):
        self._stop = True
        try:
            self.ctrl_p.send(("stop",))
        except Exception:
            pass
        self.proc.join(timeout=3)
        if self.proc.is_alive():
            self.proc.terminate()
        try:
            self.shm.close(); self.shm.unlink()
        except Exception:
            pass
        for p in (self.path, getattr(self, "_rm_later", None)):
            try:
                if p:
                    os.remove(p)
            except OSError:
                pass
