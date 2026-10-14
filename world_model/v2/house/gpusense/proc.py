"""gpusense/proc.py -- the sensing PROCESS (SIMULATION ONLY): a spawned child owns the EGL contexts, the CUDA context, the GPU noise model and
the GPU mapper; the parent (the sim thread) never imports torch and never blocks on rendering.

Pattern of house/teleop/web/render_proc.py: the model reaches the child as a .mjb; the state the renderer needs (qpos, mocap_pos/quat, eq_active)
travels WITH each request (a few KB), so the frame is a pure function of (state at the capture tick, camera pose, seed, tick) -- identical whether
the result is collected immediately or `latency_ticks` later.  The sensor model already has a 3-tick (60 ms) delivery delay: submit at the
capture tick, collect at the delivery tick, and the render time hides inside the modelled latency (async mode).

  parent                                         child
  SenseProc(model, cam_cfgs, map=...)   ------>  SenseCore(...).warmup(), GpuMapper(...).warmup()   -> ("ready", timings)
  ticket = submit(d, jobs)              ------>  set_state; per job: GPU depth + ids -> noise -> compaction  -> ("res", ticket, [frames])
  frames = collect(ticket)              <------
  submit_map(d, true_pose, est_pose)    ------>  GpuMapper.update; (publish) occ -> EDT -> gradient -> shared memory -> ("map", ticket, stats)
  latest_grid()  -> OccGrid (numpy, gradient cache filled; GM.grid_lookup_grad works unchanged)
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np

# env the child must see (the parent's own is inherited by spawn; these make EGL/threads explicit)
CHILD_ENV = {"MUJOCO_GL": "egl", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"}


def _state_of(d):
    return (d.qpos.copy(), d.mocap_pos.copy(), d.mocap_quat.copy(), d.eq_active.copy())


def _child_main(model_path, cam_cfgs, map_spec, shm_name, shm_shape, conn, env):
    os.environ.update(env)
    t00 = time.perf_counter()
    import torch                                            # noqa: E402  (child only)
    from multiprocessing import shared_memory
    from . import core as C

    core = C.SenseCore(model_path, cam_cfgs, "cuda")
    t_ctor = time.perf_counter() - t00
    core.warmup()
    t_core = time.perf_counter() - t00
    mapper, shm, views = None, None, None
    if map_spec is not None:
        from . import mapping as GM
        mapper = GM.GpuMapper(core, map_spec["cfg"], map_spec["x0"], map_spec["y0"], map_spec["ny"], map_spec["nx"], map_spec["cam"], seed=map_spec["seed"])
        shm = shared_memory.SharedMemory(name=shm_name)
        ny, nx = shm_shape
        buf = shm.buf
        hdr = np.ndarray((4,), np.float64, buffer=buf, offset=0)                # [seq, version, ...]
        o = 32
        occ = np.ndarray((ny, nx), np.uint8, buffer=buf, offset=o); o += ny * nx
        o = (o + 7) // 8 * 8
        edt = np.ndarray((ny, nx), np.float32, buffer=buf, offset=o); o += 4 * ny * nx
        gx = np.ndarray((ny, nx), np.float32, buffer=buf, offset=o); o += 4 * ny * nx
        gy = np.ndarray((ny, nx), np.float32, buffer=buf, offset=o)
        views = (hdr, occ, edt, gx, gy)
        # warm the mapper chain: one dummy update with the true poses of a no-op frame, EDT kernel compile, then reset the log-odds
        gm0 = mapper.L.clone()
        pose = np.array(map_spec.get("warm_pose", (0.0, 0.0, 0.0)))
        for _ in range(2):
            mapper.update(pose, pose)
            g = mapper.grid()
            g.edt.sum().item()
        mapper.L.copy_(gm0)
        mapper.n_frames = 0
    torch.cuda.synchronize()
    t_all = time.perf_counter() - t00
    conn.send(("ready", dict(ctor_s=t_ctor, core_warm_s=t_core - t_ctor, total_s=t_all, pid=os.getpid())))
    ppid = os.getppid()
    version = 0
    while True:
        if not conn.poll(0.5):
            if os.getppid() != ppid:
                break
            continue
        msg = conn.recv()
        k = msg[0]
        if k == "stop":
            break
        if k == "jobs":
            _, ticket, state, jobs = msg
            t0 = time.perf_counter()
            out = core.process(state, jobs)
            conn.send(("res", ticket, out, 1e3 * (time.perf_counter() - t0)))
        elif k == "map":
            _, ticket, state, true_pose, est_pose, publish = msg
            t0 = time.perf_counter()
            core.set_state(*state)
            mapper.update(true_pose, est_pose)
            t_upd = time.perf_counter()
            stats = dict(frames=mapper.n_frames)
            if publish:
                g = mapper.grid()
                occ_h = g.occ.to(torch.uint8).cpu().numpy()
                e_h, gx_h, gy_h = g.edt.cpu().numpy(), g.gx.cpu().numpy(), g.gy.cpu().numpy()
                hdr[0] += 1
                occ[:] = occ_h; edt[:] = e_h; gx[:] = gx_h; gy[:] = gy_h
                version += 1
                hdr[1] = version
                hdr[0] += 1
                stats["version"] = version
            torch.cuda.synchronize()
            stats["update_ms"] = 1e3 * (t_upd - t0)
            stats["total_ms"] = 1e3 * (time.perf_counter() - t0)
            conn.send(("map", ticket, stats))
        elif k == "reset_map":
            mapper.L.zero_()
            mapper.n_frames = 0
            conn.send(("ok", msg[1]))
    core.close()
    if shm is not None:
        shm.close()


class SenseProc:
    """parent-side handle.  `start()` blocks until the child is READY (all contexts created, kernels compiled, first renders done)."""

    def __init__(self, model, cam_cfgs, map_spec=None, tmpdir=None, device="cuda", model_path=None):
        import multiprocessing as mp
        import tempfile
        self.ctx = mp.get_context("spawn")
        jd = os.environ.get("CLAUDE_JOB_DIR")
        tmpdir = tmpdir or (os.path.join(jd, "tmp") if jd and os.path.isdir(os.path.join(jd, "tmp")) else tempfile.gettempdir())
        self.cam_cfgs = list(cam_cfgs)
        self.map_spec = map_spec
        self.own_model = model_path is None
        self.path = self._save(model, tmpdir) if model_path is None else model_path
        self.ready_info = None
        self.shm = None
        self.shape = None
        self._grid_ver = 0
        if map_spec is not None:
            from multiprocessing import shared_memory
            ny, nx = map_spec["ny"], map_spec["nx"]
            self.shape = (ny, nx)
            size = 32 + ((ny * nx + 7) // 8 * 8) + 3 * 4 * ny * nx + 64
            self.shm = shared_memory.SharedMemory(create=True, size=size)
            np.ndarray((4,), np.float64, buffer=self.shm.buf)[:] = 0.0
        self.conn, child = self.ctx.Pipe()
        self.proc = self.ctx.Process(target=_child_main, args=(self.path, self.cam_cfgs, map_spec, self.shm.name if self.shm else None, self.shape, child, dict(CHILD_ENV)),
                                     daemon=True, name="gpusense")
        self.ticket = 0
        self._res = {}
        self._cv = threading.Condition()
        self._closed = True
        self.t_submit = {}
        self.stats = dict(stall_ms=[], job_ms=[], rtt_ms=[])

    @staticmethod
    def default_tmp():
        import tempfile
        jd = os.environ.get("CLAUDE_JOB_DIR")
        return os.path.join(jd, "tmp") if jd and os.path.isdir(os.path.join(jd, "tmp")) else tempfile.gettempdir()

    @staticmethod
    def _save(m, tmpdir):
        import mujoco
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), 640)
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), 480)
        path = os.path.join(tmpdir, f"gpusense_model_{os.getpid()}_{time.time_ns()}.mjb")
        mujoco.mj_saveModel(m, path, None)
        return path

    def start_async(self):
        self._t0 = time.perf_counter()
        self.proc.start()

    def wait_ready(self, timeout=300.0):
        if not self.conn.poll(timeout):
            raise RuntimeError("gpusense child did not become READY")
        msg = self.conn.recv()
        assert msg[0] == "ready", msg
        self.ready_info = dict(msg[1], wall_s=time.perf_counter() - self._t0)
        self._closed = False
        self._reader = threading.Thread(target=self._read_loop, daemon=True, name="gpusense-reader")     # unpickling overlaps the sim's native code (GIL released)
        self._reader.start()
        return self.ready_info

    def _read_loop(self):
        while not self._closed:
            try:
                msg = self.conn.recv()
            except (EOFError, OSError):
                break
            with self._cv:
                self._res[msg[1]] = msg
                self._cv.notify_all()

    def start(self, timeout=300.0):
        self.start_async()
        return self.wait_ready(timeout)

    # ---- jobs
    def submit(self, d, jobs):
        self.ticket += 1
        self.t_submit[self.ticket] = time.perf_counter()
        self.conn.send(("jobs", self.ticket, _state_of(d), jobs))
        return self.ticket

    def collect(self, ticket, block=True):
        """-> list of per-job result dicts (None if not ready and block=False)."""
        t0 = time.perf_counter()
        with self._cv:
            while ticket not in self._res:
                if not block:
                    return None
                if not self._cv.wait(timeout=5.0) and not self.proc.is_alive():
                    raise RuntimeError("gpusense child died")
            msg = self._res.pop(ticket)
        now = time.perf_counter()
        self.stats["stall_ms"].append(1e3 * (now - t0))
        self.stats["rtt_ms"].append(1e3 * (now - self.t_submit.pop(ticket, now)))
        if msg[0] == "res":
            self.stats["job_ms"].append(msg[3])
        return msg[2]

    # ---- map
    def submit_map(self, d, true_pose, est_pose, publish=True):
        self.ticket += 1
        self.t_submit[self.ticket] = time.perf_counter()
        self.conn.send(("map", self.ticket, _state_of(d), tuple(true_pose), tuple(est_pose), bool(publish)))
        return self.ticket

    def latest_grid(self):
        """-> teleop.scene.OccGrid (gradient cache filled) of the newest published map, or None if nothing new since the last call."""
        if self.shm is None:
            return None
        ny, nx = self.shape
        buf = self.shm.buf
        hdr = np.ndarray((4,), np.float64, buffer=buf, offset=0)
        for _ in range(50):
            s0 = hdr[0]
            ver = int(hdr[1])
            if int(s0) % 2 or ver == self._grid_ver:
                if ver == self._grid_ver:
                    return None
                time.sleep(0.0002)
                continue
            o = 32
            occ = np.ndarray((ny, nx), np.uint8, buffer=buf, offset=o).astype(bool); o += ny * nx
            o = (o + 7) // 8 * 8
            edt = np.ndarray((ny, nx), np.float32, buffer=buf, offset=o).astype(np.float64); o += 4 * ny * nx
            gx = np.ndarray((ny, nx), np.float32, buffer=buf, offset=o).astype(np.float64); o += 4 * ny * nx
            gy = np.ndarray((ny, nx), np.float32, buffer=buf, offset=o).astype(np.float64)
            if hdr[0] == s0:
                self._grid_ver = ver
                from world_model.v2.house.teleop.scene import OccGrid
                ms = self.map_spec
                g = OccGrid(ms["cfg"].res, ms["x0"], ms["y0"], occ, edt)
                g._grad = (gx, gy)
                return g
        return None

    def close(self):
        try:
            self.conn.send(("stop",))
        except Exception:
            pass
        self._closed = True
        self.proc.join(timeout=5)
        if self.proc.is_alive():
            self.proc.terminate()
        if self.shm is not None:
            try:
                self.shm.close()
                self.shm.unlink()
            except Exception:
                pass
        if self.own_model:
            try:
                os.remove(self.path)
            except OSError:
                pass
