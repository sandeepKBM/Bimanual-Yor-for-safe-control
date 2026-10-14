"""stereo/link.py -- the stereo sensor in a SEPARATE PROCESS fed from a state snapshot (the pattern of teleop/web/render_proc.py).  SIMULATION ONLY.

  parent (sim loop)   StereoLink.publish(sim)   -> memcpy of qpos / mocap / eq_active + sim time + tick into a seqlock'd shared-memory vector (never blocks,
                                                   never waits for the child)
  child               loads the twin model (.mjb), builds the backend, WARMS UP, signals READY, then whenever the snapshot's SIM time has advanced by
                      1/rate_hz since its last capture: copy the latest consistent snapshot -> render pair -> backend depth -> Frame -> pipe
  parent reader       a thread stores the newest Frame (+ a bounded history); latest() / frames_since(tick) / wait_frame(tick, timeout)

Start-up cost is paid BEFORE operation: StereoLink.start() blocks until the child reports READY (model load, EGL context, backend warm-up, 2 dry captures);
`ready_s` records how long that took.  The child exits when the parent dies or on 'stop'.  GPU for rendering / a GPU backend: CUDA_VISIBLE_DEVICES 5 or 6.
"""
from __future__ import annotations

import os
import time
from collections import deque

import numpy as np

HDR = 8              # [seq, t_sim, tick, wall_pub, 0, 0, 0, 0]


def layout(m):
    return dict(nq=m.nq, nmocap=m.nmocap, neq=m.neq, size=HDR + m.nq + 3 * m.nmocap + 4 * m.nmocap + m.neq)


class Snap:
    def __init__(self, buf, lay):
        self.a = np.ndarray((lay["size"],), dtype=np.float64, buffer=buf)
        o = HDR
        self.qpos = self.a[o:o + lay["nq"]]; o += lay["nq"]
        self.mpos = self.a[o:o + 3 * lay["nmocap"]]; o += 3 * lay["nmocap"]
        self.mquat = self.a[o:o + 4 * lay["nmocap"]]; o += 4 * lay["nmocap"]
        self.eq = self.a[o:o + lay["neq"]]

    def write(self, d, t, tick):
        a = self.a
        s = a[0]
        a[0] = s + 1.0 if (int(s) % 2 == 0) else s + 2.0
        a[1:4] = (t, tick, time.time())
        self.qpos[:] = d.qpos
        self.mpos[:] = d.mocap_pos.reshape(-1)
        self.mquat[:] = d.mocap_quat.reshape(-1)
        self.eq[:] = d.eq_active
        a[0] = a[0] + 1.0

    def read(self, tries=50):
        a = self.a
        for _ in range(tries):
            s0 = a[0]
            if int(s0) % 2:
                time.sleep(0.0002)
                continue
            hdr = a[:HDR].copy()
            qp, mp, mq, eq = self.qpos.copy(), self.mpos.reshape(-1, 3).copy(), self.mquat.reshape(-1, 4).copy(), self.eq.copy()
            if a[0] == s0:
                return hdr, qp, mp, mq, eq
        return None


def _child(shm_name, lay, model_path, cfg_dict, ctrl, out, root, backend_kw):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from multiprocessing import shared_memory
    from .cfg import StereoCfg
    from .sensor import StereoSensor
    from .backends import make_backend
    shm = shared_memory.SharedMemory(name=shm_name)
    snap = Snap(shm.buf, lay)
    cfg = StereoCfg.from_dict(cfg_dict)
    m = mujoco.MjModel.from_binary_path(model_path)
    sens = StereoSensor(m, cfg, root=root, backend=make_backend(cfg, **(backend_kw or {})))
    ppid = os.getppid()
    t0 = time.perf_counter()
    got = None
    while got is None and time.perf_counter() - t0 < 30:        # the warm-up capture needs a real state: wait for the first publish
        got = snap.read()
        if got is None or got[0][0] == 0:            # seq 0 = never published
            got = None
            time.sleep(0.01)
    if got is not None:
        sens.load(got[1], got[2], got[3], got[4])
    sens.warmup()
    out.send(("ready", 1e3 * (time.perf_counter() - t0)))      # every startup overhead paid
    period = 1.0 / cfg.rate_hz
    last_t = -1e9
    last_tick = -1.0
    try:
        while True:
            if os.getppid() != ppid:
                break
            while ctrl.poll():
                if ctrl.recv()[0] == "stop":
                    return
            got = snap.read()
            if got is None or got[0][2] == last_tick or got[0][1] - last_t < period - 1e-9:
                time.sleep(0.002)
                continue
            hdr, qp, mp, mq, eq = got
            last_tick, last_t = hdr[2], hdr[1]
            t_start = time.time()
            sens.load(qp, mp, mq, eq)
            fr = sens.capture(int(hdr[2]), float(hdr[1]))
            fr["wall_pub"], fr["wall_start"], fr["wall_done"] = float(hdr[3]), t_start, time.time()
            out.send(("frame", fr))
    finally:
        sens.close()
        shm.close()


class StereoLink:
    def __init__(self, sim_model, twin_path, cfg, root="yor_root", history=64, backend_kw=None, start=True, sim=None):
        import multiprocessing as mp
        from multiprocessing import shared_memory
        import threading
        self.ctx = mp.get_context("spawn")
        self.cfg = cfg
        self.lay = layout(sim_model)
        self.shm = shared_memory.SharedMemory(create=True, size=8 * self.lay["size"])
        self.snap = Snap(self.shm.buf, self.lay)
        self.snap.a[:] = 0.0
        self.hist = deque(maxlen=history)
        self.ready_ms = None
        self.ready = threading.Event()
        self._stop = False
        self.ctrl_p, ctrl_c = self.ctx.Pipe()
        self.out_r, out_w = self.ctx.Pipe(duplex=False)
        self.proc = self.ctx.Process(target=_child, args=(self.shm.name, self.lay, str(twin_path), cfg.asdict(), ctrl_c, out_w, root, backend_kw),
                                     daemon=True, name="stereo")
        self.reader = threading.Thread(target=self._read, daemon=True, name="stereo-reader")
        self.sim_d = sim.d if sim is not None else None
        if start:
            self.start(sim)

    def start(self, sim, timeout=300.0):
        """spawn the child, feed it one real snapshot, BLOCK until READY (all start-up overhead paid)."""
        self.proc.start()
        self.reader.start()
        if sim is not None:
            self.publish_state(sim.d, sim.t, sim.k)
        if not self.ready.wait(timeout):
            raise RuntimeError("stereo child did not become READY")
        return self.ready_ms

    def _read(self):
        while not self._stop:
            try:
                kind, payload = self.out_r.recv()
            except (EOFError, OSError):
                break
            if kind == "ready":
                self.ready_ms = payload
                self.ready.set()
            else:
                payload["wall_recv"] = time.time()
                self.hist.append(payload)

    def publish_state(self, d, t, tick):
        self.snap.write(d, t, tick)

    def publish(self, sim):
        self.snap.write(sim.d, sim.t, sim.k)

    def latest(self):
        return self.hist[-1] if self.hist else None

    def frames_since(self, tick):
        return [f for f in self.hist if f["tick"] > tick]

    def wait_frame(self, min_tick, timeout=30.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            for f in reversed(self.hist):
                if f["tick"] >= min_tick:
                    return f
            time.sleep(0.002)
        return None

    def alive(self):
        return self.proc.is_alive()

    def close(self):
        self._stop = True
        try:
            self.ctrl_p.send(("stop",))
        except Exception:
            pass
        self.proc.join(timeout=5)
        if self.proc.is_alive():
            self.proc.terminate()
        try:
            self.shm.close(); self.shm.unlink()
        except Exception:
            pass
