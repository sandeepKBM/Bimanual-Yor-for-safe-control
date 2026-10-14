"""FoundationStereo depth backend (client side).  Imports only numpy + stdlib, so it works in the mujoco env.

    from world_model.v2.house.foundation.stereo_fs import Client
    c = Client(W=640, H=360, gpu=5, engine="trt", iters=12)     # spawns stereo_fs_server.py in the `foundation` env
    c.warmup()                                                    # blocks until the server signals READY (weights loaded + N dummy inferences)
    depth, valid, ms = c.depth(left_rgb_u8, right_rgb_u8, K, baseline)   # depth (H,W) float32 [m], NaN where invalid

Interface = house/stereo/backends.FoundationBackend (Client(**kw), .warmup(), .ready, .depth(left,right,K,baseline) -> (depth_m, valid, ms)).
`K` is the dict from StereoCfg.K() ({f,cx,cy,W,H,B}) or a 3x3 matrix; the pair must be rectified (parallel axes), left image first.
`ms` = client wall time of the whole call (copy-in + IPC + GPU + copy-out); Client.last_info["ms"] has the server's breakdown.

IPC: a Unix-domain socket bound to a file path (no TCP) for 4-byte-length-prefixed JSON control + one POSIX shared-memory block for the pixels.
The server (stereo_fs_server.py) runs in /data/local/ss5772/conda_envs/foundation, pinned to ONE GPU (default 5; use 5 or 6 only).
Offline: nothing here opens a network socket; the server imports no robot code.
"""
from __future__ import annotations

import atexit
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FOUNDATION_PY = "/data/local/ss5772/conda_envs/foundation/bin/python"
RUN_DIR = Path(os.environ.get("CLAUDE_JOB_DIR", "/common/home/ss5772/.claude/jobs/b9577455")) / "tmp" / "foundation"


def _layout(maxW, maxH):
    n = maxW * maxH
    o_r = 3 * n
    o_d = o_r + 3 * n
    o_v = o_d + 4 * n
    return dict(left=0, right=o_r, depth=o_d, valid=o_v, total=o_v + n)


def _send(c, obj):
    b = json.dumps(obj).encode()
    c.sendall(struct.pack("<I", len(b)) + b)


def _recv(c):
    h = b""
    while len(h) < 4:
        x = c.recv(4 - len(h))
        if not x:
            return None
        h += x
    n = struct.unpack("<I", h)[0]
    buf = b""
    while len(buf) < n:
        x = c.recv(n - len(buf))
        if not x:
            return None
        buf += x
    return json.loads(buf.decode())


class Client:
    name = "foundation"

    def __init__(self, W=640, H=360, sizes=None, gpu=5, engine="trt", iters=12, ckpt="23-51-11", vit="vitl", warmup_iters=10, max_disp=0,
                 maxwh=(1280, 720), log=None, python=FOUNDATION_PY):
        assert int(gpu) in (5, 6), "FoundationStereo is restricted to GPUs 5 and 6"
        self.W, self.H = int(W), int(H)
        self.sizes = sizes or [f"{self.W}x{self.H}"]
        self.maxW, self.maxH = maxwh
        self.lay = _layout(self.maxW, self.maxH)
        self.shm = shared_memory.SharedMemory(create=True, size=self.lay["total"], name="fs_" + uuid.uuid4().hex[:10])
        self.buf = np.ndarray((self.lay["total"],), np.uint8, buffer=self.shm.buf)
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        self.sock_path = str(RUN_DIR / f"fs_{os.getpid()}_{uuid.uuid4().hex[:6]}.sock")
        self.log_path = log or str(RUN_DIR / f"fs_server_{os.getpid()}.log")
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=str(int(gpu)), XFORMERS_DISABLED="1", PYTHONUNBUFFERED="1", TMPDIR=str(RUN_DIR))
        env.pop("PYTHONPATH", None)
        cmd = [python, "-W", "ignore", str(HERE / "stereo_fs_server.py"), "--sock", self.sock_path, "--shm", self.shm.name,
               "--maxwh", f"{self.maxW}x{self.maxH}", "--sizes", ",".join(self.sizes), "--engine", engine, "--iters", str(iters),
               "--ckpt", ckpt, "--vit", vit, "--warmup", str(warmup_iters), "--max-disp", str(max_disp)]
        self._log = open(self.log_path, "w")
        self.proc = subprocess.Popen(cmd, env=env, stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True)
        self.conn = None
        self.ready = False
        self.warm = None
        self.last_info = None
        self.ms_hist = []
        atexit.register(self.close)

    # ------------------------------------------------------------------ lifecycle
    def warmup(self, timeout=900.0):
        """block until the server is READY (weights/engine loaded, N dummy inferences done).  Idempotent.  Operation must not start before this returns."""
        if self.ready:
            return self.warm
        t0 = time.time()
        while self.conn is None:
            if self.proc.poll() is not None:
                raise RuntimeError(f"FoundationStereo server died, see {self.log_path}")
            if os.path.exists(self.sock_path):
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    s.connect(self.sock_path)
                    self.conn = s
                    break
                except OSError:
                    s.close()
            if time.time() - t0 > timeout:
                raise TimeoutError("server socket never appeared")
            time.sleep(0.2)
        self.conn.settimeout(timeout)
        msg = _recv(self.conn)
        if not msg or not msg.get("ready"):
            raise RuntimeError(f"bad READY message {msg}")
        self.conn.settimeout(120)
        self.warm = msg["warmup"]
        self.server_pid = msg.get("pid")
        self.ready = True
        self.t_ready = time.time() - t0
        return self.warm

    def close(self):
        try:
            if self.conn is not None:
                _send(self.conn, {"op": "quit"})
                self.conn.close()
                self.conn = None
        except Exception:
            pass
        try:
            if self.proc.poll() is None:
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        except Exception:
            pass
        try:
            self.buf = None
            self.shm.close()
            self.shm.unlink()
        except Exception:
            pass

    # ------------------------------------------------------------------ inference
    def depth(self, left_rgb, right_rgb, K, baseline):
        """-> (depth_m (H,W) float32 NaN=invalid, valid (H,W) bool, ms float)"""
        if not self.ready:
            raise RuntimeError("call warmup() first: operation starts only after READY")
        t0 = time.perf_counter()
        H, W = left_rgb.shape[:2]
        n = H * W
        assert W <= self.maxW and H <= self.maxH and left_rgb.dtype == np.uint8 and left_rgb.shape == right_rgb.shape
        f = float(K["f"]) if isinstance(K, dict) else float(np.asarray(K)[0, 0])
        lay = self.lay
        np.frombuffer(self.buf, np.uint8, 3 * n, lay["left"])[:] = np.ascontiguousarray(left_rgb).ravel()
        np.frombuffer(self.buf, np.uint8, 3 * n, lay["right"])[:] = np.ascontiguousarray(right_rgb).ravel()
        _send(self.conn, {"op": "depth", "W": int(W), "H": int(H), "f": f, "baseline": float(baseline)})
        r = _recv(self.conn)
        if r is None or not r.get("ok"):
            raise RuntimeError(f"server error: {r}")
        z = np.frombuffer(self.buf, np.float32, n, lay["depth"]).reshape(H, W).copy()
        ok = np.frombuffer(self.buf, np.uint8, n, lay["valid"]).reshape(H, W).astype(bool)
        ms = 1e3 * (time.perf_counter() - t0)
        self.last_info = dict(r["ms"], client_total=ms)
        self.ms_hist.append(ms)
        return z, ok, ms

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


_default = None


def depth(left_rgb, right_rgb, K, baseline):
    """module-level convenience: lazily starts a default 640x360 TRT client (blocks on READY at the first call)."""
    global _default
    if _default is None:
        _default = Client()
        _default.warmup()
    return _default.depth(left_rgb, right_rgb, K, baseline)
