"""FoundationPose model-free 6D pose backend (client side; numpy + stdlib only, works in the mujoco env).

    from world_model.v2.house.foundation.pose_fp import Client
    c = Client(sizes=["1280x720"], gpu=6); c.warmup()                         # spawns pose_fp_server.py in the `foundation` env; blocks until READY
    info = c.onboard("cup", rgbs, depths, masks, cam_in_obj, K)                # model-free: reference RGB-D views + masks + camera poses -> neural object field -> mesh
    pose, score, ms = c.register("cup", rgb, depth, mask, K)                   # once (needs a first-frame mask)
    pose, score, ms = c.track("cup", rgb, depth, K)                            # per frame; pose = ob_in_cam (4x4, OpenCV camera), mesh frame = onboarding frame

Onboarding inputs  rgbs (N,H,W,3) u8, depths (N,H,W) f32 m (0 = invalid), masks (N,H,W) u8/bool -- WHICH masks: see masks.floor_gate_mask (depth + known floor plane +
placement seed; non-privileged).  A rendered segmentation is privileged and must only be used to SCORE masks.  cam_in_obj (N,4,4): OpenCV camera pose in the object /
onboarding frame (here: odometry+FK camera poses in the world frame at scan time, which becomes the object frame).
Same IPC + READY pattern as stereo_fs.py (Unix socket on a file path + shared memory; no TCP).  GPU restricted to 5 or 6.
"""
from __future__ import annotations

import atexit
import json
import os
import socket
import struct
import subprocess
import time
import uuid
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FOUNDATION_PY = "data/envs/foundation/bin/python"
RUN_DIR = Path(os.environ.get("CLAUDE_JOB_DIR", "data/jobs")) / "tmp" / "foundation"
SCRATCH = Path("data/tmp_foundation")


def _layout(W, H):
    n = W * H
    return dict(rgb=0, depth=3 * n, mask=7 * n, total=8 * n)


def _send(c, o):
    b = json.dumps(o).encode()
    c.sendall(struct.pack("<I", len(b)) + b)


def _recv(c):
    h = b""
    while len(h) < 4:
        x = c.recv(4 - len(h))
        if not x:
            return None
        h += x
    n = struct.unpack("<I", h)[0]
    b = b""
    while len(b) < n:
        x = c.recv(n - len(b))
        if not x:
            return None
        b += x
    return json.loads(b.decode())


class Client:
    name = "foundationpose"

    def __init__(self, sizes=("1280x720",), gpu=6, maxwh=(1280, 720), warmup_iters=6, python=FOUNDATION_PY, log=None):
        assert int(gpu) in (5, 6), "FoundationPose is restricted to GPUs 5 and 6"
        self.maxW, self.maxH = maxwh
        self.lay = _layout(self.maxW, self.maxH)
        self.shm = shared_memory.SharedMemory(create=True, size=self.lay["total"], name="fp_" + uuid.uuid4().hex[:10])
        self.buf = np.ndarray((self.lay["total"],), np.uint8, buffer=self.shm.buf)
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.sock_path = str(RUN_DIR / f"fp_{os.getpid()}_{uuid.uuid4().hex[:6]}.sock")
        self.log_path = log or str(RUN_DIR / f"fp_server_{os.getpid()}.log")
        E = os.path.dirname(os.path.dirname(python))
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.update(CUDA_VISIBLE_DEVICES=str(int(gpu)), PYTHONUNBUFFERED="1", TMPDIR=str(SCRATCH), PYOPENGL_PLATFORM="egl", CC="/usr/bin/gcc", CXX="/usr/bin/g++",
                   CUDA_HOME="data/cuda_home", TORCH_CUDA_ARCH_LIST="7.5", TORCH_EXTENSIONS_DIR="data/torch_ext",
                   PATH=f"/usr/bin:/bin:{E}/bin:data/cuda_home/bin")
        cmd = [python, "-W", "ignore", str(HERE / "pose_fp_server.py"), "--sock", self.sock_path, "--shm", self.shm.name, "--maxwh", f"{self.maxW}x{self.maxH}",
               "--sizes", ",".join(sizes), "--warmup", str(warmup_iters)]
        self._log = open(self.log_path, "w")
        self.proc = subprocess.Popen(cmd, env=env, stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True, cwd=str(SCRATCH))
        self.conn = None
        self.ready = False
        self.warm = None
        self.last_info = None
        atexit.register(self.close)

    def warmup(self, timeout=1200.0):
        """block until READY (weights + nvdiffrast + warp kernels loaded, dummy register + N tracks done at the operating sizes)."""
        if self.ready:
            return self.warm
        t0 = time.time()
        while self.conn is None:
            if self.proc.poll() is not None:
                raise RuntimeError(f"FoundationPose server died, see {self.log_path}")
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
            time.sleep(0.3)
        self.conn.settimeout(timeout)
        m = _recv(self.conn)
        if not m or not m.get("ready"):
            raise RuntimeError(f"bad READY: {m}")
        self.conn.settimeout(3600)
        self.warm, self.server_pid, self.ready, self.t_ready = m["warmup"], m.get("pid"), True, time.time() - t0
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

    def _call(self, msg):
        if not self.ready:
            raise RuntimeError("call warmup() first: operation starts only after READY")
        _send(self.conn, msg)
        r = _recv(self.conn)
        if r is None or not r.get("ok"):
            raise RuntimeError(f"server error: {r}")
        return r

    # ------------------------------------------------------------------ onboarding
    def onboard(self, name, rgbs, depths, masks, cam_in_obj, K, n_step=None, save_mesh=None):
        """model-free onboarding -> object model.  Returns dict(onboard_s (wall), mesh_s, estimator_s, extents, bounds, n_vert)."""
        t0 = time.perf_counter()
        d = SCRATCH / f"onboard_{name}_{uuid.uuid4().hex[:6]}.npz"
        depths = np.asarray(depths, np.float32).copy()
        masks = np.asarray(masks).astype(np.uint8)
        masks = np.where(masks > 0, 255, 0).astype(np.uint8)
        depths[masks == 0] = 0                                   # depth outside the mask is ignored by the NOF
        np.savez(d, rgb=np.asarray(rgbs, np.uint8), depth=depths, mask=masks, cam_in_obj=np.asarray(cam_in_obj, float), K=np.asarray(K, float))
        r = self._call({"op": "onboard", "name": name, "npz": str(d), "n_step": n_step, "save_mesh": save_mesh})
        r["onboard_s"] = time.perf_counter() - t0
        try:
            d.unlink()
        except OSError:
            pass
        return r

    def load_mesh(self, name, path):
        return self._call({"op": "load_mesh", "name": name, "path": str(path)})

    # ------------------------------------------------------------------ inference
    def _put(self, rgb, depth, mask=None):
        H, W = rgb.shape[:2]
        n = H * W
        assert W <= self.maxW and H <= self.maxH
        L = self.lay
        np.frombuffer(self.buf, np.uint8, 3 * n, L["rgb"])[:] = np.ascontiguousarray(rgb, np.uint8).ravel()
        np.frombuffer(self.buf, np.float32, n, L["depth"])[:] = np.nan_to_num(np.ascontiguousarray(depth, np.float32), nan=0.0).ravel()
        if mask is not None:
            np.frombuffer(self.buf, np.uint8, n, L["mask"])[:] = np.ascontiguousarray(mask).astype(np.uint8).ravel()
        return W, H

    def register(self, name, rgb, depth, mask, K, iters=5, up_cam=None):
        """-> (ob_in_cam 4x4, score, ms).  up_cam: optional gravity-up direction in the camera frame (3,): resolves the flipped hypothesis of thin symmetric objects."""
        t0 = time.perf_counter()
        W, H = self._put(rgb, depth, mask)
        r = self._call({"op": "register", "name": name, "W": W, "H": H, "K": np.asarray(K, float).reshape(-1).tolist(), "iters": iters,
                        "up_cam": None if up_cam is None else np.asarray(up_cam, float).tolist()})
        ms = 1e3 * (time.perf_counter() - t0)
        self.last_info = dict(r["ms"], client_total=ms)
        return np.array(r["pose"]).reshape(4, 4), r["score"], ms

    def init_from_prior(self, name, rgb, depth, K, prior_pose, iters=5):
        """seed the tracker with a pose prior (4x4 ob_in_cam, e.g. arm forward kinematics) and refine it on this frame -> (pose, score, ms).  No mask / global search."""
        t0 = time.perf_counter()
        W, H = self._put(rgb, depth)
        r = self._call({"op": "init", "name": name, "W": W, "H": H, "K": np.asarray(K, float).reshape(-1).tolist(), "iters": iters,
                        "prior": np.asarray(prior_pose, float).reshape(-1).tolist()})
        ms = 1e3 * (time.perf_counter() - t0)
        self.last_info = dict(r["ms"], client_total=ms)
        return np.array(r["pose"]).reshape(4, 4), r["score"], ms

    def track(self, name, rgb, depth, K, iters=2, score=False):
        """-> (ob_in_cam 4x4, score (nan unless score=True), ms)"""
        t0 = time.perf_counter()
        W, H = self._put(rgb, depth)
        r = self._call({"op": "track", "name": name, "W": W, "H": H, "K": np.asarray(K, float).reshape(-1).tolist(), "iters": iters, "score": bool(score)})
        ms = 1e3 * (time.perf_counter() - t0)
        self.last_info = dict(r["ms"], client_total=ms)
        return np.array(r["pose"]).reshape(4, 4), r["score"], ms

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
