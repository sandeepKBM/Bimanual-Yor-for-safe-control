"""FoundationPose model-free 6D pose server.  Runs ONLY in the `foundation` conda env (torch 2.8+cu128, nvdiffrast, warp, mycpp/mycuda built for sm_75).

    CUDA_VISIBLE_DEVICES=6 python pose_fp_server.py --sock S --shm NAME --maxwh 1280x720 --sizes 1280x720 --warmup 6

Transport: Unix-domain socket bound to a FILE (mode 0600) with 4-byte-length-prefixed JSON; frames travel in one multiprocessing.shared_memory block
    [ rgb u8 (H*W*3) | depth f32 (H*W) | mask u8 (H*W) ]            (layout for the max W,H)
Onboarding views (a one-off, large) are passed as an .npz path in the request.
Messages  server->client {"ready":true,"warmup":{..}} ; client->server
   {"op":"onboard","name":..,"npz":path,"n_step":..}     -> NOF mesh (BundleSDF neural object field, model-free) + estimator; returns timings
   {"op":"load_mesh","name":..,"path":..}                  -> skip the NOF, use a mesh file
   {"op":"register","name":..,"W","H","K":[9],"iters":5}  -> {"pose":[16],"score":..,"ms":..}   ob_in_cam (OpenCV camera), mesh = onboarding frame
   {"op":"track","name":..,"W","H","K":[9],"iters":2,"score":bool} -> same, from the previous pose
   {"op":"quit"}
No network, no robot code.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import struct
import sys
import time
import traceback
from multiprocessing import shared_memory

FP = "/data/local/ss5772/models/FoundationPose"
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (FP, FP + "/bundlesdf", FP + "/bundlesdf/mycuda", HERE):
    sys.path.insert(0, p)
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
import trimesh
import yaml

if not hasattr(trimesh.Trimesh, "remove_duplicate_faces"):          # removed in newer trimesh; NOF texture code calls it
    trimesh.Trimesh.remove_duplicate_faces = lambda self: self.update_faces(self.unique_faces())

import nvdiffrast.torch as dr
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
from Utils import erode_depth, bilateral_filter_depth, set_logging_format, set_seed
import nerf_runner as _nr
from dense_octree import DenseOctreeManager

_nr.OctreeManager = DenseOctreeManager        # kaolin replacement, see dense_octree.py
logging.getLogger().setLevel(logging.WARNING)


def layout(W, H):
    n = W * H
    return dict(rgb=0, depth=3 * n, mask=3 * n + 4 * n, total=3 * n + 4 * n + n)


def send(c, o):
    b = json.dumps(o).encode()
    c.sendall(struct.pack("<I", len(b)) + b)


def recv(c):
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


class Core:
    def __init__(self):
        set_logging_format()
        logging.getLogger().setLevel(logging.WARNING)
        set_seed(0)
        t = time.perf_counter()
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.t_weights = time.perf_counter() - t
        self.est = {}

    def make_est(self, mesh):
        tmp = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=self.scorer, refiner=self.refiner, glctx=self.glctx, debug=0,
                             debug_dir="/data/local/ss5772/tmp_foundation/fp_debug")
        try:
            os.unlink(tmp.mesh_path)                  # upstream writes a scratch copy to /tmp/<uuid>.obj and never reads it back here
        except Exception:
            pass
        return tmp

    def nof_mesh(self, npz, n_step=None):
        from bundlesdf.run_nerf import run_neural_object_field
        z = np.load(npz)
        cfg = yaml.safe_load(open(FP + "/bundlesdf/config_ycbv.yml"))
        cfg["use_octree"] = 1
        cfg["save_octree_clouds"] = False
        if n_step:
            cfg["n_step"] = int(n_step)
        sd = f"/data/local/ss5772/tmp_foundation/nof_{os.getpid()}"
        rgbs = list(z["rgb"])
        depths = list(z["depth"])
        masks = list(z["mask"])
        mesh = run_neural_object_field(cfg, z["K"], rgbs, depths, masks, list(z["cam_in_obj"]), save_dir=sd, debug=0)
        return mesh

    def prep(self, rgb, depth, mask):
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return rgb, depth, mask

    def register(self, name, rgb, depth, mask, K, iters, up_cam=None, up_deg=60.0):
        e = self.est[name]
        rgb, depth, mask = self.prep(rgb, depth, mask)
        pose = e.register(K=K, rgb=rgb, depth=depth, ob_mask=mask.astype(bool), iteration=iters)
        if up_cam is not None:
            # thin / symmetric objects (a tray) have a flipped hypothesis that fits the depth as well as the true one.  Gravity (IMU / FK) is known: take the best-scoring
            # hypothesis whose mesh +z axis (the scan-time 'up') points within up_deg of `up_cam` (gravity-up in the camera frame).
            u = np.asarray(up_cam, float) / np.linalg.norm(up_cam)
            P = e.poses.data.cpu().numpy()
            ok = np.where(P[:, :3, 2] @ u > np.cos(np.radians(up_deg)))[0]
            if len(ok):
                i = int(ok[0])                                    # poses are sorted by descending score
                e.pose_last = e.poses[i]
                e.scores = e.scores[i:i + 1] if i else e.scores
                pose = (e.poses[i] @ e.get_tf_to_centered_mesh()).data.cpu().numpy()
        sc = float(torch.as_tensor(e.scores)[0].item()) if hasattr(e, "scores") else float("nan")
        return pose, sc

    def init_from_prior(self, name, rgb, depth, K, iters, prior):
        """start tracking from a pose PRIOR (e.g. forward kinematics of the arms holding the object): seed pose_last and refine with `iters` refiner passes."""
        e = self.est[name]
        c = torch.eye(4, dtype=torch.float, device="cuda")
        c[:3, 3] = torch.as_tensor(e.model_center, device="cuda", dtype=torch.float)       # inverse of get_tf_to_centered_mesh
        e.pose_last = torch.as_tensor(prior, device="cuda", dtype=torch.float) @ c
        return self.track(name, rgb, depth, K, iters, True)

    def track(self, name, rgb, depth, K, iters, want_score):
        e = self.est[name]
        rgb, depth, _ = self.prep(rgb, depth, None)
        pose = e.track_one(rgb=rgb, depth=depth, K=K, iteration=iters)
        sc = float("nan")
        if want_score:
            d = erode_depth(torch.as_tensor(depth, device="cuda", dtype=torch.float), radius=2, device="cuda")
            d = bilateral_filter_depth(d, radius=2, device="cuda")
            s, _ = self.scorer.predict(mesh=e.mesh, rgb=rgb, depth=d, K=K, ob_in_cams=e.pose_last.reshape(1, 4, 4).data.cpu().numpy(), normal_map=None,
                                       mesh_tensors=e.mesh_tensors, glctx=self.glctx, mesh_diameter=e.diameter, get_vis=False)
            sc = float(torch.as_tensor(s).reshape(-1)[0].item())
        return pose, sc


def dummy_scene(W, H, f):
    """a box on a plane rendered analytically (depth) + flat colour: only used to exercise every kernel at the operating resolution during warm-up."""
    K = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1]], np.float64)
    depth = np.full((H, W), 1.2, np.float32)
    cx, cy = W // 2, H // 2
    r = max(24, W // 16)
    depth[cy - r:cy + r, cx - r:cx + r] = 1.0
    rgb = np.full((H, W, 3), 90, np.uint8)
    rgb[cy - r:cy + r, cx - r:cx + r] = (200, 60, 60)
    mask = np.zeros((H, W), bool)
    mask[cy - r:cy + r, cx - r:cx + r] = True
    return rgb, depth, mask, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", required=True)
    ap.add_argument("--shm", required=True)
    ap.add_argument("--maxwh", default="1280x720")
    ap.add_argument("--sizes", default="1280x720")
    ap.add_argument("--warmup", type=int, default=6)
    a = ap.parse_args()
    t_start = time.perf_counter()
    torch.backends.cudnn.benchmark = False
    maxW, maxH = (int(v) for v in a.maxwh.split("x"))
    lay = layout(maxW, maxH)
    shm = shared_memory.SharedMemory(name=a.shm)
    buf = np.ndarray((lay["total"],), np.uint8, buffer=shm.buf)
    if os.path.exists(a.sock):
        os.unlink(a.sock)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(a.sock)
    os.chmod(a.sock, 0o600)
    srv.listen(1)

    core = Core()
    wu = dict(weights_s=round(core.t_weights, 2))
    # warm-up: dummy box mesh, register + N tracks at every operating size (compiles warp / nvdiffrast kernels, fills cudnn plans)
    box = trimesh.creation.box(extents=(0.12, 0.12, 0.12))
    box.visual = trimesh.visual.ColorVisuals(mesh=box, vertex_colors=np.tile(np.array([[200, 60, 60, 255]], np.uint8), (len(box.vertices), 1)))
    core.est["_warm"] = core.make_est(box)
    for s in a.sizes.split(","):
        W, H = (int(v) for v in s.split("x"))
        rgb, depth, mask, K = dummy_scene(W, H, 0.7 * H / 0.7 / 1.0 * 0.714)
        t0 = time.perf_counter()
        core.register("_warm", rgb, depth, mask, K, 5)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ts = []
        for _ in range(a.warmup):
            t = time.perf_counter()
            core.track("_warm", rgb, depth, K, 2, True)
            torch.cuda.synchronize()
            ts.append(1e3 * (time.perf_counter() - t))
        wu[s] = dict(register_ms=round(1e3 * (t1 - t0)), track_ms=[round(v) for v in ts])
        print(f"[fp-server] warm {s}: register {1e3 * (t1 - t0):.0f} ms; track(+score) last {ts[-1]:.0f} ms", flush=True)
    del core.est["_warm"]
    wu["total_s"] = round(time.perf_counter() - t_start, 2)
    wu["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    conn, _ = srv.accept()
    send(conn, {"ready": True, "warmup": wu, "pid": os.getpid()})
    print("[fp-server] READY", flush=True)

    while True:
        msg = recv(conn)
        if msg is None or msg.get("op") == "quit":
            break
        try:
            op = msg["op"]
            if op in ("onboard", "load_mesh"):
                t0 = time.perf_counter()
                if op == "onboard":
                    mesh = core.nof_mesh(msg["npz"], msg.get("n_step"))
                    mpath = msg.get("save_mesh")
                    if mpath:
                        mesh.export(mpath)
                else:
                    mesh = trimesh.load(msg["path"], force="mesh")
                t1 = time.perf_counter()
                core.est[msg["name"]] = core.make_est(mesh)
                t2 = time.perf_counter()
                send(conn, {"ok": True, "mesh_s": t1 - t0, "estimator_s": t2 - t1, "extents": mesh.extents.tolist(), "bounds": mesh.bounds.tolist(), "n_vert": int(len(mesh.vertices))})
                continue
            W, H = int(msg["W"]), int(msg["H"])
            n = W * H
            K = np.asarray(msg["K"], np.float64).reshape(3, 3)
            t0 = time.perf_counter()
            rgb = np.frombuffer(buf, np.uint8, 3 * n, lay["rgb"]).reshape(H, W, 3).copy()
            depth = np.frombuffer(buf, np.float32, n, lay["depth"]).reshape(H, W).copy()
            if op == "register":
                mask = np.frombuffer(buf, np.uint8, n, lay["mask"]).reshape(H, W).copy()
                t1 = time.perf_counter()
                pose, sc = core.register(msg["name"], rgb, depth, mask, K, int(msg.get("iters", 5)), msg.get("up_cam"))
            elif op == "init":
                t1 = time.perf_counter()
                pose, sc = core.init_from_prior(msg["name"], rgb, depth, K, int(msg.get("iters", 5)), np.asarray(msg["prior"], float).reshape(4, 4))
            elif op == "track":
                t1 = time.perf_counter()
                pose, sc = core.track(msg["name"], rgb, depth, K, int(msg.get("iters", 2)), bool(msg.get("score", False)))
            else:
                raise ValueError(op)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            send(conn, {"ok": True, "pose": np.asarray(pose, float).reshape(-1).tolist(), "score": sc, "ms": dict(upload=1e3 * (t1 - t0), infer=1e3 * (t2 - t1))})
        except Exception:
            traceback.print_exc()
            send(conn, {"ok": False, "err": traceback.format_exc()[-1000:]})
    conn.close()
    srv.close()
    try:
        os.unlink(a.sock)
    except OSError:
        pass
    del buf
    shm.close()


if __name__ == "__main__":
    main()
