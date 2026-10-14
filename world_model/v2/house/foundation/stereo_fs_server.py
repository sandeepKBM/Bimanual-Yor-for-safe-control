"""FoundationStereo depth server.  Runs ONLY in the `foundation` conda env (torch 2.8+cu128, tensorrt 10.13).

    CUDA_VISIBLE_DEVICES=5 python stereo_fs_server.py --sock /path/fs.sock --shm NAME --shm-bytes N \
         --sizes 640x360 --engine trt --iters 12 --warmup 10

Transport (no TCP): a Unix-domain stream socket bound to a FILE path (mode 0600) carries 4-byte-length-prefixed JSON control
messages; the pixels travel through one multiprocessing.shared_memory block created by the client:
    [ left u8 (H*W*3) | right u8 (H*W*3) | depth f32 (H*W) | valid u8 (H*W) ]   (laid out for the max W,H; see stereo_fs.layout)
Messages:  server -> client  {"ready": true, "warmup": {...timings...}}      once, after weights + engine load + N dummy inferences
           client -> server  {"op": "depth", "W":..,"H":..,"f":..,"baseline":..}   -> {"ok":true,"ms":{...}}
           client -> server  {"op": "quit"}
Engines: 'trt' = TensorRT FP16 engine built from the upstream ONNX export (fs_export_onnx.py / fs_build_trt.py), fixed (H,W) padded to 32;
         'graph' = PyTorch eager model replayed through a CUDA graph (fs_patches.py); 'eager' = plain PyTorch autocast.
No network, no robot code is imported here.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import time
import traceback
from multiprocessing import shared_memory

import numpy as np
import torch

FS = "/data/local/ss5772/models/FoundationStereo"
TRT_DIR = "/data/local/ss5772/models/trt"
sys.path.insert(0, FS)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def pad32(n):
    return (n + 31) // 32 * 32


def layout(maxW, maxH):
    n = maxW * maxH
    o_l = 0
    o_r = o_l + 3 * n
    o_d = o_r + 3 * n
    o_v = o_d + 4 * n
    return dict(left=o_l, right=o_r, depth=o_d, valid=o_v, total=o_v + n)


def send(conn, obj):
    b = json.dumps(obj).encode()
    conn.sendall(struct.pack("<I", len(b)) + b)


def recv(conn):
    h = b""
    while len(h) < 4:
        c = conn.recv(4 - len(h))
        if not c:
            return None
        h += c
    n = struct.unpack("<I", h)[0]
    buf = b""
    while len(buf) < n:
        c = conn.recv(n - len(buf))
        if not c:
            return None
        buf += c
    return json.loads(buf.decode())


# ------------------------------------------------------------------------------------------------ runners
class TRTRunner:
    def __init__(self, plan, H, W, dev):
        import tensorrt as trt
        self.trt = trt
        lg = trt.Logger(trt.Logger.WARNING)
        rt = trt.Runtime(lg)
        self.eng = rt.deserialize_cuda_engine(open(plan, "rb").read())
        self.ctx = self.eng.create_execution_context()
        self.H, self.W = H, W
        self.l = torch.zeros(1, 3, H, W, device=dev)
        self.r = torch.zeros(1, 3, H, W, device=dev)
        oshape = tuple(self.eng.get_tensor_shape("disp"))
        self.out = torch.zeros(oshape, device=dev)
        self.ctx.set_tensor_address("left", self.l.data_ptr())
        self.ctx.set_tensor_address("right", self.r.data_ptr())
        self.ctx.set_tensor_address("disp", self.out.data_ptr())
        self.stream = torch.cuda.Stream()

    def __call__(self, l, r):
        self.l.copy_(l)
        self.r.copy_(r)
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.ctx.execute_async_v3(self.stream.cuda_stream)
        torch.cuda.current_stream().wait_stream(self.stream)
        return self.out.view(1, 1, self.H, self.W)


class TorchRunner:
    def __init__(self, model, H, W, iters, use_graph, dev):
        self.m, self.iters, self.g = model, iters, None
        self.l = torch.zeros(1, 3, H, W, device=dev)
        self.r = torch.zeros(1, 3, H, W, device=dev)
        with torch.no_grad(), torch.autocast("cuda", enabled=True):
            if use_graph:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        model(self.l, self.r, iters=iters, test_mode=True)
                torch.cuda.current_stream().wait_stream(s)
                self.g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.g):
                    self.out = model(self.l, self.r, iters=iters, test_mode=True)

    def __call__(self, l, r):
        self.l.copy_(l)
        self.r.copy_(r)
        if self.g is not None:
            self.g.replay()
            return self.out
        with torch.no_grad(), torch.autocast("cuda", enabled=True):
            return self.m(self.l, self.r, iters=self.iters, test_mode=True)


def build_runner(args, H, W, model_holder, dev):
    Hp, Wp = pad32(H), pad32(W)
    if args.engine == "trt":
        plan = f"{TRT_DIR}/fs_{args.vit}_{Hp}x{Wp}_i{args.iters}" + (f"_d{args.max_disp}" if args.max_disp else "") + ".plan"
        if os.path.exists(plan):
            return TRTRunner(plan, Hp, Wp, dev), "trt"
        print(f"[fs-server] no engine {plan}; falling back to cuda graph", flush=True)
    if model_holder[0] is None:
        from omegaconf import OmegaConf
        from core.foundation_stereo import FoundationStereo
        import fs_patches
        name = args.ckpt
        ck = f"{FS}/pretrained_models/{name}/model_best_bp2.pth"
        cfg = OmegaConf.load(os.path.dirname(ck) + "/cfg.yaml")
        if "vit_size" not in cfg:
            cfg["vit_size"] = "vitl"
        if args.max_disp:
            cfg["max_disp"] = int(args.max_disp)
        m = FoundationStereo(cfg)
        m.load_state_dict(torch.load(ck, weights_only=False)["model"])
        m.to(dev).eval()
        fs_patches.apply(m, dev)
        model_holder[0] = m
    return TorchRunner(model_holder[0], Hp, Wp, args.iters, args.engine in ("trt", "graph"), dev), ("graph" if args.engine in ("trt", "graph") else "eager")


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", required=True)
    ap.add_argument("--shm", required=True)
    ap.add_argument("--maxwh", default="1280x720")
    ap.add_argument("--sizes", default="640x360", help="comma list of WxH to preload (engines/graphs built for each)")
    ap.add_argument("--engine", default="trt", choices=["trt", "graph", "eager"])
    ap.add_argument("--ckpt", default="23-51-11")
    ap.add_argument("--vit", default="vitl", help="engine file tag (vitl for 23-51-11, vits for 11-33-40)")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--max-disp", type=int, default=0, help="override cfg max_disp (416 trained); 0 = keep.  Needed range = f*B/z_min px")
    ap.add_argument("--min-disp", type=float, default=0.5)
    args = ap.parse_args()

    t_start = time.perf_counter()
    torch.backends.cudnn.benchmark = args.engine != "trt"
    dev = torch.device("cuda")
    maxW, maxH = (int(v) for v in args.maxwh.split("x"))
    lay = layout(maxW, maxH)
    shm = shared_memory.SharedMemory(name=args.shm)
    buf = np.ndarray((lay["total"],), np.uint8, buffer=shm.buf)

    if os.path.exists(args.sock):
        os.unlink(args.sock)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.sock)
    os.chmod(args.sock, 0o600)
    srv.listen(1)

    runners, model_holder, wu = {}, [None], {}
    for s in args.sizes.split(","):
        W, H = (int(v) for v in s.split("x"))
        t0 = time.perf_counter()
        r, kind = build_runner(args, H, W, model_holder, dev)
        t1 = time.perf_counter()
        x = torch.rand(1, 3, pad32(H), pad32(W), device=dev) * 255
        ts = []
        for _ in range(args.warmup):
            torch.cuda.synchronize()
            a = time.perf_counter()
            r(x, x.roll(-6, 3))
            torch.cuda.synchronize()
            ts.append(1e3 * (time.perf_counter() - a))
        runners[(H, W)] = r
        wu[s] = dict(kind=kind, build_s=round(t1 - t0, 2), dummy_ms=[round(v, 1) for v in ts], iters=args.iters)
        print(f"[fs-server] {s} {kind} build {t1 - t0:.1f}s warm-up last {ts[-1]:.1f} ms", flush=True)
    wu["total_s"] = round(time.perf_counter() - t_start, 2)
    wu["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    conn, _ = srv.accept()
    send(conn, {"ready": True, "warmup": wu, "pid": os.getpid()})
    print("[fs-server] READY", flush=True)

    ycache = {}
    while True:
        msg = recv(conn)
        if msg is None or msg.get("op") == "quit":
            break
        try:
            W, H, f, B = int(msg["W"]), int(msg["H"]), float(msg["f"]), float(msg["baseline"])
            n = W * H
            t0 = time.perf_counter()
            L = np.frombuffer(buf, np.uint8, 3 * n, lay["left"]).reshape(H, W, 3)
            R = np.frombuffer(buf, np.uint8, 3 * n, lay["right"]).reshape(H, W, 3)
            l = torch.from_numpy(L.copy()).to(dev).permute(2, 0, 1)[None].float()
            r = torch.from_numpy(R.copy()).to(dev).permute(2, 0, 1)[None].float()
            Hp, Wp = pad32(H), pad32(W)
            runner = runners.get((H, W))
            if runner is None:
                # on-demand size (a CUDA graph / engine for it is built now: slow, logged)
                runner, kind = build_runner(args, H, W, model_holder, dev)
                runners[(H, W)] = runner
                print(f"[fs-server] on-demand runner {W}x{H} ({kind})", flush=True)
            if (Hp, Wp) != (H, W):
                pw, ph = Wp - W, Hp - H
                pad = (pw // 2, pw - pw // 2, 0, ph)          # upstream InputPadder(force_square=False): centred in width, bottom in height
                l = torch.nn.functional.pad(l, pad, mode="replicate")
                r = torch.nn.functional.pad(r, pad, mode="replicate")
            else:
                pad = (0, 0, 0, 0)
            t1 = time.perf_counter()
            disp = runner(l, r).float()
            disp = disp[..., pad[2]:Hp - pad[3], pad[0]:Wp - pad[1]][0, 0]
            xs = ycache.get((H, W))
            if xs is None:
                xs = ycache[(H, W)] = torch.arange(W, device=dev, dtype=torch.float32)[None].expand(H, W).contiguous()
            valid = torch.isfinite(disp) & (disp > args.min_disp) & ((xs - disp) >= 0)     # right-image coordinate must be inside the image
            depth = torch.where(valid, f * B / disp.clamp(min=1e-3), torch.full_like(disp, float("nan")))
            depth_np = depth.cpu().numpy()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            np.frombuffer(buf, np.float32, n, lay["depth"])[:] = depth_np.ravel()
            np.frombuffer(buf, np.uint8, n, lay["valid"])[:] = valid.cpu().numpy().ravel().astype(np.uint8)
            t3 = time.perf_counter()
            send(conn, {"ok": True, "ms": dict(upload=1e3 * (t1 - t0), infer=1e3 * (t2 - t1), download=1e3 * (t3 - t2), total=1e3 * (t3 - t0))})
        except Exception:
            traceback.print_exc()
            send(conn, {"ok": False, "err": traceback.format_exc()[-800:]})
    conn.close()
    srv.close()
    try:
        os.unlink(args.sock)
    except OSError:
        pass
    del buf
    shm.close()


if __name__ == "__main__":
    main()
