"""people_track/detect_server.py -- OUT-OF-PROCESS neural person detector.  SIMULATION ONLY.

Runs in its OWN conda env (/data/local/ss5772/conda_envs/detect: torch 2.8 + torchvision 0.23, no repo imports, no mujoco) and serves over a
UNIX-DOMAIN socket (a file path; no TCP).  Model: torchvision Faster R-CNN ResNet50-FPN **v2**, COCO weights (BSD-3 code; weights
https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth).  Keeps the COCO 'person' class only.

STARTUP COST IS PAID BEFORE USE: load weights -> move to GPU -> N dummy inferences at every operating resolution (`--res 320x180,640x360`)
-> only then bind/listen and create the sentinel file `<sock>.ready`.  Clients wait on that file (or on the "READY" reply to a PING).

Serves MANY clients concurrently (a thread per connection, one lock around the GPU).  Protocol (little endian):  request  = 'F' | u16 W | u16 H | u32 nbytes | RGB uint8 (H*W*3)       -> reply = u32 n | f32 gpu_ms | n * 5 f32 (x1 y1 x2 y2 score)
                           request  = 'P'                                                          -> reply = b'READY'
                           request  = 'Q'                                                          -> server exits
"""
from __future__ import annotations

import argparse
import os
import socket
import struct
import sys
import time

import numpy as np


def recvall(c, n):
    buf = bytearray()
    while len(buf) < n:
        b = c.recv(n - len(buf))
        if not b:
            raise ConnectionError
        buf += b
    return bytes(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", required=True)
    ap.add_argument("--res", default="320x180,640x360")
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--thr", type=float, default=0.30, help="min score returned (the client applies its own threshold)")
    ap.add_argument("--model", default="frcnn_v2", choices=["frcnn_v2", "frcnn_mnv3"])
    ap.add_argument("--fp16", type=int, default=0)
    a = ap.parse_args()
    t_start = time.perf_counter()
    import torch
    import torchvision
    from torchvision.models import detection as DT
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    if a.model == "frcnn_v2":
        net = DT.fasterrcnn_resnet50_fpn_v2(weights=DT.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, box_score_thresh=a.thr)
    else:
        net = DT.fasterrcnn_mobilenet_v3_large_fpn(weights=DT.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1, box_score_thresh=a.thr)
    net.eval().to(dev)
    t_load = time.perf_counter() - t_start
    res = [tuple(int(v) for v in r.split("x")) for r in a.res.split(",")]

    @torch.inference_mode()
    def infer(img_u8, W, H):
        net.transform.min_size, net.transform.max_size = (H,), max(W, H)          # no internal resize: the operating resolution IS the model input
        x = torch.from_numpy(img_u8).to(dev, non_blocking=True).permute(2, 0, 1).float().div_(255.0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16, enabled=bool(a.fp16)):
            out = net([x])[0]
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        keep = out["labels"] == 1
        bx = torch.cat([out["boxes"][keep], out["scores"][keep, None]], 1).float().cpu().numpy()
        return bx, ms

    warm_ms = {}
    for (W, H) in res:
        rng = np.random.default_rng(0)
        ms = []
        for _ in range(a.warm):
            _, t = infer(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), W, H)
            ms.append(t)
        warm_ms[(W, H)] = (ms[0], float(np.median(ms[a.warm // 2:])))
    if os.path.exists(a.sock):
        os.unlink(a.sock)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(a.sock)
    s.listen(4)
    with open(a.sock + ".ready", "w") as f:
        f.write(f"model={a.model} load_s={t_load:.2f} startup_s={time.perf_counter() - t_start:.2f} first_ms/steady_ms={warm_ms}\n")
    print(f"READY model={a.model} load={t_load:.1f}s total_startup={time.perf_counter() - t_start:.1f}s warm(first,steady ms)={warm_ms}", flush=True)
    import threading
    lock = threading.Lock()                       # one GPU: inference is serialised, clients (sweep workers) are served concurrently
    stop = threading.Event()

    def serve(c):
        try:
            while True:
                op = c.recv(1)
                if not op:
                    break
                if op == b"P":
                    c.sendall(b"READY")
                elif op == b"Q":
                    stop.set()
                    break
                elif op == b"F":
                    W, H, nb = struct.unpack("<HHI", recvall(c, 8))
                    img = np.frombuffer(recvall(c, nb), np.uint8).reshape(H, W, 3).copy()
                    with lock:
                        bx, ms = infer(img, W, H)
                    c.sendall(struct.pack("<If", len(bx), ms) + bx.astype("<f4").tobytes())
        except (ConnectionError, BrokenPipeError, OSError):
            pass
        finally:
            c.close()

    s.settimeout(0.5)
    while not stop.is_set():
        try:
            c, _ = s.accept()
        except socket.timeout:
            continue
        threading.Thread(target=serve, args=(c,), daemon=True).start()
    os.unlink(a.sock)
    sys.exit(0)


if __name__ == "__main__":
    main()
