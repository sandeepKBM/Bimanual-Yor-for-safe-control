"""recard.py -- re-render ONLY the intro and end cards of an existing demo video from results.json + the saved key frames, and splice them
(ffmpeg re-encode) in place of the old ones.  Used to fix card layout without the 6-minute full re-render.   SIMULATION ONLY.

    PYTHONPATH=. python -m world_model.v2.house.zed_demo.recard [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import types
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/local/ss5772/molmospaces/zed_demo")
    ap.add_argument("--intro-s", type=float, default=3.2)
    ap.add_argument("--end-s", type=float, default=6.0)
    ap.add_argument("--old-end-s", type=float, default=6.0)
    a = ap.parse_args()
    out = Path(a.out)
    import cv2
    import imageio
    from world_model.v2.house.zed_demo import video as V, run as RUN
    R = json.loads((out / "results.json").read_text())
    ffmpeg = os.environ.get("FFMPEG", "/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/ffmpeg")
    src = out / "zed_demo.mp4"
    dur = R["video"]["seconds"]
    fr = out / "frames"
    ek = sorted(fr.glob("exec_*.png"), key=lambda p: int(p.stem.split("_")[1]))[-1]
    nk = sorted(fr.glob("naive_*.png"), key=lambda p: int(p.stem.split("_")[1]))[0]
    th = [(90, 618, cv2.resize(imageio.imread(ek), (800, 450), interpolation=cv2.INTER_AREA)),
          (1030, 618, cv2.resize(imageio.imread(nk), (800, 450), interpolation=cv2.INTER_AREA))]
    left0 = imageio.imread(out / "stereo_left.png")
    end = RUN._end_card(types.SimpleNamespace(), R, left0, None, th)
    imageio.imwrite(str(fr / "end_card.png"), end)
    ctx = types.SimpleNamespace()
    n = int(a.intro_s * V.FPS)
    tmp = out / "_cards"
    tmp.mkdir(exist_ok=True)
    w = imageio.get_writer(str(tmp / "intro.mp4"), fps=V.FPS, codec="libx264", macro_block_size=None, quality=None,
                           ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p"])
    for i in range(n):
        w.append_data(V.pipeline_card(ctx, i, n))
    w.close()
    imageio.imwrite(str(fr / "intro.png"), V.pipeline_card(ctx, n - 1, n))
    w = imageio.get_writer(str(tmp / "end.mp4"), fps=V.FPS, codec="libx264", macro_block_size=None, quality=None,
                           ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p"])
    for i in range(int(a.end_s * V.FPS)):
        w.append_data(end)
    w.close()
    t0, t1 = 3.2, dur - a.old_end_s
    dst = out / "zed_demo_recard.mp4"
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(tmp / "intro.mp4"), "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}", "-i", str(src), "-i", str(tmp / "end.mp4"),
           "-filter_complex", "[1:v]setpts=PTS-STARTPTS[m];[0:v][m][2:v]concat=n=3:v=1:a=0[v]", "-map", "[v]", "-c:v", "libx264", "-crf", "19", "-preset", "medium",
           "-pix_fmt", "yuv420p", "-r", str(V.FPS), str(dst)]
    subprocess.run(cmd, check=True)
    dst.replace(src)
    for p in tmp.glob("*"):
        p.unlink()
    tmp.rmdir()
    print("wrote", src)
    os._exit(0)


if __name__ == "__main__":
    main()
