"""people_track/video_v2.py -- head-camera RGB clip with the neural detections overlaid.  SIMULATION ONLY."""
from __future__ import annotations

import numpy as np


def write_video(path, vframes, fps=10):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    w = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=None)
    for v in vframes:
        im = Image.fromarray(v["img"])
        dr = ImageDraw.Draw(im, "RGBA")
        dr.rectangle([0, 0, im.width, 22], fill=(0, 0, 0, 150))
        info = "  ".join(f"p{i}: {r:.1f} m, {n} depth px" for i, (r, n) in enumerate(zip(v["rng"], v["npix"])))
        dr.text((6, 3), f"t={v['t']:5.1f}s  neural person detections (Faster R-CNN v2)   {info}", font=f, fill=(255, 255, 255))
        for (x1, y1, x2, y2), sc, xyt in zip(v["boxes"], v["scores"], v["dets"] + [None] * len(v["boxes"])):
            if sc < 0.5:
                continue
            dr.rectangle([x1, y1, x2, y2], outline=(255, 60, 60, 255), width=3)
            lab = f"person {sc:.2f}"
            if xyt is not None and len(v["truth"]):
                e = min(float(np.hypot(*(xyt - np.asarray(p)))) for p in v["truth"])
                lab += f"  3D err {e * 100:.0f} cm"
            dr.rectangle([x1, max(y1 - 20, 0), x1 + 8 * len(lab), max(y1 - 20, 0) + 18], fill=(255, 60, 60, 220))
            dr.text((x1 + 2, max(y1 - 20, 0) + 1), lab, font=f, fill=(255, 255, 255))
        w.append_data(np.asarray(im.convert("RGB")))
    w.close()
