"""v4: render initial / peak-intrusion / after frames for A/B/C and stitch
into a comparison montage, artifacts/ab4_render_montage.png.

Run:
    MUJOCO_GL=egl /common/users/ss5772/miniforge3/bin/conda run -n safe_env \\
        python /common/users/ss5772/bimanual/experiments/ab_smooth_vs_bangbang/render_montage_v4.py
"""
from __future__ import annotations

import os
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sim_common as sc
from run_ab4 import run_condition, CONDITIONS

ART_DIR = "/common/users/ss5772/bimanual/artifacts"
os.makedirs(ART_DIR, exist_ok=True)

CAPTURE_T = [0.0, 1.1, 5.95]


def render(model, data, width=960, height=720):
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.lookat = np.array([0.42, -0.21, 0.65])
    cam.distance = 0.9
    cam.azimuth = 160
    cam.elevation = -15
    renderer.update_scene(data, camera=cam)
    pixels = renderer.render()
    renderer.close()
    return pixels


def main():
    capture_steps = sorted(set(int(round(t / sc.DT)) for t in CAPTURE_T))
    frames = {}

    def cb(condition, k, model, data):
        frames[(condition, k)] = render(model, data)

    fig, axes = plt.subplots(3, len(CAPTURE_T), figsize=(4.6 * len(CAPTURE_T), 12.6))
    label = {"A": "A: adversarial", "B": "B: cooperative symmetric",
              "C": "C: cooperative, ego-priority"}
    phase = {0: "initial", 1: "peak intrusion", 2: "after"}
    for row, condition in enumerate(CONDITIONS):
        run_condition(condition, capture_steps=capture_steps, capture_cb=cb)
        for col, (t_target, k) in enumerate(zip(CAPTURE_T, capture_steps)):
            ax = axes[row, col]
            ax.imshow(frames[(condition, k)])
            ax.axis("off")
            ax.set_title(f"{label[condition]} -- {phase[col]} (t={k*sc.DT:.2f}s)", fontsize=11)
    fig.suptitle("v4: cooperative-intruder A/B/C comparison (ego fixed = C2 smooth HOCBF-QP)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(ART_DIR, "ab4_render_montage.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
