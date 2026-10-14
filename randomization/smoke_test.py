"""SCAFFOLD smoke test for randomization/domain_randomization.py.

Loads scenes/scene_yor_cup.xml READ-ONLY, applies `randomize()` in-memory
across ~6 seeds, renders each, checks each compiled model still has
arms+table+cup present, and writes a montage to artifacts/dr_samples.png.

Run:
    MUJOCO_GL=egl /common/users/ss5772/miniforge3/bin/conda run -n safe_env \
        python randomization/smoke_test.py
"""
from __future__ import annotations

import random
from pathlib import Path

import mujoco
import numpy as np

from randomization.domain_randomization import RandomizationConfig, randomize

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = REPO_ROOT / "scenes" / "scene_yor_cup.xml"
OUT_PATH = REPO_ROOT / "artifacts" / "dr_samples.png"
SEEDS = [0, 1, 2, 3, 4, 5]
REQUIRED_BODIES = ["left_arm_link0", "right_arm_link0", "table", "cup"]
RENDER_CAM = "front_camera"  # world-view camera defined in yor_robot_derived.mjcf
W, H = 480, 360


def render_seed(base_spec: mujoco.MjSpec, seed: int, config: RandomizationConfig):
    rng = random.Random(seed)
    spec = base_spec.copy()  # never mutate the shared read-only spec
    randomize(spec, rng, config)  # structural (distractors)
    model = spec.compile()
    randomize(model, rng, config)  # numeric (visual/dynamics/scene poses)

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    checks = {}
    for name in REQUIRED_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        checks[name] = bid != -1
    checks["nu==16"] = model.nu == 16
    checks["ncam>=4"] = model.ncam >= 4
    ok = all(checks.values())

    renderer = mujoco.Renderer(model, height=H, width=W)
    renderer.update_scene(data, camera=RENDER_CAM)
    img = renderer.render()
    renderer.close()
    return img, ok, checks


def main() -> None:
    config = RandomizationConfig()
    base_spec = mujoco.MjSpec.from_file(str(SCENE_PATH))

    images = []
    all_ok = True
    for seed in SEEDS:
        img, ok, checks = render_seed(base_spec, seed, config)
        images.append(img)
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        print(f"seed={seed} {status} mean_rgb={img.mean():.1f} checks={checks}")

    # Variety check: pairwise mean-abs-diff between renders should be
    # non-trivial (not all seeds producing an identical frame).
    diffs = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            diffs.append(np.abs(images[i].astype(np.float32) - images[j].astype(np.float32)).mean())
    print(f"pairwise mean-abs-diff: min={min(diffs):.2f} max={max(diffs):.2f} mean={np.mean(diffs):.2f}")
    variety_ok = min(diffs) > 1.0  # any two seeds differ by more than noise floor

    montage = np.zeros((2 * H, 3 * W, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        r, c = divmod(idx, 3)
        montage[r * H:(r + 1) * H, c * W:(c + 1) * W] = img

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio
    imageio.imwrite(str(OUT_PATH), montage)
    print(f"montage written to {OUT_PATH}")
    print(f"ALL_SEEDS_VALID={all_ok} VARIETY_CONFIRMED={variety_ok}")


if __name__ == "__main__":
    main()
