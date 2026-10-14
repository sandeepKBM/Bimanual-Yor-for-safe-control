"""Search for LEFT/RIGHT arm joint configs ("hover poses") on the PROPER YOR
robot such that the arm's REAL egocentric wrist camera
(left_arm_egocentric / right_arm_egocentric -- both physically exist on the
real robot, see YOR_CANONICAL.md) gets a close, oblique, ball-level-resolving
view of the cup+balls in scenes/scene_yor_cup.xml, WITHOUT the gripper's own
body physically overlapping the cup's footprint (the cup can lift up to
~0.12m and tilt ~90-115deg during the "spill_tilt" behavior; since the
teleport-scripted tilt rotates the cup body about the WORLD X axis, its
world-frame X-extent is invariant to the tilt angle -- only Y/Z change -- so
keeping the hover point's X offset from the cup large enough guarantees no
overlap regardless of tilt/lift/shake amplitude).

Method (mirrors the old cup pipeline's "empirical sweep, score by rendered
ball-pixel count" methodology, adapted to a kinematically-constrained arm
instead of a free camera pose):
  1. Pure-FK random search (no rendering) over the 6 arm joint angles: for
     each sample, compute the camera's world pose (data.cam_xpos/cam_xmat,
     exact MuJoCo values, no hand-derived optics) and score by (a) distance
     from camera to the cup's ball-pile centroid landing in a good range,
     (b) how well the camera's view ray (-Z column) is aimed at that
     centroid, (c) a hard-reject if the gripper body itself comes within a
     collision-risk margin of the cup's world-frame X-extent.
  2. Render the top candidates at the ACTUAL 64x64 recording resolution and
     count "ball-blue" pixels (b > r+15 and b > g+5, ball_mat rgba
     0.15/0.55/0.9) -- the real acceptance criterion, not just alignment.

Run: MUJOCO_GL=egl <safe_env> python scripts/probe_yor_cup_hover.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cup_task_utils import load_yor_cup_model, ball_body_ids  # noqa: E402

ARTIFACTS = REPO_ROOT / "artifacts"

CUP_POS = np.array([0.42, -0.21, 0.55])
CUP_RADIUS_OUTER = 0.0315  # wall centerline (0.030) + half wall thickness (0.0015)
BALL_TARGET = CUP_POS + np.array([0.0, 0.0, 0.045])  # ~mid ball-pile height

# Hard collision-risk margin: reject any candidate whose gripper_base body
# comes within this many meters (in world X) of the cup's own X-extent
# (cup center X +/- CUP_RADIUS_OUTER). The cup's tilt (about world X) and
# lift (world Z) NEVER change its X-extent; only shake (lateral X jitter,
# amplitude <= 0.045m) does -- so this margin already covers the worst-case
# shake amplitude plus a safety pad.
X_CLEARANCE_MARGIN = 0.09


def score_candidate(model, data, side, q):
    """side: 'left' or 'right'. q: (6,) joint angles. Returns (score, info) or
    (None, None) if hard-rejected (gripper too close to the cup's X-extent)."""
    prefix = f"{side}_arm_"
    joint_ids = [model.joint(f"{prefix}joint{i}").id for i in range(1, 7)]
    qposadr = [model.jnt_qposadr[j] for j in joint_ids]
    data.qpos[qposadr] = q
    mujoco.mj_forward(model, data)

    gripper_bid = model.body(f"{prefix}gripper_base").id
    gripper_pos = data.xpos[gripper_bid]
    x_clear = abs(gripper_pos[0] - CUP_POS[0]) - CUP_RADIUS_OUTER
    if x_clear < X_CLEARANCE_MARGIN:
        return None, None

    cam_id = model.camera(f"{prefix}egocentric").id
    cam_pos = data.cam_xpos[cam_id].copy()
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()
    view_dir = -cam_mat[:, 2]  # MuJoCo camera looks along local -Z

    to_cup = BALL_TARGET - cam_pos
    dist = float(np.linalg.norm(to_cup))
    to_cup_n = to_cup / max(dist, 1e-9)
    alignment = float(np.dot(view_dir, to_cup_n))  # 1.0 == perfectly aimed

    dist_lo, dist_hi = 0.05, 0.14
    if dist < dist_lo:
        dist_pen = (dist_lo - dist) * 10.0
    elif dist > dist_hi:
        dist_pen = (dist - dist_hi) * 10.0
    else:
        dist_pen = 0.0
    align_pen = (1.0 - alignment) * 5.0

    score = dist_pen + align_pen
    info = dict(q=q.copy(), dist=dist, alignment=alignment, x_clear=x_clear,
                cam_pos=cam_pos.copy())
    return score, info


def search(model, data, side, n=60000, seed=0, top_k=5):
    rng = np.random.default_rng(seed)
    joint_ids = [model.joint(f"{side}_arm_joint{i}").id for i in range(1, 7)]
    jr = model.jnt_range[joint_ids].copy()
    results = []
    for _ in range(n):
        q = rng.uniform(jr[:, 0], jr[:, 1])
        score, info = score_candidate(model, data, side, q)
        if score is None:
            continue
        results.append((score, info))
    results.sort(key=lambda x: x[0])
    return results[:top_k]


def render_and_count(model, data, side, q, renderer, tag):
    prefix = f"{side}_arm_"
    joint_ids = [model.joint(f"{prefix}joint{i}").id for i in range(1, 7)]
    qposadr = [model.jnt_qposadr[j] for j in joint_ids]
    data.qpos[qposadr] = q
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=f"{prefix}egocentric")
    img = renderer.render()
    b, r, g = img[..., 2].astype(int), img[..., 0].astype(int), img[..., 1].astype(int)
    ball_mask = (b > r + 15) & (b > g + 5)
    n_ball_px = int(ball_mask.sum())
    Image.fromarray(img).save(ARTIFACTS / f"cup_v2_hover_{tag}.png")
    return n_ball_px, img


def main():
    model, data = load_yor_cup_model()
    key = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    # settle balls into a resting pile before scoring/rendering
    for _ in range(int(0.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    home_qpos = data.qpos.copy()

    renderer = mujoco.Renderer(model, height=64, width=64)
    best = {}
    for side in ("left", "right"):
        data.qpos[:] = home_qpos
        mujoco.mj_forward(model, data)
        top = search(model, data, side)
        print(f"\n[{side}] top {len(top)} FK-scored candidates:")
        best_px, best_q, best_info = -1, None, None
        for score, info in top:
            data.qpos[:] = home_qpos
            n_px, _ = render_and_count(model, data, side, info["q"], renderer, f"{side}_cand")
            print(f"  score={score:.4f} dist={info['dist']:.3f} align={info['alignment']:.3f} "
                  f"x_clear={info['x_clear']:.3f} ball_px={n_px}")
            if n_px > best_px:
                best_px, best_q, best_info = n_px, info["q"], info
        data.qpos[:] = home_qpos
        n_px, img = render_and_count(model, data, side, best_q, renderer, f"{side}_best")
        best[side] = dict(q=best_q.tolist(), ball_px=n_px, dist=best_info["dist"],
                           alignment=best_info["alignment"], x_clear=best_info["x_clear"])
        print(f"[{side}] BEST: ball_px={n_px} q={np.round(best_q, 6).tolist()}")

    print("\n=== SUMMARY ===")
    for side, info in best.items():
        print(side, info)

    renderer.close()


if __name__ == "__main__":
    main()
