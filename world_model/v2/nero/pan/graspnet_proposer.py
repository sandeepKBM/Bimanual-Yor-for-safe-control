"""graspnet-baseline grasp proposer (borrowed component; free for non-commercial research use).

Run (graspnet env python, NOT the mujoco env; pin GPU 6 on westeros):
    CUDA_VISIBLE_DEVICES=6 /data/local/ss5772/graspnet/env/bin/python \
        world_model/v2/nero/pan/graspnet_proposer.py --in cloud.npz --out grasps.json [--top 50]

Input .npz : points (N,3) float32 metres, CAMERA frame (OpenCV, z forward); optional colors (N,3) in [0,1].
Output JSON: list of {score, width, height, depth, rotation (3x3 row-major list), translation (3)}, camera frame,
             NMS'd and sorted by score (descending). Collision filtering uses the full input cloud
             (ModelFreeCollisionDetector, voxel 0.01, approach_dist 0.05, thresh 0.01).
Defaults: num_point=20000, num_view=300.

Provenance:
    repo   https://github.com/graspnet/graspnet-baseline  commit 280c215129f759ed8649cb4e89fc5dfee55f4f80
    ckpt   /data/local/ss5772/graspnet/weights/checkpoint-rs.tar (RealSense)
           sha256 60680087c61cba2b6791614fef1519071e294f6dcaf99b3f581bb95f7c51a868
    env    /data/local/ss5772/graspnet/env (py3.10, torch 2.1.2+cu118, sm_75 build); patches: see /data/local/ss5772/graspnet/PATCHES.md (NMS reimplemented in numpy here)

Gripper assumptions baked into the network / code (graspnet-baseline + graspnetAPI):
    - max opening width GRASP_MAX_WIDTH = 0.10 m (loss_utils.py); predicted width clamped to [0, 0.10].
      Our claw opens to 100 mm max -> matches exactly, so there is NO margin: width == 0.10 means "at the limit".
    - depth is one of {0.01, 0.02, 0.03, 0.04} m (finger insertion past the contact point along the approach axis).
    - height is a constant 0.02 m (graspnet_baseline graspnet.py: 0.02 * ones) = finger/jaw extent along gripper z.
    - ModelFreeCollisionDetector gripper model: finger_width (thickness) 0.01 m, finger_length 0.06 m.
      Our fingers are ~80 mm long, so collision checking is optimistic by ~20 mm at the finger root: re-check with
      our own geometry/checkers before admitting a grasp.
    - Frame: gripper x = approach direction, y = closing direction, z = height; `translation` is the grasp
      centre (finger-pad point at insertion depth), rotation columns are the gripper axes in the camera frame.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

SRC = os.environ.get("GRASPNET_SRC", "/data/local/ss5772/graspnet/src")
CKPT = os.environ.get("GRASPNET_CKPT", "/data/local/ss5772/graspnet/weights/checkpoint-rs.tar")
for sub in ("models", "dataset", "utils"):
    sys.path.append(os.path.join(SRC, sub))


def load_net(ckpt=CKPT, num_view=300):
    import torch
    from graspnet import GraspNet
    net = GraspNet(input_feature_dim=0, num_view=num_view, num_angle=12, num_depth=4, cylinder_radius=0.05,
                   hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
    net.to(torch.device("cuda:0"))
    net.load_state_dict(torch.load(ckpt)["model_state_dict"])
    net.eval()
    return net


def propose(points, net=None, num_point=20000, voxel_size=0.01, approach_dist=0.05, collision_thresh=0.01,
            top=None, seed=0):
    """points (N,3) float camera-frame metres -> graspnetAPI GraspGroup (NMS'd, sorted, collision-free)."""
    import torch
    from graspnet import pred_decode
    from graspnetAPI import GraspGroup
    from collision_detector import ModelFreeCollisionDetector
    rng = np.random.RandomState(seed)
    points = np.asarray(points, np.float32)
    points = points[np.isfinite(points).all(1)]
    net = net or load_net()
    if len(points) >= num_point:
        idx = rng.choice(len(points), num_point, replace=False)
    else:
        idx = np.concatenate([np.arange(len(points)), rng.choice(len(points), num_point - len(points), replace=True)])
    cloud = torch.from_numpy(points[idx][None]).cuda()
    with torch.no_grad():
        ep = net({"point_clouds": cloud})
        preds = pred_decode(ep)
    gg = GraspGroup(preds[0].detach().cpu().numpy())
    if collision_thresh > 0:
        det = ModelFreeCollisionDetector(points, voxel_size=voxel_size)
        gg = gg[~det.detect(gg, approach_dist=approach_dist, collision_thresh=collision_thresh)]
    gg = _nms(gg).sort_by_score()
    return gg[:top] if top else gg


def _nms(gg, trans_thresh=0.03, rot_thresh=30.0 / 180.0 * np.pi):
    """Greedy NMS equivalent to graspnetAPI's C++ grasp_nms (not shipped in the pip wheel): drop a grasp if a
    higher-scored kept one is within trans_thresh (m) and rot_thresh (rad, geodesic angle of relative rotation)."""
    from graspnetAPI import GraspGroup
    order = np.argsort(-gg.scores)
    T, R = gg.translations[order], gg.rotation_matrices[order]
    keep = []
    for i in range(len(order)):
        ok = True
        for j in keep:
            if np.linalg.norm(T[i] - T[j]) < trans_thresh:
                c = (np.trace(R[i].T @ R[j]) - 1.0) / 2.0
                if np.arccos(np.clip(c, -1, 1)) < rot_thresh:
                    ok = False
                    break
        if ok:
            keep.append(i)
    return GraspGroup(gg.grasp_group_array[order[keep]])


def to_json(gg):
    return [dict(score=float(g.score), width=float(g.width), height=float(g.height), depth=float(g.depth),
                 rotation=g.rotation_matrix.tolist(), translation=g.translation.tolist()) for g in gg]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--num_point", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    d = np.load(a.inp)
    t0 = time.time()
    net = load_net()
    t1 = time.time()
    gg = propose(d["points"], net, num_point=a.num_point, top=a.top, seed=a.seed)
    t2 = time.time()
    out = to_json(gg)
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"{len(out)} grasps -> {a.out}; load {t1-t0:.1f}s infer+collision+nms {t2-t1:.2f}s; "
          f"peak GPU mem {torch.cuda.max_memory_allocated()/2**20:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
