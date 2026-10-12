"""perception_zono: point cloud -> zonotope per object (observations only). SIMULATION ONLY, CPU only; no hardware, no network.

geometry   analytic primitives (box / cylinder / sphere), object library, true SDF / surface / volume, MJCF scene builder
sensing    ray-cast rig (head_aim + wrist cams, reuses tray_rep.perception.depth_sim), full depth maps for free-space carving
segment    voxel filter, kNN outlier filter, GT segmentation by ray-hit geom, plane removal + DBSCAN (own, scipy cKDTree)
zonoutil   H-rep, containment, exact volume, point distance for zonotopes and unions
fit        PCA box, upright box, convex-hull LP zonotope (+ Girard reduction), split into a union (quantile / PA edge detector)
score      containment, over-coverage, Clear / Inside margin error, timing
run_study  the driver (multiprocessing, <= 8 workers)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
