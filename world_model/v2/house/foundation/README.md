# foundation/ -- FoundationStereo + FoundationPose (model-free) backends

SIMULATION ONLY. Models run in their own process (conda env `foundation`); the mujoco env talks to them over a Unix-domain
socket (file path, no TCP) + POSIX shared memory. Each client spawns its server and `warmup()` blocks until READY
(weights loaded + N dummy inferences). GPUs: 5 (stereo) and 6 (pose) only.

## Install recipe (done 2026-10-12)
- Env: `/data/local/ss5772/conda_envs/foundation` (mamba, conda-forge: python 3.11, cmake, ninja, eigen, boost, pybind11).
- torch 2.8.0+cu128 / torchvision 0.23.0 (PyTorch index). Includes sm_75 (2080 Ti). nvcc = /data/local/ss5772/cuda_home (12.9);
  builds must use system gcc 11 (`CC=/usr/bin/gcc`, conda's gcc 15 is rejected by nvcc) and `TORCH_CUDA_ARCH_LIST=7.5`.
- Code + weights: `/data/local/ss5772/models/{FoundationStereo,FoundationPose}` (git clone NVlabs; weights via gdown from the README Drive links;
  FS 23-51-11 = ViT-L, 11-33-40 = ViT-S; FP refiner 2023-10-28-18-33-37, scorer 2024-01-11-20-02-45). URLs in `/data/local/ss5772/models/DOWNLOADS.log`.
- FoundationPose build: `mycpp` (cmake, `build_all_conda.sh`); `bundlesdf/mycuda` (setup.py: changed `-std=c++14` -> `c++17`; common.cu `.type()` -> `.scalar_type()`).
- What was not installable and the workaround:
  - `nvdiffrast`: not on PyPI under that name -> `diffrp-nvdiffrast==0.3.3.1` (PyPI repackage of NVlabs nvdiffrast 0.3.3; JIT-compiles on first use).
  - `pytorch3d`: not on PyPI -> pure-python shim of `pytorch3d.transforms` (copied from the robotwin env) at `/data/local/ss5772/models/shims`, on sys.path via .pth; renderer/structures are stubs.
  - `kaolin`: not installable -> `dense_octree.py` (dense occupancy-grid replacement for the NOF's octree ray tracing; numerics differ slightly from upstream).
  - newer trimesh lacks `remove_duplicate_faces` (patched at runtime); added `xatlas`, `rtree`, `tensorrt-cu12==10.13.3.9`, `onnxscript`.
- FoundationStereo is monkeypatched (`fs_patches.py`) so forward() is CUDA-graph capturable (bit-identical to eager). TensorRT: `fs_export_onnx.py` + `fs_build_trt.py`
  (ViT-L 384x640 i12 FP16 engine at `/data/local/ss5772/models/trt/`; the build took ~60 min under CPU load).

## API
- `stereo_fs.Client(W,H,gpu=5,engine='trt'|'graph'|'eager',iters=12,ckpt,vit,max_disp).warmup(); .depth(left,right,K,baseline) -> (depth_m, valid, ms)`
  (matches `house/stereo/backends.FoundationBackend`). Engine 'trt' falls back to 'graph' if no plan exists.
- `pose_fp.Client(sizes,gpu=6).warmup(); .onboard(name,rgbs,depths,masks,cam_in_obj,K); .register(name,rgb,depth,mask,K,up_cam=None); .track(name,rgb,depth,K,iters=2,score=False); .init_from_prior(...)`.
  FoundationPose must see <= ~640x360 images (1280x720 OOMs the 11 GB card): crop an ROI and shift K (`bench_pose.crop_roi`).

## Onboarding masks
Non-privileged: `masks.floor_gate_mask` (depth + known floor plane + placement seed; IoU 0.98 vs rendered segmentation, which was used only to score). Depth = MuJoCo buffer + 0.5 mm noise
(not stereo depth). The first-frame register mask in the benchmark IS the rendered segmentation (privileged stand-in for a SAM/VLM prompt).

## Measured (GPU 5/6 = 2080 Ti, CPU load avg 50-85, timings include that contention)
FoundationStereo, median ms (Hz), client round trip:
| config | ms | Hz |
|---|---|---|
| ViT-L i12, 640x360, TRT FP16 | 303 | 3.3 |
| ViT-L i12, 640x360, CUDA graph | 407 (max_disp 224: 334) | 2.5-3.0 |
| ViT-L i12, 320x180, graph, max_disp 128 | 95 | 10.6 |
| ViT-S i8, 320x180, graph, max_disp 128 | 52 | 19 |
| ViT-S i8, 640x360, graph, max_disp 224 | 193 | 5.2 |
Eager (no graph) is CPU-launch-bound: ~330 ms at 320x180. 1280x720 graph capture OOMs on the shared card (not benchmarked). Server READY 8-37 s.
Depth error vs MuJoCo depth (ViT-L 640x360 TRT, speckle): median 0.1 cm at 0.3-2 m, 0.4 cm 2-3 m, 0.6 cm 3-5 m; ~6.5% pixels invalid (left strip). Flat walls (no texture): FS invalid 0.5% / median ~2 cm on walls vs
CPU SGBM 76% invalid. CPU SGBM at 640x360: ~200-290 ms (22% invalid with speckle, 72% with flat colours); at 320x180: 23 ms. Raw JSONs: `/data/local/ss5772/molmospaces/percept/foundation/bench/`.

FoundationPose (model-free, NOF 1000 steps, 16 views): READY 12 s; onboarding tray 200 s, cup 107 s (CPU-contended); register ~2.0 s; track 38 ms cup / 51 ms tray at 2 refiner iters (+30 ms with score); 3.6 GB.
Over the recorded carry (object nearly static in camera; GT depth, privileged first mask): tray 2.9 mm median / 1.6 deg axis, cup 2.5 mm / 0.9 deg, no failures.
Scripted chirp (up to ~240 deg/s, +-4 cm; GT depth): cup 1.9 mm / 1.4 deg, no failures; tray FAILS: global register flips / mis-orients it (needs a gravity-up prior; even then ~6 cm
position error), FK-prior init gives ~20 mm median with 36% frames lost; a CAD box mesh is no better, so the limit is the thin textureless plate, not the NOF mesh.
With FoundationStereo depth (640x360): cup 1.8 mm / 2.0 deg (0% fail); tray 51-65 mm, 69-93% lost.

## Unfinished
TRT engines for 192x320 (ViT-S d128 / ViT-L d128) were exported but the builds were stopped; no 1280x720 stereo numbers; no cup symmetry handling; scripts live in the job scratch dir
(`.claude/jobs/b9577455/tmp/foundation/scripts`). Licence: NVIDIA Source Code License (non-commercial / research use only, redistribution under the same licence; the FS NVIDIA-TAO variant is the commercial one, not used).
