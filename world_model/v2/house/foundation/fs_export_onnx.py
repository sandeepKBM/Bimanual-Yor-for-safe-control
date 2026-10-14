"""Export FoundationStereo to a fixed-shape ONNX (for TensorRT).  Run in the `foundation` env:
   XFORMERS_DISABLED=1 python fs_export_onnx.py --ckpt 23-51-11 --height 384 --width 640 --iters 12 --out X.onnx
Wraps upstream scripts/make_onnx.py's class (torch>=2.6 needs weights_only=False)."""
import argparse, os, sys
FS = "/data/local/ss5772/models/FoundationStereo"
sys.path.insert(0, FS); sys.path.insert(0, FS + "/scripts")
import torch
from omegaconf import OmegaConf
from make_onnx import FoundationStereoOnnx

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="23-51-11"); ap.add_argument("--height", type=int, default=384)
ap.add_argument("--width", type=int, default=640); ap.add_argument("--iters", type=int, default=12)
ap.add_argument("--max-disp", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.autograd.set_grad_enabled(False)
ck = f"{FS}/pretrained_models/{a.ckpt}/model_best_bp2.pth"
cfg = OmegaConf.load(os.path.dirname(ck) + "/cfg.yaml")
if "vit_size" not in cfg: cfg["vit_size"] = "vitl"
cfg["valid_iters"] = a.iters
if a.max_disp:
    cfg["max_disp"] = a.max_disp
m = FoundationStereoOnnx(cfg)
m.load_state_dict(torch.load(ck, weights_only=False)["model"]); m.cuda().eval()
l = torch.randn(1, 3, a.height, a.width).cuda(); r = torch.randn(1, 3, a.height, a.width).cuda()
torch.onnx.export(m, (l, r), a.out, opset_version=16, input_names=["left", "right"], output_names=["disp"])
print("saved", a.out)
