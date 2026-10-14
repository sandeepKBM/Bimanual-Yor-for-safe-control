"""Build a TensorRT FP16 engine from an ONNX file (foundation env).
usage: fs_build_trt.py in.onnx out.plan [workspace_GB]"""
import sys
import time

import tensorrt as trt

onnx_p, plan_p = sys.argv[1], sys.argv[2]
ws = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
log = trt.Logger(trt.Logger.WARNING)
b = trt.Builder(log)
net = b.create_network()
p = trt.OnnxParser(net, log)
if not p.parse_from_file(onnx_p):
    for i in range(p.num_errors):
        print(p.get_error(i))
    sys.exit(1)
cfg = b.create_builder_config()
cfg.set_flag(trt.BuilderFlag.FP16)
cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(ws * 2 ** 30))
t = time.time()
ser = b.build_serialized_network(net, cfg)
if ser is None:
    sys.exit("build failed")
open(plan_p, "wb").write(ser)
print("engine", plan_p, len(ser) / 1e6, "MB build s", round(time.time() - t))
