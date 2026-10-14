"""gpusense -- GPU depth sensing / mapping / distance field (SIMULATION ONLY).  See core.py, proc.py, mapping.py, adapters.py."""
import os

GPUSENSE = os.environ.get("BIMANUAL_GPUSENSE", "0") == "1"      # master flag; defaults OFF (bit-identical CPU paths)

HOOKS = """
Hook lines (owner applies; every one is flag-gated by BIMANUAL_GPUSENSE=1 and OFF by default):

1. people_track/perception.py  PerceptionStack.__init__
     - self.cam = CAM.HeadCamera(sim, CAM.head_spec(W=c.W, H=c.H, rate_hz=c.rate_hz, latency_ticks=c.latency_ticks))
     + from world_model.v2.house.gpusense.adapters import make_head_camera
     + self.cam = make_head_camera(sim, CAM.head_spec(W=c.W, H=c.H, rate_hz=c.rate_hz, latency_ticks=c.latency_ticks))
   (make_head_camera returns the unchanged CPU HeadCamera when the flag is off; GpuHeadCamera.render() has the same dict + static_depth().)
   Optional async (hides the render inside latency_ticks): in PerceptionStack.step replace
        frame = self.cam.render(k, seed); dets = self.det.detect(frame, ...); queue.append((k+lat, k, dets))
   by  h = self.cam.submit(k, seed) at the capture tick, queue.append((k+lat, k, h)); and at delivery
        frame = self.cam.collect(h); dets = self.det.detect(frame, self.cam, sim)   (static_depth uses the GPU image rendered with the frame).

2. grasp/assist.py  GraspAssist.__init__ (and the two cam.capture sites unchanged)
     - self.cams = {s: WristCam(m, s) for s in SIDES}
     + from world_model.v2.house.gpusense.adapters import get_hub, GpuWristCam
     + self.cams = ({s: GpuWristCam(m, s, get_hub(sim)) for s in SIDES} if GPUSENSE else {s: WristCam(m, s) for s in SIDES})
   async: fr = cam.capture(...)  ->  h = cam.submit_capture(sim.d, n, seed, sim.t, roi) at capture, fr = cam.finish_capture(h) at delivery.

3. teleop/sim.py  TeleopSim.__init__ (occ_source="perceived") -- map service instead of the one-off build:
     + hub = get_hub(self, map_cfg=MapCfg(), ...); self.map_svc = MapService(self, hub)
     every 10th tick (5 Hz):  self.map_svc.update(self.ctrl_pose());  g = self.map_svc.poll();  if g is not None: self.occ_ctrl = g; shield.occ = g
   g is a teleop.scene.OccGrid whose gradient cache is pre-filled -> GM.grid_lookup_grad(g, x, y) in shield._keep_rows is unchanged.
   Offline map build:  loc.mapper.build_map -> gpusense.mapping.build_map_gpu(scene) (returns the same PerceivedMap).

4. warm-up: Hub(...) blocks until the child reports READY (contexts, kernels, first renders).  Call it before the episode clock starts.
"""
