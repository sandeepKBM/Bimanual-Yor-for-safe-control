"""people_track -- honest person perception for the teleop shield.  SIMULATION ONLY (MuJoCo ray casting + numpy/scipy).

Replaces the shield's ground-truth `sim.people_state` with detections from the robot's own simulated head camera
(depth_sim/perception_bridge noise model), a per-person constant-velocity Kalman tracker, and an uncertainty-inflated keep-out.
See README-level notes in camera.py / detectors.py / tracker.py / perception.py / shield_pt.py and the runner run_pt.py.
"""
