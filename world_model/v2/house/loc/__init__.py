"""house/loc -- simulated LOCALIZATION + MAPPING for the teleoperated YOR.  SIMULATION ONLY (no robot SDK, no ZED SDK, no ROS, no network).

Replaces two privileged reads of the control path (the shield / zonotope base filter / capture pose):
  1. estimator.PoseEstimator : base pose from simulated swerve wheel odometry + ZED visual-inertial odometry (VIO), fused.  Truth is only the
                               SENSOR SOURCE inside the estimator (and the evaluation reference); control consumers read `.pose`.
  2. mapper                  : occupancy grid built from the head-ZED depth (stereo noise law of tray_rep.perception.depth_sim) raycast into a
                               log-odds grid in the ESTIMATED pose frame (a scripted mapping pass), + map-vs-truth error metrics.
All sensor parameters are ASSUMPTIONS (see LocCfg / MapCfg); the real YOR odometry/ZED error statistics are not known to us.
"""
