"""grasp/fuse.py -- fusion of tray-pose measurements for the friction-grasp assist (left wrist cam, right wrist cam, EXTERNAL trackers).  SIMULATION ONLY.

Every measurement is a world-frame pose (p, R) of the (static, not yet gripped) tray with a 6x6 covariance expressed in the TRAY frame
(axes: x = edge normal / approach axis, y = along the edge, z = up, then rotations about x, y, z).  `TrayFusion` is an information filter on the
static pose: sum of Lambda_i, eta_i with the covariances rotated into the world frame.  The per-axis weighting is what makes the fusion useful:
the wrist camera sees the edge normal and the height well but the along-edge position badly (measured, see `WRIST_SIGMA`), another camera
that sees the whole tray (head camera) has the opposite profile.

EXTERNAL TRACKER HOOK
    ExternalTrayPose(p, R, cov, t_meas, source)       pose of the tray centre (top-surface centre, same convention as estimate.PoseEst.p)
    GraspAssist.set_external(src)                      src is a callable  src(t, sim) -> ExternalTrayPose | None  (polled at the camera rate in PRE)
Any estimate older than `max_age_s` is dropped (the tray only stays static until it is gripped).  A FoundationPose head-camera tracker
(house/foundation/pose_fp.py, built by another agent) connects by wrapping its output in `from_cam_pose`:

    def src(t, sim):
        T_cam_obj, cov_cam = tracker.latest()          # 4x4 object-in-camera pose of the tray + its 6x6 covariance in the CAMERA frame (or None)
        T_w_cam = head_camera_world_pose(sim)          # head camera extrinsics from the arm/torso kinematics (NOT sim ground truth)
        return from_cam_pose(T_w_cam, T_cam_obj, cov_cam, t_meas=tracker.t, source="foundationpose")

`MockHeadTracker` below is the stand-in used by the tests / smoke: truth + Gaussian noise of a given covariance (PRIVILEGED: it reads the tray
body; it only stands in for the real tracker's output statistics).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from world_model.v2.house.teleop.arm import _log_R, _rotvec_to_R

# 1-sigma of the wrist-camera tray estimate in the tray frame (x edge normal [m], y along edge [m], z [m], rot x, y [rad], yaw [rad]), per hand.
# MEASURED offline (grasp/look_scan.py, 12-16 trials x 3 captures; sigma ~ 1.25 x mean|e|).  Hover pose (centre line): left x 3.0 y 13.6 z 0.2 mm
# yaw 1.4 deg, right x 1.9 y 13.0 z 0.7 mm yaw 1.3 deg -- the along-edge error IS the prior (the placement offset), i.e. unobserved.  Lateral look
# (left -5 cm, right +6 cm along the edge): left x 3.4 y 10.0, right x 1.4 y 2.8 mm.  Roll / pitch are fixed by the levelled prior.
WRIST_SIGMA = {
    "left": np.array([3.8e-3, 17.0e-3, 0.5e-3, math.radians(1.0), math.radians(1.0), math.radians(2.0)]),
    "right": np.array([2.4e-3, 16.5e-3, 0.9e-3, math.radians(1.0), math.radians(1.0), math.radians(1.7)]),
}
WRIST_SIGMA_LOOK = {
    "left": np.array([4.2e-3, 12.5e-3, 0.5e-3, math.radians(1.0), math.radians(1.0), math.radians(2.0)]),
    "right": np.array([1.8e-3, 3.6e-3, 0.9e-3, math.radians(1.0), math.radians(1.0), math.radians(1.5)]),
}
ROLLPITCH_SIGMA = math.radians(1.0)


@dataclass
class ExternalTrayPose:
    p: np.ndarray                    # (3,) world, tray centre (top surface)
    R: np.ndarray                    # (3,3) world
    cov: np.ndarray                  # (6,6) in the TRAY frame: [x y z (m)  rx ry rz (rad)]
    t_meas: float                    # sim time of the measurement
    source: str = "external"


def from_cam_pose(T_w_cam, T_cam_obj, cov_cam, t_meas, source="external", R_world_cov=None):
    """camera-frame pose + camera-frame covariance -> ExternalTrayPose (cov rotated into the tray frame)."""
    T = np.asarray(T_w_cam, float) @ np.asarray(T_cam_obj, float)
    R, p = T[:3, :3], T[:3, 3]
    Rc = np.asarray(T_w_cam, float)[:3, :3]
    B = np.kron(np.eye(2), (R.T @ Rc))          # camera-frame axes -> tray-frame axes
    cov = B @ np.asarray(cov_cam, float) @ B.T
    return ExternalTrayPose(p.copy(), R.copy(), cov, float(t_meas), source)


def wrist_cov(side, conf, scale=1.0, look=False):
    """tray-frame covariance of a wrist-camera estimate: calibrated per-axis sigma, inflated for low confidence (conf 0.85 = nominal)."""
    sig = (WRIST_SIGMA_LOOK if look else WRIST_SIGMA)[side] * scale / math.sqrt(max(conf, 0.3) / 0.85)
    return np.diag(sig ** 2)


class TrayFusion:
    """static-pose information filter.  Reference rotation = the first measurement's (levelled about world z)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.items = {}                               # key -> (p, R, cov_tray, t)

    def set(self, key, p, R, cov, t):
        """replace the source `key`'s contribution (each source is already a running estimate of the static pose)."""
        self.items[key] = (np.asarray(p, float).copy(), np.asarray(R, float).copy(), np.asarray(cov, float).copy(), float(t))

    def drop(self, key):
        self.items.pop(key, None)

    def estimate(self, t=None, max_age_s=None, keys=None):
        """(p, R, cov_tray, used keys) or None."""
        use = {k: v for k, v in self.items.items() if (keys is None or k in keys) and (max_age_s is None or t is None or t - v[3] <= max_age_s)}
        if not use:
            return None
        k0 = next(iter(use))
        R0 = use[k0][1]
        psi = math.atan2(R0[1, 0], R0[0, 0])
        Rb = np.array([[math.cos(psi), -math.sin(psi), 0.0], [math.sin(psi), math.cos(psi), 0.0], [0.0, 0.0, 1.0]])    # reference tray frame
        L = np.zeros((6, 6))
        eta = np.zeros(6)
        p0 = use[k0][0]
        for k, (p, R, cov, _) in use.items():
            x = np.concatenate([Rb.T @ (p - p0), Rb.T @ _log_R(R @ R0.T)])        # in the reference tray frame, relative to (p0, R0)
            # the measurement covariance is in the SOURCE's own tray frame; rotate into the reference tray frame
            Rs = Rb.T @ R
            B = np.kron(np.eye(2), Rs)
            Lam = np.linalg.inv(B @ cov @ B.T + 1e-12 * np.eye(6))
            L += Lam
            eta += Lam @ x
        C = np.linalg.inv(L)
        x = C @ eta
        p = p0 + Rb @ x[:3]
        R = _rotvec_to_R(Rb @ x[3:]) @ R0
        return p, R, C, list(use)


class MockHeadTracker:
    """stand-in for an external (head-camera) tray tracker: the TRUE tray pose + Gaussian noise of covariance `cov` (tray frame), at `rate_hz`.
    PRIVILEGED (reads the tray body): it only reproduces the statistics a real tracker is expected to have."""

    def __init__(self, cov=None, rate_hz=5.0, seed=0, bias_xyz=(0.0, 0.0, 0.0)):
        sig = np.array([6e-3, 3e-3, 3e-3, math.radians(2), math.radians(2), math.radians(2)])
        self.cov = np.diag(sig ** 2) if cov is None else np.asarray(cov, float)
        self.period = 1.0 / rate_hz
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 4242]))
        self.bias = np.asarray(bias_xyz, float)
        self._t_last = -1e9

    def __call__(self, t, sim):
        if t - self._t_last < self.period - 1e-9:
            return None
        self._t_last = t
        p = sim.d.xpos[sim.tray].copy()
        R = sim.d.xmat[sim.tray].reshape(3, 3).copy()
        e = self.rng.multivariate_normal(np.zeros(6), self.cov)
        p = p + R @ (e[:3] + self.bias)
        R = R @ _rotvec_to_R(e[3:])
        return ExternalTrayPose(p, R, self.cov.copy(), float(t), "mock_head_tracker")
