"""nero/base/swerve.py -- swerve controller for the YOR base in MuJoCo (SIMULATION ONLY).

Model (build_variant.variant_xml(base="swerve")): freejoint `floating_base`; per module m in (front_left, back_left, front_right, back_right)
a steer hinge `{m}_steer` (position servo kp=10 kv=1 over +-pi, YOR text) and a drive hinge `drive_{m}` (velocity servo kv=1, no force limit in YOR).
Geometry is READ from the compiled model (MODEL): steer pivots, wheel offset from the pivot (along the axle), wheel radius = hub height - lowest
wheel vertex.  Steer 0 = the wheel rolls along robot +x; +steer = CCW about +z; +drive = roll forward (checked in smoke.py).

Labels: MEASURED = YOR source value; MODEL = derived from the compiled MJCF; ASSUMED = chosen here (listed in LIMITS / README of the smoke report).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco

MODULES = ("front_left", "back_left", "front_right", "back_right")

# ---- limits ---------------------------------------------------------------------------------------------------------------------------
# MEASURED YOR/robot/yor.py:55-56 + base_motor.py:323-324: the real Base defaults max_vel = max_accel = (1.0, 1.0, 1.57) (m/s, m/s, rad/s; m/s^2, m/s^2, rad/s^2).
YOR_MAX_VEL = np.array([1.0, 1.0, 1.57])
YOR_MAX_ACCEL = np.array([1.0, 1.0, 1.57])


@dataclass
class Limits:
    # ASSUMED (carrying a tray of mugs: well inside the YOR caps; YOR teleop presets are 0.25 / 0.50 / 0.75 m/s, 1.57 rad/s (joystick.py:52-56))
    v_max: float = 0.25          # m/s   ASSUMED
    a_max: float = 0.25          # m/s^2 ASSUMED
    w_max: float = 0.5           # rad/s ASSUMED
    alpha_max: float = 0.5       # rad/s^2 ASSUMED
    steer_rate_max: float = 6.0  # rad/s ASSUMED: slew limit of the steer TARGET (avoids a step command into the position servo)
    # ASSUMED gains of the path follower (P on the pose error, on top of the profile feed-forward)
    kp_pos: float = 2.0          # 1/s
    kp_yaw: float = 2.0          # 1/s
    v_floor: float = 1e-3        # m/s: below this a module keeps its steer angle and its wheel stops
    ki_wheel: float = 4.0        # 1/s ASSUMED: integral gain on the wheel-speed error (YOR's drive servo is a bare kv=1 velocity servo; under the contact-patch
                                 # torsional friction of an in-place turn it droops ~18 %)
    i_max: float = 3.0           # rad/s ASSUMED: clamp of that integral term


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


class SwerveBase:
    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, lim: Limits | None = None):
        self.m, self.d, self.lim = m, d, lim or Limits()
        mujoco.mj_forward(m, d)
        self.steer_q = [m.jnt_qposadr[m.joint(f"{n}_steer").id] for n in MODULES]
        self.steer_v = [m.jnt_dofadr[m.joint(f"{n}_steer").id] for n in MODULES]
        self.drive_q = [m.jnt_qposadr[m.joint(f"drive_{n}").id] for n in MODULES]
        self.drive_v = [m.jnt_dofadr[m.joint(f"drive_{n}").id] for n in MODULES]
        self.a_steer = [m.actuator(f"{n}_steer_ctrl").id for n in MODULES]
        self.a_drive = [m.actuator(f"drive_{n}_ctrl").id for n in MODULES]
        self.hub = [m.body(h).id for h in ("MK4_Colson_Hub_4_X_1.5", "MK4_Colson_Hub_4_X_1.5_2", "MK4_Colson_Hub_4_X_1.5_3", "MK4_Colson_Hub_4_X_1.5_4")]
        # pivots (MODEL) in the world frame at the initial pose; robot frame := world axes rotated so that +x = (front - back)
        P = np.array([d.xanchor[m.joint(f"{n}_steer").id] for n in MODULES])
        H = np.array([d.xanchor[m.joint(f"drive_{n}").id] for n in MODULES])
        self._yaw0 = self._yaw_from(P)
        c = P[:, :2].mean(0)
        R0 = self._rot(self._yaw0)
        self.r = (P[:, :2] - c) @ R0                      # pivot offsets in the robot frame (MODEL)
        off_w = H[:, :2] - P[:, :2]
        self.off = np.linalg.norm(off_w, axis=1)           # wheel centre offset from the pivot, along the axle (MODEL)
        self.off_sign = np.sign((off_w @ R0)[:, 1])        # +1: the offset is on +y_body at steer 0
        self.R_wheel = self._wheel_radius()                # MODEL
        self.th_cmd = np.zeros(4)
        self._have_cmd = False
        self._wi = np.zeros(4)                             # integral of the wheel-speed error [rad/s]
        self.yaw_off = 0.0
        self._v = np.zeros(3)                              # current (rate-limited) body twist command
        self.yaw_nom = None                                # NOMINAL heading: turns add to it, lines hold it (no accumulation of the measured yaw error)

    # ---- model readout --------------------------------------------------------------------------------------------------
    @staticmethod
    def _rot(yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s], [s, c]])

    def _yaw_from(self, P):
        f = P[[0, 2], :2].mean(0) - P[[1, 3], :2].mean(0)
        return float(np.arctan2(f[1], f[0]))

    def _wheel_radius(self):
        m, d = self.m, self.d
        out = []
        for h, n in zip(self.hub, MODULES):
            g = [g for g in range(m.ngeom) if m.geom_bodyid[g] == h and m.geom_contype[g] | m.geom_conaffinity[g]][0]
            me = m.geom_dataid[g]
            vs = m.mesh_vert[m.mesh_vertadr[me]:m.mesh_vertadr[me] + m.mesh_vertnum[me]]
            w = vs @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            out.append(d.xpos[h][2] - w[:, 2].min())
        return float(np.mean(out))

    def pivots(self):
        d, m = self.d, self.m
        return np.array([d.xanchor[m.joint(f"{n}_steer").id][:2] for n in MODULES])

    def pose(self):
        """(x, y, yaw) of the module-pivot centre, absolute yaw of the robot +x axis (world frame; sim truth; no odometry error is modelled -- ASSUMED)."""
        P = np.array([self.d.xanchor[self.m.joint(f"{n}_steer").id] for n in MODULES])
        return np.array([*P[:, :2].mean(0), self._yaw_from(P)])

    def steer(self):
        return np.array([self.d.qpos[i] for i in self.steer_q])

    def wheel_slip(self):
        """per-wheel horizontal speed [m/s] of the tyre contact point (zero = pure rolling): v_hub + w x (-R z)."""
        m, d = self.m, self.d
        out, res = [], np.zeros(6)
        for h in self.hub:
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, h, res, 0)
            w, v = res[:3], res[3:]
            vc = v + np.cross(w, np.array([0, 0, -self.R_wheel]))
            out.append(float(np.linalg.norm(vc[:2])))
        return np.array(out)

    # ---- inverse kinematics ---------------------------------------------------------------------------------------------
    def module_cmd(self, tw):
        """body twist tw = (vx, vy, wz) [robot frame] -> (steer targets [rad], drive speeds [rad/s]) with wheel reversal (steer error <= pi/2)."""
        vx, vy, wz = tw
        th_now = self.steer()
        thd_now = np.array([self.d.qvel[i] for i in self.steer_v])
        th, qd = np.zeros(4), np.zeros(4)
        for i in range(4):
            rx, ry = self.r[i]
            v = np.array([vx - wz * ry, vy + wz * rx])     # pivot velocity (robot frame)
            sp = float(np.linalg.norm(v))
            if sp < self.lim.v_floor:
                th[i], qd[i] = th_now[i], 0.0
                continue
            t = float(np.arctan2(v[1], v[0]))
            e = float(_wrap(t - th_now[i]))
            sgn = 1.0
            if abs(e) > np.pi / 2:                         # reverse the wheel instead of swinging the module through > 90 deg
                t, e, sgn = float(_wrap(t + np.pi)), float(_wrap(e + np.pi)), -1.0
            th[i] = t
            # wheel centre sits `off` from the pivot along the axle: v_hub.u = v_pivot.u - off*sign*steer_rate (MODEL, derived in the module doc)
            vu = sgn * sp * max(np.cos(e), 0.0) - self.off_sign[i] * self.off[i] * thd_now[i]
            qd[i] = vu / self.R_wheel
        return th, qd

    def apply(self, tw, dt):
        """rate-limited twist -> steer/drive controls (writes d.ctrl).  Returns the steer targets."""
        L = self.lim
        th, qd = self.module_cmd(tw)
        mx = L.steer_rate_max * dt                         # slew-limit the steer TARGET (ASSUMED rate)
        prev = self.th_cmd if self._have_cmd else self.steer()
        self.th_cmd = prev + np.clip(_wrap(th - prev), -mx, mx)
        self._have_cmd = True
        th = self.th_cmd
        for i in range(4):
            w = self.d.qvel[self.drive_v[i]]
            self._wi[i] = np.clip(self._wi[i] + L.ki_wheel * (qd[i] - w) * dt, -L.i_max, L.i_max)
            self.d.ctrl[self.a_steer[i]] = th[i]
            self.d.ctrl[self.a_drive[i]] = qd[i] + self._wi[i]
        return th

    def stop(self):
        th_now = self.steer()
        self._wi[:] = 0.0
        self._have_cmd = False
        for i in range(4):
            self.d.ctrl[self.a_steer[i]] = th_now[i]
            self.d.ctrl[self.a_drive[i]] = 0.0


# ---- trapezoid profile ----------------------------------------------------------------------------------------------------------------
def trap_profile(dist, vmax, amax):
    """1-D trapezoid / triangle profile for |dist| -> (T, f) with f(t) = (s, v) [signed]."""
    sgn = 1.0 if dist >= 0 else -1.0
    D = abs(dist)
    ta = vmax / amax
    if D < vmax * ta:                                      # triangle
        ta = np.sqrt(D / amax)
        vp, tc = amax * ta, 0.0
    else:
        vp, tc = vmax, (D - vmax * ta) / vmax
    T = 2 * ta + tc

    def f(t):
        t = min(max(t, 0.0), T)
        if t < ta:
            s, v = 0.5 * amax * t * t, amax * t
        elif t < ta + tc:
            s, v = 0.5 * amax * ta * ta + vp * (t - ta), vp
        else:
            u = T - t
            s, v = D - 0.5 * amax * u * u, amax * u
        return sgn * s, sgn * v
    return T, f


class PathFollower:
    """turn_in_place(dyaw) / drive_line(dist, heading_rel) as closed-loop references on the sim pose.  step() returns the body twist."""

    def __init__(self, sb: SwerveBase):
        self.sb = sb
        self.kind = None

    def turn_in_place(self, dyaw):
        L = self.sb.lim
        p = self.sb.pose()
        if self.sb.yaw_nom is None:
            self.sb.yaw_nom = p[2]
        p[2] = self.sb.yaw_nom                              # the reference starts from the nominal heading
        self.kind, self.p0 = "turn", p
        self.T, self.f = trap_profile(dyaw, L.w_max, L.alpha_max)
        self.t = 0.0
        self.goal = p.copy(); self.goal[2] = p[2] + dyaw
        self.goal[:2] = self.sb.pose()[:2]
        self.sb.yaw_nom = float(self.goal[2])
        self.c0 = self.sb.pose()[:2].copy()
        return self.T

    def drive_line(self, dist, rel_heading=0.0):
        """drive `dist` m along the direction (robot +x rotated by rel_heading) at constant heading (no steering of the body)."""
        L = self.sb.lim
        p = self.sb.pose()
        if self.sb.yaw_nom is None:
            self.sb.yaw_nom = p[2]
        p[2] = self.sb.yaw_nom
        self.kind, self.p0 = "line", p
        self.dir = np.array([np.cos(p[2] + rel_heading), np.sin(p[2] + rel_heading)])
        self.T, self.f = trap_profile(dist, L.v_max, L.a_max)
        self.t = 0.0
        self.goal = p.copy(); self.goal[:2] = self.sb.pose()[:2] + dist * self.dir
        self.p0 = self.p0.copy(); self.p0[:2] = self.sb.pose()[:2]
        return self.T

    def step(self, dt):
        L, sb = self.sb.lim, self.sb
        p = sb.pose()
        self.t += dt
        s, v = self.f(self.t)
        if self.kind == "turn":
            ref_yaw = self.p0[2] + s
            wz = v + L.kp_yaw * float(_wrap(ref_yaw - p[2]))
            e = self.c0 - p[:2]                              # hold the rotation centre
            vw = L.kp_pos * e
            vb = SwerveBase._rot(p[2]).T @ vw
            tw = np.array([vb[0], vb[1], wz])
        else:
            ref = self.p0[:2] + s * self.dir
            e = ref - p[:2]
            vw = v * self.dir + L.kp_pos * e
            vb = SwerveBase._rot(p[2]).T @ vw
            tw = np.array([vb[0], vb[1], L.kp_yaw * float(_wrap(self.p0[2] - p[2]))])
        return self._limit(tw, dt)

    def _limit(self, tw, dt):
        L, sb = self.sb.lim, self.sb
        cur = sb._v
        da = np.array([L.a_max, L.a_max, L.alpha_max]) * dt
        new = cur + np.clip(tw - cur, -da, da)
        new[:2] = np.clip(new[:2], -L.v_max, L.v_max)
        new[2] = np.clip(new[2], -L.w_max, L.w_max)
        sb._v = new
        return new

    def hold_twist(self, dt):
        """P control on the pose error to this segment's goal (used in the rest gaps after a segment: removes its residual error)."""
        L, sb = self.sb.lim, self.sb
        p = sb.pose()
        e = self.goal[:2] - p[:2]
        vb = SwerveBase._rot(p[2]).T @ (L.kp_pos * e)
        return self._limit(np.array([vb[0], vb[1], L.kp_yaw * float(_wrap(self.goal[2] - p[2]))]), dt)

    def done(self):
        return self.t >= self.T

    def error(self):
        p = self.sb.pose()
        return np.linalg.norm(p[:2] - self.goal[:2]), float(_wrap(p[2] - self.goal[2]))
