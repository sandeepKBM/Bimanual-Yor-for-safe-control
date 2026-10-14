"""dynamics.py -- batched torch rollouts of the planar base + carried cup.  SIMULATION ONLY.

Base: commanded twist u = (vx, vy, wz) -> slew-limited actual twist (|dv| <= a_hw dt per axis, |u| <= v_max), midpoint integration.
  frame == "body"  : u is a BODY twist (teleop/base.py BaseSim.advance, n_sub = 1);
  frame == "world" : u is a WORLD twist of the base origin + yaw rate (zono_rows BaseFilter / smoke_s2: pose += u dt).
Cup: a point fixed at base-frame position r_c.  Its world acceleration is the finite difference of the exact point velocity
  v_pt = v_o + wz e_z x (R(yaw) r_c)       (includes the centripetal term through the change of R).
Cup contact (shield.py): the cup keeps its contact iff |g_in| <= r N/m with  g_in = -a_h + g c_h  (flat-ish tray, reach.gin_of linearised: ex ~ (1,0,-c_x)),
  r = min(kappa mu, tan theta_spill).  c_h = tray-axis horizontal part (base frame), steered by the optional tilt channels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

G = 9.81


@dataclass
class BaseModel:
    dt: float = 0.1
    dts: tuple | None = None          # optional NON-uniform step schedule [s] (len H): fine first steps = no command lag, coarse tail = long horizon
    frame: str = "body"
    a_hw: tuple = (4.0, 4.0, 8.0)
    v_max: tuple = (0.9, 0.45, 1.2)
    tilt_cap: float = 0.05            # rad, |c_h| per axis (jog governor theta_tray)
    tilt_rate: float = 0.15           # rad/s, jog_w_max
    use_tilt: bool = False

    def dt_vec(self, H, device, dtype=torch.float32):
        k = ("dt", H, str(device))
        c = self.__dict__.setdefault("_c", {})
        if k not in c:
            v = list(self.dts) if self.dts is not None else [self.dt] * H
            assert len(v) == H
            c[k] = torch.as_tensor(v, device=device, dtype=dtype)
        return c[k]

    def consts(self, device, dtype):
        """(a_hw, v_max) tensors cached per device (host->device copies are not allowed inside a CUDA-graph capture)."""
        k = (str(device), dtype)
        c = self.__dict__.setdefault("_c", {})
        if k not in c:
            c[k] = (torch.as_tensor(self.a_hw, device=device, dtype=dtype), torch.as_tensor(self.v_max, device=device, dtype=dtype))
        return c[k]


def _R(yaw):
    c, s = torch.cos(yaw), torch.sin(yaw)
    return c, s


def rollout(model: BaseModel, pose0, v0, U, r_cup, tilt0=None):
    """U (B,H,nu) with nu = 3 (+2 tilt rates dc/dt, base frame) -> dict
         pose (B,H,3) pose after each step, v (B,H,3) actual twist after each step, a (B,H,2) cup point world accel (finite diff),
         c (B,H,2) horizontal tray axis in the WORLD frame (zeros if no tilt).
       pose0 (3,), v0 (3,) tensors; r_cup (2,) cup position in the base frame."""
    B, H, nu = U.shape
    dtv = model.dt_vec(H, U.device, U.dtype)
    a_hw, vmax = model.consts(U.device, U.dtype)
    x = pose0[0].expand(B); y = pose0[1].expand(B); yaw = pose0[2].expand(B)
    v = v0.expand(B, 3)
    rc = r_cup
    body = model.frame == "body"

    def cup_vel(vv, yy):
        c, s = torch.cos(yy), torch.sin(yy)
        px = c * rc[0] - s * rc[1]; py = s * rc[0] + c * rc[1]
        if body:
            vwx = c * vv[:, 0] - s * vv[:, 1]; vwy = s * vv[:, 0] + c * vv[:, 1]
        else:
            vwx, vwy = vv[:, 0], vv[:, 1]
        return torch.stack([vwx - vv[:, 2] * py, vwy + vv[:, 2] * px], 1)

    vp_prev = cup_vel(v, yaw)
    ct = torch.zeros(B, 2, device=U.device, dtype=U.dtype) if tilt0 is None else tilt0.expand(B, 2)
    poses, vs, accs, cs = [], [], [], []
    for t in range(H):
        dt = dtv[t]
        u = torch.minimum(torch.maximum(U[:, t, :3], -vmax), vmax)
        v1 = v + torch.minimum(torch.maximum(u - v, -a_hw * dt), a_hw * dt)
        vm = 0.5 * (v + v1)
        ym = yaw + 0.5 * vm[:, 2] * dt
        if body:
            c, s = torch.cos(ym), torch.sin(ym)
            x = x + (c * vm[:, 0] - s * vm[:, 1]) * dt
            y = y + (s * vm[:, 0] + c * vm[:, 1]) * dt
        else:
            x = x + vm[:, 0] * dt
            y = y + vm[:, 1] * dt
        yaw = yaw + vm[:, 2] * dt
        v = v1
        vp = cup_vel(v, yaw)
        accs.append((vp - vp_prev) / dt)
        vp_prev = vp
        if model.use_tilt and nu >= 5:
            ct = torch.clamp(ct + torch.clamp(U[:, t, 3:5], -model.tilt_rate, model.tilt_rate) * dt, -model.tilt_cap, model.tilt_cap)
        cb, sb = torch.cos(yaw), torch.sin(yaw)
        cs.append(torch.stack([cb * ct[:, 0] - sb * ct[:, 1], sb * ct[:, 0] + cb * ct[:, 1]], 1))
        poses.append(torch.stack([x, y, yaw], 1)); vs.append(v)
    return dict(pose=torch.stack(poses, 1), v=torch.stack(vs, 1), a=torch.stack(accs, 1), c=torch.stack(cs, 1))


def slip_ratio(a, c, mu, kappa=0.8, theta_spill=0.30):
    """|g_in| / (r g) for cup accel a (B,H,2), tray axis c (B,H,2) and friction mu (M,) -> (B,H,M).   > 1 = the cup slips / spills."""
    gin = torch.linalg.norm(-a + G * c, dim=-1)                                   # (B,H)
    r = torch.clamp(kappa * mu, max=math.tan(theta_spill))                        # (M,)
    return gin[..., None] / (r * G)
