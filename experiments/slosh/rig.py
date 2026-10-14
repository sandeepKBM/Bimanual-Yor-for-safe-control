#!/usr/bin/env python
"""rig.py -- controlled cup-EXCITATION rig for the learned-slosh-constraint study.

SIMULATION ONLY (MuJoCo, offline). No robot, no hardware, no network.

Why a rig instead of FL-MPC flagship episodes: see check_init_cup_orientation.py
-- in the flagship faked-grasp episodes (cup_jitter_m=0.04), 10/24 seeds grasp the
cup ON ITS SIDE (world tilt 83-97 deg at the lift pose) and those are EXACTLY the
10 "spilled" seeds; the 14 upright seeds never spill. So flagship spill labels are
a setup artifact, not slosh. The rig drives the SAME cup + 8 balls (geometry,
contact params and physics options copied verbatim from
scenes/scene_yor_cup_collect.xml / yor_robot_derived.mjcf at load time) along
scripted 6-DoF cup trajectories spanning the plausible range, with the cup
carried EXACTLY as the demo's faked-grasp mode does it:
  * pose kinematically pinned each step (eval/scene_setup.reattach_cup_rigidly),
  * INERTIAL carry: cup free-joint qvel := backward finite-difference of the
    pinned pose (mj_differentiatePos), as scene_setup.cup_velocity_from_motion /
    run_episode(inertial_cup=True) do (commit b55c9bd),
  * balls are ordinary free bodies; they get a 0.4 s static SETTLE before any
    motion (removes the zero-velocity release transient flagged in the ladder).
Oracle: count_spilled (scripts/cup_task_utils.py) re-implemented vectorized with
the IDENTICAL cylinder test (radius 0.030, height 0.080, margin 0.002); checked
bit-equal against the original in `--selftest`.

Per-episode output (200 Hz, dt = 0.005 s, same as the demo):
  pos (T,3) cup-origin world position; quat (T,4) cup world quat (w,x,y,z);
  spilled (T,) int8 ball-out count; centroid (T,3) ball-pile centroid in the
  cup frame; zmax (T,) highest ball centre in cup frame; t_motion0 (int) first
  motion step; family (str); params (dict).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import re
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[2]
SCENE = REPO / "scenes" / "scene_yor_cup_collect.xml"
ROBOT = REPO / "scenes" / "yor_robot_derived.mjcf"
DT = 0.005
CUP_RADIUS, CUP_HEIGHT, H_MARGIN = 0.030, 0.080, 0.002   # == cup_task_utils.count_spilled defaults
N_BALLS = 8
CUP_HOME = np.array([0.0, 0.0, 1.0])
SETTLE_S, TAIL_S = 0.4, 0.6
RIM_EXIT_TOL = 0.012          # first-exit local z must be > rim - 12 mm to count as over-the-rim
V_WALL_MAX = 1.6              # m/s: max speed of any cup point (origin + rim); resample above
A_MAX = 45.0                  # m/s^2 cap on scripted peak translational accel
FAMILIES = ("translate", "shake", "tilt", "bangbang", "mixed", "vertical")


# ----------------------------------------------------------------------------- model
def _extract_body(txt: str, name: str) -> str:
    m = re.search(r'<body name="%s".*?</body>' % re.escape(name), txt, re.S)
    assert m, name
    return m.group(0)


def rig_xml() -> str:
    txt = SCENE.read_text()
    opt = re.search(r"<option[^>]*/>", ROBOT.read_text()).group(0)
    cup = _extract_body(txt, "cup")
    balls = [_extract_body(txt, f"ball_{i}") for i in range(N_BALLS)]
    # translate cup+balls so the cup origin sits at CUP_HOME (balls keep their
    # declared cup-relative offsets == count_spilled 0 layout by scene design)
    cp = np.array([float(x) for x in re.search(r'<body name="cup" pos="([^"]+)"', cup).group(1).split()])

    def shift(b):
        p = np.array([float(x) for x in re.search(r'pos="([^"]+)"', b).group(1).split()])
        q = p - cp + CUP_HOME
        return re.sub(r'pos="[^"]+"', 'pos="%.6f %.6f %.6f"' % tuple(q), b, count=1)
    body = "\n".join([shift(cup)] + [shift(b) for b in balls])
    return f"""<mujoco model="slosh_rig">
  {opt}
  <asset>
    <material name="cup_mat" rgba="0.85 0.15 0.15 0.95"/>
    <material name="ball_mat" rgba="0.15 0.55 0.9 1"/>
  </asset>
  <worldbody>
{body}
  </worldbody>
</mujoco>"""


class Rig:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_string(rig_xml())
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.cup = m.body("cup").id
        cj = m.body_jntadr[self.cup]
        self.cq, self.cv = int(m.jnt_qposadr[cj]), int(m.jnt_dofadr[cj])
        self.balls = [m.body(f"ball_{i}").id for i in range(N_BALLS)]
        self.bq = np.array([int(m.jnt_qposadr[m.body_jntadr[b]]) for b in self.balls])
        self.q0 = m.qpos0.copy()          # declared layout (cup at CUP_HOME, balls inside)
        self.ball_idx = (self.bq[:, None] + np.arange(3)[None]).ravel()

    def count_spilled_vec(self, cup_pos, cup_quat):
        R = np.empty(9); mujoco.mju_quat2Mat(R, cup_quat); R = R.reshape(3, 3)
        bp = self.d.qpos[self.ball_idx].reshape(N_BALLS, 3)
        loc = (bp - cup_pos) @ R                     # == R.T @ (b - c) per ball
        rad = np.hypot(loc[:, 0], loc[:, 1]); h = loc[:, 2]
        out = (rad > CUP_RADIUS) | (h < -H_MARGIN) | (h > CUP_HEIGHT)
        return int(out.sum()), loc

    def run(self, pos, quat, t_motion0):
        """pos (T,3), quat (T,4): scripted cup pose per step. Returns log dict."""
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self.q0
        # place balls relative to the scripted start pose (start pose is upright
        # at the origin in every family -> identity rotation, pure translation)
        d.qpos[self.ball_idx] += np.tile(pos[0] - CUP_HOME, N_BALLS)
        T = pos.shape[0]
        spilled = np.zeros(T, np.int8); cen = np.zeros((T, 3)); zmax = np.zeros(T)
        exit_step = -np.ones(N_BALLS, int); exit_z = np.full(N_BALLS, np.nan)
        prev = None
        for k in range(T):
            d.qpos[self.cq:self.cq + 3] = pos[k]
            d.qpos[self.cq + 3:self.cq + 7] = quat[k]
            if prev is None:
                d.qvel[self.cv:self.cv + 6] = 0.0
            else:   # INERTIAL carry, identical to scene_setup.cup_velocity_from_motion usage
                qv = np.zeros(m.nv)
                mujoco.mj_differentiatePos(m, qv, DT, prev, d.qpos)
                d.qvel[self.cv:self.cv + 6] = qv[self.cv:self.cv + 6]
            prev = d.qpos.copy()
            n, loc = self.count_spilled_vec(pos[k], quat[k])
            if n:
                rad = np.hypot(loc[:, 0], loc[:, 1])
                out = (rad > CUP_RADIUS) | (loc[:, 2] < -H_MARGIN) | (loc[:, 2] > CUP_HEIGHT)
                new = out & (exit_step < 0)
                exit_step[new] = k; exit_z[new] = loc[new, 2]
            spilled[k] = n; cen[k] = loc.mean(0); zmax[k] = loc[:, 2].max()
            mujoco.mj_step(m, d)
        # a GENUINE spill leaves over the rim (local z near/above the rim at first
        # exit); a first exit low on the wall / through the bottom = the pinned
        # wall teleported through a ball (tunneling artifact of kinematic carry).
        rim_exit = exit_z > (CUP_HEIGHT - RIM_EXIT_TOL)
        tunnel = bool(np.any((exit_step >= 0) & ~rim_exit))
        return dict(spilled=spilled, centroid=cen.astype(np.float32), zmax=zmax.astype(np.float32),
                    exit_step=exit_step, exit_z=exit_z, tunnel=tunnel)


# ----------------------------------------------------------------------------- motions
def minjerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def rand_dir(rng, zscale=0.5):
    v = rng.normal(size=3); v[2] *= zscale
    return v / np.linalg.norm(v)


def axis_angle_quat(axis, ang):
    axis = np.asarray(axis, float); axis = axis / (np.linalg.norm(axis) + 1e-12)
    ang = np.asarray(ang, float)
    q = np.zeros(ang.shape + (4,))
    q[..., 0] = np.cos(ang / 2); q[..., 1:] = np.sin(ang / 2)[..., None] * axis
    return q


def quat_mul_arr(a, b):
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0); w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2], -1)


def _translate_profile(rng, t, t0, amax_lo, amax_hi, n_moves):
    disp = np.zeros((len(t), 3)); tc = t0; info = []
    for _ in range(n_moves):
        A = float(np.exp(rng.uniform(np.log(amax_lo), np.log(amax_hi))))
        Tm = float(rng.uniform(0.15, 0.8))
        D = min(A * Tm**2 / 5.7735, 0.45, 0.8 * Tm)
        u = rand_dir(rng)
        disp += (D * minjerk((t - tc) / Tm))[:, None] * u
        info.append((A, Tm, D)); tc += Tm + float(rng.uniform(0.0, 0.4))
    return disp, info, tc


def make_motion(rng, family):
    """Return pos (T,3), quat (T,4), t_motion0, params. Cup starts upright at CUP_HOME."""
    motion_s = 3.0
    t = np.arange(0, SETTLE_S + motion_s + TAIL_S, DT)
    t0 = SETTLE_S
    pos = np.tile(CUP_HOME, (len(t), 1)).astype(float)
    tilt = np.zeros(len(t)); tilt_axis = rand_dir(rng, zscale=0.0); tilt_axis[2] = 0
    tilt_axis /= np.linalg.norm(tilt_axis)
    p = dict(family=family)
    if family == "translate":
        disp, info, _ = _translate_profile(rng, t, t0, 2.0, A_MAX, int(rng.integers(1, 4)))
        pos += disp; p["moves"] = info
    elif family == "shake":
        f = float(rng.uniform(0.5, 8.0)); A = float(np.exp(rng.uniform(np.log(2.0), np.log(A_MAX))))
        dur = float(rng.uniform(0.5, 2.5)); u = rand_dir(rng, 0.3)
        amp = min(A / (2 * np.pi * f) ** 2, 0.2, 1.4 / (2 * np.pi * f))
        s = np.clip((t - t0) / dur, 0, 1)
        if rng.random() < 0.5:   # shake while tilted (effective-gravity regime)
            th0 = np.radians(float(rng.uniform(20, 90)))
            tilt = th0 * minjerk((t - t0 + 0.3) / 0.3)
            p["tilt0_deg"] = np.degrees(th0)
        env = np.sin(np.pi * s) ** 2
        pos += (amp * env * np.sin(2 * np.pi * f * (t - t0)))[:, None] * u
        p.update(f=f, A=A, amp=amp, dur=dur)
    elif family == "tilt":
        th = np.radians(float(rng.uniform(30, 130))); Tt = float(rng.uniform(0.1, 1.5))
        hold = float(rng.uniform(0.2, 1.0)); back = bool(rng.random() < 0.6)
        tilt = th * minjerk((t - t0) / Tt)
        if back:
            tilt -= th * minjerk((t - t0 - Tt - hold) / Tt)
        p.update(theta_deg=np.degrees(th), Tt=Tt, hold=hold, back=back)
    elif family == "bangbang":
        # trapezoidal-velocity moves: piecewise-CONSTANT accel (infinite jerk corners)
        tc = t0; acc = np.zeros((len(t), 3)); info = []
        for _ in range(int(rng.integers(1, 4))):
            A = float(np.exp(rng.uniform(np.log(2.0), np.log(A_MAX))))
            Ta = min(float(rng.uniform(0.04, 0.3)), 1.4 / A); Tc = float(rng.uniform(0.0, 0.4)); u = rand_dir(rng)
            m1 = (t >= tc) & (t < tc + Ta); m2 = (t >= tc + Ta + Tc) & (t < tc + 2 * Ta + Tc)
            acc[m1] += A * u; acc[m2] -= A * u
            info.append((A, Ta, Tc)); tc += 2 * Ta + Tc + float(rng.uniform(0.0, 0.4))
        vel = np.cumsum(acc, 0) * DT; pos += np.cumsum(vel, 0) * DT
        p["moves"] = info
    elif family == "mixed":
        disp, info, _ = _translate_profile(rng, t, t0, 2.0, A_MAX, int(rng.integers(1, 3)))
        pos += disp
        th = np.radians(float(rng.uniform(0, 110))); Tt = float(rng.uniform(0.2, 1.2))
        tilt = th * minjerk((t - t0 - float(rng.uniform(0, 0.8))) / Tt)
        if rng.random() < 0.5:
            f = float(rng.uniform(1.0, 6.0)); A = float(rng.uniform(2.0, 25.0)); u = rand_dir(rng, 0.3)
            amp = min(A / (2 * np.pi * f) ** 2, 0.1, 0.7 / (2 * np.pi * f))
            s = np.clip((t - t0 - 0.5) / 1.5, 0, 1)
            pos += (amp * np.sin(np.pi * s) ** 2 * np.sin(2 * np.pi * f * t))[:, None] * u
            p.update(shake_f=f, shake_A=A)
        p.update(moves=info, theta_deg=np.degrees(th), Tt=Tt)
    elif family == "vertical":
        # lift/lower-and-stop moves (a_z of either sign; |a_z|>g floats the balls)
        tc = t0; info = []
        for _ in range(int(rng.integers(1, 3))):
            A = float(np.exp(rng.uniform(np.log(3.0), np.log(A_MAX)))); Tm = float(rng.uniform(0.12, 0.6))
            D = min(A * Tm**2 / 5.7735, 0.4, 0.8 * Tm); sgn = 1.0 if rng.random() < 0.5 else -1.0
            u = np.array([0.3 * rng.normal(), 0.3 * rng.normal(), sgn]); u /= np.linalg.norm(u)
            pos += (D * minjerk((t - tc) / Tm))[:, None] * u
            info.append((A, Tm, D, sgn)); tc += Tm + float(rng.uniform(0.0, 0.4))
        th = np.radians(float(rng.uniform(0, 70)))
        tilt = th * minjerk((t - t0) / float(rng.uniform(0.3, 1.2)))
        p.update(moves=info, theta_deg=np.degrees(th))
    else:
        raise ValueError(family)
    quat = axis_angle_quat(tilt_axis, tilt)
    return pos, quat, int(round(t0 / DT)), p


def max_point_speed(pos, quat):
    v = np.linalg.norm(np.diff(pos, axis=0), axis=1) / DT
    dq = np.abs(np.sum(quat[1:] * quat[:-1], axis=1)).clip(0, 1)
    w = 2 * np.arccos(dq) / DT
    return float(np.max(v + w * 0.085))


def run_one(args):
    seed, family = args
    rng = np.random.default_rng(seed)
    rig = Rig()
    for _try in range(50):     # resample until every cup point stays below V_WALL_MAX
        pos, quat, k0, p = make_motion(rng, family)
        if max_point_speed(pos, quat) <= V_WALL_MAX:
            break
    p["vmax_point"] = max_point_speed(pos, quat)
    t0 = time.perf_counter()
    log = rig.run(pos, quat, k0)
    p["wall_s"] = time.perf_counter() - t0
    return dict(seed=seed, family=family, pos=pos.astype(np.float32), quat=quat.astype(np.float32),
                t_motion0=k0, params=p, **log)


def selftest():
    """Vectorized oracle == scripts.cup_task_utils.count_spilled on random states."""
    sys.path.insert(0, str(REPO))
    from scripts import cup_task_utils as ct
    rig = Rig(); rng = np.random.default_rng(0)
    pos, quat, k0, _ = make_motion(rng, "tilt")
    m, d = rig.m, rig.d
    mujoco.mj_resetData(m, d); d.qpos[:] = rig.q0
    nbad = 0
    for k in range(0, len(pos)):
        d.qpos[rig.cq:rig.cq + 3] = pos[k]; d.qpos[rig.cq + 3:rig.cq + 7] = quat[k]
        mujoco.mj_forward(m, d)
        a = rig.count_spilled_vec(pos[k], quat[k])[0]; b = ct.count_spilled(m, d)
        nbad += int(a != b)
        mujoco.mj_step(m, d)
    print(f"selftest: {len(pos)} states, mismatches={nbad}, final spilled={b}")
    return nbad == 0


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-family", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=str, required=False)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    # family weights: spill-rich (tilt / mixed / tilted-shake) oversampled; the
    # upright translate / bangbang / vertical families are kept as HARD NEGATIVES
    # (large accelerations that an effective-gravity model may false-alarm on).
    weights = dict(translate=1, shake=2, tilt=3, bangbang=1, mixed=3, vertical=1)
    jobs = [(a.seed0 + i * 100 + j, f) for i in range(a.n_per_family)
            for j, f in enumerate(FAMILIES) for _w in range(weights[f]) if True]
    # make seeds unique across the weight repeats
    jobs = [(s * 7 + r, f) for r, (s, f) in enumerate(jobs)]
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=50) as pool:
        eps = pool.map(run_one, jobs, chunksize=4)
    print(f"{len(eps)} episodes in {time.perf_counter() - t0:.1f}s", flush=True)
    for f in FAMILIES:
        e = [x for x in eps if x["family"] == f]
        sp = [int(x["spilled"][x["t_motion0"]:].max() > 0) for x in e]
        bad = [int(x["spilled"][x["t_motion0"]] > 0) for x in e]
        tun = [int(x["tunnel"]) for x in e]
        print(f"  {f:10s} n={len(e):4d} spill-rate={np.mean(sp):.2f}  tunnel={sum(tun)}  "
              f"spilled-before-motion={sum(bad)}", flush=True)
    if a.out:
        import pickle
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "wb") as fh:
            pickle.dump(eps, fh, protocol=4)
        print("saved", a.out)


if __name__ == "__main__":
    main()
