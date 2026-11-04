#!/usr/bin/env python
"""gen_spill_data.py -- honest SPILL dataset generator for world-model (WM) training.

SIMULATION ONLY (MuJoCo, offline). No robot, no hardware, no network. New file;
imports existing modules read-only and edits nothing.

Why (see world_model/v2/data/diag_init_tilt.py and the v2 report): the old WM
data (collection/collect_cup*.py) carried a zero-velocity pinned cup, 10/24
flagship seeds had the cup ON ITS SIDE (init bug: `cup_jitter_m` shifts only the
BALLS relative to the cup -> balls spawn inside the walls -> penetration kick
tips the free cup), and spill onset was locked to the script phase / followed
collisions. This generator:

  * starts every episode with cup + contents settled UPRIGHT (asserted: tilt <
    UPRIGHT_MAX_DEG and 0 balls out after the settle; failures rejected + logged),
  * decouples spill from phase: random idle delay before motion, randomized
    families/timing/amplitude/axis (experiments/slosh/rig.make_motion, read-only)
    plus a "gentle" down-scaling so many episodes never spill,
  * decouples spill from collision: the right arm runs an INDEPENDENT random reach
    schedule (sometimes near the cup); collision labels are kept SEPARATE.

MODES (--modes, comma list with weights, e.g. "rig:0.85,flmpc:0.15")
  rig     faked_inertial carry: left arm realizes the scripted cup trajectory by
          damped-least-squares IK on the CUP point (joint-speed clamp = the
          UNMODIFIED YOR URDF limits 5,5,5,5,5,3 rad/s, joint ranges from the
          MJCF); arms are kinematic playback; cup kinematically pinned to the EE
          with INERTIAL carry (mj_differentiatePos, == rig.py / demo); balls are
          free bodies. Contents randomized (contents.RANGES).
  flmpc   the flagship FL-MPC crossing (scripts/make_cup_collect_safety_demo.py,
          read-only) with randomized aggressiveness (demo.V_EE_MAX_MS) and cube
          count; cup_jitter_m is FORCED to 0 (the jitter path is the tip-over bug);
          nominal 8-ball contents (the demo loads its own scene).
CARRY (--carry): faked_inertial (default) | real_grasp. real_grasp is wired for
  flmpc (demo build_episode(grasp_mode="real") -> control/grasp_routines
  grasp_cup_real_leveled + FL-MPC lift); for rig mode it is a STUB (raises).

OBSERVATIONS (recorded every STRIDE=13 physics steps -> 15.38 Hz WM rate; physics 200 Hz)
  img_zed    (S,S,3) uint8  cup-CENTRED crop (ZED_CROP_M metric window) of zed_cam
             (NOTE: zed_cam is the PROPOSED torso mount, scripts/zed_camera.py)
  img_lwrist (S,S,3) uint8  left_arm_egocentric
  img_rwrist (S,S,3) uint8  right_arm_egocentric
  proprio    (26,)  [left q(6), left qd(6), right q(6), right qd(6), left finger, right finger]
  action     (14,)  ABSOLUTE joint-position reference for the NEXT WM step:
             [left q_ref(6), right q_ref(6), left finger ref, right finger ref].
             rig: the IK playback reference; flmpc: the realized next-step q
             (FL-MPC commands torque; flagged per episode as action_kind).
  action_cup (6,)   commanded cup accel over the next WM interval [lin(3) m/s^2,
             ang(3) rad/s^2, world]; rig: from the script; flmpc: realized.
  cup_pose (7,) pos+quat(wxyz); cup_vel (6,) lin+ang; cup_acc (6,) (FK, realized)
LABELS (per WM step; COLLISION SEPARATE from SPILL)
  spilled (count), spill_frac, time_to_spill [s] (inf = never; 0 after onset),
  collision (arm-arm contact in the preceding interval), right_cup_contact,
  right_ball_contact; per episode: onset_step (200 Hz), onset_frame, tunnel flag,
  dense 200 Hz spilled curve (spilled_200hz, concatenated with ep_dense_start).

SHARDS: <out>/shard_XXXXX.npz (episodes concatenated along time; ep_start/ep_len
index them) + <out>/manifest.json (per-episode metadata + rejections).

Example (owner, full run -- see report for the exact command):
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=6 python world_model/v2/data/gen_spill_data.py \
      --out /path/spill_v2 --n-episodes 4000 --workers 24 --eps-per-shard 25
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import json
import time
import argparse
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import contents as C  # noqa: E402

import mujoco  # noqa: E402

REPO = C.REPO
DT = C.DT
STRIDE = 13                          # 200 Hz / 13 = 15.38 Hz WM rate (uniform dt_wm = 0.065 s)
UPRIGHT_MAX_DEG = 15.0
ZED_W, ZED_H = 640, 480
ZED_CROP_M = 0.22                    # metric side of the cup-centred window (cup is 0.06 m wide)
QD_MAX = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 3.0])   # rad/s, YOR/robot/arm/urdf/piper_description_*.xml (unmodified)
IDLE_MAX_S = 1.0
GENTLE_P = 0.25
GENTLE_SCALE = (0.1, 0.6)
REC_TAIL_S = 0.6
W_ROT_TRACK = 1.0                    # DLS weight: 1 rad orientation error ~ 1 m position error
RIG_FAMILY_W = dict(translate=1, shake=2, tilt=2, bangbang=1, mixed=2, vertical=1, pour=6)
TRANS_CAP_M = 0.15                   # max scripted cup displacement the arm is asked to realize


# ============================================================================ helpers
def quat_mul(a, b):
    r = np.empty(4); mujoco.mju_mulQuat(r, np.asarray(a, float), np.asarray(b, float)); return r


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_pow(q, s):
    """q^s (scale the rotation angle by s, same axis)."""
    q = np.asarray(q, float)
    w = np.clip(q[..., 0], -1, 1); ang = 2 * np.arccos(w)
    sn = np.sqrt(np.maximum(1 - w**2, 0))
    axis = np.where(sn[..., None] > 1e-9, q[..., 1:] / np.maximum(sn[..., None], 1e-12), np.array([1.0, 0, 0]))
    a2 = ang * s / 2
    return np.concatenate([np.cos(a2)[..., None], np.sin(a2)[..., None] * axis], -1)


def tilt_deg(q):
    R = np.empty(9); mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return float(np.degrees(np.arccos(np.clip(R[8], -1, 1))))


def quat_series_angvel(Q):
    """World angular velocity (T,3) from a quat series (backward difference)."""
    w = np.zeros((len(Q), 3))
    for k in range(1, len(Q)):
        dq = quat_mul(Q[k], quat_conj(Q[k - 1]))
        v = np.empty(3); mujoco.mju_quat2Vel(v, dq, DT); w[k] = v
    return w


class Addr:
    def __init__(self, m):
        import sim_common as sc
        self.sc = sc
        _, self.lq, self.ld = sc.joint_addrs(m, sc.LEFT_ARM_JOINTS)
        _, self.rq, self.rd = sc.joint_addrs(m, sc.RIGHT_ARM_JOINTS)
        self.lq, self.ld, self.rq, self.rd = map(np.asarray, (self.lq, self.ld, self.rq, self.rd))
        self.lf = [int(m.joint(f"left_arm_joint{i}").qposadr[0]) for i in (7, 8)]
        self.rf = [int(m.joint(f"right_arm_joint{i}").qposadr[0]) for i in (7, 8)]
        self.lfd = [int(m.joint(f"left_arm_joint{i}").dofadr[0]) for i in (7, 8)]
        self.rfd = [int(m.joint(f"right_arm_joint{i}").dofadr[0]) for i in (7, 8)]
        self.lrange = np.array([m.jnt_range[m.joint(n).id] for n in sc.LEFT_ARM_JOINTS])
        self.rrange = np.array([m.jnt_range[m.joint(n).id] for n in sc.RIGHT_ARM_JOINTS])
        self.lee = m.body("left_arm_gripper_base").id
        self.ree = m.body("right_arm_gripper_base").id
        self.cup = m.body("cup").id
        cj = m.body_jntadr[self.cup]
        self.cq, self.cv = int(m.jnt_qposadr[cj]), int(m.jnt_dofadr[cj])
        self.lact = [m.actuator(f"left_arm_joint{i}_pos").id for i in range(1, 7)]
        self.ract = [m.actuator(f"right_arm_joint{i}_pos").id for i in range(1, 7)]
        # geom classes for contact labels: 1 left arm, 2 right arm, 3 cup, 4 ball
        cls = np.zeros(m.ngeom, np.int8)
        for g in range(m.ngeom):
            bn = m.body(int(m.geom_bodyid[g])).name
            if bn.startswith("left_arm_"):
                cls[g] = 1
            elif bn.startswith("right_arm_"):
                cls[g] = 2
            elif bn == "cup":
                cls[g] = 3
            elif bn.startswith("ball_"):
                cls[g] = 4
        self.gcls = cls

    def contact_flags(self, d):
        if d.ncon == 0:
            return 0, 0, 0
        g = d.contact.geom[:d.ncon]
        a, b = self.gcls[g[:, 0]], self.gcls[g[:, 1]]
        s = set(zip(np.minimum(a, b).tolist(), np.maximum(a, b).tolist()))
        return int((1, 2) in s), int((2, 3) in s), int((2, 4) in s)


def dls_step(m, sd, bid, jac_dofs, point, p_des, q_des_quat, cur_quat, lam=0.03, w_rot=0.5):
    jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    mujoco.mj_jac(m, sd, jp, jr, point, bid)
    J = np.vstack([jp[:, jac_dofs], w_rot * jr[:, jac_dofs]])
    er = np.empty(3); mujoco.mju_quat2Vel(er, quat_mul(q_des_quat, quat_conj(cur_quat)), 1.0)
    e = np.concatenate([p_des - point, w_rot * er])
    return J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), e)


def ik_pose(m, sd, A, side, q0, p_des, quat_des, iters=200, point_fn=None):
    """Offline pose IK for the given arm EE (optionally on an offset point)."""
    qa, da, bid, rng_ = (A.lq, A.ld, A.lee, A.lrange) if side == "left" else (A.rq, A.rd, A.ree, A.rrange)
    q = q0.copy()
    for _ in range(iters):
        sd.qpos[qa] = q
        mujoco.mj_kinematics(m, sd); mujoco.mj_comPos(m, sd)
        pt = sd.xpos[bid].copy() if point_fn is None else point_fn(sd)
        if quat_des is None:
            jp = np.zeros((3, m.nv)); mujoco.mj_jac(m, sd, jp, None, pt, bid)
            J = jp[:, da]; e = p_des - pt
            dq = J.T @ np.linalg.solve(J @ J.T + 0.05**2 * np.eye(3), e)
        else:
            dq = dls_step(m, sd, bid, da, pt, p_des, quat_des, sd.xquat[bid].copy())
        q = np.clip(q + np.clip(dq, -0.2, 0.2), rng_[:, 0], rng_[:, 1])
    sd.qpos[qa] = q; mujoco.mj_kinematics(m, sd)
    pt = sd.xpos[bid].copy() if point_fn is None else point_fn(sd)
    return q, float(np.linalg.norm(p_des - pt))


def minjerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


# ============================================================================ rig mode
RIGHT_DISTAL = ("right_arm_link5", "right_arm_link6", "right_arm_gripper_base", "right_arm_link7", "right_arm_link8")
RIGHT_CUP_CLEAR = 0.10   # [m] min distance of right distal-body origins from the planned cup centre


def right_arm_schedule(m, sd, A, rng, q_home, T, cup_path, table_z, tries=5):
    """Independent random right-arm reach schedule (joint-space min-jerk), timed
    independently of the left/cup motion. 30%: stays home. Else 1-4 reaches;
    each target is 'near' the cup (7-18 cm from its start centre) w.p. 0.5 else a
    random workspace point around the right home EE. The arm is KINEMATIC, so it
    must never plough into the (pinned) cup: every reach segment is swept against
    the planned cup path (cup_path (T,3) = cup centre) and resampled / dropped if
    any distal right-arm body origin comes within RIGHT_CUP_CLEAR. Arm-ARM contact
    is allowed (labelled `collision`). Returns (T,6) and a dict of params."""
    qs = np.tile(q_home, (T, 1))
    info = dict(still=True, reaches=[], dropped=0)
    if rng.random() < 0.3:
        return qs, info
    info["still"] = False
    dist_ids = [m.body(b).id for b in RIGHT_DISTAL]
    sd.qpos[A.rq] = q_home; mujoco.mj_kinematics(m, sd)
    ee_home = sd.xpos[A.ree].copy()
    cup_c0 = cup_path[0]

    def seg_ok(q_seg, k0):
        for j in range(0, len(q_seg), 4):
            sd.qpos[A.rq] = q_seg[j]; mujoco.mj_kinematics(m, sd)
            if np.min(np.linalg.norm(sd.xpos[dist_ids] - cup_path[min(k0 + j, T - 1)], axis=1)) < RIGHT_CUP_CLEAR:
                return False
        return True

    t = float(rng.uniform(0.0, 1.5))
    qprev = q_home.copy()
    for _ in range(int(rng.integers(1, 5))):
        placed = False
        for _t in range(tries):
            if rng.random() < 0.5:
                u = rng.normal(size=3); u[2] = abs(u[2]) * 0.5; u /= np.linalg.norm(u)
                tgt = cup_c0 + u * rng.uniform(0.07, 0.18); kind = "near"
            else:
                tgt = ee_home + rng.uniform(-0.15, 0.15, 3); kind = "far"
            tgt[2] = max(tgt[2], table_z + 0.08)
            qt, err = ik_pose(m, sd, A, "right", qprev, tgt, None, iters=80)
            dist = np.max(np.abs(qt - qprev) / QD_MAX)
            dur = max(float(rng.uniform(0.4, 1.5)), 1.875 * dist)   # min-jerk peak speed = 1.875*D/T <= QD_MAX
            k0, k1 = int(t / DT), int((t + dur) / DT)
            if k0 >= T:
                break
            kk = np.arange(k0, T)
            seg = qprev + minjerk((kk - k0) / max(k1 - k0, 1))[:, None] * (qt - qprev)
            if seg_ok(seg, k0):
                qs[k0:] = seg; placed = True
                info["reaches"].append(dict(kind=kind, t0=t, dur=dur, ik_err=err))
                qprev = qt; t += dur + float(rng.uniform(0.0, 0.8))
                break
        if not placed:
            info["dropped"] += 1
    return qs, info


def make_cup_script(rng, family):
    import rig
    fam_rig = "tilt" if family == "pour" else family      # pour = rig 'tilt' profile about the wrist axis
    gp = 0.15 if family == "pour" else GENTLE_P
    for _ in range(50):
        pos, quat, k0, p = rig.make_motion(rng, fam_rig)
        gentle = rng.random() < gp
        s = float(rng.uniform(*GENTLE_SCALE)) if gentle else 1.0
        if gentle:
            pos = rig.CUP_HOME + s * (pos - rig.CUP_HOME)
            quat = quat_pow(quat, s)
        if rig.max_point_speed(pos, quat) <= rig.V_WALL_MAX:
            break
    p["gentle_scale"] = s
    idle = float(rng.uniform(0.3, 0.3 + IDLE_MAX_S))   # >=0.3 s: rig's tilted-shake starts 0.3 s before its t0
    n_idle = int(round(idle / DT))
    pos = np.concatenate([np.repeat(pos[:1], n_idle, 0), pos]) - rig.CUP_HOME
    quat = np.concatenate([np.repeat(quat[:1], n_idle, 0), quat])
    return pos, quat, k0 + n_idle, k0, p, idle


def episode_rig(seed, args):
    """faked_inertial rig-excitation episode on the full YOR scene."""
    if args.carry == "real_grasp":
        raise NotImplementedError(
            "rig mode + real_grasp is a STUB: realizing violent rig excitations through a real friction "
            "pinch needs a torque-level tracker (FL-MPC/OSC) + slip handling; use --modes flmpc with "
            "--carry real_grasp (wired through demo.build_episode(grasp_mode='real')).")
    import rig
    rng = np.random.default_rng(seed)
    fw = dict(RIG_FAMILY_W)
    if getattr(args, "family_weights", None):
        fw.update({k: float(v) for k, v in (x.split(":") for x in args.family_weights.split(","))})
    fams = list(fw); w = np.array([fw[f] for f in fams], float)
    family = str(rng.choice(fams, p=w / w.sum()))
    if getattr(args, "particle_frac", 0.0) > 0 and rng.random() < args.particle_frac:
        # liquid-like particle proxy (contents_study.particle_contents): frictionless (condim 1), soft,
        # polydisperse -- NOTE: still behaves granular (quasi-static spill ~88-90 deg, not ~59 deg liquid)
        cont = C.Contents(n_balls=int(rng.integers(40, 81)), radius=float(rng.uniform(0.0034, 0.0040)),
                          density=1000.0, friction=0.0, dampratio=1.0, condim=1, solref_tc=0.01,
                          solimp=(0.8, 0.95, 0.002), torsional=0.0, rolling=0.0, kind="particles", poly=0.3)
        if C.layout(cont) is None:
            cont = C.Contents(**C.NOMINAL)
    else:
        cont = C.sample_contents(rng) if args.randomize_contents else C.Contents(**C.NOMINAL)
    m, home, home_ctrl = C.scene_model(cont)
    d = mujoco.MjData(m); sd = mujoco.MjData(m)
    A = Addr(m); sc = A.sc
    for j in range(m.njnt):
        nm = m.joint(j).name
        if nm in home and not nm.startswith("ball_"):
            a = int(m.jnt_qposadr[j]); d.qpos[a:a + len(home[nm])] = home[nm]
    sd.qpos[:] = d.qpos
    q_rhome = d.qpos[A.rq].copy()
    mujoco.mj_forward(m, d)
    tg = m.geom("table_top").id
    table_z = float(d.geom_xpos[tg][2] + m.geom_size[tg][2])
    # ---- left start pose: LEFT_GRASP_Q lifted 0.11 m (+ small random offset), EE orientation held
    sd.qpos[A.lq] = sc.LEFT_GRASP_Q; mujoco.mj_kinematics(m, sd)
    ee0 = sd.xpos[A.lee].copy(); qee0 = sd.xquat[A.lee].copy()
    for _try in range(6):   # random start offset/yaw; shrink toward the nominal lift pose on IK failure
        sh = 1.0 - _try / 5.0
        off = np.array([sh * rng.uniform(-0.04, 0.04), sh * rng.uniform(-0.04, 0.04),
                        0.11 + sh * rng.uniform(-0.02, 0.04)])
        yaw = np.radians(sh * rng.uniform(-15, 15))
        qyaw = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        q_start, ik_err = ik_pose(m, sd, A, "left", np.array(sc.LEFT_GRASP_Q, float), ee0 + off, quat_mul(qyaw, qee0),
                                  iters=300)
        if ik_err <= 0.005:
            break
    if ik_err > 0.01:
        return None, dict(reason="start_ik", err=ik_err)
    sd.qpos[A.lq] = q_start; mujoco.mj_kinematics(m, sd)
    p7, p8 = m.geom("left_arm_link7_pad").id, m.geom("left_arm_link8_pad").id
    pad_mid = 0.5 * (sd.geom_xpos[p7] + sd.geom_xpos[p8])
    cup0 = pad_mid - np.array([0, 0, 0.55 * C.CUP_HEIGHT])
    cq0 = quat_mul(qyaw, np.array([1.0, 0, 0, 0]))
    Ree = sd.xmat[A.lee].reshape(3, 3).copy()
    rel_p = Ree.T @ (cup0 - sd.xpos[A.lee]); rel_q = quat_mul(quat_conj(sd.xquat[A.lee]), cq0)

    def cup_from_ee(dd):
        R = dd.xmat[A.lee].reshape(3, 3)
        return dd.xpos[A.lee] + R @ rel_p, quat_mul(dd.xquat[A.lee], rel_q)

    # ---- scripted cup trajectory (world), EE-independent; T includes the 0.4 s settle
    dpos0, dquat0, k_motion, k_settle, mparams, idle = make_cup_script(rng, family)
    T = len(dpos0)
    # ---- PLANNED left-arm joint trajectory: converged per-step pose IK (warm-started) so the realized
    # cup motion == the smooth rig script. Infeasible (IK residual > PLAN_ERR_MAX or joint speed > YOR
    # URDF limit) -> scale the motion amplitude down (x PLAN_SHRINK) and retry; else reject.
    # POUR mode: the scripted tilt ANGLE profile is applied about the left wrist-roll (joint6) axis through
    # its anchor (a single-joint rotation -> feasible up to the joint range), instead of rig.py's random
    # horizontal axis through the cup origin (mostly unreachable for this 6-DoF arm beyond ~40 deg).
    pour = bool(family == "pour" or (family in ("tilt", "mixed", "vertical", "shake") and rng.random() < POUR_P))
    if pour:
        import rig
        sd.qpos[A.lq] = q_start; mujoco.mj_kinematics(m, sd)
        j6 = m.joint("left_arm_joint6").id
        pax = sd.xaxis[j6].copy() * (1.0 if rng.random() < 0.5 else -1.0); piv = sd.xanchor[j6].copy()
        ang0 = 2 * np.arccos(np.clip(np.abs(dquat0[:, 0]), -1, 1))
        mparams["pour_axis"] = pax.tolist()

    tmax = float(np.linalg.norm(dpos0, axis=1).max())
    tcap = min(1.0, TRANS_CAP_M / tmax) if tmax > 0 else 1.0
    base = dict(dpos=dpos0, dquat=dquat0, T=T, k_motion=k_motion, stretch=1.0)

    def stretch_time(fac):
        """Slow the motion part (after the settle) down by `fac` (joint-speed limits bind)."""
        dp, dq_, T0 = dpos0, dquat0, len(dpos0)
        Tn = k_settle + int(np.ceil((T0 - k_settle) * fac))
        src = np.concatenate([np.arange(k_settle), k_settle + (np.arange(Tn - k_settle) / fac)])
        i0 = np.clip(np.floor(src).astype(int), 0, T0 - 1); i1 = np.clip(i0 + 1, 0, T0 - 1); w = (src - i0)[:, None]
        q1 = dq_[i1] * np.sign(np.sum(dq_[i0] * dq_[i1], 1, keepdims=True))
        qq = (1 - w) * dq_[i0] + w * q1; qq /= np.linalg.norm(qq, axis=1, keepdims=True)
        base.update(dpos=(1 - w) * dp[i0] + w * dp[i1], dquat=qq, T=Tn,
                    k_motion=k_settle + int(round((k_motion - k_settle) * fac)), stretch=fac)

    def cup_targets(sc_, sct):
        Tb = base["T"]; dpos = base["dpos"] * sct; dq_ = base["dquat"]
        if pour:
            ang = 2 * np.arccos(np.clip(np.abs(dq_[:, 0]), -1, 1))
            qr = rig.axis_angle_quat(pax, ang * sc_)
            p = np.empty((Tb, 3)); qq = np.empty((Tb, 4))
            for k in range(Tb):
                v = np.empty(3); mujoco.mju_rotVecQuat(v, cup0 - piv, qr[k])
                p[k] = piv + v + dpos[k]; qq[k] = quat_mul(qr[k], cq0)
            return p, qq
        dquat = quat_pow(dq_, sc_) if sc_ != 1.0 else dq_
        return cup0 + dpos, np.array([quat_mul(dquat[k], cq0) for k in range(Tb)])

    # joint-speed violation -> TIME-STRETCH (<= STRETCH_MAX); IK residual (reach) -> shrink translation, then all
    scale, tscale = 1.0, tcap
    if family == "pour":   # widen the pour amplitude: peak rotation ~ U(POUR_DEG) (rig 'tilt' draws 30-130 deg)
        ang_rig = float(np.degrees(2 * np.arccos(np.clip(np.abs(dquat0[:, 0]), -1, 1))).max())
        if ang_rig > 1.0:
            scale = float(rng.uniform(*POUR_DEG)) / ang_rig * (mparams.get("gentle_scale", 1.0))
    ok = False
    for _try in range(PLAN_TRIES):
        p_des, q_des = cup_targets(scale, tscale)
        QLc, perrc = plan_left(m, sd, A, q_start, p_des[::10], q_des[::10], cup_from_ee, iters=20)   # coarse pre-check
        qd_ratio = float((np.abs(np.diff(QLc, axis=0)) / (10 * DT) / QD_MAX).max())
        if perrc.max() <= PLAN_ERR_MAX and qd_ratio > 1.0 and base["stretch"] < STRETCH_MAX:
            stretch_time(min(STRETCH_MAX, base["stretch"] * qd_ratio * 1.1)); continue
        if perrc.max() > PLAN_ERR_MAX or qd_ratio > 1.0:
            tscale *= PLAN_SHRINK
            if _try >= 2:
                scale *= PLAN_SHRINK
            continue
        QL, perr = plan_left(m, sd, A, q_start, p_des, q_des, cup_from_ee)
        qd_ratio = float((np.abs(np.diff(QL, axis=0)) / DT / QD_MAX).max())
        if perr.max() <= PLAN_ERR_MAX and qd_ratio <= 1.0:
            ok = True
            break
        if perr.max() <= PLAN_ERR_MAX and base["stretch"] < STRETCH_MAX:
            stretch_time(min(STRETCH_MAX, base["stretch"] * qd_ratio * 1.1)); continue
        tscale *= PLAN_SHRINK
        if _try >= 2:
            scale *= PLAN_SHRINK
    if not ok:
        return None, dict(reason="plan_infeasible", err=float(perrc.max()), qd_ratio=qd_ratio, family=family)
    extra = float(rng.uniform(*EXTRA_STRETCH)) if base["stretch"] > 1.0 else 1.0
    if extra > 1.0 and base["stretch"] * extra <= STRETCH_MAX * EXTRA_STRETCH[1]:
        # speed-capped motions all run AT the joint-speed limit -> spill onset would lock to a fixed time
        # after motion start; a random further slow-down restores timing diversity (always still feasible)
        stretch_time(base["stretch"] * extra)
        p_des, q_des = cup_targets(scale, tscale)
        QL, perr = plan_left(m, sd, A, q_start, p_des, q_des, cup_from_ee)
    T, k_motion = base["T"], base["k_motion"]
    mparams["time_stretch"] = base["stretch"]
    mparams["plan_scale"] = scale; mparams["trans_scale"] = tscale; mparams["pour"] = pour
    cup_path = p_des + np.array([0, 0, 0.04])
    qr_sched, rinfo = right_arm_schedule(m, sd, A, rng, q_rhome, T, cup_path, table_z)
    # ---- initial state: arms, cup, balls in layout (upright)
    d.qpos[A.lq] = q_start; d.qpos[A.rq] = qr_sched[0]
    d.ctrl[A.lact] = q_start; d.ctrl[A.ract] = qr_sched[0]
    mujoco.mj_kinematics(m, d)
    cpos, cquat = cup_from_ee(d)
    d.qpos[A.cq:A.cq + 3] = cpos; d.qpos[A.cq + 3:A.cq + 7] = cquat
    orc = C.Oracle(m, cont.n_balls)
    lay = np.array(cont.layout)
    Rc = np.empty(9); mujoco.mju_quat2Mat(Rc, cquat); Rc = Rc.reshape(3, 3)
    d.qpos[orc.idx] = (cpos + lay @ Rc.T).ravel()
    for i in range(cont.n_balls):
        d.qpos[orc.bq[i] + 3:orc.bq[i] + 7] = [1, 0, 0, 0]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    # ---- main loop
    D_sp = np.zeros(T, np.int16); D_cpos = np.zeros((T, 3)); D_cquat = np.zeros((T, 4))
    D_col = np.zeros((T, 3), np.int8); D_ql = np.zeros((T, 6)); D_qr = np.zeros((T, 6))
    exit_step = -np.ones(cont.n_balls, int); exit_z = np.full(cont.n_balls, np.nan)
    snaps_q, snaps_v, snap_k = [], [], []
    q_l = q_start.copy(); q_l_prev = q_l.copy(); q_r_prev = qr_sched[0].copy()
    prev = None
    track_err = np.zeros(T)
    for k in range(T):
        q_l = QL[k]
        q_r = qr_sched[k]
        d.qpos[A.lq] = q_l; d.qpos[A.rq] = q_r
        d.qvel[A.ld] = (q_l - q_l_prev) / DT; d.qvel[A.rd] = (q_r - q_r_prev) / DT
        d.qpos[A.lf] = home["left_arm_joint7"][0], home["left_arm_joint8"][0]
        d.qpos[A.rf] = home["right_arm_joint7"][0], home["right_arm_joint8"][0]
        d.qvel[A.lfd] = 0; d.qvel[A.rfd] = 0
        d.ctrl[A.lact] = q_l; d.ctrl[A.ract] = q_r
        q_l_prev, q_r_prev = q_l.copy(), q_r.copy()
        mujoco.mj_kinematics(m, d)
        cpos, cquat = cup_from_ee(d)
        C.pin_cup(m, d, A.cq, A.cv, cpos, cquat, prev)
        prev = d.qpos.copy()
        loc = orc.local(d, cpos, cquat); out = C.Oracle.out_mask(loc)
        new = out & (exit_step < 0); exit_step[new] = k; exit_z[new] = loc[new, 2]
        D_sp[k] = int(out.sum()); D_cpos[k] = cpos; D_cquat[k] = cquat
        D_ql[k] = q_l; D_qr[k] = q_r
        track_err[k] = np.linalg.norm(cpos - p_des[k])
        if k == k_settle:          # upright + nothing out after the settle, else reject
            if tilt_deg(cquat) > UPRIGHT_MAX_DEG:
                return None, dict(reason="not_upright", tilt=tilt_deg(cquat))
        if k == k_motion and D_sp[k] > 0:   # nothing may leave during settle + idle (static cup)
            return None, dict(reason="spilled_before_motion", n=int(D_sp[k]), kind=cont.kind)
        if k >= k_settle and (k - k_settle) % STRIDE == 0:
            snaps_q.append(d.qpos.copy()); snaps_v.append(d.qvel.copy()); snap_k.append(k)
        mujoco.mj_step(m, d)
        D_col[k] = A.contact_flags(d)
    rim_exit = exit_z > (C.CUP_HEIGHT - C.RIM_EXIT_TOL)
    if np.any((exit_step >= 0) & ~rim_exit):     # a ball left through the floor/wall = sim artifact
        bad = (exit_step >= 0) & ~rim_exit
        acc_r = np.linalg.norm(np.gradient(np.gradient(D_cpos, DT, axis=0), DT, axis=0), axis=1)
        acc_s = np.linalg.norm(np.gradient(np.gradient(p_des, DT, axis=0), DT, axis=0), axis=1)
        return None, dict(reason="tunnel", exit_z=exit_z[exit_step >= 0].tolist(), family=family,
                          t_first_tunnel_from_motion=float((exit_step[bad].min() - k_motion) * DT),
                          acc_real_max=float(acc_r.max()), acc_script_max=float(acc_s.max()),
                          plan_scale=scale, contents=cont.to_dict())
    tau_ratio, qdd_max = arm_torque_check(m, sd, A, D_ql, D_qr)
    ep = dict(mode="rig", carry="faked_inertial", family=family, seed=int(seed), contents=cont.to_dict(),
              motion=_jsonable(mparams), idle_s=idle, right=rinfo, k_settle=int(k_settle), k_motion=int(k_motion),
              start_ik_err=ik_err, tau_ratio_max=tau_ratio, qdd_max=qdd_max,
              torque_feasible=bool(tau_ratio <= 1.0), track_err_med=float(np.median(track_err[k_settle:])),
              track_err_max=float(track_err.max()), action_kind="ik_joint_ref",
              tilt_cmd_max=float(max(tilt_deg(q) for q in q_des[::5])),
              tilt_real_max=float(max(tilt_deg(q) for q in D_cquat[::5])),
              tunnel=bool(np.any((exit_step >= 0) & ~rim_exit)))
    cmd_acc = np.zeros((T, 6))
    cmd_acc[1:-1, :3] = (p_des[2:] - 2 * p_des[1:-1] + p_des[:-2]) / DT**2
    wq = quat_series_angvel(q_des); cmd_acc[1:, 3:] = np.diff(wq, axis=0) / DT
    if getattr(args, "pendulum_labels", False):
        # equivalent-mechanical first-slosh-mode (liquid) proxy on the REALIZED cup motion; fill h = contents fill
        from contents_study import pendulum_run
        hh = float(np.clip(C.fill_height(cont), 0.02, 0.07))
        phi, psp, *_ = pendulum_run(D_cpos[k_settle:], D_cquat[k_settle:], hh, zeta=0.02, sub=4)
        ep["pend_h"] = hh
        pend = (phi, psp)
    else:
        pend = None
    dense = dict(spilled=D_sp, cup_pos=D_cpos, cup_quat=D_cquat, col=D_col, ql=D_ql, qr=D_qr, cmd_acc=cmd_acc, pend=pend,
                 lf=np.full(T, home["left_arm_joint7"][0]), rf=np.full(T, home["right_arm_joint7"][0]))
    return (m, A, cont, ep, dense, snaps_q, snaps_v, snap_k), None


POUR_P = 0.6
POUR_DEG = (60.0, 170.0)
STRETCH_MAX = 4.0
EXTRA_STRETCH = (1.0, 2.5)
PLAN_TRIES, PLAN_SHRINK, PLAN_ERR_MAX = 8, 0.75, 0.004   # [m] (rot error weighted 1 rad ~ 1 m)


def plan_left(m, sd, A, q0, p_des, q_des, cup_from_ee, iters=8, tol=1e-4):
    """Per-step pose IK of the CUP point/orientation, warm-started from the previous step."""
    T = len(p_des); Q = np.zeros((T, 6)); E = np.zeros(T); q = q0.copy()
    for k in range(T):
        for _ in range(iters):
            sd.qpos[A.lq] = q; mujoco.mj_kinematics(m, sd); mujoco.mj_comPos(m, sd)
            cp, cqq = cup_from_ee(sd)
            er = np.empty(3); mujoco.mju_quat2Vel(er, quat_mul(q_des[k], quat_conj(cqq)), 1.0)
            e = float(np.sqrt(np.sum((p_des[k] - cp) ** 2) + (W_ROT_TRACK * np.linalg.norm(er)) ** 2))
            if e < tol:
                break
            dq = dls_step(m, sd, A.lee, A.ld, cp, p_des[k], q_des[k], cqq, lam=0.01, w_rot=W_ROT_TRACK)
            q = np.clip(q + np.clip(dq, -0.1, 0.1), A.lrange[:, 0], A.lrange[:, 1])
        Q[k] = q; E[k] = e
    return Q, E


def arm_torque_check(m, sd, A, QL, QR, every=2):
    """Inverse-dynamics feasibility of the KINEMATIC arm playback: max over steps
    and joints of |tau_ID| / sim_common.TAU_MAX (100 Nm, unmodified YOR model).
    Arm links only (contacts disabled, payload NOT included: the cup is ~30-120 g
    at <0.3 m -> < ~0.4 Nm static, negligible vs 100 Nm). Returns (ratio, max |qdd|)."""
    TAU = float(A.sc.TAU_MAX)
    qdl = np.gradient(QL, DT, axis=0); qdr = np.gradient(QR, DT, axis=0)
    qal = np.gradient(qdl, DT, axis=0); qar = np.gradient(qdr, DT, axis=0)
    old = m.opt.disableflags
    m.opt.disableflags = old | int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    worst = 0.0
    try:
        for k in range(2, len(QL) - 2, every):
            sd.qpos[A.lq] = QL[k]; sd.qpos[A.rq] = QR[k]
            sd.qvel[:] = 0; sd.qacc[:] = 0
            sd.qvel[A.ld] = qdl[k]; sd.qvel[A.rd] = qdr[k]
            sd.qacc[A.ld] = qal[k]; sd.qacc[A.rd] = qar[k]
            mujoco.mj_inverse(m, sd)
            worst = max(worst, float(np.abs(sd.qfrc_inverse[A.ld]).max()), float(np.abs(sd.qfrc_inverse[A.rd]).max()))
    finally:
        m.opt.disableflags = old
    return worst / TAU, float(max(np.abs(qal).max(), np.abs(qar).max()))


# ============================================================================ flmpc mode
class _Recorder:
    """Duck-typed spill_monitor: never vetoes; snapshots full state at the cadence."""

    def __init__(self):
        self.q, self.v, self.k = [], [], []

    def reset(self):
        self.q, self.v, self.k = [], [], []

    def assess(self, obs):
        import safety_module as sm
        self.q.append(obs.data.qpos.copy()); self.v.append(obs.data.qvel.copy())
        self.k.append(int(round(obs.t / DT)))
        return sm.SemanticVerdict(risk=0.0, veto=False)


def episode_flmpc(seed, args):
    rng = np.random.default_rng(seed)
    from scripts import make_cup_collect_safety_demo as demo
    v_ee = float(np.exp(rng.uniform(np.log(0.18), np.log(0.6))))
    demo.V_EE_MAX_MS = v_ee
    n_cubes = int(rng.integers(2, 4))   # demo asserts >=2 IK-reachable cubes
    grasp_mode = "real" if args.carry == "real_grasp" else "faked"
    try:
        info = demo.build_episode(seed=int(seed), scenario="crossing", ee_speed_limit=True, cup_jitter_m=0.0,
                                  n_cubes=n_cubes, grasp_mode=grasp_mode)
    except Exception as e:  # e.g. real grasp failed
        return None, dict(reason="build_failed", err=repr(e)[:200])
    rec = _Recorder()
    log, _ = demo.run_episode(info, controller="flmpc", qp_enabled=True, capture=False,
                              spill_monitor=rec, vision_cadence_steps=STRIDE, ee_speed_limit=True,
                              inertial_cup=True, max_steps=args.flmpc_max_steps)
    m = info["model"]; A = Addr(m)
    k_rec0 = int(round(demo.RIGHT_START_T / DT))
    T = len(log["t"])
    tilt0 = float(np.degrees(log["true_cup_tilt_rad"][min(k_rec0, T - 1)]))
    if tilt0 > UPRIGHT_MAX_DEG:
        return None, dict(reason="not_upright", tilt=tilt0)
    sp = np.asarray(log["spilled"], np.int16)
    if sp[k_rec0] > 0:
        return None, dict(reason="spilled_before_motion", n=int(sp[k_rec0]))
    keep = [i for i, k in enumerate(rec.k) if k >= k_rec0 and k < T]
    snaps_q = [rec.q[i] for i in keep]; snaps_v = [rec.v[i] for i in keep]; snap_k = [rec.k[i] for i in keep]
    cup_pos = np.asarray(log["cup_pos"]); cup_quat = np.zeros((T, 4))
    cup_quat[:] = np.nan
    col = np.zeros((T, 3), np.int8)
    col[:, 0] = (np.asarray(log["collision"]) > 0.5)
    col[:, 1] = (np.asarray(log["right_cup_clearance_m"]) <= 0.0)
    for i, k in enumerate(snap_k):
        cup_quat[k] = snaps_q[i][A.cq + 3:A.cq + 7]
    cont = C.Contents(**C.NOMINAL)
    ep = dict(mode="flmpc", carry=args.carry, family="flmpc_crossing", seed=int(seed), contents=cont.to_dict(),
              motion=dict(v_ee_max=v_ee, n_cubes=n_cubes, grasp_mode=grasp_mode), idle_s=0.0, right={},
              k_settle=k_rec0, k_motion=k_rec0, action_kind="realized_next_q", tunnel=False,
              upright_tilt_deg=tilt0)
    dense = dict(spilled=sp, cup_pos=cup_pos, cup_quat=cup_quat, cup_axis=np.asarray(log["cup_n_axis"]), col=col,
                 ql=np.asarray(log["left_q"]),
                 qr=np.asarray(log["right_q"]), cmd_acc=None,
                 lf=np.full(T, np.nan), rf=np.full(T, np.nan))
    return (m, A, cont, ep, dense, snaps_q, snaps_v, snap_k), None


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


# ============================================================================ obs/labels
class Cams:
    def __init__(self, m, S):
        self.m, self.S = m, S
        self.r_zed = mujoco.Renderer(m, ZED_H, ZED_W)
        self.r_w = mujoco.Renderer(m, S, S)
        self.zid = m.camera("zed_cam").id
        f = 0.5 * ZED_H / np.tan(np.radians(m.cam_fovy[self.zid]) / 2)
        self.f = f

    def close(self):
        self.r_zed.close(); self.r_w.close()

    def render(self, d, cup_center):
        import cv2
        self.r_zed.update_scene(d, camera="zed_cam"); z = self.r_zed.render()
        cp = d.cam_xpos[self.zid]; Rm = d.cam_xmat[self.zid].reshape(3, 3)
        pc = Rm.T @ (cup_center - cp); depth = max(-pc[2], 0.05)
        u = ZED_W / 2 + self.f * pc[0] / depth; v = ZED_H / 2 - self.f * pc[1] / depth
        hs = max(8, int(round(self.f * ZED_CROP_M / 2 / depth)))
        pad = np.pad(z, ((hs, hs), (hs, hs), (0, 0)), mode="edge")
        uu, vv = int(round(u)) + hs, int(round(v)) + hs
        uu = int(np.clip(uu, hs, pad.shape[1] - hs)); vv = int(np.clip(vv, hs, pad.shape[0] - hs))
        crop = pad[vv - hs:vv + hs, uu - hs:uu + hs]
        crop = cv2.resize(crop, (self.S, self.S), interpolation=cv2.INTER_AREA)
        self.r_w.update_scene(d, camera="left_arm_egocentric"); lw = self.r_w.render().copy()
        self.r_w.update_scene(d, camera="right_arm_egocentric"); rw = self.r_w.render().copy()
        return crop, lw, rw, (u, v, hs)


def build_arrays(pack, args):
    m, A, cont, ep, dense, snaps_q, snaps_v, snap_k = pack
    n = len(snap_k)
    T = len(dense["spilled"])
    sp = dense["spilled"].astype(np.int16)
    k0 = ep["k_settle"]
    on = np.flatnonzero(sp[k0:] > 0)
    onset = int(k0 + on[0]) if len(on) else -1
    ks = np.array(snap_k)
    out = {}
    prop = np.zeros((n, 26), np.float32); act = np.zeros((n, 14), np.float32)
    for i in range(n):
        q, v = snaps_q[i], snaps_v[i]
        prop[i] = np.concatenate([q[A.lq], v[A.ld], q[A.rq], v[A.rd], [q[A.lf[0]]], [q[A.rf[0]]]])
    for i in range(n):
        j = min(i + 1, n - 1)
        act[i] = np.concatenate([snaps_q[j][A.lq], snaps_q[j][A.rq], [snaps_q[j][A.lf[0]]], [snaps_q[j][A.rf[0]]]])
    # realized cup kinematics (FK) at 200 Hz -> WM rate
    cp = dense["cup_pos"]
    cq = dense["cup_quat"].copy()
    if np.isnan(cq).any():   # flmpc: quats only at snapshots -> recompute from FK of snapshots
        cq = None
    vel = np.gradient(cp, DT, axis=0); acc = np.gradient(vel, DT, axis=0)
    if cq is not None:
        w = quat_series_angvel(cq); wd = np.gradient(w, DT, axis=0)
    else:
        w = np.zeros_like(cp); wd = np.zeros_like(cp)
    pose = np.zeros((n, 7), np.float32); cv = np.zeros((n, 6), np.float32); ca = np.zeros((n, 6), np.float32)
    acmd = np.zeros((n, 6), np.float32)
    if cq is None:   # flmpc: angular velocity from the (inertial-carry) cup free-joint qvel at snapshots
        for i, k in enumerate(ks):
            Rk = np.empty(9); mujoco.mju_quat2Mat(Rk, snaps_q[i][A.cq + 3:A.cq + 7])
            w[k] = Rk.reshape(3, 3) @ snaps_v[i][A.cv + 3:A.cv + 6]
        for i in range(1, len(ks)):   # angular accel: snapshot-to-snapshot difference, held over the interval
            wd[ks[i - 1] + 1:ks[i] + 1] = (w[ks[i]] - w[ks[i - 1]]) / ((ks[i] - ks[i - 1]) * DT)
    for i, k in enumerate(ks):
        qk = snaps_q[i][A.cq + 3:A.cq + 7]
        pose[i] = np.concatenate([snaps_q[i][A.cq:A.cq + 3], qk])
        cv[i] = np.concatenate([vel[k], w[k]])
        lo = max(0, k - STRIDE + 1)
        ca[i] = np.concatenate([acc[lo:k + 1].mean(0), wd[lo:k + 1].mean(0)])
        hi = min(T, k + STRIDE)
        src = dense["cmd_acc"] if dense["cmd_acc"] is not None else np.concatenate([acc, wd], 1)
        acmd[i] = src[k:hi].mean(0)
    col = dense["col"]
    colw = np.zeros((n, 3), np.int8)
    for i, k in enumerate(ks):
        lo = 0 if i == 0 else ks[i - 1] + 1
        colw[i] = col[lo:k + 1].max(0) if k + 1 > lo else col[k]
    ttsp = np.full(n, np.inf, np.float32)
    if onset >= 0:
        ttsp = np.maximum(0, (onset - ks) * DT).astype(np.float32)
    out.update(proprio=prop, action=act, action_cup=acmd, cup_pose=pose, cup_vel=cv, cup_acc=ca,
               spilled=sp[ks], spill_frac=(sp[ks] / cont.n_balls).astype(np.float32), time_to_spill=ttsp,
               collision=colw[:, 0], right_cup_contact=colw[:, 1], right_ball_contact=colw[:, 2],
               t=((ks - k0) * DT).astype(np.float32))
    # render
    if not args.no_render:
        cams = Cams(m, args.img_size)
        d = mujoco.MjData(m)
        Z = np.zeros((n, args.img_size, args.img_size, 3), np.uint8); L = np.zeros_like(Z); R = np.zeros_like(Z)
        for i in range(n):
            d.qpos[:] = snaps_q[i]; mujoco.mj_forward(m, d)
            Rc = d.xmat[A.cup].reshape(3, 3)
            Z[i], L[i], R[i], _ = cams.render(d, d.xpos[A.cup] + Rc[:, 2] * 0.04)
        cams.close()
        out.update(img_zed=Z, img_lwrist=L, img_rwrist=R)
    ep.update(n_frames=n, onset_step=onset, onset_frame=int(np.searchsorted(ks, onset)) if onset >= 0 else -1,
              onset_t_from_rec0=(onset - k0) * DT if onset >= 0 else None,
              onset_t_from_motion=(onset - ep["k_motion"]) * DT if onset >= 0 else None,
              spill_any=bool(onset >= 0), spill_final=int(sp[-1]), n_balls=cont.n_balls,
              collision_any=bool(colw[:, 0].any()), right_cup_any=bool(colw[:, 1].any()),
              right_ball_any=bool(colw[:, 2].any()), dense_len=int(T - k0))
    out["spilled_200hz"] = sp[k0:]
    # dense 200 Hz cup kinematics (from the recording start k0), for physics baselines / GRUs
    axis = np.zeros((T, 3))
    if cq is not None:
        for k in range(T):
            Rk = np.empty(9); mujoco.mju_quat2Mat(Rk, cq[k]); axis[k] = Rk.reshape(3, 3)[:, 2]
    else:
        axis = dense["cup_axis"]
    cmd = dense["cmd_acc"] if dense["cmd_acc"] is not None else np.full((T, 6), np.nan)
    out["dense"] = dict(cup_pos_200hz=cp[k0:].astype(np.float32),
                        cup_quat_200hz=(cq[k0:] if cq is not None else dense["cup_quat"][k0:]).astype(np.float32),
                        cup_axis_200hz=axis[k0:].astype(np.float32),
                        cup_vel_200hz=np.concatenate([vel, w], 1)[k0:].astype(np.float32),
                        cup_acc_200hz=np.concatenate([acc, wd], 1)[k0:].astype(np.float32),
                        cup_acc_cmd_200hz=cmd[k0:].astype(np.float32),
                        collision_200hz=dense["col"][k0:].astype(np.int8))
    if dense.get("pend") is not None:
        phi, psp = dense["pend"]
        out["pend_phi_deg"] = phi[ks - k0].astype(np.float32)
        out["pend_spill"] = np.maximum.accumulate(psp)[ks - k0].astype(np.int8)
        ep["pend_spill_any"] = bool(psp.any())
    return out, ep


# ============================================================================ shard worker
KEYS_FRAME = ("proprio", "action", "action_cup", "cup_pose", "cup_vel", "cup_acc", "spilled", "spill_frac",
              "time_to_spill", "collision", "right_cup_contact", "right_ball_contact", "t",
              "img_zed", "img_lwrist", "img_rwrist", "pend_phi_deg", "pend_spill")


def run_shard(job):
    shard_id, seeds_modes, args = job
    if not args.no_render:
        os.environ.setdefault("MUJOCO_GL", "egl")
    t0 = time.perf_counter()
    eps, rejects, arrays = [], [], {k: [] for k in KEYS_FRAME}
    dense_all, timing, dense_kin = [], [], {}
    for seed, mode in seeds_modes:
        ts = time.perf_counter()
        try:
            pack, rej = (episode_rig if mode == "rig" else episode_flmpc)(seed, args)
        except NotImplementedError:
            raise
        except Exception as e:
            pack, rej = None, dict(reason="exception", err=traceback.format_exc()[-600:])
        tsim = time.perf_counter() - ts
        if pack is None:
            rej.update(seed=int(seed), mode=mode); rejects.append(_jsonable(rej)); continue
        tr = time.perf_counter()
        out, ep = build_arrays(pack, args)
        trend = time.perf_counter() - tr
        ep.update(sim_s=tsim, obs_s=trend, shard=shard_id)
        if getattr(args, "pendulum_labels", False) and "pend_phi_deg" not in out:   # keep shard arrays aligned
            out["pend_phi_deg"] = np.full(ep["n_frames"], np.nan, np.float32)
            out["pend_spill"] = np.full(ep["n_frames"], -1, np.int8)
        for k in KEYS_FRAME:
            if k in out:
                arrays[k].append(out[k])
        dense_all.append(out["spilled_200hz"])
        for dk, dv in out["dense"].items():
            dense_kin.setdefault(dk, []).append(dv)
        eps.append(_jsonable(ep))
    if eps:
        lens = np.array([e["n_frames"] for e in eps]); starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        dl = np.array([len(x) for x in dense_all]); ds = np.concatenate([[0], np.cumsum(dl)[:-1]])
        save = {k: np.concatenate(v) for k, v in arrays.items() if v}
        is_first = np.zeros(int(lens.sum()), bool); is_first[starts] = True
        save.update(ep_start=starts, ep_len=lens, ep_seed=np.array([e["seed"] for e in eps]),
                    ep_onset_frame=np.array([e["onset_frame"] for e in eps]),
                    ep_n_balls=np.array([e["n_balls"] for e in eps]), is_first=is_first,
                    ep_index=np.repeat(np.arange(len(eps)), lens).astype(np.int32),
                    spilled_200hz=np.concatenate(dense_all), ep_dense_start=ds, ep_dense_len=dl,
                    **{k: np.concatenate(v) for k, v in dense_kin.items()})
        path = Path(args.out) / f"shard_{shard_id}.npz"
        (np.savez if args.no_compress else np.savez_compressed)(path, **save)
    return dict(shard=shard_id, eps=eps, rejects=rejects, wall_s=time.perf_counter() - t0)


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-episodes", type=int, default=20)
    ap.add_argument("--seeds", type=str, default=None,
                    help="global seed slice 'A-B' (inclusive) -- for many small Slurm jobs; overrides --seed0/--n-episodes")
    ap.add_argument("--modes", default="rig:0.85,flmpc:0.15")
    ap.add_argument("--carry", default="faked_inertial", choices=("faked_inertial", "real_grasp"))
    ap.add_argument("--seed0", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eps-per-shard", type=int, default=25)
    ap.add_argument("--img-size", type=int, default=112)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--no-compress", action="store_true")
    ap.add_argument("--no-randomize-contents", dest="randomize_contents", action="store_false")
    ap.add_argument("--flmpc-max-steps", type=int, default=None)
    ap.add_argument("--family-weights", type=str, default=None,
                    help="override rig family weights, e.g. 'pour:6,tilt:2,shake:2,mixed:2,translate:1,bangbang:1,vertical:1'")
    ap.add_argument("--particle-frac", type=float, default=0.0,
                    help="share of rig episodes with liquid-like particle contents (40-80 soft frictionless spheres)")
    ap.add_argument("--pendulum-labels", action="store_true",
                    help="add equivalent-mechanical liquid-proxy labels pend_phi_deg / pend_spill (rig mode)")
    ap.add_argument("--egl-device", type=str, default=None,
                    help="MUJOCO_EGL_DEVICE_ID for rendering (EGL index == nvidia-smi index on westeros; verified for 6)")
    args = ap.parse_args()
    if not args.no_render:
        os.environ["MUJOCO_GL"] = "egl"
        dev = args.egl_device or os.environ.get("MUJOCO_EGL_DEVICE_ID")
        if dev is None and os.environ.get("CUDA_VISIBLE_DEVICES"):
            dev = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]   # Slurm --gres=gpu:1 sets this
        if dev is None:
            sys.exit("rendering needs an explicit GPU: pass --egl-device N (never 0 on westeros = display), "
                     "or set CUDA_VISIBLE_DEVICES / MUJOCO_EGL_DEVICE_ID, or use --no-render")
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(dev)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    names, wts = zip(*[(s.split(":")[0], float(s.split(":")[1]) if ":" in s else 1.0) for s in args.modes.split(",")])
    cw = np.cumsum(np.array(wts) / sum(wts))
    if args.seeds:
        lo, hi = map(int, args.seeds.split("-"))
        seeds = np.arange(lo, hi + 1)
    else:
        seeds = args.seed0 + np.arange(args.n_episodes)
    # mode is a deterministic function of the seed (slices of a global seed range stay consistent)
    modes = [names[int(np.searchsorted(cw, np.random.default_rng([int(sd_), 7]).random(), side="right"))] for sd_ in seeds]
    items = list(zip(seeds.tolist(), modes))
    E = args.eps_per_shard
    jobs = [(f"{items[i][0]:07d}-{items[min(i + E, len(items)) - 1][0]:07d}", items[i:i + E], args)
            for i in range(0, len(items), E)]
    t0 = time.perf_counter()
    res = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(args.workers, len(jobs)), maxtasksperchild=4) as pool:
        for r in pool.imap_unordered(run_shard, jobs):
            res.append(r)
            print(f"[shard {r['shard']}] {len(r['eps'])} eps, {len(r['rejects'])} rejects, "
                  f"{r['wall_s']:.0f}s  (elapsed {time.perf_counter() - t0:.0f}s)", flush=True)
    wall = time.perf_counter() - t0
    eps = [e for r in sorted(res, key=lambda r: str(r["shard"])) for e in r["eps"]]
    rej = [x for r in res for x in r["rejects"]]
    manifest = dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"), generator="world_model/v2/data/gen_spill_data.py",
        args={k: v for k, v in vars(args).items()}, physics_dt=DT, stride=STRIDE, wm_dt=STRIDE * DT,
        img_size=args.img_size, zed_crop_m=ZED_CROP_M, zed_note="zed_cam = PROPOSED torso mount (scripts/zed_camera.py)",
        proprio_layout="[left q(6), left qd(6), right q(6), right qd(6), left finger q, right finger q]",
        action_layout="[left q_ref(6), right q_ref(6), left finger ref, right finger ref] at t+1 (absolute)",
        action_cup_layout="[lin acc(3) m/s^2, ang acc(3) rad/s^2] world, mean over next WM interval",
        labels="spilled count; spill_frac; time_to_spill s (inf never); collision / right_cup_contact / right_ball_contact SEPARATE",
        contents_ranges=C.RANGES, contents_nominal=C.NOMINAL, upright_max_deg=UPRIGHT_MAX_DEG,
        n_episodes=len(eps), n_rejected=len(rej), wall_s=wall,
        shards=sorted({f"shard_{e['shard']}.npz" for e in eps}), episodes=eps, rejected=rej)
    mname = f"manifest_{seeds[0]:07d}-{seeds[-1]:07d}.json" if args.seeds else "manifest.json"
    (out / mname).write_text(json.dumps(_jsonable(manifest), indent=1))
    print(f"done: {len(eps)} episodes, {len(rej)} rejected, wall {wall:.0f}s -> {out / mname}", flush=True)


if __name__ == "__main__":
    main()
