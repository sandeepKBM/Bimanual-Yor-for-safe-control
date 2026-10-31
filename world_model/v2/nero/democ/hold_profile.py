"""democ/hold_profile.py -- full Demo-C HOLD task profile + OPT-IN hold fixes (real friction, no weld).  SIM ONLY, CPU.

Profile (sim dt 0.005 s, controller called every sim step = 200 Hz, as grasp_smoke):
  settle 0.5 -> close 0.7 -> settle 0.3 -> lift 5 cm 1.5 -> hold1 1.0 (creep window) -> rotate box about +y to ROT_DEG 2.0
  -> hold2 0.3 -> rotate back 2.0 -> hold3 0.3 -> set down (reverse lift) 1.5 -> hold4 0.3 -> release (fingers open) 0.5 -> end 0.3.
Hands follow the box rigidly: IK of p(th) = c + Ry(th)(p0-c), R(th) = Ry(th) R0 from the ACTUAL lifted hand poses (continuous branch).

OPT-IN fixes (HoldFix; default = everything off = the 10-21 payload-ff 'internal' baseline; mpc_fl.py is NOT modified):
  * joint_k   : per-joint hold-phase impedance torque added AFTER FL-MPC,  tau += K (q_ref - q) + Kd (qd_ref - qd),
                Kd = 2 sqrt(K M_jj) (M_jj from mj_fullM at enable time), capped so Kd dt / M_jj <= 0.9 (explicit-step stability);
                units Nm/rad, Nm s/rad; sensors used: joint encoders only.  Total command is clipped to the torque limit.
  * grip_force_N : after the close phase the finger actuator is switched from the position servo (kp 200, ctrl 0) to a constant
                closing FORCE command (the real gripper takes a width + force command; magnitude <= URDF effort 10 N). Restored
                to the position servo for release.
python -m world_model.v2.nero.democ.hold_profile --mu 0.5 --mass 0.6 [--fix] [--out f.json]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco

from .. import config as C
from .. import adapters as A
from . import scene as S
from . import kin as K
from .grasp_smoke import PayloadCtl, DT, quat_angle_deg
from ..hold_test import Ctl

LIFT = 0.05
ROT_DEG = 40.0
GRIP_F_MAX = C.GRIPPER_EFFORT          # 10 N, official URDF <limit effort>
SCHED = dict(settle=0.5, close=0.7, settle2=0.3, lift=1.5, hold1=1.0, rot=2.0, hold2=0.3, back=2.0, hold3=0.3, lower=1.5,
             hold4=0.3, release=0.5, tail=0.3)


@dataclass
class HoldFix:
    joint_k: tuple | None = None          # Nm/rad per joint (7), None = off
    grip_force_N: float | None = None     # constant closing force per gripper (<= 10 N), None = position servo as before
    payload_mass: float | None = None     # assumed box mass for the payload feed-forward (None = true mass)

    def label(self):
        return "fix" if (self.joint_k is not None or self.grip_force_N is not None) else "baseline"


FIX_DEFAULT = HoldFix(joint_k=(60.0, 60.0, 40.0, 60.0, 150.0, 150.0, 150.0), grip_force_N=9.0)   # tuned on mu 0.5 / 0.6 kg only (see results JSON)


def _smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def run(p: S.DemoCParams, fix: HoldFix | None = None, payload_ff="internal", rot_deg=ROT_DEG, tau_preset=None, pad_mu=None):
    fix = fix or HoldFix()
    tau = C.tau_limits(tau_preset)
    if fix.grip_force_N is not None:
        assert 0 < fix.grip_force_N <= GRIP_F_MAX + 1e-9, "gripper force exceeds URDF effort"
    sch = dict(SCHED)
    m = S.compose(p, dt=DT)
    if pad_mu is not None:      # MuJoCo uses max(geom1, geom2) sliding friction; the gripper pads are mu=2 (class collision_gripper), so
        for s_ in C.SIDES:      # box_mu < 2 has NO effect unless the pad friction is lowered too.  This sets the EFFECTIVE pad-box mu.
            for gn in (f"{s_}_arm_finger1_col", f"{s_}_arm_finger2_col", f"{s_}_arm_gripper_base_col"):
                m.geom_friction[m.geom(gn).id, 0] = pad_mu
    pre = K.find_pregrasp(m, p)
    d = mujoco.MjData(m)
    K.set_state(m, d, p, pre["ql"], pre["qr"], K.FINGER_OPEN)
    A.disable_arm_servos(m)
    ql0, qr0 = pre["ql"], pre["qr"]
    gb = {s: m.body(f"{s}_arm_gripper_base").id for s in C.SIDES}
    box = m.body("democ_box").id
    if payload_ff == "internal":
        ctl = PayloadCtl(m, dict(left=ql0.tolist(), right=qr0.tolist()), tau, box, gb, mass=fix.payload_mass)
    else:
        ctl = Ctl(m, dict(left=ql0.tolist(), right=qr0.tolist()), tau)
    NP, FDT = ctl.fl.NP, ctl.fl.DT
    fing = {s: {m.geom(f"{s}_arm_finger1_col").id: "f1", m.geom(f"{s}_arm_finger2_col").id: "f2"} for s in C.SIDES}
    act_g = {s: m.actuator(f"{s}_arm_gripper_pos").id for s in C.SIDES}
    gain0 = {s: (m.actuator_gainprm[act_g[s]].copy(), m.actuator_biasprm[act_g[s]].copy()) for s in C.SIDES}
    bgeom = m.geom("democ_box_geom").id

    # ---- schedule
    t = {}
    tt = 0.0
    for k_, v in sch.items():
        t[k_] = (tt, tt + v)
        tt += v
    t_close1 = t["close"][1]
    t_end = tt
    n = int(round(t_end / DT))
    phases = [k_ for k_ in ("lift", "hold1", "rot", "hold2", "back", "hold3", "lower", "hold4")]

    chains = {}                                   # filled at t_lift0: "lift": {L,R}, "rot": {L,R}

    def seg(chain, key, tq):
        u = _smooth((tq - t[key][0]) / sch[key]) * (len(chain["L"]) - 1)
        i = min(int(u), len(chain["L"]) - 2)
        a = u - i
        return tuple((1 - a) * chain[s][i] + a * chain[s][i + 1] for s in "LR")

    def ref(tq):
        if "lift" not in chains:
            return ql0, qr0
        if tq < t["rot"][0]:
            return seg(chains["lift"], "lift", tq)
        if tq < t["back"][0]:
            return seg(chains["rot"], "rot", tq)
        if tq < t["lower"][0]:
            return seg(chains["rot_r"], "back", tq)
        return seg(chains["lift_r"], "lower", tq)

    def build_chains():
        dofs = (("L", "left", ctl.lq), ("R", "right", ctl.rq))
        lift = {}
        piks = {}
        for key, side, dof in dofs:
            pik = K.PoseIK(m, side); piks[key] = pik
            q = d.qpos[dof].copy(); p0, R0 = pik.ik.pose(q)
            lst = [q.copy()]
            for i in range(1, 61):
                u = i / 60
                q, e = pik.solve(p0 + np.array([0, 0, LIFT * u * u * (3 - 2 * u)]), R0, q, q_post=d.qpos[dof].copy(), k_post=0.1, iters=60, tol=1e-9)
                lst.append(q.copy())
            lift[key] = lst
        chains["lift"] = lift
        c = np.array(p.box_centre, float) + np.array([0, 0, LIFT])
        rot = {}
        marg = 9.0
        for key, side, dof in dofs:
            pik = piks[key]
            q = lift[key][-1].copy(); p0, R0 = pik.ik.pose(q)
            lst = [q.copy()]
            NW = 80
            for i in range(1, NW + 1):
                th = np.radians(rot_deg) * i / NW
                q, e = pik.solve(c + K.Ry(th) @ (p0 - c), K.Ry(th) @ R0, q, q_post=lift[key][-1], k_post=0.05, iters=60, tol=1e-9)
                if e > 1e-5:
                    raise RuntimeError(f"rotation IK failed {side} th={np.degrees(th):.1f} e={e:.2e}")
                lst.append(q.copy())
                marg = min(marg, pik.margin(q)[0])
            rot[key] = lst
        chains["rot"] = rot
        chains["rot_r"] = {s_: rot[s_][::-1] for s_ in "LR"}
        chains["lift_r"] = {s_: lift[s_][::-1] for s_ in "LR"}
        chains["margin"] = marg

    # ---- fix state
    Kj = None if fix.joint_k is None else np.asarray(fix.joint_k, float)
    Mdiag = {}
    grip_mode = "pos"

    def set_grip_mode(mode):
        nonlocal grip_mode
        for s in C.SIDES:
            a = act_g[s]
            if mode == "force":
                m.actuator_gainprm[a, :] = 0.0
                m.actuator_biasprm[a, :] = 0.0
                m.actuator_biasprm[a, 0] = -fix.grip_force_N       # force = -F (closing; same sign as the kp*(0-q) servo)
            else:
                m.actuator_gainprm[a] = gain0[s][0]
                m.actuator_biasprm[a] = gain0[s][1]
        grip_mode = mode

    # ---- logging
    ref_rel = {}
    slip = {s: [] for s in C.SIDES}               # (t, dp_m, da_deg, dvec)
    fing_f = {s: {"f1": [], "f2": []} for s in C.SIDES}
    act_force = {s: [] for s in C.SIDES}
    cnt_hist = {s: [] for s in C.SIDES}
    tau_peak = np.zeros(14)
    sag = {s: [] for s in C.SIDES}                # q - q_ref during hold1
    tau_hold = {s: [] for s in C.SIDES}
    ts_c = []
    box_z, tilt = [], []
    ctrl_open = True
    z0 = None
    t_rel0 = t["release"][0]
    for k in range(n):
        tk = k * DT
        if "lift" not in chains and tk >= t["lift"][0] - 1e-9:
            build_chains()
        qe_ref = np.stack([ref(tk + i * FDT)[0] for i in range(NP + 1)])
        qi_ref = np.stack([ref(tk + i * FDT)[1] for i in range(NP + 1)])
        ctl.qe_ref, ctl.qi_ref = qe_ref, qi_ref
        for s in C.SIDES:
            d.ctrl[act_g[s]] = K.FINGER_OPEN if (tk < t["close"][0] or tk >= t_rel0) else 0.0
        if fix.grip_force_N is not None:
            want = "force" if (t_close1 <= tk < t_rel0) else "pos"
            if want != grip_mode:
                set_grip_mode(want)
        if payload_ff == "internal" and tk >= t_close1:
            ctl.on = True
        t0 = time.perf_counter()
        te, ti = ctl.tau(d)
        ts_c.append((time.perf_counter() - t0) * 1e3)
        if Kj is not None and tk >= t_close1:
            if not Mdiag:
                Mf = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, Mf, d.qM)
                Mdiag["L"] = np.diag(Mf)[ctl.ld].copy(); Mdiag["R"] = np.diag(Mf)[ctl.rd].copy()
            r0, r1 = ref(tk), ref(tk + DT)
            for key, dof, qa, tcmd in (("L", ctl.ld, ctl.lq, te), ("R", ctl.rd, ctl.rq, ti)):
                qd_r = (r1["LR".index(key)] - r0["LR".index(key)]) / DT
                Md = np.maximum(Mdiag[key], 1e-3)
                Kd = np.minimum(2.0 * np.sqrt(Kj * Md), 0.9 * Md / DT)
                tcmd += Kj * (r0["LR".index(key)] - d.qpos[qa]) + Kd * (qd_r - d.qvel[dof])
                np.clip(tcmd, -tau, tau, out=tcmd)
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[ctl.ld] = te
        d.qfrc_applied[ctl.rd] = ti
        tau_peak = np.maximum(tau_peak, np.abs(np.concatenate([te, ti])))
        mujoco.mj_step(m, d)
        # --- contacts
        cnt = {s: 0 for s in C.SIDES}
        ff_ = {s: {"f1": 0.0, "f2": 0.0} for s in C.SIDES}
        fv = np.zeros(6)
        for j in range(d.ncon):
            c = d.contact[j]
            if bgeom not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == bgeom else c.geom1
            for s in C.SIDES:
                if other in fing[s]:
                    cnt[s] += 1
                    mujoco.mj_contactForce(m, d, j, fv)
                    ff_[s][fing[s][other]] += float(fv[0])
        for s in C.SIDES:
            cnt_hist[s].append(cnt[s])
            fing_f[s]["f1"].append(ff_[s]["f1"]); fing_f[s]["f2"].append(ff_[s]["f2"])
            act_force[s].append(float(d.actuator_force[act_g[s]]))
        R_b = d.xmat[box].reshape(3, 3).copy()
        box_z.append(float(d.xpos[box][2]))
        tilt.append(float(np.degrees(np.arccos(np.clip(R_b[2, 2], -1, 1)))))
        if tk >= t["lift"][0] - 1e-9:
            if not ref_rel:
                z0 = float(d.xpos[box][2])
                for s in C.SIDES:
                    Rg = d.xmat[gb[s]].reshape(3, 3)
                    ref_rel[s] = (Rg.T @ (d.xpos[box] - d.xpos[gb[s]]), Rg.T @ R_b)
            for s in C.SIDES:
                Rg = d.xmat[gb[s]].reshape(3, 3)
                rp, rR = ref_rel[s]
                dv = Rg.T @ (d.xpos[box] - d.xpos[gb[s]]) - rp
                slip[s].append((tk, float(np.linalg.norm(dv)), quat_angle_deg(rR, Rg.T @ R_b), dv.copy()))
        if t["hold1"][0] <= tk < t["hold1"][1]:
            for s, qa, dof, key in (("left", ctl.lq, ctl.ld, "L"), ("right", ctl.rq, ctl.rd, "R")):
                sag[s].append(d.qpos[qa] - ref(tk)["LR".index(key)])
                tau_hold[s].append((te if s == "left" else ti).copy())

    # ---- metrics
    def tsl(s, key):                              # slip samples in [t0,t1)
        return np.array([x[1:3] for x in slip[s] if t[key][0] <= x[0] < t[key][1]])

    def mm(a):
        return float(1e3 * a)

    until_rel = lambda s: np.array([x[1:3] for x in slip[s] if x[0] < t_rel0])
    res = dict(variant=fix.label(), fix=dataclasses.asdict(fix), payload_ff=payload_ff, rot_deg=rot_deg,
               mu=p.box_mu if pad_mu is None else pad_mu, pad_mu_override=pad_mu, mass=p.box_mass, joint6_margin_rad=chains["margin"])
    res["slip_max_mm"] = {s: mm(until_rel(s)[:, 0].max()) for s in C.SIDES}
    res["slip_max_deg"] = {s: float(until_rel(s)[:, 1].max()) for s in C.SIDES}
    res["slip_by_phase_mm_deg"] = {s: {ph: [mm(tsl(s, ph)[:, 0].max()), float(tsl(s, ph)[:, 1].max())] for ph in phases} for s in C.SIDES}
    creep = {}
    for s in C.SIDES:                               # linear fit of the slip VECTOR over the last 0.7 s of hold1
        w = [x for x in slip[s] if t["hold1"][1] - 0.7 <= x[0] < t["hold1"][1]]
        tv = np.array([x[0] for x in w]); V = np.stack([x[3] for x in w])
        creep[s] = float(1e3 * np.linalg.norm([np.polyfit(tv, V[:, i], 1)[0] for i in range(3)]))
    res["creep_mm_s"] = creep
    i0, i1 = int(t["hold1"][0] / DT), int(t["hold1"][1] / DT)
    res["finger_normal_N_hold1"] = {s: {f: float(np.mean(v[i0:i1])) for f, v in fing_f[s].items()} for s in C.SIDES}
    res["finger_normal_N_max"] = {s: float(max(max(v) for v in fing_f[s].values())) for s in C.SIDES}
    res["gripper_actuator_force_N"] = {s: dict(hold1_mean=float(np.mean(np.abs(act_force[s][i0:i1]))), max=float(np.max(np.abs(act_force[s])))) for s in C.SIDES}
    res["gripper_force_limit_N"] = GRIP_F_MAX
    res["tau_limit_Nm"] = float(tau.max())
    res["tau_peak_Nm"] = dict(left=tau_peak[:7].tolist(), right=tau_peak[7:].tolist(), max=float(tau_peak.max()))
    res["sag_rad_hold1_mean"] = {s: np.mean(sag[s], 0).tolist() for s in C.SIDES}
    res["tau_hold1_mean_Nm"] = {s: np.mean(tau_hold[s], 0).tolist() for s in C.SIDES}
    ir = int(t_rel0 / DT)
    res["box_z_m"] = dict(start=z0, before_release=box_z[ir - 1], end=box_z[-1])
    lost = [any(c == 0 for c in cnt_hist[s][int(t["lift"][0] / DT):ir]) for s in C.SIDES]
    res["max_contact_loss_steps"] = {s: int(sum(c == 0 for c in cnt_hist[s][int(t["lift"][0] / DT):ir])) for s in C.SIDES}
    res["box_dropped"] = bool(res["slip_max_mm"]["left"] > 30 or res["slip_max_mm"]["right"] > 30 or box_z[ir - 1] < z0 - 0.03 - LIFT
                              or max(res["max_contact_loss_steps"].values()) > 20)
    res["max_tilt_deg"] = float(max(tilt[:ir]))
    res["box_end_tilt_deg"] = tilt[-1]
    res["n_infeasible"] = int(ctl.n_infeasible); res["n_fallback"] = int(ctl.n_fallback)
    res["ctrl_ms_mean"] = float(np.mean(ts_c[5:]))
    res["control"] = dict(dt_s=DT, rate_hz=1.0 / DT, mpc_dt_s=FDT, horizon=NP)
    tol = dict(slip_mm=2.0, slip_deg=1.0, creep_mm_s=0.2)
    res["meets_target"] = bool((not res["box_dropped"]) and max(res["slip_max_mm"].values()) < tol["slip_mm"]
                               and max(res["slip_max_deg"].values()) < tol["slip_deg"] and max(creep.values()) < tol["creep_mm_s"]
                               and res["tau_peak_Nm"]["max"] <= res["tau_limit_Nm"] + 1e-6
                               and max(v["max"] for v in res["gripper_actuator_force_N"].values()) <= GRIP_F_MAX + 1e-6)
    return res


def main(argv=None):
    ap = S.add_args(argparse.ArgumentParser())
    ap.add_argument("--fix", action="store_true", help="enable the default HoldFix (joint impedance + grip force)")
    ap.add_argument("--joint-k", type=float, nargs=7, default=None)
    ap.add_argument("--grip-force", type=float, default=None)
    ap.add_argument("--payload-mass", type=float, default=None)
    ap.add_argument("--pad-mu", type=float, default=None, help="override pad (and box) friction so the EFFECTIVE contact mu = this")
    ap.add_argument("--rot-deg", type=float, default=ROT_DEG)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    fix = dataclasses.replace(FIX_DEFAULT) if a.fix else HoldFix()
    if a.joint_k is not None:
        fix.joint_k = tuple(a.joint_k)
    if a.grip_force is not None:
        fix.grip_force_N = a.grip_force
    fix.payload_mass = a.payload_mass
    res = run(S.params_from_args(a), fix=fix, rot_deg=a.rot_deg, pad_mu=a.pad_mu)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("variant", "mu", "mass", "slip_max_mm", "slip_max_deg", "creep_mm_s", "tau_peak_Nm",
                                          "gripper_actuator_force_N", "box_dropped", "n_infeasible", "meets_target")}, default=float))


if __name__ == "__main__":
    main()
