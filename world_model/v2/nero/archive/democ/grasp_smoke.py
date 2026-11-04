"""democ/grasp_smoke.py -- bimanual REAL-friction box grasp + 5 cm lift + 2 s hold under the UNMODIFIED FL-MPC (NV=7).  SIM ONLY.

Timeline (sim dt = 0.005 s, controller called every sim step, as nero/hold_test.py):  settle 0.5 s (open, at pre-grasp)
-> close 0.7 s (gripper position targets -> 0; actuator forcerange +-10 N, kp 200 as in the variant) -> settle 0.3 s
-> lift 5 cm over 1.5 s (smoothstep joint-space reference from IK of the lifted hand poses) -> hold 2 s.
NO weld by default.  `--weld` = DEBUG kinematic baseline only (labelled in the JSON); the box is welded to both gripper_base
bodies at the end of the close phase.

    python -m world_model.v2.nero.democ.grasp_smoke [--mass 0.6 --com-x 0 --mu 1.0] [--out results/democ_grasp_smoke.json]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import mujoco

from world_model.v2.nero import config as C
from world_model.v2.nero import adapters as A
from world_model.v2.nero import ik as IK
from world_model.v2.nero.hold_test import Ctl
from world_model.v2.nero.democ import scene as S
from world_model.v2.nero.democ import kin as K
from world_model.v2.nero.democ import payload as PL


class PayloadCtl(Ctl):
    """Ctl with FL-MPC's INTERNAL payload hooks: each arm gets its share of the box (democ/payload.py) as M/bias extras,
    so the payload enters tau = M a + c AND the torque-box rows.  Off until `.on = True` (set after the close phase)."""

    def __init__(self, m, hold_q, tau_vec, box_id, grip_ids, share=0.5, mass=None, **kw):
        super().__init__(m, hold_q, tau_vec, **kw)
        self.box, self.grip, self.share, self.mass, self.on = box_id, grip_ids, share, mass, False

    def tau(self, d):
        if not self.on:
            return super().tau(d)
        mujoco.mj_forward(self.m, d)
        Me, be = PL.arm_payload_terms(self.m, d, self.box, self.grip["left"], self.ld, self.share, self.mass)
        Mi, bi = PL.arm_payload_terms(self.m, d, self.box, self.grip["right"], self.rd, 1.0 - self.share, self.mass)
        pairs = self.pairs(d)
        te, ti, ok = self.fl.step(self.m, d, self.lq, self.ld, self.rq, self.rd, self.qe_ref, self.z, self.qi_ref, self.z, pairs,
                                  M_ego_extra=Me, bias_ego_extra=be, M_int_extra=Mi, bias_int_extra=bi)
        if not ok:
            self.n_infeasible += 1
        if self.fl.last_fallback:
            self.n_fallback += 1
        return te, ti

DT = 0.005
T_SETTLE, T_CLOSE, T_SETTLE2, T_LIFT, T_HOLD = 0.5, 0.7, 0.3, 1.5, 2.0
LIFT = 0.05


def quat_angle_deg(R1, R2):
    c = (np.trace(R1.T @ R2) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def _lift_refs(m, p, pre, dz=LIFT):
    """joint-space refs of the lifted hand poses, warm-started from the pre-grasp (continuous branch)."""
    tg = K.box_grasp_targets(p, 0.0, pre["jaw_sign"])
    out = {}
    for side, q0 in (("left", pre["ql"]), ("right", pre["qr"])):
        pik = K.PoseIK(m, side)
        pt, R = tg[side]
        q, e = pik.solve(pt + np.array([0, 0, dz]), R, q0, q_post=q0, k_post=0.3)
        out[side] = (q, e)
    return out


def run(p: S.DemoCParams, weld=False, tau_preset=None, verbose=False, payload_ff=False, finger_kp=None):
    # payload_ff: False/'off' | True/'external' (static half-weight torque added AFTER FL-MPC, outside its constraints) |
    #             'internal' (FL-MPC payload hooks: half the box's mass/inertia/weight per arm inside tau = M a + c and the torque box)
    payload_ff = {False: 'off', True: 'external', None: 'off'}.get(payload_ff, payload_ff)
    assert payload_ff in ('off', 'external', 'internal'), payload_ff
    tau = C.tau_limit(tau_preset)       # F3 11-03: FL-MPC torque limit, one source (config.tau_limit)
    m = S.compose(p, dt=DT)
    if finger_kp is not None:                       # DIAGNOSTIC override of the ASSUMED finger servo gain (reported in JSON)
        for s_ in C.SIDES:
            a_ = m.actuator(f"{s_}_arm_gripper_pos").id
            m.actuator_gainprm[a_, 0] = finger_kp
            m.actuator_biasprm[a_, 1] = -finger_kp
    pre = K.find_pregrasp(m, p)
    d = mujoco.MjData(m)
    K.set_state(m, d, p, pre["ql"], pre["qr"], K.FINGER_OPEN)
    A.disable_arm_servos(m)
    ql0, qr0 = pre["ql"], pre["qr"]
    lifted = dict(left=(None, float("nan")), right=(None, float("nan")))
    wp = {}          # filled at t_lift0: joint-space waypoints of the Cartesian lift from the ACTUAL (closed-chain) hand poses
    gb = {s: m.body(f"{s}_arm_gripper_base").id for s in C.SIDES}
    box = m.body("democ_box").id
    if payload_ff == 'internal':
        ctl = PayloadCtl(m, dict(left=ql0.tolist(), right=qr0.tolist()), tau, box, gb)
    else:
        ctl = Ctl(m, dict(left=ql0.tolist(), right=qr0.tolist()), tau)
    NP, FDT = ctl.fl.NP, ctl.fl.DT
    gid = {n: m.geom(n).id for n in ("democ_box_geom",)}
    fing = {s: {m.geom(f"{s}_arm_finger1_col").id: "f1", m.geom(f"{s}_arm_finger2_col").id: "f2"} for s in C.SIDES}
    base_col = {s: m.geom(f"{s}_arm_gripper_base_col").id for s in C.SIDES}
    act_g = {s: m.actuator(f"{s}_arm_gripper_pos").id for s in C.SIDES}
    bgeom = gid["democ_box_geom"]
    t_close0 = T_SETTLE
    t_close1 = t_close0 + T_CLOSE
    t_lift0 = t_close1 + T_SETTLE2
    t_lift1 = t_lift0 + T_LIFT
    t_end = t_lift1 + T_HOLD
    n = int(round(t_end / DT))

    def s_of(t):
        u = np.clip((t - t_lift0) / T_LIFT, 0, 1)
        return u * u * (3 - 2 * u)

    def cmdq(t):
        if not wp:
            return ql0, qr0
        u = np.clip((t - t_lift0) / T_LIFT, 0, 1) * (len(wp["L"]) - 1)
        i = min(int(u), len(wp["L"]) - 2); a = u - i
        return (1 - a) * wp["L"][i] + a * wp["L"][i + 1], (1 - a) * wp["R"][i] + a * wp["R"][i + 1]

    def capture_lift():
        """Cartesian lift refs: actual hand poses at the end of settle2 + smoothstep * [0,0,LIFT], IK continuation (rotation
        fixed).  Using the ACTUAL poses keeps both refs rigidly consistent with the grasped box (no internal-force reference)."""
        for key, side, dof in (("L", "left", ctl.lq), ("R", "right", ctl.rq)):
            pik = K.PoseIK(m, side)
            q = d.qpos[dof].copy()
            p0, R0 = pik.ik.pose(q)
            lst = [q.copy()]
            NW = 60
            for i in range(1, NW + 1):
                u = i / NW
                q, e = pik.solve(p0 + np.array([0, 0, LIFT * u * u * (3 - 2 * u)]), R0, q, q_post=d.qpos[dof].copy(), k_post=0.1, iters=60, tol=1e-9)
                lifted[side] = (q, e if np.isnan(lifted[side][1]) else max(e, lifted[side][1]))
                lst.append(q.copy())
            wp[key] = lst
    ref_box = None
    ref_rel = {}
    ts_c = []
    stats = dict(max_tilt=0.0, max_slip_m={"left": 0.0, "right": 0.0}, max_slip_deg={"left": 0.0, "right": 0.0})
    slip_ts = {s: [] for s in C.SIDES}
    fing_f = {s: {"f1": [], "f2": []} for s in C.SIDES}
    forces = {s: [] for s in C.SIDES}              # per-step sum of normal force on the box from each hand's pads
    ncon_hist = {s: [] for s in C.SIDES}
    welded = False
    z_box0 = None
    box_z_trace = []
    for k in range(n):
        t = k * DT
        if not wp and t >= t_lift0 - 1e-9:
            capture_lift()
        # --- reference over the MPC horizon
        qe_ref = np.stack([cmdq(t + i * FDT)[0] for i in range(NP + 1)])
        qi_ref = np.stack([cmdq(t + i * FDT)[1] for i in range(NP + 1)])
        ctl.qe_ref, ctl.qi_ref = qe_ref, qi_ref
        # --- gripper targets
        for s in C.SIDES:
            d.ctrl[act_g[s]] = K.FINGER_OPEN if t < t_close0 else 0.0
        t0 = time.perf_counter()
        if payload_ff == 'internal' and t >= t_close1:
            ctl.on = True
        te, ti = ctl.tau(d)
        ts_c.append((time.perf_counter() - t0) * 1e3)
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[ctl.ld] = te
        d.qfrc_applied[ctl.rd] = ti
        if payload_ff == 'external' and t >= t_close1:       # static feed-forward of half the box weight per arm (J_p(box COM)^T [0,0,m g/2])
            for s_, dof in (("left", ctl.ld), ("right", ctl.rd)):
                jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, jp, jr, d.xipos[box], gb[s_])
                d.qfrc_applied[dof] += jp[:, dof].T @ np.array([0.0, 0.0, 0.5 * m.body_mass[box] * 9.81])
        if weld and not welded and t >= t_close1:        # DEBUG kinematic baseline ONLY
            _weld_box(m, d)
            welded = True
        mujoco.mj_step(m, d)
        # --- contacts / forces
        cnt = {s: 0 for s in C.SIDES}
        fn = {s: 0.0 for s in C.SIDES}
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
                    fn[s] += float(fv[0])
                    ff_[s][fing[s][other]] += float(fv[0])
        for s in C.SIDES:
            ncon_hist[s].append(cnt[s]); forces[s].append(fn[s])
            fing_f[s]["f1"].append(ff_[s]["f1"]); fing_f[s]["f2"].append(ff_[s]["f2"])
        R_b = d.xmat[box].reshape(3, 3).copy()
        stats["max_tilt"] = max(stats["max_tilt"], float(np.degrees(np.arccos(np.clip(R_b[2, 2], -1, 1)))))
        box_z_trace.append(float(d.xpos[box][2]))
        if t >= t_lift0 - 1e-9:
            if ref_box is None:                            # reference = box pose in each gripper frame at lift start
                for s in C.SIDES:
                    Rg = d.xmat[gb[s]].reshape(3, 3)
                    ref_rel[s] = (Rg.T @ (d.xpos[box] - d.xpos[gb[s]]), Rg.T @ R_b)
                ref_box = d.xpos[box].copy()
            for s in C.SIDES:
                Rg = d.xmat[gb[s]].reshape(3, 3)
                rp, rR = ref_rel[s]
                dp = float(np.linalg.norm(Rg.T @ (d.xpos[box] - d.xpos[gb[s]]) - rp))
                da = quat_angle_deg(rR, Rg.T @ R_b)
                slip_ts[s].append((dp, da))
                stats["max_slip_m"][s] = max(stats["max_slip_m"][s], dp)
                stats["max_slip_deg"][s] = max(stats["max_slip_deg"][s], da)
    idx_hold = int(round(t_lift0 / DT))
    box_end = d.xpos[box].copy()
    c = np.array(ts_c[5:])
    fa = {s: np.array(forces[s]) for s in C.SIDES}
    lift_ok = (box_end[2] - p.box_centre[2])
    res = dict(
        params={k: (list(v) if isinstance(v, tuple) else v) for k, v in S.dataclasses.asdict(p).items()},
        payload_feedforward=payload_ff,
        mode="WELD_DEBUG_KINEMATIC_BASELINE" if weld else "real_friction_contact",
        dt=DT, timeline=dict(settle=T_SETTLE, close=T_CLOSE, settle2=T_SETTLE2, lift=T_LIFT, hold=T_HOLD, lift_m=LIFT),
        controller=dict(name="mpc_fl.FLMPC NV=7" + (" + payload hooks (M/bias extras, share 0.5 per arm)" if payload_ff == "internal" else " (unmodified)"), fl_kw=dict(__import__("world_model.v2.nero.hold_test", fromlist=["x"]).FL_KW),
                        tau_preset=tau_preset or C.TORQUE_PRESET, tau_limits=tau.tolist(),
                        gripper_kp=float(m.actuator_gainprm[act_g['left'], 0]), gripper_force_limit_N=10.0, finger_ctrl_closed=0.0),
        pregrasp=dict(ik_residual=[pre["el"], pre["er"]], min_proxy_h=pre["h"], jaw_sign=list(pre["jaw_sign"]),
                      joint_margin_rad=min(K.PoseIK(m, "left").margin(pre["ql"])[0], K.PoseIK(m, "right").margin(pre["qr"])[0]),
                      lift_ik_worst_residual=[lifted["left"][1], lifted["right"][1]]),
        pad_box_contacts={s: dict(mean_after_close=float(np.mean(ncon_hist[s][int(t_close1 / DT):])),
                                  min_after_close=int(np.min(ncon_hist[s][int(t_close1 / DT):])),
                                  max=int(np.max(ncon_hist[s]))) for s in C.SIDES},
        grip_normal_force_N={s: dict(mean_hold=float(fa[s][int(t_lift1 / DT):].mean()), max=float(fa[s].max()),
                                     mean_after_close=float(fa[s][int(t_close1 / DT):int(t_lift0 / DT)].mean())) for s in C.SIDES},
        slip_since_lift_start={s: dict(max_m=stats["max_slip_m"][s], max_deg=stats["max_slip_deg"][s],
                                       final_m=slip_ts[s][-1][0], final_deg=slip_ts[s][-1][1],
                                       creep_last_1s_mm=1e3 * (slip_ts[s][-1][0] - slip_ts[s][-200][0])) for s in C.SIDES},
        pad_normal_force_per_finger_hold_N={s: {k: float(np.mean(v[int(t_lift1 / DT):])) for k, v in fing_f[s].items()} for s in C.SIDES},
        max_tilt_deg=stats["max_tilt"],
        box_lift_achieved_m=float(lift_ok), box_final_xyz=box_end.tolist(),
        box_dropped=bool(box_end[2] < p.box_centre[2] - 0.03),
        n_infeasible=int(ctl.n_infeasible), n_fallback=int(ctl.n_fallback), n_steps=n,
        ctrl_ms=dict(mean=float(c.mean()), p95=float(np.percentile(c, 95)), max=float(c.max())),
    )
    # HELD criterion (explicit): not dropped, lifted >= 4 cm, final slip < 15 mm and < 6 deg wrt BOTH grippers.  This is a
    # smoke criterion, not a safety claim; the raw slip numbers are the result.
    res["held"] = bool((not res["box_dropped"]) and res["box_lift_achieved_m"] >= 0.8 * LIFT and
                       all(s["final_m"] < 0.015 and s["final_deg"] < 6.0 for s in res["slip_since_lift_start"].values()))
    return res


def _weld_box(m, d):
    """DEBUG ONLY: weld box to both gripper_base bodies via two runtime-added equality constraints is not possible without a
    model rebuild; instead teleport-free fallback: raise (labelled baseline unsupported at runtime)."""
    raise NotImplementedError("--weld baseline requires a model with weld equalities; use nero.hold_test for the welded tray baseline")


def main(argv=None):
    ap = S.add_args(argparse.ArgumentParser())
    ap.add_argument("--weld", action="store_true", help="DEBUG kinematic baseline (labelled; currently unsupported)")
    ap.add_argument("--tau-preset", default=None)
    ap.add_argument("--finger-kp", type=float, default=None, help="DIAGNOSTIC override of the ASSUMED finger servo kp (default: variant's 200)")
    ap.add_argument("--payload-ff", nargs="?", const="external", default="off", choices=["off", "external", "internal"],
                    help="external = static half-box-weight torque added after FL-MPC (old --payload-ff); internal = FL-MPC payload hooks")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "democ_grasp_smoke.json"))
    a = ap.parse_args(argv)
    res = run(S.params_from_args(a), weld=a.weld, tau_preset=a.tau_preset, payload_ff=a.payload_ff, finger_kp=a.finger_kp)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("mode", "held", "pad_box_contacts", "grip_normal_force_N", "slip_since_lift_start",
                                          "max_tilt_deg", "box_lift_achieved_m", "n_infeasible", "n_fallback", "ctrl_ms")}, indent=1, default=float))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
