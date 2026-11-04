"""democ/grasp_cert_xcheck.py -- sim cross-check of grasp_cert verdicts vs what MuJoCo does.  SIMULATION ONLY, CPU.

Hold: KINEMATIC ARMS (arm joint qpos reset to the pre-grasp IK solution + qvel=0 after every step: an infinitely stiff arm; no weld,
no mocap, NOTHING attached to the box).  Fingers are the model's real position actuators (friction contact with the box), gain
raised to kp=2000 so the 0.01 m closing deflection saturates the actuator at its forcerange (10 N) -- the hardware limit the
certificate assumes.  Timeline: 0.3 s open settle (box on pedestal) / 0.6 s close / [A: pedestal collision removed] 0.4 s both-hand hold /
release one hand (finger ctrl -> open) and watch 2.0 s.  Slip = pose of the box in the REMAINING gripper_base frame vs at release;
"stays" <=> max slip < 5 mm and < 3 deg over the 2 s.  mu is applied to box, finger and pedestal geoms (MuJoCo would otherwise use
the max with the 1.0 finger / 0.8 pedestal defaults).
Scenarios: A = one hand alone (pedestal collision disabled after the two-hand hold), B = one hand + box resting on the pedestal.
    python -m world_model.v2.nero.democ.grasp_cert_xcheck [--out ...]
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
from world_model.v2.nero.democ import scene as S
from world_model.v2.nero.democ import kin as K
from world_model.v2.nero.archive.democ import grasp_cert as GC

DT = 0.005
T_SETTLE, T_CLOSE, T_HOLD2, T_WATCH = 0.3, 0.6, 0.4, 2.0
KP_SAT = 2000.0


def rot_deg(R1, R2):
    return float(np.degrees(np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1))))


def compose_flat_pads(p):
    """S.compose with each finger's collision MESH replaced (for contact with anything) by a flat box pad on the inner face with the
    certificate's footprint: x +-PAD_HALF_W, z_g 0.0615..0.138 (finger-body y -0.0765..0), 4 mm thick inside the finger body (inner face =
    finger-body z=0).  MuJoCo gives a flat box a 4-point face contact (patch), whereas the finger mesh yields ~1 point (see header)."""
    spec = S.build_spec(p)
    for g in spec.worldbody.geoms:
        if g.name in ("democ_table_top", "democ_table_pedestal"):
            g.conaffinity |= A.ARM_BITS
    for b in spec.bodies:
        if b.name == "democ_box":
            for g in b.geoms:
                g.conaffinity |= A.ARM_BITS
    for b in spec.bodies:
        if b.name.endswith(("_finger1", "_finger2")):
            for g in b.geoms:
                if g.name.endswith("_col"):
                    cb, ca = int(g.contype), int(g.conaffinity)
                    g.contype = 0; g.conaffinity = 0
            hl = 0.5 * (GC.FINGER_Z_TIP - GC.FINGER_Z_ROOT)
            b.add_geom(name=b.name + "_pad", type=mujoco.mjtGeom.mjGEOM_BOX, size=[GC.PAD_HALF_W, hl, 0.002], pos=[0, -hl, 0.002],
                       contype=cb, conaffinity=ca, condim=3, priority=1, friction=[1.0, 0.005, 0.0001], solref=[0.005, 1.0], mass=1e-9)
    m = spec.compile()
    m.opt.timestep = DT
    return m


def build(mu, comy, pre_cache, pads="flat", mass=None, comx=0.0):
    kw = {} if mass is None else dict(box_mass=float(mass))
    p = S.DemoCParams(box_mu=mu, box_com_offset=(float(comx), comy, 0.0), **kw)
    m = compose_flat_pads(p) if pads == "flat" else S.compose(p, dt=DT)
    for g in range(m.ngeom):
        n = m.geom(g).name
        if n in ("democ_box_geom", "democ_table_pedestal", "democ_table_top") or "finger" in n and (n.endswith("_col") or n.endswith("_pad")):
            m.geom_friction[g, 0] = mu
    for s in C.SIDES:
        a = m.actuator(f"{s}_arm_gripper_pos").id
        m.actuator_gainprm[a, 0] = KP_SAT; m.actuator_biasprm[a, 1] = -KP_SAT
    A.disable_arm_servos(m)
    if "pre" not in pre_cache:
        pre_cache["pre"] = K.find_pregrasp(m, S.DemoCParams())
    return p, m, pre_cache["pre"]


def run_case(mu, comy, scenario, keep, pre_cache, m_true=0.6, pads="flat"):
    p, m, pre = build(mu, comy, pre_cache, pads)
    rel = "right" if keep == "left" else "left"
    d = mujoco.MjData(m)
    K.set_state(m, d, p, pre["ql"], pre["qr"], K.FINGER_OPEN)
    qa = {s: A.arm_addrs(m, s) for s in C.SIDES}
    hold = {"left": pre["ql"].copy(), "right": pre["qr"].copy()}
    act = {s: m.actuator(f"{s}_arm_gripper_pos").id for s in C.SIDES}
    box = m.body("democ_box").id
    gb = {s: m.body(f"{s}_arm_gripper_base").id for s in C.SIDES}
    ped = m.geom("democ_table_pedestal").id
    ped_bits = (int(m.geom_contype[ped]), int(m.geom_conaffinity[ped]))
    sfx = "pad" if pads == "flat" else "col"
    fing = {s: {m.geom(f"{s}_arm_finger1_{sfx}").id, m.geom(f"{s}_arm_finger2_{sfx}").id} for s in C.SIDES}
    bgeom = m.geom("democ_box_geom").id
    t_close0, t_close1 = T_SETTLE, T_SETTLE + T_CLOSE
    t_unped = t_close1 + (0.1 if scenario == "A" else 1e9)
    t_rel = t_close1 + T_HOLD2
    n = int(round((t_rel + T_WATCH) / DT))
    ref = None; hold_ref = None
    slip = dict(mm=0.0, deg=0.0); slip2 = dict(mm=0.0, deg=0.0)
    cert = None; sq = dict(sum_max={"left": 0.0, "right": 0.0}, act_max={"left": 0.0, "right": 0.0})
    z_trace = []
    slip_ts = []
    ped_off = False
    for k in range(n):
        t = k * DT
        for s in C.SIDES:
            closed = t >= t_close0 and not (s == rel and t >= t_rel)
            d.ctrl[act[s]] = 0.0 if closed else K.FINGER_OPEN
        if scenario == "A" and not ped_off and t >= t_unped:
            m.geom_contype[ped] = 0; m.geom_conaffinity[ped] = 0; ped_off = True
        if cert is None and t >= t_rel:
            mujoco.mj_forward(m, d)
            which = ("left", "right") if scenario == "A" else ("left", "right", "support")
            sc = GC.contacts_from_scene(m, d, which=which, mu_lo=mu)
            P = GC.Params(m_lo=m_true, m_hi=m_true, com_lo=(0, comy, 0), com_hi=(0, comy, 0))
            ok, eps, det = GC.release_ok(sc, rel, P)
            sc2 = GC.contacts_from_scene(m, d, which=which, mu_lo=mu, limit_mode="per_pad")
            ok2, eps2, _ = GC.release_ok(sc2, rel, P, compute_margin=False)
            shr = {}
            for sh in (0.7, 0.5):
                scs = GC.contacts_from_scene(m, d, which=which, mu_lo=mu, shrink=sh)
                oks, epss, _ = GC.release_ok(scs, rel, P)
                shr[str(sh)] = dict(ok=bool(oks), eps=float(epss))
            cert = dict(shrunk_patch_heuristic=shr, ok=bool(ok), eps=float(eps), ok_per_pad_limit=bool(ok2), optimistic=bool(det["optimistic_internal_force"]),
                        n_lp=det["n_lp"], elapsed_s=det["elapsed_s"], n_contacts=det["n_contacts"],
                        worst=[det["worst_vertex"], det["worst_direction"]])
            Rg = d.xmat[gb[keep]].reshape(3, 3)
            ref = (Rg.T @ (d.xpos[box] - d.xpos[gb[keep]]), Rg.T @ d.xmat[box].reshape(3, 3))
        mujoco.mj_step(m, d)
        for s in C.SIDES:                                           # kinematic arms
            d.qpos[qa[s][0]] = hold[s]; d.qvel[qa[s][1]] = 0.0
        mujoco.mj_forward(m, d)
        # squeeze bookkeeping (normal force summed over both fingers of a hand)
        fv = np.zeros(6); nsum = {s: 0.0 for s in C.SIDES}
        for j in range(d.ncon):
            c = d.contact[j]
            if bgeom not in (c.geom1, c.geom2):
                continue
            o = c.geom2 if c.geom1 == bgeom else c.geom1
            for s in C.SIDES:
                if o in fing[s]:
                    mujoco.mj_contactForce(m, d, j, fv); nsum[s] += float(fv[0])
        if t_close1 + 0.05 <= t < t_rel:
            for s in C.SIDES:
                sq["sum_max"][s] = max(sq["sum_max"][s], nsum[s]); sq["act_max"][s] = max(sq["act_max"][s], abs(float(d.actuator_force[act[s]])))
        Rk = d.xmat[gb[keep]].reshape(3, 3); Rb = d.xmat[box].reshape(3, 3)
        cur = (Rk.T @ (d.xpos[box] - d.xpos[gb[keep]]), Rk.T @ Rb)
        if t_close1 + 0.1 <= t < t_rel:
            if hold_ref is None:
                hold_ref = cur
            slip2["mm"] = max(slip2["mm"], 1e3 * float(np.linalg.norm(cur[0] - hold_ref[0]))); slip2["deg"] = max(slip2["deg"], rot_deg(hold_ref[1], cur[1]))
        if ref is not None and t >= t_rel:
            slip["mm"] = max(slip["mm"], 1e3 * float(np.linalg.norm(cur[0] - ref[0]))); slip["deg"] = max(slip["deg"], rot_deg(ref[1], cur[1]))
            z_trace.append(float(d.xpos[box][2]))
            slip_ts.append((1e3 * float(np.linalg.norm(cur[0] - ref[0])), rot_deg(ref[1], cur[1])))
    stays = bool(slip["mm"] < 5.0 and slip["deg"] < 3.0)
    return dict(pads=pads, mu=mu, com_y=comy, scenario=scenario, keep=keep, released=rel, cert=cert, sim=dict(
        slip_mm_at_1s=slip_ts[int(1.0 / DT)][0], slip_mm_final=slip_ts[-1][0], creep_last_0p5s_mm=slip_ts[-1][0] - slip_ts[-int(0.5 / DT)][0],
        slip_deg_final=slip_ts[-1][1], max_slip_mm=slip["mm"], max_slip_deg=slip["deg"], stays=stays, two_hand_phase_slip_mm=slip2["mm"], two_hand_phase_slip_deg=slip2["deg"],
        box_z_final=z_trace[-1] if z_trace else None, box_z_drop_mm=1e3 * (z_trace[0] - z_trace[-1]) if z_trace else None),
        squeeze=sq, verdict=("TP" if cert["ok"] and stays else "TN" if (not cert["ok"] and not stays) else
                             "FALSE_SAFE" if cert["ok"] else "false_unsafe"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "democ_grasp_cert_xcheck_2026-10-26.json"))
    ap.add_argument("--mus", type=float, nargs="+", default=[0.2, 0.4, 0.8])
    ap.add_argument("--coms", type=float, nargs="+", default=[0.0, 0.05, 0.10])
    ap.add_argument("--pads", nargs="+", default=["flat", "mesh"], help="flat = certificate-footprint box pads; mesh = the model's finger mesh")
    ap.add_argument("--keeps", nargs="+", default=["left", "right"])
    a = ap.parse_args(argv)
    cache = {}
    rows = []
    for pads, sc in [(pd, sc_) for pd in a.pads for sc_ in ("A", "B")]:
        for keep in a.keeps:
            for mu in a.mus:
                for cy in a.coms:
                    t0 = time.perf_counter()
                    r = run_case(mu, cy, sc, keep, cache, pads=pads)
                    r["wall_s"] = time.perf_counter() - t0
                    rows.append(r)
                    print(f"[{pads}] {sc} keep={keep} mu={mu} comy={cy:+.2f}: cert={r['cert']['ok']} (eps {r['cert']['eps']:.3f}) sim stays={r['sim']['stays']} "
                          f"slip {r['sim']['max_slip_mm']:.1f} mm {r['sim']['max_slip_deg']:.1f} deg -> {r['verdict']}", flush=True)
    for r in rows:
        for sh, v in r["cert"]["shrunk_patch_heuristic"].items():
            r.setdefault("verdict_shrink", {})[sh] = ("TP" if v["ok"] and r["sim"]["stays"] else "TN" if (not v["ok"] and not r["sim"]["stays"]) else
                                                       "FALSE_SAFE" if v["ok"] else "false_unsafe")
    summ_shr = {pd: {sh: {v: sum(r["verdict_shrink"][sh] == v and r["pads"] == pd for r in rows) for v in ("TP", "TN", "FALSE_SAFE", "false_unsafe")}
                     for sh in ("0.7", "0.5")} for pd in a.pads}
    for r in rows:
        okm = r["cert"]["ok"] and r["cert"]["eps"] >= 1.0
        r["verdict_min_margin_1"] = ("TP" if okm and r["sim"]["stays"] else "TN" if (not okm and not r["sim"]["stays"]) else "FALSE_SAFE" if okm else "false_unsafe")
    summ_mm = {pd: {v: sum(r["verdict_min_margin_1"] == v and r["pads"] == pd for r in rows) for v in ("TP", "TN", "FALSE_SAFE", "false_unsafe")} for pd in a.pads}
    summ = {pd: {v: sum(r["verdict"] == v and r["pads"] == pd for r in rows) for v in ("TP", "TN", "FALSE_SAFE", "false_unsafe")} for pd in a.pads}
    out = dict(description=__doc__, kp_sat=KP_SAT, m_true=0.6, dt=DT, timeline=dict(settle=T_SETTLE, close=T_CLOSE, hold2=T_HOLD2, watch=T_WATCH),
               summary=summ, summary_patch_shrink_heuristic=summ_shr, summary_min_margin_1_0=summ_mm, rows=rows)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(summ, "min_margin1.0:", summ_mm, "wrote", a.out)


if __name__ == "__main__":
    main()
