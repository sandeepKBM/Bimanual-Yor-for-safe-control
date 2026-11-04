"""democ/grasp_cert_compliant_xcheck.py -- sim cross-check of grasp_cert mode='exists' vs mode='compliant'.  SIMULATION ONLY, CPU.

Same physical setup as grasp_cert_xcheck (kinematic arms = infinitely stiff, flat certificate-footprint pads, 0.3 s settle / 0.6 s close /
[A: pedestal removed] 0.4 s two-hand hold / release one hand / watch 2 s), but the grip is a constant closing FORCE F_g per hand
(gain 0, bias0 = -F_g: the hold_profile.HoldFix 'grip_force_N' mode) so that F_g is a swept parameter; fmode='kp' reproduces the old
servo-saturated-at-10-N grip (the 36 original cases).
FAILURE DEFINITION (fixed before running):
  box stays  <=>  world-frame box COM displacement from its release pose < 5 mm AND box tilt (angle to its release orientation) < 3 deg
                  at every instant of the 2 s AND end speed |x(T)-x(T-0.2 s)|/0.2 s < 1 mm/s.
  The old metric (hand-relative slip < 5 mm and < 3 deg) is also recorded ('stays_old').
Verdicts per mode: TP (cert ok & stays), TN, FALSE_SAFE (cert ok & not stays), false_unsafe.
    python -m world_model.v2.nero.democ.grasp_cert_compliant_xcheck --set old36|ext180 [--procs 4] [--out ...]
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import mujoco

from world_model.v2.nero import config as C
from world_model.v2.nero import adapters as A
from world_model.v2.nero.democ import kin as K
from world_model.v2.nero.archive.democ import grasp_cert as GC
from world_model.v2.nero.archive.democ import grasp_cert_xcheck as X

DT = X.DT
T_SETTLE, T_CLOSE, T_HOLD2, T_WATCH = X.T_SETTLE, X.T_CLOSE, X.T_HOLD2, X.T_WATCH
RESULTS = Path(__file__).resolve().parents[1] / "results"
M_TRUE = 0.6
RHOS, KAPPAS = (0.5, 1.0, 2.0), (0.5, 1.0, 2.0)
# Truth-sim fix (2026-10-26, opt-in): the arms have servos disabled and are reset to their hold pose after every step, but inside a step
# they accelerate (gravity + contact, |qacc| up to ~28 rad/s^2), which leaks into every pad contact (a1 = aref - R f != 0) and makes the
# pads look 30x softer/stiffer than they are.  ARM_ARMATURE > 0 adds that armature [kg m^2] to the 14 arm DoFs so they are truly
# kinematic (|qacc| < 1e-3).  0 = the original harness (reproduces all saved 10-26 results).  Set from --arm-armature before the Pool forks.
ARM_ARMATURE = 0.0


def verdict(ok, stays):
    return "TP" if ok and stays else "TN" if (not ok and not stays) else "FALSE_SAFE" if ok else "false_unsafe"


def cert_variants(sc, rel, P, Fg, stiff0, gap_tol=GC.GAP_TOL):
    """-> dict of certificate outputs for the scene at release.  exists + compliant (default stiffness, tangential spring/creep/both) + rho x kappa grid."""
    out = {}
    t0 = time.perf_counter()
    ok, eps, det = GC.release_ok(sc, rel, P, mode="exists")
    out["exists"] = dict(ok=bool(ok), eps=float(eps), elapsed_s=time.perf_counter() - t0, n_lp=det["n_lp"], optimistic=bool(det["optimistic_internal_force"]))
    def one(st):
        t0 = time.perf_counter()
        ok, eps, det = GC.release_ok(sc, rel, P, mode="compliant", stiffness=st, gap_tol=gap_tol)
        cm = det.get("compliant", {})
        return dict(ok=bool(ok), eps=float(eps), elapsed_s=time.perf_counter() - t0, exists_ok=det.get("exists_ok"),
                    worst=det.get("worst_direction"), notes=det.get("notes"),
                    margins={k: {kk: v.get(kk) for kk in ("normal_margin_N", "friction_margin", "disp_margin", "min_fn_N", "max_ft_over_mu_fn", "max_rot_deg",
                                                          "max_trans_mm", "max_creep_surface_speed_mm_s", "singular", "worst_condition")} for k, v in cm.items()})
    out["compliant"] = one(stiff0)
    for tm in ("spring", "creep", "both"):
        out[f"compliant_{tm}"] = one(GC.Stiffness(**{**stiff0.__dict__, "tangential": tm}))
    grid = {}
    for gt_name, gt in (("gap", gap_tol), ("gap0", 0.0)):
        for rho, kap in itertools.product(RHOS, KAPPAS):
            st = GC.Stiffness(**{**stiff0.__dict__, "rho": rho, "kappa": kap, "tangential": "spring"})
            ok, eps, _ = GC.release_ok(sc, rel, P, mode="compliant", stiffness=st, gap_tol=gt)
            grid[f"spring_{gt_name}_rho{rho}_kappa{kap}"] = dict(ok=bool(ok), eps=float(eps))
        for cr, kap in itertools.product((0.5, 1.0, 2.0), KAPPAS):             # creep: rho is irrelevant (disjoint DoF sets); support/pad damping ratio matters
            st = GC.Stiffness(**{**stiff0.__dict__, "ct_s": cr * stiff0.ct, "kappa": kap, "tangential": "creep"})
            ok, eps, _ = GC.release_ok(sc, rel, P, mode="compliant", stiffness=st, gap_tol=gt)
            grid[f"creep_{gt_name}_cts{cr}_kappa{kap}"] = dict(ok=bool(ok), eps=float(eps))
    out["grid"] = grid
    return out


def scene_to_json(sc):
    return dict(g_box=list(map(float, sc.g_box)), contacts=[dict(p=c.p.tolist(), n=c.n.tolist(), mu_lo=c.mu_lo, mu_hi=c.mu_hi, f_max=c.f_max, group=c.group,
                                                                 hand=c.hand, active=c.active, pose_unc=c.pose_unc, finger=c.finger) for c in sc.contacts])


def scene_from_json(d):
    cs = [GC.Contact(np.array(c["p"]), np.array(c["n"]), c["mu_lo"], c["mu_hi"], c["f_max"], c["group"], c["hand"], c["active"], c["pose_unc"], finger=c.get("finger")) for c in d["contacts"]]
    return GC.SceneContacts(cs, np.array(d["g_box"]), np.eye(3), np.zeros(3), np.zeros(3))


def add_verdicts(row):
    stays, stays_old = row["sim"]["stays"], row["sim"]["stays_old"]
    cert = row["cert"]
    row["verdict"] = {}; row["verdict_old_metric"] = {}; row["verdict_grid"] = {}
    for key in ("exists", "compliant", "compliant_spring", "compliant_creep", "compliant_both"):
        row["verdict"][key] = verdict(cert[key]["ok"], stays)
        row["verdict_old_metric"][key] = verdict(cert[key]["ok"], stays_old)
    for g, v in cert["grid"].items():
        row["verdict_grid"][g] = verdict(v["ok"], stays)
    return row


def recompute(rows, stiff0, gap_tol):
    """re-evaluate the certificates on the SAVED release scenes (no simulation) with other constants."""
    out = []
    for r in rows:
        ms, cx = r.get("mass", M_TRUE), r.get("com_x", 0.0)
        sc = scene_from_json(r["scene"]); P = GC.Params(m_lo=ms, m_hi=ms, com_lo=(cx, r["com_y"], 0), com_hi=(cx, r["com_y"], 0))
        rel = "right" if r["keep"] == "left" else "left"
        r2 = dict(r); r2["cert"] = cert_variants(sc, rel, P, r["Fg"], stiff0, gap_tol)
        out.append(add_verdicts(r2))
    return out


def run_case(mu, comy, scenario, keep, Fg, fmode, cache, stiff0=None, gap_tol=GC.GAP_TOL, mass=M_TRUE, comx=0.0):
    stiff0 = stiff0 or GC.Stiffness()
    p, m, pre = X.build(mu, comy, cache, "flat", mass=None if mass == M_TRUE else mass, comx=comx)
    if ARM_ARMATURE:
        for s in C.SIDES:
            m.dof_armature[np.atleast_1d(A.arm_addrs(m, s)[1])] += ARM_ARMATURE
    rel = "right" if keep == "left" else "left"
    d = mujoco.MjData(m)
    K.set_state(m, d, p, pre["ql"], pre["qr"], K.FINGER_OPEN)
    qa = {s: A.arm_addrs(m, s) for s in C.SIDES}
    hold = {"left": pre["ql"].copy(), "right": pre["qr"].copy()}
    act = {s: m.actuator(f"{s}_arm_gripper_pos").id for s in C.SIDES}
    gain0 = {s: (m.actuator_gainprm[act[s]].copy(), m.actuator_biasprm[act[s]].copy()) for s in C.SIDES}
    if fmode == "force":
        for s in C.SIDES:
            m.actuator_gainprm[act[s], :] = 0.0; m.actuator_biasprm[act[s], :] = 0.0; m.actuator_biasprm[act[s], 0] = -Fg
    box = m.body("democ_box").id
    gb = {s: m.body(f"{s}_arm_gripper_base").id for s in C.SIDES}
    ped = m.geom("democ_table_pedestal").id
    bgeom = m.geom("democ_box_geom").id
    t_close0, t_close1 = T_SETTLE, T_SETTLE + T_CLOSE
    t_unped = t_close1 + (0.1 if scenario == "A" else 1e9)
    t_rel = t_close1 + T_HOLD2
    n = int(round((t_rel + T_WATCH) / DT))
    cert = None; ref = None; ped_off = False; restored = False
    xs, Rs, rel_slip = [], [], []
    sim_forces = None
    for k in range(n):
        t = k * DT
        for s in C.SIDES:
            closed = t >= t_close0 and not (s == rel and t >= t_rel)
            d.ctrl[act[s]] = 0.0 if closed else K.FINGER_OPEN
        if fmode == "force" and t >= t_rel and not restored:
            m.actuator_gainprm[act[rel]] = gain0[rel][0]; m.actuator_biasprm[act[rel]] = gain0[rel][1]; restored = True
        if scenario == "A" and not ped_off and t >= t_unped:
            m.geom_contype[ped] = 0; m.geom_conaffinity[ped] = 0; ped_off = True
        if cert is None and t >= t_rel:
            mujoco.mj_forward(m, d)
            which = ("left", "right") if scenario == "A" else ("left", "right", "support")
            P = GC.Params(m_lo=mass, m_hi=mass, com_lo=(comx, comy, 0), com_hi=(comx, comy, 0))
            try:
                sc = GC.contacts_from_scene(m, d, which=which, mu_lo=mu, f_limit=Fg)
            except ValueError as e:          # the certificate cannot model this scene (e.g. box >15 deg off the jaw axes) -> it must REJECT
                sc = None
                rej = dict(ok=False, eps=float("-inf"), elapsed_s=0.0, notes=[f"scene_not_modelable: {e}"])
                cert = {k: dict(rej) for k in ("exists", "compliant", "compliant_spring", "compliant_creep", "compliant_both")}
                cert["grid"] = {}
                scene_json = None
            if sc is not None:
                cert = cert_variants(sc, rel, P, Fg, stiff0, gap_tol)
                scene_json = scene_to_json(sc)
            Rg = d.xmat[gb[keep]].reshape(3, 3)
            ref = (Rg.T @ (d.xpos[box] - d.xpos[gb[keep]]), Rg.T @ d.xmat[box].reshape(3, 3))
            x_rel, R_rel = d.xpos[box].copy(), d.xmat[box].reshape(3, 3).copy()
        mujoco.mj_step(m, d)
        for s in C.SIDES:
            d.qpos[qa[s][0]] = hold[s]; d.qvel[qa[s][1]] = 0.0
        mujoco.mj_forward(m, d)
        if ref is not None and t >= t_rel:
            Rk = d.xmat[gb[keep]].reshape(3, 3)
            cur = (Rk.T @ (d.xpos[box] - d.xpos[gb[keep]]), Rk.T @ d.xmat[box].reshape(3, 3))
            rel_slip.append((1e3 * float(np.linalg.norm(cur[0] - ref[0])), X.rot_deg(ref[1], cur[1])))
            xs.append(d.xpos[box].copy()); Rs.append(d.xmat[box].reshape(3, 3).copy())
            if sim_forces is None and t >= t_rel + 0.3:
                sim_forces = []
                fv = np.zeros(6); Rb = d.xmat[box].reshape(3, 3)
                for j in range(d.ncon):
                    c = d.contact[j]
                    if bgeom in (c.geom1, c.geom2):
                        o = c.geom2 if c.geom1 == bgeom else c.geom1
                        mujoco.mj_contactForce(m, d, j, fv)
                        fr = c.frame.reshape(3, 3); sg = -1.0 if c.geom1 == bgeom else 1.0
                        sim_forces.append(dict(geom=m.geom(o).name, dist=float(c.dist), f_box=(Rb.T @ (sg * (fr.T @ fv[:3]))).round(4).tolist(),
                                               pos=(Rb.T @ (c.pos - d.xpos[box])).round(4).tolist()))
    xs = np.array(xs)
    disp = 1e3 * np.linalg.norm(xs - x_rel, axis=1)
    tilt = np.array([X.rot_deg(R_rel, R) for R in Rs])
    nl = int(round(0.2 / DT))
    end_speed = 1e3 * float(np.linalg.norm(xs[-1] - xs[-1 - nl])) / 0.2
    stays = bool(disp.max() < 5.0 and tilt.max() < 3.0 and end_speed < 1.0)
    rs = np.array(rel_slip)
    stays_old = bool(rs[:, 0].max() < 5.0 and rs[:, 1].max() < 3.0)
    sim = dict(max_disp_mm=float(disp.max()), disp_final_mm=float(disp[-1]), max_tilt_deg=float(tilt.max()), end_speed_mm_s=end_speed, z_drop_mm=float(1e3 * (x_rel[2] - xs[-1][2])),
               stays=stays, stays_old=stays_old, max_hand_slip_mm=float(rs[:, 0].max()), max_hand_slip_deg=float(rs[:, 1].max()))
    row = dict(mu=mu, com_y=comy, com_x=comx, mass=mass, arm_armature=ARM_ARMATURE, scenario=scenario, keep=keep, Fg=Fg, fmode=fmode, sim=sim, cert=cert, scene=scene_json, sim_forces_t0p3=sim_forces)
    return add_verdicts(row)


def _work(args):
    mu, cy, sc, keep, Fg, fmode, stiff, gap_tol, *extra = args          # extra = (mass, com_x) for the held-out set
    t0 = time.perf_counter()
    r = run_case(mu, cy, sc, keep, Fg, fmode, _CACHE, stiff, gap_tol, *extra)
    r["wall_s"] = time.perf_counter() - t0
    print(f"{sc} keep={keep} mu={mu} comy={cy:+.3f} comx={r['com_x']:+.3f} m={r['mass']} Fg={Fg} [{fmode}]: stays={r['sim']['stays']} (old {r['sim']['stays_old']}) disp {r['sim']['max_disp_mm']:.1f} mm "
          f"tilt {r['sim']['max_tilt_deg']:.1f} v_end {r['sim']['end_speed_mm_s']:.2f} | exists {r['verdict']['exists']} compliant {r['verdict']['compliant']}", flush=True)
    return r


_CACHE = {}


def case_list(which, stiff, gap_tol):
    if which == "old36":
        return [(mu, cy, sc, keep, 10.0, "kp", stiff, gap_tol) for sc in ("A", "B") for keep in ("left", "right") for mu in (0.2, 0.4, 0.8) for cy in (0.0, 0.05, 0.10)]
    if which == "ext180":
        return [(mu, cy, sc, keep, Fg, "force", stiff, gap_tol) for sc in ("A", "B") for keep in ("left", "right") for mu in (0.2, 0.4, 0.8)
                for cy in (0.0, 0.025, 0.05, 0.075, 0.10) for Fg in (6.0, 9.0, 10.0)]
    if which == "heldout256":
        # PRE-REGISTERED 2026-10-26 before any run: every value is outside the old36/ext180 training grid (mu {0.2,0.4,0.8}, com_y
        # {0,...,0.10} step 0.025, Fg {6,9,10}, mass 0.6, com_x 0).  Certificate constants frozen at GC.Stiffness() defaults and
        # GC.GAP_TOL (6e-6 m); scored once.  New axes: box mass and an off-axis COM (com_x).
        return [(mu, cy, sc, keep, Fg, "force", stiff, gap_tol, ms, cx) for sc in ("A", "B") for keep in ("left", "right") for mu in (0.3, 0.6)
                for cy in (0.0125, 0.0375, 0.0625, 0.0875) for cx in (0.0, 0.03) for ms in (0.45, 0.8) for Fg in (7.0, 8.5)]
    if which == "heldout_v2":
        # PRE-REGISTERED 2026-10-26 ~12:30, before any truth-fixed (--arm-armature) result was looked at.  A fresh set: heldout256 was
        # seen under the artifact harness.  Values are off both the training grid and heldout256 (mu 0.25/0.7, com_y 0.02..0.095,
        # com_x -0.02/0.045, mass 0.35/0.7, Fg 6.5/9.5).  Scored once with the config chosen by the rule in the _PREREG file.
        return [(mu, cy, sc, keep, Fg, "force", stiff, gap_tol, ms, cx) for sc in ("A", "B") for keep in ("left", "right") for mu in (0.25, 0.7)
                for cy in (0.02, 0.045, 0.07, 0.095) for cx in (-0.02, 0.045) for ms in (0.35, 0.7) for Fg in (6.5, 9.5)]
    raise ValueError(which)


def summarize(rows):
    keys = ("exists", "compliant", "compliant_spring", "compliant_creep", "compliant_both")
    def tab(vkey, sub):
        return {k: {v: sum(r[vkey][k] == v for r in sub) for v in ("TP", "TN", "FALSE_SAFE", "false_unsafe")} for k in keys}
    out = dict(n=len(rows), new_metric=tab("verdict", rows), old_metric=tab("verdict_old_metric", rows))
    grid = {}
    gkeys = next((r["verdict_grid"] for r in rows if r["verdict_grid"]), {})
    for g in gkeys:                      # a row without grid entries = scene not modelable = reject under every variant
        grid[g] = {v: sum(r["verdict_grid"].get(g, verdict(False, r["sim"]["stays"])) == v for r in rows) for v in ("TP", "TN", "FALSE_SAFE", "false_unsafe")}
    out["grid_rho_kappa"] = grid
    out["stays_count"] = dict(new=sum(r["sim"]["stays"] for r in rows), old=sum(r["sim"]["stays_old"] for r in rows))
    out["cert_time_s"] = dict(exists_mean=float(np.mean([r["cert"]["exists"]["elapsed_s"] for r in rows])),
                              compliant_mean=float(np.mean([r["cert"]["compliant"]["elapsed_s"] for r in rows])),
                              compliant_max=float(np.max([r["cert"]["compliant"]["elapsed_s"] for r in rows])))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="old36", choices=["old36", "ext180", "heldout256", "heldout_v2"])
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gap-tol", type=float, default=GC.GAP_TOL)
    ap.add_argument("--kn", type=float, default=None, help="override Stiffness.kn (pad normal stiffness per corner [N/m])")
    ap.add_argument("--kappa", type=float, default=None, help="override Stiffness.kappa (support/pad stiffness ratio)")
    ap.add_argument("--arm-armature", type=float, default=0.0, help="truth-sim fix: armature added to the arm DoFs (0 = original harness)")
    ap.add_argument("--recompute", default=None, help="saved rows json: re-evaluate the certificates offline (no sim) with --gap-tol")
    a = ap.parse_args(argv)
    global ARM_ARMATURE
    ARM_ARMATURE = a.arm_armature
    stiff = GC.Stiffness(**{k: v for k, v in (("kn", a.kn), ("kappa", a.kappa)) if v is not None})
    if a.recompute:
        rows = recompute(json.loads(Path(a.recompute).read_text())["rows"], stiff, a.gap_tol)
    else:
        cases = case_list(a.set, stiff, a.gap_tol)[: a.limit]
        with Pool(min(a.procs, 4), maxtasksperchild=20) as pool:
            rows = list(pool.imap(_work, cases, chunksize=1))
    out = dict(description=__doc__, set=a.set, gap_tol=a.gap_tol, arm_armature=a.arm_armature, stiffness_default=stiff.__dict__, summary=summarize(rows), rows=rows)
    path = Path(a.out) if a.out else RESULTS / f"democ_grasp_cert_compliant_{a.set}_2026-10-26.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["summary"], indent=1)); print("wrote", path)


if __name__ == "__main__":
    main()
