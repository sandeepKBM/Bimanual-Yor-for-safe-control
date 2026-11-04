"""pan/cert/contact_falsify.py -- try to break the conservative hold model (cert/contact_model.py) with the MuJoCo lift.  SIMULATION
ONLY, CPU.

Per world: the v1 skill chain (start, pinch, push, clamp) from the robot's own zed estimate, then at the lift start:
  * belief verdict  : contact_model.certify_hold over the PRIOR contents polytope x 27 pose-residual samples x g_eff corners, at
                      friction scale MU_SCALE_LO (what admission would use; the robot's knowledge only);
  * truth verdict   : the same LP at this world's true contents, true pose residual and true friction scale (model accuracy at
                      one theta, separating model conservatism from belief-set conservatism; truth is used for scoring only);
then the lift (v1 phases and the v2 MoveObject skill, both from the same snapshot) with, every step: pan slip relative to each
claw (translation of the claw's jaw point in the pan frame, and rotation), lid motion relative to the pan, pan tilt, pan COM
acceleration (50 ms central differences), and the sum of finger normal forces per claw (diagnostic for the 10 N squeeze model).

Soundness = model admits (k* >= 1) but the sim slips (thresholds applied in --report, from the raw maxima).

    python -m world_model.v2.nero.pan.cert.contact_falsify --out DIR --set full|iid|overload --procs P [--tags ...]
    python -m world_model.v2.nero.pan.cert.contact_falsify --out DIR --report
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse
import json
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

import numpy as np

PLAN = dict(push=0.07, grasp_frac=0.65, grip_force=10.0, clamp="v2")
ACC_WIN = 10           # steps (50 ms) for the acceleration differences
# report thresholds (ASSUMED; raw maxima are stored so they can be changed)
SLIP_MM = 2.0          # claw-vs-pan slip [mm] counted as a hold failure (creep budget 0.128 mm/s * 3.5 s = 0.45 mm, MEASURED sim)
SLIP_DEG = 3.0         # claw-vs-pan rotation [deg]
LID_MM = 5.0           # lid vs pan [mm] (= skills_check POST lift lid)


def _rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def _pose(d, bid):
    return d.xmat[bid].reshape(3, 3).copy(), d.xpos[bid].copy()


def _estimate_pose(ep):
    """the pan body pose the robot believes at the start: nominal pose re-aimed at the zed estimate."""
    from world_model.v2.nero.pan import observe_pose as O
    aim, Rz = O.aim_transform(ep.obs["est"], ep.p.pan_centre_xy)
    return Rz @ ep.R0, aim(ep.pos0)


def _true_delta(T_hat, T_true, anchor_w):
    """delta (dx, dy, dyaw_deg) with T_true = apply_delta(T_hat, anchor_w, delta) (contact_model convention)."""
    Rh, ph = T_hat
    Rt, pt = T_true
    Rz = Rt @ Rh.T
    yaw = float(np.degrees(np.arctan2(Rz[1, 0], Rz[0, 0])))
    cw = np.asarray(anchor_w, float)
    a = np.radians(yaw)
    Rz2 = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    t = pt - (Rz2 @ (ph - cw) + cw)
    return (float(t[0]), float(t[1]), yaw), float(np.degrees(np.arccos(np.clip(Rz[2, 2], -1, 1))))


def _true_contents(ep, pm):
    if ep.hid is None:
        return (0.0, 0.0, 0.0)
    m = float(ep.m.body_mass[ep.hid])
    Rp, pp = _pose(ep.d, ep.pan)
    c = Rp.T @ (ep.d.xpos[ep.hid] - pp) - pm.centre
    return (m, m * float(c @ pm.u), m * float(c @ pm.w))


def _mj_contacts(ep):
    """MuJoCo's own finger contacts with pan/lid: points per finger, spread, normal-force sum per claw (diagnostic only)."""
    import mujoco
    m, d = ep.m, ep.d
    tgt = {ep.pan, ep.lid}
    out, fsum = {}, {"left": 0.0, "right": 0.0}
    f6 = np.zeros(6)
    for s in ("left", "right"):
        for f in (1, 2):
            g = m.geom(f"{s}_arm_finger{f}_col").id
            pts = []
            for j in range(d.ncon):
                c = d.contact[j]
                if g in (c.geom1, c.geom2) and (m.geom_bodyid[c.geom1] in tgt or m.geom_bodyid[c.geom2] in tgt):
                    pts.append(c.pos.copy())
                    mujoco.mj_contactForce(m, d, j, f6)
                    fsum[s] += float(f6[0])
            P = np.array(pts) if pts else np.zeros((0, 3))
            out[f"{s}_f{f}"] = dict(n=len(pts), spread_mm=round(1e3 * float(np.ptp(P, axis=0).max()) if len(P) > 1 else 0.0, 2))
    return out, {k: round(v, 3) for k, v in fsum.items()}


def _lift(ep, version):
    """run the lift (v1 phases, or the v2 / v3 skill) recording the per-step maxima."""
    from world_model.v2.nero.pan import segments as S
    jp = {s: ep.m.site(f"{s}_arm_ee_grasp").id for s in ("left", "right")}
    R0p, p0p = _pose(ep.d, ep.pan)
    rel0 = {}
    for s in ("left", "right"):
        Rc, pc = _pose(ep.d, ep.gb[s])
        rel0[s] = (Rc.T @ R0p, R0p.T @ (ep.d.site_xpos[jp[s]] - p0p))      # pan rotation in the claw frame; jaw point in the pan frame
    Rc0 = {s: _pose(ep.d, ep.gb[s])[0] for s in ("left", "right")}

    def claw_rel(d):
        """right claw pose in the left claw frame: (R, jaw-point offset)."""
        RL, RR = _pose(d, ep.gb["left"])[0], _pose(d, ep.gb["right"])[0]
        return RL.T @ RR, RL.T @ (d.site_xpos[jp["right"]] - d.site_xpos[jp["left"]])

    def lid_in_pan(d):
        Rp_, pp_ = _pose(d, ep.pan)
        Rl, pl = _pose(d, ep.lid)
        return Rp_.T @ Rl, Rp_.T @ (pl - pp_)

    cr0, li0 = claw_rel(ep.d), lid_in_pan(ep.d)
    mx = dict(slip_mm={"left": 0.0, "right": 0.0}, slip_deg={"left": 0.0, "right": 0.0}, lid_mm=0.0, tilt_deg=0.0,
              claw_tilt_deg={"left": 0.0, "right": 0.0}, claw_rot_deg={"left": 0.0, "right": 0.0}, pan_tilt_rel_start_deg=0.0,
              acc=[0.0, 0.0, 0.0], fsum_max={"left": 0.0, "right": 0.0}, fsum_min={"left": 1e9, "right": 1e9},
              # 11-01: lid scored on its motion relative to the pan DURING the lift only; the clamp's earlier shove is lid_pre_mm
              lid_pre_mm=round(float(ep.pan_state()["lid_mm"]), 3), lid_lift_mm=0.0, lid_lift_deg=0.0,
              claw_rel_mm=0.0, claw_rel_deg=0.0,                            # claw-to-claw drift since the lift start (CLAW_REL)
              tau_abs_max={"left": [0.0] * 7, "right": [0.0] * 7})          # 11-01: per-joint max |commanded torque| [Nm]
    hist = []

    def hook(e, kind, i, n):
        cr, li = claw_rel(e.d), lid_in_pan(e.d)
        mx["claw_rel_mm"] = max(mx["claw_rel_mm"], 1e3 * float(np.linalg.norm(cr[1] - cr0[1])))
        mx["claw_rel_deg"] = max(mx["claw_rel_deg"], _rot_deg(cr[0] @ cr0[0].T))
        mx["lid_lift_mm"] = max(mx["lid_lift_mm"], 1e3 * float(np.linalg.norm(li[1] - li0[1])))
        mx["lid_lift_deg"] = max(mx["lid_lift_deg"], _rot_deg(li[0] @ li0[0].T))
        Rp, pp = _pose(e.d, e.pan)
        for s in ("left", "right"):
            Rc, _ = _pose(e.d, e.gb[s])
            mx["slip_mm"][s] = max(mx["slip_mm"][s], 1e3 * float(np.linalg.norm(Rp.T @ (e.d.site_xpos[jp[s]] - pp) - rel0[s][1])))
            mx["slip_deg"][s] = max(mx["slip_deg"][s], _rot_deg((Rc.T @ Rp) @ rel0[s][0].T))
            Rr = Rc @ Rc0[s].T                                             # claw rotation since the lift start (arm), world frame
            mx["claw_rot_deg"][s] = max(mx["claw_rot_deg"][s], _rot_deg(Rr))
            mx["claw_tilt_deg"][s] = max(mx["claw_tilt_deg"][s], float(np.degrees(np.arccos(np.clip(Rr[2, 2], -1, 1)))))
        Rpr = Rp @ R0p.T
        mx["pan_tilt_rel_start_deg"] = max(mx["pan_tilt_rel_start_deg"], float(np.degrees(np.arccos(np.clip(Rpr[2, 2], -1, 1)))))
        ps = e.pan_state()
        mx["lid_mm"], mx["tilt_deg"] = max(mx["lid_mm"], ps["lid_mm"]), max(mx["tilt_deg"], ps["tilt"])
        hist.append(e.d.xipos[e.pan].copy())
        for s in ("left", "right"):
            mx["tau_abs_max"][s] = np.maximum(mx["tau_abs_max"][s], np.abs(e.d.qfrc_applied[e.dofs[s]])).tolist()
        if e.k % 10 == 0:
            _, fs = _mj_contacts(e)
            for s in fs:
                mx["fsum_max"][s] = max(mx["fsum_max"][s], fs[s])
                mx["fsum_min"][s] = min(mx["fsum_min"][s], fs[s])
        return None

    if version == "v1":
        status, _ = ep.run_phases(ep.segment_phases("lift"), branch=hook)
    else:
        from world_model.v2.nero.pan.skills import pan_skills as PS
        orig = ep.run_phases
        ep.run_phases = lambda ph, branch=None, monitor=None: orig(ph, branch=hook if branch is None else branch, monitor=monitor)
        try:
            status, _ = PS.run(ep, "lift", lift=version)
        finally:
            ep.run_phases = orig
    H = np.array(hist)
    if len(H) > 2 * ACC_WIN:
        a = (H[2 * ACC_WIN:] - 2 * H[ACC_WIN:-ACC_WIN] + H[:-2 * ACC_WIN]) / (ACC_WIN * S.DT) ** 2
        mx["acc"] = [float(x) for x in np.abs(a).max(axis=0)]
    Rp, pp = _pose(ep.d, ep.pan)
    mx["rise_mm"] = round(1e3 * float(pp[2] - p0p[2]), 2)
    mx["status"] = status
    return mx


def run_world(tag, world, lifts=("v1", "v2")):
    from world_model.v2.nero.pan import segments as S
    from world_model.v2.nero.pan.cert import belief as B
    from world_model.v2.nero.archive.pan.cert import contact_model as CM
    ep = S.Episode(world, observe="zed", obs_seed=S.world_seed(tag), **PLAN)
    out = dict(tag=tag, world=world, obs=ep.obs, assumptions=CM.assumptions(), thresholds=dict(SLIP_MM=SLIP_MM, SLIP_DEG=SLIP_DEG, LID_MM=LID_MM))
    if ep.refused:
        out["refused"] = ep.refused
        return out
    pm = CM.PanModel(ep)
    T_est = _estimate_pose(ep)
    ep.run_phases(ep.segment_phases("start"))
    for seg in S.SEGMENTS:
        if seg == "lift":
            break
        ep.sense["pan_at_push_start"] = ep.d.xpos[ep.pan].copy()
        ep.run_phases(ep.segment_phases(seg))
        if seg == "pinch":
            jaw1 = ep.d.site_xpos[ep.m.site("left_arm_ee_grasp").id].copy()
    jaw2 = ep.d.site_xpos[ep.m.site("left_arm_ee_grasp").id].copy()
    T_hat = (T_est[0], T_est[1] + (jaw2 - jaw1))                 # jaw point FK displacement, heading kept (the claw twists, the pan doesn't)
    lever = float(np.linalg.norm((jaw2 - (T_hat[1] + T_hat[0] @ pm.centre))[:2]))
    T_true = _pose(ep.d, ep.pan)
    dl_true, tilt_true = _true_delta(T_hat, T_true, jaw2)
    pi_true = _true_contents(ep, pm)
    mu_true = float(world["perturb"]["mu_scale"])
    out["lift_start"] = dict(delta_true=dl_true, tilt_deg=round(tilt_true, 3), pi_true=pi_true,
                             lever_m=round(lever, 4),
                             delta_in_box=bool(max(abs(dl_true[0]), abs(dl_true[1])) <= B.AIM_POS + CM.PUSH_SLIP_POS + lever * np.sin(np.radians(B.AIM_YAW))
                                               and abs(dl_true[2]) <= B.AIM_YAW + CM.PUSH_SLIP_YAW),
                             pi_in_prior=bool(B.Contents.from_prior().contains(np.array(pi_true))))
    out["mj_contacts"], out["mj_fsum"] = _mj_contacts(ep)
    t0 = time.time()
    bel = CM.certify_hold(ep, pm, T_hat, B.Contents.from_prior().vertices(), CM.delta_grid(lever), anchor=jaw2)
    gw = CM.g_eff_corners(CM.A_BOX, CM.TILT_WIDE_DEG)
    belw = CM.certify_hold(ep, pm, T_hat, B.Contents.from_prior().vertices(), CM.delta_grid(lever), anchor=jaw2, gs=gw)
    out["belief_wide"] = {k: belw[k] for k in ("ok", "k_star", "c_margin", "lid_t", "deltas_skipped", "n_deltas")}
    out["belief"] = {k: v for k, v in bel.items() if k != "per_delta"} | dict(
        per_delta=[dict(delta=p["delta"], k=p.get("k"), c=p.get("c_margin"), c_why=p.get("c_why"), skipped=p.get("skipped", False))
                   for p in bel["per_delta"]], wall_s=round(time.time() - t0, 1))
    Rl, pl = _pose(ep.d, ep.lid)
    lid_rel = (T_true[0].T @ Rl, T_true[0].T @ (pl - T_true[1]))          # truth: the lid where it actually is (scoring only)
    kw = dict(require_contact=False, anchor=jaw2, lid_rel=lid_rel)
    tru = CM.certify_hold(ep, pm, T_hat, [pi_true], [dl_true], mu_scale=mu_true, **kw)
    tru_lo = CM.certify_hold(ep, pm, T_hat, [pi_true], [dl_true], mu_scale=CM.MU_SCALE_LO, **kw)
    truw = CM.certify_hold(ep, pm, T_hat, [pi_true], [dl_true], mu_scale=mu_true, gs=gw, **kw)
    out["truth_wide"] = dict(ok=truw["ok"], k_star=truw["k_star"], c_margin=truw["c_margin"], c_why=truw["per_delta"][0].get("c_why"),
                             lid_t=truw["lid_t"])
    pd0 = tru["per_delta"][0]
    out["truth"] = dict(ok=tru["ok"], k_star=tru["k_star"], c_margin=tru["c_margin"], c_why=pd0.get("c_why"), lid_t=tru["lid_t"],
                        c_min_fn=pd0.get("c_min_fn"), c_creep_mm_s=pd0.get("c_creep_mm_s"), ok_mu_lo=tru_lo["ok"],
                        c_margin_mu_lo=tru_lo["c_margin"], worst=tru["worst"], contacts=pd0["info"],
                        consistent=tru["per_delta"][0]["consistent"], k_star_mu_lo=tru_lo["k_star"],
                        lid_rel_mm=[round(1e3 * float(x), 2) for x in lid_rel[1]], lid_rel_deg=round(_rot_deg(lid_rel[0]), 2))
    snap = ep.snapshot("lift_start")
    out["lift"] = {}
    for v in lifts:
        ep.restore(snap)
        ep.reset_track()
        out["lift"][v] = _lift(ep, v)
    return out


def _job(job):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    tag, w, out = job[:3]
    lifts = tuple(job[3]) if len(job) > 3 else ("v1", "v2")
    t0 = time.time()
    try:
        r = run_world(tag, w, lifts=lifts)
    except Exception:
        r = dict(tag=tag, world=w, error=traceback.format_exc()[-2000:])
    r["wall_s"] = round(time.time() - t0, 1)
    p = Path(out) / "raw" / f"{tag}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(r, default=float))
    os.replace(tmp, p)
    return tag, r.get("error") is not None


def failed(lf, slip_mm=SLIP_MM, slip_deg=SLIP_DEG, lid_mm=LID_MM):
    """lid: motion relative to the pan during the lift (lid_lift_mm, 11-01); rows from before 11-01 fall back to the absolute lid_mm."""
    why = [f"slip_{s}" for s in ("left", "right") if lf["slip_mm"][s] > slip_mm or lf["slip_deg"][s] > slip_deg]
    return why + (["lid"] if lf.get("lid_lift_mm", lf["lid_mm"]) > lid_mm else [])


def claw_rel_viol(lf):
    """the CLAW_REL assumption (ASSUMED, USER 11-01) is broken; rows from before 11-01 lack the field and count as not checked."""
    from world_model.v2.nero.archive.pan.cert import contact_model as CM
    return lf.get("claw_rel_mm", 0.0) > CM.CLAW_REL_MM or lf.get("claw_rel_deg", 0.0) > CM.CLAW_REL_DEG


def report(out):
    from world_model.v2.nero.archive.pan.cert import contact_model as CM
    rows = [json.loads(p.read_text()) for p in sorted((Path(out) / "raw").glob("*.json"))]
    lines = [f"contact-model falsification: {len(rows)} worlds  ({out})",
             f"thresholds (ASSUMED): slip {SLIP_MM} mm / {SLIP_DEG} deg per claw, lid {LID_MM} mm vs the pan DURING the lift (11-01; clamp shove = lid_pre, reported apart); model assumptions: {CM.assumptions()}", ""]
    tab = dict(n=0, refused=0, error=0)
    cnt = {}
    crv = {}                                       # lift version -> [rows with CLAW_REL broken, rows with the field]
    for r in rows:
        if r.get("error"):
            tab["error"] += 1
            continue
        if r.get("refused"):
            tab["refused"] += 1
            continue
        tab["n"] += 1
        kb, kt = r["belief"]["k_star"], r["truth"]["k_star"]
        okb, okt = r["belief"]["ok"], r["truth"]["ok"]
        for v, lf in r["lift"].items():
            fl = failed(lf)
            cv = claw_rel_viol(lf)
            c = crv.setdefault(v, [0, 0])
            c[0] += int(cv)
            c[1] += int("claw_rel_mm" in lf)
            viol = lf["tilt_deg"] > CM.TILT_MAX_DEG or max(lf["acc"]) > CM.A_BOX or not r["lift_start"]["delta_in_box"] or cv
            violw = lf["tilt_deg"] > CM.TILT_WIDE_DEG or max(lf["acc"]) > CM.A_BOX or not r["lift_start"]["delta_in_box"] or cv
            okbw, oktw = r.get("belief_wide", {}).get("ok"), r.get("truth_wide", {}).get("ok")
            for name, ok_, vv in (("belief", okbw, violw), ("truth", oktw, violw)):
                key = (v, name + "_t11", "admit" if ok_ else "reject", "fail" if fl else "hold", "assump_viol" if vv else "in_box")
                cnt[key] = cnt.get(key, 0) + 1
            for name, ok_ in (("belief", okb), ("truth", okt)):
                key = (v, name, "admit" if ok_ else "reject", "fail" if fl else "hold", "assump_viol" if viol else "in_box")
                cnt[key] = cnt.get(key, 0) + 1
            ct = r["truth"].get("c_margin")
            lines.append(f"{r['tag']:>8} {v}: belief {'ADMIT' if okb else 'reject'} truth {'ADMIT' if okt else 'reject'} (k_exist={kt if kt is None else round(kt, 2)} c={ct if ct is None else round(ct, 3)} {r['truth'].get('c_why')}) "
                         f"slip L/R {lf['slip_mm']['left']:.2f}/{lf['slip_mm']['right']:.2f} mm {lf['slip_deg']['left']:.1f}/{lf['slip_deg']['right']:.1f} deg "
                         f"lid lift {lf.get('lid_lift_mm', float('nan')):.2f} (pre {lf.get('lid_pre_mm', float('nan')):.2f}, abs {lf['lid_mm']:.2f}) "
                         f"clawrel {lf.get('claw_rel_mm', float('nan')):.2f} mm/{lf.get('claw_rel_deg', float('nan')):.1f} deg "
                         f"tilt {lf['tilt_deg']:.1f}(claw tilt L/R {lf.get('claw_tilt_deg', {}).get('left', 0):.1f}/{lf.get('claw_tilt_deg', {}).get('right', 0):.1f}) "
                         f"truth11 {'ADMIT' if r.get('truth_wide', {}).get('ok') else 'reject'} acc {max(lf['acc']):.2f} rise {lf['rise_mm']} -> {fl or 'hold'}"
                         f"{'  [ASSUMPTION VIOLATED]' if viol else ''}")
    lines += ["", f"worlds run {tab['n']}, approach refused {tab['refused']}, errors {tab['error']}", "",
              "lift  verdict  model   sim   box        count"]
    for k in sorted(cnt):
        lines.append(f"{k[0]:>4}  {k[1]:>6}  {k[2]:>6}  {k[3]:>4}  {k[4]:>11}  {cnt[k]}")
    for tag in ("belief", "truth", "belief_t11", "truth_t11"):
        uns = sum(c for k, c in cnt.items() if k[1] == tag and k[2] == "admit" and k[3] == "fail" and k[4] == "in_box")
        cln = sum(c for k, c in cnt.items() if k[1] == tag and k[4] == "in_box")
        lines.append(f"{tag:>11}: UNSOUND (admit, sim fails, assumptions held) {uns}   clean rows {cln}")
    for v, (nb, nf) in sorted(crv.items()):
        lines.append(f"CLAW_REL ({CM.CLAW_REL_MM} mm / {CM.CLAW_REL_DEG} deg, ASSUMED) broken in {nb}/{nf} {v} lifts (rows without the field not checked)")
    txt = "\n".join(lines)
    (Path(out) / "SUMMARY.txt").write_text(txt)
    print(txt)


def main(argv=None):
    from world_model.v2.nero.pan.segcert import worlds
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="full", choices=["full", "iid", "overload"])
    ap.add_argument("--tags", default=None)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--lifts", default="v1,v2", help="lift skills to run from the same snapshot (v3 = coupled, 11-01)")
    a = ap.parse_args(argv)
    if a.report:
        return report(a.out)
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    ws = worlds(a.set)
    if a.tags:
        keep = a.tags.split(",")
        ws = [(t, w) for t, w in ws if t in keep]
    done = {p.stem for p in (out / "raw").glob("*.json")}
    jobs = [(t, w, str(out), a.lifts.split(",")) for t, w in ws if t not in done]
    with get_context("spawn").Pool(max(1, a.procs), maxtasksperchild=1) as pool:
        for tag, err in pool.imap_unordered(_job, jobs):
            print(tag, "ERROR" if err else "ok", flush=True)
    (out / f"_DONE_{a.set}").write_text("ok")


if __name__ == "__main__":
    main()
