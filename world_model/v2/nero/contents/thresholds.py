"""nero/contents/thresholds.py -- action thresholds v2 (11-02): claw twist cap + admission margin + heavier loads.  SIMULATION ONLY.

User-approved 11-02 (relayed: "approved for those go"):
  * claw twist cap T_cap = 1.30 Nm per claw (ASSUMED, envelope.T_CAP) so "one arm or two" is a real decision under the weld;
  * admission margin: admit only when the MODEL utilisation <= 0.9 (ASSUMED, envelope.U_ADMIT), to be sized from sim runs;
  * heavier loads: coffee tray with 2 / 4 / 6 mugs (hidden fill 0..250 ml each + hidden placement), box of cans 0.5..5 kg onto a
    shelf (objects.boxed, scenes.BoxWorld);
  * a_nom 0.5 m/s^2, mug 300 g / 250 ml / >= 1 cm freeboard, action order one arm -> two arms -> slow -> set down -> refuse, and
    the waypoints below (ASSUMED, confirmed by the user).
Each task is run per setting on the same contents sets: "v1" = the 11-01 rules (no twist row, u <= 1), "v2" = the rules above,
"v3" = CURRENT (USER 11-02 pm: the grasp is not our problem -> twist cap dropped; arm-side rows at u <= 0.9).

Tray and box use the pot's grasp points (scenes.py places the tabs there): q at W0 = the pot capture (results/contents_envelope_
2026-11-01/capture.json), the other waypoints by continuation IK along the planned moves.  o = obj_ref = the base centre, 0.0355 m
below the grasp points (scenes.py MODEL).  Guarantee: proof under MODEL at the listed waypoints (envelope.py module doc).

    python -m world_model.v2.nero.contents.thresholds [--out DIR] [--procs 8]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from .. import config as C
from ..democ import kin as K
from . import envelope as E
from . import objects as OB

CAP_FILE = E.RESULTS / "capture.json"
RESULTS = E.RESULTS.parent / "contents_thresholds_2026-11-02"
GRASP_DZ = 0.0355                                   # MODEL (scenes.py): grasp points above the object's base centre
# ASSUMED plans (user-confirmed 11-02), offsets of the object from its start pose [m], in order (continuation IK follows them)
TRAY_WPS = {"W0_liftoff": (0, 0, 0), "T1_lift5cm": (0, 0, 0.05), "T2_carry10cm": (-0.10, 0, 0.05), "T3_place": (-0.10, 0, 0.0)}
BOX_WPS = {"W0_liftoff": (0, 0, 0), "B1_lift18cm": (0, 0, 0.18), "B2_over_shelf": (-0.20, 0, 0.18), "B3_on_shelf": (-0.20, 0, 0.15)}
SETTINGS = {"v1": dict(T_cap=None, limit=1.0), "v2": dict(T_cap=E.T_CAP, limit=E.U_ADMIT),
            # v3 (USER 11-02 pm, CURRENT): grasp is not our problem -> no claw twist row; arm-side rows (16 Nm joints, 3 kg arm
            # payload) at the u <= 0.9 margin.  v1/v2 kept for the side-by-side only.
            "v3": dict(T_cap=None, limit=E.U_ADMIT)}


def capture_path(cap_pot, wps):
    """q per waypoint from the pot's W0 q, by continuation IK along the consecutive moves (orientation held)."""
    from ..pan import covered as CV
    m = CV.compose()
    q, tcp = {}, {}
    for s in cap_pot["sides"]:
        pik = K.PoseIK(m, s)
        q0 = np.asarray(cap_pot["q"]["W0_liftoff"][s], float)
        p0, R0 = pik.ik.pose(q0)
        qq, prev = q0.copy(), np.zeros(3)
        for wp, dl in wps.items():
            dl = np.asarray(dl, float)
            err = 0.0
            for i in range(1, 61):
                qq, err = pik.solve(p0 + prev + (dl - prev) * i / 60, R0, qq, q_post=q0, k_post=0.1, iters=60, tol=1e-9)
            prev = dl
            q.setdefault(wp, {})[s] = qq.tolist()
            tcp.setdefault(wp, {})[s] = pik.ik.pose(qq)[0].tolist()
            if err > 1e-4:
                q[wp][s + "_ik_err"] = float(err)
    return dict(cap_pot, q=q, tcp=tcp)


def tasks(caps):
    pot, pan = caps["pot"], caps["pan"]
    tl, tr = np.asarray(pot["tcp"]["W0_liftoff"]["left"]), np.asarray(pot["tcp"]["W0_liftoff"]["right"])
    o_ref = 0.5 * (tl + tr) - np.array([0, 0, GRASP_DZ])
    objs = E.objects_from(pot, pan)
    out = {}
    for name in ("pan", "pot"):
        spec, cap, o0, R0 = objs[name]
        out[name] = dict(spec=spec, cap=cap, o0=o0, R0=R0, wps=E.WAYPOINTS, carry=E.CARRY, liftoff=E.LIFTOFF,
                         modes=("one",) if name == "pan" else ("one", "two", "two_lever", "two_share"))
    ctray = capture_path(pot, TRAY_WPS)
    for n in (2, 4, 6):
        out[f"tray{n}"] = dict(spec=OB.coffee_tray(n), cap=ctray, o0=o_ref, R0=np.eye(3), wps=TRAY_WPS, carry=tuple(TRAY_WPS),
                               liftoff=("W0_liftoff", "T1_lift5cm"), modes=("one", "two", "two_lever", "two_share"))
    out["box"] = dict(spec=OB.boxed(), cap=capture_path(pot, BOX_WPS), o0=o_ref, R0=np.eye(3), wps=BOX_WPS, carry=tuple(BOX_WPS),
                      liftoff=("W0_liftoff", "B1_lift18cm"), modes=("one", "two", "two_lever", "two_share"))
    return out


def _job(args):
    name, setting, caps = args
    t = tasks(caps)[name]
    st = SETTINGS[setting]
    spec, cap = t["spec"], t["cap"]
    arms = E.Arms(cap)
    env = E.Envelope(spec, arms, t["o0"], t["R0"], cap["sides"], grip_mass=cap["gripper_mass"],
                     offsets={k: v for k, v in t["wps"].items()}, T_cap=st["T_cap"])
    lim = st["limit"]
    cont = spec.contents
    full = cont.scaled(cont.m_lo)
    centre = OB.Contents(cont.m_lo, cont.m_lo, np.array([[0.0, 0.0, float(cont.P[:, 2].mean())]]))
    rec = dict(ik_err={wp: {k: v for k, v in qs.items() if k.endswith("_ik_err")} for wp, qs in cap["q"].items()
                       if wp in t["wps"]})
    # the action rule uses what the robot COMMANDS: one arm, or two arms with the lever split (carry.LeverCtl); "two" (LP) is the
    # capability of the best split and "two_share" the old 50/50-at-COM payload model -- both reported, not used for the action
    act_modes = tuple(md for md in ("one", "two_lever") if md in t["modes"])
    for md in t["modes"]:
        rec[f"m_max_{md}"] = dict(
            centre_static=env.m_max(centre, t["carry"], 0.0, md, limit=lim),
            worst_com_static=env.m_max(full, t["carry"], 0.0, md, limit=lim),
            worst_com_a_nom=env.m_max(full, t["carry"], E.A_NOM, md, limit=lim),
            liftoff_worst_com_static=env.m_max(full, t["liftoff"], 0.0, md, limit=lim))
        u, det = env.worst(cont, t["carry"], E.A_NOM, md)
        rec[f"worst_{md}_full_set_a_nom"] = dict(u=round(float(u), 4), **{k: v for k, v in det.items() if k in ("wp", "m", "c", "binding", "twist", "Fz")})
    a = env.action(cont, modes=act_modes, limit=lim, carry=t["carry"], liftoff=t["liftoff"])
    rec["action_full_set"] = dict(action=a["action"], a_star=a["a_star"],
                                  rows={k: {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in a["rows"].items()})
    grid = np.round(np.linspace(cont.m_lo, cont.m_hi, 10), 3)
    rec["action_map"] = []
    for mh in grid:
        aa = env.action(cont.scaled(float(mh)), modes=act_modes, limit=lim, carry=t["carry"], liftoff=t["liftoff"])
        rec["action_map"].append(dict(m_hi=float(mh), action=aa["action"], a_star=aa["a_star"]))
    return name, setting, rec


def summary_text(res):
    L = [f"contents thresholds v2 ({res['date']}): proof under MODEL at the waypoints; see RESULT.json assumptions", ""]
    for name, o in res["objects"].items():
        L.append(f"{name}: empty {o['empty_mass']:.2f} kg, contents {o['contents_m'][0]:.2f}..{o['contents_m'][1]:.2f} kg"
                 f"  open limits {o['open_limits']}")
        for st in SETTINGS:
            r = o["settings"].get(st)
            if not r:
                continue
            mm = "  ".join(f"{k[6:]}: static {v['worst_com_static']} (centred {v['centre_static']}) a_nom {v['worst_com_a_nom']}"
                           for k, v in r.items() if k.startswith("m_max_"))
            L.append(f"  [{st}] full set -> {r['action_full_set']['action']}"
                     + (f" a*={r['action_full_set']['a_star']}" if r['action_full_set']['a_star'] else ""))
            L.append(f"       m_max (largest contents mass, every COM in the set): {mm}")
            L.append("       map m_hi -> action: " + "; ".join(f"{x['m_hi']}: {x['action']}" + (f" (a*={x['a_star']})" if x['a_star'] else "")
                                                     for x in r["action_map"]))
        L.append("")
    return "\n".join(L)


def run(out_dir=RESULTS, procs=8):
    import multiprocessing as mp
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    caps = json.loads(CAP_FILE.read_text())
    T = tasks(caps)
    res = dict(date="2026-11-02", assumptions=dict(
        grasp="ASSUMED rigid weld once closed (user 11-01)",
        T_cap=f"v2 only: ASSUMED {E.T_CAP} Nm per claw (2-norm; LP uses the inscribed box) = vizcoast claw_capacity fit extrapolated "
              f"to 40 N.  v1 and v3 (current): no claw rows",
        u_admit=f"v2, v3: ASSUMED admit at MODEL u <= {E.U_ADMIT} (controller feedback margin; to be sized from sim)",
        tau_lim="MEASURED-spec 16 Nm MIT command range (fw >= 1.11)",
        payload=f"MEASURED-spec {E.PAYLOAD_KG} kg (NERO manual) minus gripper 0.5 kg (MJCF), vertical",
        a_nom=f"ASSUMED {E.A_NOM} m/s^2 per axis", quasi_static="ASSUMED (Coriolis dropped, object level, translation only)",
        waypoints=dict(pan_pot=E.WAYPOINTS, tray=TRAY_WPS, box=BOX_WPS),
        mugs="ASSUMED count visible; 0.30 kg empty, fill 0..0.25 kg, >= 1 cm freeboard; placement anywhere inside the rim",
        box="ASSUMED 0.25 kg crate, contents 0.5..5 kg, COM anywhere in the interior",
        guarantee="proof under MODEL over contents set x acceleration box at the listed waypoints; between waypoints, placement "
                  "error, slosh, contents sliding NOT covered"), objects={})
    for name, t in T.items():
        sp = t["spec"]
        res["objects"][name] = dict(empty_mass=round(float(sp.empty_mass), 4), contents_m=(sp.contents.m_lo, sp.contents.m_hi),
                                    open_limits={o.name: round(o.a_max, 3) for o in sp.open_limits}, labels=sp.contents.labels,
                                    plausible=sp.plausible, settings={})
    jobs = [(n, st, caps) for n in T for st in SETTINGS]
    with mp.get_context("spawn").Pool(min(procs, len(jobs)), maxtasksperchild=1) as pool:
        for name, st, rec in pool.imap_unordered(_job, jobs):
            res["objects"][name]["settings"][st] = rec
            print(f"done {name} {st}", flush=True)
    (out_dir / "RESULT.json").write_text(json.dumps(res, indent=1, default=float))
    (out_dir / "RESULT.txt").write_text(summary_text(res))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args(argv)
    t0 = time.time()
    res = run(a.out, a.procs)
    print(summary_text(res))
    print(f"wall {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
