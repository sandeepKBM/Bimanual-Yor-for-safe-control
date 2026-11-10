"""nero/contents/carry_cert.py -- per-cell segment admission for the tray and box tasks (step 2 + 3).  SIMULATION ONLY.

The robot's belief about the hidden contents is a CELL: a contents-mass interval x the full COM set (placement unknown).
  tray{n}   n mugs visible (ASSUMED), fill per mug in [0, 0.125] or [0.125, 0.25] kg, mugs anywhere inside the rim
  box       contents mass in [0.5, 1.0], [1.0, 1.5], ..., [4.5, 5.0] kg, COM anywhere in the interior
Per cell:
  1. STATIC rows, proof under MODEL (exact corner checks over the cell's contents vertices x the acceleration box):
       torque_payload        ARM-SIDE envelope u* (joint torque vs 16 Nm, arm vertical load vs 3 kg - gripper) of the COMMANDED
                             lever split (carry.LeverCtl) at path samples of every move, <= U_ADMIT
       support_dest          the total COM over the destination support (object footprint x support top) for every vertex
       open_content          tray: planned peak acceleration <= the quasi-static spill limit at the fullest fill of the cell
     Action (ASSUMED rule, user-confirmed): two arms at a_nom; else two arms slowed to a* (largest a with the torque row
     passing, capped by the spill limit); else refuse.  The one-arm utilisation is REPORTED (u_one_nom) but not acted on: the
     carry sim welds both claws, a one-arm carry is not built.
     USER 11-02 (afternoon): the grasp is not our problem -> NO claw rows.  The claw twist cap (envelope.T_CAP, approved that morning)
     is dropped from admission; T_cap=... / --twist-cap reproduces the old rule only.  Failures must come from how the arms carry.
  2. DYNAMIC rows, sampled: worlds theta drawn uniformly from the cell, each run at the admitted speed with the stored backups
     branched every 0.25 s (carry.world_trial) -> pan/cert/admit.certify(..., static=...) per segment, with the Clopper-Pearson
     bound on the dynamic failure rate.

    python -m world_model.v2.nero.contents.carry_cert --tasks tray2,tray4,tray6,box --n 20 --procs 10 --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import numpy as np

from .. import config as C
from .. import gripper as GR
from . import carry as CR
from . import envelope as E
from . import objects as OB
from . import scenes as SC
from . import thresholds as TH

TRAY_FILL_CELLS = ((0.0, 0.05), (0.05, 0.125), (0.125, 0.25))   # kg per mug, ASSUMED cell split
BOX_CELLS = tuple((float(round(a, 2)), float(round(a + 0.5, 2))) for a in np.arange(0.5, 5.0, 0.5))
POT_CELLS = ((0.02, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.25), (1.25, 1.5))   # ASSUMED cell split of the THETA mass box 0.02..1.5 kg (kg of puck)
N_PER_MOVE = 6
A_MIN = 0.05                                            # m/s^2, ASSUMED: slower than this = refuse (the task stalls)


def plan_key(task):
    """which carry.PLANS entry / support applies: "tray" (tray2/4/6) | "pot" | "box"."""
    return "tray" if task.startswith("tray") else ("pot" if task == "pot" else "box")


def world_default(task):
    """the default world of a task (the static admission's)."""
    k = plan_key(task)
    return SC.TrayWorld(n_cups=int(task[4:])) if k == "tray" else (SC.PotWorld() if k == "pot" else SC.BoxWorld())


def cells(task):
    if task == "pot":
        return [(c, ) for c in POT_CELLS]
    return [(f, ) for f in TRAY_FILL_CELLS] if task.startswith("tray") else [(c, ) for c in BOX_CELLS]


def cell_spec(task, cell):
    if task == "pot":
        return SC.pot_spec(*cell)
    if task.startswith("tray"):
        n = int(task[4:])
        sp = OB.coffee_tray(n)
        lo, hi = cell
        sp.contents = OB.Contents(n * (OB.CUP["empty"] + lo), n * (OB.CUP["empty"] + hi), sp.contents.P, sp.contents.labels)
        D, H = OB.CUP["D"], OB.CUP["H"]
        h_fill = SC.CUP_BOT + hi / (SC.WATER * np.pi * SC.CUP_RIN ** 2)          # fullest fill of the cell (scenes.py geometry)
        sp.open_limits = [OB.OpenLimit("cup_spill", OB.cup_spill_limit(D, H, h_fill), "a_max = g 2 (H - h) / D, fullest fill of the cell")]
        return sp
    sp = OB.boxed(*cell)
    return sp


def path_offsets(task):
    """object offsets along the planned moves: N_PER_MOVE samples per move (the geometric path is the straight line; smoothstep
    timing).  -> ({key: offset}, {seg: [keys]})"""
    offs, per_seg, cur = {"p0": np.zeros(3)}, {}, np.zeros(3)
    k = 0
    for seg, dl in CR.plan_moves(plan_key(task)):
        keys = []
        for i in range(1, N_PER_MOVE + 1):
            k += 1
            offs[f"p{k}"] = cur + np.asarray(dl, float) * i / N_PER_MOVE
            keys.append(f"p{k}")
        per_seg[seg] = ["p0" if not per_seg else list(per_seg.values())[-1][-1]] + keys
        cur = cur + np.asarray(dl, float)
    return offs, per_seg


def support_row(task, spec, contents, world, disp=None):
    """total COM over the destination support for every contents vertex (linear in pi -> exact).  disp: object displacement start -> goal [m] (default: sum of PLANS)."""
    from ...stack import tasks as TK
    td = TK.TASKS.get(task)
    caps = json.loads(TH.CAP_FILE.read_text())["pot"]
    tl, tr = np.asarray(caps["tcp"]["W0_liftoff"]["left"]), np.asarray(caps["tcp"]["W0_liftoff"]["right"])
    o_ref = 0.5 * (tl + tr) - np.array([0, 0, TH.GRASP_DZ])
    if td is not None and td.base:                 # table-behind task (11-09): the goal support (counter2) of the swerve scene; dest = the goal xy (MODEL: scene.place_pose_xy)
        from ..base import scene as BS
        sup = BS.support_tops_behind(world)[td.goal_support]
        dest = np.array([*BS.place_pose_xy(world), o_ref[2]], float)
    else:
        tops = SC.support_tops(world)
        sup = tops["shelf"] if plan_key(task) == "box" else tops["counter"]
        dest = o_ref + (sum(np.asarray(dl, float) for _, dl in CR.plan_moves(plan_key(task))) if disp is None else np.asarray(disp, float))
    disc = None
    if task == "pot":     # MODEL: the pot's contact is its bottom disc, radius R + t (pan/scene.py:102, cylinder geom "pan_bottom")
        r_c = SC.PS.ASSUMED["pan_radius"] + SC.PS.ASSUMED["pan_wall_t"]
        hx = hy = r_c       # bounding square of the disc; the disc test below is the binding one
        if os.environ.get("BIMANUAL_LEGACY_POT_SQUARE") == "1":     # pre-11-09: the inscribed square (half side r / sqrt 2); refused contents from ~1 kg (checker audit 11-06, S3)
            hx = hy = r_c / np.sqrt(2.0)
        else:
            disc = r_c
    else:
        hx, hy = (OB.TRAY["lx"] / 2, OB.TRAY["ly"] / 2) if task.startswith("tray") else (OB.BOX["lx"] / 2, OB.BOX["ly"] / 2)
    x0, x1 = max(dest[0] - hx, sup["x"][0]), min(dest[0] + hx, sup["x"][1])
    y0, y1 = max(dest[1] - hy, sup["y"][0]), min(dest[1] + hy, sup["y"][1])
    worst, worst_v = np.inf, None
    for pi in contents.pi_vertices():
        M = spec.empty_mass + pi[0]
        c = dest[:2] + (spec.empty_mass * spec.empty_com[:2] + pi[1:3]) / M
        marg = min(c[0] - x0, x1 - c[0], c[1] - y0, y1 - c[1])
        if disc is not None:    # COM over (disc ∩ support top).  The test ||M (c - dest)|| <= disc M is convex in pi, so its worst case over the contents polytope is at a vertex: exact
            marg = min(marg, disc - float(np.linalg.norm(c - dest[:2])))
        if marg < worst:
            worst, worst_v = marg, dict(m=round(float(pi[0]), 3), c=(pi[1:4] / max(pi[0], 1e-9)).round(4).tolist())
    return dict(ok=bool(worst >= 0.0), margin_m=round(float(worst), 4), worst=worst_v, support=sup, region=[[x0, x1], [y0, y1]],
                **({} if disc is None else dict(disc=dict(centre=[round(float(v), 4) for v in dest[:2]], radius=round(float(disc), 4)))),
                guarantee="proof under MODEL (rigid object on a flat support: at rest iff the total COM is over the contact "
                          "rectangle (pot: bottom disc); placement at the nominal destination, placement error NOT covered)")


def refused_static(task, cell, spec, why):
    """static verdict when the planner REFUSES the goal (closed_chain, 11-05): no path, no admission.  Same keys the callers read."""
    cont = spec.contents
    return dict(task=task, cell=list(cell), contents_m=[cont.m_lo, cont.m_hi], action="refuse (unreachable)", a_admit=None, a_open=min([ol.a_max for ol in spec.open_limits], default=np.inf),
                u_nom=float("inf"), worst_nom={}, u_one_nom=float("inf"), worst_one={}, T_cap=None, rows={}, planner="closed_chain", why=why)


def static_cell(task, cell, world, T_cap=None, spec=None, planner=None, wp=None):
    """T_cap=None (default, user 11-02 pm): arm-side rows only.  T_cap=E.T_CAP reproduces the morning rule (claw twist row).
    spec: an ObjectSpec for an arbitrary contents set (11-03, probe.py: the interval a probe measured); default = the cell's.
    planner (11-05, None -> carry.resolve_planner): "scripted" = the legacy PLANS samples + continuation IK (unchanged); "closed_chain" = the samples and joint
    configurations of the PLANNED path (stack/planners/closed_chain.static_path); an unreachable goal gives a refused verdict (refused_static).
    wp (11-05): a closed_chain.WaypointPlan written by the proposer: implies planner "closed_chain"; the rows are per its segment names (the place row = its last move)."""
    spec = cell_spec(task, cell) if spec is None else spec
    planner = "closed_chain" if wp is not None else CR.resolve_planner(planner)
    caps = json.loads(TH.CAP_FILE.read_text())
    if planner == "closed_chain":
        from ...stack.planners import closed_chain as CCP
        try:
            offs, per_seg, cap = CCP.static_path(task, wp)
            disp = CCP.goal_offset(task, wp)
        except CCP.PlanRefused as e:
            return refused_static(task, cell, spec, str(e))
    else:
        offs, per_seg = path_offsets(task)
        cap = TH.capture_path(caps["pot"], offs)
        disp = None
    arms = E.Arms(cap)
    tl, tr = np.asarray(caps["pot"]["tcp"]["W0_liftoff"]["left"]), np.asarray(caps["pot"]["tcp"]["W0_liftoff"]["right"])
    o_ref = 0.5 * (tl + tr) - np.array([0, 0, TH.GRASP_DZ])
    env = E.Envelope(spec, arms, o_ref, np.eye(3), ["left", "right"], grip_mass=caps["pot"]["gripper_mass"], offsets=offs, T_cap=T_cap)
    cont = spec.contents
    a_open = min([ol.a_max for ol in spec.open_limits], default=np.inf)
    keys = list(offs)
    u_nom, det_nom = env.worst(cont, keys, E.A_NOM, "two_lever")
    u_one, det_one = env.worst(cont, keys, E.A_NOM, "one")          # reported only (one-arm carry not built)
    row_name = "torque_payload" if T_cap is None else "torque_payload_twist"
    row_guar = (f"proof under MODEL (rigid weld ASSUMED, quasi-static, NERO MJCF arms, config.tau_limit() torque, 3 kg payload, "
                + ("no claw rows (user 11-02)" if T_cap is None else f"claw twist {T_cap} Nm ASSUMED")
                + f", admit u <= {E.U_ADMIT} MODEL, USER 11-09)")
    if u_nom <= E.U_ADMIT and E.A_NOM <= a_open:
        a_adm, action = E.A_NOM, "two arms"
    else:
        a_t = env.a_max(cont, keys, "two_lever", limit=E.U_ADMIT)
        a_adm = None if a_t is None else min(a_t, a_open)
        action = "two arms, slow" if (a_adm is not None and a_adm >= A_MIN) else "refuse"
        if action == "refuse":
            a_adm = None
    rows = {}
    for seg, ks in per_seg.items():
        aa = a_adm if a_adm is not None else E.A_NOM
        u, det = env.worst(cont, ks, aa, "two_lever")
        u_lp, _ = env.worst(cont, ks, aa, "two")
        rows[seg] = {row_name: dict(
            ok=bool(u <= E.U_ADMIT), u_max=round(float(u), 4), u_best_split=round(float(u_lp), 4), a=aa, n_samples=len(ks),
            worst={k: v for k, v in det.items() if k in ("wp", "m", "c", "a", "Fz", "twist")},
            guarantee=row_guar + f" over the cell x acceleration box at {len(ks)} path samples")}
        if a_open < np.inf:
            rows[seg]["open_content"] = dict(ok=bool(aa <= a_open), a_plan=aa, a_open=round(float(a_open), 4),
                                             guarantee="proof under MODEL (quasi-static liquid surface; slosh NOT covered)")
    rows["release"] = dict(support_dest=support_row(task, spec, cont, world, disp))
    rows[list(per_seg)[-1] if wp is not None else "place"]["support_dest"] = rows["release"]["support_dest"]
    if action != "refuse" and not rows["release"]["support_dest"]["ok"]:
        action = "refuse (placement)"
        a_adm = None
    out = dict(task=task, cell=list(cell), contents_m=[cont.m_lo, cont.m_hi], action=action, a_admit=a_adm, a_open=a_open,
               u_nom=round(float(u_nom), 4), worst_nom={k: v for k, v in det_nom.items() if k in ("wp", "m", "c", "Fz", "twist")},
               u_one_nom=round(float(u_one), 4), worst_one={k: v for k, v in det_one.items() if k in ("wp", "m", "c", "worst_joint", "binding")},
               T_cap=T_cap, rows=rows)
    if planner != "scripted":
        out["planner"] = planner
    if wp is not None:
        out["wp"] = wp.key_data()
    return out


def sample_pot_world(cell, rng):
    """THETA draws (pan/theta.py): mass uniform on the cell (ASSUMED box), (c_r, c_phi) uniform (MODEL: uniform over the C_R disc), lid_mu_scale uniform (ASSUMED);
    handle length fixed (POT_HANDLE_LEN ASSUMED).  dx, dy, yaw and mu_scale (start-pose / global friction) are NOT sampled (mu_scale stays 1.0)."""
    from ..pan.theta import THETA, world_from_theta
    th = dict(dx=0.0, dy=0.0, yaw_deg=0.0, mu_scale=1.0, handle_len=SC.POT_HANDLE_LEN, mass=float(rng.uniform(*cell)),
              c_r=float(rng.uniform(*THETA["c_r"][:2])), c_phi=float(rng.uniform(*THETA["c_phi"][:2])),
              lid_mu_scale=float(rng.uniform(*THETA["lid_mu_scale"][:2])))
    wd = world_from_theta(th)
    return SC.PotWorld(handle_len=wd["handle_len"], puck_kg=wd["hidden_mass"], puck_uw=tuple(wd["placement"]), lid_mu_scale=wd["perturb"]["lid_mu_scale"])


def sample_world(task, cell, rng):
    if task == "pot":
        return sample_pot_world(cell, rng)
    if task.startswith("tray"):
        n = int(task[4:])
        w = SC.sample_tray_world(n, rng, fill_max=cell[1])
        f = rng.uniform(cell[0], cell[1], n)
        w.fills_kg = [float(x) for x in f]
        return w
    hx = SC.BoxWorld().box_outer[0] / 2 - SC.BoxWorld().wall_t - 0.04
    hy = SC.BoxWorld().box_outer[1] / 2 - SC.BoxWorld().wall_t - 0.06
    return SC.BoxWorld(contents_kg=float(rng.uniform(*cell)), contents_xy=(float(rng.uniform(-hx, hx)), float(rng.uniform(-hy, hy))))


def _trial(args):
    task, cell, i, seed, a, m_hat, fc, *rest = args
    ws = rest[0] if rest else None
    rng = np.random.default_rng(seed)
    w = sample_world(task, cell, rng)
    try:
        r = CR.world_trial(w, f"{task}_{cell[0]}_{i}", a_plan=a, m_hat=m_hat, fric_comp=fc, weld_solref=ws)
    except Exception as e:                                  # a crash is reported, never silently dropped
        r = dict(tag=f"{task}_{cell[0]}_{i}", error=repr(e), theta=CR._theta(w), segments=[])
    r["cell"], r["seed"] = list(cell), seed
    return task, tuple(cell), r


WELD_STIFF = C.WELD_STIFF    # weld solref [s, dampratio]: 4 ms (>= 2 x the 1 ms physics step); MuJoCo default 0.02 s bends 2-3 deg (11-03: one source, config)


TASK_ID = {"tray2": 2, "tray4": 4, "tray6": 6, "box": 9, "pot": 7}


def cell_seed(task, cell, i):
    """world seed fixed by (task, cell, index) -- NOT by which cells a rule admits -- so two rules sample the same worlds in a cell."""
    return int(np.random.SeedSequence([TASK_ID[task], int(round(1000 * cell[0])), int(round(1000 * cell[1])), i]).generate_state(1)[0])


def run(tasks, n, procs, out, fric_comp=None, weld_stiff=None, T_cap=None, only=None):
    """only: set of cell keys "task|lo-hi" to simulate (the rest get the static rows only); None = every admitted cell."""
    import multiprocessing as mp
    from ..pan.cert import admit as AD
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cp = C.carry_physics()                                  # 11-03 item 6: None = the shared default (fric_comp + stiff weld; legacy pair under BIMANUAL_LEGACY_CARRY_PHYS=1)
    fric_comp = bool(cp["fric_comp"]) if fric_comp is None else fric_comp
    weld_stiff = (cp["weld_solref"] is not None) if weld_stiff is None else weld_stiff
    res = dict(date=time.strftime("%Y-%m-%d"), n_per_cell=n, assumptions=dict(
        grasp="ASSUMED rigid weld (user 11-01)", **GR.provenance("rigid weld"),
        T_cap=("none: no claw rows (user 11-02 pm, arm-side rows only)" if T_cap is None else f"ASSUMED {T_cap} Nm (old 11-02 am rule)"),
        seeds="carry_cert.cell_seed(task, cell, i)", u_admit=f"MODEL {E.U_ADMIT} (rated motor / payload limits, USER 11-09)", a_nom=E.A_NOM,
        a_backup=CR.A_BACKUP, a_min=A_MIN, controller="carry.LeverCtl (FL-MPC + lever-split payload ff), config.tau_limit() torque box",
        claw_rows_in_sim="recorded, NOT in U (closed-chain internal stress under the weld; see carry.py doc)",
        spill_sim="quasi-static proxy from sim mug acceleration; slosh NOT modelled",
        worlds="theta ~ uniform on the cell (tray: scenes.sample_tray_world placement + uniform fills; box: uniform mass + xy)",
        proposer="none (scripted plan; no LLM in this run)",
        fric_comp=(f"carry.FRIC_COMP {CR.FRIC_COMP} (wrapper after FL-MPC; ASSUMED gains)" if fric_comp else "off"),
        weld=(f"solref {WELD_STIFF} (stiff; MODEL choice for the ASSUMED rigid grasp)" if weld_stiff else "MuJoCo default solref (0.02, 1)")),
        cells={})
    if CR.resolve_planner() != "scripted":
        res["assumptions"]["planner"] = f"{CR.resolve_planner()} (stack/planners/closed_chain.py; its labelled assumptions: closed_chain.assumptions())"
    jobs = []
    for task in tasks:
        world0 = world_default(task)
        for (cell,) in cells(task):
            st = static_cell(task, cell, world0, T_cap=T_cap)
            key = f"{task}|{cell[0]}-{cell[1]}"
            res["cells"][key] = dict(static=st, records=[], accel=cell_accel(task, st["a_admit"]))
            print(f"static {key}: {st['action']} a={st['a_admit']} u_nom={st['u_nom']} u_one={st['u_one_nom']}", flush=True)
            if st["a_admit"] is not None and (only is None or key in only):
                for i in range(n):
                    jobs.append((task, cell, i, cell_seed(task, cell, i), st["a_admit"], 0.5 * sum(st["contents_m"]), fric_comp,
                                 WELD_STIFF if weld_stiff else None))
    (out / "STATIC.json").write_text(json.dumps(res, indent=1, default=float))
    with mp.get_context("spawn").Pool(procs, maxtasksperchild=3) as pool:
        for task, cell, r in pool.imap_unordered(_trial, jobs):
            key = f"{task}|{cell[0]}-{cell[1]}"
            res["cells"][key]["records"].append(r)
            with open(out / "trials.jsonl", "a") as f:
                f.write(json.dumps(dict(key=key, **r), default=float) + "\n")
    for key, c in res["cells"].items():
        recs = [r for r in c["records"] if "error" not in r]
        c["n_errors"] = len(c["records"]) - len(recs)
        from world_model.v2.nero import fl_log as FLL
        c["fl"] = FLL.aggregate([s["nominal"] for r in recs for s in r.get("segments", [])])   # 11-03 F2: nominal runs of all sampled worlds
        c["cert"] = {}
        if not recs:
            continue
        for seg in CR.SEGS:
            stat = {k: v for k, v in c["static"]["rows"].get(seg, {}).items()}
            static = dict(ok=all(v["ok"] for v in stat.values()), rows=stat)
            c["cert"][seg] = AD.certify(recs, seg, family=(0.0,), static=static)
            _count_pre_hold(c["cert"][seg], recs)
        c["n_no_world_segments"] = sum(1 for ce in c["cert"].values() if ce.get("no_world"))   # 11-03 item 12a: counted apart from passes
        res["n_no_world_segments"] = res.get("n_no_world_segments", 0) + c["n_no_world_segments"]
    (out / "RESULT.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "RESULT.txt").write_text(summary(res))
    return res


def cell_accel(task, a, planner=None):
    """11-03 item 10: per segment, the acceleration the admitted plan actually commands (carry.commanded_accel: T_MIN clamps the move time, so
    a_cmd_peak <= a when the move is short).  None when the cell is refused."""
    if a is None:
        return None
    if CR.resolve_planner(planner) == "closed_chain":
        from ...stack.planners import closed_chain as CCP
        return CCP.accel_records(task, a)
    return {seg: CR.commanded_accel(float(np.linalg.norm(dl)), a) for seg, dl in CR.plan_moves(plan_key(task))}


def _count_pre_hold(ce, recs):
    """11-03 item 12c: a U entry during the 0.4 s load-transfer hold (record 'pre_hold_viol') is a failure of every segment of that world; the
    certificate counts it (before, only tilt_sweep did).  Adds n_pre_hold_viol, turns ok False, appends counterexamples, refreshes eps_upper."""
    bad = [r for r in recs if r.get("pre_hold_viol")]
    ce["n_pre_hold_viol"] = len(bad)
    if not bad:
        return ce
    from ..pan.segcert import cp_upper
    ce["ok"] = False
    ce.setdefault("counterexamples", [])
    for r in bad:
        ce["counterexamples"].append(dict(tag=r.get("tag"), theta=r.get("theta"), where="pre_hold", t=r["pre_hold_viol"].get("t"),
                                          why=r["pre_hold_viol"].get("kind")))
    k = len({x.get("tag") for x in ce["counterexamples"]})
    n = ce.get("n_worlds", 0)
    ce["eps_upper"] = round(cp_upper(k, n, ce.get("eps_conf", 0.95)), 4) if n else 1.0
    return ce


def summary(res):
    L = [f"carry certificate per cell ({res['date']}), n = {res['n_per_cell']} sampled worlds per admitted cell", ""]
    for key, c in res["cells"].items():
        st = c["static"]
        L.append(f"{key}: contents {st['contents_m'][0]:.2f}..{st['contents_m'][1]:.2f} kg -> {st['action']}"
                 + (f" (a = {st['a_admit']:.3f} m/s^2)" if st["a_admit"] else "") + f"  u_nom(a_nom) = {st['u_nom']}")
        if c.get("accel"):
            L.append("    commanded peak a per segment: " + ", ".join(f"{sg} {v['a_cmd_peak']:.3f}" + (" (T_MIN clamp)" if v["clamped"] else "") for sg, v in c["accel"].items()))
        sup = st["rows"].get("release", {}).get("support_dest")                      # refused_static (closed_chain, unreachable) has no rows
        if sup:
            L.append(f"    support at destination: margin {sup['margin_m'] * 1e3:.1f} mm (proof under MODEL)")
        elif st.get("why"):
            L.append(f"    planner refused: {st['why']}")
        for seg, ce in c.get("cert", {}).items():
            cx = ce.get("counterexamples", [])
            L.append(f"    {seg:8s} ok={ce['ok']}{'  [no world qualified]' if ce.get('no_world') else ''}{('  pre_hold_viol=' + str(ce['n_pre_hold_viol'])) if ce.get('n_pre_hold_viol') else ''}  n={ce['n_worlds']}  eps<={ce.get('eps_upper')}  {('cx: ' + ', '.join(str(x.get('why') or x.get('row')) for x in cx[:4])) if cx else ''}")
        f_ = c.get("fl") or {}
        if f_.get("n_with"):
            L.append(f"    FL-MPC (nominal runs): infeasible {f_['n_infeasible']}/{f_['steps']} steps (relaxed {f_['n_relaxed']}, zero {f_['n_zero']}, "
                     f"scale-shrunk {f_['n_scale_shrunk']}); TRUE min arm-arm {f_['armarm_min_m']:.3f} m, min static {f_['static_min_m']:.3f} m, "
                     f"pen steps {f_['n_armarm_pen']}/{f_['n_static_pen']}, max |tau|/limit {f_['tau_ratio_max']:.2f}")
        if c.get("n_errors"):
            L.append(f"    ERRORS: {c['n_errors']} trials crashed (see trials.jsonl)")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="tray2,tray4,tray6,box")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--out", default="/data/local/ss5772/contents/carry_cert_2026-11-02")
    ap.add_argument("--static-only", action="store_true")
    ap.add_argument("--fric-comp", action=argparse.BooleanOptionalAction, default=None,
                    help="joint friction compensation + integral wrapper (carry.FRIC_COMP); default = config.carry_physics()")
    ap.add_argument("--weld-stiff", action=argparse.BooleanOptionalAction, default=None,
                    help=f"weld solref {WELD_STIFF} instead of the MuJoCo default; default = config.carry_physics()")
    ap.add_argument("--twist-cap", action="store_true", help=f"old 11-02 am rule: claw twist row at {E.T_CAP} Nm (default: arm-side only)")
    ap.add_argument("--only", default=None, help="comma list of cell keys task|lo-hi to simulate (default: every admitted cell)")
    ap.add_argument("--planner", choices=CR.PLANNERS, default=None,
                    help="plan the static rows are certified on and the sampled worlds execute (default: env BIMANUAL_PLANNER, else scripted = legacy PLANS)")
    a = ap.parse_args(argv)
    if a.planner is not None:
        import os
        os.environ["BIMANUAL_PLANNER"] = a.planner                  # inherited by the spawned workers
    T_cap = E.T_CAP if a.twist_cap else None
    if a.static_only:
        for task in a.tasks.split(","):
            world0 = world_default(task)
            for (cell,) in cells(task):
                st = static_cell(task, cell, world0, T_cap=T_cap)
                rn = "torque_payload" if T_cap is None else "torque_payload_twist"
                print(task, cell, st["action"], st["a_admit"], "u_nom", st["u_nom"], "u_one", st["u_one_nom"], st["worst_one"].get("binding"),
                      "support", st["rows"].get("release", {}).get("support_dest", {}).get("margin_m"),
                      {s: r[rn]["u_max"] for s, r in st["rows"].items() if rn in r}, flush=True)
        return
    t0 = time.time()
    only = set(a.only.split(",")) if a.only else None
    res = run(a.tasks.split(","), a.n, a.procs, a.out, fric_comp=a.fric_comp, weld_stiff=a.weld_stiff, T_cap=T_cap, only=only)
    print(summary(res))
    print(f"wall {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
