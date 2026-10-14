"""flagship/simulate.py -- run ONE side (shield off | on) of the coffee run headless and record everything the renderer needs.  SIMULATION ONLY.

    cd /common/users/ss5772/bimanual ; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=$PWD
    PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
    $PY -m world_model.v2.house.flagship.simulate --side raw    --seed 0 --perception off
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<free, not 0> $PY -m world_model.v2.house.flagship.simulate --side shield --seed 0 --perception live
    $PY -m world_model.v2.house.flagship.simulate --side shield --seed 0 --perception cached        # CPU only (re-uses perception_*.pkl)
    ... --side shield --perception live|cached --place-backend zono_rows --arm-zono-rows --out <NEW dir>   # + FL-MPC zonotope rows on both arms and the held tray in PLACE/RELEASE

Outputs (OUT=/data/local/ss5772/molmospaces/flagship): run_<side>_s<seed>.npz (per teleop tick 50 Hz: qpos, mocap_pos/quat, ctrl, stand pose, scalars) +
run_<side>_s<seed>.json (metrics, events, mission log) + perception/perception_{counter,desk}.pkl (live mode).
Same closed-loop operator model (same seed) on both sides.
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("BIMANUAL_FLMPC_IMPL", "rt")      # exact Goldfarb-Idnani QP FL-MPC (owner decision 2026-10-10); BIMANUAL_FLMPC_IMPL=py = old dual-ascent QP
if "--perception" in sys.argv and sys.argv[sys.argv.index("--perception") + 1] == "live":
    os.environ.setdefault("MUJOCO_GL", "egl")           # only the two stereo captures render; physics is CPU

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from world_model.v2.house.teleop import base as BS, shield as SH
from world_model.v2.house.teleop.sim import Cmd, TICK

from . import scene as FS
from . import mission as MS

OUT = FS.OUT
LABELS = ["none", "off", "auto", "wall", "person", "cup_slip", "cup_jerk", "hw_slew", "speed_cap", "ir_veto", "brake_fallback", "other"]


def label_id(lbl):
    if lbl is None:
        return 0
    if lbl.startswith("person"):
        return LABELS.index("person")
    return LABELS.index(lbl) if lbl in LABELS else LABELS.index("other")


def _tray_pos_obs(sim):
    """BIMANUAL_PROPRIO_ONLY: tray position for the stand / spot placement from the HANDS' forward kinematics (arm encoders + the grasp offset:
    the weld relpose of the held tray, or the friction assist's wrist-camera offset, or -- before any grasp -- the nominal hold relpose).
    No read of the tray body."""
    a = getattr(sim, "assist", None)
    if a is not None and a.cfg.proprio:
        r = a.tray_from_hands()
        if r is not None:
            return np.asarray(r[0], float)
    ps = []
    for s in ("left", "right"):
        b = sim.m.body(f"{s}_arm_gripper_base").id
        e = sim.eq[s]
        rel = sim.m.eq_data[e, 3:6] if (sim.grasped and e >= 0 and int(sim.d.eq_active[e])) else sim.of.nominal[s][3:6]
        ps.append(sim.d.xpos[b] + sim.d.xmat[b].reshape(3, 3) @ rel)
    return 0.5 * (ps[0] + ps[1])


def run_side(shield_on, seed=0, max_s=330.0, perception="off", out=OUT, tag=None, start="full", place_backend="zed", cafe_walker=False, verbose=True,
             tail_s=1.0, quiet_every=500, arm_zono_rows=False, people="truth", people_kw=None, lang=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    side = "shield" if shield_on else "raw"
    tag = tag or f"{side}_s{seed}"
    t_wall0 = time.time()
    wpts = None
    walker_used = cafe_walker and start == "full"
    # ---- world
    if start == "full":
        sim = FS.new_sim(seed, cafe_walker=walker_used)
        start_phase = "OUT"
    elif start == "counter":
        from world_model.v2.house.office import scene as OS
        wpts = OS.office_meta()["waypoints"]["cafe_counter_pick"]
        pre = np.array(wpts["xy"]) - MS.PREDOCK_M * np.array([math.cos(wpts["heading"]), math.sin(wpts["heading"])])
        sim = FS.new_sim(seed, cafe_walker=False, spawn=(pre[0], pre[1], wpts["heading"]))
        start_phase = "SENSE_C"
    elif start == "desk":
        from world_model.v2.house.office import scene as OS
        wd = OS.office_meta()["waypoints"]["lab_desk_place"]
        sim = FS.new_sim(seed, cafe_walker=False, spawn=(wd["xy"][0], wd["xy"][1] - 0.15, wd["heading"]), grasped=True, counter=False)
        start_phase = "CREEP_D"
    else:
        raise ValueError(start)
    m, d = sim.m, sim.d
    wp = sim.scene.waypoints
    # ---- operator / shield / services
    if shield_on and perception != "off":
        from .perception import PerceptionService
        services = PerceptionService(perception, cache_dir=out / "perception", backend=place_backend, seed=seed, log=lambda s: print(s, flush=True))
    else:
        services = MS.NullServices()
    lang_services = None
    if lang is not None:                     # --lang: language layer (grounding + spec authoring + mu prior at the SENSE phases); default off = bit-identical
        if not (shield_on and perception != "off"):
            raise SystemExit("--lang needs --side shield and --perception live|cached (it grounds names on the perception captures)")
        from world_model.v2.house.lang import cli as LCLI
        services = lang_services = LCLI.make_lang_services(services, lang, out, sim=sim, log=lambda s: print(s, flush=True))
    op = MS.CoffeeRunOperator(seed, sim.scene, services, start_phase=start_phase)
    shield = SH.Shield(None, occ=sim.occ, a_hw=sim.a_hw) if shield_on else SH.Passthrough()
    if shield_on and people != "truth":      # head-camera KF tracker as the shield's people source (people_track); sim.people_state stays the metric truth
        sim.people_source = people
        shield = sim.make_shield(None, **dict(dict(seed=seed), **(people_kw or {})))
    if shield_on:
        shield.docks = [wp["cafe_counter_pick"][:2], wp["lab_desk_place"][:2]]
    if lang_services is not None:
        lang_services.on_result = lambda lr, kind: (shield.set_lang(lr), print(f"[lang] shield: mu set [{lr.mu.lo:.3f}, {lr.mu.hi:.3f}], IR spec status={lr.spec.status} ({kind})", flush=True))
    passthrough = SH.Passthrough()
    n_max = int(max_s / TICK) + 400
    nq, nu = m.nq, m.nu
    R = dict(qpos=np.zeros((n_max, nq), np.float32), mocap_pos=np.zeros((n_max, m.nmocap, 3), np.float32),
             mocap_quat=np.zeros((n_max, m.nmocap, 4), np.float32), ctrl=np.zeros((n_max, nu), np.float32), stand=np.zeros((n_max, 4), np.float32))
    S = {k: np.zeros(n_max) for k in ("t", "wall", "pdist", "tray_tilt", "cup_tilt", "interv", "interv_aux", "a_cup", "ir_margin", "lift_h", "beads_out", "bump_cum",
                                      "cup_lost", "ctl_ms", "infeasible", "grasped")}
    S["pose"] = np.zeros((n_max, 3)); S["v"] = np.zeros((n_max, 3)); S["u_op"] = np.zeros((n_max, 3)); S["u_safe"] = np.zeros((n_max, 3))
    S["tray_acc"] = np.zeros((n_max, 3)); S["rel"] = np.zeros((n_max, 3)); S["tray_pos"] = np.zeros((n_max, 3))
    S["phase"] = np.zeros(n_max, np.int16); S["label"] = np.zeros(n_max, np.int16); S["qp"] = np.zeros(n_max, np.int16)
    stand_id = m.body("tp_stand").id
    FS.set_stand(m, None)
    bump_prev = 0
    stand_set = False
    place_guard = None
    arm_rows = None                       # zono_rows.arm_rows.ArmZonoRows (arm_zono_rows only); kept after the autonomous phases end for the summary
    arm_rows_on = False
    done_t = None
    k = 0
    n_ticks = int(max_s / TICK)
    proprio_only = os.environ.get("BIMANUAL_PROPRIO_ONLY", "0") == "1"       # tray position for placement from hand FK, not the tray body
    for k in range(n_ticks):
        obs = sim.observe()
        obs["tray_pos"] = _tray_pos_obs(sim) if proprio_only else d.xpos[sim.tray].copy()
        obs["bump_new"] = sim.bump_contacts - bump_prev
        bump_prev = sim.bump_contacts
        cmd = op.step(obs)
        if op._pending_capture:
            op._pending_capture = None
        if op.phase == "PLACE" and op.stand is not None and not stand_set:
            stand_set = True
            st = op.stand
            FS.set_stand(m, st["xy"], st["z_under"] - 0.003, st["yaw"])
        if place_guard is None and place_backend == "zono_rows" and op.phase == "PLACE" and (op.plan_d or {}).get("guard_zonos") is not None:
            from world_model.v2.house.zono_rows.flagship_place import PlaceGuard
            place_guard = PlaceGuard(sim, op.plan_d["guard_zonos"], op.plan_d.get("guard_names"))
        if place_guard is not None and op.phase == "PLACE":              # zono_rows backend only: zonotope clearance CBF on the PLACE base twist
            cmd.twist = place_guard.filter(sim, cmd.twist, TICK)
        if arm_zono_rows and shield_on and hasattr(services, "results"):   # FL-MPC zonotope rows on the ARMS (+ held tray) during the autonomous phases
            want = op.phase in MS.AUTONOMOUS and "desk" in services.results
            if want and not arm_rows_on:
                from world_model.v2.house.zono_rows.flagship_place import obstacle_zonos
                zs_arm, nm_arm = obstacle_zonos(services.results["desk"])
                arm_rows = sim.ctl.set_zono(zs_arm, nm_arm)
                arm_rows_on = True
                print(f"[arm_zono_rows] FL-MPC rows ON in {op.phase}: {len(zs_arm)} desk zonotopes, max_rows={arm_rows.max_rows}, d_act={arm_rows.d_act}", flush=True)
            elif not want and arm_rows_on:
                sim.ctl.clear_zono()
                arm_rows_on = False
            if arm_rows_on:
                sim.ctl.zono_tray = bool(sim.grasped)
        if op.phase in MS.AUTONOMOUS:
            safe, info = passthrough.filter(sim, cmd)
            info = dict(info, label="auto")
            if shield_on and people != "truth":          # keep the tracker / observed-space map alive through the autonomous phases
                shield.perc.step(sim)
        else:
            if shield_on:
                shield.cfg.use_cup = bool(sim.grasped)
                shield.cfg.use_ir_veto = bool(sim.grasped)
            safe, info = shield.filter(sim, cmd)
        t0 = time.perf_counter()
        o = sim.tick(safe)
        ctl_ms = (time.perf_counter() - t0) * 1000 / 4
        # ---- record
        R["qpos"][k] = d.qpos; R["mocap_pos"][k] = d.mocap_pos; R["mocap_quat"][k] = d.mocap_quat; R["ctrl"][k] = d.ctrl
        sb = m.body_pos[stand_id]
        R["stand"][k] = (sb[0], sb[1], sb[2], 1.0 if sb[2] > -1.0 else 0.0)
        S["t"][k] = o["t"]; S["wall"][k] = o["wall"]; S["pdist"][k] = o["pdist"]; S["tray_tilt"][k] = o["tray_tilt"]; S["cup_tilt"][k] = o["cup_tilt"]
        S["interv"][k] = info["intervention_norm"]; S["interv_aux"][k] = info.get("intervention_aux", 0.0); S["a_cup"][k] = info.get("a_cup", np.nan)
        S["ir_margin"][k] = info.get("ir_margin", np.nan); S["lift_h"][k] = o["lift_h"]; S["beads_out"][k] = o["beads_out"]; S["bump_cum"][k] = o["bump_contacts"]
        S["cup_lost"][k] = 1.0 if sim.cup_lost_t is not None else 0.0; S["ctl_ms"][k] = ctl_ms; S["infeasible"][k] = float(o["infeasible"]); S["grasped"][k] = float(sim.grasped)
        S["pose"][k] = o["pose"]; S["v"][k] = o["v"]; S["u_op"][k] = cmd.twist; S["u_safe"][k] = safe.twist; S["tray_acc"][k] = o["tray_acc"]
        S["rel"][k] = o["rel"]; S["tray_pos"][k] = d.xpos[sim.tray]
        S["phase"][k] = op.phase_id; S["label"][k] = label_id(info.get("label")); S["qp"][k] = {"all": 0, "no_jerk": 1, "no_cup": 2, "brake": 3, "off": 4}.get(info.get("qp"), 4)
        if verbose and k % quiet_every == 0:
            print(f"[{side}] t={o['t']:6.1f} ph={op.phase:8s} pose=({o['pose'][0]:.2f},{o['pose'][1]:.2f},{math.degrees(o['pose'][2]):.0f}) v={np.round(o['v'], 2)} "
                  f"beads={o['beads_out']} wall={o['wall']:.2f} pd={o['pdist']:.2f} lift={o['lift_h']:.3f} lab={info.get('label')} wall_s={time.time() - t_wall0:.0f}", flush=True)
        if op.done and done_t is None:
            done_t = o["t"]
        if done_t is not None and o["t"] - done_t >= tail_s:
            break
    n = k + 1
    R = {a: v[:n] for a, v in R.items()}
    S = {a: v[:n] for a, v in S.items()}
    meta = dict(side=side, seed=seed, n_ticks=n, tick=TICK, start=start, perception=perception if shield_on else "off", place_backend=place_backend,
                model_hash=sim.rig.info.get("model_hash"), nq=nq, nmocap=m.nmocap, mocap_names=[m.body(int(np.flatnonzero(m.body_mocapid == i)[0])).name for i in range(m.nmocap)],
                labels=LABELS, phases=MS.PHASES, events=op.events, wall_s=time.time() - t_wall0, done=bool(op.done), done_t=done_t, place_info=op.place_info,
                walker=walker_used)
    np.savez_compressed(out / f"run_{tag}.npz", **{f"s_{a}": v for a, v in S.items()}, **R)
    summ = summarize(S, meta, sim)
    summ["flmpc_impl"] = os.environ.get("BIMANUAL_FLMPC_IMPL")
    if place_guard is not None:
        summ["place_guard"] = place_guard.summary()
    if arm_zono_rows:
        summ["arm_zono_rows"] = arm_rows.summary() if arm_rows is not None else dict(n_calls=0, note="never armed (no desk perception result / autonomous phase not reached)")
    try:
        summ["flmpc_qp"] = sim.ctl.qp_stats()
    except Exception as e:
        summ["flmpc_qp"] = dict(error=str(e))
    if lang_services is not None:
        summ["lang"] = {k: dict(lr.hud(), json=f"lang_{k}_lang.json") for k, lr in lang_services.lang.items()}
    if shield_on and perception != "off" and hasattr(services, "results"):
        summ["perception"] = {kk: dict(t_sim=r["t_sim"], n_zonotopes=len(r["zonos"]), gt_recovered=r["gt_recovered"], capture_wall_s=r.get("capture_wall_s"),
                                       timing=r["timing"], approach=r.get("approach"),
                                       plan={a: b for a, b in (r.get("plan") or {}).items() if a not in ("map", "path_t", "naive_path_t")}) for kk, r in services.results.items()}
    (out / f"run_{tag}.json").write_text(json.dumps(dict(meta=meta, metrics=summ), indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    if verbose:
        print(json.dumps(summ, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))[:3000], flush=True)
    return meta, summ


def episodes(mask, gap=15):
    """number of contiguous True runs in a bool array (runs separated by < gap Falses are merged)."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return 0
    return int(1 + (np.diff(idx) > gap).sum())


def summarize(S, meta, sim):
    t = S["t"]
    auto = np.isin(S["phase"], [MS.PHASE_ID[p] for p in MS.AUTONOMOUS + ("DONE",)])
    wall = np.where(auto, 9.0, S["wall"])       # footprint-vs-occupancy clearance is meaningless while the tray is placed over the desk (autonomous phases)
    acc = np.hypot(S["tray_acc"][:, 0], S["tray_acc"][:, 1])
    held = S["grasped"] > 0
    pd = S["pdist"]
    iv = S["interv"]
    on = iv > 0.02
    ph = S["phase"]
    P = MS.PHASE_ID
    bead_t = [float(x) for x in sim.bead_out_t[sim.bead_out]]
    first = lambda mask: (float(t[np.argmax(mask)]) if mask.any() else None)
    s = dict(
        side=meta["side"], sim_s=float(t[-1]), done=meta["done"], mission_time_s=meta["done_t"], wall_s=meta["wall_s"],
        beads_spilled=int(S["beads_out"][-1]), n_beads=int(sim.rig.n_beads), bead_out_times_s=bead_t, first_spill_t=first(S["beads_out"] > 0),
        cup_lost=bool(S["cup_lost"][-1] > 0), cup_lost_t=sim.cup_lost_t,
        person_contact_ticks=int((pd <= 0).sum()), person_contact_episodes=episodes(pd <= 0), first_person_contact_t=first(pd <= 0),
        min_person_clearance_m=float(pd.min()), person_within_0p3m_ticks=int((pd < 0.3).sum()),
        wall_contact_ticks=int((wall <= 0).sum()), wall_contact_episodes=episodes(wall <= 0), first_wall_contact_t=first(wall <= 0),
        min_wall_clearance_m=float(wall.min()),
        physical_house_contacts=int(S["bump_cum"][-1]), physical_collision_episodes=episodes(np.diff(np.r_[0.0, S["bump_cum"]]) > 0),
        peak_tray_accel_ms2_held=float(acc[held].max()) if held.any() else None,
        p99_tray_accel_ms2_held=float(np.percentile(acc[held], 99)) if held.any() else None,
        max_tray_tilt_deg_held=float(S["tray_tilt"][held].max()) if held.any() else None,
        intervention_fraction_gt2pct=float(on.mean()), intervention_mean_norm=float(iv.mean()), intervention_episodes=episodes(on, gap=25),
        intervention_fraction_when_held=float(on[held].mean()) if held.any() else None,
        intervention_labels_ticks={LABELS[i]: int((on & (S["label"] == i)).sum()) for i in range(len(LABELS)) if (on & (S["label"] == i)).any()},
        flmpc_infeasible_steps=int(S["infeasible"].sum()), ctl_ms_median=float(np.median(S["ctl_ms"])),
        placed=None,
    )
    # phase timings
    ev = meta["events"]
    s["phase_times_s"] = {a: float(b) for b, a in ev}
    # placement outcome: tray resting on the stand? (tray z within 6 mm of the stand top + tray xy within 6 cm)
    try:
        tray_p = sim.d.xpos[sim.tray]
        stand_p = sim.d.xpos[sim.m.body("tp_stand").id]
        s["tray_final_xyz"] = tray_p.tolist()
        s["stand_xyz"] = stand_p.tolist()
        s["placed_on_stand"] = bool(stand_p[2] > -1 and np.hypot(*(tray_p[:2] - stand_p[:2])) < 0.06 and abs(tray_p[2] - 0.006 - stand_p[2]) < 0.01)
        s["cup_on_tray_final"] = bool(np.linalg.norm(sim.cup_rel()[0][:2]) < 0.12)
    except Exception as e:
        s["placed_err"] = str(e)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["raw", "shield"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-s", type=float, default=330.0)
    ap.add_argument("--perception", choices=["off", "live", "cached"], default="off")
    ap.add_argument("--place-backend", choices=["zed", "zono_rows"], default="zed")
    ap.add_argument("--start", choices=["full", "counter", "desk"], default="full")
    ap.add_argument("--walker", action="store_true", help="add the optional cafeteria walker (pauses freeze in teleop.sim.set_people: stalls the shield side)")
    ap.add_argument("--no-walker", action="store_true", help="(default, kept for compatibility)")
    ap.add_argument("--arm-zono-rows", action="store_true", help="FL-MPC zonotope HOCBF rows on both arms + the held tray during PLACE / RELEASE (shield side with a perception result; "
                                                                  "zono_rows.arm_rows via ArmCtl.set_zono).  Default off = bit-identical")
    ap.add_argument("--people", choices=["truth", "tracker"], default="truth", help="people source of the shield side (tracker = head-camera KF tracker; the truth stays the metric)")
    ap.add_argument("--people-look", action="store_true", help="--people tracker: look-before-move glance")
    ap.add_argument("--people-conformal", default=None, help="--people tracker: conformal margin json (people_track.calib)")
    ap.add_argument("--lang", default=None, metavar="INSTRUCTION", help="language layer: a natural-language instruction; at the SENSE phases a VLM grounds the perceived zonotopes, authors the IR spec "
                                                                          "and a mu prior (shield spec + mu set + per-object zonotope clearances).  Default off = bit-identical")
    ap.add_argument("--lang-backend", choices=["auto", "live", "mock"], default="auto", help="auto = live VLM daemon if up, else mock (logged)")
    ap.add_argument("--lang-image", choices=["main", "twin"], default="main", help="image shown to the VLM: main = flat-colour render of the main sim through the perception camera's exact pose; twin = the speckled stereo twin's left image")
    ap.add_argument("--lang-cache", default=None, help="VLM response cache dir (default <out>/lang_cache): replays are offline and reproducible")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    run_side(a.side == "shield", a.seed, a.max_s, a.perception, a.out, a.tag, a.start, a.place_backend, cafe_walker=a.walker and not a.no_walker, arm_zono_rows=a.arm_zono_rows,
             people=a.people, people_kw=dict(look=a.people_look, conformal=a.people_conformal),
             lang=(dict(instruction=a.lang, backend=a.lang_backend, cache=a.lang_cache, image=a.lang_image) if a.lang else None))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
