"""MEASURED base-stop card (USER 11-16: "go with a and build the base stop check"; plan step C).  SIMULATION ONLY, offline, CPU, no VLM, no hardware.

What it measures.  The run-2 bookcase plan (Codex reply 3, replayed from the run-2 cache with no network) runs in world i with the base backup ON
(BIMANUAL_BASE_BACKUP=1: base segment backup = [base_stop, hold]).  executor.run_segments already runs the stored backup from EVERY branch point of the
base segment (every 0.25 s, counterfactual, from the branch snapshot); a logging backup_runner records, per stop:
  p0, v_cmd (sb._v: the rate-limited body twist command at the branch), level caps (v, a, w, alpha) of the segment's level,
  model stop distance  d_lin = |v_xy|^2 / (2 a),  d_ang = w^2 / (2 alpha)   (the driver.abort rate limit at the level's a / alpha)
  actual displacement from p0 to the landed pose (base_stop_log: T_s, at_rest), extra = actual - model (positive = overshoot beyond the model)
  settle = T_s - max(|v|/a, |w|/alpha)
  achieved peak |a_xy| and |alpha| during the stop (second differences of the base pose over WIN steps; a measurement, noisy)
  U kinds after the backup (viol_all), in_R.
  drift (11-17): during each stop (every control step of stop + hold) the SIDEWAYS displacement (perpendicular to the base heading at the branch, max |.|) and the YAW change from the branch
  heading (max |.|, recorded for line stops; a turn stop's yaw change is its intended motion, already in extra_ang): `lat_drift_m`, `yaw_drift_rad`.
Aggregate (CARD.json): n, max extra (m, rad), max settle, n not at rest, U kinds, achieved peaks, worlds, levels, src, and extra_lateral_m / extra_yaw_rad (max over n of the
drift; omitted when a stop record lacks them: the rows that need them are then MISSING).  "max over n", never a bound.
Leaves out: tracking error of the nominal route before the branch (TRACK_MARGIN_M = 0 in the sweep), worlds the admission refuses (no carry), other plans.

    PYTHONPATH=. MUJOCO_GL=osmesa python world_model/v2/nero/results/base_stop_card_2026-11-16/measure_base_stop.py --world 2 --out OUT
    ... --aggregate OUT   (after the worlds)
"""
import argparse
import glob
import json
import math
import os
import pathlib
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RUN2 = HERE.parent / "bookcase_codex_2026-11-16" / "run2"
WIN = 10                     # control steps (50 ms) per achieved-acceleration estimate (MEASUREMENT window, not a bound)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def measure(world_i, out, level):
    """11-17 (USER: fresh runs, the card is a property of the base per speed level): the run-2 bookcase plan is loaded from its saved compiled form (no
    proposer, no replay) and carried at the FORCED level (behind_job.LEVELS = (level,)).  Measurement, not admission: Rule 1 / planned-path gates are
    switched off for this script only (BIMANUAL_LEGACY_RULE1_SKIP / _NO_T4 / _NO_PATH_ROWS), since they need this very card."""
    for k in ("BIMANUAL_BASE_BACKUP", "BIMANUAL_NO_NETWORK", "BIMANUAL_LEGACY_RULE1_SKIP", "BIMANUAL_LEGACY_NO_T4", "BIMANUAL_LEGACY_NO_PATH_ROWS"):
        os.environ[k] = "1"
    os.environ.setdefault("BIMANUAL_PLANNER", "closed_chain")
    from world_model.v2.nero.contents import carry as CR, probe as P, behind_job as BJ
    from world_model.v2.stack.planners import base_route as BR
    from world_model.v2.stack.proposer import compile as PC
    levels = BR.default_backend().levels()
    task = "box_bookcase"
    method = P.PROBE_METHOD
    cal = P.load_calib(P.default_calib(method), method)["eps"]
    plan = json.loads((RUN2 / "plans" / "box_bookcase.json").read_text())["plan"]
    jp = PC.compile_plan(plan, task, level=level).job_plan("refuse")
    jp["plan_dict"] = plan
    BJ.LEVELS = (level,)
    stops = []

    def runner(ep, seg, tr, phases):
        if not phases or phases[0][0] != "base_stop":
            ep.run_phases(phases)
            return
        sb = ep.sb
        lv = (getattr(ep, "last_accel", None) or {}).get("level")
        v, a, w, al = levels[lv]
        p0 = np.asarray(sb.pose(), float)
        vc = np.asarray(sb._v, float).copy()
        poses = []
        hook = lambda e: poses.append(np.asarray(e.sb.pose(), float))
        ep.step_hooks.append(hook)
        try:
            ep.run_phases(phases)
        finally:
            ep.step_hooks.remove(hook)
        log = (getattr(ep, "base_stop_log", None) or [{}])[-1]
        p1 = np.asarray(log.get("pose", sb.pose()), float)
        v_lin, w_ang = float(np.linalg.norm(vc[:2])), abs(float(vc[2]))
        d_lin_model, d_ang_model = v_lin ** 2 / (2 * a), w_ang ** 2 / (2 * al)
        d_lin, d_ang = float(np.linalg.norm(p1[:2] - p0[:2])), abs(_wrap(float(p1[2] - p0[2])))
        lat_drift = yaw_drift = 0.0
        if poses:
            P0_ = np.vstack([p0[None, :], np.array(poses)])
            dxy = P0_[:, :2] - p0[None, :2]
            lat_drift = float(np.max(np.abs(-dxy[:, 0] * math.sin(p0[2]) + dxy[:, 1] * math.cos(p0[2]))))
            if w_ang <= 1e-9:                                   # a line stop: any yaw change is drift; a turn stop's yaw is the motion itself (extra_ang)
                yaw_drift = float(max(abs(_wrap(float(y - p0[2]))) for y in P0_[:, 2]))
        acc = alp = 0.0
        if len(poses) > 2 * WIN:
            P_ = np.array(poses)
            dt = WIN * CR.DT
            vv = (P_[WIN:] - P_[:-WIN]) / dt
            vv[:, 2] = np.array([_wrap(x * dt) / dt for x in vv[:, 2]])
            aa = (vv[WIN:] - vv[:-WIN]) / dt
            acc, alp = float(np.max(np.linalg.norm(aa[:, :2], axis=1))), float(np.max(np.abs(aa[:, 2])))
        ok, why = ep.in_R()
        stops.append(dict(world=world_i, seg=seg, t_branch=(None if tr is None else round(float(tr), 3)), level=lv, caps=dict(v=v, a=a, w=w, alpha=al),
                          v_cmd=[round(float(x), 5) for x in vc], d_lin_model=d_lin_model, d_ang_model=d_ang_model, d_lin=d_lin, d_ang=d_ang,
                          extra_lin=d_lin - d_lin_model, extra_ang=d_ang - d_ang_model, T_s=log.get("T_s"), at_rest=log.get("at_rest"),
                          settle_s=(None if log.get("T_s") is None else float(log["T_s"]) - max(v_lin / a, w_ang / al)),
                          lat_drift_m=lat_drift, yaw_drift_rad=yaw_drift, acc_peak=acc, alpha_peak=alp, U=sorted({(x.get("kind") if isinstance(x, dict) else str(x)) for x in (ep.track.get("viol_all") or [])}),
                          in_R=bool(ok), R_fail=why))

    orig = CR.CarryEpisode.__init__

    def init(self, *aa, **kw):
        orig(self, *aa, **kw)
        self.backup_runner = runner

    CR.CarryEpisode.__init__ = init
    t0 = time.time()
    r = BJ.behind_job((task, world_i, float(cal["EPS_LO"]), float(cal["EPS_FREE"]), {**jp, "probe_method": method}), method=method)
    CR.CarryEpisode.__init__ = orig
    rec = dict(world=world_i, level_forced=level, outcome=r.get("outcome"), refused_by=r.get("refused_by"), error=r.get("error"), M_true=r.get("M_true"),
               base_level=r.get("base_level"), any_U=r.get("any_U"), n_stops=len(stops), stops=stops, wall_s=round(time.time() - t0, 1))
    (out / f"world_{level}_{world_i}.json").write_text(json.dumps(rec, indent=1, default=str))
    print(f"world {world_i}: outcome {rec['outcome']} level {rec['base_level']} stops {len(stops)} {rec['wall_s']} s", flush=True)


def aggregate(out):
    recs = [json.loads(pathlib.Path(f).read_text()) for f in sorted(glob.glob(str(out / "world_L*_*.json")))]      # 11-17 fresh per-level runs only
    S = [s for r in recs for s in r["stops"]]
    if not S:
        raise SystemExit("no stops recorded")
    mx = lambda k: max(float(s[k]) for s in S if s.get(k) is not None)
    has_drift = all(s.get("lat_drift_m") is not None and s.get("yaw_drift_rad") is not None for s in S)
    card = dict(
        n=len(S), worlds=[dict(world=r["world"], level_forced=r.get("level_forced"), M_true=r["M_true"], outcome=r["outcome"], level=r["base_level"], n_stops=r["n_stops"]) for r in recs],
        extra_overshoot_m=max(0.0, mx("extra_lin")), extra_overshoot_rad=max(0.0, mx("extra_ang")), settle_s=max(0.0, mx("settle_s")),
        T_s_max=mx("T_s"), n_not_at_rest=sum(1 for s in S if not s.get("at_rest")), n_not_in_R=sum(1 for s in S if not s.get("in_R")),
        U_kinds=sorted({u for s in S for u in s["U"]}), acc_peak_max=mx("acc_peak"), alpha_peak_max=mx("alpha_peak"),
        levels=sorted({s["level"] for s in S if s.get("level")}),
        tag="MEASURED", statistic="max over n stops (not a bound)",
        src=f"{HERE / 'measure_base_stop.py'}: run-2 bookcase plan (Codex reply 3, saved compiled plan), carried at each forced level, base backup ON, every 0.25 s branch point of the base segment; card values = max over ALL levels listed (conservative for each)",
        noise_model="none (sim, deterministic per world); achieved accelerations from second differences of the base pose over 50 ms windows",
        leaves_out="nominal-route tracking error before the branch (TRACK_MARGIN_M = 0); refused worlds; other plans and levels than those listed")
    per = {}
    for lv in sorted({s["level"] for s in S if s.get("level")}):
        SL = [s for s in S if s.get("level") == lv]
        m = lambda k: max(float(s[k]) for s in SL if s.get(k) is not None)
        per[lv] = dict(n=len(SL), extra_overshoot_m=max(0.0, m("extra_lin")), extra_overshoot_rad=max(0.0, m("extra_ang")), settle_s=max(0.0, m("settle_s")),
                       T_s_max=m("T_s"), n_not_at_rest=sum(1 for s in SL if not s.get("at_rest")), acc_peak_max=m("acc_peak"), alpha_peak_max=m("alpha_peak"),
                       U_kinds=sorted({u for s in SL for u in s["U"]}))
        if has_drift:
            per[lv].update(extra_lateral_m=max(0.0, m("lat_drift_m")), extra_yaw_rad=max(0.0, m("yaw_drift_rad")))
    card["per_level"] = per                  # 11-17: per level (USER "measure per level"); the top-level values are the max over all levels (what the rows read today)
    if has_drift:
        card.update(extra_lateral_m=max(0.0, mx("lat_drift_m")), extra_yaw_rad=max(0.0, mx("yaw_drift_rad")))
    else:
        print("WARNING: some stop records lack lat_drift_m / yaw_drift_rad (measured before 11-17): the card has no extra_lateral_m / extra_yaw_rad; re-measure every world")
    (out / "CARD.json").write_text(json.dumps(card, indent=1))
    print(json.dumps({k: v for k, v in card.items() if k not in ("src", "noise_model", "leaves_out")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int)
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--level", default="L2")
    a = ap.parse_args()
    o = pathlib.Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    aggregate(o) if a.aggregate else measure(a.world, o, a.level)
