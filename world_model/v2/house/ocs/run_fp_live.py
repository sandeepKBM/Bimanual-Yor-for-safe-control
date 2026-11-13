"""ocs/run_fp_live.py -- ONE live h5 carry episode (shield ON, as ocs/record.record_carry) with the LIVE FoundationPose cup tracker producing the WorldState online.  SIMULATION ONLY.

    export PYTHONPATH=<repo>:<repo>/experiments/ab_smooth_vs_bangbang:<repo>/experiments/realtime
    CUDA_VISIBLE_DEVICES=5 MUJOCO_GL=egl python -m world_model.v2.house.ocs.run_fp_live --seed 3 --max-s 20 --mode fp        # live FP (head ZED render + mask + FoundationPose in a worker thread)
    CUDA_VISIBLE_DEVICES=5 MUJOCO_GL=egl python -m world_model.v2.house.ocs.run_fp_live --seed 3 --max-s 20 --mode nofp      # baseline: same loop, blob-only anchored tracker (loop-rate reference)

The control loop is the recording's (operator -> Shield.filter(truth people) -> TeleopSim.tick = 4 FL-MPC substeps (200 Hz) + one base step (50 Hz)).  Each tick the perception
pipeline (BaseProducer, TrayFKProducer, anchored cup tracker with the LiveFPSource first / wrist blob as fallback; no people producer) builds the WorldState from a LiveView.
The shield / controller do NOT consume this state (logged only).  CLOCK: lock-step -- the sim runs ~5-10x slower than real time, so FP latency is simulated in sim time from the
worker's measured WALL latency (see fp_live.py); the worker really runs concurrently with the sim thread.  Truth is used only (a) by the renderer (what the camera sees) and (b) here, after the
fact, to score.  Outputs <out>/<mode>_s<seed>/{ticks.csv, ticks.npz, events.json, summary.json}.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("BIMANUAL_TELEOP_FAST", "1")

import numpy as np

OUT = Path("data/molmospaces/ocs/fp_live")
TICK = 0.02


class CachedLiveView:
    """LiveView whose noisy encoders are drawn ONCE per tick (every producer + the FP snapshot read the same encoder sample, as in the recording)."""

    def __new__(cls, *a, **k):
        from world_model.v2.house.ocs.views import LiveView

        class _V(LiveView):
            def encoders(self_):
                if getattr(self_, "_enc", None) is None:
                    self_._enc = LiveView.encoders(self_)
                return self_._enc
        return _V(*a, **k)


def pct(v, q):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return float(np.percentile(v, q)) if len(v) else None


def run(seed=3, max_s=20.0, tail_s=1.0, mode="fp", gpu=5, out=OUT, clock="lockstep", lat_scale=1.0, verbose=True, prewarm=False, holdoff=0.5):
    from world_model.v2.house.teleop.sim import TeleopSim
    from world_model.v2.house.teleop import operator as OP, shield as SH
    from world_model.v2.house.teleop.arm import ENC_Q
    from world_model.v2.house.loc.estimator import PoseEstimator
    from world_model.v2.house.ocs import producers as P, fp_live as FL, record as REC
    from world_model.v2.house.ocs.views import fk_for_sim
    t_wall0 = time.time()
    outd = Path(out) / f"{mode}{'_prewarm' if prewarm else ''}{'_holdoff' if (mode == 'fp' and holdoff) else ''}_s{seed}"
    outd.mkdir(parents=True, exist_ok=True)

    src = None
    if mode == "fp":
        be = FL.HeadFPBackend(seed, gpu=gpu)
        src = FL.LiveFPSource(be, rate_hz=10.0, clock=clock, lat_scale=lat_scale, blob_holdoff_s=holdoff or None)
        src.start()                                                  # FP server warm + render ready BEFORE the control loop starts (builds in the worker thread)
        print(f"[fp_live] FP worker READY {time.time() - t_wall0:.0f}s", flush=True)

    sim = TeleopSim("h5", grasped=True, seed=seed, with_people=True, people_clock="time")      # = record.record_carry
    sc = sim.scene
    wp = sc.waypoints
    op = OP.SloppyOperator(seed, "carry", sc.out_route, pick_pose=wp["cafe_counter_pick"], place_pose=wp["lab_desk_place"], params=None,
                           back_route=sc.back_route, back_mode=sc.back_mode)
    shield = SH.Shield(None, occ=sim.occ_ctrl, a_hw=sim.a_hw)
    shield.docks = [wp["cafe_counter_pick"][:2]]
    sp = wp["lab_start"]
    loc = PoseEstimator(None, seed=seed, true_pose0=(sp[0], sp[1], sp[2]))
    sensors = REC.LiveSensors(sim, seed)                             # wrist blob fallback (the head frame / people path is not used: no people producer)
    fk = fk_for_sim(sim)
    enc_rng = np.random.default_rng(70_000 + seed)
    pipe = P.perception_pipeline(None, people=False, names=False, cup_kind="sim", cup_tracker="anchored_fp_live" if mode == "fp" else "anchored", fp_live=src)

    if src is not None and prewarm:                                  # register on the standstill frame BEFORE the loop (deployment: acquire the cup while stopped)
        t_pw = time.time()
        r0 = src.prewarm(CachedLiveView(sim, sensors, loc, fk, np.random.default_rng(1)))
        print(f"[fp_live] prewarm register ok={r0 and r0['ok']} mode={r0 and r0['mode']} {time.time() - t_pw:.1f}s", flush=True)
    n_max = int(max_s / TICK)
    keys = ["t_op", "t_perc", "t_shield", "t_sim", "t_loop"]
    T = {k: np.zeros(n_max) for k in keys}
    tr = dict(t=np.zeros(n_max), base=np.zeros((n_max, 3)), tray_p=np.zeros((n_max, 3)), tray_R=np.zeros((n_max, 9)), cup_p=np.zeros((n_max, 3)),
              qL=np.zeros((n_max, 6)), qR=np.zeros((n_max, 6)), lift_enc=np.zeros(n_max))
    est = dict(p=np.full((n_max, 3), np.nan), half=np.full((n_max, 3), np.nan), vis=np.zeros(n_max, bool), t_meas=np.full(n_max, np.nan), rel=np.full((n_max, 3), np.nan),
               r_eff=np.full(n_max, np.nan))
    arrived = np.full(n_max, -1, np.int64)                           # index into src.events of a measurement DELIVERED this tick (-1 none)
    done_t = None
    n = 0
    t_loop0 = time.perf_counter()
    for k in range(n_max):
        tk0 = time.perf_counter()
        obs = sim.observe()
        cmd = op.step(obs)
        tk1 = time.perf_counter()
        # ---- perception at the START of the tick (what the shield reads): live state -> WorldState
        view = CachedLiveView(sim, sensors, loc, fk, enc_rng)
        ws = pipe.step(view)
        enc = view.encoders()
        tk2 = time.perf_counter()
        d = sim.d
        Rt = d.xmat[sim.tray].reshape(3, 3)
        tr["t"][k], tr["base"][k] = sim.t, (sim.base.x, sim.base.y, sim.base.yaw)
        tr["tray_p"][k], tr["tray_R"][k], tr["cup_p"][k] = d.xpos[sim.tray], d.xmat[sim.tray], d.xpos[sim.cup]
        tr["qL"][k], tr["qR"][k], tr["lift_enc"][k] = enc["qL"], enc["qR"], enc["lift"]
        c = ws.objects.get("cup")
        if c is not None:
            est["p"][k] = c.pose.p
            est["half"][k] = np.abs(np.asarray(c.unc.pos.G)).sum(1)
            est["vis"][k], est["t_meas"][k] = c.visible, c.t_meas
            est["rel"][k] = Rt.T @ (d.xpos[sim.cup] - d.xpos[sim.tray])                       # EVAL ONLY
        safe, info = shield.filter(sim, cmd)
        tk3 = time.perf_counter()
        o = sim.tick(safe)
        loc.update(sim.t, sim.pose(), TICK)
        tk4 = time.perf_counter()
        T["t_op"][k], T["t_perc"][k], T["t_shield"][k], T["t_sim"][k], T["t_loop"][k] = tk1 - tk0, tk2 - tk1, tk3 - tk2, tk4 - tk3, tk4 - tk0
        n = k + 1
        if verbose and k % 50 == 0:
            ev = src.events if src is not None else []
            print(f"[fp_live] tick {k} t={sim.t:.2f}s wall {time.perf_counter() - t_loop0:.0f}s  sim {1e3 * T['t_sim'][max(0, k - 49):k + 1].mean():.0f} ms/tick  "
                  f"perc {1e3 * T['t_perc'][max(0, k - 49):k + 1].mean():.1f} ms/tick  FP jobs {len(ev)}  delivered {src.n_delivered if src else 0}", flush=True)
        if op.done and done_t is None:
            done_t = o["t"]
        if done_t is not None and o["t"] - done_t >= tail_s:
            break
    wall_loop = time.perf_counter() - t_loop0
    if src is not None:                                              # collect the jobs still running (after the loop; the control loop never waited)
        t_end = time.time()
        while src._inflight and time.time() - t_end < 60:
            time.sleep(0.05)
            src._drain(sim.t)
        src._drain(sim.t)
        src.stop()
    return dict(n=n, T={k: v[:n] for k, v in T.items()}, tr={k: v[:n] for k, v in tr.items()}, est={k: v[:n] for k, v in est.items()}, src=src, seed=seed, mode=mode,
                wall_loop=wall_loop, wall_total=time.time() - t_wall0, outd=outd, fk=fk, max_s=max_s, clock=clock, sim_t=float(sim.t))


def score(R):
    """truth-based evaluation (EVAL ONLY) + per-event / per-tick tables + summary.  Returns summary dict; writes the output files."""
    n, T, tr, est, src, seed, outd, fk = R["n"], R["T"], R["tr"], R["est"], R["src"], R["seed"], R["outd"], R["fk"]
    events = []
    delivered_at = {}
    if src is not None:
        for e in sorted(src.events, key=lambda e: e["t_cap"]):
            k = int(e["k"])
            ev = dict(k=k, t_cap=float(e["t_cap"]), recv_t=float(e.get("recv_t", np.nan)), latency_sim=float(e["latency_sim"]), latency_wall=float(e["latency_wall"]),
                      mask_px=int(e["mask_px"]), mode=int(e["mode"]), ok=bool(e["ok"]), delivered=bool(e.get("delivered", False)), superseded=bool(e.get("superseded", False)), expired=bool(e.get("expired", False)),
                      late=bool(e.get("late", False)), render_ms=float(e.get("render_ms", np.nan)), mask_ms=float(e.get("mask_ms", np.nan)), fp_ms=float(e.get("fp_ms", np.nan)),
                      fp_call_ms=float(e.get("ms", np.nan)), r_eff=float(e.get("r_eff", np.nan)), queue_ms=1e3 * (e["wall_start"] - e["wall_submit"]),
                      error=e.get("error"), err_true_mm=None, err_fk_mm=None, tilt_deg=None)
            if e["ok"] and k < n:
                Rt = tr["tray_R"][k].reshape(3, 3)
                p_true = Rt.T @ (tr["cup_p"][k] - tr["tray_p"][k])
                ev["err_true_mm"] = float(1e3 * np.linalg.norm(np.asarray(e["p"]) - p_true))
                c, s = math.cos(tr["base"][k, 2]), math.sin(tr["base"][k, 2])
                Rb = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
                p_base = Rb.T @ (tr["cup_p"][k] - np.array([tr["base"][k, 0], tr["base"][k, 1], 0.0]))
                pt, Rf = fk.tray(tr["qL"][k], tr["qR"][k], float(tr["lift_enc"][k]))
                ev["err_fk_mm"] = float(1e3 * np.linalg.norm(np.asarray(e["p"]) - Rf.T @ (p_base - pt)))
            events.append(ev)
            if ev["delivered"]:
                delivered_at[int(round(ev["recv_t"] / TICK))] = ev
    # ---- per-tick table
    ecup = np.linalg.norm((est["p"] - est["rel"])[:, :2], axis=1) * 1e3          # estimator (xy) vs true cup-in-tray xy at the tick, mm
    contained = (np.abs((est["p"] - est["rel"])[:, :2]) <= est["half"][:, :2] + 1e-9).all(1)
    rows = []
    for k in range(n):
        ev = delivered_at.get(k)
        rows.append(dict(k=k, t=round(float(tr["t"][k]), 4), arrived=int(ev is not None), capture_t=None if ev is None else round(ev["t_cap"], 4),
                         recv_t=None if ev is None else round(ev["recv_t"], 4), latency_s=None if ev is None else round(ev["recv_t"] - ev["t_cap"], 4),
                         mask_px=None if ev is None else ev["mask_px"], fp_mode=None if ev is None else ev["mode"],
                         fp_err_fk_mm=None if ev is None or ev["err_fk_mm"] is None else round(ev["err_fk_mm"], 3),
                         fp_err_true_mm=None if ev is None or ev["err_true_mm"] is None else round(ev["err_true_mm"], 3),
                         est_err_xy_mm=None if not np.isfinite(ecup[k]) else round(float(ecup[k]), 3), set_contains_truth=int(contained[k]) if np.isfinite(ecup[k]) else None,
                         cup_visible=int(est["vis"][k]), t_meas=None if not np.isfinite(est["t_meas"][k]) else round(float(est["t_meas"][k]), 4),
                         set_half_x_mm=None if not np.isfinite(est["half"][k, 0]) else round(1e3 * float(est["half"][k, 0]), 2),
                         wall_ms_op=round(1e3 * T["t_op"][k], 2), wall_ms_perc=round(1e3 * T["t_perc"][k], 2), wall_ms_shield=round(1e3 * T["t_shield"][k], 2),
                         wall_ms_sim=round(1e3 * T["t_sim"][k], 2), wall_ms_loop=round(1e3 * T["t_loop"][k], 2)))
    with open(outd / "ticks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.savez(outd / "ticks.npz", **{f"T_{k}": v for k, v in T.items()}, **{f"tr_{k}": v for k, v in tr.items()}, **{f"est_{k}": v for k, v in est.items()}, est_err_xy_mm=ecup)
    (outd / "events.json").write_text(json.dumps(events, indent=1))
    # ---- summary
    dlv = [e for e in events if e["delivered"]]
    ok_ev = [e for e in events if e["ok"]]
    n_due = src.n_due if src is not None else 0
    t_loop = T["t_loop"]
    wall = lambda a: dict(p50_ms=1e3 * pct(a, 50), p95_ms=1e3 * pct(a, 95), mean_ms=1e3 * float(np.mean(a)), max_ms=1e3 * float(np.max(a)))
    due = np.array([math.floor((k + 1) * 10.0 * TICK) > math.floor(k * 10.0 * TICK) for k in range(n)])
    S = dict(seed=R["seed"], mode=R["mode"], clock=R["clock"], n_ticks=n, sim_s=R["sim_t"], wall_loop_s=R["wall_loop"], wall_total_s=R["wall_total"], realtime_factor=R["wall_loop"] / max(R["sim_t"], 1e-9),
             tick_wall=dict(loop=wall(t_loop), sim_tick=wall(T["t_sim"]), shield=wall(T["t_shield"]), op=wall(T["t_op"]), perception=wall(T["t_perc"]), perception_due_ticks=wall(T["t_perc"][due]), perception_other_ticks=wall(T["t_perc"][~due]),
                           loop_due_ticks=wall(t_loop[due]), loop_other_ticks=wall(t_loop[~due])),
             achieved_rates_wall_hz=dict(base_loop_p50=1.0 / pct(t_loop, 50), base_loop_mean=n / float(t_loop.sum()), arm_p50=4.0 / pct(T["t_sim"], 50), arm_mean=4.0 * n / float(T["t_sim"].sum()),
                                         budget_base_hz=50.0, budget_arm_hz=200.0),
             frac_ticks_loop_within_20ms=float((t_loop <= TICK).mean()), frac_ticks_perception_within_2ms=float((T["t_perc"] <= 0.002).mean()),
             est_cup_err_xy_mm=dict(p50=pct(ecup, 50), p95=pct(ecup, 95)), set_contains_truth_frac=float(contained[np.isfinite(ecup)].mean()) if np.isfinite(ecup).any() else None,
             cup_visible_frac=float(est["vis"].mean()))
    if src is not None:
        S["fp"] = dict(n_due=n_due, n_submitted=src.n_submitted, n_skipped_busy=src.n_skipped, n_jobs=len(events), n_ok=len(ok_ev), n_delivered=len(dlv), n_superseded=src.n_superseded, n_late=src.n_late, n_expired=src.n_expired,
                       coverage_of_due=len(dlv) / max(n_due, 1), coverage_of_submitted=len(ok_ev) / max(src.n_submitted, 1),
                       submit_ms=dict(p50=pct(src.submit_ms, 50), max=max(src.submit_ms or [0.0])),
                       err_fk_mm=dict(p50=pct([e["err_fk_mm"] for e in ok_ev], 50), p95=pct([e["err_fk_mm"] for e in ok_ev], 95)),
                       err_true_mm=dict(p50=pct([e["err_true_mm"] for e in ok_ev], 50), p95=pct([e["err_true_mm"] for e in ok_ev], 95)),
                       latency_sim_s=dict(p50=pct([e["recv_t"] - e["t_cap"] for e in dlv], 50), p95=pct([e["recv_t"] - e["t_cap"] for e in dlv], 95)),
                       latency_wall_s=dict(p50=pct([e["latency_wall"] for e in events], 50), p95=pct([e["latency_wall"] for e in events], 95)),
                       stage_ms_p50=dict(render=pct([e["render_ms"] for e in events], 50), mask=pct([e["mask_ms"] for e in events], 50), fp=pct([e["fp_ms"] for e in events], 50),
                                         queue=pct([e["queue_ms"] for e in events], 50)),
                       stage_ms_p95=dict(render=pct([e["render_ms"] for e in events], 95), mask=pct([e["mask_ms"] for e in events], 95), fp=pct([e["fp_ms"] for e in events], 95)),
                       fp_call_ms=dict(p50=pct([e["fp_call_ms"] for e in events if e["mode"] > 0], 50), p95=pct([e["fp_call_ms"] for e in events if e["mode"] > 0], 95)),
                       mask_px=dict(p50=pct([e["mask_px"] for e in events], 50), min=min([e["mask_px"] for e in events] or [0])),
                       modes=dict(register=sum(e["mode"] == 1 for e in events), track=sum(e["mode"] == 2 for e in events), none=sum(e["mode"] == 0 for e in events)),
                       r_eff_mm=dict(p50=pct([1e3 * e["r_eff"] for e in dlv], 50), p95=pct([1e3 * e["r_eff"] for e in dlv], 95)),
                       errors=[e["error"] for e in events if e["error"]][:3])
    (outd / "summary.json").write_text(json.dumps(S, indent=1))
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--max-s", type=float, default=20.0)
    ap.add_argument("--mode", choices=["fp", "nofp"], default="fp")
    ap.add_argument("--gpu", type=int, default=5)
    ap.add_argument("--prewarm", action="store_true", help="register once on the standstill frame before the loop")
    ap.add_argument("--holdoff", type=float, default=0.5, help="fp: the wrist-blob fallback is suppressed for this many s after a delivered FP measurement (0 = off: blob overrides FP on its own tick grid)")
    ap.add_argument("--clock", choices=["lockstep", "wall"], default="lockstep")
    ap.add_argument("--lat-scale", type=float, default=1.0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    assert a.gpu == 5, "this job is restricted to GPU 5"
    R = run(a.seed, a.max_s, 1.0, a.mode, a.gpu, a.out, a.clock, a.lat_scale, prewarm=a.prewarm, holdoff=a.holdoff)
    S = score(R)
    print(json.dumps(S, indent=1), flush=True)
    print(f"[fp_live] DONE {R['outd']}", flush=True)


if __name__ == "__main__":
    main()
