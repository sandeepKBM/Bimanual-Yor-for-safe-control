"""Does FL-MPC survive a realistic model of the real arm's actuation?
SIMULATION ONLY -- no hardware, no network, no piperlib/piper_sdk import.

Runs the UNMODIFIED scripts/make_cup_collect_safety_demo.run_episode(
controller="flmpc") with experiments/actuation/actuation_model.ActuationModel
injected WITHOUT editing any existing file:

  INJECTION: the demo applies the (filtered, TAU_MAX-clipped) arm torque by
  writing data.qfrc_applied[left/right_dofadr] and then calls
  `mujoco.mj_step(model, data)` exactly once per control step, through its own
  module-level `mujoco` name. We temporarily rebind `demo.mujoco` to a proxy
  whose mj_step first calls ActuationModel.apply (reads qfrc_applied[arm dofs]
  = commanded torque, overwrites it with what the actuator would realize),
  then the real mujoco.mj_step. Every other attribute forwards to the real
  module; no other module's mujoco reference is touched. Proof of activity:
  hook-call count == logged steps, and (for 'ideal') bit-identical metrics.

  FIX (i) "plan within the real limits": monkeypatches sim_common.TAU_MAX
  (scalar, read live by the demo's box/clip and FL-MPC) and wraps
  mpc_fl._assemble_fl / FLMPC._torque_safe_scale so FL-MPC's torque-box
  rows use PER-JOINT limits (8,8,8,6.4,6.4,6.4) -- again no file edited.

Usage:
  python experiments/actuation/run_actuation_eval.py --conds baseline,b,c,d --seeds 0
  python experiments/actuation/run_actuation_eval.py --conds all --seeds 0-49 --workers 56 \
      --out $CLAUDE_JOB_DIR/tmp/actuation_n50.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO / "experiments" / "actuation"), str(REPO / "scripts"), str(REPO),
           str(REPO / "experiments" / "ab_smooth_vs_bangbang")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from actuation_model import (ActuationConfig, ActuationModel, piper_ff_clip_limits,  # noqa: E402
                             PIPER_DEFAULT_KP, PIPER_DEFAULT_KD)

PIPER_LIM = piper_ff_clip_limits()     # [8, 8, 8, 6.4, 6.4, 6.4]
RAW8 = np.full(6, 8.0)


def make_conditions(kp_pd=None, kd_pd=None, latency=0):
    kp_pd = np.full(6, 60.0) if kp_pd is None else np.asarray(kp_pd, float)
    kd_pd = np.full(6, 2.0) if kd_pd is None else np.asarray(kd_pd, float)
    mit = dict(ff_limit=RAW8, quantize=True, latency_steps=latency)
    return {
        "baseline": None,                                                                 # no hook at all
        "ideal":    ActuationConfig(name="ideal"),                                        # hook passthrough (must == baseline)
        "b":  ActuationConfig(name="b_clip_q_gcON",  semantics="clip", overflow="clip", gravcomp=True,  **mit),
        "c":  ActuationConfig(name="c_clip_q_gcOFF", semantics="clip", overflow="clip", gravcomp=False, **mit),
        "d":  ActuationConfig(name="d_drop_q_gcOFF", semantics="drop", overflow="wrap", gravcomp=False, **mit),
        "gcoff": ActuationConfig(name="gcOFF_only", gravcomp=False),                     # isolates gravcomp removal
        # fix (i): FL-MPC plans inside the real per-joint limits (+ c's actuation)
        "fix_i": ActuationConfig(name="fix_i_plan_in_limits", semantics="clip", overflow="clip", gravcomp=False, **mit),
        # fix (ii): MIT PD+ff, ref = integrated planned accel, tau_ff = M a + c clipped/quantized
        # WARNING (seed-0 smoke): kp=60 run did not finish in 75 min (>8x slower than every other
        # condition -> PD-vs-replanning double feedback); prefer fix_ii_offset below.
        "fix_ii": ActuationConfig(name="fix_ii_pd_ff", semantics="clip", overflow="clip", gravcomp=False,
                                  kp=kp_pd, kd=kd_pd, **mit),
        "fix_ii_default_gains": ActuationConfig(name="fix_ii_pd_ff_kp10", semantics="clip", overflow="clip",
                                                gravcomp=False, kp=np.full(6, PIPER_DEFAULT_KP),
                                                kd=np.full(6, PIPER_DEFAULT_KD), **mit),
        # fix (ii-b): torque-through-position-offset (MIT fields only; see ActuationConfig.pd_ref)
        "fix_ii_offset": ActuationConfig(name="fix_ii_offset_kp100", semantics="clip", overflow="clip",
                                         gravcomp=False, kp=np.full(6, 100.0), kd=np.full(6, 1.0),
                                         pd_ref="offset", **mit),
        "fix_ii_offset_lat2": ActuationConfig(name="fix_ii_offset_kp100_lat10ms", semantics="clip",
                                              overflow="clip", gravcomp=False, kp=np.full(6, 100.0),
                                              kd=np.full(6, 1.0), pd_ref="offset",
                                              **{**mit, "latency_steps": 2}),
        "c_lat2": ActuationConfig(name="c_clip_q_gcOFF_lat10ms", semantics="clip", overflow="clip",
                                  gravcomp=False, **{**mit, "latency_steps": 2}),
        # latency attribution: pure transport delay on an otherwise ideal (gcON, unlimited) actuator
        "lat1_only": ActuationConfig(name="lat5ms_only", latency_steps=1),
        "lat2_only": ActuationConfig(name="lat10ms_only", latency_steps=2),
        "c_lat1": ActuationConfig(name="c_clip_q_gcOFF_lat5ms", semantics="clip", overflow="clip",
                                  gravcomp=False, **{**mit, "latency_steps": 1}),
        "fix_ii_offset_lat1": ActuationConfig(name="fix_ii_offset_kp100_lat5ms", semantics="clip",
                                              overflow="clip", gravcomp=False, kp=np.full(6, 100.0),
                                              kd=np.full(6, 1.0), pd_ref="offset",
                                              **{**mit, "latency_steps": 1}),
        "fix_i_ii": ActuationConfig(name="fix_i+ii", semantics="clip", overflow="clip", gravcomp=False,
                                    kp=kp_pd, kd=kd_pd, **mit),
    }


PLAN_IN_LIMITS = {"fix_i", "fix_i_ii"}


class _MjProxy:
    def __init__(self, real, hook):
        self._real, self._hook, self.calls = real, hook, 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    def mj_step(self, m, d, *a, **k):
        self.calls += 1
        self._hook(m, d)
        return self._real.mj_step(m, d, *a, **k)


def _install_plan_in_limits(sc, mpc_fl, lim):
    """FIX (i) monkeypatches; returns an undo() closure."""
    saved = (sc.TAU_MAX, mpc_fl._assemble_fl, mpc_fl.FLMPC.__dict__["_torque_safe_scale"])
    orig = mpc_fl._assemble_fl
    L = np.asarray(lim, float)

    def assemble_perjoint(*args):
        H, f, a_ineq, b_ineq = orig(*args)
        c_e, c_i, NV = args[21], args[23], int(args[25])
        base = b_ineq.shape[0] - 4 * NV
        for j in range(NV):
            b_ineq[base + 2 * j] = L[j] - c_e[j]
            b_ineq[base + 2 * j + 1] = L[j] + c_e[j]
            b_ineq[base + 2 * NV + 2 * j] = L[j] - c_i[j]
            b_ineq[base + 2 * NV + 2 * j + 1] = L[j] + c_i[j]
        return H, f, a_ineq, b_ineq

    def safe_scale(M_arm, c_arm, a0, tau_max):
        delta = M_arm @ a0
        lam = 1.0
        for j in range(delta.shape[0]):
            dj = float(delta[j])
            if dj == 0.0:
                continue
            lj = ((L[j] - c_arm[j]) / dj) if dj > 0.0 else ((-L[j] - c_arm[j]) / dj)
            lam = min(lam, lj)
        return max(0.0, min(1.0, lam)) * a0

    sc.TAU_MAX = float(L.max())
    mpc_fl._assemble_fl = assemble_perjoint
    mpc_fl.FLMPC._torque_safe_scale = staticmethod(safe_scale)

    def undo():
        sc.TAU_MAX, mpc_fl._assemble_fl = saved[0], saved[1]
        mpc_fl.FLMPC._torque_safe_scale = saved[2]
    return undo


def jerk_rms(p, dt):
    if p.shape[0] < 4:
        return 0.0
    j = np.diff(p, n=3, axis=0) / dt ** 3
    return float(np.sqrt(np.mean(np.sum(j ** 2, axis=1))))


def run_one(job):
    seed, cond = job["seed"], job["cond"]
    t0 = time.perf_counter()
    try:
        if job.get("demo_dir"):
            # frozen snapshot of the demo (git HEAD), so concurrent live edits
            # by other agents cannot move the baseline
            sys.path.insert(0, job["demo_dir"])
            if str(REPO / "eval") not in sys.path:
                sys.path.insert(1, str(REPO / "eval"))
        import make_cup_collect_safety_demo as demo
        if job.get("demo_dir"):
            assert str(Path(demo.__file__).resolve()).startswith(job["demo_dir"]), demo.__file__
        import sim_common as sc
        import mpc_fl
        conds = make_conditions(job.get("kp"), job.get("kd"), job.get("latency", 0))
        cfg = conds[cond]
        # SAME episode config as comprehensive_eval.run_one's "flmpc" variant
        # (reset every call: spawn workers are reused).
        demo.N_CUBES = job.get("n_cubes") or 6   # 0/None -> demo default (6, IK-filtered)
        demo.V_EE_MAX_MS = 0.28
        demo.FL_W_JERK_OVERRIDE = None
        demo.FL_W_POS_INT_OVERRIDE = None
        sc.D_MIN = 0.03
        sc.ALPHA1 = sc.ALPHA2 = 30.0
        sc.COMBINED_RADIUS = sc.R_EGO + sc.R_INT + sc.D_MIN
        # --ctrlfixes F5 (delay compensation) / --ctrlfix-parts: only passed when requested,
        # so the default call is exactly the pre-existing one (library defaults).
        cf_kw = {}
        if job.get("delay_comp") or job.get("ctrlfix_parts"):
            parts = job.get("ctrlfix_parts") or "delaycomp"
            demo.CTRLFIX_PARTS = demo.parse_ctrlfix_parts(parts)
            # compensate the condition's OWN transport delay (e.g. lat1_only -> 1 step)
            demo.CTRLFIX_DELAY_STEPS = (int(cfg.latency_steps) if (job.get("delay_comp") and cfg is not None)
                                        else 0)
            demo.CTRLFIX_TAU_MAX_VEC = (None if job.get("ctrlfix_tau_max") is None
                                        else [float(x) for x in job["ctrlfix_tau_max"]])
            cf_kw = dict(ctrlfixes=True)
        if job.get("bugfixes") is not None:
            cf_kw["bugfixes"] = bool(job["bugfixes"])
        info = demo.build_episode(seed=seed, scenario="crossing", ee_speed_limit=True,
                                  cup_jitter_m=job.get("cup_jitter", 0.04), n_cubes=job.get("n_cubes") or None,
                                  **cf_kw)
        model, data = info["model"], info["data"]
        act = None
        undo = None
        real_mj = demo.mujoco
        proxy = None
        if cfg is not None:
            act = ActuationModel(model, cfg, [(info["left_qposadr"], info["left_dofadr"]),
                                              (info["right_qposadr"], info["right_dofadr"])])
            act.install(model)
            proxy = _MjProxy(real_mj, lambda m, d: act.apply(m, d, sc.DT))
            demo.mujoco = proxy
        if cond in PLAN_IN_LIMITS:
            undo = _install_plan_in_limits(sc, mpc_fl, PIPER_LIM)
        try:
            log, _ = demo.run_episode(info, controller="flmpc", qp_enabled=True, capture=False,
                                      ee_speed_limit=True, table_floor=True, max_steps=job.get("max_steps"))
        finally:
            demo.mujoco = real_mj
            if undo is not None:
                undo()
            if act is not None:
                act.uninstall(model)
        m = demo.summarize(log)
        dt = sc.DT
        n = len(log["t"])
        row = dict(seed=seed, cond=cond, ok=True, n_steps=n,
                   delay_comp=bool(job.get("delay_comp")), ctrlfix_parts=job.get("ctrlfix_parts"),
                   comp_delay_steps=int(demo.CTRLFIX_DELAY_STEPS) if cf_kw.get("ctrlfixes") else 0,
                   right_ee_speed_max_ms=float(m["right_ee_speed_max_ms"]),
                   placed_strict_final=int(m.get("placed_strict_final", -1)),
                   fl_relaxed_steps=int(m.get("fl_relaxed_steps", 0)),
                   fl_taumax_relaxed_steps=int(m.get("fl_taumax_relaxed_steps", 0)),
                   collision=bool(m["collision"]), h_min=float(m["h_min"]),
                   placed_final=int(m["placed_final"]), spilled_max=int(m["spilled_max"]),
                   cup_tilt_max_deg=float(np.degrees(np.max(log["left_cup_tilt_rad"]))),
                   qp_infeas_frac=float(max(m["qp_infeasible_frac_left"], m["qp_infeasible_frac_right"])),
                   jerk_rms_right_ee=jerk_rms(np.asarray(log["right_ee_pos"]), dt),
                   jerk_rms_left_ee=jerk_rms(np.asarray(log["left_ee_pos"]), dt),
                   cup_z_min=float(m["cup_z_min"]), wall_s=time.perf_counter() - t0)
        if "true_cup_tilt_rad" in log and np.any(log["true_cup_tilt_rad"]):
            row["true_cup_tilt_max_deg"] = float(np.degrees(np.max(log["true_cup_tilt_rad"])))
        if act is not None:
            row["hook_calls"] = proxy.calls
            row["injection_active"] = bool(proxy.calls == n)
            row.update(act.stats())
            cmd_l, real_l = np.array(act.cmd_hist[0]), np.array(act.real_hist[0])
            cmd_r, real_r = np.array(act.cmd_hist[1]), np.array(act.real_hist[1])
            row["max_abs_cmd_minus_real"] = float(max(np.abs(cmd_l - real_l).max(), np.abs(cmd_r - real_r).max()))
            # when do saturations happen (phase timing, per arm), first/last sat time
            lim = act._eff_limit()
            if lim is not None:
                t = np.asarray(log["t"])[:len(cmd_l)]
                for side, cmd in (("left", cmd_l), ("right", cmd_r)):
                    sat = np.abs(cmd) > lim + 1e-9
                    anyj = sat.any(1)
                    row[f"{side}_sat_t_first"] = float(t[anyj][0]) if anyj.any() else None
                    row[f"{side}_sat_frac_pre_cross"] = float(anyj[t < demo.RIGHT_START_T].mean()) if (t < demo.RIGHT_START_T).any() else 0.0
                    row[f"{side}_sat_frac_cross"] = float(anyj[t >= demo.RIGHT_START_T].mean()) if (t >= demo.RIGHT_START_T).any() else 0.0
            if job.get("dump_dir"):
                np.savez_compressed(Path(job["dump_dir"]) / f"act_{cond}_s{seed}.npz",
                                    t=np.asarray(log["t"]), cmd_l=cmd_l, real_l=real_l, cmd_r=cmd_r, real_r=real_r,
                                    left_q=np.asarray(log["left_q"]), right_q=np.asarray(log["right_q"]),
                                    h_min=np.asarray(log["h_min"]), tilt=np.asarray(log["left_cup_tilt_rad"]),
                                    collision=np.asarray(log["collision"]))
        return row
    except Exception as e:
        return dict(seed=seed, cond=cond, ok=False, err=str(e)[:300], tb=traceback.format_exc()[-1500:],
                    wall_s=time.perf_counter() - t0)


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="baseline,ideal,b,c,d")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-cubes", type=int, default=3,
                    help="3 = comprehensive_eval config; 0 = demo default (N_CUBES=6, IK-filtered -> the flagship seed-0 run)")
    ap.add_argument("--cup-jitter", type=float, default=0.04)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--latency", type=int, default=0)
    ap.add_argument("--kp", type=float, default=None)
    ap.add_argument("--kd", type=float, default=None)
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--delay-comp", choices=("off", "on"), default="off",
                    help="--ctrlfixes F5: FL-MPC forward-predicts the state over --latency steps with the "
                         "committed accelerations (only the 'delaycomp' ctrlfix part unless --ctrlfix-parts).")
    ap.add_argument("--ctrlfix-parts", default=None,
                    help="optional comma list of make_cup_collect_safety_demo ctrlfix parts to enable")
    ap.add_argument("--bugfixes", choices=("default", "on", "off"), default="default")
    ap.add_argument("--ctrlfix-tau-max", default=None,
                    help="F6: FL-MPC per-joint torque limits (comma list of 6, or 'piper' = 8,8,8,6.4,6.4,6.4); "
                         "needs --ctrlfix-parts including taumax")
    ap.add_argument("--out", default=None)
    ap.add_argument("--demo-src", choices=("live", "head"), default="live",
                    help="live = scripts/make_cup_collect_safety_demo.py as on disk; head = frozen "
                         "`git show <rev>:` snapshot under experiments/actuation/_snap/ (import-only copy)")
    ap.add_argument("--demo-rev", default="HEAD")
    a = ap.parse_args()
    demo_dir = None
    if a.demo_src == "head":
        import subprocess
        snap = Path(__file__).resolve().parent / "_snap" / a.demo_rev.replace("/", "_") / "x"
        snap.mkdir(parents=True, exist_ok=True)
        src = subprocess.run(["git", "-C", str(REPO), "show", f"{a.demo_rev}:scripts/make_cup_collect_safety_demo.py"],
                             check=True, capture_output=True, text=True).stdout
        (snap / "make_cup_collect_safety_demo.py").write_text(src)
        demo_dir = str(snap)
        print(f"[demo-src] frozen snapshot {a.demo_rev} -> {snap}", flush=True)
    conds = list(make_conditions().keys()) if a.conds == "all" else a.conds.split(",")
    jobs = [dict(seed=s, cond=c, n_cubes=a.n_cubes, cup_jitter=a.cup_jitter, max_steps=a.max_steps,
                 latency=a.latency, kp=None if a.kp is None else [a.kp] * 6,
                 kd=None if a.kd is None else [a.kd] * 6, dump_dir=a.dump_dir, demo_dir=demo_dir,
                 delay_comp=(a.delay_comp == "on"), ctrlfix_parts=a.ctrlfix_parts,
                 ctrlfix_tau_max=(None if a.ctrlfix_tau_max is None else
                                  (PIPER_LIM.tolist() if a.ctrlfix_tau_max == "piper"
                                   else [float(x) for x in a.ctrlfix_tau_max.split(",")])),
                 bugfixes=(None if a.bugfixes == "default" else a.bugfixes == "on"))
            for s in parse_seeds(a.seeds) for c in conds]
    if a.dump_dir:
        Path(a.dump_dir).mkdir(parents=True, exist_ok=True)
    fout = open(a.out, "a") if a.out else None
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=3) as pool:
            it = pool.imap_unordered(run_one, jobs)
            rows = []
            for r in it:
                rows.append(r)
                print(json.dumps(r), flush=True)
                if fout:
                    fout.write(json.dumps(r) + "\n"); fout.flush()
    else:
        rows = []
        for j in jobs:
            r = run_one(j)
            rows.append(r)
            print(json.dumps(r), flush=True)
            if fout:
                fout.write(json.dumps(r) + "\n"); fout.flush()
    # aggregate
    print("\ncond            n  coll%  h_min_mean  h_min_worst placed_mean tilt_max_deg jerkR  drops  sat_frac_max", flush=True)
    for c in conds:
        g = [r for r in rows if r.get("ok") and r["cond"] == c]
        if not g:
            continue
        sat = [max(max(r.get("left_sat_frac", [0])), max(r.get("right_sat_frac", [0]))) for r in g]
        print(f"{c:14s} {len(g):3d} {100*np.mean([r['collision'] for r in g]):5.1f} "
              f"{np.mean([r['h_min'] for r in g]):11.4f} {np.min([r['h_min'] for r in g]):11.4f} "
              f"{np.mean([r['placed_final'] for r in g]):11.2f} {np.max([r['cup_tilt_max_deg'] for r in g]):12.2f} "
              f"{np.mean([r['jerk_rms_right_ee'] for r in g]):6.1f} {np.sum([r.get('drops_total', 0) for r in g]):6d} "
              f"{np.mean(sat):8.3f}", flush=True)


if __name__ == "__main__":
    main()
