"""robust/smoke.py -- smoke tests for the perception-robustness harness. SIMULATION / SAVED DATA ONLY (CPU, <= a few cores).
Only has to RUN and be DETERMINISTIC; the numbers it prints are plumbing, not evidence.

  python -m world_model.v2.tray_rep.robust.smoke checks          # unit checks + determinism + equivalence gate vs cl_eval_mj
  python -m world_model.v2.tray_rep.robust.smoke test1           # both setups x both arms x 4 conditions, a few states each, + VAL oracle + conformal tau
  python -m world_model.v2.tray_rep.robust.smoke test2           # 5 filters + F0, both setups, a few carries; one tiny tune cell
  python -m world_model.v2.tray_rep.robust.smoke test3           # latency sweep, budget (few states), reports
  python -m world_model.v2.tray_rep.robust.smoke all
Output: <OUT>/smoke/  (jsonl, summaries, report tables and plots, smoke_summary.json)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K          # noqa: F401
from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep.store import Store
from world_model.v2.tray_rep.robust import corrupt as CR
from world_model.v2.tray_rep.robust import util as RU

SM = RU.OUT / "smoke"
SUMMARY = SM / "smoke_summary.json"


def _save(key, val):
    d = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    d[key] = val
    RU.atomic_json(d, SUMMARY)


def _same(a, b, skip=("wall_s", "filt_s", "sim_s")):
    ka = {k: v for k, v in a.items() if k not in skip}
    kb = {k: v for k, v in b.items() if k not in skip}
    return ka == kb


# ===================================================================================== checks
def checks(workers=1):
    from world_model.v2.tray_rep.robust import test1 as T1, run2, tune
    from world_model.v2.tray_rep.robust import rfilters as RF
    res = {}
    # 1. Corruptor == env.Tracker (same stream), exact pass-through == stored obs, determinism
    rng = np.random.default_rng(1)
    xy = np.cumsum(rng.normal(0, 0.002, (120, 2)), 0)
    tr = E.Tracker(0.003, 3, 30.0, 0.02); tr.reset(xy[0], seed=5); a = [tr.out()]
    for k in range(1, 120):
        a.append(tr.update(xy[k]))
    def stream(cond, burst=None):
        cr = CR.Corruptor(CR.parse_cond(cond), 5, burst); inf = [cr.reset(xy[0], (0, 0))]
        for k in range(1, 120):
            inf.append(cr.update(xy[k], (0, 0)))
        return inf
    s1 = stream("trk")
    res["corruptor_equals_env_tracker"] = bool(np.abs(np.array(a) - np.array([i["ball"] for i in s1])).max() == 0.0)
    res["corruptor_deterministic"] = bool(all(np.array_equal(i["ball"], j["ball"]) for i, j in zip(s1, stream("trk"))))
    sb = stream("trkB10f", (40, 10))
    ph = [i["phase"] for i in sb]
    res["burst_phases_seen"] = sorted(set(ph))
    res["burst_freezes_ball"] = bool(np.array_equal(sb[44]["ball"], sb[48]["ball"]))
    st = Store(RU.data_dir("piper"), "test")
    ep = 3
    L = st.ep_info(ep)["lens"]
    priv, obs = st.rows("priv", ep, 0, L), st.rows("obs", ep, 0, L)
    ball, _, _ = CR.corrupt_prefix(CR.parse_cond("off"), 0, None, priv)
    res["off_equals_stored_exact_obs"] = bool(np.abs(ball - obs[:, 9:13]).max() < 1e-6)
    # 2. tracker set grows during dropout, support shape, centre
    cond = CR.parse_cond("trkB10f")
    ts = CR.TrackerSet(cond)
    for i in sb:
        ts.step(i)
    sig = [ts.sigma_now(a_)[0] for a_ in (0, 2, 5, 10, 20)]
    res["tracker_set_sigma_by_age_0_2_5_10_20"] = [round(x, 4) for x in sig]
    res["tracker_set_grows"] = bool(np.all(np.diff(sig) > 0))
    sup = ts.support(sb[-1]["ball"], 10)
    res["support_shape_centre_ok"] = bool(sup.shape == (9, 4) and np.allclose(sup[0], sb[-1]["ball"]))
    res["scale_calibration_trkB10f"] = tune.calibrate_scale("piper", "trkB10f", n_eps=20)
    # 3. equivalence gate: Test-1 loop with cond `off` (piper models, piper data) == cl_eval_mj.evaluate(WMGreedy) on the same states
    from world_model.v2.tray_rep import cl_eval_mj as CL
    from world_model.v2.tray_rep.bridge import filters as BF
    from world_model.v2.tray_rep.wm.policy import WMGreedy
    renv = RU.make_clean_env(st.meta)
    orc = T1.load_oracle(T1.oracle_path("piper", "test"))
    wm, net, _ = BF.load_arm(RU.runs_root(), "piper", "N0", 0)
    pol = WMGreedy(wm, net, "cpu", renv.U, renv.alpha)
    n = 8
    r_cl, per = CL.evaluate(pol, RU.data_dir("piper"), orc, "test", 50, max_states=n)
    lt = __import__("world_model.v2.tray_rep.robust.lat", fromlist=["x"]).load_tracks(RU.runs_root(), "piper", "N0", [0], renv.U, renv.alpha)[0]
    sel = T1.population(orc, n, 0)
    mine = [T1.run_state(renv, st, orc, int(j), lt, CR.parse_cond("off")) for j in sel]
    f_mine = np.array([m["fail"] for m in mine])
    res["equiv_cl_eval_states"] = int(len(sel))
    res["equiv_same_states"] = bool(np.array_equal(per["sel"], sel))
    res["equiv_fail_agree"] = int((f_mine == per["fail"]).sum())
    res["equiv_dmin_maxdiff"] = float(np.abs(np.array([m["dmin"] for m in mine]) - per["dmin"]).max())
    # 4. run_state determinism
    m2 = [T1.run_state(renv, st, orc, int(j), lt, CR.parse_cond("trkB10s")) for j in sel[:3]]
    m3 = [T1.run_state(renv, st, orc, int(j), lt, CR.parse_cond("trkB10s")) for j in sel[:3]]
    res["test1_run_state_deterministic"] = bool(all(_same(a_, b_, ("wall_s",)) for a_, b_ in zip(m2, m3)))
    # 5. run2: worker count and resume do not change records (rob and aci filters, 4 carries)
    for spec in ("rob:arm=Rc3k,thr=0.1,kappa=1", "aci:arm=Rc3k,thr=0.1"):
        outs = []
        for tag, w in (("det1", 1), ("det2", 2)):
            _, recs = run2.run(spec, "piper_trk", 0, "trkB10f", "val", 4, w, tag, SM, None, False, quiet=True)
            outs.append(sorted(recs, key=lambda r: r["ep"]))
        _, again = run2.run(spec, "piper_trk", 0, "trkB10f", "val", 4, 1, "det1", SM, None, False, quiet=True)   # resume = no new work
        res[f"run2_{spec.split(':')[0]}_1worker_eq_2workers"] = bool(all(_same(a_, b_) for a_, b_ in zip(*outs)))
        res[f"run2_{spec.split(':')[0]}_resume_eq"] = bool(all(_same(a_, b_) for a_, b_ in zip(outs[0], sorted(again, key=lambda r: r["ep"]))))
    # 6. no hardware modules loaded
    bad = [m for m in sys.modules if m.startswith(("YOR.robot", "YOR.extra", "xarm", "piper_sdk", "rospy", "serial", "can"))]
    res["hardware_modules_loaded"] = bad
    res["pass"] = bool(res["corruptor_equals_env_tracker"] and res["corruptor_deterministic"] and res["off_equals_stored_exact_obs"]
                       and res["tracker_set_grows"] and res["support_shape_centre_ok"] and res["equiv_same_states"]
                       and res["test1_run_state_deterministic"] and not bad
                       and all(v for k, v in res.items() if k.startswith("run2_")))
    _save("checks", res)
    print(json.dumps(res, indent=1, default=float), flush=True)
    return res


# ===================================================================================== test1
def test1_smoke(workers=1):
    from world_model.v2.tray_rep.robust import test1 as T1, report
    t0 = time.time()
    out = {}
    for setup in RU.SETUPS:
        for arm in RU.ARMS:
            for cond in ("off", "trk", "trkB10s", "trkB10f"):
                recs = T1.run(setup, arm, 0, cond, "test", 6, 3, workers, "smoke", SM)
                out[f"{setup}/{arm}/{cond}"] = dict(n=len(recs), fail=int(sum(r["fail"] for r in recs)),
                                                    phases=np.bincount([r["phase"] for r in recs], minlength=3).tolist())
    # VAL oracle (24 candidate states) -> test1 on VAL -> conformal tau
    import subprocess
    py = "/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python"
    vo = SM / "oracle" / "val_smoke.npz"
    if not vo.exists():
        subprocess.check_call([py, "-m", "world_model.v2.tray_rep.robust.oracle_val", "--n", "24", "--tag", "val_smoke", "--out", str(SM)],
                              cwd=str(RU.OUT.parents[2] / "bimanual"), env=dict(__import__("os").environ, PYTHONPATH="/common/users/ss5772/bimanual"))
    recs = T1.run("piper_trk", "Rc3k", 0, "trk", "val", None, None, workers, "smoke", SM, oracle=str(vo))
    tau, n_safe, r = T1.conformal_tau(recs, 0.2)
    out["conformal_tau_eps0.2_on_VAL_smoke"] = dict(n=len(recs), tau=tau, n_safe=n_safe, emp_rate=r)
    report.report1("smoke", "test", SM)
    out["wall_s"] = time.time() - t0
    _save("test1", out)
    print(json.dumps(out, indent=1, default=float), flush=True)
    return out


# ===================================================================================== test2
def test2_smoke(workers=1):
    from world_model.v2.tray_rep.robust import run2, tune, report
    from world_model.v2.tray_rep.robust import rfilters as RF
    t0 = time.time()
    out = {}
    for setup in RU.SETUPS:
        tau = RU.chosen_tau(setup, "Rc3k")
        specs = [("f0", RF.spec_str("plain", "Rc3k", -1e9)), ("plain", RF.spec_str("plain", "Rc3k", tau)),
                 ("conf", RF.spec_str("conf", "Rc3k", 0.0)), ("epi", RF.spec_str("epi", "Rc3k", -0.01)),
                 ("aci", RF.spec_str("aci", "Rc3k", tau, alpha=0.1)), ("rob", RF.spec_str("rob", "Rc3k", tau, scale=0.4))]
        for cond in ("trk", "trkB10f"):
            for kind, sp in specs:
                s, _ = run2.run(sp, setup, 0, cond, "test", 6, workers, "smoke", SM, None, False, quiet=True)
                out[f"{setup}/{cond}/{kind}"] = dict(drop=s["drop"]["k"], n=s["drop"]["n"], interv=round(s["interv_rate"], 4),
                                                     s_per_carry=round(s["cpu_s_per_ep"], 2))
    # one tiny tune cell + test on it
    ch = tune.tune_cell("piper_trk", "Rc3k", 0, "trkB10f", n_val=6, out_root=SM, tag="smoke", workers=workers)
    out["tune_cell"] = {k: (dict(thr=v.get("thr"), rho=v.get("rho"), rho_target=v.get("rho_target"), drop_val=v.get("drop_val"))
                            if isinstance(v, dict) and "thr" in v else v) for k, v in ch.items()}
    tune.test_cell("piper_trk", "Rc3k", 0, "trkB10f", 6, SM, "smoke", workers=workers)
    report.report2("smoke", "test", SM)
    out["wall_s"] = time.time() - t0
    _save("test2", out)
    print(json.dumps(out, indent=1, default=float), flush=True)
    return out


# ===================================================================================== test3
def test3_smoke(workers=1):
    from world_model.v2.tray_rep.robust import test3, budget, report
    t0 = time.time()
    out = {}
    for setup in RU.SETUPS:
        out[f"sweep_{setup}"] = test3.run(setup, "Rc3k", 0, (0, 3, 5), ("plain", "rob"), 6, workers, "smoke", SM, "test", base_tag="smoke")
    out["budget"] = budget.run("piper", 4, workers, SM, "smoke")
    budget.run("piper_trk", 4, workers, SM, "smoke")
    report.report3("smoke", "test", SM, "smoke")
    out["wall_s"] = time.time() - t0
    _save("test3", out)
    print(json.dumps(out, indent=1, default=float), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["checks", "test1", "test2", "test3", "all"])
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    SM.mkdir(parents=True, exist_ok=True)
    for c, f in (("checks", checks), ("test1", test1_smoke), ("test2", test2_smoke), ("test3", test3_smoke)):
        if a.cmd in (c, "all"):
            f(a.workers)


if __name__ == "__main__":
    main()
