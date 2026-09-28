#!/usr/bin/env python
"""report.py -- final results table (rig held-out test) + flagship FL-MPC
transfer check + latent-fidelity check. SIM DATA ONLY.

  python experiments/slosh/report.py [--flagship /path/traj_clean.npz]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import slosh_models as sm  # noqa: E402
from data_prep import load_rig, load_flagship, labels  # noqa: E402
from evaluate import evaluate, fmt_row, HEADER  # noqa: E402

ORDER = ["a0_static_tilt", "a1_eff_gravity", "b2_cup_osc_physics_w24.6", "b1_world_pendulum",
         "b2_cup_oscillator", "b3_world_lti", "c_pendulum_residual", "c_pendulum_dyn_residual", "d_gru"]


def smooth_pos(pos, k=5):
    ker = np.ones(k) / k
    return np.stack([np.convolve(np.pad(pos[:, i], k // 2, mode="edge"), ker, "valid")[:len(pos)] for i in range(3)], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data" / "rig_v1.pkl"))
    ap.add_argument("--flagship", default="/common/home/ss5772/.claude/jobs/f1580deb/tmp/traj_clean.npz")
    ap.add_argument("--H", type=int, default=50)
    a = ap.parse_args()
    D = load_rig(a.data)
    phys = np.load(HERE / "data" / "scores_phys.npz")
    P = json.loads(str(phys["params"]))
    scores = {k: phys[k] for k in phys.files if k not in ("params", "metrics")}
    for f in sorted((HERE / "data").glob("scores_nets_*.npz")):
        z = np.load(f); scores.update({k: z[k] for k in z.files})
    pos = D["onset"] >= 0
    print(f"RIG data: {len(pos)} trajectories ({pos.sum()} spilled), test {D['test'].sum()} "
          f"({(D['test'] & pos).sum()} spilled); tunneling-dropped {D['n_tunnel_dropped']}; T={D['T']} @5ms")
    print("fitted params:", json.dumps(P))
    res = {}
    for H in (20, a.H, 100):
        print(f"\n### held-out test, label = spill within H={H} steps ({H*5} ms)")
        print(HEADER)
        for k in ORDER:
            if k in scores:
                m = evaluate(scores[k], D, H=H); res[(k, H)] = m
                print(fmt_row(k, m))
    # ---- bootstrap (resample TEST trajectories) 95% CIs at H
    y, valid = labels(D["onset"], D["k0"], D["T"], a.H)
    ti = np.where(D["test"])[0]; rng = np.random.default_rng(0)
    print(f"\n### bootstrap 95% CI over test trajectories (200 reps), H={a.H}")
    for k in ORDER:
        if k not in scores:
            continue
        s = scores[k]; es = np.where(valid, s, -np.inf).max(-1)
        c_list, e_list = [], []
        for _ in range(200):
            b = rng.choice(ti, len(ti), replace=True)
            vv = valid[b][:, ::4]
            c_list.append(sm.auroc(s[b][:, ::4][vv], y[b][:, ::4][vv])); e_list.append(sm.auroc(es[b], pos[b]))
        print(f"  {k:28s} causal AUROC [{np.percentile(c_list, 2.5):.3f}, {np.percentile(c_list, 97.5):.3f}]  "
              f"ep-AUROC [{np.percentile(e_list, 2.5):.3f}, {np.percentile(e_list, 97.5):.3f}]")
    K = sm.kinematics(D["pos"], D["n"])
    pk =np.degrees(np.max(np.where(labels(D["onset"], D["k0"], D["T"], a.H)[1], K["theta0"], 0), -1))
    hard = (pk > 60) & (pk < 110)
    print(f"\n### HARD subset (peak static tilt 60-110 deg): test n={(D['test'] & hard).sum()} "
          f"spilled={(D['test'] & hard & pos).sum()}, H={a.H}")
    print(HEADER)
    for k in ORDER:
        if k in scores:
            print(fmt_row(k, evaluate(scores[k], D, H=a.H, mask=hard)))

    # ---- latent fidelity: does the model's slosh angle track the ball pile?
    y, valid = labels(D["onset"], D["k0"], D["T"], a.H)
    te = D["test"]
    cen = D["centroid"]; lat = np.hypot(cen[..., 0], cen[..., 1])     # pile lateral offset in cup frame
    from scipy.stats import spearmanr
    print("\n### latent fidelity: Spearman(model angle, ball-pile lateral offset) on test valid steps")
    for k in ("a0_static_tilt", "a1_eff_gravity", "b1_world_pendulum", "b2_cup_oscillator", "b3_world_lti"):
        m = valid[te][:, ::4]
        print(f"  {k:28s} rho = {spearmanr(scores[k][te][:, ::4][m], lat[te][:, ::4][m])[0]:.3f}")

    # ---- flagship FL-MPC transfer (faked grasp, zero-velocity pin, pre-inertial data)
    if Path(a.flagship).exists():
        eps = load_flagship(a.flagship)
        print(f"\n### FLAGSHIP FL-MPC episodes ({len(eps)}; faked grasp, zero-vel pin, oracle-fixed)")
        up = []
        for e in eps:
            Ke = sm.kinematics(smooth_pos(e["pos"])[None], e["n"][None])
            e["theta0_rel"] = float(np.degrees(Ke["theta0"][0, e["k0"]]))
            e["s"] = {"a0_static_tilt": Ke["theta0"][0], "a1_eff_gravity": Ke["theta_eff"][0],
                      "b1_world_pendulum": sm.risk(sm.pendulum_world(Ke, P["b1_world_pendulum"]["w"], P["b1_world_pendulum"]["zeta"]),
                                                   P["b1_world_pendulum"]["kappa"])[0],
                      "b3_world_lti": sm.risk(sm.lti_world(Ke, P["b3_world_lti"]["w"], P["b3_world_lti"]["zeta"]),
                                              P["b3_world_lti"]["kappa"])[0]}
            e["peak_a"] = float(np.max(np.linalg.norm(Ke["a"][0, e["k0"]:], axis=-1)))
        sp = np.array([e["onset"] >= 0 for e in eps]); t0 = np.array([e["theta0_rel"] for e in eps])
        print(f"  spilled {sp.sum()}/{len(eps)}; cup tilt at release: spilled {np.round(t0[sp], 0)} deg; "
              f"clean {np.round(t0[~sp], 1)} deg")
        print(f"  peak |a_cup| after release (clean eps): {np.round([e['peak_a'] for e in eps if e['onset'] < 0], 1)} m/s^2")
        for k in ("a0_static_tilt", "a1_eff_gravity", "b1_world_pendulum", "b3_world_lti"):
            es = []
            for e in eps:
                end = e["onset"] if e["onset"] >= 0 else len(e["s"][k])
                es.append(np.max(e["s"][k][e["k0"]:end]) if end > e["k0"] else e["s"][k][e["k0"]])
            es = np.array(es); thr = res[(k, a.H)]["thr"]
            print(f"  {k:22s} ep-AUROC(24 eps)={sm.auroc(es, sp):.3f} | rig thr={np.degrees(thr):.0f}deg: "
                  f"alarms on clean {np.mean(es[~sp] >= thr):.2f}, on spilled {np.mean(es[sp] >= thr):.2f} | "
                  f"clean max {np.degrees(es[~sp].max()):.0f}deg")


if __name__ == "__main__":
    main()
