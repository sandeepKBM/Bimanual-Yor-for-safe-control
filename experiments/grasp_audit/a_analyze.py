"""Analyze Part A npz logs: EE tracking errors, speed spike root cause, cup tilt, left-EE accel demand.
usage: python a_analyze.py A_seed0.npz [...]"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import quat_mul, quat_conj, quat_angle  # noqa: E402

V_CAP = 0.28


def pct(x, ps=(50, 95, 99, 100)):
    x = np.asarray(x)
    return " ".join(f"p{p}={np.percentile(x, p):.3g}" for p in ps)


def analyze(path, bounds=None):
    z = np.load(path, allow_pickle=True)
    dt = float(z["dt"]); t0r = float(z["right_start_t"])
    t = z["log_t"]; tr = z["log_right_track_t"]
    n = len(t)
    rp = z["log_right_ee_pos"]
    fd = np.linalg.norm(np.diff(rp, axis=0), axis=1) / dt           # summarize()'s metric
    vtrue = np.linalg.norm(z["r_v"], axis=1)                         # J*qvel at the same instants
    k = int(np.argmax(fd))
    print(f"== {path}")
    print(f"right_ee_speed_max_ms (FD, summarize) = {fd[k]:.3f} at step {k}->{k+1}, t={t[k]:.3f}s, t_r={tr[k]:.3f}")
    print(f"  true |v| (mj_objectVelocity) at k,k+1 = {vtrue[k]:.3f}, {vtrue[k+1]:.3f}; max true |v| = {vtrue.max():.3f} "
          f"at step {int(np.argmax(vtrue))} t={t[int(np.argmax(vtrue))]:.3f}")
    w = slice(max(0, k - 6), min(n, k + 8))
    eq = z["eq_active"]
    names = z["eq_names"]
    print("  step  t     t_r    |v|fd   |v|true  pinned eq_active                 fl_feas fl_relax |qd_r|max  |tau_r|max  |qfrc_con_r|max")
    for j in range(w.start, w.stop):
        fdj = fd[j] if j < len(fd) else np.nan
        print(f"  {j:5d} {t[j]:.3f} {tr[j]:+.3f} {fdj:7.3f} {vtrue[j]:7.3f}   {int(z['log_cube_pinned'][j])}   "
              f"{''.join(str(int(a)) for a in eq[j])}  {int(z['fl_feasible'][j])} {int(z['fl_relaxed'][j])} "
              f"{np.abs(z['qvel_r'][j]).max():7.2f} {np.abs(z['tau_r'][j]).max():8.2f} {np.abs(z['qfrc_constraint_r'][j]).max():9.2f}")
    # all FD speed exceedances
    over = fd > V_CAP * 1.05
    idx = np.nonzero(over)[0]
    print(f"  FD speed > {V_CAP*1.05:.3f}: {over.sum()} steps ({100*over.mean():.2f}%); true |v| > cap: "
          f"{(vtrue > V_CAP*1.05).sum()} steps")
    if len(idx):
        runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
        for r in runs[:12]:
            j = r[np.argmax(fd[r])]
            print(f"    run steps {r[0]}-{r[-1]} ({len(r)}): peak FD {fd[j]:.3f} true {vtrue[j]:.3f} t={t[j]:.3f} t_r={tr[j]:.3f} "
                  f"pinned {int(z['log_cube_pinned'][j])}->{int(z['log_cube_pinned'][min(j+1,n-1)])} fl_relax {int(z['fl_relaxed'][j])}")
    # tracking error (right, after start)
    cross = tr >= 0
    e_r = np.linalg.norm(rp - z["ref_r"], axis=1)
    ang_r = np.array([np.degrees(quat_angle(quat_mul(quat_conj(z["ref_rq"][i]), z["r_quat"][i]))) for i in range(n)])
    print(f"  RIGHT EE pos err [mm] (t_r>=0): {pct(e_r[cross]*1e3)} ; ori err [deg]: {pct(ang_r[cross])}")
    lp = z["log_left_ee_pos"]
    lift = t < t0r
    e_l = np.linalg.norm(lp - z["ref_l"], axis=1)
    ang_l = np.array([np.degrees(quat_angle(quat_mul(quat_conj(z["ref_lq"][i]), z["l_quat"][i]))) for i in range(n)])
    print(f"  LEFT EE pos err [mm] LIFT (t<{t0r}): {pct(e_l[lift]*1e3)} ; ori [deg]: {pct(ang_l[lift])}")
    print(f"  LEFT EE deviation from q_lift hold [mm] CROSSING: {pct(e_l[cross]*1e3)} ; ori [deg]: {pct(ang_l[cross])}")
    tilt = np.degrees(z["log_left_cup_tilt_rad"])
    print(f"  cup tilt (EE-axis, log_left_cup_tilt) [deg]: all {pct(tilt)} ; crossing {pct(tilt[cross])} ; "
          f"frac>5.73deg (crossing) = {np.mean(tilt[cross] > 5.73):.3f}")
    if "log_true_cup_tilt_rad" in z:
        tt = np.degrees(z["log_true_cup_tilt_rad"])
        print(f"  TRUE cup-body tilt [deg]: all {pct(tt)} ; crossing {pct(tt[cross])}")
    # left EE (= cup carrier) accelerations: FD of exact velocities
    lv, lw = z["l_v"], z["l_w"]
    la = np.linalg.norm(np.diff(lv, axis=0), axis=1) / dt
    lal = np.linalg.norm(np.diff(lw, axis=0), axis=1) / dt
    c1 = cross[1:]
    print(f"  LEFT EE |a| [m/s^2] crossing: {pct(la[c1])} ; LIFT: {pct(la[~c1])}")
    print(f"  LEFT EE |alpha| [rad/s^2] crossing: {pct(lal[c1])} ; LIFT: {pct(lal[~c1])}")
    print(f"  LEFT EE |v| [m/s] crossing: {pct(np.linalg.norm(lv, axis=1)[cross])} ; |w| [rad/s]: "
          f"{pct(np.linalg.norm(lw, axis=1)[cross])}")
    # cup-body accel (faked: kinematically pinned, so = EE-rigid-point accel)
    cv = z["cup_v"]
    ca = np.linalg.norm(np.diff(cv, axis=0), axis=1) / dt
    print(f"  CUP body |a| [m/s^2] crossing: {pct(ca[c1])}")
    # smoothed (20 ms) accel, to separate 1-step numerical spikes from sustained demand
    ker = np.ones(4) / 4
    las = np.linalg.norm(np.stack([np.convolve(np.diff(lv[:, i]) / dt, ker, "same") for i in range(3)], 1), axis=1)
    print(f"  LEFT EE |a| 20ms-smoothed crossing: {pct(las[c1])}")
    print(f"  fl relaxed-fallback steps: {int(z['fl_relaxed'].sum())}, infeasible steps: {int((1-z['fl_feasible'][cross]).sum())}")
    return dict(k=k, fd=fd, vtrue=vtrue)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
