"""v5: A (adversarial) vs v4-C (decentralized, hand-tuned ego-priority) vs MPC
(joint cooperative MPC, no hand-tuned margin) plots from
experiments/ab_smooth_vs_bangbang/logs/v5_{A,C,MPC}.npz into
/common/users/ss5772/bimanual/artifacts/ab5_*.png.

Run:
    MUJOCO_GL=egl /common/users/ss5772/miniforge3/bin/conda run -n safe_env \\
        python /common/users/ss5772/bimanual/experiments/ab_smooth_vs_bangbang/make_plots_v5.py
"""
from __future__ import annotations

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sim_common as sc

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
ART_DIR = "/common/users/ss5772/bimanual/artifacts"
os.makedirs(ART_DIR, exist_ok=True)

CONDITIONS = ["A", "v4-C", "MPC"]
FILE_TAG = {"A": "A", "v4-C": "C", "MPC": "MPC"}
COLOR = {"A": "#D55E00", "v4-C": "#009E73", "MPC": "#0072B2"}
LABEL = {"A": "A: adversarial (open-loop intruder)",
         "v4-C": "v4-C: decentralized, hand-tuned ego-priority margin",
         "MPC": "MPC: joint cooperative, no hand-tuned margin"}

logs = {c: dict(np.load(os.path.join(LOG_DIR, f"v5_{FILE_TAG[c]}.npz"))) for c in CONDITIONS}
with open(os.path.join(LOG_DIR, "v5_summary.json")) as f:
    summary = json.load(f)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})


def savefig(fig, name):
    path = os.path.join(ART_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


# (a) min arm-arm distance / h over time
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for c in CONDITIONS:
    ax1.plot(logs[c]["t"], logs[c]["dist"], color=COLOR[c], label=LABEL[c], lw=1.3)
ax1.set_xlabel("time [s]"); ax1.set_ylabel("min arm-arm sphere distance [m]")
ax1.set_title("(a1) Min arm-arm distance")
ax1.legend(fontsize=7, loc="upper right")

for c in CONDITIONS:
    ax2.plot(logs[c]["t"], logs[c]["h"], color=COLOR[c], label=LABEL[c], lw=1.3)
ax2.axhline(0.0, color="black", ls="--", lw=1, alpha=0.6, label="h=0 (safety boundary)")
ax2.set_xlabel("time [s]"); ax2.set_ylabel("barrier value h = dist - combined_radius [m]")
ax2.set_title("(a2) HOCBF barrier value h(t) (ego-vs-intruder, worst pair)")
ax2.legend(fontsize=7, loc="upper right")
fig.suptitle("v5: does a JOINT/centralized cooperative MPC beat v4-C's hand-tuned margin?", fontsize=10)
savefig(fig, "ab5_distance_h.png")

# (b) box drift (trans + rot)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for c in CONDITIONS:
    ax1.plot(logs[c]["t"], logs[c]["box_drift_trans"] * 100.0, color=COLOR[c], label=LABEL[c], lw=1.3)
ax1.axhline(5.0, color="black", ls="--", lw=1, alpha=0.6, label="drop threshold (5cm)")
ax1.set_xlabel("time [s]"); ax1.set_ylabel("box translation drift [cm]")
ax1.set_title("(b1) Box drift vs gripper (translation)")
ax1.legend(fontsize=7, loc="upper left")

for c in CONDITIONS:
    ax2.plot(logs[c]["t"], np.degrees(logs[c]["box_drift_rot"]), color=COLOR[c], label=LABEL[c], lw=1.3)
ax2.set_xlabel("time [s]"); ax2.set_ylabel("box rotation drift [deg]")
ax2.set_title("(b2) Box drift vs gripper (rotation)")
ax2.legend(fontsize=7, loc="upper left")
savefig(fig, "ab5_box_drift.png")

# (c) EE jerk overlays (ego + intruder)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for c in CONDITIONS:
    ee = logs[c]["ee_pos"]
    v = np.gradient(ee, sc.DT, axis=0); a = np.gradient(v, sc.DT, axis=0); j = np.gradient(a, sc.DT, axis=0)
    jn = np.linalg.norm(j, axis=1)
    ax1.plot(logs[c]["t"], jn, color=COLOR[c], label=LABEL[c], lw=1.0, alpha=0.9)
ax1.set_yscale("log")
ax1.set_xlabel("time [s]"); ax1.set_ylabel("ego EE jerk [m/s^3] (log)")
ax1.set_title("(c1) Ego (grasping, LEFT) EE jerk")
ax1.legend(fontsize=7)

for c in CONDITIONS:
    ee = logs[c]["right_ee_pos"]
    v = np.gradient(ee, sc.DT, axis=0); a = np.gradient(v, sc.DT, axis=0); j = np.gradient(a, sc.DT, axis=0)
    jn = np.linalg.norm(j, axis=1)
    ax2.plot(logs[c]["t"], jn, color=COLOR[c], label=LABEL[c], lw=1.0, alpha=0.9)
ax2.set_yscale("log")
ax2.set_xlabel("time [s]"); ax2.set_ylabel("intruder EE jerk [m/s^3] (log)")
ax2.set_title("(c2) Intruder (free, RIGHT) EE jerk")
ax2.legend(fontsize=7)
savefig(fig, "ab5_ee_jerk.png")

# (d) intruder goal progress
fig, ax = plt.subplots(figsize=(9, 4.2))
for c in CONDITIONS:
    gap = np.linalg.norm(logs[c]["ee_pos"] - logs[c]["right_ee_pos"], axis=1)
    ax.plot(logs[c]["t"], gap * 100.0, color=COLOR[c], label=LABEL[c], lw=1.3)
ax.set_xlabel("time [s]"); ax.set_ylabel("ego-intruder EE-EE gap [cm]")
ax.set_title("(d) Intrusion depth over time")
ax.legend(fontsize=8)
savefig(fig, "ab5_intruder_progress.png")

# (e) solve time (MPC only)
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(logs["MPC"]["t"], logs["MPC"]["solve_time"] * 1000.0, color=COLOR["MPC"], lw=1.0)
ax.axhline(sc.DT * 1000.0, color="black", ls="--", lw=1, alpha=0.6,
           label=f"physics step budget ({sc.DT*1000:.1f}ms)")
ax.set_xlabel("time [s]"); ax.set_ylabel("MPC QP solve time [ms]")
ax.set_title("(e) Joint MPC per-step solve time (real-time plausibility)")
ax.legend(fontsize=8)
savefig(fig, "ab5_solve_time.png")

# (f) summary bars
fig, axes = plt.subplots(1, 4, figsize=(15, 4.0))
metrics = [("min_h", "min h [m] (0=safety boundary)"),
           ("box_drift_trans_max", "max box drift [m]"),
           ("rms_ee_jerk_ego", "ego RMS EE jerk [m/s^3] (log)"),
           ("qp_infeasible_frac_ego", "QP infeasible frac")]
for ax, (key, title) in zip(axes, metrics):
    vals = [summary[c][key] for c in CONDITIONS]
    bars = ax.bar(CONDITIONS, vals, color=[COLOR[c] for c in CONDITIONS])
    ax.set_title(title, fontsize=9)
    if "jerk" in key:
        ax.set_yscale("log")
    if key == "min_h":
        ax.axhline(0.0, color="black", ls="--", lw=1)
fig.suptitle("v5 summary: A (adversarial) vs v4-C (decentralized, hand-tuned) vs MPC (joint, no margin)",
             fontsize=11)
savefig(fig, "ab5_summary_bars.png")

print("\nSummary table:")
keys = ["min_h", "collision", "box_dropped", "box_drift_trans_max", "box_drift_rot_max_deg",
        "rms_ee_jerk_ego", "rms_ee_jerk_int", "intruder_progress_frac_vs_A",
        "qp_infeasible_frac_ego", "hdot_sign_changes_per_s"]
for k in keys:
    print(f"{k:28s}  " + "  ".join(f"{c}={summary[c][k]!s:>10}" for c in CONDITIONS))
