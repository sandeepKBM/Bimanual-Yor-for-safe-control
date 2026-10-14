"""v3: Generate the A/B/C/D (C0/C1/C2/C3) comparison plots from
experiments/ab_smooth_vs_bangbang/logs/*.npz (re-pointed to
scenes/scene_yor_canonical.xml, the real 0.185m-spacing YOR rig) into
/common/users/ss5772/bimanual/artifacts/ab3_*.png.

Deliberately a SEPARATE script from make_plots.py/make_plots_v2.py (which
read the same logs/*.npz paths under the OLD scene and write ab_*.png /
ab2_*.png) so v1/v2's plot PNGs are never overwritten.

Run:
    MUJOCO_GL=egl /common/users/ss5772/miniforge3/bin/conda run -n safe_env \\
        python /common/users/ss5772/bimanual/experiments/ab_smooth_vs_bangbang/make_plots_v3.py
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

CONDITIONS = ["C0", "C1", "C2", "C3"]
# Okabe-Ito colorblind-safe categorical colors, fixed assignment per condition.
COLOR = {"C0": "#949494", "C1": "#D55E00", "C2": "#0072B2", "C3": "#009E73"}
LABEL = {"C0": "C0: no safety", "C1": "C1: bang-bang HJ-style",
         "C2": "C2: smooth HOCBF-QP (hard)", "C3": "C3: geometric fabric (soft)"}

logs = {c: dict(np.load(os.path.join(LOG_DIR, f"{c}.npz"))) for c in CONDITIONS}
with open(os.path.join(LOG_DIR, "summary.json")) as f:
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


# ---------------------------------------------------------------------------
# (a) min arm-arm distance / h over time
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for c in CONDITIONS:
    ax1.plot(logs[c]["t"], logs[c]["dist"], color=COLOR[c], label=LABEL[c], lw=1.3)
ax1.axhline(sc.COMBINED_RADIUS, color="black", ls="--", lw=1, alpha=0.6, label="combined radius (r_ego+r_int+d_min)")
ax1.set_xlabel("time [s]"); ax1.set_ylabel("min arm-arm sphere distance [m]")
ax1.set_title("(a1) Min arm-arm distance")
ax1.legend(fontsize=7.5, loc="upper right")

for c in CONDITIONS:
    ax2.plot(logs[c]["t"], logs[c]["h"], color=COLOR[c], label=LABEL[c], lw=1.3)
ax2.axhline(0.0, color="black", ls="--", lw=1, alpha=0.6, label="h=0 (safety boundary)")
ax2.set_xlabel("time [s]"); ax2.set_ylabel("barrier value h = dist - combined_radius [m]")
ax2.set_title("(a2) HOCBF barrier value h(t)")
ax2.legend(fontsize=7.5, loc="upper right")
fig.suptitle("v3: analytic-distance barrier -- arm-arm safety margin (aggressive oscillating intruder)", fontsize=10)
savefig(fig, "ab3_distance_h.png")

# ---------------------------------------------------------------------------
# (b) EE jerk over time
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.2))
for c in CONDITIONS:
    ee = logs[c]["ee_pos"]
    v = np.gradient(ee, sc.DT, axis=0)
    a = np.gradient(v, sc.DT, axis=0)
    j = np.gradient(a, sc.DT, axis=0)
    jn = np.linalg.norm(j, axis=1)
    ax.plot(logs[c]["t"], jn, color=COLOR[c], label=LABEL[c], lw=1.0, alpha=0.9)
ax.set_yscale("log")
ax.set_xlabel("time [s]"); ax.set_ylabel("EE jerk magnitude [m/s^3] (log scale)")
ax.set_title("(b) Ego end-effector jerk -- soft fabric vs smooth HOCBF-QP vs bang-bang")
ax.legend(fontsize=9)
savefig(fig, "ab3_ee_jerk.png")

# ---------------------------------------------------------------------------
# (c) grasp (finger-box) contact force over time
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.2))
for c in CONDITIONS:
    ax.plot(logs[c]["t"], logs[c]["contact_force"], color=COLOR[c], label=LABEL[c], lw=1.1, alpha=0.9)
ax.set_xlabel("time [s]"); ax.set_ylabel("total finger-box contact force [N]")
ax.set_title("(c) Grasp contact force (left fingers on box_center) -- payload-coupled dynamics")
ax.legend(fontsize=9)
savefig(fig, "ab3_grasp_force.png")

# ---------------------------------------------------------------------------
# (d) box pose drift over time (translation + rotation)
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for c in CONDITIONS:
    ax1.plot(logs[c]["t"], logs[c]["box_drift_trans"] * 100.0, color=COLOR[c], label=LABEL[c], lw=1.3)
ax1.axhline(5.0, color="black", ls=":", lw=1, alpha=0.5, label="box_dropped threshold (5cm)")
ax1.set_xlabel("time [s]"); ax1.set_ylabel("box translation drift vs gripper [cm]")
ax1.set_title("(d1) Box position drift (grasp frame)")
ax1.legend(fontsize=7.5)

for c in CONDITIONS:
    ax2.plot(logs[c]["t"], logs[c]["box_drift_rot"] * 180.0 / np.pi, color=COLOR[c], label=LABEL[c], lw=1.3)
ax2.set_xlabel("time [s]"); ax2.set_ylabel("box rotation drift vs gripper [deg]")
ax2.set_title("(d2) Box orientation drift (grasp frame)")
ax2.legend(fontsize=7.5)
fig.suptitle("v3 (real 0.185m geometry): grasp retention under the oscillating intruder -- "
             "C1/C2/C3 all exceed the 5cm drop threshold here (tighter rig, see RESULTS_v3.md)", fontsize=9.5)
savefig(fig, "ab3_box_drift.png")

# ---------------------------------------------------------------------------
# Summary bar chart of headline metrics
# ---------------------------------------------------------------------------
metrics_panels = [
    ("rms_ee_jerk", "RMS EE jerk [m/s^3]", False),
    ("rms_torque_rate", "RMS torque-rate [Nm/s]", False),
    ("min_h", "min h [m] (>=0 is collision-free)", False),
    ("box_drift_trans_max", "max box translation drift [m]", False),
]
fig, axes = plt.subplots(1, 4, figsize=(16, 4.0))
for ax, (key, ylabel, _log) in zip(axes, metrics_panels):
    vals = [summary[c][key] for c in CONDITIONS]
    colors = [COLOR[c] for c in CONDITIONS]
    bars = ax.bar(CONDITIONS, vals, color=colors)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(key, fontsize=9)
    if key == "min_h":
        ax.axhline(0.0, color="black", lw=1, ls="--", alpha=0.7)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3g}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8,
                    xytext=(0, 2), textcoords="offset points")
fig.suptitle("v3 headline metrics: C0 (none) | C1 (bang-bang, hard) | C2 (HOCBF-QP, hard) | C3 (fabric, SOFT)", fontsize=11)
savefig(fig, "ab3_summary_bars.png")

# ---------------------------------------------------------------------------
# Summary table figure (all metrics, exact values)
# ---------------------------------------------------------------------------
keys = [k for k in summary["C0"].keys() if k != "condition"]
fig, ax = plt.subplots(figsize=(13, 0.4 * len(keys) + 1.2))
ax.axis("off")
cell_text = []
for k in keys:
    row = []
    for c in CONDITIONS:
        v = summary[c][k]
        if isinstance(v, bool):
            row.append(str(v))
        elif isinstance(v, float):
            row.append(f"{v:.4g}")
        else:
            row.append(str(v))
    cell_text.append(row)
table = ax.table(cellText=cell_text, rowLabels=keys, colLabels=[LABEL[c] for c in CONDITIONS],
                  cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1.0, 1.3)
for c_idx, c in enumerate(CONDITIONS):
    table[(0, c_idx)].set_facecolor(COLOR[c])
    table[(0, c_idx)].get_text().set_color("white")
ax.set_title("v3 full metric table: C0 | C1 | C2 | C3  (C3 = soft-safety fabric, see safety_type row)", fontsize=12, pad=14)
savefig(fig, "ab3_metric_table.png")

print("\nAll v3 plots written to", ART_DIR)
