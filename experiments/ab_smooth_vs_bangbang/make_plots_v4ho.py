"""v4-HO: A (adversarial) vs C (v4-C baseline) vs C_HO (v4-C + higher-order
smoothness) plots from experiments/ab_smooth_vs_bangbang/logs/v4ho_{A,C,C_HO}.npz
into /common/users/ss5772/bimanual/artifacts/ab4ho_*.png.
"""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
ART_DIR = "/common/users/ss5772/bimanual/artifacts"
os.makedirs(ART_DIR, exist_ok=True)

CONDITIONS = ["A", "C", "C_HO"]
COLOR = {"A": "#D55E00", "C": "#009E73", "C_HO": "#0072B2"}
LABEL = {"A": "A: adversarial", "C": "v4-C (baseline)", "C_HO": "v4-C + higher-order smoothness"}

logs = {c: dict(np.load(os.path.join(LOG_DIR, f"v4ho_{c}.npz"))) for c in CONDITIONS}
with open(os.path.join(LOG_DIR, "v4ho_summary.json")) as f:
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


# (a) hdot over time -- chatter proxy, zoom on C vs C_HO (A is on a very
# different scale / regime, shown for reference on its own y-scale panel)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for c in CONDITIONS:
    ax1.plot(logs[c]["t"], logs[c]["hdot"], color=COLOR[c], label=LABEL[c], lw=1.0, alpha=0.85)
ax1.set_xlabel("time [s]"); ax1.set_ylabel("hdot [m/s]")
ax1.set_title("(a1) hdot(t), all three")
ax1.legend(fontsize=8)

for c in ["C", "C_HO"]:
    ax2.plot(logs[c]["t"], logs[c]["hdot"], color=COLOR[c], label=LABEL[c], lw=1.1)
ax2.set_xlabel("time [s]"); ax2.set_ylabel("hdot [m/s]")
ax2.set_title("(a2) hdot(t), C vs C_HO (zoom): does HO cut chatter further?")
ax2.legend(fontsize=8)
fig.suptitle("v4-HO: hdot sign-changes/s (chatter) -- "
             f"A={summary['A']['hdot_sign_changes_per_s']:.2f}  "
             f"C={summary['C']['hdot_sign_changes_per_s']:.2f}  "
             f"C_HO={summary['C_HO']['hdot_sign_changes_per_s']:.2f}", fontsize=10)
savefig(fig, "ab4ho_hdot_chatter.png")

# (b) EE jerk overlays (ego + intruder), C vs C_HO
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for c in CONDITIONS:
    ee = logs[c]["ee_pos"]
    v = np.gradient(ee, 0.005, axis=0); a = np.gradient(v, 0.005, axis=0); j = np.gradient(a, 0.005, axis=0)
    jn = np.linalg.norm(j, axis=1)
    ax1.plot(logs[c]["t"], jn, color=COLOR[c], label=LABEL[c], lw=1.0, alpha=0.9)
ax1.set_yscale("log")
ax1.set_xlabel("time [s]"); ax1.set_ylabel("ego EE jerk [m/s^3] (log)")
ax1.set_title("(b1) Ego (grasping) EE jerk")
ax1.legend(fontsize=7.5)

for c in CONDITIONS:
    ee = logs[c]["right_ee_pos"]
    v = np.gradient(ee, 0.005, axis=0); a = np.gradient(v, 0.005, axis=0); j = np.gradient(a, 0.005, axis=0)
    jn = np.linalg.norm(j, axis=1)
    ax2.plot(logs[c]["t"], jn, color=COLOR[c], label=LABEL[c], lw=1.0, alpha=0.9)
ax2.set_yscale("log")
ax2.set_xlabel("time [s]"); ax2.set_ylabel("intruder EE jerk [m/s^3] (log)")
ax2.set_title("(b2) Intruder (free) EE jerk")
ax2.legend(fontsize=7.5)
savefig(fig, "ab4ho_ee_jerk.png")

# (c) box drift comparison
fig, ax = plt.subplots(figsize=(7, 4.2))
for c in CONDITIONS:
    ax.plot(logs[c]["t"], logs[c]["box_drift_trans"] * 100.0, color=COLOR[c], label=LABEL[c], lw=1.3)
ax.axhline(5.0, color="black", ls="--", lw=1, alpha=0.6, label="drop threshold (5cm)")
ax.set_xlabel("time [s]"); ax.set_ylabel("box translation drift [cm]")
ax.set_title("(c) Box drift: HO does not touch grasp retention")
ax.legend(fontsize=8)
savefig(fig, "ab4ho_box_drift.png")

print("\nSummary table:")
keys = ["min_h", "collision", "box_dropped", "box_drift_trans_max",
        "rms_ee_jerk_ego", "max_ee_jerk_ego", "rms_ee_jerk_int", "max_ee_jerk_int",
        "hdot_sign_changes_per_s", "intruder_progress_frac_vs_A", "avg_step_time_ms"]
for k in keys:
    print(f"{k:28s}  " + "  ".join(f"{c}={summary[c][k]!s:>10}" for c in CONDITIONS))
