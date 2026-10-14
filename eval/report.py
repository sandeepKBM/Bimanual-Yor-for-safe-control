"""Aggregate eval/run_eval.py's per-episode CSV into a per-config summary
table (csv+json) and the 4-config comparison bar plot -- the "does the world
model help" figure template this eval harness exists to produce.

Run:
    /common/users/ss5772/miniforge3/bin/conda run -n safe_env \\
        python /common/users/ss5772/bimanual/eval/report.py

Reads eval/results/all_episodes.csv (written by run_eval.py).
Writes eval/results/summary.csv, eval/results/summary.json,
artifacts/eval_compare.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_EVAL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EVAL_DIR.parent
RESULTS_DIR = _EVAL_DIR / "results"
ARTIFACTS_DIR = _REPO_ROOT / "artifacts"

CONFIG_ORDER = ["none", "geometric_only", "semantic_only", "both"]

# (column, aggregator, display label) -- aggregator "rate" = mean of a 0/1
# (or bool) column * 100; "mean" = plain mean, ignoring NaN.
PANELS = [
    ("success", "rate", "Task success rate (%)"),
    ("collision", "rate", "Collision rate (%)"),
    ("spilled", "rate", "Spill rate (%)"),
    ("cup_dropped", "rate", "Grasp-dropped rate (%)"),
    ("min_h", "mean", "Mean episode min h [m]\n(HJ-style safety margin)"),
    ("rms_ee_jerk", "mean", "Mean RMS EE jerk"),
    ("mean_solve_time_ms", "mean", "Mean per-step solve time [ms]"),
    ("semantic_veto_frac", "mean", "Mean semantic-veto fraction"),
]


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config in CONFIG_ORDER:
        sub = df[df["config"] == config]
        if len(sub) == 0:
            continue
        row = {"config": config, "n_episodes": int(len(sub))}
        for col, kind, _label in PANELS:
            vals = sub[col].astype(float)
            if kind == "rate":
                row[f"{col}_rate_pct"] = float(np.nanmean(vals) * 100.0)
            else:
                row[f"{col}_mean"] = float(np.nanmean(vals))
        # precision/recall: only meaningful where the semantic role is active
        prec = sub["semantic_precision"].astype(float)
        rec = sub["semantic_recall"].astype(float)
        row["semantic_precision_mean"] = float(np.nanmean(prec)) if prec.notna().any() else None
        row["semantic_recall_mean"] = float(np.nanmean(rec)) if rec.notna().any() else None
        row["qp_infeasible_frac_mean"] = float(np.nanmean(sub["qp_infeasible_frac"].astype(float)))
        rows.append(row)
    return pd.DataFrame(rows).set_index("config").reindex(
        [c for c in CONFIG_ORDER if c in df["config"].unique()])


def make_plot(summary: pd.DataFrame, out_path: Path) -> None:
    configs = list(summary.index)
    n_panels = len(PANELS)
    ncols = 4
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    colors = ["#8a8f98", "#4c78a8", "#e2a33a", "#54a24b"]  # none/geo/sem/both

    for i, (col, kind, label) in enumerate(PANELS):
        ax = axes[i]
        key = f"{col}_rate_pct" if kind == "rate" else f"{col}_mean"
        vals = [summary.loc[c, key] if c in summary.index else np.nan for c in configs]
        bars = ax.bar(configs, vals, color=colors[: len(configs)])
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.2g}", (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=7)
    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Cup-task safety-stack comparison (SMOKE, placeholder modules)\n"
                  "none / geometric_only / semantic_only / both", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    csv_path = RESULTS_DIR / "all_episodes.csv"
    if not csv_path.exists():
        raise SystemExit(f"no episodes found at {csv_path}; run eval/run_eval.py first")
    df = pd.read_csv(csv_path)
    summary = aggregate(df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_csv = RESULTS_DIR / "summary.csv"
    summary_json = RESULTS_DIR / "summary.json"
    summary.to_csv(summary_csv)
    with open(summary_json, "w") as f:
        json.dump(json.loads(summary.reset_index().to_json(orient="records")), f, indent=2)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = ARTIFACTS_DIR / "eval_compare.png"
    make_plot(summary, plot_path)

    print(summary.to_string())
    print(f"\nWrote {summary_csv}\nWrote {summary_json}\nWrote {plot_path}")


if __name__ == "__main__":
    main()
