"""Aggregate the fixed SPILL-scenario 4-config comparison (eval/run_eval.py's
none/geometric_only/semantic_only/both, run via world_model.run_wm_eval_harness
with the REAL trained WM spill monitor swapped in for semantic_only/both) into
a summary table + the "does the WM help" figure.

Unlike eval/report.py (which reads eval/results/all_episodes.csv from a
single run_eval.py invocation), this reads and merges several batched
per-config/per-seed-range CSVs under eval/results/real_wm/*/all_episodes.csv
-- the 8-seed run was split into foreground batches to stay under the tool
timeout (the real WM monitor's render+RSSM cost makes semantic_only/both far
slower per-episode than the placeholder oracle).

Run:
    conda run -n mujoco_ur5e python eval/report_spill.py

Reads eval/results/real_wm/*/all_episodes.csv.
Writes eval/results/real_wm/all_episodes.csv (merged),
eval/results/real_wm/summary.csv, eval/results/real_wm/summary.json,
artifacts/does_wm_help_spill.png.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_EVAL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EVAL_DIR.parent
REAL_WM_DIR = _EVAL_DIR / "results" / "real_wm"
ARTIFACTS_DIR = _REPO_ROOT / "artifacts"

CONFIG_ORDER = ["none", "geometric_only", "semantic_only", "both"]

# spilled is the HEADLINE metric (the semantic failure geometry can't see);
# collision/min_h are reported to show the geometric layer's role is
# unchanged/inactive in this scenario.
PANELS = [
    ("success", "rate", "Task success rate (%)"),
    ("spilled", "rate", "Spill rate (%)  <- headline"),
    ("collision", "rate", "Collision rate (%)"),
    ("min_h", "mean", "Mean episode min h [m]\n(HJ-style safety margin)"),
    ("semantic_veto_frac", "mean", "Mean semantic-veto fraction"),
    ("mean_solve_time_ms", "mean", "Mean per-step QP solve time [ms]"),
]


def load_merged() -> list[dict]:
    csvs = sorted(REAL_WM_DIR.glob("*/all_episodes.csv"))
    if not csvs:
        raise SystemExit(f"no batch CSVs found under {REAL_WM_DIR}/*/all_episodes.csv")
    rows = []
    seen = set()
    fieldnames = None
    for p in csvs:
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = fieldnames or reader.fieldnames
            for row in reader:
                key = (row["config"], row["seed"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    merged_path = REAL_WM_DIR / "all_episodes.csv"
    with open(merged_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Merged {len(csvs)} batch CSVs ({len(rows)} episode rows) -> {merged_path}")
    return rows


def _to_bool_float(v: str) -> float:
    if v in ("True", "False"):
        return 1.0 if v == "True" else 0.0
    try:
        return float(v)
    except ValueError:
        return float("nan")


def aggregate(rows: list[dict]) -> dict:
    summary = {}
    for config in CONFIG_ORDER:
        sub = [r for r in rows if r["config"] == config]
        if not sub:
            continue
        entry = {"n_episodes": len(sub)}
        for col, kind, _label in PANELS:
            vals = np.array([_to_bool_float(r[col]) for r in sub], dtype=float)
            if kind == "rate":
                entry[f"{col}_rate_pct"] = float(np.nanmean(vals) * 100.0)
            else:
                entry[f"{col}_mean"] = float(np.nanmean(vals))
        summary[config] = entry
    return summary


def make_plot(summary: dict, out_path: Path) -> None:
    configs = [c for c in CONFIG_ORDER if c in summary]
    n_panels = len(PANELS)
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    colors = ["#8a8f98", "#4c78a8", "#e2a33a", "#54a24b"]  # none/geo/sem/both

    for i, (col, kind, label) in enumerate(PANELS):
        ax = axes[i]
        key = f"{col}_rate_pct" if kind == "rate" else f"{col}_mean"
        vals = [summary[c].get(key, np.nan) for c in configs]
        bars = ax.bar(configs, vals, color=colors[: len(configs)])
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.2g}", (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=8)
    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Does the world model help? -- SPILL scenario (geometry-blind failure)\n"
        "LEFT arm: fast cup tilt transfer (accel/tilt-induced spill) -- RIGHT arm: idle (no intruder)\n"
        "semantic_only/both use the REAL trained WM spill monitor (world_model/wm_semantic_monitor.py)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    rows = load_merged()
    summary = aggregate(rows)

    summary_csv = REAL_WM_DIR / "summary.csv"
    summary_json = REAL_WM_DIR / "summary.json"
    fieldnames = ["config"] + sorted({k for v in summary.values() for k in v.keys()})
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for config in CONFIG_ORDER:
            if config in summary:
                w.writerow({"config": config, **summary[config]})
    with open(summary_json, "w") as f:
        json.dump([{"config": c, **summary[c]} for c in CONFIG_ORDER if c in summary], f, indent=2)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = ARTIFACTS_DIR / "does_wm_help_spill.png"
    make_plot(summary, plot_path)

    for config in CONFIG_ORDER:
        if config in summary:
            print(config, summary[config])
    print(f"\nWrote {summary_csv}\nWrote {summary_json}\nWrote {plot_path}")


if __name__ == "__main__":
    main()
