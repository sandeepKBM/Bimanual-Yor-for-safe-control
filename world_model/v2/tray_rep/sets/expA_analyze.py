"""sets/expA_analyze.py -- verdict of Experiment A (bimanual_data/tray_rep/sets/expA/PREREG_A_FROZEN.md). SAVED DATA ONLY. CPU.

Is the analytic set filter F5 good enough to skip HJ? F5 against the bridge arms (F0-F4, re-run at their frozen VAL choices so that
every arm has path progress), 600 TEST carries, seeds 0-2 where the arm has a noisy input or a model seed.

  PRIMARY (piper_trk):  K1  95% CI of drop(F5) - drop(F3 Rc3k) entirely above 0      -> KILL
                        K2  mean progress(F5) <= mean progress(F2)  (matches F3 only by freezing) -> KILL
                        otherwise PASS.
  SECONDARY (piper, noise-free): the same two tests, reported with the same rule.
  Ablation: F5 with the raw friction map (calib=0) - F5, both setups.  mu bins: drop rate per arm in 5 log bins.
  Reproduction: the re-run F0-F4 records against the stored bridge TEST records (old fields only).

Bootstrap: hierarchical paired; per iteration resample the 600 carries and the 3 seeds with replacement; arms with no noisy input and
no model seed (F0, F2, and every arm in the noise-free setup except F1/F3) are seed-independent and are replicated across seeds.
B = 10000, numpy default_rng(20261008), percentile 95% interval.

  python -m world_model.v2.tray_rep.sets.expA_analyze [--root R]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep.bridge import metrics as M
from world_model.v2.tray_rep.bridge import run as R

ROOT = Path("/common/users/ss5772/bimanual_data/tray_rep/sets/expA")
STORED = Path("/common/users/ss5772/bimanual_data/tray_rep/bridge")
B, RNG_SEED, N_TEST = 10000, 20261008, 600
MU_EDGES = np.exp(np.linspace(np.log(5e-4), np.log(6e-3), 6))
OLD_FIELDS = ("dropped", "drop_tick", "n_ticks", "n_interv", "dev_norm_sum")

# arm -> (spec, seeds, n_eps); frozen VAL choices from bridge/main/chosen_params.json
ARMS = {
    "piper_trk": {
        "F0": ("F0", (0,), N_TEST),
        "F1": ("F1:l_align=4,l_prox=0", (0, 1, 2), N_TEST),
        "F2": ("F2:theta_max=0.05,a_cap=0.4", (0,), N_TEST),
        "F3_N0": ("F3:arm=N0,tau=0.1", (0, 1, 2), N_TEST),
        "F3_Rc3k": ("F3:arm=Rc3k,tau=0.1", (0, 1, 2), N_TEST),
        "F4": ("F4:check_every=5", (0,), 100),
        "F5": ("F5", (0, 1, 2), N_TEST),
        "F5_raw": ("F5:calib=0", (0, 1, 2), N_TEST),
    },
    "piper": {
        "F0": ("F0", (0,), N_TEST),
        "F1": ("F1:l_align=4,l_prox=10", (0,), N_TEST),
        "F2": ("F2:theta_max=0.05,a_cap=0.4", (0,), N_TEST),
        "F3_N0": ("F3:arm=N0,tau=0.05", (0, 1, 2), N_TEST),
        "F3_Rc3k": ("F3:arm=Rc3k,tau=0.05", (0, 1, 2), N_TEST),
        "F4": ("F4:check_every=5", (0,), 100),
        "F5": ("F5", (0,), N_TEST),
        "F5_raw": ("F5:calib=0", (0,), N_TEST),
    },
}


def load(root, setup, spec, seeds, n, split="test", tag="main"):
    """{seed: [record by ep 0..n-1]}; None if any record is missing."""
    out = {}
    for s in seeds:
        _, pj, _ = R.paths(root, tag, split, setup, spec, s)
        rows = {r["ep"]: r for r in R.read_jsonl(pj)} if pj.exists() else {}
        if not all(e in rows for e in range(n)):
            return None
        out[s] = [rows[e] for e in range(n)]
    return out


def mat(recs, key, seeds=(0, 1, 2)):
    """[n_carries, 3] array of a per-carry field; a seed-independent arm (one seed) is replicated across the 3 seeds."""
    cols = [np.array([float(r[key]) for r in recs[s]]) for s in sorted(recs)]
    if len(cols) == 1:
        cols = cols * len(seeds)
    return np.stack(cols, 1)


def boot_diff(XA, XB, rng):
    """hierarchical paired bootstrap of mean(XA) - mean(XB); XA, XB [C, S]. Returns (point, lo, hi)."""
    C, S = XA.shape
    D = XA - XB
    vals = np.empty(B)
    for b in range(B):
        c = rng.integers(0, C, C)
        s = rng.integers(0, S, S)
        vals[b] = D[c][:, s].mean()
    return float(D.mean()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mu_bins(recs):
    mu = np.array([r["mu"] for r in recs[sorted(recs)[0]]])
    d = mat(recs, "dropped").mean(1)
    idx = np.clip(np.digitize(mu, MU_EDGES[1:-1]), 0, len(MU_EDGES) - 2)
    return [(float(d[idx == k].mean()) if (idx == k).any() else float("nan"), int((idx == k).sum())) for k in range(len(MU_EDGES) - 1)]


def reproduction(root, setup, spec, seeds, n):
    """count records whose OLD fields differ from the stored bridge TEST run (same spec, setup, seed)."""
    a, b = load(root, setup, spec, seeds, n), load(STORED, setup, spec, seeds, n)
    if a is None or b is None:
        return None
    bad = sum(any(ra[k] != rb[k] for k in OLD_FIELDS) for s in seeds for ra, rb in zip(a[s], b[s]))
    return dict(mismatched=int(bad), n=int(n * len(seeds)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args()
    root = Path(a.root)
    rng = np.random.default_rng(RNG_SEED)
    res, L = {}, []
    for setup in ("piper_trk", "piper"):
        L += [f"## {setup} ({'PRIMARY' if setup == 'piper_trk' else 'secondary'})", "",
              "| arm | spec | seeds | n | drop | interv | dev (norm) | progress | goal | ms/tick mean (max) |", "|---|---|---|---|---|---|---|---|---|---|"]
        data, rs = {}, {}
        for arm, (spec, seeds, n) in ARMS[setup].items():
            recs = load(root, setup, spec, seeds, n)
            if recs is None:
                L.append(f"| {arm} | `{spec}` | {seeds} | {n} | MISSING | | | | | |")
                continue
            data[arm] = recs
            sm = M.summarize([r for s in seeds for r in recs[s]])
            drop = sm["drop"]["rate"]
            g = sm["goal"]["rate"] if "goal" in sm else None
            ms = [r.get("x_f5_ms_mean") for s in seeds for r in recs[s] if "x_f5_ms_mean" in r]
            mx = [r.get("x_f5_ms_max") for s in seeds for r in recs[s] if "x_f5_ms_max" in r]
            rs[arm] = dict(spec=spec, n=n, seeds=list(seeds), drop=drop, interv=sm["interv_rate"], dev=sm["mean_dev_norm"],
                           progress=sm.get("progress"), goal=g, ms_mean=float(np.mean(ms)) if ms else None,
                           ms_max=float(np.max(mx)) if mx else None, mu_bins=mu_bins(recs))
            L.append(f"| {arm} | `{spec}` | {list(seeds)} | {n} | {100*drop:.1f}% | {100*sm['interv_rate']:.1f}% | "
                     f"{sm['mean_dev_norm']:.3f} | {sm.get('progress', float('nan')):.3f} | {g if g is None else round(float(g), 3)} | "
                     + (f"{np.mean(ms):.1f} ({np.max(mx):.0f})" if ms else "-") + " |")
        out = dict(arms=rs)
        # contrasts on the 600 paired carries
        con = {}
        pairs = [("F5", "F3_Rc3k", "dropped"), ("F5", "F3_Rc3k", "progress"), ("F5", "F2", "progress"), ("F5", "F2", "dropped"),
                 ("F5", "F1", "dropped"), ("F5", "F3_N0", "dropped"), ("F5_raw", "F5", "dropped"), ("F5_raw", "F5", "progress"),
                 ("F5", "F0", "dropped")]
        for x, y, k in pairs:
            if x in data and y in data and ARMS[setup][x][2] == ARMS[setup][y][2]:
                con[f"{x}-{y}:{k}"] = boot_diff(mat(data[x], k), mat(data[y], k), rng)
        if "F5" in data and "F4" in data:                      # F4 ran on the first 100 carries
            f5 = {s: v[:100] for s, v in data["F5"].items()}
            con["F5-F4:dropped(first100)"] = boot_diff(mat(f5, "dropped"), mat(data["F4"], "dropped"), rng)
        out["contrasts"] = con
        L += ["", "| contrast | point | 95% CI |", "|---|---|---|"]
        for k, (p, lo, hi) in con.items():
            sc = 100 if k.endswith("dropped") or "dropped(" in k else 1
            u = " pp" if sc == 100 else ""
            L.append(f"| {k} | {sc*p:+.3f}{u} | [{sc*lo:+.3f}, {sc*hi:+.3f}] |")
        # verdict
        k1 = k2 = None
        if "F5-F3_Rc3k:dropped" in con:
            k1 = con["F5-F3_Rc3k:dropped"][1] > 0
        if "F5" in rs and "F2" in rs and rs["F5"]["progress"] is not None and rs["F2"]["progress"] is not None:
            k2 = rs["F5"]["progress"] <= rs["F2"]["progress"]
        if k1 is None or k2 is None:
            v = "INCOMPLETE"
        else:
            v = "KILL" if (k1 or k2) else "PASS"
        out["verdict"] = dict(K1_F5_worse_than_F3=k1, K2_F5_freezes=k2, verdict=v)
        pred = {}
        if "F5" in rs and "F3_Rc3k" in rs:
            pred = dict(drop_le_F3=rs["F5"]["drop"] <= rs["F3_Rc3k"]["drop"], drop_le_13_6=rs["F5"]["drop"] <= 0.136,
                        dev_le_0_12=rs["F5"]["dev"] <= 0.12,
                        progress_ge_F3=(rs["F5"]["progress"] or 0) >= (rs["F3_Rc3k"]["progress"] or 0),
                        ms_mean_lt_20=(rs["F5"]["ms_mean"] or 1e9) < 20.0)
        out["predictions"] = pred
        L += ["", f"**{'PRIMARY' if setup == 'piper_trk' else 'Secondary'} verdict ({setup}): {v}** "
                  f"(K1 F5 significantly worse than F3 Rc3k on drops: {k1}; K2 F5 progress <= F2 progress: {k2})",
              f"Predictions: {pred}", "", "Drop rate by mu bin (edges " + ", ".join(f"{x:.2g}" for x in MU_EDGES) + "):", "",
              "| arm | " + " | ".join(f"bin {k}" for k in range(5)) + " |", "|---" * 6 + "|"]
        for arm, r in rs.items():
            L.append(f"| {arm} | " + " | ".join(f"{100*d:.1f}% ({n})" for d, n in r["mu_bins"]) + " |")
        rep = {arm: reproduction(root, setup, *ARMS[setup][arm]) for arm in ("F0", "F1", "F2", "F3_N0", "F3_Rc3k", "F4")}
        out["reproduction"] = rep
        L += ["", "Reproduction of the stored bridge TEST records (old fields " + ", ".join(OLD_FIELDS) + "): "
              + "; ".join(f"{k} {v['mismatched']}/{v['n']} differ" if v else f"{k} n/a" for k, v in rep.items()), ""]
        res[setup] = out
    md = "\n".join(L) + "\n"
    (root / "analysis").mkdir(parents=True, exist_ok=True)
    (root / "analysis" / "tables_A.md").write_text(md)
    (root / "analysis" / "summary_A.json").write_text(json.dumps(res, indent=1, default=float))
    print(md, flush=True)


if __name__ == "__main__":
    main()
