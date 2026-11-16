#!/usr/bin/env python
"""score_ladder.py -- scorer for the WM-requirement spill-predictor LADDER
(experiments/wm_requirement/spill_predictors.py). Consumes a .npz produced by
collect_ladder.py and answers "how powerful must the WM be?" by measuring
where the accuracy-vs-rung curve SATURATES.

For each episode, runs every rung in spill_predictors.LADDER and scores it
against the ground-truth ball oracle via spill_predictors.score_predictor
(per-episode AUROC + lead-time-to-1.0-threshold + whether-detected).

Aggregation across episodes (per rung):
  - AUROC (pooled): scores and per-step spill-warning labels are POOLED
    across ALL episodes (spilled AND never-spilled) into one vector pair
    before computing AUROC via spill_predictors._auroc -- NOT a naive mean of
    per-episode AUROCs, which would have to silently drop every never-spilled
    episode (an episode with no positive steps has an undefined per-episode
    AUROC). Pooling is exactly how a never-spilled episode's steps still
    count: they add pure negatives to the SAME negative pool that spilled
    episodes' positives must be ranked above, so a rung that is trigger-happy
    on calm episodes gets penalized even though those episodes never
    "resolve" a per-episode AUROC of their own.
  - detection rate: fraction of SPILLED episodes where score crosses >=1.0
    (the analytic spill threshold) at some point before onset.
  - median lead time (steps, ms): median, over DETECTED spill episodes only,
    of score_predictor's lead_steps (how many steps before onset the score
    first crossed 1.0).

Also runs a small grid scan of spill_predictors.CupConfig's rung-2 knobs
(omega in {0.5x, 1x, 2x} of the Abramson-formula value, zeta in
{0.05, 0.15, 0.30} -- 9 combos) and reports the best by pooled AUROC (ties
broken by detection rate, then median lead-time).

Run:
  /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python \\
    experiments/wm_requirement/score_ladder.py \\
    --data experiments/wm_requirement/traj_smoke.npz --lead-window-steps 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SELF_DIR = Path(__file__).resolve().parent
if str(SELF_DIR) not in sys.path:
    sys.path.insert(0, str(SELF_DIR))

from spill_predictors import LADDER, CupConfig, score_predictor, _auroc  # noqa: E402  (ladder physics; DO NOT modify)


def corrected_onset(spilled_curve: np.ndarray, settle: int = 200) -> int | None:
    """Spill ONSET = first step (after a settle window) where the ball-out count
    rises PERMANENTLY above its settled post-lift baseline. CRITICAL (2026-09-10,
    see memory wm-requirement-ladder): count_spilled reads 6-8/8 balls as "out"
    from step 0 because the free balls spill during the initial cup LIFT, not the
    dodges -- so the baseline is that lift-loss floor (median over the settle
    window), and a genuine DYNAMIC (dodge-induced) spill is a PERMANENT increase
    ABOVE it. Episodes whose count never rises above the floor had only lift-loss
    (no dynamic spill) -> onset None -> pure negatives. This replaces the
    collector's stored onset, which used base=count[0] and mislabeled episodes."""
    c = np.asarray(spilled_curve).astype(int)
    T = len(c)
    if T == 0:
        return None
    s = min(settle, T - 1)
    base = int(np.round(np.median(c[:s]))) if s > 0 else int(c[0])
    for t in range(s, T):
        if c[t] > base and int(c[t:].min()) > base:
            return t
    return None


def load_episodes(data_path: str) -> list[dict]:
    npz = np.load(data_path)
    seeds = npz["seeds"]
    dt = float(npz["dt"])
    has_curve = f"spilled_curve_{int(seeds[0])}" in npz.files if len(seeds) else False
    episodes = []
    for s in seeds:
        s = int(s)
        # Prefer the CORRECTED onset recomputed from the raw per-step spill curve
        # (permanent rise above the settled lift-loss floor); fall back to the
        # collector's stored onset only for old npz files without the raw curve.
        if has_curve:
            onset = corrected_onset(npz[f"spilled_curve_{s}"])
        else:
            onset_raw = int(npz[f"onset_{s}"])
            onset = None if onset_raw < 0 else onset_raw
        episodes.append(dict(
            seed=s,
            n_axis=npz[f"n_axis_{s}"],
            a_cup=npz[f"a_cup_{s}"],
            onset=onset,
            T=int(npz[f"T_{s}"]),
            spilled_max=int(npz[f"spilled_max_{s}"]),
            dt=dt,
        ))
    return episodes


def _labels_for(T: int, onset: int | None, lead_window_steps: int) -> np.ndarray:
    """SAME label construction as spill_predictors.score_predictor's
    internal logic (duplicated here, not imported, since it's a local block
    inside that function) -- 1 for the lead_window_steps steps immediately
    before spill onset, 0 elsewhere (all 0 if the episode never spills)."""
    labels = np.zeros(T, dtype=int)
    if onset is not None and onset > 0:
        lo = max(0, onset - lead_window_steps)
        labels[lo:onset] = 1
    return labels


def score_rung(fn, episodes: list[dict], cfg: CupConfig, lead_window_steps: int) -> dict:
    pooled_scores = []
    pooled_labels = []
    leads = []          # lead_steps for DETECTED spilled episodes
    n_spilled = 0
    n_detected = 0
    for ep in episodes:
        scores = fn(ep["n_axis"], ep["a_cup"], ep["dt"], cfg)
        res = score_predictor(scores, ep["onset"], ep["T"], lead_window_steps)
        labels = _labels_for(ep["T"], ep["onset"], lead_window_steps)
        pooled_scores.append(scores)
        pooled_labels.append(labels)
        if res["spilled"]:
            n_spilled += 1
            if res["lead_steps"] is not None:
                n_detected += 1
                leads.append(res["lead_steps"])
    pooled_scores = np.concatenate(pooled_scores) if pooled_scores else np.array([])
    pooled_labels = np.concatenate(pooled_labels) if pooled_labels else np.array([], dtype=int)
    auroc = _auroc(pooled_scores, pooled_labels)
    detect_rate = (n_detected / n_spilled) if n_spilled else float("nan")
    median_lead_steps = float(np.median(leads)) if leads else None
    return dict(auroc=auroc, detect_rate=detect_rate, median_lead_steps=median_lead_steps,
                n_spilled=n_spilled, n_detected=n_detected, n_total=len(episodes))


def _fmt_lead(median_lead_steps, dt):
    if median_lead_steps is None:
        return "n/a"
    return f"{median_lead_steps:.0f} steps ({median_lead_steps * dt * 1000.0:.0f} ms)"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--lead-window-steps", type=int, default=40)
    args = ap.parse_args()

    episodes = load_episodes(args.data)
    dt = episodes[0]["dt"] if episodes else 0.005
    n_spilled_total = sum(1 for ep in episodes if ep["spilled_max"] > 0)
    print(f"loaded {len(episodes)} episodes from {args.data} "
          f"({n_spilled_total}/{len(episodes)} spilled), dt={dt*1000:.1f} ms, "
          f"lead_window_steps={args.lead_window_steps} ({args.lead_window_steps*dt*1000:.0f} ms)\n")

    cfg_default = CupConfig()
    print(f"[cfg] default: omega={cfg_default.omega_slosh():.1f} rad/s, zeta={cfg_default.zeta}, "
          f"theta_max={np.degrees(cfg_default.theta_max):.1f} deg\n")

    print("=== accuracy-vs-rung (default CupConfig) ===")
    header = f"{'rung':16s} {'AUROC(pooled)':>13s} {'detect_rate':>11s} {'median_lead':>22s} {'n_spilled':>9s}"
    print(header)
    print("-" * len(header))
    rows = {}
    for name, fn in LADDER.items():
        r = score_rung(fn, episodes, cfg_default, args.lead_window_steps)
        rows[name] = r
        print(f"{name:16s} {r['auroc']:>13.3f} {r['detect_rate']:>11.3f} "
              f"{_fmt_lead(r['median_lead_steps'], dt):>22s} {r['n_spilled']:>4d}/{r['n_total']:<4d}")
    print()

    # --- rung-2 mini hyperparameter scan ---
    omega_formula = CupConfig().omega_slosh()
    omega_mults = (0.5, 1.0, 2.0)
    zetas = (0.05, 0.15, 0.30)
    print("=== rung2_slosh mini-scan (omega in {0.5x,1x,2x} formula, zeta in {0.05,0.15,0.30}) ===")
    scan_header = f"{'omega_mult':>10s} {'omega':>8s} {'zeta':>6s} {'AUROC':>8s} {'detect_rate':>11s} {'median_lead':>22s}"
    print(scan_header)
    print("-" * len(scan_header))
    best = None
    fn_rung2 = LADDER["rung2_slosh"]
    for mult in omega_mults:
        for zeta in zetas:
            cfg = CupConfig(omega=mult * omega_formula, zeta=zeta)
            r = score_rung(fn_rung2, episodes, cfg, args.lead_window_steps)
            print(f"{mult:>10.1f} {cfg.omega_slosh():>8.1f} {zeta:>6.2f} {r['auroc']:>8.3f} "
                  f"{r['detect_rate']:>11.3f} {_fmt_lead(r['median_lead_steps'], dt):>22s}")
            key = (r["auroc"] if not np.isnan(r["auroc"]) else -1.0,
                   r["detect_rate"] if not np.isnan(r["detect_rate"]) else -1.0,
                   r["median_lead_steps"] if r["median_lead_steps"] is not None else -1.0)
            if best is None or key > best[0]:
                best = (key, mult, zeta, cfg, r)
    _, best_mult, best_zeta, best_cfg, best_r = best
    print(f"\nBEST rung2 config: omega_mult={best_mult}x (omega={best_cfg.omega_slosh():.1f} rad/s), "
          f"zeta={best_zeta}  ->  AUROC(pooled)={best_r['auroc']:.3f}, "
          f"detect_rate={best_r['detect_rate']:.3f}, median_lead={_fmt_lead(best_r['median_lead_steps'], dt)}")

    print("\n=== summary ===")
    a0, a1, a2 = rows["rung0_static"]["auroc"], rows["rung1_effgrav"]["auroc"], best_r["auroc"]
    print(f"rung0 (static tilt)       AUROC={a0:.3f}")
    print(f"rung1 (effective gravity) AUROC={a1:.3f}")
    print(f"rung2 (1st slosh, best)   AUROC={a2:.3f}")
    if not any(np.isnan(x) for x in (a0, a1, a2)):
        if a2 > a1 + 0.02 and a1 <= a0 + 0.02:
            print("-> rung2 closes a gap rung1 does NOT: the WM needs at least a "
                  "2nd-order oscillatory latent driven by cup acceleration.")
        elif a1 > a0 + 0.02 and a2 <= a1 + 0.02:
            print("-> rung1 closes the gap and rung2 adds little: a memoryless "
                  "(feedforward) effective-gravity signal suffices; no oscillator needed.")
        elif a2 <= a1 + 0.02 and a1 <= a0 + 0.02:
            print("-> none of the analytic rungs move the needle much: the balls' "
                  "true dynamics are higher-D/nonlinear than this ladder captures.")
        else:
            print("-> mixed/non-monotonic result on this (tiny) sample -- rerun on more seeds.")


if __name__ == "__main__":
    main()
