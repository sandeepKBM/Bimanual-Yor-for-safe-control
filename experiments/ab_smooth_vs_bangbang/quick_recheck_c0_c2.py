"""Task 5: quick C0 + C2 re-check with the corrected (multi-keypoint,
contype/conaffinity-fixed) collision proxy. Not the full C0-C3 sweep --
just enough to confirm whether the "naive methods fail" finding
(C0 collides, C2 also collides+drops the box) survives the fidelity fix.
"""
import json
import sys
sys.path.insert(0, "/common/users/ss5772/bimanual/experiments/ab_smooth_vs_bangbang")
import run_ab as ab


def main():
    summary = {}
    for condition in ["C0", "C2"]:
        print(f"=== running {condition} (corrected proxy) ===")
        log = ab.run_condition(condition)
        m = ab.compute_metrics(condition, log)
        summary[condition] = m
        print(json.dumps(m, indent=2))
    print("\n=== quick summary ===")
    for c in ["C0", "C2"]:
        m = summary[c]
        print(f"{c}: collision={m['collision']} min_h={m['min_h']:.4f} min_dist={m['min_dist']:.4f} "
              f"box_dropped={m['box_dropped']} box_drift_max={m['box_drift_trans_max']:.4f}")


if __name__ == "__main__":
    main()
