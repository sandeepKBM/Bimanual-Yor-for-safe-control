"""agentchord_probe/summarize.py -- table of run results from runs/<run>/episodes.jsonl (machine-written; no hand-entered numbers).

Per run: n, their success, strict success (pour tasks), injected drops, episodes with a monitor firing, crashes, silent failures.
Silent failure = their_success or strict success is False AND no monitor fired in that episode.
Usage: python -m world_model.v2.agentchord_probe.summarize RUN_GLOB [--json OUT]
"""
import argparse, glob, json, math, os

ROOT = "/data/local/ss5772/agentchord/runs"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(run_dir):
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "episodes.jsonl"))]
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    n = len(rows)
    their = sum(bool(r.get("their_success")) for r in rows)
    has_strict = any("strict_success" in r for r in rows)
    strict = sum(bool(r.get("strict_success")) for r in rows) if has_strict else None
    fired = sum(r.get("n_monitor_fired", 0) > 0 for r in rows)
    inj_eps = sum(r.get("n_injected", 0) > 0 for r in rows)
    crash = sum(r.get("error") is not None for r in rows)
    fail = [r for r in rows if not r.get("their_success") or (has_strict and not r.get("strict_success"))]
    # silent = failed with no monitor firing AND no exception (a crash is loud, so it is counted under crashes only)
    silent = sum(r.get("n_monitor_fired", 0) == 0 and r.get("error") is None for r in fail)
    hw = open(os.path.join(run_dir, "hw_audit.txt")).read().strip() if os.path.exists(os.path.join(run_dir, "hw_audit.txt")) else "?"
    return dict(run=os.path.basename(run_dir), task=meta["task"], p=meta["p"], mode=meta["drop_mode"], n=n,
                their=their, their_ci=wilson(their, n), strict=strict, strict_ci=wilson(strict, n) if has_strict else None,
                eps_with_drop=inj_eps, drops=sum(r.get("n_injected", 0) for r in rows), eps_monitor_fired=fired, crashes=crash,
                failures=len(fail), silent_failures=silent, errors=sorted({(r.get("error") or "")[:90] for r in rows if r.get("error")}),
                mean_wall_s=sum(r.get("wall_s", 0) for r in rows) / max(n, 1), hw_audit=hw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glob")
    ap.add_argument("--json")
    a = ap.parse_args()
    out = [summarize(d) for d in sorted(glob.glob(os.path.join(ROOT, a.glob))) if os.path.exists(os.path.join(d, "episodes.jsonl"))]
    print("| run | task | p | n | their success | strict success | eps w/ drop | eps monitor fired | crashes | silent fails | hw |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in out:
        st = f"{s['strict']}/{s['n']}" if s["strict"] is not None else "n/a"
        print(f"| {s['run']} | {s['task']} | {s['p']} | {s['n']} | {s['their']}/{s['n']} | {st} | {s['eps_with_drop']} | "
              f"{s['eps_monitor_fired']} | {s['crashes']} | {s['silent_failures']} | {s['hw_audit'][:5]} |")
    for s in out:
        if s["errors"]:
            print(f"- {s['run']} errors: {s['errors']}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
