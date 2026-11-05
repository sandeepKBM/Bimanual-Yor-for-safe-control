"""nero/contents/same_worlds.py -- ours (probe -> admit -> act) vs AgentChord-style (act, no checks) on the SAME worlds.  SIMULATION ONLY.
USER 11-05: "count payload rating in U and run same worlds in both methods".

Worlds: probe._world(task, probe._seed("e2e", task, i)) -- uniform on the full prior (probe.PRIOR), the same worlds as the 11-03
probe e2e run (i < 20), so ours can be checked against that run.
U for BOTH methods = carry.py's U plus the per-claw payload row (strict_claw=True, T_cap=None: |Fz| > (3 kg - gripper) g, from the
weld constraint wrench, unfiltered, every control step; no twist row).  Set in each worker by wrapping CarryEpisode.
  theirs = baseline_ungated._trial: carry at a_nom (0.5 m/s^2, ASSUMED), pose monitors, no force/payload check, no backups.
           Payload feedforward mass = prior mid-contents (ASSUMED: it has no estimate of the contents).
  ours   = probe._e2e_job: probe ladder (eps MEASURED, CALIB.json) -> static admission on the probe's set -> carry with stored
           backups, or refuse and let go in place.
    python -m world_model.v2.nero.contents.same_worlds run --tasks box,tray6 --n 20 --procs 4 --out DIR
    python -m world_model.v2.nero.contents.same_worlds summary --out DIR
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import time

CALIB = "/data/local/ss5772/contents/probe_calib_2026-11-03/CALIB.json"


def _init_worker():
    """payload row in U for both methods: every CarryEpisode built in this worker gets strict_claw=True (T_cap stays None)."""
    from world_model.v2.nero.contents import carry as CR
    base = CR.CarryEpisode
    if getattr(base, "_same_worlds_strict", False):
        return

    class _Strict(base):
        _same_worlds_strict = True

        def __init__(self, *a, **k):
            k["strict_claw"] = True
            super().__init__(*a, **k)
    CR.CarryEpisode = _Strict


def CR_resolve():
    from world_model.v2.nero.contents import carry as CR
    return CR.resolve_planner(None)


def _job(args):
    method, task, i, eps_lo, eps_free = args
    _init_worker()
    from world_model.v2.nero.contents import probe as P, baseline_ungated as BU, carry_cert as CC
    seed = P._seed("e2e", task, i)
    if method == "ours":
        r = P._e2e_job((task, i, eps_lo, eps_free))
    else:
        BU.seed_of = lambda t, c, k: P._seed("e2e", t, k)          # same world as ours: CC.sample_world(task, PRIOR[task], rng(seed))
        lo, hi = P.PRIOR[task]
        r = BU._trial((task, (lo, hi), i, 0.5 * (lo + hi), True, CC.WELD_STIFF))
    r.update(method=method, seed_e2e=seed, planner=CR_resolve())
    return r


def _first_U(r):
    if r.get("method") == "theirs":
        v = r.get("first_U")
        return v and f"{v.get('seg')}:{v.get('kind')}"
    if r.get("probe_viol"):
        return f"probe:{r['probe_viol'].get('kind')}"
    for s in r.get("carry", {}).get("segments", []):
        v = s.get("nominal", {}).get("viol")
        if v:
            return f"{s.get('seg')}:{v.get('kind')}"
    return None


def summary(out):
    out = pathlib.Path(out)
    rows = [json.loads(l) for l in open(out / "trials.jsonl")]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["task"], r["method"])].append(r)
    lines = ["# same worlds: ours (probe -> admit -> act) vs AgentChord-style (act)  -- machine-written by same_worlds.summary", "",
             "U includes the per-claw payload row (|Fz| > (3 kg - gripper) g, unfiltered). Worlds: probe e2e seeds, full prior.", "",
             "| task | method | n | errors | carried (no U) | refused | any U | U kinds (first per run) | truth in probe set |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (task, method), rs in sorted(by.items()):
        ok = [r for r in rs if "error" not in r]
        kinds = collections.Counter(_first_U(r) for r in ok if r.get("any_U"))
        if method == "ours":
            carried = sum(r.get("outcome") == "carried" for r in ok)
            refused = sum(str(r.get("outcome", "")).startswith("refused") for r in ok)
            sound = f"{sum(bool(r.get('theta_sound')) for r in ok)}/{len(ok)}"
        else:
            carried = sum(not r.get("any_U") for r in ok)
            refused, sound = 0, "-"
        lines.append(f"| {task} | {method} | {len(rs)} | {len(rs) - len(ok)} | {carried} | {refused} | {sum(bool(r.get('any_U')) for r in ok)} "
                     f"| {dict(kinds) or '-'} | {sound} |")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--tasks", default="box,tray6")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--calib", default=CALIB)
    ap.add_argument("--planner", choices=("scripted", "closed_chain"), default=None,
                    help="plan BOTH methods execute and are admitted on (carry.resolve_planner; default = env BIMANUAL_PLANNER, else scripted = the legacy PLANS)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "summary":
        return summary(a.out)
    if a.planner is not None:
        os.environ["BIMANUAL_PLANNER"] = a.planner               # workers are spawned after this: they inherit it (one plan for ours and theirs)
    from world_model.v2.nero.contents import carry as CR0
    planner = CR0.resolve_planner(None)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cal = json.loads(pathlib.Path(a.calib).read_text())["eps"]          # as probe.main e2e reads it
    eps_lo, eps_free = float(cal["EPS_LO"]), float(cal["EPS_FREE"])
    jobs = [(m, t, i, eps_lo, eps_free) for i in range(a.n) for t in a.tasks.split(",") for m in ("theirs", "ours")]
    (out / "CONFIG.json").write_text(json.dumps(dict(tasks=a.tasks, n=a.n, procs=a.procs, calib=a.calib, eps_lo=eps_lo,
                                                     eps_free=eps_free, u_payload="strict_claw=True, T_cap=None", planner=planner), indent=1))
    print(f"{len(jobs)} jobs", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs, initializer=_init_worker, maxtasksperchild=3) as pool, \
            open(out / "trials.jsonl", "a") as f:
        for r in pool.imap_unordered(_job, jobs):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            print(f"{r.get('method')} {r.get('task')} i={r.get('i')} anyU={r.get('any_U')} {r.get('outcome', '')} "
                  f"{'ERR ' + r['error'] if 'error' in r else ''} t={time.time() - t0:.0f}s", flush=True)
    print(f"wall {time.time() - t0:.0f} s", flush=True)
    summary(out)


if __name__ == "__main__":
    main()
