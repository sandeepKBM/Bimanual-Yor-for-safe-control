"""boed/smoke_checks.py -- plumbing and determinism checks of the BOED harness on a handful of VAL carries. CPU ONLY, SIMULATION ONLY.

  python -m world_model.v2.tray_rep.boed.smoke_checks [--n 6] [--workers 2]     (needs <out>/smoke/table_<setup>.json from `launch_local --stage smoke`)

Checks (both setups unless noted):
  1. P2 == bridge F3: the BOED stored-carry policy reproduces the bridge's F3 run (drop tick, interventions, path progress) carry for carry.
  2. determinism: P3, P4, P1 and a table cell give identical records with 1 worker and with several workers (all fields but wall time).
  3. resume: a run of n/2 carries followed by a run of n equals one run of n.
  4. the certificate does not touch the real env (state, tracker, obs identical before and after `certify`).
  5. the table fit is monotone along mu_roll; the intensity rule picks the fastest meeting eps (and the slowest if none).
  6. the belief: prior is uniform; a synthetic likelihood moves it; calibration helpers agree.
  7. bridge: the new path-progress fields are present and the old ones unchanged (F0 completion equals the stored-carry value).
  8. no hardware code was imported.
Writes <out>/smoke/checks.json. Exit code 0 = all pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep.boed import run as R
from world_model.v2.tray_rep.boed import policy as PO
from world_model.v2.tray_rep.boed import model as MD

TIMING = ("wall_s",)


def recs(out, tag, split, setup, pol, seed):
    rows = R.read_all(out, tag, split, setup, pol, seed)
    return [{k: v for k, v in r.items() if k not in TIMING} for r in rows]


def go(out, tag, pol, setup, seed, n, workers, table, split="val", **kw):
    return R.run(pol, setup, seed, split, n, workers, tag, out, table=table, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=str(R.OUT))
    a = ap.parse_args()
    out = Path(a.out)
    sm = out / "smoke"
    n, W = a.n, a.workers
    res, ok_all = {}, True

    def check(name, ok, detail=""):
        nonlocal ok_all
        ok_all &= bool(ok)
        res[name] = dict(ok=bool(ok), detail=detail)
        print(("PASS " if ok else "FAIL ") + name, detail, flush=True)

    # 1. P2 == bridge F3 ---------------------------------------------------------------------------------------------
    from world_model.v2.tray_rep.bridge import run as BR
    for setup in ("piper", "piper_trk"):
        tau = R.default_tau(setup)
        go(out, "chk_p2", "P2", setup, 0, n, 1, None)
        BR.run(f"F3:arm=Rc3k,tau={tau:g}", setup, 0, "val", n, 1, "chk_bridge", out)
        a_ = recs(out, "chk_p2", "val", setup, "P2", 0)
        from world_model.v2.tray_rep.bridge.filters import spec_tag
        bj = BR.read_jsonl(out / "chk_bridge" / "val" / setup / f"{spec_tag(f'F3:arm=Rc3k,tau={tau:g}')}__s0.jsonl")
        b_ = sorted(bj, key=lambda r: r["ep"])
        fields = ("drop_tick", "n_interv", "dropped", "progress", "goal_tick", "n_ticks")
        diff = [(x["ep"], f) for x, y in zip(a_, b_) for f in fields if x[f] != y[f]]
        check(f"p2_equals_bridge_f3_{setup}", len(a_) == len(b_) == n and not diff, f"{len(a_)} carries, {sum(x['dropped'] for x in a_)} drops, mismatches {diff[:4]}")

    # 2. determinism (1 worker vs several) -------------------------------------------------------------------------------
    for setup in ("piper", "piper_trk"):
        tab = sm / f"table_{setup}.json"
        if not tab.exists():
            check(f"table_exists_{setup}", False, f"{tab} missing: run launch_local --stage smoke first")
            continue
        for pol in ("P3", "P4", "P1", "FIX:I=0.55,j=1"):
            go(out, "chk_a", pol, setup, 0, n, 1, tab, n_nodes=4)
            go(out, "chk_b", pol, setup, 0, n, W, tab, n_nodes=4)
            ra, rb = recs(out, "chk_a", "val", setup, pol, 0), recs(out, "chk_b", "val", setup, pol, 0)
            ra, rb = sorted(ra, key=lambda r: r["ep"]), sorted(rb, key=lambda r: r["ep"])
            check(f"determinism_{setup}_{pol.split(':')[0]}", ra == rb and len(ra) == n, f"{len(ra)} vs {len(rb)} records identical={ra == rb}")

    # 3. resume ----------------------------------------------------------------------------------------------------------
    tab = sm / "table_piper_trk.json"
    if tab.exists():
        go(out, "chk_r", "P3", "piper_trk", 0, n // 2, W, tab, n_nodes=4)
        go(out, "chk_r", "P3", "piper_trk", 0, n, W, tab, n_nodes=4)
        rr = sorted(recs(out, "chk_r", "val", "piper_trk", "P3", 0), key=lambda r: r["ep"])
        rf = sorted(recs(out, "chk_a", "val", "piper_trk", "P3", 0), key=lambda r: r["ep"])
        check("resume_equals_fresh", rr == rf and len(rr) == n, f"{len(rr)} records identical={rr == rf}")

    # 4. the certificate does not touch the real env ---------------------------------------------------------------------
    from world_model.v2.tray_rep.boed import episode as EP, probe as PB
    for setup in ("piper", "piper_trk"):
        ctx = EP.Ctx(setup, 0, "val")
        run = EP.Run(ctx, ctx.store.ep_info(1)["seed"])
        run.kf = PB.make_kf(ctx.trk, ctx.noisefree)
        run.kf.feed(run.obs[0], 0)
        before = ctx.renv.snapshot().arrays()
        o_before = run.obs[:1].copy()
        run.certify(False)
        after = ctx.renv.snapshot().arrays()
        same = all(np.array_equal(before[k], after[k]) for k in before)
        check(f"certificate_leaves_real_env_untouched_{setup}", same and np.array_equal(o_before, run.obs[:1]) and run.n == 0,
              f"cert_ok={run.info['cert_ok']} dmin={run.info['cert_dmin']:.3f} label={run.info['cert_label']}")

    # 5. table fit and rule ----------------------------------------------------------------------------------------------
    I = np.array(PO.IGRID); nodes = PO.NODES
    rng = np.random.default_rng(0)
    d = np.array([[min(1, 0.5 * i ** 2 * (0.0006 / m) ** 1.5) for m in nodes] for i in I])
    n_ = np.full(d.shape, 30.0)
    k_ = rng.binomial(30, d)
    T = PO.Table(I, nodes, k_, n_)
    mono = bool(np.all(np.diff(T.drop, axis=1) <= 1e-9))
    i_all = T.choose(np.full(MD.N_GRID, 1 / MD.N_GRID), 1.01)
    i_none = T.choose(PO.w_worst(), -1.0)
    check("table_fit_monotone", mono, "drop falls with mu after the fit (speed axis left free)")
    check("intensity_rule", i_all == 0 and i_none == len(I) - 1, f"eps 101% -> fastest (idx {i_all}); eps<0 -> slowest (idx {i_none})")

    # 6. belief ----------------------------------------------------------------------------------------------------------
    b = MD.Belief()
    ll = -0.5 * ((MD.LOG_MU - np.log(0.002)) / 0.2) ** 2
    b.update(ll)
    check("belief_prior_and_update", abs(MD.Belief().w.sum() - 1) < 1e-12 and np.allclose(MD.Belief().w, 1 / MD.N_GRID) and abs(b.mean_log() - np.log(0.002)) < 0.2
          and b.covers(0.002, 0.9) and b.sd_log() < 0.3, f"mean {np.exp(b.mean_log()):.5f} sd {b.sd_log():.3f}")

    # 7. bridge fields ---------------------------------------------------------------------------------------------------
    from world_model.v2.tray_rep.bridge import run as BR2
    BR2.run("F0", "piper", 0, "val", 4, 1, "chk_bridge", out)
    r0 = BR2.read_jsonl(out / "chk_bridge" / "val" / "piper" / "F0__s0.jsonl")
    old = ("ep", "seed", "L", "onset_stored", "dropped", "drop_tick", "tunnel", "n_ticks", "n_interv", "dev_raw_sum", "dev_norm_sum", "completion", "mu")
    new = ("path_len_nom", "path_len_done", "progress", "goal_tick", "goal_reached")
    check("bridge_old_and_new_fields", all(all(k in r for k in old + new) for r in r0) and all(r["drop_tick"] == r["onset_stored"] or r["onset_stored"] < 0 for r in r0)
          and all(0.0 <= r["progress"] <= 1.0 for r in r0), f"progress {[round(r['progress'], 3) for r in r0]}")

    # 8. no hardware ------------------------------------------------------------------------------------------------------
    bad = [m for m in sys.modules if m.startswith("YOR.robot") or m.startswith("YOR.extra") or m.split(".")[0] in ("piper_sdk", "can", "xarm", "rclpy", "serial")]
    check("no_hardware_modules_imported", not bad, str(bad))

    sm.mkdir(parents=True, exist_ok=True)
    (sm / "checks.json").write_text(json.dumps(res, indent=1))
    print("ALL PASS" if ok_all else "SOME FAILED", flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
