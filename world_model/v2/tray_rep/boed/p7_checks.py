"""boed/p7_checks.py -- checks and plumbing smoke of P7 / P7s. CPU ONLY, SIMULATION / SAVED DATA ONLY. VAL carries only (never TEST).

  python -m world_model.v2.tray_rep.boed.p7_checks --stage all --workers 2 [--out <new dir>]
Stages
  unit      the decision rule, the frozen tables (monotone along mu), no hardware module imported
  equiv     P1, P3, P6 unchanged: n VAL carries, both setups, frozen tables:  (a) new code vs a pristine copy of the code taken before the edits
            (--pristine <dir with world_model/...>), (b) new code vs the stored smoke records of the original harness (smoke tables)
  determ    P7 / P7s: 1 worker == 2 workers; a resumed run == a fresh run
  certvalid the set certificate at pickup vs the shadow certificate, and vs MuJoCo: for every carry run the REAL probe at the carry's own mu and at
            the worst mu; where the set certificate passes the true minimum edge distance must be >= 0 (no drop). Variants of the certificate.
  smoke     20 VAL carries x {P6, P3, P7, P7s}, piper_trk, seed 0 (plumbing, not evidence)
Writes <out>/checks/<stage>.json (+ smoke.md). The frozen tables are read from <frozen>/main/table_<setup>.json and never written.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep.boed import run as R
from world_model.v2.tray_rep.boed import policy as PO
from world_model.v2.tray_rep.boed import model as MD
from world_model.v2.tray_rep.boed import p7 as P7

FROZEN = K.ROOT / "boed"
OUT = K.ROOT / "boed_p7"
TIMING = ("wall_s",)
SETUPS = ("piper", "piper_trk")


def table_path(setup):
    return FROZEN / "main" / f"table_{setup}.json"


def recs(out, tag, split, setup, pol, seed=0):
    return sorted(({k: v for k, v in r.items() if k not in TIMING} for r in R.read_all(out, tag, split, setup, pol, seed)), key=lambda r: r["ep"])


def jeq(a, b):
    """record lists equal after a json round trip (nan == nan)."""
    return json.dumps(a, sort_keys=True, default=float) == json.dumps(b, sort_keys=True, default=float)


def go(out, tag, pol, setup, n, workers, table, n_nodes=PO.N_NODES, seed=0, split="val"):
    return R.run(pol, setup, seed, split, n, workers, tag, out, table=table, n_nodes=n_nodes)


def go_pristine(pristine, out, tag, pol, setup, n, table, n_nodes=PO.N_NODES, seed=0):
    """the same run with the pre-edit code (a copy of the tree), as a subprocess: cwd and PYTHONPATH = the copy."""
    env = dict(os.environ, PYTHONPATH=str(pristine))
    c = [sys.executable, "-m", "world_model.v2.tray_rep.boed.run", "--policy", pol, "--setup", setup, "--seed", str(seed), "--split", "val", "--n-eps", str(n),
         "--workers", "1", "--tag", tag, "--out", str(out), "--table", str(table), "--n-nodes", str(n_nodes)]
    r = subprocess.run(c, cwd=pristine, env=env, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    return r.stdout


# ------------------------------------------------------------------------------------------------------------------ stages
def stage_unit(a, res, check):
    I = np.array(PO.IGRID)
    d = np.array([[min(1.0, 0.6 * i ** 2 * (0.0006 / m) ** 1.5) for m in PO.NODES] for i in I])
    T = PO.Table(I, PO.NODES, np.round(d * 100), np.full(d.shape, 100.0))
    b = MD.Belief()
    mu_q, idx = P7.decide(T, b, 0.02)
    check("rule_prior_is_worst_node", abs(mu_q - float(MD.MU_GRID[0])) < 1e-12 and idx == P7.pick_worst_case(T, MD.MU_GRID[0], 0.02), f"mu_q {mu_q:.5f} -> I {T.igrid[idx]}")
    # a sharp posterior at mu = 0.004: the 2.5% quantile is near it, and a faster I is chosen than for the prior
    b2 = MD.Belief(); b2.update(-0.5 * ((MD.LOG_MU - np.log(0.004)) / 0.05) ** 2)
    mu2, idx2 = P7.decide(T, b2, 0.02)
    check("rule_sharp_posterior_faster", mu2 > 0.003 and idx2 <= idx, f"mu_q {mu2:.5f} -> I {T.igrid[idx2]} (prior: {T.igrid[idx]})")
    # fastest meeting eps / slowest if none / agrees with brute force at every belief node
    ok = True
    for j in range(MD.N_GRID):
        dd = T.drop_at(MD.MU_GRID[j])
        want = next((i for i in range(len(I)) if dd[i] <= 0.02), len(I) - 1)
        ok &= P7.pick_worst_case(T, MD.MU_GRID[j], 0.02) == want
    check("rule_is_fastest_meeting_eps_at_mu_q", ok)
    check("rule_slowest_if_none", P7.pick_worst_case(T, MD.MU_GRID[0], -1.0) == len(I) - 1)
    check("rule_frac_scaling_monotone", P7.pick_worst_case(T, MD.MU_GRID[0], 0.02, 0.2) <= P7.pick_worst_case(T, MD.MU_GRID[0], 0.02, 1.0))
    for setup in SETUPS:
        t = PO.Table.load(table_path(setup))
        mono = bool(np.all(np.diff(t.drop, axis=1) <= 1e-9))
        check(f"frozen_table_monotone_decreasing_in_mu_{setup}", mono, f"drop(I,mu) non-increasing in mu: {mono}; drop at mu_lo per I {np.round(t.drop[:, 0], 3).tolist()}")
        I_lo = int(P7.pick_worst_case(t, MD.MU_GRID[0], 0.02))
        I_pop = int(t.choose(PO.w_prior(), 0.02))
        check(f"frozen_table_pop_vs_worst_{setup}", True, f"P1 (mu_lo) I={t.igrid[I_lo]}, P6 (population) I={t.igrid[I_pop]}")
    bad = [m for m in sys.modules if m.startswith("YOR.robot") or m.startswith("YOR.extra") or m.split(".")[0] in ("piper_sdk", "can", "xarm", "rclpy", "serial")]
    from world_model.v2.tray_rep.boed import launch_local, episode  # noqa: F401
    bad = [m for m in sys.modules if m.startswith("YOR.robot") or m.startswith("YOR.extra") or m.split(".")[0] in ("piper_sdk", "can", "xarm", "rclpy", "serial")]
    check("no_hardware_modules_imported", not bad, str(bad))


def stage_equiv(a, res, check):
    n, out = a.n, Path(a.out)
    for setup in SETUPS:
        tab = table_path(setup)
        for pol in ("P1", "P3", "P6"):
            go(out, "chk_equiv_new", pol, setup, n, a.workers, tab)
            new = recs(out, "chk_equiv_new", "val", setup, pol)
            if a.pristine:
                go_pristine(a.pristine, out, "chk_equiv_old", pol, setup, n, tab)
                old = recs(out, "chk_equiv_old", "val", setup, pol)
                check(f"equiv_pristine_{setup}_{pol}", jeq(new, old) and len(new) == n, f"{len(new)} vs {len(old)} records identical={jeq(new, old)}")
            # (b) vs the stored smoke records of the original harness (smoke tables: 4 mu nodes)
            sm = FROZEN / "smoke"
            stored = [r for r in recs(FROZEN, "smoke", "val", setup, pol) if r["ep"] < n]
            go(out, "chk_equiv_smk", pol, setup, n, a.workers, sm / f"table_{setup}.json", n_nodes=4)
            new_s = recs(out, "chk_equiv_smk", "val", setup, pol)
            check(f"equiv_stored_smoke_{setup}_{pol}", jeq(new_s, stored) and len(new_s) == n, f"{len(new_s)} vs {len(stored)} stored records identical={jeq(new_s, stored)}")


def stage_determ(a, res, check):
    n, out = a.n, Path(a.out)
    for setup in SETUPS:
        tab = table_path(setup)
        for pol in ("P7", "P7s"):
            go(out, "chk_det1", pol, setup, n, 1, tab)
            go(out, "chk_det2", pol, setup, n, 2, tab)
            r1, r2 = recs(out, "chk_det1", "val", setup, pol), recs(out, "chk_det2", "val", setup, pol)
            check(f"determinism_1_vs_2_workers_{setup}_{pol}", jeq(r1, r2) and len(r1) == n, f"{len(r1)} vs {len(r2)} records identical={jeq(r1, r2)}")
            go(out, "chk_det_res", pol, setup, n // 2, 2, tab)
            go(out, "chk_det_res", pol, setup, n, 2, tab)
            rr = recs(out, "chk_det_res", "val", setup, pol)
            check(f"resume_equals_fresh_{setup}_{pol}", jeq(rr, r1) and len(rr) == n, f"{len(rr)} records identical={jeq(rr, r1)}")


def stage_certvalid(a, res, check):
    from world_model.v2.tray_rep.boed import episode as EP, probe as PB
    N = a.n_cert
    MUS = [float(MD.MU_GRID[0]), 0.001, 0.002, 0.004, float(MD.MU_GRID[-1])]
    variants = {"default": P7.P7Spec(),
                "F5_eps_a_0.08": dataclasses.replace(P7.P7Spec(), eps_a=0.08),
                "F5_eps_a_0.08_late3": dataclasses.replace(P7.P7Spec(), eps_a=0.08, trig_late=3),
                "literal_full_hold": dataclasses.replace(P7.P7Spec(), trigger_aware=0),
                "F5_gentle_backup": dataclasses.replace(P7.P7Spec(), backup=0)}
    out = {}
    for setup in SETUPS:
        ctx = EP.Ctx(setup, 0, "val")
        certs = {k: P7.SetCertificate(ctx.renv, ctx.spec, v) for k, v in variants.items()}
        rows = []
        for ep in range(N):
            seed = ctx.store.ep_info(ep)["seed"]
            run = EP.Run(ctx, seed)
            run.kf = PB.make_kf(ctx.trk, ctx.noisefree); run.kf.reset_prior(sx=0.006, sv=0.01); run.kf.feed(run.obs[0], 0)
            mu_w = run.belief.lower_mu(P7.Q_CERT)
            row = dict(ep=ep, mu=run.mu)
            for k, sc in certs.items():
                t0 = time.perf_counter(); r = sc.check(run.kf, run.obs[0], mu_w)
                row[f"set_{k}"] = bool(r["ok"]); row[f"dmin_{k}"] = r["dmin"]
                if k == "default":
                    row["set_ms"] = 1e3 * (time.perf_counter() - t0)
            t0 = time.perf_counter(); row["shadow"] = bool(run.certify(False)); row["shadow_dmin"] = run.info["cert_dmin"]; row["shadow_ms"] = 1e3 * (time.perf_counter() - t0)
            # the real probe (no carry): own mu, then each mu of MUS
            true = {}
            for mu in [run.mu] + MUS:
                rn = EP.Run(ctx, seed, mu_override=mu)
                kf2 = PB.make_kf(ctx.trk, ctx.noisefree); kf2.reset_prior(sx=0.006, sv=0.01)
                pr = PB.run_probe(ctx.renv, ctx.spec, kf2, False)
                true[mu] = (pr["dmin"], pr["dropped"], int(pr["ctrl"].n_hold))
            row["true_own"] = true[run.mu][0]
            row["true_by_mu"] = {f"{mu:.5f}": true[mu][0] for mu in MUS}
            row["true_min_all"] = min(v[0] for v in true.values())
            row["any_drop"] = any(v[1] for v in true.values())
            rows.append(row)
        summ = dict(n=len(rows))
        for k in list(variants) + ["shadow"]:
            key = "shadow" if k == "shadow" else f"set_{k}"
            ps = [r for r in rows if r[key]]
            summ[k] = dict(pass_rate=len(ps) / len(rows), n_pass=len(ps),
                           true_min_edge_dist_cm_own_mu=(100 * min(r["true_own"] for r in ps)) if ps else None,
                           true_min_edge_dist_cm_all_mu=(100 * min(r["true_min_all"] for r in ps)) if ps else None,
                           n_true_dmin_below_0=sum(r["true_min_all"] < 0 for r in ps), n_true_dmin_below_2cm=sum(r["true_min_all"] < 0.02 for r in ps),
                           n_probe_drops=sum(r["any_drop"] for r in ps))
        summ["set_ms_mean"] = float(np.mean([r["set_ms"] for r in rows])); summ["shadow_ms_mean"] = float(np.mean([r["shadow_ms"] for r in rows]))
        summ["agree_set_default_vs_shadow"] = float(np.mean([r["set_default"] == r["shadow"] for r in rows]))
        summ["set_default_pass_shadow_fail"] = sum(r["set_default"] and not r["shadow"] for r in rows)
        out[setup] = dict(summary=summ, rows=rows)
        d = summ["default"]
        check(f"set_cert_never_less_conservative_than_it_claims_{setup}", d["n_true_dmin_below_0"] == 0 and d["n_probe_drops"] == 0 and d["n_pass"] > 0,
              f"{d['n_pass']}/{N} pass (shadow {summ['shadow']['n_pass']}/{N}); true min edge dist where it passes: own mu {d['true_min_edge_dist_cm_own_mu']} cm, "
              f"any mu in {{5e-4..6e-3}} {d['true_min_edge_dist_cm_all_mu']} cm; <2 cm: {d['n_true_dmin_below_2cm']}; drops {d['n_probe_drops']}")
    return out


def stage_smoke(a, res, check):
    out = Path(a.out)
    setup, n = "piper_trk", a.n
    tab = table_path(setup)
    lines = [f"smoke: {n} VAL carries, {setup}, value seed 0, frozen VAL table (in-sample for VAL: plumbing, not evidence)", "",
             "| policy | drop | t_complete_s (non-dropped) | I_rel hist (1/.75/.55/.4) | cert pass | probes run | probe drops | s/carry |", "|---|---|---|---|---|---|---|---|"]
    table = {}
    for pol in ("P6", "P3", "P7", "P7s"):
        t0 = time.perf_counter()
        s = go(out, "smoke", pol, setup, n, a.workers, tab)
        rs = recs(out, "smoke", "val", setup, pol)
        I = [r["I_rel"] for r in rs]
        hist = [int(sum(abs(i - g) < 1e-9 for i in I if i is not None)) for g in PO.IGRID]
        tc = [r["t_complete_s"] for r in rs if not r["dropped"]]
        cp = [r.get("cert_pass", r.get("cert_ok")) for r in rs]
        cp = [c for c in cp if c is not None]
        row = dict(n=len(rs), drops=int(sum(r["dropped"] for r in rs)), t_mean=float(np.mean(tc)) if tc else None, I_hist=hist,
                   cert_pass=(float(np.mean(cp)) if cp else None), probes=int(sum(bool(r.get("probe_run")) for r in rs)),
                   probe_drops=int(sum(bool(r.get("probe_drop")) for r in rs)), s_per_carry=float(np.mean([r["wall_s"] for r in R.read_all(out, "smoke", "val", setup, pol, 0)])))
        if pol in ("P7", "P7s"):
            row["mu_q_range"] = [float(min(r["mu_q"] for r in rs)), float(max(r["mu_q"] for r in rs))]
            row["rules"] = {k: int(sum(r.get("rule") == k for r in rs)) for k in ("worstcase_q", "p6_fallback")}
        table[pol] = row
        lines.append(f"| {pol} | {row['drops']}/{row['n']} | {row['t_mean']:.2f} | {hist} | {row['cert_pass'] if row['cert_pass'] is None else round(row['cert_pass'], 2)} | "
                     f"{row['probes']} | {row['probe_drops']} | {row['s_per_carry']:.1f} |")
    (out / "checks").mkdir(parents=True, exist_ok=True)
    (out / "checks" / "smoke.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    check("smoke_ran", all(v["n"] == n for v in table.values()), str({k: v["n"] for k, v in table.items()}))
    return table


STAGES = ("unit", "equiv", "determ", "certvalid", "smoke")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all")
    ap.add_argument("--n", type=int, default=0, help="carries per run (default 6; smoke 20)")
    ap.add_argument("--n-cert", type=int, default=60)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--pristine", default="", help="a copy of the code tree from before the P7 edits (dir containing world_model/); enables equiv (a)")
    a = ap.parse_args()
    a.n_user = a.n
    out = Path(a.out)
    assert out.resolve() != FROZEN.resolve(), "never write into the frozen folder"
    (out / "checks").mkdir(parents=True, exist_ok=True)
    stages = STAGES if a.stage == "all" else a.stage.split(",")
    ok_all, results = True, {}

    for st in stages:
        res = {}

        def check(name, ok, detail=""):
            nonlocal ok_all
            ok_all &= bool(ok)
            res[name] = dict(ok=bool(ok), detail=detail)
            print(("PASS " if ok else "FAIL ") + name, detail, flush=True)

        a.n = a.n_user or (20 if st == "smoke" else 6)
        extra = globals()[f"stage_{st}"](a, res, check)
        (out / "checks" / f"{st}.json").write_text(json.dumps(dict(checks=res, extra=extra), indent=1, default=float))
        results[st] = res
    print("ALL PASS" if ok_all else "SOME FAILED", flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
