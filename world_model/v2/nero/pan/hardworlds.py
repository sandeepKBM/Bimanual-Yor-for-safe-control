"""pan/hardworlds.py -- systematic hard-world generation for demo D1 (covered pot, two side handles).  SIMULATION ONLY, CPU.

Spec: docs/paper/demos_2026-11-01.md section 2 (USER 11-01: "get that running").  Stage 1 implemented here:
  1. Theta box (THETA, every range labelled) -> scrambled Sobol design, fixed seed (independent of any proposer).
  2. Corner plans of the plan menu: grasp_frac {0.45, 0.85} x grip force {10, 40} N x lift {v1 independent, v3 coupled}.
  3. Per (world, plan): continuous margins (slip mm / deg in each claw, lid, rise, reach) + torque vs the NERO MIT range.
  4. Tiers per world: easy (all corner plans hold) / boundary (split) / infeasible (none hold); failure mode of each failed plan
     = the margin that is most negative.
Next stages (not here yet): adaptive boundary sampling where plans disagree, per-plan worst-case search, contact-artefact re-check,
selection + freeze + PREREG.

Robot setup for every plan (same physics for all, rule 8): payload-aware lift (Episode lift_ctl="adaptive"), claw force limit 40 N
(NERO reseller rating, ASSUMED; grip force is the plan's command), physics step 1 ms (5 substeps per 5 ms control step; > 10 N grip
needs it, claw40/RESULT.txt), observe_pose at zed noise with a per-world seed.

    python -m world_model.v2.nero.pan.hardworlds stage1 --out DIR --n 64 --procs 10
    python -m world_model.v2.nero.pan.hardworlds summary --out DIR
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from .. import config as C

# ---- Theta: (lo, hi, label).  Raised to the user 11-01 (demos_2026-11-01.md sec. 3); USER said "get that running" 11-01.
THETA = {
    "dx": (-0.015, 0.015, "ASSUMED (prior box, unchanged)"),
    "dy": (-0.015, 0.015, "ASSUMED (prior box, unchanged)"),
    "yaw_deg": (-8.0, 8.0, "ASSUMED (prior box, unchanged)"),
    "mu_scale": (0.4, 1.2, "ASSUMED (widened from 0.6-1.2: wet/greasy)"),
    "mass": (0.02, 1.5, "ASSUMED (widened from 0-0.4 kg; 0.02 = an almost empty pot)"),
    "c_r": (0.0, 1.0, "MODEL: contents centre radius as a fraction of C_R (uniform over the disc via sqrt)"),
    "c_phi": (0.0, 1.0, "MODEL: contents centre angle / 2 pi"),
    "lid_mu_scale": (0.5, 1.0, "ASSUMED (wet lid)"),
    "handle_len": (0.09, 0.13, "ASSUMED (stove-clearance tier; 0.09 refused 6/9 on 11-01)"),
}
C_R = 0.06            # m, MODEL: pot inner radius 0.09 - puck radius 0.025 - 5 mm
SOBOL_SEED = 11
CORNER_FRACS = (0.45, 0.85)
CORNER_FORCES = (10.0, 40.0)
LIFTS = ("v1", "v3")
CLAW_N = C.CLAW_N_DEFAULT   # 40 N, MEASURED vendor rating (user-confirmed 11-01; see config)
NSUB = 5              # 1 ms physics step
TAU_MIT = 16.0        # Nm, NERO MIT feed-forward command range fw >= 1.11 (pyAgxArm firmware_reference) -- ASSUMED as the limit
RISE_MIN_MM = 40.0    # = handle_grasp 'held' rule (0.8 x LIFT)


def design(n, seed=SOBOL_SEED):
    """-> [(tag, theta dict)] scrambled Sobol over THETA."""
    from scipy.stats import qmc
    keys = list(THETA)
    u = qmc.Sobol(len(keys), scramble=True, seed=seed).random(n)
    out = []
    for i, row in enumerate(u):
        th = {k: float(THETA[k][0] + x * (THETA[k][1] - THETA[k][0])) for k, x in zip(keys, row)}
        out.append((f"H{i:03d}", th))
    return out


def world_from_theta(th):
    r, phi = C_R * np.sqrt(th["c_r"]), 2 * np.pi * th["c_phi"]
    return dict(placement=[float(r * np.cos(phi)), float(r * np.sin(phi))], hidden_mass=th["mass"], handle_len=th["handle_len"],
                perturb=dict(dx=th["dx"], dy=th["dy"], yaw_deg=th["yaw_deg"], mu_scale=th["mu_scale"], lid_mu_scale=th["lid_mu_scale"]))


def _patch_substeps(nsub):
    """mujoco proxy for the pan modules: each mj_step = nsub substeps of dt/nsub, applied torque held (as claw40 / physics_ab)."""
    import mujoco
    from . import segments as S, handle_grasp as HG, scene as SC, under_rim as UR

    class _Sub:
        def __getattr__(self, name):
            return getattr(mujoco, name)

        @staticmethod
        def mj_step(m, d):
            dt = m.opt.timestep
            m.opt.timestep = dt / nsub
            try:
                for _ in range(nsub):
                    mujoco.mj_step(m, d)
            finally:
                m.opt.timestep = dt

    if nsub > 1:
        for mod in (S, HG, SC, UR):
            mod.mujoco = _Sub()


def margins(lf, F):
    """continuous margins (> 0 = inside the limit) for one lift record."""
    return dict(slip_mm=F.SLIP_MM - max(lf["slip_mm"].values()), slip_deg=F.SLIP_DEG - max(lf["slip_deg"].values()),
                lid=F.LID_MM - lf.get("lid_lift_mm", lf["lid_mm"]), rise=lf["rise_mm"] - RISE_MIN_MM)


def run_job(job):
    """one world x one pinch (frac, force) -> both lifts from the same post-pinch snapshot.  Writes outdir/jobs/<key>.json."""
    tag, th, frac, force, outdir = job[:5]
    contact = job[5] if len(job) > 5 else "base"
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    key = f"{tag}_f{int(round(frac * 100))}_F{int(force)}"
    path = os.path.join(outdir, "jobs", key + ".json")
    if os.path.exists(path):
        return key, "cached"
    _patch_substeps(NSUB // C.substeps_default())   # 11-01: the scene now substeps by default -> no extra proxy (legacy: x5)
    from . import segments as S
    from .cert import contact_falsify as F
    t0 = time.time()
    rec = dict(tag=tag, theta=th, frac=frac, force=force, claw_N=CLAW_N, nsub=NSUB, lift_ctl="adaptive", contact=contact)
    try:
        ep = S.Episode(world_from_theta(th), observe="zed", obs_seed=S.world_seed(tag), obj="pot", grasp_frac=frac,
                       grip_force=force, claw_N=CLAW_N, lift_ctl="adaptive")
        rec["obs_residual"] = None if not ep.obs else dict(mm=ep.obs["residual_mm"], deg=ep.obs["residual_yaw_deg"])
        rec["refused"] = ep.refused
        rec["cand"] = (getattr(ep, "pregrasp_margin", None) or {}).get("cand")   # grasp actually used: [left, right] (frac, yaw, roll)
        apply_contact(ep.m, contact)
        if not ep.refused:
            ep.run_phases(ep.segment_phases("start"))
            ep.run_phases(ep.segment_phases("pinch"))
            snap = ep.snapshot("lift_start")
            rec["lift"] = {}
            for v in LIFTS:
                ep.restore(snap)
                ep.reset_track()
                lf = F._lift(ep, v)
                mg = margins(lf, F)
                fail = list(F.failed(lf)) + (["rise"] if mg["rise"] < 0 else [])
                tau = max(max(lf["tau_abs_max"]["left"]), max(lf["tau_abs_max"]["right"]))
                rec["lift"][v] = dict(fail=fail, margins=mg, tau_abs_max_Nm=tau, tau_mit_margin=TAU_MIT - tau,
                                      slip_mm=lf["slip_mm"], slip_deg=lf["slip_deg"], lid_lift_mm=lf.get("lid_lift_mm"),
                                      tilt_deg=lf["tilt_deg"], rise_mm=lf["rise_mm"], claw_rel_mm=lf.get("claw_rel_mm"))
    except Exception as e:                     # recorded, never silently dropped
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path + ".tmp", "w").write(json.dumps(rec, default=float))
    os.replace(path + ".tmp", path)
    return key, "done"


def plan_outcomes(rec):
    """-> {plan_id: (held bool | None, mode str | None, margins | None)} for one job record."""
    base = f"f{int(round(rec['frac'] * 100))}_F{int(rec['force'])}"
    if rec.get("error"):
        return {f"{base}_{v}": (None, "error", None) for v in LIFTS}
    if rec.get("refused"):
        return {f"{base}_{v}": (False, "reach", None) for v in LIFTS}
    out = {}
    for v, r in rec["lift"].items():
        mg = r["margins"]
        held = not r["fail"]
        mode = None if held else min((k for k in mg), key=lambda k: mg[k])
        out[f"{base}_{v}"] = (held, mode, mg)
    return out


def summarize(outdir):
    import glob
    worlds = {}
    for p in glob.glob(os.path.join(outdir, "jobs", "*.json")):
        rec = json.load(open(p))
        w = worlds.setdefault(rec["tag"], dict(theta=rec["theta"], plans={}))
        w["plans"].update(plan_outcomes(rec))
    n_plans = len(CORNER_FRACS) * len(CORNER_FORCES) * len(LIFTS)
    rows, tiers, modes = [], {}, {}
    for tag in sorted(worlds):
        w = worlds[tag]
        pl = w["plans"]
        done = [v for v in pl.values() if v[0] is not None]
        if len(pl) < n_plans:
            tier = "incomplete"
        elif any(v[0] is None for v in pl.values()):
            tier = "error"
        else:
            nh = sum(v[0] for v in done)
            tier = "easy" if nh == n_plans else ("infeasible" if nh == 0 else "boundary")
        tiers[tier] = tiers.get(tier, 0) + 1
        ms = sorted({v[1] for v in pl.values() if v[1]})
        for mo in ms:
            modes[mo] = modes.get(mo, 0) + 1
        rows.append(dict(tag=tag, tier=tier, held=sum(bool(v[0]) for v in pl.values()), n=len(pl), modes=ms, theta=w["theta"],
                         plans={k: dict(held=v[0], mode=v[1]) for k, v in sorted(pl.items())}))
    lines = [f"hard worlds stage 1: {outdir}", f"worlds {len(rows)}; tiers {tiers}; worlds with each failure mode {modes}",
             "per-parameter mean, holding-plan share weighted (does a parameter push worlds toward failure?):"]
    full = [r for r in rows if r["tier"] in ("easy", "boundary", "infeasible")]
    if full:
        for k in THETA:
            x = np.array([r["theta"][k] for r in full])
            h = np.array([r["held"] / r["n"] for r in full])
            c = float(np.corrcoef(x, h)[0, 1]) if np.std(x) > 0 and np.std(h) > 0 else float("nan")
            lines.append(f"  {k:13s} corr(theta, share of plans held) = {c:+.2f}")
        plan_ids = sorted(full[0]["plans"])
        lines.append("per plan: held / worlds")
        for pid in plan_ids:
            lines.append(f"  {pid:14s} {sum(bool(r['plans'][pid]['held']) for r in full)}/{len(full)}")
    open(os.path.join(outdir, "worlds.jsonl"), "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
    open(os.path.join(outdir, "SUMMARY.txt"), "w").write("\n".join(lines) + "\n")
    return "\n".join(lines)


CONTACTS = {
    "base": "finger condim 3 (scene), 1 ms step",
    # physics_ab/twist_mj (11-01): 1 ms + condim 4 + torsional 0.003 m = 97 % of the analytic twist capacity (base 1 ms: 92 %)
    "c4t003": "finger condim 4, torsional friction 0.003 m (MODEL check: best twist match), 1 ms step",
}


def apply_contact(m, contact):
    """contact-model variant for the artefact check (after Episode construction; the pinch happens later)."""
    if contact == "base":
        return
    if contact != "c4t003":
        raise ValueError(contact)
    for s in ("left", "right"):
        for i in (1, 2):
            g = m.geom(f"{s}_arm_finger{i}_col").id
            m.geom_condim[g] = 4
            m.geom_friction[g, 1] = 0.003


def norm_margin(mg):
    """one number per lift: the most-violated limit, each margin divided by its limit (< 0 = failed)."""
    from .cert import contact_falsify as F
    return min(mg["slip_mm"] / F.SLIP_MM, mg["slip_deg"] / F.SLIP_DEG, mg["lid"] / F.LID_MM, mg["rise"] / RISE_MIN_MM)


def to_u(th):
    return np.array([(th[k] - THETA[k][0]) / (THETA[k][1] - THETA[k][0]) for k in THETA])


def from_u(u):
    u = np.clip(u, 0.0, 1.0)
    return {k: float(THETA[k][0] + x * (THETA[k][1] - THETA[k][0])) for k, x in zip(THETA, u)}


def cross_moment(th):
    """MODEL feature: contents mass x offset ACROSS the handle line [kg m] (drives the swing about the two-claw axis)."""
    r, phi = C_R * np.sqrt(th["c_r"]), 2 * np.pi * th["c_phi"]
    return th["mass"] * abs(r * np.sin(phi))


def run_pool(jobs, procs, label=""):
    ctx = mp.get_context("spawn")
    with ctx.Pool(procs, maxtasksperchild=3) as pool:
        for i, (key, st) in enumerate(pool.imap_unordered(run_job, jobs), 1):
            print(f"{label}[{i}/{len(jobs)}] {key} {st}", flush=True)


def load(outdir):
    """-> {tag: dict(theta, plans={plan_id: (held, mode, margins)}, recs=[...])}"""
    import glob
    out = {}
    for p in glob.glob(os.path.join(outdir, "jobs", "*.json")):
        rec = json.load(open(p))
        w = out.setdefault(rec["tag"], dict(theta=rec["theta"], plans={}, recs=[]))
        w["plans"].update(plan_outcomes(rec))
        w["recs"].append(rec)
    return out


def artefact(src, out, tags, procs, contact="c4t003"):
    """step 6: re-run the given worlds' corner plans with the accurate contact model; compare verdicts plan by plan."""
    W = load(src)
    jobs = [(t, W[t]["theta"], fr, fo, out, contact) for t in tags for fr in CORNER_FRACS for fo in CORNER_FORCES]
    run_pool(jobs, procs, "artefact ")
    A = load(out)
    lines, agree, tot, flips = [f"artefact check: {contact} = {CONTACTS[contact]}; src {src}"], 0, 0, []
    for t in tags:
        for pid, (h, mo, _) in sorted(W[t]["plans"].items()):
            if pid not in A.get(t, {}).get("plans", {}):
                continue
            h2, mo2, _ = A[t]["plans"][pid]
            tot += 1
            agree += int(h == h2)
            if h != h2:
                flips.append(f"{t} {pid}: base {'HOLD' if h else 'fail:' + str(mo)} -> {contact} {'HOLD' if h2 else 'fail:' + str(mo2)}")
    lines += [f"verdict agreement {agree}/{tot} plan runs", "flips:"] + ["  " + f for f in flips]
    open(os.path.join(out, "ARTEFACT.txt"), "w").write("\n".join(lines) + "\n")
    return "\n".join(lines)


def _knn_predict(U, Y, Q, k=7):
    """distance-weighted k-NN hold probability; U (n, d) train, Y (n,) 0/1, Q (m, d) queries."""
    D = np.linalg.norm(Q[:, None, :] - U[None, :, :], axis=2)
    idx = np.argsort(D, axis=1)[:, :k]
    w = 1.0 / (np.take_along_axis(D, idx, axis=1) + 1e-3)
    return (w * Y[idx]).sum(1) / w.sum(1)


def stage4(src, out, n, procs, pool_n=4096, seed=12, k=7, min_dist=0.15):
    """adaptive boundary sampling: predict each corner plan's hold probability over a large Sobol pool (k-NN on stage-1 outcomes in
    the unit box + the cross-moment feature), score = variance ACROSS plans + 0.5 mean p(1-p) + 0.02 distance to stage-1 worlds,
    take the top-scoring candidates greedily with a minimum spread.  Heuristic selection (no guarantee); the outcomes are simulated."""
    from scipy.stats import qmc
    W = load(src)
    tags = sorted(t for t, w in W.items() if all(v[0] is not None for v in w["plans"].values()))
    pids = sorted(W[tags[0]]["plans"])
    f = lambda th: np.append(to_u(th), cross_moment(th) / (THETA["mass"][1] * C_R))
    U = np.array([f(W[t]["theta"]) for t in tags])
    Y = {p: np.array([float(bool(W[t]["plans"][p][0])) for t in tags]) for p in pids}
    pool = qmc.Sobol(len(THETA), scramble=True, seed=seed).random(pool_n)
    ths = [from_u(u) for u in pool]
    Q = np.array([f(th) for th in ths])
    P = np.stack([_knn_predict(U, Y[p], Q, k) for p in pids], axis=1)        # (pool, plans)
    # var across plans = they confidently DISAGREE; mean p(1-p) = each is unsure (var + mean p(1-p) == m(1-m), so weight them apart);
    # small exploration term = distance to the nearest stage-1 world (breaks ties toward unexplored regions)
    dmin = np.linalg.norm(pool[:, None, :] - np.array([to_u(W[t]["theta"]) for t in tags])[None, :, :], axis=2).min(axis=1)
    score = P.var(axis=1) + 0.5 * (P * (1 - P)).mean(axis=1) + 0.02 * dmin
    chosen = []
    for i in np.argsort(-score):
        if all(np.linalg.norm(pool[i] - pool[j]) >= min_dist for j in chosen):
            chosen.append(i)
        if len(chosen) == n:
            break
    des = [(f"B{j:03d}", ths[i]) for j, i in enumerate(chosen)]
    os.makedirs(out, exist_ok=True)
    json.dump(dict(stage=4, src=src, pool_n=pool_n, seed=seed, k=k, min_dist=min_dist, plans=pids,
                   worlds=[(t, th, float(score[i]), P[i].round(3).tolist()) for (t, th), i in zip(des, chosen)]),
              open(os.path.join(out, "design.json"), "w"), indent=1)
    run_pool([(t, th, fr, fo, out) for t, th in des for fr in CORNER_FRACS for fo in CORNER_FORCES], procs, "stage4 ")
    return summarize(out)


def stage5(src, out, procs, pinches=((0.85, 40.0), (0.45, 40.0)), lift="v3", pop=10, gens=3, elite=3, seed=13, sigma0=0.15):
    """per-plan worst-case search (cross-entropy method in the unit box): minimise the normalised margin of plan (pinch, lift),
    starting from that plan's worst stage-1 world.  A counterexample = a world where the plan fails; recorded with its mode."""
    W = load(src)
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    report = []
    for fr, fo in pinches:
        pid = f"f{int(round(fr * 100))}_F{int(fo)}_{lift}"
        scored = [(norm_margin(w["plans"][pid][2]) if w["plans"].get(pid, (None, None, None))[2] else -1.0, t)
                  for t, w in W.items() if pid in w["plans"] and w["plans"][pid][0] is not None]
        mu = to_u(W[min(scored)[1]]["theta"])
        sd = np.full(len(THETA), sigma0)
        hist = []
        for g in range(gens):
            cand = [np.clip(mu + sd * rng.standard_normal(len(THETA)), 0, 1) for _ in range(pop)]
            tags = [f"W{pid}_g{g}_{i:02d}" for i in range(pop)]
            run_pool([(t, from_u(u), fr, fo, out) for t, u in zip(tags, cand)], procs, f"stage5 {pid} g{g} ")
            R = load(out)
            res = []
            for t, u in zip(tags, cand):
                h, mo, mg = R[t]["plans"][pid]
                v = -1.0 if mg is None else norm_margin(mg)
                res.append((v, t, u, h, mo))
                hist.append(dict(gen=g, tag=t, margin=v, held=h, mode=mo, cross_moment=cross_moment(from_u(u)), theta=from_u(u)))
            res.sort(key=lambda x: x[0])
            E = np.array([x[2] for x in res[:elite]])
            mu, sd = E.mean(0), np.maximum(E.std(0), 0.03)
        fails = [h for h in hist if not h["held"]]
        report.append(dict(plan=pid, evals=len(hist), fails=len(fails), worst=min(hist, key=lambda h: h["margin"]),
                           modes=sorted({h["mode"] for h in fails}), mildest_fail=max(fails, key=lambda h: h["margin"]) if fails else None))
    json.dump(report, open(os.path.join(out, "STAGE5.json"), "w"), indent=1, default=float)
    lines = [f"stage 5 worst-case search ({pop} x {gens} per plan, CEM, seed {seed}); src {src}"]
    for r in report:
        w = r["worst"]
        lines.append(f"  {r['plan']}: {r['fails']}/{r['evals']} evaluated worlds fail; modes {r['modes']}; worst margin {w['margin']:+.2f} "
                     f"({w['mode']}; m={w['theta']['mass']:.2f} kg, cross-moment {w['cross_moment']:.3f} kg m, mu {w['theta']['mu_scale']:.2f}, "
                     f"handle {100 * w['theta']['handle_len']:.1f} cm)")
    open(os.path.join(out, "STAGE5.txt"), "w").write("\n".join(lines) + "\n")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage1", "summary", "artefact", "stage4", "stage5"])
    ap.add_argument("--src", default=None, help="stage-1 output dir (artefact / stage4 / stage5)")
    ap.add_argument("--tags", default=None, help="artefact: comma list of world tags")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--only", default=None, help="comma list of world tags (smoke)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "stage1":
        des = design(a.n)
        if a.only:
            keep = set(a.only.split(","))
            des = [x for x in des if x[0] in keep]
        json.dump(dict(theta=THETA, n=a.n, sobol_seed=SOBOL_SEED, corner_fracs=CORNER_FRACS, corner_forces=CORNER_FORCES,
                       lifts=LIFTS, claw_N=CLAW_N, nsub=NSUB, tau_mit=TAU_MIT, rise_min_mm=RISE_MIN_MM, lift_ctl="adaptive",
                       worlds=des), open(os.path.join(a.out, "design.json"), "w"), indent=1)
        jobs = [(tag, th, fr, fo, a.out) for tag, th in des for fr in CORNER_FRACS for fo in CORNER_FORCES]
        ctx = mp.get_context("spawn")
        with ctx.Pool(a.procs, maxtasksperchild=3) as pool:
            for i, (key, st) in enumerate(pool.imap_unordered(run_job, jobs), 1):
                print(f"[{i}/{len(jobs)}] {key} {st}", flush=True)
    elif a.cmd == "artefact":
        print(artefact(a.src, a.out, a.tags.split(","), a.procs))
        return
    elif a.cmd == "stage4":
        print(stage4(a.src, a.out, a.n, a.procs))
        return
    elif a.cmd == "stage5":
        print(stage5(a.src, a.out, a.procs))
        return
    print(summarize(a.out))


if __name__ == "__main__":
    main()
