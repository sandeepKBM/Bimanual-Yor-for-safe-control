"""vlm/analyze.py -- bake-off analysis: python -m world_model.v2.vlm.analyze <model_key> [<model_key> ...]
reads out/bake_<model>.jsonl + scenes/meta.json; prints tables and writes results/<model>.json (observations only)."""
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from world_model.v2.vlm import phys2real_prior as PR
from world_model.v2.vlm import prompts as P
from world_model.v2.vlm import spec_eval as SE

DATA = Path("/common/users/ss5772/bimanual_data/vlm")
TRUTH_KEY = {"mu_roll": "mu_roll", "mu": "mu", "ball_radius": "ball_radius", "bead_friction": "bead_friction", "n_beads": "n_beads",
             "bead_radius": "bead_radius"}


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def spearman(a, b):
    if len(a) < 4:
        return None
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def calib(rows):
    """rows: list of (truth, mean, sigma) ; sigma>0 or None. log-space: sigma_log = sigma/mean (delta method)."""
    rows = [(t, m, s) for t, m, s in rows if m and m > 0 and s and s > 0]
    if not rows:
        return None
    e = np.array([np.log(m / t) for t, m, s in rows])
    sl = np.array([s / m for t, m, s in rows])
    z = np.abs(e) / sl
    return dict(n=len(rows), cov1=float((z <= 1).mean()), cov2=float((z <= 2).mean()), conf_wrong=float((z > 2).mean()),
                med_z=float(np.median(z)), med_sigma_log=float(np.median(sl)), spearman_sigma_abserr=spearman(sl, np.abs(e)))


def analyze_params(out, meta, hint):
    tag = "h" if hint else "n"
    res = {}
    for s in meta:
        recs = [out[k] for k in (f"p|{s['id']}|{tag}|{i}" for i in range(len(s["images"]))) if k in out]
        if not recs:
            continue
        agg = PR.aggregate(recs, s["kind"])
        for p, a in agg.items():
            res.setdefault((p, s["set"]), []).append((s["truth"][TRUTH_KEY[p]], a, s))
    table = {}
    for (p, sset), L in sorted(res.items()):
        n_samples = sum(a["n_total"] for _, a, _ in L)
        n_parsed = sum(len(a["samples"]) for _, a, _ in L)
        ok = [(t, a) for t, a, _ in L if a["mean"]]
        errs = np.array([np.log(a["mean"] / t) for t, a in ok]) if ok else np.array([])
        row = dict(n_scenes=len(L), parse_rate=n_parsed / max(n_samples, 1), n_ok=len(ok))
        if len(ok):
            lt = np.log([t for t, _ in ok])
            row.update(bias_ln=float(np.median(errs)), mae_ln=float(np.abs(errs).mean()),
                       const_mae_ln=float(np.abs(lt - np.median(lt)).mean()),   # reference: best constant guess (in-sample log-median of truths)
                       spearman_est_truth=spearman(np.log([a["mean"] for _, a in ok]), np.log([t for t, _ in ok])),
                       est_median=float(np.median([a["mean"] for _, a in ok])), truth_median=float(np.median([t for t, _ in ok])),
                       sigma_vlm_median=float(np.median([a["sigma_vlm"] for _, a in ok])),
                       calib_sigma_vlm=calib([(t, a["mean"], a["sigma_vlm"]) for t, a in ok]),
                       calib_std_est=calib([(t, a["mean"], a["std_est"]) for t, a in ok]),
                       std_est_zero_frac=float(np.mean([(a["std_est"] or 0) < 1e-12 for _, a in ok])),
                       calib_per_sample=calib([(t, e, sg) for t, a in ok for e, sg in a["samples"]]),
                       calib_sigma_vlm_rand_only=None)
        table[f"{p}|{sset}"] = row
    return table


def analyze_specs(out, meta):
    res = {}
    for pre, name in (("s0", "greedy"), ("s1", "sampled")):
        by = defaultdict(list)
        for s in meta:
            o = out.get(f"{pre}|{s['id']}")
            if not o:
                continue
            for t in o["texts"]:
                j = SE.extract_json(t)
                r = dict(json=j is not None, compile=False, eval=False, sem=False, hand=False, text=t[:600], spec=j)
                if j is not None:
                    c = SE.check(j, s["kind"])
                    r.update(compile=c["compile"], eval=c["eval"], sem=c["sem"], error=c["error"], margins=c["margins"],
                             probes=c.get("probes_caught"))
                    if s["kind"] == "tray_ball":
                        r["hand"] = sorted(SE.atoms(j)) == sorted(SE.atoms(SE.TRAY_SPEC_FULL))
                by[s["kind"]].append(r)
        for kind, rs in by.items():
            res[f"{name}|{kind}"] = dict(n=len(rs), **{k: float(np.mean([r[k] for r in rs])) for k in ("json", "compile", "eval", "sem", "hand")},
                                         errors=sorted({(r.get("error") or "")[:60] for r in rs if r.get("error")})[:4],
                                         example=next((r["spec"] for r in rs if r["sem"]), None) or next((r["text"] for r in rs), None))
    return res


def analyze_latency(out):
    by = defaultdict(list)
    for k, o in out.items():
        typ = k.split("|")[0]
        by[typ].append(o)
    res = {}
    for typ, L in by.items():
        sec = np.array([o["seconds"] for o in L])
        tok = np.array([sum(o["n_new_tokens"]) for o in L])
        res[typ] = dict(n=len(L), sec_median=float(np.median(sec)), sec_mean=float(sec.mean()), new_tokens_median=float(np.median(tok)),
                        tok_per_s=float(tok.sum() / sec.sum()), prompt_tokens=int(np.median([o["n_prompt_tokens"] for o in L])),
                        peak_vram_gb=max(o["peak_vram_gb"] for o in L))
    return res


def main(models, tag="bake"):
    meta = json.load(open(DATA / "scenes" / "meta.json"))
    (DATA / "results").mkdir(exist_ok=True)
    for m in models:
        f = DATA / "out" / f"{tag}_{m}.jsonl"
        if not f.exists():
            f = Path(str(f) + ".partial")
        if not f.exists():
            print("missing", m); continue
        out = {o["id"]: o for o in (json.loads(l) for l in open(f) if l.strip())}
        R = dict(model=m, n_requests=len(out), params_hint=analyze_params(out, meta, True), params_nohint=analyze_params(out, meta, False),
                 specs=analyze_specs(out, meta), latency=analyze_latency(out))
        (DATA / "results" / (f"{m}.json" if tag == "bake" else f"{tag}_{m}.json")).write_text(json.dumps(R, indent=1, default=str))
        print(f"\n######## {m}  ({len(out)} requests)")
        for v in ("params_hint", "params_nohint"):
            print(f"-- {v}: param|set  n parse bias_ln mae_ln spearman | sigma_vlm: cov1 cov2 confwrong | std_est: cov1 cov2 confwrong zero")
            for k, r in R[v].items():
                if "bias_ln" not in r:
                    print(f"  {k:22s} n={r['n_scenes']} parse={r['parse_rate']:.2f} (no valid estimates)"); continue
                a, b = r["calib_sigma_vlm"], r["calib_std_est"]
                f2 = lambda c, ks: " ".join(f"{c[x]:.2f}" if c and c[x] is not None else "  - " for x in ks)
                sp = r["spearman_est_truth"]
                print(f"  {k:22s} n={r['n_ok']:2d} parse={r['parse_rate']:.2f} bias={r['bias_ln']:+.2f} mae={r['mae_ln']:.2f} "
                      f"const_mae={r['const_mae_ln']:.2f} rho={'-' if sp is None else format(sp, '+.2f')} | vlm: {f2(a, ['cov1', 'cov2', 'conf_wrong'])} "
                      f"| std: {f2(b, ['cov1', 'cov2', 'conf_wrong'])} zero={r['std_est_zero_frac']:.2f}")
        print("-- specs: kind n json compile eval sem hand(tray_ball)")
        for k, r in R["specs"].items():
            print(f"  {k:24s} n={r['n']:3d} json={r['json']:.2f} compile={r['compile']:.2f} eval={r['eval']:.2f} sem={r['sem']:.2f} hand={r['hand']:.2f}")
        print("-- latency:", {t: (round(v['sec_median'], 1), round(v['tok_per_s'], 1), v['peak_vram_gb']) for t, v in R["latency"].items()})


if __name__ == "__main__":
    args = sys.argv[1:]
    tag = "bake"
    if args and args[0].startswith("--tag="):
        tag, args = args[0].split("=")[1], args[1:]
    main(args, tag)
