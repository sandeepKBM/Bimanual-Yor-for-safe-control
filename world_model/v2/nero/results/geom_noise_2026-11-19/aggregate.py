"""Aggregate DIR/runs/*.json -> DIR/errors.json, DIR/ERRORS.md, DIR/halfwidth_<noise>.json (the format adapters.geom_probe_run.err_table_from_study reads:
{target: {field: {"max": half_width, "n": n_observed}}}; top_z is not a stack.geometry field and is left out of it).
python -m world_model.v2.nero.results.geom_noise_2026-11-19.aggregate DIR"""
import glob
import json
import os
import sys

import numpy as np

from world_model.v2.nero.contents import geom_probe as GP

ORDER = ("cx", "cy", "cz", "sx", "sy", "sz", "yaw", "top_z")
HW_FIELDS = ORDER[:-1]


def main(out):
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out, "runs", "*.json")))]
    agg = {}     # task -> noise -> entity -> field -> dict
    cell = {}
    for r in recs:
        for e, er in r["entities"].items():
            for f, fr in er["fields"].items():
                cell.setdefault((r["task"], r["noise"], e, f), []).append((r["world"], r["seed"], fr))
    for (task, nz, e, f), L in cell.items():
        obs = [(w, s, fr["error"]) for w, s, fr in L if fr["observed"]]
        a = np.array([x[2] for x in obs], float)
        d = dict(n_observed=len(obs), n_unobserved=len(L) - len(obs))
        if len(obs):
            ab = np.abs(a)
            k = int(np.argmax(ab))
            d.update(p50=float(np.percentile(ab, 50)), p95=float(np.percentile(ab, 95)), max_abs=float(ab.max()), max_pos=float(max(a.max(), 0.0)), min_neg=float(min(a.min(), 0.0)),
                     worst_world=obs[k][0], worst_seed=obs[k][1])
        agg.setdefault(task, {}).setdefault(nz, {}).setdefault(e, {})[f] = d
    meta = dict(
        guarantee="sampled", n_records=len(recs), worlds=sorted({r["world"] for r in recs}), seeds=sorted({r["seed"] for r in recs}),
        cams=recs[0]["cams"] if recs else None, image_size=recs[0]["image_size"] if recs else None,
        noise_model=dict(zed2i_lab=dict(sigma_z="z^2*SIGMA_D_PX/(f_px*BASELINE_M)", SIGMA_D_PX=GP.SIGMA_D_PX, BASELINE_M=GP.BASELINE_M, LAB_CORR_FRAC=GP.LAB_CORR_FRAC,
                                        LAB_CORR_PX_720=GP.LAB_CORR_PX_720, LAB_DROP=GP.LAB_DROP, LAB_CALIB_M=GP.LAB_CALIB_M, LAB_CALIB_DEG=GP.LAB_CALIB_DEG,
                                        status="ASSUMED (user approved 11-19) except BASELINE_M = datasheet"),
                         clean="no noise, deterministic (seed 0 only)"),
        units="metres; yaw in degrees (folded mod 90)")
    json.dump(dict(meta=meta, errors=agg), open(os.path.join(out, "errors.json"), "w"), indent=1)
    hw = {}
    for task, nzd in agg.items():
        for nz, ed in nzd.items():
            t = {}
            for e, fd in ed.items():
                t[e] = {f: dict(max=(v["max_abs"] * (np.pi / 180 if f == "yaw" else 1)), n=v["n_observed"]) for f, v in fd.items() if f in HW_FIELDS and v["n_observed"]}
            hw.setdefault(nz, {}).update(t)        # one file per noise level (tasks share entity names; last task wins -- one task per study today)
    for nz, t in hw.items():
        json.dump(t, open(os.path.join(out, f"halfwidth_{nz}.json"), "w"), indent=1)    # yaw half-width in RADIANS (stack.geometry units)
    L = ["# Depth box fit: error table (sampled)", "",
         f"Guarantee: **sampled** over the drawn worlds {meta['worlds']} x seeds {meta['seeds']} ({len(recs)} runs).  Cameras: {meta['cams']}; image size {meta['image_size']} (w, h).",
         "Truth = geom poses read from the model.  Errors in metres, yaw in degrees.  \"max over n\" = max |error| over the n observed fits (a measured maximum, not a bound).", "",
         "Noise model (zed2i_lab): " + json.dumps(meta["noise_model"]["zed2i_lab"]), "Clean: no noise, deterministic, seed 0 only.", ""]
    for task, nzd in agg.items():
        for nz, ed in nzd.items():
            L += [f"## {task} / {nz}", "", "| entity | field | n obs | n unobs | p50 | p95 | max over n | max + | max - | worst world |", "|---|---|---|---|---|---|---|---|---|---|"]
            for e in ["crate"] + [x for x in ed if x != "crate"]:
                for f in ORDER:
                    v = ed.get(e, {}).get(f)
                    if not v:
                        continue
                    if v["n_observed"]:
                        L.append(f"| {e} | {f} | {v['n_observed']} | {v['n_unobserved']} | {v['p50']:.4g} | {v['p95']:.4g} | max over n = {v['n_observed']}: {v['max_abs']:.4g} | {v['max_pos']:.4g} | {v['min_neg']:.4g} | w{v['worst_world']} s{v['worst_seed']} |")
                    else:
                        L.append(f"| {e} | {f} | 0 | {v['n_unobserved']} | - | - | - | - | - | - |")
            L.append("")
    L += ["Half-widths for err_table_from_study: `halfwidth_<noise>.json` = max |error| over n per (entity, field), yaw in radians, n = observed count."]
    open(os.path.join(out, "ERRORS.md"), "w").write("\n".join(L) + "\n")
    print(f"{len(recs)} runs -> {out}/errors.json, ERRORS.md, halfwidth_*.json")


if __name__ == "__main__":
    main(sys.argv[1])
