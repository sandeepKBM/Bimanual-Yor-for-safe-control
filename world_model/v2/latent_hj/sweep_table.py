"""sweep_table.py -- aggregate scenario_*.json sweeps into per-(mode, setting) rows, apply a
selection rule, and emit the FINAL-run commands.

  python -m world_model.v2.latent_hj.sweep_table tune  <glob-or-json> ... [--rule-spill 0.1] [--out sel.json]
  python -m world_model.v2.latent_hj.sweep_table table <glob-or-json> ...        (no selection)
  python -m world_model.v2.latent_hj.sweep_table final --sel sel.json --seeds 200-249 --out-dir DIR  (prints commands)
  python -m world_model.v2.latent_hj.sweep_table apply --sel sel.json <glob-or-json> ...  (rows of the selected settings only)
  python -m world_model.v2.latent_hj.sweep_table tunecmds --seeds 100-129 [--sbatch DIR]   (A: tuning commands)
  python -m world_model.v2.latent_hj.sweep_table paretocmds --sel sel.json --seeds 200-249 [--sbatch DIR]  (B2)

Setting key per row:
  * CBF modes (f_cbf_sample, g_cbf_row): (alpha, eps)  -- carried in the row (cbf_alpha / cbf_eps)
  * c_switch / d_value / e_margin: eps -- row key 'eps' if present, else parsed from the file name
    ('...eps0.5...' / '..._e0.5_...'); scenario_heldout30_v1.json == eps 0.3 (legacy, no tag)
  * a_none / b_box: no knob.
Selection rule (default): among a mode's settings with spill_rate <= --rule-spill (3/30 = 0.10)
(optionally --max-coll collisions), take the max mean stroke_reach (tie -> lower spill, then lower jerk). If no setting
qualifies, take the min-spill setting (tie -> max reach) and flag it 'FALLBACK'.
SIMULATION-ONLY analysis of saved json.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CBF = ("f_cbf_sample", "g_cbf_row")
KNOB = ("c_switch", "d_value", "e_margin") + CBF
ORDER = ("a_none", "b_box", "c_switch", "e_margin", "d_value", "f_cbf_sample", "g_cbf_row")
LEG = Path(__file__).parent / "results"


def _eps_from_name(fn):
    m = re.search(r"(?:eps|_e)([0-9]+\.?[0-9]*)", Path(fn).name)
    if m:
        return float(m.group(1).rstrip("."))
    if "heldout30_v1" in fn:
        return 0.3
    return None


def load(paths):
    rows = []
    for p in paths:
        for f in sorted(glob.glob(p)) or [p]:
            for r in json.load(open(f)):
                if not r.get("ok", True):
                    continue
                r = dict(r); r["_file"] = f
                m = r["mode"]
                if m in CBF:
                    r["_cfg"] = (float(r["cbf_alpha"]), float(r["cbf_eps"]), int(r.get("cbf_N", 0)), str(r.get("cbf_rows", "block1")))
                elif m in KNOB:
                    e = r.get("eps", _eps_from_name(f))
                    r["_cfg"] = (float(e),) if e is not None else (None,)
                else:
                    r["_cfg"] = ()
                rows.append(r)
    return rows


def _m(rs, k, f=np.mean):
    v = [r[k] for r in rs if k in r and r[k] is not None]
    return float(f(v)) if v else float("nan")


def aggregate(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["mode"], r["_cfg"])].append(r)
    out = []
    for (mode, cfg), rs in g.items():
        seeds = sorted({r["seed"] for r in rs})
        out.append(dict(
            mode=mode, cfg=cfg, n=len(rs), seeds=(seeds[0], seeds[-1]),
            spill=int(sum(r["spill"] for r in rs)), coll=int(sum(r["collision"] for r in rs)),
            reach=_m(rs, "stroke_reach"), track_rms_cm=_m(rs, "track_rms_cm"), jerk=_m(rs, "jerk_rms"),
            da_p99=_m(rs, "da_p99"), ovr_rel=_m(rs, "ovr_rel"), ovr_p99=_m(rs, "ovr_p99"),
            fb=_m(rs, "fb_frac"), active=_m(rs, "engaged_frac"), fms=_m(rs, "filter_ms_mean"),
            ms=_m(rs, "ms_mean"), wall=_m(rs, "wall_s"), infeas=_m(rs, "qp_infeas_steps")))
    out.sort(key=lambda d: (ORDER.index(d["mode"]) if d["mode"] in ORDER else 99, d["cfg"]))
    return out


def cfg_str(mode, cfg):
    if mode in CBF:
        return "alpha=%g eps=%g N=%d rows=%s" % cfg
    if mode in KNOB:
        return "eps=%s" % (cfg[0],)
    return "-"


def print_table(agg):
    print("| mode | setting | n | spill | coll | reach | jerk | da_p99 | ovr_rel | fb_frac | active | filt ms | step ms | wall s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    f = lambda x, d=2: "-" if x != x else f"{x:.{d}f}"
    for a in agg:
        print(f"| {a['mode']} | {cfg_str(a['mode'], a['cfg'])} | {a['n']} | {a['spill']}/{a['n']} | {a['coll']} | "
              f"{f(a['reach'])} | {f(a['jerk'], 0)} | {f(a['da_p99'], 1)} | {f(a['ovr_rel'])} | {f(a['fb'])} | "
              f"{f(a['active'])} | {f(a['fms'], 1)} | {f(a['ms'], 1)} | {f(a['wall'], 0)} |")


def select(agg, rule_spill, max_coll=-1):
    sel = {}
    for mode in KNOB:
        cand = [a for a in agg if a["mode"] == mode]
        if not cand:
            continue
        ok = [a for a in cand if a["spill"] <= rule_spill * a["n"] + 1e-9 and (max_coll < 0 or a["coll"] <= max_coll)]
        if ok:
            b = max(ok, key=lambda a: (round(a["reach"], 6), -a["spill"], -a["jerk"]))
            flag = "ok"
        else:
            b = min(cand, key=lambda a: (a["spill"], -a["reach"]))
            flag = "FALLBACK(no setting met rule)"
        sel[mode] = dict(cfg=list(b["cfg"]), spill=b["spill"], n=b["n"], reach=b["reach"], coll=b["coll"], flag=flag)
    return sel


def final_cmds(sel, seeds, out_dir, ckpt, workers):
    base = ("$PY -u -m world_model.v2.latent_hj.scenario --seeds {seeds} --workers {w} --ckpt {ck} "
            "--out-dir {od} --refresh 1 --a-cap 60 --trust 10").format(seeds=seeds, w=workers, ck=ckpt, od=out_dir)
    cmds = [f"{base} --modes a_none,b_box --tag final_ref"]
    for mode, s in sel.items():
        cfg = s["cfg"]
        if mode in CBF:
            cmds.append(f"{base} --modes {mode} --cbf-alphas {cfg[0]:g} --cbf-epss {cfg[1]:g} "
                        f"--n-samples {cfg[2]} --cbf-rows {cfg[3]} --tag final_{mode}")
        else:
            cmds.append(f"{base} --modes {mode} --eps-v {cfg[0]:g} --eps-l {cfg[0]:g} --tag final_{mode}")
    return cmds


# ------------------------------------------------------------------ command / sbatch generators
EPS_KNOB = (0.15, 0.3, 0.5, 0.7, 0.9)                 # c_switch / d_value / e_margin (the existing Pareto grid)
CBF_ALPHAS = (0.0, 0.5, 0.85, 0.925, 0.975)           # 0.0 == "any Q >= eps"; paper range is 0.85-0.925
CBF_EPSS = (0.05, 0.15, 0.3, 0.5, 0.7)                # paper 0.05-0.2, extended up the V scale (V in [-2, 1])


def _base(seeds, workers, ckpt, out_dir):
    return ("$PY -u -m world_model.v2.latent_hj.scenario --seeds {s} --workers {w} --ckpt {ck} --out-dir {od} "
            "--refresh 1 --a-cap 60 --trust 10").format(s=seeds, w=workers, ck=ckpt, od=out_dir)


def tune_cmds(seeds, workers, ckpt, out_dir, with_cde=False, cbf_n=7600):
    b = _base(seeds, workers, ckpt, out_dir)
    cmds = []
    if with_cde:      # optional: regenerate (results/scenario_heldout30_v1.json + scenario_pareto_eps*.json already hold seeds 100-129)
        for m in ("c_switch", "d_value", "e_margin"):
            for e in EPS_KNOB:
                cmds.append(f"{b} --modes {m} --eps-v {e:g} --eps-l {e:g} --tag tune_{m}_eps{e:g}")
    for m in CBF:
        for e in CBF_EPSS:
            cmds.append(f"{b} --modes {m} --cbf-alphas {','.join('%g' % x for x in CBF_ALPHAS)} --cbf-epss {e:g} "
                        f"--n-samples {cbf_n} --tag tune_{m}_e{e:g}")
    return cmds


def pareto_cmds(sel, seeds, workers, ckpt, out_dir):
    """Full eps-Pareto curves on FRESH seeds: every method swept over its safety knob (CBF modes at
    the alpha/N/rows selected on the tuning seeds), plus the no-filter / tilt-box references."""
    b = _base(seeds, workers, ckpt, out_dir)
    cmds = [f"{b} --modes a_none,b_box --tag par_ref"]
    for m in ("c_switch", "d_value", "e_margin"):
        for e in EPS_KNOB:
            cmds.append(f"{b} --modes {m} --eps-v {e:g} --eps-l {e:g} --tag par_{m}_eps{e:g}")
    for m in CBF:
        cfg = sel[m]["cfg"]
        cmds.append(f"{b} --modes {m} --cbf-alphas {cfg[0]:g} --cbf-epss {','.join('%g' % x for x in CBF_EPSS)} "
                    f"--n-samples {cfg[2]} --cbf-rows {cfg[3]} --tag par_{m}_a{cfg[0]:g}")
    return cmds


def write_sbatch(cmds, d, name, hours=12, throttle=24):
    d = Path(d); d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}_tasks.txt").write_text("\n".join(cmds) + "\n")
    (d / f"{name}.sbatch").write_text(f"""#!/bin/bash
# LatentCBF sweep, ilab CPU array: one scenario invocation per array task (--workers 2 = the 2 CPUs a job gets).
# Runbook (docs/runbooks/slurm.md): no --cpus-per-task, generous --time, finite work, _DONE sentinel.
# CPU-only, simulation-only (MuJoCo, no GL, no hardware, no network beyond the shared /common FS).
# Submit (from westeros):  scripts/offload/submit_via_slurm_gateway.sh {d / (name + '.sbatch')}
#SBATCH --job-name={name}
#SBATCH --mem=8G
#SBATCH --time={hours}:00:00
#SBATCH --array=1-{len(cmds)}%{throttle}
#SBATCH --output={d}/{name}_%A_%a.out
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
source /common/users/ss5772/latent_hj_scratch/env.sh     # LATENT_HJ_SCR, $PY, caches on /common/users, cd repo
export TMPDIR=/common/users/ss5772/bimanual_data/latentcbf/tmp; mkdir -p $TMPDIR
echo "host=$(hostname) task=${{SLURM_ARRAY_TASK_ID}} start=$(date -u +%FT%TZ)"
CMD=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {d}/{name}_tasks.txt)
echo "$CMD"
eval "$CMD"
echo "exit=$? done=$(date -u +%FT%TZ)"
date -u +"%Y-%m-%dT%H:%M:%SZ" > {d}/_DONE_{name}_${{SLURM_ARRAY_TASK_ID}}
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("tune", "table", "final", "apply", "tunecmds", "paretocmds"))
    ap.add_argument("files", nargs="*")
    ap.add_argument("--rule-spill", type=float, default=0.10, help="max spill RATE (3/30 = 0.10)")
    ap.add_argument("--max-coll", type=int, default=-1, help="optional: also require collisions <= this (default off)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sel", default=None)
    ap.add_argument("--seeds", default="200-249")
    ap.add_argument("--out-dir", default="/common/users/ss5772/bimanual_data/latentcbf/final")
    ap.add_argument("--ckpt", default="$LATENT_HJ_SCR/v1/value.pt")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--with-cde", action="store_true", help="tunecmds: also regenerate c/d/e (existing files cover them)")
    ap.add_argument("--sbatch", default=None, help="tunecmds/paretocmds: dir to write <name>_tasks.txt + <name>.sbatch")
    a = ap.parse_args()
    if a.cmd == "tunecmds":
        cmds = tune_cmds(a.seeds, a.workers, a.ckpt, a.out_dir, a.with_cde)
        print("\n".join(cmds))
        if a.sbatch:
            write_sbatch(cmds, a.sbatch, "latentcbf_tune")
        return
    if a.cmd == "paretocmds":
        cmds = pareto_cmds(json.load(open(a.sel)), a.seeds, a.workers, a.ckpt, a.out_dir)
        print("\n".join(cmds))
        if a.sbatch:
            write_sbatch(cmds, a.sbatch, "latentcbf_pareto")
        return
    if a.cmd == "final":
        sel = json.load(open(a.sel))
        print("\n".join(final_cmds(sel, a.seeds, a.out_dir, a.ckpt, a.workers)))
        return
    rows = load(a.files)
    if a.cmd == "apply":
        sel = json.load(open(a.sel))
        keep = []
        for r in rows:
            m = r["mode"]
            if m not in sel or list(r["_cfg"]) == list(sel[m]["cfg"][:len(r["_cfg"])]):
                keep.append(r)
        rows = keep
    agg = aggregate(rows)
    print_table(agg)
    if a.cmd == "tune":
        sel = select(agg, a.rule_spill, a.max_coll)
        print("\nSELECTION (max reach s.t. spill rate <= %.3f%s):" % (
            a.rule_spill, "" if a.max_coll < 0 else " and collisions <= %d" % a.max_coll))
        for m, s in sel.items():
            print(f"  {m:13s} {cfg_str(m, tuple(s['cfg']))}  spill {s['spill']}/{s['n']}  reach {s['reach']:.3f}  coll {s['coll']}  {s['flag']}")
        if a.out:
            Path(a.out).write_text(json.dumps(sel, indent=1))
            print("wrote", a.out)


if __name__ == "__main__":
    main()
