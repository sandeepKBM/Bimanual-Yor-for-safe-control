"""pan/cert/runtime.py -- the gated robot policy (plan 2026-10-29, Phase C).  SIMULATION ONLY, CPU.

For each segment k:
  gate   start segment k only if the library certificate for k is ok and the robot's estimate lies in the certified cell;
         otherwise REFUSE: run the stored backup of the last executed segment from where the robot is (certified from that
         segment's end), or, at B0, do nothing (pan untouched).
  arm    the stored backup of k (fixed retreat from the certificate) is built before k starts.
  fire   if the runtime monitor aborts k, or the observable boundary check after k fails, run the armed backup verbatim.
Truth (U entered, R_V2, task success) is used only to score.

    python -m world_model.v2.nero.pan.cert.runtime --cert DIR/certificate.json --out OUT --set full --procs P [--ungated]
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse
import json
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

PLAN = dict(push=0.07, grasp_frac=0.65, grip_force=10.0, clamp="v2")
LIFT_MIN = 0.8    # ASSUMED (as demo_cycle): lift boundary = both claws rose >= LIFT_MIN * LIFT by encoders


def _in_cell(est, cell):
    keys = ("dx", "dy", "yaw_deg")
    return all(cell["est_lo"][i] - 1e-12 <= est[k] <= cell["est_hi"][i] + 1e-12 for i, k in enumerate(keys))


def _score(ep, out, label):
    ok, mg, st = ep.in_R(reach=True, version="v2")
    out.update(outcome=label, end_inR=ok, end_R_fail=[k for k, v in mg.items() if v <= 0], end_state=st, U_entered=ep.track["viol"])
    return out


def run_task(world, tag, cert, gated=True):
    from .. import segments as S
    from . import backups as BK
    ep = S.Episode(world, observe="zed", obs_seed=S.world_seed(tag), **PLAN)
    out = dict(tag=tag, world=world, obs=ep.obs, gated=gated, refused=ep.refused, steps=[])
    if ep.refused:
        return _score(ep, out, "refused at B0 (robot's own pre-grasp check)")
    if gated and not _in_cell(ep.obs["est"], cert["cell"]):
        return _score(ep, out, "refused at B0 (estimate outside the certified cell)")
    ep.reset_track()
    ep.run_phases(ep.segment_phases("start"))
    last = None
    for seg in S.SEGMENTS:
        c = cert["segments"][seg]
        if gated and not c["ok"]:
            if last is None:
                return _score(ep, out, f"refused before {seg} (not certified); pan untouched")
            bk = BK.make(ep, last, cert["segments"][last]["backup_retreat"])
            ep.run_phases(list(bk.phases))
            out["steps"].append(dict(seg=seg, gate="refused", backup_of=last, backup=bk.describe()))
            return _score(ep, out, f"refused before {seg} (not certified); backup of {last} run")
        d = c.get("backup_retreat", c.get("best_retreat", 0.03))
        bk = BK.make(ep, seg, d)                                  # armed before the segment starts
        status, info = ep.run_phases(ep.segment_phases(seg), monitor=S.default_monitor(seg))
        obs = ep.observe_boundary(seg) if status == "done" else None
        bok = S.boundary_ok(ep, seg, dict(nominal=dict(status=status, observed=obs)))
        if seg == "lift" and status == "done":                   # encoder rise check (boundary_ok has none for the lift)
            rise = min(float(ep.tcp(s)[2] - ep.mem["lift_start"][s][2]) for s in S.SIDES)
            bok = bok and rise >= LIFT_MIN * S.LIFT
        step = dict(seg=seg, gate="admitted" if gated else "ungated", status=status, why=info.get("why"), t_abort=info.get("t"),
                    boundary_ok=bok, success_truth=ep.success(seg) if status == "done" else False, backup=bk.describe())
        out["steps"].append(step)
        if status != "done" or not bok:
            ep.run_phases(list(bk.phases))
            step["fired"] = True
            return _score(ep, out, f"backup fired in {seg}")
        last = seg
    out["task_success"] = bool(ep.success("lift"))
    out.update(outcome="lifted", U_entered=ep.track["viol"])
    return out


def _job(job):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    tag, w, cert, gated = job
    t0 = time.time()
    try:
        r = run_task(w, tag, cert, gated)
    except Exception:
        r = dict(tag=tag, world=w, error=traceback.format_exc()[-1500:])
    r["wall_s"] = round(time.time() - t0, 1)
    return r


def main(argv=None):
    from ..segcert import worlds
    ap = argparse.ArgumentParser()
    ap.add_argument("--cert", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="full", choices=["full", "iid", "overload"])
    ap.add_argument("--tags", default=None)
    ap.add_argument("--ungated", action="store_true")
    ap.add_argument("--procs", type=int, required=True)
    a = ap.parse_args(argv)
    cert = json.loads(Path(a.cert).read_text())
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    ws = worlds(a.set)
    if a.tags:
        keep = a.tags.split(",")
        ws = [(t, w) for t, w in ws if t in keep]
    done = {p.stem for p in (out / "raw").glob("*.json")}
    jobs = [(t, w, cert, not a.ungated) for t, w in ws if t not in done]
    with get_context("spawn").Pool(a.procs, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_job, jobs):
            (out / "raw" / f"{r['tag']}.json").write_text(json.dumps(r, default=float))
            print(r["tag"], r.get("outcome"), r.get("end_inR"), r.get("task_success"), (r.get("U_entered") or {}).get("kind"),
                  r.get("error", "")[-300:], flush=True)
    (out / f"_DONE_{a.set}").write_text("ok")


if __name__ == "__main__":
    main()
