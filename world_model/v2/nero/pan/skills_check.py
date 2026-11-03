"""pan/skills_check.py -- check each action skill ALONE against its contract (building blocks before the certificate, user
2026-10-29).  SIMULATION ONLY, CPU.  No backups, no monitor, no certificate: the nominal skill chain from the robot's own zed
estimate, with metrics sampled every SAMPLE_DT inside each skill, then each backup skill run once from that skill's end.

Contract = pre / post / never-U.  A skill whose precondition did not hold (an earlier skill failed its post) is still run
and recorded, but scored as "pre not met", so a failure is charged to the skill that caused it.

    python -m world_model.v2.nero.pan.skills_check --out DIR --set full|iid|overload --procs P [--tags ...]
    python -m world_model.v2.nero.pan.skills_check --out DIR --report
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

import numpy as np

PLAN = dict(push=0.07, grasp_frac=0.65, grip_force=10.0, clamp="v2")
SAMPLE_DT = 0.05

# Postconditions (ASSUMED, proposed to the user 2026-10-29; change here, every result carries this dict).
POST = dict(
    pinch=dict(pan_move_mm=1.0, lid_change_mm=0.5),
    push=dict(push_frac=(0.8, 1.2), yaw_change_deg=2.0, tilt_max_deg=1.0, lid_change_mm=2.0),
    clamp=dict(pan_move_mm=2.0, lid_change_mm=1.0),
    lift=dict(rise_frac=0.8, rise_diff_mm=5.0, tilt_max_deg=3.0, lid_max_mm=5.0),
    release_left=dict(pan_move_mm=1.0, lid_change_mm=0.5),
    right_back=dict(pan_move_mm=1.0, lid_change_mm=1.0),
    lower=dict(tilt_end_deg=1.0, lid_change_mm=1.0, dz_end_mm=3.0),
)


def _state(ep):
    from . import observe_pose as O
    ps = ep.pan_state()
    yaw = O.true_pose(ep.m, ep.d, ep.p, ep.pos0, ep.R0)["yaw_deg"]
    return dict(xy=ep.d.xpos[ep.pan][:2].copy(), yaw=float(yaw), tilt=ps["tilt"], lid=ps["lid_mm"], dz=ps["dz_mm"],
                tcpz={s: float(ep.tcp(s)[2]) for s in ("left", "right")})


def _run(ep, phases, skill_v2=None):
    """run phases (or, with skill_v2 = a skill name, the v2 skill), sampling the pan every SAMPLE_DT (via the branch hook /
    the step loop, which records and never snapshots)."""
    from . import segments as S
    every = max(1, int(round(SAMPLE_DT / S.DT)))
    mx = dict(tilt=0.0, lid=0.0, v=0.0)

    def hook(e, kind, i, n):
        if e.k % every == 0:
            ps = e.pan_state()
            mx["tilt"], mx["lid"], mx["v"] = max(mx["tilt"], ps["tilt"]), max(mx["lid"], ps["lid_mm"]), max(mx["v"], ps["v"])
        return None

    if skill_v2 is None:
        status, info = ep.run_phases(phases, branch=hook)
        return status, mx
    from .skills import pan_skills as PS
    orig = ep.run_phases

    def hooked(ph, branch=None, monitor=None):          # the v2 executor steps through run_phases: sample there too
        return orig(ph, branch=hook if branch is None else branch, monitor=monitor)

    ep.run_phases = hooked
    try:
        status, info = PS.run(ep, skill_v2)
    finally:
        ep.run_phases = orig
    mx["v2_status"], mx["v2_why"], mx["v2_log"] = status, info.get("why"), info.get("log")
    return status, mx


def _metrics(ep, s0, s1, mx, seg):
    from . import segments as S
    cs = ep.contacts()
    t = cs["touch"]
    out = dict(pan_move_mm=float(np.linalg.norm(s1["xy"] - s0["xy"]) * 1e3), yaw_change_deg=s1["yaw"] - s0["yaw"],
               lid_change_mm=s1["lid"] - s0["lid"], lid_end_mm=s1["lid"], lid_max_mm=mx["lid"], tilt_max_deg=mx["tilt"],
               tilt_end_deg=s1["tilt"], dz_end_mm=s1["dz"], pan_v_max=mx["v"], U=ep.track["viol"],
               fingers={f"{s}{k}": sorted(t[(s, k)]) for s in S.SIDES for k in (1, 2)})
    if seg == "push":
        out["push_m"] = float(-(ep.d.xpos[ep.pan][:2] - s0["xy"]) @ ep.u_true[:2])
    if seg == "lift":
        r = {s: s1["tcpz"][s] - s0["tcpz"][s] for s in S.SIDES}
        out.update(rise_mm={s: v * 1e3 for s, v in r.items()}, rise_diff_mm=abs(r["left"] - r["right"]) * 1e3,
                   support=cs["support"] / ep.W)
    return out


def check_post(seg, mt):
    """-> list of failed contract items (empty = post holds).  never-U is part of every contract."""
    from . import segments as S
    P, f = POST[seg], []
    if mt["U"]:
        f.append("U:" + mt["U"]["kind"])
    fg = mt["fingers"]
    if seg == "pinch":
        if not ("pan" in fg["left1"] and "pan" in fg["left2"]):
            f.append("grip")
    if seg == "push":
        lo, hi = P["push_frac"]
        if not lo * PLAN["push"] <= mt["push_m"] <= hi * PLAN["push"]:
            f.append("push_dist")
        if abs(mt["yaw_change_deg"]) > P["yaw_change_deg"]:
            f.append("yaw")
        if mt["tilt_max_deg"] > P["tilt_max_deg"]:
            f.append("tilt")
        if not ("pan" in fg["left1"] or "pan" in fg["left2"]):
            f.append("grip")
    if seg == "clamp":
        if not ("pan" in fg["right1"] and "pan" in fg["right2"]):
            f.append("grip")
    if seg == "lift":
        if min(mt["rise_mm"].values()) < P["rise_frac"] * S.LIFT * 1e3:
            f.append("rise")
        if mt["rise_diff_mm"] > P["rise_diff_mm"]:
            f.append("rise_diff")
        if mt["tilt_max_deg"] > P["tilt_max_deg"]:
            f.append("tilt")
        if mt["lid_max_mm"] > P["lid_max_mm"]:
            f.append("lid")
    if seg in ("release_left", "right_back"):
        if mt["pan_move_mm"] > P["pan_move_mm"]:
            f.append("pan_move")
    if seg == "lower":
        if mt["tilt_end_deg"] > P["tilt_end_deg"]:
            f.append("tilt")
        if abs(mt["dz_end_mm"]) > P["dz_end_mm"]:
            f.append("not_down")
    if seg in ("pinch", "clamp") and mt["pan_move_mm"] > P["pan_move_mm"]:
        f.append("pan_move")
    if "lid_change_mm" in P and abs(mt["lid_change_mm"]) > P["lid_change_mm"]:
        f.append("lid")
    return f


def backup_skills(ep, seg):
    """the backup building blocks reachable from this skill's end (cert/backups pieces, each run alone, in order)."""
    from . import segments as S
    from .cert import backups as BK
    rel = [("open", "left", BK.OPEN_T, BK.RELEASE_OPEN), ("retreat_fixed", "left", 0.03), ("hold", BK.SETTLE_T)]
    rb = [("open", "right", BK.OPEN_T), ("move_to", {"right": lambda: ep.mem["clamp_start"]["right"]}, BK.RIGHT_BACK_T,
                                         S.APPROACH_IN), ("hold", 0.2)]
    low = [("move_to", {s: (lambda s=s: ep.mem["lift_start"][s]) for s in S.SIDES}, BK.LOWER_T, S.LIFT), ("hold", 0.5),
           ("payload", False)]
    stop = [("stop", BK.STOP_T)]
    return {"pinch": [("release_left", stop + rel)], "push": [("release_left", stop + rel)],
            "clamp": [("right_back", stop + rb), ("release_left", rel)],
            "lift": [("lower", stop + low), ("right_back", rb), ("release_left", rel)]}[seg]


def run_world(tag, world, skills="v1"):
    from . import segments as S
    ep = S.Episode(world, observe="zed", obs_seed=S.world_seed(tag), **PLAN)
    out = dict(tag=tag, world=world, obs=ep.obs, post_contract=POST, skills_version=skills, skills=[])
    out["approach"] = dict(ok=not ep.refused, why=ep.refused)
    if ep.refused:
        return out
    ep.reset_track()
    ep.run_phases(ep.segment_phases("start"))
    pre_ok = True
    for seg in S.SEGMENTS:
        ep.reset_track()
        ep.sense["pan_at_push_start"] = ep.d.xpos[ep.pan].copy()
        s0 = _state(ep)
        status, mx = _run(ep, ep.segment_phases(seg), skill_v2=seg if skills == "v2" else None)
        mt = _metrics(ep, s0, _state(ep), mx, seg)
        fail = check_post(seg, mt) + ([f"v2:{status}"] if skills == "v2" and status != "done" else [])
        if skills == "v2":
            mt["v2"] = dict(status=mx["v2_status"], why=mx["v2_why"], log=mx["v2_log"])
        rec = dict(skill=seg, pre_ok=pre_ok, post_fail=fail, metrics=mt, backups=[])
        end = ep.snapshot("end")
        for name, ph in backup_skills(ep, seg):          # backup skills chained from this skill's end (each scored alone)
            ep.reset_track()
            b0 = _state(ep)
            _, bmx = _run(ep, ph)
            bm = _metrics(ep, b0, _state(ep), bmx, name)
            rec["backups"].append(dict(skill=name, post_fail=check_post(name, bm), metrics=bm))
        ep.restore(end)
        out["skills"].append(rec)
        pre_ok = pre_ok and not fail
        if mt["U"]:
            break
    return out


def _job(job):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    tag, w, sk = job
    t0 = time.time()
    try:
        r = run_world(tag, w, sk)
    except Exception:
        r = dict(tag=tag, world=w, error=traceback.format_exc()[-1500:])
    r["wall_s"] = round(time.time() - t0, 1)
    return r


def report(out):
    import collections
    recs = [json.loads(p.read_text()) for p in sorted((Path(out) / "raw").glob("*.json"))]
    groups = collections.defaultdict(list)
    for r in recs:
        groups[r["tag"][0] if not r["tag"].startswith("O") else "O"].append(r)
    lines = []
    for g, rs in sorted(groups.items()):
        n = len(rs)
        ap = sum(r.get("approach", {}).get("ok", False) for r in rs)
        lines.append(f"## group {g}*  n = {n}  approach ok {ap}/{n}  errors {sum('error' in r for r in rs)}")
        why = collections.Counter((r.get("approach") or {}).get("why", "")[:60] for r in rs if not (r.get("approach") or {}).get("ok"))
        for k, v in why.most_common():
            lines.append(f"    approach refused {v}: {k}")
        tally = collections.defaultdict(lambda: [0, 0, collections.Counter()])
        for r in rs:
            for sk in r.get("skills", []):
                for name, pf, pre in [(sk["skill"], sk["post_fail"], sk["pre_ok"])] + \
                        [(f"{b['skill']}@{sk['skill']}", b["post_fail"], sk["pre_ok"] and not sk["post_fail"]) for b in sk["backups"]]:
                    t = tally[name]
                    if not pre:
                        t[2]["(pre not met)"] += 1
                        continue
                    t[0] += 1
                    t[1] += not pf
                    for x in pf:
                        t[2][x] += 1
        for name, (nr, ok, c) in tally.items():
            lines.append(f"    {name:22s} post holds {ok}/{nr}   fails {dict(c)}")
    txt = "\n".join(lines)
    (Path(out) / "SUMMARY.txt").write_text(txt + "\n\npost contract (ASSUMED): " + json.dumps(POST) + "\n")
    print(txt)


def main(argv=None):
    from .segcert import worlds
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="full", choices=["full", "iid", "overload"])
    ap.add_argument("--tags", default=None)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--skills", default="v1", choices=["v1", "v2"])
    a = ap.parse_args(argv)
    if a.report:
        return report(a.out)
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    ws = worlds(a.set)
    if a.tags:
        keep = a.tags.split(",")
        ws = [(t, w) for t, w in ws if t in keep]
    done = {p.stem for p in (out / "raw").glob("*.json")}
    jobs = [(t, w, a.skills) for t, w in ws if t not in done]
    with get_context("spawn").Pool(a.procs, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_job, jobs):
            (out / "raw" / f"{r['tag']}.json").write_text(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
            print(r["tag"], r.get("approach"), [(s["skill"], s["post_fail"], [(b["skill"], b["post_fail"]) for b in s["backups"]])
                                                 for s in r.get("skills", [])], r.get("error", "")[-300:], r["wall_s"], flush=True)
    (out / f"_DONE_{a.set}").write_text("ok")


if __name__ == "__main__":
    main()
