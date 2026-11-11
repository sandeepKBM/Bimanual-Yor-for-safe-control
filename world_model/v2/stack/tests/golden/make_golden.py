"""stack/tests/golden/make_golden.py -- record TODAY's checker / goal / U / R numbers as a frozen fixture (migration step 0).  SIMULATION ONLY.

Nothing in the repo is edited; this script only calls the current code and writes JSON next to itself.  It records the numbers the unified
stack/tasks, goal, checks and outcome modules must reproduce (admissions, U and R bit-identical; goal and success are new or now uniform).
Re-run (from the worktree root, threads pinned):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
        /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python world_model/v2/stack/tests/golden/make_golden.py [--procs 6] [--only a,b]
Files (see README.md): admission.json, probe_admission.json, runtime_episodes.json, goal_success.json, pan.json, coffee_tray.json.
Every float is stored at full precision (json default); np types are converted with float()/int()/list().
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import multiprocessing as mp
import os
import pathlib
import subprocess
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[4]          # worktree root: golden -> tests -> stack -> v2 -> world_model -> ROOT  (parents[4] of .../world_model/v2/stack/tests/golden)
sys.path.insert(0, str(ROOT))

TASKS = ("box", "tray2", "tray4", "tray6", "pot")
PLANNERS = ("scripted", "closed_chain")
SAMPLE_EVERY = 400              # control steps (DT = 5 ms) between logged _check / _monitor samples (0.5 s)
AC_XY_TOL, AC_Z_TOL = 0.05, 0.02    # copy of agentchord_probe/nero/execute.py (ac_success is an inline expression there, not a function)


def san(x):
    """JSON-safe copy: np scalars/arrays, dataclasses, tuples, sets, non-str keys; floats kept at full precision."""
    import numpy as np
    if isinstance(x, dict):
        return {(k if isinstance(k, str) else repr(k)): san(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [san(v) for v in x]
    if isinstance(x, np.ndarray):
        return san(x.tolist())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return dict(__class__=type(x).__name__, **{f.name: san(getattr(x, f.name)) for f in dataclasses.fields(x)})
    if isinstance(x, pathlib.Path):
        return str(x)
    if hasattr(x, "tolist"):
        return san(x.tolist())
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


def meta():
    def sh(*a):
        try:
            return subprocess.check_output(a, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    # GOLDEN_GIT_SHA: set when run from a pristine export of a commit (no .git there); the committed fixture was made that way from cac058c
    # because other work was editing the tracked files of the live worktree while the runs were going.
    return dict(git_sha=sh("git", "rev-parse", "HEAD") or os.environ.get("GOLDEN_GIT_SHA"), git_branch=sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
                git_dirty_tracked=(bool(sh("git", "status", "--porcelain", "--untracked-files=no")) if sh("git", "rev-parse", "HEAD") else None),
                date=time.strftime("%Y-%m-%d %H:%M:%S"), python=sys.version.split()[0],
                env_flags={k: v for k, v in sorted(os.environ.items()) if k.startswith(("BIMANUAL", "PL_", "OMP_", "OPENBLAS_", "MKL_", "NUMBA_", "MUJOCO_"))},
                make_golden="world_model/v2/stack/tests/golden/make_golden.py")


def write(name, data, extra_meta=None):
    out = dict(meta=dict(meta(), **(extra_meta or {})), data=san(data))
    (HERE / name).write_text(json.dumps(out, indent=1))
    print(f"wrote {name} ({(HERE / name).stat().st_size / 1e3:.0f} kB)", flush=True)


def guarded(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:       # a crash is recorded, never silently dropped
        return dict(error=repr(e), tb=traceback.format_exc()[-600:])


# ------------------------------------------------------------------------------------------------- 1. contents admission
def admission():
    from world_model.v2.nero.contents import carry as CR, carry_cert as CC, probe as P
    from world_model.v2.stack.planners import closed_chain as CCP
    out = dict(planners=list(PLANNERS), tasks=list(TASKS), cells={}, support_row={}, c_admit_max={})
    for task in TASKS:
        cs = CC.cells(task)
        pick = [cs[0][0], cs[len(cs) // 2][0], cs[-1][0]]                          # 3 cells: lightest, middle, heaviest
        for planner in PLANNERS:
            c_rows = []
            for i, cell in enumerate(pick):
                cell = tuple(cell)
                seed = P._seed("e2e", task, i)
                world = P._world(task, seed)                              # same world helper as probe.py / same_worlds.py
                st = guarded(CC.static_cell, task, cell, world, planner=planner)
                c_rows.append(dict(task=task, planner=planner, cell=list(cell), world_i=i, world_seed=seed, world_theta=guarded(CR._theta, world),
                                   static_cell=st))
                spec = CC.cell_spec(task, cell)
                disp = None if planner == "scripted" else guarded(CCP.goal_offset, task)
                out["support_row"][f"{task}|{planner}|{i}"] = dict(
                    task=task, planner=planner, cell=list(cell), world_i=i, disp=disp,
                    row=guarded(CC.support_row, task, spec, spec.contents, world, None if isinstance(disp, dict) else disp),
                    row_default_disp=guarded(CC.support_row, task, spec, spec.contents, world, None))
            out["cells"][f"{task}|{planner}"] = c_rows
            out["c_admit_max"][f"{task}|{planner}"] = dict(task=task, planner=planner, prior=list(P.prior_contents(task)),
                                                           value=guarded(P.c_admit_max, task, planner=planner))
            print("admission", task, planner, flush=True)
    return out


# ------------------------------------------------------------------------------------------------- 2. probe admission
def probe_admission():
    import types
    from world_model.v2.nero.contents import envelope as E, probe as P
    from world_model.v2.stack.planners import closed_chain as CCP  # noqa: F401 (import side effects as in the probe)
    out = dict(consts=dict(U_ADMIT=E.U_ADMIT, A_NOM=E.A_NOM, O_MAX=P.O_MAX), rung_torque_u={}, admissible={}, interval={})
    old = os.environ.get("BIMANUAL_PLANNER")
    try:
        for planner in PLANNERS:
            os.environ["BIMANUAL_PLANNER"] = planner                       # rung_torque_u reads the process default planner
            for task in TASKS:
                me = P.empty_mass(task)
                lo, hi = P.prior_contents(task)
                grid = sorted({round(me + lo + f * (hi - lo), 4) for f in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)} | {round(me, 4), round(me + 0.5 * lo, 4), 3.0 if task == "box" else 1.0})
                out["rung_torque_u"][f"{task}|{planner}"] = dict(task=task, planner=planner, empty_mass=me, prior=[lo, hi],
                                                                 m=[dict(m_ff=m, u=guarded(P.rung_torque_u, task, m)) for m in grid])
                ns = lambda m_lo: types.SimpleNamespace(task=task, M_lo=m_lo)      # Probe.admissible reads only self.task, self.M_lo
                rows = []
                for m_lo in (me + lo, me + lo + 0.5 * (hi - lo)):
                    for m_to in (m_lo, m_lo + 0.5, m_lo + 1.0, m_lo + 1.5):
                        ok, det = P.Probe.admissible(ns(m_lo), m_to)
                        rows.append(dict(M_lo=m_lo, m_to=m_to, admissible=ok, detail=det))
                out["admissible"][f"{task}|{planner}"] = dict(task=task, planner=planner, rows=rows)
                print("probe_admission", task, planner, flush=True)
    finally:
        if old is None:
            os.environ.pop("BIMANUAL_PLANNER", None)
        else:
            os.environ["BIMANUAL_PLANNER"] = old
    for task in TASKS:
        lo, hi = P.prior_contents(task)
        ivs = [(lo, lo + 0.5 * (hi - lo)), (lo + 0.5 * (hi - lo), hi)]       # 2 intervals per task
        for planner in PLANNERS:
            rows = []
            for c_lo, c_hi in ivs:
                rows.append(dict(c_lo=c_lo, c_hi=c_hi, interval_spec=P.interval_spec(task, c_lo, c_hi),
                                 static_interval=guarded(P.static_interval, task, c_lo, c_hi, planner=planner)))
            out["interval"][f"{task}|{planner}"] = dict(task=task, planner=planner, rows=rows)
    return out


# ------------------------------------------------------------------------------------------------- 3/4. episodes: U, R, goal, success
def _tilt(CR, R):
    return CR._tilt_deg(R.reshape(3, 3))


def _carry_state(ep, CR, np, mujoco):
    """the quantities CarryEpisode._check reads, at this instant (read-only)."""
    d = ep.d
    st = dict(k=int(ep.k), obj_pos=d.xpos[ep.obj].copy(), z0=float(ep.z0), obj_tilt_deg=_tilt(CR, d.xmat[ep.obj]),
              cup_tilt_deg=[_tilt(CR, d.xmat[b]) for b in ep.cups],
              n_contacts=int(d.ncon), min_contact_dist=float(min([c.dist for c in d.contact[:d.ncon]], default=0.0)))
    for s in CR.SIDES:
        F, T = ep.claw_wrench(s)
        st[f"claw_F_{s}"], st[f"claw_T_{s}"] = F, T
    if ep.lid is not None:
        rel, lt = ep._lid_rel()
        st["lid_rel"], st["lid_tilt_deg"] = rel, float(lt)
    return st


def _track_scalars(tr):
    return {k: v for k, v in tr.items() if isinstance(v, (int, float)) or k == "viol"}


def _instrument(log):
    """class-level wrapper of CarryEpisode._check: log inputs (t, tau ratio, state) and outputs (ep.track['viol'], track scalars) every
    SAMPLE_EVERY-th call and at every first U entry; also in_R at the sampled steps.  Reads only; the original runs unchanged."""
    import mujoco
    import numpy as np
    from world_model.v2.nero.contents import carry as CR
    base = CR.CarryEpisode
    orig_check, orig_init = base._check, base.__init__
    if getattr(base, "_golden", False):
        return
    base._golden = True

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        self._glog = []
        self._gcalls = 0
        log.append(self)

    def check(self, t, r):
        self._gcalls += 1
        sample = (self._gcalls % SAMPLE_EVERY == 0)
        pre = dict(t=t, tau_ratio=r, state=_carry_state(self, CR, np, mujoco)) if (sample or self.track["viol"] is None) else None
        had = self.track["viol"] is not None
        orig_check(self, t, r)
        new_u = (not had) and self.track["viol"] is not None
        if pre is not None and (sample or new_u):
            okR, why = self.in_R()
            self._glog.append(dict(call=self._gcalls, kind=("first_U" if new_u else "sample"), inputs=pre,
                                   outputs=dict(viol=self.track["viol"], track=_track_scalars(self.track), in_R=[okR, why])))
    base._check, base.__init__ = check, init


def _final(ep, CR, np, mujoco):
    okR, why = ep.in_R()
    return dict(inputs=dict(state=_carry_state(ep, CR, np, mujoco), tau_ratio=None),
                outputs=dict(viol=ep.track["viol"], track=_track_scalars(ep.track), in_R=[okR, why]))


def episode_job(args):
    """args = (method, task, i).  method "ours" = same_worlds._job (probe -> admit -> carry | refuse); "theirs" = same_worlds ungated (act, no checks).
    closed_chain planner, default probe (poslift), default calib (as same_worlds main).  -> record with the _check/in_R log and the goal calls."""
    method, task, i = args          # method: ours | theirs (ungated baseline_ungated._trial) | theirs_ac (AgentChord cached graph)
    os.environ["BIMANUAL_PLANNER"] = "closed_chain"
    t0 = time.time()
    try:
        import mujoco
        import numpy as np
        from world_model.v2.nero.contents import carry as CR, probe as P, same_worlds as SW
        from world_model.v2.agentchord_probe.nero import execute as X
        log = []
        SW._init_worker()                                                  # strict_claw payload row in U, as both methods run in same_worlds
        _instrument(log)
        cal = json.loads(pathlib.Path(P.default_calib()).read_text())["eps"]
        eps_lo, eps_free = float(cal["EPS_LO"]), float(cal["EPS_FREE"])
        if method == "theirs_ac":          # AgentChord's cached compiled graph (offline, no LLM), as same_worlds --theirs agentchord; U exercised on the heavy box
            r = X.same_world_trial(task, i, AC_PLANS / task)
            r.update(method="theirs", seed_e2e=P._seed("e2e", task, i), planner=CR.resolve_planner(None))
        else:
            r = SW._job((method, task, i, eps_lo, eps_free))
        ep = log[-1]
        fin = _final(ep, CR, np, mujoco)
        okR, whyR = fin["outputs"]["in_R"]
        gc = X.goal_check(ep, task)
        pre_hold = (r.get("carry") or {}).get("pre_hold_viol", r.get("pre_hold_viol"))
        done = (r.get("outcome") in ("carried", "completed")) if method != "theirs" else True     # ungated: the plan always runs to its end
        goal_ok, succ, anyU = X.success_ours(done, okR, ep.track["viol"], pre_hold, gc)
        ac = bool(done and gc["tilt_deg"] < 45 and gc["mug_tilt_deg"] < 45 and gc["xy_err"] <= AC_XY_TOL and gc["z_err"] <= AC_Z_TOL)
        # stack/eval/harness._contents_job's outcome expression (copied: inline there), on the carry record of this episode
        harness_outcome = None
        carry = r.get("carry")
        if carry is not None:
            segs = carry["segments"]
            viol = next((dict(seg=s["seg"], **{k: v for k, v in s["nominal"]["viol"].items() if isinstance(v, (str, int, float))}) for s in segs if s["nominal"]["viol"]), None)
            complete = len(segs) == len(CR.SEGS)
            harness_outcome = "violation" if (viol or carry.get("pre_hold_viol")) else ("ok" if (carry["end_in_R"][0] and complete) else "fail")
        slim = {k: r.get(k) for k in ("task", "i", "seed", "seed_e2e", "method", "planner", "outcome", "why", "M_true", "Theta", "theta_sound", "decision",
                                      "any_U", "first_U", "refuse_end_in_R", "theta_world", "probe_method", "peak_u", "probe_viol", "pre_hold_viol", "end_in_R",
                                      "truth_total_kg", "reason", "goal", "ac_success", "goal_ok", "success_ours", "R_fail", "n_edges_executed", "monitors_fired",
                                      "rejections", "monitor_errors", "track", "sim_s")}
        if carry is not None:
            slim["carry_end_in_R"], slim["carry_pre_hold_viol"] = carry["end_in_R"], carry["pre_hold_viol"]
            slim["carry_segments"] = [dict(seg=s["seg"], nominal_viol=s["nominal"]["viol"], nominal_end_in_R=s["nominal"].get("end_in_R"),
                                           nominal_scalars={k: v for k, v in s["nominal"].items() if isinstance(v, (int, float))}) for s in carry["segments"]]
        if "segments" in r and method == "theirs":
            slim["segments"] = [dict(seg=s["seg"], viol=s["viol"], end_in_R=s["end_in_R"], R_fail=s["R_fail"]) for s in r["segments"]]
        rec = dict(method=method, task=task, i=i, record=slim,
                   check_log=dict(every=SAMPLE_EVERY, n_check_calls=ep._gcalls, samples=ep._glog), end=fin,
                   goal=dict(inputs=dict(task=task, done=done, okR=okR, viol=ep.track["viol"], pre_hold_viol=pre_hold, gc=gc),
                             goal_check=gc, goal_ok=goal_ok, success_ours=succ, any_U=anyU, ac_success=ac, probe_outcome=r.get("outcome"),
                             harness_outcome=harness_outcome,
                             end_pose=dict(obj_pos=ep.d.xpos[ep.obj].copy(), z0=ep.z0, world_obj_xy=list(ep.world.obj_xy))),
                   wall_s=round(time.time() - t0, 1))
        return san(rec)
    except Exception as e:
        return dict(method=method, task=task, i=i, error=repr(e), tb=traceback.format_exc()[-1500:])


AC_PLANS = pathlib.Path("/data/local/ss5772/contents/agentchord_plans_sol_2026-11-05")      # AgentChord's cached compiled graphs (box, pot, tray6)
EP_JOBS = [("ours", t, 0) for t in TASKS] + [("theirs", "box", 0), ("theirs", "tray6", 0), ("ours", "box", 2)]
# AgentChord's cached graph needs langchain_core: run in THEIR venv (/data/local/ss5772/agentchord/venv_nero/bin/python make_golden.py --only episodes_ac)
EP_JOBS_AC = [("theirs_ac", "box", 0), ("theirs_ac", "box", 1), ("theirs_ac", "box", 5), ("theirs_ac", "tray6", 0), ("theirs_ac", "pot", 0)]
# box i = 0, 1, 5: the 11-08 demos3_regen "theirs" rows with any_U (heavy boxes); i = 2: ours carried (no U); tray6 / pot: the other tasks without a stored U row


def episodes(procs, jobs=None):
    ctx = mp.get_context("spawn")
    with ctx.Pool(procs, maxtasksperchild=1) as pool:
        res = pool.map(episode_job, EP_JOBS if jobs is None else jobs, chunksize=1)
    return res


def write_episodes(res, jobs, suffix=""):
    write(f"runtime_episodes{suffix}.json", [dict(method=r["method"], task=r["task"], i=r["i"], error=r.get("error"), tb=r.get("tb"), record=r.get("record"),
                                                 check_log=r.get("check_log"), end=r.get("end"), wall_s=r.get("wall_s")) for r in res],
          dict(jobs=jobs, planner="closed_chain", probe="poslift (default)", sample_every_steps=SAMPLE_EVERY))
    write(f"goal_success{suffix}.json", [dict(method=r["method"], task=r["task"], i=r["i"], error=r.get("error"), **(r.get("goal") or {})) for r in res],
          dict(note="ac_success is an inline expression in execute.run_world; replicated here with AC_XY_TOL/AC_Z_TOL. harness_outcome replicates stack/eval/harness._contents_job's inline expression on the carry record (ours only)."))


# ------------------------------------------------------------------------------------------------- 5. pan
def _pan_instrument(ep, log):
    orig = ep._monitor
    cnt = [0]

    def mon(t, tau_ratio):
        orig(t, tau_ratio)
        cnt[0] += 1
        if cnt[0] % SAMPLE_EVERY == 0:
            ps, cs = ep._last
            log.append(dict(call=cnt[0], t=t, tau_ratio=tau_ratio, pan_state=ps,
                            contacts=dict(support=cs["support"], coll=cs["coll"], fz=cs["fz"], squeeze=cs["squeeze"]),
                            out=dict(margin=ep.track["margin"], worst=ep.track["worst"], viol=ep.track["viol"])))
    ep._monitor = mon


def _pan_run(tag, world, obj, segs):
    from world_model.v2.nero.pan import segments as S
    rec = dict(tag=tag, world=world, obj=obj, steps=[])
    ep = S.Episode(world, observe=None, obj=obj)
    log = []
    _pan_instrument(ep, log)
    rec["refused"] = ep.refused
    rec["R_start_noreach"] = ep.in_R(reach=False, version="v2")
    rec["R_start_reach_v1"] = guarded(ep.in_R, reach=True)
    ep.run_phases(ep.segment_phases("start"))
    for seg in segs:
        ep.reset_track()
        ep.sense["pan_at_push_start"] = ep.d.xpos[ep.pan].copy()
        n0 = len(log)
        status = guarded(ep.run_phases, ep.segment_phases(seg))
        step = dict(seg=seg, status=status, success=guarded(ep.success, seg), track=dict(margin=ep.track["margin"], worst=ep.track["worst"], viol=ep.track["viol"]),
                    in_R_v1_noreach=ep.in_R(reach=False), in_R_v2_noreach=ep.in_R(reach=False, version="v2"), pan_state=ep.pan_state(),
                    monitor_samples=log[n0:])
        rec["steps"].append(step)
        print("pan", tag, seg, flush=True)
        if ep.track["viol"]:
            break
    return rec


def pan_job(which):
    try:
        os.environ.setdefault("MUJOCO_GL", "osmesa")
        from world_model.v2.nero.pan import segments as S
        if which == "pan_corner":     # the world of nero/pan/tests/test_segments.py, obj "pan"
            world = dict(placement="far_left", hidden_mass=0.4, perturb=dict(dx=0.015, dy=-0.015, yaw_deg=8.0, mu_scale=0.6))
            return san(_pan_run(which, world, "pan", ("pinch", "push")))
        from world_model.v2.stack.universes.pot import PotUniverse
        w = PotUniverse().worlds("nominal")[0]
        return san(dict(_pan_run("pot_nominal", w.native, "pot", S.SEGMENTS), theta=w.theta))
    except Exception as e:
        return dict(which=which, error=repr(e), tb=traceback.format_exc()[-1500:])


def pan_certify():
    """cert/admit.certify (sampled) on stored world_trials records (tracked in git: nero/results/pan_cert_v2_2026-10-29/smokeB2/raw, world_trials format with a family per segment)."""
    from world_model.v2.nero.pan.cert import admit as AD
    d = ROOT / "world_model/v2/nero/results/pan_cert_v2_2026-10-29/smokeB2/raw"
    out = dict(source=str(d.relative_to(ROOT)), files={}, certify={})
    recs = []
    for f in ("I03.json", "V10.json", "V11.json"):
        p = d / f
        out["files"][f] = dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)
        recs.append(json.loads(p.read_text()))
    segs = sorted({s["seg"] for r in recs for s in r["segments"]})
    for seg in segs:
        fam = (0.03, 0.01)                         # the family these smoke records were run with (the default BK.FAMILY has 0.06.. -> KeyError)
        out["certify"][seg] = dict(family=list(fam), conf_0_95=guarded(AD.certify, recs, seg, family=fam),
                                   conf_0_9=guarded(AD.certify, recs, seg, family=fam, conf=0.9))
    out["segments_in_records"] = {f: [(s["seg"], s["nominal"]["status"], s["nominal"]["viol"]) for s in r["segments"]] for f, r in zip(("I03", "V10", "V11"), recs)}
    return out


# ------------------------------------------------------------------------------------------------- 6. coffee tray
def coffee_tray():
    """stored ac_run records (/data/local/ss5772/agentchord/runs/*/episodes.jsonl, key tray_check = coffee_tray_check output) + a replay of
    coffee_tray_check on poses rebuilt from the stored values (tray0 = identity at origin; tray tilt is NOT stored, only the flag: 0 deg if level, 15 deg else)."""
    import numpy as np
    runs = pathlib.Path("/data/local/ss5772/agentchord/runs")
    geom = json.loads((ROOT / "world_model/v2/agentchord_probe/coffee_tray/geometry.json").read_text())
    tray_g = geom["tray"]
    inner_x, inner_y = tray_g["lx"] / 2 - tray_g["rim_t"], tray_g["ly"] / 2 - tray_g["rim_t"]
    fwd = np.asarray(geom["forward"], float)

    def check(tray, cap, tray0, loose=False):          # coffee_tray_check body, verbatim (the module needs embodichain to import)
        rel = np.linalg.inv(tray) @ cap
        target = tray0[:2, 3] + fwd
        out = {
            "capsule_in_tray": bool(abs(rel[0, 3]) <= inner_x and abs(rel[1, 3]) <= inner_y and abs(rel[2, 3]) <= 0.04),
            "tray_at_target": bool(np.linalg.norm(tray[:2, 3] - target) <= 0.03),
            "tray_level": bool(np.degrees(np.arccos(np.clip(tray[2, 2], -1, 1))) < 10),
            "capsule_rel_tray": rel[:3, 3].round(4).tolist(),
            "tray_xy_err": float(np.linalg.norm(tray[:2, 3] - target)),
            "tray_moved_forward": bool(tray[0, 3] - tray0[0, 3] >= 0.05 and abs(tray[1, 3] - tray0[1, 3]) <= 0.05),
            "tray_dxy": (tray[:2, 3] - tray0[:2, 3]).round(4).tolist(),
        }
        placed = out["tray_moved_forward"] if loose else out["tray_at_target"]
        out["success_strict"] = out["capsule_in_tray"] and out["tray_at_target"] and out["tray_level"]
        out["success_loose"] = out["capsule_in_tray"] and out["tray_moved_forward"] and out["tray_level"]
        out["success"] = out["capsule_in_tray"] and placed and out["tray_level"]
        return out

    cands = []
    for p in sorted(runs.glob("*/episodes.jsonl")):
        for ln in p.read_text().splitlines():
            try:
                r = json.loads(ln)
            except Exception:
                continue
            tc = r.get("tray_check")
            if isinstance(tc, dict) and "capsule_rel_tray" in tc:
                cands.append((p.parent.name, r, tc))
    wanted = [("strict_success", lambda t: t.get("success_strict")), ("loose_only", lambda t: t.get("success_loose") and not t.get("success_strict")),
              ("not_level", lambda t: not t.get("tray_level")), ("capsule_out", lambda t: not t.get("capsule_in_tray")),
              ("level_capsule_in_not_moved", lambda t: t.get("tray_level") and t.get("capsule_in_tray") and not t.get("tray_moved_forward"))]
    items = []
    for label, pred in wanted:
        hit = next(((n, r, t) for n, r, t in cands if pred(t)), None)
        if hit is None:
            items.append(dict(label=label, found=False))
            continue
        name, r, tc = hit
        tray0 = np.eye(4)
        tray = np.eye(4)
        tray[:2, 3] = tc["tray_dxy"]
        if not tc["tray_level"]:
            a = np.radians(15.0)
            tray[:3, :3] = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        capm = np.eye(4)
        capm[:3, 3] = (tray @ np.array([*tc["capsule_rel_tray"], 1.0]))[:3]
        loose = "min" in name                                          # run dirs of the minimal-prompt variant (CoffeeTrayMin = loose_success)
        replay = check(tray, capm, tray0, loose=loose)
        same = {k: (bool(np.allclose(replay[k], tc[k], atol=1e-3)) if isinstance(tc[k], (float, list)) else replay[k] == tc[k]) for k in tc if k in replay}
        items.append(dict(label=label, found=True, run=name, episode=r.get("episode"), seed=r.get("seed"), their_success=r.get("their_success"),
                          stored_check_detail=tc, replay_inputs=dict(tray0=tray0, tray=tray, capsule=capm, loose=loose, tilt_assumed_deg=(0.0 if tc["tray_level"] else 15.0)),
                          replay_check_detail=replay, replay_matches_stored_fields=same))
    return dict(source=str(runs), n_records_with_check=len(cands),
                note=("Raw poses are NOT stored in ac_run records, only the check_detail dict (capsule_rel_tray, tray_dxy, flags).  tray0, the tray pose and the capsule pose "
                      "were rebuilt from those values (tray0 = identity; tilt 0 deg if tray_level else 15 deg, ASSUMED); the replay reproduces the stored fields except tray_xy_err "
                      "(stored value uses the true tray0)."), items=items)


# ------------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--only", default="admission,probe_admission,episodes,pan,coffee_tray")
    a = ap.parse_args(argv)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    only = set(a.only.split(","))
    t0 = time.time()
    if "admission" in only:
        write("admission.json", admission())
    if "probe_admission" in only:
        write("probe_admission.json", probe_admission())
    if "episodes" in only:
        write_episodes(episodes(a.procs), EP_JOBS)
    if "episodes_ac" in only:
        write_episodes(episodes(a.procs, EP_JOBS_AC), EP_JOBS_AC, "_theirs_ac")
    if "pan" in only:
        eps = [pan_job(w) for w in ("pan_corner", "pot_nominal")]       # in-process: Episode spawns its own pool (no daemonic parent allowed)
        write("pan.json", dict(episodes=eps, certify=guarded(pan_certify)))
    if "coffee_tray" in only:
        write("coffee_tray.json", coffee_tray())
    print(f"total wall {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
