"""tilt_measure.py -- MEASURE the peak object tilt of the nominal carry per (cell, acceleration level)  (checker step K4).  SIMULATION ONLY.

    python -m world_model.v2.nero.contents.tilt_measure --task box --a 0.25,0.5 --n 20 --procs N --out DIR [--cells 0.5-1.0,1.0-1.5]

For each cell of carry_cert.cells(task) and each a: n episodes, world = carry_cert.sample_world(task, cell, rng(seed)), seed = cell_seed(task, cell, i)
(the same worlds as the certification), nominal carry = hold 0.4 s + all SEGS (world_trial, branch_backups=False), m_hat = the cell's centre, the
default carry physics.  Per episode: peak obj_tilt_deg per segment (track obj_tilt_max, reset per segment) and overall.  Output: episodes.jsonl
(one line per episode, restartable), TABLE.json, README.md.  An episode that crashes is recorded with `error` and excluded from the statistics
(counted in n_errors).  A segment sequence cut short by a U violation (the run stops) is recorded (`stopped_at`) and its peak covers the segments run."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import time

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

NOISE = "simulation only: nominal carry (no sensing noise, no injected disturbance); worlds i.i.d. uniform on the cell; peak of the object z-axis tilt from vertical"
LEAVES_OUT = ("weld compliance of the real gripper (sim grasp = stiff rigid weld, config.WELD_STIFF), sensing / estimation error, base motion, "
              "slosh, controller latency, worlds outside the cell")


def u_limit_deg(task):
    """the task's U limit on obj_tilt_deg (stack/tasks.py, pure data); a peak tilt >= it counts as over (conservative for strict limits too)."""
    from world_model.v2.stack import tasks as TK
    return float(dict(TK.task(task).U)["obj_tilt_deg"].hi)


def cell_key(task, cell):
    return f"{task}|{cell[0]}-{cell[1]}"


def _episode(args):
    task, cell, a, i, seed, planner = args
    os.environ["BIMANUAL_PLANNER"] = planner
    import numpy as np
    from . import carry as CR
    from . import carry_cert as CC
    t0 = time.time()
    key = cell_key(task, cell)
    base = dict(key=key, task=task, cell=list(cell), a=a, i=i, seed=seed, planner=planner)
    try:
        w = CC.sample_world(task, cell, np.random.default_rng(seed))
        c = CC.cell_spec(task, cell).contents
        r = CR.world_trial(w, f"{key}_a{a}_{i}", a_plan=a, m_hat=0.5 * (c.m_lo + c.m_hi), branch_backups=False)
        segs = {s["seg"]: s["nominal"]["obj_tilt_max"] for s in r["segments"]}
        viol = next(((s["seg"], s["nominal"]["viol"]) for s in r["segments"] if s["nominal"]["viol"]), None)
        out = dict(base, tilt_seg=segs, tilt_max=max(segs.values()) if segs else None, pre_hold_viol=r.get("pre_hold_viol"),
                   stopped_at=list(viol) if viol else None, wall_s=round(time.time() - t0, 1))
    except Exception as e:
        out = dict(base, error=repr(e), wall_s=round(time.time() - t0, 1))
    return out


def _git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=pathlib.Path(__file__).parent).strip()
    except Exception:
        return "?"


def rot_summary(ok):
    """per-piece max over the episodes of RotTrack's swing / twist / drift (MEASURED, a max over n: not a bound), or None when no episode carries `rot`.
    n_missing_rot = ok episodes without a `rot` record (the consumer treats > 0 as MISSING)."""
    import numpy as np
    have = [r for r in ok if r.get("rot")]
    if not have:
        return None
    seg = {}
    for r in have:
        for s, v in r["rot"]["seg"].items():
            x = seg.setdefault(s, dict(swing_deg=0.0, twist_deg=0.0, drift_m=0.0, drift_time_matched_m=0.0, n_episodes=0))
            x["swing_deg"], x["twist_deg"] = max(x["swing_deg"], v["swing_deg"]), max(x["twist_deg"], v["twist_deg"])
            x["drift_time_matched_m"] = max(x["drift_time_matched_m"], v.get("drift_time_matched_m", 0.0))
            x["drift_m"], x["n_episodes"] = max(x["drift_m"], v["drift_m"]), x["n_episodes"] + 1
    piv = np.array([r["rot"]["pivot_obj"] for r in have])
    x0 = np.array([r["rot"]["x0"] for r in have])
    R0 = np.array([r["rot"]["R0"] for r in have])
    return dict(per_seg_max=seg, n_rot=len(have), n_missing_rot=len(ok) - len(have), pivot_obj=piv.mean(0).tolist(),
                pivot_obj_spread_m=float(np.linalg.norm(piv - piv.mean(0), axis=1).max()), x0=x0.mean(0).tolist(),
                x0_spread_m=float(np.linalg.norm(x0 - x0.mean(0), axis=1).max()), R0=R0[0].tolist(),
                R0_spread_rad=float(max(np.arccos(np.clip((np.trace(R0[0].T @ Rk) - 1) / 2, -1, 1)) for Rk in R0)),
                n_unattributed_steps=int(sum(r["rot"]["n_unattributed"] for r in have)), n_pre_unfolded=int(sum(r["rot"].get("n_pre_unfolded", 0) for r in have)),
                unattributed_max_deg=dict(swing=max(r["rot"]["unattributed_max_deg"]["swing"] for r in have), twist=max(r["rot"]["unattributed_max_deg"]["twist"] for r in have)),
                what="RotTrack (tilt_measure.py): per plan piece, max over episodes of swing / twist [deg] of R_act R_nom^T and pivot drift [m] (pivot = object-fixed claw midpoint at k = 0)")


def build_table(rows, tasks, a_levels, n, meta):
    import numpy as np
    cells = {}
    for r in rows:
        cells.setdefault(r["key"], {}).setdefault(repr(float(r["a"])), []).append(r)
    tab = {}
    for key, lv in cells.items():
        task = key.split("|")[0]
        ul = u_limit_deg(task)
        levels = {}
        for a, rs in lv.items():
            ok = sorted((r for r in rs if "error" not in r and r.get("tilt_max") is not None), key=lambda r: r["i"])
            t = np.array([r["tilt_max"] for r in ok])
            n_err, n_stop = len(rs) - len(ok), sum(1 for r in ok if r.get("stopped_at"))
            n_over = int((t >= ul).sum()) if len(t) else 0
            n_fail = n_err + sum(1 for r in ok if r.get("stopped_at") or r["tilt_max"] >= ul)      # union: errored, stopped early, or peak >= U
            levels[a] = dict(n=len(ok), n_total=len(rs), n_errors=n_err, n_stopped=n_stop, n_over=n_over, n_fail=n_fail, u_limit_deg=ul,
                             p50=round(float(np.percentile(t, 50)), 4) if len(t) else None, p95=round(float(np.percentile(t, 95)), 4) if len(t) else None,
                             max=round(float(t.max()), 4) if len(t) else None,
                             per_seg_max={s: round(max(r["tilt_seg"].get(s, 0.0) for r in ok), 4) for s in (("lift", "carry", "place", "release") + tuple(sorted({k for r in ok for k in r["tilt_seg"]} - {"lift", "carry", "place", "release"})))} if ok else None,
                             seeds=[r["seed"] for r in rs])
            rot = rot_summary(ok)
            if rot is not None:
                levels[a]["rot"] = rot
        tab[key] = dict(task=task, planner=meta["planner"], u_limit_deg=ul, levels=levels)
    return dict(cells=tab, noise_model=NOISE, leaves_out=LEAVES_OUT, n_requested=n, a_levels=a_levels, tasks=tasks, **meta)


# ---------------------------------------------------------------------------------------------------------------------------- per-plan measurement (11-17, USER "measure per plan")
N_PLAN_DEFAULT = 20           # ASSUMED: worlds per plan measurement (= the table measurer's default n); a max over n, not a bound
SEED_SALT = "tilt_measure_plan_v1_salt_a7f3"      # fixed salt: the measurement worlds are never the run's own worlds (probe._seed("e2e", ...) / behind_world seeds)
_PLAN_CACHE = {}              # IN-RUN cache only (USER 11-17: no cross-run disk replay; every run measures its plan fresh): (task, plan_id, cell, a, level, n, code) -> table


def plan_seeds(task, plan_id, cell, n, avoid=()):
    """n seeds = sha256(salt | task | plan_id | cell | i) (re-hashed on a clash with `avoid` or the run's world seeds probe._seed("e2e", base task, j), j < 512)."""
    import hashlib
    from . import probe as PR
    from ...stack import tasks as TK
    bt = TK.fixed_task(task)
    bad = {int(x) for x in avoid} | {PR._seed("e2e", bt, j) for j in range(512)}
    out = []
    for i in range(n):
        k = 0
        while True:
            h = int(hashlib.sha256(f"{SEED_SALT}|{task}|{plan_id}|{cell[0]}-{cell[1]}|{i}|{k}".encode()).hexdigest()[:8], 16)
            if h not in bad and h not in out:
                out.append(h)
                break
            k += 1
    return out


def _plan_world(task, cell, seed):
    """a world drawn i.i.d. uniform from the CURRENT belief cell (carry_cert.sample_world on the base task), with the task's world_kw as smoke.behind_world applies them."""
    import numpy as np
    from dataclasses import replace
    from . import carry_cert as CC
    from ...stack import tasks as TK
    bt = TK.fixed_task(task)
    w = CC.sample_world(bt, cell, np.random.default_rng(seed))
    if bt == "box" and TK.TASKS[task].base:
        w = replace(w, **dict(TK.task("box_behind").world_kw))
    if task in TK.TASKS:
        w = replace(w, **dict(TK.task(task).world_kw))
    return w


def swing_twist(Rrel):
    """swing-twist split of a WORLD-frame rotation Rrel = R_swing @ R_twist (R_twist about world z, R_swing about a horizontal axis) -> (swing, |twist|) [rad].
    swing = angle(Rrel e_z, e_z) (= the obj_tilt_deg of a level nominal pose); twist = the yaw left after removing the swing."""
    import numpy as np
    v = Rrel[:, 2]
    sw = float(np.arccos(np.clip(v[2], -1.0, 1.0)))
    ax = np.array([-v[1], v[0], 0.0])                      # e_z x v
    s = float(np.linalg.norm(ax))
    if s < 1e-12:
        Rs = np.eye(3)
    else:
        k = ax / s
        Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        Rs = np.eye(3) + np.sin(sw) * Kx + (1 - np.cos(sw)) * Kx @ Kx
    Rt = Rs.T @ Rrel
    return sw, abs(float(np.arctan2(Rt[1, 0], Rt[0, 0])))


class RotTrack:
    """MEASURED (11-17, collision row under load tilt): per control step, the deviation of the carried object from where the closed_chain plan puts it, as the
    collision row (adapters/collision_rows) places it: nominal pose = the model's start pose (m.qpos0 of the object's free joint, level) translated by the active piece's planned object offset p[kk]
    (kk = the time-matched sample, clipped to the piece) and, after the base move, the PLANNED base transform (geo.info["base"]), exactly as collision_rows._Scene.set_state.
    Pivot = the object-fixed point at the claw midpoint at k = 0 (mean of the two *_arm_ee_grasp sites, in the ACTUAL k = 0 object frame), stored in the object frame (pivot_obj).
    The k = 0 deviation of the actual pose from qpos0 (settling) is recorded (k0_dev) and is part of the measured swing / drift.
    Per step: R_rel = R_act R_nom^T split into swing (horizontal axis) and twist (about z) [swing_twist]; drift_time_matched = |actual pivot - nominal pivot at the time-matched sample| (holds the
    tracking lag along the path); drift_m (used by the row) = distance of the actual pivot to the nearest point of THE SAME piece's nominal pivot path (linear between
    samples), which is what the row needs (static world: the set of poses matters, not their timing; the row covers that piece's samples and the segments between them).  Steps counted: weld on, base driver not active, a plan piece active (attributed to that piece); the steps before the first piece
    (the pre-lift hold) are attributed to the FIRST piece at its sample 0.  Welded steps outside these (base driving, no active piece after the first) are counted, not attributed."""

    def __init__(self, ep):
        import numpy as np
        import mujoco
        m, d = ep.m, ep.d
        xa, Ra = d.xpos[ep.obj].copy(), d.xmat[ep.obj].reshape(3, 3).copy()          # actual pose at k = 0 (may have settled a little off the model pose)
        cm = 0.5 * (d.site_xpos[ep.site["left"]] + d.site_xpos[ep.site["right"]])
        self.o_c = Ra.T @ (cm - xa)                                                   # pivot: object-fixed point at the claw midpoint at k = 0
        j = next(j for j in range(m.body_jntadr[ep.obj], m.body_jntadr[ep.obj] + m.body_jntnum[ep.obj]) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
        adr = int(m.jnt_qposadr[j])
        self.x0 = np.array(m.qpos0[adr:adr + 3], float)                               # NOMINAL start pose = the model's qpos0 (what collision_rows places at p = 0)
        R0 = np.zeros(9)
        mujoco.mju_quat2Mat(R0, np.array(m.qpos0[adr + 3:adr + 7], float))
        self.R0 = R0.reshape(3, 3)
        self.c0 = self.x0 + self.R0 @ self.o_c                                         # nominal pivot at p = 0
        self.k0_dev = dict(dx_m=float(np.linalg.norm(xa - self.x0)), swing_twist_deg=[float(np.degrees(v)) for v in swing_twist(Ra @ self.R0.T)])
        self.pre, self.stats, self.seen = [], {}, False
        self.n_base = self.n_unattr = 0
        self.unattr_max = dict(swing=0.0, twist=0.0)
        self._np = np

    def _nominal(self, cc, name, p):
        np = self._np
        P, Rn = self.c0 + np.asarray(p, float), self.R0
        names = list(cc.pieces)
        wp = getattr(cc.inp, "wp", None)
        if wp is not None and getattr(wp, "base", None) and names.index(name) >= wp.base[0][0]:
            from world_model.v2.stack.planners import base_route as BR
            b = cc.geo.info["base"]
            c0, ce = np.array(b["start_pose"][:2], float), np.array(b["end_pose"][:2], float)
            dl = float(b["end_pose"][2] - b["start_pose"][2])
            P = np.r_[ce + BR.rot(dl) @ (P[:2] - c0), P[2]]
            Rz = np.array([[np.cos(dl), -np.sin(dl), 0.0], [np.sin(dl), np.cos(dl), 0.0], [0.0, 0.0, 1.0]])
            Rn = Rz @ Rn
        return P, Rn

    def _poly(self, cc, name):
        """the nominal pivot path of piece `name` (one node per planned sample) -> (A (n,3), B (n,3)) segment ends (a single node: A = B)."""
        if not hasattr(self, "_polys"):
            self._polys = {}
        if name not in self._polys:
            np = self._np
            Pn = np.array([self._nominal(cc, name, q)[0] for q in cc.pieces[name].p])
            self._polys[name] = (Pn[:-1], Pn[1:]) if len(Pn) > 1 else (Pn, Pn)
        return self._polys[name]

    def _add(self, name, x, R, P, Rn, cc):
        np = self._np
        sw, tw = swing_twist(R @ Rn.T)
        Pa = x + R @ self.o_c
        dr_t = float(np.linalg.norm(Pa - P))                                      # time-matched
        A, B = self._poly(cc, name)                                               # set-based: nearest point of this piece's nominal pivot path (linear between samples = the row's model)
        AB = B - A
        u = np.clip(np.einsum("ij,ij->i", Pa - A, AB) / np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-18), 0.0, 1.0)
        dr = float(np.linalg.norm(A + u[:, None] * AB - Pa, axis=1).min())
        s = self.stats.setdefault(name, dict(swing=0.0, twist=0.0, drift=0.0, drift_t=0.0, n=0))
        s["swing"], s["twist"], s["drift"], s["drift_t"], s["n"] = max(s["swing"], sw), max(s["twist"], tw), max(s["drift"], dr), max(s["drift_t"], dr_t), s["n"] + 1

    def __call__(self, ep):
        d = ep.d
        if not all(d.eq_active[ep.weld_id[s]] for s in ("left", "right")):
            return
        x, R = d.xpos[ep.obj].copy(), d.xmat[ep.obj].reshape(3, 3).copy()
        drv = getattr(ep, "driver", None)
        if drv is not None and getattr(drv, "active", False):
            self.n_base += 1
            return
        ap = getattr(ep, "_active_piece", None)
        cc = getattr(ep, "cc", None)
        name = None
        if ap is not None and cc is not None:
            pc, t0 = ap
            name = next((k for k, v in cc.pieces.items() if v is pc), None)
        if name is None:
            if not self.seen:
                self.pre.append((x, R))
            else:
                self.n_unattr += 1
                sw, tw = swing_twist(R @ self.R0.T)
                self.unattr_max = dict(swing=max(self.unattr_max["swing"], sw), twist=max(self.unattr_max["twist"], tw))
            return
        self.seen = True
        from . import carry as CR
        kk = min(max(int(round(ep.k - t0 / CR.DT)), 0), pc.n)
        P, Rn = self._nominal(cc, name, pc.p[kk])
        self._add(name, x, R, P, Rn, cc)

    def result(self, ep):
        """-> JSON dict (degrees / metres)."""
        import math
        cc = getattr(ep, "cc", None)
        if cc is not None and self.pre:
            first = list(cc.pieces)[0]
            P, Rn = self._nominal(cc, first, cc.pieces[first].p[0])
            for x, R in self.pre:
                self._add(first, x, R, P, Rn, cc)
        seg = {k: dict(swing_deg=math.degrees(v["swing"]), twist_deg=math.degrees(v["twist"]), drift_m=v["drift"], drift_time_matched_m=v["drift_t"], n_steps=v["n"]) for k, v in self.stats.items()}
        return dict(seg=seg, pivot_obj=self.o_c.tolist(), x0=self.x0.tolist(), R0=self.R0.tolist(), c0=self.c0.tolist(), k0_dev=self.k0_dev, n_pre=len(self.pre) if cc is not None else 0,
                    n_pre_unfolded=0 if cc is not None else len(self.pre), n_base_steps=self.n_base, n_unattributed=self.n_unattr,
                    unattributed_max_deg=dict(swing=math.degrees(self.unattr_max["swing"]), twist=math.degrees(self.unattr_max["twist"])))


def _plan_episode(args):
    """one nominal carry of the GIVEN compiled plan (base move included for base tasks; the episode path of behind_job: CarryEpisode(wp=, base="swerve") -> set_belief -> hold -> run_segments).
    11-17: also the per-piece rotation / pivot drift of the load (RotTrack, record key `rot`) for the collision row's tilt cover (adapters/collision_rows)."""
    task, plan_id, wp, cell, a, i, seed, level, *rest = args
    strict_claw = bool(rest[0]) if rest else False               # 11-17: the run's own strict_claw (probe_select passes ep.strict_claw); default False = as before
    os.environ["BIMANUAL_PLANNER"] = "closed_chain"
    import dataclasses
    from . import carry as CR
    from . import carry_cert as CC
    from ...stack import tasks as TK
    t0 = time.time()
    key = cell_key(task, cell)
    base = dict(key=key, task=task, cell=list(cell), a=a, i=i, seed=seed, planner="closed_chain", plan_id=plan_id, level=level)
    try:
        w = _plan_world(task, cell, seed)
        if level is not None and getattr(wp, "base", None):
            wp = dataclasses.replace(wp, base=tuple((x, r, level) for x, r, _ in wp.base))
        td = TK.TASKS[task]
        kw = dict(base="swerve", goal_support=td.goal_support) if td.base else {}
        ep = CR.CarryEpisode(w, m_hat=0.5 * (cell[0] + cell[1]), fric_comp=True, weld_solref=CC.WELD_STIFF, wp=wp, a_plan=a, strict_claw=strict_claw, **kw)
        rt = RotTrack(ep)                                                    # 11-17: per-piece load rotation + pivot drift (collision row tilt cover)
        ep.step_hooks.append(rt)
        ep.set_belief(0.5 * (cell[0] + cell[1]), a_plan=a)
        ep.run_phases([("hold", 0.4)])
        r = CR.run_segments(ep, dict(segments=[]), branch_backups=False)
        segs = {s["seg"]: s["nominal"].get("obj_tilt_max", 0.0) for s in r["segments"]}
        viol = next(((s["seg"], s["nominal"]["viol"]) for s in r["segments"] if s["nominal"]["viol"]), None)
        if viol is None and ep.track["viol"]:                                # a U outside the segment records (base flags, pre-hold)
            viol = ("episode", CR._jsonable(ep.track["viol"]))
        out = dict(base, tilt_seg=segs, tilt_max=max(segs.values()) if segs else None, pre_hold_viol=r.get("pre_hold_viol"),
                   stopped_at=list(viol) if viol else None, rot=rt.result(ep), wall_s=round(time.time() - t0, 1))
    except Exception as e:
        out = dict(base, error=repr(e), wall_s=round(time.time() - t0, 1))
    return out


def measure_plan(task, wp, cell, a, n=N_PLAN_DEFAULT, seed_base="", procs=1, out_dir=None, level=None, plan_id="", strict_claw=False):
    """MEASURE the peak object tilt of THIS compiled plan (closed_chain WaypointPlan, base move included for base tasks) at plan acceleration a (and base `level`) over n worlds drawn
    i.i.d. from the belief cell `cell` = (lo, hi) (seeds plan_seeds(...), recorded), nominal carry, default physics.  Same per-episode records / TABLE.json schema as main().
    Always measured fresh in this run (in-run memo only).  Writes OUT_DIR/{episodes.jsonl, TABLE.json}.  -> (table dict, info dict(table_path, seeds, n, wall_s, procs)).
    n is ASSUMED (a max over n is a measurement, not a bound over the cell); noise model = NOISE (sim only) with base motion now INCLUDED for base tasks."""
    cell = (float(cell[0]), float(cell[1]))
    ck = (task, plan_id, cell, round(float(a), 6), level, int(n), _git_head(), bool(strict_claw))
    if ck in _PLAN_CACHE:
        return _PLAN_CACHE[ck]
    t0 = time.time()
    seeds = plan_seeds(task, plan_id or seed_base, cell, n)
    jobs = [(task, plan_id, wp, cell, float(a), i, seeds[i], level, bool(strict_claw)) for i in range(n)]
    out = pathlib.Path(out_dir) if out_dir else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    rows = []
    if procs and procs > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(int(procs), maxtasksperchild=3) as pool:
            rows = list(pool.imap_unordered(_plan_episode, jobs))
    else:
        rows = [_plan_episode(j) for j in jobs]
    rows.sort(key=lambda r: r["i"])
    meta = dict(planner="closed_chain", git_head=_git_head(), file=str(out / "TABLE.json") if out else None, date=time.strftime("%Y-%m-%d"),
                env={k: v for k, v in os.environ.items() if k.startswith("BIMANUAL_")}, carry_physics="carry_cert default (config.carry_physics())",
                plan_id=plan_id, level=level, n_tag="ASSUMED (n worlds per plan)", base_motion="INCLUDED for base tasks (CarryEpisode base=swerve, the behind_job carry path)",
                strict_claw=bool(strict_claw), seeds_note="sha256(salt | task | plan_id | cell | i): never the run's own world seeds", seeds=seeds)
    tab = build_table(rows, [task], [float(a)], n, meta)
    tab["leaves_out"] = LEAVES_OUT.replace("base motion, ", "")
    info = dict(table_path=meta["file"], seeds=seeds, n=n, wall_s=round(time.time() - t0, 1), procs=int(procs or 1))
    if out is not None:
        with open(out / "episodes.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
        (out / "TABLE.json").write_text(json.dumps(tab, indent=1, default=float))
    _PLAN_CACHE[ck] = (tab, info)
    return tab, info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="box", help="comma list: box, tray2/4/6, pot")
    ap.add_argument("--a", default="0.25,0.5", help="comma list of a_plan levels (m/s^2)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--planner", default=None, help="scripted | closed_chain: sets BIMANUAL_PLANNER in the workers (default: the env / scripted)")
    ap.add_argument("--cells", default="", help="comma list of 'lo-hi' cell filters (default: every cell)")
    a = ap.parse_args(argv)
    import multiprocessing as mp
    from . import carry_cert as CC
    if a.planner:
        os.environ["BIMANUAL_PLANNER"] = a.planner
    from . import carry as CR
    planner = CR.resolve_planner()                          # validated; spawn workers inherit the env
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in a.task.split(",") if t]
    levels = [float(x) for x in a.a.split(",") if x]
    flt = {x for x in a.cells.split(",") if x}
    jl = out / "episodes.jsonl"
    rows = [json.loads(l) for l in jl.read_text().splitlines() if l.strip()] if jl.exists() else []
    if any(r.get('planner', planner) != planner for r in rows):
        raise SystemExit(f'{jl} holds episodes of another planner than {planner!r}: use a separate --out')
    done = {(r["key"], repr(float(r["a"])), r["i"]) for r in rows}
    jobs = []
    for task in tasks:
        for (cell,) in CC.cells(task):
            if flt and f"{cell[0]}-{cell[1]}" not in flt:
                continue
            for lv in levels:
                for i in range(a.n):
                    if (cell_key(task, cell), repr(lv), i) not in done:
                        jobs.append((task, cell, lv, i, CC.cell_seed(task, cell, i), planner))
    print(f"{len(jobs)} episodes to run ({len(rows)} already done)", flush=True)
    with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=3) as pool, open(jl, "a") as f:
        for r in pool.imap_unordered(_episode, jobs):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            rows.append(r)
            print(f"{r['key']} a={r['a']} i={r['i']} tilt={r.get('tilt_max')} {r.get('error', '')} {r['wall_s']}s", flush=True)
    meta = dict(planner=planner, git_head=_git_head(), file=str(out / "TABLE.json"), date=time.strftime("%Y-%m-%d"),
                env={k: v for k, v in os.environ.items() if k.startswith("BIMANUAL_")}, carry_physics="carry_cert default (config.carry_physics())")
    tab = build_table(rows, tasks, levels, a.n, meta)
    tmp = out / "TABLE.json.tmp"
    tmp.write_text(json.dumps(tab, indent=1, default=float))
    tmp.replace(out / "TABLE.json")
    (out / "README.md").write_text(
        f"# Peak object tilt of the nominal carry (K4)\n\nMeasured: peak object tilt (deg) over hold + lift/carry/place/release, per cell x a_plan, n={a.n} worlds per row, "
        f"worlds = carry_cert.sample_world with carry_cert.cell_seed (seeds in TABLE.json). Statistics: p50 / p95 / max over n (a max over n, not a bound).\n\n"
        f"Noise model: {NOISE}.\n\nLeaves out: {LEAVES_OUT}.\n\nplanner {planner}; git HEAD {meta['git_head']}; BIMANUAL_* env: {meta['env']}.\n"
        f"Consumer: stack/checks/tilt.py tilt_row (next HIGHER measured level; errored / stopped-early / over-U episodes count as failures; CP bound).\n")
    print("wrote", out / "TABLE.json", flush=True)


if __name__ == "__main__":
    main()
