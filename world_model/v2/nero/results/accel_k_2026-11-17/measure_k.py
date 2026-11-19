"""k = ACHIEVED / PLANNED object acceleration on the motions Rule 1 certifies (user ruling 11-17, row side).  MEASUREMENT only (sim, no checker edits).

Per episode (fresh seed, recorded): world = carry_cert.sample_world(task, cell, rng); CarryEpisode(w, a_plan=a, planner="closed_chain");
ep.set_belief(cell centre, a_plan=a)  (REQUIRED: else the arms' feed-forward uses the prior centre); hold 0.4 s; then the executor's run_segments (certs=None = the planner's
STORED backups): each segment's nominal run records the branch snapshots (every 0.25 s inside the move + t = 0 after the first step + the end), and the counterfactual family
restores each snapshot and runs the stored backup (stop -> retrace of this and earlier moves at A_BACKUP -> hold -> weld off -> retreat -> hold).  That IS "an interrupt at each
Rule 1 branch time, then the stored backup" (same snapshots, same recovery as executor.run_segments(interrupt=...)).  One step hook traces the object body's linear velocity
(MuJoCo free-joint qvel, world frame) every control step (DT = 5 ms) with the phase label.

Achieved object acceleration at step i, window w steps (1, 4, 10 = 5, 20, 50 ms): (v_i - v_{i-w}) / (w DT), per world axis and as the norm; v is the first difference of the object
velocity (the 5 ms case = the first difference of the linear velocity over one control step).  Windows never cross a trace start (a restore); a window is attributed to the phase of its END step.
Planned: Piece.pdd of the piece being tracked (ep._active_piece).  Two denominators, both reported:
  k_plan = |achieved| / a_ref  with a_ref = max |pdd| over the active piece (all axes, all steps) = the planned peak per-axis object acceleration of THAT piece
           (= its TOPP-RA cap: a for the nominal pieces, the retrace/advance a_ (A_BACKUP) for the backup pieces; the stop uses the piece it interrupted);
  k_a    = |achieved| / a  (the nominal per-axis box Rule 1's torque/payload rows are certified on, envelope.accel_corners(a)).
Phases: nominal_piece | piece_boundary (kk <= 10 steps after a piece start, or <= 10 steps after a piece end) | piece_end (the last 10 steps of a piece, arrival at rest) | hold ; backup: stop | retrace_piece | piece_boundary | hold |
post_release (weld / retreat / jretreat: the object is let go; kept in the phase rows, excluded from the kind-level max).  The "release" segment is excluded from the kind level.
A stored backup has no read; its "hold" after the stop is the hold a read_then_retrace program would read in.

MEASURED, statistic "max over n (not a bound)", noise_model "sim, deterministic per world"; leaves out: real-robot tracking, sensing, worlds outside the cells.
usage: measure_k.py --task box --cells 0.5-1.0,2.0-2.5 --n 20 --procs 8 --out DIR --a 0.5 [--table-only]
"""
import os, sys, json, time, argparse, pathlib, math

HERE = pathlib.Path(__file__).resolve().parent
WIN = (1, 4, 10)                               # steps: 5 / 20 / 50 ms
WMS = {1: 5, 4: 20, 10: 50}
SEED_BASE = 1117000000                          # 11-17 batch; seed = SEED_BASE + round(100 lo) * 1000 + i  (fresh: no earlier run used this range)
BOUNDARY_STEPS = 10
POST = ("weld", "retreat", "jretreat")


def cell_seed(cell, i):
    return SEED_BASE + int(round(100 * cell[0])) * 1000 + int(i)


def _setup_env(out):
    os.environ.setdefault("BIMANUAL_NO_NETWORK", "1")
    os.environ["BIMANUAL_PLANNER"] = "closed_chain"
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    os.environ["BIMANUAL_PLAN_CACHE"] = str(pathlib.Path(out) / "plan_cache")      # never the repo's cache
    pathlib.Path(os.environ["BIMANUAL_PLAN_CACHE"]).mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------------------------------------------------------ one episode
class Tracer:
    """step hook: records (phase, v, piece tag, kk, a_ref, planned pdd) per step into the current trace."""

    def __init__(self, ep, CR):
        self.ep, self.CR = ep, CR
        m = ep.m
        self.da = int(m.jnt_dofadr[m.body_jntadr[ep.obj]])
        self.cur = None
        self.done = []
        self._ref = {}
        self.last_ref = None
        ep.step_hooks.append(self.hook)

    def a_ref(self, pc):
        r = self._ref.get(id(pc))
        if r is None:
            r = self._ref[id(pc)] = max(float(abs(pc.pdd).max()), 1e-9)
        return r

    def start(self, seg, kind, tr, a_default):
        self.finish()
        d = self.ep.d
        ap = getattr(self.ep, "_active_piece", None)
        self.last_ref = self.a_ref(ap[0]) if ap else a_default
        self.cur = dict(seg=seg, kind=kind, tr=tr, ph="nominal", v0=d.qvel[self.da:self.da + 3].copy(), rows=[], viol=None, a_default=a_default)

    def phase(self, name):
        if self.cur is not None:
            self.cur["ph"] = name

    def finish(self):
        if self.cur is not None:
            self.cur["viol"] = {k: v.get("n") for k, v in (self.ep.track.get("viol_all") or {}).items()} if self.cur["kind"] == "backup" else None
            self.done.append(self.cur)
            self.cur = None

    def hook(self, e):
        c = self.cur
        if c is None:
            return
        DT = self.CR.DT
        ap = getattr(e, "_active_piece", None)
        v = e.d.qvel[self.da:self.da + 3].copy()
        kk, pid, pdd, ref, inpiece = -1, 0, None, self.last_ref, False
        if ap is not None:
            pc, t0 = ap
            kk = int(round(e.k - t0 / DT))
            ref = self.last_ref = self.a_ref(pc)
            pid = id(pc)
            if 0 < kk <= pc.n:
                inpiece = True
                pdd = pc.pdd[kk].copy()
            elif kk > pc.n:
                inpiece = False
        ph = c["ph"]
        if c["kind"] == "nominal":
            if inpiece:
                ph = "piece_boundary" if kk <= BOUNDARY_STEPS else ("piece_end" if kk > pc.n - BOUNDARY_STEPS else "nominal_piece")
            elif ap is not None and 0 < kk - pc.n <= BOUNDARY_STEPS:
                ph = "piece_boundary"
            else:
                ph = "hold"
        elif ph in ("retrace", "advance"):
            if (inpiece and kk <= BOUNDARY_STEPS) or (ap is not None and 0 < kk - pc.n <= BOUNDARY_STEPS):
                ph = "piece_boundary"
            else:
                ph = "piece_end" if (inpiece and kk > pc.n - BOUNDARY_STEPS) else "retrace_piece"
        elif ph in POST:
            ph = "post_release"
        c["rows"].append((ph, v, pid, kk, ref, pdd))


def _stats_of_trace(tr, a):
    """per phase, per window: max over the trace of the achieved accel per axis / norm (absolute, k_plan, k_a) and where it occurs."""
    import numpy as np
    DT = 0.005
    rows = tr["rows"]
    if not rows:
        return {}
    V = np.vstack([tr["v0"]] + [r[1] for r in rows])
    ref = np.array([r[4] for r in rows])
    out = {}
    for w in WIN:
        if len(rows) < w:
            continue
        acc = (V[w:] - V[:-w]) / (w * DT)                       # acc[j] belongs to row j + w - 1 (the END step of the window)
        for j in range(acc.shape[0]):
            r = rows[j + w - 1]
            ph = r[0]
            ab = np.abs(acc[j])
            nm = float(np.linalg.norm(acc[j]))
            kp_ax, kp_nm = ab / ref[j + w - 1], nm / ref[j + w - 1]
            ka_ax, ka_nm = ab / a, nm / a
            d = out.setdefault((ph, w), dict(n=0, abs_axis=np.zeros(3), abs_norm=0.0, kp_axis=np.zeros(3), kp_norm=0.0, ka_axis=np.zeros(3), ka_norm=0.0, at_kp=None, at_ka=None,
                                              plan_peak_axis=np.zeros(3), a_ref=0.0))
            d["n"] += 1
            d["abs_axis"] = np.maximum(d["abs_axis"], ab)
            d["abs_norm"] = max(d["abs_norm"], nm)
            d["kp_axis"] = np.maximum(d["kp_axis"], kp_ax)
            d["ka_axis"] = np.maximum(d["ka_axis"], ka_ax)
            if kp_nm > d["kp_norm"]:
                d["kp_norm"] = float(kp_nm)
                d["at_kp"] = dict(step=j + w - 1, kk=int(r[3]))
            if ka_nm > d["ka_norm"]:
                d["ka_norm"] = float(ka_nm)
                d["at_ka"] = dict(step=j + w - 1, kk=int(r[3]))
            d["a_ref"] = max(d["a_ref"], float(ref[j + w - 1]))
            if r[5] is not None:
                d["plan_peak_axis"] = np.maximum(d["plan_peak_axis"], np.abs(r[5]))
    return out


def _merge(dst, src, key, tr):
    """keep, per (seg, kind, phase, window), the max over the branches of this segment (and which branch)."""
    for (ph, w), d in src.items():
        k = (tr["seg"], tr["kind"], ph, w)
        cur = dst.get(k)
        if cur is None:
            dst[k] = dict(d, n_branches=1, branch_at=tr["tr"])
            dst[k]["n_steps"] = d["n"]
            continue
        cur["n_branches"] += 1
        cur["n_steps"] += d["n"]
        for f in ("abs_axis", "kp_axis", "ka_axis", "plan_peak_axis"):
            import numpy as np
            cur[f] = np.maximum(cur[f], d[f])
        cur["abs_norm"] = max(cur["abs_norm"], d["abs_norm"])
        cur["a_ref"] = max(cur["a_ref"], d["a_ref"])
        for f, at in (("kp_norm", "at_kp"), ("ka_norm", "at_ka")):
            if d[f] > cur[f]:
                cur[f], cur[at] = d[f], d[at]
                cur["branch_at"] = tr["tr"]


def _jsonable(o):
    import numpy as np
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, np.ndarray):
        return [round(float(x), 5) for x in o]
    if isinstance(o, (np.floating, float)):
        return round(float(o), 5)
    if isinstance(o, np.integer):
        return int(o)
    return o


def run_episode(job):
    """job = dict(task, cell, i, seed, a, out).  -> the episode record (also written atomically to out/eps/)."""
    t0 = time.time()
    import numpy as np
    from types import SimpleNamespace
    from world_model.v2.nero.contents import carry as CR, carry_cert as CC
    from world_model.v2.stack import executor as EX
    from world_model.v2.stack.planners.closed_chain import ClosedChainPlanner
    from world_model.v2.stack.types import SkillCall
    task, cell, a, seed = job["task"], tuple(job["cell"]), float(job["a"]), int(job["seed"])
    rec_out = dict(task=task, cell=list(cell), a=a, i=job["i"], seed=seed)
    try:
        w = CC.sample_world(task, cell, np.random.default_rng(seed))
        ep = CR.CarryEpisode(w, a_plan=a, planner="closed_chain")
        ep.set_belief(0.5 * (cell[0] + cell[1]), a_plan=a)        # REQUIRED (feed-forward at the cell centre, as probe_select.py does)
        ep.run_phases([("hold", 0.4)])
        plan = ClosedChainPlanner().plan(None, SkillCall("carry_task", dict(task=task, a_plan=ep.a_plan)), SimpleNamespace(episode=ep))
        tracer = Tracer(ep, CR)
        orig_store = ep.store
        ctx = dict(seg=None)

        def store(seg):
            orig_store(seg)
            ctx["seg"] = seg
            tracer.start(seg, "nominal", 0.0, a)
        ep.store = store

        def runner(ep_, seg, tr, phases):                          # the family branch: snapshot already restored
            tracer.start(seg, "backup", float("inf") if tr is None else float(tr), a)
            for ph in phases:
                tracer.phase(ph[0])
                EX.run_phases(ep_, [ph])
        ep.backup_runner = runner
        nominal_u = []
        orig_rp = ep.run_phases

        def rp(phases, branch=None):                                # run_segments stops the plan after a nominal U violation; we measure the whole plan anyway:
            orig_rp(phases, branch)                                 # note the first U of the window (viol_all keeps every kind), then clear it so the loop continues
            v = ep.track["viol"]
            if v is not None:
                nominal_u.append(dict(kind=v["kind"], t=v.get("t")))
                ep.track["viol"] = None
        ep.run_phases = rp
        rec = dict(segments=[])
        out = EX.run_segments(ep, plan, rec, True)                  # certs=None: the planner's stored backups
        tracer.finish()
        agg = {}
        for tr in tracer.done:
            _merge(agg, _stats_of_trace(tr, a), None, tr)
        viol = dict(nominal={}, backup={})
        for tr in tracer.done:
            if tr["kind"] == "backup" and tr["viol"]:
                for k, n in tr["viol"].items():
                    viol["backup"][k] = viol["backup"].get(k, 0) + 1
        for s in out["segments"]:
            for k in (s["nominal"].get("viol_all") or {}):
                viol["nominal"][k] = viol["nominal"].get(k, 0) + 1
        rec_out.update(ok=True, phases=_jsonable({"|".join(map(str, k)): v for k, v in agg.items()}),
                       segs=[s["seg"] for s in out["segments"]], n_traces=len(tracer.done), U=viol,
                       fam={s["seg"]: dict(n=s["family"][0.0]["n"], n_bad=s["family"][0.0]["n_bad"]) for s in out["segments"]},
                       nominal_viol=nominal_u, viol_cleared_to_continue=True)
    except Exception as e:
        import traceback
        rec_out.update(ok=False, error=repr(e), tb=traceback.format_exc()[-1500:])
    rec_out["wall_s"] = round(time.time() - t0, 1)
    fn = pathlib.Path(job["out"]) / "eps" / f"{task}_{cell[0]}-{cell[1]}_a{a}_{job['i']:04d}.json"
    tmp = fn.with_suffix(".tmp%d" % os.getpid())
    tmp.write_text(json.dumps(rec_out))
    os.replace(tmp, fn)
    return rec_out


# ------------------------------------------------------------------------------------------------------------------------------ table
def _pct(xs):
    import numpy as np
    if not xs:
        return None
    x = np.asarray(xs, float)
    return dict(n=len(xs), p50=round(float(np.percentile(x, 50)), 4), p95=round(float(np.percentile(x, 95)), 4), max=round(float(x.max()), 4))


def build_table(out, args=None):
    out = pathlib.Path(out)
    eps = [json.load(open(f)) for f in sorted((out / "eps").glob("*.json"))]
    good = [e for e in eps if e.get("ok")]
    groups = {}
    for e in good:
        groups.setdefault((e["task"], tuple(e["cell"]), e["a"]), []).append(e)
    rows = []
    for (task, cell, a), es in sorted(groups.items()):
        for kind in ("nominal", "backup"):
            # per episode: max over segments / branches / phases (kind level), the phase of the max, plus per-phase maxima
            kvals = {w: {v: [] for v in ("kp_axis", "kp_norm", "ka_axis", "ka_norm", "abs_norm")} for w in WIN}
            phase_of_max = {w: [] for w in WIN}
            per_phase = {}
            seg_of_max = {w: [] for w in WIN}
            for e in es:
                best = {w: (-1.0, None, None) for w in WIN}
                for key, d in e["phases"].items():
                    seg, kd, ph, w = key.split("|")
                    w = int(w)
                    if kd != kind:
                        continue
                    pp = per_phase.setdefault((ph, w), {v: [] for v in ("kp_axis", "kp_norm", "ka_axis", "ka_norm", "abs_norm")})
                    pp["kp_axis"].append(max(d["kp_axis"])); pp["ka_axis"].append(max(d["ka_axis"]))
                    pp["kp_norm"].append(d["kp_norm"]); pp["ka_norm"].append(d["ka_norm"]); pp["abs_norm"].append(d["abs_norm"])
                    if seg == "release" or ph == "post_release":
                        continue
                    if d["kp_norm"] > best[w][0]:
                        best[w] = (d["kp_norm"], ph, seg)
                for w in WIN:
                    mx = {v: 0.0 for v in kvals[w]}
                    for key, d in e["phases"].items():
                        seg, kd, ph, ww = key.split("|")
                        if kd != kind or int(ww) != w or seg == "release" or ph == "post_release":
                            continue
                        mx["kp_axis"] = max(mx["kp_axis"], max(d["kp_axis"])); mx["ka_axis"] = max(mx["ka_axis"], max(d["ka_axis"]))
                        mx["kp_norm"] = max(mx["kp_norm"], d["kp_norm"]); mx["ka_norm"] = max(mx["ka_norm"], d["ka_norm"]); mx["abs_norm"] = max(mx["abs_norm"], d["abs_norm"])
                    for v in mx:
                        kvals[w][v].append(mx[v])
                    phase_of_max[w].append(best[w][1]); seg_of_max[w].append(best[w][2])
            nU = sum(1 for e in es if e["U"][kind])
            row = dict(task=task, cell=list(cell), a=a, kind=kind, n=len(es), n_with_U_monitor=nU,
                       U_kinds={k: sum(1 for e in es if k in e["U"][kind]) for k in sorted({k for e in es for k in e["U"][kind]})},
                       k_all_phases={f"{WMS[w]}ms": {v: _pct(kvals[w][v]) for v in kvals[w]} for w in WIN},
                       phase_of_max={f"{WMS[w]}ms": dict(argmax_episode=(phase_of_max[w][int(max(range(len(es)), key=lambda i: kvals[w]["kp_norm"][i]))] if es else None),
                                                         seg_of_argmax=(seg_of_max[w][int(max(range(len(es)), key=lambda i: kvals[w]["kp_norm"][i]))] if es else None),
                                                         hist={p: phase_of_max[w].count(p) for p in sorted(set(phase_of_max[w]), key=str)}) for w in WIN},
                       per_phase={f"{ph}|{WMS[w]}ms": {v: _pct(x) for v, x in pp.items()} for (ph, w), pp in sorted(per_phase.items())})
            rows.append(row)
    table = dict(tag="MEASURED", statistic="max over n (not a bound)", noise_model="sim, deterministic per world",
                 leaves_out=["real-robot tracking", "sensing", "worlds outside the cells"],
                 definitions=dict(k_plan="|achieved| / a_ref, a_ref = max |Piece.pdd| of the active piece (its TOPP-RA cap); k_a = |achieved| / a (Rule 1 box); "
                                         "*_axis = max over the 3 world axes of the per-axis ratio; *_norm = norm of the acceleration vector / denominator; abs_norm in m/s^2; "
                                         "each episode contributes its max over segments, branch times and phases (kind level: excludes the release segment and post_release phases); "
                                         "p50 / p95 / max are over episodes",
                                  windows="5 / 20 / 50 ms = 1 / 4 / 10 control steps of 5 ms; first difference of the object linear velocity", backups="the planner's stored backup (certs=None)"),
                 n_episodes=len(eps), n_failed=len(eps) - len(good), failed=[dict(task=e["task"], cell=e["cell"], i=e["i"], seed=e["seed"], error=e["error"]) for e in eps if not e.get("ok")][:20],
                 seeds={f"{e['task']}_{e['cell'][0]}-{e['cell'][1]}_a{e['a']}": [] for e in eps}, wall_s=_pct([e["wall_s"] for e in eps]), rows=rows)
    for e in eps:
        table["seeds"][f"{e['task']}_{e['cell'][0]}-{e['cell'][1]}_a{e['a']}"].append(e["seed"])
    tmp = out / "TABLE.json.tmp"
    tmp.write_text(json.dumps(table, indent=1))
    os.replace(tmp, out / "TABLE.json")
    tmp = out / "episodes.jsonl.tmp"
    with open(tmp, "w") as f:
        for e in eps:
            f.write(json.dumps(e) + "\n")
    os.replace(tmp, out / "episodes.jsonl")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="box")
    ap.add_argument("--cells", default="2.0-2.5")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--a", type=float, default=0.5)
    ap.add_argument("--table-only", action="store_true")
    args = ap.parse_args()
    out = pathlib.Path(args.out).resolve()
    (out / "eps").mkdir(parents=True, exist_ok=True)
    _setup_env(out)
    sys.path.insert(0, str(HERE.parents[4]))
    if not args.table_only:
        cells = [tuple(float(x) for x in c.split("-")) for c in args.cells.split(",")]
        done = {f.name for f in (out / "eps").glob("*.json")}
        jobs = []
        for cell in cells:
            for i in range(args.n):
                name = f"{args.task}_{cell[0]}-{cell[1]}_a{args.a}_{i:04d}.json"
                if name not in done:                                    # restartable: skip episodes already written
                    jobs.append(dict(task=args.task, cell=cell, i=i, seed=cell_seed(cell, i), a=args.a, out=str(out)))
        print(f"{len(jobs)} episodes to run ({len(done)} present), procs={args.procs}", flush=True)
        import multiprocessing as mp
        t0 = time.time()
        if jobs:
            with mp.Pool(args.procs, maxtasksperchild=3) as pool:
                for r in pool.imap_unordered(run_episode, jobs):
                    print(f"  {r['task']} {r['cell']} i={r['i']} seed={r['seed']} ok={r['ok']} wall={r['wall_s']}s" + ("" if r["ok"] else f" ERR {r['error']}"), flush=True)
        print(f"sweep wall {time.time() - t0:.0f}s", flush=True)
    tab = build_table(out)
    print(f"TABLE.json: {tab['n_episodes']} episodes, {tab['n_failed']} failed", flush=True)


if __name__ == "__main__":
    main()
