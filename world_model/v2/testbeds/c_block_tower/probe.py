"""probe.py -- feasibility probe for the block-tower-on-a-plate testbed (SIMULATION ONLY, CPU, no GPU/render).
Stages (see README.md):
  grid     nominal-carry failure rate: replay spill_v2 plate trajectories (optionally tilt-scaled) over a config grid
  savable  savable-set / hold-still / lead-time on pre-failure states (search-certified, same library as bead_rep.oracle)
  gap      reduced analytic model vs MuJoCo on identical branches (85-set + hold + brake)
  ramp     friction identification probe: slow tilt ramp -> first-slip tilt vs atan(mu); mass and noslip checks
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import argparse, glob, json, multiprocessing as mp, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from world_model.v2.testbeds.c_block_tower import tower_env as T  # noqa: E402
from world_model.v2.latent_hj import spill_data as sd  # noqa: E402

OUT = Path("/common/users/ss5772/bimanual_data/testbeds/c_block_tower")
HORIZON = 100          # ticks = 2 s
CHECK_EVERY = 2        # branch failure check cadence (ticks)
FATALS = [0]           # per-process count of MuJoCo FatalErrors swallowed inside branches


# ---------------------------------------------------------------------------------------- data
def test_shards(n):
    ps = sorted(glob.glob(str(sd.DATA / "shard_*.npz")))
    te = [p for i, p in enumerate(ps) if i % 5 == 4]       # same held-out split rule as spill_data.shard_split
    return te[:n]


def scale_tilt(quat, axis, s):
    """tilt-scaled copy of a plate trajectory: polar angle of the axis x s (azimuth kept), yaw of q0 kept."""
    if s == 1.0:
        return quat
    th = np.arccos(np.clip(axis[:, 2], -1, 1)) * s
    phi = np.arctan2(axis[:, 1], axis[:, 0])
    cs = np.stack([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)], 1)
    q = np.empty_like(quat)
    q0 = quat[0]
    z = np.array([0, 0, 1.0])
    for k in range(len(cs)):
        c = cs[k]
        d = float(z @ c)
        m = np.array([1 + d, *np.cross(z, c)]) if d > -1 + 1e-9 else np.array([0, 1.0, 0, 0])
        m /= np.linalg.norm(m)
        w1, x1, y1, z1 = m; w2, x2, y2, z2 = q0
        q[k] = [w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def prep_episode(ep, s):
    quat = scale_tilt(ep["quat"], ep["axis"], s) if s != 1.0 else ep["quat"]
    pos = ep["pos"]
    vb = np.zeros_like(pos)
    vb[1:] = (pos[1:] - pos[:-1]) / T.DT
    return pos, quat, vb


def replay(env, pos, quat, vb, seed, snap_ticks=None):
    """pin the logged 200 Hz plate pose, blocks free. -> l per tick, first failure tick (-1 none), snapshots."""
    env.rng = np.random.default_rng(seed)
    env.reset(pos[0], quat[0])
    Tn = (len(pos) // 4) * 4
    nt = Tn // 4
    l = np.zeros(nt, np.float32)
    snaps = {}
    want = set(snap_ticks) if snap_ticks is not None else set()
    for k in range(Tn):
        env.set_kin(pos[k], quat[k], vb[k])
        j = k // 4
        if k % 4 == 0 and j in want:
            snaps[j] = env.snapshot()
        env.phys_step()
        if k % 4 == 0:
            l[j], _ = env.margin()
    neg = np.where(l < 0)[0]
    return l, (int(neg[0]) if len(neg) else -1), snaps


# ---------------------------------------------------------------------------------------- stage grid
def _grid_task(a):
    path, ei, n, mu, s = a
    ep = sd.load_dense(path)[ei]
    env = T.TowerEnv(T.TowerCfg(n_blocks=n, mu=mu))
    pos, quat, vb = prep_episode(ep, s)
    l, ft, _ = replay(env, pos, quat, vb, ep["seed"])
    return (n, mu, s, int(ft >= 0), float(l.min()))


def stage_grid(nshard, nproc, Ns=(2, 3, 4), mus=(0.3, 0.5, 0.8), ss=(1.0, 0.5, 0.3)):
    paths = test_shards(nshard)
    tasks = []
    for p in paths:
        ne = len(np.load(p)["ep_seed"])
        for ei in range(ne):
            for n in Ns:
                for mu in mus:
                    for s in ss:
                        tasks.append((p, ei, n, mu, s))
    t0 = time.time()
    with mp.Pool(nproc, maxtasksperchild=50) as P:
        res = P.map(_grid_task, tasks, chunksize=8)
    out = {}
    for n, mu, s, f, lm in res:
        out.setdefault(f"N{n}_mu{mu}_s{s}", []).append(f)
    tab = {k: dict(fail_rate=float(np.mean(v)), n=len(v)) for k, v in out.items()}
    wall = time.time() - t0
    print(f"grid: {len(tasks)} replays, wall {wall:.0f}s", flush=True)
    for k, v in tab.items():
        print(k, v, flush=True)
    return dict(table=tab, n_tasks=len(tasks), wall_s=wall, cpu_s=wall * nproc)


# ---------------------------------------------------------------------------------------- branches / oracle
def run_branch(env, snap, seq=None, tail_brake=False, hold=False, T_=HORIZON):
    """restore snap, apply a (len<=T,6) tick action sequence (then brake / zero), return (ok, fail_tick, min_margin)."""
    env.restore(snap)
    mn = 9.0
    for t in range(T_):
        if seq is not None and t < len(seq):
            u = seq[t]
        elif tail_brake:
            u = T.brake(env)
        else:
            u = np.zeros(6)
        chk = (t % CHECK_EVERY == CHECK_EVERY - 1) or t == T_ - 1
        try:
            l, f = env.tick(u, check=chk)
        except Exception:                  # mujoco FatalError (box-box contact overflow): count as a failed branch
            FATALS[0] += 1
            return False, env.tick_n, -1.0
        if chk:
            mn = min(mn, l)
        if f:
            return False, env.fail_tick, mn
    return True, -1, mn


def certify(env, snap, U, rng, n_rand=100):
    """search-certified savable (an under-approximation, like bead_rep.oracle): library = hold, brake,
    84 other held constants, n_rand random 2-segment (3-15 ticks each, uniform in U) + brake tail."""
    n_eval = 1
    ok_h, ft_h, m_h = run_branch(env, snap)
    out = dict(hold_ok=int(ok_h), hold_fail_tick=ft_h, cert=int(ok_h), kind=0, n_eval=1, best_margin=m_h)
    if ok_h:
        out["kind"] = 1
        return out
    n_eval += 1
    ok, _, m = run_branch(env, snap, tail_brake=True)
    if ok:
        return {**out, "cert": 1, "kind": 2, "n_eval": n_eval, "best_margin": m}
    best = m
    zi = int(np.argmin(np.abs(U).sum(1)))
    for i in range(len(U)):
        if i == zi:
            continue
        n_eval += 1
        ok, _, m = run_branch(env, snap, seq=np.repeat(U[i][None], HORIZON, 0))
        best = max(best, m)
        if ok:
            return {**out, "cert": 1, "kind": 3, "n_eval": n_eval, "best_margin": m}
    for _ in range(n_rand):
        n_eval += 1
        k1, k2 = rng.integers(3, 16, 2)
        i1, i2 = rng.integers(0, len(U), 2)
        seq = np.concatenate([np.repeat(U[i1][None], k1, 0), np.repeat(U[i2][None], k2, 0)])
        ok, _, m = run_branch(env, snap, seq=seq, tail_brake=True)
        best = max(best, m)
        if ok:
            return {**out, "cert": 1, "kind": 4, "n_eval": n_eval, "best_margin": m}
    return {**out, "n_eval": n_eval, "best_margin": best}


def _savable_task(a):
    path, ei, n, mu, s, back, n_rand, extra = a
    ep = sd.load_dense(path)[ei]
    env = T.TowerEnv(T.TowerCfg(n_blocks=n, mu=mu))
    pos, quat, vb = prep_episode(ep, s)
    l, ft, _ = replay(env, pos, quat, vb, ep["seed"])
    U = T.fallback_U()
    rng = np.random.default_rng(ep["seed"] + 7)
    rows = []
    if ft >= 0:
        ticks = [ft - b for b in back if ft - b >= 20]
        if not ticks:
            return dict(seed=ep["seed"], fail=1, ft=ft, rows=[])
        _, _, snaps = replay(env, pos, quat, vb, ep["seed"], snap_ticks=ticks)
        for tk in sorted(snaps):
            t0 = time.perf_counter()
            r = certify(env, snaps[tk], U, rng, n_rand)
            r.update(tick=tk, back=ft - tk, kind_ep="pre", l0=float(l[tk]), wall=time.perf_counter() - t0)
            rows.append(r)
    elif extra:                                  # non-failing episode: a few uniform reference states
        tk = [int(x) for x in rng.integers(20, len(l) - 1, extra)]
        _, _, snaps = replay(env, pos, quat, vb, ep["seed"], snap_ticks=tk)
        for t_ in sorted(snaps):
            r = certify(env, snaps[t_], U, rng, n_rand)
            r.update(tick=t_, back=-1, kind_ep="ok_ep", l0=float(l[t_]), wall=0.0)
            rows.append(r)
    return dict(seed=ep["seed"], fail=int(ft >= 0), ft=ft, rows=rows)


def stage_savable(nshard, nproc, n, mu, s, max_fail_eps, back, n_rand, extra=0):
    paths = test_shards(nshard)
    tasks = []
    for p in paths:
        ne = len(np.load(p)["ep_seed"])
        tasks += [(p, ei, n, mu, s, back, n_rand, extra) for ei in range(ne)]
    t0 = time.time()
    with mp.Pool(nproc, maxtasksperchild=3) as P:
        res = []
        for r in P.imap_unordered(_savable_task, tasks, chunksize=1):
            res.append(r)
            if sum(x["fail"] for x in res) >= max_fail_eps and extra == 0:
                P.terminate()
                break
    wall = time.time() - t0
    fe = [r for r in res if r["fail"]]
    rows = [x for r in fe for x in r["rows"]]
    ok_rows = [x for r in res if not r["fail"] for x in r["rows"]]
    summ = summarize(rows)
    summ_ok = summarize(ok_rows) if ok_rows else None
    # lead time: per failing episode, smallest back such that no state at <= back (closer to failure) is certified
    leads = []
    for r in fe:
        rr = sorted(r["rows"], key=lambda x: -x["back"])      # earliest first
        cert_back = [x["back"] for x in rr if x["cert"] and not x["hold_ok"]]
        # "first unsavable" = earliest sampled state from which every later sampled state is NOT certified-savable
        # and which is itself not certified; lead = back of that state.
        first = None
        for i, x in enumerate(rr):
            if not x["cert"] and all(not y["cert"] for y in rr[i:]):
                first = x["back"]; break
        if first is not None and len(rr) > 1:
            leads.append(first)
    print(f"savable: {len(res)} eps done, {len(fe)} failing, {len(rows)} states, wall {wall:.0f}s", flush=True)
    return dict(cfg=dict(n=n, mu=mu, s=s, back=back, n_rand=n_rand), n_eps=len(res), n_fail_eps=len(fe),
                summary=summ, uniform_ok_ep_states=summ_ok, lead_ticks=leads,
                lead_median=float(np.median(leads)) if leads else None,
                wall_s=wall, cpu_s=wall * nproc, rows=rows)


def summarize(rows):
    if not rows:
        return {}
    a = lambda k: np.array([x[k] for x in rows])
    hold = a("hold_ok").astype(bool)
    cert = a("cert").astype(bool)
    return dict(n=len(rows), hold_ok_frac=float(hold.mean()), savable_frac=float((cert & ~hold).mean()),
                uncertified_frac=float((~cert).mean()), n_savable=int((cert & ~hold).sum()),
                mean_n_eval=float(a("n_eval").mean()), kinds={int(k): int((a("kind") == k).sum()) for k in range(5)})


# ---------------------------------------------------------------------------------------- stage gap
KS = (1, 2, 3, 5, 8)


def _branch_set(U):
    """(name, seq, tail) for hold, brake, and the 84 other constants; each T=100 ticks."""
    zi = int(np.argmin(np.abs(U).sum(1)))
    out = [("hold", None, "zero"), ("brake", None, "brake")]
    out += [(f"u{i}", np.repeat(U[i][None], HORIZON, 0), "zero") for i in range(len(U)) if i != zi]
    return out


def _gap_task(a):
    path, ei, n, mu, s, back = a
    from world_model.v2.testbeds.c_block_tower import analytic as AN
    ep = sd.load_dense(path)[ei]
    env = T.TowerEnv(T.TowerCfg(n_blocks=n, mu=mu))
    pos, quat, vb = prep_episode(ep, s)
    l, ft, _ = replay(env, pos, quat, vb, ep["seed"])
    if ft < 0:
        return []
    ticks = [ft - b for b in back if ft - b >= 20]
    _, _, snaps = replay(env, pos, quat, vb, ep["seed"], snap_ticks=ticks)
    U = T.fallback_U()
    BS = _branch_set(U)
    rows = []
    for tk in sorted(snaps):
        env.restore(snaps[tk])
        P, _ = env.block_pose_plate_frame()
        k0 = snaps[tk].kin
        mj_fail, an = [], {K: [] for K in KS}
        margin_an = []
        for name, seq, tail in BS:
            ok, f_t, mn = run_branch(env, snaps[tk], seq=seq, tail_brake=(tail == "brake"))
            mj_fail.append(f_t if not ok else -1)
            sl, tp = AN.branch_margins(k0[:3], k0[7:10], k0[3:7], P, env.cfg, mu, seq=seq, tail=tail)
            for K in KS:
                an[K].append(AN.predict_fail(sl, tp, K))
            margin_an.append(float(np.minimum(sl, tp).min()))
        rows.append(dict(seed=ep["seed"], tick=tk, back=ft - tk, mj=mj_fail, an={K: an[K] for K in KS}, margin_an=margin_an))
    return rows


def stage_gap(nshard, nproc, n, mu, s, back, max_eps):
    paths = test_shards(nshard)
    tasks = []
    for p in paths:
        ne = len(np.load(p)["ep_seed"])
        tasks += [(p, ei, n, mu, s, back) for ei in range(ne)]
    t0 = time.time()
    rows = []
    neps = 0
    with mp.Pool(nproc, maxtasksperchild=3) as P:
        for r in P.imap_unordered(_gap_task, tasks, chunksize=1):
            if r:
                rows += r
                neps += 1
            if neps >= max_eps:
                P.terminate()
                break
    wall = time.time() - t0
    mj = np.array([r["mj"] for r in rows])                       # (S, B)
    mjf = mj >= 0
    res = {}
    for K in KS:
        an = np.array([r["an"][K] for r in rows])
        anf = an >= 0
        tp, fp, fn_, tn = int((anf & mjf).sum()), int((anf & ~mjf).sum()), int((~anf & mjf).sum()), int((~anf & ~mjf).sum())
        both = anf & mjf
        terr = np.abs(an[both] - mj[both]) if both.any() else np.array([np.nan])
        # state level: analytic-certified = some branch predicted OK; compare with MuJoCo-certified
        a_cert, m_cert = (~anf).any(1), (~mjf).any(1)
        # plan-aware physics baseline: take the analytic-best branch (max margin); success = MuJoCo says it is OK
        marg = np.array([r["margin_an"] for r in rows])
        pick = np.argmax(marg, 1)
        succ = ~mjf[np.arange(len(rows)), pick]
        # same but only K-consistent: pick any analytic-OK branch (first by margin among OK)
        res[f"K{K}"] = dict(tp=tp, fp=fp, fn=fn_, tn=tn, branch_acc=(tp + tn) / mj.size,
                            precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn_, 1),
                            fail_tick_abs_err_median=float(np.nanmedian(terr)),
                            state_cert_agree=float((a_cert == m_cert).mean()), a_cert=float(a_cert.mean()),
                            m_cert=float(m_cert.mean()),
                            savable_states_mj=int((m_cert & ~mjf[:, 0]).sum()),
                            analytic_pick_success_on_mj_savable=float(succ[m_cert & mjf[:, 0]].mean()) if (m_cert & mjf[:, 0]).any() else None,
                            analytic_pick_success_all=float(succ.mean()))
    print(f"gap: {len(rows)} states, {mj.size} branches, wall {wall:.0f}s", flush=True)
    return dict(cfg=dict(n=n, mu=mu, s=s, back=back), n_states=len(rows), n_branches=int(mj.size), mj_branch_fail_frac=float(mjf.mean()),
                results=res, wall_s=wall, cpu_s=wall * nproc, n_eps=neps)


# ---------------------------------------------------------------------------------------- stage sens
SENS_CFGS = {"mu0.3": dict(mu=0.3), "mu0.5": dict(mu=0.5), "mu0.8": dict(mu=0.8), "solref0.02": dict(solref=(0.02, 1.0)),
             "noslip10": dict(noslip=10), "mass3x": dict(block_mass=0.18), "offset0": dict(offset_std=0.0), "offset6mm": dict(offset_std=0.006)}


def _sens_task(a):
    path, ei, s = a
    ep = sd.load_dense(path)[ei]
    pos, quat, vb = prep_episode(ep, s)
    out = {}
    for k, kw in SENS_CFGS.items():
        env = T.TowerEnv(T.TowerCfg(n_blocks=3, **kw))
        try:
            l, ft, _ = replay(env, pos, quat, vb, ep["seed"])
        except Exception:
            ft = -2
        out[k] = ft
    return out


def stage_sens(nshard, nproc, s):
    tasks = []
    for p in test_shards(nshard):
        tasks += [(p, ei, s) for ei in range(len(np.load(p)["ep_seed"]))]
    with mp.Pool(nproc, maxtasksperchild=5) as P:
        res = P.map(_sens_task, tasks, chunksize=2)
    F = {k: np.array([r[k] for r in res]) for k in SENS_CFGS}
    out = {k: dict(fail_rate=float((v >= 0).mean()), fatal=int((v == -2).sum())) for k, v in F.items()}
    base = F["mu0.5"] >= 0
    for k, v in F.items():
        out[k]["outcome_flip_vs_mu0.5"] = float(((v >= 0) != base).mean())
    out["mu_dependent_eps(0.3 fail xor 0.8 fail)"] = float(((F["mu0.3"] >= 0) != (F["mu0.8"] >= 0)).mean())
    both = (F["mu0.3"] >= 0) & (F["mu0.8"] >= 0)
    out["median_fail_tick_shift_mu0.3_vs_0.8"] = float(np.median(F["mu0.3"][both] - F["mu0.8"][both])) if both.any() else None
    out["n_eps"] = len(res)
    for k, v in out.items():
        print(k, v, flush=True)
    return out


# ---------------------------------------------------------------------------------------- stage ramp
def ramp_onset(cfg, w=0.1, az=0.0, tick_max=400, thr=0.003):
    """tilt the (static) plate about the y axis at w rad/s; return tilt at which any block has moved > thr [m]
    relative to the plate frame from its start, and tilt at which failure fires."""
    env = T.TowerEnv(T.TowerCfg(**cfg), seed=0)
    env.reset(env.home, np.array([1.0, 0, 0, 0]), offsets=np.zeros((cfg.get("n_blocks", 3), 2)))
    for _ in range(40):
        env.tick(np.zeros(6))
    P0, _ = env.block_pose_plate_frame()
    t_slip = t_fail = None
    for t in range(tick_max):
        env.tick(np.array([0, 0, 0, 0, w, 0.0]))
        P, _ = env.block_pose_plate_frame()
        th = float(np.arccos(env.axis()[2]))
        if t_slip is None and np.abs(P[:, :2] - P0[:, :2]).max() > thr:
            t_slip = th
        if env.failed and t_fail is None:
            t_fail = th
            break
    return t_slip, t_fail


def stage_ramp():
    out = {}
    for w in (0.1, 0.03):
        for mu in (0.3, 0.5, 0.8):
            for n in (1, 3):
                for ms, ns, mass in (("def", 0, 0.06), ("noslip", 10, 0.06), ("mass3x", 0, 0.18)):
                    if w == 0.03 and ms != "def":
                        continue
                    ts, tf = ramp_onset(dict(n_blocks=n, mu=mu, noslip=ns, block_mass=mass), w=w, tick_max=int(0.9 / w / 0.02))
                    th_slide = float(np.arctan(mu))
                    th_tip = float(np.arctan(0.025 / (n * 0.02)))
                    out[f"w{w}_mu{mu}_N{n}_{ms}"] = dict(slip_tilt=ts, fail_tilt=tf, analytic_slide=th_slide,
                                                         analytic_tip=th_tip, analytic_min=min(th_slide, th_tip))
                    print(f"w {w} mu {mu} N {n} {ms}: slip {ts} fail {tf} | analytic slide {th_slide:.3f} tip {th_tip:.3f}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["grid", "savable", "gap", "sens", "ramp", "selftest"])
    ap.add_argument("--nshard", type=int, default=8)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--s", type=float, default=0.5)
    ap.add_argument("--max-fail-eps", type=int, default=60)
    ap.add_argument("--n-rand", type=int, default=100)
    ap.add_argument("--extra", type=int, default=0)
    ap.add_argument("--back", type=int, nargs="+", default=[5, 10, 20, 30, 40, 50])
    ap.add_argument("--tag", default="")
    ap.add_argument("--Ns", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--mus", type=float, nargs="+", default=[0.3, 0.5, 0.8])
    ap.add_argument("--ss", type=float, nargs="+", default=[1.0, 0.5, 0.3])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "selftest":
        print("kin_step numba vs bead_env max err", T.selftest(), flush=True)
    elif a.stage == "grid":
        r = stage_grid(a.nshard, a.nproc, a.Ns, a.mus, a.ss)
        (OUT / "tmp").mkdir(exist_ok=True)
        json.dump(r, open(OUT / "tmp" / f"grid{a.tag}.json", "w"), indent=1)
    elif a.stage == "savable":
        r = stage_savable(a.nshard, a.nproc, a.n, a.mu, a.s, a.max_fail_eps, a.back, a.n_rand, a.extra)
        json.dump(r, open(OUT / "tmp" / f"savable{a.tag}.json", "w"), indent=1)
        print(json.dumps({k: v for k, v in r.items() if k != "rows"}, indent=1), flush=True)
    elif a.stage == "gap":
        r = stage_gap(a.nshard, a.nproc, a.n, a.mu, a.s, a.back, a.max_fail_eps)
        json.dump(r, open(OUT / "tmp" / f"gap{a.tag}.json", "w"), indent=1)
        print(json.dumps(r, indent=1), flush=True)
    elif a.stage == "sens":
        r = stage_sens(a.nshard, a.nproc, a.s)
        json.dump(r, open(OUT / "tmp" / f"sens{a.tag}.json", "w"), indent=1)
    elif a.stage == "ramp":
        r = stage_ramp()
        json.dump(r, open(OUT / "tmp" / f"ramp{a.tag}.json", "w"), indent=1)
