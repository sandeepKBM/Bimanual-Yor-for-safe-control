"""nero/contents/probe_al.py -- active-learning choice of the lift-probe rungs (11-03).  SIMULATION ONLY.

Same physics, same Probe class, same SOUND update as probe.Probe.run.  Only the CHOICE of the next rung force and the stop rule change.

Two beliefs:
  Theta = [M_lo, M_hi] on total carried mass: the sound set (probe.Probe rules).  Used for admission (safety) and the decision.
  particle belief over M (grid of N_PART points on the prior box + weights): used ONLY to pick the probe.  Reweighted by the measurement
  likelihood fitted on calib.jsonl (MEASURED) and clipped to Theta.  The selector simulates reads from it (no MuJoCo inside).

Objectives:  decision (maximise P(Theta decides) per rung used, expected over the particle belief, greedy roll-out tail),
             estimate (minimise the expected sound width per rung),  fixed (Probe.run's schedule: M_lo, then +DM).
Decision of a finished probe:  carry iff Theta is free-confirmed and M_hi <= empty + c_admit_max(task); else refuse.

    python -m world_model.v2.nero.contents.probe_al run --records R --answers A --calib CALIB.json --calib-rows calib.jsonl \
        --ceilings CEILINGS.json --prior flat|vlm|oracle --objective decision|estimate|fixed --worlds all|N --procs P --out OUT.jsonl
    python -m world_model.v2.nero.contents.probe_al score --runs OUT1 OUT2 --baseline traces.jsonl --ceilings C --md out.md
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from world_model.v2.stack import tasks as TK

from . import envelope as E
from . import probe as PR

N_PART = 2000        # ASSUMED: particle grid size on the prior box
K_SAMP = 128         # ASSUMED: truth/read samples per candidate evaluation (common random numbers across candidates)
STEP = 0.125         # ASSUMED: candidate force grid [kg]; also the minimum increase over the last rung
RHO = 0.2            # ASSUMED: weight of the flat component mixed into a VLM / oracle prior
VLM_Z = 1.645        # VLM interval read as a 90% central interval of a normal (ASSUMED)
SD_MIN_PRIOR = 0.05  # ASSUMED: floor of the VLM prior sd [kg]
ORACLE_SD = 0.05     # ASSUMED: sd of the oracle prior [kg] (reference arm only)
W_TARGET = 0.2       # ASSUMED: estimate objective stops at sound width <= this [kg]
MIN_GAIN = 0.02      # ASSUMED: estimate objective stops when no candidate cuts the expected width by this [kg]
H_ROLL = 14          # ASSUMED: roll-out horizon (rungs) of the decision objective
MAX_RUNGS = 40       # ASSUMED: hard cap on rungs per episode
SD_FLOOR = 0.03      # ASSUMED: floor of the likelihood sd [kg] (the calibration reads are noise-free and very tight)
ZONE = 0.15          # ASSUMED: half-width of the on-counter/lifted transition zone in f(m) - (M + b) [kg]
ASSUMPTIONS = dict(N_PART=N_PART, K_SAMP=K_SAMP, STEP=STEP, RHO=RHO, VLM_Z=VLM_Z, SD_MIN_PRIOR=SD_MIN_PRIOR, ORACLE_SD=ORACLE_SD,
                   W_TARGET=W_TARGET, MIN_GAIN=MIN_GAIN, H_ROLL=H_ROLL, MAX_RUNGS=MAX_RUNGS, SD_FLOOR=SD_FLOOR, ZONE=ZONE,
                   O_MAX=PR.O_MAX, DM=PR.DM, FREE_RATIO=PR.FREE_RATIO, TAU_NOISE=PR.TAU_NOISE)


def family(task):
    return "box" if task == "box" else "tray"


# ------------------------------------------------------------------------------------------------- likelihood (MEASURED on calib)
def fit_likelihood(rows):
    """L | (m_ff, M): L = min(f(m_ff), M + b) + N(0, sd_zone^2 + sd_noise^2).
    f = quadratic in m_ff fitted on on-counter rungs with M - m_ff > 0.5 (gap grows with the push: joint friction),
    b, sd_free from the rungs flagged free, sd_on / sd_tr = residual sd of the remaining rungs by zone.  Per family (box / tray)."""
    out = {}
    for fam in ("box", "tray"):
        rs = [r for r in rows if "error" not in r and family(r["task"]) == fam]
        pts = [(x["m_ff"], x["L"], r["M_true"], x["free"], x.get("noise_bound", 0.082)) for r in rs for x in r["rungs"]]
        if not pts:
            continue
        m, L, M, fr, nb = (np.array(c, float) for c in zip(*pts))
        fr = fr.astype(bool)
        on = (~fr) & (M - m > 0.5)
        coef = np.polyfit(m[on], L[on], 2)
        b = float(np.mean(L[fr] - M[fr])) if fr.any() else 0.0
        sd_free = float(np.std(L[fr] - M[fr])) if fr.sum() > 2 else 0.0
        f = np.polyval(coef, m)
        d = f - (M + b)
        det = np.minimum(f, M + b)
        res = L - det
        tr = (np.abs(d) <= ZONE)
        on_z, fr_z = (d < -ZONE), (d > ZONE)
        out[fam] = dict(coef=[float(c) for c in coef], b=b,
                        sd_on=float(np.std(res[on_z])) if on_z.sum() > 2 else 0.0,
                        sd_tr=float(np.std(res[tr])) if tr.sum() > 2 else 0.1,
                        sd_free=max(sd_free, float(np.std(res[fr_z])) if fr_z.sum() > 2 else 0.0),
                        nb=float(np.median(nb)), n_rungs=int(len(m)), n_worlds=len(rs), n_free_rungs=int(fr.sum()),
                        n_fit_on=int(on.sum()), n_zone=[int(on_z.sum()), int(tr.sum()), int(fr_z.sum())])
    return dict(fam=out, label="MEASURED: fitted on calib.jsonl (noise-free reads, calibration worlds only); zone width and sd floor ASSUMED",
                SD_FLOOR=SD_FLOOR, ZONE=ZONE)


class Lik:
    def __init__(self, params):
        self.p = params["fam"]

    def nb(self, fam):
        return self.p[fam]["nb"]

    def f(self, fam, m):
        return np.polyval(self.p[fam]["coef"], m)

    def mean_sd(self, fam, m, M):
        q = self.p[fam]
        f = self.f(fam, m)
        lift = M + q["b"]
        d = f - lift
        sd = np.where(d < -ZONE, q["sd_on"], np.where(d > ZONE, q["sd_free"], q["sd_tr"]))
        # MODEL: the read is the mean of N_READ samples of sum_j Pz_j (tau noise U(+-TAU_NOISE)) / g; with equal |Pz_j| over the 14 joints
        # its sd is nb / sqrt(14 * 3 * N_READ), and nb = (worst case) sum|Pz| TAU_NOISE / g
        sd_noise = q["nb"] / np.sqrt(14.0 * 3.0 * PR.N_READ)
        return np.minimum(f, lift), np.sqrt(np.maximum(sd, SD_FLOOR) ** 2 + sd_noise ** 2)


# ------------------------------------------------------------------------------------------------- sound update (= Probe.run)
def sound_update(lo, hi, L, nb, prev_L, last_m, m, eps_lo, eps_free):
    """exactly Probe.run's rule; works on scalars or arrays.  last_m = m_ff of the previous rung (0.0 if none); prev_L None/nan = none.
    Returns lo, hi, flat (=free)."""
    pl = np.nan if prev_L is None else prev_L
    flat = np.asarray(~np.isnan(pl) & ((L - pl) < PR.FREE_RATIO * (m - last_m)))
    e = eps_free + nb
    lo_f, hi_f = np.maximum(lo, L - e), np.minimum(hi, L + e)
    lo_o = np.maximum(lo, L - eps_lo - nb)
    return np.where(flat, lo_f, lo_o), np.where(flat, hi_f, hi), flat


class Ctx:
    def __init__(self, task, lik, ceil_c, eps_lo, eps_free, m_umax):
        self.task, self.fam, self.lik = task, family(task), lik
        self.empty = PR.empty_mass(task)
        self.ceil_total = self.empty + ceil_c                       # carry iff total hi <= this (MODEL: probe.c_admit_max)
        self.eps_lo, self.eps_free, self.m_umax = eps_lo, eps_free, m_umax
        self.nb = lik.nb(self.fam)

    def m_adm(self, lo):
        """largest admissible force on Theta with lower bound lo (= Probe.admissible: torque row and overshoot row)."""
        return np.minimum(np.asarray(lo) + PR.O_MAX, self.m_umax)

    def decided(self, lo, hi, free, last_m=0.0):
        """Theta decides: free-confirmed, or lo above the carry ceiling, or the torque cap lies at or below lo (no admissible rung can
        ever exceed the mass, so free can never be proven)."""
        return free | (lo > self.ceil_total) | (self.m_umax <= lo + 1e-9)

    def dead(self, lo, last_m):
        """no admissible rung at least STEP above the last one exists (undecided dead end: carry not proven)."""
        return self.m_adm(lo) < last_m + STEP - 1e-9


_UMAX = {}


def m_umax(task, tol=1e-3):
    """largest force with rung_torque_u <= U_ADMIT (bisection; u is monotone in m: tests/test_probe.py).  MODEL."""
    if task not in _UMAX:
        a, b = 0.0, 8.0
        while b - a > tol:
            c = 0.5 * (a + b)
            a, b = (c, b) if PR.rung_torque_u(task, c) <= E.U_ADMIT else (a, c)
        _UMAX[task] = a
    return _UMAX[task]


# ------------------------------------------------------------------------------------------------- particle belief
class Belief:
    def __init__(self, lo, hi, prior="flat", iv=None, M_true=None, n=N_PART):
        self.M = np.linspace(lo, hi, n)
        flat = np.ones(n) / n
        if prior == "vlm" and iv is not None:
            a, b = iv
            sd = max((b - a) / (2 * VLM_Z), SD_MIN_PRIOR)
            g = np.exp(-0.5 * ((self.M - 0.5 * (a + b)) / sd) ** 2)
            w = (1 - RHO) * g / g.sum() + RHO * flat if g.sum() > 0 else flat
        elif prior == "oracle":
            g = np.exp(-0.5 * ((self.M - M_true) / ORACLE_SD) ** 2)
            w = (1 - RHO) * g / g.sum() + RHO * flat
        else:
            w = flat
        self.w = w / w.sum()

    def clip(self, lo, hi):
        w = np.where((self.M >= lo) & (self.M <= hi), self.w, 0.0)
        if w.sum() <= 0:                                        # Theta excludes every particle: restart flat inside Theta
            w = ((self.M >= lo) & (self.M <= hi)).astype(float)
            if w.sum() <= 0:
                w = np.ones_like(self.M)
        self.w = w / w.sum()

    def update(self, lik, fam, m, L):
        mu, sd = lik.mean_sd(fam, m, self.M)
        ll = -0.5 * ((L - mu) / sd) ** 2 - np.log(sd)
        w = self.w * np.exp(ll - ll.max())
        self.w = w / w.sum() if w.sum() > 0 else self.w

    def sample(self, rng, k=K_SAMP):
        c = np.cumsum(self.w)
        u = (np.arange(k) + rng.uniform(size=k)) / k              # stratified
        return self.M[np.minimum(np.searchsorted(c, u * c[-1]), len(self.M) - 1)]


# ------------------------------------------------------------------------------------------------- analytic rollout (no MuJoCo)
def rollout(ctx, Ms, Z, lo0, hi0, prev_L, last_m, m1, n_steps, tail):
    """vectorised over truth samples Ms.  Rung 1 = m1; rung 2 = m1 + DM (the pair, if n_steps >= 2); afterwards (tail) the greedy policy
    'largest admissible force'.  Each sample uses its own Theta for admission and stops when Theta decides or no rung can follow.
    Returns lo, hi, free, decided, n_used, peak."""
    K = len(Ms)
    lo, hi = np.full(K, lo0, float), np.full(K, hi0, float)
    pL = np.full(K, np.nan if prev_L is None else prev_L)
    lm = np.full(K, last_m, float)
    free = np.zeros(K, bool)
    act = np.ones(K, bool)
    nused = np.zeros(K)
    peak = np.full(K, last_m, float)
    for j in range(min(n_steps if not tail else H_ROLL, Z.shape[1])):
        madm = ctx.m_adm(lo)
        if j == 0:
            m = np.full(K, m1)
        elif j == 1:
            m = np.minimum(m1 + PR.DM, madm)
        else:
            m = madm
        ok = act & (m >= lm + STEP - 1e-9) & (m >= lo - 1e-9) & (m <= madm + 1e-9)
        act = ok
        if not act.any():
            break
        mu, sd = ctx.lik.mean_sd(ctx.fam, m, Ms)
        L = mu + sd * Z[:, j]
        nlo, nhi, fl = sound_update(lo, hi, L, ctx.nb, pL, lm, m, ctx.eps_lo, ctx.eps_free)
        lo, hi = np.where(act, nlo, lo), np.where(act, nhi, hi)
        free |= act & fl
        nused += act
        peak = np.where(act, m, peak)
        pL, lm = np.where(act, L, pL), np.where(act, m, lm)
        dec = ctx.decided(lo, hi, free, lm)
        act = act & ~dec & ~ctx.dead(lo, lm)
        if not act.any():
            break
    dec = ctx.decided(lo, hi, free, lm)
    return lo, hi, free, dec, nused, peak


def candidates(ctx, lo, last_m, ideas=()):
    mmax = float(ctx.m_adm(lo))
    start = lo if last_m <= 0 else last_m + STEP
    c = list(np.arange(start, mmax + 1e-9, STEP)) + [mmax] + [float(x) for x in ideas]
    c = sorted({round(float(x), 4) for x in c if x >= start - 1e-9 and x <= mmax + 1e-9 and x >= lo - 1e-9})
    return c


def choose(ctx, belief, lo, hi, prev_L, last_m, objective, rng, ideas=()):
    """returns (m or None, info).  None = stop (info['why'])."""
    cand = candidates(ctx, lo, last_m, ideas)
    if not cand:
        return None, dict(why="rung_not_admissible")
    Ms = belief.sample(rng)
    Z = rng.standard_normal((len(Ms), H_ROLL))                    # common random numbers across candidates
    sc = []
    for m1 in cand:
        if objective == "decision":
            _, _, free, dec, nu, _ = rollout(ctx, Ms, Z, lo, hi, prev_L, last_m, m1, 2, tail=True)
            p = float(np.mean(dec))
            sc.append((p / max(float(np.mean(nu)), 1e-9), p, float(np.mean(nu)), m1))
        else:
            ns = 1 if prev_L is not None else 2
            nlo, nhi, free, dec, nu, _ = rollout(ctx, Ms, Z, lo, hi, prev_L, last_m, m1, ns, tail=False)
            gain = (hi - lo) - float(np.mean(nhi - nlo))
            sc.append((gain / max(float(np.mean(nu)), 1e-9), gain, float(np.mean(nu)), m1))
    sc.sort(reverse=True)
    best = sc[0]
    info = dict(n_cand=len(cand), top=[dict(m=s[3], score=round(s[0], 4), gain_or_p=round(s[1], 4), rungs=round(s[2], 2)) for s in sc[:3]])
    if objective == "decision" and best[1] <= 0.0:
        return None, dict(info, why="no_gain")
    if objective == "estimate" and best[1] < MIN_GAIN:
        return None, dict(info, why="no_gain")
    return best[3], info


# ------------------------------------------------------------------------------------------------- the AL probe loop
def run_loop(pr, ctx, objective, belief, rng, ideas=()):
    """drives a probe.Probe-like object (rung, admissible, M_lo, M_hi, m_ff, ep.ctl, ep.run_phases).  Returns (why, free)."""
    pr.ep.ctl.mass = 0.0
    pr.ep.run_phases([("hold", 0.3)])
    prev_L, free, why = None, False, None
    while True:
        if len(pr.rungs) >= MAX_RUNGS:
            why = "max_rungs"
            break
        last_m = pr.rungs[-1]["m_ff"] if pr.rungs else 0.0
        info = {}
        if objective == "fixed":
            m = pr.M_lo if not pr.rungs else pr.m_ff + PR.DM
        else:
            if objective == "decision" and ctx.decided(pr.M_lo, pr.M_hi, False, last_m):
                why = "decided"
                break
            if objective == "estimate" and pr.M_hi - pr.M_lo <= W_TARGET:
                why = "width_target"
                break
            m, info = choose(ctx, belief, pr.M_lo, pr.M_hi, prev_L, last_m, objective, rng, ideas)
            if m is None:
                why = info.get("why", "stop")
                break
        ok, rows = pr.admissible(m)
        if not ok:
            why = "rung_not_admissible"
            break
        L, nb, rise = pr.rung(m)
        dm = m - last_m
        lo, hi, flat = sound_update(pr.M_lo, pr.M_hi, L, nb, prev_L, last_m, m, ctx.eps_lo, ctx.eps_free)
        pr.M_lo, pr.M_hi, free = float(lo), float(hi), bool(flat)
        rec = dict(m_ff=round(m, 4), L=round(L, 4), noise_bound=round(nb, 4), rise_mm=round(rise, 3), **rows, free=free,
                   Theta=[round(pr.M_lo, 4), round(pr.M_hi, 4)])
        if info:
            rec["sel"] = info
        pr.rungs.append(rec)
        prev_L = L
        if belief is not None:
            belief.update(ctx.lik, ctx.fam, m, L)
            belief.clip(pr.M_lo, pr.M_hi)
        if free:
            why = "free"
            break
    return why, free


def decide(ctx, lo, hi, free):
    return "carry" if (free and hi <= ctx.ceil_total + 1e-9) else "refuse"


def _job(a):
    task, seed, rid, prior, objective, lik_params, ceil_c, eps, iv, ideas = a
    from . import carry_cert as CC
    t0 = time.time()
    try:
        w = CC.sample_world(task, TK.prior_box(task), np.random.default_rng(seed))
        ep = PR._episode(w)
        M = PR.true_mass(ep)
        pr = PR.make_probe(ep, task, eps[0], eps[1], rng=np.random.default_rng(seed + 1), noise=True, method="ladder")   # AL chooses push-ladder rung forces: ladder only
        lik = Lik(lik_params)
        ctx = Ctx(task, lik, ceil_c, eps[0], eps[1], m_umax(task))
        bel = Belief(pr.M_lo, pr.M_hi, prior, iv, M) if objective != "fixed" else Belief(pr.M_lo, pr.M_hi, "flat")
        why, free = run_loop(pr, ctx, objective, bel, np.random.default_rng(seed + 2), ideas)
        dec = decide(ctx, pr.M_lo, pr.M_hi, free)
        truth_adm = bool(M - ctx.empty <= ceil_c + 1e-9)
        return dict(task=task, seed=seed, id=rid, prior=prior, objective=objective, M_true=M, empty=ctx.empty, why=why, rungs=pr.rungs,
                    final=[pr.M_lo, pr.M_hi], free=free, decision=dec, truth_admissible=truth_adm,
                    false_safe=bool(dec == "carry" and not truth_adm), missed=bool(dec == "refuse" and truth_adm),
                    truth_in_theta=bool(pr.M_lo - 1e-9 <= M <= pr.M_hi + 1e-9), n_rungs=len(pr.rungs),
                    peak_m_ff=max((r["m_ff"] for r in pr.rungs), default=0.0), viol=ep.track["viol"], track=pr.track,
                    ceil_total=ctx.ceil_total, vlm_iv=iv, wall_s=round(time.time() - t0, 1))
    except Exception as e:
        import traceback
        return dict(task=task, seed=seed, id=rid, prior=prior, objective=objective, error=repr(e), tb=traceback.format_exc()[-800:])


# ------------------------------------------------------------------------------------------------- score
def _pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def row_from_trace(t, ceil_c):
    """baseline arm (fixed ladder with no stop rule) -> the same fields as an AL row."""
    empty = t["empty"]
    ct = empty + ceil_c
    free = bool(t["rungs"] and t["rungs"][-1]["free"])
    lo, hi = t["final"]
    dec = "carry" if (free and hi <= ct + 1e-9) else "refuse"
    adm = bool(t["M_true"] - empty <= ceil_c + 1e-9)
    return dict(task=t["task"], seed=t["seed"], why=t["why"], free=free, final=[lo, hi], decision=dec, truth_admissible=adm,
                false_safe=bool(dec == "carry" and not adm), missed=bool(dec == "refuse" and adm),
                truth_in_theta=bool(lo - 1e-9 <= t["M_true"] <= hi + 1e-9), n_rungs=len(t["rungs"]),
                peak_m_ff=max((r["m_ff"] for r in t["rungs"]), default=0.0), ceil_total=ct)


def table(arms):
    L = ["arm | task | n | decided % (free, or lo > ceiling, or no admissible rung left) | carry | false-safe | missed | rungs p50/max | "
         "peak force p50 | final width p50 | truth in Theta"]
    L.append("---|---|---|---|---|---|---|---|---|---|---")
    for name, rows in arms.items():
        for task in sorted({r["task"] for r in rows}):
            rs = [r for r in rows if r["task"] == task and "error" not in r]
            if not rs:
                continue
            dec = [r["free"] or r["final"][0] > r["ceil_total"] or r.get("why") in ("decided", "rung_not_admissible") or r.get("why") is None
                   for r in rs]
            L.append(f"{name} | {task} | {len(rs)} | {100 * np.mean(dec):.0f} | {sum(r['decision'] == 'carry' for r in rs)} | "
                     f"{sum(r['false_safe'] for r in rs)} | {sum(r['missed'] for r in rs)} | "
                     f"{_pct([r['n_rungs'] for r in rs], 50):.0f}/{max(r['n_rungs'] for r in rs)} | "
                     f"{_pct([r['peak_m_ff'] for r in rs], 50):.2f} | {_pct([r['final'][1] - r['final'][0] for r in rs], 50):.2f} | "
                     f"{sum(r['truth_in_theta'] for r in rs)}/{len(rs)}")
    return "\n".join(L)


def _score(a):
    ceil = json.loads(pathlib.Path(a.ceilings).read_text())["c_admit_max"]
    arms = {}
    for p in a.runs:
        rows = [json.loads(l) for l in open(p)]
        if not rows:
            continue
        r0 = next((r for r in rows if "error" not in r), rows[0])
        arms[f"{r0.get('objective')}/{r0.get('prior')}"] = rows
    if a.baseline:
        tr = [json.loads(l) for l in open(a.baseline)]
        sel = {(r["task"], r["seed"]) for rows in arms.values() for r in rows}
        arms["baseline fixed ladder"] = [row_from_trace(t, ceil[t["task"]]) for t in tr
                                         if "error" not in t and (not sel or (t["task"], t["seed"]) in sel)]
    s = table(arms)
    s += "\n\nerrors: " + ", ".join(f"{k}={sum('error' in r for r in v)}" for k, v in arms.items())
    print(s)
    if a.md:
        pathlib.Path(a.md).write_text(f"probe_al score ({time.strftime('%Y-%m-%d')})\nASSUMED: {json.dumps(ASSUMPTIONS)}\n\n" + s + "\n")


def _run(a):
    import multiprocessing as mp
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cal = PR.load_calib(a.calib, "ladder")["eps"]            # AL = ladder only
    eps = (cal["EPS_LO"], cal["EPS_FREE"])
    ceil = json.loads(pathlib.Path(a.ceilings).read_text())["c_admit_max"]
    lik_params = fit_likelihood([json.loads(l) for l in open(a.calib_rows)])
    lik_params.update(date=time.strftime("%Y-%m-%d"), eps=cal, calib_rows=str(a.calib_rows), assumptions=ASSUMPTIONS)
    pathlib.Path(str(out) + ".lik.json").write_text(json.dumps(lik_params, indent=1))
    recs = [json.loads(l) for l in open(a.records)]
    ans = {}
    if a.answers:
        for l in open(a.answers):
            r = json.loads(l)
            p = r.get("parsed") or {}
            if p.get("total_mass_kg"):
                ans[r["id"]] = (p["total_mass_kg"], p.get("probe_forces") or [])
    tasks = set(a.tasks.split(",")) if a.tasks else None
    recs = [r for r in recs if tasks is None or r["task"] in tasks]
    if a.prior != "vlm":
        recs = [r for r in recs if r["view"] == "torso"]
    recs = sorted(recs, key=lambda r: (r["task"], int(r["seed"]), r["view"]))
    if a.worlds != "all":
        keep = []
        for r in recs:
            if len({(k["task"], k["seed"]) for k in keep} | {(r["task"], r["seed"])}) <= int(a.worlds):
                keep.append(r)
        recs = keep
    jobs = []
    for r in recs:
        iv, ideas = ans.get(r["id"], (None, []))
        jobs.append((r["task"], int(r["seed"]), r["id"], a.prior, a.objective, lik_params, ceil[r["task"]], eps, iv, ideas))
    print(f"{len(jobs)} episodes", flush=True)
    with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=3) as pool, open(out, "a") as f:
        for r in pool.imap_unordered(_job, jobs):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            print(r["task"], r["seed"], r.get("why"), r.get("decision"), r.get("n_rungs"), r.get("error", "")[:100], flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--records", required=True)
    r.add_argument("--answers", default=None)
    r.add_argument("--calib", required=True)
    r.add_argument("--calib-rows", required=True)
    r.add_argument("--ceilings", required=True)
    r.add_argument("--prior", choices=["flat", "vlm", "oracle"], default="flat")
    r.add_argument("--objective", choices=["decision", "estimate", "fixed"], default="decision")
    r.add_argument("--worlds", default="all")
    r.add_argument("--tasks", default=None)
    r.add_argument("--procs", type=int, default=2)
    r.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--baseline", default=None)
    s.add_argument("--ceilings", required=True)
    s.add_argument("--md", default=None)
    a = ap.parse_args(argv)
    _run(a) if a.cmd == "run" else _score(a)


if __name__ == "__main__":
    main()
