"""nero/contents/probe.py -- soft-to-strong lift probe ("probe ladder"), user 11-02/03.  SIMULATION ONLY.

USER 11-02: "given any bound it should start conservatively ... poke softly and see and then learn and keep strengthening the bound".

Setting: object grasped (rigid weld, ASSUMED 11-01) and still resting on the counter; the arm reference stays at the rest pose (no
commanded motion).  Rung k ramps the payload feedforward (LeverCtl lever split, total mass m_ff,k) over T_RAMP, holds T_HOLD, and
reads the vertical load the two claws carry from joint torques (robot-observable: encoders, commanded torque, the arm's own model):

    L = sum_s  e_z . (J_s^T)^+ (tau_cmd,s - bias_s(q)) / g        J_s = 6x7 TCP Jacobian, bias_s = own-arm gravity + Coriolis (MODEL)
    (+ uniform joint-torque noise +-TAU_NOISE per joint and sample, ASSUMED; its worst case is propagated into the bound, MODEL)

What a rung teaches (belief Theta_t = interval of the total carried mass M; the object's empty mass is known):
  still on the counter: the arms carry L and the counter the rest (>= 0)      =>  M >= L - EPS_LO      (lower bound rises)
  hanging free:  L stops growing while m_ff grows (dL < FREE_RATIO dm)         =>  M in [L - EPS_FREE, L + EPS_FREE]
  EPS_LO, EPS_FREE: MEASURED by `calibrate` (max over n worlds of the read residual) + the propagated noise bound (MODEL).

Rung admission on Theta_t (checked BEFORE the rung runs; the soft-first part):
  torque     static u at the rest pose with the full feedforward m_ff,k <= U_ADMIT          proof under MODEL (quasi-static; in steady
             state the arms carry min(M, ~m_ff) so the feedforward bounds the load)
  overshoot  m_ff,k - M_lo(Theta_t) <= O_MAX                                                 ASSUMED; bounds how far and how fast a
             lighter-than-pushed object rises (rise / tilt / spill measured per rung in `calibrate`, sampled)
The first rung pushes no more than the LIGHTEST object in Theta_0 (overshoot <= 0 by construction).
Stop: free -> interval; M_lo > the heaviest carry admissible (verdict decided: refuse, never lifted); next rung not admissible ->
refuse; the static verdict is already the same for all of Theta_0 -> no probe at all.
After the probe: refuse = ramp the push down, let go in place (weld off, retreat); admit = set the belief, ramp to the new nominal
feedforward and run the carry segments (carry.run_segments, stored backups) in the SAME episode.

    python -m world_model.v2.nero.contents.probe calibrate --tasks box,tray6,tray4 --n 30 --procs 4 --out DIR
    python -m world_model.v2.nero.contents.probe e2e --tasks box,tray6 --n 30 --procs 4 --calib DIR/CALIB.json --out DIR2
    python -m world_model.v2.nero.contents.probe onepoke --tasks box,tray6 --n 10 --procs 4 --out DIR3     (ablation)
"""
from __future__ import annotations

import argparse
import functools
import json
import pathlib
import time

import mujoco
import numpy as np

from . import carry as CR
from . import carry_cert as CC
from . import envelope as E
from . import objects as OB
from . import scenes as SC
from . import thresholds as TH

G = OB.G
DM = 0.25            # ASSUMED: push step between rungs [kg of feedforward]
O_MAX = 1.0          # ASSUMED: largest push above the current lower bound [kg]
T_RAMP = 0.3         # design: ramp time per rung [s]
T_HOLD = 0.4         # design: hold before reading [s]
N_READ = 40          # design: readings averaged over the last N_READ control steps of the hold (0.2 s)
FREE_RATIO = 0.2     # ASSUMED: "free" when the read grows by less than FREE_RATIO x the push step (one rung; calibrated)
TAU_NOISE = 0.05     # ASSUMED: joint-torque reading noise, uniform +-TAU_NOISE per joint [Nm] (real NERO value unknown)
PRIOR = {"box": (0.5, 5.0), "tray6": (0.0, 0.25), "tray4": (0.0, 0.25), "tray2": (0.0, 0.25)}   # sample_world cells = full prior


# ------------------------------------------------------------------------------------------------- specs and static verdicts
def n_cups(task):
    return int(task[4:]) if task.startswith("tray") else 0


def empty_mass(task):
    return OB.TRAY["mass"] if task.startswith("tray") else OB.BOX["mass"]


def prior_contents(task):
    """contents mass interval of the full prior (tray: mugs + fill; box: the block)."""
    if task.startswith("tray"):
        n = n_cups(task)
        return n * OB.CUP["empty"], n * (OB.CUP["empty"] + OB.CUP["fill_max_kg"])
    return OB.BOX_CONTENTS_KG


def interval_spec(task, c_lo, c_hi):
    """ObjectSpec for contents mass in [c_lo, c_hi].  Tray: the total fill is known only as a total, so one mug may hold all of
    it: per-mug fill <= min(fill_max, total fill hi) sets the spill limit (MODEL, conservative)."""
    if not task.startswith("tray"):
        return OB.boxed(c_lo, c_hi)
    n = n_cups(task)
    sp = OB.coffee_tray(n)
    sp.contents = OB.Contents(c_lo, c_hi, sp.contents.P, dict(sp.contents.labels, m=f"probe interval {c_lo:.3f}..{c_hi:.3f} kg"))
    f_hi = float(np.clip(c_hi - n * OB.CUP["empty"], 0.0, OB.CUP["fill_max_kg"]))
    h_fill = SC.CUP_BOT + f_hi / (SC.WATER * np.pi * SC.CUP_RIN ** 2)
    sp.open_limits = [OB.OpenLimit("cup_spill", OB.cup_spill_limit(OB.CUP["D"], OB.CUP["H"], h_fill),
                                   "a_max = g 2 (H - h) / D, one mug holding the whole fill hi")]
    return sp


def static_interval(task, c_lo, c_hi, world=None):
    world = world if world is not None else (SC.TrayWorld(n_cups=n_cups(task)) if task.startswith("tray") else SC.BoxWorld())
    return CC.static_cell(task, (round(c_lo, 4), round(c_hi, 4)), world, spec=interval_spec(task, c_lo, c_hi))


@functools.lru_cache(maxsize=None)
def c_admit_max(task, tol=0.01):
    """largest contents mass c with the point interval [c, c] admitted (bisection; admission is monotone in the mass: MODEL --
    torque rows grow with m, the spill limit tightens with the fill).  None if even the lightest prior member is refused."""
    lo, hi = prior_contents(task)
    if static_interval(task, lo, lo)["action"].startswith("refuse"):
        return None
    if not static_interval(task, hi, hi)["action"].startswith("refuse"):
        return hi
    a, b = lo, hi
    while b - a > tol:
        c = 0.5 * (a + b)
        a, b = (c, b) if not static_interval(task, c, c)["action"].startswith("refuse") else (a, c)
    return a


@functools.lru_cache(maxsize=None)
def _rest_envelope(task):
    spec = interval_spec(task, *prior_contents(task))
    caps = json.loads(TH.CAP_FILE.read_text())
    offs, _ = CC.path_offsets(task)
    cap = TH.capture_path(caps["pot"], offs)
    tl, tr = np.asarray(caps["pot"]["tcp"]["W0_liftoff"]["left"]), np.asarray(caps["pot"]["tcp"]["W0_liftoff"]["right"])
    o_ref = 0.5 * (tl + tr) - np.array([0, 0, TH.GRASP_DZ])
    return E.Envelope(spec, E.Arms(cap), o_ref, np.eye(3), ["left", "right"], grip_mass=caps["pot"]["gripper_mass"], offsets=offs)


def rung_torque_u(task, m_ff):
    """static u at the rest pose (path key p0, a = 0) carrying total mass m_ff (contents m_ff - empty, COM anywhere in P): MODEL."""
    c = max(m_ff - empty_mass(task), 0.0)
    P = interval_spec(task, *prior_contents(task)).contents.P
    u, _ = _rest_envelope(task).worst(OB.Contents(c, c, P), ["p0"], 0.0, "two_lever")
    return float(u)


# ------------------------------------------------------------------------------------------------- the probe itself
def true_mass(ep):
    return float(sum(ep.m.body_subtreemass[b] for b in ep.carried))


class Probe:
    def __init__(self, ep, task, eps_lo, eps_free, rng=None, noise=True):
        self.ep, self.task = ep, task
        self.eps_lo, self.eps_free = float(eps_lo), float(eps_free)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.noise = bool(noise)
        self.tau = {s: np.zeros(7) for s in CR.SIDES}
        orig = ep._step

        def step():                                   # keep the torque the robot commanded this step (robot-side quantity)
            orig()
            self.tau["left"] = ep.d.qfrc_applied[ep.ctl.ld].copy()
            self.tau["right"] = ep.d.qfrc_applied[ep.ctl.rd].copy()
        ep._step = step
        m_e = empty_mass(task)
        c_lo, c_hi = prior_contents(task)
        self.M_lo, self.M_hi = m_e + c_lo, m_e + c_hi           # Theta_0 (total carried mass)
        self.m_ff = 0.0
        self.rungs = []
        self.track = dict(rise_mm=0.0, tilt=0.0, spill=0.0)

    def read(self):
        """(load L [kg], worst-case noise bound [kg]) from one control step."""
        ep, m, d = self.ep, self.ep.m, self.ep.d
        mujoco.mj_forward(m, d)
        Fz, nb = 0.0, 0.0
        for s in CR.SIDES:
            dofs = ep.dofs[s]
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jp, jr, ep.site[s])
            Pz = np.linalg.pinv(np.vstack([jp[:, dofs], jr[:, dofs]]).T)[2]       # row of (J^T)^+ giving F_z
            r = self.tau[s] - d.qfrc_bias[dofs]
            if self.noise:
                r = r + self.rng.uniform(-TAU_NOISE, TAU_NOISE, size=7)
            Fz += float(Pz @ r)
            nb += float(np.abs(Pz).sum()) * TAU_NOISE
        return Fz / G, nb / G

    def _ramp(self, m_to, T):
        ep = self.ep
        n = max(int(round(T / CR.DT)), 1)
        m0 = self.m_ff
        for i in range(n):
            ep.ctl.mass = m0 + (m_to - m0) * (i + 1) / n
            ep._step()
        self.m_ff = m_to

    def rung(self, m_to):
        ep = self.ep
        self._ramp(m_to, T_RAMP)
        reads, nbs = [], []
        nh = int(round(T_HOLD / CR.DT))
        for i in range(nh):
            ep._step()
            if i >= nh - N_READ:
                L, nb = self.read()
                reads.append(L)
                nbs.append(nb)
        rise = 1e3 * (float(ep.d.xpos[ep.obj][2]) - ep.z0)
        self.track["rise_mm"] = max(self.track["rise_mm"], rise)
        self.track["tilt"] = max(self.track["tilt"], ep.track["obj_tilt_max"])
        self.track["spill"] = max(self.track["spill"], ep.track["spill_ratio"])
        return float(np.mean(reads)), float(max(nbs)), rise

    def admissible(self, m_to):
        u = rung_torque_u(self.task, m_to)
        over = m_to - self.M_lo
        return (u <= E.U_ADMIT and over <= O_MAX + 1e-9), dict(u=round(u, 4), overshoot=round(over, 4))

    def run(self, M_stop=None):
        """ladder until free / decided / not admissible.  M_stop: refuse once M_lo > M_stop (the heaviest admissible total)."""
        self.ep.ctl.mass = 0.0
        self.ep.run_phases([("hold", 0.3)])
        prev_L, n_flat, why = None, 0, None
        m_next = self.M_lo                                   # first rung: the lightest object Theta_0 allows (overshoot 0)
        while True:
            if M_stop is not None and self.M_lo > M_stop:
                why = "decided_refuse"
                break
            ok, rows = self.admissible(m_next)
            if not ok:
                why = "rung_not_admissible"
                break
            L, nb, rise = self.rung(m_next)
            dm = m_next - (self.rungs[-1]["m_ff"] if self.rungs else 0.0)
            flat = prev_L is not None and (L - prev_L) < FREE_RATIO * dm
            n_flat = n_flat + 1 if flat else 0
            rec = dict(m_ff=round(m_next, 4), L=round(L, 4), noise_bound=round(nb, 4), rise_mm=round(rise, 3), **rows,
                       free=n_flat >= 1)
            if n_flat >= 1:                                 # hanging free: interval read
                e = self.eps_free + nb
                self.M_lo, self.M_hi = max(self.M_lo, L - e), min(self.M_hi, L + e)
            else:                                           # still (partly) on the counter: lower bound only
                self.M_lo = max(self.M_lo, L - self.eps_lo - nb)
            rec["Theta"] = [round(self.M_lo, 4), round(self.M_hi, 4)]
            self.rungs.append(rec)
            prev_L = L
            if n_flat >= 1:
                why = "free"
                break
            m_next = self.m_ff + DM
        return why


# ------------------------------------------------------------------------------------------------- jobs
def _seed(kind, task, i):
    return int(np.random.SeedSequence([{"calib": 11, "e2e": 22, "onepoke": 33}[kind], CC.TASK_ID[task], i]).generate_state(1)[0])


def _world(task, seed):
    return CC.sample_world(task, PRIOR[task], np.random.default_rng(seed))


def _episode(world):
    return CR.CarryEpisode(world, m_hat=0.0, fric_comp=True, weld_solref=CC.WELD_STIFF)


def _calib_job(args):
    """noise-free ladder to free (no stop rule): record per-rung read vs truth."""
    task, i = args
    seed = _seed("calib", task, i)
    t0 = time.time()
    try:
        w = _world(task, seed)
        ep = _episode(w)
        M = true_mass(ep)
        pr = Probe(ep, task, eps_lo=0.0, eps_free=0.0, noise=False)
        pr.M_lo = empty_mass(task)                          # calibration: start soft from the empty object, ignore the stop rule
        why = pr.run(M_stop=None)
        free = [r for r in pr.rungs if r["free"]]
        on = [r for r in pr.rungs if not r["free"]]
        return dict(task=task, i=i, seed=seed, M_true=M, why=why, rungs=pr.rungs,
                    res_free=(free[-1]["L"] - M) if free else None,
                    res_on_max=max((r["L"] - M for r in on), default=None),
                    viol=ep.track["viol"], **pr.track, wall_s=round(time.time() - t0, 1))
    except Exception as e:
        return dict(task=task, i=i, seed=seed, error=repr(e))


def calibrate_summary(rows):
    out = {}
    for task in sorted({r["task"] for r in rows}):
        rs = [r for r in rows if r["task"] == task and "error" not in r]
        fr = [r["res_free"] for r in rs if r["res_free"] is not None]
        on = [r["res_on_max"] for r in rs if r["res_on_max"] is not None]
        out[task] = dict(n=len(rs), n_free=len(fr), n_error=sum("error" in r for r in rows if r["task"] == task),
                         free_res_max_abs=round(max(map(abs, fr)), 4) if fr else None,
                         free_res_p50=round(float(np.median(fr)), 4) if fr else None,
                         on_res_max=round(max(on), 4) if on else None,
                         n_viol=sum(bool(r["viol"]) for r in rs), rise_mm_max=round(max(r["rise_mm"] for r in rs), 2) if rs else None,
                         tilt_max=round(max(r["tilt"] for r in rs), 2) if rs else None,
                         spill_max=round(max(r["spill"] for r in rs), 3) if rs else None,
                         not_free=[r["why"] for r in rs if r["why"] != "free"])
    fr_all = [abs(r["res_free"]) for r in rows if r.get("res_free") is not None]
    on_all = [r["res_on_max"] for r in rows if r.get("res_on_max") is not None]
    eps = dict(EPS_FREE=round(max(fr_all), 4) if fr_all else None, EPS_LO=round(max(max(on_all), 0.0), 4) if on_all else None,
               label="MEASURED: max over n calibration worlds (noise-free reads; the ASSUMED torque-noise bound is added per read)")
    return dict(per_task=out, eps=eps)


def _e2e_job(args):
    task, i, eps_lo, eps_free = args
    seed = _seed("e2e", task, i)
    t0 = time.time()
    try:
        w = _world(task, seed)
        ep = _episode(w)
        M = true_mass(ep)
        cmax = c_admit_max(task)
        M_stop = None if cmax is None else empty_mass(task) + cmax
        pr = Probe(ep, task, eps_lo, eps_free, rng=np.random.default_rng(seed), noise=True)
        if not static_interval(task, *prior_contents(task), w)["action"].startswith("refuse"):
            why = "free"                                     # verdict already "carry" on all of Theta_0: no probe needed
            pr.M_lo, pr.M_hi = empty_mass(task) + prior_contents(task)[0], empty_mass(task) + prior_contents(task)[1]
        else:
            why = pr.run(M_stop=M_stop)
        probe_viol = ep.track["viol"]
        sound = bool(pr.M_lo - 1e-9 <= M <= pr.M_hi + 1e-9)
        rec = dict(task=task, i=i, seed=seed, M_true=M, why=why, rungs=pr.rungs, Theta=[pr.M_lo, pr.M_hi], theta_sound=sound,
                   probe_viol=probe_viol, probe_track=dict(pr.track), theta_world=CR._theta(w))
        decision = None
        if why == "free":
            c_lo, c_hi = pr.M_lo - empty_mass(task), pr.M_hi - empty_mass(task)
            st = static_interval(task, c_lo, c_hi, w)
            decision = dict(action=st["action"], a_admit=st["a_admit"], u_nom=st["u_nom"], contents=[c_lo, c_hi])
        rec["decision"] = decision
        if decision and decision["a_admit"] is not None:
            ep.set_belief(0.5 * (decision["contents"][0] + decision["contents"][1]), a_plan=decision["a_admit"])
            m_nom = ep.m_nom
            ep.ctl.mass = pr.m_ff
            pr._ramp(m_nom, 0.3)
            ep.run_phases([("hold", 0.4)])
            ep.ctl.mass = m_nom
            carry = CR.run_segments(ep, dict(segments=[]), branch_backups=True)
            rec["carry"] = dict(segments=carry["segments"], end_in_R=carry["end_in_R"], pre_hold_viol=carry["pre_hold_viol"])
            rec["outcome"] = "carried" if (carry["end_in_R"][0] and not any(s["nominal"]["viol"] for s in carry["segments"])) else "carry_failed"
        else:
            pr._ramp(0.0, 0.5)
            ep.run_phases([("hold", 0.3), ("weld", False), ("retreat", CR.RETREAT, 0.8), ("hold", 0.5)])
            okR, whyR = ep.in_R()
            rec["refuse_end_in_R"] = [okR, whyR]
            rec["outcome"] = "refused_in_R" if okR and not ep.track["viol"] else "refused_bad"
        rec["any_U"] = bool(probe_viol or ep.track["viol"] or any(s["nominal"]["viol"] for s in rec.get("carry", {}).get("segments", [])))
        rec["wall_s"] = round(time.time() - t0, 1)
        return rec
    except Exception as e:
        import traceback
        return dict(task=task, i=i, seed=seed, error=repr(e), tb=traceback.format_exc()[-800:])


def _onepoke_job(args):
    """ablation: one push straight to the strongest torque-admissible feedforward (no ladder), on the same object."""
    task, i = args
    seed = _seed("onepoke", task, i)
    try:
        w = _world(task, seed)
        ep = _episode(w)
        M = true_mass(ep)
        m_top = empty_mass(task)
        while rung_torque_u(task, m_top + DM) <= E.U_ADMIT and m_top < 8.0:
            m_top += DM
        pr = Probe(ep, task, 0.0, 0.0, noise=False)
        ep.ctl.mass = 0.0
        ep.run_phases([("hold", 0.3)])
        L, nb, rise = pr.rung(m_top)
        return dict(task=task, i=i, seed=seed, M_true=M, m_push=m_top, L=L, rise_mm=pr.track["rise_mm"], tilt=pr.track["tilt"],
                    spill=pr.track["spill"], viol=ep.track["viol"])
    except Exception as e:
        return dict(task=task, i=i, seed=seed, error=repr(e))


def e2e_summary(rows):
    L = ["task | n | probe stop: free / decided refuse / rung refused | Theta contains truth | outcome: carried / carry_failed / "
         "refused_in_R / refused_bad | any U | rungs p50/max | final width kg p50/max | probe rise mm max | errors"]
    for task in sorted({r["task"] for r in rows}):
        rs = [r for r in rows if r["task"] == task and "error" not in r]
        if not rs:
            continue
        c = lambda k, v: sum(r.get(k) == v for r in rs)
        nr = [len(r["rungs"]) for r in rs]
        wid = [r["Theta"][1] - r["Theta"][0] for r in rs]
        L.append(f"{task} | {len(rs)} | {c('why', 'free')} / {c('why', 'decided_refuse')} / {c('why', 'rung_not_admissible')} | "
                 f"{sum(r['theta_sound'] for r in rs)}/{len(rs)} | {c('outcome', 'carried')} / {c('outcome', 'carry_failed')} / "
                 f"{c('outcome', 'refused_in_R')} / {c('outcome', 'refused_bad')} | {sum(r['any_U'] for r in rs)} | "
                 f"{int(np.median(nr))}/{max(nr)} | {np.median(wid):.3f}/{max(wid):.3f} | "
                 f"{max(r['probe_track']['rise_mm'] for r in rs):.2f} | {sum('error' in r for r in rows if r['task'] == task)}")
    return "\n".join(L)


def _pool_run(fn, jobs, procs, out_file):
    import multiprocessing as mp
    rows = []
    with mp.get_context("spawn").Pool(procs, maxtasksperchild=3) as pool, open(out_file, "a") as f:
        for r in pool.imap_unordered(fn, jobs):
            rows.append(r)
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            print(f"{r.get('task')} {r.get('i')} {r.get('why', '')} {r.get('outcome', '')} {r.get('error', '')[:120]}", flush=True)
    return rows


ASSUMPTIONS = dict(DM=f"{DM} kg ASSUMED push step", O_MAX=f"{O_MAX} kg ASSUMED max push above the lower bound",
                   FREE_RATIO=f"{FREE_RATIO} ASSUMED free test (one rung)", TAU_NOISE=f"+-{TAU_NOISE} Nm/joint ASSUMED uniform",
                   grasp="ASSUMED rigid weld (user 11-01), stiff solref (0.004, 1)", controller="LeverCtl + FRIC_COMP (ASSUMED gains)",
                   rows="arm-side static rows (user 11-02 pm), U_ADMIT 0.9 ASSUMED", worlds="uniform on the full prior per task")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "e2e", "onepoke"])
    ap.add_argument("--tasks", default="box,tray6")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--calib")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = a.tasks.split(",")
    t0 = time.time()
    if a.cmd == "calibrate":
        rows = _pool_run(_calib_job, [(t, i) for t in tasks for i in range(a.n)], a.procs, out / "calib.jsonl")
        s = calibrate_summary(rows)
        s.update(date=time.strftime("%Y-%m-%d"), assumptions=ASSUMPTIONS, n_per_task=a.n, wall_s=round(time.time() - t0))
        (out / "CALIB.json").write_text(json.dumps(s, indent=1, default=float))
        print(json.dumps(s, indent=1, default=float))
    elif a.cmd == "e2e":
        cal = json.loads(pathlib.Path(a.calib).read_text())["eps"]
        jobs = [(t, i, cal["EPS_LO"], cal["EPS_FREE"]) for t in tasks for i in range(a.n)]
        rows = _pool_run(_e2e_job, jobs, a.procs, out / "e2e.jsonl")
        s = e2e_summary(rows)
        hdr = (f"probe ladder end-to-end ({time.strftime('%Y-%m-%d')}), worlds uniform on the full prior, n = {a.n} per task, held-out "
               f"seeds (calibration used different worlds)\neps (MEASURED, {a.calib}): {cal}\nassumptions: {ASSUMPTIONS}\n"
               f"static admission ceiling (MODEL): " + ", ".join(f"{t} contents <= {c_admit_max(t)}" for t in tasks) + "\n")
        (out / "SUMMARY.txt").write_text(hdr + s + f"\nwall {round(time.time() - t0)} s\n")
        print(hdr + s, flush=True)
    else:
        rows = _pool_run(_onepoke_job, [(t, i) for t in tasks for i in range(a.n)], a.procs, out / "onepoke.jsonl")
        (out / "SUMMARY.txt").write_text("\n".join(json.dumps({k: r.get(k) for k in ("task", "i", "M_true", "m_push", "rise_mm", "tilt",
                                                                                    "spill", "viol", "error")}, default=float)
                                                   for r in rows) + "\n")


if __name__ == "__main__":
    main()
