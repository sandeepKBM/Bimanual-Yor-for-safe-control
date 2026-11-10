"""nero/contents/probe_liftoff.py -- lift-off detection options for the weighing probe, tested separately.  SIMULATION ONLY.
Diagnosis (11-05 rise calibration): the arms hold a fixed pose reference while the push ramps, so the position loop (and the ki term of
carry.FRIC_COMP) cancels part of the push; lift-off needs push ~ M + 0.2-0.5 kg, which the payload row forbids for heavy boxes / tray6.
Variants (probe.py, carry.py, probe_variants.py are NOT edited; probe.Probe is subclassed, patched in per worker):
  control  today's Probe (push ladder, load-plateau free test), plus per-rung extras (claw rise, peak payload utilisation, probe time).
  poslift  option A, "lift by position, read while holding": feed-forward ctl.mass = M_lo (current lower bound of the total mass);
           height ladder: both claws move up DZ over T_MOVE, hold T_HOLD, read L (mean of the last N_READ steps).  Each rung: M_lo <- max(M_lo,
           L - EPS_LO - nb) and ctl.mass is ramped to the new M_lo.  FREE when the last two dL = L_k - L_{k-1} are both < FREE_TOL in
           magnitude (3 rungs agree) -> interval [L - EPS_FREE - nb, L + EPS_FREE + nb].  decided_refuse when M_lo > M_stop; rung_not_admissible
           when the static torque row u(M_lo) > U_ADMIT; z_cap at Z_MAX (lower bound only); abort when the sim-truth claw payload row fires
           (ep.track["viol"]).  In every case the claws are moved back down to the start height before returning.
  softz    option B, "soft vertically while weighing": today's push ladder, but (1) before each rung both claws' commanded z is re-based to
           the MEASURED z (a T_REBASE ramp, path continuation IK, z only), so the vertical position error is ~0 when the push starts and the
           loop does not cancel it; (2) for the probe, ep.fc kp and ki are set to 0 (Coulomb term kept) and restored afterwards (instance
           attribute, carry.py untouched).  It does NOT soften x/y/orientation (their references stay), does NOT change the FL-MPC tracker
           gains, and after lift-off the reference z (= measured z at the rung start) still acts as a spring through the FL-MPC tracker.
           O_MAX and the u row are kept (probe.Probe.admissible).
Every episode still uses the existing tracker (carry.LeverCtl, FL-MPC, pre-11-07).  U includes the per-claw payload row (same_worlds._init_worker).
    python -m world_model.v2.nero.contents.probe_liftoff calibrate --variant poslift --tasks box,tray6,pot --n 20 --procs 4 --out DIR/poslift
    python -m world_model.v2.nero.contents.probe_liftoff e2e --variant poslift --calib DIR/poslift/CALIB.json --n 20 --procs 4 --out DIR2/poslift
    python -m world_model.v2.nero.contents.probe_liftoff table --root DIR
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                              # pinned before numpy / numba load (spawned workers re-import this module)

import mujoco
import numpy as np

DZ = float(os.environ.get("PL_DZ", 0.001))   # ASSUMED, USER-approved 11-09 (poslift): height step between rungs [m]
Z_MAX = float(os.environ.get("PL_Z_MAX", 0.020))   # ASSUMED, USER-approved 11-09 (poslift): hard cap on the commanded claw rise [m]; env PL_Z_MAX for the slow/high run (11-09)
FREE_TOL = float(os.environ.get("PL_FREE_TOL", 0.05))   # ASSUMED, USER-approved 11-09 (poslift): "free" when two consecutive |dL| < FREE_TOL [kg]; calibrate reports the measured dL spread to justify it
T_MOVE = 0.3          # design (= probe.T_RAMP): time for one height step / ff ramp [s]
T_REBASE = 0.1        # ASSUMED (softz): time to re-base the commanded z to the measured z before a rung [s]
T_HOLD_POSLIFT = float(os.environ.get("PL_T_HOLD", 1.2))   # ASSUMED, USER-approved 11-09 (poslift): hold per height rung [s]
N_FREE_CONFIRM = 2    # ASSUMED, USER-approved 11-09 (poslift): number of consecutive flat dL needed
VARIANTS = ("control", "poslift", "softz", "poslift_mixed")
MIXED_DZ = (0.001, 0.0005, 0.00025, 0.0001)   # calibration only (poslift_mixed): per-rung step drawn from these [m] to MEASURE dL/dz at the sizes the adaptive probe uses (0.1 mm = the proposed DZ_MIN, ASSUMED)
PHYSICS = ("tracker = carry.LeverCtl (FL-MPC, pre-11-07, the existing tracker); U includes the per-claw payload row (strict_claw); "
           "claw rise = encoders+FK, object rise = sim truth, peak_u = sim-truth weld wrench / F_cap")


def _make_class(variant):
    from world_model.v2.nero.contents import probe as P, carry as CR
    from world_model.v2.nero.democ import kin as K
    if not hasattr(P, "_Probe_orig"):
        P._Probe_orig = P.Probe                                # workers run several jobs: always subclass the ORIGINAL class
    Orig = P._Probe_orig

    class Base(Orig):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._extra, self.peak_u, self.probe_time_s = [], 0.0, None
            _LAST["pr"] = self

        def _claw_z(self):
            mujoco.mj_forward(self.ep.m, self.ep.d)
            return float(np.mean([self.ep.d.site_xpos[self.ep.site[s]][2] for s in CR.SIDES]))

        def _begin(self):
            self.ep.ctl.mass = 0.0
            self.ep.run_phases([("hold", 0.3)])
            self._z_rest, self._k0 = self._claw_z(), self.ep.k

        def _end(self):
            self.probe_time_s = round((self.ep.k - self._k0) * CR.DT, 3)

        def _rec_extra(self, e):
            e["peak_u"] = round(e["peak_u"], 4)
            self.peak_u = max(self.peak_u, e["peak_u"])

    class Control(Base):
        def rung(self, m_to):
            ep = self.ep
            fz0, ep.track["Fz_max"] = ep.track["Fz_max"], 0.0
            out = super().rung(m_to)
            e = dict(claw_rise_mm=round(1e3 * (self._claw_z() - self._z_rest), 3), peak_u=ep.track["Fz_max"] / ep.F_cap)
            ep.track["Fz_max"] = max(fz0, ep.track["Fz_max"])
            self._rec_extra(e)
            self._extra.append(e)
            return out

        def run(self, M_stop=None):
            self._begin()
            try:
                why = super().run(M_stop)                      # probe.Probe.run (it repeats the 0.3 s hold; harmless, same as today)
            finally:
                self._end()
            for r, e in zip(self.rungs, self._extra):
                r.update(e)
            return why

    class SoftZ(Control):
        def _rebase(self):
            """commanded claw z <- measured claw z (z only; continuation IK from the last commanded q)."""
            ep = self.ep
            t = ep.k * CR.DT
            dz = {}
            for s in CR.SIDES:
                q = ep.paths[s][-1][2][-1] if ep.paths[s] else ep.q0[s]
                z_cmd = float(K.PoseIK(ep.m, s).ik.pose(q.copy())[0][2])
                dz[s] = float(ep.d.site_xpos[ep.site[s]][2]) - z_cmd
            for s in CR.SIDES:
                ep._add_path(s, t, t + T_REBASE, np.array([0.0, 0.0, dz[s]]))
            for _ in range(max(int(round(T_REBASE / CR.DT)), 1)):
                ep._step()
            self._last_rebase_mm = {s: round(1e3 * v, 3) for s, v in dz.items()}

        def rung(self, m_to):
            self._rebase()
            out = super().rung(m_to)
            self._extra[-1]["rebase_mm"] = self._last_rebase_mm
            return out

        def run(self, M_stop=None):
            ep = self.ep
            fc0, ei0 = ep.fc, {s: ep.e_int[s].copy() for s in CR.SIDES}
            if ep.fc is not None:
                ep.fc = dict(ep.fc, kp=0.0, ki=0.0)             # Coulomb term kept; instance attribute only
                for s in CR.SIDES:
                    ep.e_int[s][:] = 0.0
            try:
                return super().run(M_stop)
            finally:
                ep.fc = fc0
                for s in CR.SIDES:
                    ep.e_int[s][:] = 0.0

    class PosLift(Base):
        MIXED = False

        def _dz_k(self, idx):
            if not self.MIXED:
                return DZ
            if not hasattr(self, "_mix_rng"):
                self._mix_rng = np.random.default_rng(int(round(1e6 * P.true_mass(self.ep))))   # deterministic per world
            return float(self._mix_rng.choice(MIXED_DZ))

        def _move(self, dz, ramp_to=None):
            """both claws up dz [m] over T_MOVE (path continuation IK); the ff ramps to ramp_to over the same time."""
            ep = self.ep
            t = ep.k * CR.DT
            if abs(dz) > 0:
                for s in CR.SIDES:
                    ep._add_path(s, t, t + T_MOVE, np.array([0.0, 0.0, dz]))
            self._ramp_during(ramp_to, T_MOVE)

        def _ramp_during(self, m_to, T):
            ep = self.ep
            n = max(int(round(T / CR.DT)), 1)
            m0 = ep.ctl.mass
            m_to = m0 if m_to is None else m_to
            for i in range(n):
                ep.ctl.mass = m0 + (m_to - m0) * (i + 1) / n
                ep._step()
            self.m_ff = ep.ctl.mass

        def run(self, M_stop=None):
            ep = self.ep
            self._begin()
            self.m_ff = 0.0
            zc, idx, prev_L, flat_mm, why = 0.0, 0, None, 0.0, None
            try:
                while True:
                    if M_stop is not None and self.M_lo > M_stop:
                        why = "decided_refuse"
                        break
                    ok, rows = self.admissible(self.M_lo)
                    if not ok:
                        why = "rung_not_admissible"
                        break
                    dz_k = self._dz_k(idx)
                    if zc + dz_k > Z_MAX + 1e-9:
                        why = "z_cap"
                        break
                    fz0, ep.track["Fz_max"] = ep.track["Fz_max"], 0.0
                    dz_k = dz_k if idx > 0 else 0.0
                    self._move(dz_k, ramp_to=self.M_lo)
                    zc += dz_k
                    reads, nbs, nh = [], [], int(round(T_HOLD_POSLIFT / CR.DT))
                    for i in range(nh):
                        ep._step()
                        if i >= nh - P.N_READ:
                            L, nb = self.read()
                            reads.append(L)
                            nbs.append(nb)
                    L, nb = float(np.mean(reads)), float(max(nbs))
                    peak_u = ep.track["Fz_max"] / ep.F_cap
                    ep.track["Fz_max"] = max(fz0, ep.track["Fz_max"])
                    dL = None if prev_L is None else L - prev_L
                    rate = None if dL is None else dL / (dz_k * 1e3)               # [kg/mm]
                    flat_mm, free = P.flat_update(flat_mm, dL, dz_k)
                    rise_obj = 1e3 * (float(ep.d.xpos[ep.obj][2]) - ep.z0)
                    self.track["rise_mm"] = max(self.track["rise_mm"], rise_obj)
                    self.track["tilt"] = max(self.track["tilt"], ep.track["obj_tilt_max"])
                    self.track["spill"] = max(self.track["spill"], ep.track["spill_ratio"])
                    rec = dict(m_ff=round(self.m_ff, 4), z_cmd_mm=round(1e3 * zc, 3), claw_rise_mm=round(1e3 * (self._claw_z() - self._z_rest), 3),
                               rise_mm=round(rise_obj, 3), L=round(L, 4), dL=None if dL is None else round(dL, 4), noise_bound=round(nb, 4),
                               peak_u=round(peak_u, 4), **rows, free=free,
                               dz_mm=round(1e3 * dz_k, 5), rate_kg_per_mm=None if rate is None else round(rate, 4))
                    self.peak_u = max(self.peak_u, peak_u)
                    if free:
                        e = self.eps_free + nb
                        self.M_lo, self.M_hi = max(self.M_lo, L - e), min(self.M_hi, L + e)
                    else:
                        self.M_lo = max(self.M_lo, L - self.eps_lo - nb)
                    rec["Theta"] = [round(self.M_lo, 4), round(self.M_hi, 4)]
                    self.rungs.append(rec)
                    prev_L = L
                    if ep.track["viol"]:
                        why = "free" if free else "abort"
                        rec["viol"] = ep.track["viol"]
                        break
                    if free:
                        why = "free"
                        break
                    idx += 1
            finally:
                if zc > 0:                                      # back to the start height (ff stays at M_lo <= M in expectation)
                    t = ep.k * CR.DT
                    for s in CR.SIDES:
                        ep._add_path(s, t, t + T_MOVE, np.array([0.0, 0.0, -zc]))
                    self._ramp_during(None, T_MOVE)
                    ep.run_phases([("hold", 0.3)])
                self._end()
            return why

    class PosLiftMixed(PosLift):
        MIXED = True

    return dict(control=Control, softz=SoftZ, poslift=PosLift, poslift_mixed=PosLiftMixed)[variant]


_LAST = {}


def _job(args):
    cmd, variant, task, i, eps_lo, eps_free = args
    from world_model.v2.nero.contents import same_worlds as SW, probe as P
    SW._init_worker()                                          # payload row in U (strict_claw=True, T_cap=None)
    P.Probe = _make_class(variant)
    _LAST.clear()
    r = P._calib_job((task, i), method="ladder") if cmd == "calibrate" else P._e2e_job((task, i, eps_lo, eps_free), method="ladder")   # variants subclass the ladder Probe (make_probe("ladder") returns the patched P.Probe)
    pr = _LAST.get("pr")
    if pr is not None and "error" not in r:
        r.update(probe_time_s=pr.probe_time_s, peak_u=round(pr.peak_u, 4),
                 interval=[round(pr.M_lo, 4), round(pr.M_hi, 4)], truth_inside=bool(pr.M_lo - 1e-9 <= r["M_true"] <= pr.M_hi + 1e-9))
    r.update(variant=variant)
    return r


def _run(cmd, variant, tasks, n, procs, out, eps_lo=0.0, eps_free=0.0):
    import multiprocessing as mp
    jobs = [(cmd, variant, t, i, eps_lo, eps_free) for i in range(n) for t in tasks]
    rows = []
    with mp.get_context("spawn").Pool(procs, maxtasksperchild=3) as pool, open(out / f"{cmd}.jsonl", "a") as f:
        for r in pool.imap_unordered(_job, jobs):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            rows.append(r)
            print(f"{variant} {cmd} {r.get('task')} i={r.get('i')} why={r.get('why')} {r.get('outcome', '')} "
                  f"{'ERR ' + r['error'] if 'error' in r else ''}", flush=True)
    return rows


def kstep_summary(rows, tol=0.05):
    """MEASURED load-rate bound K_STEP = max over rungs still on the counter (L < M_true - tol) of dL/dz [kg/mm], pooled over tasks (rows need dz_mm per rung,
    i.e. poslift / poslift_mixed calibration rows; the first rung has dL = None and is skipped).  Also: per task, per dz size, and the rate-test false-free count
    (a rung where `free` is declared although L < M_true - tol; n = rungs with free declared)."""
    per, by_dz, allr, ff, nfree = {}, {}, [], 0, 0
    for r in rows:
        if "rungs" not in r or r.get("M_true") is None:
            continue
        for k in r["rungs"]:
            if k.get("dL") is None or k.get("dz_mm") in (None, 0):
                continue
            if k["L"] < r["M_true"] - tol:
                rate = k["dL"] / k["dz_mm"]
                per.setdefault(r["task"], []).append(rate)
                by_dz.setdefault(k["dz_mm"], []).append(rate)
                allr.append(rate)
            if k.get("free"):
                nfree += 1
                ff += k["L"] < r["M_true"] - tol
    if not allr:
        return None
    lab = ", ".join(f"{t} {max(v):.4g} n={len(v)}" for t, v in sorted(per.items()))
    dzl = ", ".join(f"dz {d} mm: {max(v):.4g} n={len(v)}" for d, v in sorted(by_dz.items()))
    return dict(K_STEP=dict(value=round(max(allr), 4), n=len(allr), unit="kg/mm",
                            label=f"MEASURED: max per-rung dL/dz over rungs with L < M_true - {tol} (still on the counter), tasks pooled ({lab}); by step size ({dzl})"),
                FALSE_FREE=dict(n_false=int(ff), n_free_rungs=int(nfree), label="rungs declared free (rate test) while L < M_true - 0.05"))


def _dL_stats(rows):
    """spread of consecutive dL (poslift): max |dL| on the rungs that confirmed free vs min |dL| on non-flat rungs (justifies FREE_TOL)."""
    flat, moving = [], []
    for r in rows:
        if "error" in r:
            continue
        for k in r.get("rungs", []):
            if k.get("dL") is None:
                continue
            (flat if (r["why"] == "free" and k["free"]) else moving).append(abs(k["dL"]))
    return dict(dL_flat_max=round(max(flat), 4) if flat else None, dL_moving_min=round(min(moving), 4) if moving else None,
                dL_moving_p10=round(float(np.percentile(moving, 10)), 4) if moving else None, n_flat=len(flat), n_moving=len(moving))


def table(root):
    root = pathlib.Path(root)
    L = ["| variant | cmd | task | n | free / rung_not_adm / decided_refuse / z_cap / abort | truth in set | width kg p50 | peak u max | "
         "obj rise mm max | tilt max | spill max | viol | probe s p50 | err |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for vd in sorted(p for p in root.iterdir() if p.is_dir()):
        for cmd in ("calibrate", "e2e"):
            f = vd / f"{cmd}.jsonl"
            if not f.exists():
                continue
            rows = [json.loads(l) for l in open(f)]
            for task in sorted({r["task"] for r in rows}):
                rs = [r for r in rows if r["task"] == task and "error" not in r]
                ne = sum(1 for r in rows if r["task"] == task and "error" in r)
                if not rs:
                    L.append(f"| {vd.name} | {cmd} | {task} | 0 | - | - | - | - | - | - | - | - | - | {ne} |")
                    continue
                c = lambda v: sum(r["why"] == v for r in rs)
                wid = [r["interval"][1] - r["interval"][0] for r in rs if "interval" in r]
                tr = [r["probe_track"] if "probe_track" in r else r for r in rs]
                viol = sum(bool(r.get("probe_viol", r.get("viol"))) for r in rs)
                L.append(f"| {vd.name} | {cmd} | {task} | {len(rs)} | {c('free')} / {c('rung_not_admissible')} / {c('decided_refuse')} / {c('z_cap')} / "
                         f"{c('abort')} | {sum(bool(r.get('truth_inside')) for r in rs)}/{len(rs)} | {np.median(wid):.2f} | "
                         f"{max(r.get('peak_u', 0) for r in rs):.2f} | {max(t['rise_mm'] for t in tr):.2f} | {max(t['tilt'] for t in tr):.2f} | "
                         f"{max(t['spill'] for t in tr):.2f} | {viol} | {np.median([r.get('probe_time_s') or 0 for r in rs]):.1f} | {ne} |")
    (root / "TABLE.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "e2e", "table"])
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--tasks", default="box,tray6,pot")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--calib")
    ap.add_argument("--out")
    ap.add_argument("--root")
    a = ap.parse_args(argv)
    if a.cmd == "table":
        return table(a.root)
    from world_model.v2.nero.contents import probe as P
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks, t0 = a.tasks.split(","), time.time()
    asm = dict(DZ=f"{DZ} m ASSUMED", Z_MAX=f"{Z_MAX} m ASSUMED", FREE_TOL=f"{FREE_TOL} kg ASSUMED", T_MOVE=f"{T_MOVE} s design", T_HOLD_POSLIFT=f"{T_HOLD_POSLIFT} s ASSUMED",
               T_REBASE=f"{T_REBASE} s ASSUMED", N_FREE_CONFIRM=f"{N_FREE_CONFIRM} ASSUMED", physics=PHYSICS, **P.ASSUMPTIONS)
    if a.cmd == "calibrate":
        rows = _run("calibrate", a.variant, tasks, a.n, a.procs, out)
        s = P.calibrate_summary(rows)
        s.update(variant=a.variant, date=time.strftime("%Y-%m-%d"), n_per_task=a.n, wall_s=round(time.time() - t0), assumptions=asm,
                 dL=_dL_stats(rows))
        ks = kstep_summary(rows) if a.variant in ("poslift", "poslift_mixed") else None
        if ks:
            s.update(ks)
        (out / "CALIB.json").write_text(json.dumps(s, indent=1, default=float))
        print(json.dumps(s["eps"]), json.dumps(s["per_task"], default=float)[:1500], flush=True)
    else:
        cal = json.loads(pathlib.Path(a.calib).read_text())["eps"]
        (out / "CALIB_USED.json").write_text(json.dumps(dict(cal, source=a.calib, assumptions=asm), indent=1))
        _run("e2e", a.variant, tasks, a.n, a.procs, out, cal["EPS_LO"], cal["EPS_FREE"])
        print(f"wall {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
