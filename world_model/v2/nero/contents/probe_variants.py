"""nero/contents/probe_variants.py -- probe-ladder "free" test variants, tested separately (USER 11-05: options 1, 2 and 4).  SIMULATION ONLY.
Why: with today's test ("free" = the load read L grows by < FREE_RATIO x the push step) the ladder never sees boxes above ~2 kg total or
the 6-mug tray go free: L keeps rising ~0.57 kg/kg after lift-off, and the overshoot cap (box) or the rest-pose torque row (tray6) stops it.
Variants (probe.py is NOT edited; Probe is subclassed and patched in per worker):
  control  option 4: today's probe (load-plateau test, T_HOLD 0.4 s).  eps = the 11-03 calibration.
  rise     option 1: free once the CLAW sites rose >= RISE_MM above their rest height at the end of a rung's hold.  The claw height is
           robot-observable (encoders + FK; the claws are welded to the object).  RISE_MM ASSUMED.
  settle   option 2: today's load test with a longer hold per rung (T_HOLD_SETTLE) so L settles before the next push.  ASSUMED.
Each variant is calibrated on the calibration seeds (eps MEASURED for that variant), then run end to end on the probe e2e seeds (the same
worlds as same_worlds.py).  U includes the payload row for every variant (same_worlds._init_worker).
    python -m world_model.v2.nero.contents.probe_variants calibrate --variant rise --n 20 --procs 4 --out DIR
    python -m world_model.v2.nero.contents.probe_variants e2e --variant rise --calib DIR/CALIB.json --n 20 --procs 4 --out DIR
    python -m world_model.v2.nero.contents.probe_variants table --root ROOT
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import mujoco
import numpy as np

RISE_MM = 2.0          # ASSUMED (option 1): claw rise above rest that counts as "off the counter" [mm]
T_HOLD_SETTLE = 1.2    # ASSUMED (option 2): hold per rung before reading [s] (today 0.4)
VARIANTS = {"control": dict(free="load", t_hold=None), "rise": dict(free="rise", t_hold=None), "settle": dict(free="load", t_hold=T_HOLD_SETTLE)}
CONTROL_CALIB = "/data/local/ss5772/contents/probe_calib_2026-11-03/CALIB.json"   # ladder variants: the ladder CALIB (= probe.LADDER_CALIB)


def _make_class(variant):
    from world_model.v2.nero.contents import probe as P, carry as CR
    cfg = VARIANTS[variant]
    if not hasattr(P, "_Probe_orig"):
        P._Probe_orig = P.Probe                                # workers run several jobs: always subclass the ORIGINAL class

    class ProbeV(P._Probe_orig):
        def _claw_z(self):
            ep = self.ep
            mujoco.mj_forward(ep.m, ep.d)
            return float(np.mean([ep.d.site_xpos[ep.site[s]][2] for s in CR.SIDES]))

        def rung(self, m_to):
            if cfg["t_hold"] is None and cfg["free"] == "load":
                out = super().rung(m_to)                       # control: byte-for-byte today's rung
                self._claw_rise = 1e3 * (self._claw_z() - self._z_rest)
                return out
            t_hold_old = P.T_HOLD
            try:
                if cfg["t_hold"] is not None:
                    P.T_HOLD = cfg["t_hold"]
                out = super().rung(m_to)
            finally:
                P.T_HOLD = t_hold_old
            self._claw_rise = 1e3 * (self._claw_z() - self._z_rest)
            return out

        def run(self, M_stop=None):
            """probe.Probe.run with the variant's free test; records claw_rise_mm per rung."""
            self.ep.ctl.mass = 0.0
            self.ep.run_phases([("hold", 0.3)])
            self._z_rest = self._claw_z()
            prev_L, n_flat, why = None, 0, None
            m_next = self.M_lo
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
                if cfg["free"] == "rise":
                    free = self._claw_rise >= RISE_MM
                else:
                    free = prev_L is not None and (L - prev_L) < P.FREE_RATIO * dm
                n_flat = n_flat + 1 if free else 0
                rec = dict(m_ff=round(m_next, 4), L=round(L, 4), noise_bound=round(nb, 4), rise_mm=round(rise, 3),
                           claw_rise_mm=round(self._claw_rise, 3), **rows, free=n_flat >= 1)
                if n_flat >= 1:
                    e = self.eps_free + nb
                    self.M_lo, self.M_hi = max(self.M_lo, L - e), min(self.M_hi, L + e)
                else:
                    self.M_lo = max(self.M_lo, L - self.eps_lo - nb)
                rec["Theta"] = [round(self.M_lo, 4), round(self.M_hi, 4)]
                self.rungs.append(rec)
                prev_L = L
                if n_flat >= 1:
                    why = "free"
                    break
                m_next = self.m_ff + P.DM
            return why
    return ProbeV


def _job(args):
    cmd, variant, task, i, eps_lo, eps_free = args
    from world_model.v2.nero.contents import same_worlds as SW, probe as P
    SW._init_worker()                                          # payload row in U (strict_claw=True, T_cap=None)
    P.Probe = _make_class(variant)
    r = P._calib_job((task, i), method="ladder") if cmd == "calibrate" else P._e2e_job((task, i, eps_lo, eps_free), method="ladder")   # variants subclass the ladder Probe (make_probe("ladder") returns the patched P.Probe)
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


def table(root):
    """machine-written comparison of the variants' e2e runs (same worlds)."""
    root = pathlib.Path(root)
    L = ["| variant | task | n | free (upper bound) | truth in set | width kg p50/max | carried | refused | any U | rungs p50 | "
         "probe time s p50 | rise mm max | eps lo/free |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for vd in sorted(p for p in root.iterdir() if (p / "e2e.jsonl").exists()):
        cal = json.loads((vd / "CALIB_USED.json").read_text())
        rows = [json.loads(l) for l in open(vd / "e2e.jsonl")]
        for task in sorted({r["task"] for r in rows}):
            rs = [r for r in rows if r["task"] == task and "error" not in r]
            if not rs:
                continue
            wid = [r["Theta"][1] - r["Theta"][0] for r in rs]
            nr = [len(r["rungs"]) for r in rs]
            L.append(f"| {vd.name} | {task} | {len(rs)} | {sum(r['why'] == 'free' for r in rs)} | {sum(bool(r['theta_sound']) for r in rs)} | "
                     f"{np.median(wid):.2f}/{max(wid):.2f} | {sum(r.get('outcome') == 'carried' for r in rs)} | "
                     f"{sum(str(r.get('outcome', '')).startswith('refused') for r in rs)} | {sum(bool(r.get('any_U')) for r in rs)} | "
                     f"{int(np.median(nr))} | {np.median([r.get('wall_s', 0) for r in rs]):.0f} | "
                     f"{max(r['probe_track']['rise_mm'] for r in rs):.1f} | {cal['EPS_LO']}/{cal['EPS_FREE']} |")
    (root / "TABLE.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "e2e", "table"])
    ap.add_argument("--variant", choices=list(VARIANTS))
    ap.add_argument("--tasks", default="box,tray6")
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
    if a.cmd == "calibrate":
        rows = _run("calibrate", a.variant, tasks, a.n, a.procs, out)
        s = P.calibrate_summary(rows)
        s.update(variant=a.variant, date=time.strftime("%Y-%m-%d"), n_per_task=a.n, wall_s=round(time.time() - t0),
                 assumptions=dict(RISE_MM=f"{RISE_MM} mm ASSUMED", T_HOLD_SETTLE=f"{T_HOLD_SETTLE} s ASSUMED", **P.ASSUMPTIONS))
        (out / "CALIB.json").write_text(json.dumps(s, indent=1, default=float))
        print(json.dumps(s["eps"]), json.dumps(s["per_task"], default=float)[:1500], flush=True)
    else:
        cal = json.loads(pathlib.Path(a.calib).read_text())["eps"]
        (out / "CALIB_USED.json").write_text(json.dumps(dict(cal, source=a.calib), indent=1))
        _run("e2e", a.variant, tasks, a.n, a.procs, out, cal["EPS_LO"], cal["EPS_FREE"])
        print(f"wall {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
