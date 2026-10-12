"""robust/budget.py -- TEST 3 support: the DERIVED latency budget, and the oracle's own tolerable delay. SIMULATION / SAVED DATA ONLY
(MuJoCo CPU; no hardware, no network).

Idea. A monitor that sees the ball `Delta` ticks late starts its fallback from the state `Delta` ticks later. In between the
tray follows the NOMINAL command. The fallback still works if that later state is still certified savable. So per
certified-savable state s_t of a nominal carry:
  lead(s)      = onset - t      ticks until the nominal carry drops the ball (stored data; the task statement quotes a median of 3.5
                                ticks from the tray studies; we do not re-derive that number, it is a parameter)
  D(s)         = the largest Delta (even, the snapshot grid has stride 2) such that the oracle still labels (e, t + Delta)
                 savable (or hold_ok) for every Delta' <= Delta. The ORACLE tolerable delay: it uses the library search, so it
                 is an upper bound on what any fallback tolerates (the learned fallback is no better than the best library element).
  rec(s)       = lead(s) - D(s) the time the fallback itself needs; i.e. D is "lead minus recovery" measured, not assumed.
  turn(s)      = ticks until the ball stops approaching the edge under the best library sequence (first tick the distance does not shrink);
                 a second, direct measure of the recovery time (needs one MuJoCo rollout per state).
  fb_bridge    = median number of fallback ticks of F3 Rc3k on carries it SAVED in the bridge TEST runs (existing jsonl, no re-run):
                 the duration the learned fallback runs. It bounds recovery from above (it includes late un-intervening).
NOTE: the measured lead (onset - t, median about 48 ticks in the smoke) is a different quantity from the task's 3.5 ticks; the
derived budget below uses the task's number as given, so it can be strongly negative. Read it with the measured D.
Derived budget (ticks) = lead_given - recovery, with lead_given = 3.5 (parameter) and recovery = median turn(s) (oracle) or fb_bridge.
Negative = no budget: the fallback needs longer than the lead. Both are printed; the plot (report.py) draws the measured D too.

  python -m world_model.v2.tray_rep.robust.budget run --n-states 100 --workers 2
  python -m world_model.v2.tray_rep.robust.budget summarize
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K          # noqa: F401
from world_model.v2.tray_rep.store import Store
from world_model.v2.tray_rep import oracle as OR
from world_model.v2.tray_rep.robust import util as RU

LEAD_GIVEN = 3.5
G = {}
DMAX = 40


def _init(setup):
    G["st"] = Store(RU.data_dir(setup), "test")
    G["renv"] = RU.make_clean_env(G["st"].meta)
    G["orc"] = OR.load_oracle(RU.data_dir(setup) / "oracle" / "test_v0.npz")


def _work(j):
    st, renv, orc = G["st"], G["renv"], G["orc"]
    e, t = int(orc["ep"][j]), int(orc["t"][j])
    info = st.ep_info(e)
    onset, L = info["onset"], info["lens"]
    snap, hid = st.snapshot_at(e, t)
    renv.set_mu_roll(hid)
    rec = dict(j=int(j), ep=e, t=t, onset=onset, lead=(onset - t if onset >= 0 else -1), hold_ftk=int(orc["hold_ftk"][j]), D=-1,
               D_cap=False, turn=-1)
    # --- turnaround of the best library sequence
    n = len(renv.U)
    i, dh = int(orc["best_i"][j]), int(orc["best_dh"][j])
    sn = snap.snap
    seqs = OR.best_sequence(sn.kin[:3].copy(), sn.kin[7:10].copy(), sn.kin[3:7].copy(), renv.U, dh if dh > 0 else 10, OR.K_H,
                            renv.a_max, renv.w_max, snap.w_slew, renv.alpha)
    seq = seqs[i] if 0 <= i < n + 2 else seqs[n + 1]
    renv.restore(snap, mu_roll=hid)
    d_prev, tmin = renv.env.dist(), 60
    for k in range(60):
        r = renv.tick(np.asarray(seq[min(k, len(seq) - 1)], np.float64))
        if r["d"] >= d_prev - 1e-6:                    # the ball stops approaching the edge: first tick the distance does not shrink
            tmin = k + 1
            break
        d_prev = r["d"]
        if r["failed"]:
            break
    rec["turn"] = int(tmin)
    # --- tolerable delay D: certify the nominal continuation at t + 2, t + 4, ...
    D = 0
    for dl in range(2, DMAX + 1, 2):
        tt = t + dl
        if tt > L - 2:
            rec["D_cap"] = True
            D = dl - 2
            break
        try:
            s2, h2 = st.snapshot_at(e, tt)
        except KeyError:
            rec["D_cap"] = True
            D = dl - 2
            break
        if float(st.rows("l", e, tt, tt + 1)[0]) <= 0:
            break
        renv.set_mu_roll(h2)
        lab = OR.certify(renv, s2, (OR.D_HOLD,))["label"]
        if lab == OR.UNCERT:
            break
        D = dl
        if dl == DMAX:
            rec["D_cap"] = True
    rec["D"] = int(D)
    return rec


def run(setup="piper", n_states=100, workers=1, out_root=RU.OUT, tag="main"):
    orc = OR.load_oracle(RU.data_dir(setup) / "oracle" / "test_v0.npz")
    sav = np.flatnonzero(orc["label"] == OR.SAVABLE)
    sel = sav[np.linspace(0, len(sav) - 1, min(n_states, len(sav))).astype(int)]
    pj = Path(out_root) / "budget" / tag / f"states_{setup}.jsonl"
    pj.parent.mkdir(parents=True, exist_ok=True)
    done = {r["j"] for r in RU.read_jsonl(pj)}
    todo = [int(j) for j in sel if int(j) not in done]
    RU.log(f"[budget] n={len(sel)} todo={len(todo)} -> {pj}")
    t0 = time.perf_counter()
    if todo:
        with open(pj, "a") as f:
            if workers <= 1:
                _init(setup)
                for i, j in enumerate(todo):
                    f.write(json.dumps(_work(j)) + "\n"); f.flush()
            else:
                with Pool(workers, initializer=_init, initargs=(setup,)) as P:
                    for r in P.imap_unordered(_work, todo):
                        f.write(json.dumps(r) + "\n"); f.flush()
    RU.log(f"[budget] {time.perf_counter() - t0:.0f}s")
    return summarize(setup, out_root, tag)


def bridge_fallback_ticks(setup="piper_trk", arm="Rc3k", tau=None):
    """median fallback ticks of F3 on carries it saved (nominal drops, filter does not), pooled over the 3 model seeds."""
    tau = RU.chosen_tau(setup, arm) if tau is None else tau
    vals, n_saved = [], 0
    for s in RU.SEEDS:
        p = RU.BRIDGE / "test" / setup / f"F3__arm-{arm}_tau-{tau:g}__s{s}.jsonl"
        for r in RU.read_jsonl(p):
            if r["onset_stored"] >= 0 and not r["dropped"]:
                vals.append(r.get("x_fallback_ticks", 0))
                n_saved += 1
    return dict(n_saved=n_saved, median=float(np.median(vals)) if vals else float("nan"),
                p25=float(np.percentile(vals, 25)) if vals else float("nan"),
                p75=float(np.percentile(vals, 75)) if vals else float("nan"), tau=tau, source=str(RU.BRIDGE / "test" / setup))


def summarize(setup="piper", out_root=RU.OUT, tag="main", lead_given=LEAD_GIVEN):
    recs = RU.read_jsonl(Path(out_root) / "budget" / tag / f"states_{setup}.jsonl")
    if not recs:
        return {}
    g = lambda k: np.array([r[k] for r in recs], float)
    q = lambda a: [float(x) for x in np.percentile(a, [10, 25, 50, 75, 90])]
    lead, D, turn = g("lead"), g("D"), g("turn")
    ok = lead >= 0
    fb = {s: bridge_fallback_ticks(s) for s in RU.SETUPS}
    out = dict(n=len(recs), lead_given=lead_given, lead_pct_10_25_50_75_90=q(lead[ok]) if ok.any() else None,
               D_pct=q(D), D_capped_frac=float(np.mean([r["D_cap"] for r in recs])), turn_pct=q(turn),
               recovery_from_D_pct=q((lead - D)[ok]) if ok.any() else None,
               recovery_turn_median=float(np.median(turn)),
               bridge_fallback=fb,
               derived_budget_ticks=dict(
                   from_turn=float(lead_given - np.median(turn)),
                   from_bridge_trk=float(lead_given - fb["piper_trk"]["median"]),
                   measured_D_median=float(np.median(D))),
               how=("lead_given is the task's median 3.5 ticks (not re-derived). recovery = median ticks the oracle's best library "
                    "sequence needs until the ball stops approaching the edge (turn), or the median fallback ticks of the learned F3 on "
                    "carries it saved (bridge TEST jsonl). measured_D = oracle tolerable delay: largest even Delta with the nominal "
                    "continuation at t+Delta still certified savable."))
    RU.atomic_json(out, Path(out_root) / "budget" / tag / f"summary_{setup}.json")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "summarize"])
    ap.add_argument("--setup", default="piper", choices=RU.SETUPS)
    ap.add_argument("--n-states", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", default=str(RU.OUT))
    a = ap.parse_args()
    out = run(a.setup, a.n_states, a.workers, a.out, a.tag) if a.cmd == "run" else summarize(a.setup, a.out, a.tag)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
